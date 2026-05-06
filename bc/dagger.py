"""
DAgger: Dataset Aggregation for Ground Robot Waypoint Navigation
Reference: Ross et al., AISTATS 2011

This module implements the DAgger algorithm with several key enhancements:
  1. Unconditional Beta Decay: Beta decays unconditionally after an initial
     BC warmup period (pure-expert rollouts). This ensures the student policy
     is exposed to its own failure modes during training.

  2. Observation Stacking: Stacks the last STACK_N frames of observation features.
     This gives the neural network an implicit understanding of obstacle velocity
     and acceleration context, allowing it to better mimic the expert MPC's 
     constant-velocity obstacle prediction.

  3. Per-dimension Loss Tracking: Evaluates mean-squared error for both 
     steering (omega) and throttle (a) independently to diagnose potential
     learning imbalances.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import copy
import time
import sys
from tqdm import tqdm
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EnvConfig
from env.robot_env import GroundRobotEnv
from mpc.mpc import MPCController

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
DAGGER_ITERS      = 40 # Number of DAgger iterations (rollout + aggregate + train + eval)
ROLLOUT_STEPS     = 8000 # Number of steps to collect with the current policy each DAgger iteration
MAX_EPISODE_STEPS = 800 # Max steps per episode during rollouts (80s)
EVAL_EPISODES     = 20 # Number of episodes to evaluate the policy after each DAgger iteration

# Beta Decay Strategy: Warmup with pure expert, then unconditionally decay
BETA_WARMUP_ITERS = 5 # Number of initial DAgger iterations to hold beta=1.0 (pure expert) for BC warmup
BETA_INIT         = 1.0 # Initial beta (probability of using expert action during rollouts)
BETA_DECAY        = 0.85 # Multiplicative decay of beta after each DAgger iteration post-warmup (floor 0.05)

CURRICULUM_START  = 0.6 # Starting difficulty for rollouts (0.0 = easiest, 1.0 = full range). Linearly annealed to CURRICULUM_END across DAgger_ITERS. Adjust as needed based on your environment's difficulty curve.
CURRICULUM_END    = 1.0 # Ending difficulty for rollouts. Should be ≤ 1.0. Setting to 1.0 means the final DAgger iterations will sample from the full difficulty range, which is usually desirable to ensure the policy learns to handle the hardest scenarios.

# Observation stacking length
STACK_N           = 5

# Model and training
HIDDEN_DIM        = 512
NUM_LAYERS        = 5
LEARNING_RATE     = 1e-4
BATCH_SIZE        = 256
MAX_EPOCHS        = 300
PATIENCE          = 30

NEAR_OBS_MULTIPLIER = 4
NEAR_OBS_DIST_MULT  = 3.0
MAX_DATASET_SIZE  = 200_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
N_OBS      = EnvConfig.NUM_OBSTACLES
SINGLE_DIM = 4 + 6 * N_OBS
OBS_DIM    = SINGLE_DIM * STACK_N
ACT_DIM    = 2


def parse_obs(obs: np.ndarray) -> np.ndarray:
    """
    Converts raw environment observations into a policy-ready feature vector.
    
    Extracts the robot's local frame coordinates, transforms waypoint target and 
    obstacle positions/velocities into relative metrics, and serializes the state 
    for the neural network.
    
    Args:
        obs (np.ndarray): Raw environment observation array.
        
    Returns:
        np.ndarray: Parsed feature vector containing local waypoint offsets, 
                    robot speed, and relative localized obstacle data.
    """
    x, y, theta, v = obs[0:4]
    wx, wy          = obs[4:6]
    obstacles_flat  = obs[6:]

    c, s = np.cos(theta), np.sin(theta)

    def to_local(dx, dy):
        return c * dx + s * dy, -s * dx + c * dy

    dx_wp, dy_wp = to_local(wx - x, wy - y)
    dist_wp      = float(np.linalg.norm([dx_wp, dy_wp]))

    parsed_obstacles = []
    for i in range(N_OBS):
        base                 = i * 5
        ox, oy, ovx, ovy, r = obstacles_flat[base:base + 5]
        odx,  ody            = to_local(ox - x,  oy - y)
        ovx_loc, ovy_loc     = to_local(ovx, ovy)
        odist                = float(np.linalg.norm([odx, ody]))
        parsed_obstacles.append((odist, [odx, ody, ovx_loc, ovy_loc, r, odist]))

    parsed_obstacles.sort(key=lambda item: item[0])

    obs_features = []
    for _, feat in parsed_obstacles:
        obs_features.extend(feat)

    return np.array([dx_wp, dy_wp, v, dist_wp] + obs_features, dtype=np.float32)


def obs_to_expert_args(obs: np.ndarray):
    """
    Parses a raw observation into components suitable for the MPC expert.

    The MPC controller expects specific partitions of the state representation:
    the agent's kinematics, the active waypoint representation, and obstacles.

    Args:
        obs (np.ndarray): The raw state observation from the environment.

    Returns:
        tuple: (robot_state, w_active, obstacles), where each is a numpy array
               compatible with the MPC expert.
    """
    robot_state = obs[0:4].astype(np.float64)
    w_active    = obs[4:6].astype(np.float64)
    obstacles   = obs[6:6 + N_OBS * 5].reshape(N_OBS, 5).astype(np.float64)
    return robot_state, w_active, obstacles


def _is_near_obstacle(obs: np.ndarray) -> bool:
    """
    Checks if the robot is critically near any obstacle.
    
    Used to upsample dangerous states in the dataset aggregation phase.
    
    Args:
        obs (np.ndarray): The raw observation data.
        
    Returns:
        bool: True if an obstacle is within the proximity multiplier.
    """
    x, y           = obs[0], obs[1]
    obstacles_flat = obs[6:]
    for i in range(N_OBS):
        base            = i * 5
        ox, oy, _, _, r = obstacles_flat[base:base + 5]
        dist            = np.linalg.norm([ox - x, oy - y])
        if dist < NEAR_OBS_DIST_MULT * (EnvConfig.ROBOT_RADIUS + r):
            return True
    return False


# ──────────────────────────────────────────────
# Obs stack
# ──────────────────────────────────────────────
class ObsStack:
    """
    Maintains a rolling window of STACK_N parsed feature vectors.
    Zero-padded at episode start.

    Provides the neural network implicit access to dynamic temporal information:
      - obstacle velocity (positional delta across frames)
      - robot acceleration (speed delta)
      - recent heading changes
    This contextual historical data helps the model mimic the MPC's predictive behavior.
    """
    def __init__(self, n: int = STACK_N, feature_dim: int = SINGLE_DIM):
        self.n   = n
        self.dim = feature_dim
        self.buf = deque(maxlen=n)
        self.reset()

    def reset(self):
        self.buf.clear()
        for _ in range(self.n):
            self.buf.append(np.zeros(self.dim, dtype=np.float32))

    def push(self, feat: np.ndarray) -> np.ndarray:
        self.buf.append(feat)
        return np.concatenate(list(self.buf), axis=0)


# ──────────────────────────────────────────────
# Policy Network
# ──────────────────────────────────────────────
class PolicyNet(nn.Module):
    """
    MLP-based policy network mapping historical observation stacks to control actions.
    
    Contains a shared trunk and separates the final layer into two dedicated prediction 
    heads for angular velocity (omega) and linear acceleration (a) bounded by EnvConfig limits.
    """
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()]
        self.trunk      = nn.Sequential(*layers)
        self.head_omega = nn.Linear(hidden_dim, 1)
        self.head_a     = nn.Linear(hidden_dim, 1)
        self.omega_max  = float(EnvConfig.OMEGA_MAX)
        self.a_max      = float(EnvConfig.A_MAX)

    def forward(self, x):
        h     = self.trunk(x)
        omega = torch.tanh(self.head_omega(h)) * self.omega_max
        a     = torch.tanh(self.head_a(h))     * self.a_max
        return torch.cat([omega, a], dim=1)

    @torch.no_grad()
    def predict(self, stacked_feature: np.ndarray) -> np.ndarray:
        x = torch.tensor(stacked_feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return self.forward(x).squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────
# Dataset management
# ──────────────────────────────────────────────
def aggregate_dataset(all_states, all_actions,
                      new_states, new_actions, new_obs_raw, max_size):
    """
    Appends newly collected rollout trajectories into the cumulative DAgger dataset.
    
    To improve the policy's robustness, it applies an oversampling strategy 
    (data augmentation) for states where the robot is near an obstacle. Oldest records 
    are dropped if the dataset exceeds its allowed maximum capacity (max_size).
    """
    near_mask  = np.array([_is_near_obstacle(o) for o in new_obs_raw])
    danger_s   = new_states[near_mask]
    danger_a   = new_actions[near_mask]

    aug_s = new_states.copy()
    aug_a = new_actions.copy()
    for _ in range(NEAR_OBS_MULTIPLIER - 1):
        if len(danger_s) > 0:
            aug_s = np.concatenate([aug_s, danger_s], axis=0)
            aug_a = np.concatenate([aug_a, danger_a], axis=0)

    all_states  = np.concatenate([all_states,  aug_s], axis=0)
    all_actions = np.concatenate([all_actions, aug_a], axis=0)

    if len(all_states) > max_size:
        all_states  = all_states[-max_size:]
        all_actions = all_actions[-max_size:]

    return all_states, all_actions


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_policy(policy, states, actions, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """
    Trains the provided policy on the aggregated dataset using Behavioral Cloning.

    Uses normalized actions as targets to maintain scale parity during MSE loss 
    backpropagation. Employs Adam optimization, learning rate reduction on plateau,
    and early stopping. The validation subset dictates the best saved weights.

    Args:
        policy (nn.Module): The policy neural network model to be trained.
        states (np.ndarray): Historical stacked observations.
        actions (np.ndarray): Target expert actions to map observations to.
        max_epochs (int): Number of epochs to train.
        patience (int): Number of epochs with no improvement to trigger early stop.

    Returns:
        tuple: (trained_policy, corresponding_action_mean, corresponding_action_std)
    """
    action_mean  = actions.mean(axis=0)
    action_std   = actions.std(axis=0) + 1e-6
    actions_norm = (actions - action_mean) / action_std

    X = torch.tensor(states,       dtype=torch.float32)
    Y = torch.tensor(actions_norm, dtype=torch.float32)

    dataset  = TensorDataset(X, Y)
    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=0, pin_memory=True)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    a_mean_t = torch.tensor(action_mean, device=DEVICE)
    a_std_t  = torch.tensor(action_std,  device=DEVICE)

    best_val   = float('inf')
    best_w     = copy.deepcopy(policy.state_dict())
    pat_count  = 0
    best_epoch = 0
    last_omega = last_a = 0.0

    pbar = tqdm(range(max_epochs), desc="  [Train]", leave=False)
    for epoch in pbar:
        policy.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred      = policy(xb)
            pred_norm = (pred - a_mean_t) / a_std_t
            nn.functional.mse_loss(pred_norm, yb).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        policy.eval()
        v_omega = v_a = n_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb    = xb.to(DEVICE), yb.to(DEVICE)
                pred      = policy(xb)
                pred_norm = (pred - a_mean_t) / a_std_t
                v_omega  += nn.functional.mse_loss(pred_norm[:, 0], yb[:, 0]).item() * len(xb)
                v_a      += nn.functional.mse_loss(pred_norm[:, 1], yb[:, 1]).item() * len(xb)
                n_val    += len(xb)

        v_omega /= n_val; v_a /= n_val
        val_loss = (v_omega + v_a) / 2.0
        last_omega, last_a = v_omega, v_a
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val  = val_loss
            best_w    = copy.deepcopy(policy.state_dict())
            pat_count = 0
            best_epoch = epoch + 1
        else:
            pat_count += 1
            if pat_count >= patience:
                break

        pbar.set_postfix({"ω": f"{v_omega:.4f}", "a": f"{v_a:.4f}",
                          "pat": f"{pat_count}/{patience}"})
    pbar.close()

    policy.load_state_dict(best_w)
    print(f"    [Train] best val  ω={last_omega:.4f}  a={last_a:.4f}  @ epoch {best_epoch}")
    return policy, action_mean, action_std


# ──────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────
def rollout(env, policy, expert, beta, n_steps, max_ep_steps,
            difficulty=1.0, action_mean=None, action_std=None):
    """
    Executes episodes in the environment executing the current policy alongside an expert.
    
    DAgger specific rollout logic. At every step the environment's observation is 
    passed to both the expert MPC and the active network policy. An action is 
    selected with probability beta from the expert, and (1 - beta) from the policy.

    The returned payload is solely used for aggregating ground truth labels from 
    the expert trajectory data collection.
    
    Args:
        env (gym.Env): The simulated environment wrapper.
        policy (nn.Module): Current version of the policy net.
        expert (MPCController): MPC expert producing oracle actions.
        beta (float): Probability parameter controlling the expert intervention rate.
        n_steps (int): Total raw transitions to accumulate in iteration.
        max_ep_steps (int): Terminal truncation length.
        difficulty (float): Curriculum parameter applied to env resetting.

    Returns:
        tuple: Arrays (stacked_features, expert_actions_generated, direct_observations)
    """
    all_stacked, all_expert_actions, all_raw_obs = [], [], []
    steps_done = 0
    pbar  = tqdm(total=n_steps, desc="  [Rollout]", leave=False)
    stack = ObsStack()

    while steps_done < n_steps:
        obs, _ = env.reset(options={"difficulty": difficulty})
        stack.reset()
        ep_steps = 0

        while steps_done < n_steps:
            feat    = parse_obs(obs)
            stacked = stack.push(feat)

            robot_state, w_active, obstacles = obs_to_expert_args(obs)
            expert_action = expert.get_action(robot_state, w_active, obstacles).astype(np.float32)

            action = expert_action if np.random.rand() < beta else policy.predict(stacked)

            obs, _, terminated, truncated, _ = env.step(action)

            all_stacked.append(stacked.copy())
            all_expert_actions.append(expert_action)
            all_raw_obs.append(obs)

            steps_done += 1
            ep_steps   += 1
            pbar.update(1)

            if terminated or truncated or ep_steps >= max_ep_steps:
                if steps_done < n_steps:
                    obs, _ = env.reset(options={"difficulty": difficulty})
                    stack.reset()
                    ep_steps = 0
                else:
                    break

    pbar.close()
    return (np.array(all_stacked,        dtype=np.float32),
            np.array(all_expert_actions, dtype=np.float32),
            np.array(all_raw_obs,        dtype=np.float32))


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(env, policy, n_episodes=EVAL_EPISODES):
    """
    Evaluates the trained single-agent policy without expert assistance.

    Computes task metrics by testing the neural policy network strictly on actual
    unseen episodes. It records goals reached, waypoint progression, and collisions.
    
    Args:
        env (gym.Env): The simulated environment wrapper.
        policy (nn.Module): Current version of the policy net.
        n_episodes (int): Count of evaluation runs over different randomized scenarios.
        
    Returns:
        dict: A dictionary storing rates (0-1) for goal success, waypoint success, 
              and collisions.
    """
    policy.eval()
    goals = collisions = wps_reached = wps_possible = 0
    stack = ObsStack()

    pbar = tqdm(range(n_episodes), desc="  [Eval]", leave=False)
    for _ in pbar:
        obs, _ = env.reset()
        stack.reset()
        done = False; ep_steps = 0; prev_idx = 0; ep_wps = 0
        info = {"waypoint_idx": 0, "num_waypoints": EnvConfig.NUM_WAYPOINTS,
                "collision": False, "goal_reached": False}

        while not done and ep_steps < MAX_EPISODE_STEPS:
            action = policy.predict(stack.push(parse_obs(obs)))
            obs, _, terminated, truncated, info = env.step(action)
            ep_steps += 1
            done = terminated or truncated

            curr_idx = info["waypoint_idx"]
            if curr_idx > prev_idx:
                ep_wps  += curr_idx - prev_idx
            prev_idx = curr_idx

            if info["goal_reached"]: goals += 1; ep_wps = info["num_waypoints"]
            if info["collision"]:    collisions += 1

        wps_reached  += ep_wps
        wps_possible += info["num_waypoints"]

    pbar.close()
    return {
        "goal_success_rate":     goals       / n_episodes,
        "waypoint_success_rate": wps_reached / max(wps_possible, 1),
        "collision_rate":        collisions  / n_episodes,
    }


# ──────────────────────────────────────────────
# Main DAgger loop
# ──────────────────────────────────────────────
def run_dagger(save_dir="dagger_checkpoints"):
    """
    Main loop orchestrating the DAgger framework execution.

    Initializes the agent's environment, policy, and expert. Iteratively requests
    rollouts scaling in challenge via the predefined curriculum while updating the 
    underlying expert beta decay. Saves checkpoints after evaluating progress 
    where performance yields a global network maximum on validation.
    
    Args:
        save_dir (str): Relative directory identifying where to save neural weights.
    """
    os.makedirs(save_dir, exist_ok=True)

    train_env = GroundRobotEnv(render_mode="rgb_array")
    eval_env  = GroundRobotEnv(render_mode="rgb_array")
    expert    = MPCController()
    policy    = PolicyNet().to(DEVICE)

    all_states  = np.empty((0, OBS_DIM), dtype=np.float32)
    all_actions = np.empty((0, ACT_DIM), dtype=np.float32)
    action_mean = np.zeros(ACT_DIM, dtype=np.float32)
    action_std  = np.ones(ACT_DIM,  dtype=np.float32)

    best_policy    = copy.deepcopy(policy)
    best_goal_rate = -1.0
    best_iter      = 0
    beta           = BETA_INIT

    print(f"DAgger v3 — device={DEVICE}, iters={DAGGER_ITERS}")
    print(f"OBS_DIM={OBS_DIM} (stack={STACK_N}×{SINGLE_DIM}), ACT_DIM={ACT_DIM}")
    print(f"Beta: hold 1.0 for {BETA_WARMUP_ITERS} warmup iters, "
          f"then decay ×{BETA_DECAY} unconditionally (floor 0.05)\n")

    for n in range(1, DAGGER_ITERS + 1):

        if n <= BETA_WARMUP_ITERS:
            difficulty = CURRICULUM_START
        else:
            progress   = (n - BETA_WARMUP_ITERS) / (DAGGER_ITERS - BETA_WARMUP_ITERS)
            difficulty = CURRICULUM_START + (CURRICULUM_END - CURRICULUM_START) * min(progress * 2, 1.0)

        print(f"{'='*60}")
        print(f"Iteration {n}/{DAGGER_ITERS}   β={beta:.3f}   difficulty={difficulty:.2f}")

        t0 = time.time()
        new_s, new_a, new_obs_raw = rollout(
            train_env, policy, expert, beta,
            ROLLOUT_STEPS, MAX_EPISODE_STEPS, difficulty,
            action_mean, action_std
        )
        print(f"  Collected {len(new_s)} steps  ({time.time()-t0:.1f}s)")
        print(f"  Student drove ~{(1-beta)*100:.0f}% of steps")

        all_states, all_actions = aggregate_dataset(
            all_states, all_actions, new_s, new_a, new_obs_raw, MAX_DATASET_SIZE)
        print(f"  Dataset: {len(all_states)} samples")

        t0 = time.time()
        policy, action_mean, action_std = train_policy(policy, all_states, all_actions)
        print(f"  Retrained in {time.time()-t0:.1f}s")

        t0 = time.time()
        metrics = evaluate(eval_env, policy, EVAL_EPISODES)
        print(f"  Eval:  goal={metrics['goal_success_rate']:.2f}  "
              f"wp={metrics['waypoint_success_rate']:.2f}  "
              f"collision={metrics['collision_rate']:.2f}  "
              f"({time.time()-t0:.1f}s)")

        if metrics["goal_success_rate"] > best_goal_rate:
            best_goal_rate = metrics["goal_success_rate"]
            best_policy    = copy.deepcopy(policy)
            best_iter      = n
            
            torch.save(best_policy.state_dict(), os.path.join(save_dir, "best_policy.pt"))
            print(f"  ★ New best (iter {n})")

        torch.save(policy.state_dict(), os.path.join(save_dir, f"policy_iter_{n:02d}.pt"))

        # Update beta
        if n >= BETA_WARMUP_ITERS:
            beta = max(beta * BETA_DECAY, 0.05)
            print(f"  β → {beta:.3f}")
        else:
            print(f"  β held at {beta:.3f} (warmup {n}/{BETA_WARMUP_ITERS})")

    train_env.close()
    eval_env.close()
    print(f"\n{'='*60}")
    print(f"Done. Best iter {best_iter}  goal_rate={best_goal_rate:.2f}")
    return best_policy


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def load_policy(checkpoint_path):
    policy = PolicyNet().to(DEVICE)
    ckpt   = torch.load(checkpoint_path, map_location=DEVICE)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return (policy,
            ckpt.get("action_mean", np.zeros(ACT_DIM, dtype=np.float32)),
            ckpt.get("action_std",  np.ones(ACT_DIM,  dtype=np.float32)))


def run_trained_policy(checkpoint_path, n_episodes=5):
    policy, _, _ = load_policy(checkpoint_path)
    env   = GroundRobotEnv(render_mode="human")
    stack = ObsStack()
    for ep in range(n_episodes):
        obs, _ = env.reset(); stack.reset()
        done = False; total_r = 0.0
        while not done:
            action = policy.predict(stack.push(parse_obs(obs)))
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r; done = terminated or truncated
        print(f"Episode {ep+1}: reward={total_r:.1f}")
    env.close()


if __name__ == "__main__":
    run_dagger(save_dir="dagger_checkpoints")
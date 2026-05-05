"""
DAgger: Dataset Aggregation for Ground Robot Waypoint Navigation
Reference: Ross et al., AISTATS 2011

Algorithm (Slide 15):
  1. Initialize D = ∅, policy π₁ (random MLP weights)
  2. For iteration n = 1..N:
     a. Roll out πmix = βn·π* + (1-βn)·πn  (β anneals to 0)
     b. Collect uncertain states Sn
     c. Query MPC expert for labels → Dn = {(s, π*(s)) : s ∈ Sn}
     d. Aggregate D ← D ∪ Dn
     e. Train/Val split (80/20) + early stopping → πn+1
     f. Evaluate πn+1 in pure unassisted simulation
  3. Return best πi by unassisted test performance
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EnvConfig
from env.robot_env import GroundRobotEnv
from mpc.mpc import MPCController

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
DAGGER_ITERS      = 15       # N: number of DAgger outer iterations
ROLLOUT_STEPS     = 500      # steps collected per DAgger iteration (per rollout)
MAX_EPISODE_STEPS = 300      # hard cap on a single episode during rollout
EVAL_EPISODES     = 10       # unassisted evaluation episodes per iteration

# Policy network
HIDDEN_DIM        = 256
NUM_LAYERS        = 3
LEARNING_RATE     = 3e-4
BATCH_SIZE        = 256
MAX_EPOCHS        = 200      # per retraining pass (early stopping kicks in)
PATIENCE          = 15       # early-stopping patience (epochs without val improvement)

# Beta schedule: β_n = β_0 * decay^n  (probability of querying expert vs. learner)
BETA_INIT         = 1.0
BETA_DECAY        = 0.7      # after ~5 iters, learner is mostly driving

# Curriculum: ramp obstacle difficulty over DAgger iterations.
# difficulty ∈ [0.0, 1.0] — passed to env.reset(options={"difficulty": d})
# 0.0 = slow/few obstacles; 1.0 = full config speed/count
CURRICULUM_START  = 0.3      # difficulty at iteration 1
CURRICULUM_END    = 1.0      # difficulty at final iteration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Helper: parse raw gymnasium observation
# ──────────────────────────────────────────────
def parse_obs(obs: np.ndarray):
    """
    Observation layout (from robot_env.py):
        [x, y, theta, v]                    → robot_state  (4)
        [wx, wy]                            → active waypoint (2)
        [px,py,vx,vy,r] * NUM_OBSTACLES     → obstacles (5*N)

    Returns the compact relative-feature vector used as NN input (Slide 16):
        p_rel   = (wx - x, wy - y)          (2)
        v                                   (1)
        Δwk     = norm(p_rel)               (1)
        For each obstacle: (dx, dy, vx, vy, r, dist)  (6*N)
    Total: 4 + 6*N
    """
    x, y, theta, v = obs[0:4]
    wx, wy = obs[4:6]
    obstacles_flat = obs[6:]

    dx_wp = wx - x
    dy_wp = wy - y
    dist_wp = float(np.linalg.norm([dx_wp, dy_wp]))

    obs_features = []
    N = EnvConfig.NUM_OBSTACLES
    for i in range(N):
        base = i * 5
        ox, oy, ovx, ovy, r = obstacles_flat[base:base+5]
        odx = ox - x
        ody = oy - y
        odist = float(np.linalg.norm([odx, ody]))
        obs_features.extend([odx, ody, ovx, ovy, r, odist])

    feature = np.array([dx_wp, dy_wp, v, dist_wp] + obs_features, dtype=np.float32)
    return feature


def obs_to_expert_args(obs: np.ndarray):
    """
    Decompose gymnasium obs into (current_state, w_active, obstacles)
    as expected by MPCController.get_action().
    """
    robot_state = obs[0:4].astype(np.float64)       # [x, y, theta, v]
    w_active    = obs[4:6].astype(np.float64)        # [wx, wy]
    N = EnvConfig.NUM_OBSTACLES
    obstacles   = obs[6:6 + N*5].reshape(N, 5).astype(np.float64)
    return robot_state, w_active, obstacles


OBS_DIM = 4 + 6 * EnvConfig.NUM_OBSTACLES   # feature vector size
ACT_DIM = 2                                  # [omega, a]


# ──────────────────────────────────────────────
# Policy Network (feedforward MLP, Slide 16)
# ──────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)

        # Output bounds (tanh squeeze)
        self.omega_max = float(EnvConfig.OMEGA_MAX)
        self.a_max     = float(EnvConfig.A_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        omega = torch.tanh(raw[:, 0:1]) * self.omega_max
        a     = torch.tanh(raw[:, 1:2]) * self.a_max
        return torch.cat([omega, a], dim=1)

    @torch.no_grad()
    def predict(self, feature: np.ndarray) -> np.ndarray:
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return self.forward(x).squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────
# Training utility (with 80/20 split + early stopping)
# ──────────────────────────────────────────────
def train_policy(policy: PolicyNet,
                 states: np.ndarray,
                 actions: np.ndarray,
                 max_epochs: int = MAX_EPOCHS,
                 patience: int = PATIENCE) -> PolicyNet:
    """
    Train policy on (states, actions) dataset.
    Uses 80/20 train/val split and early stopping.
    Returns the model with best validation loss (Slide 15, step 9).
    """
    X = torch.tensor(states,  dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)

    dataset = TensorDataset(X, Y)
    val_size   = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss  = float('inf')
    best_weights   = copy.deepcopy(policy.state_dict())
    patience_count = 0

    pbar = tqdm(range(max_epochs), desc="  [Train]", leave=False)
    for epoch in pbar:
        # ── train ──
        policy.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(policy(xb), yb)
            loss.backward()
            optimizer.step()

        # ── validate ──
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(policy(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss  = val_loss
            best_weights   = copy.deepcopy(policy.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
                
        pbar.set_postfix({"val_loss": f"{val_loss:.5f}", "best": f"{best_val_loss:.5f}", "pat": f"{patience_count}/{patience}"})
        
    pbar.close()    

    policy.load_state_dict(best_weights)
    print(f"    [Train] best val_loss={best_val_loss:.5f}  (stopped at epoch {epoch+1})")
    return policy


# ──────────────────────────────────────────────
# Rollout: collect (state, expert_action) pairs
# ──────────────────────────────────────────────
def rollout(env: GroundRobotEnv,
            policy: PolicyNet,
            expert: MPCController,
            beta: float,
            n_steps: int,
            max_ep_steps: int,
            difficulty: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute πmix = β·π* + (1-β)·πn for n_steps total steps.
    At every step, regardless of who drove, we query the expert for the label.
    difficulty ∈ [0, 1]: scales obstacle speed range for curriculum training.
    Returns arrays of features and expert actions collected.
    """
    all_features = []
    all_expert_actions = []

    steps_done = 0
    pbar = tqdm(total=n_steps, desc="  [Rollout]", leave=False)
    while steps_done < n_steps:
        obs, _ = env.reset(options={"difficulty": difficulty})
        ep_steps = 0

        while steps_done < n_steps and ep_steps < max_ep_steps:
            feature = parse_obs(obs)

            # ── decide who drives (β-mixing) ──
            if np.random.rand() < beta:
                # Expert drives
                robot_state, w_active, obstacles = obs_to_expert_args(obs)
                action = expert.get_action(robot_state, w_active, obstacles).astype(np.float32)
            else:
                # Learner drives
                action = policy.predict(feature)

            # ── always query expert for the label at this state ──
            robot_state, w_active, obstacles = obs_to_expert_args(obs)
            expert_action = expert.get_action(robot_state, w_active, obstacles).astype(np.float32)

            all_features.append(feature)
            all_expert_actions.append(expert_action)

            obs, _, terminated, truncated, _ = env.step(action)
            steps_done += 1
            ep_steps   += 1
            pbar.update(1)

            if terminated or truncated:
                break
    pbar.close()

    return np.array(all_features, dtype=np.float32), \
           np.array(all_expert_actions, dtype=np.float32)


# ──────────────────────────────────────────────
# Evaluation: pure unassisted simulation (Slide 15, step 10)
# ──────────────────────────────────────────────
def evaluate(env: GroundRobotEnv,
             policy: PolicyNet,
             n_episodes: int = EVAL_EPISODES) -> dict:
    """
    Run the learner policy with zero expert intervention.
    Returns a dict with goal_success_rate, waypoint_success_rate, collision_rate.
    """
    policy.eval()
    goals      = 0
    collisions = 0
    wps_reached_total = 0
    wps_total         = 0

    pbar = tqdm(range(n_episodes), desc="  [Eval]", leave=False)
    for _ in pbar:
        obs, _ = env.reset()
        done   = False
        ep_wps = 0
        ep_steps = 0
        total_wps = EnvConfig.NUM_WAYPOINTS

        while not done and ep_steps < MAX_EPISODE_STEPS:
            feature = parse_obs(obs)
            action  = policy.predict(feature)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_steps += 1
            done = terminated or truncated

            # Count waypoints reached via reward signal
            if reward >= 9.0:    # waypoint reward is +10 (minus time penalty ~0.1)
                ep_wps += 1
            if reward >= 59.0:   # full goal bonus (+50) on top of last waypoint
                goals += 1
            if reward <= -49.0:  # collision penalty is -50
                collisions += 1

        wps_reached_total += ep_wps
        wps_total         += total_wps
        
    pbar.close()

    return {
        "goal_success_rate":     goals      / n_episodes,
        "waypoint_success_rate": wps_reached_total / max(wps_total, 1),
        "collision_rate":        collisions / n_episodes,
    }


# ──────────────────────────────────────────────
# Main DAgger Loop
# ──────────────────────────────────────────────
def run_dagger(save_dir: str = "dagger_checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    # Use "rgb_array" for headless PyBullet (DIRECT mode); "human" opens GUI
    train_env = GroundRobotEnv(render_mode="rgb_array")
    eval_env  = GroundRobotEnv(render_mode="rgb_array")

    expert = MPCController()
    policy = PolicyNet().to(DEVICE)

    # Aggregated dataset
    all_states  = np.empty((0, OBS_DIM),  dtype=np.float32)
    all_actions = np.empty((0, ACT_DIM),  dtype=np.float32)

    best_policy     = copy.deepcopy(policy)
    best_goal_rate  = -1.0
    best_iter       = 0

    print(f"DAgger starting — device={DEVICE}, iters={DAGGER_ITERS}")
    print(f"OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}\n")

    for n in range(1, DAGGER_ITERS + 1):
        beta = BETA_INIT * (BETA_DECAY ** (n - 1))

        # Linear curriculum: ramp difficulty from CURRICULUM_START → CURRICULUM_END
        t = (n - 1) / max(DAGGER_ITERS - 1, 1)
        difficulty = CURRICULUM_START + t * (CURRICULUM_END - CURRICULUM_START)

        print(f"{'='*60}")
        print(f"Iteration {n}/{DAGGER_ITERS}   β={beta:.3f}   difficulty={difficulty:.2f}")

        # ── Step 1: Rollout πmix, collect labels ──
        t0 = time.time()
        new_states, new_actions = rollout(
            train_env, policy, expert, beta,
            n_steps=ROLLOUT_STEPS, max_ep_steps=MAX_EPISODE_STEPS,
            difficulty=difficulty
        )
        print(f"  Collected {len(new_states)} steps  ({time.time()-t0:.1f}s)")

        # ── Step 2: Aggregate ──
        all_states  = np.concatenate([all_states,  new_states],  axis=0)
        all_actions = np.concatenate([all_actions, new_actions], axis=0)
        print(f"  Dataset size: {len(all_states)} samples")

        # ── Step 3: Retrain πn+1 ──
        t0 = time.time()
        policy = train_policy(policy, all_states, all_actions)
        print(f"  Retrained in {time.time()-t0:.1f}s")

        # ── Step 4: Unassisted evaluation ──
        t0 = time.time()
        metrics = evaluate(eval_env, policy, n_episodes=EVAL_EPISODES)
        print(f"  Eval ({EVAL_EPISODES} eps, {time.time()-t0:.1f}s): "
              f"goal={metrics['goal_success_rate']:.2f}  "
              f"wp={metrics['waypoint_success_rate']:.2f}  "
              f"collision={metrics['collision_rate']:.2f}")

        # ── Track best ──
        if metrics["goal_success_rate"] > best_goal_rate:
            best_goal_rate = metrics["goal_success_rate"]
            best_policy    = copy.deepcopy(policy)
            best_iter      = n
            torch.save(best_policy.state_dict(),
                       os.path.join(save_dir, "best_policy.pt"))
            print(f"  ★ New best policy saved (iter {n})")

        # Save checkpoint for every iteration
        torch.save(policy.state_dict(),
                   os.path.join(save_dir, f"policy_iter_{n:02d}.pt"))

    train_env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print(f"DAgger complete. Best policy from iteration {best_iter} "
          f"(goal_rate={best_goal_rate:.2f})")
    print(f"Saved to: {os.path.join(save_dir, 'best_policy.pt')}")
    return best_policy


# ──────────────────────────────────────────────
# Inference helper (use after training)
# ──────────────────────────────────────────────
def load_policy(checkpoint_path: str) -> PolicyNet:
    policy = PolicyNet().to(DEVICE)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    policy.eval()
    return policy


def run_trained_policy(checkpoint_path: str, n_episodes: int = 5):
    """Visualise the trained DAgger policy in the PyBullet GUI."""
    policy = load_policy(checkpoint_path)
    env    = GroundRobotEnv(render_mode="human")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        total_reward = 0.0
        while not done:
            feature = parse_obs(obs)
            action  = policy.predict(feature)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep+1}: total_reward={total_reward:.1f}")

    env.close()


# ──────────────────────────────────────────────
if __name__ == "__main__":
    best = run_dagger(save_dir="dagger_checkpoints")

"""
DAgger: Dataset Aggregation for Ground Robot Waypoint Navigation
Reference: Ross et al., AISTATS 2011

Fixes for 0 goal-rate / high collision-rate training:
  1. ROLLOUT_STEPS: 500 → 2000. More diverse states per iter.
  2. MAX_EPISODE_STEPS: 300 → 600. Give the learner time to actually reach waypoints.
  3. BETA_DECAY: 0.7 → 0.85. Slower annealing — keep expert guiding longer
     so early data quality stays high while dataset grows.
  4. CURRICULUM_START: 0.3 → 0.2. Start even easier so the policy sees
     solvable episodes first and builds a navigation prior before obstacles
     get hard.
  5. Dataset capping at 20k samples with reservoir-style recency bias:
     old data is downsampled so recent (harder) rollouts dominate training.
  6. EVAL capped at MAX_EPISODE_STEPS=600 to match rollout budget.
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
DAGGER_ITERS      = 15
ROLLOUT_STEPS     = 2000     # ↑ was 500 — more state diversity per iter
MAX_EPISODE_STEPS = 600      # ↑ was 300 — 60s at dt=0.1, enough to finish 5 WPs
EVAL_EPISODES     = 10

HIDDEN_DIM        = 256
NUM_LAYERS        = 3
LEARNING_RATE     = 3e-4
BATCH_SIZE        = 256
MAX_EPOCHS        = 200
PATIENCE          = 20       # slightly more patience for larger datasets

BETA_INIT         = 1.0
BETA_DECAY        = 0.85     # ↑ was 0.7 — slower annealing, expert guides longer

CURRICULUM_START  = 0.2      # ↓ was 0.3 — start easier
CURRICULUM_END    = 1.0

# Reservoir cap: once dataset exceeds this, keep only the most recent N samples.
# Prevents old easy-difficulty data from drowning out recent hard-difficulty data.
MAX_DATASET_SIZE  = 20_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
def parse_obs(obs: np.ndarray) -> np.ndarray:
    """
    Transform raw gymnasium obs into robot-local feature vector.
    All spatial quantities are rotated into the robot's body frame so
    the MLP sees LEFT/RIGHT/AHEAD directly rather than global coordinates.

    Output shape: (4 + 6*N_obs,)
        [dx_wp, dy_wp, v, dist_wp,
         dx_o, dy_o, vx_o, vy_o, r_o, dist_o ...]
    """
    x, y, theta, v = obs[0:4]
    wx, wy = obs[4:6]
    obstacles_flat = obs[6:]

    c, s = np.cos(theta), np.sin(theta)

    def to_local(dx, dy):
        return c * dx + s * dy, -s * dx + c * dy

    dx_wp, dy_wp = to_local(wx - x, wy - y)
    dist_wp = float(np.linalg.norm([dx_wp, dy_wp]))

    obs_features = []
    N = int(len(obstacles_flat) / 5)
    for i in range(N):
        base = i * 5
        ox, oy, ovx, ovy, r = obstacles_flat[base:base + 5]
        odx, ody         = to_local(ox - x, oy - y)
        ovx_loc, ovy_loc = to_local(ovx, ovy)
        odist = float(np.linalg.norm([odx, ody]))
        obs_features.extend([odx, ody, ovx_loc, ovy_loc, r, odist])

    return np.array([dx_wp, dy_wp, v, dist_wp] + obs_features, dtype=np.float32)


def obs_to_expert_args(obs: np.ndarray):
    robot_state = obs[0:4].astype(np.float64)
    w_active    = obs[4:6].astype(np.float64)
    N = EnvConfig.NUM_OBSTACLES
    obstacles   = obs[6:6 + N * 5].reshape(N, 5).astype(np.float64)
    return robot_state, w_active, obstacles


OBS_DIM = 4 + 6 * EnvConfig.NUM_OBSTACLES
ACT_DIM = 2


# ──────────────────────────────────────────────
# Policy Network
# ──────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)
        self.omega_max = float(EnvConfig.OMEGA_MAX)
        self.a_max     = float(EnvConfig.A_MAX)

    def forward(self, x):
        raw   = self.net(x)
        omega = torch.tanh(raw[:, 0:1]) * self.omega_max
        a     = torch.tanh(raw[:, 1:2]) * self.a_max
        return torch.cat([omega, a], dim=1)

    @torch.no_grad()
    def predict(self, feature: np.ndarray) -> np.ndarray:
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return self.forward(x).squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────
# Dataset management
# ──────────────────────────────────────────────
def aggregate_dataset(all_states, all_actions, new_states, new_actions, max_size):
    """
    Append new data and trim to max_size by dropping the OLDEST samples.
    This gives recent (harder) rollouts more influence over training,
    preventing easy early-iteration data from dominating.
    """
    all_states  = np.concatenate([all_states,  new_states],  axis=0)
    all_actions = np.concatenate([all_actions, new_actions], axis=0)
    if len(all_states) > max_size:
        all_states  = all_states[-max_size:]
        all_actions = all_actions[-max_size:]
    return all_states, all_actions


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_policy(policy, states, actions, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    X = torch.tensor(states,  dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)

    dataset    = TensorDataset(X, Y)
    val_size   = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=0, pin_memory=True)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss  = float('inf')
    best_weights   = copy.deepcopy(policy.state_dict())
    patience_count = 0

    pbar = tqdm(range(max_epochs), desc="  [Train]", leave=False)
    for epoch in pbar:
        policy.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(policy(xb), yb).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(policy(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss  = val_loss
            best_weights   = copy.deepcopy(policy.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        pbar.set_postfix({"val": f"{val_loss:.5f}",
                          "best": f"{best_val_loss:.5f}",
                          "pat": f"{patience_count}/{patience}"})
    pbar.close()

    policy.load_state_dict(best_weights)
    print(f"    [Train] best val_loss={best_val_loss:.5f}  (epoch {epoch + 1})")
    return policy


# ──────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────
def rollout(env, policy, expert, beta, n_steps, max_ep_steps, difficulty=1.0):
    """
    Execute pi_mix = beta*pi* + (1-beta)*pi_n.
    One expert call per step (reused as label regardless of who drives).
    """
    all_features, all_expert_actions = [], []
    steps_done = 0
    pbar = tqdm(total=n_steps, desc="  [Rollout]", leave=False)

    while steps_done < n_steps:
        obs, _ = env.reset(options={"difficulty": difficulty})
        ep_steps = 0

        while steps_done < n_steps and ep_steps < max_ep_steps:
            feature = parse_obs(obs)
            robot_state, w_active, obstacles = obs_to_expert_args(obs)

            expert_action = expert.get_action(robot_state, w_active, obstacles).astype(np.float32)
            action = expert_action if np.random.rand() < beta else policy.predict(feature)

            obs, _, terminated, truncated, info = env.step(action)

            all_features.append(feature)
            all_expert_actions.append(expert_action)
                
            steps_done += 1
            ep_steps   += 1
            pbar.update(1)

            if terminated or truncated:
                break

    pbar.close()
    return (np.array(all_features,       dtype=np.float32),
            np.array(all_expert_actions, dtype=np.float32))


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(env, policy, n_episodes=EVAL_EPISODES):
    """
    Pure unassisted evaluation. Reads progress from info dict, not reward thresholds.
    """
    policy.eval()
    goals = collisions = wps_reached = wps_possible = 0

    pbar = tqdm(range(n_episodes), desc="  [Eval]", leave=False)
    for _ in pbar:
        obs, _   = env.reset()
        done     = False
        ep_steps = 0
        prev_idx = 0
        ep_wps   = 0
        info     = {"waypoint_idx": 0,
                    "num_waypoints": EnvConfig.NUM_WAYPOINTS,
                    "collision": False,
                    "goal_reached": False}

        while not done and ep_steps < MAX_EPISODE_STEPS:
            action = policy.predict(parse_obs(obs))
            obs, _, terminated, truncated, info = env.step(action)
            ep_steps += 1
            done = terminated or truncated

            curr_idx = info["waypoint_idx"]
            if curr_idx > prev_idx:
                ep_wps += curr_idx - prev_idx
            prev_idx = curr_idx

            if info["goal_reached"]:
                goals  += 1
                ep_wps  = info["num_waypoints"]
            if info["collision"]:
                collisions += 1

        wps_reached  += ep_wps
        wps_possible += info["num_waypoints"]

    pbar.close()
    return {
        "goal_success_rate":     goals      / n_episodes,
        "waypoint_success_rate": wps_reached / max(wps_possible, 1),
        "collision_rate":        collisions  / n_episodes,
    }


# ──────────────────────────────────────────────
# Main DAgger loop
# ──────────────────────────────────────────────
def run_dagger(save_dir="dagger_checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    train_env = GroundRobotEnv(render_mode="rgb_array")
    eval_env  = GroundRobotEnv(render_mode="rgb_array")
    expert    = MPCController()
    policy    = PolicyNet().to(DEVICE)

    all_states  = np.empty((0, OBS_DIM), dtype=np.float32)
    all_actions = np.empty((0, ACT_DIM), dtype=np.float32)

    best_policy    = copy.deepcopy(policy)
    best_goal_rate = -1.0
    best_iter      = 0

    print(f"DAgger starting — device={DEVICE}, iters={DAGGER_ITERS}")
    print(f"OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}, max_dataset={MAX_DATASET_SIZE}\n")

    for n in range(1, DAGGER_ITERS + 1):
        beta       = BETA_INIT * (BETA_DECAY ** (n - 1))
        difficulty = 1.0

        print(f"{'='*60}")
        print(f"Iteration {n}/{DAGGER_ITERS}   β={beta:.3f}   difficulty={difficulty:.2f}")

        t0 = time.time()
        new_s, new_a = rollout(train_env, policy, expert, beta,
                               ROLLOUT_STEPS, MAX_EPISODE_STEPS, difficulty)
        print(f"  Collected {len(new_s)} steps  ({time.time()-t0:.1f}s)")

        all_states, all_actions = aggregate_dataset(
            all_states, all_actions, new_s, new_a, MAX_DATASET_SIZE)
        print(f"  Dataset size: {len(all_states)} samples")

        t0 = time.time()
        policy = train_policy(policy, all_states, all_actions)
        print(f"  Retrained in {time.time()-t0:.1f}s")

        t0 = time.time()
        metrics = evaluate(eval_env, policy, EVAL_EPISODES)
        print(f"  Eval ({EVAL_EPISODES} eps, {time.time()-t0:.1f}s):  "
              f"goal={metrics['goal_success_rate']:.2f}  "
              f"wp={metrics['waypoint_success_rate']:.2f}  "
              f"collision={metrics['collision_rate']:.2f}")

        if metrics["goal_success_rate"] > best_goal_rate:
            best_goal_rate = metrics["goal_success_rate"]
            best_policy    = copy.deepcopy(policy)
            best_iter      = n
            torch.save(best_policy.state_dict(), os.path.join(save_dir, "best_policy.pt"))
            print(f"  ★ New best policy saved (iter {n})")

        torch.save(policy.state_dict(), os.path.join(save_dir, f"policy_iter_{n:02d}.pt"))

    train_env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print(f"DAgger complete.  Best from iter {best_iter}  (goal_rate={best_goal_rate:.2f})")
    return best_policy


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def load_policy(checkpoint_path):
    policy = PolicyNet().to(DEVICE)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    policy.eval()
    return policy


def run_trained_policy(checkpoint_path, n_episodes=5):
    """Visualise the trained DAgger policy in the PyBullet GUI."""
    policy = load_policy(checkpoint_path)
    env    = GroundRobotEnv(render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, total_r = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(policy.predict(parse_obs(obs)))
            total_r += r
            done = terminated or truncated
        print(f"Episode {ep+1}: reward={total_r:.1f}")
    env.close()


if __name__ == "__main__":
    run_dagger(save_dir="dagger_checkpoints")
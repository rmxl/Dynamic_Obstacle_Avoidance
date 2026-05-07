import time
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from config import EnvConfig
from env.robot_env import GroundRobotEnv
from mpc.mpc import MPCController

class Evaluator:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.reset_episode()
        self.all_episodes_data = []

    def reset_episode(self, start_pos=None):
        self.episode_data = {
            "path_length": 0.0,
            "waypoints_reached": 0,
            "goal_success": False,
            "collision": False,
            "min_clearance": float('inf'),
            "inference_times_ms": [],
            "time_to_goal_s": 0.0
        }
        self.last_pos = start_pos

    def step_update(self, pos, current_wp_idx, num_waypoints, obstacles, r_robot, terminated, reward, inference_time_ms, info):
        if self.last_pos is not None:
            self.episode_data["path_length"] += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos.copy()
        
        self.episode_data["waypoints_reached"] = int(current_wp_idx)
        
        # Use the environment's ground truth info dict instead of guessing from reward
        if info.get("goal_reached", False):
            self.episode_data["goal_success"] = True
            self.episode_data["waypoints_reached"] = int(num_waypoints)

        if info.get("collision", False):
            self.episode_data["collision"] = True

        for obs in obstacles:
            obs_pos = obs[0:2]
            r_obs = float(obs[4])
            dist = float(np.linalg.norm(pos - obs_pos))
            clearance = dist - float(r_robot) - r_obs
            
            if clearance >= 0.0 and clearance < self.episode_data["min_clearance"]:
                self.episode_data["min_clearance"] = clearance
                
        self.episode_data["inference_times_ms"].append(float(inference_time_ms))
        self.episode_data["time_to_goal_s"] += EnvConfig.DT

    def finish_episode(self):
        avg_inf = float(np.mean(self.episode_data["inference_times_ms"])) if self.episode_data["inference_times_ms"] else 0.0
        self.episode_data["avg_inference_time_ms"] = avg_inf
        
        if self.episode_data["min_clearance"] == float('inf'):
            self.episode_data["min_clearance"] = 999.0
            
        self.all_episodes_data.append(self.episode_data.copy())
        
    def save_results(self, filename="evaluation_metrics.json"):
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(self.all_episodes_data, f, indent=4)
        print(f"Saved evaluation metrics to {path}")

def compute_aggregate_metrics(json_file_path):
    if not os.path.exists(json_file_path):
        return None
        
    with open(json_file_path, "r") as f:
        data = json.load(f)
        
    if len(data) == 0:
        return None
        
    total_episodes = len(data)
    num_waypoints = EnvConfig.NUM_WAYPOINTS
    
    total_waypoints_possible = total_episodes * num_waypoints
    total_waypoints_reached = sum([ep["waypoints_reached"] for ep in data])
    waypoint_success_rate = total_waypoints_reached / total_waypoints_possible
    
    goal_successes = sum([1 for ep in data if ep.get("goal_success", False)])
    goal_success_rate = goal_successes / total_episodes
    
    collisions = sum([1 for ep in data if ep.get("collision", False)])
    collision_rate = collisions / total_episodes
    
    timeouts = sum([1 for ep in data if not ep.get("goal_success", False) and not ep.get("collision", False)])
    timeout_rate = timeouts / total_episodes
    
    # Only average metrics across actual successful runs to prevent crashing runs from skewing it
    successful_episodes = [ep for ep in data if ep.get("goal_success", False)]
    
    if successful_episodes:
        avg_path_length = np.mean([ep["path_length"] for ep in successful_episodes])
        avg_time_to_goal_s = np.mean([ep["time_to_goal_s"] for ep in successful_episodes])
    else:
        avg_path_length = 0.0
        avg_time_to_goal_s = 0.0
    
    clearances = [ep["min_clearance"] if ep["min_clearance"] != 999.0 else float('inf') for ep in data]
    min_clearance = min(clearances) if clearances else float('inf')
    
    all_inference_times = []
    for ep in data:
        all_inference_times.extend(ep.get("inference_times_ms", []))
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    
    return {
        "waypoint_success_rate": waypoint_success_rate * 100.0,
        "goal_success_rate": goal_success_rate * 100.0,
        "collision_rate": collision_rate * 100.0,
        "timeout_rate": timeout_rate * 100.0,
        "avg_path_length": avg_path_length,
        "avg_time_to_goal_s": avg_time_to_goal_s,
        "min_clearance": min_clearance,
        "avg_inference_time_ms": avg_inference_time,
        "num_episodes": total_episodes
    }

# Try to import DAgger / BC policy
try:
    from bc.dagger import PolicyNet, parse_obs, ObsStack
    import torch
    DAGGER_AVAILABLE = True
except ImportError:
    DAGGER_AVAILABLE = False


def evaluate_method(method, num_episodes, model_path=None, headless=True, max_steps=500):
    print(f"Evaluating {method.upper()} over {num_episodes} episodes...")
    
    env = GroundRobotEnv(render_mode="rgb_array" if headless else "human")
    
    save_dir = os.path.join("results", method)
    evaluator = Evaluator(save_dir=save_dir)
    
    agent = None
    
    if method == "mpc":
        agent = MPCController(dt=EnvConfig.DT, horizon=10, r_robot=EnvConfig.ROBOT_RADIUS)
        obs_stack = None
    elif method in ["dagger", "bc"]:
        if not DAGGER_AVAILABLE:
            raise ImportError("Could not import DAgger policy dependencies.")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Provide --model_path.")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = PolicyNet().to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()
        obs_stack = ObsStack()
    else:
        raise NotImplementedError(f"Evaluation for method '{method}' is not implemented yet.")
        
    pbar = tqdm(range(num_episodes), desc=f"Evaluating {method.upper()}")
    
    for ep in pbar:
        obs, _ = env.reset()
        evaluator.reset_episode(start_pos=env.robot_state[0:2])
        if obs_stack is not None:
            obs_stack.reset()
        
        for step in range(max_steps):
            step_start_time = time.time()
            
            # Predict Action
            if method == "mpc":
                robot_state = obs[0:4]
                w_active = obs[4:6]
                obstacles = obs[6:].reshape((EnvConfig.NUM_OBSTACLES, 5))
                action = agent.get_action(robot_state, w_active, obstacles)
            elif method in ["dagger", "bc"]:
                feature = parse_obs(obs)
                stacked_feature = obs_stack.push(feature)
                action = agent.predict(stacked_feature)
                obstacles = obs[6:].reshape((EnvConfig.NUM_OBSTACLES, 5)) # for evaluator
            else:
                action = np.zeros(2)
                
            inference_time_ms = (time.time() - step_start_time) * 1000.0
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update Evaluator
            evaluator.step_update(
                pos=env.robot_state[0:2],
                current_wp_idx=env.current_waypoint_idx,
                num_waypoints=EnvConfig.NUM_WAYPOINTS,
                obstacles=obstacles,
                r_robot=env.r_robot,
                terminated=(terminated or truncated),
                reward=reward,
                inference_time_ms=inference_time_ms,
                info=info
            )
            
            if terminated or truncated or step == max_steps - 1:
                evaluator.finish_episode()
                break
                
    env.close()
    
    # Save the episode JSON
    evaluator.save_results()
    print("\nEvaluation Complete.")
    
    # Compute and print aggregate metrics directly
    json_path = os.path.join(save_dir, "evaluation_metrics.json")
    metrics = compute_aggregate_metrics(json_path)
    
    if metrics:
        print("\n" + "="*80)
        print(f" AGGREGATE RESULTS: {method.upper()} ({num_episodes} Episodes)")
        print("="*80)
        print(f"  Goal Success Rate:      {metrics['goal_success_rate']:.2f}%")
        print(f"  Waypoint Success Rate:  {metrics['waypoint_success_rate']:.2f}%")
        print(f"  Collision Rate:         {metrics['collision_rate']:.2f}%")
        print(f"  Timeout Rate:           {metrics['timeout_rate']:.2f}%")
        print(f"  Average Time to Goal:   {metrics['avg_time_to_goal_s']:.2f}s")
        print(f"  Average Path Length:    {metrics['avg_path_length']:.2f}m")
        print(f"  Minimum Clearance:      {metrics['min_clearance']:.4f}m")
        print(f"  Avg Inference Time:     {metrics['avg_inference_time_ms']:.2f}ms/step")
        print("="*80 + "\n")

    # Save aggregate metrics to a separate JSON file using method name
    aggregate_metrics_path = os.path.join(save_dir, f"aggregate_metrics_{method}.json")
    with open(aggregate_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved aggregate metrics to {aggregate_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Navigation Algorithm over many episodes")
    parser.add_argument("--method", type=str, default="mpc", choices=["mpc", "irl", "bc", "dagger"],
                        help="Algorithm to evaluate")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI (not recommended for >10 episodes)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained policy .pt file (for dagger/bc)")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode before truncation")
    
    args = parser.parse_args()
    evaluate_method(
        method=args.method, 
        num_episodes=args.episodes, 
        model_path=args.model_path, 
        headless=not args.gui,
        max_steps=args.max_steps
    )
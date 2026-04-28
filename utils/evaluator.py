import numpy as np
import json
import os

class Evaluator:
    """
    Evaluator to track and save navigation metrics according to the slides:
    - Waypoint success rate
    - Goal success rate
    - Path length vs optimal
    - Collision rate
    - Minimum clearance
    - Inference time (ms per step)
    - Time to goal (episode duration)
    """
    def __init__(self, save_dir="utils"):
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

    def step_update(self, pos, current_wp_idx, num_waypoints, obstacles, r_robot, terminated, reward, inference_time_ms):
        # 1. Path Length
        if self.last_pos is not None:
            self.episode_data["path_length"] += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos.copy()
        
        # 2. Waypoints
        self.episode_data["waypoints_reached"] = int(current_wp_idx)
        if terminated and reward >= 40.0:  # Goal success defined by high reward upon reaching last wp
            self.episode_data["goal_success"] = True
            
        # 3. Collision Rate (defined by massive penalty)
        if terminated and reward <= -40.0:
            self.episode_data["collision"] = True

        # 4. Minimum Clearance (dist - r_safe = dist - r_robot - r_obs)
        for obs in obstacles:
            obs_pos = obs[0:2]
            r_obs = float(obs[4])
            dist = float(np.linalg.norm(pos - obs_pos))
            clearance = dist - float(r_robot) - r_obs
            if clearance < self.episode_data["min_clearance"]:
                self.episode_data["min_clearance"] = clearance
                
        # 5. Inference Times
        self.episode_data["inference_times_ms"].append(float(inference_time_ms))
        
        # 6. Time to target
        self.episode_data["time_to_goal_s"] += 0.1 # assuming dt=0.1

    def finish_episode(self):
        avg_inf = float(np.mean(self.episode_data["inference_times_ms"])) if self.episode_data["inference_times_ms"] else 0.0
        self.episode_data["avg_inference_time_ms"] = avg_inf
        
        # Sanitize infinity if no minimum clearance was calculated
        if self.episode_data["min_clearance"] == float('inf'):
            self.episode_data["min_clearance"] = 999.0
            
        self.all_episodes_data.append(self.episode_data.copy())
        
    def save_results(self, filename="evaluation_metrics.json"):
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(self.all_episodes_data, f, indent=4)
        print(f"Saved evaluation metrics to {path}")

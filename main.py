import time
import numpy as np
from env.robot_env import GroundRobotEnv
from mpc.mpc import MPCController
from utils.evaluator import Evaluator

def run_mpc_episode(env, mpc, evaluator, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0
    
    print(f"Starting MPC Run. Initial Robot State: {env.robot_state}")
    start_time = time.time()
    
    evaluator.reset_episode(start_pos=env.robot_state[0:2])
    
    for step in range(max_steps):
        # Extract components from observation
        robot_state = obs[0:4] # [x, y, theta, v]
        w_active = obs[4:6]    # [wx, wy]
        
        # Reshape obstacles back to (N, 5)
        num_obstacles = env.num_obstacles
        obstacles_flat = obs[6:]
        obstacles = obstacles_flat.reshape((num_obstacles, 5))
        
        # 1. Fetch expert action from MPC optimizer
        step_start_time = time.time()
        action = mpc.get_action(robot_state, w_active, obstacles)
        inference_time_ms = (time.time() - step_start_time) * 1000.0
        
        # 2. Apply action to environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 3. Update Metrics Evaluator
        evaluator.step_update(
            pos=robot_state[0:2],
            current_wp_idx=env.current_waypoint_idx,
            num_waypoints=env.num_waypoints if hasattr(env, "num_waypoints") else 3,
            obstacles=obstacles,
            r_robot=env.r_robot,
            terminated=terminated,
            reward=reward,
            inference_time_ms=inference_time_ms
        )
        
        if step % 20 == 0:
            print(f"Step {step}: pos=({robot_state[0]:.2f}, {robot_state[1]:.2f}), "
                  f"action=omega:{action[0]:.2f}, a:{action[1]:.2f}, "
                  f"dist_to_wp={np.linalg.norm(robot_state[:2] - w_active):.2f}")
            
        if terminated or truncated:
            print(f"Episode finished at step {step}. Total reward: {total_reward:.2f}")
            evaluator.finish_episode()
            break
            
    print(f"MPC Execution Time: {(time.time() - start_time):.2f} seconds")
    return total_reward

if __name__ == "__main__":
    from config import EnvConfig
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run Ground Robot Navigation")
    parser.add_argument("--method", type=str, default="mpc", choices=["mpc", "irl", "bc", "dagger"],
                        help="The method to run: mpc, irl, bc, or dagger")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--headless", action="store_true", help="Run strictly headless without PyBullet GUI")
    args = parser.parse_args()
    
    test_env = GroundRobotEnv(render_mode="rgb_array" if args.headless else "human")
    
    # Save results in a separate dynamically named folder
    save_dir = os.path.join("results", args.method)
    evaluator = Evaluator(save_dir=save_dir)
    
    if args.method == "mpc":
        # Initialize MPC with dt and Horizon
        mpc_controller = MPCController(dt=EnvConfig.DT, horizon=10, r_robot=EnvConfig.ROBOT_RADIUS)
        
        # Run test episodes
        for run in range(args.episodes):
            print(f"\n--- Episode {run+1}/{args.episodes} ---")
            run_mpc_episode(test_env, mpc_controller, evaluator)
            
    else:
        print(f"Method '{args.method}' is not fully implemented yet in main.py!")
        
    evaluator.save_results()


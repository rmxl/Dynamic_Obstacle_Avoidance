import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from config import EnvConfig

class GroundRobotEnv(gym.Env):
    """
    PyBullet Environment for Ground Robot Waypoint Navigation with Dynamic Obstacles
    State: [x, y, theta, v]
    Action: [omega, a] 
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": EnvConfig.RENDER_FPS}
    
    def __init__(self, render_mode="human"):
        super().__init__()
        
        self.dt = EnvConfig.DT
        self.r_robot = EnvConfig.ROBOT_RADIUS
        
        # Action limits
        self.omega_max = EnvConfig.OMEGA_MAX
        self.a_max = EnvConfig.A_MAX
        self.v_min = EnvConfig.V_MIN
        self.v_max = EnvConfig.V_MAX
        
        # Scenario settings
        self.waypoint_tolerance = EnvConfig.WAYPOINT_TOLERANCE
        self.num_obstacles = EnvConfig.NUM_OBSTACLES
        
        # Continuous action space: [omega, a]
        self.action_space = spaces.Box(
            low=np.array([-self.omega_max, -self.a_max], dtype=np.float32),
            high=np.array([self.omega_max, self.a_max], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: 
        # Robot state [x, y, theta, v] (4)
        # Current waypoint [wx, wy] (2)
        # Obstacles [px, py, vx, vy, r] * num_obstacles (5 * n)
        obs_dim = 4 + 2 + self.num_obstacles * 5
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.client = None
        self.robot_id = None
        self.obstacle_ids = []
        self.waypoint_ids = []
        
        # Connect to PyBullet
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
    def _create_cylinder(self, radius, height, mass, color, pos):
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        body_id = p.createMultiBody(baseMass=mass,
                                    baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=pos)
        return body_id
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # difficulty ∈ [0, 1] scales obstacle speed range linearly
        difficulty = 1.0
        if options is not None:
            difficulty = float(np.clip(options.get("difficulty", 1.0), 0.0, 1.0))
        obs_speed_min = EnvConfig.OBS_SPEED_RANGE[0] * difficulty
        obs_speed_max = EnvConfig.OBS_SPEED_RANGE[1] * difficulty

        if self.client is None:
            return self._get_obs(), {}
            
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Initialize randomized robot state
        start_x = np.random.uniform(1.0, 3.0)
        start_y = np.random.uniform(1.0, 3.0)
        start_theta = np.random.uniform(-np.pi, np.pi)
        self.robot_state = np.array([start_x, start_y, start_theta, 0.0], dtype=np.float32)
        
        self.robot_id = self._create_cylinder(self.r_robot, 0.2, 10.0, [0, 0, 1, 1], [start_x, start_y, 0.1])
        
        # Initialize randomized waypoints
        self.waypoints = []
        for _ in range(EnvConfig.NUM_WAYPOINTS):
            wx = np.random.uniform(EnvConfig.WORKSPACE_X_BOUNDS[0] + 2.0, EnvConfig.WORKSPACE_X_BOUNDS[1] - 2.0)
            wy = np.random.uniform(EnvConfig.WORKSPACE_Y_BOUNDS[0] + 2.0, EnvConfig.WORKSPACE_Y_BOUNDS[1] - 2.0)
            self.waypoints.append(np.array([wx, wy]))
            
        self.current_waypoint_idx = 0
        
        self.waypoint_ids = []
        for i, wp in enumerate(self.waypoints):
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=self.waypoint_tolerance, rgbaColor=[0, 1, 0, 0.3])
            wp_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=[wp[0], wp[1], 0.1])
            self.waypoint_ids.append(wp_id)
            
            # Number the waypoint
            p.addUserDebugText(str(i + 1), [wp[0], wp[1], 0.5], textColorRGB=[0, 0, 0], textSize=1.0)
            
        # Draw Workspace Boundaries
        xb = EnvConfig.WORKSPACE_X_BOUNDS
        yb = EnvConfig.WORKSPACE_Y_BOUNDS
        corners = [
            [xb[0], yb[0], 0.05],
            [xb[1], yb[0], 0.05],
            [xb[1], yb[1], 0.05],
            [xb[0], yb[1], 0.05]
        ]
        for i in range(4):
            p.addUserDebugLine(corners[i], corners[(i+1)%4], lineColorRGB=[0.8, 0, 0.8], lineWidth=3.0)
            
            
        # Initialize randomized dynamic obstacles [px, py, vx, vy, radius]
        self.obstacles = np.zeros((self.num_obstacles, 5), dtype=np.float32)
        self.obstacle_ids = []
        
        for i in range(self.num_obstacles):
            # Ensure obstacles don't spawn exactly on the robot
            valid_spawn = False
            while not valid_spawn:
                ox = np.random.uniform(EnvConfig.WORKSPACE_X_BOUNDS[0], EnvConfig.WORKSPACE_X_BOUNDS[1])
                oy = np.random.uniform(EnvConfig.WORKSPACE_Y_BOUNDS[0], EnvConfig.WORKSPACE_Y_BOUNDS[1])
                if np.linalg.norm(np.array([ox, oy]) - np.array([start_x, start_y])) > 3.0:
                    valid_spawn = True
                    
            r = np.random.uniform(EnvConfig.OBS_RADIUS_RANGE[0], EnvConfig.OBS_RADIUS_RANGE[1])
            vx = np.random.uniform(obs_speed_min, obs_speed_max)
            vy = np.random.uniform(obs_speed_min, obs_speed_max)
            
            self.obstacles[i] = [ox, oy, vx, vy, r]
            obs_id = self._create_cylinder(r, 0.2, 10.0, [1, 0, 0, 1], [ox, oy, 0.1])
            self.obstacle_ids.append(obs_id)
            
        # Set camera to get a top-down view centered on workspace
        center_x = (EnvConfig.WORKSPACE_X_BOUNDS[1] - EnvConfig.WORKSPACE_X_BOUNDS[0]) / 2
        center_y = (EnvConfig.WORKSPACE_Y_BOUNDS[1] - EnvConfig.WORKSPACE_Y_BOUNDS[0]) / 2
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[center_x, center_y, 0])
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.concatenate([
            self.robot_state,
            self.waypoints[self.current_waypoint_idx],
            self.obstacles.flatten()
        ])
        return obs.astype(np.float32)
        
    def step(self, action):
        omega, a = action
        
        # Clip actions to stay within bounds
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        a = np.clip(a, -self.a_max, self.a_max)
        
        # 1. Update Robot State
        x, y, theta, v = self.robot_state
        v_new = np.clip(v + a * self.dt, self.v_min, self.v_max)
        theta_new = theta + omega * self.dt
        x_new = x + v_new * np.cos(theta_new) * self.dt
        y_new = y + v_new * np.sin(theta_new) * self.dt
        
        self.robot_state = np.array([x_new, y_new, theta_new, v_new], dtype=np.float32)
        
        if self.client is not None:
            quat = p.getQuaternionFromEuler([0, 0, theta_new])
            p.resetBasePositionAndOrientation(self.robot_id, [x_new, y_new, 0.1], quat)
            
            # 2. Update Obstacles
            for i, obs in enumerate(self.obstacles):
                # Randomly change velocity for robust unpredictable obstacles
                if np.random.rand() < EnvConfig.OBS_DIR_CHANGE_PROB:
                    obs[2] = np.random.uniform(EnvConfig.OBS_SPEED_RANGE[0], EnvConfig.OBS_SPEED_RANGE[1])
                    obs[3] = np.random.uniform(EnvConfig.OBS_SPEED_RANGE[0], EnvConfig.OBS_SPEED_RANGE[1])
                    
                obs[0] += obs[2] * self.dt
                obs[1] += obs[3] * self.dt
                
                # Bounce off walls
                if obs[0] - obs[4] <= EnvConfig.WORKSPACE_X_BOUNDS[0] or obs[0] + obs[4] >= EnvConfig.WORKSPACE_X_BOUNDS[1]:
                    obs[2] *= -1
                    obs[0] = np.clip(obs[0], EnvConfig.WORKSPACE_X_BOUNDS[0] + obs[4], EnvConfig.WORKSPACE_X_BOUNDS[1] - obs[4])
                if obs[1] - obs[4] <= EnvConfig.WORKSPACE_Y_BOUNDS[0] or obs[1] + obs[4] >= EnvConfig.WORKSPACE_Y_BOUNDS[1]:
                    obs[3] *= -1
                    obs[1] = np.clip(obs[1], EnvConfig.WORKSPACE_Y_BOUNDS[0] + obs[4], EnvConfig.WORKSPACE_Y_BOUNDS[1] - obs[4])
                    
                p.resetBasePositionAndOrientation(self.obstacle_ids[i], [obs[0], obs[1], 0.1], [0, 0, 0, 1])
                
            p.stepSimulation()
            
            if self.render_mode == "human":
                time.sleep(self.dt)
        else:
            # 2. Minimal Update Obstacles
            for obs in self.obstacles:
                # Randomly change velocity (5% chance) for robust unpredictable obstacles
                if np.random.rand() < 0.05:
                    obs[2] = np.random.uniform(EnvConfig.OBS_SPEED_RANGE[0], EnvConfig.OBS_SPEED_RANGE[1])
                    obs[3] = np.random.uniform(EnvConfig.OBS_SPEED_RANGE[0], EnvConfig.OBS_SPEED_RANGE[1])

                obs[0] += obs[2] * self.dt
                obs[1] += obs[3] * self.dt
            
                # Bounce off walls
                if obs[0] - obs[4] <= EnvConfig.WORKSPACE_X_BOUNDS[0] or obs[0] + obs[4] >= EnvConfig.WORKSPACE_X_BOUNDS[1]:
                    obs[2] *= -1
                    obs[0] = np.clip(obs[0], EnvConfig.WORKSPACE_X_BOUNDS[0] + obs[4], EnvConfig.WORKSPACE_X_BOUNDS[1] - obs[4])
                if obs[1] - obs[4] <= EnvConfig.WORKSPACE_Y_BOUNDS[0] or obs[1] + obs[4] >= EnvConfig.WORKSPACE_Y_BOUNDS[1]:
                    obs[3] *= -1
                    obs[1] = np.clip(obs[1], EnvConfig.WORKSPACE_Y_BOUNDS[0] + obs[4], EnvConfig.WORKSPACE_Y_BOUNDS[1] - obs[4])
        
        # 3. Check Waypoint Reaching
        robot_pos = self.robot_state[0:2]
        current_wp = self.waypoints[self.current_waypoint_idx]
        dist_to_wp = np.linalg.norm(robot_pos - current_wp)
        
        reward = -0.1 # Time penalty
        terminated = False
        truncated = False
        
        goal_reached = False
        has_collided = False
        out_of_bounds = False
        
        if dist_to_wp <= self.waypoint_tolerance:
            if self.client is not None:
                p.resetBasePositionAndOrientation(self.waypoint_ids[self.current_waypoint_idx], [0, 0, -10], [0, 0, 0, 1])
                
            reward += 10.0 # Reward for reaching waypoint
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
            else:
                reward += 50.0 # Reached all waypoints
                terminated = True
                goal_reached = True
                
        # 4. Check Collisions
        if self.client is not None:
            for obs_id in self.obstacle_ids:
                contacts = p.getContactPoints(self.robot_id, obs_id)
                if len(contacts) > 0:
                    reward -= 50.0
                    terminated = True
                    has_collided = True
                    break
        else:
            for obs in self.obstacles:
                obs_pos = obs[0:2]
                obs_r = obs[4]
                dist_to_obs = np.linalg.norm(robot_pos - obs_pos)
                if dist_to_obs <= (self.r_robot + obs_r):
                    reward -= 50.0
                    terminated = True
                    has_collided = True
                    break
                    
        # 5. Check out of bounds
        if robot_pos[0] < EnvConfig.WORKSPACE_X_BOUNDS[0] or robot_pos[0] > EnvConfig.WORKSPACE_X_BOUNDS[1] or \
           robot_pos[1] < EnvConfig.WORKSPACE_Y_BOUNDS[0] or robot_pos[1] > EnvConfig.WORKSPACE_Y_BOUNDS[1]:
            terminated = True
            out_of_bounds = True
            reward -= 50.0
                    
        # Dist to target penalty
        dist_to_target = np.linalg.norm(robot_pos - self.waypoints[self.current_waypoint_idx])
        reward -= 0.01 * dist_to_target
        
        # Determine explicit success: Can't have a goal success if there's a collision on the same frame.
        actual_goal_reached = goal_reached and not has_collided and not out_of_bounds
        
        info = {
            "waypoint_idx": self.current_waypoint_idx,
            "num_waypoints": len(self.waypoints),
            "collision": has_collided,
            "out_of_bounds": out_of_bounds,
            "goal_reached": actual_goal_reached,
        }
        return self._get_obs(), reward, terminated, truncated, info
        
    def render(self):
        pass # Handled internally by PyBullet GUI
    
    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
import numpy as np

class EnvConfig:
    # Simulation
    DT = 0.1
    RENDER_FPS = 30
    
    # Workspace
    WORKSPACE_X_BOUNDS = (0.0, 15.0)
    WORKSPACE_Y_BOUNDS = (0.0, 15.0)
    
    # Robot
    ROBOT_RADIUS = 0.3
    OMEGA_MAX = 1.0
    A_MAX = 1.0
    V_MIN = 0.0
    V_MAX = 2.0
    
    # Task
    NUM_WAYPOINTS = 5
    WAYPOINT_TOLERANCE = 0.5
    
    # Obstacles
    NUM_OBSTACLES = 5  # Increased for more challenge
    OBS_RADIUS_RANGE = (0.3, 0.6)
    OBS_SPEED_RANGE = (-1.0, 1.0)
    
    # Reward / Penalties
    REWARD_WAYPOINT = 10.0
    REWARD_GOAL = 50.0
    PENALTY_COLLISION = -50.0
    PENALTY_TIME = -0.1
    PENALTY_DIST_MULTIPLIER = 0.01
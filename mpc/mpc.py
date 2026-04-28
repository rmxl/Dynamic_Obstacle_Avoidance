import numpy as np
from scipy.optimize import minimize
from config import EnvConfig

class MPCController:
    """
    Model Predictive Control (MPC) for Ground Robot Waypoint Navigation.
    Formulation based on the project presentation specs.
    """
    def __init__(self, dt=EnvConfig.DT, horizon=10, r_robot=EnvConfig.ROBOT_RADIUS):
        self.dt = dt
        self.H = horizon
        self.r_robot = r_robot
        
        # MPC Weights (Lambda values)
        self.lambda1 = 1.0   # Waypoint tracking
        self.lambda2 = 0.1   # Control effort penalty
        self.lambda3 = 100.0 # Obstacle penalty
        self.lambda4 = 5.0   # Terminal waypoint distance penalty
        
        self.r_safe = 0.5    # Buffer radius for obstacle avoidance (r_robot + r_obs + margin)
        
        # Action bounds
        self.omega_bounds = (-EnvConfig.OMEGA_MAX, EnvConfig.OMEGA_MAX)
        self.a_bounds = (-EnvConfig.A_MAX, EnvConfig.A_MAX)
        self.v_bounds = (EnvConfig.V_MIN, EnvConfig.V_MAX)
        
    def _unicycle_dynamics(self, state, action):
        """Discrete unicycle kinematics: f(st, ut)"""
        x, y, theta, v = state
        omega, a = action
        
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        v_new = np.clip(v + a * self.dt, self.v_bounds[0], self.v_bounds[1])
        
        return np.array([x_new, y_new, theta_new, v_new])

    def _predict_obstacles(self, obstacles):
        """
        Constant-velocity model for obstacle prediction over horizon H.
        obstacles: [px, py, vx, vy, radius] for each obstacle
        Returns: traj of shape (H, num_obs, 2)
        """
        num_obs = len(obstacles)
        obs_traj = np.zeros((self.H, num_obs, 2))
        
        for k in range(self.H):
            for i, obs in enumerate(obstacles):
                px, py, vx, vy, _ = obs
                obs_traj[k, i, 0] = px + k * self.dt * vx
                obs_traj[k, i, 1] = py + k * self.dt * vy
                
        return obs_traj

    def _cost_function(self, u_flat, current_state, w_active, obstacles):
        """
        Computes the total cost over the prediction horizon.
        u_flat: flattened actions array of shape (H * 2,)
        """
        u = u_flat.reshape((self.H, 2))
        state = current_state.copy()
        
        # Predict dynamic obstacles
        obs_traj = self._predict_obstacles(obstacles)
        
        total_cost = 0.0
        
        for k in range(self.H):
            action = u[k]
            state = self._unicycle_dynamics(state, action)
            p_tk = state[:2]
            
            # 1. Waypoint Tracking Cost
            dist_to_wp = np.linalg.norm(p_tk - w_active)
            total_cost += self.lambda1 * (dist_to_wp ** 2)
            
            # 2. Control Effort Cost
            total_cost += self.lambda2 * np.linalg.norm(action) ** 2
            
            # 3. Obstacle Penalty Cost (Soft Constraints form per Slide 9)
            for i, obs in enumerate(obstacles):
                r_obs = obs[4]
                # Robustness: Expand the safety margin linearly over the prediction horizon.
                # This accounts for the accumulating uncertainty of random swerves or wall bounces.
                safe_dist = self.r_robot + r_obs + 0.2 + (k * 0.05)
                
                obs_pred_pos = obs_traj[k, i]
                d_tk = np.linalg.norm(p_tk - obs_pred_pos)
                
                # Penalty max(0, r_safe - d_tk)^2
                penalty = max(0, safe_dist - d_tk)
                total_cost += self.lambda3 * (penalty ** 2)
                
            # 4. Terminal Cost (only at end of horizon)
            if k == self.H - 1:
                total_cost += self.lambda4 * (dist_to_wp ** 2)
                
        return total_cost

    def get_action(self, current_state, w_active, obstacles):
        """
        Solve the optimal control problem and return the first action.
        """
        v = current_state[3]
        
        # Initial guess: accelerate if slow, otherwise maintain speed
        u_guess = np.zeros(self.H * 2)
        for i in range(self.H):
            u_guess[i*2] = 0.0 # omega
            u_guess[i*2 + 1] = EnvConfig.A_MAX if v < 0.5 else 0.0 # a
            
        # Bounds for each action timestep
        bounds = [self.omega_bounds, self.a_bounds] * self.H
        
        # Run optimization
        res = minimize(
            fun=self._cost_function,
            x0=u_guess,
            args=(current_state, w_active, obstacles),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-3, 'disp': False, 'maxiter': 50}
        )
        
        # Extracted action (even if technically 'unsuccessful' due to maxiter, 
        # the partial solve is vastly superior to a hard brake)
        optimal_u = res.x.reshape((self.H, 2))
        
        # Deadlock Override: If the robot is effectively stopped and the optimizer outputs 
        # a braking/negative acceleration, it's trapped in a zero-gradient optimization valley.
        # We manually force it to turn towards the waypoint and accelerate forward.
        action = optimal_u[0]
        if v < 0.1 and action[1] <= 0.01:
            rx, ry, rtheta = current_state[0], current_state[1], current_state[2]
            target_angle = np.arctan2(w_active[1] - ry, w_active[0] - rx)
            # Shortest angle diff
            angle_diff = (target_angle - rtheta + np.pi) % (2 * np.pi) - np.pi
            
            # Rotate aggressively and pulse the gas
            action[0] = np.clip(angle_diff * 2.0, self.omega_bounds[0], self.omega_bounds[1])
            action[1] = self.a_bounds[1]  # A_MAX
            
        return action

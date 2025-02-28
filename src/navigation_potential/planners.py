
import numpy as np
from navigation_potential.environments import GridEnvironment
from navigation_potential.fields import VectorField

class PathPlanner:
    """Class for planning paths using vector fields."""
    
    def __init__(self, environment : GridEnvironment, vector_field : VectorField):
        """Initialize with environment and vector field."""
        self.env = environment
        self.vector_field = vector_field
        
    def plan_path(self, step_size=0.5, max_steps=1000, target_threshold=2.0):
        """Generate a trajectory from start to target."""
        # Initialize trajectory with starting point
        trajectory = [np.array(self.env.start, dtype=float)]
        current_pos = np.array(self.env.start, dtype=float)
        
        # Iterate until reaching target or max steps
        for _ in range(max_steps):
            # Check if close enough to target
            dist_to_target = np.linalg.norm(current_pos - np.array(self.env.target))
            if dist_to_target < target_threshold:
                trajectory.append(np.array(self.env.target))
                return np.array(trajectory), True  # Success
            
            # Get vector direction at current position
            vx, vy = self.vector_field.get_vector_at(current_pos)
            
            # If vector is near zero, try random step
            if abs(vx) < 1e-10 and abs(vy) < 1e-10:
                angle = np.random.uniform(0, 2 * np.pi)
                vx, vy = np.cos(angle), np.sin(angle)
            
            # Compute next position
            next_pos = current_pos + step_size * np.array([vy, vx])
            
            # Check for obstacle collision
            dist_to_obs = self.env.get_interp_distance(next_pos)
            if dist_to_obs < 0.5:  # Safety margin
                # Try to adjust direction to avoid obstacle
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                best_dist = 0
                best_pos = None
                
                for angle in angles:
                    # Try different directions around the blocked path
                    test_vx = np.cos(angle)
                    test_vy = np.sin(angle)
                    test_pos = current_pos + step_size * np.array([test_vy, test_vx])
                    
                    # Check if in bounds
                    if 0 <= test_pos[0] < self.env.size and 0 <= test_pos[1] < self.env.size:
                        test_dist = self.env.get_interp_distance(test_pos)
                        if test_dist > best_dist:
                            best_dist = test_dist
                            best_pos = test_pos
                
                if best_pos is not None and best_dist > 0.5:
                    next_pos = best_pos
                else:
                    # If no good direction found, terminate
                    return np.array(trajectory), False  # Failure
            
            # Ensure next position is within bounds
            next_pos[0] = max(0, min(self.env.size-1, next_pos[0]))
            next_pos[1] = max(0, min(self.env.size-1, next_pos[1]))
            
            # Add to trajectory and update current position
            trajectory.append(next_pos)
            current_pos = next_pos
            
        return np.array(trajectory), False  # Reached max steps without success
    
    @property
    def trajectory_length(self):
        """Calculate the length of the last planned trajectory."""
        if not hasattr(self, 'trajectory') or len(self.trajectory) < 2:
            return 0
        return np.sum(np.linalg.norm(self.trajectory[1:] - self.trajectory[:-1], axis=1))


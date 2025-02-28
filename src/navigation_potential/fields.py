from navigation_potential.environments import GridEnvironment

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
import numpy as np

from abc import ABC, abstractmethod
import heapq


class ScalarField(ABC):
    """Abstract base class for scalar fields."""
    
    def __init__(self, environment : GridEnvironment):
        """Initialize with an environment."""
        self.env = environment
        self.size = environment.size
        self.field = None
        self._field_interp = None
    
    @abstractmethod
    def compute(self):
        """Compute the scalar field."""
        pass
    
    def get_value_at(self, position : np.ndarray):
        """Get interpolated scalar value at a position."""
        if self.field is None:
            raise ValueError("Scalar field not computed yet")
            
        if self._field_interp is None :
            self._create_interpolator()
            
        return self._field_interp([position[0], position[1]])[0]
    
    def _create_interpolator(self):
        """Create interpolator for the scalar field."""
        self._field_interp = RegularGridInterpolator(
            (np.arange(self.size), np.arange(self.size)), 
            self.field, 
            bounds_error=False, 
            fill_value=0
        )


class APFScalarField(ScalarField):
    """Artificial Potential Field scalar field."""
    
    def __init__(self, environment : GridEnvironment, k_att : float=1.0, k_rep : float=100.0, rho0 : float=10.0):
        """Initialize with parameters."""
        super().__init__(environment)
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho0 = rho0
        
    def compute(self):
        """Compute the APF scalar field (potential function)."""
        # Create meshgrid for vectorized computations
        y, x = np.mgrid[0:self.size, 0:self.size]
        
        # Attractive potential
        dx = self.env.target[1] - x
        dy = self.env.target[0] - y
        dist_to_target = np.sqrt(dx**2 + dy**2)
        attractive = 0.5 * self.k_att * dist_to_target**2
        
        # Repulsive potential
        repulsive = np.zeros_like(attractive)
        mask = self.env.distance_map < self.rho0
        
        if np.any(mask):
            repulsive[mask] = 0.5 * self.k_rep * (1/self.env.distance_map[mask] - 1/self.rho0)**2
        
        # Total potential field
        self.field = attractive + repulsive
        
        return self.field

class LaplacianScalarField(ScalarField):
    """Laplacian scalar field."""
    
    def __init__(self, environment : GridEnvironment, alpha : float=1.0):
        """Initialize with environment and boundary value."""
        super().__init__(environment)
        self.alpha = alpha  # Value at obstacle boundaries
    
    def compute(self):
        """Compute the Laplacian scalar field."""
        # Create a mask for the free space
        free_space = ~self.env.grid.copy()
        
        # Create an extended grid that includes domain boundaries as obstacles
        extended_grid = np.ones((self.size + 2, self.size + 2), dtype=bool)
        extended_grid[1:-1, 1:-1] = self.env.grid
        
        # Create target and obstacle boundary conditions
        phi = np.ones_like(self.env.grid, dtype=float) * self.alpha  # Default all to alpha
        target_mask = np.zeros_like(self.env.grid, dtype=bool)
        
        # Dilate target point slightly to make a small target region
        target_region = np.zeros_like(self.env.grid, dtype=bool)
        target_region[max(0, self.env.target[0]-1):min(self.size, self.env.target[0]+2), 
                     max(0, self.env.target[1]-1):min(self.size, self.env.target[1]+2)] = True
        target_mask = target_region & free_space
        
        # Identify obstacle boundary points (adjacent to free space)
        obstacle_boundary = np.zeros_like(self.env.grid, dtype=bool)
        for i in range(self.size):
            for j in range(self.size):
                if self.env.grid[i, j]:  # If this is an obstacle
                    # Check if any neighbor is free space
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size and not self.env.grid[ni, nj]:
                            obstacle_boundary[i, j] = True
                            break
        
        # Also identify domain boundary points
        domain_boundary = np.zeros_like(self.env.grid, dtype=bool)
        domain_boundary[0, :] = True  # Bottom boundary
        domain_boundary[-1, :] = True  # Top boundary
        domain_boundary[:, 0] = True  # Left boundary
        domain_boundary[:, -1] = True  # Right boundary
        
        # Combined boundary
        combined_boundary = obstacle_boundary | domain_boundary
        
        # Set boundary conditions
        phi[target_mask] = 0.0  # Target has value 0
        phi[combined_boundary] = self.alpha  # Obstacles and domain boundaries have value alpha
        
        # Create the Laplacian operator (using 5-point stencil)
        indices = np.arange(self.size*self.size).reshape(self.size, self.size)
        ind_I, ind_J, ind_V = [], [], []
        
        for i in range(self.size):
            for j in range(self.size):
                # Only set up equations for non-boundary, non-target points in free space
                if free_space[i, j] and not target_mask[i, j] and not domain_boundary[i, j]:
                    idx = indices[i, j]
                    ind_I.append(idx)
                    ind_J.append(idx)
                    ind_V.append(4.0)
                    
                    # Add neighbor contributions
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            nidx = indices[ni, nj]
                            ind_I.append(idx)
                            ind_J.append(nidx)
                            ind_V.append(-1.0)
        
        # Create sparse matrix and RHS
        L = sparse.csr_matrix((ind_V, (ind_I, ind_J)), shape=(self.size*self.size, self.size*self.size))
        b = np.zeros(self.size*self.size)
        
        # Add boundary conditions to RHS
        for i in range(self.size):
            for j in range(self.size):
                if free_space[i, j] and not target_mask[i, j] and not domain_boundary[i, j]:
                    idx = indices[i, j]
                    # Check neighbors for boundary conditions
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            if target_mask[ni, nj]:
                                b[idx] += 0.0  # Target boundary condition (0)
                            elif self.env.grid[ni, nj] or domain_boundary[ni, nj]:
                                b[idx] += self.alpha  # Obstacle or domain boundary condition
        
        # Solve the linear system for non-boundary points
        mask = (free_space & ~target_mask & ~domain_boundary).flatten()
        x = np.zeros(self.size*self.size)
        
        # Set known boundary values
        x[target_mask.flatten()] = 0.0  # Target has value 0
        x[(self.env.grid | domain_boundary).flatten()] = self.alpha  # Obstacles and boundaries have value alpha
        
        # Solve for unknown values
        if np.any(mask):
            L_sub = L[mask][:, mask]
            b_sub = b[mask]
            x_sub = spsolve(L_sub, b_sub)
            x[mask] = x_sub
        
        # Reshape the solution
        self.field = x.reshape(self.size, self.size)
        
        return self.field
class EikonalScalarField(ScalarField):
    """Eikonal scalar field."""
    
    def __init__(self, environment, sigma=5.0):
        """Initialize with parameters."""
        super().__init__(environment)
        self.sigma = sigma
        
    def compute(self):
        """Compute the Eikonal scalar field (distance/time field)."""
        # Create an extended grid that includes domain boundaries as obstacles
        extended_grid = np.ones((self.size + 2, self.size + 2), dtype=bool)
        extended_grid[1:-1, 1:-1] = self.env.grid
        
        # Create a distance map that accounts for domain boundaries
        # First compute distance to original obstacles
        distance_map = self.env.distance_map.copy()
        
        # Then account for domain boundaries by computing distance to edges
        y_indices, x_indices = np.indices((self.size, self.size))
        distance_to_left = x_indices
        distance_to_right = self.size - 1 - x_indices
        distance_to_bottom = y_indices
        distance_to_top = self.size - 1 - y_indices
        
        # Minimum distance to any domain boundary
        distance_to_boundary = np.minimum.reduce([
            distance_to_left, distance_to_right, 
            distance_to_bottom, distance_to_top
        ])
        
        # Final distance is the minimum of distance to obstacles and distance to boundary
        distance_map = np.minimum(distance_map, distance_to_boundary)
        
        # Create speed function (slower near obstacles and boundaries)
        speed = 1.0 - np.exp(-distance_map**2 / (2*self.sigma**2))
        speed = np.clip(speed, 0.1, 1.0)  # Ensure positive speed
        
        # Initialize field with infinity
        self.field = np.ones((self.size, self.size)) * np.inf
        
        # Set the status for each point:
        # 0: Far (not processed)
        # 1: Considered (in narrow band)
        # 2: Frozen (processed)
        status = np.zeros((self.size, self.size), dtype=int)
        
        # Mark obstacles and domain boundary as frozen
        status[self.env.grid] = 2  # Obstacles
        
        # Also mark points at the domain boundary
        status[0, :] = 2  # Bottom edge
        status[-1, :] = 2  # Top edge
        status[:, 0] = 2  # Left edge
        status[:, -1] = 2  # Right edge
        
        # Initialize narrow band with the target point
        narrow_band = []
        
        # Make sure target is not on the boundary or an obstacle
        target_y, target_x = self.env.target
        if 0 < target_y < self.size-1 and 0 < target_x < self.size-1 and not self.env.grid[target_y, target_x]:
            self.field[target_y, target_x] = 0.0
            status[target_y, target_x] = 2  # Mark target as frozen
            
            # Add neighbors of target to narrow band
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = target_y + di, target_x + dj
                if 0 <= ni < self.size and 0 <= nj < self.size and status[ni, nj] == 0:
                    # Compute initial time estimate
                    self.field[ni, nj] = np.sqrt(di**2 + dj**2) / speed[ni, nj]
                    status[ni, nj] = 1  # Mark as considered
                    heapq.heappush(narrow_band, (self.field[ni, nj], (ni, nj)))
        else:
            # If target is on boundary or obstacle, find closest free point
            free_points = np.where((~self.env.grid) & 
                                   (y_indices > 0) & (y_indices < self.size-1) & 
                                   (x_indices > 0) & (x_indices < self.size-1))
            if len(free_points[0]) > 0:
                # Find point closest to original target
                distances = (free_points[0] - target_y)**2 + (free_points[1] - target_x)**2
                closest_idx = np.argmin(distances)
                alt_target_y, alt_target_x = free_points[0][closest_idx], free_points[1][closest_idx]
                
                self.field[alt_target_y, alt_target_x] = 0.0
                status[alt_target_y, alt_target_x] = 2  # Mark as frozen
                
                # Add neighbors to narrow band
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = alt_target_y + di, alt_target_x + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size and status[ni, nj] == 0:
                        self.field[ni, nj] = np.sqrt(di**2 + dj**2) / speed[ni, nj]
                        status[ni, nj] = 1  # Mark as considered
                        heapq.heappush(narrow_band, (self.field[ni, nj], (ni, nj)))
            else:
                # No valid free points - this is an error condition
                print("Warning: No valid free points for Eikonal field initialization")
                return np.zeros((self.size, self.size))
        
        # Main Fast Marching loop
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while narrow_band:
            # Get point with minimum time
            current_time, (i, j) = heapq.heappop(narrow_band)
            
            # Check if already processed (can happen due to duplicates in heap)
            if status[i, j] == 2:
                continue
                
            # Mark as frozen
            status[i, j] = 2
            self.field[i, j] = current_time
            
            # Update neighbors
            for di, dj in directions:
                ni, nj = i + di, j + dj
                # Skip boundary points
                if (ni == 0 or ni == self.size-1 or nj == 0 or nj == self.size-1):
                    continue
                    
                if 0 <= ni < self.size and 0 <= nj < self.size and status[ni, nj] != 2:
                    # Compute time to this neighbor using upwind scheme
                    t_values = []
                    
                    # Check x-direction neighbors
                    if 0 <= ni-1 < self.size and status[ni-1, nj] == 2:
                        t_values.append(self.field[ni-1, nj])
                    if 0 <= ni+1 < self.size and status[ni+1, nj] == 2:
                        t_values.append(self.field[ni+1, nj])
                    
                    tx = min(t_values) if t_values else np.inf
                    
                    # Check y-direction neighbors
                    t_values = []
                    if 0 <= nj-1 < self.size and status[ni, nj-1] == 2:
                        t_values.append(self.field[ni, nj-1])
                    if 0 <= nj+1 < self.size and status[ni, nj+1] == 2:
                        t_values.append(self.field[ni, nj+1])
                    
                    ty = min(t_values) if t_values else np.inf
                    
                    # Solve quadratic equation for the arrival time
                    if tx == np.inf and ty == np.inf:
                        continue
                    
                    a = 0
                    b = 0
                    c = -1.0/(speed[ni, nj]**2)
                    
                    if tx != np.inf:
                        a += 1
                        b -= 2*tx
                        c += tx**2
                    
                    if ty != np.inf:
                        a += 1
                        b -= 2*ty
                        c += ty**2
                    
                    if a == 0:  # Should not happen given the checks above
                        continue
                    
                    # Solve quadratic: at^2 + bt + c = 0
                    discriminant = b**2 - 4*a*c
                    if discriminant < 0:  # No real solution, should not happen
                        continue
                        
                    new_time = (-b + np.sqrt(discriminant)) / (2*a)
                    
                    # Update if a better time is found
                    if new_time < self.field[ni, nj]:
                        self.field[ni, nj] = new_time
                        
                        # Add to narrow band if not already there
                        if status[ni, nj] == 0:
                            status[ni, nj] = 1
                            heapq.heappush(narrow_band, (new_time, (ni, nj)))
                        else:
                            # For points already in narrow band, we need to add them again
                            heapq.heappush(narrow_band, (new_time, (ni, nj)))
        
        # Set high values for boundaries and obstacles if not already set
        self.field[self.env.grid] = np.max(self.field) * 2
        self.field[0, :] = np.max(self.field) * 2  # Bottom edge
        self.field[-1, :] = np.max(self.field) * 2  # Top edge
        self.field[:, 0] = np.max(self.field) * 2  # Left edge
        self.field[:, -1] = np.max(self.field) * 2  # Right edge
        
        # Fix any remaining infinity values
        mask = np.isinf(self.field)
        if np.any(mask):
            valid_mask = ~np.isinf(self.field)
            i_valid, j_valid = np.where(valid_mask)
            i_invalid, j_invalid = np.where(mask)
            
            # Find nearest valid point for each invalid point
            for i, j in zip(i_invalid, j_invalid):
                distances = (i_valid - i)**2 + (j_valid - j)**2
                idx = np.argmin(distances)
                self.field[i, j] = self.field[i_valid[idx], j_valid[idx]]
        
        return self.field

class VectorField:
    """Class for vector fields derived from scalar fields."""
    
    def __init__(self, scalar_field : ScalarField):
        """Initialize with a scalar field."""
        self.scalar_field = scalar_field
        self.env = scalar_field.env
        self.size = scalar_field.size
        self.fx = None
        self.fy = None
        self._fx_interp = None
        self._fy_interp = None
        
    def compute(self, normalize : bool=False):
        """Compute the vector field as the gradient of the scalar field."""
        if self.scalar_field.field is None:
            self.scalar_field.compute()
            
        # Calculate the gradient (negative for fields where lower values are better)
        self.fy, self.fx = np.gradient(-self.scalar_field.field)
        
        if normalize:
            # Normalize vector field
            magnitude = np.sqrt(self.fx**2 + self.fy**2)
            magnitude[magnitude < 1e-10] = 1.0  # Avoid division by zero
            self.fx = self.fx / magnitude
            self.fy = self.fy / magnitude
        
        return self.fx, self.fy
    
    def get_vector_at(self, position : np.ndarray):
        """Get interpolated vector at a position."""
        if self.fx is None or self.fy is None:
            self.compute()
            
        if self._fx_interp is None or self._fy_interp is None:
            self._create_interpolators()
            
        vx = self._fx_interp([position[0], position[1]])[0]
        vy = self._fy_interp([position[0], position[1]])[0]
        
        # Normalize
        magnitude = np.sqrt(vx**2 + vy**2)
        if magnitude > 1e-10:
            vx /= magnitude
            vy /= magnitude
            
        return vx, vy
    
    def _create_interpolators(self):
        """Create interpolators for the vector field."""
        self._fx_interp = RegularGridInterpolator(
            (np.arange(self.size), np.arange(self.size)), 
            self.fx, 
            bounds_error=False, 
            fill_value=0
        )
        self._fy_interp = RegularGridInterpolator(
            (np.arange(self.size), np.arange(self.size)), 
            self.fy, 
            bounds_error=False, 
            fill_value=0
        )

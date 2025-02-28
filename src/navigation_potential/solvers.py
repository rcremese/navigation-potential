from navigation_potential.environments import GridEnvironment
from navigation_potential.fields import ScalarField

from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy as np
from abc import ABC, abstractmethod
import heapq

class FieldSolver(ABC):
    """Abstract base class for scalar field solvers."""
    def __init__(self, environment : GridEnvironment):
        assert isinstance(environment, GridEnvironment), "Environment must be an instance of GridEnvironment"
        self.environment = environment

    @abstractmethod
    def solve(self, **kwargs) -> ScalarField:
        """
        Solve for a scalar field.

        Args:
            **kwargs: Additional solver-specific parameters.

        Returns:
            ScalarField: The computed scalar field.
        """
        pass

class APFSolver(FieldSolver):
    """Artificial Potential Field solver."""

    def solve(self, k_att=1.0, k_rep=100.0, rho0=10.0) -> ScalarField:
        """
        Compute an Artificial Potential Field.

        Args:
            k_att: Attractive force coefficient.
            k_rep: Repulsive force coefficient.
            rho0: Obstacle influence range.
            environment (GridEnvironment, optional): Grid environment.

        Returns:
            ScalarField: Computed APF scalar field.
        """
        # Create scalar field with same dimensions as environment
        size = self.environment.size
        field = ScalarField([0, 0], [size-1, size-1], [size, size])

        # Create meshgrid for vectorized computations
        y, x = np.mgrid[0:size, 0:size]

        # Attractive potential (toward target)
        dx = self.environment.target[1] - x
        dy = self.environment.target[0] - y
        dist_to_target = np.sqrt(dx**2 + dy**2)
        attractive = 0.5 * k_att * dist_to_target**2

        # Repulsive potential (away from obstacles)
        repulsive = np.zeros_like(attractive)
        mask = self.environment.distance_map < rho0

        if np.any(mask):
            repulsive[mask] = 0.5 * k_rep * (1/self.environment.distance_map[mask] - 1/rho0)**2

        # Total potential field
        field.values = attractive + repulsive

        return field

class LaplaceSolver(FieldSolver):
    """
    Laplacian field solver.

    Computes a scalar field by solving the Laplace equation (∇²ϕ = 0) with
    Dirichlet boundary conditions: ϕ = 0 at target, ϕ = alpha at obstacles.
    Always computes at the same resolution as the environment grid.
    """
    def solve(self, alpha : float=1.0) -> ScalarField:
        """
        Compute the Laplacian scalar field.

        Args:
            alpha: Value at obstacle boundaries.

        Returns:
            ScalarField: Computed Laplacian scalar field.
        """
        # Create scalar field at the same resolution as environment
        size = self.environment.size
        field = ScalarField([0, 0], [size-1, size-1], [size, size])

        # Use the environment grid directly
        grid = self.environment.grid.copy()
        free_space = ~grid

        # Create target mask
        target_mask = self._create_target_mask(self.environment, free_space)

        # Initialize the linear system and solve
        phi = self._solve_laplace_equation(grid, free_space, target_mask, alpha)

        # Set the computed field
        field.values = phi

        return field

    def _create_target_mask(self, environment : GridEnvironment, free_space : np.ndarray) -> np.ndarray:
        """
        Create a mask for the target region.

        Args:
            environment: Environment containing target information.
            free_space: Mask of free space in the grid.

        Returns:
            numpy.ndarray: Boolean mask of the target region.
        """
        size = environment.size

        # Create target region (small area around target point)
        target_region = np.zeros((size, size), dtype=bool)
        target_y, target_x = environment.target

        # Create a small region around the target
        target_region[max(0, target_y-1):min(size, target_y+2),
                     max(0, target_x-1):min(size, target_x+2)] = True

        # Ensure target is in free space
        target_mask = target_region & free_space

        return target_mask

    def _setup_linear_system(self, grid, free_space, target_mask):
        """
        Set up the linear system for the Laplace equation.

        Args:
            grid: Grid of obstacles.
            free_space: Mask of free space in the grid.
            target_mask: Mask of the target region.

        Returns:
            tuple: (indices, I, J, V, b) components for the linear system
        """
        size = self.environment.size

        # Create index mapping for flattened grid
        indices = np.arange(size*size).reshape(size, size)

        # Initialize sparse matrix components and right-hand side
        I, J, V = [], [], []
        b = np.zeros(size*size)

        for i in range(size):
            for j in range(size):
                # Only set up equations for free space that isn't the target
                if free_space[i, j] and not target_mask[i, j]:
                    idx = indices[i, j]

                    # Center cell coefficient (standard 5-point stencil)
                    I.append(idx)
                    J.append(idx)
                    V.append(4.0)

                    # Add neighbor contributions
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            nidx = indices[ni, nj]
                            I.append(idx)
                            J.append(nidx)
                            V.append(-1.0)

        return indices, I, J, V, b

    def _apply_boundary_conditions(self, indices, b, grid, free_space, target_mask, alpha):
        """
        Apply boundary conditions to the right-hand side of the linear system.

        Args:
            indices: Index mapping for flattened grid.
            b: Right-hand side vector.
            grid: Grid of obstacles.
            free_space: Mask of free space in the grid.
            target_mask: Mask of the target region.
            alpha: Value at obstacle boundaries.

        Returns:
            numpy.ndarray: Updated right-hand side vector.
        """
        size = self.environment.size

        # Apply boundary conditions to interior points adjacent to boundaries
        for i in range(size):
            for j in range(size):
                if free_space[i, j] and not target_mask[i, j]:
                    idx = indices[i, j]

                    # Check neighbors for boundary conditions
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if target_mask[ni, nj]:
                                b[idx] += 0.0  # Target boundary (0)
                            elif grid[ni, nj]:
                                b[idx] += alpha  # Obstacle boundary

        # Apply domain boundary conditions
        for i in range(size):
            for j in range(size):
                if free_space[i, j] and not target_mask[i, j]:
                    idx = indices[i, j]
                    # Check if on edge
                    if i == 0 or i == size-1 or j == 0 or j == size-1:
                        b[idx] += alpha  # Treat domain boundary like obstacle

        return b

    def _solve_laplace_equation(self, grid, free_space, target_mask, alpha):
        """
        Solve the Laplace equation with the given boundary conditions.

        Args:
            grid: Grid of obstacles.
            free_space: Mask of free space in the grid.
            target_mask: Mask of the target region.
            alpha: Value at obstacle boundaries.

        Returns:
            numpy.ndarray: Solution to the Laplace equation.
        """
        size = self.environment.size

        # Initialize solution with boundary values
        phi = np.ones((size, size), dtype=float) * alpha  # Default all to alpha
        phi[target_mask] = 0.0  # Target has value 0
        phi[grid] = alpha  # Obstacles have value alpha

        # Set up the linear system
        indices, I, J, V, b = self._setup_linear_system(grid, free_space, target_mask)

        # Apply boundary conditions
        b = self._apply_boundary_conditions(indices, b, grid, free_space, target_mask, alpha)

        # Create sparse matrix
        L = sparse.csr_matrix((V, (I, J)), shape=(size*size, size*size))

        # Identify unknown values (free space that isn't target)
        mask = (free_space & ~target_mask).flatten()
        x = np.zeros(size*size)

        # Set known boundary values
        x[target_mask.flatten()] = 0.0
        x[grid.flatten()] = alpha

        # Solve for unknown values
        if np.any(mask):
            try:
                # Extract submatrix for unknown values
                L_sub = L[mask][:, mask]
                b_sub = b[mask]

                # Solve the linear system
                x_sub = spsolve(L_sub, b_sub)
                x[mask] = x_sub
            except Exception as e:
                print(f"Warning: Sparse solver failed ({e}), using fallback method")
                # Fallback to iterative solver (simple Jacobi iteration)
                x[mask] = self._solve_iteratively(L, b, x, mask, 1000, 1e-6)

        # Reshape the solution
        return x.reshape(size, size)

    def _solve_iteratively(self, L, b, x, mask, max_iter=1000, tol=1e-6):
        """
        Solve the linear system iteratively if the direct solver fails.

        Args:
            L: System matrix.
            b: Right-hand side vector.
            x: Initial guess.
            mask: Mask for unknown values.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.

        Returns:
            numpy.ndarray: Solution vector.
        """
        # Extract diagonal
        D = L.diagonal()[mask]
        D_inv = 1.0 / D

        # Jacobi iteration
        x_old = x[mask].copy()
        for _ in range(max_iter):
            # Compute residual
            r = b[mask] - L[mask, :].dot(x)

            # Update solution
            x_new = x_old + D_inv * r

            # Check convergence
            if np.max(np.abs(x_new - x_old)) < tol:
                break

            x_old = x_new.copy()
            x[mask] = x_new

        return x[mask]

class EikonalSolver(FieldSolver):
    """
    Eikonal equation solver using the Fast Marching Method.

    Computes the minimum time-to-target field by solving the Eikonal equation:
    |∇T(x)| = 1/F(x) where F(x) is the speed function and T is the time field.
    """

    def solve(self, sigma : float=5.0):
        """
        Compute the Eikonal scalar field (time-to-target field).

        Args:
            environment: Grid environment.
            sigma: Parameter controlling obstacle influence on speed.

        Returns:
            ScalarField: Computed Eikonal scalar field.
        """
        size = self.environment.size

        # Create the output field
        field = ScalarField(np.zeros(2) + 0.5, np.ones(2) * size - 0.5, [size, size])

        # Create speed function based on distance to obstacles
        speed = self._create_speed_function(self.environment, sigma)

        # Solve the Eikonal equation
        time_field = self._solve_eikonal_2d(self.environment, speed)

        # Set the computed field
        field.values = time_field

        return field

    def _create_speed_function(self, environment : GridEnvironment, sigma : float):
        """
        Create a speed function that slows down near obstacles and boundaries.

        Args:
            environment: Grid environment.
            sigma: Controls how much obstacles affect speed.

        Returns:
            numpy.ndarray: Speed function values.
        """
        size = environment.size

        # Get distance map to obstacles
        distance_map = environment.distance_map.copy()

        # Account for domain boundaries
        distance_to_boundary = self._compute_boundary_distance(size)

        # Final distance is minimum of distance to obstacles and boundaries
        distance_map = np.minimum(distance_map, distance_to_boundary)

        # Create speed function (slower near obstacles and boundaries)
        speed = 1.0 - np.exp(-distance_map**2 / (2*sigma**2))
        speed = np.clip(speed, 0.1, 1.0)  # Ensure positive speed

        return speed

    def _compute_boundary_distance(self, size):
        """
        Compute distance to domain boundaries for each point.

        Args:
            size: Size of the grid.

        Returns:
            numpy.ndarray: Distance to nearest boundary.
        """
        y_indices, x_indices = np.indices((size, size))

        # Distance to each boundary
        distance_to_left = x_indices
        distance_to_right = size - 1 - x_indices
        distance_to_bottom = y_indices
        distance_to_top = size - 1 - y_indices

        # Minimum distance to any boundary
        return np.minimum.reduce([
            distance_to_left, distance_to_right,
            distance_to_bottom, distance_to_top
        ])

    def _solve_eikonal_2d(self, environment, speed):
        """
        Solve the 2D Eikonal equation using Fast Marching Method.

        Args:
            environment: Grid environment.
            speed: Speed function values.

        Returns:
            numpy.ndarray: Time field solution.
        """
        size = environment.size

        # Initialize time field and status array
        time_field, status = self._initialize_arrays(environment)

        # Initialize narrow band starting from target
        narrow_band = self._initialize_narrow_band(environment, time_field, status, speed)

        # Main Fast Marching loop
        self._fast_marching_loop(narrow_band, time_field, status, speed, size)

        # Fix any remaining infinity values and set boundaries
        self._finalize_field(time_field, environment)

        return time_field

    def _initialize_arrays(self, environment):
        """
        Initialize the time field and status arrays.

        Args:
            environment: Grid environment.

        Returns:
            tuple: (time_field, status)
        """
        size = environment.size

        # Initialize time field with infinity
        time_field = np.ones((size, size)) * np.inf

        # Set the status for each point:
        # 0: Far (not processed)
        # 1: Considered (in narrow band)
        # 2: Frozen (processed)
        status = np.zeros((size, size), dtype=int)

        # Mark obstacles and domain boundary as frozen
        status[environment.grid] = 2  # Obstacles
        status[0, :] = 2  # Bottom edge
        status[-1, :] = 2  # Top edge
        status[:, 0] = 2  # Left edge
        status[:, -1] = 2  # Right edge

        return time_field, status

    def _initialize_narrow_band(self, environment, time_field, status, speed):
        """
        Initialize the narrow band with points around the target.

        Args:
            environment: Grid environment.
            time_field: Time field array.
            status: Status array.
            speed: Speed function values.

        Returns:
            list: Narrow band (priority queue).
        """
        size = environment.size
        narrow_band = []
        target_y, target_x = environment.target

        # Make sure target is not on boundary or obstacle
        if not environment.grid[target_y, target_x]:
            # Set target time to zero
            time_field[target_y, target_x] = 0.0
            status[target_y, target_x] = 2  # Mark as frozen

            # Add neighbors to narrow band
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = target_y + dy, target_x + dx
                if 0 <= ny < size and 0 <= nx < size and status[ny, nx] == 0:
                    # Compute initial time estimate
                    time = np.sqrt(dy**2 + dx**2) / speed[ny, nx]
                    time_field[ny, nx] = time
                    status[ny, nx] = 1  # Mark as considered
                    heapq.heappush(narrow_band, (time, (ny, nx)))

        return narrow_band

    def _fast_marching_loop(self, narrow_band, time_field, status, speed, size):
        """
        Execute the main Fast Marching loop.

        Args:
            narrow_band: Priority queue of points to process.
            time_field: Time field array.
            status: Status array.
            speed: Speed function values.
            size: Size of the grid.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while narrow_band:
            # Get point with minimum time
            current_time, (i, j) = heapq.heappop(narrow_band)

            # Skip if already processed
            if status[i, j] == 2:
                continue

            # Mark as frozen
            status[i, j] = 2
            time_field[i, j] = current_time

            # Update neighbors
            for di, dj in directions:
                ni, nj = i + di, j + dj

                # Check if in bounds, not processed, and not obstacle
                if (0 <= ni < size and 0 <= nj < size and
                    status[ni, nj] != 2 and status[ni, nj] != -1):

                    # Skip boundary points
                    if ni == 0 or ni == size-1 or nj == 0 or nj == size-1:
                        continue

                    # Compute new arrival time
                    new_time = self._update_point(ni, nj, time_field, status, speed)

                    # Update if better time found
                    if new_time < time_field[ni, nj]:
                        time_field[ni, nj] = new_time

                        # Add to narrow band
                        if status[ni, nj] == 0:
                            status[ni, nj] = 1
                        heapq.heappush(narrow_band, (new_time, (ni, nj)))

    def _update_point(self, i, j, time_field, status, speed):
        """
        Update the arrival time at point (i,j) using the upwind scheme.

        Args:
            i, j: Point coordinates.
            time_field: Time field array.
            status: Status array.
            speed: Speed function values.

        Returns:
            float: Updated arrival time.
        """
        # Get minimum arrival times from neighbors
        tx = ty = np.inf

        # Check x-direction neighbors
        if status[i-1, j] == 2:
            tx = min(tx, time_field[i-1, j])
        if status[i+1, j] == 2:
            tx = min(tx, time_field[i+1, j])

        # Check y-direction neighbors
        if status[i, j-1] == 2:
            ty = min(ty, time_field[i, j-1])
        if status[i, j+1] == 2:
            ty = min(ty, time_field[i, j+1])

        # If no frozen neighbors, can't update
        if tx == np.inf and ty == np.inf:
            return np.inf

        # Solve quadratic equation for arrival time
        return self._solve_eikonal_quadratic(tx, ty, speed[i, j])

    def _solve_eikonal_quadratic(self, tx, ty, speed_value):
        """
        Solve the quadratic equation for the Eikonal update.

        Args:
            tx: Minimum x-direction time.
            ty: Minimum y-direction time.
            speed_value: Speed at the point.

        Returns:
            float: Solution to the quadratic equation.
        """
        a = 0
        b = 0
        c = -1.0/(speed_value**2)

        if tx != np.inf:
            a += 1
            b -= 2*tx
            c += tx**2

        if ty != np.inf:
            a += 1
            b -= 2*ty
            c += ty**2

        if a == 0:  # No valid neighbors
            return np.inf

        # Solve quadratic: at^2 + bt + c = 0
        discriminant = b**2 - 4*a*c
        if discriminant < 0:  # No real solution
            return np.inf

        return (-b + np.sqrt(discriminant)) / (2*a)

    def _finalize_field(self, time_field, environment):
        """
        Set boundary values and handle any remaining infinity values.

        Args:
            time_field: Time field array.
            environment: Grid environment.
        """
        # Set high values for obstacles and boundaries
        max_time = np.max(time_field[~np.isinf(time_field)]) if np.any(~np.isinf(time_field)) else 1.0
        high_value = max_time * 2

        time_field[environment.grid] = high_value
        time_field[0, :] = high_value  # Bottom edge
        time_field[-1, :] = high_value  # Top edge
        time_field[:, 0] = high_value  # Left edge
        time_field[:, -1] = high_value  # Right edge

        # Fix any remaining infinity values
        mask = np.isinf(time_field)
        if np.any(mask):
            valid_mask = ~np.isinf(time_field)
            i_valid, j_valid = np.where(valid_mask)
            i_invalid, j_invalid = np.where(mask)

            # Find nearest valid point for each invalid point
            for i, j in zip(i_invalid, j_invalid):
                distances = (i_valid - i)**2 + (j_valid - j)**2
                idx = np.argmin(distances)
                time_field[i, j] = time_field[i_valid[idx], j_valid[idx]]

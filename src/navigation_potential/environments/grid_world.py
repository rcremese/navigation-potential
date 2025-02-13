import numpy as np


class GridWorld:
    def __init__(
        self,
        grid_size: tuple[int],
        spacing: float | tuple[float] = 1.0,
        seed: int | None = None,
    ):
        """Initialise the grid world.
        The grid world is a grid of cells, each cell can be either free or occupied by an obstacle.
        The grid world is defined by its size and the spacing between cells. The spacing can be a scalar or a tuple of scalars.

        Args:
            grid_size (tuple[int]): The size of the grid world.
            spacing (float | tuple[float]): The spacing between cells.
            seed (int, optional): The seed for the random number generator. Defaults to None.
        """
        assert isinstance(grid_size, tuple), "Grid size must be a tuple of integers"
        assert all(isinstance(size, int) for size in grid_size), (
            "Grid size must be a tuple of positive integers"
        )
        if any(size <= 0 for size in grid_size):
            raise ValueError("Grid size must be a tuple of positive integers")
        self.grid_size: tuple[int] = grid_size
        self.dim: int = len(grid_size)
        assert isinstance(spacing, (float, tuple)), (
            "Spacing must be a float or a tuple of floats"
        )
        if isinstance(spacing, float):
            spacing = self.dim * (spacing,)
        assert len(spacing) == self.dim, (
            "Spacing must have the same dimension as the grid size"
        )
        assert all(isinstance(sp, float) and sp > 0 for sp in spacing), (
            "Spacing must be a positive float or a tuple of floats"
        )
        self.spacing: tuple[float] = spacing
        self.grid: np.ndarray = None
        self.initial_pos: np.ndarray = None
        self.goal_pos: np.ndarray = None
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.initialize()

    def initialize(self):
        self.grid = self.generate_grid()
        # Add inital and goal positions
        self.goal_pos = self.sample_position()
        self.initial_pos = self.sample_position()
        # Ensure initial and goal positions are different and not obstacles
        while np.array_equal(self.initial_pos, self.goal_pos):
            self.initial_pos = self.sample_position()

    def generate_grid(self) -> np.ndarray:
        return np.zeros(self.grid_size, dtype=int)

    def sample_position(self) -> np.ndarray:
        free_pos = self.free_positions
        if free_pos.size == 0:
            raise ValueError("No free positions available")
        return free_pos[self.rng.randint(free_pos.shape[0])]

    def set_obstacle(self, position: tuple[int]):
        assert isinstance(position, tuple), "Position must be a tuple"
        assert len(position) == self.dim, f"Position must be a tuple of len {self.dim}"
        if any([pos < 0 for pos in position]) or any(
            [pos >= self.grid_size[i] for i, pos in enumerate(position)]
        ):
            raise IndexError("Obstacle position out of bounds")
        self.grid[position] = 1

    def is_obstacle(self, position: tuple[int]) -> bool:
        assert isinstance(position, tuple), "Position must be a tuple"
        assert len(position) == self.dim, f"Position must be a  tuple of len {self.dim}"
        if any([pos < 0 for pos in position]) or any(
            [pos >= self.grid_size[i] for i, pos in enumerate(position)]
        ):
            raise IndexError("Obstacle position out of bounds")
        return self.grid[position] == 1

    @property
    def free_positions(self) -> np.ndarray:
        return np.argwhere(1 - self.grid)

    def __repr__(self):
        return f"{self.__class__.__name__}(grid_size={self.grid_size}, spacing={self.spacing},seed={self.seed})"


class RandomObstacleGrid(GridWorld):
    def __init__(
        self,
        grid_size: int,
        obstacle_density: float,
        spacing: tuple[float] | float = 1.0,
        seed=None,
    ):
        if obstacle_density < 0 or obstacle_density > 0.9:
            raise ValueError("Obstacle density must be between 0 and 0.9")
        self.obstacle_density = obstacle_density
        super().__init__(grid_size, spacing, seed)

    def generate_grid(self) -> np.ndarray:
        obstacle_grid = self.rng.choice(
            [0, 1],
            size=self.grid_size,
            p=[1 - self.obstacle_density, self.obstacle_density],
        )

        while obstacle_grid.sum() >= np.prod(self.grid_size) - 2:
            obstacle_grid = self.rng.choice(
                [0, 1],
                size=self.grid_size,
                p=[1 - self.obstacle_density, self.obstacle_density],
            )

        return obstacle_grid

    def __repr__(self):
        return f"{__class__.__name__}(grid_size={self.grid_size}, obstacle_density={self.obstacle_density}, spacing={self.spacing}, seed={self.seed})"


# Example usage:
if __name__ == "__main__":
    # Create a 2D and 3D grid world without obstacles
    grid_2d = GridWorld(grid_size=(5, 5), seed=None)
    grid_3d = GridWorld(grid_size=(5, 10, 15), seed=None)
    print("2D Grid World:", grid_2d, "\n")
    print("3D Grid World:", grid_3d, "\n")
    # Create a 2D and 3D grid world with random obstacles
    obstacle_grid_2d = RandomObstacleGrid(
        grid_size=(5, 5), obstacle_density=0.3, seed=None
    )
    obstacle_grid_3d = RandomObstacleGrid(
        grid_size=(5, 10, 15), obstacle_density=0.3, seed=None
    )
    print("2D Grid World with Obstacles:", obstacle_grid_2d, "\n")
    print("3D Grid World with Obstacles:", obstacle_grid_3d, "\n")

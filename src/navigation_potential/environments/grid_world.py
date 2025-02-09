import abc
import numpy as np


class GridWorld(abc.ABC):
    def __init__(self, grid_size: int, seed=None):
        assert isinstance(grid_size, int), "Grid size must be an integer"
        if grid_size <= 1:
            raise ValueError(
                "Grid size must be greater than one in order to place initial and goal positions"
            )
        self.grid_size = grid_size
        self.dim: int = None
        self.grid: np.ndarray = None
        self.initial_pos: np.ndarray = None
        self.goal_pos: np.ndarray = None
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample_position(self) -> np.ndarray:
        position = self.rng.randint(0, self.grid_size, size=self.dim)
        while self.is_obstacle(tuple(position)):
            position = self.rng.randint(0, self.grid_size, size=self.dim)
        return position

    def set_obstacle(self, position: tuple[int]):
        assert isinstance(position, tuple), "Position must be a tuple"
        assert len(position) == self.dim, "Position must be a 1D array of shape (dim,)"
        if np.any([pos < 0 for pos in position]) or np.any(
            [pos >= self.grid_size for pos in position]
        ):
            raise IndexError("Obstacle position out of bounds")
        self.grid[position] = 1

    def initialize(self):
        self.grid = self.generate_grid()
        # Add inital and goal positions
        self.goal_pos = self.sample_position()
        self.initial_pos = self.sample_position()
        # Ensure initial and goal positions are different and not obstacles
        while np.array_equal(self.initial_pos, self.goal_pos):
            self.initial_pos = self.sample_position()

    def is_obstacle(self, position: tuple[int]) -> bool:
        assert isinstance(position, tuple), "Position must be a tuple"
        assert len(position) == self.dim, "Position must be a 1D array of shape (dim,)"
        if np.any([pos < 0 for pos in position]) or np.any(
            [pos >= self.grid_size for pos in position]
        ):
            raise IndexError("Obstacle position out of bounds")
        return self.grid[position] == 1

    @property
    def free_positions(self) -> np.ndarray:
        return np.argwhere(self.grid)

    @abc.abstractmethod
    def generate_grid(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def print_grid(self):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(grid_size={self.grid_size}, seed={self.seed})"
        )


class Grid2D(GridWorld):
    def __init__(self, grid_size=5, seed=None):
        super().__init__(grid_size, seed)
        self.dim = 2
        self.initialize()

    def generate_grid(self):
        return np.zeros((self.grid_size, self.grid_size), dtype=int)

    def print_grid(self):
        grid_view = self.grid.tolist()
        for i, row in enumerate(grid_view):
            row_print = []
            for j, cell in enumerate(row):
                car = "X" if cell == 1 else "."
                if np.array_equal(self.initial_pos, (i, j)):
                    car = "I"
                elif np.array_equal(self.goal_pos, (i, j)):
                    car = "G"
                row_print.append(car)
            print(" ".join([car for car in row_print]))
        print(f"Initial position: {self.initial_pos}")
        print(f"Goal position: {self.goal_pos}")


class Grid3D(GridWorld):
    def __init__(self, grid_size=5, seed=None):
        super().__init__(grid_size, seed)
        self.dim = 3
        self.initialize()

    def generate_grid(self):
        return np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=int)

    def print_grid(self):
        grid_view = self.grid.tolist()
        for i, layer in enumerate(grid_view):
            print(f"Layer {i}:")
            for j, row in enumerate(layer):
                row_print = []
                for k, cell in enumerate(row):
                    car = "X" if cell == 1 else "."
                    if np.array_equal(self.initial_pos, (i, j, k)):
                        car = "I"
                    elif np.array_equal(self.goal_pos, (i, j, k)):
                        car = "G"
                    row_print.append(car)
                print(" ".join(row_print))
        print(f"Initial position: {self.initial_pos}")
        print(f"Goal position: {self.goal_pos}")


# Example usage:
if __name__ == "__main__":
    # Create a 2D grid world with obstacles
    grid_2d = Grid2D(grid_size=5, seed=None)
    print("2D Grid World:")
    grid_2d.print_grid()
    print("\n")

    # Create a 3D grid world with obstacles
    grid_3d = Grid3D(grid_size=3, seed=None)
    print("3D Grid World:")
    grid_3d.print_grid()

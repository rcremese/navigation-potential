from navigation_potential.environments.grid_world import Grid2D, Grid3D
import numpy as np


class RandomObstacle2D(Grid2D):
    def __init__(self, grid_size: int, obstacle_density: float, seed=None):
        if obstacle_density < 0 or obstacle_density >= 1:
            raise ValueError("Obstacle density must be between 0 and 1")
        self.obstacle_density = obstacle_density
        super().__init__(grid_size, seed)

    def generate_grid(self) -> np.ndarray:
        return np.random.choice(
            [0, 1],
            size=self.dim * (self.grid_size,),
            p=[1 - self.obstacle_density, self.obstacle_density],
        )

    def __repr__(self):
        return f"{__class__.__name__}(grid_size={self.grid_size}, obstacle_density={self.obstacle_density}, seed={self.seed})"


class RandomObstacle3D(Grid3D):
    def __init__(self, grid_size: int, obstacle_density: float, seed=None):
        if obstacle_density < 0 or obstacle_density >= 1:
            raise ValueError("Obstacle density must be between 0 and 1")
        self.obstacle_density = obstacle_density
        super().__init__(grid_size, seed)

    def generate_grid(self) -> np.ndarray:
        return np.random.choice(
            [0, 1],
            size=self.dim * (self.grid_size,),
            p=[1 - self.obstacle_density, self.obstacle_density],
        )

    def __repr__(self):
        return f"{__class__.__name__}(grid_size={self.grid_size}, obstacle_density={self.obstacle_density}, seed={self.seed})"


# Example usage:
if __name__ == "__main__":
    # Create a 2D grid world with obstacles
    grid_2d = RandomObstacle2D(grid_size=5, obstacle_density=0.5, seed=None)
    print("2D Grid World:")
    grid_2d.print_grid()
    print("\n")

    # Create a 3D grid world with obstacles
    grid_3d = RandomObstacle3D(grid_size=5, obstacle_density=0.5, seed=None)
    print("3D Grid World:")
    grid_3d.print_grid()

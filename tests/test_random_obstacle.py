import pytest
import numpy as np
from navigation_potential.environments import RandomObstacleGrid


@pytest.fixture
def random_obstacle_2d():
    return RandomObstacleGrid(grid_size=(100,100), obstacle_density=0.5, seed=0)


@pytest.fixture
def random_obstacle_3d():
    return RandomObstacleGrid(grid_size=(5,100,150), obstacle_density=0.5, seed=0)


def test_init(
    random_obstacle_2d: RandomObstacleGrid, random_obstacle_3d: RandomObstacleGrid
):
    assert random_obstacle_2d.obstacle_density == 0.5
    assert random_obstacle_3d.obstacle_density == 0.5


def test_generate_grid(
    random_obstacle_2d: RandomObstacleGrid, random_obstacle_3d: RandomObstacleGrid
):
    grid_2d = random_obstacle_2d.generate_grid()
    assert isinstance(grid_2d, np.ndarray)
    assert grid_2d.shape == (100, 100)
    grid_3d = random_obstacle_3d.generate_grid()
    assert isinstance(grid_3d, np.ndarray)
    assert grid_3d.shape == (5, 100, 150)


def test_obstacle_density(
    random_obstacle_2d: RandomObstacleGrid, random_obstacle_3d: RandomObstacleGrid
):
    grid_2d = random_obstacle_2d.generate_grid()
    obstacle_count = np.count_nonzero(grid_2d == 1)
    assert np.isclose(obstacle_count / np.prod(grid_2d.shape), 0.5, atol=0.1)
    grid_3d = random_obstacle_3d.generate_grid()
    obstacle_count = np.count_nonzero(grid_3d == 1)
    assert np.isclose(obstacle_count / np.prod(grid_3d.shape), 0.5, atol=0.1)


def test_obstacle_density_edge_cases():
    grid = RandomObstacleGrid(grid_size=(5,5), obstacle_density=0.0, seed=0)
    grid_2d = grid.generate_grid()
    assert np.count_nonzero(grid_2d == 1) == 0
    grid = RandomObstacleGrid(grid_size=(5,5), obstacle_density=0.9, seed=10)
    grid_2d = grid.generate_grid()
    obstacle_count = np.count_nonzero(grid_2d == 1)
    assert np.isclose(obstacle_count / (5**2), 0.9, atol=0.1)
    grid = RandomObstacleGrid(grid_size=(5, 5 , 5), obstacle_density=0.0, seed=0)
    grid_3d = grid.generate_grid()
    assert np.count_nonzero(grid_3d == 1) == 0
    grid = RandomObstacleGrid(grid_size=(5,5,5), obstacle_density=0.9, seed=10)
    grid_3d = grid.generate_grid()
    obstacle_count = np.count_nonzero(grid_3d == 1)
    assert np.isclose(obstacle_count / (5**3), 0.9, atol=0.1)


def test_obstacle_density_out_of_range():
    with pytest.raises(ValueError):
     RandomObstacleGrid(grid_size=(5,5), obstacle_density=-0.5)
    with pytest.raises(ValueError):
     RandomObstacleGrid(grid_size=(5,5), obstacle_density=1.5)
    with pytest.raises(ValueError):
        RandomObstacleGrid(grid_size=(5,5,5), obstacle_density=-0.5)
    with pytest.raises(ValueError):
        RandomObstacleGrid(grid_size=(5,5,5), obstacle_density=1.5)

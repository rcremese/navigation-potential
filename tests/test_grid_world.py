import pytest
from navigation_potential.environments.grid_world import Grid2D, Grid3D
import numpy as np


@pytest.fixture
def grid_2d():
    return Grid2D(grid_size=5, seed=42)


@pytest.fixture
def grid_3d():
    return Grid3D(grid_size=3, seed=42)


def test_initialization(grid_2d, grid_3d):
    assert grid_2d.grid_size == 5
    assert grid_3d.grid_size == 3
    assert grid_2d.dim == 2
    assert grid_3d.dim == 3


def test_grid_generation(grid_2d: Grid2D, grid_3d: Grid3D):
    assert isinstance(grid_2d.grid, np.ndarray)
    assert grid_2d.grid.shape == (5, 5)
    assert isinstance(grid_3d.grid, np.ndarray)
    assert grid_3d.grid.shape == (3, 3, 3)


def test_sample_position(grid_2d: Grid2D, grid_3d: Grid3D):
    pos_2d = grid_2d.sample_position()
    pos_3d = grid_3d.sample_position()
    assert pos_2d.shape == (2,)
    assert pos_3d.shape == (3,)


def test_set_obstacle(grid_2d: Grid2D, grid_3d: Grid3D):
    pos_2d = (1, 1)
    pos_3d = (1, 1, 1)

    grid_2d.set_obstacle(pos_2d)
    grid_3d.set_obstacle(pos_3d)
    assert grid_2d.grid[pos_2d] == 1
    assert grid_3d.grid[pos_3d] == 1
    assert grid_2d.is_obstacle(pos_2d)
    assert grid_3d.is_obstacle(pos_3d)


def test_print_grid(capsys, grid_2d: Grid2D, grid_3d: Grid3D):
    # This is more complex to test as it involves printing to [the console.
    # However, we can verify that it doesn't throw any exceptions.
    grid_2d.print_grid()
    captured = capsys.readouterr()
    assert captured.out
    grid_3d.print_grid()
    captured = capsys.readouterr()
    assert captured.out


def test_free_positions(grid_2d: Grid2D, grid_3d: Grid3D):
    free_2d = grid_2d.free_positions
    free_3d = grid_3d.free_positions
    assert isinstance(free_2d, np.ndarray)
    assert isinstance(free_3d, np.ndarray)


def test_edge_cases():
    # Test with a grid size of 1.
    with pytest.raises(ValueError):
        grid_2d = Grid2D(grid_size=0, seed=42)
        grid_3d = Grid3D(grid_size=0, seed=42)

    # Test setting an obstacle at a position that is out of bounds.
    grid_2d = Grid2D(grid_size=5, seed=42)
    with pytest.raises(IndexError):
        grid_2d.set_obstacle((5, 5))

    grid_3d = Grid3D(grid_size=3, seed=42)
    with pytest.raises(IndexError):
        grid_3d.set_obstacle((3, 2, 1))

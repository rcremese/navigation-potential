from navigation_potential.environments import GridWorld

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d_environment(env : GridWorld) -> plt.Figure:
    """
    Plot a 2D environment with a goal, an initial position, and obstacles.

    Args:
    - grid (numpy array): A 2D grid representing the environment, where 1s indicate obstacles.
    - initial_position (tuple): The coordinates of the initial position.
    - goal_position (tuple): The coordinates of the goal position.
    """
    assert env.dim == 2, "This function only supports 2D environments"
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the free space
    ax.add_patch(patches.Rectangle((0, 0), env.grid_size[0], env.grid_size[1], edgecolor='black', facecolor='lightgreen'))

    # Plot the obstacles
    for i, j in env.obstacle_positions:
        ax.add_patch(patches.Rectangle((i, j), 1, 1, edgecolor='black', facecolor='red'))

    # Plot the initial position
    ax.plot(env.initial_pos[0] + 0.5, env.initial_pos[1] + 0.5, 'bo', markersize=10, label='Initial position')

    # Plot the goal position
    ax.plot(env.goal_pos[0] + 0.5, env.goal_pos[1] + 0.5, 'go', markersize=10, label='Goal position')

    # Set the title and labels
    ax.set_title('2D Environment')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set the limits of the axes to the size of the grid
    ax.set_xlim(0, env.grid_size[0])
    ax.set_ylim(0, env.grid_size[1])
    ax.set_aspect('equal')

    # Add a legend
    ax.legend()

    return fig


def visualize_3d_environment(env : GridWorld) -> plt.Figure:
    """
    Plot a 3D environment with a goal, an initial position, and obstacles.

    Args:
    - grid (3D numpy array): A 3D grid representing the environment, where 1s indicate obstacles.
    - initial_position (tuple): The coordinates of the initial position.
    - goal_position (tuple): The coordinates of the goal position.
    """
    assert env.dim == 3, "The environment must be 3D"
    # Create a figure and axis
    fig = plt.figure()
    ax : Axes3D = fig.add_subplot(111, projection='3d')

    # Plot the obstacles as voxels
    ax.voxels(env.grid, edgecolors='k', facecolors='red', alpha=0.5)

    # Plot the initial position
    ax.scatter(*env.initial_pos, c='b', marker='o', s=100, label='Initial position')

    # Plot the goal position
    ax.scatter(*env.goal_pos, c='g', marker='o', s=100, label='Goal position')

    # Set the title and labels
    ax.set_title('3D Environment')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, env.grid_size[0])
    ax.set_ylim(0, env.grid_size[1])
    ax.set_zlim(0, env.grid_size[2])

    # Add a legend
    ax.legend()
    return fig
    
def visualize_environment(env : GridWorld) -> plt.Figure:
    if env.dim == 2:
        fig = visualize_2d_environment(env)
    elif env.dim == 3:
        fig = visualize_3d_environment(env)
    else:
        raise NotImplementedError("Only 2D and 3D environments are supported for now")
    return fig

if __name__ == "__main__":
    from navigation_potential.environments import RandomObstacleGrid
    # Create a random obstacle grid environment in 2D
    env_2d = RandomObstacleGrid(grid_size=(20,10), obstacle_density=0.3, seed=42)
    print("Initial position:", env_2d.initial_pos, "Goal position:", env_2d.goal_pos)
    fig1 = visualize_environment(env_2d)
    
    env_3d = RandomObstacleGrid(grid_size=(20,10,10), obstacle_density=0.1, seed=42)
    print("Initial position:", env_3d.initial_pos, "Goal position:", env_3d.goal_pos)
    fig2 = visualize_environment(env_3d)
    plt.show()

from navigation_potential.environments import GridEnvironment
from navigation_potential.fields import ScalarField, VectorField

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

class Visualizer:
    """Class for visualizing environments, vector fields, and trajectories."""
    
    def __init__(self, environment : GridEnvironment):
        """Initialize with environment."""
        assert isinstance(environment, GridEnvironment), "Environment must be an instance of GridEnvironment"
        self.env = environment

    def plot_environment(self) -> plt.Axes:
        """Plot the environment grid."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot obstacles
        ax.imshow(self.env.grid, cmap='binary', origin='lower')
        
        # Plot start and target
        ax.plot(self.env.start[1], self.env.start[0], 'bo', markersize=12, label='Start')
        ax.plot(self.env.target[1], self.env.target[0], 'go', markersize=12, label='Target')
        ax.legend()
        return ax
    
    def plot_scalar_field(self, scalar_field : ScalarField, title : str | None =None, ax : plt.Axes | None =None, alpha : float=0.7) -> plt.Axes:
        """Plot a scalar field."""
        if ax is None:
            ax = self.plot_environment()
        
        # Plot scalar field
        if scalar_field.field is not None:
            cmap = 'inferno'
            im = ax.imshow(scalar_field.field, origin='lower', cmap=cmap, alpha=alpha)
            plt.colorbar(im, ax=ax, label='Field Value')
        
        if title:
            ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
        
        return ax
    
    def plot_vector_field(self, vector_field : VectorField, title : str | None = None, ax : plt.Axes | None =None) -> plt.Axes:
        """Plot a vector field."""
        if ax is None:
            ax = self.plot_environment()
        # Plot vector field
        step = max(1, self.env.size // 25)  # Sample the vector field for clearer visualization
        y, x = np.mgrid[0:self.env.size:step, 0:self.env.size:step]
        fx_sampled = vector_field.fx[::step, ::step]
        fy_sampled = vector_field.fy[::step, ::step]
        
        # Color arrows by magnitude for better visualization
        magnitude = np.sqrt(fx_sampled**2 + fy_sampled**2)
        norm = Normalize()
        norm.autoscale(magnitude)
        
        ax.quiver(x, y, fx_sampled, fy_sampled, magnitude, cmap='inferno', norm=norm, scale=25, width=0.002)
                
        if title:
            ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
        
        return ax
            
    
    def plot_scalar_and_vector_fields(self, scalar_field : ScalarField, vector_field : VectorField, title : str | None = None, ax : plt.Axes | None =None) -> plt.Axes:
        """Plot scalar and vector fields."""
        if ax is None:
            ax = self.plot_environment()
        ax = self.plot_scalar_field(scalar_field,  title=title, ax=ax)
        ax = self.plot_vector_field(vector_field,  title=title, ax=ax)
        return ax

    
    def plot_trajectory(self, trajectory : np.ndarray, success : bool, label : str | None =None, ax : plt.Axes | None =None) -> plt.Axes:
        """Plot a trajectory."""
        if ax is None:
            ax = self.plot_environment()
        # Calculate trajectory length
        length = np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)) if len(trajectory) > 1 else 0
        
        # Plot trajectory
        if label:
            label = f'{label} ({"Success" if success else "Failed"}, Length: {length:.1f})'
        else:
            label = f'Trajectory ({"Success" if success else "Failed"}, Length: {length:.1f})'
            
        ax.plot(trajectory[:, 1], trajectory[:, 0], color='green', linewidth=2, label=label)
        
        ax.legend()
        ax.set_aspect('equal')
        
        return ax
    
    def compare_trajectories(self, trajectories : list[np.ndarray], labels : list[str] | None =None, success_flags : list[bool] | None =None, colors : list[str] | None =None) -> plt.Axes:
        """Compare multiple trajectories."""
        ax = self.plot_environment()
        # Default colors if not provided
        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        # Plot trajectories
        for i, trajectory in enumerate(trajectories):
            label = labels[i] if labels and i < len(labels) else f'Trajectory {i+1}'
            success = success_flags[i] if success_flags and i < len(success_flags) else True
            color = colors[i % len(colors)]
            
            length = np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)) if len(trajectory) > 1 else 0
            ax.plot(trajectory[:, 1], trajectory[:, 0], color, linewidth=2, 
                   label=f'{label} ({"Success" if success else "Failed"}, Length: {length:.1f})')
        
        ax.set_title("Trajectory Comparison")
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        return ax

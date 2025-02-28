from navigation_potential.environments import GridEnvironment
from navigation_potential.fields import ScalarField, VectorField

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

class Visualizer:
    """Class for visualizing environments, fields, and trajectories."""

    def __init__(self, environment : GridEnvironment):
        """Initialize with environment."""
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


    def plot_scalar_field(self, scalar_field : ScalarField, title=None, ax=None, alpha=0.7, cmap="inferno"):
        """
        Plot a scalar field.

        Args:
            scalar_field: ScalarField object to visualize.
            title: Title for the plot.
            ax: Matplotlib axes for plotting (creates new figure if None).
            alpha: Transparency of the scalar field.
            cmap: Colormap to use.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        if ax is None:
            ax = self.plot_environment()

        # Plot scalar field with proper scaling
        field_extent = [scalar_field.min_bounds[1], scalar_field.max_bounds[1], scalar_field.min_bounds[0], scalar_field.max_bounds[0]]
        im = ax.imshow(scalar_field.values, origin='lower', cmap=cmap, alpha=alpha, extent=field_extent)
        plt.colorbar(im, ax=ax, label='Field Value')

        if title:
            ax.set_title(f"{title} ({scalar_field.shape})")
        ax.legend()
        ax.set_aspect('equal')
        return ax

    def plot_vector_field(self, vector_field : VectorField, title=None, ax=None, density=25, color='black'):
        """
        Plot a vector field.

        Args:
            vector_field: VectorField object to visualize.
            title: Title for the plot.
            ax: Matplotlib axes for plotting (creates new figure if None).
            density: Density of vectors to display (higher = fewer vectors).
            color: Color of the vectors.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        if ax is None:
            ax = self.plot_environment()
        # Plot vector field
        field_height, field_width = vector_field.shape
        step = max(1, min(field_height, field_width) // density)

        # Create grid scaled to environment size
        scale_y = self.env.size / field_height
        scale_x = self.env.size / field_width

        y_indices = np.arange(0, field_height, step)
        x_indices = np.arange(0, field_width, step)
        Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')

        # Scale grid points to environment coordinates
        Y_env = Y * scale_y
        X_env = X * scale_x

        # Sample vector field
        fx_sampled = vector_field.fx[::step, ::step]
        fy_sampled = vector_field.fy[::step, ::step]

        # Color arrows by magnitude for better visualization
        magnitude = np.sqrt(fx_sampled**2 + fy_sampled**2)
        norm = Normalize()
        norm.autoscale(magnitude)

        ax.quiver(X_env, Y_env, fx_sampled, fy_sampled, magnitude,
                 cmap='inferno', norm=norm, scale=25, width=0.002, pivot='mid')

        if title:
            ax.set_title(f"{title} ({vector_field.shape})")
        ax.legend()
        ax.set_aspect('equal')

        return ax

    def plot_scalar_and_vector_field(self, scalar_field, vector_field, title=None, ax=None):
        """
        Plot both scalar field and its derived vector field.

        Args:
            scalar_field: ScalarField object.
            vector_field: VectorField object.
            title: Title for the plot.
            ax: Matplotlib axes for plotting (creates new figure if None).

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        if ax is None:
            ax = self.plot_environment()

        # Plot scalar field
        self.plot_scalar_field(scalar_field, title=None, ax=ax, alpha=0.5)

        # Overlay vector field
        self.plot_vector_field(vector_field, title=None, ax=ax, density=20)

        if title:
            ax.set_title(f"{title} ({scalar_field.shape})")

        return ax

    def plot_trajectory(self, trajectory, success, label=None, ax=None, color='b'):
        """
        Plot a trajectory.

        Args:
            trajectory: Trajectory coordinates.
            success: Whether the trajectory reached the target.
            label: Label for the plot.
            ax: Matplotlib axes for plotting.
            color: Color of the trajectory.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        if ax is None:
            ax = self.plot_environment()
        # Calculate trajectory length
        length = np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)) if len(trajectory) > 1 else 0

        # Plot trajectory
        if label:
            label = f'{label} ({"Success" if success else "Failed"}, Length: {length:.1f})'
        else:
            label = f'Trajectory ({"Success" if success else "Failed"}, Length: {length:.1f})'

        ax.plot(trajectory[:, 1], trajectory[:, 0], color='-', linewidth=2, label=label)

        ax.legend()
        ax.set_aspect('equal')

        return ax

    def compare_trajectories(self, trajectories, labels=None, success_flags=None, colors=None):
        """
        Compare multiple trajectories.

        Args:
            trajectories: List of trajectory arrays.
            labels: List of labels for each trajectory.
            success_flags: List of success flags for each trajectory.
            colors: List of colors for each trajectory.

        Returns:
            matplotlib.axes.Axes: The plot axes.
        """
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot background
        ax.imshow(self.env.grid, cmap='binary', origin='lower', extent=[0, self.env.size, 0, self.env.size])

        # Plot start and target
        ax.plot(self.env.start[1], self.env.start[0], 'go', markersize=12, label='Start')
        ax.plot(self.env.target[1], self.env.target[0], 'ro', markersize=12, label='Target')

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

    def compare_scalar_fields(self, fields : list[ScalarField], labels : list[str]=None, title=None, cmaps=None):
        """
        Compare multiple scalar fields.

        Args:
            fields: List of ScalarField objects.
            labels: List of labels for each field.
            title: Overall title for the plot.
            cmaps: List of colormaps for each field.

        Returns:
            matplotlib.figure.Figure: The figure with all plots.
        """
        n_fields = len(fields)
        cols = min(3, n_fields)
        rows = (n_fields + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_fields == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Default colormaps if not provided
        if cmaps is None:
            cmaps = ['inferno'] * n_fields
        elif(isinstance(cmaps, str)):
            cmaps = [cmaps] * n_fields
        elif(len(cmaps) < n_fields):
            cmaps.extend(['inferno'] * (n_fields - len(cmaps)))

        # Plot each field
        for i, field in enumerate(fields):
            if i < len(axes):
                ax = axes[i]
                label = labels[i] if labels and i < len(labels) else f'Field {i+1}'
                cmap = cmaps[i] if cmaps and i < len(cmaps) else 'inferno'

                # Plot field
                im = ax.imshow(field.values, origin='lower', cmap=cmap,
                              extent=[0, self.env.size, 0, self.env.size])
                plt.colorbar(im, ax=ax)

                # Add field details
                res_text = f"Resolution: {field.shape}"
                ax.set_title(f"{label}\n{res_text}")
                ax.set_aspect('equal')

        # Hide unused subplots
        for i in range(n_fields, len(axes)):
            axes[i].axis('off')

        if title:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)

        return fig

    def compare_vector_fields(self, fields : list[VectorField], labels=None, title=None, density=15):
        """
        Compare multiple vector fields.

        Args:
            fields: List of VectorField objects.
            labels: List of labels for each field.
            title: Overall title for the plot.
            density: Density of vectors to display.

        Returns:
            matplotlib.figure.Figure: The figure with all plots.
        """
        n_fields = len(fields)
        cols = min(3, n_fields)
        rows = (n_fields + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_fields == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each field
        for i, field in enumerate(fields):
            if i < len(axes):
                ax = axes[i]
                label = labels[i] if labels and i < len(labels) else f'Field {i+1}'

                # Plot environment
                ax.imshow(self.env.grid, cmap='binary', origin='lower',
                        extent=[0, self.env.size, 0, self.env.size])

                # Plot vector field
                field_height, field_width = field.shape
                step = max(1, min(field_height, field_width) // density)

                # Create grid scaled to environment size
                scale_y = self.env.size / field_height
                scale_x = self.env.size / field_width

                y_indices = np.arange(0, field_height, step)
                x_indices = np.arange(0, field_width, step)
                Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')

                # Scale grid points to environment coordinates
                Y_env = Y * scale_y
                X_env = X * scale_x

                # Sample vector field
                fx_sampled = field.fx[::step, ::step]
                fy_sampled = field.fy[::step, ::step]

                # Plot the vector field
                magnitude = np.sqrt(fx_sampled**2 + fy_sampled**2)
                ax.quiver(X_env, Y_env, fx_sampled, fy_sampled, magnitude,
                         cmap='inferno', scale=25, width=0.002)

                # Add field details
                res_text = f"Resolution: {field.shape}"
                ax.set_title(f"{label}\n{res_text}")
                ax.set_aspect('equal')

        # Hide unused subplots
        for i in range(n_fields, len(axes)):
            axes[i].axis('off')

        if title:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)

        return fig

from scipy.interpolate import RegularGridInterpolator
import numpy as np
from abc import ABC, abstractmethod

class SampledField(ABC):
    """Abstract base class for sampled fields."""
    def __init__(self, min_bounds, max_bounds, resolution):
        """
        Initialize the sampled field.

        Args:
            min_bounds (np.ndarray): Minimum bounds of the field.
            max_bounds (np.ndarray): Maximum bounds of the field.
            resolution (np.ndarray): Resolution of the field in each dimension.
        """
        self.min_bounds = np.array(min_bounds)
        self.max_bounds = np.array(max_bounds)
        self.resolution = np.array(resolution)
        self.shape = tuple(resolution)
        self._interpolator = None

    @abstractmethod
    def at(self, position : np.ndarray) -> float:
        """Get interpolated scalar value at a position."""
        pass

    @abstractmethod
    def resample(self, factor : int) -> 'SampledField':
        """Resample the field at a new resolution."""
        pass

class ScalarField(SampledField):
    def __init__(self, min_bounds, max_bounds, resolution):
        """
        Initialize the sampled field.

        Args:
            min_bounds (np.ndarray): Minimum bounds of the field.
            max_bounds (np.ndarray): Maximum bounds of the field.
            resolution (np.ndarray): Resolution of the field in each dimension.
        """
        super().__init__(min_bounds, max_bounds, resolution)
        self.values = np.zeros(self.shape)

    """Class representing a scalar field."""
    def at(self, position) -> float:
        """Get interpolated scalar value at a position."""
        if self._interpolator is None:
            self._create_interpolator()

        normalized_pos = (position - self.min_bounds) / (self.max_bounds - self.min_bounds)
        return self._interpolator(normalized_pos)[0]

    def resample(self, factor) -> 'ScalarField':
        """
        Resample the field at a new resolution.

        Args:
            factor (int): Positive for upsampling, negative for downsampling.
                          The absolute value is the scaling factor.

        Returns:
            ScalarField: A new scalar field with resampled resolution.
        """
        if factor == 0:
            return self

        if factor > 0:
            # Upsampling
            new_resolution = self.resolution * factor
        else:
            # Downsampling
            new_resolution = self.resolution // abs(factor)
            # Ensure resolution is at least 1 in each dimension
            new_resolution = np.maximum(new_resolution, 1)

        new_field = ScalarField(self.min_bounds, self.max_bounds, new_resolution)

        # Create coordinates for original and new grids
        old_coords = [np.linspace(0, 1, dim) for dim in self.shape]
        new_coords = [np.linspace(0, 1, dim) for dim in new_field.shape]

        # Create interpolator for the original field
        interp = RegularGridInterpolator(old_coords, self.values)

        # Create meshgrid for new coordinates
        if len(new_field.shape) == 2:
            Y, X = np.meshgrid(new_coords[0], new_coords[1], indexing='ij')
            points = np.column_stack((Y.ravel(), X.ravel()))
            new_field.values = interp(points).reshape(new_field.shape)
        elif len(new_field.shape) == 3:
            Z, Y, X = np.meshgrid(new_coords[0], new_coords[1], new_coords[2], indexing='ij')
            points = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))
            new_field.values = interp(points).reshape(new_field.shape)
        else:
            raise ValueError("Resampling only supported for 2D and 3D fields")

        return new_field

    def _create_interpolator(self):
        """Create interpolator for the scalar field."""
        coords = [np.linspace(0, 1, dim) for dim in self.shape]
        self._interpolator = RegularGridInterpolator(
            coords, self.values, bounds_error=False, fill_value=None
        )

class VectorField(SampledField):
    """Class representing a vector field."""

    def __init__(self, min_bounds, max_bounds, resolution):
        """
        Initialize the vector field.

        Args:
            min_bounds (np.ndarray): Minimum bounds of the field.
            max_bounds (np.ndarray): Maximum bounds of the field.
            resolution (np.ndarray): Resolution of the field in each dimension.
        """
        super().__init__(min_bounds, max_bounds, resolution)

        # Initialize field components
        self.fx = np.zeros(self.shape)
        self.fy = np.zeros(self.shape)


    def at(self, position):
        """
        Get interpolated vector at a position.

        Args:
            position: Position to evaluate the vector field at.

        Returns:
            tuple: (vx, vy) vector components at the position.
        """
        if self._interpolators is None:
            self._create_interpolators()

        normalized_pos = (position - self.min_bounds) / (self.max_bounds - self.min_bounds)
        vx = self._interpolators[0](normalized_pos)[0]
        vy = self._interpolators[1](normalized_pos)[0]

        return vx, vy

    def resample(self, factor):
        """
        Resample the field at a new resolution.

        Args:
            factor (int): Positive for upsampling, negative for downsampling.
                          The absolute value is the scaling factor.

        Returns:
            VectorField: A new vector field with resampled resolution.
        """
        if factor == 0:
            return self

        if factor > 0:
            # Upsampling
            new_resolution = self.resolution * factor
        else:
            # Downsampling
            new_resolution = self.resolution // abs(factor)
            # Ensure resolution is at least 1 in each dimension
            new_resolution = np.maximum(new_resolution, 1)

        new_field = VectorField(self.min_bounds, self.max_bounds, new_resolution)

        # Create coordinates for original and new grids
        old_coords = [np.linspace(0, 1, dim) for dim in self.shape]
        new_coords = [np.linspace(0, 1, dim) for dim in new_field.shape]

        # Create interpolators for the original field components
        fx_interp = RegularGridInterpolator(old_coords, self.fx)
        fy_interp = RegularGridInterpolator(old_coords, self.fy)

        # Create meshgrid for new coordinates
        if len(new_field.shape) == 2:
            Y, X = np.meshgrid(new_coords[0], new_coords[1], indexing='ij')
            points = np.column_stack((Y.ravel(), X.ravel()))
            new_field.fx = fx_interp(points).reshape(new_field.shape)
            new_field.fy = fy_interp(points).reshape(new_field.shape)
        elif len(new_field.shape) == 3:
            Z, Y, X = np.meshgrid(new_coords[0], new_coords[1], new_coords[2], indexing='ij')
            points = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))
            new_field.fx = fx_interp(points).reshape(new_field.shape)
            new_field.fy = fy_interp(points).reshape(new_field.shape)
        else:
            raise ValueError("Resampling only supported for 2D and 3D fields")

        return new_field

    def _create_interpolators(self):
        """Create interpolators for the vector field components."""
        coords = [np.linspace(0, 1, dim) for dim in self.shape]
        self._interpolators = [
            RegularGridInterpolator(coords, self.fx, bounds_error=False, fill_value=None),
            RegularGridInterpolator(coords, self.fy, bounds_error=False, fill_value=None)
        ]


def create_vector_field_from_scalar(scalar_field : ScalarField, normalize : bool = False) -> VectorField:
    """
    Compute a vector field from a scalar field's gradient.

    Args:
        scalar_field: ScalarField to compute gradient from.

    Returns:
        VectorField: Vector field representing the gradient.
    """
    # Create a vector field with the same parameters
    vector_field = VectorField(
        scalar_field.min_bounds,
        scalar_field.max_bounds,
        scalar_field.resolution
    )

    # Compute the gradient (negative for fields where lower values are better)
    fy, fx = np.gradient(-scalar_field.values)

    # Normalize vector field
    if normalize:
        magnitude = np.sqrt(fx**2 + fy**2)
        magnitude[magnitude < 1e-10] = 1.0  # Avoid division by zero
    else:
        magnitude = np.ones_like(fx)

    vector_field.fx = fx / magnitude
    vector_field.fy = fy / magnitude

    return vector_field

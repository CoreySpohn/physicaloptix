"""Private geometry extraction for square separable propagation kernels."""

import numpy as np


def _grid_axis(grid):
    """Host x coordinates for a centred square, isotropic separable grid.

    Host evaluation preserves float64 construction of Fourier kernels even
    when JAX is configured for float32. General affine grids are resampled
    at image boundaries, not passed to the separable propagation kernels.
    """
    if grid.shape[0] != grid.shape[1] or (
        grid.center_x != (grid.shape[1] - 1) / 2
        or grid.center_y != (grid.shape[0] - 1) / 2
    ):
        raise ValueError("separable propagation requires a centred square grid")
    return (np.arange(grid.shape[1]) - grid.center_x) * grid.step

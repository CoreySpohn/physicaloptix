"""Geometry extraction for square separable propagation kernels."""

import numpy as np


def grid_axis(grid):
    """Host x coordinates for a centred square, isotropic separable grid.

    Sampling decisions use host coordinates. Propagation imports this axis
    at the field precision when it builds a Fourier kernel. General affine
    grids are resampled at image boundaries.

    Args:
        grid: A centred square ``PixelGrid`` with equal steps on both axes.

    Returns:
        The host x coordinates, a NumPy array of ``grid.shape[1]`` samples.

    Raises:
        ValueError: The grid is not a centred square.
    """
    if grid.shape[0] != grid.shape[1] or (
        grid.center_x != (grid.shape[1] - 1) / 2
        or grid.center_y != (grid.shape[0] - 1) / 2
    ):
        raise ValueError("separable propagation requires a centred square grid")
    return (np.arange(grid.shape[1]) - grid.center_x) * grid.step

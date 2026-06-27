"""Render optixstuff primaries into dLux apertures (the optixstuff -> dLux seam).

This is where the (unavoidable) translation from an optixstuff hardware
description to a dLux object lives -- once, dispatched by primary type, so callers
work with optixstuff objects and never construct dLux apertures by hand.
"""

import functools

import dLux as dl
import numpy as np
from optixstuff import SegmentedPrimary, SimplePrimary


@functools.singledispatch
def to_dlux_aperture(primary):
    """Render an optixstuff primary into a dLux aperture layer.

    Register a new primary type with ``@to_dlux_aperture.register`` rather than
    branching here -- O(primary types), not a growing if/elif.

    Args:
        primary: An :class:`optixstuff.AbstractPrimary` concrete instance.

    Returns:
        A dLux aperture layer (e.g. ``MultiAperture`` or ``CircularAperture``).
    """
    raise NotImplementedError(f"no dLux adapter for {type(primary).__name__}")


@to_dlux_aperture.register
def _(primary: SegmentedPrimary):
    centres = np.asarray(primary.segment_centres_m)
    seg_rmax = float(primary.segment_flat_to_flat_m) / np.sqrt(3.0)
    segments = []
    for x, y in centres:
        transform = dl.CoordTransform(
            translation=[float(x), float(y)], rotation=float(np.pi / 6)
        )  # flat-top hexagons
        segments.append(
            dl.RegPolyAperture(
                nsides=6, rmax=seg_rmax, transformation=transform, softening=1.0
            )
        )
    return dl.MultiAperture(segments, normalise=True)


@to_dlux_aperture.register
def _(primary: SimplePrimary):
    # no segment geometry -> model the circumscribing circle
    return dl.CircularAperture(radius=float(primary.diameter_m) / 2.0, normalise=True)

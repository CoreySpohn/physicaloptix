"""Pupil geometry: the YAML loader and the segmented-aperture rasterizer.

``load_primary_yaml`` reads a small pupil-geometry YAML (circumscribed and
inscribed diameters plus a hexagonal segmentation block) into an optixstuff
``SegmentedPrimary``, deriving the collecting area from the segment geometry.
The EAC-1 parameters ship with the package (``eac1_primary``), sourced from
the design-survey baseline pupil FITS headers.

``rasterize_primary`` renders the segmented aperture to a gray-pixel pupil
array using the same conventions as the design-survey baseline pupils: a
half-pixel-offset pixel grid spanning the circumscribed diameter, flat-top
hexagonal segments tested with inclusive apothem half-planes, and mean
supersampling on a half-offset subpixel lattice. Built once in numpy at
construction time; convert to a jax array (and normalize) at use.
"""

from pathlib import Path

import numpy as np
import yaml
from optixstuff import SegmentedPrimary

_EAC1_YAML = Path(__file__).parent / "data" / "eac1_primary.yaml"


def load_primary_yaml(path):
    """Load a pupil-geometry YAML into a ``SegmentedPrimary``.

    The schema::

        name: EAC-1
        circumscribed_diameter_m: 7.2
        inscribed_diameter_m: 5.96        # optional
        segmentation:
          type: hexagonal
          n_rings: 2
          segment_point_to_point_m: 1.65
          gap_m: 0.004

    The collecting area is derived from the segment geometry
    (``n_segments * (3 sqrt(3) / 8) * point_to_point**2``), so every number
    in the primary traces to the YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        The ``SegmentedPrimary``.
    """
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    segmentation = config["segmentation"]
    if segmentation.get("type") != "hexagonal":
        raise ValueError(
            "only hexagonal segmentation is supported, got "
            f"{segmentation.get('type')!r}"
        )
    n_rings = int(segmentation["n_rings"])
    n_segments = 1 + 3 * n_rings * (n_rings + 1)
    point_to_point = float(segmentation["segment_point_to_point_m"])
    hex_area = 3.0 * np.sqrt(3.0) / 8.0 * point_to_point**2

    inscribed = config.get("inscribed_diameter_m")
    return SegmentedPrimary(
        diameter_m=float(config["circumscribed_diameter_m"]),
        area_m2=n_segments * hex_area,
        n_rings=n_rings,
        n_segments=n_segments,
        segment_gap_m=float(segmentation["gap_m"]),
        segment_point_to_point_m=point_to_point,
        inscribed_diameter_m=None if inscribed is None else float(inscribed),
    )


def eac1_primary():
    """The EAC-1 segmented primary from the bundled geometry YAML."""
    return load_primary_yaml(_EAC1_YAML)


def _segment_setup(primary, npix, extent_m, supersample):
    """Shared grid, subpixel offsets, and hexagon geometry for rasterizing."""
    if primary.segment_point_to_point_m is None:
        raise ValueError(
            "rasterizing segments needs an exact segment size: set "
            "segment_point_to_point_m on the primary"
        )
    extent = float(extent_m) if extent_m is not None else float(primary.diameter_m)
    delta = extent / npix
    coords = (np.arange(npix) - npix / 2 + 0.5) * delta
    offsets = ((np.arange(supersample) + 0.5) / supersample - 0.5) * delta
    circum = float(primary.segment_point_to_point_m)
    apothem = np.cos(np.pi / 6) * circum / 2
    thetas = np.pi / 2 + np.arange(3) * np.pi / 3
    centres = np.asarray(primary.segment_centres_m, dtype=float)
    half_box = circum / 2 + delta
    return coords, offsets, apothem, thetas, centres, half_box


def _segment_block(cx, cy, coords, offsets, apothem, thetas, half_box):
    """The gray coverage of one flat-top hexagon over its bounding window.

    Returns ``(iy, ix, block)`` (the window row/col indices and the mean
    subpixel coverage on that window), or ``None`` if the segment falls off
    the grid.
    """
    ix = np.flatnonzero(np.abs(coords - cx) <= half_box)
    iy = np.flatnonzero(np.abs(coords - cy) <= half_box)
    if not ix.size or not iy.size:
        return None
    block = np.zeros((iy.size, ix.size))
    for dy in offsets:
        y = coords[iy] + dy - cy
        for dx in offsets:
            x = coords[ix] + dx - cx
            inside = np.ones((iy.size, ix.size), dtype=bool)
            for theta in thetas:
                projection = (np.cos(theta) * x)[np.newaxis, :] + (np.sin(theta) * y)[
                    :, np.newaxis
                ]
                inside &= projection**2 <= apothem**2
            block += inside
    return iy, ix, block / offsets.size**2


def rasterize_primary(primary, npix, *, extent_m=None, supersample=16):
    """Render a segmented primary to a gray-pixel pupil array.

    Args:
        primary: A ``SegmentedPrimary`` with an exact segment size
            (``segment_point_to_point_m``).
        npix: Output array side length in pixels.
        extent_m: Spatial extent of the array in metres; defaults to the
            circumscribed diameter, matching the design-survey pupils.
        supersample: Subpixel samples per axis (16 matches the survey).

    Returns:
        The pupil as an ``(npix, npix)`` float array in [0, 1].
    """
    coords, offsets, apothem, thetas, centres, half_box = _segment_setup(
        primary, npix, extent_m, supersample
    )
    pupil = np.zeros((npix, npix))
    for cx, cy in centres:
        rendered = _segment_block(cx, cy, coords, offsets, apothem, thetas, half_box)
        if rendered is None:
            continue
        iy, ix, block = rendered
        pupil[np.ix_(iy, ix)] += block
    return pupil


def rasterize_segments(primary, npix, *, extent_m=None, supersample=16):
    """Render each segment to its own gray-pixel mask.

    Like ``rasterize_primary`` but returns the per-segment stack rather than
    the summed pupil, so a segment-local basis (piston, tip, tilt) can key
    each mode to a single segment's pixels. The stack sums exactly to
    ``rasterize_primary``, and for a nonzero gap the masks are disjoint.

    Args:
        primary: A ``SegmentedPrimary`` with an exact segment size.
        npix: Output side length in pixels.
        extent_m: Spatial extent in metres; defaults to the circumscribed
            diameter.
        supersample: Subpixel samples per axis.

    Returns:
        An ``(n_segments, npix, npix)`` float array in [0, 1], ordered as
        ``primary.segment_centres_m`` (centre segment first).
    """
    coords, offsets, apothem, thetas, centres, half_box = _segment_setup(
        primary, npix, extent_m, supersample
    )
    masks = np.zeros((len(centres), npix, npix))
    for seg, (cx, cy) in enumerate(centres):
        rendered = _segment_block(cx, cy, coords, offsets, apothem, thetas, half_box)
        if rendered is None:
            continue
        iy, ix, block = rendered
        masks[seg][np.ix_(iy, ix)] = block
    return masks


def normalize_unit_energy(pupil, dx_m):
    """Scale a pupil so its field carries unit energy (one photon).

    The design-survey convention: divide by
    ``sqrt(sum(|pupil|**2) * dx**2)`` so that
    ``sum(|E|**2 dx**2) == 1``.

    Args:
        pupil: The pupil amplitude array.
        dx_m: Pixel pitch in metres.

    Returns:
        The normalized pupil array.
    """
    energy = np.sum(np.abs(pupil) ** 2) * dx_m**2
    return pupil / np.sqrt(energy)

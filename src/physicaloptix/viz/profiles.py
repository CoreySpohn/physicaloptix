"""Radial contrast profiles: three statistical conventions, one binning.

``plot_contrast_profile`` is a BRIDGE, deliberately thin. eyepiece already
owns everything about drawing a contrast curve -- the line, the shaded
IWA/OWA exclusion regions, the labeled reference floors, and the
once-per-axes annotation bookkeeping that keeps a second call from
duplicating them. ``hwoutils.radial`` owns the ring geometry. physicaloptix
adds only what neither can know: how to pull a plottable 2D map and a
lambda/D pixel scale out of a ``Field``, and how to reduce the pixels in a
ring by a statistic other than the mean.

The three statistics share ONE per-pixel binning pass. Three incompatible
radial conventions are common in this kind of analysis code -- a
per-integer-pixel-radius ``bincount`` mean, a median over a fixed count of
equal-width annuli, and a fixed-``nbins`` mean -- and measured against each
other on the same map they disagree by tens of percent pervasively, not just
in the core. Hence ``stat`` re-aggregates one shared bin-index array rather
than dispatching to three implementations: ring geometry must be identical
across statistics, or "median vs mean" silently also means "one ring
definition vs another".
"""

import matplotlib.pyplot as plt
import numpy as np
from hwoutils.radial import radial_distance

from physicaloptix.core import Field
from physicaloptix.viz import _require
from physicaloptix.viz.fields import _intensity_from, _resolve

_STATS = ("mean", "median", "max")

# plot_radial/plot_contrast_curve are unit-agnostic by design and never guess
# an axis label. This function is not: pixscale_lod is lambda/D per pixel by
# contract, so the abscissa's units ARE known here and the label is set.
_LOD_SEPARATION_XLABEL = r"$r$ [$\lambda/D$]"


def _radial_bins(shape, nbins, center=None):
    """The shared ring geometry: bin centers in pixels and a per-pixel index.

    Replicates ``hwoutils.radial.radial_profile``'s own geometry -- its
    ``linspace(0, max_radius, nbins + 1)`` edge construction and its
    ``digitize``-then-clip pixel assignment -- rather than inventing a
    second one, so ``stat="mean"`` reproduces that function exactly (pinned
    by a test) and ``"median"``/``"max"`` differ from it ONLY in the
    reduction, never in which pixels fall in which ring.

    Args:
        shape: ``(ny, nx)`` of the map being profiled.
        nbins: Number of radial bins. None uses hwoutils' own default,
            ``max(ny, nx) // 2``.
        center: ``(cy, cx)`` center in index space. None uses
            ``radial_distance``'s geometric center, ``((ny-1)/2, (nx-1)/2)``.

    Returns:
        A ``(bin_centers, inds, nbins)`` triple. ``bin_centers`` are the
        ring centers in PIXELS (the caller applies the pixel scale).
        ``inds`` is a flat, 0-based bin index per pixel, matching the
        flattened map.
    """
    ny, nx = shape
    if nbins is None:
        nbins = int(max(ny, nx) // 2)

    radius = np.asarray(radial_distance((ny, nx), center), dtype=float)
    max_radius = float(radius.max())
    bin_edges = np.linspace(0.0, max_radius, nbins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    inds = np.clip(np.digitize(radius.ravel(), bin_edges), 1, nbins) - 1
    return bin_centers, inds, nbins


def _reduce_bins(values, inds, nbins, stat):
    """Reduce flat ``values`` within each of ``nbins`` rings by ``stat``.

    Empty rings return 0.0, matching ``hwoutils.radial.radial_profile``'s
    own ``where(counts > 0, ..., 0.0)`` guard. A ring can only be empty when
    ``nbins`` is fine enough that a ring falls between sample radii; on this
    library's half-pixel-offset grid convention there is no pixel at exactly
    r = 0, which is what makes the innermost ring the one at risk (see the
    binning-convention note).

    Args:
        values: Flat array of per-pixel values.
        inds: Flat 0-based bin index per pixel, from :func:`_radial_bins`.
        nbins: Number of radial bins.
        stat: One of "mean", "median", "max".

    Returns:
        The profile, shape ``(nbins,)``.

    Raises:
        ValueError: ``stat`` is not one of the three supported statistics.
    """
    if stat not in _STATS:
        msg = f"unknown stat: {stat!r}; use 'mean', 'median', or 'max'"
        raise ValueError(msg)

    counts = np.bincount(inds, minlength=nbins).astype(float)
    if stat == "mean":
        # hwoutils' own formula, verbatim, so the default path is bit-for-bit
        # the library function it is meant to reproduce.
        sums = np.bincount(inds, weights=values, minlength=nbins)
        return np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)

    # median/max need the members of each ring, not just their sum. One
    # stable sort by bin index groups them; searchsorted gives the slice
    # boundaries. Same `inds`, so same rings as the mean above.
    order = np.argsort(inds, kind="stable")
    sorted_values = values[order]
    edges = np.searchsorted(inds[order], np.arange(nbins + 1))
    reducer = np.median if stat == "median" else np.max
    return np.array(
        [
            reducer(sorted_values[edges[b] : edges[b + 1]]) if counts[b] > 0 else 0.0
            for b in range(nbins)
        ]
    )


def _validate_ax(ax):
    """Raise if ``ax`` is an axes ARRAY rather than the single axes required.

    The axes-shape contract applies here even though this function draws
    into one axes: handing it a ``(2,)`` array from ``plt.subplots`` should
    say so, not fail later inside matplotlib with an unrelated message.

    Raises:
        ValueError: ``ax`` is array-like, naming both the expected and the
            received shape.
    """
    if ax is None or isinstance(ax, plt.Axes):
        return
    shape = np.asarray(ax, dtype=object).shape
    msg = f"plot_contrast_profile: expected a single ax, got axes shape {shape}"
    raise ValueError(msg)


def _validate_floors(floors):
    """Raise if any floor is a callable rather than a precomputed curve.

    ``floors`` are precomputed ``(r, y, label)`` curves and deliberately not
    callables: the leakage floors this serves are SCALED MEASURED PROFILES,
    not analytic functions of radius, so a callable signature would
    advertise a generality the data cannot honor.

    Raises:
        ValueError: An entry is callable, or is not a 3-tuple.
    """
    if floors is None:
        return
    for i, floor in enumerate(floors):
        if callable(floor):
            msg = (
                f"plot_contrast_profile: floors[{i}] is callable; floors take "
                "precomputed (r, y, label) curves, not functions of radius"
            )
            raise ValueError(msg)
        if len(floor) != 3:
            msg = (
                f"plot_contrast_profile: floors[{i}] has {len(floor)} entries; "
                "each floor is an (r, y, label) triple"
            )
            raise ValueError(msg)


def plot_contrast_profile(
    field_or_map,
    *,
    pixscale_lod=None,
    iwa_lod=None,
    owa_lod=None,
    stat="mean",
    ax=None,
    nbins=None,
    floors=None,
    line_kw=None,
):
    """Draw the radial contrast profile of a focal-plane map.

    The radial statistic is computed here; the drawing is delegated whole to
    ``eyepiece.plot_contrast_curve``, which owns the curve, the shaded
    IWA/OWA exclusion regions, and the labeled floor curves. That delegation
    is what makes annotations behave on a reused axes: eyepiece tracks, per
    axes and per marker kind, what it has already drawn, so calling this
    function twice on one ``ax`` to compare two profiles draws two curves
    but only one pair of IWA/OWA spans.

    ``stat="mean"`` (default), ``"median"``, and ``"max"`` are three
    reductions over ONE shared ring geometry -- ``hwoutils.radial``'s
    binning, re-aggregated -- so switching statistic changes only the
    statistic. Reading a "median profile" against a "mean profile" computed
    under a different ring definition is the failure mode this prevents; see
    ``_radial_bins``.

    Units: ``pixscale_lod`` is lambda/D per pixel and the abscissa is
    labeled accordingly. Keep it that way. eyepiece's IWA/OWA shading
    reaches the edge of the axes through a fixed-width data-coordinate
    rectangle, which is scale-dependent and silently wrong for abscissa
    values of order >= 1e9 or <= 1e-8; separations in lambda/D are order
    1-30 and clear of it, but the same profile plotted in meters or radians
    is not.

    Args:
        field_or_map: A focal-plane ``physicaloptix.core.Field`` (its grid
            supplies the pixel scale), or a bare 2D array-like -- including
            a ``SpeckleProcess``-produced map such as
            ``moments.mean_map`` -- which carries no scale of its own and
            therefore REQUIRES ``pixscale_lod``. A complex array is reduced
            to intensity; a chromatic Field is weight-summed to intensity,
            exactly as ``contrast_row`` does.
        pixscale_lod: Pixel scale in lambda/D per pixel. None takes the
            Field's own ``grid.dx``; None with a bare array raises, rather
            than silently plotting a profile against pixel indices while
            labeling the axis in lambda/D.
        iwa_lod: Inner working angle in lambda/D, shaded by eyepiece. None
            omits it.
        owa_lod: Outer working angle in lambda/D. None omits it.
        stat: "mean" (default), "median", or "max".
        ax: Axes to draw into. None creates a new figure and axes.
        nbins: Number of radial bins. None uses ``hwoutils``' own default,
            ``max(ny, nx) // 2``. Exposed deliberately: the disagreement
            between binning conventions is itself a strong, non-monotonic
            function of ``nbins``, so a map whose structure
            has a characteristic spacing (Airy rings, a speckle correlation
            length) wants a bin width chosen against that spacing rather
            than a default inherited unexamined.
        floors: Optional iterable of precomputed ``(r, y, label)`` reference
            curves, forwarded to eyepiece and drawn dashed. Callables are
            rejected; see ``_validate_floors``.
        line_kw: Extra kwargs passed to the main curve's ``ax.plot``,
            applied last -- the place to put a ``label`` or an explicit
            ``color``.

    Returns:
        The ``PlotResult`` from ``eyepiece.plot_contrast_curve``.
        ``artists["line"]`` is the profile curve, whose x data is the ring
        centers in lambda/D. ``artists["fill"]``/``artists["text"]`` carry
        the IWA/OWA shading and labels, present only on the call that
        actually drew them; ``artists["lines"]`` likewise for the floors.

    Raises:
        ValueError: An unknown ``stat``; a bare array with no
            ``pixscale_lod``; a callable or malformed entry in ``floors``;
            or an axes ARRAY passed as ``ax``.
    """
    ep = _require.eyepiece()
    if stat not in _STATS:
        msg = f"unknown stat: {stat!r}; use 'mean', 'median', or 'max'"
        raise ValueError(msg)
    _validate_ax(ax)
    _validate_floors(floors)

    data, _, _ = _resolve(field_or_map, None, "intensity")
    values = np.asarray(_intensity_from(data), dtype=float)

    native = float(field_or_map.grid.dx) if isinstance(field_or_map, Field) else None
    scale = native if pixscale_lod is None else float(pixscale_lod)
    if scale is None:
        msg = (
            "plot_contrast_profile: a bare map carries no pixel scale; pass "
            "pixscale_lod= (lambda/D per pixel), or pass a Field whose grid "
            "supplies it"
        )
        raise ValueError(msg)

    bin_centers, inds, resolved_nbins = _radial_bins(values.shape, nbins)
    profile = _reduce_bins(values.ravel(), inds, resolved_nbins, stat)
    separations = bin_centers * scale

    return ep.plot_contrast_curve(
        separations,
        profile,
        ax=ax,
        iwa=iwa_lod,
        owa=owa_lod,
        floors=floors,
        line_kw=line_kw,
        xlabel=_LOD_SEPARATION_XLABEL,
    )

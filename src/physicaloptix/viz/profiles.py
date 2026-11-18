"""Draw radial contrast profiles using the shared image-grid binning."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.radial import radial_profile

from physicaloptix.core import Field
from physicaloptix.viz import _require
from physicaloptix.viz.fields import _intensity_from

_LOD_SEPARATION_XLABEL = r"$r$ [$\lambda/D$]"


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

    hwoutils computes the radial statistic and eyepiece draws the curve,
    IWA/OWA spans, and labeled floor curves.

    Args:
        field_or_map: Focal Field or Image on a lambda/D grid, or a bare
            2D map. Complex data become intensity, and a chromatic Field
            is wavelength-weighted before binning.
        pixscale_lod: Lambda/D per pixel for a bare map with no grid.
        iwa_lod: Inner working angle in lambda/D, shaded by eyepiece. None
            omits it.
        owa_lod: Outer working angle in lambda/D. None omits it.
        stat: "mean" (default), "median", or "max".
        ax: Axes to draw into. None creates a new figure and axes.
        nbins: Number of radial bins. None uses the hwoutils default.
        floors: Optional iterable of precomputed ``(r, y, label)`` reference
            curves, forwarded to eyepiece and drawn dashed. Callables are
            rejected; see ``_validate_floors``.
        line_kw: Extra kwargs passed to the main curve's ``ax.plot``,
            applied last -- the place to put a ``label`` or an explicit
            ``color``.

    Returns:
        The ``PlotResult`` from ``eyepiece.plot.plot_contrast_curve``.
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
    _validate_ax(ax)
    _validate_floors(floors)

    image = (
        field_or_map.intensity() if isinstance(field_or_map, Field) else field_or_map
    )
    scale = 1.0
    if isinstance(image, Image):
        if image.grid.unit != "lod" or pixscale_lod is not None:
            raise ValueError(
                "pass an Image or Field on a lod grid without pixscale_lod"
            )
        if np.iscomplexobj(image.data):
            image = replace(image, data=_intensity_from(np.asarray(image.data)))
    else:
        if pixscale_lod is None:
            raise ValueError("a bare map requires pixscale_lod")
        scale = float(pixscale_lod)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("pixscale_lod must be positive and finite")
        values = _intensity_from(np.asarray(image))
        grid = PixelGrid.regular(values.shape, 1, "pixel")
        image = Image(values, grid, (), Measure.intensive(), "contrast")

    separations, profile, _ = radial_profile(image, stat=stat, nbins=nbins)

    return ep.plot.plot_contrast_curve(
        separations * scale,
        profile,
        ax=ax,
        iwa=iwa_lod,
        owa=owa_lod,
        floors=floors,
        line_kw=line_kw,
        xlabel=_LOD_SEPARATION_XLABEL,
    )

"""Field -> picture bridges (intensity, phase, complex, declared cut).

``plot_field`` turns a physicaloptix ``Field`` (or a bare array) into a
picture by delegating the pixel work to eyepiece: ``imshow_log`` for
intensity, a direct ``ax.imshow`` for phase (eyepiece has no phase-specific
image primitive of its own), and ``show_field`` for the four-panel complex
view. physicaloptix supplies only what eyepiece cannot know: how to pull a
2D array and a native-unit extent out of a Field (chromatic summing or
channel selection), and where to hang eyepiece's own unit-labeling helpers.
"""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np

from physicaloptix.core import Field
from physicaloptix.viz import _require

# eyepiece owns the unit vocabulary (label_lod, label_arcsec, label_au) for
# every plane it has a helper for; see _label_plane, which calls
# eyepiece.label_lod for a focal-plane Field rather than restating that
# vocabulary here. This dict stays populated ONLY for plane kinds eyepiece
# has no helper for.
_LABELS = {
    "pupil": "pupil coordinate [D]",  # no eyepiece helper as of 2026-08-11
}

_CUT_AXES_RECT = (0.0, -0.45, 1.0, 0.35)
_PHASE_ZERO_RATIO = 1e-10


def _resolve(field_or_map, channel, kind):
    """Pull a plottable array, extent, and plane out of a Field or array.

    Args:
        field_or_map: A ``physicaloptix.core.Field`` or a bare array-like.
        channel: Wavelength index into a chromatic Field's leading axis.
            None sums to intensity when ``kind="intensity"``, and is left
            for the caller to reject for any other kind.
        kind: One of "intensity", "phase", "complex"; only consulted to
            decide what a channel-less chromatic Field does.

    Returns:
        A ``(data, extent, plane)`` tuple. ``data`` is a 2D array: the
        weight-summed intensity when the chromatic-no-channel-intensity
        case applies, otherwise the Field's native (complex) data, or
        whatever a bare array already carries. ``extent`` is the
        ``(left, right, bottom, top)`` tuple from the Field's grid, or
        None for a bare array. ``plane`` is the ``PlaneKind`` value string
        (e.g. "focal"), or None for a bare array.

    Raises:
        ValueError: A chromatic Field with ``kind != "intensity"`` is
            given no ``channel`` -- there is no single 2D array to draw.
    """
    if not isinstance(field_or_map, Field):
        return np.asarray(field_or_map), None, None

    data = np.asarray(field_or_map.data)
    half = float(field_or_map.grid.extent)
    extent = (-half, half, -half, half)
    plane = field_or_map.plane.value

    if data.ndim == 3:
        if channel is not None:
            data = data[channel]
        elif kind == "intensity":
            return np.asarray(field_or_map.intensity()), extent, plane
        else:
            msg = f"chromatic Field with kind={kind!r} requires channel="
            raise ValueError(msg)
    return data, extent, plane


def _label_plane(ep, ax, plane):
    """Apply eyepiece's axis labels for a plane, falling back to _LABELS."""
    if plane == "focal":
        ep.label_lod(ax)
    elif plane in _LABELS:
        label = _LABELS[plane]
        ax.set_xlabel(label)
        ax.set_ylabel(label)


def _intensity_from(data):
    """``|data|^2`` for complex input; the array itself, as float, otherwise."""
    if np.iscomplexobj(data):
        return data.real**2 + data.imag**2
    return np.asarray(data, dtype=float)


def _plot_phase(ax, data, extent, cmap, colorbar, cbar_label, imshow_kw, cbar_kw):
    """Draw phase, NaN-masked below a fraction of the peak intensity.

    Mirrors the legacy ``render_path`` convention: phase is undefined
    where there is no light, so pixels below ``_PHASE_ZERO_RATIO`` of the
    peak intensity are masked rather than rendered as numerical noise.

    Returns:
        An ``(ax, artists)`` pair; ``artists`` holds ``"image"`` and, if
        drawn, ``"cbar"``.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    intensity = _intensity_from(data)
    peak = float(intensity.max())
    phase = np.where(intensity > peak * _PHASE_ZERO_RATIO, np.angle(data), np.nan)

    resolved_cmap = cmap if cmap is not None else hwostyle.cmaps.phase
    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    im = ax.imshow(
        phase, cmap=resolved_cmap, vmin=-np.pi, vmax=np.pi, extent=extent, **kw
    )
    artists = {"image": im}
    if colorbar:
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        label = "rad" if cbar_label is None else cbar_label
        cb = ax.figure.colorbar(im, cax=cax, label=label, **(cbar_kw or {}))
        artists["cbar"] = cb
    return ax, artists


def _cut_profile(image_data, cut, extent):
    """The 1D slice through the image center, its coordinate axis, and mark.

    Returns:
        A ``(coords, profile, mark)`` tuple: 1D coordinate array, the
        sliced 1D data, and the coordinate the slice was taken at (for the
        dashed cut-mark line).

    Raises:
        ValueError: ``cut`` is not "x" or "y".
    """
    npix_row, npix_col = image_data.shape
    if cut == "x":
        idx = npix_row // 2
        profile = image_data[idx, :]
        n = npix_col
    elif cut == "y":
        idx = npix_col // 2
        profile = image_data[:, idx]
        n = npix_row
    else:
        msg = f"unknown cut: {cut!r}; use 'x' or 'y'"
        raise ValueError(msg)

    lo, hi = (extent[0], extent[1]) if extent is not None else (-0.5, n - 0.5)
    step = (hi - lo) / n
    coords = lo + step * (np.arange(n) + 0.5)
    mark = coords[min(idx, n - 1)]
    return coords, profile, mark


def _add_cut(ax, image_data, extent, cut, floor, semilog):
    """Append the declared-cut panel: a dashed mark plus a profile curve.

    Returns:
        An ``(ax_cut, line)`` pair: the new inset Axes (x-linked to `ax`
        via `inset_axes`'s own `sharex`, never by calling `.sharex()` on
        the caller's `ax`) and the cut curve's `Line2D`.
    """
    coords, profile, mark = _cut_profile(image_data, cut, extent)
    if cut == "x":
        ax.axhline(mark, color="0.4", ls="--", lw=1.0)
    else:
        ax.axvline(mark, color="0.4", ls="--", lw=1.0)

    ax_cut = ax.inset_axes(_CUT_AXES_RECT, sharex=ax)
    if semilog:
        (line,) = ax_cut.semilogy(coords, np.clip(profile, floor, None))
    else:
        (line,) = ax_cut.plot(coords, profile)
    return ax_cut, line


def plot_field(
    field_or_map,
    *,
    ax=None,
    kind="intensity",
    channel=None,
    extent=None,
    plane=None,
    cut=None,
    cmap=None,
    vmin=None,
    vmax=None,
    floor=1e-20,
    colorbar=True,
    cbar_label=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a Field (or bare array) as intensity, phase, or a complex mosaic.

    Delegates the pixel work to eyepiece: ``imshow_log`` for
    ``kind="intensity"``, a direct ``ax.imshow`` for ``kind="phase"``
    (eyepiece has no phase-specific image primitive), and ``show_field``
    for ``kind="complex"``. A chromatic Field with no ``channel`` sums to
    intensity for ``kind="intensity"`` and raises for any other kind,
    since there is then no single 2D array to hand eyepiece. Axis labels
    come from eyepiece (``label_lod`` for a focal-plane Field); a
    physicaloptix-local fallback covers only the plane kinds eyepiece has
    no helper for (currently pupil).

    Args:
        field_or_map: A ``physicaloptix.core.Field`` or a bare 2D (or,
            with ``channel``, 3D) array-like.
        ax: Axes to draw the single-panel kinds into. None creates a new
            figure. Ignored for ``kind="complex"``, which always owns its
            own 2x2 figure (``eyepiece.show_field``'s own contract).
        kind: "intensity" (default), "phase", or "complex".
        channel: Wavelength index into a chromatic Field's leading axis.
        extent: ``(left, right, bottom, top)`` override; None uses the
            Field's grid extent, or None for a bare array.
        plane: ``PlaneKind`` value string override (e.g. "focal") for axis
            labeling; None uses the Field's own plane, or no labeling for
            a bare array.
        cut: None, "x", or "y" -- append a declared 1D cut through the
            image center along that axis. Not supported with
            ``kind="complex"``.
        cmap: Colormap override.
        vmin: Norm lower bound (``kind="intensity"`` only).
        vmax: Norm upper bound (``kind="intensity"`` only).
        floor: Log-floor clip, used for ``kind="intensity"`` and for the
            cut curve when ``kind="intensity"``.
        colorbar: Whether to attach a colorbar.
        cbar_label: Colorbar label.
        imshow_kw: Extra kwargs passed to the underlying ``imshow`` call.
        cbar_kw: Extra kwargs passed to the colorbar.

    Returns:
        A ``PlotResult`` for ``kind="intensity"``/``"phase"`` without
        ``cut``. A ``MosaicResult`` with flat ``axes`` ``[image, cut]``
        (image first) when ``cut`` is given. A ``MosaicResult`` with
        ``(2, 2)`` axes for ``kind="complex"``.

    Raises:
        ValueError: ``cut`` is given with ``kind="complex"``; ``cut`` is
            not one of None, "x", "y"; ``kind`` is not one of "intensity",
            "phase", "complex"; or a chromatic Field with
            ``kind != "intensity"`` is given no ``channel``.
    """
    ep = _require.eyepiece()

    if cut is not None and kind == "complex":
        msg = "cut is not supported with kind='complex'"
        raise ValueError(msg)
    if cut is not None and cut not in ("x", "y"):
        msg = f"unknown cut: {cut!r}; use 'x' or 'y'"
        raise ValueError(msg)

    data, resolved_extent, resolved_plane = _resolve(field_or_map, channel, kind)
    extent = resolved_extent if extent is None else extent
    plane = resolved_plane if plane is None else plane

    if kind == "complex":
        result = ep.show_field(data, extent=extent)
        for panel_ax in result.axes.ravel():
            _label_plane(ep, panel_ax, plane)
        return result

    if cut is not None and ax is None:
        width = plt.rcParams["figure.figsize"][0]
        _, ax = plt.subplots(figsize=(width, width * 1.5), layout="constrained")

    if kind == "phase":
        panel_ax, artists = _plot_phase(
            ax, data, extent, cmap, colorbar, cbar_label, imshow_kw, cbar_kw
        )
        image_data = artists["image"].get_array()
        result = ep.PlotResult(ax=panel_ax, artists=artists)
        semilog = False
    elif kind == "intensity":
        image_data = _intensity_from(data)
        result = ep.imshow_log(
            image_data,
            ax=ax,
            extent=extent,
            floor=floor,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar=colorbar,
            cbar_label=cbar_label,
            imshow_kw=imshow_kw,
            cbar_kw=cbar_kw,
        )
        semilog = True
    else:
        msg = f"unknown kind: {kind!r}; use 'intensity', 'phase', or 'complex'"
        raise ValueError(msg)

    _label_plane(ep, result.ax, plane)

    if cut is None:
        return result

    ax_cut, line = _add_cut(result.ax, image_data, extent, cut, floor, semilog)
    artists = dict(result.artists)
    artists["line"] = line
    axes = np.array([result.ax, ax_cut], dtype=object)
    return ep.MosaicResult(axes=axes, artists=artists)

"""Field -> picture bridges (intensity, phase, complex, declared cut).

``plot_field`` turns a physicaloptix ``Field`` (or a bare array) into a
picture by delegating the pixel work to eyepiece: ``imshow_log`` for
intensity, a direct ``ax.imshow`` for phase (eyepiece has no phase-specific
image primitive of its own), and ``show_field`` for the four-panel complex
view. physicaloptix supplies only what eyepiece cannot know: how to pull a
2D array and a native-unit extent out of a Field (chromatic summing or
channel selection), where to hang eyepiece's own unit-labeling helpers, and
the declared 1D cut through the image -- direction-correct for a rectangular
array on an asymmetric extent, not just the square/symmetric fixtures.
"""

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from physicaloptix.core import Field
from physicaloptix.viz import _require

# eyepiece owns the unit vocabulary (label_lod, label_arcsec, label_au) for
# every plane it has a helper for; see _label_plane, which calls
# eyepiece.label_lod for a focal-plane Field rather than restating that
# vocabulary here. This dict stays populated ONLY for plane kinds eyepiece
# has no helper for.
_LABELS = {
    "pupil": ("$x$ [D]", "$y$ [D]"),  # no eyepiece helper as of 2026-08-11
}

_KINDS = ("intensity", "phase", "complex")
_CUTS = ("x", "y")
_PHASE_ZERO_RATIO = 1e-10
# Owned-figure cut inset: below the image (shares x) for a "x" cut, beside
# it (shares y) for a "y" cut -- see _draw_cut_curve for why "y" is
# transposed rather than reusing the "below, sharex" geometry.
_CUT_RECT_BELOW = (0.0, -0.45, 1.0, 0.35)
_CUT_RECT_RIGHT = (1.15, 0.0, 0.35, 1.0)


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
        xlabel, ylabel = _LABELS[plane]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


def _intensity_from(data):
    """``|data|^2`` for complex input; the array itself, as float, otherwise."""
    if np.iscomplexobj(data):
        return data.real**2 + data.imag**2
    return np.asarray(data, dtype=float)


def _masked_phase(data):
    """NaN-masked phase: undefined below ``_PHASE_ZERO_RATIO`` of peak intensity."""
    intensity = _intensity_from(data)
    peak = float(intensity.max())
    return np.where(intensity > peak * _PHASE_ZERO_RATIO, np.angle(data), np.nan)


def _panel_array(data, kind):
    """The 2D array a panel of ``kind`` draws, from raw (possibly complex) data."""
    return _masked_phase(data) if kind == "phase" else _intensity_from(data)


def _resolved_phase_cmap(cmap, ax):
    """A phase Colormap for `ax`, masked (NaN) pixels painted its facecolor.

    Matches ``eyepiece.show_field``'s own phase panel.
    """
    value = cmap if cmap is not None else hwostyle.cmaps.phase
    resolved = matplotlib.colormaps[value] if isinstance(value, str) else value
    return resolved.with_extremes(bad=ax.get_facecolor())


def _plot_phase(ax, data, extent, cmap, colorbar, cbar_label, imshow_kw, cbar_kw):
    """Draw phase, NaN-masked below a fraction of the peak intensity.

    Mirrors the legacy ``render_path`` convention: phase is undefined
    where there is no light, so pixels below ``_PHASE_ZERO_RATIO`` of the
    peak intensity are masked rather than rendered as numerical noise.

    Returns:
        An ``(ax, artists, update)`` triple. ``artists`` holds ``"image"``
        and, if drawn, ``"cbar"``. ``update(new_data)`` recomputes the
        masked phase from fresh raw data and calls ``set_data``, never
        creating a new artist.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    resolved_cmap = _resolved_phase_cmap(cmap, ax)
    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    im = ax.imshow(
        _masked_phase(data),
        cmap=resolved_cmap,
        vmin=-np.pi,
        vmax=np.pi,
        extent=extent,
        **kw,
    )
    artists = {"image": im}
    if colorbar:
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        label = "rad" if cbar_label is None else cbar_label
        cb = ax.figure.colorbar(im, cax=cax, label=label, **(cbar_kw or {}))
        artists["cbar"] = cb

    def update(new_data):
        im.set_data(_masked_phase(new_data))

    return ax, artists, update


def _draw_panel(
    ep,
    ax,
    data,
    kind,
    extent,
    cmap,
    vmin,
    vmax,
    floor,
    colorbar,
    cbar_label,
    imshow_kw,
    cbar_kw,
):
    """Draw the single-panel image for ``kind="intensity"`` or ``"phase"``.

    Returns:
        A ``(result, image_data, semilog)`` tuple: ``result`` is the
        ``PlotResult``; ``image_data`` is the exact 2D array drawn (what
        the cut mechanism slices); ``semilog`` says whether the cut curve
        should use a log value-axis (True for intensity, False for the
        signed phase, which can be negative or NaN).
    """
    if kind == "phase":
        panel_ax, artists, update = _plot_phase(
            ax, data, extent, cmap, colorbar, cbar_label, imshow_kw, cbar_kw
        )
        image_data = artists["image"].get_array()
        result = ep.PlotResult(ax=panel_ax, artists=artists, update=update)
        return result, image_data, False

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
    return result, image_data, True


def _pixel_centers(lo, hi, n):
    """``n`` half-pixel-offset sample coordinates spanning edge extent ``(lo, hi)``."""
    step = (hi - lo) / n
    return lo + step * (np.arange(n) + 0.5)


def _cut_profile(image_data, cut, extent):
    """The 1D slice through the image center, its coordinate axis, and mark.

    ``cut="x"`` holds a row fixed and slices along columns, so the
    profile's abscissa is the image's X range (``extent[0:2]``) and the
    fixed row's OWN coordinate -- the cut-mark value, drawn with
    ``axhline`` -- is a Y value (``extent[2:4]``). ``cut="y"`` is the
    transpose: the profile's abscissa is Y, the mark is an X value, drawn
    with ``axvline``. Reading both from ``extent[0:2]`` regardless of
    direction is only correct for a square array on a symmetric extent --
    exactly what every existing fixture happens to be, and exactly what a
    rectangular array or an asymmetric extent breaks.

    Args:
        image_data: The 2D array the image panel drew.
        cut: "x" or "y".
        extent: ``(left, right, bottom, top)``, or None (a bare array with
            no extent: pixel-index coordinates are used instead).

    Returns:
        A ``(coords, profile, mark)`` tuple: the 1D coordinate array along
        the cut, the sliced 1D data, and the physical coordinate the slice
        was taken at.

    Raises:
        ValueError: ``cut`` is not "x" or "y".
    """
    npix_row, npix_col = image_data.shape
    if cut == "x":
        idx, fixed_n = npix_row // 2, npix_row
        profile, coord_n = image_data[idx, :], npix_col
        coord_lo_hi = extent[0:2] if extent is not None else (-0.5, coord_n - 0.5)
        mark_lo_hi = extent[2:4] if extent is not None else (-0.5, fixed_n - 0.5)
    elif cut == "y":
        idx, fixed_n = npix_col // 2, npix_col
        profile, coord_n = image_data[:, idx], npix_row
        coord_lo_hi = extent[2:4] if extent is not None else (-0.5, coord_n - 0.5)
        mark_lo_hi = extent[0:2] if extent is not None else (-0.5, fixed_n - 0.5)
    else:
        msg = f"unknown cut: {cut!r}; use 'x' or 'y'"
        raise ValueError(msg)

    coords = _pixel_centers(*coord_lo_hi, coord_n)
    mark = _pixel_centers(*mark_lo_hi, fixed_n)[idx]
    return coords, profile, mark


def _draw_mark(image_ax, cut, mark):
    """The dashed cut-mark line on the image.

    A row mark for "x" (axhline), a column mark for "y" (axvline).
    """
    if cut == "x":
        return image_ax.axhline(mark, color="0.4", ls="--", lw=1.0)
    return image_ax.axvline(mark, color="0.4", ls="--", lw=1.0)


def _draw_cut_curve(cut_ax, coords, profile, cut, floor, semilog):
    """Draw the profile into ``cut_ax``, transposed for ``cut="y"``.

    A ``cut="x"`` panel's coordinate is the image's X axis, the same
    physical axis as ``cut_ax``'s own x by construction, so the standard
    orientation applies: coordinate on x, value on y. A ``cut="y"``
    panel's coordinate is the image's Y axis -- a DIFFERENT physical axis
    than ``cut_ax``'s default x -- so it is transposed (value on x,
    coordinate on y) so that a genuine ``sharey`` with the image is
    possible instead of a physically wrong ``sharex``.

    Returns:
        The cut curve's ``Line2D``.
    """
    value = np.clip(profile, floor, None) if semilog else profile
    if cut == "x":
        plotter = cut_ax.semilogy if semilog else cut_ax.plot
        (line,) = plotter(coords, value)
    else:
        plotter = cut_ax.semilogx if semilog else cut_ax.plot
        (line,) = plotter(value, coords)
    return line


def _owned_cut_figsize(cut):
    """Figure size reserving canvas room for the cut panel in its direction."""
    width = plt.rcParams["figure.figsize"][0]
    return (width, width * 1.5) if cut == "x" else (width * 1.5, width)


def _owned_cut_axes(image_ax, cut):
    """A new inset axes hung off ``image_ax``.

    Below (sharing x) for ``cut="x"``, beside it (sharing y) for
    ``cut="y"``. This is OUR OWN new axes sharing at construction time,
    never ``.sharex()``/``.sharey()`` called on a caller-supplied axes.
    """
    if cut == "x":
        return image_ax.inset_axes(_CUT_RECT_BELOW, sharex=image_ax)
    return image_ax.inset_axes(_CUT_RECT_RIGHT, sharey=image_ax)


def _link_cut_limits(image_ax, cut_ax, cut):
    """Match ``cut_ax``'s relevant limits to ``image_ax``'s.

    By explicit ``set_xlim``/``set_ylim`` -- the caller-supplied-axes path
    never calls ``.sharex()``/``.sharey()``, which would raise on a
    pre-shared axes.
    """
    if cut == "x":
        cut_ax.set_xlim(image_ax.get_xlim())
    else:
        cut_ax.set_ylim(image_ax.get_ylim())


def _make_cut_update(result, cut, extent, floor, semilog, mark_line, line, kind):
    """Build the cut ``MosaicResult``'s ``update(new_data)``.

    Refreshes the image panel (via its own reusable transform), the
    dashed cut-mark position, and the cut curve -- never creating a new
    artist.

    Args:
        result: The image panel's own ``PlotResult`` (its ``.update`` does
            the panel-specific redraw: floor-reclip for intensity,
            re-mask for phase).
        cut: "x" or "y".
        extent: The image's extent, for re-deriving coordinates.
        floor: Log-floor clip, reapplied to the cut curve when ``semilog``.
        semilog: Whether the cut curve uses a log value-axis.
        mark_line: The dashed cut-mark ``Line2D`` on the image.
        line: The cut curve's ``Line2D``.
        kind: "intensity" or "phase" -- which raw-to-panel transform to
            reapply to ``new_data``.

    Returns:
        ``update(new_data)``, where ``new_data`` is raw data in the same
        representation ``_resolve`` would hand ``plot_field`` (complex or
        real, matching the original Field/array's dtype) -- NOT the
        already-reduced intensity/phase array ``_draw_panel`` produced.
    """

    def update(new_data):
        new_image_data = _panel_array(new_data, kind)
        result.update(new_image_data)
        coords, profile, mark = _cut_profile(np.asarray(new_image_data), cut, extent)
        value = np.clip(profile, floor, None) if semilog else profile
        if cut == "x":
            mark_line.set_ydata([mark, mark])
            line.set_data(coords, value)
        else:
            mark_line.set_xdata([mark, mark])
            line.set_data(value, coords)

    return update


def _validate_call(kind, cut, ax, axes, fig):
    """All ``plot_field`` precondition checks, raised before any figure work.

    Raises:
        ValueError: An unknown ``kind``/``cut``, an unsupported
            ``kind``/``cut`` combination, or an unsupported combination of
            ``ax``/``axes``/``fig``.
    """
    if kind not in _KINDS:
        msg = f"unknown kind: {kind!r}; use 'intensity', 'phase', or 'complex'"
        raise ValueError(msg)
    if cut is not None and cut not in _CUTS:
        msg = f"unknown cut: {cut!r}; use 'x' or 'y'"
        raise ValueError(msg)
    if cut is not None and kind == "complex":
        msg = "cut is not supported with kind='complex'"
        raise ValueError(msg)
    if ax is not None and axes is not None:
        msg = "pass only one of ax= or axes="
        raise ValueError(msg)
    if fig is not None and kind != "complex":
        msg = "fig= is only used with kind='complex'"
        raise ValueError(msg)

    if kind == "complex":
        if ax is not None:
            msg = (
                "kind='complex' draws a 2x2 mosaic and cannot draw into a "
                "single ax=; pass axes= of shape (2, 2) or fig=, or let "
                "plot_field own the figure"
            )
            raise ValueError(msg)
        if axes is not None:
            shape = np.asarray(axes).shape
            if shape != (2, 2):
                msg = (
                    f"plot_field: kind='complex' expected axes shape (2, 2), "
                    f"got {shape}"
                )
                raise ValueError(msg)
        return

    if cut is None:
        if axes is not None:
            msg = "axes= is only used with kind='complex' or cut=; pass ax= instead"
            raise ValueError(msg)
        return

    if ax is not None:
        msg = (
            "cut= needs two axes; pass axes=(image_ax, cut_ax) instead of "
            "ax=, or leave ax=None to let plot_field own the figure"
        )
        raise ValueError(msg)
    if axes is not None and len(axes) != 2:
        msg = f"plot_field: cut= expected 2 axes [image, cut], got {len(axes)}"
        raise ValueError(msg)


def plot_field(
    field_or_map,
    *,
    ax=None,
    axes=None,
    fig=None,
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

    ``cut`` appends a declared 1D slice through the image center. With no
    ``ax``/``axes``, ``plot_field`` owns the figure and hangs the cut
    panel off the image via an inset (reserving canvas room by figsize):
    below the image for ``cut="x"`` (shares x), beside it for ``cut="y"``
    (shares y; the panel is transposed -- coordinate on its own y axis --
    since a "y" cut's coordinate is not the same physical axis as the
    image's x). To place the two panels yourself, reserve them and pass
    ``axes=(image_ax, cut_ax)``::

        fig, (image_ax, cut_ax) = plt.subplots(2, 1, figsize=(4, 6))
        plot_field(field, axes=(image_ax, cut_ax), cut="x")

    ``plot_field`` never carves the cut axes out of a single caller-owned
    ``ax``: pass ``ax=`` and ``cut=`` together and it raises, naming
    ``axes=`` as the caller path. The caller path sets ``cut_ax``'s
    matching limits by explicit ``set_xlim``/``set_ylim``, never
    ``.sharex()``/``.sharey()`` on a caller-supplied axes.

    Args:
        field_or_map: A ``physicaloptix.core.Field`` or a bare 2D (or,
            with ``channel``, 3D) array-like.
        ax: Axes to draw a single-panel kind into (no ``cut``). None
            creates a new figure. Mutually exclusive with ``axes``, and
            with ``kind="complex"`` or ``cut=`` (use ``axes=`` for those).
        axes: For ``kind="complex"``, a ``(2, 2)`` array of Axes (forwarded
            to ``eyepiece.show_field``). For ``cut=``, a length-2
            ``(image_ax, cut_ax)`` pair. None lets ``plot_field`` own the
            figure. Mutually exclusive with ``ax``.
        fig: A Figure or SubFigure ``eyepiece.show_field`` builds its 2x2
            grid inside. Only used with ``kind="complex"``; ignored when
            ``axes`` is also given.
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
        (image first) when ``cut`` is given, whose ``.update(new_data)``
        refreshes the image, the cut-mark, and the cut curve together. A
        ``MosaicResult`` with ``(2, 2)`` axes for ``kind="complex"``.

    Raises:
        ValueError: An unknown ``kind`` or ``cut``; ``cut`` given with
            ``kind="complex"``; ``ax``/``axes``/``fig`` combined in an
            unsupported way (message names the fix); ``axes`` of the
            wrong shape/length for the active mode; or a chromatic Field
            with ``kind != "intensity"`` given no ``channel``.
    """
    ep = _require.eyepiece()
    _validate_call(kind, cut, ax, axes, fig)

    data, resolved_extent, resolved_plane = _resolve(field_or_map, channel, kind)
    extent = resolved_extent if extent is None else extent
    plane = resolved_plane if plane is None else plane

    if kind == "complex":
        result = ep.show_field(data, fig=fig, axes=axes, extent=extent)
        for panel_ax in result.axes.ravel():
            _label_plane(ep, panel_ax, plane)
        return result

    if cut is None:
        result, _, _ = _draw_panel(
            ep,
            ax,
            data,
            kind,
            extent,
            cmap,
            vmin,
            vmax,
            floor,
            colorbar,
            cbar_label,
            imshow_kw,
            cbar_kw,
        )
        _label_plane(ep, result.ax, plane)
        return result

    owned = axes is None
    if owned:
        figsize = _owned_cut_figsize(cut)
        _, image_ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        image_ax, cut_ax = axes[0], axes[1]

    result, image_data, semilog = _draw_panel(
        ep,
        image_ax,
        data,
        kind,
        extent,
        cmap,
        vmin,
        vmax,
        floor,
        colorbar,
        cbar_label,
        imshow_kw,
        cbar_kw,
    )
    _label_plane(ep, result.ax, plane)

    if owned:
        cut_ax = _owned_cut_axes(result.ax, cut)

    coords, profile, mark = _cut_profile(image_data, cut, extent)
    mark_line = _draw_mark(result.ax, cut, mark)
    line = _draw_cut_curve(cut_ax, coords, profile, cut, floor, semilog)
    if not owned:
        _link_cut_limits(result.ax, cut_ax, cut)

    artists = dict(result.artists)
    artists["line"] = line
    update = _make_cut_update(
        result, cut, extent, floor, semilog, mark_line, line, kind
    )
    axes_arr = np.array([result.ax, cut_ax], dtype=object)
    return ep.MosaicResult(axes=axes_arr, artists=artists, update=update)


def draw_dark_zone(ax, iwa_lod, owa_lod, *, line_kw=None):
    """Ring the inner and outer working angles on a 2D map.

    eyepiece's ``plot_contrast_curve`` marks the same two working angles
    on a 1D contrast curve as shaded spans (a neutral tone blended from
    the axes' own facecolor/text color, drawn once per axes and tracked
    so a second call does not duplicate it -- see that function's
    docstring). A shaded span has no 2D equivalent that would not either
    obscure the map or double as a second definition of the same
    quantity, so this draws rings instead: the same restrained,
    non-data-color idiom eyepiece intends for a working-angle marker,
    carried over as a single hwostyle palette color (its private neutral
    tone is not part of eyepiece's public surface) rather than a color
    that could be mistaken for a plotted curve.

    Args:
        ax: Axes to draw into, in the map's native coordinates centered
            on the optical axis (typically lambda/D).
        iwa_lod: Inner working angle radius, in the same units as `ax`.
        owa_lod: Outer working angle radius, in the same units as `ax`.
        line_kw: Extra kwargs passed to both `Circle` constructors,
            applied last.

    Returns:
        A dict with keys "iwa" and "owa", each the `Circle` patch added
        to `ax`.
    """
    kw = {
        "fill": False,
        "linestyle": "--",
        "color": hwostyle.palette.cyan,
        **(line_kw or {}),
    }
    iwa_circle = Circle((0.0, 0.0), iwa_lod, **kw)
    owa_circle = Circle((0.0, 0.0), owa_lod, **kw)
    ax.add_patch(iwa_circle)
    ax.add_patch(owa_circle)
    return {"iwa": iwa_circle, "owa": owa_circle}


def contrast_row(
    fields_or_maps,
    *,
    telescope_peak=None,
    titles=None,
    axes=None,
    norm_policy="shared",
    vmin=None,
    vmax=None,
    cbar_label="contrast (I / telescope peak)",
    annulus=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a row of contrast maps, comparable or independently scaled.

    Each entry of ``fields_or_maps`` is normalized through ``_resolve``
    (the same path ``plot_field`` uses) and, for a Field, reduced to
    intensity exactly as ``plot_field(kind="intensity")`` does; a bare
    array passes through unchanged. ``telescope_peak``, when given,
    divides every map before plotting, turning intensity into contrast
    (I / telescope peak). The conventional pinned window for a contrast
    colorbar is 1e-13 to 1e-8 -- pass that range as ``vmin``/``vmax`` to
    compare maps against an absolute, paper-fixed scale. Pinning both
    ends neutralizes ``norm_policy`` by construction: "shared" already
    forces one norm built from the pinned bounds, and "independent"
    builds each panel's norm from the same pinned bounds too, so the two
    policies draw identically once both `vmin` and `vmax` are fixed.

    ``norm_policy="shared"`` delegates to ``eyepiece.compare_row``, which
    builds one norm object from the min/max across all maps (or the
    pinned ``vmin``/``vmax``) and hands it to every panel by identity --
    panels are then directly, pixel-for-pixel comparable, not merely
    rescaled to look alike. ``norm_policy="independent"`` instead calls
    ``eyepiece.imshow_log`` once per panel, so each map gets its own norm
    derived from its own data (or the same pinned bounds, if given).

    Args:
        fields_or_maps: Sequence of ``physicaloptix.core.Field``
            instances or bare 2D array-likes, one per panel.
        telescope_peak: Divides every map before plotting. None leaves
            the maps as given.
        titles: Optional sequence of per-panel titles, same length as
            `fields_or_maps`.
        axes: Axes to draw into, one per panel; shape must be `(n,)` for
            `n = len(fields_or_maps)`. None creates a new row of panels.
        norm_policy: "shared" (default) or "independent"; see above.
        vmin: Norm lower bound. None derives it from the data.
        vmax: Norm upper bound. None derives it from the data.
        cbar_label: Label for the colorbar(s).
        annulus: Optional `(iwa_lod, owa_lod)` pair; when given,
            `draw_dark_zone` is called on every panel.
        imshow_kw: Extra kwargs passed to each panel's `imshow` call.
        cbar_kw: Extra kwargs passed to each colorbar.

    Returns:
        A `MosaicResult` with `axes` shape `(n,)`. `artists["image"]` is
        a list of `AxesImage`, one per panel. `artists["cbar"]` is the
        single shared `Colorbar` for `norm_policy="shared"`, or a list of
        one `Colorbar` per panel for `norm_policy="independent"`.
        `artists["annulus"]`, present only when `annulus` is given, is a
        list of one `draw_dark_zone` result dict per panel.

    Raises:
        ValueError: `axes` is given with a shape other than `(n,)`, or
            `norm_policy` is not "shared" or "independent".
    """
    ep = _require.eyepiece()

    maps = []
    for item in fields_or_maps:
        data, _, _ = _resolve(item, None, "intensity")
        intensity = _intensity_from(data)
        if telescope_peak is not None:
            intensity = intensity / telescope_peak
        maps.append(intensity)

    n = len(maps)
    if axes is not None:
        axes = np.atleast_1d(axes)
        if axes.shape != (n,):
            msg = f"contrast_row: expected axes shape ({n},), got {axes.shape}"
            raise ValueError(msg)

    if norm_policy == "shared":
        result = ep.compare_row(
            maps,
            titles,
            axes=axes,
            norm="log",
            vmin=vmin,
            vmax=vmax,
            cbar_label=cbar_label,
            imshow_kw=imshow_kw,
            cbar_kw=cbar_kw,
        )
    elif norm_policy == "independent":
        if axes is None:
            _, panel_axes = plt.subplots(1, n, layout="constrained", squeeze=False)
            axes = panel_axes[0]
        ims = []
        cbars = []
        for i, m in enumerate(maps):
            panel = ep.imshow_log(
                m,
                ax=axes[i],
                vmin=vmin,
                vmax=vmax,
                cbar_label=cbar_label,
                imshow_kw=imshow_kw,
                cbar_kw=cbar_kw,
            )
            if titles is not None:
                axes[i].set_title(titles[i])
            ims.append(panel.artists["image"])
            if "cbar" in panel.artists:
                cbars.append(panel.artists["cbar"])
        artists = {"image": ims}
        if cbars:
            artists["cbar"] = cbars
        result = ep.MosaicResult(axes=axes, artists=artists)
    else:
        msg = f"unknown norm_policy: {norm_policy!r}; use 'shared' or 'independent'"
        raise ValueError(msg)

    if annulus is None:
        return result

    iwa_lod, owa_lod = annulus
    rings = [draw_dark_zone(panel_ax, iwa_lod, owa_lod) for panel_ax in result.axes]
    artists = dict(result.artists)
    artists["annulus"] = rings
    return ep.MosaicResult(axes=result.axes, artists=artists)

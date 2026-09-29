"""field_columns: the complex field at successive planes, amplitude over phase.

One column per plane (or per side of an element within a plane), the
amplitude ``|E|`` on top and the phase below, with the operation that
carries each column into the next written in the gap between them. Every
scale is a property of the whole sequence, never of the columns drawn, so
any subset of the columns (``columns=``) draws each panel exactly as the
full sequence does. That is what lets a figure show one element at a time
and still match a figure that shows them all.

Two thresholds are absolute, not per panel. The amplitude norm is one log
range for every amplitude panel, by default the peak ``|E|`` of the whole
sequence down four decades (the same ``1e-8`` intensity floor ``plot_path``
uses, as an amplitude). The phase is masked where ``|E|`` falls below one
absolute amplitude, by default the bottom of that norm, so a panel with
little light shows little phase rather than its numerical noise scaled up
to a full cycle.
"""

from collections.abc import Mapping
from itertools import pairwise

import numpy as np
from matplotlib.transforms import Bbox

from physicaloptix.core import Field
from physicaloptix.viz import _require
from physicaloptix.viz.fields import _label_plane, _resolve, _resolved_phase_cmap

# Default amplitude range: four decades below the sequence peak, the
# amplitude of plot_path's 1e-8 intensity floor.
_AMP_DECADES = 4.0
# Owned-figure geometry, in inches. Every gap has one width, wide enough
# for a short operator label; a pair is marked by its label, not by spacing.
_PANEL_IN = 1.6
_GAP_IN = 0.64
_ROW_GAP_IN = 0.12
_LEFT_IN = 1.0
_RIGHT_IN = 0.2
_RIGHT_KEY_IN = 0.95
_TOP_IN = 0.35
_HEADER_IN = 0.3
_BOTTOM_IN = 0.55
_KEY_WIDTH = 0.06
_KEY_PAD = 0.1
_PHASE_TICKS = (-np.pi, 0.0, np.pi)
_PHASE_TICKLABELS = (r"$-\pi$", "0", r"$\pi$")


def _keyed(fields):
    """``(keys, items)`` of a mapping, or of a sequence keyed by index."""
    if isinstance(fields, Mapping):
        return list(fields), list(fields.values())
    items = list(fields)
    return list(range(len(items))), items


def _complex_data(item, key):
    """The 2D complex array, extent and plane of one column's field.

    Raises:
        ValueError: The field is not 2D (a chromatic Field has no single
            complex field to draw; select a wavelength first).
    """
    raw = np.asarray(item.data if isinstance(item, Field) else item)
    if raw.ndim != 2:
        msg = (
            f"field_columns: column {key!r} has shape {raw.shape}; each "
            "column is one 2D complex field (select a wavelength of a "
            "chromatic Field first)"
        )
        raise ValueError(msg)
    return _resolve(item, None, "phase")


def _validate_selection(keys, columns, gaps, pairs, intensity):
    """Resolve ``columns`` to sequence indices and check every argument.

    Returns:
        The selected indices into the full sequence, increasing.

    Raises:
        ValueError: An unknown or out-of-order column, a ``gaps`` of the
            wrong length, a pair whose two columns are not neighbors, or an
            intensity key not in the sequence.
    """
    index = {key: i for i, key in enumerate(keys)}
    chosen = keys if columns is None else list(columns)
    unknown = [c for c in chosen if c not in index]
    if unknown:
        msg = f"field_columns: unknown columns {unknown}; known: {keys}"
        raise ValueError(msg)
    picks = [index[c] for c in chosen]
    if not picks or any(b <= a for a, b in pairwise(picks)):
        msg = (
            "field_columns: columns= must name at least one column, each "
            f"once, in sequence order {keys}; got {chosen}"
        )
        raise ValueError(msg)
    if gaps is not None and len(gaps) != len(keys) - 1:
        msg = (
            f"field_columns: gaps= has {len(gaps)} entries; a sequence of "
            f"{len(keys)} columns has {len(keys) - 1} gaps"
        )
        raise ValueError(msg)
    for label, pair in (pairs or {}).items():
        a, b = pair
        if a not in index or b not in index or index[b] != index[a] + 1:
            msg = (
                f"field_columns: pair {label!r} must name two neighboring "
                f"columns, before then after; got {pair}"
            )
            raise ValueError(msg)
    missing = [k for k in intensity if k not in index]
    if missing:
        msg = f"field_columns: unknown intensity columns {missing}"
        raise ValueError(msg)
    return picks


def _pair_roles(pairs):
    """``{key: (label, "before" | "after")}`` for every paired column."""
    roles = {}
    for label, (before, after) in (pairs or {}).items():
        roles[before] = (label, "before")
        roles[after] = (label, "after")
    return roles


def _owned_axes(n, header, right_key):
    """A new figure holding a 2 x ``n`` panel grid at fixed inch geometry.

    The panels are placed explicitly, not by a layout engine: the gaps
    between columns carry labels, so their width is part of the drawing.
    """
    import matplotlib.pyplot as plt

    right = _RIGHT_KEY_IN if right_key else _RIGHT_IN
    top = _TOP_IN + (_HEADER_IN if header else 0.0)
    width = _LEFT_IN + n * _PANEL_IN + (n - 1) * _GAP_IN + right
    height = _BOTTOM_IN + 2.0 * _PANEL_IN + _ROW_GAP_IN + top
    fig = plt.figure(figsize=(width, height))
    axes = np.empty((2, n), dtype=object)
    for row, y in ((0, _BOTTOM_IN + _PANEL_IN + _ROW_GAP_IN), (1, _BOTTOM_IN)):
        for col in range(n):
            x = _LEFT_IN + col * (_PANEL_IN + _GAP_IN)
            rect = [x / width, y / height, _PANEL_IN / width, _PANEL_IN / height]
            axes[row, col] = fig.add_axes(rect)
    return axes


def _between(left, right):
    """A draw-time bbox from ``left``'s right edge to ``right``'s left edge.

    Vertically it spans the space between the amplitude and phase rows of
    the left column, so ``(0.5, 0.5)`` in it is the middle of the gap.
    Evaluated at draw time, so a layout engine that moves the axes after
    this call still gets the label centered.
    """
    amp_left, phase_left = left
    amp_right = right[0]

    def bbox(renderer):
        a = amp_left.get_window_extent(renderer)
        p = phase_left.get_window_extent(renderer)
        r = amp_right.get_window_extent(renderer)
        return Bbox.from_extents(a.x1, p.y1, r.x0, a.y0)

    return bbox


def _spanning(top_axes):
    """A draw-time bbox around the given amplitude-row axes."""

    def bbox(renderer):
        return Bbox.union([ax.get_window_extent(renderer) for ax in top_axes])

    return bbox


def _phase_image(data, phase_floor):
    """Phase in radians, NaN where ``|E|`` is below ``phase_floor``."""
    return np.where(np.abs(data) >= phase_floor, np.angle(data), np.nan)


def _key(ax, image, side, label, ticks=None, ticklabels=None):
    """A colorbar in a thin inset beside ``ax``, on the given side."""
    x0 = -_KEY_PAD - _KEY_WIDTH if side == "left" else 1.0 + _KEY_PAD
    cax = ax.inset_axes([x0, 0.0, _KEY_WIDTH, 1.0])
    cbar = ax.figure.colorbar(image, cax=cax, label=label)
    cax.yaxis.set_ticks_position(side)
    cax.yaxis.set_label_position(side)
    if ticks is not None:
        cbar.set_ticks(list(ticks))
        cbar.set_ticklabels(list(ticklabels))
    return cbar


def field_columns(
    fields,
    *,
    columns=None,
    gaps=None,
    pairs=None,
    titles=None,
    intensity=(),
    intensity_range=None,
    extents=None,
    vmin=None,
    vmax=None,
    phase_floor=None,
    amp_cmap=None,
    phase_cmap=None,
    blank_color=None,
    colorbar=True,
    axes=None,
    imshow_kw=None,
):
    """Draw complex fields at successive planes as amplitude-over-phase columns.

    ``fields`` is the whole sequence, in propagation order. ``columns``
    picks the ones to draw; every scale (the amplitude norm, the phase
    threshold) is computed from the whole sequence, so a subset draws each
    of its panels exactly as the full sequence draws it. Draw the full
    chain for the complete figure and a two- or three-column subset for a
    figure about one element, from the same arguments.

    The top panel of a column is ``|E|`` on one log norm shared by every
    amplitude panel; the bottom panel is the phase on the fixed range
    ``[-pi, pi]``, masked (drawn in ``blank_color``) where ``|E|`` is
    below ``phase_floor``. A column named in ``intensity`` draws
    ``I = |E|^2`` on top instead, on one intensity norm of its own: the
    plane where the intensity is taken.

    ``gaps`` labels what carries each column into the next (a transform, a
    multiplication by an element), one entry per gap of the full sequence.
    A label is drawn between two drawn columns only when they are neighbors
    in the full sequence; drawn columns with skipped planes between them
    get no label. ``pairs`` marks the two sides of an element in one plane:
    its columns are titled "before" and "after", and the pair's label (the
    element's name) is written over them.

    Args:
        fields: The complex fields in propagation order, as a mapping of
            column key to field, or a sequence (keys are then the indices).
            Each field is a 2D array-like or a monochromatic
            ``physicaloptix.core.Field``. Normalize them before the call
            (pupil planes relative to the incident amplitude, focal planes
            to the square root of a reference peak, say): one norm covers
            every amplitude panel.
        columns: Keys of the columns to draw, in sequence order. None draws
            them all.
        gaps: One label per gap of the full sequence (``len(fields) - 1``
            entries), each a string or None. None draws no gap labels.
        pairs: Mapping of an element label to its ``(before, after)``
            column keys, which must be neighbors. None pairs nothing.
        titles: Mapping of column key to title, drawn over the top panel.
            Overrides the "before"/"after" title of a paired column.
        intensity: Keys of the columns whose top panel is ``|E|^2``.
        intensity_range: ``(vmin, vmax)`` of the intensity norm. None uses
            the square of the amplitude range.
        extents: Mapping of column key to ``(left, right, bottom, top)``, or
            one extent for every column. None uses a Field's own grid, or
            pixel indices for a bare array.
        vmin: Bottom of the amplitude norm. None is ``vmax`` times
            ``1e-4``.
        vmax: Top of the amplitude norm. None is the peak ``|E|`` over
            every non-intensity column of the full sequence.
        phase_floor: Absolute amplitude below which the phase is masked.
            None uses the bottom of the amplitude norm.
        amp_cmap: Colormap for the amplitude and intensity panels. None
            uses the semantic "intensity" colormap, as
            ``eyepiece.show_field`` does for its amplitude panel.
        phase_cmap: Colormap for the phase panels. None uses the semantic
            "phase" colormap.
        blank_color: Color of the masked phase pixels. None uses the phase
            axes' own facecolor.
        colorbar: Whether to draw the keys: amplitude and phase beside the
            first column (on its left), intensity beside the last intensity
            column (on its right).
        axes: A ``(2, n)`` array of Axes, amplitude row first, for the
            ``n`` drawn columns. None creates a new figure with the panels
            at a fixed size and every gap one width. The gap labels
            and pair labels are centered on the axes at draw time, so a
            caller's own layout is respected.
        imshow_kw: Extra kwargs passed to every ``imshow`` call.

    Returns:
        A ``MosaicResult`` with ``axes`` of shape ``(2, n)``.
        ``artists["image"]`` is the list of ``AxesImage``, one per panel in
        ``axes.flat`` order (the top row, then the phase row).
        ``artists["title"]`` is the list of column titles drawn.
        ``artists["text"]`` is the list of gap labels, left to right,
        followed by the pair labels. ``artists["cbar"]``, when drawn, is
        the list of keys ``[amplitude, phase]`` plus the intensity key when
        an intensity column is drawn. ``update(new_fields)`` redraws the
        drawn columns from a new full sequence under the same norms and
        phase threshold.

    Raises:
        ValueError: An unknown or out-of-order column, a ``gaps`` of the
            wrong length, a pair of non-neighboring columns, an unknown
            intensity key, a column that is not one 2D field, or ``axes``
            of the wrong shape.
    """
    ep = _require.eyepiece()
    keys, items = _keyed(fields)
    intensity = tuple(intensity)
    picks = _validate_selection(keys, columns, gaps, pairs, intensity)
    n = len(picks)
    if axes is not None:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != (2, n):
            msg = f"field_columns: expected axes shape (2, {n}), got {axes.shape}"
            raise ValueError(msg)

    resolved = [_complex_data(item, key) for key, item in zip(keys, items, strict=True)]
    datas = [r[0] for r in resolved]
    if vmax is None:
        peaks = [
            float(np.abs(d).max())
            for key, d in zip(keys, datas, strict=True)
            if key not in intensity
        ]
        vmax = max(peaks) if peaks else 1.0
    if vmin is None:
        vmin = vmax * 10.0**-_AMP_DECADES
    if phase_floor is None:
        phase_floor = vmin
    if intensity_range is None:
        intensity_range = (vmin**2, vmax**2)

    roles = _pair_roles(pairs)
    if axes is None:
        drawn = {keys[i] for i in picks}
        header = any(drawn & set(pair) for pair in (pairs or {}).values())
        axes = _owned_axes(n, header, bool(drawn & set(intensity)))

    kw = dict(imshow_kw or {})
    images = [None] * (2 * n)
    titles_drawn = []
    for col, i in enumerate(picks):
        key = keys[i]
        data, own_extent, plane = resolved[i]
        if isinstance(extents, Mapping):
            extent = extents.get(key, own_extent)
        else:
            extent = own_extent if extents is None else extents
        top_ax, phase_ax = axes[0, col], axes[1, col]
        if key in intensity:
            lo, hi = intensity_range
            top = ep.imshow_log(
                np.abs(data) ** 2,
                ax=top_ax,
                extent=extent,
                floor=lo,
                vmin=lo,
                vmax=hi,
                cmap=amp_cmap,
                colorbar=False,
                imshow_kw=kw,
            )
        else:
            top = ep.imshow_log(
                np.abs(data),
                ax=top_ax,
                extent=extent,
                floor=vmin,
                vmin=vmin,
                vmax=vmax,
                cmap=amp_cmap,
                colorbar=False,
                imshow_kw=kw,
            )
        images[col] = top.artists["image"]
        if blank_color is not None:
            phase_ax.set_facecolor(blank_color)
        phase_im = phase_ax.imshow(
            _phase_image(data, phase_floor),
            cmap=_resolved_phase_cmap(phase_cmap, phase_ax),
            vmin=-np.pi,
            vmax=np.pi,
            extent=extent,
            **{"interpolation": "nearest", "origin": "lower", **kw},
        )
        images[n + col] = phase_im
        for ax in (top_ax, phase_ax):
            ax.set_yticks([])
        top_ax.tick_params(labelbottom=False)
        _label_plane(ep, phase_ax, plane)
        top_ax.set_xlabel("")
        phase_ax.set_ylabel("")
        title = (titles or {}).get(key, roles.get(key, (None, None))[1])
        if title is not None:
            titles_drawn.append(top_ax.set_title(title))

    texts = []
    for col in range(n - 1):
        a, b = picks[col], picks[col + 1]
        word = None if gaps is None or b != a + 1 else gaps[a]
        if word is None:
            continue
        left = (axes[0, col], axes[1, col])
        right = (axes[0, col + 1], axes[1, col + 1])
        texts.append(
            axes[0, col].annotate(
                word,
                xy=(0.5, 0.5),
                xycoords=_between(left, right),
                ha="center",
                va="center",
                annotation_clip=False,
            )
        )
    for label, pair in (pairs or {}).items():
        drawn = [col for col, i in enumerate(picks) if keys[i] in pair]
        if not drawn:
            continue
        size = float(axes[0, drawn[0]].title.get_fontsize())
        above = 1.3 * size + 10.0
        header = axes[0, drawn[0]].annotate(
            label,
            xy=(0.5, 1.0),
            xycoords=_spanning([axes[0, col] for col in drawn]),
            xytext=(0.0, above),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=size,
            annotation_clip=False,
        )
        texts.append(header)

    artists = {"image": images, "title": titles_drawn, "text": texts}
    if colorbar:
        amp_col = next(
            (c for c, i in enumerate(picks) if keys[i] not in intensity), None
        )
        cbars = []
        if amp_col is not None:
            cbars.append(_key(axes[0, 0], images[amp_col], "left", "$|E|$"))
        cbars.append(
            _key(
                axes[1, 0],
                images[n],
                "left",
                "phase [rad]",
                _PHASE_TICKS,
                _PHASE_TICKLABELS,
            )
        )
        drawn_intensity = [c for c, i in enumerate(picks) if keys[i] in intensity]
        if drawn_intensity:
            last = drawn_intensity[-1]
            cbars.append(_key(axes[0, last], images[last], "right", "$I = |E|^2$"))
        artists["cbar"] = cbars

    def update(new_fields):
        _, new_items = _keyed(new_fields)
        for col, i in enumerate(picks):
            data = _complex_data(new_items[i], keys[i])[0]
            if keys[i] in intensity:
                top = np.clip(np.abs(data) ** 2, intensity_range[0], None)
            else:
                top = np.clip(np.abs(data), vmin, None)
            images[col].set_data(top)
            images[n + col].set_data(_phase_image(data, phase_floor))

    return ep.MosaicResult(axes=axes, artists=artists, update=update)

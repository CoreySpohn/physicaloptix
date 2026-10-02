"""plot_propagation: a field along the direction of travel, drawn as a map.

A propagation map stacks the transverse cut of a field at successive
distances into one picture: the distance ``z`` runs to the right, as on an
optical bench, and the transverse coordinate ``x`` runs up. The array is
indexed ``[z, x]``, one row per distance, so it is the natural output of a
loop that propagates a 1D field to each ``z`` in turn.

Cells are drawn with ``pcolormesh`` at their true edges, computed from the
pixel-center coordinates, so a nonuniform or logarithmic ``z`` (a map over
Fresnel number, say) is drawn without resampling or interpolation. Bench
elements -- an opaque screen with openings, a phase screen drawn as its
phase, a marked plane -- are drawn in data coordinates on top of the map.

The signed field ``Re E`` gets a diverging colormap centered on the axes
background: zero is drawn as the background itself, positive values run to
the starlight role color, negative values to a muted complement of it at
the same lightness distance from the background. A sequential map from the
background to the starlight color would put zero at its middle, a faint
starlight tint, so an unlit region would read as faintly lit.
"""

import colorsys

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hwostyle.colors import hex_to_rgb, relative_luminance, rgb_to_hex
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_hex
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from physicaloptix.viz import _require
from physicaloptix.viz.columns import _hatch_color, _hatch_patch
from physicaloptix.viz.fields import _resolved_phase_cmap

_KINDS = ("real", "intensity", "amplitude", "phase")
_NORMS = ("linear", "log")
_SCALES = ("linear", "log")
_ELEMENT_KINDS = ("screen", "phase", "plane")
# A log norm with only its top given spans this many decades.
_LOG_DECADES = 4.0
# Phase is masked where |E| falls below this fraction of the first frame's
# peak |E|, unless an absolute phase_floor is given.
_PHASE_FLOOR_REL = 1e-3
_HATCH = "////"
# Screens and phase strips are this fraction of the z axis wide (of its
# log span on a log axis), centered on their plane.
_ELEMENT_WIDTH = 0.015
# Neutral levels from the background (0) to the text color (1): an opaque
# screen is hardware, a marked plane must read over a bright field.
_SCREEN_LEVEL = 0.5
_PLANE_LEVEL = 0.75
# The negative side of the signed map keeps this fraction of the starlight
# color's saturation: a quieter hue than the positive side, at equal
# lightness distance from the background. At the lightness of a bright
# starlight color the sRGB gamut holds little blue chroma anyway, so on a
# dark background the complement is pale whatever this factor is.
_NEGATIVE_SATURATION = 0.85
# Plane lines are this fraction of the style's line width: scenery, not data.
_PLANE_LW = 0.6
# An odd table puts one entry exactly at the midpoint, so zero draws as the
# background color itself rather than its nearest neighbor.
_SIGNED_N = 257
_PHASE_TICKS = (-np.pi, 0.0, np.pi)
_PHASE_TICKLABELS = (r"$-\pi$", "0", r"$\pi$")
_CBAR_LABELS = {
    "real": r"Re $E$",
    "intensity": r"$|E|^2$",
    "amplitude": r"$|E|$",
    "phase": "phase [rad]",
}


def _lightness(color):
    """CIE L* of a color, from its WCAG relative luminance."""
    y = relative_luminance(to_hex(color))
    delta = 6.0 / 29.0
    f = y ** (1.0 / 3.0) if y > delta**3 else y / (3.0 * delta**2) + 4.0 / 29.0
    return 116.0 * f - 16.0


def _complement(positive, target):
    """The muted complement of ``positive`` with lightness ``target``.

    The hue is rotated half a turn and the saturation scaled by
    ``_NEGATIVE_SATURATION``; the HLS lightness is then bisected until the
    color's CIE L* matches ``target``.
    """
    hue, _, sat = colorsys.rgb_to_hls(*hex_to_rgb(to_hex(positive)))
    hue = (hue + 0.5) % 1.0
    sat *= _NEGATIVE_SATURATION
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _lightness(colorsys.hls_to_rgb(hue, mid, sat)) < target:
            lo = mid
        else:
            hi = mid
    return rgb_to_hex(colorsys.hls_to_rgb(hue, 0.5 * (lo + hi), sat))


def _signed_cmap(background, positive):
    """A diverging map: muted complement, ``background`` at 0.5, ``positive``.

    Both ends sit at the same CIE L* distance from the background (the
    complement takes the positive color's lightness, on whichever side of
    the background that lies), so equal magnitudes of either sign read
    equally strong and only the hue tells the sign.
    """
    negative = _complement(positive, _lightness(positive))
    return LinearSegmentedColormap.from_list(
        "propagation_signed",
        [negative, to_hex(background), to_hex(positive)],
        N=_SIGNED_N,
    )


def _edges(centers, scale, name):
    """Cell edges of pixel-center coordinates on a linear or log axis.

    Interior edges are the arithmetic (linear) or geometric (log) midpoints
    of neighboring centers; the two outer edges are placed half a
    neighboring cell beyond the end centers, in the same sense. Works for
    increasing or decreasing coordinates and for nonuniform spacing.

    Raises:
        ValueError: Fewer than two centers, centers that are not strictly
            monotonic, or non-positive centers on a log axis.
    """
    c = np.asarray(centers, dtype=float)
    if c.ndim != 1 or c.size < 2:
        msg = f"plot_propagation: {name} needs at least two pixel centers"
        raise ValueError(msg)
    step = np.diff(c)
    if not (np.all(step > 0) or np.all(step < 0)):
        msg = f"plot_propagation: {name} must be strictly monotonic"
        raise ValueError(msg)
    if scale == "log":
        if np.any(c <= 0):
            msg = f"plot_propagation: a log {name} axis needs positive centers"
            raise ValueError(msg)
        return np.exp(_edges(np.log(c), "linear", name))
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - 0.5 * step[0]], mid, [c[-1] + 0.5 * step[-1]]])


def _strip_edges(z0, z_edges, z_scale):
    """The z extent of a screen or phase strip centered on ``z0``."""
    if z_scale == "log":
        span = abs(np.log(z_edges[-1]) - np.log(z_edges[0]))
        half = 0.5 * _ELEMENT_WIDTH * span
        return z0 * np.exp(-half), z0 * np.exp(half)
    half = 0.5 * _ELEMENT_WIDTH * abs(z_edges[-1] - z_edges[0])
    return z0 - half, z0 + half


def _neutral(ax, level):
    """A tone ``level`` of the way from the axes background to the text color."""
    back = np.asarray(matplotlib.colors.to_rgb(ax.get_facecolor()))
    front = np.asarray(matplotlib.colors.to_rgb(matplotlib.rcParams["text.color"]))
    return tuple(back + (front - back) * float(level))


def _validate(kind, norm, z_scale, vlim):
    """Argument checks that need no data.

    Raises:
        ValueError: An unknown ``kind``, ``norm`` or ``z_scale``; a log norm
            for a signed or cyclic quantity; a ``vlim`` for the phase (its
            range is fixed); or a ``vlim`` pair for the signed field.
    """
    if kind not in _KINDS:
        msg = f"plot_propagation: unknown kind {kind!r}; use one of {_KINDS}"
        raise ValueError(msg)
    if norm not in _NORMS:
        msg = f"plot_propagation: unknown norm {norm!r}; use one of {_NORMS}"
        raise ValueError(msg)
    if z_scale not in _SCALES:
        msg = f"plot_propagation: unknown z_scale {z_scale!r}; use one of {_SCALES}"
        raise ValueError(msg)
    if norm == "log" and kind in ("real", "phase"):
        msg = f"plot_propagation: norm='log' needs a positive quantity, not {kind!r}"
        raise ValueError(msg)
    if kind == "phase" and vlim is not None:
        msg = "plot_propagation: the phase range is fixed at [-pi, pi]; omit vlim"
        raise ValueError(msg)
    if kind == "real" and vlim is not None and np.ndim(vlim) != 0:
        msg = (
            "plot_propagation: kind='real' takes one vlim, the symmetric "
            "range [-vlim, vlim], so that zero stays at the colormap center"
        )
        raise ValueError(msg)


def _quantity(field, kind):
    """The real 2D array ``[z, x]`` a map of ``kind`` draws (phase unmasked)."""
    if kind == "real":
        return np.real(field)
    if kind == "amplitude":
        return np.abs(field)
    if kind == "intensity":
        if np.iscomplexobj(field):
            return field.real**2 + field.imag**2
        return np.asarray(field, dtype=float)
    return np.angle(field)


def _limits(values, kind, norm, vlim):
    """``(vmin, vmax)`` of the norm, from ``vlim`` or from the first frame."""
    if kind == "phase":
        return -np.pi, np.pi
    if kind == "real":
        top = float(np.nanmax(np.abs(values))) if vlim is None else float(vlim)
        top = top if top > 0 else 1.0
        return -top, top
    if vlim is not None and np.ndim(vlim) != 0:
        lo, hi = (float(v) for v in vlim)
        return lo, hi
    top = float(np.nanmax(values)) if vlim is None else float(vlim)
    top = top if top > 0 else 1.0
    if norm == "log":
        return top * 10.0**-_LOG_DECADES, top
    return 0.0, top


def _phase_strip_values(spec, x_edges, x_lo, x_hi):
    """``(edges, phase)`` of a phase strip from its values or callable.

    An array gives the phase at the centers of equal cells spanning
    ``[x_lo, x_hi]``. A callable is evaluated at cell centers spaced at the
    map's own x pitch across ``[x_lo, x_hi]``.
    """
    if callable(spec):
        pitch = float(np.min(np.abs(np.diff(x_edges))))
        n = max(2, int(np.ceil(abs(x_hi - x_lo) / pitch)))
        edges = np.linspace(x_lo, x_hi, n + 1)
        phase = np.asarray(spec(0.5 * (edges[1:] + edges[:-1])), dtype=float)
    else:
        phase = np.asarray(spec, dtype=float).ravel()
        edges = np.linspace(x_lo, x_hi, phase.size + 1)
    return edges, np.angle(np.exp(1j * phase))


def _draw_elements(ax, elements, x_edges, z_edges, z_scale, rasterized):
    """Draw the bench elements; return ``{"fill", "collection", "lines", "text"}``.

    Raises:
        ValueError: An element of unknown kind or the wrong length.
    """
    drawn = {"fill": [], "collection": [], "lines": [], "text": []}
    x_lo_map, x_hi_map = sorted((x_edges[0], x_edges[-1]))
    span = x_hi_map - x_lo_map
    # Screens run well past the map so their ends never show; the axes clip.
    bar_lo, bar_hi = x_lo_map - span, x_hi_map + span
    for i, element in enumerate(elements):
        kind = element[0] if len(element) else None
        gid = f"propagation/{kind}/{i}"
        if kind == "screen" and len(element) == 3:
            _, z0, openings = element
            za, zb = _strip_edges(float(z0), z_edges, z_scale)
            cuts = [e for pair in sorted(openings) for e in sorted(pair)]
            stops = [bar_lo, *cuts, bar_hi]
            for lo, hi in zip(stops[0::2], stops[1::2], strict=True):
                if hi <= lo:
                    continue
                bar = Rectangle(
                    (za, lo),
                    zb - za,
                    hi - lo,
                    facecolor=_neutral(ax, _SCREEN_LEVEL),
                    edgecolor="none",
                    zorder=4,
                    gid=gid,
                )
                drawn["fill"].append(ax.add_patch(bar))
        elif kind == "phase" and len(element) == 5:
            _, z0, x_lo, x_hi, spec = element
            za, zb = _strip_edges(float(z0), z_edges, z_scale)
            edges, phase = _phase_strip_values(spec, x_edges, x_lo, x_hi)
            strip = ax.pcolormesh(
                [za, zb],
                edges,
                phase[:, None],
                cmap=hwostyle.cmaps.phase,
                vmin=-np.pi,
                vmax=np.pi,
                shading="flat",
                rasterized=rasterized,
                zorder=5,
            )
            strip.set_gid(gid)
            drawn["collection"].append(strip)
        elif kind == "plane" and len(element) == 3:
            _, z0, label = element
            line = ax.axvline(
                float(z0),
                color=_neutral(ax, _PLANE_LEVEL),
                ls="--",
                lw=_PLANE_LW * matplotlib.rcParams["lines.linewidth"],
                zorder=6,
            )
            line.set_gid(gid)
            drawn["lines"].append(line)
            if label:
                text = ax.annotate(
                    label,
                    xy=(float(z0), 1.0),
                    xycoords=blended_transform_factory(ax.transData, ax.transAxes),
                    xytext=(3.0, -3.0),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    zorder=7,
                    gid=gid,
                )
                drawn["text"].append(text)
        else:
            msg = (
                f"plot_propagation: element {i} {element!r} is not one of "
                "('screen', z0, openings), ('phase', z0, x_lo, x_hi, values), "
                "('plane', z0, label)"
            )
            raise ValueError(msg)
    return drawn


def plot_propagation(
    field,
    x,
    z,
    *,
    ax=None,
    kind="real",
    norm="linear",
    vlim=None,
    cmap=None,
    z_scale="linear",
    z_label=None,
    x_label=None,
    elements=(),
    phase_floor=None,
    rasterized=True,
    colorbar=True,
    cbar_label=None,
):
    """Draw a field over distance and transverse position as a map.

    ``field`` is indexed ``[z, x]``: row ``j`` is the transverse cut at
    distance ``z[j]``. The map puts ``z`` on the horizontal axis, the
    direction of travel to the right as on an optical bench, and ``x`` on
    the vertical axis, so the drawn array is ``field.T``. ``x`` and ``z``
    are pixel-center coordinates; the cell edges are their midpoints
    (geometric midpoints on a log ``z`` axis), and every cell is drawn at
    its true edges with ``pcolormesh``, never interpolated. Decreasing
    coordinates are kept as given, so a ``z`` that decreases (a Fresnel
    number, which falls with distance) draws with its axis inverted.

    The four kinds:

    - ``"real"``: the instantaneous field ``Re E`` on a symmetric range
      ``[-vlim, vlim]``, through a diverging map centered on the axes
      background color: zero is the background, positive values run to the
      starlight role color (``hwostyle.roles.star``), negative values to a
      muted complement of it (hue rotated half a turn, saturation reduced)
      at the same CIE lightness distance from the background. Unlit regions
      therefore read as background, and the two signs read equally strong.
      The map is built at call time from the active hwostyle mode.
    - ``"intensity"``: ``|E|^2`` (or a real input as given), through the
      hwostyle intensity colormap, linear or log.
    - ``"amplitude"``: ``|E|``, likewise.
    - ``"phase"``: ``arg E`` on ``[-pi, pi]`` through the hwostyle phase
      colormap, hatched where ``|E|`` falls below ``phase_floor``, so an
      unlit region shows no phase rather than its numerical noise.

    ``elements`` draws the bench on top of the map, in data coordinates:

    - ``("screen", z0, [(x_lo, x_hi), ...])``: an opaque screen seen edge
      on, a neutral bar across all ``x`` except the listed openings.
    - ``("phase", z0, x_lo, x_hi, values)``: a thin strip colored by the
      wrapped phase of an element (a lens or a mask drawn as the phase
      screen it is, not as glass). ``values`` is an array of phases at the
      centers of equal cells spanning ``[x_lo, x_hi]``, or a callable of
      ``x`` evaluated at the map's own ``x`` pitch.
    - ``("plane", z0, label)``: a dashed line marking a plane (a focal
      plane, a Talbot plane), with ``label`` written at its top when not
      empty.

    Screens and strips are ``1.5%`` of the ``z`` axis wide (of its log span
    on a log axis), centered on ``z0``. Every artist of element ``i`` has
    the gid ``"propagation/<kind>/<i>"``.

    Example::

        res = plot_propagation(
            field, x, z,
            elements=[("screen", 0.0, [(-8, 8)]), ("plane", 40.0, "focal plane")],
        )
        for frame in frames:          # same grid, a new instant
            res.update(frame)

    Args:
        field: 2D array-like indexed ``[z, x]``, complex (or real for
            ``kind="real"`` or ``"intensity"``).
        x: 1D transverse pixel centers, ``field.shape[1]`` of them.
        z: 1D axial pixel centers, ``field.shape[0]`` of them, positive for
            ``z_scale="log"``.
        ax: Axes to draw into. None creates a new figure.
        kind: ``"real"`` (default), ``"intensity"``, ``"amplitude"`` or
            ``"phase"``.
        norm: ``"linear"`` (default) or ``"log"`` (intensity and amplitude
            only).
        vlim: The color range. For ``"real"`` a positive scalar, the
            symmetric range ``[-vlim, vlim]``. For ``"intensity"`` and
            ``"amplitude"`` a ``(vmin, vmax)`` pair or a scalar ``vmax``
            (with ``vmin`` 0 for a linear norm, ``vmax`` times ``1e-4`` for
            a log norm). Not accepted for ``"phase"``, whose range is fixed.
            None takes the range from this first frame (the peak ``|Re E|``,
            or the peak of the quantity); it then stays fixed, so frames
            passed to ``update`` are drawn on the first frame's range and an
            animation does not rescale.
        cmap: Colormap override. None uses the signed map for ``"real"``,
            ``hwostyle.cmaps.intensity`` for intensity and amplitude, and
            ``hwostyle.cmaps.phase`` for phase.
        z_scale: ``"linear"`` (default) or ``"log"`` for the ``z`` axis.
        z_label: Label of the ``z`` axis. None writes ``"$z$"``; an empty
            string leaves it blank.
        x_label: Label of the ``x`` axis. None writes ``"$x$"``.
        elements: Sequence of bench element tuples, described above.
        phase_floor: Absolute ``|E|`` below which ``kind="phase"`` is
            hatched. None uses ``1e-3`` of this first frame's peak ``|E|``,
            fixed thereafter.
        rasterized: Rasterize the map and phase strips in vector output.
            A vector cell per sample makes a PDF of megabytes.
        colorbar: Whether to attach a colorbar beside the axes.
        cbar_label: Colorbar label. None names the quantity drawn.

    Returns:
        A ``PlotResult``. ``artists["image"]`` is the map's ``QuadMesh``,
        whose array is ``field.T`` reduced to the drawn quantity.
        ``artists["cbar"]`` is the colorbar, when drawn.
        ``artists["fill"]``, when used, lists the phase-mask hatch (for
        ``kind="phase"``, gid ``"propagation/blank"``) and then the screen
        bars in element order. ``artists["collection"]`` lists the phase
        strips, ``artists["lines"]`` the plane lines and ``artists["text"]``
        their labels, each in element order. ``update(field)`` redraws the
        map from a new field on the same grid under the same norm and
        phase floor, adding no artist.

    Raises:
        ValueError: An unknown ``kind``, ``norm`` or ``z_scale``; a log
            norm for ``"real"`` or ``"phase"``; a ``vlim`` for ``"phase"``
            or a ``vlim`` pair for ``"real"``; a ``field`` that is not 2D or
            does not match ``(len(z), len(x))``; coordinates that are not
            strictly monotonic (or not positive on a log axis); or an
            element of unknown form. ``update`` raises on a field of a
            different shape.
    """
    ep = _require.eyepiece()
    _validate(kind, norm, z_scale, vlim)
    data = np.asarray(field)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    shape = (z.size, x.size)
    if data.shape != shape:
        msg = (
            f"plot_propagation: field has shape {data.shape}; expected "
            f"(len(z), len(x)) = {shape}, indexed [z, x]"
        )
        raise ValueError(msg)
    x_edges = _edges(x, "linear", "x")
    z_edges = _edges(z, z_scale, "z")

    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    values = _quantity(data, kind)
    vmin, vmax = _limits(values, kind, norm, vlim)
    if kind == "phase" and phase_floor is None:
        peak = float(np.max(np.abs(data)))
        phase_floor = _PHASE_FLOOR_REL * peak

    if cmap is not None:
        resolved = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    elif kind == "real":
        resolved = _signed_cmap(ax.get_facecolor(), hwostyle.roles.star)
    elif kind == "phase":
        resolved = _resolved_phase_cmap(None, ax)
    else:
        value = hwostyle.cmaps.intensity
        resolved = matplotlib.colormaps[value] if isinstance(value, str) else value
    resolved = resolved.with_extremes(bad=ax.get_facecolor())

    artists = {}
    fills = []
    if kind == "phase":
        hatch = _hatch_patch(ax, _HATCH, _hatch_color(None))
        hatch.set_gid("propagation/blank")
        fills.append(hatch)
        resolved = resolved.with_extremes(bad=(0.0, 0.0, 0.0, 0.0))

    def drawn(vals, raw):
        if kind == "phase":
            vals = np.where(np.abs(raw) >= phase_floor, vals, np.nan)
            return np.ma.masked_invalid(vals.T)
        if norm == "log":
            return np.clip(vals, vmin, None).T
        return vals.T

    norm_obj = LogNorm(vmin, vmax) if norm == "log" else Normalize(vmin, vmax)
    mesh = ax.pcolormesh(
        z_edges,
        x_edges,
        drawn(values, data),
        cmap=resolved,
        norm=norm_obj,
        shading="flat",
        rasterized=rasterized,
        zorder=1,
    )
    artists["image"] = mesh
    if z_scale == "log":
        ax.set_xscale("log")

    parts = _draw_elements(ax, elements, x_edges, z_edges, z_scale, rasterized)
    fills.extend(parts["fill"])
    if fills:
        artists["fill"] = fills
    for key in ("collection", "lines", "text"):
        if parts[key]:
            artists[key] = parts[key]

    ax.set_xlim(z_edges[0], z_edges[-1])
    ax.set_ylim(x_edges[0], x_edges[-1])
    ax.set_xlabel("$z$" if z_label is None else z_label)
    ax.set_ylabel("$x$" if x_label is None else x_label)

    if colorbar:
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        label = _CBAR_LABELS[kind] if cbar_label is None else cbar_label
        cbar = ax.figure.colorbar(mesh, cax=cax, label=label)
        if kind == "phase":
            cbar.set_ticks(list(_PHASE_TICKS))
            cbar.set_ticklabels(list(_PHASE_TICKLABELS))
        artists["cbar"] = cbar

    def update(new_field):
        new = np.asarray(new_field)
        if new.shape != shape:
            msg = (
                f"plot_propagation: update got shape {new.shape}; the map "
                f"is drawn on shape {shape}"
            )
            raise ValueError(msg)
        mesh.set_array(drawn(_quantity(new, kind), new))

    return ep.PlotResult(ax=ax, artists=artists, update=update)

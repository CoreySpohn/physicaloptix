"""Views of diffraction between the near field and the far field.

Three pictures of Fresnel diffraction, each in units of the wavelength:

- ``plot_near_to_far`` draws the brightness behind a slit over the Fresnel
  number ``N_F = (b/2)^2 / (lambda z)`` (log, falling with distance, so the
  far field is at the right) and position across the slit, or angle, as a
  propagation map; ``prepare_near_to_far`` computes that map by the exact
  1D angular spectrum.
- ``plot_fresnel_zones`` draws a round opening face-on, colored by the
  phase of the path from each of its points to a point on the axis behind
  it: each ring of half a turn is a Fresnel zone, and the opening holds
  about ``N_F`` of them.
- ``plot_cornu`` draws the Cornu spiral ``C(v) + i S(v)`` with a window of
  it thick and the window's chord as an arrow: in units of
  ``sqrt(lambda z / 2)`` a slit picks a window of the spiral, and the
  brightness behind it relative to the incident wave is half the squared
  chord.
"""

import math
from typing import NamedTuple

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from scipy.special import fresnel

from physicaloptix.viz import _require
from physicaloptix.viz.fields import _resolved_phase_cmap
from physicaloptix.viz.propagation import _neutral, plot_propagation

_AXES = ("position", "angle")
_NORMALIZE = (None, "row")
_NF_TICKS = (20.0, 10.0, 3.0, 1.0, 0.3, 0.1)
_NF_LABEL = r"Fresnel number $N_F=(b/2)^2/(\lambda z)$, farther right"
_X_LABELS = {
    "position": "position across the slit, $x/b$",
    "angle": r"angle, $\sin\theta=X/z$ ($\lambda/b$)",
}
# Slit edges: dotted lines in a bright neutral, this fraction of the base width.
_EDGE_LEVEL, _EDGE_LW = 0.85, 0.8
# Zones: the opening's rim in the hardware neutral, the zone boundaries
# thin and bright, the screen around the opening hatched in a dim neutral.
_RIM_LEVEL, _RING_LEVEL, _SCREEN_LEVEL = 0.7, 0.85, 0.3
_RIM_LW, _RING_LW = 1.2, 0.6
_SCREEN_OUTER = 1.25
_ZONE_LIM = 1.27
_HATCH = "////"
# Cornu: the spiral thin in a mid neutral, the window thick, the eyes (the
# spiral's limit points) as plus signs.
_SPIRAL_LEVEL, _EYE_LEVEL = 0.5, 0.6
_SPIRAL_LW, _WINDOW_LW = 0.7, 2.2
_SPIRAL_SAMPLES, _WINDOW_SAMPLES = 3001, 801
_CHORD_HEAD = 0.7


class NearToFar(NamedTuple):
    """A slit's brightness behind it, ready for ``plot_near_to_far``.

    The fields are in ``plot_near_to_far``'s argument order, so
    ``plot_near_to_far(*prepared)`` draws it.

    Attributes:
        intensity: Brightness indexed ``[row, x]``, relative to the incident
            wave (or to each row's peak, when prepared with
            ``normalize="row"``).
        x: Position across the slit in units of its width ``b``, or angle
            ``sin(theta)`` in units of ``lambda / b``.
        nf: Fresnel number of each row.
        axis: ``"position"`` or ``"angle"``, what ``x`` holds.
    """

    intensity: np.ndarray
    x: np.ndarray
    nf: np.ndarray
    axis: str


def _angular_spectrum_1d(field, pitch, z, pad):
    """Propagate a 1D field by ``z`` (wavelengths) on a window ``pad`` times wider.

    Returns:
        ``(x, propagated)`` on the padded grid, ``x`` centered like the input.
    """
    n = field.size
    total = int(pad * n)
    start = (total - n) // 2
    padded = np.zeros(total, dtype=complex)
    padded[start : start + n] = field
    f = np.fft.fftfreq(total, d=pitch)
    # Decays past f = 1 / lambda: the evanescent frequencies.
    kz = np.sqrt((1.0 - f**2).astype(complex))
    out = np.fft.ifft(np.fft.fft(padded) * np.exp(2j * np.pi * z * kz))
    x = (np.arange(total) - start - (n - 1) / 2) * pitch
    return x, out


def prepare_near_to_far(
    nf,
    *,
    width=40.0,
    pitch=0.2,
    pad=120,
    x_half=3.0,
    angles=None,
    normalize=None,
):
    """A slit's brightness from the near field to the far field.

    A unit plane wave through a slit of ``width`` wavelengths is propagated
    to the distance ``z = (width/2)^2 / N_F`` of each row by the exact
    angular-spectrum transfer function ``exp(i 2 pi z sqrt(1/lambda^2 -
    f^2))``, on a window ``pad`` times the slit's width so the spread stays
    inside the periodic window. Each row is read across position, or, with
    ``angles``, interpolated at ``X = sin(theta) z``, so a pattern fixed in
    angle draws as horizontal bands.

    The padding must hold the far-field spread at the smallest Fresnel
    number: a slit of 40 wavelengths at 0.2 per sample and ``N_F = 0.1``
    needs about 120 (30 still wraps the side bands onto the axis by 8%).

    Example::

        nf = np.geomspace(26.0, 0.08, 241)
        res = plot_near_to_far(*prepare_near_to_far(nf))

    Args:
        nf: Fresnel numbers of the rows, positive (any order; decreasing
            draws the far field at the right).
        width: Slit width ``b``, in wavelengths.
        pitch: Sample pitch, in wavelengths.
        pad: Padding factor of the propagation window, at least 1.
        x_half: Half-width of the kept positions, in units of ``b``
            (position axis only).
        angles: None for a position axis, or ``sin(theta)`` of each column
            in units of ``lambda / b`` for an angle axis.
        normalize: None for brightness relative to the incident wave, or
            ``"row"`` to divide each row by its largest value.

    Returns:
        A ``NearToFar``.

    Raises:
        ValueError: Non-positive Fresnel numbers, a slit narrower than two
            samples, a padding below 1, or an unknown ``normalize``.
    """
    nf = np.asarray(nf, dtype=float).ravel()
    if nf.size == 0 or np.any(nf <= 0):
        msg = "prepare_near_to_far: Fresnel numbers must be positive"
        raise ValueError(msg)
    if normalize not in _NORMALIZE:
        msg = f"prepare_near_to_far: normalize must be None or 'row', not {normalize!r}"
        raise ValueError(msg)
    n = round(width / pitch)
    if n < 2:
        msg = "prepare_near_to_far: the slit must span at least two samples"
        raise ValueError(msg)
    if pad < 1:
        msg = f"prepare_near_to_far: pad must be at least 1, not {pad!r}"
        raise ValueError(msg)
    slit = np.ones(n, dtype=complex)
    theta = None if angles is None else np.asarray(angles, dtype=float).ravel()
    keep = None
    rows = []
    for value in nf:
        z = (0.5 * width) ** 2 / value
        x, out = _angular_spectrum_1d(slit, pitch, z, pad)
        if theta is None:
            if keep is None:
                keep = np.abs(x) <= x_half * width
            row = np.abs(out[keep]) ** 2
        else:
            row = np.interp(theta * z / width, x, np.abs(out) ** 2)
        rows.append(row / row.max() if normalize == "row" else row)
    if theta is None:
        return NearToFar(np.array(rows), x[keep] / width, nf, "position")
    return NearToFar(np.array(rows), theta, nf, "angle")


def plot_near_to_far(
    intensity,
    x,
    nf,
    axis="position",
    *,
    ax=None,
    vlim=None,
    cmap=None,
    nf_ticks=_NF_TICKS,
    edges=None,
    edge_kw=None,
    x_label=None,
    z_label=None,
    elements=(),
    colorbar=True,
    cbar_label=None,
    rasterized=True,
    gid="near-to-far",
):
    """Draw a slit's brightness over log Fresnel number and position or angle.

    A propagation map (``plot_propagation`` with ``kind="intensity"`` on a
    log axis) with the Fresnel number across, ticked at round values, and
    position across the slit (or angle) up. Because ``N_F`` falls with
    distance, a decreasing ``nf`` puts the slit at the left and the far
    field at the right. On a position axis the slit's edges are dotted, so
    the near field's sharp shadow reads against them; the regimes read off
    it as a geometric shadow at large ``N_F``, Fresnel ripples near 1, and
    the far-field shape, only widening, well below 1. On an angle axis, a
    pattern fixed in angle is a set of horizontal bands.

    The axes and the map carry gids (``gid`` and ``"<gid>/image"``), so a
    figure that carries the map into another can find and check it.

    Example::

        res = plot_near_to_far(*prepare_near_to_far(np.geomspace(26, 0.08, 241)))
        theta = np.linspace(-8.0, 8.0, 241)
        prep = prepare_near_to_far(nf, angles=theta, normalize="row")
        plot_near_to_far(*prep, ax=right, vlim=1.0)

    Args:
        intensity: Brightness indexed ``[row, x]``, one row per Fresnel
            number.
        x: Column centers: position in units of the slit width, or angle.
        nf: Fresnel number of each row, positive and strictly monotonic.
        axis: ``"position"`` (default) or ``"angle"``; sets the default
            ``x_label`` and ``edges``.
        ax: Axes to draw into. None creates a new figure.
        vlim: Color range, as for ``plot_propagation`` with
            ``kind="intensity"``: a ``(vmin, vmax)`` pair or a scalar
            ``vmax``. None uses the peak.
        cmap: Colormap override. None uses ``hwostyle.cmaps.intensity``.
        nf_ticks: Fresnel numbers ticked on the horizontal axis (those
            inside the drawn range), labeled plainly; minor ticks are off.
            None keeps matplotlib's log ticks.
        edges: Heights of the dotted edge lines. None draws the slit's
            edges, ``(-0.5, 0.5)``, on a position axis and none on an angle
            axis.
        edge_kw: Extra keyword arguments for the edge lines, applied last.
        x_label: Label of the vertical axis. None names position or angle.
        z_label: Label of the horizontal axis. None names the Fresnel
            number and its definition.
        elements: Bench elements passed to ``plot_propagation`` (a
            ``("plane", N_F, label)`` marks a Fresnel number).
        colorbar: Whether to attach a colorbar beside the axes.
        cbar_label: Colorbar label. None uses ``plot_propagation``'s.
        rasterized: Rasterize the map in vector output.
        gid: The axes' gid and the prefix of the artists' gids.

    Returns:
        A ``PlotResult``: the ``plot_propagation`` artists (``"image"``,
        ``"cbar"``, and those of ``elements``), with the edge lines (gids
        ``"<gid>/edge/<i>"``) appended to ``artists["lines"]`` after any
        plane lines. ``update(intensity)`` redraws the map on the same grid
        and color range.

    Raises:
        ValueError: An unknown ``axis``, or anything ``plot_propagation``
            rejects (a shape mismatch, a non-monotonic or non-positive
            ``nf``).
    """
    ep = _require.eyepiece()
    if axis not in _AXES:
        msg = f"plot_near_to_far: axis must be one of {_AXES}, not {axis!r}"
        raise ValueError(msg)
    res = plot_propagation(
        intensity,
        x,
        nf,
        ax=ax,
        kind="intensity",
        vlim=vlim,
        cmap=cmap,
        z_scale="log",
        z_label=_NF_LABEL if z_label is None else z_label,
        x_label=_X_LABELS[axis] if x_label is None else x_label,
        elements=elements,
        colorbar=colorbar,
        cbar_label=cbar_label,
        rasterized=rasterized,
    )
    ax = res.ax
    ax.set_gid(gid)
    res.artists["image"].set_gid(f"{gid}/image")
    if nf_ticks is not None:
        lo, hi = sorted(ax.get_xlim())
        ticks = [t for t in nf_ticks if lo <= t <= hi]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
    if edges is None:
        edges = (-0.5, 0.5) if axis == "position" else ()
    kw = {
        "color": _neutral(ax, _EDGE_LEVEL),
        "ls": ":",
        "lw": _EDGE_LW * matplotlib.rcParams["lines.linewidth"],
        **(edge_kw or {}),
    }
    artists = dict(res.artists)
    lines = list(artists.get("lines", []))
    for i, height in enumerate(edges):
        line = ax.axhline(float(height), **kw)
        line.set_gid(f"{gid}/edge/{i}")
        lines.append(line)
    if lines:
        artists["lines"] = lines
    return ep.PlotResult(ax=ax, artists=artists, update=res.update)


def _zone_radii(z, radius):
    """Radii (wavelengths) inside ``radius`` where the path grows by m half waves."""
    m_max = (np.hypot(z, radius) - z) / 0.5
    m = np.arange(1, int(m_max) + 1)
    return np.sqrt((z + 0.5 * m) ** 2 - z**2)


def _bowl_phase(x, z, radius):
    """The path's phase ``2 pi (R - z)`` across the opening, masked outside it."""
    rho = radius * np.hypot(*np.meshgrid(x, x))
    phase = np.angle(np.exp(2j * np.pi * (np.hypot(z, rho) - z)))
    return np.ma.masked_where(rho > radius, phase)


def plot_fresnel_zones(
    nf,
    *,
    radius=50.0,
    ax=None,
    samples=241,
    max_rings=None,
    cmap=None,
    screen=True,
    ring_color=None,
    rim_color=None,
    gid="fresnel-zones",
):
    """Draw a round opening face-on, colored by its Fresnel zones.

    Seen from a point on the axis at distance ``z`` behind an opening of
    radius ``a``, the path from a point of the opening at radius ``rho`` is
    ``R = sqrt(z^2 + rho^2)``. The opening is colored by the phase of the
    path's excess, ``2 pi (R - z)`` (exact, not the paraxial bowl), so each
    ring across which the path grows by half a wavelength, a Fresnel zone,
    is half a turn of the phase colormap, and the zone boundaries, where
    ``R - z`` is a whole number of half waves, are drawn as thin circles.
    The opening holds about ``N_F = a^2 / (lambda z)`` zones; equal-area
    rings, they crowd toward the rim. The screen around the opening is
    hatched and its rim is drawn in the hardware neutral.

    Lengths are in wavelengths and the drawing in units of the radius, so
    ``radius`` sets only how far from paraxial the zones are: ``z =
    radius^2 / nf``.

    Example::

        res = plot_fresnel_zones(6.0, max_rings=12)
        for nf in np.geomspace(6.0, 0.5, 48):   # the point moves away
            res.update(nf)

    Args:
        nf: Fresnel number of the opening, ``a^2 / (lambda z)``, positive.
        radius: The opening's radius ``a``, in wavelengths.
        ax: Axes to draw into. None creates a new figure.
        samples: Pixels across the opening's diameter.
        max_rings: Number of boundary circles made. None makes as many as
            this ``nf`` needs; an animation toward larger ``nf`` (more
            zones) needs room for the most it reaches.
        cmap: Phase colormap override. None uses the hwostyle phase map.
        screen: Whether to draw the hatched screen around the opening.
        ring_color: Color of the zone boundaries. None uses an 85% neutral.
        rim_color: Color of the rim. None uses the 70% hardware neutral.
        gid: Prefix of the artists' gids.

    Returns:
        A ``PlotResult``. ``artists["image"]`` is the phase image (gid
        ``"<gid>/image"``), ``artists["fill"]`` the hatched screen (gid
        ``"<gid>/screen"``, when drawn), and ``artists["ellipse"]`` the
        boundary circles (gids ``"<gid>/ring/<i>"``, innermost first,
        unused ones hidden) followed by the rim (``"<gid>/rim"``).
        ``update(nf)`` redraws the phase and moves the boundary circles for
        a new Fresnel number.

    Raises:
        ValueError: A non-positive ``nf`` or ``radius``, fewer than two
            samples, or an ``nf`` (first or in ``update``) with more zone
            boundaries than ``max_rings``.
    """
    ep = _require.eyepiece()
    if not radius > 0:
        msg = f"plot_fresnel_zones: radius must be positive, not {radius!r}"
        raise ValueError(msg)
    if samples < 2:
        msg = f"plot_fresnel_zones: samples must be at least 2, not {samples!r}"
        raise ValueError(msg)

    def boundaries(value):
        if not value > 0:
            msg = f"plot_fresnel_zones: nf must be positive, not {value!r}"
            raise ValueError(msg)
        z = radius**2 / value
        return z, _zone_radii(z, radius) / radius

    z, bounds = boundaries(nf)
    n_rings = len(bounds) if max_rings is None else int(max_rings)
    if len(bounds) > n_rings:
        msg = (
            f"plot_fresnel_zones: nf = {nf} has {len(bounds)} zone boundaries; "
            f"max_rings is {n_rings}"
        )
        raise ValueError(msg)
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    base = float(matplotlib.rcParams["lines.linewidth"])
    x = np.linspace(-1.0, 1.0, samples)
    half = 0.5 * (x[1] - x[0])
    extent = (x[0] - half, x[-1] + half, x[0] - half, x[-1] + half)

    artists = {}
    if screen:
        wedge = Wedge(
            (0.0, 0.0),
            _SCREEN_OUTER,
            0.0,
            360.0,
            width=_SCREEN_OUTER - 1.0,
            facecolor="none",
            edgecolor=_neutral(ax, _SCREEN_LEVEL),
            hatch=_HATCH,
            lw=0,
            zorder=0,
            gid=f"{gid}/screen",
        )
        artists["fill"] = [ax.add_patch(wedge)]
    image = ax.imshow(
        _bowl_phase(x, z, radius),
        extent=extent,
        origin="lower",
        interpolation="nearest",
        # Outside the opening is transparent, so the hatched screen shows.
        cmap=_resolved_phase_cmap(cmap, ax).with_extremes(bad=(0.0, 0.0, 0.0, 0.0)),
        vmin=-math.pi,
        vmax=math.pi,
        zorder=1,
    )
    image.set_gid(f"{gid}/image")
    artists["image"] = image
    rings = []
    for i in range(n_rings):
        ring = Circle(
            (0.0, 0.0),
            float(bounds[i]) if i < len(bounds) else 0.5,
            facecolor="none",
            edgecolor=_neutral(ax, _RING_LEVEL) if ring_color is None else ring_color,
            lw=_RING_LW * base,
            zorder=3,
            visible=i < len(bounds),
            gid=f"{gid}/ring/{i}",
        )
        rings.append(ax.add_patch(ring))
    rim = Circle(
        (0.0, 0.0),
        1.0,
        facecolor="none",
        edgecolor=_neutral(ax, _RIM_LEVEL) if rim_color is None else rim_color,
        lw=_RIM_LW * base,
        zorder=4,
        gid=f"{gid}/rim",
    )
    artists["ellipse"] = [*rings, ax.add_patch(rim)]
    ax.set(xlim=(-_ZONE_LIM, _ZONE_LIM), ylim=(-_ZONE_LIM, _ZONE_LIM), aspect="equal")
    ax.axis("off")

    def update(new_nf):
        new_z, new_bounds = boundaries(new_nf)
        if len(new_bounds) > n_rings:
            msg = (
                f"plot_fresnel_zones: update nf = {new_nf} has {len(new_bounds)} "
                f"zone boundaries; {n_rings} circles were made (pass max_rings)"
            )
            raise ValueError(msg)
        image.set_data(_bowl_phase(x, new_z, radius))
        for i, ring in enumerate(rings):
            ring.set_visible(i < len(new_bounds))
            if i < len(new_bounds):
                ring.set_radius(float(new_bounds[i]))

    return ep.PlotResult(ax=ax, artists=artists, update=update)


def _cornu(v):
    """The Cornu spiral ``C(v) + i S(v)`` (Fresnel integrals of ``pi v^2 / 2``)."""
    s, c = fresnel(np.asarray(v, dtype=float))
    return c + 1j * s


def plot_cornu(
    window=None,
    *,
    ax=None,
    v_max=6.0,
    lim=0.85,
    color=None,
    spiral_color=None,
    chord_color=None,
    eyes=True,
    gid="cornu",
):
    """Draw the Cornu spiral with a window of it and the window's chord.

    The spiral is ``C(v) + i S(v)``, the Fresnel integrals of ``pi v^2 / 2``:
    the running sum of the arrows from a strip of an opening, in units of
    ``sqrt(lambda z / 2)`` across it. It winds into its two eyes at
    ``+-(1 + i)/2``. An opening from ``v0`` to ``v1`` picks that window of
    the spiral, drawn thick, and the chord from its start to its end is the
    sum of the window's arrows: the brightness behind the opening, relative
    to the incident wave, is half the squared chord (the whole spiral, eye
    to eye, is the unobstructed wave, chord ``1 + i``).

    For a slit of width ``b`` and Fresnel number ``N_F = (b/2)^2 / (lambda
    z)`` seen from position ``X`` (in units of ``b``) the window is
    ``((-1/2 - X) sqrt(8 N_F), (1/2 - X) sqrt(8 N_F))``; for a straight edge
    it runs from ``v0`` to infinity.

    Example::

        res = plot_cornu((-2.45, 2.45))        # a slit of N_F = 3, on axis
        res.update((-4.9, 0.0))                # the same slit, seen from its edge

    Args:
        window: ``(v0, v1)`` of the highlighted stretch; either end may be
            infinite (an edge). None draws the spiral alone.
        ax: Axes to draw into (made equal-aspect and frameless). None
            creates a new figure.
        v_max: The thin spiral runs from ``-v_max`` to ``v_max``, and an
            infinite window end is drawn to it.
        lim: Half-width of the drawn plane, centered on 0.
        color: Color of the window. None uses ``hwostyle.roles.star``.
        spiral_color: Color of the thin spiral. None uses a 50% neutral.
        chord_color: Color of the chord. None uses the text color.
        eyes: Whether to mark the eyes with plus signs.
        gid: Prefix of the artists' gids.

    Returns:
        A ``PlotResult``. ``artists["lines"]`` lists the spiral (gid
        ``"<gid>/spiral"``), then the window (``"<gid>/window"``) and the
        eyes (``"<gid>/eyes"``) when drawn; ``artists["arrow"]`` is the
        chord (``[FancyArrowPatch]``, gid ``"<gid>/chord"``) when a window
        is given. With a window, ``update(window)`` moves the window and
        its chord.

    Raises:
        ValueError: A window whose ends are equal or both infinite on the
            same side.
    """
    ep = _require.eyepiece()

    def parts(win):
        v0, v1 = (float(v) for v in win)
        if v0 == v1:
            msg = f"plot_cornu: the window {win!r} is empty"
            raise ValueError(msg)
        lo, hi = np.clip(sorted((v0, v1)), -v_max, v_max)
        if v0 > v1:
            lo, hi = hi, lo
        stretch = _cornu(np.linspace(lo, hi, _WINDOW_SAMPLES))
        return stretch, complex(_cornu(v0)), complex(_cornu(v1))

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    base = float(matplotlib.rcParams["lines.linewidth"])
    spiral = _cornu(np.linspace(-v_max, v_max, _SPIRAL_SAMPLES))
    (thin,) = ax.plot(
        spiral.real,
        spiral.imag,
        color=_neutral(ax, _SPIRAL_LEVEL) if spiral_color is None else spiral_color,
        lw=_SPIRAL_LW * base,
        gid=f"{gid}/spiral",
    )
    lines = [thin]
    artists = {}
    update = None
    if window is not None:
        stretch, start, end = parts(window)
        (thick,) = ax.plot(
            stretch.real,
            stretch.imag,
            color=hwostyle.roles.star if color is None else color,
            lw=_WINDOW_LW * base,
            zorder=3,
            gid=f"{gid}/window",
        )
        lines.append(thick)
        chord = ep.phasor(
            [end - start],
            ax=ax,
            starts=[start],
            colors=matplotlib.rcParams["text.color"]
            if chord_color is None
            else chord_color,
            head_scale=_CHORD_HEAD,
            cross=False,
            lim=lim,
            arrow_kw={"gid": f"{gid}/chord"},
        )
        artists["arrow"] = chord.artists["arrow"]

        def update(new_window):
            new_stretch, new_start, new_end = parts(new_window)
            thick.set_data(new_stretch.real, new_stretch.imag)
            chord.update([new_end - new_start], starts=[new_start])

    if eyes:
        (marks,) = ax.plot(
            [0.5, -0.5],
            [0.5, -0.5],
            "+",
            color=_neutral(ax, _EYE_LEVEL),
            ms=matplotlib.rcParams["lines.markersize"],
            gid=f"{gid}/eyes",
        )
        lines.append(marks)
    artists["lines"] = lines
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    ax.axis("off")
    return ep.PlotResult(ax=ax, artists=artists, update=update)

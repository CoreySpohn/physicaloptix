"""Side views of optical trains drawn from traced rays.

``ray_train`` draws a train of thin lenses with paraxial rays and filled
beams; ``mirror_train`` draws a train of curved mirrors with exactly traced
rays and beams, each mirror shown as its used piece over its dashed parent
surface; ``plot_mirror`` draws one such mirror. All three draw in data
coordinates ``(z, y)``: ``z`` along the bench to the right, ``y`` up. The
tracers they call are in ``physicaloptix.viz.raytrace``.

A beam is the light between two edge rays. Between the surfaces both edge
rays are straight, so the beam is filled one quadrilateral per segment;
where the edge rays cross (a focus) the quadrilateral is a bow tie, which
fills as two triangles meeting at the crossing, so a beam pinches to a
point at every focus, as light does, rather than as an envelope that keeps
a width there.
"""

import numbers

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.patches import Ellipse

from physicaloptix.viz import _require
from physicaloptix.viz.propagation import _neutral
from physicaloptix.viz.raytrace import (
    beam_polygons,
    strip_polygons,
    trace_mirrors,
    trace_paraxial,
)

# Beams are a translucent fill of the ray color, under the rays.
_BEAM_ALPHA = 0.28
_BEAM_Z, _LENS_Z, _MIRROR_Z, _RAY_Z = 2, 3, 4, 5
# Lens glyphs: a faint ellipse, its face a low neutral at this opacity and
# its edge a brighter one; the edge is this fraction of the base line width.
_LENS_FACE_LEVEL, _LENS_FACE_ALPHA, _LENS_EDGE_LEVEL = 0.25, 0.6, 0.7
_LENS_EDGE_LW = 0.56
# A lens glyph spans this factor times the largest height a ray reaches at
# any lens, and is this fraction of the z span wide, by default.
_LENS_HALF_PAD = 1.2
_LENS_WIDTH_FRAC = 0.01
# Mirrors: the used piece thick in the hardware neutral, the parent surface
# thin, dashed and fainter; widths are fractions of the base line width.
_MIRROR_LEVEL, _PARENT_LEVEL = 0.7, 0.45
_MIRROR_LW, _PARENT_LW = 3.0, 0.8
_PARENT_DASH = (0, (3, 2))
# The used piece found from the hits is padded by this much on each side.
_USED_PAD = 0.05


def _is_linestyle(value):
    """A single line style: a name, or a dash pattern ``(offset, (on, off...))``."""
    if isinstance(value, str):
        return True
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], numbers.Number)
        and isinstance(value[1], (tuple, list))
    )


def _is_number(value):
    return isinstance(value, numbers.Number)


def _per_item(value, n, default, name, owner, single):
    """One value per item: ``default`` for None, a single value repeated, or a list.

    Raises:
        ValueError: A sequence whose length is not ``n``.
    """
    if value is None:
        return [default] * n
    if single(value):
        return [value] * n
    values = list(value)
    if len(values) != n:
        msg = f"{owner}: {name} has {len(values)} entries for {n} items"
        raise ValueError(msg)
    return values


def _styles(owner, n, colors, linestyles, linewidths, zorder, default_color):
    lw = float(matplotlib.rcParams["lines.linewidth"])
    return (
        _per_item(colors, n, default_color, "colors", owner, is_color_like),
        _per_item(linestyles, n, "-", "linestyles", owner, _is_linestyle),
        _per_item(linewidths, n, lw, "linewidths", owner, _is_number),
        _per_item(zorder, n, _RAY_Z, "zorder", owner, _is_number),
    )


def _draw_rays(ax, owner, polylines, styles):
    colors, linestyles, linewidths, zorders = styles
    lines = []
    for i, ((z, y), color, ls, lw, zo) in enumerate(
        zip(polylines, colors, linestyles, linewidths, zorders, strict=True)
    ):
        (line,) = ax.plot(
            z,
            y,
            color=color,
            ls=ls,
            lw=lw,
            zorder=zo,
            solid_capstyle="butt",
            gid=f"{owner}/ray/{i}",
        )
        lines.append(line)
    return lines


def _draw_beams(ax, owner, polygon_sets, colors, alpha):
    fills = []
    for i, (polys, color) in enumerate(zip(polygon_sets, colors, strict=True)):
        coll = PolyCollection(
            polys,
            facecolors=to_rgba(color, alpha),
            edgecolors="none",
            zorder=_BEAM_Z,
        )
        coll.set_gid(f"{owner}/beam/{i}")
        fills.append(ax.add_collection(coll, autolim=True))
    return fills


def ray_train(
    lenses,
    rays=(),
    *,
    z_start,
    z_end,
    z_ref=0.0,
    beams=(),
    ax=None,
    colors=None,
    linestyles=None,
    linewidths=None,
    zorder=None,
    beam_colors=None,
    beam_alpha=_BEAM_ALPHA,
    lens_half=None,
    lens_width=None,
    lens_glyphs=True,
):
    """Draw a train of thin lenses with paraxial rays and filled beams.

    Each ray is traced by ``trace_paraxial`` from ``z_start`` to ``z_end``
    through every lens between them and drawn as its polyline. Each beam is
    the light between two edge rays of one slope, filled translucent beneath
    the rays, one quadrilateral per segment between lenses, so it pinches
    to a point at a focus. A beam's edge rays are not drawn unless also
    listed in ``rays``, so the edges can take their own styles.

    In a chain of 4f relays the marginal rays of an on-axis star (a beam of
    slope 0) meet at every focus, and the chief ray of an off-axis source
    (height 0, nonzero slope) crosses the axis at every pupil image.

    Example::

        lenses = [(1.5, 1.5), (4.5, 1.5), (7.5, 1.5)]
        res = ray_train(
            lenses,
            [(0.0, 0.15)],                     # an off-axis chief ray
            z_start=-1.0,
            z_end=9.0,
            beams=[(0.5, -0.5, 0.0)],          # the on-axis star's beam
            linestyles=[(0, (4, 2))],
        )
        res.update(rays=[(0.0, 0.10)])         # the source moves

    Args:
        lenses: ``(z, f)`` thin lenses, position and focal length.
        rays: ``(height, slope)`` of each drawn ray at ``z_ref``.
        z_start: Where every ray and beam starts.
        z_end: Where every ray and beam ends.
        z_ref: The plane where heights are given, before the first lens.
        beams: ``(top, bottom, slope)`` of each filled beam: its edge rays'
            heights at ``z_ref`` and their common slope.
        ax: Axes to draw into. None creates a new figure.
        colors: Ray color, one for all or one per ray. None uses the
            starlight role color, ``hwostyle.roles.star``.
        linestyles: Ray line style (a name or a dash pattern), one for all
            or one per ray. None draws solid rays.
        linewidths: Ray width in points, one for all or one per ray. None
            uses ``rcParams["lines.linewidth"]``.
        zorder: Ray z-order, one for all or one per ray. None uses 5, above
            the lenses (3) and the beams (2).
        beam_colors: Beam color, one for all or one per beam. None uses the
            starlight role color.
        beam_alpha: Opacity of the beam fill.
        lens_half: Half-height of the lens glyphs, one for all or one per
            lens. None uses 1.2 times the largest height any ray or beam
            edge reaches at any lens.
        lens_width: Width of the lens glyphs along ``z``. None uses 1% of
            ``z_end - z_start``.
        lens_glyphs: Whether to draw the lenses.

    Returns:
        A ``PlotResult``. ``artists["lines"]`` lists the rays (gid
        ``"ray-train/ray/<i>"``) in order; ``artists["fill"]`` lists one
        ``PolyCollection`` per beam (gid ``"ray-train/beam/<i>"``);
        ``artists["ellipse"]`` lists the lens glyphs (gid
        ``"ray-train/lens/<i>"``) in the order given. Keys with nothing
        drawn are absent. ``update(rays=None, beams=None)`` retraces new
        ``(height, slope)`` rays or ``(top, bottom, slope)`` beams, as many
        as first drawn, through the same lenses, keeping every style; None
        keeps the current ones.

    Raises:
        ValueError: ``z_end`` not after ``z_start``; a per-ray, per-beam or
            per-lens style of the wrong length; ``update`` with a different
            number of rays or beams.
    """
    ep = _require.eyepiece()
    owner = "ray-train"
    lenses = [(float(z), float(f)) for z, f in lenses]
    rays = [tuple(r) for r in rays]
    beams = [tuple(b) for b in beams]
    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    def traced_rays(specs):
        return [
            trace_paraxial(h, s, lenses, z_start, z_end, z_ref=z_ref) for h, s in specs
        ]

    def traced_beams(specs):
        polys, edges = [], []
        for top, bottom, slope in specs:
            up = trace_paraxial(top, slope, lenses, z_start, z_end, z_ref=z_ref)
            down = trace_paraxial(bottom, slope, lenses, z_start, z_end, z_ref=z_ref)
            polys.append(beam_polygons(up, down))
            edges += [up, down]
        return polys, edges

    ray_lines = traced_rays(rays)
    beam_polys, beam_edges = traced_beams(beams)
    star = hwostyle.roles.star
    styles = _styles(owner, len(rays), colors, linestyles, linewidths, zorder, star)
    fill_colors = _per_item(
        beam_colors, len(beams), star, "beam_colors", owner, is_color_like
    )

    artists = {}
    fills = _draw_beams(ax, owner, beam_polys, fill_colors, beam_alpha)
    if fills:
        artists["fill"] = fills
    lines = _draw_rays(ax, owner, ray_lines, styles)
    if lines:
        artists["lines"] = lines

    if lens_glyphs and lenses:
        if lens_half is None:
            # Interior vertices are the heights at the lenses crossed.
            heights = [abs(v) for _z, y in ray_lines + beam_edges for v in y[1:-1]]
            reach = max(heights, default=0.0)
            lens_half = _LENS_HALF_PAD * reach if reach > 0 else 1.0
        halves = _per_item(lens_half, len(lenses), 1.0, "lens_half", owner, _is_number)
        width = lens_width
        if width is None:
            width = _LENS_WIDTH_FRAC * (z_end - z_start)
        lw = _LENS_EDGE_LW * float(matplotlib.rcParams["lines.linewidth"])
        glyphs = []
        for i, ((z, _f), half) in enumerate(zip(lenses, halves, strict=True)):
            patch = Ellipse(
                (z, 0.0),
                width,
                2.0 * float(half),
                facecolor=to_rgba(_neutral(ax, _LENS_FACE_LEVEL), _LENS_FACE_ALPHA),
                edgecolor=_neutral(ax, _LENS_EDGE_LEVEL),
                lw=lw,
                zorder=_LENS_Z,
                gid=f"{owner}/lens/{i}",
            )
            glyphs.append(ax.add_patch(patch))
        artists["ellipse"] = glyphs
    ax.autoscale_view()

    def update(rays=None, beams=None):
        if rays is not None:
            new = [tuple(r) for r in rays]
            if len(new) != len(lines):
                msg = f"ray_train: update got {len(new)} rays for {len(lines)} drawn"
                raise ValueError(msg)
            for line, (z, y) in zip(lines, traced_rays(new), strict=True):
                line.set_data(z, y)
        if beams is not None:
            new = [tuple(b) for b in beams]
            if len(new) != len(fills):
                msg = f"ray_train: update got {len(new)} beams for {len(fills)} drawn"
                raise ValueError(msg)
            for coll, polys in zip(fills, traced_beams(new)[0], strict=True):
                coll.set_verts(polys)

    return ep.PlotResult(ax=ax, artists=artists, update=update)


def plot_mirror(
    mirror,
    used,
    parent=None,
    *,
    ax=None,
    color=None,
    parent_color=None,
    linewidth=None,
    zorder=_MIRROR_Z,
    gid="mirror",
):
    """Draw a curved mirror: the used piece solid, its parent surface dashed.

    An off-axis mirror is a piece cut from a larger parent surface (an
    off-axis parabola from a parabola centered on its axis). The piece is
    drawn thick in the hardware neutral; the parent, when given, thin,
    dashed and fainter beneath it, so the reader sees which surface the
    piece belongs to and where its axis and focus lie.

    Args:
        mirror: A ``Mirror`` (``raytrace.parabola``, ``raytrace.hyperbola``).
        used: ``(lo, hi)`` of the mirror's curve coordinate (``y``, or
            ``z`` for a parabola with its axis along ``y``) for the piece.
        parent: ``(lo, hi)`` of the same coordinate for the parent surface,
            or None to leave it out.
        ax: Axes to draw into. None creates a new figure.
        color: Color of the piece. None uses a neutral 70% of the way from
            the axes background to the text color.
        parent_color: Color of the parent. None uses the 45% neutral.
        linewidth: Width of the piece in points. None uses 3 times
            ``rcParams["lines.linewidth"]``; the parent is 0.8 of the base
            width.
        zorder: z-order of the piece; the parent is drawn one below.
        gid: Prefix of the gids, ``"<gid>/used"`` and ``"<gid>/parent"``.

    Returns:
        A ``PlotResult``. ``artists["lines"]`` is ``[used]`` or ``[used,
        parent]``.
    """
    ep = _require.eyepiece()
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    base = float(matplotlib.rcParams["lines.linewidth"])
    lines = []
    uc = mirror.curve(*used)
    (piece,) = ax.plot(
        uc[:, 0],
        uc[:, 1],
        color=_neutral(ax, _MIRROR_LEVEL) if color is None else color,
        lw=_MIRROR_LW * base if linewidth is None else linewidth,
        zorder=zorder,
        solid_capstyle="butt",
        gid=f"{gid}/used",
    )
    lines.append(piece)
    if parent is not None:
        pc = mirror.curve(*parent)
        (whole,) = ax.plot(
            pc[:, 0],
            pc[:, 1],
            color=_neutral(ax, _PARENT_LEVEL) if parent_color is None else parent_color,
            lw=_PARENT_LW * base,
            ls=_PARENT_DASH,
            zorder=zorder - 1,
            gid=f"{gid}/parent",
        )
        lines.append(whole)
    return ep.PlotResult(ax=ax, artists={"lines": lines})


def _hit_range(polylines, index, axis, pad):
    values = [float(p[index, axis]) for p in polylines]
    return min(values) - pad, max(values) + pad


def mirror_train(
    mirrors,
    rays,
    *,
    end=None,
    beams=(),
    used=None,
    parents=None,
    ax=None,
    colors=None,
    linestyles=None,
    linewidths=None,
    zorder=None,
    beam_colors=None,
    beam_alpha=_BEAM_ALPHA,
    mirror_color=None,
    pad=_USED_PAD,
):
    """Draw a train of curved mirrors with exactly traced rays and beams.

    Each ray is traced by ``trace_mirrors`` off every mirror in order and
    drawn as its polyline. Each beam fills the light between two of the
    rays, one quadrilateral per segment (every ray meets the same mirrors,
    so the segments pair up). Each mirror is drawn by ``plot_mirror``: by
    default its used piece spans the rays' hits on it, padded by ``pad``,
    and its parent surface is drawn where ``parents`` gives a range.

    Example::

        first = parabola((4.0, 0.0), 1.0, piece=(1.0, 3.0))
        second = parabola((2.0, 0.0), 1.0, opens=+1.0, piece=(-3.0, -1.0))
        res = mirror_train(
            [first, second],
            [((0.0, 2.5), (1.0, 0.0)), ((0.0, 1.5), (1.0, 0.0))],
            end=("z", 5.6),
            beams=[(0, 1)],
            parents=[(-2.8, 2.8), (-2.8, 2.8)],
        )

    Args:
        mirrors: ``Mirror`` objects in the order the rays meet them.
        rays: ``(point, direction)`` of each ray's start.
        end: Where every ray stops after the last mirror, as for
            ``trace_mirrors``: ``("z", value)``, ``("y", value)``, a length,
            or None.
        beams: ``(i, j)`` index pairs of the rays whose light is filled.
        used: The used piece of each mirror, a ``(lo, hi)`` range of its
            curve coordinate, or None for the span of the rays' hits on it
            padded by ``pad``. One entry per mirror, or None for all.
        parents: The parent range of each mirror, or None to leave its
            parent out. One entry per mirror, or None for no parents.
        ax: Axes to draw into. None creates a new figure.
        colors: Ray color, one for all or one per ray. None uses
            ``hwostyle.roles.star``.
        linestyles: Ray line style, one for all or one per ray. A ray may
            take ``"none"`` to be traced (for a beam edge or a mirror's
            piece) but not drawn.
        linewidths: Ray width in points, one for all or one per ray.
        zorder: Ray z-order, one for all or one per ray (default 5).
        beam_colors: Beam color, one for all or one per beam. None uses
            ``hwostyle.roles.star``.
        beam_alpha: Opacity of the beam fill.
        mirror_color: Color of every used piece. None uses the 70% neutral.
        pad: Padding of an automatic used piece, in data units.

    Returns:
        A ``PlotResult``. ``artists["lines"]`` lists the rays in order (gid
        ``"mirror-train/ray/<i>"``), then each mirror's used piece and its
        parent when drawn (gids ``"mirror-train/mirror/<k>/used"`` and
        ``".../parent"``). ``artists["fill"]`` lists one ``PolyCollection``
        per beam (gid ``"mirror-train/beam/<i>"``).

    Raises:
        ValueError: A ray that misses a mirror; a ``used`` or ``parents``
            list of the wrong length; a per-ray or per-beam style of the
            wrong length.
    """
    ep = _require.eyepiece()
    owner = "mirror-train"
    mirrors = list(mirrors)
    rays = list(rays)
    beams = [tuple(b) for b in beams]
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    polylines = [trace_mirrors(p, d, mirrors, end=end)[0] for p, d in rays]
    star = hwostyle.roles.star
    styles = _styles(owner, len(rays), colors, linestyles, linewidths, zorder, star)
    fill_colors = _per_item(
        beam_colors, len(beams), star, "beam_colors", owner, is_color_like
    )
    used = _per_item(used, len(mirrors), None, "used", owner, lambda v: v is None)
    parents = _per_item(
        parents, len(mirrors), None, "parents", owner, lambda v: v is None
    )

    artists = {}
    fills = _draw_beams(
        ax,
        owner,
        [strip_polygons(polylines[i], polylines[j]) for i, j in beams],
        fill_colors,
        beam_alpha,
    )
    if fills:
        artists["fill"] = fills
    lines = _draw_rays(ax, owner, [(p[:, 0], p[:, 1]) for p in polylines], styles)
    for k, mirror in enumerate(mirrors):
        piece = used[k]
        if piece is None:
            piece = _hit_range(polylines, k + 1, mirror.piece_axis, pad)
        res = plot_mirror(
            mirror,
            piece,
            parents[k],
            ax=ax,
            color=mirror_color,
            gid=f"{owner}/mirror/{k}",
        )
        lines.extend(res.artists["lines"])
    artists["lines"] = lines
    ax.autoscale_view()
    return ep.PlotResult(ax=ax, artists=artists)

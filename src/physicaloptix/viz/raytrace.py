"""Two-dimensional ray tracing for side views of optical trains.

A side view is drawn in the plane ``(z, y)``: ``z`` runs along the bench,
the direction of travel, and ``y`` runs up. Two tracers fill it.

- The paraxial tracer follows a ray through thin lenses. A ray is a height
  and a slope ``dy/dz``; free space adds slope times distance to the
  height, and a thin lens of focal length ``f`` subtracts height over ``f``
  from the slope. Between lenses the ray is straight, so it is returned as
  a polyline whose vertices are its ends and the lenses it crosses.
- The exact tracer follows a ray off curved mirrors. A ray is a point and a
  direction; it meets a mirror where the mirror's implicit equation
  ``G(z, y) = 0`` holds (found by a scan along the ray and a root find),
  and it leaves along the mirror image of its direction in the surface
  normal. The mirrors provided are pieces of conics: a parabola sends rays
  parallel to its axis to its focus, and a hyperbola sends rays aimed at
  one of its foci to the other.

These are drawing tools, not differentiable models: they use NumPy and
SciPy only, so they import without the viz extra. The views that draw
their output are in ``physicaloptix.viz.rays``.
"""

import math
from itertools import pairwise

import numpy as np
from scipy.optimize import brentq

# The ray scan starts this far along the ray, so a ray leaving a mirror's
# surface does not meet that surface again at its own start.
_T_MIN = 1e-6
# Central-difference step of the surface normal.
_NORMAL_STEP = 1e-6
# Absolute tolerance of the root find along the ray.
_XTOL = 1e-12


def trace_paraxial(height, slope, lenses, z_start, z_end, *, z_ref=0.0):
    """A paraxial ray through thin lenses, as a polyline.

    The ray passes the reference plane ``z_ref`` at ``height`` with
    ``slope`` (``dy/dz``). Free space adds slope times distance to the
    height; a thin lens of focal length ``f`` at ``z`` subtracts
    ``height / f`` from the slope. A lens at or before ``z_start``, or at or
    after ``z_end``, is not crossed.

    Example::

        lenses = [(1.5, 1.5), (4.5, 1.5)]          # a 4f relay
        z, y = trace_paraxial(0.5, 0.0, lenses, -1.0, 6.0)

    Args:
        height: Height at ``z_ref``.
        slope: Slope before the first lens crossed.
        lenses: ``(z, f)`` pairs, in any order (they are crossed in
            increasing ``z``).
        z_start: Where the polyline starts.
        z_end: Where it ends, after ``z_start``.
        z_ref: The plane where ``height`` is given, before the first lens
            crossed (the ray is straight back to ``z_start`` from it).

    Returns:
        ``(z, y)``: arrays of the polyline's vertices, the start, every lens
        crossed and the end.

    Raises:
        ValueError: ``z_end`` not after ``z_start``.
    """
    if not z_end > z_start:
        msg = f"trace_paraxial: z_end ({z_end}) must be after z_start ({z_start})"
        raise ValueError(msg)
    z, y, u = z_start, height + slope * (z_start - z_ref), slope
    zs, ys = [z], [y]
    for zl, f in sorted(lenses, key=lambda lens: lens[0]):
        if zl <= z_start:
            continue
        if zl >= z_end:
            break
        y += u * (zl - z)
        z = zl
        zs.append(z)
        ys.append(y)
        u -= y / f
    y += u * (z_end - z)
    zs.append(z_end)
    ys.append(y)
    return np.array(zs), np.array(ys)


def beam_polygons(top, bottom, breaks=None):
    """Quadrilaterals of a beam between two paraxial edge rays.

    Between consecutive break points both rays are straight, so each segment
    of the beam is a quadrilateral. Where the rays cross (a focus) it is a
    bow tie, which a polygon fill draws as two triangles meeting at the
    crossing, so the beam pinches to a point there.

    Args:
        top: A traced ray ``(z, y)``, as ``trace_paraxial`` returns.
        bottom: The other edge ray.
        breaks: Increasing ``z`` values that include every vertex of both
            rays between the ends. None uses the union of the two rays'
            vertices over the range they share.

    Returns:
        A list of ``(4, 2)`` vertex arrays ``(z, y)``, one per segment.
    """
    if breaks is None:
        lo = max(top[0][0], bottom[0][0])
        hi = min(top[0][-1], bottom[0][-1])
        both = np.concatenate([top[0], bottom[0]])
        breaks = np.unique(both[(both >= lo) & (both <= hi)])
    polys = []
    for za, zb in pairwise(breaks):
        polys.append(
            np.array(
                [
                    (za, float(np.interp(za, top[0], top[1]))),
                    (zb, float(np.interp(zb, top[0], top[1]))),
                    (zb, float(np.interp(zb, bottom[0], bottom[1]))),
                    (za, float(np.interp(za, bottom[0], bottom[1]))),
                ]
            )
        )
    return polys


def strip_polygons(ray_a, ray_b):
    """Quadrilaterals of a beam between two polylines with matching vertices.

    Segment ``i`` of the beam joins segment ``i`` of each ray, so the two
    rays must have met the same surfaces in the same order, as rays traced
    through one train by ``trace_mirrors`` do.

    Args:
        ray_a: ``(n, 2)`` polyline vertices ``(z, y)``.
        ray_b: ``(n, 2)`` polyline vertices of the other edge ray.

    Returns:
        A list of ``(4, 2)`` vertex arrays, one per segment.

    Raises:
        ValueError: Polylines with different numbers of vertices.
    """
    a = np.asarray(ray_a, dtype=float)
    b = np.asarray(ray_b, dtype=float)
    if a.shape != b.shape:
        msg = (
            f"strip_polygons: the rays have {len(a)} and {len(b)} vertices; "
            "a beam joins rays that met the same surfaces"
        )
        raise ValueError(msg)
    return [
        np.array([a0, a1, b1, b0])
        for (a0, a1), (b0, b1) in zip(pairwise(a), pairwise(b), strict=True)
    ]


class Mirror:
    """A curved mirror in the side-view plane: an implicit curve and its piece.

    The mirror is the set ``G(z, y) = 0``. A ray meets only the piece of it
    whose coordinate ``piece_axis`` (0 for ``z``, 1 for ``y``) lies in
    ``piece``; the rest of the curve is the parent surface, which can be
    drawn but reflects nothing. ``curve(lo, hi)`` gives points along the
    curve over a range of the same coordinate, for drawing.

    The ``parabola`` and ``hyperbola`` constructors build the conic mirrors;
    any other surface can be built directly from its ``G`` and ``curve``.

    Attributes:
        g: ``G(z, y)``, zero on the mirror; it must accept NumPy arrays.
        curve: ``curve(lo, hi, n=120)``, an ``(n, 2)`` array of points
            ``(z, y)`` over ``[lo, hi]`` of the coordinate ``piece_axis``.
        piece: ``(lo, hi)`` of the coordinate ``piece_axis`` that the mirror
            occupies, or None for the whole curve.
        piece_axis: 0 when the curve is parametrized by ``z``, 1 by ``y``.
        focus: The focus ``(z, y)`` of a parabola, else None.
    """

    def __init__(self, g, curve, *, piece=None, piece_axis=1, focus=None):
        """Store the surface; see the class docstring for the fields."""
        if piece_axis not in (0, 1):
            msg = f"Mirror: piece_axis must be 0 (z) or 1 (y), not {piece_axis!r}"
            raise ValueError(msg)
        self.g = g
        self.curve = curve
        self.piece = None if piece is None else (float(piece[0]), float(piece[1]))
        self.piece_axis = piece_axis
        self.focus = focus

    def contains(self, point):
        """Whether ``point`` lies on the mirror's piece (by its coordinate)."""
        if self.piece is None:
            return True
        lo, hi = self.piece
        return lo <= point[self.piece_axis] <= hi

    def normal(self, point):
        """The unit normal of the surface at ``point``, by central differences."""
        z, y = point
        h = _NORMAL_STEP
        gz = (self.g(z + h, y) - self.g(z - h, y)) / (2 * h)
        gy = (self.g(z, y + h) - self.g(z, y - h)) / (2 * h)
        n = np.array([gz, gy], dtype=float)
        return n / np.linalg.norm(n)


def parabola(vertex, focal, *, axis="z", opens=-1.0, piece=None):
    """A parabolic mirror with its axis along ``z`` or along ``y``.

    With ``axis="z"`` the surface is ``z = z_v + opens (y - y_v)^2 / (4 f)``:
    ``opens = -1`` makes its concave side face ``-z`` (focus at ``z_v - f``),
    ``+1`` makes it face ``+z``. With ``axis="y"`` the roles of ``z`` and
    ``y`` swap: ``y = y_v + opens (z - z_v)^2 / (4 f)``. A ray parallel to
    the axis on the concave side reflects through the focus, so an off-axis
    piece (``piece`` away from the vertex) turns a collimated beam to a
    focus beside it.

    Example::

        # A collimated beam along +z at height 2 turns 90 degrees down to
        # the focus (1, 0) of a parabola facing -z with its vertex at z = 2.
        oap = parabola((2.0, 0.0), 1.0, piece=(1.0, 3.0))

    Args:
        vertex: The vertex ``(z_v, y_v)``.
        focal: Focal length (vertex to focus), positive.
        axis: ``"z"`` (default) or ``"y"``, the direction of the axis.
        opens: ``-1`` (default) or ``+1``, the side the concave face looks
            toward along the axis.
        piece: ``(lo, hi)`` of the coordinate across the axis (``y`` for
            ``axis="z"``, ``z`` for ``axis="y"``) that the mirror occupies,
            or None for the whole parabola.

    Returns:
        A ``Mirror`` whose ``focus`` is set.

    Raises:
        ValueError: An unknown ``axis``, ``opens`` not +-1, or a focal length
            that is not positive.
    """
    if axis not in ("z", "y"):
        msg = f"parabola: axis must be 'z' or 'y', not {axis!r}"
        raise ValueError(msg)
    if opens not in (-1, 1):
        msg = f"parabola: opens must be -1 or +1, not {opens!r}"
        raise ValueError(msg)
    if not focal > 0:
        msg = f"parabola: focal length must be positive, not {focal!r}"
        raise ValueError(msg)
    zv, yv = (float(v) for v in vertex)
    if axis == "z":

        def z_of(y):
            return zv + opens * (np.asarray(y) - yv) ** 2 / (4.0 * focal)

        def curve(lo, hi, n=120):
            y = np.linspace(lo, hi, n)
            return np.column_stack([z_of(y), y])

        return Mirror(
            lambda z, y: z - z_of(y),
            curve,
            piece=piece,
            piece_axis=1,
            focus=(zv + opens * focal, yv),
        )

    def y_of(z):
        return yv + opens * (np.asarray(z) - zv) ** 2 / (4.0 * focal)

    def curve_y(lo, hi, n=120):
        z = np.linspace(lo, hi, n)
        return np.column_stack([z, y_of(z)])

    return Mirror(
        lambda z, y: y - y_of(z),
        curve_y,
        piece=piece,
        piece_axis=0,
        focus=(zv, yv + opens * focal),
    )


def hyperbola(focus_a, focus_b, vertex_z, *, axis_y=0.0, piece=None):
    """One branch of a hyperbolic mirror whose foci lie on a line along ``z``.

    The foci are ``(focus_a, axis_y)`` and ``(focus_b, axis_y)``; the branch
    is the one through ``(vertex_z, axis_y)``, which must lie between the
    foci. On it the distance to ``focus_b`` exceeds the distance to
    ``focus_a`` by ``2a`` (``a`` the vertex's distance from the center), so a
    ray aimed at ``focus_a`` reflects from the branch toward ``focus_b``:
    the convex secondary of a Cassegrain telescope, with ``focus_a`` the
    primary's focus.

    Args:
        focus_a: ``z`` of the focus the branch curves around.
        focus_b: ``z`` of the other focus.
        vertex_z: ``z`` of the branch's vertex, between the foci and nearer
            ``focus_a``.
        axis_y: Height of the line through the foci.
        piece: ``(lo, hi)`` of ``y`` that the mirror occupies, or None.

    Returns:
        A ``Mirror``.

    Raises:
        ValueError: A vertex not strictly between the foci and nearer
            ``focus_a``.
    """
    za, zb = float(focus_a), float(focus_b)
    center = 0.5 * (za + zb)
    a = abs(vertex_z - center)
    c = 0.5 * abs(zb - za)
    lo, hi = sorted((za, zb))
    if not (lo < vertex_z < hi) or abs(vertex_z - za) >= abs(vertex_z - zb):
        msg = (
            f"hyperbola: vertex_z ({vertex_z}) must lie between the foci "
            f"({za}, {zb}) and nearer focus_a"
        )
        raise ValueError(msg)
    b = math.sqrt(c * c - a * a)
    side = math.copysign(1.0, za - zb)
    ya = float(axis_y)

    def g(z, y):
        return np.hypot(z - zb, y - ya) - np.hypot(z - za, y - ya) - 2.0 * a

    def curve(lo, hi, n=120):
        y = np.linspace(lo, hi, n)
        return np.column_stack(
            [center + side * a * np.sqrt(1.0 + ((y - ya) / b) ** 2), y]
        )

    return Mirror(g, curve, piece=piece, piece_axis=1)


def _hit(point, direction, mirror, t_max, steps):
    """Distance along a unit direction to the first crossing of the piece, or None."""
    p = np.asarray(point, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    ts = np.linspace(_T_MIN, t_max, steps)
    vals = np.asarray(mirror.g(p[0] + ts * d[0], p[1] + ts * d[1]), dtype=float)
    sign = np.sign(vals)
    for i in np.flatnonzero(sign[:-1] * sign[1:] < 0):
        t = brentq(
            lambda t: float(mirror.g(*(p + t * d))), ts[i], ts[i + 1], xtol=_XTOL
        )
        if mirror.contains(p + t * d):
            return t
    return None


def _reflect(direction, normal):
    """The mirror image of a direction in a surface of unit normal ``normal``."""
    d = np.asarray(direction, dtype=float)
    return d - 2.0 * np.dot(d, normal) * normal


def trace_mirrors(point, direction, mirrors, *, end=None, t_max=40.0, steps=4000):
    """A ray off a sequence of mirrors, exactly, as a polyline.

    The ray meets each mirror in turn where the mirror's ``G`` changes sign
    along it (a scan of ``steps`` points out to ``t_max``, refined by a
    root find) on the mirror's piece, and leaves along its direction
    reflected in the surface normal there.

    Example::

        oap = parabola((2.0, 0.0), 1.0, piece=(1.0, 3.0))
        pts, d = trace_mirrors((0.0, 2.0), (1.0, 0.0), [oap], end=("y", -1.0))
        # pts[-2] is the hit on the mirror; the ray then passes the focus.

    Args:
        point: Start point ``(z, y)``.
        direction: Start direction ``(dz, dy)``, any length.
        mirrors: ``Mirror`` objects in the order the ray meets them.
        end: Where the polyline stops after the last mirror: ``("z", value)``
            or ``("y", value)`` for a plane, a number for a length along the
            ray, or None for no last segment.
        t_max: Farthest distance searched along the ray for each mirror.
        steps: Number of scan points along ``t_max``; a mirror crossed twice
            within one step can be missed.

    Returns:
        ``(points, direction)``: the vertices as an ``(n, 2)`` array (the
        start, each hit and the end) and the final unit direction.

    Raises:
        ValueError: A ray that misses a mirror's piece, or an ``end`` plane
            the final ray runs parallel to.
    """
    p = np.asarray(point, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    pts = [p]
    for i, mirror in enumerate(mirrors):
        t = _hit(p, d, mirror, t_max, steps)
        if t is None:
            msg = f"trace_mirrors: the ray misses mirror {i}"
            raise ValueError(msg)
        p = p + t * d
        d = _reflect(d, mirror.normal(p))
        pts.append(p)
    if end is not None:
        if isinstance(end, tuple):
            index = 0 if end[0] == "z" else 1
            if d[index] == 0:
                msg = f"trace_mirrors: the final ray never reaches {end!r}"
                raise ValueError(msg)
            t = (end[1] - p[index]) / d[index]
        else:
            t = float(end)
        pts.append(p + t * d)
    return np.array(pts), d


def ray_crossing(a, b):
    """Where two straight rays cross, each given as ``(point, direction)``.

    Solves ``p_a + s d_a = p_b + t d_b``. With the last segments of two
    traced chief rays this locates an image of the pupil; with two rays
    from one object point, the image of that point.

    Returns:
        The crossing ``(z, y)`` as an array.

    Raises:
        numpy.linalg.LinAlgError: Parallel rays.
    """
    pa, da = (np.asarray(v, dtype=float) for v in a)
    pb, db = (np.asarray(v, dtype=float) for v in b)
    m = np.column_stack([da, -db])
    s, _ = np.linalg.solve(m, pb - pa)
    return pa + s * da

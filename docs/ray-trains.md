---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Ray trains

A side view of an optical train shows where its beams go: which planes hold
an image of the pupil, which hold a focus, and how wide the beam is at each
surface. `physicaloptix.viz` draws two kinds, both from traced rays rather
than from envelopes, so every beam is where the geometry puts it and pinches
to a point at a focus.

- `ray_train` draws thin lenses with paraxial rays. A ray is a height and a
  slope; free space adds slope times distance to the height, and a thin lens
  of focal length $f$ subtracts height over $f$ from the slope.
- `mirror_train` draws curved mirrors with rays traced exactly in two
  dimensions. A ray meets a mirror where the mirror's implicit equation
  $G(z, y) = 0$ holds and leaves along the mirror image of its direction in
  the surface normal. Each mirror is drawn as the piece in use over its dashed
  parent surface.

The tracers themselves, `physicaloptix.viz.raytrace`, need only NumPy and
SciPy. Lengths below are in units of the beam diameter $D$, and the drawn
angles are exaggerated, as in any schematic side view.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

from physicaloptix.viz import mirror_train, ray_train
from physicaloptix.viz.raytrace import hyperbola, parabola

hwostyle.use("dark")
# Docs-build only, to keep the page images small.
plt.rcParams["savefig.dpi"] = 120
```

## Pupils and foci in a relay chain

A 4f relay puts a lens of focal length $f$ halfway between two planes $2f$
apart, so every pupil plane is imaged onto the next pupil plane and every
focus onto the next focus. The two kinds of plane are told apart by two rays.
The marginal rays of an on-axis star, the edges of its collimated beam, meet
at every focus. The chief ray of an off-axis source, through the center of
the entrance pupil, crosses the axis at every image of the pupil. Below, the
star's beam is filled between its edge rays and the planet's chief ray is
dashed; the planes sit at $z = 0, 3, 6, 9$.

```{code-cell} python
f = 1.5
planes = [0.0, 3.0, 6.0, 9.0]
lenses = [(0.5 * (a + b), f) for a, b in zip(planes[:-1], planes[1:])]
star, planet = hwostyle.roles.star, hwostyle.roles.planet

fig, ax = plt.subplots(figsize=(7.0, 2.6), layout="constrained")
relay = ray_train(
    lenses,
    [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.15)],
    z_start=-1.25,
    z_end=9.0,
    beams=[(0.5, -0.5, 0.0)],
    colors=[star, star, planet],
    linestyles=["-", "-", (0, (4, 2))],
    ax=ax,
)
for z, kind in zip(planes, ["pupil", "focus", "pupil", "focus"]):
    ax.axvline(z, color="0.5", lw=0.6, ls=":")
    ax.text(z, 0.68, kind, ha="center", va="bottom", fontsize=8)
ax.set(xlabel="$z$ ($D$)", ylabel="$y$ ($D$)", ylim=(-0.8, 0.85))
```

The chief ray is at height $f\theta$ at the first focus, the image of the
planet, and on the axis at the pupil image, where the planet's light fills the
same circle as the star's. The relay also turns everything over: the top of
the beam at the entrance pupil is at the bottom of the pupil image.

`update` retraces new rays through the same lenses with the same artists, so
a sweep of the source angle is an animation of one figure:

```{code-cell} python
relay.update(rays=[(0.5, 0.0), (-0.5, 0.0), (0.0, 0.08)])
fig
```

## Two off-axis parabolas, folded

A parabola sends rays parallel to its axis through its focus. A piece cut
from it away from the axis, an off-axis parabola, does the same while turning
the beam aside, so the focus sits beside the beam rather than in it. Two such
pieces sharing a focus relay a collimated beam with nothing in its way. The
dashed curves are the parent parabolas; only the solid pieces are made.

```{code-cell} python
first = parabola((4.0, 0.0), 1.0, piece=(1.0, 3.0))
second = parabola((2.0, 0.0), 1.0, opens=+1.0, piece=(-3.0, -1.0))
starts = [((-0.4, h), (1.0, 0.0)) for h in (2.5, 2.0, 1.5)]

fig, ax = plt.subplots(figsize=(4.0, 3.6), layout="constrained")
mirror_train(
    [first, second],
    starts,
    end=("z", 5.6),
    beams=[(0, 2)],
    parents=[(-2.8, 2.8), (-2.8, 2.8)],
    linestyles=["-", "none", "-"],
    ax=ax,
)
ax.plot(*first.focus, "o", color=star, mec="white", mew=0.6)
ax.set(xlim=(-0.5, 5.7), ylim=(-3.1, 3.1), aspect="equal")
ax.axis("off")
```

The middle ray is traced but not drawn (line style `"none"`), so it still
counts toward the span of each used piece. A ray entering at height $h$
leaves at $-h$: the two mirrors are mirror images of each other in the plane
of their shared focus.

## An off-axis telescope

A Cassegrain telescope folds its focus behind the primary with a convex
hyperbolic secondary: light converging on the primary's focus meets the
secondary first, and a hyperbola sends rays aimed at one of its foci to the
other. Built from off-axis pieces of both, nothing blocks the beam. A third
mirror, a parabola with its axis along $y$ and its focus at the telescope's
focus, makes the beam collimated again.

```{code-cell} python
primary = parabola((10.0, 0.0), 4.0, piece=(1.6, 4.0))
secondary = hyperbola(primary.focus[0], 12.0, 7.5, piece=(0.0, 1.72))
collimator = parabola((12.0, 1.0), 1.0, axis="y", piece=(13.2, 14.8))
heights = (3.8, 2.8, 1.8)
rays = [((-0.6, h), (1.0, 0.0)) for h in heights]

fig, ax = plt.subplots(figsize=(7.0, 3.8), layout="constrained")
mirror_train(
    [primary, secondary, collimator],
    rays,
    end=("y", -2.6),
    beams=[(0, 2)],
    parents=[(-3.9, 3.9), (-1.8, 1.8), (12.8, 15.0)],
    linestyles=["-", (0, (1, 1.5)), "-"],
    ax=ax,
)
ax.plot(12.0, 0.0, "o", color=star, mec="white", mew=0.6)
ax.set(xlim=(-0.7, 16.2), ylim=(-3.0, 4.6), aspect="equal")
ax.axis("off")
```

Each mirror's used piece spans the hits of the drawn rays, padded slightly;
pass `used=` to set the pieces instead.

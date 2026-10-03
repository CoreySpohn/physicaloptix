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

# Near field and far field

Light through an opening makes a sharp shadow just behind it and a spread
pattern far away, and the Fresnel number
$N_F = a^2 / (\lambda z)$ of an opening of half-width $a$ seen from distance
$z$ says which. Three views in `physicaloptix.viz` draw the change, each in
units of the wavelength:

- `plot_near_to_far` draws a slit's brightness over the Fresnel number and
  position (or angle), from a map that `prepare_near_to_far` computes by the
  exact angular spectrum;
- `plot_fresnel_zones` draws an opening face-on, colored by the phase of the
  path from each of its points to a point behind it, so the Fresnel zones
  show as rings;
- `plot_cornu` draws the Cornu spiral, whose windows give the brightness
  behind a slit or an edge.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import fresnel

from physicaloptix.viz import (
    plot_cornu,
    plot_fresnel_zones,
    plot_near_to_far,
    prepare_near_to_far,
)

hwostyle.use("dark")
# Docs-build only, to keep the page images small.
plt.rcParams["savefig.dpi"] = 120
```

## A slit from near to far

`prepare_near_to_far` propagates a unit plane wave through a slit of width
$b$ to the distance of each Fresnel number $N_F = (b/2)^2/(\lambda z)$ with
the transfer function $\exp\!\left(i 2\pi z\sqrt{1/\lambda^2 - f^2}\right)$,
on a window wide enough to hold the far-field spread. Because $N_F$ falls
with distance, a decreasing list of Fresnel numbers draws the slit at the
left and the far field at the right; the dotted lines are the slit's edges.

```{code-cell} python
nf = np.geomspace(26.0, 0.08, 161)
near_far = prepare_near_to_far(nf, width=40.0, pitch=0.2, pad=120)
theta = np.linspace(-8.0, 8.0, 241)
in_angle = prepare_near_to_far(
    nf, width=40.0, pitch=0.2, pad=120, angles=theta, normalize="row"
)

fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
res = plot_near_to_far(*near_far, ax=left, cbar_label="brightness / incident")
plot_near_to_far(
    *in_angle, ax=right, vlim=1.0, cbar_label="brightness / brightest in row"
)
```

On the left, the near field keeps the slit's shadow with ripples at its
edges, the ripples merge near $N_F = 1$, and well below it the pattern only
widens. On the right the same propagation is read against angle,
$\sin\theta = X/z$ in units of $\lambda/b$, each distance scaled to its
brightest point: the far-field pattern, fixed in angle, is the set of
horizontal bands at the right, and the near field's fixed width closes in.

The axes and the map carry gids (`"near-to-far"` and `"near-to-far/image"`
by default), so a figure that carries the map elsewhere can find it again:

```{code-cell} python
res.ax.get_gid(), res.artists["image"].get_gid()
```

## Fresnel zones

Seen from a point at distance $z$ on the axis behind a round opening of
radius $a$, the path from a point at radius $\rho$ is
$R = \sqrt{z^2 + \rho^2}$. Coloring the opening by the phase of
$R - z$ turns each ring across which the path grows by half a wavelength, a
Fresnel zone, into half a turn of the phase colormap. The zones have equal
areas, $\pi\lambda z$, so they crowd toward the rim, and the opening holds
about $N_F$ of them. The thin circles are the zone boundaries.

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), layout="constrained")
for ax, value in zip(axes, (1.0, 2.0, 6.0)):
    plot_fresnel_zones(value, ax=ax)
    ax.set_title(f"$N_F$ = {value:g}", fontsize=9)
```

Each pair of neighboring zones sends light to the axis with opposite signs,
so an even number of zones nearly cancels on the axis and an odd number
leaves the field of one. `update(nf)` redraws the phase and moves the
boundary circles; pass `max_rings` when an animation adds zones.

## The Cornu spiral

The running sum of the arrows from the strips of an opening, in units of
$\sqrt{\lambda z / 2}$ across it, traces the Cornu spiral
$C(v) + i\,S(v)$, the Fresnel integrals of $\pi v^2 / 2$, which winds into
its eyes at $\pm(1 + i)/2$. An opening from $v_0$ to $v_1$ picks that window
of the spiral; the chord across it is the sum of the window's arrows, and the
brightness behind the opening, relative to the incident wave, is half the
squared chord. The whole spiral is the unobstructed wave, chord $1 + i$ and
brightness 1.

A slit of Fresnel number $N_F$ seen from position $X$ (in units of its
width) is the window $\big((-\tfrac12 - X)\sqrt{8N_F},
(\tfrac12 - X)\sqrt{8N_F}\big)$; a straight edge seen from its shadow line is
the window from 0 to infinity, a quarter of the incident brightness.

```{code-cell} python
nf_slit = 3.0
scale = np.sqrt(8.0 * nf_slit)
windows = {
    "slit, on axis": (-0.5 * scale, 0.5 * scale),
    "slit, from its edge": (-scale, 0.0),
    "edge, from its shadow line": (0.0, np.inf),
}


def spiral(v):
    s, c = fresnel(v)
    return c + 1j * s


fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), layout="constrained")
for ax, (name, (v0, v1)) in zip(axes, windows.items()):
    plot_cornu((v0, v1), ax=ax)
    brightness = abs(spiral(v1) - spiral(v0)) ** 2 / 2.0
    ax.set_title(f"{name}: {brightness:.2f}", fontsize=9)
```

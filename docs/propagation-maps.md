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

# Propagation maps

A propagation map shows a field along its direction of travel: the
transverse cut of the field at each distance $z$, stacked into one picture
with $z$ running to the right, as on an optical bench, and the transverse
coordinate $x$ running up. `physicaloptix.viz.plot_propagation` draws one
from an array indexed `[z, x]`, which is what a loop that propagates a 1D
field to each distance in turn produces.

Each cell is drawn at its true edges, computed from the pixel-center
coordinates, so a logarithmic or otherwise nonuniform $z$ needs no
resampling, and nothing is interpolated. Bench elements are drawn on top in
data coordinates: an opaque screen with openings, a phase screen drawn as its
phase, and a dashed marked plane.

The three maps below are computed on the page with NumPy. Each propagates a
1D scalar field by the angular spectrum: the field is Fourier transformed,
multiplied by the exact transfer function
$\exp\!\left(i 2\pi z \sqrt{1/\lambda^2 - f^2}\right)$, and transformed
back. Lengths are in wavelengths ($\lambda = 1$).

```{code-cell} python
import eyepiece as ep
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

from physicaloptix.viz import plot_propagation

hwostyle.use("dark")
# Docs-build only, to keep the page images small.
plt.rcParams["savefig.dpi"] = 120


def angular_spectrum(field, dx, z, pad=1):
    """Propagate a 1D field by z (wavelengths) on a window pad times wider."""
    n = field.size
    total = pad * n
    start = (total - n) // 2
    padded = np.zeros(total, dtype=complex)
    padded[start : start + n] = field
    f = np.fft.fftfreq(total, d=dx)
    kz = np.sqrt((1.0 - f**2).astype(complex))  # decays past f = 1 / lambda
    out = np.fft.ifft(np.fft.fft(padded) * np.exp(2j * np.pi * z * kz))
    return out[start : start + n]
```

## A slit from near to far

A plane wave through a slit of width $b$ casts a sharp shadow just behind it
and spreads into the far-field $\mathrm{sinc}^2$ pattern far away. The
distance that separates the two regimes is set by the Fresnel number
$N_F = (b/2)^2 / (\lambda z)$, so the natural axis is $N_F$ on a log scale.
The rows below are taken at Fresnel numbers from 20 down to 0.1. Because
$N_F$ falls with distance, the decreasing `z` passed in draws with its axis
inverted: the slit is off to the left and the far field is on the right.

```{code-cell} python
b, dx = 16.0, 0.125
n_window = 4096  # the window holds the far-field spread at N_F = 0.1
x_full = (np.arange(n_window) - n_window / 2 + 0.5) * dx
slit = (np.abs(x_full) <= b / 2).astype(complex)
keep = np.abs(x_full) <= 3.0 * b

nf = np.geomspace(20.0, 0.1, 64)
rows = [angular_spectrum(slit, dx, (b / 2) ** 2 / value)[keep] for value in nf]
slit_map = np.abs(np.array(rows)) ** 2  # indexed [z, x]

fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
res = plot_propagation(
    slit_map,
    x_full[keep] / b,
    nf,
    ax=ax,
    kind="intensity",
    z_scale="log",
    z_label=r"Fresnel number $N_F = (b/2)^2/(\lambda z)$, farther to the right",
    x_label="$x / b$",
    elements=[("plane", 1.0, "$N_F = 1$")],
    cbar_label="intensity / incident",
)
```

The near field keeps the slit's edges, with Fresnel ripples inside the lit
region. Around $N_F = 1$ the ripples merge, and well past it the pattern
only widens: the far field is the same shape at every distance, scaled.

## A lens focusing a plane wave

A lens is a phase screen: it delays the field at each height by the optical
path that makes every point of the lens one path length from the focus. The
`"phase"` element draws it that way, a thin strip colored by the wrapped
phase, beside an opaque screen that limits the beam to the lens width, and a
`"plane"` element marks the focal plane. The map is the instantaneous field
$\mathrm{Re}\,E$, whose colormap is centered on the background: zero is the
background color, positive values run to the starlight color, and negative
values run to a muted complement of it at the same lightness, so the unlit
regions read as dark and the two signs read as equally strong.

```{code-cell} python
width, focal = 20.0, 40.0
pitch = 0.1
x = (np.arange(300) - 149.5) * pitch
z = np.arange(-8.0, 62.0, pitch) + pitch / 2


def lens_phase(u):
    return -2.0 * np.pi * (np.sqrt(u**2 + focal**2) - focal)


behind = np.where(np.abs(x) <= width / 2, np.exp(1j * lens_phase(x)), 0.0)
lens_map = np.array(
    [
        np.full(x.size, np.exp(2j * np.pi * zj)) if zj < 0 else
        angular_spectrum(behind, pitch, zj, pad=8)
        for zj in z
    ]
)

fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
lens = plot_propagation(
    lens_map,
    x,
    z,
    ax=ax,
    kind="real",
    vlim=2.0,
    elements=[
        ("screen", 0.0, [(-width / 2, width / 2)]),
        ("phase", 0.0, -width / 2, width / 2, lens_phase),
        ("plane", focal, "focal plane"),
    ],
    cbar_label="field / incident",
)
ax.set_aspect("equal")
```

The map's `update` redraws it from a new field on the same grid, keeping the
color range of the first frame, so an animation over one period does not
rescale. Advancing the time by $t$ periods multiplies the field by
$e^{-i 2\pi t}$:

```{code-cell} python
frames = np.arange(8) / 8.0


def draw(_fig, t):
    lens.update(lens_map * np.exp(-2j * np.pi * t))


anim = ep.animate(fig, draw, frames, fps=8)
HTML(anim.jshtml(dpi=72))
```

## A Talbot carpet

A periodic field images itself as it propagates. For a grating of period
$p$, the paraxial Talbot length is $z_T = 2p^2/\lambda$: the field repeats at
$z_T$, repeats shifted by half a period at $z_T/2$, and at $z_T/4$ a pure
phase ripple has turned into the strongest intensity ripple. The field below
starts as a plane wave that meets the phase ripple
$\exp\left(i\phi_0 \cos(2\pi x/p)\right)$ with $\phi_0 = 0.5$ rad at
$z = 0$, two periods across a periodic window, so no padding is needed. The
map is far coarser in $z$ than a wavelength, so the field is drawn with its
carrier $e^{i 2\pi z}$ removed: the slowly varying envelope, whose phase is
the ripple's. The intensity map is the carpet; the phase map below it shows
the phase ripple coming back at the half and full planes, where the
intensity is flat again.

```{code-cell} python
p, phi0 = 16.0, 0.5
z_talbot = 2.0 * p**2
pitch = p / 128
x = (np.arange(256) + 0.5) * pitch - p
z = np.linspace(-0.04, 1.05, 300) * z_talbot


def ripple(u):
    return phi0 * np.cos(2.0 * np.pi * u / p)


start = np.exp(1j * ripple(x))
carpet = np.array(
    [
        np.ones(x.size) if zj < 0 else
        angular_spectrum(start, pitch, zj) * np.exp(-2j * np.pi * zj)
        for zj in z
    ]
)
planes = [
    ("plane", z_talbot / 4, "$z_T/4$"),
    ("plane", z_talbot / 2, "$z_T/2$"),
    ("plane", z_talbot, "$z_T$"),
]

fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(7.0, 4.6), sharex=True, layout="constrained"
)
plot_propagation(
    carpet,
    x,
    z,
    ax=top,
    kind="intensity",
    z_label="",
    elements=[("phase", 0.0, -p, p, ripple), *planes],
    cbar_label="intensity",
)
plot_propagation(
    carpet,
    x,
    z,
    ax=bottom,
    kind="phase",
    elements=planes,
)
```

The exact transfer function differs slightly from the paraxial one for the
diffracted orders, so the revival at $z_T$ is close to, not exactly, the
starting field; the difference shrinks as $p/\lambda$ grows.

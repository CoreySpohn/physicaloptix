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

# Speckle boiling, prepared once

A boiling dark hole is shown in several forms: a paper still, a strip of
selected epochs, a movie, and a clip in a talk. Each form needs the same
science: the speckle field evaluated at every epoch, the deterministic floor
added back, the frames referenced to the telescope peak, one display scale for
the whole sequence, and the dark-zone trace. `physicaloptix.viz.prepare_speckles`
does that work once and returns an [eyepiece](https://eyepiece.readthedocs.io)
prepared `Sequence`. Every output below reads that one sequence, so none of
them evaluates the field again, and they cannot disagree about the data.

The preparation imports NumPy, hwoutils, and `eyepiece.prepared` only.
Drawing is left to eyepiece's renderers: `eyepiece.mpl` for figures and
movies, and `eyepiece.manim` for native Manim clips. The `viz` extra
(`pip install 'physicaloptix[viz]'`) installs eyepiece with its Matplotlib
renderer; the Manim renderer also needs `eyepiece[manim]`. The records, scales, and sample-and-hold time rules
are described in eyepiece's
[prepared views guide](https://eyepiece.readthedocs.io/en/latest/prepared-views.html).

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece.mpl as mpl
from eyepiece.prepared import find_element
from eyepiece.style import snapshot_profile
from physicaloptix.viz import boiling_strip, prepare_speckles

hwostyle.use("dark")
# Docs-build only, to keep the page images small. A figure script keeps the
# style's print resolution and omits this line.
plt.rcParams["savefig.dpi"] = 120

# One profile for every output on this page, taken once from the active
# style and sized from its rc settings.
PROFILE = snapshot_profile(
    text_size_pt=plt.rcParams["font.size"],
    stroke_width_pt=plt.rcParams["lines.linewidth"],
)
```

## A speckle field to prepare

`prepare_speckles` is typed on the speckle-field protocol, not on a concrete
class: anything with `realize(wavelength_nm=..., time_s=...)` returning a
`(y, x)` map and a `pixel_scale_lod` drives it. The synthetic field below
follows that protocol. A static pupil phase error behind an ideal
coronagraph sets the deterministic field `e_nom`, and twelve sine ripples
drift on top of it with periods between half an hour and two hours.

As the protocol requires, `realize` returns only the change in intensity over
the deterministic floor $|E_\mathrm{nom}|^2$, in flux-fraction units. The
field also carries `e_nom` and `normalization`, which is what lets the
preparation add the floor back. The field counts its own evaluations so the
page can show when the science runs.

```{code-cell} python
N_PIX, PUPIL_PIX = 64, 32
PIXSCALE_LOD = PUPIL_PIX / N_PIX  # lambda/D per focal-plane pixel
u = np.arange(N_PIX) - N_PIX / 2.0
uu, vv = np.meshgrid(u, u)
pupil = (np.hypot(uu, vv) < PUPIL_PIX / 2.0).astype(float)
cycles_u, cycles_v = uu / PUPIL_PIX, vv / PUPIL_PIX
# Half a pixel of tilt puts the optical axis between the four central
# pixels, where the prepared image extent places zero.
half_pixel = np.exp(-1j * np.pi * (uu + vv) / N_PIX)


def focal(pupil_field):
    """Far field of a pupil-plane field, scaled by the clear aperture's area."""
    shifted = np.fft.ifftshift(pupil_field * half_pixel)
    return np.fft.fftshift(np.fft.fft2(shifted)) / pupil.sum()


# The peak of the unocculted PSF on this grid, which references contrast.
TELESCOPE_PEAK = float(np.max(np.abs(focal(pupil)) ** 2))


class DriftingSpeckles:
    """A synthetic speckle field that follows the speckle-field protocol."""

    pixel_scale_lod = PIXSCALE_LOD
    # Total focal-plane flux of the clear aperture, in the units of |e_nom|^2.
    normalization = N_PIX**2 / pupil.sum()

    def __init__(self, n_ripples=12, seed=0):
        rng = np.random.default_rng(seed)
        # A static phase error whose amplitude falls as one over frequency.
        white = np.fft.fft2(rng.normal(size=(N_PIX, N_PIX)))
        k = np.hypot(*np.meshgrid(np.fft.fftfreq(N_PIX), np.fft.fftfreq(N_PIX)))
        static = np.real(np.fft.ifft2(white / np.maximum(k, 1.0 / N_PIX)))
        self.static_rad = 4e-4 * static / static[pupil > 0].std()
        # Drifting ripples, 3 to 11 cycles across the pupil.
        freq = rng.uniform(3.0, 11.0, n_ripples)
        angle = rng.uniform(0.0, 2.0 * np.pi, n_ripples)
        offset = rng.uniform(0.0, 2.0 * np.pi, n_ripples)
        self.ripples = np.stack(
            [
                np.cos(
                    2.0 * np.pi * f * (np.cos(a) * cycles_u + np.sin(a) * cycles_v) + o
                )
                for f, a, o in zip(freq, angle, offset)
            ]
        )
        self.amplitude_rad = 1.5e-4 * rng.uniform(0.5, 1.0, n_ripples)
        self.period_s = rng.uniform(1800.0, 7200.0, n_ripples)
        self.phase_rad = rng.uniform(0.0, 2.0 * np.pi, n_ripples)
        self.e_nom = self._field(self.static_rad)
        self.calls = 0

    def _field(self, phase_rad):
        # An ideal coronagraph removes the unaberrated field exactly.
        return focal(pupil * (np.exp(1j * phase_rad) - 1.0))

    def realize(self, *, wavelength_nm, time_s=0.0):
        self.calls += 1
        drift = self.amplitude_rad * np.sin(
            2.0 * np.pi * time_s / self.period_s + self.phase_rad
        )
        phase = self.static_rad + np.tensordot(drift, self.ripples, axes=1)
        total = np.abs(self._field(phase)) ** 2
        return (total - np.abs(self.e_nom) ** 2) / self.normalization


field = DriftingSpeckles()
```

## Preparing the sequence

Thirty exposures arrive a few minutes apart, with one forty-minute pause in
the middle. `prepare_speckles` evaluates the field once per epoch into one
allocated cube, adds the floor reconstructed from `e_nom` and `normalization`
(`include_floor=True` is the default), and divides by `telescope_peak`, the
peak of the unocculted PSF, so the frames are contrast. `trace=(3.0, 11.0)`
computes the mean over that annulus in $\lambda/D$ for every frame, from the
valid finite pixels only.

```{code-cell} python
rng = np.random.default_rng(0)
gaps_s = rng.uniform(150.0, 330.0, 29)
gaps_s[17] = 2400.0  # a pause between exposures
times_s = np.concatenate([[0.0], np.cumsum(gaps_s)])

sequence = prepare_speckles(
    field,
    times_s=times_s,
    wavelength_nm=550.0,
    telescope_peak=TELESCOPE_PEAK,
    trace=(3.0, 11.0),
    clock_fmt="t = {value:.2f} {unit}",
)
image = find_element(sequence.template, "image")
print(f"{len(sequence.times)} epochs, times in {sequence.time_unit!r}")
print(f"{image.scale.kind} scale from {image.scale.vmin:.2g} to {image.scale.vmax:.2g}")
print("field evaluations:", field.calls)
```

The prepared frames are a total intensity, so the scale is logarithmic. Its
bounds span the 1st to the 99.9th percentile of every positive, finite, valid
sample in the whole sequence, which keeps a few near-zero pixels from taking
the color range. Every frame, strip panel, and movie frame uses this one
scale: renormalizing each epoch would make a quietly boiling dark hole and
one degrading by an order of magnitude look the same. Pass `bounds=` to pin
the window instead, for example to compare two runs.

Times are held in the largest unit that keeps the numbers readable, here
hours. The template carries stable element IDs: an `"image"` panel with a
`"clock"` label and an `"annulus"` region, over a `"trace"` panel holding the
full `"series"` path and the one-point `"active"` datum.

## A still

`sequence.frame(i)` is the complete state at one epoch. Rendering it with
`eyepiece.mpl.render` gives a figure on the full-sequence scale. The epoch
below is the first one after the pause; the straight segment in the trace is
the pause itself, a stretch with no samples in it.

```{code-cell} python
fig, axes = plt.subplots(
    2, 1, figsize=(4.0, 5.2), height_ratios=[3.0, 1.0], layout="constrained"
)
still = mpl.render(sequence.frame(18), axes=axes, profile=PROFILE)
```

The returned `MplResult` maps each element ID to its artist, so a single
part is adjusted through its handle. For example,
`still.parts["annulus"].set_visible(False)` hides the annulus outline.

## A strip of epochs

A comparison between chosen epochs belongs in a strip. `boiling_strip`
accepts the prepared sequence directly and draws the selected epochs' image
panels side by side, each labelled once, by its clock, with its acquisition
time. Every panel
shares one extent and one scale, so one colorbar and one y label serve them
all.

```{code-cell} python
strip = boiling_strip(sequence, indices=[0, 10, 18, 29], profile=PROFILE)
```

Given raw input rather than a sequence, `boiling_strip` prepares it itself.
Preparing once and passing the sequence is what lets one preparation serve
a strip, a movie, and a talk.

## A movie

`eyepiece.mpl.animate` plays the sequence in physical time. The call below
asks for three seconds at ten frames per second: thirty output frames spaced
evenly from the first epoch to the last, each showing the most recent epoch
acquired at or before its time. Across the pause the display holds the last
exposure before it, rather than inventing states that were never observed.
The image, the trace datum, and the clock always name the same epoch.

```{code-cell} python
fig, axes = plt.subplots(
    2, 1, figsize=(3.6, 4.6), height_ratios=[3.0, 1.0], layout="constrained"
)
movie = mpl.animate(sequence, run_time=3.0, fps=10, axes=axes, profile=PROFILE)
html = movie.jshtml(dpi=100)
plt.close(fig)
print("output frames:", movie.n_frames)
print("field evaluations:", field.calls)
HTML(html)
```

The field was evaluated thirty times, once per epoch, during preparation.
The still, the strip, and the movie added no evaluations. `animate_speckles`
wraps preparation and this call in one step, for a movie that needs no other
output.

## A signed delta

The quantity decides the scale, not the sign of the data. With
`include_floor=False` the frames are the drift alone, a signed change in
contrast, and the preparation gives them a symmetric scale about zero with
the diverging residual colormap. Positive and negative excursions of equal
size then read as equally strong.

By default a delta's window spans the largest valid magnitude in the whole
sequence. Here the brightest few drifting speckles set that bound and leave
most of the field near zero, so the figure pins a narrower window with
`bounds=`.
A delta's bounds must be symmetric, and values beyond them are clipped to
the ends of the colormap, not masked.

```{code-cell} python
delta = prepare_speckles(
    field,
    times_s=times_s,
    wavelength_nm=550.0,
    telescope_peak=TELESCOPE_PEAK,
    include_floor=False,
    bounds=(-2e-9, 2e-9),
    clock_fmt="t = {value:.2f} {unit}",
)
frame = delta.frame(18)
print(frame.quantity, frame.scale.kind, (frame.scale.vmin, frame.scale.vmax))

fig, ax = plt.subplots(figsize=(3.8, 3.1), layout="constrained")
drift = mpl.render(frame, ax=ax, profile=PROFILE)
```

The clock stays readable over the near-white zero of the diverging map
because a label drawn over an image carries a halo in the background color.

A precomputed `(n_t, y, x)` cube enters through the same function. The
preparation cannot know what such a cube holds, so the caller declares it:
`sample_kind="instantaneous"` (exposure-integrated frames are refused) and
`quantity="total"` or `"delta"`. A cube is borrowed as given, without a copy
or a dtype change, and a `numpy.ma.MaskedArray` has its mask captured as
validity, so masked pixels stay invalid in every output.

Below, the contrast frames prepared earlier stand in for a cube loaded from
disk, with the occulted core inside $2\,\lambda/D$ masked.

```{code-cell} python
frames = np.stack(
    [find_element(sequence.frame(i), "image").data for i in range(len(times_s))]
)
x_lod = (np.arange(N_PIX) - (N_PIX - 1) / 2.0) * PIXSCALE_LOD
core = np.hypot(*np.meshgrid(x_lod, x_lod)) < 2.0
cube = np.ma.masked_array(frames, mask=np.broadcast_to(core, frames.shape))
bare = prepare_speckles(
    cube, times_s=times_s, sample_kind="instantaneous", quantity="total"
)
first = bare.frame(0)
print(f"valid pixels in frame 0: {first.valid.mean():.3f}")
try:
    prepare_speckles(cube, times_s=times_s, quantity="total")
except ValueError as err:
    print("refused:", err)
```

## In a talk

The same sequence plays as a native Manim clip through `eyepiece.manim`,
with a profile sized for slides. The code below is shown rather than run,
because rendering Manim media needs system libraries a documentation builder
does not have.

```python
import manim
import eyepiece.manim as em
from matplotlib.colors import to_hex

TALK = snapshot_profile()  # 24 pt text and 1.5 pt strokes


class Boiling(manim.Scene):
    def construct(self):
        self.camera.background_color = manim.ManimColor(to_hex(TALK.background_color))
        clip = em.animate(sequence, profile=TALK)
        self.add(clip.mobject)
        self.play(clip.playback(run_time=8))
```

Render it with Manim's caching disabled (`manim --disable_caching`). The
playback follows the same physical-time schedule as the Matplotlib movie, so
a slide and a movie of the same duration and frame rate show the same
epochs. Eyepiece's
[Manim guide](https://eyepiece.readthedocs.io/en/latest/manim.html) covers the
parts, reveals, and Manim Slides.

## Migrating from the earlier boiling functions

Earlier releases drew `boiling_strip` and `animate_speckles` directly from a
field or a cube. Both now prepare a sequence first and render it through
eyepiece, which changes the following behavior.

- A bare `(n_t, y, x)` cube must declare what it holds:
  `sample_kind="instantaneous"` and `quantity="total"` or `"delta"`. A field
  still declares both itself, deriving the quantity from `include_floor`.
- The scale follows the declared quantity, not the sign of the data. A total
  always takes a log scale and a delta always takes a symmetric diverging
  scale, even when none of its values are negative. Earlier releases chose
  the diverging scale only when a frame contained a negative value.
- `boiling_strip` no longer accepts `titles=` or `imshow_kw=`. Restyle a
  panel through its artist in `result.parts` instead, and set the time text
  with `clock_fmt=`.
- `boiling_strip` returns an `eyepiece.mpl.MplResult`, not a
  `MosaicResult`. Its parts are keyed by element ID: `parts["0/image"]` is
  the first panel's image and `parts["0/clock"]` its time label.
- Each panel's acquisition time is a label inside the panel rather than an
  axes title, in the strip and in the movie alike.
- `animate_speckles` plays in physical time. It takes `run_time=` (by
  default `len(times_s) / fps`) and spaces its output frames evenly from the
  first epoch to the last, holding the most recent epoch at each frame.
  Evenly spaced epochs play as before; unevenly spaced epochs hold through a
  long gap instead of advancing one epoch per output frame.

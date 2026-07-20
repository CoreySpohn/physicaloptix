# Conventions

The choices below are shared by every grid, field, and propagator in
physicaloptix. They are stated once here so the tutorials can assume them. Terms
in this page are defined in the [glossary](glossary).

## Coordinates and units

Each plane has its own natural unit, and a {class}`~physicaloptix.Grid` carries
its coordinates in that unit -- never in metres or pixels.

- A **pupil** grid, `Grid.pupil(npix)`, spans one aperture diameter: its
  coordinates run over roughly $[-0.5, 0.5]$ in units of the pupil diameter, so
  a clear circular aperture is the disk $x^2 + y^2 \le 0.25$.
- A **focal** grid, `Grid.focal(npix, pixel_scale)`, is sampled in
  {term}`lambda/D`: its `pixel_scale` is the {term}`lambda/D` per pixel, and its
  coordinates are angular separations in {term}`lambda/D`.
- **Wavelengths** are in nanometres everywhere they appear
  (`wavelength_nm`, `Spectrum`).

Because focal coordinates are in {term}`lambda/D`, a monochromatic result is
achromatic by construction: the same map applies at any wavelength once the
angular unit is fixed. Turning it into a physical angle or a detector position
is a wavelength-aware conversion the consumer does.

## The half-pixel-offset grid

Every grid is symmetric about its centre with **no sample at the origin**:

$$ x_i = \left(i - \tfrac{\mathrm{npix}}{2} + \tfrac{1}{2}\right)\,\mathrm{d}x,
\qquad i = 0, \dots, \mathrm{npix}-1. $$

The centre falls between the four central pixels (index $(\mathrm{npix}-1)/2$),
which is the shared radial convention `hwoutils.radial.radial_distance` uses, so
a {func}`~physicaloptix.stats.dark_zone_mask` annulus lines up with the PSF
centre. The missing sample at $r = 0$ is deliberate: it steps around the
$\mathrm{atan2}$ singularity at the core of a {term}`vortex coronagraph` mask,
where the gradient would otherwise be undefined.

## Plane tags

A {class}`~physicaloptix.Field` carries a {class}`~physicaloptix.PlaneKind` tag
(`PUPIL`, `FOCAL`, `INTERMEDIATE`, or `DETECTOR`). Propagators and elements
declare which plane they consume and which they produce, and the check runs when
the {class}`~physicaloptix.OpticalPath` is built. A mis-wired train -- a
focal-plane mask handed a pupil field -- fails at construction with a clear
message rather than returning a quietly wrong number at run time.

## Fields, intensity, and contrast

A `Field`'s `data` is the complex wavefront: its amplitude is the field
magnitude and its argument is the phase. Intensity is `field.intensity()`
($|\,\text{data}\,|^2$), and energy is the intensity integrated with the grid's
cell-area weight $\mathrm{d}x^2$.

Deep results are quoted as {term}`contrast`: intensity divided by the
unaberrated telescope PSF peak, so that a raw stellar peak is $1$ and an Earth
twin sits near $10^{-10}$. The normalization is always the *telescope* peak
(the aperture with no coronagraph), which is the quantity the speckle layer and
{func}`~physicaloptix.read_detector` both reference.

## Wavefront error

Wavefront error is an {term}`optical path difference` in **nanometres**, the
same length unit as the wavelength. A {class}`~physicaloptix.ModeBasis` returns
its modes already scaled so that a unit coefficient produces its stated
`rms_nm` of root-mean-square wavefront error over the aperture. An OPD of $h$
nanometres is a phase of $2\pi h / \lambda$; {func}`~physicaloptix.linearize`
applies that factor when it builds the sensitivity `G`.

## Precision

Deep contrast lives below the float32 floor, so any run that reaches for
$10^{-10}$ contrast -- a coronagraph null, the coherent speckle cross term --
needs float64:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

The forward propagation is float32-safe; the deep-null and pinning quantities
are not. Every tutorial that quotes a deep number sets this flag in its first
cell.

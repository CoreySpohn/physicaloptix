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

Every grid is symmetric about its centre; for the **even** `npix` used
throughout, there is no sample at the origin:

$$ x_i = \left(i - \tfrac{\mathrm{npix}}{2} + \tfrac{1}{2}\right)\,\mathrm{d}x,
\qquad i = 0, \dots, \mathrm{npix}-1. $$

The centre falls between the four central pixels (index $(\mathrm{npix}-1)/2$),
which is the shared radial convention `hwoutils.radial.radial_distance` uses, so
a {func}`~physicaloptix.stats.dark_zone_mask` annulus lines up with the PSF
centre. The missing sample at $r = 0$ is deliberate: it steps around the
$\mathrm{atan2}$ singularity at the core of a {term}`vortex coronagraph` mask,
where the gradient would otherwise be undefined. Odd `npix` is legal but places
a sample exactly at $r = 0$, forfeiting that protection -- use even grids in
any chain containing an azimuthal-phase mask.

## Signs and phases

One phase convention runs through every element and propagator, and every
formula imported from a textbook must be checked against it before use.

- **Accumulated path is positive phase.** An optical path difference of $h$
  nanometres multiplies the field by $\exp(+i\,2\pi h/\lambda)$
  ({class}`~physicaloptix.PhaseScreen`), and the exact angular-spectrum kernel
  advances a propagating wave with positive phase,
  $\exp\!\big(+i\,2\pi(z/\lambda)\sqrt{1-\lambda^2\nu^2}\big)$
  ({class}`~physicaloptix.Fresnel`). This is the $e^{+ikz}$ spatial
  convention, equivalently an implied $e^{-i\omega t}$ time dependence.
- **The forward Fourier kernel is $e^{-2i\pi u x}$** (pupil to focal;
  `cmft_fwd`), and the adjoint uses $e^{+2i\pi x u}$. Together with the sign
  above this fixes which way a tilt moves the PSF; the tilt column of `G` is
  pinned against the Fourier shift theorem in the [validation](validation)
  suite.
- **The paraxial Fresnel transfer is $\exp(-i\pi\alpha\nu^2)$** with
  $\alpha = \lambda z/D^2$ -- the transverse correction that accompanies the
  $+ikz$ carrier -- and beyond the evanescent cutoff the wave decays, never
  grows. Pinned by the Talbot phase-to-amplitude conversion law in the
  [validation](validation) suite.
- **Imported formulas are unverified until sign-checked.** Much of the optics
  literature -- notably the thin-film and coating lineage after Macleod, and
  any $e^{+i\omega t}$ engineering text -- uses the complex conjugate of this
  convention, and transcribing such a formula verbatim silently flips every
  phase it produces. The operational check: an index-matched slab of index
  $n$ and thickness $d$ is pure propagation and must multiply the field by
  $\exp(+i\,2\pi n d/\lambda)$; if a transcription yields the minus sign,
  conjugate it. When adding any element that imprints phase, pin its sign
  with an anchor test of this kind rather than trusting the source. The
  Macleod thin-film characteristic matrix is the complex conjugate of this
  convention, which is why {mod}`physicaloptix.coatings` negates its `1j`
  factors when it builds `multilayer_response`'s per-layer matrix.
- **A lossless beamsplitter reflects in quadrature.** `BeamSplitter.energy`
  uses the symmetric convention $t = \sqrt{1-R}$, $r = i\sqrt{R}$: the two
  ports differ by 90 degrees, so any future recombination is phased
  consistently. A binary-mask split (`from_mask`) is the amplitude complement
  ($t = m$, $r = 1 - m$) and is valid only for binary masks.

## Orientation and parity

The physics code shares one orientation system; state it before comparing
with any external code or transcribing any formula.

- **Sky-upright, everywhere.** A pupil tilt of $+p$ cycles per diameter moves
  the focal peak to $+p$ {term}`lambda/D`, and the return to a pupil plane is
  the adjoint transform, never a second forward transform -- so no plane in a
  chain is ever parity-flipped. The Lyot pupil is upright relative to the
  entrance pupil (the physical two-transform relay inverts it); formulas
  imported from treatments that carry the physical inversion must be
  re-expressed in upright coordinates.
- **Rows are y, ascending upward.** `data` is `(y, x)` with both axes sampled
  on the same ascending coordinates, so row 0 sits at $y = -\mathrm{extent}$.
  Display with `origin="lower"`; matplotlib's default renders every image
  vertically flipped.
- **Azimuth is counterclockwise from $+x$** ($\theta = \mathrm{atan2}(y, x)$)
  for vortex masks and Zernike modes alike, and a positive vortex charge
  applies $e^{+i\ell\theta}$. The on-axis null is blind to the sign of
  $\ell$; the handedness matters whenever it couples to anything external.
- **Off-axis offsets run along $+x$**, and exporting a coronagraph with a 1D
  offset list asserts azimuthal symmetry -- table readers rotate those PSFs
  to arbitrary position angles. A design without that symmetry must export
  2D offsets. The optical axis of an emitted map sits at pixel coordinate
  $(\mathrm{npix}-1)/2$ (0-based), which `emit_yip` records as
  XCENTER/YCENTER.

## Transform normalization and sampling gates

The MFT pair approximates the continuous Fourier transform: `cmft_fwd`
carries the input cell area $\mathrm{d}x^2$ and `cmft_bwd` the output's
$\mathrm{d}u^2$, so the pair is adjoint under the grids' weighted inner
products, **not unitary**. Round trip and Parseval hold only on complete
conjugate grids; on a zoomed focal grid the backward-forward composition is
a band-limited projection. The unaberrated focal peak amplitude therefore
equals the pupil integral -- $\pi/4$ for a unit-amplitude clear circle --
not 1. Grids are square, with one uniform ascending coordinate vector
serving both axes; the transforms infer the integration weight from the
first coordinate difference.

Sampling gates run once, at construction, against declared wavelengths, and
are never re-evaluated against a propagated field's spectrum. A chromatic
chain must declare its band edges: the red edge to
{class}`~physicaloptix.Fresnel` (`max_wavelength_nm`; the chirp worsens with
$\lambda$) and the blue edge to a fixed-grid `Fraunhofer`
(`min_wavelength_nm`; the scaled kernel is densest there). Omitting them
gates only the design wavelength.

## Wavelength binding and chromatic layout

Wavefront error is stored achromatically -- an OPD map in nanometres -- and
the wavelength binds late: the phasor $\exp(+i\,2\pi\,\mathrm{OPD}/\lambda)$
is formed where the optic is applied, once per wavelength. This OPD phasor
is the special case of a more general per-mode complex dispersion kernel
$D_k(\lambda)$ (log-amplitude and phase, applied as
$\exp(\sum_k c_k D_k(\lambda) B_k)$) that a
{class}`~physicaloptix.DispersiveScreen` supplies for non-OPD chromatic
responses such as coatings. A chromatic
{class}`~physicaloptix.Field` carries `data` of shape `(nlam, y, x)` with a
{class}`~physicaloptix.Spectrum`; elements and propagators act on the
wavelength axis slice by slice (nothing couples wavelength channels), and the
`Spectrum.weights` enter only at incoherent summation (`Field.intensity`).
Speckle sensitivity stacks keep the mode axis third from the end -- `G` is
`(m, y, x)` monochromatic or `(w, m, y, x)` chromatic -- so a mode count is
always `G.shape[-3]`, never `G.shape[0]`.

`Spectrum.weights` sum to one: chromatic intensity is a weighted band
*average*, never a band integral, and hand-built spectra must preserve the
normalization (only the constructors enforce it). `Spectrum.midpoint_band`
reproduces the design-survey band sampling (bin midpoints, endpoints
excluded) and is required for any survey-comparable result;
`Spectrum.tophat` includes the endpoints and is for standalone use. When a
`Fraunhofer` projects a chromatic field onto a fixed reference-wavelength
grid, each slice's amplitude carries the factor
$\lambda_{\mathrm{ref}}/\lambda$: fixed-grid intensity is a surface
brightness, not a per-pixel photon count. A band-averaged product must also
state its angular space -- YIP band images average on one fixed
reference-wavelength grid, while a chromatic-built `PathCoronagraph`
averages in each wavelength's native {term}`lambda/D`; prefer per-wavelength
instances in a `MultiBandCoronagraph` for chromatic work.

A monochromatic field carries no wavelength of its own: the *element's*
`wavelength_nm` performs the OPD-to-phase conversion, and nothing can
cross-check it against the wavelength the caller has in mind. A chromatic
field's spectrum overrides the element's stored wavelength entirely.

## Plane tags

A {class}`~physicaloptix.Field` carries a {class}`~physicaloptix.PlaneKind` tag
(`PUPIL`, `FOCAL`, `INTERMEDIATE`, or `DETECTOR`). Propagators and elements
declare which plane they consume and which they produce, and the check runs when
the {class}`~physicaloptix.OpticalPath` is built. A mis-wired train -- a
focal-plane mask handed a pupil field -- fails at construction with a clear
message rather than returning a quietly wrong number at run time.

## Validation runs at construction, not at mutation

Every gate in the library -- plane chaining, shape and kind checks, the
beamsplitter energy check, the propagator sampling gates -- runs in
`__check_init__` when an object is built. `eqx.tree_at` bypasses them all by
design; that is what makes per-step coefficient swaps free inside `jit`. The
contract: swap only leaf values of unchanged shape and meaning -- in
practice, `ModeBasis.coeffs` -- and reconstruct through the constructor for
anything else, because a `tree_at` swap of a mode stack, a grid, or a basis
kind is applied unvalidated. Optical layouts are also pytrees, and pytrees
are trees, not DAGs: an optic shared by two arms must appear exactly once,
in the trunk of an `OpticalSystem`; flattened per-channel views duplicate
the trunk and silently decouple under `jit`.

## Fields, intensity, and contrast

A `Field`'s `data` is the complex wavefront: its amplitude is the field
magnitude and its argument is the phase. Intensity is `field.intensity()`
($|\,\text{data}\,|^2$), and energy is the intensity integrated with the grid's
cell-area weight $\mathrm{d}x^2$.

Deep results are quoted as {term}`contrast`, but three normalizations
coexist in the stack and must not be mixed:

- **Per-pixel flux fraction** (the seam convention: the
  `AbstractCoronagraph` maps {class}`~physicaloptix.PathCoronagraph` serves,
  the YIP packages table readers consume, and -- since the 2026-07 seam fix
  -- what every `AbstractSpeckleField.realize` returns): the fraction of
  pre-coronagraph stellar flux landing in each pixel,
  $I\,\mathrm{d}u^2/E_{\mathrm{in}}$. An image-simulation consumer
  multiplies these maps by the star's total photon rate. The speckle classes
  derive this normalization at construction from the stored primitives
  (`input_energy`, `pixel_scale_lod`), which `linearize` records from the
  field it is handed.
- **Peak-referenced contrast** (a derived *view*, and the
  {func}`~physicaloptix.read_detector` reference): intensity divided by the
  unaberrated *telescope* PSF peak (the aperture with no coronagraph), so a
  raw stellar peak is $1$ and an Earth twin sits near $10^{-10}$. Obtain it
  from a flux-fraction field via
  {meth}`~physicaloptix.AnalyticSpeckleField.peak_contrast` with
  {func}`~physicaloptix.telescope_peak`; compute the peak **on the same
  focal grid as the data being normalized** (the half-pixel grid has no
  sample at the exact centre, so the sampled maximum is
  sampling-dependent, which is why the peak is never stored). The two
  conventions differ by the peak-pixel fraction (tens of times at typical
  sampling).
- **Intensity density** (`emit_yip` map values before the cell-area factor):
  intensity per $(\lambda_0/D)^2$ for unit pre-coronagraph energy, the
  design-survey pipeline's internal convention -- larger than the per-pixel
  fractions shipped survey packages carry by exactly the pixel cell area.
  Check which convention a package uses before serving it to a table reader.

{func}`~physicaloptix.read_detector` is sample-in, electron-out: one map
sample is one detector pixel (no area integration), and its `flux` argument
is the photon rate of the pixel where intensity equals 1 -- the stellar
peak pixel's rate, not the star's total rate.

When a normalization crosses a library seam, pass the primitive references
(input energy, pixel scale), never a pre-derived scalar: derived ratios such
as the peak-pixel fraction are sampling-dependent, and a bare number carries
no record of which convention it encodes. This is now the implemented
contract: speckle constructors take `input_energy` (there is no raw
`normalization` kwarg -- legacy call sites fail loudly), and npz/YIP exports
record `input_energy` and `pixel_scale_lod` beside their maps.

## Wavefront error

Wavefront error is an {term}`optical path difference` in **nanometres**, the
same length unit as the wavelength. A {class}`~physicaloptix.ModeBasis` is an
unnormalized container; the normalization contract belongs to the mode
constructors in `physicaloptix.elements.modes`, and it is **per mode, over
that mode's own support**: a unit coefficient produces `rms_nm` of RMS
wavefront error over the aperture disk for Zernike and Fourier modes, and
`ptt_nm` over the *individual segment* for the piston/tip/tilt basis, whose
over-aperture RMS is smaller by the square root of the segment-to-aperture
area ratio. Several unit coefficients add in quadrature.

An OPD of $h$ nanometres is a phase of $2\pi h/\lambda$;
{func}`~physicaloptix.linearize` applies that factor when it builds the
sensitivity `G`, and it injects the perturbation **at the input field's
plane** -- linearizing about an interior stage means linearizing the
sub-path from that stage onward. Coefficient units follow the basis kind:
`"opd"` coefficients are lengths in nanometres; `"amplitude"` coefficients
are dimensionless fractional amplitude ($E(1 + B\,\epsilon)$) with
wavelength-independent sensitivity columns. The analytic method is exact
because every stage is linear in the field; any future nonlinear stage makes
it silently wrong (cross-check against `method="jvp"`).

## The speckle process

A speckle realization is a **flux-fraction delta**: the wavefront-error
excess over the deterministic floor $|E_{\mathrm{nom}}|^2$, expressed as
per-pixel flux fractions ($\Delta I\,\mathrm{d}u^2/E_{\mathrm{in}}$) -- the
consumer's stellar-intensity map already carries the floor, so adding it
here would double count. The divisor is derived **once, at construction**
from the stored primitives (`normalization = input_energy /
pixel_scale_lod**2`), so the jitted `realize` hot path carries no unit
branching; because `eqx.tree_at` bypasses `__init__`, mutating
`input_energy` structurally does *not* re-derive it -- rebuild the field to
change photometry. Peak-referenced contrast is the
{meth}`~physicaloptix.AnalyticSpeckleField.peak_contrast` view. The default
`coherent=False` returns the strictly positive incoherent halo only; the
speckle-pinning cross term, which sets the bright-tail statistics, requires
`coherent=True` and float64.

The time axis is a frozen spectral synthesis: a draw fixes amplitudes and
phases, after which `realize` is deterministic and differentiable in time --
realizations vary by key, never by repeated calls at the same time. The
temporal PSD is the knee form $(1 + (f/f_{\mathrm{knee}})^2)^{s/2}$, and a
decorrelation time $\tau$ maps to $f_{\mathrm{knee}} = 1/(2\pi\tau)$ (the
Ornstein-Uhlenbeck identification, exact for $s = -2$). `epoch_jd` anchors
the clock: set it at or near the first observation -- the J2000 default puts
current-epoch observations $\sim 10^9$ seconds from the origin, which
destroys phase precision under float32 downstream.

Two draw statistics exist, and the defaults straddle them:
`draw(renormalize=True)` (the default) honors the wavefront-error budget
exactly per draw and is mildly sub-Gaussian, while `moments()` defaults to
describing the exact-Gaussian ensemble (`renormalize=False` draws) -- so
with both defaults a Monte-Carlo ensemble undershoots the predicted variance
by the closed-form kurtosis correction. Match the flags when comparing
drawn ensembles against closed-form moments.

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

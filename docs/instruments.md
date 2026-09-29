# Instrument builders

`physicaloptix.instruments` holds narrow builders that assemble an
{class}`~physicaloptix.OpticalPath` for one real instrument from an exported
optical prescription. There is no generic instrument registry: each builder
reproduces the plane sequence of one reference calculation, so the two codes
can be compared plane by plane. There are two: JWST NIRCam imaging and
round-mask coronagraphy, built to match STPSF, and the compact Roman
Coronagraph train, built to match the Roman preflight PROPER compact
prescription.

## The NIRCam path

{func}`~physicaloptix.instruments.build_nircam` takes a
{class}`~physicaloptix.instruments.NIRCamConfig` (wavelength, source
position, stage toggles and sampling) and a
{class}`~physicaloptix.instruments.NIRCamInputs` (the prescription arrays),
and returns the path and its entrance-plane source field.

| Stage | Plane | What it applies | When |
|-------|-------|-----------------|------|
| `entrance_pupil` | pupil | pupil amplitude transmission | always |
| `ote_opd` | pupil | fixed telescope (OTE) OPD, before the mask | `ote_opd=True` |
| `inversion` | pupil | reflection in y (`CoordinateInversion`) | always |
| `focal_mask` | pupil to pupil | pupil, focal-plane mask, Lyot-plane pupil | `focal_mask=True` |
| `lyot_stop` | pupil | Lyot-stop amplitude transmission | `lyot_stop=True` |
| `si_wfe` | pupil | instrument (SI) support times its OPD phasor, after the Lyot stop | `si_wfe=True` |
| `science_focal` | focal | MFT to `detector_npix * oversample` samples per side | always |

The OTE OPD acts before the focal mask and the SI wavefront error after the
Lyot stop. They are separate optics: folding them into one entrance screen
changes the coronagraphic image, because the occulter redistributes only the
light that reaches the mask plane with the OTE error. Every stage can be
tapped by name in `OpticalPath.propagate`, and
{func}`~physicaloptix.instruments.mask_plane_field` forms the field just after
the mask from the `inversion` tap.

```python
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from physicaloptix.instruments import (
    NIRCamConfig,
    NIRCamInputs,
    build_nircam,
    integrate_detector_pixels,
)

bundle = Path("path/to/bundle")
resolved = json.loads((bundle / "manifest.json").read_text())["configuration"]
inputs = NIRCamInputs.from_bundle(bundle)
config = NIRCamConfig(
    wavelength_nm=resolved["wavelength_nm"],
    pixel_scale_arcsec=resolved["pixelscale_arcsec"],
    detector_npix=resolved["fov_pixels"],
    oversample=resolved["detector_oversample"],
    source_position_arcsec=(1.0, 0.5),
    focal_mask=True,
    lyot_stop=True,
)
path, field = build_nircam(config, inputs)
out, _ = path.propagate(field)
energy = out.intensity() * out.grid.weights   # fraction of pupil energy per sample
pixels = integrate_detector_pixels(energy, config.oversample)
```

## Frames and the y inversion

- The pupil arrays are in the entrance-pupil frame (+x = +V2, +y = +V3,
  rows are y).
- The `inversion` stage reflects the field in y, as STPSF does after the
  telescope. The mask plane, the exit pupil (Lyot stop, SI wavefront error)
  and the detector are therefore in the entrance frame with y reversed. Lyot
  and SI arrays are given in that exit frame, as STPSF applies them.
- Source positions are in the detector-oriented output frame (x right, y up),
  like STPSF's `source_offset_x` / `source_offset_y`. The tilt is applied at
  the entrance pupil with its y component reflected, so the image lands at
  `+position`. The focal mask is centered at its own `center_arcsec` in the
  same frame, independent of the source.
- The output grid (x along array axis 1) is STPSF's undistorted `OVERSAMP`
  frame, and intensities agree with it sample by sample, including the parity
  of x-asymmetric wavefronts. As sky angles, x is along -V2 and y along +V3:
  the V2/V3 axes with the parity of the SIAF ideal frame (`VIdlParity = -1`).
  STPSF reaches the science ("sci", DMS) frame from it with a rotation by the
  aperture's `V3IdlYAngle` and the SIAF distortion (`add_distortion=True`),
  with no reflection. For the NIRCam apertures the ideal and sci axes share
  their parity, so a science-frame image maps onto this grid without a flip.
  The raw detector ("det") frame is a different frame: for NRCA5 its x axis
  is reversed relative to sci (`DetSciParity = -1`).

This stage is the one explicit exception to the library's rule that no plane
in a chain is parity-flipped; see [Conventions](conventions.md).

## OPD sign mapping

`NIRCamInputs` holds OPDs in this library's convention: nanometers of
accumulated optical path (a positive OPD is a delayed wavefront), phasor
$\exp(+2\pi i\,\mathrm{OPD}/\lambda)$ and forward kernel
$\exp(-2\pi i\,u x)$. STPSF (POPPY) writes OPDs in the Wyant-Creath sense, a
positive OPD being an advanced wavefront, with forward kernel
$\exp(+2\pi i\,u x)$ and the same phasor. The same physical surface error
therefore has opposite OPD signs in the two codes, and the same physical field
is represented by complex-conjugate amplitudes; intensities agree sample by
sample. The bundle readers convert each OPD once, at the import boundary:

$$
\mathrm{OPD}_\mathrm{nm} = -10^{9}\,\mathrm{OPD}_\mathrm{m}.
$$

Real amplitude transmissions (pupil, mask, Lyot stop, SI support) are
conjugation-invariant and are used as given.

## Normalization

The source field is scaled so the field just after the entrance pupil carries
unit energy, the normalization STPSF applies with `normalize='first'`. No
later renormalization happens, so `out.intensity() * out.grid.weights` is the
fraction of entrance-pupil energy per output sample, and finite-field, mask
and Lyot losses are kept.

## The focal-plane mask

{class}`~physicaloptix.instruments.BandLimitedRoundMask` is the round
band-limited occulter (MASK210R, MASK335R, MASK430R): amplitude transmission
$t(r) = 1 - (2 J_1(s)/s)^2$ with $s = \sigma r$ truncated at the second zero
of $J_1$, followed by rectangular override regions (the ND squares, the opaque
holder edge and the glass edge). It is evaluated analytically at each
wavelength.

The mask stage carries the pupil field $E$ to a mask-plane window with the MFT
pair $F$ / $B$ and returns $E + B[(M - 1) F E]$ to the Lyot plane (a Babinet
form). The NIRCam masks are not compactly supported (the holder and glass
edges run across any window), so the default window is the pupil raster's full
Nyquist band, on which $B F$ is the identity and the result equals the direct
propagation $B[M F E]$. Narrower windows (`mask_extent_lod`) are for
convergence studies only.

## The prescription bundle

`NIRCamInputs.from_bundle(path)` reads a self-describing directory:

| File | Content |
|------|---------|
| `manifest.json` | bundle name, each file's name and SHA-256, the analytic focal-mask record, and the band node sets |
| `pupil_amplitude.fits` | pupil amplitude transmission (required) |
| `ote_opd.fits` | OTE OPD in meters, entrance frame |
| `lyot_stop.fits` | Lyot-stop amplitude transmission, exit frame |
| `si_wfe_opd.fits` | SI OPD in meters, exit frame, at header `WAVELEN`; extension `SUPPORT` holds its amplitude support; extensions `OPD_<FILTER>_N<n>` hold per-node cubes |
| `focal_mask.fits` | the mask as STPSF sampled it, checked against the analytic record |

Pupil-grid files carry `BUNIT`, `PIXSCALE` in m/pix, `NPIX`, the optical axis
at `CENTERX = CENTERY = (NPIX - 1) / 2` and `PHASECNV` (the STPSF phasor).
Every file is checked against its manifest hash before use, and a unit, grid,
convention or checksum mismatch raises. The full layout is in the
`from_bundle` docstrings.

## Band integration

{class}`~physicaloptix.instruments.NIRCamBand` holds wavelength nodes, their
relative photon weights and, optionally, an SI OPD per node (the SI wavefront
error depends on wavelength). `NIRCamBand.from_bundle(path, nlambda)` reads one
node set. {func}`~physicaloptix.instruments.nircam_band_image` builds and
propagates every node independently (its own $\lambda/D$ scale, OPD phase,
source tilt, mask sampling and SI OPD) onto the single angular output grid set
by the pixel scale, and sums the energy images with weights normalized to unit
sum. The result stays a fraction of entrance-pupil energy; the weights do not
set absolute throughput.

## Detector pixels

{func}`~physicaloptix.instruments.integrate_detector_pixels` integrates an
oversampled energy image over detector pixels once, by summing the
`oversample x oversample` samples inside each pixel: the composite midpoint
rule, and STPSF's `DET_SAMP` rule. Its error falls as `oversample**-2`; no
further pixel kernel may be applied afterwards. It returns a block sum of
energies, whereas {func}`physicaloptix.ifs.pixel_integrate` returns the window
mean of an intensity: on aligned windows the first is `oversample**2` times
the second.

## Validation

Data-free tests pin the conventions (signed tilt and OPD anchors on even and
odd grids, the y inversion, the unity-mask identity, band nodes propagated
independently, second-order pixel convergence). The cross-code comparison
with STPSF references is the instrument benchmark in `tests/benchmark/`,
described with its environment variables and gate on the
[validation](validation.md) page.

## The Roman compact train

{class}`~physicaloptix.instruments.RomanCompact` is the monochromatic compact
model of the Roman Coronagraph: the same plane sequence as the compact
prescription distributed with the Roman preflight PROPER models, on the
prescription's own grids. It holds the arrays read from that prescription
(entrance pupil, shaped-pupil mask, focal-plane mask, Lyot stop, and optional
DM surface maps) and the grid sizes that go with them;
`RomanCompact.propagate` returns every stage as an array.

| Stage | Plane | What it applies |
|-------|-------|-----------------|
| `entrance` | pupil | pupil amplitude normalized to unit total intensity, source tilt |
| `dm1` | pupil | DM1 surface as twice-surface phase |
| `dm2` | pupil | Fresnel relay to DM2, DM2 surface |
| `back_to_dm1` | pupil | Fresnel relay back to DM1, trimmed to `n_big` |
| `pupil_mask` | pupil | shaped-pupil mask (when present) |
| `fpm` | focal | field at the focal-plane mask, before the mask |
| `fpm_exit` | pupil | field after the mask, trimmed to `n_small` |
| `lyot` | pupil | Lyot-stop amplitude |
| `image` | focal | Fraunhofer transform to `output_dim` samples, transposed |

Two mask forms are supported. A shaped-pupil design (`kind="spc"`) applies a
real mask directly between a forward and a backward matrix Fourier transform,
returning onto a grid `pupil_diam_pix` samples wide. A hybrid-Lyot design
(`kind="hlc"`) applies its complex mask in Babinet form: the field times the
mask's clear transmission, plus the backward transform of the forward field
times the mask minus one over the patterned region.

### Conventions

Every stage array is in prescription units (the physicaloptix field times the
plane's sample spacing, so the sum of squared magnitudes is the fraction of
entrance energy) and in prescription orientation. Arrays are centered at the
integer sample `n // 2`; {func}`~physicaloptix.instruments.roman.proper_trim`
crops and pads about that center. Three differences between the two codes are
handled inside the model rather than left to the caller:

- **Half-pixel grids.** On even grids physicaloptix samples sit half a sample
  from the prescription's. The source tilt is re-referenced by a constant
  phase and the image by the resulting linear phase, so complex fields match,
  not only intensities. The shaped-pupil return grid must therefore be an
  integer number of samples with the parity of `n_big`, or the forward and
  backward half-sample phases do not cancel; the constructor rejects any other
  combination.
- **Transform sign.** The prescription's transform to the mask plane uses a
  `+i` kernel and physicaloptix uses `-i`, so masks are applied rotated by
  180 degrees. Masks must be square with an odd side, for which the rotation
  is exact.
- **Output transpose.** The prescription transposes its final image, so the
  source offset `source_x_lod` lands on output columns and `source_y_lod` on
  output rows. Offsets are in lambda0/D at `lam0_nm` and scale as
  `lam0_nm / wavelength_nm` at other wavelengths.

DM surfaces enter as surface-height maps on the `n_small` grid; the reflected
wavefront carries twice the surface. The model does not convert actuator
commands to surfaces: influence functions, actuator registration and the DM2
orientation belong to whatever produced the maps.

### Validation

Data-free tests state the prescription's matrix Fourier transform and its
final focusing step directly in NumPy and require the model's mask-plane,
mask-exit and image fields to match them to $10^{-12}$ of their peak, for an
asymmetric shaped-pupil mask and a patterned complex hybrid-Lyot mask. Further
tests pin the integer-centered trim, entrance normalization, energy
conservation on an odd grid, the tilt phase and its direction on output rows
and columns, the Fresnel round trip, the twice-surface DM phase at each DM
plane, and the grid checks above. A cross-code comparison with the PROPER
compact prescription itself needs the prescription package and is not part of
the test suite; see the [validation](validation.md) page.

# Related work and prior art

physicaloptix did not appear from nothing. It sits in an active ecosystem of
open-source physical-optics libraries, and it borrows many ideas from them.
This page records where the core ideas come from and how physicaloptix compares
to the tools it most resembles, so that a reader arriving from any of them can
place it quickly.

## Why a purpose-built library

physicaloptix exists to serve one thing well: the HWO direct-imaging simulation
stack. That stack is under active development, and its interfaces change often
as the mission concept matures (and we change our mind about what we should be
doing). In that setting, owning a small, readable propagation library that can
move in step with the stack was the path of least resistance. A tool understood
end to end is easier to evolve alongside fast-moving requirements than a
general-purpose dependency, and owning it lets the coronagraph-specific
behavior be tuned exactly where it matters for HWO: focal sampling decoupled
from the pupil array so that deep nulls near $10^{-11}$ resolve, a
focal-plane-mask propagator that stays sampled through the phase singularity of
a vortex, and sampling adequacy that is checked when a path is built rather
than trusted at run time.

None of this is a departure from the wider ecosystem. physicaloptix began as a
thin wrapper around [dLux](https://github.com/LouisDesdoigts/dLux) and still
shares its central paradigm: an optical system is an Equinox `Module` PyTree
whose leaves are the physics, so the whole forward model is one pure function
that can be composed with `jax.jit`, `jax.vmap`, and `jax.grad`. It builds
directly on the ideas gathered below, complements those libraries rather than
competing with them, and several of its pieces are candidates to contribute back
upstream.

## How it compares

| Library | Backend and autodiff | Primary domain | Data model | Coronagraph focus |
|---|---|---|---|---|
| **physicaloptix** | JAX + Equinox, differentiable | HWO coronagraph propagation | plane-tagged `Field` on a static `Grid` | first class: owned multi-scale vortex, `PathCoronagraph`, live or freeze-to-table |
| [dLux](https://github.com/LouisDesdoigts/dLux) | JAX + Equinox, differentiable | general differentiable optical modelling, phase retrieval | `Wavefront` through composed `Optics` layers | by composition |
| [chromatix](https://github.com/chromatix-team/chromatix) | JAX + Equinox, differentiable | computational and microscopy wave optics | typed `Field` with explicit wavelength and polarization axes | not coronagraph specific |
| [hcipy](https://docs.hcipy.org) | NumPy (newer parts JAX tested) | high-contrast imaging and adaptive optics | `Field` = data + `Grid` | strong: Lyot, vortex, adaptive optics |
| [poppy](https://poppy-optics.readthedocs.io) | NumPy with optional acceleration | astronomical PSF, Fraunhofer and Fresnel | `Wavefront` with a `PlaneType` tag | Lyot style via semi-analytic MFT |
| [prysm](https://prysm.readthedocs.io) | NumPy | physical and first-order optics, interferometer data | arrays on coordinate grids | segmented apertures, deformable mirrors |

The short version: physicaloptix is closest to dLux and chromatix in machinery
(JAX, Equinox, autodiff), and closest to hcipy and poppy in purpose
(high-contrast coronagraphy). It is the intersection of those two groups.

## Ideas we build on

Almost every abstraction in physicaloptix has a clear ancestor. Naming them is
both an acknowledgment and a map for anyone who wants to go deeper.

- **Plane-tagged fields on an explicit grid.** The separation of field values
  from their sampling comes from hcipy, whose `Field` is data plus a `Grid`.
  The plane tag that a propagator validates comes from poppy, whose
  `PlaneType` is set when a wavefront is created and checked inside each optic.
  physicaloptix combines the two: a static `Grid` plus a `PlaneKind` tag that
  is checked when a path is built.
- **A matrix Fourier transform with free output sampling.** The semi-analytic
  MFT of Soummer et al. (2007) decouples the focal sampling from the pupil
  array size and is the backbone of Lyot-style coronagraph propagation. It is
  implemented in poppy (`matrix_dft`) and in hcipy. physicaloptix carries the
  continuous-Fourier weights exactly, so band sums and Babinet subtraction
  cancel to the deep floor.
- **The mode-basis primitive.** hcipy models a deformable mirror and modal
  wavefront error as a fixed basis cube times a coefficient vector.
  physicaloptix uses one `ModeBasis` for Zernike modes, band-limited Fourier
  and deformable-mirror modes, and segment piston-tip-tilt alike.
- **The multi-scale vortex.** The multi-resolution propagation that keeps a
  vortex focal-plane mask sampled through its central phase singularity is the
  algorithm hcipy pioneered, tiling focal grids of increasing resolution around
  the singularity.
- **Functional-to-sampled optic duality.** Building an optic as a callable over
  coordinates gives resolution independence and soft, differentiable edges;
  freezing it to a static array gives speed. This dual representation is
  poppy's `fixed_sampling_optic` and prysm's cached apertures.
- **Segmented apertures.** Per-segment local coordinate frames and the reuse of
  the few unique segment grids come from prysm's composite hexagonal aperture.
- **The differentiable-program paradigm.** The idea that a telescope forward
  model is a PyTree whose every physical parameter is a differentiable leaf is
  the shared foundation of dLux and chromatix.

## References

- Soummer, Pueyo, Sivaramakrishnan and Vanderbei (2007), "Fast computation of
  Lyot-style coronagraph propagation", Optics Express 15, 15935.
- Por, Haffert, Radhakrishnan, Doelman, Van Kooten and Bos (2018), "High
  Contrast Imaging for Python (HCIPy): an open-source adaptive optics and
  coronagraph simulator", Proc. SPIE 10703.
- Perrin, Soummer, Elliott, Lallo and Sivaramakrishnan (2012), "Simulating
  point spread functions for the James Webb Space Telescope with WebbPSF",
  Proc. SPIE 8442 (the poppy propagation engine).
- dLux: <https://github.com/LouisDesdoigts/dLux>
- chromatix: <https://github.com/chromatix-team/chromatix>
- prysm: <https://prysm.readthedocs.io>
- hcipy: <https://docs.hcipy.org>

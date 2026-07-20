# Installation

## Basic install

```bash
pip install physicaloptix
```

This pulls in physicaloptix and its JAX-CPU dependencies (JAX, Equinox,
optixstuff, hwoutils). For most analysis tasks -- building a coronagraph
path, propagating PSFs, running sampling gates -- the CPU build is fine.

## GPU install

GPU acceleration matters when you `vmap` a propagation over many
wavelengths, phase-screen realizations, or off-axis field points. JAX with
CUDA 12:

```bash
pip install physicaloptix jax[cuda12]
```

On a fresh Linux machine with a CUDA-capable GPU, that single command gets
you running. On a shared system you may need to point JAX at the right CUDA
runtime; see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for details.

## Double precision

physicaloptix propagation is validated in double precision. The continuous-FT
MFT pair and the construction-time sampling gates assume float64, so enable
x64 **before any other JAX-using import**:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Running in single precision will not raise, but the deep contrast floors that
the coronagraph propagation is built to resolve (nulls near 1e-11) are below
the float32 noise floor.

## Working from source

For development inside the HWO workspace:

```bash
git clone https://github.com/CoreySpohn/physicaloptix
cd physicaloptix
uv sync --all-packages
```

`uv` and `--all-packages` ensure every workspace dependency (optixstuff,
hwoutils, ...) installs as an editable workspace member.

To build these docs, install the `docs` extra and run Sphinx:

```bash
uv sync --extra docs
uv run sphinx-build docs docs/_build/html
```

## Where physicaloptix sits in the stack

physicaloptix is a downstream consumer of `optixstuff` and a functional
sibling of `yippy`; both back the same optixstuff `AbstractCoronagraph`
slot.

| Package | Role |
|---|---|
| `optixstuff` | Hardware description: primary aperture, coronagraph interface, detector, filters |
| `hwoutils` | Shared unit conversions, transforms, JAX configuration |
| `yippy` | Sampled-YIP PSF synthesis (the interpolated sibling of live propagation) |
| `tiptilt` | Wavefront-error generation and wavefront control, built on the `(E_nom, G)` linearization |
| `coronagraphoto` | 2D coronagraphic image simulation |
| `jaxedith` | Exposure-time and yield calculations |
| `skyscapes` | Scene model: stars, planets, disks, zodi |

`pip install physicaloptix` pulls `optixstuff` and `hwoutils` as transitive
dependencies. If you `uv sync` from source, use `--all-packages` to get the
workspace-editable versions.

## Verifying the install

```python
import physicaloptix

print(physicaloptix.__version__)
```

The first JAX import takes 10-20 s on cold cache as XLA initializes. This is
normal.

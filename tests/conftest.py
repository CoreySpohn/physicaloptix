"""Shared fixtures for physicaloptix tests."""

import importlib.util
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils import enable_x64, set_platform
from hypothesis import settings
from optixstuff import FocalResponse
from optixstuff.grids import Frame, PixelGrid
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import _grid_axis
from physicaloptix.elements import ModeBasis, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

# The deep-contrast path is x64-mandatory (f32 floors the dark hole at ~3e-4),
# and the suite must be deterministic: pin tests to CPU x64. Builder runs opt
# into GPU explicitly outside the test suite.
set_platform("cpu")
enable_x64()

TEST_FRAME = Frame(500.0, 1.0, "circumscribed")

EAC1_CACHE_ENV = "PHYSICALOPTIX_EAC1_CACHE"
_DATA_DIRS = (Path(__file__).parent / "data",)


def find_data_file(name):
    """Locate a reference data file in ``tests/data/``."""
    for directory in _DATA_DIRS:
        path = directory / name
        if path.exists():
            return path
    return None


@pytest.fixture(scope="session")
def dense_speckle_export():
    """The dense-basis (E_nom, G) reference export, if available locally."""
    path = find_data_file("speckle_dense_eac1.npz")
    if path is None:
        pytest.skip("dense-basis speckle export not found")
    return np.load(path)


@pytest.fixture(scope="session")
def eac1_cache():
    """The cds_pipeline EAC-1 AAVC reference cache (masks + reference PSFs).

    Looked up from ``$PHYSICALOPTIX_EAC1_CACHE``, then ``tests/data/``.
    Gate tests skip when no copy is available
    (e.g. CI).
    """
    env = os.environ.get(EAC1_CACHE_ENV)
    if env and Path(env).exists():
        return np.load(env)
    path = find_data_file("cds_eac1_ref.npz")
    if path is not None:
        return np.load(path)
    pytest.skip(f"cds EAC-1 reference cache not found (set {EAC1_CACHE_ENV})")


def _disk_pupil(npup=24):
    grid = PixelGrid.pupil(npup)
    x = np.asarray(_grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    return grid, ((xx**2 + yy**2) <= 0.25).astype(float)


@pytest.fixture
def small_path():
    """Disk stop + Fraunhofer, native lambda/D output (no reference wavelength)."""
    grid, disk = _disk_pupil()
    return OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
                ),
            ),
            Stage(
                "science",
                Fraunhofer(grid_in=grid, grid_out=PixelGrid.focal(32, 0.5, TEST_FRAME)),
            ),
        )
    )


@pytest.fixture
def opd_basis():
    """Unit-normal 4-mode OPD basis (nm); the seam test's tolerance assumes
    the unit normalization -- do not rescale B here."""
    rng = np.random.default_rng(0)
    return ModeBasis(
        B=jnp.asarray(rng.standard_normal((4, 24, 24))), coeffs=jnp.zeros(4)
    )


@pytest.fixture
def mono_field():
    grid, disk = _disk_pupil()
    return Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )


@pytest.fixture
def chromatic_field():
    """Three-band field with wavelength-identical slices (a flat input)."""
    grid, disk = _disk_pupil()
    spectrum = Quadrature.build(
        (Band(500.0 * (1 - 0.2 / 2), 500.0 * (1 + 0.2 / 2)),), order=3
    )
    data = jnp.broadcast_to(jnp.asarray(disk).astype(complex), (3, 24, 24))
    return Field(data=data, grid=grid, plane=PlaneKind.PUPIL, spectrum=spectrum)


# --- speckle-field protocol fixtures (viz preparation and boiling tests) ----

SPECKLE_NPIX = 16


class ProtocolFake:
    """The whole speckle-field protocol and nothing else.

    If the viz layer needs more than ``realize`` and ``pixel_scale_lod``, it
    is typed on a class rather than on the protocol, and a replayed cube or a
    maintained residual would need an adapter. ``calls`` counts evaluations,
    so a test can pin that preparation evaluates each epoch exactly once.
    """

    grid = PixelGrid.focal(SPECKLE_NPIX, 0.25, TEST_FRAME)

    def __init__(self):
        self.calls = 0

    def realize(self, *, wavelength_nm, elapsed_s=0.0):
        # Asymmetric in x/y AND time-varying: a uniform or static fixture
        # cannot tell a shared norm from a per-panel one, nor an x/y
        # conflation from a correct one.
        self.calls += 1
        y, x = np.ogrid[:SPECKLE_NPIX, :SPECKLE_NPIX]
        from optixstuff.grids import Image, Measure

        return Image(
            jnp.asarray(1e-9 * (1.0 + 0.5 * np.sin(elapsed_s)) * (1.0 + x + 3.0 * y)),
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )


@pytest.fixture
def protocol_fake():
    """A fresh ProtocolFake with its evaluation counter at zero."""
    return ProtocolFake()


@pytest.fixture
def analytic_speckle_field():
    """Factory for a small real AnalyticSpeckleField (the concrete protocol end)."""

    def build(*, coherent=False, seed=0):
        rng = np.random.default_rng(seed)
        shape = (SPECKLE_NPIX, SPECKLE_NPIX)
        e_nom = jnp.asarray(
            rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=complex
        )
        g = jnp.asarray(
            rng.normal(size=(3, *shape)) + 1j * rng.normal(size=(3, *shape)),
            dtype=complex,
        )
        return analytic_from_density(
            e_nom * 1e-4,
            g * 1e-5,
            amplitudes=jnp.asarray(rng.random((3, 2))),
            frequencies_hz=jnp.asarray([1e-3, 3e-3]),
            phases=jnp.asarray(rng.random((3, 2)) * 2.0 * np.pi),
            input_energy=1.0,
            grid=speckle_grid((e_nom * 0.0001).shape[-2:], 0.25, 500.0),
            reference_wavelength_nm=500.0,
            coherent=coherent,
        )

    return build


def as_image(data):
    """Per-pixel energy fixture on a declared unit arcsecond grid."""
    from optixstuff.grids import Image, Measure

    data = jnp.asarray(data)
    return Image(
        data,
        PixelGrid.regular(data.shape[-2:], 1.0, "arcsec"),
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )


def speckle_grid(shape, step=0.25, reference=500.0):
    """Declared geometry for synthetic density-amplitude fixtures."""
    return PixelGrid.regular(
        shape, step, "lod", frame=Frame(reference, 1.0, "circumscribed")
    )


def analytic_from_density(
    e_nom,
    G,
    amplitudes,
    frequencies_hz,
    phases,
    *,
    input_energy,
    grid,
    reference_wavelength_nm,
    epoch_d=0.0,
    coherent=False,
    wavelengths_nm=None,
):
    """Construct a deterministic synthesis through the density producer boundary."""
    import equinox as eqx
    import jax

    from physicaloptix.speckle import SpeckleProcess

    process = SpeckleProcess(
        FocalResponse(e_nom, G, grid, input_energy, reference_wavelength_nm),
        1.0,
        1e-3,
        epoch_d=epoch_d,
        coherent=coherent,
        wavelengths_nm=wavelengths_nm,
        n_freq=max(2, amplitudes.shape[-1]),
    )
    field = process.draw(jax.random.PRNGKey(0))
    return eqx.tree_at(
        lambda f: (f.amplitudes, f.frequencies_hz, f.phases),
        field,
        (amplitudes, frequencies_hz, phases),
    )


# Reference-data gate and executed-case accounting for the benchmark tier.
#
# PHYSICALOPTIX_REQUIRE_REFERENCES=1 turns every missing reference into a
# failure (not a skip) and makes a run that collected benchmark cases but
# executed none a failure too. Without it, a missing reference skips with a
# reason naming the environment variable to set. When benchmark cases are
# collected, the terminal summary reports how many executed and lists them,
# so a green run that compared nothing is visible. The gate covers only
# tests collected under tests/benchmark/; the design-survey tests in
# tests/validation/ keep their own skip-when-absent behavior. It lives here,
# not in a benchmark conftest, so that one module named conftest serves the
# helpers the test files import.

REQUIRE_ENV = "PHYSICALOPTIX_REQUIRE_REFERENCES"
BUNDLE_ENV = "PHYSICALOPTIX_NIRCAM_BUNDLE"
REFERENCE_ENV = "PHYSICALOPTIX_NIRCAM_REFERENCE"
EXECUTED = []
BENCHMARK_COLLECTED = []


def references_required():
    return os.environ.get(REQUIRE_ENV, "") == "1"


def require_path(env_name, *, marker):
    """Directory from ``env_name`` containing ``marker``, or skip / fail."""
    raw = os.environ.get(env_name)
    problem = None
    if not raw:
        problem = f"{env_name} is not set"
    elif not (Path(raw).expanduser() / marker).is_file():
        problem = f"{env_name}={raw} has no {marker}"
    if problem is None:
        return Path(raw).expanduser()
    message = f"benchmark reference data absent: {problem}"
    if references_required():
        pytest.fail(f"{message} ({REQUIRE_ENV}=1 requires it)", pytrace=False)
    pytest.skip(f"{message}; set {env_name} (and {REQUIRE_ENV}=1 to require it)")


@pytest.fixture(scope="session")
def nircam_bundle_dir():
    """The exported NIRCam prescription bundle (``manifest.json`` inside)."""
    return require_path(BUNDLE_ENV, marker="manifest.json")


@pytest.fixture(scope="session")
def nircam_reference_dir():
    """The STPSF references: ``<case>.fits`` and ``band/manifest.json``."""
    return require_path(REFERENCE_ENV, marker="band/manifest.json")


@pytest.fixture
def record_case():
    """Record one executed comparison: ``record_case(case_id, summary)``."""

    def record(case_id, summary):
        EXECUTED.append((case_id, summary))

    return record


def pytest_collection_modifyitems(items):
    BENCHMARK_COLLECTED[:] = [
        item for item in items if "benchmark" in Path(str(item.path)).parent.parts
    ]


def pytest_terminal_summary(terminalreporter):
    if not BENCHMARK_COLLECTED:
        return
    gate = "on" if references_required() else "off"
    terminalreporter.write_sep(
        "-", f"benchmark cases executed: {len(EXECUTED)} ({REQUIRE_ENV} {gate})"
    )
    for case_id, summary in EXECUTED:
        terminalreporter.write_line(f"  {case_id}: {summary}")


def pytest_sessionfinish(session, exitstatus):
    if (
        BENCHMARK_COLLECTED
        and references_required()
        and not EXECUTED
        and exitstatus in (0, 5)
    ):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print(f"\n{REQUIRE_ENV}=1 but no benchmark case executed: failing the run")

"""Mirror-train construction: PSD surfaces and the equivalent-space Fresnel chain."""

import dataclasses
from importlib import resources

import jax.numpy as jnp
import numpy as np
import yaml

from physicaloptix.core import PlaneKind
from physicaloptix.elements import ModeBasis, PhaseScreen
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms.fresnel import Fresnel

REFLECTION_OPD_FACTOR = 2.0
"""Surface height to optical path at near-normal incidence on reflection."""


def synthesize_psd_surface(seed, grid, *, rms_nm, k_min=1.0, k_max=None, slope=-2.5):
    """A band-limited power-law-PSD surface height map (nm) on a pupil grid.

    Fourier amplitudes follow ``k**(slope / 2)`` (PSD ``k**slope``) between
    ``k_min`` and ``k_max`` cycles/pupil with uniform random phases; the real
    map is zero-mean and rescaled to exactly ``rms_nm``. Deterministic per
    ``seed``. A generic polished-surface placeholder: swap in measured maps
    when metrology exists.
    """
    if k_min <= 0:
        raise ValueError(f"k_min must be positive, got {k_min}")
    if k_max is None:
        k_max = grid.npix / 4
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(grid.npix, d=grid.dx)
    kk = np.hypot(*np.meshgrid(k, k))
    amplitude = np.zeros_like(kk)
    band = (kk >= k_min) & (kk <= k_max)
    if not band.any():
        raise ValueError(f"no grid frequencies fall in [{k_min}, {k_max}]")
    amplitude[band] = kk[band] ** (slope / 2.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, kk.shape)
    surface = np.real(np.fft.ifft2(amplitude * np.exp(1j * phases)))
    surface -= surface.mean()
    return surface * (rms_nm / surface.std())


@dataclasses.dataclass(frozen=True)
class MirrorSpec:
    """One mirror of an equivalent-collimated-space train.

    ``alpha = lambda z / D**2`` is the conjugation parameter at the train's
    reference wavelength; ``surface_nm`` / ``drift_basis.B`` are SURFACE
    height maps in nm (the builder applies ``REFLECTION_OPD_FACTOR``).
    ``provenance`` records where alpha came from ("exact", "assumption",
    "representative").

    Attributes:
        name: Stage-name stem for this mirror's figure/drift stages.
        alpha: Conjugation parameter at the train's reference wavelength.
        surface_nm: Static SURFACE height map (nm), or ``None`` to omit the
            figure stage.
        drift_basis: A ``ModeBasis`` (SURFACE height, nm) of the caller's
            drift modes, or ``None`` to omit the drift stage.
        provenance: Where ``alpha`` came from ("exact", "assumption",
            "representative", or "unspecified").
    """

    name: str
    alpha: float
    surface_nm: object = None
    drift_basis: object = None
    provenance: str = "unspecified"


def build_mirror_train(
    specs, grid, *, wavelength_nm, beam_diameter_m, npad=1, on_undersampled="warn"
):
    """Chain mirrors into an ``OpticalPath`` of Fresnel hops + phase screens.

    Mirrors are ordered by ascending ``alpha``; an ``alpha == 0`` mirror acts
    at the pupil before any hop. Hop distances are alpha deltas converted by
    ``z = alpha * beam_diameter_m**2 / (wavelength_nm * 1e-9)``, and a closing
    ``-alpha_max`` hop returns the chain to the pupil plane.

    Args:
        specs: Iterable of ``MirrorSpec``.
        grid: The (shared) propagation grid every stage uses.
        wavelength_nm: Reference wavelength (nm) for both the alpha-to-meters
            conversion and the phase-screen OPD-to-phase conversion.
        beam_diameter_m: Physical beam diameter (m) at every plane.
        npad: Real-domain zero-pad factor forwarded to each ``Fresnel`` hop.
        on_undersampled: Sampling-gate policy forwarded to each ``Fresnel``
            hop ("raise", "warn", or "record").

    Returns:
        The assembled ``OpticalPath``.
    """
    ordered = sorted(specs, key=lambda s: s.alpha)
    if any(s.alpha < 0 for s in ordered):
        raise ValueError("mirror alpha must be non-negative")
    z_unit = beam_diameter_m**2 / (wavelength_nm * 1e-9)

    def hop(name, d_alpha, plane_in, plane_out):
        return Stage(
            name,
            Fresnel(
                grid=grid,
                distance_m=d_alpha * z_unit,
                beam_diameter_m=beam_diameter_m,
                wavelength_nm=wavelength_nm,
                plane_in=plane_in,
                plane_out=plane_out,
                npad=npad,
                on_undersampled=on_undersampled,
            ),
        )

    def screens(spec, plane):
        out = []
        if spec.surface_nm is not None:
            opd = REFLECTION_OPD_FACTOR * jnp.asarray(spec.surface_nm)
            basis = ModeBasis(B=opd[None], coeffs=jnp.ones(1))
            out.append(
                Stage(
                    f"{spec.name}_figure",
                    PhaseScreen(basis, grid, wavelength_nm=wavelength_nm, plane=plane),
                )
            )
        if spec.drift_basis is not None:
            basis = ModeBasis(
                B=REFLECTION_OPD_FACTOR * spec.drift_basis.B,
                coeffs=spec.drift_basis.coeffs,
                kind=spec.drift_basis.kind,
            )
            out.append(
                Stage(
                    f"{spec.name}_drift",
                    PhaseScreen(basis, grid, wavelength_nm=wavelength_nm, plane=plane),
                )
            )
        return out

    stages, current_alpha, plane = [], 0.0, PlaneKind.PUPIL
    for spec in ordered:
        if spec.alpha > 0:
            stages.append(
                hop(
                    f"hop_to_{spec.name}",
                    spec.alpha - current_alpha,
                    plane,
                    PlaneKind.INTERMEDIATE,
                )
            )
            current_alpha, plane = spec.alpha, PlaneKind.INTERMEDIATE
        stages.extend(screens(spec, plane))
    if current_alpha > 0:
        stages.append(hop("hop_to_exit_pupil", -current_alpha, plane, PlaneKind.PUPIL))
    return OpticalPath(stages=tuple(stages))


def load_train_yaml(path=None):
    """Load a train geometry config; ``None`` loads the bundled EAC-1 file.

    Args:
        path: Path to a train-geometry YAML file, or ``None`` to load the
            package's bundled EAC-1 config.

    Returns:
        The parsed config dict, with keys ``reference_wavelength_nm``,
        ``beam_diameter_m``, and ``mirrors`` (a list of per-mirror dicts).
    """
    if path is None:
        text = (
            resources.files("physicaloptix") / "data" / "eac1_ci_train.yaml"
        ).read_text()
    else:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    config = yaml.safe_load(text)
    for key in ("reference_wavelength_nm", "beam_diameter_m", "mirrors"):
        if key not in config:
            raise ValueError(f"train config missing {key!r}")
    return config

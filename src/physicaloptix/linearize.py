"""linearize(): one entry point for the (E_nom, G) linearization.

The path is linear in the field, so the first-order focal response to an OPD
mode is exact: ``G_k = i (2 pi / lambda) Path(B_k * E_in)`` (the analytic
method, cheapest and the default). ``jvp`` and ``jacfwd`` differentiate the
full nonlinear map ``eps -> Path(E_in * exp(i 2 pi (B . eps) / lambda))`` at
``eps = 0`` and exist as autodiff cross-checks; all three agree to roundoff
on a linear chain.

Amplitude bases (``kind="amplitude"``) linearize the fractional-amplitude
map ``E * (1 + B . eps)``: their columns are ``Path(B_k * E_in)`` with no
phase factor, so G is achromatic for them.

Memory policy: mode stacks at pupil resolution are the wall (a dense basis at
2048^2 in complex128 is tens of MB per mode), so the analytic method streams
mode chunks through a vmapped propagation when the full stack would exceed
``memory_budget_bytes``; ``jvp`` is a host-side loop, memory-flat by
construction. The chromatic (per-band or lambda-scaled) extension is not
implemented yet; the ``kind`` tag is recorded so the lambda-scaling shortcut,
when it lands, applies only to OPD bases (its ``(lambda0/lambda)^2`` rule
comes from the phase factor and does not apply to amplitude modes).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from physicaloptix.core import Field
from physicaloptix.speckle import SpeckleProcess


class Linearization(eqx.Module):
    """The (E_nom, G) product of ``linearize``.

    Attributes:
        e_nom: Complex nominal focal field, shape ``(y, x)``.
        G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``,
            per unit mode coefficient (an OPD basis in the same length unit
            as ``wavelength_nm``).
        wavelength_nm: Design wavelength the phase factor used.
        method: The method that actually ran (``auto`` resolves before this
            is recorded).
        kind: The basis kind the columns were built from (``"opd"``).
        pixel_scale_lod: The output plane's pixel scale (lambda/D per pixel for
            a focal output) -- the grid ``e_nom`` / ``G`` are sampled on, so a
            speckle field built from this product carries its true plate scale.
    """

    e_nom: Array
    G: Array
    wavelength_nm: float = eqx.field(static=True)
    method: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    pixel_scale_lod: float = eqx.field(static=True)

    @property
    def n_modes(self):
        """Number of sensitivity columns."""
        return self.G.shape[0]

    def to_speckle_process(
        self,
        *,
        normalization,
        per_mode_rms=None,
        knee_hz=None,
        decorr_hours=None,
        total_rms=None,
        **kwargs,
    ):
        """Package (E_nom, G) as a ``SpeckleProcess``.

        Either give the process parameters directly (``per_mode_rms`` +
        ``knee_hz``) or the decorrelation parameterization
        (``decorr_hours`` + ``total_rms``).

        Args:
            normalization: Intensity that maps to unit contrast (the
                telescope PSF peak the focal field is referenced to).
            per_mode_rms: Per-mode rms drift (with ``knee_hz``).
            knee_hz: Temporal PSD knee frequency (with ``per_mode_rms``).
            decorr_hours: Decorrelation time (with ``total_rms``).
            total_rms: Total WFE budget, split evenly over modes.
            **kwargs: Forwarded to ``SpeckleProcess``.

        Returns:
            The parameter object whose ``draw(key)`` yields
            ``AnalyticSpeckleField`` realizations.
        """
        kwargs.setdefault("pixel_scale_lod", self.pixel_scale_lod)
        if decorr_hours is not None:
            return SpeckleProcess.from_decorrelation(
                self.e_nom,
                self.G,
                decorr_hours=decorr_hours,
                total_rms=total_rms,
                normalization=normalization,
                **kwargs,
            )
        return SpeckleProcess(
            self.e_nom,
            self.G,
            per_mode_rms,
            knee_hz,
            normalization,
            **kwargs,
        )


def _phase_factor(wavelength_nm):
    return 1j * 2.0 * jnp.pi / wavelength_nm


def _replace_data(field, data):
    return Field(data=data, grid=field.grid, plane=field.plane, spectrum=field.spectrum)


def perturbed_map(path, field, basis, wavelength_nm):
    """The nonlinear map ``eps -> E_focal`` the linearization approximates.

    OPD modes perturb the phase (``E * exp(i 2 pi (B . eps) / lambda)``);
    amplitude modes perturb the field multiplicatively
    (``E * (1 + B . eps)``, fractional amplitude, achromatic).
    """

    def run(eps):
        mode_map = jnp.tensordot(eps, basis.B, axes=1)
        if basis.kind == "opd":
            data = field.data * jnp.exp(_phase_factor(wavelength_nm) * mode_map)
        else:
            data = field.data * (1.0 + mode_map)
        out, _ = path.propagate(_replace_data(field, data))
        return out.data

    return run


def linearize(
    path,
    field,
    basis,
    *,
    wavelength_nm,
    method="auto",
    chunk_size=None,
    memory_budget_bytes=4 * 2**30,
):
    """Build the (E_nom, G) linearization of a path around ``field``.

    Args:
        path: The ``OpticalPath`` (or any object with ``propagate``); every
            stage must be linear in the field for the analytic method.
        field: The unperturbed input field; the OPD perturbation applies at
            this plane.
        basis: An OPD ``ModeBasis`` in the same length unit as
            ``wavelength_nm``.
        wavelength_nm: Design wavelength for the phase factor.
        method: ``"analytic"`` (default via ``"auto"``), ``"jvp"``, or
            ``"jacfwd"``.
        chunk_size: Modes per propagation batch for the analytic method;
            ``None`` batches all modes (subject to ``memory_budget_bytes``
            under ``"auto"``).
        memory_budget_bytes: When ``method="auto"``, the mode-stack size
            above which the analytic method streams chunks instead of
            batching everything.

    Returns:
        A ``Linearization``.
    """
    e_nom_field, _ = path.propagate(field)
    n_modes = basis.n_modes

    resolved = method
    if method == "auto":
        resolved = "analytic"
        estimate = n_modes * field.data.size * 16
        if chunk_size is None and estimate > memory_budget_bytes:
            per_mode = field.data.size * 16
            chunk_size = max(1, int(memory_budget_bytes // per_mode))

    if resolved == "analytic":
        # OPD columns carry the phase factor; amplitude columns are the
        # propagated fractional-amplitude modes themselves (achromatic).
        factor = _phase_factor(wavelength_nm) if basis.kind == "opd" else 1.0
        propagate_stack = jax.vmap(
            lambda data: path.propagate(_replace_data(field, data))[0].data
        )

        def columns(mode_chunk):
            return propagate_stack(factor * mode_chunk * field.data)

        if chunk_size is None:
            g = columns(basis.B)
        else:
            g = jnp.concatenate(
                [
                    columns(basis.B[start : start + chunk_size])
                    for start in range(0, n_modes, chunk_size)
                ]
            )
    elif resolved == "jvp":
        run = perturbed_map(path, field, basis, wavelength_nm)
        zero = jnp.zeros(n_modes)
        cols = [
            jax.jvp(run, (zero,), (jnp.zeros(n_modes).at[k].set(1.0),))[1]
            for k in range(n_modes)
        ]
        g = jnp.stack(cols)
    elif resolved == "jacfwd":
        run = perturbed_map(path, field, basis, wavelength_nm)
        jacobian = jax.jacfwd(run)(jnp.zeros(n_modes))
        g = jnp.moveaxis(jacobian, -1, 0)
    else:
        raise ValueError(f"method must be auto/analytic/jvp/jacfwd, got {method!r}")

    return Linearization(
        e_nom=e_nom_field.data,
        G=g,
        wavelength_nm=float(wavelength_nm),
        method=resolved,
        kind=basis.kind,
        pixel_scale_lod=float(e_nom_field.grid.dx),
    )


def linearity_residual(path, field, basis, linearization, eps):
    """Relative error of the linear model at coefficients ``eps``.

    ``|E(eps) - (E_nom + G eps)| / |E(eps)|`` over the focal plane -- the
    small-phase validity check (scales as ``eps^2``).
    """
    eps = jnp.asarray(eps)
    run = perturbed_map(path, field, basis, linearization.wavelength_nm)
    exact = run(eps)
    linear = linearization.e_nom + jnp.tensordot(eps, linearization.G, axes=1)
    return float(jnp.linalg.norm(exact - linear) / jnp.linalg.norm(exact))

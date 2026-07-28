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
construction.

Chromatic linearization: passing ``wavelengths_nm`` (matching
``field.spectrum.wavelengths_nm``) builds a per-band stack, ``e_nom``
``(w, y, x)`` and ``G`` ``(w, m, y, x)``, one column set per wavelength. The
hardcoded OPD phase law is replaced by a per-mode, per-wavelength complex
factor table (``dispersion``, shape ``(m, w)``): the default derives from
``basis.kind`` (``i 2 pi / lambda_w`` for ``"opd"``, ``1`` for
``"amplitude"``), or an explicit table (e.g.
:meth:`physicaloptix.elements.DispersiveScreen.kernel_at`) stands in for a
non-OPD chromatic response. The perturbation is applied at the INPUT plane
``field`` is given at (the same scope the mono method has always had); a
mode's effect elsewhere in the path is only captured through
``dispersion`` when that stage's own response is linear and diagonal in the
mode's spatial map -- see :class:`physicaloptix.elements.DispersiveScreen`
for the case this makes exact. The per-mode dispersion factors are otherwise
independent across wavelength bands (no cross-band coupling is modeled).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from physicaloptix.core import Field
from physicaloptix.speckle import SpeckleProcess


class Linearization(eqx.Module):
    """The (E_nom, G) product of ``linearize``.

    Monochromatic (no ``wavelengths_nm``): ``e_nom`` is ``(y, x)`` and ``G``
    is ``(m, y, x)``, per unit mode coefficient (an OPD basis in the same
    length unit as ``wavelength_nm``). Chromatic (``wavelengths_nm`` set):
    ``e_nom`` is ``(w, y, x)`` and ``G`` is ``(w, m, y, x)`` -- one column
    stack per wavelength band.

    Attributes:
        e_nom: Complex nominal focal field, shape ``(y, x)`` mono or
            ``(w, y, x)`` chromatic.
        G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``
            mono or ``(w, m, y, x)`` chromatic.
        wavelength_nm: The recorded design wavelength; per-band phase
            factors come from ``wavelengths_nm`` when chromatic.
        method: The method that actually ran (``auto`` resolves before this
            is recorded).
        kind: The basis kind the columns were built from (``"opd"`` or
            ``"amplitude"``).
        pixel_scale_lod: The output plane's pixel scale (lambda/D per pixel for
            a focal output) -- the grid ``e_nom`` / ``G`` are sampled on, so a
            speckle field built from this product carries its true plate scale.
        input_energy: Total energy of the input field at the plane the
            linearization was built from (``field.energy()``): the
            pre-coronagraph photometric reference that, with
            ``pixel_scale_lod``, converts intensity densities to per-pixel
            flux fractions. A scalar for a mono field, or one value per
            wavelength band (``field.energy()`` is per-band for a chromatic
            field) as a tuple.
        wavelengths_nm: Channel wavelengths, shape ``(w,)``, for a chromatic
            linearization; ``None`` for a mono one.
        dispersion: The resolved per-mode, per-wavelength complex factor
            table, shape ``(m, w)``, for a chromatic linearization (whether
            supplied explicitly or derived from ``basis.kind``); ``None``
            for a mono one.
    """

    e_nom: Array
    G: Array
    wavelength_nm: float = eqx.field(static=True)
    method: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    pixel_scale_lod: float = eqx.field(static=True)
    input_energy: float | tuple[float, ...] = eqx.field(static=True)
    wavelengths_nm: Array | None = None
    dispersion: Array | None = None

    @property
    def n_modes(self):
        """Number of sensitivity columns (correct in both mono and chromatic layout)."""
        return self.G.shape[-3]

    def to_speckle_process(
        self,
        *,
        per_mode_rms=None,
        knee_hz=None,
        decorr_hours=None,
        total_rms=None,
        **kwargs,
    ):
        """Package (E_nom, G) as a ``SpeckleProcess``.

        Either give the process parameters directly (``per_mode_rms`` +
        ``knee_hz``) or the decorrelation parameterization
        (``decorr_hours`` + ``total_rms``). The process inherits the recorded
        photometric primitives (``input_energy``, ``pixel_scale_lod``), so
        realized maps are per-pixel flux fractions with no further input;
        pass ``input_energy=...`` explicitly to re-reference (e.g. when a
        coronagraph mask was baked into the input field instead of living in
        the path).

        Args:
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
        kwargs.setdefault("input_energy", self.input_energy)
        if decorr_hours is not None:
            return SpeckleProcess.from_decorrelation(
                self.e_nom,
                self.G,
                decorr_hours=decorr_hours,
                total_rms=total_rms,
                **kwargs,
            )
        return SpeckleProcess(
            self.e_nom,
            self.G,
            per_mode_rms,
            knee_hz,
            **kwargs,
        )


def _phase_factor(wavelength_nm):
    return 1j * 2.0 * jnp.pi / wavelength_nm


def _replace_data(field, data):
    return Field(data=data, grid=field.grid, plane=field.plane, spectrum=field.spectrum)


def _factors(basis, wavelength_nm, wavelengths_nm, dispersion):
    """Per-mode complex factors: shape (m,) mono or (m, w) chromatic."""
    if wavelengths_nm is None:
        if dispersion is not None:
            raise ValueError("dispersion requires wavelengths_nm")
        scalar = _phase_factor(wavelength_nm) if basis.kind == "opd" else 1.0
        return jnp.full((basis.n_modes,), scalar, dtype=complex)
    wavelengths = jnp.asarray(wavelengths_nm, dtype=float)
    if dispersion is not None:
        d = jnp.asarray(dispersion, dtype=complex)
        if d.shape != (basis.n_modes, wavelengths.shape[0]):
            raise ValueError(
                f"dispersion shape {d.shape} must be (n_modes, n_wavelengths) "
                f"= ({basis.n_modes}, {wavelengths.shape[0]})"
            )
        return d
    if basis.kind == "opd":
        return jnp.broadcast_to(
            _phase_factor(wavelengths), (basis.n_modes, wavelengths.shape[0])
        )
    return jnp.ones((basis.n_modes, wavelengths.shape[0]), dtype=complex)


def perturbed_map(
    path, field, basis, wavelength_nm, *, wavelengths_nm=None, dispersion=None
):
    """The nonlinear map ``eps -> E_focal`` the linearization approximates.

    OPD modes perturb the phase (``E * exp(i 2 pi (B . eps) / lambda)``);
    amplitude modes perturb the field multiplicatively
    (``E * (1 + B . eps)``, fractional amplitude, achromatic). With
    ``wavelengths_nm`` set, both kinds use the exp-form
    ``E * exp(sum_k eps_k D_k(lambda) B_k)`` built from the per-mode
    dispersion factors (``D_k``: the default derived from ``basis.kind``, or
    the supplied ``dispersion`` table). For ``kind="amplitude"`` this differs
    from the mono ``E * (1 + B . eps)`` at finite ``eps`` (the exp-form is
    only exact to first order there); the first derivative at ``eps = 0`` --
    what ``linearize`` actually uses -- is identical.
    """
    if wavelengths_nm is not None:
        factors = _factors(basis, wavelength_nm, wavelengths_nm, dispersion)

    def run(eps):
        if wavelengths_nm is None:
            mode_map = jnp.tensordot(eps, basis.B, axes=1)
            if basis.kind == "opd":
                data = field.data * jnp.exp(_phase_factor(wavelength_nm) * mode_map)
            else:
                data = field.data * (1.0 + mode_map)
        else:
            mode_log = jnp.einsum("m,mw,myx->wyx", eps, factors, basis.B)
            data = field.data * jnp.exp(mode_log)
        out, _ = path.propagate(_replace_data(field, data))
        return out.data

    return run


def linearize(
    path,
    field,
    basis,
    *,
    wavelength_nm=None,
    method="auto",
    chunk_size=None,
    memory_budget_bytes=4 * 2**30,
    wavelengths_nm=None,
    dispersion=None,
):
    """Build the (E_nom, G) linearization of a path around ``field``.

    Args:
        path: The ``OpticalPath`` (or any object with ``propagate``); every
            stage must be linear in the field for the analytic method.
        field: The unperturbed input field; the OPD perturbation applies at
            this plane. Chromatic (``field.spectrum`` set) requires
            ``wavelengths_nm``; mono requires it to be absent.
        basis: An OPD ``ModeBasis`` in the same length unit as
            ``wavelength_nm`` (or an ``"amplitude"`` basis).
        wavelength_nm: Design wavelength for the phase factor. Required for
            a mono field; optional for a chromatic one, where it defaults to
            ``float(wavelengths_nm[0])`` and is then record-only (the
            per-band factors come from ``wavelengths_nm`` / ``dispersion``).
        method: ``"analytic"`` (default via ``"auto"``), ``"jvp"``, or
            ``"jacfwd"``.
        chunk_size: Modes per propagation batch for the analytic method;
            ``None`` batches all modes (subject to ``memory_budget_bytes``
            under ``"auto"``).
        memory_budget_bytes: When ``method="auto"``, the mode-stack size
            above which the analytic method streams chunks instead of
            batching everything.
        wavelengths_nm: Channel wavelengths, shape ``(w,)``, matching
            ``field.spectrum.wavelengths_nm``; set this to build a chromatic
            linearization.
        dispersion: Optional complex ``(m, w)`` table of per-mode factors
            evaluated at ``wavelengths_nm``; default derives from
            ``basis.kind`` (``i 2 pi / lambda_w`` for ``"opd"``, ``1`` for
            ``"amplitude"``). Only meaningful with ``wavelengths_nm`` set.

    Returns:
        A ``Linearization``.
    """
    if dispersion is not None and wavelengths_nm is None:
        raise ValueError("dispersion requires wavelengths_nm")
    if wavelengths_nm is not None:
        wavelengths_nm = jnp.asarray(wavelengths_nm, dtype=float)
        if field.spectrum is None:
            raise ValueError(
                "wavelengths_nm requires a chromatic field (with a spectrum)"
            )
        if not np.allclose(
            np.asarray(field.spectrum.wavelengths_nm), np.asarray(wavelengths_nm)
        ):
            raise ValueError("wavelengths_nm must match field.spectrum.wavelengths_nm")
        if wavelength_nm is None:
            wavelength_nm = float(wavelengths_nm[0])
    else:
        if field.spectrum is not None:
            raise ValueError(
                "linearize is monochromatic without wavelengths_nm: pass "
                "wavelengths_nm for a chromatic field or slice per band"
            )
        if wavelength_nm is None:
            raise ValueError(
                "wavelength_nm is required for a monochromatic linearization"
            )

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
        # Chunked in lockstep with B: a closed-over full-size factor table
        # would raise at chunked sizes.
        factors = _factors(basis, wavelength_nm, wavelengths_nm, dispersion)
        propagate_stack = jax.vmap(
            lambda data: path.propagate(_replace_data(field, data))[0].data
        )

        if wavelengths_nm is None:

            def columns(mode_chunk, factor_chunk):
                return propagate_stack(
                    factor_chunk[:, None, None] * mode_chunk * field.data
                )
        else:

            def columns(mode_chunk, factor_chunk):
                return propagate_stack(
                    factor_chunk[:, :, None, None]
                    * mode_chunk[:, None]
                    * field.data[None]
                )

        if chunk_size is None:
            g = columns(basis.B, factors)
        else:
            g = jnp.concatenate(
                [
                    columns(basis.B[s : s + chunk_size], factors[s : s + chunk_size])
                    for s in range(0, n_modes, chunk_size)
                ]
            )
    elif resolved == "jvp":
        run = perturbed_map(
            path,
            field,
            basis,
            wavelength_nm,
            wavelengths_nm=wavelengths_nm,
            dispersion=dispersion,
        )
        zero = jnp.zeros(n_modes)
        cols = [
            jax.jvp(run, (zero,), (jnp.zeros(n_modes).at[k].set(1.0),))[1]
            for k in range(n_modes)
        ]
        g = jnp.stack(cols)
    elif resolved == "jacfwd":
        run = perturbed_map(
            path,
            field,
            basis,
            wavelength_nm,
            wavelengths_nm=wavelengths_nm,
            dispersion=dispersion,
        )
        jacobian = jax.jacfwd(run)(jnp.zeros(n_modes))
        g = jnp.moveaxis(jacobian, -1, 0)
    else:
        raise ValueError(f"method must be auto/analytic/jvp/jacfwd, got {method!r}")

    # All three methods produce a leading mode axis; under the chromatic map
    # that is (m, w, y, x), normalized here to the documented (w, m, y, x).
    if wavelengths_nm is not None:
        g = jnp.moveaxis(g, 1, 0)

    if wavelengths_nm is None:
        input_energy = float(field.energy())
        resolved_dispersion = None
    else:
        input_energy = tuple(float(e) for e in field.energy())
        resolved_dispersion = _factors(basis, wavelength_nm, wavelengths_nm, dispersion)

    return Linearization(
        e_nom=e_nom_field.data,
        G=g,
        wavelength_nm=float(wavelength_nm),
        method=resolved,
        kind=basis.kind,
        pixel_scale_lod=float(e_nom_field.grid.dx),
        input_energy=input_energy,
        wavelengths_nm=wavelengths_nm,
        dispersion=resolved_dispersion,
    )


def linearity_residual(path, field, basis, linearization, eps):
    """Relative error of the linear model at coefficients ``eps``.

    ``|E(eps) - (E_nom + G eps)| / |E(eps)|`` over the focal plane -- the
    small-phase validity check (scales as ``eps^2``). Generalizes to a
    chromatic ``linearization`` by rebuilding the chromatic
    ``perturbed_map`` (from the recorded ``wavelengths_nm`` / ``dispersion``)
    and contracting ``G``'s mode axis, which is no longer the leading axis.
    """
    eps = jnp.asarray(eps)
    run = perturbed_map(
        path,
        field,
        basis,
        linearization.wavelength_nm,
        wavelengths_nm=linearization.wavelengths_nm,
        dispersion=linearization.dispersion,
    )
    exact = run(eps)
    if linearization.wavelengths_nm is None:
        model = linearization.e_nom + jnp.tensordot(eps, linearization.G, axes=1)
    else:
        model = linearization.e_nom + jnp.einsum("m,wmyx->wyx", eps, linearization.G)
    return float(jnp.linalg.norm(exact - model) / jnp.linalg.norm(exact))

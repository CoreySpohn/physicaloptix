"""linearize(): one entry point for the (E_nom, G) linearization.

The path is linear in the field, so the first-order focal response to an OPD
mode is exact: ``G_k = i (2 pi / lambda) Path(B_k * E_in)`` (the analytic
method, cheapest and the default). ``jvp`` and ``jacfwd`` differentiate the
full nonlinear map ``eps -> Path(E_in * exp(i 2 pi (B . eps) / lambda))`` at
``eps = 0`` and exist as autodiff cross-checks; all three agree to roundoff
on a linear chain.

Amplitude bases (``kind="amplitude"``) linearize the fractional-amplitude
map ``E * (1 + B . eps)``: their columns are ``Path(B_k * E_in)`` with no
phase factor, so G is achromatic for them. A single-plane focal response
carries the basis kind as its wavelength law. A chromatic response carries
``wavelength_law=None`` because each node already includes its dispersion.

Memory policy: mode stacks at pupil resolution are the wall (a dense basis at
2048^2 in complex128 is tens of MB per mode), so the analytic method streams
mode chunks through a vmapped propagation when the full stack would exceed
``memory_budget_bytes``; ``jvp`` is a host-side loop, memory-flat by
construction.

Chromatic linearization: passing ``wavelengths_nm`` (matching
``field.spectrum.nodes_nm[0]``) builds a per-band stack, ``e_nom``
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
from hwoutils.constants import two_pi
from hwoutils.numerics import same
from jaxtyping import Array
from optixstuff import FocalResponse
from optixstuff.grids import PixelGrid

from physicaloptix.command import command_response
from physicaloptix.core import Field
from physicaloptix.speckle import SpeckleProcess


class Linearization(eqx.Module):
    """The (E_nom, G) product of ``linearize``.

    Monochromatic (no ``wavelengths_nm``): ``e_nom`` is ``(y, x)`` and ``G``
    is ``(m, y, x)``, per unit mode coefficient (an OPD basis in the same
    length unit as ``wavelength_nm``). Chromatic (``wavelengths_nm`` set):
    ``e_nom`` is ``(w, y, x)`` and ``G`` is ``(w, m, y, x)`` -- one column
    stack per wavelength band. A mono product needs a known basis kind to
    supply the focal response's wavelength law. A chromatic product supplies
    a tabulated response with no additional scaling.

    Attributes:
        e_nom: Complex nominal focal field, shape ``(y, x)`` mono or
            ``(w, y, x)`` chromatic, in density-amplitude units.
        G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``
            mono or ``(w, m, y, x)`` chromatic.
        wavelength_nm: The recorded design wavelength; per-band phase
            factors come from ``wavelengths_nm`` when chromatic.
        method: The method that actually ran (``auto`` resolves before this
            is recorded).
        kind: The basis kind the columns were built from (``"opd"`` or
            ``"amplitude"``).
        grid: The actual output grid, including its Frame, on which
            ``e_nom`` and ``G`` are sampled. A
            speckle field built from this product carries its true plate scale.
        input_energy: Total energy of the input field at the plane the
            linearization was built from (``field.energy()``): the
            pre-coronagraph photometric reference that, with
            ``grid``, converts intensity densities to per-pixel
            flux fractions. A scalar for a mono field, or one value per
            wavelength band (``field.energy()`` is per-band for a chromatic
            field) as a tuple.
        wavelengths_nm: Channel wavelengths, shape ``(w,)``, for a chromatic
            linearization; ``None`` for a mono one.
        dispersion: The resolved per-mode, per-wavelength complex factor
            table, shape ``(m, w)``, for a chromatic linearization (whether
            supplied explicitly or derived from ``basis.kind``); ``None``
            for a mono one.
        perturbation_stage: The name of the stage whose optical command
            was differentiated (the plane-aware route), or ``None`` for the
            input-plane injection route.
    """

    e_nom: Array
    G: Array
    wavelength_nm: float = eqx.field(static=True)
    method: str = eqx.field(static=True)
    kind: str | None = eqx.field(static=True)
    grid: PixelGrid
    input_energy: float | tuple[float, ...] = eqx.field(static=True)
    wavelengths_nm: Array | None = None
    dispersion: Array | None = None
    perturbation_stage: str | None = eqx.field(static=True, default=None)

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
        photometric primitives (``input_energy``, ``grid``) and,
        for a chromatic linearization, the recorded ``wavelengths_nm``, so
        realized maps are per-pixel flux fractions (per channel, chromatic)
        with no further input; pass ``input_energy=...`` explicitly to
        re-reference (e.g. when a coronagraph mask was baked into the input
        field instead of living in the path).

        Args:
            per_mode_rms: Per-mode rms drift (with ``knee_hz``).
            knee_hz: Temporal PSD knee frequency (with ``per_mode_rms``).
            decorr_hours: Decorrelation time (with ``total_rms``).
            total_rms: Total WFE budget, split evenly over modes.
            **kwargs: Forwarded to ``SpeckleProcess``.

        Returns:
            The parameter object whose ``draw(key)`` yields
            ``AnalyticSpeckleField`` realizations.

        Raises:
            ValueError: A monochromatic product lacks a known basis kind.
        """
        grid = kwargs.pop("grid", self.grid)
        reference_wavelength_nm = kwargs.pop(
            "reference_wavelength_nm",
            self.wavelength_nm
            if self.wavelengths_nm is None
            else grid.frame.reference_wavelength_nm,
        )
        law = None
        if self.wavelengths_nm is None:
            if self.kind not in ("opd", "amplitude"):
                raise ValueError(
                    f"unknown monochromatic basis kind {self.kind!r}; linearize "
                    "with wavelengths_nm to obtain a tabulated response"
                )
            law = self.kind
        response = FocalResponse(
            self.e_nom,
            self.G,
            grid,
            jnp.asarray(
                kwargs.pop("input_energy", self.input_energy), dtype=self.G.real.dtype
            ),
            reference_wavelength_nm,
            wavelength_law=law,
        )
        kwargs.setdefault("wavelengths_nm", self.wavelengths_nm)
        if decorr_hours is not None:
            return SpeckleProcess.from_decorrelation(
                response, decorr_hours=decorr_hours, total_rms=total_rms, **kwargs
            )
        return SpeckleProcess(response, per_mode_rms, knee_hz, **kwargs)


def _phase_factor(wavelength_nm):
    return 1j * two_pi / wavelength_nm


def _replace_data(field, data):
    return Field(data=data, grid=field.grid, plane=field.plane, spectrum=field.spectrum)


def _factors(basis, wavelength_nm, wavelengths_nm, dispersion, field_data):
    """Per-mode complex factors: shape (m,) mono or (m, w) chromatic."""
    if wavelengths_nm is None:
        if dispersion is not None:
            raise ValueError("dispersion requires wavelengths_nm")
        scalar = _phase_factor(wavelength_nm) if basis.kind == "opd" else 1.0
        return jnp.full((basis.n_modes,), scalar, dtype=field_data.dtype)
    wavelengths = jnp.asarray(wavelengths_nm, dtype=field_data.real.dtype)
    if dispersion is not None:
        d = jnp.asarray(dispersion, dtype=field_data.dtype)
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
    return jnp.ones((basis.n_modes, wavelengths.shape[0]), dtype=field_data.dtype)


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

    Args:
        path: Optical path that propagates the perturbed field to the focal plane.
        field: Unperturbed input field.
        basis: Mode basis; its kind selects the phase or amplitude perturbation.
        wavelength_nm: Design wavelength in nanometers for the phase factor of a
            monochromatic map.
        wavelengths_nm: Channel wavelengths in nanometers, shape (w,), for the chromatic
            exp-form; None for the monochromatic map.
        dispersion: Optional complex (m, w) table of per-mode factors at wavelengths_nm;
            the default derives from basis.kind.

    Returns:
        Function of the coefficient vector eps, shape (m,), that returns the complex
        propagated field data.
    """
    if wavelengths_nm is not None:
        factors = _factors(basis, wavelength_nm, wavelengths_nm, dispersion, field.data)

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
    basis=None,
    *,
    wavelength_nm=None,
    method="auto",
    chunk_size=None,
    memory_budget_bytes=4 * 2**30,
    wavelengths_nm=None,
    dispersion=None,
    perturbation_stage=None,
):
    """Build the (E_nom, G) linearization of a path around ``field``.

    ``wavelengths_nm`` must be passed explicitly rather than inferred from
    ``field.spectrum``: it is a deliberate opt-in, so a caller cannot
    silently get a 4-D chromatic ``G`` back from what looks like a
    monochromatic call.

    Two ways to place the perturbed modes: ``basis`` injects them at the
    INPUT plane ``field`` is given at; ``perturbation_stage`` instead
    differentiates with respect to a named stage's current command
    (an additive delta around its current values, propagated through the
    remainder of the path). The stage route is correct for modes living at
    any plane, at the cost of requiring ``method="jvp"`` or ``"jacfwd"``
    (the analytic shortcut assumes an input-plane injection).

    Args:
        path: The ``OpticalPath`` (or any object with ``propagate`` and,
            for ``perturbation_stage``, ``stages``); every stage must be
            linear in the field for the analytic method.
        field: The unperturbed input field; the OPD perturbation applies at
            this plane (or, with ``perturbation_stage``, is propagated from
            it). Chromatic (``field.spectrum`` set) requires
            ``wavelengths_nm``; mono requires it to be absent.
        basis: An OPD ``ModeBasis`` in the same length unit as
            ``wavelength_nm`` (or an ``"amplitude"`` basis). Required
            unless ``perturbation_stage`` is given, and mutually exclusive
            with it.
        wavelength_nm: Design wavelength for the phase factor. Required for
            a mono field; optional for a chromatic one, where it defaults to
            ``float(wavelengths_nm[0])`` and is then record-only (the
            per-band factors come from ``wavelengths_nm`` / ``dispersion``).
        method: ``"analytic"`` (default via ``"auto"``), ``"jvp"``, or
            ``"jacfwd"``. ``perturbation_stage`` requires ``"jvp"`` or
            ``"jacfwd"``.
        chunk_size: Modes per propagation batch for the analytic method;
            ``None`` batches all modes (subject to ``memory_budget_bytes``
            under ``"auto"``).
        memory_budget_bytes: When ``method="auto"``, the mode-stack size
            above which the analytic method streams chunks instead of
            batching everything.
        wavelengths_nm: Channel wavelengths, shape ``(w,)``, matching
            ``field.spectrum.nodes_nm[0]``; set this to build a chromatic
            linearization.
        dispersion: Optional complex ``(m, w)`` table of per-mode factors
            evaluated at ``wavelengths_nm``; default derives from
            ``basis.kind`` (``i 2 pi / lambda_w`` for ``"opd"``, ``1`` for
            ``"amplitude"``). Only meaningful with ``wavelengths_nm`` set.
            Not used by the ``perturbation_stage`` route (the stage's own
            per-wavelength physics runs unchanged).
        perturbation_stage: Name of a stage in ``path.stages`` whose command
            to differentiate, instead of injecting
            ``basis`` at the input plane. Mutually exclusive with ``basis``;
            ``method="analytic"``/``"auto"`` raise ``NotImplementedError``
            for this route.

    Returns:
        A ``Linearization``.
    """
    if perturbation_stage is not None:
        if basis is not None:
            raise ValueError(
                "pass either basis (input-plane injection) or "
                "perturbation_stage (a stage's own modes), not both"
            )
        if method in ("auto", "analytic"):
            raise NotImplementedError(
                "analytic remaining-path injection is not built; use "
                "method='jacfwd' (or 'jvp') with perturbation_stage"
            )
    elif basis is None:
        raise ValueError("basis is required without perturbation_stage")

    if dispersion is not None and wavelengths_nm is None:
        raise ValueError("dispersion requires wavelengths_nm")
    if wavelengths_nm is not None:
        if field.spectrum is None:
            raise ValueError(
                "wavelengths_nm requires a chromatic field (with a spectrum)"
            )
        field_wavelengths = np.asarray(field.spectrum.nodes_nm[0])
        query_wavelengths = np.asarray(wavelengths_nm)
        if field_wavelengths.shape != query_wavelengths.shape or not np.all(
            same(field_wavelengths, query_wavelengths)
        ):
            raise ValueError("wavelengths_nm must match field.spectrum.nodes_nm[0]")
        wavelengths_nm = jnp.asarray(wavelengths_nm, dtype=field.data.real.dtype)
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

    if method not in ("auto", "analytic", "jvp", "jacfwd"):
        raise ValueError(f"method must be auto/analytic/jvp/jacfwd, got {method!r}")
    resolved = method
    if perturbation_stage is not None:
        response = command_response(path, field, perturbation_stage)
        e_nom_field = response._propagate(response.element.command)
        kind = response.element.command_kind
        if resolved == "jvp" and chunk_size is None:
            chunk_size = 1
        g = response.materialize(chunk_size)
    else:
        e_nom_field, _ = path.propagate(field)
        kind = basis.kind

    if method == "auto":
        resolved = "analytic"
        estimate = basis.n_modes * field.data.size * 16
        if chunk_size is None and estimate > memory_budget_bytes:
            per_mode = field.data.size * 16
            chunk_size = max(1, int(memory_budget_bytes // per_mode))

    if resolved == "analytic":
        # OPD columns carry the phase factor; amplitude columns are the
        # propagated fractional-amplitude modes themselves (achromatic).
        # Chunked in lockstep with B: a closed-over full-size factor table
        # would raise at chunked sizes.
        factors = _factors(basis, wavelength_nm, wavelengths_nm, dispersion, field.data)
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
                    for s in range(0, basis.n_modes, chunk_size)
                ]
            )
    elif perturbation_stage is None:
        run = perturbed_map(
            path,
            field,
            basis,
            wavelength_nm,
            wavelengths_nm=wavelengths_nm,
            dispersion=dispersion,
        )
        n_modes = basis.n_modes
        zero = jnp.zeros(n_modes, dtype=field.data.real.dtype)
        if resolved == "jvp":
            g = jnp.stack(
                [
                    jax.jvp(run, (zero,), (zero.at[k].set(1.0),))[1]
                    for k in range(n_modes)
                ]
            )
        else:
            g = jnp.moveaxis(jax.jacfwd(run)(zero), -1, 0)

    # All three methods produce a leading mode axis; under the chromatic map
    # that is (m, w, y, x), normalized here to the documented (w, m, y, x).
    if wavelengths_nm is not None:
        g = jnp.moveaxis(g, 1, 0)

    if wavelengths_nm is None:
        input_energy = float(field.energy())
        resolved_dispersion = None
    else:
        input_energy = tuple(float(e) for e in field.energy())
        # The perturbation_stage route ignores dispersion (the stage's own
        # per-wavelength physics runs unchanged), so there is no derived
        # table that describes what was actually applied; record None
        # rather than a plausible-looking but unused value.
        resolved_dispersion = (
            None
            if perturbation_stage is not None
            else _factors(basis, wavelength_nm, wavelengths_nm, dispersion, field.data)
        )

    return Linearization(
        e_nom=e_nom_field.data,
        G=g,
        wavelength_nm=float(wavelength_nm),
        method=resolved,
        kind=kind,
        grid=e_nom_field.grid,
        input_energy=input_energy,
        wavelengths_nm=wavelengths_nm,
        dispersion=resolved_dispersion,
        perturbation_stage=perturbation_stage,
    )


def linearize_stages(
    path,
    field,
    stage_names,
    *,
    wavelength_nm=None,
    method="jacfwd",
    wavelengths_nm=None,
):
    """Concatenate per-stage plane-aware linearizations along the mode axis.

    Builds one ``perturbation_stage`` :func:`linearize` per name in
    ``stage_names`` and stacks their ``G`` columns, so a caller can budget a
    sensitivity matrix across several stages (e.g. two mirrors' drift bases)
    without re-deriving the column layout by hand.

    Args:
        path: The ``OpticalPath``.
        field: The unperturbed input field.
        stage_names: Names of stages in ``path.stages`` to linearize, in the
            order their columns should appear.
        wavelength_nm: Forwarded to :func:`linearize`.
        method: Forwarded to :func:`linearize`; must be ``"jvp"`` or
            ``"jacfwd"`` (the ``perturbation_stage`` route has no analytic
            method yet).
        wavelengths_nm: Forwarded to :func:`linearize`; set for a chromatic
            linearization.

    Returns:
        ``(Linearization, {stage_name: slice})``: the concatenated
        linearization (``e_nom`` and the recorded normalization primitives
        are shared across stages by construction, so those are checked
        rather than merged) and each stage's column slice into ``G``'s mode
        axis. The merged ``Linearization`` records
        ``perturbation_stage=",".join(stage_names)`` (a truthy marker so
        ``linearity_residual``'s input-plane-only guard fires on it too --
        every ``linearize_stages`` output is stage-route by construction --
        and honest provenance of which stages contributed) and
        ``dispersion=None`` (no single per-mode table is coherent once
        stages with different mode counts / kinds are concatenated). Stages
        of different kinds merge only on wavelength nodes, where the merged
        ``kind`` is ``None``; a monochromatic merge requires one kind.
    """
    parts = [
        linearize(
            path,
            field,
            wavelength_nm=wavelength_nm,
            method=method,
            wavelengths_nm=wavelengths_nm,
            perturbation_stage=name,
        )
        for name in stage_names
    ]
    first = parts[0]
    for part in parts[1:]:
        if part.method != first.method:
            raise ValueError("linearize_stages: stage parts disagree on method")
        # A tabulated product carries each column's own wavelength response,
        # so stages of different kinds merge on wavelength nodes only.
        if part.kind != first.kind and wavelengths_nm is None:
            raise ValueError("linearize_stages: stage parts disagree on kind")
        if part.wavelength_nm != first.wavelength_nm:
            raise ValueError("linearize_stages: stage parts disagree on wavelength_nm")
        if part.grid != first.grid:
            raise ValueError("linearize_stages: stage parts disagree on grid")
        if part.input_energy != first.input_energy:
            raise ValueError("linearize_stages: stage parts disagree on input_energy")
        if not np.array_equal(np.asarray(part.e_nom), np.asarray(first.e_nom)):
            raise ValueError("linearize_stages: stage parts disagree on e_nom")
        if first.wavelengths_nm is not None and not np.array_equal(
            np.asarray(part.wavelengths_nm), np.asarray(first.wavelengths_nm)
        ):
            raise ValueError("linearize_stages: stage parts disagree on wavelengths_nm")

    axis = 0 if wavelengths_nm is None else 1
    g = jnp.concatenate([part.G for part in parts], axis=axis)
    slices, start = {}, 0
    for name, part in zip(stage_names, parts, strict=True):
        m = part.G.shape[axis]
        slices[name] = slice(start, start + m)
        start += m

    merged = Linearization(
        e_nom=first.e_nom,
        G=g,
        wavelength_nm=first.wavelength_nm,
        method=first.method,
        kind=first.kind if all(part.kind == first.kind for part in parts) else None,
        grid=first.grid,
        input_energy=first.input_energy,
        wavelengths_nm=first.wavelengths_nm,
        dispersion=None,
        perturbation_stage=",".join(stage_names),
    )
    return merged, slices


def linearity_residual(path, field, basis, linearization, eps):
    """Relative error of the linear model at coefficients ``eps``.

    ``|E(eps) - (E_nom + G eps)| / |E(eps)|`` over the focal plane -- the
    small-phase validity check (scales as ``eps^2``). Generalizes to a
    chromatic ``linearization`` by rebuilding the chromatic
    ``perturbed_map`` (from the recorded ``wavelengths_nm`` / ``dispersion``)
    and contracting ``G``'s mode axis, which is no longer the leading axis.

    Args:
        path: Optical path used to build the linearization.
        field: Unperturbed input field the linearization was built around.
        basis: Mode basis the linearization was built with.
        linearization: Linearization to check; it must be an input-plane one, with no
            perturbation stage.
        eps: Coefficient vector, shape (m,), at which the linear model is compared with
            the nonlinear map.

    Returns:
        Relative error, a dimensionless float: the norm of the model minus exact
        difference divided by the norm of the exact field.
    """
    if linearization.perturbation_stage is not None:
        raise NotImplementedError(
            "linearity_residual checks input-plane linearizations; for a "
            "perturbation_stage route, cross-check method='jvp' vs "
            "method='jacfwd' or use a finite difference"
        )
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

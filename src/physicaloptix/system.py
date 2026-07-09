"""Multi-path optics: the two-port beamsplitter (and, above it, the fork).

A linear ``OpticalPath`` is a single fold, so a fork -- a wavefront-sensor arm,
a dual-channel science split, a rejected-light low-order-sensor feed -- lives
one level above it. ``BeamSplitter`` divides one field between two output
ports while conserving energy; the path guard rejects it as a ``Stage`` (it is
multi-output), so it belongs to the forked container, not the fold.

Port conventions: ``"transmit"`` and ``"reflect"``. A lossless split satisfies
``|t|^2 + |r|^2 = 1`` pointwise (checked at construction against the
``on_violation`` policy); the symmetric lossless convention places the two
ports in phase quadrature, which ``energy()`` bakes in (``r = i sqrt(R)``).

The physical reject port is only shipped where it is physical: a BINARY mask
(the Babinet complement -- an occulter's reject is the occulted core) via
``from_mask``. An absorbing grey apodizer has no coherent reject port (the
blocked light is gone), so ``from_mask`` refuses a non-binary mask rather than
inventing a fictitious sensing field.
"""

import warnings

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.path import OpticalPath, Stage, _planes

_ENERGY_TOL = 1e-10
PORTS = ("transmit", "reflect")


class BeamSplitter(eqx.Module):
    """A two-output-port energy split at a single plane.

    A bare plane-carrying module (like the vortex and the Zernike sensor), not
    an ``Element``: ``__call__`` returns a dict of ports, so it cannot be a
    stage of the linear fold. It routes energy in place -- same plane, same
    grid, no propagation.

    Attributes:
        t: Amplitude transmission -- scalar ``()``, spatial ``(y, x)``, or
            spectral-spatial ``(nlam, y, x)``. ``None`` for a dichroic.
        r: Amplitude reflection, same rank rules. ``None`` for a dichroic.
        wavelengths_nm: Dichroic table wavelengths (``None`` otherwise).
        transmit_table: Dichroic intensity-transmittance table, interpolated
            against the incident field's spectrum at call time.
        grid: The grid the splitter sits on.
        plane: The plane the splitter sits in.
        on_violation: Energy-gate policy: ``"raise"``, ``"warn"``, ``"record"``.
    """

    t: Array | None
    r: Array | None
    wavelengths_nm: Array | None
    transmit_table: Array | None
    grid: Grid
    plane: PlaneKind = eqx.field(static=True)
    on_violation: str = eqx.field(static=True, default="raise")
    multi_output: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        *,
        t=None,
        r=None,
        wavelengths_nm=None,
        transmit_table=None,
        grid,
        plane,
        on_violation="raise",
    ):
        """Build a splitter from concrete amplitudes or a dichroic table.

        Args:
            t: Concrete amplitude transmission (scalar, ``(y, x)``, or
                ``(nlam, y, x)``); mutually exclusive with the table.
            r: Concrete amplitude reflection, same ranks.
            wavelengths_nm: Dichroic table wavelengths.
            transmit_table: Dichroic intensity transmittance per table
                wavelength (amplitudes derived, energy-conserving).
            grid: The grid the splitter sits on.
            plane: The plane the splitter sits in.
            on_violation: Energy-gate policy (``raise``/``warn``/``record``).
        """
        if on_violation not in ("raise", "warn", "record"):
            raise ValueError(
                f"on_violation must be 'raise', 'warn', or 'record', "
                f"got {on_violation!r}"
            )
        tabled = wavelengths_nm is not None or transmit_table is not None
        concrete = t is not None or r is not None
        if tabled == concrete:
            raise ValueError(
                "provide either concrete (t, r) or a dichroic "
                "(wavelengths_nm, transmit_table), not both or neither"
            )
        if concrete:
            for name, amp in (("t", t), ("r", r)):
                if amp is None:
                    raise ValueError(f"missing amplitude {name!r}")
                if jnp.ndim(amp) not in (0, 2, 3):
                    raise ValueError(
                        f"{name} has rank {jnp.ndim(amp)}; a splitter "
                        "amplitude must be scalar (), spatial (y, x), or "
                        "spectral-spatial (nlam, y, x) -- a bare (nlam,) is "
                        "ambiguous"
                    )
                if jnp.ndim(amp) >= 2 and jnp.shape(amp)[-2:] != (
                    grid.npix,
                    grid.npix,
                ):
                    raise ValueError(
                        f"{name} shape {jnp.shape(amp)} does not match grid "
                        f"({grid.npix}, {grid.npix})"
                    )
            t = jnp.asarray(t)
            r = jnp.asarray(r)
        else:
            wavelengths_nm = jnp.asarray(wavelengths_nm)
            transmit_table = jnp.asarray(transmit_table)
            if wavelengths_nm.ndim != 1 or wavelengths_nm.shape != (
                transmit_table.shape[0],
            ):
                raise ValueError(
                    "dichroic table must be 1D with matching lengths, got "
                    f"{wavelengths_nm.shape} and {transmit_table.shape}"
                )
        self.t = t
        self.r = r
        self.wavelengths_nm = wavelengths_nm
        self.transmit_table = transmit_table
        self.grid = grid
        self.plane = plane
        self.on_violation = on_violation
        self.multi_output = True
        # Host-side energy gate on concrete amplitudes (a dichroic conserves
        # by construction: amplitudes are derived from one transmittance).
        residual = float(self.energy_residual)
        if residual > _ENERGY_TOL:
            message = (
                f"BeamSplitter energy residual max||t|^2+|r|^2 - 1| = "
                f"{residual:.3e} exceeds {_ENERGY_TOL:.0e}"
            )
            if on_violation == "raise":
                raise ValueError(message)
            if on_violation == "warn":
                warnings.warn(message, stacklevel=2)

    @property
    def energy_residual(self):
        """Max pointwise deviation of ``|t|^2 + |r|^2`` from one at build."""
        if self.t is None:
            return 0.0  # dichroic: r is derived from t, conserving exactly
        total = np.abs(np.asarray(self.t)) ** 2 + np.abs(np.asarray(self.r)) ** 2
        return float(np.max(np.abs(total - 1.0)))

    @classmethod
    def energy(cls, reflectance, *, grid, plane, on_violation="raise"):
        """A grey (achromatic) energy split sending ``reflectance`` aside.

        Uses the symmetric lossless convention ``t = sqrt(1 - R)``,
        ``r = i sqrt(R)`` (the two ports in quadrature), so a future
        recombination is phased consistently.

        Args:
            reflectance: Fraction of the intensity sent to the reflect port.
            grid: The grid the splitter sits on.
            plane: The plane the splitter sits in.
            on_violation: Energy-gate policy.

        Returns:
            A ready ``BeamSplitter``.
        """
        reflectance = float(reflectance)
        if not 0.0 <= reflectance <= 1.0:
            raise ValueError(f"reflectance must be in [0, 1], got {reflectance}")
        return cls(
            t=jnp.asarray(np.sqrt(1.0 - reflectance), dtype=complex),
            r=jnp.asarray(1j * np.sqrt(reflectance)),
            grid=grid,
            plane=plane,
            on_violation=on_violation,
        )

    @classmethod
    def from_mask(cls, mask, *, grid, plane, on_violation="raise"):
        """A binary-mask split: ``transmit = mask``, ``reflect = 1 - mask``.

        The transmit port is bit-identical to ``SampledOptic(mask)``, so the
        science channel is unchanged; the reflect port is the exact Babinet
        complement (an occulting mask's reject is the occulted core -- the
        low-order-sensor feed). A non-binary (grey) mask is refused: an
        absorbing apodizer has no coherent reject port.

        Args:
            mask: Binary transmission mask, shape ``(npix, npix)``.
            grid: The grid the mask is sampled on.
            plane: The plane the mask sits in.
            on_violation: Energy-gate policy.

        Returns:
            A ready ``BeamSplitter``.
        """
        arr = np.asarray(mask)
        if not np.all((np.abs(arr) < 1e-12) | (np.abs(arr - 1.0) < 1e-12)):
            raise ValueError(
                "from_mask requires a binary mask: an absorbing grey "
                "apodizer has no coherent reject port (model it as a "
                "through-path SampledOptic instead)"
            )
        mask = jnp.asarray(mask)
        return cls(
            t=mask,
            r=1.0 - mask,
            grid=grid,
            plane=plane,
            on_violation=on_violation,
        )

    @classmethod
    def dichroic(cls, *, wavelengths_nm, transmittance, grid, plane):
        """A color split: intensity transmittance interpolated per wavelength.

        Amplitudes are derived at call time against the incident field's
        spectrum (``t = sqrt(T)``, ``r = sqrt(1 - T)``), so the split is
        energy-conserving by construction and always registered to the
        actual band.

        Args:
            wavelengths_nm: Table wavelengths (1D, ascending).
            transmittance: Intensity transmittance in [0, 1] per table point.
            grid: The grid the splitter sits on.
            plane: The plane the splitter sits in.

        Returns:
            A ready ``BeamSplitter``.
        """
        return cls(
            wavelengths_nm=wavelengths_nm,
            transmit_table=jnp.clip(jnp.asarray(transmittance), 0.0, 1.0),
            grid=grid,
            plane=plane,
        )

    def _amplitudes(self, field):
        """The (t, r) amplitudes to apply to this particular field."""
        if self.t is not None:
            if self.t.ndim == 3 and (
                field.spectrum is None or self.t.shape[0] != len(field.spectrum)
            ):
                raise ValueError(
                    "a spectral-spatial (nlam, y, x) splitter needs a "
                    "chromatic field with matching nlam"
                )
            return self.t, self.r
        if field.spectrum is None:
            raise ValueError(
                "a dichroic BeamSplitter needs a chromatic field (a spectrum "
                "defines the colors to route)"
            )
        transmit = jnp.interp(
            field.spectrum.wavelengths_nm,
            self.wavelengths_nm,
            self.transmit_table,
        )[:, jnp.newaxis, jnp.newaxis]
        return jnp.sqrt(transmit), jnp.sqrt(1.0 - transmit)

    def __call__(self, field):
        """Split a field into its two ports.

        Args:
            field: The incident field (validated against plane and grid).

        Returns:
            ``{"transmit": Field, "reflect": Field}`` on the same plane/grid.
        """
        validate_field(field, plane=self.plane, grid=self.grid, context="BeamSplitter")
        t, r = self._amplitudes(field)
        return {
            "transmit": Field(
                data=field.data * t,
                grid=field.grid,
                plane=field.plane,
                spectrum=field.spectrum,
            ),
            "reflect": Field(
                data=field.data * r,
                grid=field.grid,
                plane=field.plane,
                spectrum=field.spectrum,
            ),
        }


class Branch(eqx.Module):
    """One arm of a forked system: a port binding plus its own downstream path.

    Attributes:
        name: The arm's name (tap namespace and output key).
        port: Which splitter port feeds it (``"transmit"`` or ``"reflect"``).
        path: The arm's own ``OpticalPath`` (its DMs, masks, and non-common
            aberrations). An empty path is a bare detector arm at the split
            plane.
    """

    name: str = eqx.field(static=True)
    port: str = eqx.field(static=True)
    path: OpticalPath


class SplitterPort(eqx.Module):
    """A single-output view of one splitter port (the flattening shim).

    Wraps ``(split, port)`` as a plain ``Field -> Field`` operator so a
    trunk + one branch can be composed into an ordinary ``OpticalPath``
    (``OpticalSystem.as_channel_path``). The sibling port's field is simply
    not computed.
    """

    split: BeamSplitter
    port: str = eqx.field(static=True)

    @property
    def plane_in(self):
        """The splitter's plane (a port selection does not re-propagate)."""
        return self.split.plane

    @property
    def plane_out(self):
        """Same plane out: energy routing only."""
        return self.split.plane

    def __call__(self, field):
        """Apply the splitter and keep this port."""
        return self.split(field)[self.port]


class OpticalSystem(eqx.Module):
    """A forked optical layout: one shared trunk feeding named branches.

    The trunk is propagated ONCE per call; the splitter divides its output
    among the branches (each bound to one port), and every branch propagates
    its own downstream path. Because the whole system is one pytree, a shared
    deformable mirror lives in the TRUNK as a single leaf -- one command, one
    gradient, seen by every branch. (Do not place one mirror object into two
    branch paths: pytrees are trees, not DAGs, and the copies silently
    decouple under jit.)

    Attributes:
        trunk: The shared upstream ``OpticalPath`` (may be empty for a pure
            entrance split).
        split: The ``BeamSplitter`` at the fork.
        branches: The arms, each a ``Branch`` bound to a splitter port.
    """

    trunk: OpticalPath
    split: BeamSplitter
    branches: tuple

    def __check_init__(self):
        """Reject bad names/ports and plane discontinuities at build."""
        names = [branch.name for branch in self.branches]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate branch names in {names}")
        if "trunk" in names:
            raise ValueError("'trunk' is reserved; rename the branch")
        for branch in self.branches:
            if branch.port not in PORTS:
                raise ValueError(
                    f"branch '{branch.name}' binds unknown port "
                    f"{branch.port!r}; ports are {PORTS}"
                )
        if self.trunk.stages:
            _, trunk_out = _planes(self.trunk.stages[-1].op)
            if trunk_out is not self.split.plane:
                raise ValueError(
                    f"trunk ends in the {trunk_out.value} plane but the "
                    f"splitter sits in the {self.split.plane.value} plane"
                )
        for branch in self.branches:
            if branch.path.stages:
                first_in, _ = _planes(branch.path.stages[0].op)
                if first_in is not self.split.plane:
                    raise ValueError(
                        f"branch '{branch.name}' starts in the "
                        f"{first_in.value} plane but the splitter feeds the "
                        f"{self.split.plane.value} plane"
                    )

    def _parse_taps(self, taps):
        """Split namespaced tap names into per-scope tuples."""
        trunk_names = {stage.name for stage in self.trunk.stages}
        branch_names = {
            branch.name: {stage.name for stage in branch.path.stages}
            for branch in self.branches
        }
        trunk_taps, branch_taps = [], {name: [] for name in branch_names}
        for tap in taps:
            scope, _, stage = tap.partition("/")
            if scope == "trunk" and stage in trunk_names:
                trunk_taps.append(stage)
            elif scope in branch_names and stage in branch_names[scope]:
                branch_taps[scope].append(stage)
            else:
                known = sorted(
                    [f"trunk/{s}" for s in trunk_names]
                    + [f"{b}/{s}" for b, stages in branch_names.items() for s in stages]
                )
                raise ValueError(f"unknown tap {tap!r}; taps are {known}")
        return tuple(trunk_taps), {k: tuple(v) for k, v in branch_taps.items()}

    def as_channel_path(self, name):
        """Flatten the trunk plus one branch into a plain ``OpticalPath``.

        The returned path is ``trunk stages + a single-output SplitterPort
        shim + the branch's stages``, so every existing single-path tool
        (``linearize``, a control loop, ``PathCoronagraph``) runs on a channel
        verbatim. NOTE: flattening DUPLICATES the trunk pytree per channel, so
        use a flattened path for per-channel work only; joint control of a
        shared trunk mirror across channels must go through this container
        (one trunk leaf), never through several flattened copies.

        Args:
            name: The branch to flatten.

        Returns:
            ``(path, index_map)``: the flattened ``OpticalPath`` and a dict
            mapping every stage name to its index in it (the port shim is
            ``"<name>_port"``).
        """
        by_name = {branch.name: branch for branch in self.branches}
        if name not in by_name:
            raise ValueError(f"unknown branch {name!r}; branches are {sorted(by_name)}")
        branch = by_name[name]
        port_stage = Stage(
            f"{name}_port", SplitterPort(split=self.split, port=branch.port)
        )
        stages = (*self.trunk.stages, port_stage, *branch.path.stages)
        path = OpticalPath(stages=stages)
        index_map = {stage.name: i for i, stage in enumerate(stages)}
        return path, index_map

    def propagate(self, field, *, taps=()):
        """Propagate the trunk once, split, and run every branch.

        Args:
            field: The entrance field (must match the trunk's first plane).
            taps: Static tuple of namespaced stage names to record --
                ``"trunk/<stage>"`` or ``"<branch>/<stage>"``.

        Returns:
            ``(outputs, tapped)``: a dict mapping each branch name to its
            output field, and the namespaced tapped fields.
        """
        trunk_taps, branch_taps = self._parse_taps(taps)
        trunk_out, trunk_tapped = self.trunk.propagate(field, taps=trunk_taps)
        ports = self.split(trunk_out)
        tapped = {f"trunk/{k}": v for k, v in trunk_tapped.items()}
        outputs = {}
        for branch in self.branches:
            out, branch_tapped = branch.path.propagate(
                ports[branch.port], taps=branch_taps[branch.name]
            )
            outputs[branch.name] = out
            tapped.update({f"{branch.name}/{k}": v for k, v in branch_tapped.items()})
        return outputs, tapped

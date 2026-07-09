"""Multi-channel linearization: the shared trunk hoisted once.

For an ``OpticalSystem`` whose shared wavefront modes live in the TRUNK (at
the entrance, or on a named trunk ``PhaseScreen``), every channel sees the
SAME driving coefficients through its OWN downstream optics. Because each
stage is linear in the field, the shared-mode column of channel ``i`` is
exact:

    G_i[k] = factor * P_i(port_i(T_post(B_k * V)))

with ``V`` the field at the shared plane, ``T_post`` the trunk remainder,
``port_i`` the splitter amplitude, ``P_i`` the branch, and ``factor`` the OPD
phase factor ``i 2 pi / lambda`` (one, for an amplitude basis). The trunk
remainder is propagated ONCE per mode and every branch reuses the stack --
the hoist. An INTERIOR shared stage matters: a naive entrance injection is
wrong whenever optics precede the shared plane (they do not commute with the
mode multiply).

The blocks feed two consumers: correlated per-channel speckle (the same
shared draw through each channel's ``g_shared``) and cross-channel
feed-forward control. Monochromatic, like ``linearize``; run per sub-band
for a broadband model.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from physicaloptix.elements import PhaseScreen
from physicaloptix.linearize import _phase_factor, _replace_data
from physicaloptix.path import OpticalPath


class ChannelLinearization(eqx.Module):
    """One channel's nominal field and shared-mode sensitivity block.

    Attributes:
        name: The branch name.
        e_nom: Complex nominal field at the channel's terminal plane.
        g_shared: Shared-mode sensitivity ``d(E_channel)/d(mode)``, shape
            ``(m, y, x)``, per unit mode coefficient.
        pixel_scale_lod: The terminal grid's pixel scale.
    """

    name: str = eqx.field(static=True)
    e_nom: Array
    g_shared: Array
    pixel_scale_lod: float = eqx.field(static=True)

    @property
    def n_modes(self):
        """Number of shared-mode columns."""
        return self.g_shared.shape[0]


class MultiChannelLinearization(eqx.Module):
    """The per-channel ``(e_nom, g_shared)`` blocks of one shared basis.

    Attributes:
        channels: One ``ChannelLinearization`` per branch.
        wavelength_nm: Design wavelength of the phase factor.
        kind: The shared basis kind (``"opd"`` or ``"amplitude"``).
    """

    channels: tuple
    wavelength_nm: float = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    @property
    def names(self):
        """The channel names, in branch order."""
        return tuple(channel.name for channel in self.channels)

    def __getitem__(self, name):
        """The named channel's block."""
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"unknown channel {name!r}; channels are {self.names}")


def linearize_shared(
    system,
    field,
    *,
    wavelength_nm,
    shared_stage=None,
    basis=None,
    chunk_size=None,
):
    """Per-channel shared-mode blocks of a forked system, trunk hoisted once.

    Args:
        system: The ``OpticalSystem``.
        field: The unperturbed entrance field (monochromatic).
        wavelength_nm: Design wavelength for the OPD phase factor.
        shared_stage: Name of the trunk ``PhaseScreen`` carrying the shared
            modes; the injection happens at ITS plane, linearized about its
            current coefficients. Mutually exclusive with ``basis``.
        basis: A ``ModeBasis`` applied at the ENTRANCE plane instead (only
            correct when nothing precedes the shared modes).
        chunk_size: Modes per propagation batch; ``None`` batches all.

    Returns:
        A ``MultiChannelLinearization``.
    """
    if field.spectrum is not None:
        raise ValueError(
            "linearize_shared is monochromatic (like linearize); run it per "
            "sub-band for a broadband model"
        )
    if (shared_stage is None) == (basis is None):
        raise ValueError("provide exactly one of shared_stage or basis")

    if shared_stage is not None:
        names = [stage.name for stage in system.trunk.stages]
        if shared_stage not in names:
            raise ValueError(
                f"unknown shared_stage {shared_stage!r}; trunk stages are {names}"
            )
        idx = names.index(shared_stage)
        screen = system.trunk.stages[idx].op
        if not isinstance(screen, PhaseScreen):
            raise TypeError(
                f"shared_stage {shared_stage!r} is not a PhaseScreen; got "
                f"{type(screen).__name__}"
            )
        basis = screen.basis
        pre = OpticalPath(stages=system.trunk.stages[: idx + 1])
        post = OpticalPath(stages=system.trunk.stages[idx + 1 :])
    else:
        pre = OpticalPath(stages=())
        post = system.trunk

    shared_plane_field, _ = pre.propagate(field)
    factor = _phase_factor(wavelength_nm) if basis.kind == "opd" else 1.0
    injected = factor * basis.B * shared_plane_field.data  # (m, y, x)

    def to_split_plane(data):
        out, _ = post.propagate(_replace_data(shared_plane_field, data))
        return out.data

    def stack(fn, block):
        if chunk_size is None:
            return jax.vmap(fn)(block)
        return jnp.concatenate(
            [
                jax.vmap(fn)(block[start : start + chunk_size])
                for start in range(0, block.shape[0], chunk_size)
            ]
        )

    trunk_stack = stack(to_split_plane, injected)  # (m, y, x) at the split
    trunk_out, _ = system.trunk.propagate(field)
    outputs, _ = system.propagate(field)

    channels = []
    for branch in system.branches:

        def through_channel(data, branch=branch):
            ports = system.split(_replace_data(trunk_out, data))
            out, _ = branch.path.propagate(ports[branch.port])
            return out.data

        g_shared = stack(through_channel, trunk_stack)
        out_field = outputs[branch.name]
        channels.append(
            ChannelLinearization(
                name=branch.name,
                e_nom=out_field.data,
                g_shared=g_shared,
                pixel_scale_lod=float(out_field.grid.dx),
            )
        )
    return MultiChannelLinearization(
        channels=tuple(channels),
        wavelength_nm=float(wavelength_nm),
        kind=basis.kind,
    )


def ncpa_differential_opd(system, name_a, name_b):
    """The differential (non-common-path) OPD map between two branches.

    Sums each branch's ``PhaseScreen`` OPD maps (``B . coeffs``) and returns
    branch ``a`` minus branch ``b`` -- the aberration a sensor in one arm can
    never see about the other, the irreducible cross-channel floor.

    Args:
        system: The ``OpticalSystem``.
        name_a: First branch name.
        name_b: Second branch name.

    Returns:
        The differential OPD map, shape ``(npix, npix)``, in the basis's
        length unit (nm by convention).
    """
    by_name = {branch.name: branch for branch in system.branches}
    for name in (name_a, name_b):
        if name not in by_name:
            raise ValueError(f"unknown branch {name!r}; branches are {sorted(by_name)}")

    def branch_opd(branch):
        npix = system.split.grid.npix
        total = jnp.zeros((npix, npix))
        for stage in branch.path.stages:
            if isinstance(stage.op, PhaseScreen):
                screen = stage.op
                total = total + jnp.tensordot(
                    screen.basis.coeffs, screen.basis.B, axes=1
                )
        return total

    return branch_opd(by_name[name_a]) - branch_opd(by_name[name_b])

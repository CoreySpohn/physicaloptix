"""Multi-channel linearization: the shared trunk hoisted once.

Every channel sees the same shared command through its own downstream
optics. A named trunk stage is differentiated through its real optical
response at its current command. The prefix is evaluated once, and the
splitter and branches share the trunk output and its tangent. An explicitly
supplied basis instead perturbs the field at the entrance plane.

The blocks feed two consumers: correlated per-channel speckle (the same
shared draw through each channel's ``g_shared``) and cross-channel
feed-forward control. Monochromatic, like ``linearize``; run per sub-band
for a broadband model.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from optixstuff.grids import PixelGrid

from physicaloptix.command import command_response
from physicaloptix.elements import PhaseScreen
from physicaloptix.linearize import _replace_data, linearize
from physicaloptix.path import OpticalPath


class ChannelLinearization(eqx.Module):
    """One channel's nominal field and shared-mode sensitivity block.

    Attributes:
        name: The branch name.
        e_nom: Complex nominal field at the channel's terminal plane.
        g_shared: Shared-mode sensitivity ``d(E_channel)/d(mode)``, shape
            ``(m, y, x)``, per unit mode coefficient.
        grid: The terminal grid including its Frame.
    """

    name: str = eqx.field(static=True)
    e_nom: Array
    g_shared: Array
    grid: PixelGrid

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
        input_energy: Total energy of the shared entrance field
            (``field.energy()``) -- one photometric reference for every
            channel; each channel pairs it with its own ``grid``
            to convert density amplitudes into per-pixel amplitudes.
    """

    channels: tuple
    wavelength_nm: float = eqx.field(static=True)
    kind: str | None = eqx.field(static=True)
    input_energy: float = eqx.field(static=True)

    @property
    def names(self):
        """The channel names, in branch order."""
        return tuple(channel.name for channel in self.channels)

    def __getitem__(self, name):
        """The named channel's block.

        Args:
            name: Channel name.

        Returns:
            The ChannelLinearization with that name.
        """
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
        shared_stage: Name of the commanded trunk element. The derivative
            is evaluated at its current command. Mutually exclusive with
            ``basis``.
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
        response = command_response(system, field, shared_stage)
        outputs = response._propagate(response.element.command)
        blocks = response.materialize(chunk_size)
        kind = response.element.command_kind
    else:
        # An explicitly supplied entrance basis is a perturbation of the
        # input field. Reuse the single-path analytic linearization.
        trunk = linearize(
            system.trunk,
            field,
            basis,
            wavelength_nm=wavelength_nm,
            method="analytic",
            chunk_size=chunk_size,
        )
        trunk_out, _ = system.trunk.propagate(field)
        downstream = eqx.tree_at(lambda s: s.trunk, system, OpticalPath(()))
        outputs, _ = downstream.propagate(trunk_out)
        kind = trunk.kind

        def through_channels(data):
            fields, _ = downstream.propagate(_replace_data(trunk_out, data))
            return {name: out.data for name, out in fields.items()}

        if chunk_size is None:
            blocks = jax.vmap(through_channels)(trunk.G)
        else:
            blocks = jax.lax.map(through_channels, trunk.G, batch_size=chunk_size)

    channels = tuple(
        ChannelLinearization(
            name=branch.name,
            e_nom=outputs[branch.name].data,
            g_shared=blocks[branch.name],
            grid=outputs[branch.name].grid,
        )
        for branch in system.branches
    )
    return MultiChannelLinearization(
        channels=channels,
        wavelength_nm=float(wavelength_nm),
        kind=kind,
        input_energy=float(field.energy()),
    )


def ncpa_differential_opd(system, name_a, name_b):
    """The differential (non-common-path) OPD map between two branches.

    Sum the applied ``PhaseScreen`` OPD maps in each branch and return
    branch ``a`` minus branch ``b``.

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
        npix = system.split.grid.shape[0]
        total = jnp.zeros((npix, npix))
        for stage in branch.path.stages:
            if isinstance(stage.op, PhaseScreen):
                total = total + stage.op.opd()
        return total

    return branch_opd(by_name[name_a]) - branch_opd(by_name[name_b])

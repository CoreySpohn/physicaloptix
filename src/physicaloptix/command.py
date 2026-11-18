"""Derivatives of propagated complex fields with respect to optic commands."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from physicaloptix.core import Field
from physicaloptix.elements.base import AbstractCommandableElement
from physicaloptix.path import OpticalPath
from physicaloptix.system import OpticalSystem


class CommandResponse(eqx.Module):
    """A field derivative at the selected element's current command.

    The field at the element's input is propagated once at construction.
    Each derivative runs the commanded element and the downstream optics
    through their ordinary propagation, sharing the trunk in a forked system.
    """

    element: AbstractCommandableElement
    field: Field
    downstream: OpticalPath | OpticalSystem

    def _propagate(self, command: Array) -> Field | dict[str, Field]:
        field = self.element.with_command(command)(self.field)
        return self.downstream.propagate(field)[0]

    def jvp(self, tangent: ArrayLike) -> Array | dict[str, Array]:
        """Apply the Jacobian to one command-shaped tangent.

        Return the complex density-amplitude derivative on the output grid,
        or a dict of those arrays keyed by channel for an OpticalSystem.
        """
        command = self.element.command
        if not jnp.issubdtype(command.dtype, jnp.inexact):
            command = command.astype(float)

        def propagated_data(command):
            outputs = self._propagate(command)
            if isinstance(outputs, dict):
                return {name: field.data for name, field in outputs.items()}
            return outputs.data

        return jax.jvp(
            propagated_data, (command,), (jnp.asarray(tangent, dtype=command.dtype),)
        )[1]

    def materialize(self, chunk_size: int | None = None) -> Array | dict[str, Array]:
        """Return the Jacobian with a leading command axis at every output.

        Chunking bounds the number of propagated tangents in a batch. None
        batches all commands. The remaining axes match the output field data.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        command = self.element.command
        indices = jnp.arange(command.size)

        def column(index):
            return self.jvp(jax.nn.one_hot(index, command.size, dtype=command.dtype))

        if chunk_size is None or chunk_size >= command.size:
            return jax.vmap(column)(indices)
        return jax.lax.map(column, indices, batch_size=chunk_size)


def command_response(
    path_or_system: OpticalPath | OpticalSystem, field: Field, stage: str
) -> CommandResponse:
    """Differentiate the command of a named path stage or system trunk stage."""
    system = path_or_system if isinstance(path_or_system, OpticalSystem) else None
    path = path_or_system if system is None else system.trunk
    names = tuple(step.name for step in path.stages)
    if stage not in names:
        raise ValueError(f"no stage named {stage!r}; stages are {names}")
    index = names.index(stage)
    element = path.stages[index].op
    if not isinstance(element, AbstractCommandableElement):
        raise TypeError(f"stage {stage!r} does not expose an optical command")
    before = OpticalPath(path.stages[:index])
    after = OpticalPath(path.stages[index + 1 :])
    downstream = (
        after if system is None else eqx.tree_at(lambda s: s.trunk, system, after)
    )
    return CommandResponse(element, before.propagate(field)[0], downstream)

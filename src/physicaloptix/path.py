"""The optical path: named, plane-tagged stages folded once.

``OpticalPath.propagate`` is one Python fold over a static tuple of stages, so
plane validation, tapped intermediates, and (later) linearization are views
of the same fold rather than separate code paths. The fold is unrolled at
trace time by construction -- each stage retags the field's static plane and
grid, so the pytree structure changes stage to stage and a ``lax.scan`` over
stages is structurally impossible (and unnecessary at ~10 stages).

The tap set is a static argument: taps off is the production hot path,
bit-identical and cost-free; taps on returns the same result plus the named
intermediate fields (which carry their plane tags and grids, so a path
renderer consumes them directly).
"""

import equinox as eqx

from physicaloptix.linearize import linearize as _linearize


class Stage(eqx.Module):
    """A named step of the path: an element or propagator."""

    name: str = eqx.field(static=True)
    op: eqx.Module


def _planes(op):
    """The (input, output) planes an op consumes and produces."""
    if hasattr(op, "plane_in"):
        return op.plane_in, op.plane_out
    return op.plane, op.plane


class OpticalPath(eqx.Module):
    """A linear chain of named stages with construction-time plane checking."""

    stages: tuple

    def __check_init__(self):
        """Reject duplicate names and plane-inconsistent chains at build."""
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names in {names}")
        current = None
        for stage in self.stages:
            if getattr(stage.op, "multi_output", False):
                raise ValueError(
                    f"stage '{stage.name}' is a multi-output op and cannot "
                    "be a Stage of the linear fold; use OpticalSystem to "
                    "fork a path"
                )
            plane_in, plane_out = _planes(stage.op)
            if current is not None and plane_in is not current:
                raise ValueError(
                    f"stage '{stage.name}' expects the {plane_in.value} "
                    f"plane but the chain is in the {current.value} plane"
                )
            current = plane_out

    def propagate(self, field, *, taps=()):
        """Fold the field through every stage.

        Args:
            field: The input field (must match the first stage's plane).
            taps: Static tuple of stage names whose output fields to record.

        Returns:
            ``(field, tapped)``: the output field and a dict mapping each
            tapped stage name to the field just after that stage (empty when
            ``taps`` is empty).
        """
        names = {stage.name for stage in self.stages}
        unknown = [t for t in taps if t not in names]
        if unknown:
            raise ValueError(
                f"unknown tap name(s) {unknown}; stages are {sorted(names)}"
            )
        tapped = {}
        for stage in self.stages:
            field = stage.op(field)
            if stage.name in taps:
                tapped[stage.name] = field
        return field, tapped

    def linearize(self, field, basis, **kwargs):
        """Build the (E_nom, G) linearization of this path around ``field``.

        See :func:`physicaloptix.linearize.linearize` for the arguments
        (``wavelength_nm`` is required; ``method`` defaults to ``"auto"``).
        """
        return _linearize(self, field, basis, **kwargs)

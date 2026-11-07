"""Late-bound wavelength: ops that fix their wavelength after construction.

Some stages bind the wavelength when they are built (a phase screen converts
its OPD at one wavelength, a sampled mask is tabulated at one wavelength's
lambda/D scale). An op that derives from ``AbstractChromaticOp`` instead
exposes ``at_wavelength``, which returns the op ready to act at a given
wavelength. ``OpticalPath.at_wavelength`` maps it over the stages, so one
path object, holding every shared parameter exactly once, serves every
wavelength node.
"""

import abc

import equinox as eqx


class AbstractChromaticOp(eqx.Module):
    """An op whose wavelength is bound late, by ``at_wavelength``."""

    @abc.abstractmethod
    def at_wavelength(self, wavelength_nm):
        """The op bound to ``wavelength_nm`` (nanometers, a concrete float).

        Args:
            wavelength_nm: Wavelength in nanometers.

        Returns:
            An op that acts at that wavelength. Dynamic leaves are shared
            with ``self``, not copied.
        """


def _op_planes(op):
    if hasattr(op, "plane_in"):
        return op.plane_in, op.plane_out
    return op.plane, op.plane


class ChromaticTable(AbstractChromaticOp):
    """One op per wavelength node, selected by ``at_wavelength``.

    For stages that bind the wavelength in their sampled data (a mask
    transmission tabulated at each node's lambda/D scale, a per-node
    instrument phasor, an output grid in each node's units). Each entry is
    built exactly as the single-wavelength stage would be; the table only
    picks one. Lookup is by exact float equality with the node wavelengths,
    so the same tuple must feed the table and the loop that binds it.

    A table cannot act on a field itself: calling it raises until the path
    has been bound with ``OpticalPath.at_wavelength``.

    Attributes:
        ops: The per-node ops, all acting between the same planes.
        wavelengths_nm: Node wavelengths in nanometers, one per op, distinct.
    """

    ops: tuple
    wavelengths_nm: tuple[float, ...] = eqx.field(static=True, converter=tuple)

    def __check_init__(self):
        """Validate node counts, distinct wavelengths and common planes."""
        if len(self.ops) < 1:
            raise ValueError("a ChromaticTable needs at least one op")
        if len(self.ops) != len(self.wavelengths_nm):
            raise ValueError(
                f"need one wavelength per op: {len(self.ops)} ops, "
                f"{len(self.wavelengths_nm)} wavelengths"
            )
        if len(set(self.wavelengths_nm)) != len(self.wavelengths_nm):
            raise ValueError(
                f"table wavelengths must be distinct, got {self.wavelengths_nm}"
            )
        planes = {_op_planes(op) for op in self.ops}
        if len(planes) != 1:
            raise ValueError(
                "every op of a ChromaticTable must act between the same planes"
            )

    @property
    def plane_in(self):
        """The plane the ops consume."""
        return _op_planes(self.ops[0])[0]

    @property
    def plane_out(self):
        """The plane the ops produce."""
        return _op_planes(self.ops[0])[1]

    def at_wavelength(self, wavelength_nm):
        """The op built for exactly ``wavelength_nm``.

        Args:
            wavelength_nm: One of the table's node wavelengths.

        Returns:
            The node's op (shared, not copied).

        Raises:
            KeyError: ``wavelength_nm`` is not a node of the table.
        """
        for wavelength, op in zip(self.wavelengths_nm, self.ops, strict=True):
            if wavelength == wavelength_nm:
                return op
        raise KeyError(
            f"wavelength {wavelength_nm} nm is not a table node; "
            f"nodes are {self.wavelengths_nm}"
        )

    def __call__(self, field):
        """Refuse to act: the table must be bound to a wavelength first."""
        raise ValueError(
            "a ChromaticTable is unbound; call OpticalPath.at_wavelength(...) "
            "before propagating"
        )

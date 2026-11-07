"""One path, many wavelength nodes: the polychromatic fold.

``PolychromaticPath`` holds a single ``OpticalPath`` whose wavelength-bound
stages are late-bound (``AbstractChromaticOp``, ``ChromaticTable``) and loops
over wavelength nodes, binding the path at each node and summing the
weighted energy images. Because every node reads the same path object, a
parameter shared across wavelengths (an OPD offset, a stop shift) is one
leaf: its gradient is the sum over nodes by construction, not by bookkeeping.

The loop is outer over nodes and inner over sources, one unit of work per
(node, source), each optionally rematerialized (``eqx.filter_checkpoint``) so
the backward pass holds about one propagation at a time.
"""

import equinox as eqx
import numpy as np

from physicaloptix.core import Field
from physicaloptix.path import OpticalPath
from physicaloptix.sources import PointSource


def _floats(values):
    return tuple(float(v) for v in values)


def _node_image(path, entrance, source, scale, wavelength_nm, pupil_diameter_m, parity):
    """Energy image of one source at one node, times ``scale``."""
    field = source.entrance_field(entrance, wavelength_nm, pupil_diameter_m, parity)
    out, _ = path.at_wavelength(wavelength_nm).propagate(field)
    return scale * out.intensity() * out.grid.weights


class PolychromaticPath(eqx.Module):
    """A late-bound path folded over wavelength nodes and point sources.

    Attributes:
        path: The ``OpticalPath``; its late-bound stages are bound per node.
        entrance: The untilted, normalized entrance ``Field`` (mono), the
            argument ``PointSource.entrance_field`` tilts.
        wavelengths_nm: Node wavelengths in nanometers. Any
            ``ChromaticTable`` in the path must hold these exact values.
        weights: Non-negative relative node weights; ``image`` normalizes
            them to unit sum on the host.
        pupil_diameter_m: Pupil diameter in meters, the lambda/D scale of the
            source positions.
        source_parity: ``(x, y)`` signs applied to source positions before
            the tilt, for paths that reflect an axis (see
            ``PointSource.entrance_field``).
        checkpoint: Rematerialize each (node, source) unit in the backward
            pass instead of storing its intermediates.
    """

    path: OpticalPath
    entrance: Field
    wavelengths_nm: tuple[float, ...] = eqx.field(static=True, converter=_floats)
    weights: tuple[float, ...] = eqx.field(static=True, converter=_floats)
    pupil_diameter_m: float = eqx.field(static=True)
    source_parity: tuple[int, int] = eqx.field(
        static=True, default=(1, 1), converter=tuple
    )
    checkpoint: bool = eqx.field(static=True, default=True)

    def __check_init__(self):
        """Validate the nodes and weights, and that the path binds at every node."""
        n_nodes = len(self.wavelengths_nm)
        if n_nodes < 1:
            raise ValueError("need at least one wavelength node")
        if len(self.weights) != n_nodes:
            raise ValueError(
                f"need one weight per node: {n_nodes} nodes, "
                f"{len(self.weights)} weights"
            )
        if not all(w > 0 for w in self.wavelengths_nm):
            raise ValueError(
                f"node wavelengths must be positive, got {self.wavelengths_nm}"
            )
        if any(w < 0 for w in self.weights) or not sum(self.weights) > 0:
            raise ValueError(
                "node weights must be non-negative with a positive sum, "
                f"got {self.weights}"
            )
        if len(self.source_parity) != 2:
            raise ValueError(f"source_parity must be (x, y), got {self.source_parity}")
        if self.entrance.spectrum is not None:
            raise ValueError("the entrance field must be monochromatic")
        for wavelength_nm in self.wavelengths_nm:
            self.path.at_wavelength(wavelength_nm)

    def image(self, sources):
        """The weighted energy image of the sources over all nodes.

        Each (node, source) is propagated through the path bound at the node
        wavelength; the energy images ``flux * w_k * s_k * I`` are summed in
        node-major order, with ``w`` the unit-sum node weights and ``s`` the
        source's ``spectral_weights`` (one when ``None``).

        Args:
            sources: A ``PointSource`` or a sequence of them.

        Returns:
            The image as a fraction of entrance energy per output sample, on
            the path's output grid.

        Raises:
            ValueError: No sources, or a source's ``spectral_weights`` does
                not have one entry per node.
        """
        if isinstance(sources, PointSource):
            sources = (sources,)
        if len(sources) == 0:
            raise ValueError("need at least one source")
        n_nodes = len(self.wavelengths_nm)
        for source in sources:
            if (
                source.spectral_weights is not None
                and source.spectral_weights.shape[0] != n_nodes
            ):
                raise ValueError(
                    f"spectral_weights has {source.spectral_weights.shape[0]} "
                    f"entries for {n_nodes} nodes"
                )
        weights = np.asarray(self.weights) / np.sum(self.weights)
        unit = eqx.filter_checkpoint(_node_image) if self.checkpoint else _node_image
        total = None
        for k, (wavelength_nm, weight) in enumerate(
            zip(self.wavelengths_nm, weights, strict=True)
        ):
            for source in sources:
                scale = weight * source.flux
                if source.spectral_weights is not None:
                    scale = scale * source.spectral_weights[k]
                image = unit(
                    self.path,
                    self.entrance,
                    source,
                    scale,
                    wavelength_nm,
                    self.pupil_diameter_m,
                    self.source_parity,
                )
                total = image if total is None else total + image
        return total

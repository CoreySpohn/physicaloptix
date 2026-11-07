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

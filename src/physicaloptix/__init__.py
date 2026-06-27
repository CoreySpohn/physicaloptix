"""physicaloptix -- physical optics (PSFs and diffraction) for the HWO suite.

A downstream consumer of optixstuff that produces PSFs via dLux, parallel to
coronagraphoto (image sim) and jaxedith (ETC). optixstuff stays free of dLux and
physical optics; diffraction lives here, with dLux as the (hidden, swappable)
backend.

``DLuxCoronagraph`` implements optixstuff's ``AbstractCoronagraph``, so
coronagraphoto / jaxedith get dLux-propagated PSFs by dependency injection: build
one from an optixstuff primary and hand it over, no dLux in sight.
"""

from physicaloptix._version import __version__
from physicaloptix.apertures import to_dlux_aperture
from physicaloptix.coronagraph import DLuxCoronagraph, psf

__all__ = [
    "DLuxCoronagraph",
    "__version__",
    "psf",
    "to_dlux_aperture",
]

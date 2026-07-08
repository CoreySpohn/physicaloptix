"""Mode-basis constructors: physically meaningful ``ModeBasis`` stacks.

These build the ``B`` stack that ``linearize`` propagates into the sensitivity
``G``. Two contracts they all honour, both easy to get silently wrong:

- **Units.** ``linearize`` reads an OPD basis in the SAME length unit as the
  wavelength (nanometres), so every constructor returns ``B`` already scaled to
  nanometres. A coefficient of 1 on a mode produces that mode's ``*_nm``
  amount of wavefront error, not a dimensionless shape.
- **Grid.** The modes live on the dimensionless pupil ``Grid`` the field uses
  (coordinates in pupil diameters), not on a metre grid.
"""

import math

import jax.numpy as jnp
import numpy as np

from physicaloptix.apertures import rasterize_segments
from physicaloptix.elements.basis import ModeBasis


def _noll_to_nm(j):
    """Map a 1-based Noll index to the Zernike ``(n, m)`` (signed azimuth)."""
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    n = 0
    remainder = j - 1
    while remainder > n:
        n += 1
        remainder -= n
    m = (-1) ** j * ((n % 2) + 2 * ((remainder + ((n + 1) % 2)) // 2))
    return n, m


def _zernike_radial(n, m, rho):
    """The Zernike radial polynomial ``R_n^{|m|}(rho)`` (unnormalized)."""
    m = abs(m)
    radial = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coefficient = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        radial += coefficient * rho ** (n - 2 * k)
    return radial


def _zernike(n, m, rho, theta):
    """A single unnormalized Zernike ``Z_n^m`` on ``(rho, theta)``."""
    radial = _zernike_radial(n, m, rho)
    if m > 0:
        return radial * np.cos(m * theta)
    if m < 0:
        return radial * np.sin(-m * theta)
    return radial


def segment_ptt_basis(primary, grid, *, ptt_nm=1.0, supersample=16):
    """Per-segment piston, tip, and tilt modes as an OPD ``ModeBasis``.

    Three modes per segment (piston, then the x- and y-tilts), ordered by
    ``primary.segment_centres_m`` (centre segment first). Each mode is confined
    to its own segment and normalized so a coefficient of 1 gives ``ptt_nm`` of
    area-weighted RMS wavefront error over that segment. This is the dominant
    segment-phasing basis for a segmented aperture (the PASTIS control set).

    Args:
        primary: A ``SegmentedPrimary`` with an exact segment size.
        grid: The pupil ``Grid`` the field is sampled on (sets ``npix`` and the
            dimensionless tilt coordinates).
        ptt_nm: Per-mode RMS amplitude in nanometres for a unit coefficient.
        supersample: Subpixel samples per axis for the segment masks.

    Returns:
        A ``ModeBasis`` with ``B`` of shape ``(3 * n_segments, npix, npix)`` in
        nanometres, zero coefficients, and ``kind="opd"``.
    """
    npix = grid.npix
    masks = rasterize_segments(primary, npix, supersample=supersample)
    coords = np.asarray(grid.coords)
    x_grid, y_grid = np.meshgrid(coords, coords)
    centres = np.asarray(primary.segment_centres_m, dtype=float) / float(
        primary.diameter_m
    )

    modes = []
    for mask, (cx, cy) in zip(masks, centres, strict=True):
        area = mask.sum()
        for ramp in (np.ones_like(mask), x_grid - cx, y_grid - cy):
            shape = mask * ramp
            rms = np.sqrt((shape**2).sum() / area)
            modes.append(ptt_nm * shape / rms)

    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def fourier_dm_basis(grid, *, n_actuators, k_min=1.0, k_max=None, rms_nm=1.0):
    """Band-limited cosine/sine modes of a Fourier deformable mirror.

    A deformable mirror with ``n_actuators`` across the pupil controls spatial
    frequencies up to its Nyquist ``n_actuators / 2`` cycles per aperture, and a
    pupil-phase frequency of ``k`` cycles per aperture places a focal-plane
    speckle at radius ``k`` lambda/D. The mode set spanning
    ``k_min <= |k| <= k_max`` is therefore exactly the control basis that carves
    a dark hole over that annulus. Each half-plane integer frequency
    ``(kx, ky)`` contributes a cosine and a sine mode (the two focal
    quadratures a two-sided or broadband hole needs), each normalized so a unit
    coefficient gives ``rms_nm`` of RMS wavefront error over the aperture.

    Args:
        grid: The pupil ``Grid`` the field is sampled on.
        n_actuators: Actuators across the pupil; sets the Nyquist frequency cap
            ``n_actuators / 2`` on the controllable region.
        k_min: Smallest controlled frequency (inner working angle in lambda/D).
        k_max: Largest controlled frequency; clamped to the Nyquist cap and
            defaulting to it when ``None`` (outer working angle in lambda/D).
        rms_nm: Per-mode RMS amplitude in nanometres for a unit coefficient.

    Returns:
        A ``ModeBasis`` with ``B`` of shape ``(n_modes, npix, npix)`` in
        nanometres and zero coefficients, ``n_modes`` even (cosine/sine pairs).

    Raises:
        ValueError: If the band selects no frequencies.
    """
    nyquist = n_actuators / 2.0
    upper = nyquist if k_max is None else min(k_max, nyquist)
    coords = np.asarray(grid.coords)
    x_grid, y_grid = np.meshgrid(coords, coords)
    aperture = (x_grid**2 + y_grid**2) <= 0.25

    modes = []
    kcap = int(np.floor(upper))
    for kx in range(0, kcap + 1):
        for ky in range(-kcap, kcap + 1):
            if kx == 0 and ky <= 0:  # keep one of each +/- pair (half-plane)
                continue
            if not (k_min <= math.hypot(kx, ky) <= upper):
                continue
            arg = 2.0 * np.pi * (kx * x_grid + ky * y_grid)
            for shape in (np.cos(arg), np.sin(arg)):
                rms = np.sqrt((shape[aperture] ** 2).mean())
                modes.append(rms_nm * shape / rms)

    if not modes:
        raise ValueError(f"no modes in band [{k_min}, {upper}] cycles per aperture")
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def zernike_basis(grid, n_modes, *, rms_nm=1.0, diameter=1.0):
    """Noll-ordered Zernike modes over a circular pupil as an OPD ``ModeBasis``.

    Mode ``j`` (1-based Noll) is evaluated on the aperture disk, zeroed
    outside it, and normalized so a coefficient of 1 gives ``rms_nm`` of RMS
    wavefront error over the aperture. Modes are mutually orthogonal over the
    aperture to the discretization limit.

    Args:
        grid: The pupil ``Grid`` the field is sampled on.
        n_modes: Number of Noll modes, starting from piston (j = 1).
        rms_nm: Per-mode RMS amplitude in nanometres for a unit coefficient.
        diameter: Aperture diameter in grid units (1.0 fills the pupil grid,
            whose coordinates span one diameter).

    Returns:
        A ``ModeBasis`` with ``B`` of shape ``(n_modes, npix, npix)`` in
        nanometres, zero coefficients, and ``kind="opd"``.
    """
    coords = np.asarray(grid.coords)
    x_grid, y_grid = np.meshgrid(coords, coords)
    radius = diameter / 2.0
    r = np.sqrt(x_grid**2 + y_grid**2)
    aperture = r <= radius
    rho = np.where(aperture, r / radius, 0.0)
    theta = np.arctan2(y_grid, x_grid)
    npupil = aperture.sum()

    modes = []
    for j in range(1, n_modes + 1):
        n, m = _noll_to_nm(j)
        z = _zernike(n, m, rho, theta) * aperture
        rms = np.sqrt((z**2).sum() / npupil)
        modes.append(rms_nm * z / rms)

    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(n_modes))

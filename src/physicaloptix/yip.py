"""The yield-input-package (YIP) emitter: freeze a path to YIP tables.

Emits the standard coronagraph yield data package -- ``stellar_intens`` (band
images versus stellar angular diameter), ``offax_psf`` (band images versus
source offset), and ``sky_trans`` (extended-source transmission) -- in the
FITS layout the table-backed readers consume. Recipes follow the design-survey
reference implementation exactly:

- Band images are the equal-weight mean of per-wavelength intensities on ONE
  fixed angular grid (reference-wavelength lambda/D).
- A stellar diameter is 50 pointings drawn area-uniformly over the stellar
  disk (radius ``sqrt(U)/2`` in diameter units, uniform azimuth, seeded per
  diameter), each applied as a fixed-angle 2D tilt, incoherently averaged.
- Sky transmission is stochastic: complex-Gaussian pupil screens at uniform
  random in-band wavelengths, the field scaled by ``lambda / lambda_ref``,
  intensities accumulated and normalized by ``npix^2 / n_screens / 2`` --
  which makes an open telescope transmit exactly one.

The input field is normalized to unit pre-coronagraph energy on the
dimensionless pupil grid, so map values are in the survey convention. This is
builder-tier code: a full-scale package is many propagations; freeze once and
serve from the tables.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from physicaloptix.core import Field, Spectrum
from physicaloptix.sources import point_source
from physicaloptix.transforms import Fraunhofer

DEFAULT_STELLAR_DIAMETERS = np.concatenate([[0.0], np.logspace(-3, 0, 11)])


def _fits():
    try:
        from astropy.io import fits
    except ImportError as err:
        raise ImportError(
            "the YIP emitter writes FITS files and needs astropy (pip install astropy)"
        ) from err
    return fits


def _unit_energy(field):
    energy = jnp.sum(field.data.real**2 + field.data.imag**2) * field.grid.weights
    return Field(
        data=field.data / jnp.sqrt(energy),
        grid=field.grid,
        plane=field.plane,
        spectrum=field.spectrum,
    )


def _band_image(core_path, science, source):
    """Equal-weight band-averaged intensity on the fixed angular grid."""
    lyot, _ = core_path.propagate(source)
    return np.asarray(science(lyot).intensity())


def _make_header(
    fits,
    *,
    science_grid,
    spectrum,
    reference_wavelength_nm,
    n_pointings,
    diameter_m,
    inscribed_diameter_m,
    obscured,
    design_name,
    time_multiplier_detection,
    time_multiplier_characterization,
):
    def tidy(value):
        # Full-precision float noise (0.9500000000000001) overflows the
        # 80-char card and truncates the comment the readers parse units
        # from; nine decimals is far beyond any physical meaning here.
        return float(np.round(value, 9))

    center = science_grid.npix / 2 - 0.5  # 00LL pixel coordinate of the axis
    lambda_um = tidy(reference_wavelength_nm * 1e-3)
    wavelengths = np.asarray(spectrum.wavelengths_nm)
    header = fits.Header()
    header["DESIGN"] = design_name
    header["PIXSCALE"] = (tidy(science_grid.dx), "pixel scale in units of lambda0/D")
    header["LAMBDA"] = (lambda_um, "central wavelength of the bandpass in microns")
    header["MINLAM"] = (
        tidy(wavelengths.min() * 1e-3),
        "shortest wavelength of the bandpass in microns",
    )
    header["MAXLAM"] = (
        tidy(wavelengths.max() * 1e-3),
        "longest wavelength of the bandpass in microns",
    )
    header["XCENTER"] = (center, "center of the image in 00LL pixel coordinates")
    header["YCENTER"] = (center, "center of the image in 00LL pixel coordinates")
    header["OBSCURED"] = (
        obscured,
        "obscured aperture-area fraction vs circular",
    )
    header["JITTER"] = (0, "RMS jitter per axis in mas")
    header["N_LAM"] = (
        len(spectrum),
        "number of wavelength samples",
    )
    header["N_STAR"] = (n_pointings, "number of points across stellar diameter")
    header["D"] = (diameter_m, "circumscribed telescope diameter in meters")
    header["D_INSC"] = (
        diameter_m if inscribed_diameter_m is None else inscribed_diameter_m,
        "inscribed diameter of the telescope in meters",
    )
    header["TMULDET"] = (
        time_multiplier_detection,
        "time multiplier for detection observations",
    )
    header["TMULCHAR"] = (
        time_multiplier_characterization,
        "time multiplier for characterization obs",
    )
    return header


def emit_yip(
    output_dir,
    core_path,
    input_field,
    *,
    science_grid,
    spectrum,
    reference_wavelength_nm,
    offsets_lod,
    diameter_m,
    stellar_diams_lod=None,
    n_pointings=50,
    n_sky_screens=1000,
    sky_band_nm=None,
    seed=0,
    inscribed_diameter_m=None,
    obscured=0.0,
    design_name="physicaloptix",
    time_multiplier_detection=1.0,
    time_multiplier_characterization=1.0,
):
    """Freeze a coronagraph path to a yield data package on disk.

    Args:
        output_dir: Directory to write the package into (created if needed).
        core_path: ``OpticalPath`` from entrance pupil to Lyot plane.
        input_field: The entrance-pupil field (normalized internally to unit
            pre-coronagraph energy).
        science_grid: The fixed angular science grid, in
            reference-wavelength lambda/D.
        spectrum: The evaluation band (equal weights match the survey mean).
        reference_wavelength_nm: Band-center wavelength defining the grid.
        offsets_lod: 1D array of off-axis source offsets along +x, in
            reference lambda/D.
        diameter_m: Circumscribed telescope diameter (header metadata).
        stellar_diams_lod: Stellar angular diameters; defaults to the survey
            list ``[0] + logspace(-3, 0, 11)``.
        n_pointings: Point sources per resolved stellar diameter.
        n_sky_screens: Stochastic screens for the sky transmission map.
        sky_band_nm: ``(min, max)`` wavelength range the sky screens draw
            from. The survey convention draws over the FULL band
            (``center * (1 +- bandwidth / 2)``), which is wider than the
            midpoint-sampled spectrum; defaults to the spectrum's range.
        seed: Seed for the (deterministic) pointing and screen draws.
        inscribed_diameter_m: Inscribed diameter (header; defaults to
            ``diameter_m``).
        obscured: Obscured aperture-area fraction (header).
        design_name: Design name (header).
        time_multiplier_detection: Header metadata.
        time_multiplier_characterization: Header metadata.

    Returns:
        Dict with the emitted arrays (``stellar_intens``,
        ``stellar_intens_diam_list``, ``offax_psf``,
        ``offax_psf_offset_list``, ``sky_trans``).
    """
    fits = _fits()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stellar_diams_lod is None:
        stellar_diams_lod = DEFAULT_STELLAR_DIAMETERS
    stellar_diams_lod = np.asarray(stellar_diams_lod, dtype=float)
    offsets_lod = np.asarray(offsets_lod, dtype=float)

    base = _unit_energy(input_field)
    science = Fraunhofer(
        grid_in=base.grid,
        grid_out=science_grid,
        reference_wavelength_nm=reference_wavelength_nm,
        min_wavelength_nm=float(np.asarray(spectrum.wavelengths_nm).min()),
    )

    # -- off-axis PSFs ------------------------------------------------------
    offax = np.stack(
        [
            _band_image(
                core_path,
                science,
                point_source(
                    base,
                    spectrum=spectrum,
                    separation_lod=float(offset),
                    reference_wavelength_nm=reference_wavelength_nm,
                ),
            )
            for offset in offsets_lod
        ]
    )

    # -- stellar intensity vs angular diameter ------------------------------
    stellar = []
    for diameter in stellar_diams_lod:
        if diameter == 0.0:
            stellar.append(
                _band_image(
                    core_path,
                    science,
                    point_source(base, spectrum=spectrum),
                )
            )
            continue
        rng = np.random.default_rng(seed)  # per-diameter, as the survey does
        radius = np.sqrt(rng.uniform(0, 1, n_pointings)) / 2
        theta = rng.uniform(0, 2 * np.pi, n_pointings)
        image = 0.0
        for r, t in zip(radius, theta, strict=True):
            position = (
                float(r * np.cos(t) * diameter),
                float(r * np.sin(t) * diameter),
            )
            image = image + _band_image(
                core_path,
                science,
                point_source(
                    base,
                    spectrum=spectrum,
                    position_lod=position,
                    reference_wavelength_nm=reference_wavelength_nm,
                ),
            )
        stellar.append(image / n_pointings)
    stellar = np.stack(stellar)

    # -- stochastic sky transmission ----------------------------------------
    rng = np.random.default_rng(seed)
    if sky_band_nm is None:
        wavelengths = np.asarray(spectrum.wavelengths_nm, dtype=float)
        min_wl, max_wl = wavelengths.min(), wavelengths.max()
    else:
        min_wl, max_wl = float(sky_band_nm[0]), float(sky_band_nm[1])
    npix = base.grid.npix
    sky = 0.0
    for _ in range(n_sky_screens):
        screen = rng.standard_normal((npix, npix)) + 1j * rng.standard_normal(
            (npix, npix)
        )
        wavelength = rng.uniform(min_wl, max_wl)
        single = Spectrum(
            wavelengths_nm=jnp.array([wavelength]), weights=jnp.array([1.0])
        )
        data = base.data * jnp.asarray(screen) * (wavelength / reference_wavelength_nm)
        source = Field(
            data=data[None, ...],
            grid=base.grid,
            plane=base.plane,
            spectrum=single,
        )
        lyot, _ = core_path.propagate(source)
        out = science(lyot).data[0]
        sky = sky + np.asarray(out.real**2 + out.imag**2)
    sky = sky * (npix**2 / n_sky_screens / 2.0)

    # -- write the package ---------------------------------------------------
    header = _make_header(
        fits,
        science_grid=science_grid,
        spectrum=spectrum,
        reference_wavelength_nm=reference_wavelength_nm,
        n_pointings=n_pointings,
        diameter_m=diameter_m,
        inscribed_diameter_m=inscribed_diameter_m,
        obscured=obscured,
        design_name=design_name,
        time_multiplier_detection=time_multiplier_detection,
        time_multiplier_characterization=time_multiplier_characterization,
    )
    fits.writeto(output_dir / "stellar_intens.fits", stellar, header, overwrite=True)
    fits.writeto(
        output_dir / "stellar_intens_diam_list.fits",
        stellar_diams_lod,
        header,
        overwrite=True,
    )
    fits.writeto(output_dir / "offax_psf.fits", offax, header, overwrite=True)
    fits.writeto(
        output_dir / "offax_psf_offset_list.fits",
        offsets_lod,
        header,
        overwrite=True,
    )
    fits.writeto(output_dir / "sky_trans.fits", sky, header, overwrite=True)

    return {
        "stellar_intens": stellar,
        "stellar_intens_diam_list": stellar_diams_lod,
        "offax_psf": offax,
        "offax_psf_offset_list": offsets_lod,
        "sky_trans": sky,
    }

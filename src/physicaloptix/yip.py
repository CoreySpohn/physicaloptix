"""The yield-input-package (YIP) emitter: freeze a path to YIP tables.

Emits the standard coronagraph yield data package -- ``stellar_intens`` (band
images versus stellar angular diameter), ``offax_psf`` (band images versus
source offset), and ``sky_trans`` (extended-source transmission) -- in the
FITS layout the table-backed readers consume. Recipes follow the design-survey
reference implementation exactly:

- Band images are the quadrature-weighted mean of per-wavelength intensities on ONE
  fixed angular grid (reference-wavelength lambda/D).
- A stellar diameter is 50 pointings drawn area-uniformly over the stellar
  disk (radius ``sqrt(U)/2`` in diameter units, uniform azimuth, seeded per
  diameter), each applied as a fixed-angle 2D tilt, incoherently averaged.
- Sky transmission is stochastic: complex-Gaussian pupil screens at uniform
  random in-band wavelengths, the field scaled by ``lambda / lambda_ref``,
  intensities accumulated and normalized by ``npix^2 / n_screens / 2`` --
  which makes an open telescope transmit exactly one.

The input field is normalized to unit pre-coronagraph energy on the
dimensionless pupil grid, and ``stellar_intens`` / ``offax_psf`` values are
per-pixel flux fractions (intensity density times the pixel cell area) --
the convention shipped survey packages carry; ``sky_trans`` samples
dimensionless transmission and carries no area factor. This is
builder-tier code: a full-scale package is many propagations; freeze once and
serve from the tables.
"""

from dataclasses import replace

import numpy as np
from hwoutils.grids import Frame, Image, Measure
from yippy import write_yip
from yippy.writer import yip_header

from physicaloptix._sky import uniform_sky_image, unit_energy
from physicaloptix.sources import point_source
from physicaloptix.transforms import Fraunhofer

DEFAULT_STELLAR_DIAMETERS = np.concatenate([[0.0], np.logspace(-3, 0, 11)])


def _band_image(core_path, science, source):
    """Area-scaled point intensity on the fixed angular science grid.

    The YIP encodes each pointing as a centre sample times cell area,
    before the incoherent average over a finite star's pointings.
    """
    lyot, _ = core_path.propagate(source)
    return np.asarray(science.forward(lyot).intensity().integrated().data)


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
        spectrum: A one-band Quadrature, for example
            ``Quadrature.build((band,), core_path, order=4)``. Its dimensional
            weights define the mean; an explicit midpoint rule reproduces the
            survey sampling.
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
        Point densities, transmission and sample axes under the YIPPackage names.
    """
    science_grid = replace(
        science_grid, frame=Frame(reference_wavelength_nm, diameter_m, "circumscribed")
    )

    if stellar_diams_lod is None:
        stellar_diams_lod = DEFAULT_STELLAR_DIAMETERS
    stellar_diams_lod = np.asarray(stellar_diams_lod, dtype=float)
    offsets_lod = np.asarray(offsets_lod, dtype=float)

    if len(spectrum.bands) != 1:
        raise ValueError("a yield package requires a one-band quadrature")
    active = np.asarray(spectrum.weights_nm[0]) != 0
    nodes_nm = np.asarray(spectrum.nodes_nm[0])[active]
    base = unit_energy(input_field)
    science = Fraunhofer(
        grid_in=base.grid,
        grid_out=science_grid,
        reference_wavelength_nm=reference_wavelength_nm,
        min_wavelength_nm=float(nodes_nm.min()),
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
    band = spectrum.bands[0]
    science_band_nm = (band.lo_nm, band.hi_nm)
    sky_band_nm = science_band_nm if sky_band_nm is None else sky_band_nm
    sky = uniform_sky_image(
        core_path,
        science,
        base,
        n_screens=n_sky_screens,
        seed=seed,
        sky_band_nm=sky_band_nm,
    )

    # -- write the package ---------------------------------------------------
    products = {
        "stellar_intensity": Image(
            stellar,
            science_grid,
            ("stellar_diameter",),
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="band_limited",
            outside="reflect",
        ).density("grid"),
        "stellar_diameters_lod": stellar_diams_lod,
        "offaxis_psfs": Image(
            offax,
            science_grid,
            ("source",),
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="band_limited",
        ).density("grid"),
        "offsets_xy_lod": np.column_stack((offsets_lod, np.zeros_like(offsets_lod))),
        "sky_transmission": sky,
    }

    def product_header(band_nm, n_wavelengths):
        return yip_header(
            science_grid,
            design=design_name,
            inscribed_diameter_m=diameter_m
            if inscribed_diameter_m is None
            else inscribed_diameter_m,
            min_wavelength_nm=band_nm[0],
            max_wavelength_nm=band_nm[1],
            obscured=obscured,
            jitter_mas=0.0,
            n_wavelengths=n_wavelengths,
            n_pointings=n_pointings,
            time_multiplier_detection=time_multiplier_detection,
            time_multiplier_characterization=time_multiplier_characterization,
        )

    headers = dict.fromkeys(products, product_header(science_band_nm, nodes_nm.size))
    headers["sky_transmission"] = product_header(sky_band_nm, n_sky_screens)
    # Keep the original area-scaled samples on disk. Re-encoding the native
    # density after averaging resolved-star pointings can round a second time.
    write_yip(
        output_dir,
        {
            **products,
            "stellar_intensity": stellar,
            "offaxis_psfs": offax,
            "headers": headers,
            "sample_kind": "point",
        },
    )
    return products

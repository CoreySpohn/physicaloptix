# Changelog

## [4.0.0](https://github.com/CoreySpohn/physicaloptix/compare/v3.0.0...v4.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **physicaloptix:** close propagation, instrument and IFS forwards at their working precision
* **physicaloptix:** keep Fresnel detuning and spatial phase across the float32 grid, condition OU exposure averaging, refuse float64-only statistics only
* **detector:** ReparameterizedDetector raises on photon_counting=True, because its draws and log_prob follow the Gaussian density.
* **speckle:** a monochromatic linearization becomes an opd or amplitude FocalResponse by its basis kind and refuses an unknown kind; DispersiveScreen.command_kind is None, so a dispersive command is linearized on wavelength nodes; AnalyticSpeckleField refuses amplitudes, phases or frequencies whose shapes do not match the response modes; at_wavelength no longer precompensates the interpolated sensitivity.
* **interop:** PathCoronagraph.lyot_field, input_energy, _records and _sky are removed in favour of calibration and recalibrated(); an explicit grid always returns pixel integrals, including at native pitch; from_path refuses a native pitch above lambda_min / (2 L lambda_ref) and an extent outside the pupil Nyquist radius at the shortest active wavelength; PolychromaticPath.image raises on unequal node grids; point_source takes a traced position and an explicitly positioned chromatic source requires reference_wavelength_nm.
* **ifs:** psflet_pack passes template_origin to PsfletPack and requires an optixstuff that declares it; the pack preflight uses the shortest wavelength; psflet_template is host-side (no jit or vmap).
* **ifs:** center PSFlet templates on their centroid and write pack format 2
* **nircam:** a provenance record now compares equal only to another provenance record; compare it with a plain mapping through to_dict(). NIRCamInputs and NIRCamBand have explicit constructors with the same parameters and order as before, and complex FITS header values are recorded as {"real": float, "imag": float}.
* **speckle:** AnalyticSpeckleField.coefficients and peak_contrast, and ChromaticAnalyticSpeckleField.realize and peak_contrast, take keyword-only elapsed_s, seconds since the field's clock origin, in place of time_d; positional coefficients(t) raises TypeError. Synthesis no longer reads epoch_d; convert scene days with field.elapsed_s(time_d). prepare_speckles and the boiling views pass times_s to realize as elapsed_s unchanged, so a protocol field must accept the elapsed_s keyword.
* **nircam:** NIRCamInputs.provenance and NIRCamBand.provenance refuse at construction, naming the key path, any key that is not a str and any value other than an exact str, bool, int, finite float or None, or a list, tuple or str-keyed mapping of them: NumPy scalars and arrays, Decimal, NaT and other datetime values, sets and frozensets, complex, bytes, infinite floats (NaN was already refused) and objects with their own hashing or equality. Records holding 1, 1.0 and True now compare unequal. to_dict() returns the record as plain dicts and lists for JSON.
* **leaves:** PointSource position_arcsec and spectral_weights, BeamSplitter wavelengths_nm and transmit_table, NIRCamInputs ote_opd_nm and si_opd_nm, and NIRCamBand si_opd_nm are strong float leaves of the active dtype (complex input is refused); RomanCompact pupil and lyot, and NIRCamInputs pupil_amplitude, lyot_amplitude and si_support, are strong leaves of their own kind in the active precision.
* **nircam:** NIRCamInputs and NIRCamBand refuse at construction a provenance holding a NumPy array, a set or another unhashable value, a NaN number, or a mapping whose keys mix types or cannot be ordered; each refusal names the key.
* **leaves:** SampledOptic transmission (float or complex) and shift_px, SampledPhaseScreen opd_nm, PointSource flux, BeamSplitter t and r, RomanCompact pupil_mask, fpm and the four DM maps, and every RomanFull arrays value change from NumPy or weakly typed arrays to strong jax.Array leaves (kind kept: a complex phasor stays complex, a mask stays boolean). NIRCamInputs.provenance and NIRCamBand.provenance change from a dict leaf to a static read-only hashable mapping: item reads work, mutation and json.dumps on it no longer do, lists inside become tuples.
* **interop:** render only what the model holds, zero invalid sources, bound product grids by pupil Nyquist
* **interop:** public uniform_sky_profile on PathCoronagraph
* **yip:** yippy spellings for emit_yip arguments and dark_zone_mask working angles
* **interop:** rename sep_step_lod to separation_step_lod and follow the trimmed slot
* **coronagraph:** serve the finite-star check from stellar_map only
* **coronagraph:** drop the outer working angle field and make uniform_sky_response a concrete method on PathCoronagraph
* **transforms:** name the Fraunhofer and Fresnel wavelength range ends low_nm and high_nm
* name every wavelength interval end low_nm and high_nm
* use the single Rates type
* read quadrature weights by name
* import grids, spectral and radial_profile from optixstuff and own the random fields module
* psflet_pack returns an optixstuff PsfletPack
* store YIP and field intensities with explicit measures and integrate through footprints
* replace Spectrum with one-row quadratures
* compute the live uniform-sky response from the incoherent pupil screen and expose point energy separately
* differentiate commanded elements through their own call with command_response and hold speckle fields as a FocalResponse
* write YIP products through yippy and integrate hard-edged leakage over the exact region in tests
* take raster extents, radii and radial profiles from the hwoutils grid
* take photometry from optixstuff and reject finite stars in PathCoronagraph.stellar_map
* **ifs:** remove save_psflet_pack; coronachrome owns the PSFlet pack format
* build PSD surfaces on hwoutils random fields; add ou_exposure_neff; colours from eyepiece
* binning from hwoutils, one FFT and MFT helper each, and no propagator call aliases
* drop the deprecated path renderer and the unweighted speckle option; geometric centres come from hwoutils
* **speckle:** build analytic speckle fields on the shared linear speckle field
* build fields, spectra, coronagraph records and detectors on the shared grid, quadrature and detector types

### Features

* build fields, spectra, coronagraph records and detectors on the shared grid, quadrature and detector types ([e5c9ccc](https://github.com/CoreySpohn/physicaloptix/commit/e5c9ccc70a0deab3b530d93a67c63e1151dfa655))
* compute the live uniform-sky response from the incoherent pupil screen and expose point energy separately ([0a3df8d](https://github.com/CoreySpohn/physicaloptix/commit/0a3df8d2933b77abbfb3af7842c71a30613a0d79))
* **coronagraph:** drop the outer working angle field and make uniform_sky_response a concrete method on PathCoronagraph ([96a26a7](https://github.com/CoreySpohn/physicaloptix/commit/96a26a7fb41acbcec7aaf357c395f995992ca594))
* **coronagraph:** serve the finite-star check from stellar_map only ([c122012](https://github.com/CoreySpohn/physicaloptix/commit/c122012932cf883ab874865af8523d6da4c80ad2))
* differentiate commanded elements through their own call with command_response and hold speckle fields as a FocalResponse ([44c9e9b](https://github.com/CoreySpohn/physicaloptix/commit/44c9e9b1e90b518bd36921df456fc48703e1c260))
* **interop:** attach the uniform-sky profile to every calibrated performance record ([5d813ff](https://github.com/CoreySpohn/physicaloptix/commit/5d813ff913b234cc654e38dc4302a11cb5311ffe))
* **interop:** declare a path coronagraph band-averaged when its build spectrum has width ([29d07f4](https://github.com/CoreySpohn/physicaloptix/commit/29d07f40406b006e95c4272704ac8c7db7ea8008))
* **interop:** public uniform_sky_profile on PathCoronagraph ([3dc4520](https://github.com/CoreySpohn/physicaloptix/commit/3dc4520c9302e9c8e9ac8b9568249aa11fa07a97))
* **interop:** serve the live path on one fixed-angle grid with pixel-integral reads and a calibration snapshot ([fc75b24](https://github.com/CoreySpohn/physicaloptix/commit/fc75b241c8bd177cd4d696144b53f4593b82a48c))
* optical products declare band-limited reconstruction so resampling preserves aperture photometry ([19f0fbc](https://github.com/CoreySpohn/physicaloptix/commit/19f0fbc98fa8ed257995236c584ed15b87f560e9))
* **speckle:** analytic speckle fields take elapsed seconds on their own clock ([2853631](https://github.com/CoreySpohn/physicaloptix/commit/28536313a9b57258dae66918e10aa871877fc66f))
* **speckle:** build analytic speckle fields on the shared linear speckle field ([dfaa8f3](https://github.com/CoreySpohn/physicaloptix/commit/dfaa8f3dcc64fec39fb2f077093bc441405d0c78))
* **speckle:** carry the basis wavelength law into speckle responses ([b88cbfe](https://github.com/CoreySpohn/physicaloptix/commit/b88cbfef811f134da8479b3f594c19e1edc4f60b))
* stellar and sky maps declare that they continue past the field ([2f96c35](https://github.com/CoreySpohn/physicaloptix/commit/2f96c35c3f1249373c9790ae90e23fa405d52eed))
* **yip:** yippy spellings for emit_yip arguments and dark_zone_mask working angles ([6155ec2](https://github.com/CoreySpohn/physicaloptix/commit/6155ec271a7de9c3086850c2ea4d3edfc9e2a654))


### Bug Fixes

* **detector:** refuse photon counting on the reparameterized Gaussian detector ([ac6e4c1](https://github.com/CoreySpohn/physicaloptix/commit/ac6e4c1dc58825cf4b2049e6fe173e45bad26706))
* **ifs:** center PSFlet templates on their centroid and write pack format 2 ([461814a](https://github.com/CoreySpohn/physicaloptix/commit/461814a181fb3493b451a9cd0c0d1f400330aa67))
* **ifs:** declare the centroid template origin and check both shifted band endpoints ([2f040aa](https://github.com/CoreySpohn/physicaloptix/commit/2f040aae52119c12b2dc91f2e74e0022f63458b1))
* **interop:** render only what the model holds, zero invalid sources, bound product grids by pupil Nyquist ([4a48561](https://github.com/CoreySpohn/physicaloptix/commit/4a48561deb1788ba2e5c995100df399c71059c99))
* **interop:** resample the sky map on a traced grid instead of comparing it on the host ([47e2a41](https://github.com/CoreySpohn/physicaloptix/commit/47e2a41aaf1c09e4149ca2f030e81c370a5bbe6c))
* **interop:** scale products to the build band centre; float32 build-wavelength and splitter tolerances ([2be2cf8](https://github.com/CoreySpohn/physicaloptix/commit/2be2cf83570f9c2a1801fb4d3f9cc4e137bb6ca9))
* **interop:** state calibration records, aperture rules and YIP geometry in build-centre lambda/D ([d240fc5](https://github.com/CoreySpohn/physicaloptix/commit/d240fc5e9070694ce5e3250e2d1731d5c47c59da))
* **leaves:** store sampled optics, sources, splitters and Roman arrays as strong leaves; freeze NIRCam provenance ([5d0199e](https://github.com/CoreySpohn/physicaloptix/commit/5d0199e379e682def7d83d0a725c34f32c45f991))
* **leaves:** store source positions, dichroic tables, Roman pupils and NIRCam prescription arrays as strong leaves ([c793618](https://github.com/CoreySpohn/physicaloptix/commit/c79361854cd049aee3cf7bc912059b62fcf8d716))
* **nircam:** compare provenance only to records and forward subclass fields ([0dd3359](https://github.com/CoreySpohn/physicaloptix/commit/0dd33599e830a69587c1e74449bc01542d1d000a))
* **nircam:** hold only plain JSON-like values in provenance and add to_dict ([6d8ffde](https://github.com/CoreySpohn/physicaloptix/commit/6d8ffded591ea06086acdc6195d46d2882d98ee9))
* **nircam:** record complex FITS cards and report provenance refusals by key path ([aac1adf](https://github.com/CoreySpohn/physicaloptix/commit/aac1adf457849d705bf960be11ceddf8588007fa))
* **nircam:** refuse provenance values that cannot hash stably, naming the key ([7511550](https://github.com/CoreySpohn/physicaloptix/commit/7511550ee0b4f22bbf21e5a3ef3858987d3c2b8b))
* **physicaloptix:** close propagation, instrument and IFS forwards at their working precision ([b0e778b](https://github.com/CoreySpohn/physicaloptix/commit/b0e778ba33bd40ff3eed309ac26c5304a637020e))
* **physicaloptix:** keep Fresnel detuning and spatial phase across the float32 grid, condition OU exposure averaging, refuse float64-only statistics only ([b638b80](https://github.com/CoreySpohn/physicaloptix/commit/b638b803fb4e727e5d8861999614ace95057d0c2))
* **physicaloptix:** read dispersive kernels and dichroic tables with interp_linear for jax 0.12.1 ([cc91f58](https://github.com/CoreySpohn/physicaloptix/commit/cc91f581de8a06696c211064c082d0248688ec45))
* **precision:** derive working dtypes from the inputs and enforce the precision guard ([f96b05e](https://github.com/CoreySpohn/physicaloptix/commit/f96b05ed9e1d486cd75bb06bdae2d32eaeb3e760))


### Code Refactoring

* binning from hwoutils, one FFT and MFT helper each, and no propagator call aliases ([836336f](https://github.com/CoreySpohn/physicaloptix/commit/836336ffed8448bd0b5f85abb1e6f67b4186d8e0))
* build PSD surfaces on hwoutils random fields; add ou_exposure_neff; colours from eyepiece ([71ce0e4](https://github.com/CoreySpohn/physicaloptix/commit/71ce0e434081190936c543ad6ad70f7aeaad0e7f))
* drop the deprecated path renderer and the unweighted speckle option; geometric centres come from hwoutils ([a39de1a](https://github.com/CoreySpohn/physicaloptix/commit/a39de1a1488e5ff3709dfe3f72796658afca7dd9))
* **ifs:** remove save_psflet_pack; coronachrome owns the PSFlet pack format ([ca5ee67](https://github.com/CoreySpohn/physicaloptix/commit/ca5ee676fd0a9f38e2844a56066bf2c3b8de5f5a))
* import grids, spectral and radial_profile from optixstuff and own the random fields module ([7494b78](https://github.com/CoreySpohn/physicaloptix/commit/7494b78e15dda9d15e8985bdb796ad43f391eaad))
* **interop:** rename sep_step_lod to separation_step_lod and follow the trimmed slot ([eeefac6](https://github.com/CoreySpohn/physicaloptix/commit/eeefac615b5e824b973c4e835ae12b1f23ea6bbf))
* name every wavelength interval end low_nm and high_nm ([3c7dda4](https://github.com/CoreySpohn/physicaloptix/commit/3c7dda4445dec5abdfb67887471643888a9bdc26))
* psflet_pack returns an optixstuff PsfletPack ([5807e81](https://github.com/CoreySpohn/physicaloptix/commit/5807e8191eae9b3706b4cb0f04281f461896fbbc))
* read quadrature weights by name ([29411fe](https://github.com/CoreySpohn/physicaloptix/commit/29411fe91db33cfd7ad53ea72d6eaa08f322cc69))
* replace Spectrum with one-row quadratures ([50281de](https://github.com/CoreySpohn/physicaloptix/commit/50281de72058a2345c6fa76fba054d5934cb127a))
* store YIP and field intensities with explicit measures and integrate through footprints ([fe6177c](https://github.com/CoreySpohn/physicaloptix/commit/fe6177caf9d87a008382bdf02596fb225272b87a))
* take photometry from optixstuff and reject finite stars in PathCoronagraph.stellar_map ([1baa198](https://github.com/CoreySpohn/physicaloptix/commit/1baa1980b15f09508768f5b749efe66ea55feb1c))
* take raster extents, radii and radial profiles from the hwoutils grid ([fd5847e](https://github.com/CoreySpohn/physicaloptix/commit/fd5847ee8fed0b0e7fbe2bc183d6571f882f7b5d))
* **transforms:** name the Fraunhofer and Fresnel wavelength range ends low_nm and high_nm ([400f867](https://github.com/CoreySpohn/physicaloptix/commit/400f867ac7aee3d9e68dc20f73e95651476ca7e5))
* use the single Rates type ([7dd4306](https://github.com/CoreySpohn/physicaloptix/commit/7dd4306cbb829dc1d1f1f551aee37056edb70581))
* write YIP products through yippy and integrate hard-edged leakage over the exact region in tests ([d06d18f](https://github.com/CoreySpohn/physicaloptix/commit/d06d18f3147369bc89bbd5d82bfaa25aca4abf98))

## [3.0.0](https://github.com/CoreySpohn/physicaloptix/compare/v2.3.1...v3.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **speckle:** the speckle clock origin is days from the scene epoch

### Features

* **elements:** add a differentiable shift_px to SampledOptic ([5e7bb7d](https://github.com/CoreySpohn/physicaloptix/commit/5e7bb7db0104bf76021d3f4fac93c8e60e0f08a7))
* **elements:** add SampledPhaseScreen, a dense OPD leaf with a late-bound wavelength ([aee6c63](https://github.com/CoreySpohn/physicaloptix/commit/aee6c63ee490f95bd61a69c5d60e37ac97b32a6d))
* **nircam:** add build_nircam_band, the band as one path with shared seams ([189d401](https://github.com/CoreySpohn/physicaloptix/commit/189d4013228183573b9c9c3141a2176479d96a2f))
* **nircam:** zero-padded FFT focal-mask path on the full band, selected by mask_transform ([fcc8638](https://github.com/CoreySpohn/physicaloptix/commit/fcc8638e6d721ef61362f4c11e5533bde8c71356))
* **path:** add ChromaticTable for per-node ops selected by wavelength ([c09271f](https://github.com/CoreySpohn/physicaloptix/commit/c09271f33670e0dcdef519be9890cd61edbcf58d))
* **path:** add PolychromaticPath folding one late-bound path over nodes and sources ([739af5c](https://github.com/CoreySpohn/physicaloptix/commit/739af5ca79f250a22d1f3f3c0b609ace22e6c077))
* **path:** address a stage by name with OpticalPath.stage ([1a5228c](https://github.com/CoreySpohn/physicaloptix/commit/1a5228cf5a66de4da17d0c286d84a2d2cb95238c))
* **path:** bind the wavelength late with AbstractChromaticOp and OpticalPath.at_wavelength ([0794b98](https://github.com/CoreySpohn/physicaloptix/commit/0794b9889aea6803198acc06410ab5576f991714))
* **sources:** add PointSource owning its position, flux and spectral weights ([8ef333b](https://github.com/CoreySpohn/physicaloptix/commit/8ef333b8e48e178878fa117164c28b27b7fc1afd))
* **speckle:** the speckle clock origin is days from the scene epoch ([d0c3166](https://github.com/CoreySpohn/physicaloptix/commit/d0c31664f9a3266cc1e2df7b20a6db04bfa440a8))


### Bug Fixes

* **nircam:** accept scalar positions in BandLimitedRoundMask.transmission ([205c43c](https://github.com/CoreySpohn/physicaloptix/commit/205c43c32345f6caac1f0bc418a844568ba011ba))

## [2.3.1](https://github.com/CoreySpohn/physicaloptix/compare/v2.3.0...v2.3.1) (2026-11-05)


### Bug Fixes

* **interop:** accept stellar_diam_lod in core_mean_intensity ([4dbc6ef](https://github.com/CoreySpohn/physicaloptix/commit/4dbc6efa5a95532cab65def7c4638ac2ac6fc7e0))
* **physicaloptix:** assert focal mask is present before use ([4d5afa5](https://github.com/CoreySpohn/physicaloptix/commit/4d5afa578522c84f021317437d1fc8abff5ca1fb))

## [2.3.0](https://github.com/CoreySpohn/physicaloptix/compare/v2.2.0...v2.3.0) (2026-10-03)


### Features

* **viz:** add plot_propagation, a field-along-z map with bench elements and a background-centered signed colormap ([9b3701f](https://github.com/CoreySpohn/physicaloptix/commit/9b3701fd29914024735e4f10e07d5c24f8671c4d))
* **viz:** add ray_train and mirror_train side views with paraxial and conic-mirror tracers, and near-to-far, Fresnel-zone and Cornu views ([da896e5](https://github.com/CoreySpohn/physicaloptix/commit/da896e528606901c9f4bc9ecc141a58ca37e6c27))


### Bug Fixes

* **deps:** require eyepiece 0.7.0 for the viz extra, whose Cornu view draws with phasor ([c2cd05e](https://github.com/CoreySpohn/physicaloptix/commit/c2cd05ed2a05471f10c1f10ad1d4229b07536bc3))

## [2.2.0](https://github.com/CoreySpohn/physicaloptix/compare/v2.1.0...v2.2.0) (2026-09-30)


### Features

* **roman:** add flight DM wavefront terms to the compact model and document the full train ([d5eb0bc](https://github.com/CoreySpohn/physicaloptix/commit/d5eb0bcd41f8482afd7a4d1a3780153787ba4edc))
* **roman:** add PROPER-convention transforms, masks and resampling weights ([df3fb93](https://github.com/CoreySpohn/physicaloptix/commit/df3fb93800cf199d7e0d6845cfec58141fa4d52f))
* **roman:** compile and run the full Roman train on pilot-beam grids ([c3d1b44](https://github.com/CoreySpohn/physicaloptix/commit/c3d1b449d9143c9e9d07e977af3dbc699e2d9815))
* **roman:** plan PROPER-style pilot-beam propagation and run it in JAX ([4a72f9b](https://github.com/CoreySpohn/physicaloptix/commit/4a72f9b5eb1e700aeb92ed710129bcc7f4ef99ec))


### Bug Fixes

* **roman:** normalize batched transforms by the grid, expose the full-train build helpers, and state the full-train agreement ([87c4bb3](https://github.com/CoreySpohn/physicaloptix/commit/87c4bb3009c0477094adca788f57de2df3829dec))

## [2.1.0](https://github.com/CoreySpohn/physicaloptix/compare/v2.0.0...v2.1.0) (2026-09-29)


### Features

* **instruments:** add the Roman DM model: influence-function surface maps and voltage-to-height conversion ([7d26d54](https://github.com/CoreySpohn/physicaloptix/commit/7d26d54bfd62be3c62b0befcf19aef5380dbc545))
* **viz:** give the field_columns no-light hatch its own hatch_color, keeping blank_color the flat fill ([c2b6a9a](https://github.com/CoreySpohn/physicaloptix/commit/c2b6a9a7d579c324f2ca2de3d77160180b84b69e))
* **viz:** let field_columns take caller-placed keys, hang gap labels on the figure, and hatch the no-light phase pixels ([5ffd710](https://github.com/CoreySpohn/physicaloptix/commit/5ffd710afc2e7d835ca6081a7dd0742d7a0724b5))


### Bug Fixes

* **deps:** require eyepiece 0.5.0 in the viz extra, the first release with eyepiece.prepared ([9a9119f](https://github.com/CoreySpohn/physicaloptix/commit/9a9119f5de691ce5a7a3d010a7f2ebbf0fadeac0))

## [2.0.0](https://github.com/CoreySpohn/physicaloptix/compare/v1.2.0...v2.0.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **viz:** share prepared speckle outputs

### Features

* **elements:** noll_to_nm and zernike_name map Noll indices to orders and names ([073268d](https://github.com/CoreySpohn/physicaloptix/commit/073268dd8e9891694461a488e46c5caba082b3e1))
* **instruments:** add NIRCam band integration over independent wavelength nodes and single detector pixel integration ([ab95e08](https://github.com/CoreySpohn/physicaloptix/commit/ab95e08cc6ac470ba7d83b935f78cd9198a0c700))
* **instruments:** add NIRCam pupil-path builder and STPSF bundle reader ([8a755ad](https://github.com/CoreySpohn/physicaloptix/commit/8a755ade134364dea78a63b258a434080c74979e))
* **instruments:** add NIRCam round focal mask, Lyot stop and post-mask SI WFE ([7885ec9](https://github.com/CoreySpohn/physicaloptix/commit/7885ec92d241234355dc561d7607cd980dbcc955))
* **instruments:** add the Roman Coronagraph compact train matched to the PROPER compact prescription ([7c73b23](https://github.com/CoreySpohn/physicaloptix/commit/7c73b230f572f27e1aacd10fe60a418deb38342b))
* **speckle:** public eps accessor for the drifting mode coefficients ([3383d40](https://github.com/CoreySpohn/physicaloptix/commit/3383d405c6c55bcfb16af44547cc5566f87b1eff))
* **viz:** add field_columns for amplitude-over-phase columns of a field sequence ([3c6571d](https://github.com/CoreySpohn/physicaloptix/commit/3c6571dadaebc7d2c02c9f62245fa1750b42c920))
* **viz:** boiling_strip and animate_speckles on the speckle-field protocol ([a38566b](https://github.com/CoreySpohn/physicaloptix/commit/a38566b0078e37494b42629681f40372ad87f911))
* **viz:** plot_zernike_pyramid lays Zernike modes out by radial and azimuthal order ([0261778](https://github.com/CoreySpohn/physicaloptix/commit/026177894c4294031dab754fd9a749e21e7764a8))
* **viz:** share prepared speckle outputs ([286484e](https://github.com/CoreySpohn/physicaloptix/commit/286484e92c59b3b11ba5eab5ae404b72deace783))
* **viz:** the five stats shapes -- contrast profile, process, ensemble, field ellipse, mode gallery ([76e4082](https://github.com/CoreySpohn/physicaloptix/commit/76e4082168f1cbc6796f536eb14687f98db7bb11))


### Bug Fixes

* **instruments:** document non-compact NIRCam mask support and pin the full-band requirement ([af896d8](https://github.com/CoreySpohn/physicaloptix/commit/af896d8e5ac357be9f6e02e2f1b29b2300ae9c8c))
* **instruments:** rename the NIRCam detector block sum to integrate_detector_pixels, take m2nm from hwoutils, and state the benchmark gate scope and by-construction containment ([9df1d98](https://github.com/CoreySpohn/physicaloptix/commit/9df1d983566809bb125079c303dc99b80322b9cd))
* **transforms:** make chromatic Fraunhofer.backward the fixed-grid adjoint at every wavelength ([97f5060](https://github.com/CoreySpohn/physicaloptix/commit/97f50605b5ed325b4991b60bb727994c206f13c2))
* **viz:** retire the eyepiece colorbar and figure-sizing workarounds ([bfddbd3](https://github.com/CoreySpohn/physicaloptix/commit/bfddbd356e43df1deb85d5fffd1c44ac073c1831))
* **viz:** strip prepared speckle sequences directly ([9f92439](https://github.com/CoreySpohn/physicaloptix/commit/9f924399ee7b354f8a86be2073876782adb7ae2e))
* **viz:** validate times before evaluating and label strip panels by their clock ([a787927](https://github.com/CoreySpohn/physicaloptix/commit/a787927762d008fcac80e4c0f4687b200fab00e3))

## [1.2.0](https://github.com/CoreySpohn/physicaloptix/compare/v1.1.0...v1.2.0) (2026-08-12)


### Features

* **diagnostics:** quadrature audit for the paired-quadrature eta bound ([ecc3b13](https://github.com/CoreySpohn/physicaloptix/commit/ecc3b131a43b765456f0f1e87a65d2a4f3ac7f44))
* **speckle:** Photometry audit of the normalization primitives and derived divisor ([542637b](https://github.com/CoreySpohn/physicaloptix/commit/542637bfcbdebdffe0d2238997aed78cc4c8c4fb))
* **speckle:** public delta_e complex-increment accessor ([8c69f97](https://github.com/CoreySpohn/physicaloptix/commit/8c69f97bd96e74f92a103404fb12f52ec982ea8a))

## [1.1.0](https://github.com/CoreySpohn/physicaloptix/compare/v1.0.1...v1.1.0) (2026-08-12)


### Features

* **api:** export the stats module ([e15346b](https://github.com/CoreySpohn/physicaloptix/commit/e15346ba4b0f1744d695b8a86a06d0301830ec04))
* **coatings:** transfer-matrix stack response and thickness dispersion kernels ([4776d86](https://github.com/CoreySpohn/physicaloptix/commit/4776d8676a88a67463ba837ed66a795f6742c50a))
* **docs:** migrate Basics to plot_path, executed output-free ([04f56f3](https://github.com/CoreySpohn/physicaloptix/commit/04f56f3dcdd71dab918a2096ffb05e10e2d9e412))
* **elements:** DispersiveScreen with tabulated complex dispersion kernel ([a00e874](https://github.com/CoreySpohn/physicaloptix/commit/a00e874c14b343186e42b082b0344ab5cebb3fde))
* **linearize:** chromatic G stacks via per-mode dispersion factors ([e5e64b0](https://github.com/CoreySpohn/physicaloptix/commit/e5e64b09b333d3dca586969200f2d51a0c33f9c4))
* **linearize:** plane-aware perturbation_stage route and linearize_stages ([5025e8e](https://github.com/CoreySpohn/physicaloptix/commit/5025e8ea4973beb03703c2750d3eacd9866e3e0f))
* **package:** export the mirror-train and plane-aware linearize API ([8901dda](https://github.com/CoreySpohn/physicaloptix/commit/8901ddab686021c3ae27c4ef96f3fa3b629ea7f5))
* **robustness:** PASTIS sensitivity matrix and the exact improper dark-zone variance budget ([7afc643](https://github.com/CoreySpohn/physicaloptix/commit/7afc643fda02de76e36a81a34f2ffd71ad89b53d))
* **speckle:** band correlation, n_eff, impropriety views and probe joint covariance ([4472b3d](https://github.com/CoreySpohn/physicaloptix/commit/4472b3d2c49c8534c2e5c88e83b40be03aab47d9))
* **speckle:** chromatic (w, m, y, x) layout through SpeckleProcess.draw ([0054ef6](https://github.com/CoreySpohn/physicaloptix/commit/0054ef608d3c304c58a312fd8db1e8f65f522325))
* **speckle:** closed-form exposure-averaging count per mode ([d9d6091](https://github.com/CoreySpohn/physicaloptix/commit/d9d609196dcaa3308db66975adfde52a1a25f048))
* **speckle:** cross_band_moments exact joint band-pair statistics ([c1012b8](https://github.com/CoreySpohn/physicaloptix/commit/c1012b8b116b89377382bf504c4f608b73376d59))
* **speckle:** derive flux-fraction normalization from recorded input-energy primitives ([25d19a2](https://github.com/CoreySpohn/physicaloptix/commit/25d19a2b0b18b553911cb265d236a012a40c9f7e))
* **speckle:** per-mode temporal PSDs and the closed-form moment self-oracle ([b1803af](https://github.com/CoreySpohn/physicaloptix/commit/b1803af922159510822a8ca0f890153aa1e8df4b))
* **speckle:** scale_e_nom option for lambda-scaled channels ([7fcd63b](https://github.com/CoreySpohn/physicaloptix/commit/7fcd63b3ededf704e9fa867ddaf17809434f6cec))
* **stats:** absorb delta-referenced Beckmann and genchi2 speckle laws, scipy-free ([0242132](https://github.com/CoreySpohn/physicaloptix/commit/02421325ca3ca2862373ca61fa7b577e2e2bbbba))
* **stats:** add generalized chi-square survival function (Gil-Pelaez) ([c65029d](https://github.com/CoreySpohn/physicaloptix/commit/c65029de9b0e40054b6efa5da205620dda0f8612))
* **trains:** band-limited power-law PSD surface synthesis ([e5679d3](https://github.com/CoreySpohn/physicaloptix/commit/e5679d3994dc031e50381efc2f852c1b93afe5a0))
* **trains:** equivalent-space mirror-train builder with bundled EAC-1 geometry ([1990455](https://github.com/CoreySpohn/physicaloptix/commit/1990455d03129c7e8c39d74cd120c78bda0cee87))
* **viz:** contrast_row with shared/independent norm policies and dark-zone rings ([2ea1141](https://github.com/CoreySpohn/physicaloptix/commit/2ea1141b196869733ef65f2c8eb9fb725184d13a))
* **viz:** convert viz to a lazy package; render_path frozen in _legacy ([56e79bb](https://github.com/CoreySpohn/physicaloptix/commit/56e79bbfba0ecbac4b14b52885b875db3d092c44))
* **viz:** deprecate render_path in favor of plot_path ([b350e62](https://github.com/CoreySpohn/physicaloptix/commit/b350e62191f7f620b3418397fdcb26b3ffe008e0))
* **viz:** minimap greyed you-are-here rail ([ae55392](https://github.com/CoreySpohn/physicaloptix/commit/ae55392690f44d56aa87eb13a26b4b3231a01c14))
* **viz:** plot_field bridge with chromatic handling and declared cut ([9689a0d](https://github.com/CoreySpohn/physicaloptix/commit/9689a0d7dbeca64dbe9b1c09f80b0c138e54d76c))
* **viz:** plot_path typed rail and panels on eyepiece.rail with native_dpi ([dc1f8d7](https://github.com/CoreySpohn/physicaloptix/commit/dc1f8d76b90e9164839a788c2c6aae00543023c2))


### Bug Fixes

* American spelling throughout (nanometres, centre, grey, honour), match optixstuff's segment_centers_m rename ([0368ad7](https://github.com/CoreySpohn/physicaloptix/commit/0368ad74ea842e4aeaf12238f6153c4114fff014))
* **linearize:** guard dispersion-without-wavelengths on every method ([5c1b8d7](https://github.com/CoreySpohn/physicaloptix/commit/5c1b8d76e35a56a10fb04ffd06a0ebdd6b4492a5))
* **linearize:** mark merged stage-route linearizations and drop stale dispersion ([282f16d](https://github.com/CoreySpohn/physicaloptix/commit/282f16d20650fe1524eda3b1cefb72c65463df9d))
* **linearize:** reject linearity_residual on perturbation_stage linearizations ([f9cda5d](https://github.com/CoreySpohn/physicaloptix/commit/f9cda5dc78f7fa702b08c1cd9ea4b46cb340542b))
* **package:** drop exports of names not yet defined in committed modules ([c2a5a8b](https://github.com/CoreySpohn/physicaloptix/commit/c2a5a8b90e279891db850e64cf43beb7fc057ad3))
* post-review polish for chromatic-optics dispersion feature ([11b758b](https://github.com/CoreySpohn/physicaloptix/commit/11b758bb2be61c2029219e4407b6fdc716cbfd3e))
* **speckle:** joint_covariance mask-shape guard, docs literal-block, tau_s coverage ([8a85c37](https://github.com/CoreySpohn/physicaloptix/commit/8a85c378d8f712c26636a4b4f2448c5e0f60af88))
* **speckle:** weight spectral lines by S(f) df so the temporal kernel is the PSD's transform ([dafbefd](https://github.com/CoreySpohn/physicaloptix/commit/dafbefd495a43787f618a413398f370ebdb66d54))
* **stats:** tighten genchi2_sf accuracy, fix its u-grid bias, and de-jargon test comments ([dc59e7c](https://github.com/CoreySpohn/physicaloptix/commit/dc59e7cd0f33d5dd668306ee0f54e03c4e32ee11))
* **trains:** reject degenerate frequency bands ([893fa21](https://github.com/CoreySpohn/physicaloptix/commit/893fa217be0edee793d3f6e03aa6b665119c2789))
* **viz:** correct system highlight, subfigure escape, rail alignment, and chromatic phase in plot_path ([e180f62](https://github.com/CoreySpohn/physicaloptix/commit/e180f626eff1a572182df716d68b7b6695ef6fd2))
* **viz:** direction-correct declared cut and axes-shape contract for plot_field ([e4368c2](https://github.com/CoreySpohn/physicaloptix/commit/e4368c2f30f9d0f115d0b99956083b16a01f69dc))
* **viz:** family-safe ring color and native-extent threading in contrast_row ([c19f608](https://github.com/CoreySpohn/physicaloptix/commit/c19f60849bec322719961b440fe037d1ee0e848e))
* **viz:** render focal-plane maps with origin lower ([4809228](https://github.com/CoreySpohn/physicaloptix/commit/4809228a4d647c35be713bbb9b84349e2ed4639d))
* **viz:** stop double-masking phase on cut update and clip cut='y' panel ([fcb4828](https://github.com/CoreySpohn/physicaloptix/commit/fcb48285591fd5d9538d08527ce5e0354114da6e))
* **viz:** tighten axes-shape guards and clarify docs for R2 review items ([0c4cb05](https://github.com/CoreySpohn/physicaloptix/commit/0c4cb0523c6fc5b4ac539b2afbf7ffa12a6c5666))
* **viz:** unconditional branch labels, positive highlight/kinds tests, named channel range in plot_path ([67fa487](https://github.com/CoreySpohn/physicaloptix/commit/67fa4876303fd730ca410059d952dbc445a74716))
* **yip:** emit per-pixel flux fractions and repair validation data paths ([f87346f](https://github.com/CoreySpohn/physicaloptix/commit/f87346fc1828867a2b5fe563a759f33be96b1960))

## [1.0.1](https://github.com/CoreySpohn/physicaloptix/compare/v1.0.0...v1.0.1) (2026-07-22)


### Bug Fixes

* **vortex:** palindromic Hann window keeps the level-handoff taper point-symmetric ([854201f](https://github.com/CoreySpohn/physicaloptix/commit/854201fb74dfa2691bc1a4ce46ec0699f122c23e))

## [1.0.0](https://github.com/CoreySpohn/physicaloptix/compare/v0.1.0...v1.0.0) (2026-07-20)


### ⚠ BREAKING CHANGES

* **interop:** PathCoronagraph builds at a true wavelength and serves the sampling-explicit contract natively -- native-grid table members removed

### Features

* **ifs:** full-array coherent detector scene -- per-lenslet two-axis MFTs, centroid-table accumulation ([76cf0aa](https://github.com/CoreySpohn/physicaloptix/commit/76cf0aae8f09aba5065eca23eb7583bc3e79608f))
* **interop:** PathCoronagraph builds at a true wavelength and serves the sampling-explicit contract natively -- native-grid table members removed ([de2a025](https://github.com/CoreySpohn/physicaloptix/commit/de2a0257615da133d333d177a88b3c5aa9b89f0c))
* **speckle:** chromatic wavelength channels -- stacked per-channel e_nom/G, nearest-channel realize, lambda-scaled broadened() ([ceff56a](https://github.com/CoreySpohn/physicaloptix/commit/ceff56a7f720297bb395e2a19f200f58058d0c8a))

## [0.1.0](https://github.com/CoreySpohn/physicaloptix/compare/v0.0.1...v0.1.0) (2026-07-16)


### Features

* **apertures:** YAML pupil loader and owned segmented rasterizer -- bundled EAC-1 geometry, survey-convention gray-pixel rendering, unit-energy normalization; pupil validation gates ([717809d](https://github.com/CoreySpohn/physicaloptix/commit/717809dfff6c6db9b1967b035de1967153d9d012))
* **broadband:** chromatic propagation -- Spectrum.tophat, point_source with per-wavelength tilts and late-bound OPD phasors, fixed-angular Fraunhofer (reference_wavelength_nm); EAC-1 broadband null and fixed-angle planet validation ([39d45b2](https://github.com/CoreySpohn/physicaloptix/commit/39d45b210341afd29947d5de80e9014cd1a7a01a))
* **core:** greenfield owned core -- Grid/Field, cmft pair, Fraunhofer with build-time sampling gates, SampledOptic, MultiScaleVortex port, OpticalTrain with static taps; EAC-1 acceptance gates in tests/validation ([8138f9d](https://github.com/CoreySpohn/physicaloptix/commit/8138f9d3ac050245de3b9213e43815e61694a9e5))
* **detector:** photon shot + Gaussian read noise measurement model ([302bc67](https://github.com/CoreySpohn/physicaloptix/commit/302bc6747b11478f0d34f09d7967fe69e8ef0b25))
* **elements:** PhaseScreen -- a mode-basis pupil phasor exp(i 2pi (coeffs.B)/lambda), the commandable/differentiable deformable-mirror & aberration stage (coeffs swapped per step via tree_at) ([3e1feb0](https://github.com/CoreySpohn/physicaloptix/commit/3e1feb03dc82adcab7479811cb0bf54eb8d273c1))
* **fresnel:** near-field angular-spectrum propagator with construction-time sampling gate, paraxial/exact kernels, chromatic + padding, and a physics V&V suite ([2847be8](https://github.com/CoreySpohn/physicaloptix/commit/2847be86802b9ba3befc343fc9d87b922558ec6d))
* **ifs:** single-lenslet wave-optics chain and PSFlet template pack emitter (format v1) ([bebb3d7](https://github.com/CoreySpohn/physicaloptix/commit/bebb3d7876c1e42d5774eb4bf4eb9a8ab6528c5e))
* **interop:** PathCoronagraph -- OpticalPath behind AbstractCoronagraph with derived IWA/throughput/core curves; retire DLuxCoronagraph and the dLux dependency ([dd568a0](https://github.com/CoreySpohn/physicaloptix/commit/dd568a02b3cd480a30e11ff68eea326f6d80bb1c))
* **linearize:** amplitude-mode Jacobian columns -- kind='amplitude' bases linearize E(1+B.eps), achromatic and exactly linear ([48a2b40](https://github.com/CoreySpohn/physicaloptix/commit/48a2b40f4ab9a898e70157379857fb915117ef01))
* **linearize:** unified (E_nom, G) entry point -- analytic/jvp/jacfwd with memory-policy streaming, ModeBasis, diff_spec, stats module, SpeckleProcess bridge; G-export reproduction gate ([069a85c](https://github.com/CoreySpohn/physicaloptix/commit/069a85cc640d9214a3aa508f586498875fc2f3dd))
* **modes:** band-limited Fourier deformable-mirror basis ([8c8d20a](https://github.com/CoreySpohn/physicaloptix/commit/8c8d20a85f55abeb79fb53ec448437993b4a0bcf))
* **modes:** mode-basis constructors -- zernike_basis (Noll) + segment_ptt_basis (PASTIS) in nm on the pupil grid, per-segment rasterizer, linearize round-trips clean ([b521837](https://github.com/CoreySpohn/physicaloptix/commit/b5218377dfd347e7088b3b61df1859c2563451e4))
* **multichannel:** linearize_shared -- trunk-hoisted per-channel shared-mode blocks (entrance + interior stage) and ncpa_differential_opd ([99072c2](https://github.com/CoreySpohn/physicaloptix/commit/99072c2e12e068c9d382e7e2a7459f337abe45e4))
* **path:** reject multi-output ops as stages at construction (fork lives in OpticalSystem) ([d2371f8](https://github.com/CoreySpohn/physicaloptix/commit/d2371f84d4e1378d6dc68c64964bc016e35c8222))
* **phase-screen:** apply per-wavelength phase for chromatic fields (broadband DM/aberration support) ([37fb944](https://github.com/CoreySpohn/physicaloptix/commit/37fb9449ad4f47279ecb55f4e36020391f2cfe8b))
* **speckle:** add AnalyticSpeckleField generator ([676ef4e](https://github.com/CoreySpohn/physicaloptix/commit/676ef4e4d6fa91307f11e8874527c0ace456b5d8))
* **speckle:** add SpeckleProcess parameter object with draw(key) ensembles ([ac3713e](https://github.com/CoreySpohn/physicaloptix/commit/ac3713e8fff2ce73ab7ec3aa69d46e2f501b0676))
* **system:** BeamSplitter -- two-port energy split with construction-time conservation gate, Babinet from_mask, quadrature energy split, and call-time dichroic routing ([776daa2](https://github.com/CoreySpohn/physicaloptix/commit/776daa24aca22abe9a51b7e825c388110c2ab50b))
* **system:** Branch/SplitterPort/OpticalSystem -- shared trunk propagated once feeding named branch paths, namespaced taps, and the as_channel_path flattening adapter ([018670e](https://github.com/CoreySpohn/physicaloptix/commit/018670e0dc8e3c396f5bf37d4fff39a2291fb0e1))
* **viz:** render_path -- glyph rail + per-stage field panels consuming tapped propagation ([bae0723](https://github.com/CoreySpohn/physicaloptix/commit/bae0723042e7265f12d9e4564570ef567ecbca0c))
* **yip:** yield-input-package emitter -- stellar_intens/offax_psf/sky_trans in the survey recipes (area-uniform disk pointings, stochastic sky screens, band-averaged fixed-angular images), yippy round-trip verified; 2D point_source positions; CPU-pinned deterministic test suite ([8129dd8](https://github.com/CoreySpohn/physicaloptix/commit/8129dd84a7efe12da056ad6958b5380ca2e6d1bc))
* **zernike-wfs:** Zernike low-order wavefront sensor forward model ([9766d3b](https://github.com/CoreySpohn/physicaloptix/commit/9766d3bbe592d258692a9fcc9fe406abf3fcd783))


### Bug Fixes

* **broadband:** chromatic slices carry the 1/lambda amplitude factor; midpoint_band + sky_band_nm -- seed-matched cds YIP cross-check lands at the engine floor with absolute scales 1.0000 ([4400de0](https://github.com/CoreySpohn/physicaloptix/commit/4400de0c7830271eb1a5847e57557fe9b7828689))
* **speckle,linearize:** stable coherent cross term (no catastrophic cancellation) + linearize stamps the focal pixel scale onto Linearization and to_speckle_process -- adversarial-review hardening ([5739b3b](https://github.com/CoreySpohn/physicaloptix/commit/5739b3b443fd19cb9653856f5bd2d4eadfe50428))

## 0.0.1 (2026-06-27)


### Features

* Initial commit ([8ce773d](https://github.com/CoreySpohn/physicaloptix/commit/8ce773d3a96b000e1bbaf0525030c1a8801333c6))


### Miscellaneous Chores

* release 0.0.1 ([155d963](https://github.com/CoreySpohn/physicaloptix/commit/155d96329592b79d33db01981350de5023da7c07))

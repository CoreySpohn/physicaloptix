# Changelog

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

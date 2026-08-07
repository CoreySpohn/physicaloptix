# Changelog

## [1.1.0](https://github.com/CoreySpohn/physicaloptix/compare/v1.0.1...v1.1.0) (2026-08-07)


### Features

* **coatings:** transfer-matrix stack response and thickness dispersion kernels ([4776d86](https://github.com/CoreySpohn/physicaloptix/commit/4776d8676a88a67463ba837ed66a795f6742c50a))
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
* **trains:** band-limited power-law PSD surface synthesis ([e5679d3](https://github.com/CoreySpohn/physicaloptix/commit/e5679d3994dc031e50381efc2f852c1b93afe5a0))
* **trains:** equivalent-space mirror-train builder with bundled EAC-1 geometry ([1990455](https://github.com/CoreySpohn/physicaloptix/commit/1990455d03129c7e8c39d74cd120c78bda0cee87))


### Bug Fixes

* American spelling throughout (nanometres, centre, grey, honour), match optixstuff's segment_centers_m rename ([0368ad7](https://github.com/CoreySpohn/physicaloptix/commit/0368ad74ea842e4aeaf12238f6153c4114fff014))
* **linearize:** guard dispersion-without-wavelengths on every method ([5c1b8d7](https://github.com/CoreySpohn/physicaloptix/commit/5c1b8d76e35a56a10fb04ffd06a0ebdd6b4492a5))
* **linearize:** mark merged stage-route linearizations and drop stale dispersion ([282f16d](https://github.com/CoreySpohn/physicaloptix/commit/282f16d20650fe1524eda3b1cefb72c65463df9d))
* **linearize:** reject linearity_residual on perturbation_stage linearizations ([f9cda5d](https://github.com/CoreySpohn/physicaloptix/commit/f9cda5dc78f7fa702b08c1cd9ea4b46cb340542b))
* **package:** drop exports of names not yet defined in committed modules ([c2a5a8b](https://github.com/CoreySpohn/physicaloptix/commit/c2a5a8b90e279891db850e64cf43beb7fc057ad3))
* post-review polish for chromatic-optics dispersion feature ([11b758b](https://github.com/CoreySpohn/physicaloptix/commit/11b758bb2be61c2029219e4407b6fdc716cbfd3e))
* **speckle:** joint_covariance mask-shape guard, docs literal-block, tau_s coverage ([8a85c37](https://github.com/CoreySpohn/physicaloptix/commit/8a85c378d8f712c26636a4b4f2448c5e0f60af88))
* **speckle:** weight spectral lines by S(f) df so the temporal kernel is the PSD's transform ([dafbefd](https://github.com/CoreySpohn/physicaloptix/commit/dafbefd495a43787f618a413398f370ebdb66d54))
* **trains:** reject degenerate frequency bands ([893fa21](https://github.com/CoreySpohn/physicaloptix/commit/893fa217be0edee793d3f6e03aa6b665119c2789))
* **viz:** render focal-plane maps with origin lower ([4809228](https://github.com/CoreySpohn/physicaloptix/commit/4809228a4d647c35be713bbb9b84349e2ed4639d))
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

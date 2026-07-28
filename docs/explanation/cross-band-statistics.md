# Cross-band speckle statistics

A chromatic {class}`~physicaloptix.SpeckleProcess` drives every wavelength
channel with the **same** real mode trajectory: a wavefront error in
nanometres is one physical surface, so only the response `G(lambda)` differs
per band. The channels are therefore not independent noise draws -- they are
one jointly improper (non-circular) complex Gaussian field over pixel and
wavelength, and their complete second-order description needs two kernels,
not one:

    Gamma_ij(r) = sum_k rms_k^2 G_k(i, r) conj(G_k(j, r))     (covariance)
    P_ij(r)     = sum_k rms_k^2 G_k(i, r) G_k(j, r)           (pseudo-covariance)

{meth}`~physicaloptix.SpeckleProcess.cross_band_moments` returns both, along
with the exact joint intensity covariance of the flux-fraction deltas
(complex Gaussian moment theorem, noncentral; the circular zero-offset case
is Goodman's classic `|Gamma|^2`, and the improper second-order algebra
follows Schreier & Scharf, *Statistical Signal Processing of Complex-Valued
Data*, 2010):

    N_i N_j Cov[delta_i, delta_j] = |Gamma_ij|^2 + |P_ij|^2
        + 2 Re[conj(A_i) A_j Gamma_ij + conj(A_i) conj(A_j) P_ij]

with `A_i` the nominal field and the heterodyne bracket present only for a
`coherent` process. The band-diagonal reproduces the monochromatic
{meth}`~physicaloptix.SpeckleProcess.moments` exactly.

## What to read off it

- {meth}`~physicaloptix.CrossBandMoments.correlation` is the derived spectral
  correlation `rho(lambda_i, lambda_j)` per pixel -- the quantity spectral
  differential imaging and spectral deconvolution depend on, here computed
  from the optical model instead of fitted with a free correlation length.
  It is generically **non-stationary**: a function of both wavelengths, not
  of their separation, with sharp structure wherever an element's dispersion
  swings quickly (a dichroic transition band imprints a decorrelation line
  no single-length stationary kernel can represent).
- {meth}`~physicaloptix.CrossBandMoments.n_eff` is the effective number of
  independent spectral looks per pixel (a participation ratio; the form for
  independent patterns is Goodman's). Because the spectrum is a vector of
  quadratic forms in one `m`-dimensional Gaussian, it is capped at
  `m (m + 3) / 2` no matter how finely the band is sampled -- and it is
  exactly 1 in the achromatic limit.
- {meth}`~physicaloptix.CrossBandMoments.impropriety` is the band-pair degree
  of impropriety `|P_ij| / sqrt(Gamma_ii Gamma_jj)`, bounded by 1. For a
  statistically homogeneous (translation-invariant) residual the cos and sin
  quadratures of every spatial frequency carry equal independent power and
  the pseudo-covariance cancels at **every** pair of wavelengths -- band
  separation cannot manufacture impropriety. A nonzero value diagnoses
  quadrature-coupled (localized, non-translation-invariant) drift, at the
  same pair of bands where it would appear monochromatically.
- {meth}`~physicaloptix.SpeckleProcess.joint_covariance` supplies the full
  `(band, pixel) x (band, pixel)` covariance on a small pixel set, for
  statistics that mix pixels and channels (band-weighted aperture sums,
  spatio-spectral matched filters). The per-pixel maps are its coincident
  diagonal.

## Limits worth knowing

- **Achromatic limit.** If `G(lambda) = G(lambda_0) lambda_0 / lambda` (the
  {func}`~physicaloptix.lambda_scaled_channels` rule) and the statistics are
  incoherent, every band pair is perfectly correlated -- exactly, in any
  basis. With a static field the same holds only if `e_nom` co-scales with
  `G`; the lambda-scaling shortcut holds `e_nom` frozen by default, which
  itself decorrelates bands slightly under coherent statistics (the
  heterodyne and speckle terms scale differently with wavelength) -- pass
  `scale_e_nom=True` to {func}`~physicaloptix.lambda_scaled_channels` for
  the co-scaled variant that restores the exact limit. A correlation length
  fitted to a frozen lambda-scaled simulation with a static field partly
  measures that artifact.
- **Two-time statistics.** `tau_s` damps both kernels by the synthesis
  autocorrelation `rho_k(tau)`, giving the exact lagged covariance
  `Cov[delta_i(t), delta_j(t + tau_s)]`; the mean is time-independent.
- **Ensembles.** The formulas describe the exact-Gaussian
  `draw(renormalize=False)` ensemble; `renormalized=True` adds the
  closed-form sub-Gaussian correction for per-draw-rms-exact ensembles
  (equal-time only).
- **Memory.** The maps scale as `w^2 * y * x` and `joint_covariance` as
  `(w * p)^2` (capped at `w * p <= 4096`).

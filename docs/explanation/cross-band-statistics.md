# Cross-band speckle statistics

A chromatic {class}`~physicaloptix.SpeckleProcess` drives every wavelength channel
with the **same** real mode trajectory: a wavefront error expressed in
nanometers is one physical surface, and only the per-channel response
$G(\lambda)$ differs across the band. Two channels of a chromatic process are
therefore not two independent noise draws with a fitted correlation between
them. They are one field over the joint space of pixel and wavelength, driven
by a single random vector, and the correlation between any two channels is a
consequence of the optical design rather than a free parameter. This page
derives that joint law, explains the two-kernel structure it needs, and works
through the limits that anchor it against the familiar monochromatic case.

Throughout, $x = (r, \lambda)$ denotes a pixel-wavelength pair, $\delta(x)$ the
flux-fraction delta {meth}`~physicaloptix.AnalyticSpeckleField.realize` returns,
and $\epsilon \in \mathbb{R}^m$ the real mode-coefficient vector with
$\mathrm{rms}_k = \sqrt{\mathrm{Var}(\epsilon_k)}$. The field itself is the
linear model

$$
\delta E(x) = \sum_k \epsilon_k\,G_k(x),
$$

with $G_k(x)$ the (generally complex) sensitivity of pixel $r$ at wavelength
$\lambda$ to mode $k$, computed once by {func}`~physicaloptix.linearize` and
stacked across channels. Because {meth}`~physicaloptix.SpeckleProcess.draw`
synthesizes one $\epsilon(t)$ with no wavelength axis at all and only
{meth}`~physicaloptix.AnalyticSpeckleField.realize` selects the per-channel
$(E_{\mathrm{nom}}, G)$ afterward, this shared-$\epsilon$ structure is not a
modeling choice: it is how the generator is built, and it holds for any
chromatic process constructed this way.

## Why the field needs two kernels, not one

A complex random variable built from real Gaussian coefficients does not
behave like the textbook circular speckle field, and the reason is visible
already for two ordinary real numbers. If $X$ and $Y$ are jointly Gaussian and
zero-mean, Isserlis' theorem (the Gaussian moment factorization) gives

$$
\mathrm{Cov}(X^2, Y^2) = 2\,\mathrm{Cov}(X, Y)^2,
$$

so a single real number, $\mathrm{Cov}(X, Y)$, fixes how the two squares
covary. A complex number $z = X + iY$ built from two such reals carries two
real degrees of freedom, and its intensity $|z|^2 = X^2 + Y^2$ needs correspondingly
more information to describe how it correlates with another intensity: not
one complex number but two, the ordinary covariance
$\Gamma = \mathbb{E}[z_1\,\overline{z_2}]$ and the pseudo-covariance
$P = \mathbb{E}[z_1 z_2]$. Schreier and Scharf (2010) develop this second-order
theory for general complex random variables and call $P$ the complementary
covariance; a variable with $P \neq 0$ is improper, or noncircular, and $|P|/\Gamma$
is what they name the degree of impropriety. They also point out that optics
already had its own name for the pair: "the standard correlation function is
called the phase-insensitive correlation, and the complementary correlation
function is called the phase-sensitive correlation." $\Gamma$ and $P$ are
exactly that pair, carried across wavelength as well as position.

For a chromatic residual, the two kernels at a pair of channels $i, j$ are

$$
\Gamma_{ij}(r) = \sum_k \mathrm{rms}_k^2\,G_k(i, r)\,\overline{G_k(j, r)},
\qquad
P_{ij}(r) = \sum_k \mathrm{rms}_k^2\,G_k(i, r)\,G_k(j, r).
$$

{meth}`~physicaloptix.SpeckleProcess.cross_band_moments` builds both from `G`
in one pass and reuses them for the mean, the covariance, and every derived
view below. $\Gamma_{ij}$ reduces to the familiar single-band $\Gamma$ on the
diagonal ($i = j$); $P_{ij}$ is the genuinely new object a chromatic process
introduces, because two channels driven by the same real vector share exactly
the correlation structure that makes $P$ nonzero in the first place.

The single-mode limit makes the mechanism concrete. With only one real mode
($m = 1$), $\Gamma_{ij} = \mathrm{rms}^2\,G(i)\,\overline{G(j)}$ and
$P_{ij} = \mathrm{rms}^2\,G(i)\,G(j)$ differ only by which factor is
conjugated, so $|P_{ij}| = |\Gamma_{ij}|$ at every pixel and every band pair.
One real degree of freedom driving both channels makes the field as improper
as it can possibly be (the Cauchy-Schwarz bound in
{meth}`~physicaloptix.CrossBandMoments.impropriety` below is saturated).
Adding independent modes is what tames this: the individual phases of
$G_k(i)\,G_k(j)$ can point in different directions and partially cancel in the
sum that builds $P_{ij}$, while $|G_k(i)\,G_k(j)|$ always adds constructively
in the sum that builds $\Gamma_{ij}$. A wide-mode residual is therefore
generically closer to circular than a narrow one, and the rank ceiling in the
effective-channel-count section below is the precise statement of how much
closer it can get.

## The joint intensity law

With $A_i = E_{\mathrm{nom}}(i)$ the deterministic nominal field in channel $i$
and $N_i$ the per-channel flux-fraction normalization, the exact joint
covariance of the deltas {meth}`~physicaloptix.AnalyticSpeckleField.realize`
returns is

$$
N_i N_j\,\mathrm{Cov}[\delta_i, \delta_j] = |\Gamma_{ij}|^2 + |P_{ij}|^2
+ 2\,\mathrm{Re}\!\left[\overline{A_i} A_j\,\Gamma_{ij}
+ \overline{A_i}\,\overline{A_j}\,P_{ij}\right],
$$

with the heterodyne bracket present only for a `coherent` process and
$N_i\,\mathbb{E}[\delta_i] = \Gamma_{ii}$. The first two terms are the
zero-offset (speckle-only) part: they are exact and noncentral, not an
approximation, and the proof is three lines of the complex Gaussian moment
theorem applied to the linear-plus-quadratic form in $\epsilon$ (the
cross term between the linear part and the quadratic part vanishes because
$\epsilon$ has zero third moments, and $|\Gamma_{ij}|^2 + |P_{ij}|^2$ is the
quadratic-quadratic term). In the classical zero-offset, circular case this
reduces to Goodman's $|\Gamma|^2$; the noncentral heterodyne bracket, and the
instantiation of both on the shared-$\epsilon$ modal kernels across
wavelength, is what this library adds.

The formula is a genuine generalization rather than a new one, because the
band-diagonal reproduces {meth}`~physicaloptix.SpeckleProcess.moments`
exactly. Setting $i = j$ makes $A_i = A_j = A$ and writes $A$ in polar form as
$A = \sqrt{I_C}\,e^{i\phi_C}$, so the heterodyne bracket collapses to

$$
2\,\mathrm{Re}\!\left[|A|^2\,\Gamma + \overline{A}^2 P\right]
= 2\,I_C\,\Gamma + 2\,I_C\,\mathrm{Re}\!\left[P\,e^{-2i\phi_C}\right]
= 4\,I_C\,\mathrm{Var}(X),
$$

using the pinning-quadrature variance
$\mathrm{Var}(X) = \tfrac{1}{2}\left(\Gamma + \mathrm{Re}[P\,e^{-2i\phi_C}]\right)$
that {meth}`~physicaloptix.SpeckleProcess.moments` already names. That is the
same heterodyne term the monochromatic law uses, recovered as a special case
rather than assumed, and it is why {meth}`~physicaloptix.SpeckleProcess.cross_band_moments`
can be tested against `moments()` at machine precision on the diagonal.

Because time enters the law only through the modal variance, replacing
$\mathrm{rms}_k^2$ with $\mathrm{rms}_k^2\,\rho_k(\tau)$ in both kernels (with
$\rho_k$ the per-mode {meth}`~physicaloptix.SpeckleProcess.autocorrelation`)
turns the same expression into the exact two-time covariance
$\mathrm{Cov}[\delta_i(t), \delta_j(t + \tau)]$: the argument does not care
whether the two evaluation points differ in wavelength, in time, or in both,
because the modal covariance $\mathrm{rms}_k^2\,\rho_k(\tau)$ is the only
place either one enters. `cross_band_moments`'s `tau_s` keyword is exactly
this substitution, with `tau_s=0.0` giving $\rho_k(0) = 1$ and the equal-time
container.

## The band-pair correlation

{meth}`~physicaloptix.CrossBandMoments.correlation` normalizes the covariance
into $\rho_{ij}(r) = \mathrm{Cov}[\delta_i, \delta_j] / \sqrt{\mathrm{Cov}[\delta_i,\delta_i]\,\mathrm{Cov}[\delta_j,\delta_j]}$,
the quantity that spectral differential imaging and spectral deconvolution
depend on, computed here from the optical model rather than fitted as a free
correlation length. It is generically **non-stationary**: a function of
$\lambda$ and $\lambda'$ separately, not of their separation $\lambda -
\lambda'$, because $\rho_{ij}$ is the normalized Gram kernel of the curve
$\lambda \mapsto C_a^{1/2}\,g(r, \lambda)$ through mode space, and that curve
has no reason to trace out a constant-speed path. A stationary kernel
(the fitted squared-exponential correlation length used elsewhere in the
literature) would require it to. Where an element's dispersion swings
quickly, such as a dichroic transition band, the curve's local speed spikes
and $\rho_{ij}$ can fall sharply between two adjacent channels while staying
high on either side of the transition: a decorrelation line that a single
correlation length cannot represent no matter how it is tuned, because a
stationary model has exactly one length for the whole band.

## Effective number of independent channels

{meth}`~physicaloptix.CrossBandMoments.n_eff` reports the participation ratio
of the band-band intensity covariance $\Sigma$ at each pixel,

$$
N_{\mathrm{eff}}(r) = \frac{(\mathrm{tr}\,\Sigma)^2}{\mathrm{tr}(\Sigma^2)},
$$

the same functional form Goodman (2015, section 7.7, eq. 7.7-19) derives for
the effective number of independent speckle realizations of unequal mean
intensity. There the sum runs over independent exposures; here it runs over
the eigenvalues of one band-band covariance matrix, so $N_{\mathrm{eff}}$
answers a related but distinct question: not how many independent looks an
exposure sequence provides, but how many independent spectral looks one
snapshot provides. $N_{\mathrm{eff}} = 1$ when every channel is perfectly
correlated (the achromatic limit below), and it grows toward the channel
count $w$ as the channels decorrelate, but it cannot grow without bound: the
spectrum at fixed $r$ is a vector of quadratic-plus-linear forms in the same
$m$-dimensional Gaussian $\epsilon$, so the band-band covariance has rank at
most $m(m+1)/2$ from the quadratic (speckle) part, plus $m$ more directions
when a nonzero nominal field adds the heterodyne term, for a hard ceiling of

$$
N_{\mathrm{eff}} \le \frac{m(m+3)}{2}.
$$

No amount of spectral resolution can raise this ceiling, because it is set by
how many independent random numbers ($\epsilon$'s $m$ entries) drive every
channel, not by how finely the band is sampled. A spectrograph with far more
channels than $m(m+3)/2$ therefore cannot be resolving $m(m+3)/2 + 1$ genuinely
independent chromatic degrees of freedom: the extra channels are, in this
exact sense, redundant with the ones already measured. This is the cross-band
analog of the low-rank residual-field structure Pogorelyuk, Kasdin, and
Rowley (2019) exploit for state estimation, here applied to the wavelength
axis of the same underlying object.

## Cross-band impropriety

{meth}`~physicaloptix.CrossBandMoments.impropriety` is the band-pair
generalization of Schreier and Scharf's degree of impropriety,

$$
\eta_{ij}(r) = \frac{|P_{ij}(r)|}{\sqrt{\Gamma_{ii}(r)\,\Gamma_{jj}(r)}},
$$

bounded by $\eta_{ij} \le 1$ from the Cauchy-Schwarz inequality applied to
$P_{ij}$ against the two diagonal kernels, with equality forced whenever the
modal covariance has rank 1 (the single-mode limit worked through above). The
more useful bound in practice is the other direction: for a statistically
homogeneous, translation-invariant residual, decomposing each spatial
frequency into its cosine and sine quadrature responses shows that the
same-frequency contribution to $P_{ij}$ carries a prefactor proportional to
the difference between the two quadratures' variances, and vanishes exactly
when that difference is zero. This holds at **every** pair of wavelengths,
because the quadrature balance is a property of $\epsilon$'s statistics
alone and does not depend on the deterministic, per-band response
$G_k(i, r)$. A homogeneous drift therefore cannot manufacture cross-band
impropriety even though the two channels share the same random vector:
$\eta_{ij}$ stays small at every band pair, for the same reason the
single-band impropriety is small for a homogeneous residual. A nonzero
$\eta_{ij}$ diagnoses the opposite case, a residual whose cosine and sine
quadratures at some spatial frequency are unequal in variance, and it flags
that at the same pair of bands the effect would already be visible
monochromatically.

## The full spatio-spectral covariance

The three views above summarize the coincident-pixel diagonal of a larger
object. {meth}`~physicaloptix.SpeckleProcess.joint_covariance` returns the
complete $\mathrm{Cov}[\delta(\mathrm{band}\ i, \mathrm{pixel}\ p),\,
\delta(\mathrm{band}\ j, \mathrm{pixel}\ q)]$ over a selected set of pixels,
shape $(w, p, w, p)$, needed for any statistic that mixes channels and
pixels at once, such as a band-weighted aperture sum or a matched filter
built across wavelength. Because its memory cost scales as $(w p)^2$, the
pixel selection is capped at $w p \le 4096$; the per-pixel maps from
{meth}`~physicaloptix.SpeckleProcess.cross_band_moments` are this object's
coincident-pixel diagonal and are the right tool whenever the cross-pixel
terms are not needed.

## The achromatic limit, and an artifact in the standard shortcut

An achromatic optical path in native $\lambda/D$ coordinates is the sanity
anchor for all of the above, and it is worth deriving in full because it also
explains a real subtlety in {func}`~physicaloptix.lambda_scaled_channels`.
On such a path the response scales as $G(\lambda) = c_\lambda\,G(\lambda_0)$
with $c_\lambda = \lambda_0/\lambda$, so both kernels inherit the same scale
factor at each channel, $\Gamma_{ij} = c_i c_j\,\Gamma_{00}$ and
$P_{ij} = c_i c_j\,P_{00}$ (writing $c_i \equiv c_{\lambda_i}$ and
$\Gamma_{00}$, $P_{00}$ for the reference-band values). Write
$K_2 = |\Gamma_{00}|^2 + |P_{00}|^2$ for the speckle-only weight at the
reference band and, for a `coherent` process with a **frozen** nominal field
$A_i \equiv A_0$ (the default {func}`~physicaloptix.lambda_scaled_channels`
behavior, which does not rescale $E_{\mathrm{nom}}$ with wavelength), write
$K_1 = 2\,\mathrm{Re}\!\left[|A_0|^2\,\Gamma_{00} + \overline{A_0}^2\,P_{00}\right]$
for the heterodyne weight. Substituting into the joint law gives, up to the
overall (wavelength-independent, so canceling in a ratio) normalization,

$$
\mathrm{Cov}[\delta_i, \delta_j] \propto c_i^2 c_j^2\,K_2 + c_i c_j\,K_1,
$$

because the frozen $A_0$ contributes no extra factor of $c$ to the heterodyne
term while the speckle-only term picks up one factor of $c$ from each of the
two kernels it squares. Dividing by the standard deviations gives the
correlation in closed form,

$$
\rho_{ij} = \frac{c_i c_j\,K_2 + K_1}{\sqrt{\left(c_i^2 K_2 + K_1\right)\left(c_j^2 K_2 + K_1\right)}}.
$$

Setting $K_1 = 0$ (the incoherent case) collapses this to exactly 1 for any
pair of channels, which is the expected achromatic limit and holds in any
basis, improper or not. With $K_1 > 0$, writing $u = c_i^2 K_2$ and
$v = c_j^2 K_2$ turns the inequality $\rho_{ij} \le 1$ into
$2\sqrt{uv} \le u + v$, the arithmetic-geometric mean inequality, with
equality only at $c_i = c_j$. **The frozen nominal field therefore strictly
decorrelates bands under coherent statistics whenever the pinned heterodyne
variance is nonzero and the two channels differ**, an artifact of the
approximation rather than a physical effect. A small worked example makes
the size of it concrete: with $K_1 = K_2$ (heterodyne and speckle-only
weights equal at the reference band) and a 10% wavelength mismatch, $c_i = 1$
and $c_j = 0.9$, the formula gives $\rho_{ij} = 1.9 / \sqrt{2 \times 1.81}
\approx 0.9984$, a small but nonzero and entirely spurious decorrelation
between two channels of an ideal achromatic instrument. Passing
`scale_e_nom=True` to {func}`~physicaloptix.lambda_scaled_channels` co-scales
$A_i = c_i A_0$ along with $G$, which makes the heterodyne term pick up the
same $c_i c_j$ factor as the speckle-only term (both become
proportional to $c_i^2 c_j^2$), restoring $\rho_{ij} = 1$ exactly. A spectral
correlation length measured from a frozen, lambda-scaled simulation with a
static nominal field is therefore partly measuring this frozen-leakage
artifact rather than the instrument's own chromaticity, and the default
stays frozen only for backward compatibility with recorded results that
predate this fix.

## Two-time statistics and ensemble conventions

`tau_s` damps both kernels by the synthesis autocorrelation
$\rho_k(\tau)$ from {meth}`~physicaloptix.SpeckleProcess.autocorrelation`,
giving the exact lagged covariance $\mathrm{Cov}[\delta_i(t), \delta_j(t +
\tau)]$ derived above; the mean is time-independent because it depends only
on $\Gamma_{ii}$, which never carries a cross-time argument. The formulas
describe the exact-Gaussian ensemble produced by `draw(renormalize=False)`.
Passing `renormalized=True` adds the closed-form correction for the
per-draw-rms-exact ensemble from `draw(renormalize=True)` (the default draw
statistics), mirroring `moments(renormalized=True)`'s single-band correction;
this correction is derived for equal time only and `cross_band_moments`
raises if it is requested together with a nonzero `tau_s`. The per-pixel
maps scale as $w^2 y x$ in memory and
{meth}`~physicaloptix.SpeckleProcess.joint_covariance` as $(w p)^2$, capped at
$w p \le 4096$; a wide-band process on a fine grid should go through a
mask-reduced or per-pair workflow rather than materializing the full maps.

## References

- Goodman, Joseph W. (2015), *Statistical Optics*, 2nd ed., John Wiley & Sons,
  Hoboken, New Jersey.
- Schreier, Peter J. and Scharf, Louis L. (2010), *Statistical Signal
  Processing of Complex-Valued Data: The Theory of Improper and Noncircular
  Signals*, Cambridge University Press.
- Racine, René, Walker, Gordon A. H., Nadeau, Daniel, Doyon, René and Marois,
  Christian (1999), "Speckle noise and the detection of faint companions",
  Publications of the Astronomical Society of the Pacific 111, 587.
- Pogorelyuk, Leonid, Kasdin, N. Jeremy and Rowley, Clarence W. (2019),
  "Reduced order estimation of the speckle electric field history for
  space-based coronagraphs", The Astrophysical Journal 881, 126.

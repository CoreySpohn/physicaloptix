# Glossary

Terms used throughout the tutorials and reference, in plain language. Where a
notebook uses one of these for the first time, this page is the place to look
it up.

```{glossary}
pupil plane
  The plane where the telescope aperture lives. A field here is the wavefront
  across the entrance aperture. physicaloptix measures pupil coordinates in
  aperture diameters.

focal plane
  The plane where an image forms, one Fourier transform away from the pupil. A
  field here is the point-spread function of the light that reached it.
  physicaloptix measures focal coordinates in {term}`lambda/D`.

lambda/D
  The natural angular unit of diffraction, the ratio of wavelength to aperture
  diameter $D$. A telescope cannot resolve detail finer than about one
  $\lambda/D$, and the first dark ring of the {term}`Airy pattern` falls at
  $1.22\,\lambda/D$.

point-spread function
  Abbreviated PSF. The image a single point source (a star) forms after passing
  through an optical system. For a clear circular aperture it is the
  {term}`Airy pattern`.

Airy pattern
  The point-spread function of an unobstructed circular aperture: a bright core
  ringed by faint diffraction rings, with intensity
  $[2 J_1(\pi r)/(\pi r)]^2$ and dark rings at $r = 1.22, 2.23, 3.24\,\lambda/D$.

optical path difference
  Abbreviated OPD. The wavefront error expressed as a length (how far the
  wavefront departs from a perfect one), usually in nanometers. It converts to
  a phase by $\phi = 2\pi\,\mathrm{OPD}/\lambda$.

wave
  A unit of optical path difference equal to one wavelength. An error of "0.1
  waves" is an OPD of $0.1\lambda$, or a phase of $0.2\pi$ radians.

wavefront error
  The departure of the actual wavefront from a perfect (flat, for a point
  source at infinity) one, from mirror figure, misalignment, or thermal drift.
  Measured as an {term}`optical path difference`, often as a root-mean-square
  over the aperture.

Strehl ratio
  The peak intensity of an aberrated point-spread function relative to a perfect
  one. A number between 0 and 1 that summarizes image quality; for small errors
  it follows the Marechal approximation
  $S \approx \exp[-(2\pi\sigma/\lambda)^2]$ with $\sigma$ the root-mean-square
  wavefront error.

matrix Fourier transform
  Abbreviated MFT. A discrete Fourier transform written as a pair of matrix
  multiplications, which lets the output (focal) sampling be chosen
  independently of the input (pupil) array size. physicaloptix uses a
  continuous-transform-weighted variant (cMFT) so the transform is
  energy-consistent.

aperture
  The opening that admits light, the pupil-plane amplitude. A clear circular
  aperture gives an {term}`Airy pattern`; a real telescope aperture is
  segmented and obstructed.

apodizer
  A pupil-plane element that grades the aperture transmission smoothly to
  reshape the point-spread function, often used to darken the diffraction rings.

Lyot stop
  A pupil-plane mask placed after a {term}`focal-plane mask` in a coronagraph.
  It blocks the starlight the mask has diffracted to the edge of the re-imaged
  pupil while passing most of the planet light.

focal-plane mask
  An element at the {term}`focal plane` of a coronagraph that acts on the
  starlight core, either blocking it (an occulter) or, for a {term}`vortex
  coronagraph`, winding its phase so it diffracts outside the pupil.

coronagraph
  An instrument that suppresses the light of an on-axis star so that a much
  fainter, nearby source (a planet) can be seen. It trades away starlight near
  the axis, quantified by the {term}`inner working angle` and {term}`throughput`.

vortex coronagraph
  A coronagraph whose {term}`focal-plane mask` applies a spiral phase ramp that
  winds $2\pi$ around its center some integer number of times. On-axis starlight
  is sent entirely outside the geometric pupil, where a {term}`Lyot stop` blocks
  it.

vortex charge
  The integer number of times a {term}`vortex coronagraph` mask winds its phase
  by $2\pi$ around the center. A higher, even charge nulls a more forgiving
  on-axis field (better against a finite stellar size and low-order aberration)
  but pushes the {term}`inner working angle` outward.

inner working angle
  Abbreviated IWA. The smallest separation at which a coronagraph passes a
  useful fraction of a planet's light, conventionally where the
  {term}`throughput` reaches half its far-field value. Closer than the IWA the
  planet is suppressed along with the star.

outer working angle
  Abbreviated OWA. The largest separation a coronagraph is designed to serve,
  set by the field of view or the correctable region of the wavefront control.

throughput
  The fraction of an off-axis point source's light that survives the
  coronagraph and reaches the focal-plane core. It rises from zero through the
  {term}`inner working angle`.

contrast
  The brightness of a source relative to the star, or of a residual relative to
  the stellar peak. An Earth twin sits near a contrast of $10^{-10}$.

dark hole
  A region of the focal plane where wavefront control has driven the residual
  starlight down to a deep {term}`contrast`, opening a window to search for
  planets. Its extent is set by the {term}`deformable mirror` control band.

deformable mirror
  Abbreviated DM. A mirror whose surface is adjustable by an array of actuators,
  used to correct {term}`wavefront error` and dig a {term}`dark hole`. It can
  write only the spatial frequencies its actuators resolve.

quadrature
  The real and imaginary parts of the scattered focal field. A pupil phase error
  scatters into the imaginary quadrature; an amplitude error into the real one.
  A single pupil deformable mirror writes only the phase quadrature, which is
  why a full dark hole needs a second, out-of-pupil mirror.

speckle
  A residual starlight spot in the focal plane, created when {term}`wavefront
  error` scatters light out of the stellar core. Speckles, not the ideal
  point-spread function, set the deep contrast floor of a coronagraph.

speckle pinning
  The amplification of a {term}`speckle` where it overlaps a bright static
  field: because speckles are coherent with the starlight, their intensity
  responds linearly (not quadratically) to a small wavefront drift there.

quasi-static speckle
  A {term}`speckle` that varies on minute-to-hour timescales (thermal drift,
  {term}`non-common-path aberration`). Unlike fast speckles, it does not average
  away over an exposure and survives differential imaging, setting the
  post-processing floor.

non-common-path aberration
  Abbreviated NCPA. Wavefront error in the science path that the wavefront
  sensor does not see, so the control loop cannot correct it directly. A leading
  source of {term}`quasi-static speckle`.

coherent
  Light that can interfere: the {term}`speckle` field is coherent with the
  starlight, which is what makes {term}`speckle pinning` and dark-hole control
  possible. Incoherent contributions (different wavelengths, separate sources)
  add in intensity, not in field.

Yield Input Package
  Abbreviated YIP. A precomputed table of coronagraph point-spread functions and
  transmission maps. The [yippy](https://yippy.readthedocs.io/) library
  interpolates a YIP; physicaloptix is its live-propagation sibling.

AbstractCoronagraph
  The interface (defined in optixstuff) that a coronagraph presents to the rest
  of the simulation suite: an on-axis PSF, a {term}`throughput` curve, an
  {term}`inner working angle`. Both a live-propagation `PathCoronagraph` and a
  sampled {term}`Yield Input Package` fill the same slot interchangeably.
```

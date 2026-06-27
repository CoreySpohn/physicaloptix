# physicaloptix

Physical optics -- PSFs and diffraction -- for the HWO direct-imaging
simulation suite.

`physicaloptix` turns an {mod}`optixstuff` hardware description into
point-spread functions by wave-optics propagation, using
[dLux](https://github.com/LouisDesdoigts/dLux) as the hidden, swappable
backend. It is a downstream consumer of optixstuff, parallel to
{mod}`coronagraphoto` (image simulation) and
[jaxEDITH](https://github.com/CoreySpohn/jaxedith) (exposure-time and yield),
so optixstuff itself stays free of diffraction code.

The central type, {class}`~physicaloptix.DLuxCoronagraph`, implements
optixstuff's `AbstractCoronagraph`. Build one from an optixstuff primary and
hand it to any downstream tool, which consumes it as an `AbstractCoronagraph`
and never touches dLux.

## What physicaloptix is NOT

- **Not a hardware model.** The telescope / coronagraph / detector description
  lives in {mod}`optixstuff`; physicaloptix consumes it.
- **Not a PSF interpolator.** That's {mod}`yippy`'s job (a sampled YIP table).
  physicaloptix is its functional sibling -- live propagation -- and both back
  the same `AbstractCoronagraph` slot.
- **Not a scene model.** Stars, planets, disks, and zodi live in
  {mod}`skyscapes`.

## Quickstart

```python
import physicaloptix as po

coro = po.DLuxCoronagraph.from_primary(primary)       # optixstuff in, dLux hidden
psf = coro.on_axis_psf(600.0, pixel_scale_rad, npix)   # PSF out
```

```{toctree}
:hidden:
:maxdepth: 2

autoapi/index
```

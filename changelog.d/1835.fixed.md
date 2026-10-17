### Fixed — Match forward subpixel smoothing to run (#1373)

`forward()` accepts subpixel smoothing on uniform and graded single-device meshes,
using the same permittivity tensor as `run()`.
Single-device `run()` supports grad/jit/vmap for source and material values.
On uniform meshes, `run()` and `forward()` refuse Kerr with smoothing.

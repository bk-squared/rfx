### Fixed — Match forward subpixel smoothing to run (#1373)

`forward()` accepts subpixel smoothing on uniform and graded single-device meshes,
using the same permittivity tensor as `run()`.
Single-device `run()` supports grad/jit/vmap for source and material values.
Traced declared shape dimensions raise an explanatory error on every entry.
On uniform meshes, `run()` and `forward()` refuse Kerr with smoothing.

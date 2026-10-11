### Fixed — Plane-wave PEC sheet seams

Uniform `run()` now realizes conductor edges using the TF/SF boundary flags
used by its time step, closing missing transverse seam edges on full-span PEC sheets.

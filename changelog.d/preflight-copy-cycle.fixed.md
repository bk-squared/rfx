### Fixed — a run no longer keeps preflight's whole-domain conductor masks alive until garbage collection (#PR)

- A `run()` or `forward()` with preflight on now frees preflight's copy of the model, and the whole-domain conductor masks it holds, when preflight returns, not at the next cyclic collection.
- Each released mask is 1 byte per cell; the time loop no longer runs beside them.

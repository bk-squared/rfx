### Fixed — preflight no longer keeps whole-domain conductor masks alive until garbage collection (#PR)

- A run with preflight enabled now frees the preflight copy and its whole-domain conductor masks when preflight returns, not at the next cyclic collection.
- `compute_waveguide_s_matrix` with a material override releases its preflight copy and conductor masks at the same point.
- Each released conductor mask saves 1 byte per cell.

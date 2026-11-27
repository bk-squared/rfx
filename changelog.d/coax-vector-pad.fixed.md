### Fixed — Coaxial ports run with vector padding (#PR)

- Uniform coaxial-port runs no longer raise `ValueError: vector padding: carry entry 'coax_sparam_accs' has no pad rule` before the first step on RTX A6000, 3090 and 3080 with jax 0.10.2 when the grid's cell count is a multiple of 4 and its z extent is not.
- `coax_sparam_accs`, `coax_time_records`, `coax_table` and `coax_freqs` now pass through padding unchanged. Results on other cards and grids do not change.

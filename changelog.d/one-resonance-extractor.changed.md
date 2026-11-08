### Changed — resonance lists and ring-down use one matrix-pencil body (#PR)

- `rfx.matrix_pencil`, `Result.find_resonances` and `matrix_pencil_from_probe` now use the same matrix-pencil code as `rfx.ringdown`. On decaying modes, frequencies and Q are unchanged to float64 rounding: at most 6e-13 relative in frequency and 2.5e-7 in Q on the test records.
- On a lossless record the decay rate is only rounding, so its sign and the reported Q (already a windowing artefact) can change. The frequency is unchanged.
- Records with a peak below about 1e-27 now return their modes. Before, a fixed singular-value floor lost them.
- An all-NaN record returns an empty list instead of raising `LinAlgError`.

### Fixed — weak-port ring-down gradient identification (#1419)

- `RingdownSpec(identification_probes=...)` adds read-only field channels to
  pole identification in `run()`, `forward()`, and early stopping; port
  residues and S assembly retain their channel order. Unsupported placements
  and lanes refuse the request. Reports expose per-pole channel amplitude shares.
- `gradient_witness` normalizes each parameter leaf independently; `bin_axis`
  judges frequency bins separately, skipping and counting zero-reference bins.
  Its bar remains 1e-2. A weak-port PEC cavity pins the gradient failure and
  its recovery with one interior identification probe.

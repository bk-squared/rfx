### Added — the microstrip S-matrix on several devices (#PR)

- `compute_msl_s_matrix(..., devices=...)` returns the same S-matrix and settling witness as one device, on uniform and graded meshes.
- Multi-device runs now return the per-step plane records the settling witness reads, each device contributing only the cells it owns.
- `eps_override`, `checkpoint_every` and `checkpoint_segments` are refused by name when combined with `devices`.

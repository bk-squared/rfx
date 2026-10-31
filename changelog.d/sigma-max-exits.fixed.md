### Fixed — The non-passive S warning now also covers the coaxial one-port reflection and the distributed, graded, subgridded and ring-down `run()`/`forward()` S (#PR)

- `compute_coaxial_line_reflection`, distributed `run()` and `forward()`, graded `forward()`, subgridded `run()` and the
  completed S of `run(ringdown=...)` / `forward(ringdown=...)` returned S with no singular-value check. They now warn,
  like the other port exits since #2314, when the largest singular value of the returned S exceeds 1 + 1e-2.
- A bin where no column is all-finite is judged by its largest finite |S_ij|, a lower bound of the largest singular
  value; a mixed result on the wave channel with a multi-cell wire port is now judged too. S and every result field
  are unchanged.

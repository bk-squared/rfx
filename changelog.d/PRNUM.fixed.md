### Fixed — `run()` and `forward()` give a point source in a closed box the same drive (#PRNUM)

- A soft point source with `amplitude_kind='current'` (the `add_source` default) adds Cb·w/dV to its edge each step.
  On a PEC or other non-absorbing boundary `run()` formed it as (Cb/dV)·w and `forward()` as (1/dV)·(Cb·w): the
  same number rounded differently, one float32 unit in the last place on about a third of the samples. In a lossless
  cavity that difference accumulates: probe fields of the two entries drifted 20–50 float32 units apart in 200 steps.
- Both entries now build the drive with one rule that depends only on `amplitude_kind`, on every boundary and path.
  `run()` on a closed boundary is unchanged; `forward()`, and `run()` on CPML/UPML, move by float32 rounding only.

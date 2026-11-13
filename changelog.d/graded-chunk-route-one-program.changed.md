### Changed — graded runs with progress or a stop rule give the plain run's numbers (#PR)

- On a graded mesh, `run(report_every=...)`, `run(until_decay=...)`, `run(until_identified=True)` and a `stop_fn`
  now step each chunk with the same scan as the plain run, so the record equals the plain run of the same length bit
  for bit. Before, the chunked program differed in the last bits while a source drove.
- What moves: a probe record by up to 2.1e-5 of its peak on a driven 600-step scene and up to 2.1e-3 of its peak on an
  11772-step lossy-wall cavity; S-parameters and port records by up to 6e-7.
- An `until_decay` run can stop one or two energy checks earlier or later (50 to 100 steps in the tests) where its
  energy sat within rounding of the threshold.

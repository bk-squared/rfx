# rfx Validation Scripts

These scripts compare rfx results with analytic calculations, other solvers,
or a second rfx configuration. They are test cases, not tutorials. Start with
`examples/README.md` if you are learning the software.

Each comparison covers the geometry, mesh, frequency range, source, and output
implemented by that script. A passing result does not establish accuracy for a
different setup.

## Run a validation case

Cross-solver comparisons live in `tests/crossval/`, with frozen reference
records and per-case pytest instructions. For the RT/Duroid patch, see
`tests/crossval/rt5880_patch/test_rt5880_patch.py`. The former
`validation/crossval/` scripts, manifest and script exit-code convention were
removed; use the pytest result and the case's stated scope.

When reporting a result, include the script path, commit, geometry, mesh,
frequency range, comparator, and numerical thresholds. Do not generalize a
successful case to a different port, material model, boundary, or mesh without
an applicable comparison.

## Other validation directories

- `research/subgrid/12_subgrid_disjoint_prototype.py` is a short-duration
  research check for a topology that is not long-time energy-stable.
- `research/subgrid/13_subgrid_material_validation.py` demonstrates the guarded
  subgrid boundary and checks the same material setup on a uniform mesh.
  SBP-SAT/subgridding is not a supported feature.
- `tmtt_paper/` contains paper-support scripts. Values identified there as
  targets are experiment goals, not measured results. See
  `tmtt_paper/README.md` for the status of each script.

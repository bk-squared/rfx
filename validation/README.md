# rfx Validation Scripts

These scripts compare rfx results with analytic calculations, other solvers,
or a second rfx configuration. They are test cases, not tutorials. Start with
`examples/README.md` if you are learning the software.

Each comparison covers the geometry, mesh, frequency range, source, and output
implemented by that script. A passing result does not establish accuracy for a
different setup.

## Run a validation case

Run scripts from the repository root, one case at a time:

```bash
python validation/crossval/15_patch_antenna_rt5880.py
```

The process exit code has a specific meaning:

- `0`: every configured numerical check passed, including a required external
  comparison when the case has one.
- `1`: a simulation, numerical check, or required execution step failed.
- `2`: the rfx checks completed, but a required reference file or external
  solver was unavailable. This result is inconclusive, not a pass.

See `crossval/manifest.json` for each script's dependencies, expected exit
codes, numerical-check files, and generated artifacts.

## Available cases

| Script | What it checks | Reference and interpretation limit |
| --- | --- | --- |

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

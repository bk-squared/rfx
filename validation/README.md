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
| `crossval/15_patch_antenna_rt5880.py` | Probe-fed patch on a 3.175 mm RT/Duroid 5880 laminate, uniform mesh | A same-geometry rfx-versus-OpenEMS integration study on a different laminate from the former cv05 patch case, removed 2026-09-21, which it does not replace. It checks a coarse-mesh envelope (resonance within 8% of the OpenEMS dip, a −40 dB settling witness, passivity on both sides, broadside directivity within 3 dB) and asserts the substrate rasterizes to exactly four cells before quoting a resonance. Issue #812 (round 2): its 8 % f0 gate is blind to the #740 one-plane-ground realization (`validation/crossval/_15_patch_results/rfx_one_plane_ground_b29f9de7.json::f_harminv_hz = 2471858108` Hz vs `validation/crossval/_15_patch_results/openems.json::f_dip_hz = 2330000000` Hz, +6.09 %); only the build assertion catches it, and no ratio-band window with provenance could be derived — recorded as an honest STOP, gates unchanged. **PRE-2.0 (#931).** The "#740 one-plane-ground realization" that sentence names no longer exists: from 2.0 a conductor is a volume, a sheet or a wire by declaration, a one-cell PEC Box realizes walls at both drawn faces, and cv15's ground is declared as a sheet on the substrate floor. The cited `rfx_one_plane_ground_b29f9de7.json` figure stays as the dated record of what the old realization cost; it is not reproducible on the current tree, and the case's negative control is now a wrong DECLARED ground plane rather than a realization flag. The 8 % f0 gate's blindness is a property of the gate and is unchanged by any of that. Patch accuracy is not established here; patch evidence remains with the committed tests the former cv05 patch case named (regression locks, not accuracy claims). |

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

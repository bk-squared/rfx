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
| `crossval/20_msl_phase_referee.py` | MSL de-embedded phase (`angle(S21)`) vs an external OpenEMS referee, matched-mesh (`dx=50um`) RO4350B thru line | OpenEMS `MSL_NotchFilter.py` tutorial reproduce-gate plus a matched-mesh thru-line comparison against rfx's own `compute_msl_s_matrix`; resolves whether rfx's previously un-gated MSL phase was a time-convention or reference-plane mismatch (both tools' DFT kernels are identical at the source — it was reference-plane, resolved by matching the measurement plane to rfx's own probe-0 coordinate). Gated: each solver's own `angle(S21)` against its own measured `beta` (intra-run self-consistency, 3.0°/200 ps) over 3.0–4.5 GHz; each solver's measured `beta` against the Hammerstad–Jensen quasi-TEM closed form of the realized board (2.0%; measured 0.94% rfx, 0.31% openEMS — pre-2.0 realization; re-measured under #931); and — since 2026-09-01, issue #812 — the **raw** cross-solver `angle(S21)` difference itself (3.0°; historical pre-#931 run-2 pair measured 0.342°), which was previously reported only. The committed replay with the current rfx leg and historical run-2 OpenEMS leg reads 0.5308° raw phase difference and analytic-beta errors of 1.4122% rfx / 0.3068% OpenEMS (`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv20.run2_openems_with_current_rfx_fixture.cross_solver_max_abs_raw_phase_diff_deg = 0.5308`); this is not a fresh matched-board OpenEMS solve after #931, which remains outstanding. The self-consistency leg alone cannot fail on a coherent phase-velocity error: a factor-2 error reads `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv20.blindness.audit_construction_e1_max_phase_dev_deg = 0.0647` degrees against its 3° gate. The 500 MHz bin is excluded — the line is ~0.03 guide wavelengths there, where port extraction is ill-conditioned. Reported, not gated: the dispersion-corrected residual, which is provably blind to the same error class. The `CalcPort` reference-plane-referral rotation is unexercised on this fixture and untested off-grid. Applies to one geometry, one mesh, single drive. |

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

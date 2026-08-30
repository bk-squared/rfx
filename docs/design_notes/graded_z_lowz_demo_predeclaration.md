# graded-z efficiency demo — wide low-Z escape hatch: PRE-DECLARATION

Date: 2026-08-29 (written and committed BEFORE any FDTD run)
Item: backlog `graded-z` — the ONE permitted R2-tight attempt (STOP fence:
`2026-07-22_todo_plan_inventory.md:33`, RA-5 in
`20260713_research_agenda_principle_based.md`; ledger entry
`rfx-known-issues.md` "Graded-z efficiency demo ... honestly deferred").
Author agent branch: `agent/graded-z-lowz-demo`.

## Question under test

Does the PRODUCTION non-uniform-z profile machinery
(`rfx.auto_config._make_dz_profile`, called exactly as `auto_configure`
step 4 calls it — thirds rule + `smooth_grading(max_ratio=1.3)`, default
`min_cells_per_feature=4`, NO hand-tuned profile) deliver a z-cell saving
on a wide low-Z microstrip (the documented escape hatch: W ≈ 6·h_sub,
~20–25 Ω, dx = W/8, so h_sub/4 < dx) while preserving the committed MSL
accuracy envelope?

## Fixture (all absolute physical coordinates)

Identical physical geometry in both arms; the committed thru-line fixture
class (`tests/test_msl_port_integration.py::test_msl_thru_line_passive_gate`,
`scripts/diagnostics/msl_thru_mesh_convergence.py`), re-dimensioned to the
escape-hatch line:

- Substrate: eps_r = 3.38 (RO4003C-class, lossless), h_sub = 254 µm,
  filling z ∈ [0, 254 µm] over the whole board.
- Trace: PEC, W = 6·h_sub = 1524 µm wide, full line length, thickness
  h_sub/4 = 63.5 µm (declared physically, same in both arms), i.e.
  Box (0, y_c−W/2, 254 µm) → (lx, y_c+W/2, 317.5 µm).
- Line: L = 10 mm, port margin 2 mm each end → lx = 14 mm.
- ly = W + 2·(2·h_sub + 8·dx) = 5.588 mm (committed fixed-clearance rule).
- Nominal z column: h_sub + 1.5 mm air (committed LZ rule); the graded
  arm's realized column is whatever the production profile sums to.
- Boundary: BoundarySpec(x=cpml, y=cpml, z=(lo=pec ground, hi=cpml)),
  cpml_layers = 8.
- In-plane cell: dx = W/8 = 190.5 µm (the escape hatch's declared rule).
- Ports: `add_msl_port` (production laplace mode), width=1524 µm,
  height=254 µm, at (2.0 mm, y_c, 0) direction +x (driven and matched) and
  (12.0 mm, y_c, 0) direction −x; port impedance = analytic
  Hammerstad-Jensen Z0 of the line (matched-thru demo; analytic anchor,
  not a tuned value).
- Analytic anchors (rfx.microstrip, computed pre-run):
  Z0_HJ = 25.436 Ω, eps_eff_HJ = 2.877.
- Band: freq_max = 5 GHz, n_freqs = 30 (default grid 0.5–5 GHz);
  GATE BAND 3.0–4.5 GHz (committed thru-gate window).
- num_periods = 12 (committed CPU thru-gate precedent), preflight ON and
  fully surfaced, `enforce_passivity=False` (issue #470: diagnostics read
  the RAW extraction), raw phasor dump enabled for the V·I checks.

## The two arms

- Arm A (uniform-z baseline): `dz_profile = full(nz_A, h_sub/4)` —
  the NU code path with uniform fine cells (mirrors the committed NU gate
  test's isolation rationale), nz_A = round(sum(arm-B profile)/(h_sub/4))
  so both arms span the same realized z column to within one cell.
  h_sub/(h_sub/4) = 4 exactly → aligned mesh (the documented
  mesh-alignment rule prefers h_sub/4).
- Arm B (graded-z): `dz_profile = _make_dz_profile(analyze_features(
  geometry, materials).z_features, h_sub + 1.5 mm, dx)` — the production
  machinery verbatim, consumed as-is.

## Metrics

1. z-cell count ratio nz_A/nz_B (same realized column height).
2. z-cost ratio (nz_B/dt_B)/(nz_A/dt_A) — cell-steps per unit simulated
   time attributable to the z axis; dt from the realized grids
   (NU dt = 0.99/(c·sqrt(1/dx_min² + 1/dy_min² + 1/dz_min²))).
3. Per arm: band max|S11|, band mean|S21|, median Re(Z0_extracted),
   eps_eff_extracted = (Re(β)·c/ω)² over the gate band.
4. Arm agreement: per-bin |S11| and |S21| differences over the gate band
   (reported as witnesses; each arm is gated ABSOLUTELY per the committed
   NU-gate rationale — "NU is validated against physics on its own grid",
   never bit-for-bit against the sibling grid).
5. Sub-item (b) (todo-inventory): S11 re-referenced to the EXTRACTED Z0,
   S11_reref = (Zin − Z0_ext)/(Zin + Z0_ext) with Zin = V0/I0 from the raw
   probe-0 phasors — reported as a witness (no committed threshold exists;
   it is not gated).

## FALSIFIERS (declared before any run; thresholds are the documented ones)

Accuracy (evaluated per arm over the 3.0–4.5 GHz gate band, on the raw
extraction, on bins passing the documented `np.all(reliable, axis=0)`
screen):

- F1: max|S11| ≥ 0.10 → FAIL.
  [Committed thru envelope `thru_max_s11=0.10`,
  `scripts/diagnostics/build_msl_broad_e5_envelope.py` THRESH, pinned by
  `tests/test_msl_broad_e5_envelope_gates.py`.]
- F2: mean|S21| ≤ 0.95 → FAIL. [Same envelope, `thru_mean_s21_min=0.95`.]
- F3: |median Re(Z0_ext) − Z0_HJ| / Z0_HJ ≥ 0.05 → FAIL.
  [Same envelope, `thru_z0_rel_err_med=0.05`. Context: the documented
  aligned-mesh Z0 bias at h_sub/4 is −3.8% (PR #535 sweep) — the gate has
  real headroom only on an aligned mesh.]
- F4: eps_eff_ext outside (1.0, 3.38) anywhere in the gate band → FAIL
  (hard physical bound for a quasi-TEM microstrip mode, not a tolerance).

Efficiency (deterministic, from the realized grids; recorded at run time):

- F5: z-cell ratio nz_A/nz_B < 2.0 → the ledger's "~3× z-savings" claim
  FAILS. [Threshold set BELOW the documented ~3× claim; not generous.]
- F6: z-cost ratio (nz_B/dt_B)/(nz_A/dt_A) ≥ 1.0 → the graded arm is not
  cheaper than the uniform-fine-z arm at equal xy mesh → the efficiency
  claim FAILS on its own terms.

Boundedness protocol (ledger |S11|≤1 ⇔ Re(V/I)≥0 theorem, DO-NOT-REPEAT
entry): any |S11| > 1 bin is interpreted ONLY after checking the sign of
Re(V0/I0) at that port's probe plane. Re < 0 ⇒ resolution-bound
measurement artifact — reported as such; NO reference/geometry "fix" is
attempted (that class of fix was debunked 4×).

Never re-anchored: the gate set above is fixed before the run. In
particular no gate is ever re-specified onto a spectral feature found
after the fact (the "never dip-at-9.3" fence, issue #118 category error).

## Validity preconditions (distinct from falsifiers)

- `sim.preflight()` runs for both arms; every advisory is printed into the
  log and this note's results section. Never suppressed.
- Settling: every drive's `settling_db` ≤ −40 dB
  (docs/guides/simulation_methodology.md ring-down rule). An unsettled
  record is NOT interpreted and does NOT consume the attempt: the
  escalation is a VESSL YAML for both arms (design_only outcome).
- Reliability: gate-band bins failing `np.all(reliable, axis=0)` are
  excluded from band statistics and counted in the log; if the ENTIRE gate
  band is unreliable the measurement is invalid (escalate, not falsify).

## Attempt-spend rule

The one R2-tight attempt is SPENT the moment falsifier verdicts F1–F6 are
evaluated on a valid record (F5/F6 are deterministic mesh facts and are
recorded regardless). After any verdict: no fixture change, no parameter
change, no rerun. A fired falsifier closes the demo with an honest
closure note.

## Pilot (declared)

One timing-only pilot of arm B at num_periods=2 to project wallclock.
Its S/Z0 outputs are not read or recorded — wallclock only. If the
projected full run exceeds ~20 min CPU, emit the VESSL YAML instead of
running (attempt stays live).

## Design-time predictions (deterministic meshing arithmetic, recorded
BEFORE the run so the run cannot be accused of hindsight)

Computed from the production profile for this fixture (no FDTD involved):

- Graded profile: nz_B = 24 cells, sum = 2.380 mm (the machinery INFLATES
  the nominal 1.754 mm column: `smooth_grading` inserts transition cells
  without preserving total length — the documented issue-#48 behavior,
  pinned by `tests/test_smooth_grading_preserve.py`, and
  `_make_dz_profile` does not pass `preserve_regions`).
- dz_min = 21.17 µm (the thirds-rule 1/3-cell) → dt_B = 6.905e-14 s vs
  dt_A = 1.897e-13 s: the graded arm needs 2.75× more steps.
- The substrate top (254 µm) falls MID-CELL at fraction 0.346 of the
  cell (239.35 → 281.68 µm) — inside preflight's documented mixed-cell
  danger zone [0.10, 0.40] — because the smoothing insertion inside the
  substrate block destroys the exact snap `_make_dz_profile` constructs.
- Predicted F5: nz_A/nz_B = 37/24 = 1.54 < 2.0 → F5 expected to FIRE.
- Predicted F6: z-cost ratio = 1.78 ≥ 1.0 → F6 expected to FIRE.

These predictions, if confirmed by the realized grids, falsify the
efficiency half of the demo at the meshing level. The FDTD run still
executes (it is the same single attempt) to record the ACCURACY half —
whether the production graded profile even preserves the committed
envelope — because that is durable evidence for the NU lane either way.

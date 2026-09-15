# RECOMPUTE — section T (tests/crossval, chain battery, gate policy), #931

Branch `feat/931-t5-crossval-tests`; worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T5-crossval-tests`.

## Submitted by section T

| case | VESSL run id | yaml | command inside the job | expected runtime |
|---|---|---|---|---|
| WR-90 chain battery, post-change re-measure | **369367259196** | `scripts/vessl_931/T/rfx-931-post-chain-battery.yaml` (preset `gpu-rtx4090`, `JAX_PLATFORMS=cpu`) | `python scripts/diagnostics/waveguide_chain_battery_measure.py --out-dir $OUT --stages {cells,ad_fd,plane_shift} ... ` then `--stages assemble --fixture-out $OUT/fixture.json` | ~3-6 h wall (run 3 used 4 h + 4 h + 1 h timeouts; three rungs x three DUTs x two lanes, then the AD/FD and plane-shift legs) |

Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-chain-battery-<ts>/`,
with `issue931-post-chain-battery.latest` naming the directory. The job reads
the working tree of this worktree over NFS, so the commit it measures is
whatever is checked out when it starts; `commit.txt` in the output records it.

**Why it must run.** Two things changed for this battery and neither can be
re-pinned by hand:

1. `compute_waveguide_s_matrix` no longer folds `pec_mask` into `sigma = 1e10`;
   it applies the realized PEC edges (`rfx/api/_sparams.py`, "#931 §1.7: the
   interior PEC of this lane is the realized edge set"). So
   `design_override(kind="sigma")` no longer carries the fold — re-stamping it
   would put a lossy volume on top of an already-shorted conductor — and the
   `("pec_short", "sigma")` AD leg now differentiates through a **different
   quantity**. Its gradient is not the old gradient scaled; it is a different
   measurement.
2. The 2-cell PEC short gains a wall at its far face (`x = 63.50 mm`). The
   reflecting near face is unchanged, so `|S11| ~ 1` and
   `pec_short_phase_oracle_deg` should hold — but "should" is the word the
   contract exists to remove, so it is measured.

**Pre-declared, before the run.** The `thru` and `slab` legs must come back
unchanged to within their existing tolerances: neither declares a conductor,
and the absorber depth is unchanged (measured build-time: `guide_source` is
`("domain_faces", "domain_faces")` at every rung, `fc_TE10 = 6.55714 GHz`,
`cpml_layers = 17 / 34 / 68`). `referee_pec_short` must stay inside
`[0.99, 1.03)` with mean within 0.02. The `("pec_short", "sigma")` AD leg is
expected to MOVE and is report-only until re-declared. If the thru or slab legs
move, the diagnosis is wrong and the change is not the PEC short.

The job runs the build-time witnesses (`assert_dut_realizes_its_faces`,
`assert_oracle_anchors_are_realized`) before it spends any solve time, so a
geometry regression fails in seconds rather than in hours.

Fixtures are NOT committed from this phase (the ingest phase does that). The
artifact is written to the run directory only.

## Result of run 369367259196 (finished 2026-09-07 11:54 UTC)

Output: `/root/workspace/claude-workspace/rfx/runs/issue931-post-chain-battery-20260907T105351Z/`
(all four stage `.rc` files 0; `fixture.json` 507 kB; 184 verdicts, 6 fail).
The fixture is NOT committed here — the ingest phase decides that. Compare
against `tests/fixtures/waveguide_chain_battery/fixture_guide_cell_aperture.json`,
which is the run-2 config this builder ships (`SHIFT_PAIR_NAME =
sign_discriminating_pair`); `fixture.json` is run 1 and differs for reasons
that have nothing to do with #931.

### Measured against the pre-declaration, item by item

| pre-declared | measured | verdict |
|---|---|---|
| `thru` and `slab` legs unchanged | every S entry of all 12 thru/slab cells agrees to ≤ 1.1e-5 (float32 noise); no verdict moved | **held** |
| `referee_pec_short` inside [0.99, 1.03), mean within 0.02 | coarse\|false min/max/mean |S11| 0.99702 / 1.00235 / 0.99965 (was 0.99702 / 1.00236 / 0.99965) | **held** |
| `|S11|` unchanged — the reflecting NEAR face does not move | max |ΔS11| over the six pec_short cells ≤ 6.8e-6; the implied reference-plane move is 0.00000 mm on every bin | **held** |
| the far face gains a wall | `S22` moves by a phase equal to **exactly one cell**: −2.5263 mm at dx = 2.54, −1.2646 at dx = 1.27, −0.6323 at dx = 0.635 (per-bin spread ≤ 0.07 mm, the Yee β dispersion) | **held, and it is the §5 one-cell witness on a third case** |
| the `("pec_short", "sigma")` AD leg MOVES and is report-only | it did not move: g_AD −6.4283 vs −6.4214, rel(AD, FD) 4.9e-4 against a 0.05 gate, verdict pass. Re-stamping is gone and the gradient is the same quantity to four digits | **prediction wrong, stated as measured** |
| — (not predicted) | the nine `forward_identity|*|flux|*` failures the run-2 fixture carries are all **gone**: the override no longer has to re-carry a sigma fold, so the traced and untraced forwards are the same solve | improvement |
| — (not predicted) | six `ad_vs_fd|pec_short|*|eps|*` legs FAIL with `g_fd = nan` | new red, cause below |

### The six NaN legs: traced, and not the realization change

`f_minus` is NaN; `f_plus`, the AD gradient and the primal are all finite. The
minus arm evaluates the pec_short θ window (vacuum) at `eps_r = 0.95`, and dt
is picked at 0.99 of the `eps_r = 1` Courant limit: `0.99 / sqrt(0.95) =
1.0157`. The FD reference has always been 1.6 % outside the stability limit,
at every rung — `tests/_waveguide_chain_battery_fixture.py::
eps_fd_step_courant_ratio` now reports that number without solving anything
(1.015719 for pec_short, 0.498123 for slab, every rung).

Two arms, measured locally (coarse and mid rungs, x64, θ = −0.05):

| rung | pec_short | thru (NO conductor anywhere) |
|---|---|---|
| coarse, 713 → 8545 steps | bounded, max|S| = 1.00254 | bounded, max|S| = 1.00196 |
| mid, 1425 steps | NaN | **NaN** |

A guide with no conductor in it blows up the same way, so `eps_r < 1` is
sufficient and the ownership contract is not the cause. What the contract
removed is what used to hide it on THIS DUT: the lane no longer folds the
short into a `sigma = 1e10` volume, and that lossy block sat one cell from the
window and damped the growing mode.

Section T did not change `THETA0_EPS` or `FD_STEP_EPS` to make the leg pass.
That is a re-declaration of a measurement the battery's predeclaration fixes
(a central difference AT the shipped fixture), so it is the PI's call; the
options are in `docs/design_notes/931_migration/T-tests-crossval.md` §3.

## Not submitted by section T — owned elsewhere, named so nothing is lost

| case | owner | what section T is waiting for |
|---|---|---|
| `rfx-931-post-cv19` | crossval-D | `19_wr90_iris_filter_aghanim.py --write-fixture` on the migrated builder. Until it lands, six tests in `test_wr90_iris_filter_gates.py` skip with that name, and `_PINS_REPINNED_FOR_931 = False` holds the four hard pins. |
| `rfx-931-post-cv18` | crossval-D | `18_wr90_iris_modematch.py --write-fixture` with the aperture `- 1` deleted and a `t_c = 1` row added (design note §5 one-cell witness). Two tests skip naming it. |
| `rfx-931-post-cv15` | crossval-C | cv15 re-solved on sheet declarations. All eight tests in `test_crossval_cv15_wall_planes.py` skip until the script imports. |
| `rfx-931-post-cv05` | crossval-A | **DONE 2026-09-07 (VESSL 369367259302, phase-2b ingest).** `_ENVELOPES_REDERIVED_FOR_931 = True`; the three slow gates run. `D_ABS_TOL_DB` held at 1.0 (measured 0.0659 dB), `F_RES_REL_LO/HI` +0.06/+0.16 -> -0.02/+0.09 (measured +3.51 %, was +11.3 %), mode-pair band unchanged. The resonance band now straddles zero, so that gate is no longer sign-locked — a change of kind, stated in its docstring. Constants moved to lines 157/163/164. |
| `rfx-931-post-cv06b` | crossval-B | cv06b re-solved with the trace and stub as sheets. The Z0-anchor test skips naming it; the run-log/live width comparison reds if the log is not refreshed with the declaration. |
| `rfx-931-post-cv07`, cv20 Stage B | crossval-B / E | same foil decision; annotated in the test files. |

## Explicitly NOT re-run (measured, contradicts the tests-crossval inventory)

Five GPU re-runs the inventory asked for are unnecessary. Every one of these
builds its metal with the low-level `rasterize(grid, [(shape, 1.0, sigma)])`
CELL FILL, which design note §1.8 fences out of the ownership contract:

* cv16 ka sweep (`validation/crossval/16_pec_sphere_mie_ka_sweep.py:283`)
* RCS Mie reference (`tests/fixtures/rcs_mie_e4`)
* RCS sphere three-way (`tests/fixtures/rcs_sphere_three_way`)
* RCS cube vs Bempp (`tests/fixtures/rcs_cube_bem/generate.py:58`)
* WR-90 T-junction E4/E5
  (`scripts/diagnostics/build_waveguide_tjunction_broad_e5_envelope.py:41-42, 64-65`)

Also not re-run, with the reason traced to the producer: the WR-90 broad-E4/E5
and NU-E4 `pec_short` legs. The short is drawn two cells thick at `x = 145 mm`
on a 1 mm mesh, so both faces are on node planes; the contract adds a wall at
the FAR face only, behind a total reflector, and the reflecting near face is
where it has always been.

## Verification of section T's own files (2026-09-07, this pod)

`JAX_PLATFORMS=cpu pytest -n 4`, the default fast lane (`not gpu and not slow
and not slow_physics`), the 25 changed files run in four batches because the
whole `tests/crossval` directory needs longer than the 20-minute cap this pod
allows:

| batch | result |
|---|---|
| 13 files (coax/pmc/cv09/cv14/cv15/cv06b/cv23/witness/meep/msl x3/cv05) | 230 passed, 10 skipped, 126 s |
| 9 files (patch mode id, five RCS lanes, sheen, NU-E4, T-junction) | 116 passed, 5 s |
| `test_wr90_iris_filter_gates.py` + `test_wr90_iris_modematch_gates.py` | 59 passed, 6 skipped, 651 s |
| `test_waveguide_broad_e5.py` | 36 passed, 20 s |
| the three unchanged crossval files that import section T's helpers (cv22, cv24, NU-E5) | 77 passed, 4 s |

441 passed, 16 skipped, 0 failed. Every skip names the VESSL run that will
un-skip it (`rfx-931-post-cv05/06b/15/18/19`).

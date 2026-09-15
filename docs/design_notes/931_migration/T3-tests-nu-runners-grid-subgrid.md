# T3 — tests/unit/{nonuniform,runners,grid,subgrid} under the lattice ownership contract

Branch `feat/931-t3-nu-runners-grid-subgrid`. Inventory: `tests-nu-runners-grid-subgrid.json`
(47 sites), corrected by `critic.json`. This note records the decisions the design
note left to the migrating agent, and the sites that belong to another owner.

## Decisions taken here

The inventory asked the PI four questions. Three are answered by the design
note as it stands; the fourth is answered by the branch. All four are written
into the fixtures themselves so the next reader does not have to re-derive them.

1. **Does the ownership change reach MATERIAL Box sampling?** No. §1.8:
   "Dielectric sampling (node, half-open) … unchanged". So
   `test_vmap_sweep_dft_planes.py::test_exact_hi_face_touch_matches_run_via_shared_fallback`
   keeps its hi-face-drop assertion, and the #627 CPML pad fallback and #655
   node repair stay live. The fixture says it is fenced, so the assert is not
   mistaken for an oversight. rfx does carry two conventions for one `Box`
   primitive — centre-sampled for a PEC volume, node half-open for a
   dielectric — and §1.1 states that on purpose (they coincide for
   node-aligned corners).

2. **Is a high-sigma MATERIAL box a conductor under the contract?** Two
   different answers, and the difference is the API level:
   * `sim.add(shape, material=<sigma above threshold>)` goes through
     `classify_pec_entry` and IS a declared conductor — that is the WR-90
     iris in `test_nonuniform_pec_scatterer_limit.py` (`sigma=1e7`);
   * a raw `sigma` stamp into the material arrays — `box.mask(grid)` +
     `jnp.where(mask, 1e10, sigma)` in `test_simulation.py`'s T-junction, and
     `rasterize(grid, [(plate, 1.0, PEC_SIGMA)])` in
     `test_x64_scan_carry_dtypes.py` — is a sigma FILL, fenced out by §1.8 as
     a lossy volume model.
   Both files now say which they are. The T-junction's declared `a = 0.04`
   against a realized 0.042 channel is the #868 class and stays open under
   the follow-up issue §1.8 names; it is not repaired here and not papered
   over.

3. **Does (S) cover LOSSY thin conductors?** No. §1.8 fences the DC fold
   (`sigma_bulk < 1e6`, no `f0`) out as a lossy volume model. So
   `test_vmap_sweep.py::_make_cpml_sim`'s conductor keeps its finite 2x2-cell
   box, `test_thin_conductor_fixture_is_live`'s `sigma_eff = 175 S/m`
   assertion stands, and the calibrated `n_steps = 200` gate needs no
   re-measurement. The word "sheet" is removed from those docstrings: the box
   is two cells thick in BOTH y and z, so it has no single normal axis and
   was never a sheet — only the pre-#931 rule made it look like one.

4. **Are DOMAIN FACES inside the realized-edge-set function?** No, §1.8.
   `test_adi.py::test_adi_cavity_resonance`'s `a_eff = (Nx-1)*dx` and
   `test_distributed.py::TestLegacyPmapPadXFaceAppliers` are domain-boundary
   PEC and unchanged. Both now say so, because a domain wall and a body wall
   really are two mechanisms and the compensation in the ADI oracle looks
   exactly like the body-rule compensations the contract abolishes.

Two more the design note did not raise:

5. **`conformal_pec` is not a realization knob.** §1.5 forbids per-ENTRY
   realization keywords; `conformal_pec` is a run-level subpixel model
   (Dey-Mittra) layered on the realized edge set and fenced by §1.8, so it is
   not in the `two_plane` deletion list. It is still honoured on the uniform
   lane and warn-dropped on NU/subgrid/distributed, i.e. the lanes agree
   about realization and disagree about subpixel treatment.
   `test_silent_drop_warnings.py` records the distinction.

6. **`pec_mask_override` is a VOLUME override.** So the geom|override union
   in `test_distributed_nu_kernel.py` is a union of cell occupancies and
   stays a boolean OR. A sheet cannot be expressed through it; the
   distributed-NU forward lane refuses one loudly instead
   (`test_distributed_nu_pec_mask_lane_parity.py`).

## Sites that are another owner's

* `rfx/api/_preflight.py` — the "declared conductor realizes no wall plane"
  ERROR (inventory MISSING GATE 1) is preflight's, owner **P**. The test-side
  half is done: the two fixtures in this group that declared a conductor and
  realized NOTHING (`test_nu_progress_chunking.py`, the two zero-thickness
  thin conductors in `test_auto_config.py`) now assert a realized wall plane
  at build time, so they can no longer pass vacuous.
* `test_inplane_grading_guards.py`'s under-resolution advisory sentence ("a
  1.5 mm PEC volume in a 500 um-cell region is 3 cells across") is scored by
  preflight from the body. The fixture is redrawn on nodes and asserts the
  realized span; whether preflight scores it from realized walls or from the
  Box mask is owner **P**'s call. The assertion here is qualitative and fires
  either way.

  **Finding for owner P, measured while redrawing it.**
  `_validate_mesh_quality`'s `_local_cell` resolves a body whose face lands
  exactly on a band boundary to the FINER neighbouring band. On the fixture's
  profile (`[12x250um, 8x500um, 8x125um, 12x250um]`, coarse band
  `[3.0, 7.0) mm`) the same 3-coarse-cell PEC volume scores:

      drawn 3.0 -> 4.5 mm (lo face ON the band's first node): 0 advisories
      drawn 3.5 -> 5.0 mm (one coarse cell inside):           2 advisories

  Same body, same 3 cells, same realized-vs-drawn span. The 3.0 mm case is
  scored against the 250 um cell on the fine side of the boundary — the
  exact "a body in a coarse region judged by a fine cell it never sees"
  failure the #743 check exists to prevent, reappearing at the band edge.
  The fixture is drawn one coarse cell inside the band so it keeps testing
  what it names; the tie itself is NOT worked around and is not fixed here.
* `validation/crossval/05_patch_antenna.py` and
  `scripts/diagnostics/patch_tutorial_rfx.py` — owner **X-A**.
  `test_patch_uniform_fine_substrate.py::build_uniform_fine_z` is the
  grid-build lock those two imitate, and it has dropped its two reserved
  metal cells here (see below). The live cv05 migration is X-A's.
* `tests/unit/sparams/test_settling_witness.py` keeps its own copy of the
  `_msl_thru` geometry migrated in `test_run_progress_reporting.py`; owner
  **T (sparams group)**.
* `rfx/surrogate.py::export_geometry_sdf` iterates `sim._geometry` only, so a
  sheet declared through `add_thin_conductor` never reaches it and vanishes
  from exported training data with no error. A zero-thickness Box DOES
  register (the exporter's containment test is closed), which is what
  `test_amr_surrogate.py` now pins. **The `add_thin_conductor` gap is a real
  defect and is not fixed here — `rfx/surrogate.py` is nobody's file in this
  phase.**

## Findings — measured on this branch, NOT fixed here

### 1. The distributed-NU lane mis-realizes a body on the seam cell

**Owner: `rfx/runners/distributed_nu.py`.** Exposed by the contract, not
caused by it.

Two fixtures in `test_distributed_nu_kernel.py` drew a PEC body three cells
wide across the slab seam, with the comment "so the tangential mask in
apply_pec_mask sees a PEC neighbour (otherwise the thin-sheet rule preserves
the field — no double-zeroing risk to detect)". That padding existed only to
make the pre-#931 neighbour rule fire. Under §1.2 one occupied cell owns
every edge incident to it, so the fixtures were reduced to the single seam
cell they are named for — and the lanes then disagree.

Class B final-step relative error, 16x8x8 grid, 2 ranks, 30 steps, gate
`RTOL_PROBE_B = 5e-5`, one PEC cell at `(i, ny//2, nz//2)`:

| body | rel. error |
|---|---|
| one cell AT the seam | **2.107e-01** |
| one cell 3 inside rank 1 | 0.000e+00 |
| one cell 3 inside rank 0 | 1.729e-05 |
| three cells straddling the seam (the old fixture) | 2.131e-06 |

Pre-declared falsifier before the measurement: *if the same one-cell body
placed away from the seam lands inside the gate while the seam placement
does not, the divergence is seam-specific.* It does; it is. The three-cell
row is why the defect was invisible — with a real occupied cell on both
sides of the seam, every edge is owned by a rank that holds it as a real
cell, so the ghost convention is never asked the hard question.

The hard-mask and soft-occupancy twins fail with the same magnitude, so it
is one root cause, in the shard/ghost handling rather than in the rule
(`test_distributed_nu_pec_mask_lane_parity.py`'s edge-count battery, which
applies the masks to a static field without a halo exchange, passes on the
same one-cell bodies).

Both tests carry `pytest.mark.xfail(strict=True)` with this table as the
reason. `strict` is the point: when the lane is fixed the marker turns red
and has to be deleted, and nobody can quietly restore the three-cell
padding instead.

**CLOSED 2026-09-07 (ingest).** The lane was fixed on
`feat/931-core-distributed-seam` and merged into the base at `b096d464`
(fix `ac782d4f`). Root cause: the scan body exchanged the E ghost rows
BEFORE the PEC stages, and those stages act on real cells only, so a body
whose cell is rank 1's FIRST real cell had its seam-plane Ey/Ez zeroed only
in rank 1's copy — rank 0's ghost copy was taken before that, and rank 0's
next H update at its last real cell read the stale plane. The exchange is
now the last stage of the E half-step, matching the placement the H half
already gives the PMC face. That is why the three-cell fixture never saw
it: with body cells on both sides of the seam the stale read sits inside
the body and feeds only PEC edges.

The strict markers did what they were for — they went red on the fix and
were deleted in the same commit that made them pass. The one-cell fixtures
stay one cell; `strict` is what stopped the padding from coming back
instead.

### 2. `test_runner_import_binding.py::test_coax_then_refplane_order_does_not_leak_fake_run`

Red in the slow lane only (`-m ""`), not caused by #931, not fixed here.
The nested pytest it launches PASSES (`83 passed, 7 deselected`, returncode
0); the test then asserts `"failed" not in result.stdout`, and the substring
appears inside a warning emitted by
`solve_two_port_from_wave_amplitudes` — "…or one drive that **failed** to
excite". A summary-line check instead of a whole-stdout substring check
would fix it. Left alone because the brittleness predates this branch and
the warning text belongs to another owner.

### 3. The MSL thru fixture does not witness its own trace

Measured on this branch while reading the re-measured `Z0` / `beta` for
`test_run_progress_reporting.py::_msl_thru`. NOT caused by #931, NOT fixed
here, and no gate in this group moves — every assertion in that file
compares two runs of the SAME fixture. It is recorded because the migration
puts a SHEET where a one-cell PEC Box was, and the obvious question — did
the trace survive the change — turns out to be unanswerable with this
fixture.

`compute_msl_s_matrix(freqs = 2-18 GHz, num_periods = 6)`, dx = 0.2 mm,
eps_r 2.2 substrate 0.8 mm, ports at x = 2 and 10 mm. Logs:
`/root/workspace/claude-workspace/rfx/runs/issue931-post-t3-msl-witness-20260907T141743Z/`
(`three_traces.log`, `shorting_wall_control.log`), local CPU runs:

| declared trace | \|S21\| @10.7 GHz | beta[6] | Re Z0[6] |
|---|---|---|---|
| 1.2 mm sheet on z = 0.8 mm (the migrated fixture) | 0.999992 | 196.064 | 59.72 |
| 3.6 mm sheet, same plane | 0.999992 | 196.064 | 97.02 |
| 1.2 mm one-cell PEC Box (walls at 0.8 and 1.0 mm) | 0.999992 | 196.064 | 53.28 |

`max|dS|` 1.4e-07 between the two sheet widths and 4.2e-07 between the
sheet and the volume — float32 noise, on a matrix reading `|S11|` exactly
0.000000. Only `Z0` moves, and it moves with the REALIZED width, so it is
read off the geometry rather than off the fields.

**The lane is not blind — the control says so.** Pre-declared before the
run: *if a conductor the fixture cannot ignore also leaves S unmoved, the
S-matrix path is not reading the realized geometry at all; if it moves S,
the trace insensitivity is the fixture's, not the lane's.* A PEC Box drawn
across the whole guide at mid-line takes `|S21|` from 0.999992 to
**0.000000** (`max|dS| = 1.000`). So the realized edges do reach this
solve; what this fixture cannot resolve is its own trace, which for a
matched thru changes neither `|S11| = 0` nor `|S21| = 1`.

Two known mechanisms are enough to explain the saturation and neither is
new here:

* `compute_msl_s_matrix` projects S onto the passive set by default, and
  its own docstring warns the projection "is NOT small where the raw
  extraction is bad" (it names a thru whose raw sigma_max ran 1.19-1.91).
  Whether `S_raw` separates the three declarations is UNMEASURED — that is
  the one cheap check left for whoever owns the extractor;
* `beta/k0` = 0.872 on 8 of the 12 bins (eps_eff 0.76, below vacuum) and
  does not move with `num_periods` 6 -> 20, while `Re Z0` swings -4533 to
  98 ohm across the band. This is *consistent with* `rfx-known-issues.md`
  2026-08-20, "MSL extractor reports non-physical Z0 / eps_eff on a real
  board — DIAGNOSTIC-ONLY (does NOT reach S11)": same extractor, same model
  class, same non-physical eps_eff. Not a new defect and not a #931 one.

**One thing this measurement settles positively**, and the critic's
`critic.json` listed it as blocking ~18 rows in other groups: the MSL trace
detector no longer refuses a sheet. Removing the trace from the fixture
raises `compute_msl_s_matrix: no realized PEC trace conductor found above
the substrate top for MSL port 'p1'` — the message reads *realized*, and a
sheet-declared trace passes it. The "declare the MSL trace as a sheet"
migrations elsewhere are unblocked.

**Still open, for the stage that owns the solver lanes**: every one of
these runs printed `_assemble_materials (uniform lane): PEC sheets/wires
were classified but the caller passed no pec_sheets/pec_wires collector, so
they are absent from the returned pec_mask`. That is the design note's own
§6 out-parameter warning firing on a production path, not a test artifact.


## Recompute

See `RECOMPUTE.md` beside this file.

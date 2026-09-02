# THRU singular-value dx ladder — pre-declaration (issue #819, candidate 1)

Status: PRE-DECLARATION, committed before any rung is run. One attempt under
R2 at the RF/EM threshold (N >= 1): this note names the falsifier, the outcome
table and the validity gates once; none of them is widened afterwards. The
lane that runs it does not interpret the result — harvesting is a separate
round with its own note. Append-only.

Governing plan: `docs/design_notes/v18_waveguide_s_chain_plan.md`, section
"WP8 — lumped/wire tail". Issue #819 stays open through this lane.

## 1. What is measured

Fixture: the 2-port wire THRU of
`tests/test_lumped_twoport_vi_validation_battery.py::_build_thru` — air
microstrip, trace width 5 mm at height 1 mm over a `pec_faces` ground, ports
at x = 8 mm and 24 mm (16 mm line), dx = 0.5 mm, CPML 8 layers on x/y and
z-hi, 4000 steps, Gaussian pulse f0 = 5 GHz bandwidth 0.8, nine bins 3–7 GHz.

Recorded state (the battery's `test_thru_passivity_singular_values`
docstring, 2026-08-29): the largest singular value of the 2x2 S-matrix is
1.003227 at 3 GHz and falls monotonically to 0.9874 at 7 GHz. f64 fields,
2x/4x step counts and complex128 algebra all reproduce the excess; the
mechanism is unidentified. Gate `_THRU_MAX_SINGULAR_VALUE = 1.01` and every
measured number stay as they are.

Observable: `sv(f)` = largest singular value of `S[:, :, f]`. The primary
quantity is the excess at 3 GHz,

    e(dx) = sv(3 GHz; dx) - 1,

recorded at e1 = 0.003227 for dx = 0.5 mm. Every bin's sv(f) is recorded at
every rung; the outcome table below reads only e(dx).

## 2. The ladder — what changes and what is held

| quantity | dx | dx/2 | dx/4 | rule |
|---|---|---|---|---|
| cell size | 0.5 mm | 0.25 mm | 0.125 mm | the only intended variable |
| `cpml_layers` | 8 | 16 | 32 | physical absorber thickness 4.0 mm held |
| `n_steps` | 4000 | 8000 | 16000 | physical run time T = 4000·dt(0.5 mm) held; dt = 0.99·dx/(c·sqrt 3) is linear in dx, so the step count doubles per halving (computed from the grid's own dt, both recorded) |
| trace thickness | 1 cell | 1 cell | 1 cell | sheet realization at every rung, see 2.1 |
| trace overhang past each port column | 0.5 mm | 0.5 mm | 0.5 mm | physical (1, 2, 4 cells — the ">= 1 cell" rasterization rule of the fixture docstring holds at every rung) |
| port `extent` | 1.0 mm | 1.0 mm | 1.0 mm | physical; rasterizes to 3 / 5 / 9 cells, top cell inside the sheet (dead by the Ampere quench), n_live 2 / 4 / 8 |
| domain, width, height, port x, Z0 = 50 ohm, pulse, bins, boundary spec, precision (x64 OFF, default field dtype) | held | held | held | verbatim from the fixture |

Extractor: `sim.run(n_steps, compute_s_params=True, s_param_freqs=...)`, the
fixture's own call, which routes the 2-port wire set through
`rfx/probes/sparam_driver.py::compute_lumped_wire_s_matrix_via_scan`
(`rfx/runners/uniform.py`, the multi-port wire branch). No reference-plane
opt-in, no de-embedding, no normalization change.

At the dx rung the built geometry is the battery fixture byte for byte
(overhang 0.5 mm = 1 cell, sheet = 1 cell, 8 layers, 4000 steps); gate G1
below checks that the number reproduces.

### 2.1 Two cell-unit lengths and why the sheet is kept

The fixture states two lengths in cells, not metres: the trace thickness
(`H` to `H + dx`) and the overhang (`X1 - dx`, `X2 + dx`). A refinement
study should hold the physical object fixed, so both need a decision.

Trace thickness — kept at ONE cell at every rung. Memory
`project_thin_pec_sheet_live_ez_edge` (2026-08-19): a one-cell PEC sheet
leaves its own Ez edge live (`apply_pec_mask` shorts Ez only where
`pec & roll(pec, +/-1, z)`), while a slab of two or more cells shorts Ez
inside it. Holding 0.5 mm of physical thickness would make the dx/2 rung a
two-cell slab, flipping the operator class between the first two rungs; the
ladder would then measure that flip and the discretization at once. Keeping
a sheet keeps one operator class. The price, named here: the physical strip
thickness shrinks with dx (t/h = 0.5 → 0.25 → 0.125), a small change in the
effective strip width and hence in Zc. It is a smooth perturbation, not a
class change, and it is the confounder this design accepts.

Overhang — made physical, 0.5 mm. It is line length beyond the feed post,
not an operator choice; its only constraint is >= 1 cell, which 0.5 mm
satisfies at every rung.

The slab variant (0.5 mm thickness held, operator class flips) is the named
alternative if this ladder ends non-closing. It is not run here.

## 3. Outcome table — written before any rung runs

Let e1, e2, e4 be the 3 GHz excess at dx, dx/2, dx/4. Residual floor:
`floor = 1e-5` (four times the recorded f32-vs-f64 spread of 2.5e-6 on this
number). Every comparison below uses `max(|e|, floor)` in log space, as the
R2 numeric rule prescribes; a sign change of e at any rung is a change, not a
reduction.

| outcome | condition | what follows |
|---|---|---|
| A — discretization | e1, e2, e4 all the same sign, and `e1/e2 >= 2` and `e2/e4 >= 2` (each halving at least halves the excess) | the excess is Yee/CPML discretization; fitted order per pair `p = log2(e_coarse/e_fine)`; the envelope is bounded by `e(dx) <= e1 · (dx / 0.5 mm)^p_min` with `p_min` the smaller of the two; the 1.01 gate stays as the dx envelope pin, a finer-dx gate is a separate decision |
| B — discretization refuted | `(max(e) - min(e)) < 0.20 · e1` across the three rungs (less than 20 % change) | mesh resolution is not the mechanism; candidate 2 of the issue (same extractor, a different port family) is the next single attempt and needs its own pre-declaration |
| C — non-closing | anything else: a sign flip, a non-monotone e, a reduction on only one pair, a ratio between 1.25 and 2 | STOP. Re-read memory, list architectures, write the redesign before any further run. No third mechanism without a new written falsifier |

Expected value: not known. This is a measurement, not a prediction; no
outcome is favoured here, and the existing gate pins the envelope until the
harvest round records what landed.

## 4. Validity gates — a failure makes the ladder uninterpretable, it is not a physics result

- G1 (fixture identity): the dx rung reproduces `sv_max = 1.003227` to
  `|delta| < 1e-5` with the three witness probes present (probes are read-only;
  the S path is unchanged, and this gate is the proof), and its preflight
  advisory set contains exactly the battery's three codes
  (`pec_faces_finite_pec`, `wire_port_dead_extent_cells` x 2) plus whatever
  the probes add, quoted verbatim.
- G2 (self-similarity): every rung's preflight advisory multiset contains
  those three codes. A missing `wire_port_dead_extent_cells` means the
  port/sheet relation changed with dx — the rung is invalid.
- G3 (settling): per-drive `settling_db <= -40 dB` at every rung (the repo's
  ring-down rule). A rung above -40 dB is truncation-suspect; the ladder is
  not read before that rung is re-run at double `n_steps` — a run-length
  witness inside the same attempt, not a new mechanism.
- G4 (rasterization): finite-PEC cell count scales 4x per halving (a sheet:
  length x width), wire port cells 3 / 5 / 9, live cells 2 / 4 / 8, CPML
  layers 8 / 16 / 32. Recorded from the assembled mask and the live-cell
  enumerator, not assumed.
- G5 (provenance): every rung's JSON carries git SHA, jax version and
  backend, x64 flag, field dtype, wall time and the verbatim preflight and
  warning text. A rung without these is not a rung.

## 5. Settling witness

Three Ez point probes at z = 0.5 mm (mid-gap): under port 1, mid-line, under
port 2. The production driver's per-drive forward returns the probe record in
its raw dict, so a spy on `Simulation._forward_from_materials` captures it per
driven port without touching the driver. Definition, the same as the MSL
lane's witness in `rfx/api/_sparams.py` (search `_ratio_db`):

    settling_db(drive) = max over probes of 10·log10( mean(E^2 over the last 10 % of the record) / max(E^2) )

The main pass of `run()` (both ports excited at once) is recorded too, as a
fourth number, labelled separately.

## 6. Not done in this lane

No gate or measured number is touched. Nothing is added to the waveguide
battery (this is a lumped/wire fixture, `ROADMAP.md:39`). No result is
interpreted here. Only one VESSL run is launched; the dx/4 rung is never run
on CPU.

## 7. Memory (R1)

- `docs/agent-memory/rfx-known-issues.md`, entry "Added 2026-09-01 —
  #778/#779 stack merged" (the plan cites it at `:137-141` of main 1c38b0d7;
  the ledger has grown since, so the entry is named by heading): "f64 fields,
  4x/2x step counts and complex128 algebra all reproduce 1.003227 (falling
  monotonically to 0.9874 at 7 GHz), so the excess is systematic with
  mechanism unidentified → filed as #819. Gate value (1.01) and every measured
  number unchanged." Consistent — this ladder is that issue's candidate 1,
  and it moves neither the gate nor a measured number.
- same file, entry "Added 2026-07-10 — #313 RESOLVED" (plan `:4278-4306`):
  "Default path and ALL diagonals byte-frozen (bitwise witnesses)"; the
  reference-plane path is opt-in and never de-embeds with omega/c or a
  nominal 50 ohm. Consistent — the ladder runs the default path with no
  opt-in and no de-embedding.
- `feedback_negated_closing_keyword`: "keep the number away from any form of
  fix/close/resolve". Applied — the PR title and body say the issue stays
  open and never pair its number with those words; the study is named by
  its physics.
- `project_thin_pec_sheet_live_ez_edge`: the sheet/slab operator-class
  distinction that decides 2.1.
- `feedback_persist_before_the_optional_stage`: each rung writes its JSON
  before anything else happens in the job; a later rung failing does not
  lose an earlier one.

## 8. R3 pre-launch line

R3: memory=rfx-known-issues.md "#778/#779 stack merged" (#819 entry) +
"#313 RESOLVED" entry + feedback_negated_closing_keyword +
project_thin_pec_sheet_live_ez_edge | R2-attempts=1 (this single pre-declared
attempt on the mesh-resolution mechanism; the three earlier witnesses were
precision and record-length checks, a different mechanism family) |
falsifier=G1 — the dx rung on CPU, run before launch with the witness probes
present, reproduces sv_max 1.003227 within 1e-5; the number and the wall time
are quoted in the PR body.

## 9. RESULTS (appended 2026-09-02 after the single VESSL run; sections 1–8 unchanged)

Run: VESSL 369367257803 (`rfx-thru-sv-dx-ladder`, remilab-c0 gpu-rtx4090,
image nvcr.io/nvidia/jax:24.10-py3, `scripts/vessl_thru_singular_value_dx_ladder.yaml`),
rfx pinned at 08828189 (the PR #858 head; main carries it as 30fad4bb). All
three rungs returned rc=0; each rung's JSON was persisted before the next
rung started. Wall time 6.6 / 12.2 / 160.2 s total for dx / dx/2 / dx/4.

Artifacts:
- committed: `tests/fixtures/thru_singular_value_dx_ladder/rung_dx_over_{1,2,4}.json`
  (byte-identical copies of the run's JSONs; sha256 in that directory's
  README), the adjudication `verdict.json` next to them, and the replay gate
  `tests/test_thru_singular_value_dx_ladder_replay.py` (fast lane, no FDTD:
  re-derives sv(f) from the stored S matrices and the outcome-table verdict
  from the stored excesses, compares both with `verdict.json`).
- originals: `claude-workspace/rfx/runs/thru_sv_dx_ladder/20260902T101202Z-08828189/`
  on the personal-workspaces NFS mount (`rung_dx_over_{1,2,4}.{json,log,rc}`).
- job log: `docs/vessl-logs/rfx-thru-sv-dx-ladder_369367257803_completed.log`
  in the primary checkout (gitignored, local only).

Provenance (every rung JSON, key `provenance`): jax 0.4.33.dev20241023+e3c6d6430,
backend gpu (cuda:0), x64 False, field_dtype null (the default), python 3.10.12.
The CI pin is jax 0.6.2 on CPU; the dx rung on that stack (PR #858 body) gave
sv_max 1.0032274707981712, this GPU run 1.0032274714899068 — a spread of
6.9e-10, four orders below the G1 tolerance.

### 9.1 Validity gates (section 4) — all pass, the ladder is readable

- G1 PASS. dx rung sv_max = 1.0032274714899068 at 3 GHz; delta vs the recorded
  1.003227 = +4.7e-7 (gate 1e-5). Preflight codes exactly
  `[pec_faces_finite_pec, wire_port_dead_extent_cells, wire_port_dead_extent_cells]`,
  `extra_codes = []` — the three witness probes add no advisory.
- G2 PASS. The same three codes at dx/2 and dx/4, `extra_codes = []` at every
  rung. The wire-port advisory text changes only in its counts: n = 3 / 5 / 9
  cells, 1 dead at every rung, n_live 2 / 4 / 8, and the pre-#318 phantom
  termination it quotes 33.3 / 40.0 / 44.4 ohm.
- G3 PASS. `settling_db` per drive (worst probe = `mid_line` at every rung):
  dx −138.3 / −141.8 dB (main pass −140.4); dx/2 −134.7 / −134.8 (main −132.0);
  dx/4 −129.2 / −126.2 (main −133.0). Every value is at least 86 dB below the
  −40 dB rule; no run-length rerun was needed.
- G4 PASS. Finite-PEC cells 340 / 1360 / 5440 (×4.00 per halving), wire-port
  cells 3 / 5 / 9 with the top cell dead, live 2 / 4 / 8, CPML layers 8 / 16 / 32
  (4.0 mm held), grids 81×57×29 / 161×113×57 / 321×225×113.
- G5 PASS. SHA, jax version and backend, x64 flag, field dtype, wall time,
  preflight messages and warnings are in every JSON verbatim.

Preflight, verbatim at the dx rung (three advisories; the same text at the other
rungs with the count substitutions listed under G2, quoted in full in each JSON's
`preflight.messages_verbatim`):

> pec_faces={z_lo} creates an INFINITE PEC boundary AND the geometry contains finite PEC objects. For antennas or finite-GP structures, the pec_faces boundary makes the ground plane cover the entire domain face, which changes the physics (cavity vs radiating antenna). If you need a finite ground plane, remove pec_faces and use an explicit PEC Box instead.

> Wire port at (0.008, 0.01, 0.0) (extent 0.001) rasterizes to n=3 cells of which 1 land inside PEC geometry ['pec'] (n_live/n = 2/3). Dead cells are shorted by the PEC and are excluded from the port's resistance distribution, drive injection, and wave normalization (issue #318 fix): the port terminates at 50 ohm across its 2 live cells. (rfx versions before the #318 fix counted all 3 cells and physically terminated at Z0*(n_live/n) = 33.3 ohm — the issue-#313 finding.) Verify the extent was MEANT to end on/inside the conductor, and keep the midpoint V/I probe cell live; to silence, shorten the extent or move the port so none of its rasterized cells land on PEC (per the assembled geometry -- not a cell-center guess; a thin PEC sheet snaps to its nearest grid NODE, which can be a full cell away from the sheet's midpoint).

> (the same text for the wire port at (0.024, 0.01, 0.0))

Warnings, verbatim, the same set at every rung (`warnings_verbatim`): one
"UserWarning: [run] preflight found 3 advisory issue(s) - pass skip_preflight=True
to suppress:" carrying the three advisories above, and twelve of
"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in
astype is not available, and will be truncated to dtype float32. To enable more
dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell
environment variable. See https://github.com/google/jax#current-gotchas for more."
— the x64-off truncation warning this fixture always emits (PR #858 recorded the
same twelve on CPU).

### 9.2 Outcome table (section 3) applied — verdict C, non-closing

e(dx) = sv(3 GHz) − 1, from `singular_values.excess_3ghz`:

| rung | sv_max at 3 GHz | e | sign |
|---|---|---|---|
| dx = 0.5 mm | 1.0032274714899068 | +3.2274714899e-3 | + |
| dx/2 = 0.25 mm | 1.0003216974938964 | +3.2169749390e-4 | + |
| dx/4 = 0.125 mm | 0.9991541764781098 | −8.4582352189e-4 | − |

Floor 1e-5, comparisons on max(|e|, floor) in log space: e1/e2 = 10.03
(log2 = 3.33); e2/e4 = 0.380 (log2 = −1.39) and the sign changes. |e4| = 8.46e-4
is above the floor, so the floor does not absorb the crossing.

- A (discretization): FALSE. "All the same sign" fails at dx/4; "e2/e4 ≥ 2"
  fails (0.38). The first pair on its own satisfies the A condition (same sign,
  ratio 10.03, fitted p = 3.33), but the table requires both pairs and no
  envelope is quoted from one.
- B (discretization refuted): FALSE. max(e) − min(e) = 4.07e-3 against
  0.20·e1 = 6.45e-4.
- C (non-closing): TRUE — "a sign flip" and "a non-monotone e" both hold.

**Verdict: C — non-closing.** The excess keeps its sign and falls 10.03× on the
first halving (dx → dx/2), then changes sign on the second (dx/2 → dx/4,
|e4| = 8.46e-4 above the 1e-5 floor). The ladder bounds the excess — every
bin at every rung is below the 1.01 gate (max 1.0032275, at dx), and every
bin at dx/4 is below unity — but its order cannot be fitted. STOP; redesign
before any further rung (9.5). Under R2 at the RF/EM threshold this attempt on
the mesh-resolution mechanism is closed; a second needs a named new falsifier
or an identified defect of attempt 1, in writing.

Gate `_THRU_MAX_SINGULAR_VALUE = 1.01` and every measured number stay as they
are. The replay test imports the live constant and asserts it is 1.01.

### 9.3 Every frequency bin (`singular_values.max_per_bin`, `min_per_bin`)

| f (GHz) | sv_max dx | sv_max dx/2 | sv_max dx/4 | sv_min dx | sv_min dx/2 | sv_min dx/4 |
|---|---|---|---|---|---|---|
| 3.0 | 1.0032275 | 1.0003217 | 0.9991542 | 0.9877287 | 0.9921246 | 0.9938213 |
| 3.5 | 1.0029304 | 0.9994327 | 0.9981012 | 0.9824581 | 0.9879052 | 0.9898787 |
| 4.0 | 1.0021866 | 0.9981704 | 0.9967294 | 0.9768735 | 0.9828919 | 0.9849299 |
| 4.5 | 1.0010032 | 0.9966007 | 0.9950798 | 0.9716669 | 0.9776580 | 0.9795427 |
| 5.0 | 0.9993989 | 0.9947540 | 0.9931775 | 0.9675501 | 0.9729531 | 0.9745898 |
| 5.5 | 0.9973740 | 0.9926218 | 0.9909909 | 0.9650987 | 0.9694705 | 0.9708771 |
| 6.0 | 0.9948537 | 0.9901288 | 0.9884380 | 0.9645975 | 0.9676708 | 0.9689025 |
| 6.5 | 0.9916542 | 0.9870927 | 0.9853552 | 0.9659964 | 0.9676088 | 0.9686952 |
| 7.0 | 0.9873984 | 0.9832293 | 0.9814624 | 0.9689178 | 0.9690032 | 0.9698638 |

- sv_max(f) is monotone decreasing in f at every rung (`monotone_decreasing_in_f`
  True ×3).
- The unity crossing moves down in frequency with refinement: between 4.5 and
  5.0 GHz at dx, between 3.0 and 3.5 GHz at dx/2, and at dx/4 no bin exceeds
  unity (max 0.9991542 at 3 GHz).
- The whole sv_max(f) curve moves down with each halving: dx → dx/2 by
  2.91e-3 (3 GHz) … 4.17e-3 (7 GHz); dx/2 → dx/4 by 1.17e-3 … 1.77e-3. The
  ratio of successive differences is 2.49 at 3 GHz and 2.36 at 7 GHz. This
  is arithmetic on the record, not a pre-declared observable, and it decides
  nothing here; 9.5 uses it to name the next falsifier.
- sv_min moves up: 0.98773 → 0.99212 → 0.99382 at 3 GHz.

### 9.4 Column power, reciprocity and |S| across the ladder

- Column power Σ_i |S_ij|² per drive column: max over bins 0.99124 / 0.99251 /
  0.99300, min 0.95635 / 0.95264 / 0.95185 — below 1 everywhere, moving by
  < 2e-3 per halving; the 1.02 plausibility ceilings are never approached. At
  3 GHz the two columns differ by 4.1e-4 / 5.5e-5 / 2.2e-6: the port-1 /
  port-2 asymmetry of the coarse rung vanishes with refinement.
- Reciprocity max_f |S21 − S12|: 2.67e-4 / 5.66e-5 / 1.24e-5, ×4.7 and ×4.6
  per halving. It moved with dx.
- |S11| at 3 GHz 0.0093 / 0.0171 / 0.0344 (|S22| 0.0176 / 0.0176 / 0.0343);
  |S11| at 7 GHz 0.290 / 0.392 / 0.486; |S21| at 7 GHz 0.934 / 0.894 / 0.846.
  The fixture's reflection is not held across the ladder: it grows with
  refinement, by 0.20 in |S11| at 7 GHz from dx to dx/4. Section 2.1 accepted
  one confounder — the sheet thickness t/h = 0.5 → 0.25 → 0.125, called "a
  small change in Zc" — and this record does not bear out "small". Which of
  the thickness, the port's live-cell count (2 / 4 / 8) or something else
  moves |S11| is not measured by this ladder; the slab variant of 2.1 is the
  design that separates the first of them.

### 9.5 STOP — memory re-read, alternative architectures, redesign (R2)

Memory (R1), re-read after the verdict:
- `docs/agent-memory/rfx-known-issues.md`, "#778/#779 stack merged" (#819
  entry): "the excess is systematic with mechanism unidentified". Consistent.
  This ladder adds two facts: the 3 GHz excess is not invariant to dx (10× on
  the first halving), so it is not a fixed extraction offset; and it crosses
  zero, so "excess over unity" is the wrong quantity to fit an order to.
- `feedback_gate_can_bind_artifact`: "a green physics test proves nothing
  about physics until you show the gate can FAIL on wrong physics". The 1.01
  gate was nowhere near binding on any rung; the replay test locks the
  record, not the physics.
- `feedback_label_mechanism_provenance`: "never prescribe a fix for a
  mechanism you have not instrumented". 9.4 names candidates and attributes
  nothing.
- `project_thin_pec_sheet_live_ez_edge`: the reason the sheet was kept (2.1);
  the slab variant stays the named alternative.
- `feedback_quote_the_measure_with_the_number`: each number above is tied to
  its measure (excess over unity, successive difference, |S11|).

Alternative architectures — not parameter tweaks — for the next SINGLE
pre-declared attempt. One of them, its own note, one run:

1. Same fixture and rungs, a different observable. The record shows sv_max(f)
   converging from above to a limit below unity, so an observable that does
   not reference unity can carry an order: the successive difference
   Δ_k(f) = sv(dx_k; f) − sv(dx_k/2; f) per bin, or 1 − sv_min(f), which
   converges from below and never crosses. The named new falsifier is a fourth
   rung dx/8: the two recorded pairs predict Δ(dx/4 → dx/8) ≈ Δ(dx/2 → dx/4) /
   2.4 per bin, and the pre-declaration fixes the ratio window and the floor
   before the run. The recorded pairs supply the prediction; only the new rung
   is evidence. Cost: 65M cells × 32000 steps × 3 passes, about 16× the dx/4
   wall (roughly 45 min on the same GPU) — GPU lane only.
2. The slab variant of 2.1 (0.5 mm physical thickness held; the Ez-shorting
   operator class flips between dx and dx/2 by design). It separates the
   thickness confounder named in 9.4 from the discretization.
3. Candidate 2 of #819 (same extractor, a different port family). It asks
   whether the excess is the extractor's or this geometry's; it does not
   fit an order on this fixture, so it follows, rather than replaces, the
   observable question.

This note recommends 1: it re-uses the three rungs as the prediction basis
and needs one run. The choice is the PI's; nothing is launched here.

### 9.6 Not done in this round

No gate or measured number touched (`_THRU_MAX_SINGULAR_VALUE = 1.01` asserted
by the replay test). No rung re-run. Nothing folded into the waveguide
battery. Issue #819 stays open; this note is its candidate-1 record.

R3: memory=rfx-known-issues.md "#778/#779 stack merged" (#819 entry) +
feedback_gate_can_bind_artifact + feedback_label_mechanism_provenance +
project_thin_pec_sheet_live_ez_edge + feedback_quote_the_measure_with_the_number
| R2-attempts=1 (closed as non-closing; no second run on this mechanism) |
falsifier=`tests/test_thru_singular_value_dx_ladder_replay.py` re-derives sv(f)
from the stored S matrices and the A/B/C verdict from the stored excesses and
compares both with `verdict.json` (ran; passes).

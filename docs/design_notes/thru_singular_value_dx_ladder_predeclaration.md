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

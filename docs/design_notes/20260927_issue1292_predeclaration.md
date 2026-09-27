# Issue 1292: pre-declaration of the measurements

Status: written before the first solve of this campaign, on branch `meas/1292-locked-results` at
`a55ec1e7` (origin/main, 2026-09-27). Once committed it does not change. Anything learned later
that changes a row goes into a dated addendum at the end, written before the affected runs (the
pattern of `coax_chain_battery_predeclaration.md`).

This note fixes references, ladders, observables, expected outcomes and falsifiers. It states no
verdicts. The leader decides each row.

## The question

Commit 76f68f9f (#1213) changed how the Yee grid represents a material interface. An E component
on the interface now takes the mean ε and σ of its four incident cells; before, the cell that
owned the edge decided. According to the ledger entry for #1210, the tangential E edges on a
substrate–air plane used to take ε = 1, which solved the substrate half a cell thin. Five locked
results on structures with a dielectric or lossy interface have been quarantined since. For each
one, the measurement is designed to show whether the current value is closer to the physical
answer, judged against an independent reference, or against the converged value of the same
drawing where no independent reference exists.

## Trees, platform, code

| label | commit | role |
|---|---|---|
| T0 | cd237692 | parent of #1213 |
| T1 | 76f68f9f | #1213 |
| T2 | c0b83539 | parent of #1012 |
| T3 | f7b3270d | #1012 (CPML magnetic half-cell grading) |
| T4 | a55ec1e7 | origin/main at the branch point |

- **Tree sweep.** Each row's own fixture, at the fixture's mesh, is run on T0–T4. It runs on CPU
  (the weekly lane's platform), JAX 0.6.2, numpy 2.2.6, float32. The sweep separates #1213 (T0→T1)
  from #1012 (T2→T3) and from everything else, which answers A2 and A3 per row.
- **Ladders** run on T0 and T4 on GPU (JAX 0.6.2 with CUDA 12, float32). The coarsest rung of
  every ladder that coincides with its fixture also runs in the CPU tree sweep, so the difference
  between the two platforms is measured at one point.
- rfx comes from `git archive <sha>` of each tree and is asserted by import path. The fixture
  builders are imported from main's copies of the test modules. Between T0 and T4 those files
  differ only by the xfail markers of #1291 (checked with `git log cd237692..a55ec1e7 -- <file>`
  for the five test modules, the v173a harness, `tests/_msl_fixture_qualification.py` and
  `tests/unit/sparams/test_msl_port_integration.py`).
- #1278 (213a6f60) changed the automatic MSL probe ladder. It sits between T1 and T2, so it
  moves any fixture whose MSL port auto-resolves its probes (R3's fed arm, R5). The tree sweep
  records each tree's resolved offsets. The R5 ladder pins them explicitly.

Common rules:

- **Realized geometry is read from the built grid and stored in each record**: layer cells, foil
  and sheet planes, patch edge counts, probe planes.
- **Settling.** Every record states its end-of-record level relative to its peak. A frequency
  read from a record that has not fallen below −40 dB is flagged and not used.
- **The v2 bar.** Mesh convergence needs at least three rungs with the trend stated. Curves are
  judged by their trend over frequency. The limits are 2 dB in magnitude, 1 % in frequency and
  1 % in electrical length.
- **Deep nulls (PI ruling, 2026-09-21).** A quantity at or below −20 dB is not compared in dB.
  Inside a null's core the observable is the null's frequency, and a thru's reflection is held to
  the −20 dB bound.
- **Convergence estimate.** Fit `q(h) = q_inf + a·h^p` to the rungs: p is free when there are
  four or more rungs; with three rungs it is Richardson with the observed order. Report `q_inf`,
  `p` and the fit residual. A non-monotone sequence gets no extrapolation; only its trend is
  stated.

## R1: PEC-backed lossy layer

A plane wave at normal incidence hits an ε_r 4, σ 1.4 S/m layer backed by PEC. The layer is drawn
4.69 mm thick. At dx = 0.5 mm the fixture rounds it to 9 cells, 4.50 mm (`position_to_index`,
read without a solve). The observable is |Γ(f)| from the fixture's two-run plateau extractor, read
at the fixture's 21 bins over 6–10 GHz and on a fine 4–12 GHz grid of 161 bins for the null.

**Reference.** The exact 1-D TMM of the PEC-backed layer: the fixture's `_tmm_reflection`, whose
physical limits `test_tmm_oracle_physical_limits` checks. It is evaluated at the drawn thickness,
at the realized thickness and at a best-fit thickness. TMM facts, computed before any solve:

| layer thickness | null frequency | null depth | TMM \|Γ(8 GHz)\| |
|---|---:|---:|---:|
| 4.50 mm | 8.774 GHz | −20.6 dB | 0.155 |
| 4.69 mm | 8.417 GHz | −18.9 dB | 0.132 |
| 4.75 mm | 8.311 GHz | −18.4 dB | 0.130 |

**Ladder.** dx = 0.5, 0.25 and 0.125 mm, with 10, 20 and 40 CPML layers so the absorber stays
5 mm thick. The interface plane (40 mm), the domain and the PEC backing are whole cells at every
rung. Two layers are solved per rung:

- (a) the drawn 4.69 mm layer, realized as 4.50, 4.75 and 4.75 mm (read without a solve);
- (b) a whole-cell 4.50 mm layer of 9, 18 and 36 cells, where drawn equals realized.

Trees T0 and T4. Recorded per run: |Γ| per bin; null frequency (parabolic refinement on the fine
grid) and null depth; the best-fit TMM thickness `d_fit` (least squares on |Γ| over 6–10 GHz with
ε and σ held); the mean and maximum distance to TMM(realized) and to TMM(drawn); the settling of
the lossy run.

**Expected under A1.** Before #1213, the Ez nodes on the layer's front plane took the layer's ε,
which puts the effective front face half a cell ahead of the plane. `d_fit` then comes out near
d_realized + dx/2: for the whole-cell layer, 4.75, 4.625 and 4.5625 mm, an offset that halves per
halving of dx. After #1213 those nodes take (1 + 4)/2, which puts the face on the plane, so
`d_fit` should equal d_realized to well under dx/2 at every rung. On T4 the null frequency should
be within 1 % of TMM(d_realized) at every rung.

**Falsifier.** Either of these shows a defect:

- on T4, `d_fit − d_realized` ≥ 0.25·dx at the finest rung, or not shrinking with dx;
- on T4, the null frequency more than 1 % from TMM(d_realized) at the finest rung.

The fixed bar |Γ(8 GHz)| < 0.15 compares a 4.69 mm drawing with a solve that realizes 4.50 mm.
That mismatch is recorded as a fact next to the numbers; it is not tested.

## R2: copper foil on FR4 (the v173a composition lock)

A 28 × 36 mm PEC foil sheet lies on a 1.5 mm, ε_r 4.3 slab over a PEC ground, in a
50 × 50 × 20 mm box with 4 mm of CPML. An interior Ez current source sits under the foil, with a
probe beside it. The observable is the lock's quantity: the strongest Harminv mode in 1.5–3.5 GHz,
near 1.98 GHz. For scale only, c/(2 · 36 mm · √4.3) = 2.01 GHz.

**Reference.** No closed form reaches 1 % on this truncated board, so cavity-model values are
quoted for scale and never used as the judge. The reference is the mesh-converged frequency of
the same drawing.

**Ladder.**

- dx = 0.5, 0.25, 1/6 and 0.125 mm, which puts 3, 6, 9 and 12 cells across the substrate.
- Every face is on a node at every rung: x 11/39 mm, y 7/43 mm, z 1.5 mm.
- The CPML is held at 4 mm.
- The record duration is held at 5.72 ns (the fixture's 6000 steps at 0.5 mm) and read through
  the harness's own ring-down window.
- Trees T0 and T4.
- Control: T4 at dx = 0.5 mm with twice the record length, to state how much the extractor depends
  on record length.

**Expected under A1.** T0 and T4 extrapolate to the same limit within 0.2 %. At 0.5 mm, T4 is
closer to that limit than T0 is. T0 approaches it at first order.

**Falsifier.** Either of these shows a defect:

- at 0.5 mm, |f_T4 − f_inf| ≥ |f_T0 − f_inf|;
- the two trees' limits differ by more than 0.3 %.

## R3: edge-fed patch

**Leg A reference.** The mesh-converged TM010 of the isolated (unfed) patch. The Balanis value on
the realized raster, which is the lock's anchor, is computed at each rung for context only and is
never the judge.

**Why the ladder uses a redrawn board.** In x and y the fixture's own board is off the lattice.
Its 8.595 × 10.129 mm patch realizes 42 × 50 edges at h/4 and a different count at every other
mesh, so a ladder on it would compare different patches.

**The whole-cell board.** h = 787 µm, ε_r 3.38. Distances are in units of h/2 = 393.5 µm:

| item | size in h/2 | size in mm |
|---|---|---|
| patch L × W | 21 × 25 | 8.2635 × 9.8375 |
| domain x × y × z | 76 × 47 × 32 | 29.906 × 18.4945 × 12.592 |
| patch corner (x, y) | (33, 11) | (12.9855, 4.3285) |
| ground plane height z | 10 | 3.935 |

- The patch equals the fixture's realized electrical patch at N = 4.
- The foils lie on the laminate faces.
- The CPML is 2N layers, which holds it at 1.574 mm.
- Every plane is on a node for every even N.
- The rungs are N = h/dx = 4, 6, 8 and 10. Only the unfed arm is run, with 200 periods and the
  fixture's census and parity extractor.
- Trees T0 and T4. The fixture board itself, both arms at N = 4, is in the tree sweep.

**Expected under A1** (same form as R2): both trees converge to one limit within 0.3 %, T4 is
closer at N = 4, and T0 approaches at first order.

**Falsifier** (same form as R2): at N = 4, |f_T4 − f_inf| ≥ |f_T0 − f_inf|; or the two limits
differ by more than 0.3 %.

**Leg B: the feed pull f_fed/f_unfed − 1 at N = 4 on the fixture board.** A bisection over the
first-parent range from 529fab2b (09-21, weekly lane −7.067 %) to 3247dc0e (09-23, weekly lane
−7.216 %). Of the 39 commits in the range, 25 touch `rfx/`. The lock module is byte-identical
across the range; it changed only in #1291, afterwards.

- **Round 1 (CPU, run in parallel).** 529fab2b, 12da9104, 77f7094c (the parent of #1178),
  e7f7e027 (#1178, the candidate named in #1281), 02ff8849 and 3247dc0e.
- **Validity check.** The two endpoints have to reproduce the weekly numbers to the printed
  digits; if they do not, the environment differs, and the report says so.
- **Rounds 2 and later.** Binary search inside every interval where the pull moved, down to
  adjacent commits. Every step larger than 0.005 pp is named: this lane is deterministic, with
  bit-identical probe series across thread counts (the lock's docstring).
- **Checking A3.** The range ends before #1213. The tree sweep's T0 → T1 pair gives #1213's own
  effect on the pull.

## R4: microstrip S matrix, end to end, on a graded mesh

**Structure.** The test's builder: a 254 µm, ε_r 3.66 substrate under a 600 µm PEC trace sheet.
The ports are at 2 and 12 mm, and the S reference planes are the probe-0 planes at 5 and 9 mm,
leaving 4 mm of line between them. The mesh has dx = dy = 50 µm and a graded z profile whose
substrate is 2 × 50 plus 4 × 38.5 µm. There are 10 bins over 0.5–5 GHz, with num_periods 12.

**Independent reference.** The closed-form microstrip, computed before any solve:

- Hammerstad–Jensen 1980 (t = 0): Z0 = 47.667 Ω, quasi-static ε_eff = 2.8758.
- Kirschning–Jansen dispersion: ε_eff = 2.8760, 2.8796 and 2.8839 at 0.5, 3 and 5 GHz.
- The rfx S matrix is referenced at both ports to the simplified formula
  `hammerstad_jensen_z0_eps_eff`, which gives Z_ref = 47.895 Ω.

**Observables per run.**

- (a) The solved line impedance Z_s(f) and ε_eff(f), obtained by inverting the 2 × 2 S of a
  uniform reciprocal line referenced to Z_ref (through ABCD).
- (b) Electrical length (v2 bar item 5): the least-squares slope of unwrapped arg S21 over the
  bins where |S21| > −20 dB, against the same slope of the closed-form phase
  −2πf·√ε_eff,KJ(f)·L/c.
- (c) |S11| per bin, held to the −20 dB bound (deep-null ruling: a thru's reflection).
- (d) S element by element against the committed golden (the lock's quantity) and against the
  same tree's finest rung.

**Ladder.** Refinement r = 1, 1.5 and 2 (dx = 50, 33.3 and 25 µm), with the dz profile scaled.
Every plane is on a node at every rung: the substrate top, the trace edges at 1.4 and 2.0 mm, the
ports and the probe planes. Probe offset and spacing are 60r and 20r cells, and the CPML is 16r
layers. r = 3 is added if the measured cost of the first r = 2 run puts it under 6 GPU-hours per
tree. Trees T0 and T4.

**Expected under A1.**

- The T4 − T0 difference in S shrinks with refinement at least as fast as dx.
- At r = 1, T4's electrical length is closer than T0's both to the closed form and to the
  converged value.
- **Evidence already on record that points the other way (not a finding of this campaign).** In
  the #1253 battery (a uniform 300 µm board), after #1213 the solved Z0 sat further from the
  simplified closed form at every rung: 49.3, 51.1 and 52.0 Ω against 53.1 Ω. The R4 ladder states
  which tree's Z_s converges toward the HJ 1980 value, and from which side.

**Falsifier.** Either of these shows a defect:

- the T4 − T0 difference in S does not shrink (ratio ≥ 0.8 per rung);
- at the finest rung, T4's ε_eff or electrical length is more than 1 % from Kirschning–Jansen
  while T0's is within 1 %.

The #1253 battery's electrical length and Z_s trends are quoted beside R4's, read from the
ledger.

## R5: microstrip with a floating resistive sheet

**Structure** (read on main without a solve):

- RO4350B, 254 µm, with a PEC trace sheet on the laminate face, at DX = h/3.
- The lossy element is a separate node-thin resistive sheet, made by `add_thin_conductor` with
  `surface_impedance_f0` = 3.5 GHz. Its sheet resistance Rs0 is frozen at f0 and it has no
  reactance.
- The sheet is 35 × 16 cells (x 4.57–7.54 mm, 1.35 mm wide) on z node 6, which is 254 µm above
  the trace plane, in air.
- Probe planes resolved on main: port 1 at x = 53u, port 2 at 89u (u = h/3; offset 29, spacing 2).

**Observables.**

- The mean |S21| over the 3.0–4.5 GHz gate band, and its drop per Rs0 step through
  {1e-6, 1, 5} Ω/sq.
- The absorbed fraction 1 − |S11|² − |S21|².
- The dispersion pin: the β-implied shift f·|β_f0(1e-6) − β_pec|/β_pec in each gate bin, also as a
  fraction of f.

**Reference.**

- **The suggested formula does not describe this structure.** α_c = Rs/(Z0·W) is the loss of a
  line whose strip is the resistive conductor. Here the strip is PEC and the resistive sheet
  floats above it.
- **Size of that mismatch.** Applied anyway, the formula predicts Δα = 4 Ω / (47.9 Ω × 0.6 mm)
  = 139 Np/m for the 1 → 5 step. Over the 2.96 mm sheet that is a |S21| drop of about 0.34, while
  the measured drop is about 2e-4. The formula is not used.
- **What is used instead.** No closed form for a resistive plate floating over a microstrip is
  known to me. The reference is therefore the mesh-converged 1 → 5 drop of the same drawing, and
  the Rs sweep below is a diagnostic.
- **Open question.** This was sent to the leader on 2026-09-27. If the answer changes this row, it
  is recorded as an addendum before the R5 ladder runs.

**Ladder.** A whole-cell redraw in u = h/3 = 84.667 µm:

| item | size |
|---|---|
| substrate | 3u |
| trace width | 7u = 592.7 µm (600 µm would be 7.09u) |
| LX | 142u |
| port margin | 24u |
| line length | 94u |
| LY | 35u |
| LZ | 21u |
| sheet | x 54u–89u, y 9u–26u (17 cells, symmetric about the trace), plane 6u |

- DX = u/n for n = 1, 2 and 3.
- Probe offset and spacing are pinned to 29n and 2n on both trees, so both trees use the same
  instrument.
- The CPML is 8n layers.
- Rs0 ∈ {1e-6, 1, 5} and no sheet.
- Trees T0 and T4.

**Diagnostic sweep.** The fixture drawing at DX = h/3 on T0–T4, with
Rs0 ∈ {1e-6, 0.5, 1, 2, 3, 5, 10, 25}, no sheet, and the PEC sheet. It shows where the 1 → 5 step
sits on the curve of absorbed power against Rs0.

**Expected under A1.** At n = 1, T4's 1 → 5 drop is closer than T0's to the converged drop, and
the T4 − T0 difference shrinks with refinement.

**Falsifier.** The T4 − T0 difference in the 1 → 5 drop does not shrink with refinement (ratio
≥ 0.8 per rung). The reverse ordering at n = 1 is recorded as a fact.

## Assumptions carried from the brief (measured, never stated as findings)

- A1: #1213 moved each row toward the physical answer.
- A2: #1012 moved R1–R3 and R5 further (the tree sweep's T2 → T3 pair).
- A3: Leg B's move is not #1213 (the bisection, and the T0 → T1 pair).

## Records

- Measurement scripts, per-run JSON, logs and figures go to `bk-squared/rfx-archive` under
  `rfx/records/20260927-1292-<topic>/`.
- Every VESSL run's provider log is saved before the run is deleted.
- This repository gets no records.

## Addendum 1 (2026-09-27, before any solve of this campaign)

Written after build-only checks of every drawing; no field has been solved yet.

**R2, record length.** The fixture's record is 6000 steps, 5.72 ns. It ends while the ring-down is
still at 0.768 of its post-source peak (−2.3 dB; `ringdown_tail_to_peak` in
`tests/data/v173a_aligned_composition.json`). Under this note's settling rule, a frequency read
from that record is flagged and not used. So every R2 rung is solved for 84 ns: a mode near
2 GHz with Q ≈ 105 falls about 40 dB in 77 ns. Two readings come from the same solve:

- (i) the lock's reading, the harness's `_measure` on the first 5.72 ns (a prefix of the long
  record); where the undecimated matrix pencil does not fit in memory (post-source prefix longer
  than 6000 samples, that is dx < 0.25 mm), it uses harminv with decimation `"auto"`, and the
  record says so;
- (ii) the settled reading, harminv (decimation `"auto"`, 1.5–3.5 GHz, min_Q 5) on the whole
  post-source record, together with its end-of-record level.

The convergence estimate and the falsifier use (ii). The pair (i)/(ii) replaces the
"twice the record length" control. Rungs 0.5, 0.25 and 1/6 mm run first. The 0.125 mm rung runs
only if the measured wall time of the 1/6 mm rung puts it under 12 GPU-hours per tree; otherwise R2
has three rungs. The CPU tree sweep runs the 84 ns record at 0.5 mm, which yields both readings.

**R5, the realized sheet (correction to the Structure paragraph).** The sheet's realized mask
counts nodes, not cells: 35 × 16 nodes is 34 × 15 cells, x 54u–88u = 4.572–7.451 mm, y 10u–25u
(1.27 mm, centred on the trace), plane 6u. The trace realizes 7 cells (y nodes 14–21, 592.7 µm);
the drawn width is 600 µm. The trace runs over x nodes 0–141 and stops one node short of the x_hi
domain face. Probe 0 lies at x = 53u for port 1 and 89u for port 2, one cell outside each end of
the sheet. The α_c illustration becomes Δα = 139 Np/m over 2.88 mm, a |S21| drop of about 0.33;
this does not change the conclusion that the formula does not describe the structure.

**R5, the redraw (replaces the ladder table).** The domain is 142 × 36 × 21 u. The trace is a
node-plane sheet on x 0–141u, y 14u–21u. The sheet's nodes are x 54u–88u, y 10u–25u, on plane 6u.
The ports sit at x 24u and 118u, y 17.5u, width 7u; probe offset and spacing are pinned to 29n and
2n. The CPML is 8n. At n = 1 this redraw realizes the fixture's own node masks for the trace, the
sheet and the probes (checked at build time on main). Whether its S equals the fixture's at n = 1
is checked in the first job and recorded; the port source can realize differently, because the
drawn position is 17.5u against 17.54u and the width 7u against 7.09u.

**R3.** The whole-cell board's solver patch, read from the sheet footprint, is 42 × 50, 63 × 75,
84 × 100 and 105 × 125 edges at N = 4, 6, 8 and 10: 8.2635 × 9.8375 mm at every rung, with
N cells of ε_r 3.38 between the foil planes. The lock's `shape.mask` node count carries a ±1 at
some rungs (for example 76 instead of 75 at N = 6), so the Balanis context value is computed from
the sheet footprint: 9.5411 GHz at every rung, and for the fixture board 9.5411 GHz (the lock's
own anchor is 9.3305 GHz, computed from node counts).

**R4.** The generalized builder at r = 1 reproduces the fixture's grid (313 × 101 × 63, same dz,
same dt). At r = 1.5, 2 and 3 the grids are 6.7, 15.7 and 52.7 M cells. Preflight on the refined
rungs reports the trace sheet solved 0.35 cell wider at each free edge (for example 617.5 µm at
r = 1.5). This is recorded, not acted on.

## Addendum 2 (2026-09-27, before the R3 N = 6 re-run)

**R3, whole-cell board, rung N = 6: a defect in my drawing.** In the first attempt the laminate
realized 7 cells instead of 6. Its upper face, 10·(h/2) + h, sits at 36.00000000000001 cells in
floating point, so the laminate also took the cell above the patch plane. Preflight printed "PEC
sheet … realizes on node plane 48 … the cells on BOTH sides of that plane carry the same
dielectric … buried half a cell inside the dielectric" for that rung only; N = 4, 8 and 10, and the
fixture board, have no such message. The first-attempt N = 6 value on main was 8.8200 GHz, and all
three identified modes sat 1.6–2.2 % below the neighbouring rungs.

The drawing now places both laminate faces 0.001 cell inside their node planes, and every record
asserts the laminate's cell count in a column outside the patch (N cells). At N = 4, 8 and 10 the
realized masks are unchanged, and their first-attempt records stand. N = 6 is re-run on T0 and
T4. The first-attempt N = 6 records are kept and marked as defective; they are not used in the
convergence estimate. R2 (3/6/9/12 cells) and the R5 redraw (3/6/9 cells) were checked the same
way at build time and are correct.

## Addendum 3 (2026-09-27, before the R2 re-run)

**R2, record length.** The first 84 ns ladder solve, main at dx = 1/6 mm, ended with the ring-down
at −35.0 dB of its post-source peak; its settled reading is flagged and not used (common rules).
The envelope of that record falls at a steady 0.446 dB/ns, reaching −40 dB about 89 ns after the
ring-down window opens. So every R2 rung is re-run at 110 ns, where the same slope puts the end
near −48 dB. The 84 ns records are kept: their prefix readings (i) are valid, and their settled
readings are marked as unsettled.

**R2, fourth rung.** The 1/6 mm solve took 43 min at 84 ns, which puts 0.125 mm at 110 ns near
3 GPU-hours per tree, under addendum 1's 12-hour limit. The 0.125 mm rung therefore runs on both
trees.

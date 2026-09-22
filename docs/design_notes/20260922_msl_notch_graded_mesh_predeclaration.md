# The MSL notch filter on a band-graded mesh — pre-declaration

**Status:** pre-declaration, committed before any arm runs. Windows in §5 are
frozen from this commit; results are appended in a "Results" section, never
edited into §5. One attempt per arm (R2). Lane: nu-mesh. Tracker: #810 (the
graded-mesh S-parameter question) and #928 (the case).
**Tree:** branch `nu/msl-notch-graded` on top of `crossval/msl-notch-filter`
(PR #1177, the case this arm extends); main 12da9104 is contained.
**Written by** the NU lane leader (bk-workspace, Fable) on 2026-09-22 after
reading the case, the MSL port's graded-mesh paths, the band-profile builder,
the edge-offset builder and the ledger entries named below, and after one
build-only feasibility run on CPU (§2).

## 0. The question, and what changes for a user

A 50 Ω microstrip (600 µm trace on 254 µm of lossless εr 3.66) with a 12 mm
open stub at its middle is a notch filter: the stub is a quarter wave near
3.67 GHz and shorts the line there. On rfx's uniform staircase mesh the notch
converges to openEMS's own tutorial value from above, but at an order of about
0.65 in the cell size, because the 600 µm trace realizes 4, 9 and 14 node rows
at h/2, h/4 and h/6: the finest affordable rung (h/6, 13.8 M cells, 661 s on
an RTX 4090) still reads the notch 2.16 % high, and the next rung did not finish
in 73 min (ledger 2026-09-22; PR #1177). The user-facing number is therefore a
stub notch frequency about 2 % high at any affordable uniform mesh.

This note asks one question: **does a mesh that is fine only where the metal
edges and the substrate are — a three-axis band-graded mesh — converge the
same board to the same reference at a cost a user can pay?** The answer is a
number the support matrix can carry for "MSL S-matrix + nonuniform mesh"
(today: "no external nonuniform comparison") and the load-bearing case #810
asked for.

## 1. Structure and reference (facts, with their sources)

| item | value | source |
|---|---|---|
| substrate | 254 µm, εr 3.66, lossless | `tests/crossval/msl_notch_filter/test_msl_notch_filter.py` constants |
| trace / stub width | 600 µm, zero-thickness PEC sheets on the substrate-top node plane | same |
| stub | 12 mm open, along +y from the trace's far edge, centred at x = 12.0 mm | same |
| box (rfx's own) | 24.0 × 18.8 × 1.754 mm; arms 10 mm each side of the stub; ports 2 mm from the x faces; CPML 8 layers on x, y and z_hi; PEC ground at z_lo | same |
| reference | openEMS 0.37.0 running its own `MSL_NotchFilter.py` on 50 mm arms; judged stage `stage_b_fine`: notch **3.67436 GHz**; band 2–7 GHz | `reference/openems_tutorial.json`, `reference/PROVENANCE.md` |
| bar (v2) | notch within 1 %; \|S21\| within 2 dB where both curves are above −20 dB; mesh statement: last two rungs within 1 % | the case's `FREQ_BAR`, `MAG_BAR_DB`, `DEEP_NULL_DB`, `LADDER_AGREEMENT` |
| witnesses on rfx's record | ring-down ≤ −40 dB per driven run; passivity excess ≤ 0.01 | the case's `SETTLING_DB`, `PASSIVITY_EXCESS_BAR` |
| uniform ladder (for the cost and cross-ladder comparison) | h/2, h/4, h/6 → 3.8813, 3.8034, 3.7537 GHz; 0.79 / 4.59 / 13.8 M cells; 24 / 132 / 661 s; extrapolated limit 3.67 GHz | ledger `rfx-known-issues.md` 2026-09-22 entry; PR #1177 body |
| edge rule | a PEC sheet's in-plane edge is solved 0.35 cell beyond its last node (0.31–0.37 for a strip over ground) | `rfx/mesh_edges.py` docstring, `EDGE_OFFSET` |
| MSL port on a graded mesh | `mode="laplace"` only; per-cell sizes read at the trace's first width cell and the substrate's first normal cell, uniform inside the port box assumed; probe planes must be set explicitly when the propagation axis is graded | `rfx/sparams/msl.py:280-296`, `rfx/sources/msl_port.py:755-830`, preflight text of §2 |
| prior art | a graded transition adjacent to the substrate SPLITS the microstrip mode (cv05, 2026-07-16): the fine band must contain the whole substrate with no transition next to it | ledger history 2026-07-16; #1083 results note §8 |

## 2. What the leader verified on this tree (FACT) and what is not yet measured (ASSUMPTION)

FACT — a build-only run (no time step) of the board below on the current
tree, `scratchpad/notch/graded_feasibility.py`, CPU, jax 0.10.2:

- profiles: x 212 cells (50–127 µm, worst ratio 1.260), y 185 cells (50–127 µm,
  1.262), z 24 cells (42.3–110.7 µm, 1.378); interior 941,280 cells against
  13.8 M for uniform h/6; `dt` = 8.961e-14 s (set by the 42.33 µm substrate
  cell, the same as uniform h/6);
- every declared edge — stub x 11.7 / 12.3 mm, trace y 1.55 / 2.15 mm, stub end
  y 14.15 mm, substrate top z 254 µm — landed on a node to 0.000 nm;
- the product realized one sheet plane (k = 6, z = 254.0 µm), zero PEC volume
  cells, trace 13 node rows spanning 600.0 µm, stub 13 node columns spanning
  600.0 µm, stub joined at the trace's far row, stub length 12000.0 µm;
- preflight fired: `nu_grading_reaches_absorber` on x_lo, x_hi, y_lo, y_hi
  (plateau cells solved 0.8 % off the 127 µm pin; the y_lo runway too short
  for the ramp), the sheet-edge advisory (600 µm drawn, solved as 635 µm,
  +5.83 %, both sheets), the MSL probe-interval auto-solve SKIPPED because the
  propagation axis is graded, and the substrate box 3.5e-9 nm past the y_hi
  absorber boundary (float dust of the profile sum).
- the uniform case at h/2 resolves `n_probe_offset = 14`, `n_probe_spacing =
  6`, `n_probes = 5` (127 µm cells: 1.778 mm offset, 0.762 mm spacing, deepest
  probe 4.826 mm from the port plane), read from the built simulation's port
  entries without solving.

ASSUMPTION (not measured; the arms measure it):

- the NU runner's throughput per cell is within 2× of the uniform runner's (the
  cost claim depends on it);
- the graded ladder converges to the same limit as the uniform ladder (§5 W3);
- the 0.35-cell edge offset applies to this strip over ground with 50/42 µm
  cells as it did on the 2-D fin (0.31–0.37 measured there);
- the MSL Laplace feed is unaffected by the ramp cells outside the fine y band
  (the port box is trace ± 1·h = ±254 µm, inside the ±500 µm band).

## 3. The mesh (arms and rungs)

Coarse cell `C = 127 µm` everywhere outside the bands (the case's h/2; 170
cells per guide wavelength at 7 GHz). Every x/y profile starts and ends with
**at least 9 cells of exactly `C`** (8 is the absorber's runway; one more so
the ramp's first cell is not the ninth), the box lengths are derived from the
profiles (the domain is rfx's own, so its x and y extents may grow by up to
one coarse cell to make the runways exact), and the geometry is drawn to the
profile sums. In-plane ratio cap 1.3, z cap 1.4 (the two validated caps).

Fine bands (protected, uniform inside):

| axis | band | contains |
|---|---|---|
| x | stub centre ± (300 µm + 500 µm) | the stub's two x edges |
| y | trace ± 500 µm around the line, and stub end ± 500 µm | the line's two y edges and the stub's open end |
| z | 0 → 2h = 508 µm, cell `FZ = h / n_z` | the substrate (n_z cells, top face on a node) and n_z air cells above the sheet |

Two ways of placing the in-plane nodes at a metal edge, each an arm:

- **on-node**: a node ON each drawn edge (the feasibility build of §2);
- **offset**: a node `0.35 F` INSIDE the metal at each free edge, the sheet
  drawn at its true size, so its solved edge lands on the drawn edge (the
  `rfx.mesh_edges` rule, openEMS's thirds rule). With n cells across the
  600 µm width the fine cell is `F = 600 µm / (n + 0.7)`.

Rungs (the ladder is on the offset arm):

| rung | n_z (substrate cells) | FZ | n (cells across 600 µm) | F | expected interior cells |
|---|---|---|---|---|---|
| A-on | 6 | 42.33 µm | 12 | 50.00 µm | 0.94 M (measured, §2) |
| A-off | 6 | 42.33 µm | 12 | 47.24 µm | ≈ 1.0 M |
| B-off | 8 | 31.75 µm | 16 | 35.93 µm | ≈ 1.6 M |
| C-off | 12 | 21.17 µm | 24 | 24.29 µm | ≈ 3 M |

Plus **A-off on 15.08 mm arms** (the case's arm-length witness, at the coarsest
rung only). Five solves in total.

Ports: `n_probe_offset = 14`, `n_probe_spacing = 6`, `n_probes = 5` set
explicitly on both ports (the values the uniform case resolves at `C`, §2), so
the probe planes sit at the same physical x as the uniform case's; every probe
plane lies inside the uniform-`C` region (deepest 6.83 mm < 11.2 mm, the x
band's start). `mode="laplace"` (the default). `n_freqs = 400`, `num_periods =
20`, `enforce_passivity = False` — the case's values.

## 4. Instrument

`validation/research/multiband_nu/msl_notch_graded.py`, importing the case
module (`tests.crossval.msl_notch_filter.test_msl_notch_filter`) for the board
constants, the notch estimator, `_compare`, the arm-length witness and the
reference loader — nothing re-typed. It adds `profiles(rung, placement)`,
`build_graded(...)`, a graded-aware realized reader (node coordinates from
`tests._realized_geometry._node_line`, never `xs[1] - xs[0]`), `run_arm(...)`
and a JSON writer. Output: `validation/research/multiband_nu/results/msl_notch_graded.json`
(per arm: profiles, realized geometry, preflight lines verbatim, S11/S21 on
the 400-point grid, Z0, settling, passivity excess, wall time, cells,
`rfx.__file__`, git sha/dirty, argv, run id from the submitter) and a figure.
The CLI refuses an existing output (the lane's rule).

Refusals before any solve (instrument checks, not windows; a refusal is fixed
in the instrument and recorded, the windows do not move):

- R1 every intended node (edges ± offset, substrate top, band edges) is a node
  to 1e-9 m; every adjacent ratio ≤ 1.3 in x/y and ≤ 1.4 in z; the first and
  last 9 cells of x and y equal `C` bit-exactly;
- R2 `preflight()` emits NO `nu_grading_reaches_absorber` line and no
  geometry-in-absorber line;
- R3 realized: exactly one sheet plane at z = h; zero PEC volume cells; trace
  rows == stub columns == n + 1; stub attached at the trace's far row and
  contiguous; the stub's open end on the intended node; on the offset arm the
  outermost realized node of each edge is the offset node (0.35 F inside);
- R4 after the solve, the case's witnesses: ring-down ≤ −40 dB on every driven
  run, passivity excess ≤ 0.01 — a rung failing either is not compared.

Always-on CPU test `tests/unit/nonuniform/test_msl_notch_graded_build.py`:
builds A-on and A-off (no solve), asserts R1–R3 through the instrument's own
functions. Mutations the PR body reports: (a) the R3 check disabled → red;
(b) the stub drawn one coarse cell short with every helper call kept → red;
(c) the offset arm built with the node ON the edge → red on the offset check.

VESSL: `scripts/vessl_msl_notch_graded.yaml`, GPU preset copied from
`scripts/vessl_gpu_suite.yaml`, one job per arm submitted through
`scripts/vessl_submit.sh` (run id recorded by the submitter), log tee'd and
collected under a trap, pytest not involved (the instrument is a script).
Concurrency per the queue rule (agent-rules research/CLAUDE.md 2026-09-22).

## 5. Frozen windows

- **W1 mesh statement (offset ladder):** `|f_C − f_B| / f_B ≤ 1 %` (the case's
  `LADDER_AGREEMENT`), with the three notches monotone along A-off → B-off →
  C-off. Not monotone or > 1 %: the graded ladder has not converged; the
  comparison in W2 is still printed but the arm's verdict is "not converged".
- **W2 comparison (finest offset rung against the reference):**
  `|f_C − 3.67436 GHz| / 3.67436 ≤ 1 %` and `max |ΔdB| ≤ 2 dB` over the
  reference bins in 2–7 GHz where both curves are above −20 dB — the case's
  bar, applied to the graded mesh unchanged.
- **W3 cross-ladder consistency:** the offset ladder's Richardson limit (order
  fitted on A-off, B-off, C-off; the limit from B and C at that order) agrees
  with the uniform ladder's extrapolated 3.67 GHz (ledger 2026-09-22) to
  within 1 %. W1 and W2 holding while W3 fires is a finding, not a pass: the
  two lanes converge to different boards.
- **W4 arm-length witness (A-off, 10 vs 15.08 mm arms):** `|S21|` moves ≤ 0.5 dB
  where both curves are above −20 dB (the case's `ARM_WITNESS_BAR_DB`).
- **Reported, no window:** `f_A-on − f_A-off` (what the 0.35-cell edge offset is
  worth on this structure); per-rung cells, wall time and the cost ratio to
  uniform h/6 (13.8 M cells, 661 s); fitted line Z0 per rung.

## 6. Expectations, written before the run

A-off within 1.5 % of the reference (the uniform h/6 rung is at +2.16 % with a
realized width of 592.7 µm; the offset arm realizes 600 µm and puts the stub's
open end on a node). The ladder monotone. C-off inside W2. Rung A wall time
under 1/8 of uniform h/6's 661 s if the NU runner's throughput is within 2× of
the uniform runner's (§2 assumption). A-on below A-off in frequency by a few
tenths of a percent (a 635 µm solved width lowers Z0 and lengthens the open
end's fringing). If C-off converges (W1) but misses W2, the residual is the
graded lane's own and is recorded as such; no rung is re-run with a different
band, cap, probe or window.

## 7. Cost

Five GPU solves. Rung A ≈ 1 M cells at the h/6 time step (uniform h/6: 13.8 M
cells, 661 s); rung C ≈ 3 M cells at half that time step. Under one GPU-hour
in total on the assumption of §2. Instrument tests: seconds on CPU.

## 8. What the arms can and cannot claim

Can: "on the MSL notch filter, a mesh fine only across the metal edges (n cells
across 600 µm, node 0.35 cell inside the edge) and through the substrate (n_z
cells) reproduces the openEMS notch to X % at Y cells and Z s, converging along
n_z = 6 / 8 / 12", as one support-matrix sentence for "MSL S-matrix +
nonuniform mesh" and as #810's load-bearing case. Cannot: any other port
family; grading that reaches an absorber (the runways are uniform by
construction); phase (not judged by the case); the general in-plane envelope
(one structure, ratio ≤ 1.3, three rungs); float64 (the NU runner threads
float32 fields only).

## 9. Regression requirement

No `rfx/` change. The case module is imported, not edited (a mesh option in the
case itself is a later PR, after #1177 lands and after this note's results
section exists). `tests/crossval/msl_notch_filter/` byte-identical to PR
#1177's head at every commit of this branch. The always-on build test of §4
passes on CPU; `scripts/ci/local.sh` green.

## 10. Who does what

Implementation: one Opus instance from this note alone. The leader reads the
JSON, the figure and the preflight text, and writes the Results section and
every sentence that interprets a number. Review: a separate Opus instance,
fresh eyes, one round; P1/P2 fixes back to the same reviewer. Documentation
mismatches go to #1171.

## Results (facts)

Appended after the runs; section 5 above is unchanged.  Every number below is read from
`validation/research/multiband_nu/results/msl_notch_graded.json` by `markdown_tables()` in
the instrument, and re-derived from the same file by `tests/unit/nonuniform/test_msl_notch_graded_replay.py`.

### R.0 What the instrument had to change, and why

Five facts, recorded because each one changed something section 3 or section 4
states. None of them moves a section 5 window.

1. **The line sits 2.159 mm from the y_lo face, not the case's 1.55 mm, and the
   box is 19.409 mm deep in y instead of 18.800 mm.** Section 3 asks for at
   least nine cells of exactly 127 um against each in-plane absorber face and
   for a fine band reaching 500 um beyond each metal edge. On the case's board
   the y band starts 500 um below the line, at 1.05 mm, and nine coarse cells
   need 1.143 mm: the runway does not fit under the band. A fine y_lo runway is
   not available either, because `make_nonuniform_grid` refuses a y profile
   whose two end cells differ (`rfx/nonuniform.py`, the CPML boundary-cell
   contract). 17 x 127 um = 2.159 mm leaves room for the runway plus the
   longest transition any rung needs (C_off: 9 x 127 um + 336 um). Trace width,
   stub length, arm length, port margin, the 4.65 mm above the stub's open end,
   the substrate and the box height are the case's, unchanged.
2. **The z column above 2h holds nine cells of one SOLVED size, not of 127 um.**
   The 1.246 mm of air above the fine z band cannot hold nine 127 um cells plus
   a ratio-1.4 transition down to the substrate cell at any rung: B_off needs at
   least 235 um for the transition and has 230 um. The tail cell is solved per
   rung instead (table R.1). Every one is below the declared coarse cell and the
   nine are bit-identical, so the z_hi absorber still stands on a uniform
   runway, which is what the absorber's own preflight check asks for.
3. **The fine band's margin is the smallest whole number of fine cells that
   covers the declared 500 um**, so the realized margin is 500.0 um at A_on and
   519.7, 503.0 and 510.1 um at the three offset rungs (table R.1). A band is a
   whole number of fine cells laid down from the node the metal must own; a
   margin of exactly 500 um would put the band edge off the node line.
4. **Every face is drawn to the node the profile produced, not to the
   arithmetic that asked for it.** On rung B the substrate top's node lands
   5e-20 m below 254 um -- float dust in a cumulative sum of eight equal cells
   -- and a box drawn to 254 um exactly then swallows the 31.75 um cell above
   the trace plane: the board solved carries 285.75 um of dielectric and the
   trace sheet is buried half a cell inside it. rfx's preflight names that
   condition, and it now joins the R2 refusal list alongside the two the note
   declares.
5. **The first attempt recorded no commit, and every arm was re-run.** VESSL
   run 369367263251 produced a valid A_off measurement whose provenance block
   carried the string `<unavailable: ... exit status 128>` where the commit
   should be: the job exports the pinned tree with `git archive` into scratch,
   so `git rev-parse` inside it has nothing to read. The instrument now takes
   the commit from the submitter and REFUSES an arm it cannot name a commit
   for, before the solve. The five arms in the tables below are the second
   attempt, all at one commit. Four arms have a first attempt, and it read the
   same notch to every printed digit:

   | arm | attempt 1 (GHz) | VESSL run | attempt 2 (GHz) | VESSL run |
   |---|---|---|---|---|
   | A_on | 3.732318 | 369367263257 | 3.732318 | 369367263265 |
   | A_off | 3.747389 | 369367263251 | 3.747389 | 369367263264 |
   | B_off | 3.732214 | 369367263258 | 3.732214 | 369367263266 |
   | C_off | 3.716635 | 369367263261 | 3.716635 | 369367263283 |

   The first attempt's logs, curves and figures are kept off-repo under
   `rfx-nu/runs/attempt1-*`.

### R.1 The mesh each arm solved

| arm | rung | placement | F (um) | FZ (um) | z tail cell (um) | band margin (fine cells) | interior cells | grid | dt (fs) |
|---|---|---|---|---|---|---|---|---|---|
| A_on | A | on-node | 50.0000 | 42.3333 | 114.7303 | 10 | 966,720 | 212x190x24 | 89.6116 |
| A_off | A | offset | 47.2441 | 42.3333 | 114.7303 | 11 | 996,384 | 214x194x24 | 86.6012 |
| B_off | B | offset | 35.9281 | 31.7500 | 111.0135 | 14 | 1,383,648 | 224x213x29 | 65.5056 |
| C_off | C | offset | 24.2915 | 21.1667 | 109.6286 | 21 | 2,403,500 | 250x253x38 | 44.0446 |
| A_off_longarms | A | offset | 47.2441 | 42.3333 | 114.7303 | 11 | 1,368,864 | 294x194x24 | 86.6012 |

### R.2 What the lattice realized

| arm | sheet plane z (um) | PEC volume cells | line node rows | stub node cols | metal node span (um) | stub length (um) | node inside the drawn edge (um) |
|---|---|---|---|---|---|---|---|
| A_on | 254.0000 | 0 | 13 | 13 | 600.0000 | 12000.0000 | 0.0000 |
| A_off | 254.0000 | 0 | 13 | 13 | 566.9291 | 12000.0000 | 16.5354 |
| B_off | 254.0000 | 0 | 17 | 17 | 574.8503 | 12000.0000 | 12.5749 |
| C_off | 254.0000 | 0 | 25 | 25 | 582.9960 | 12000.0000 | 8.5020 |
| A_off_longarms | 254.0000 | 0 | 13 | 13 | 566.9291 | 12000.0000 | 16.5354 |

### R.3 What each arm measured

| arm | notch (GHz) | depth (dB) | -10 dB BW (MHz) | fitted Z0 (ohm) | worst settling (dB) | worst passivity excess | wall (s) | cells / uniform h6 | wall / uniform h6 |
|---|---|---|---|---|---|---|---|---|---|
| A_on | 3.73232 | -51.07 | 770.4 | 47.00 | -92.89 | 0.00455 | 101.4 | 0.1134 | 0.1534 |
| A_off | 3.74739 | -54.88 | 786.6 | 48.74 | -95.48 | 0.00430 | 106.1 | 0.1166 | 0.1605 |
| B_off | 3.73221 | -51.84 | 782.4 | 48.58 | -95.10 | 0.00439 | 333.1 | 0.1526 | 0.5039 |
| C_off | 3.71663 | -50.81 | 777.8 | 48.38 | -94.89 | 0.00443 | 1007.7 | 0.2455 | 1.5246 |
| A_off_longarms | 3.74731 | -54.90 | 787.6 | 48.58 | -94.57 | 0.00416 | 129.5 | 0.1569 | 0.1959 |

Bars, for reading the two witness columns: ring-down -40 dB, passivity excess 0.01.  The uniform h/6 rung this cost is divided by is 13,800,000 cells and 661 s (pre-declaration section 1).  Both cell counts in that ratio are GRID cells, absorber pad included, which is what the uniform ladder's own record counted; table R.1's interior count is the smaller number the pre-declaration's section 3 quotes.

### R.4 The frozen windows

**W1 mesh statement (A_off -> B_off -> C_off).**

| notch A_off (GHz) | notch B_off (GHz) | notch C_off (GHz) | last two rungs apart (%) | bar (%) | monotone | verdict |
|---|---|---|---|---|---|---|
| 3.74739 | 3.73221 | 3.71663 | 0.4174 | 1 | True | HELD |

**W2 comparison (C_off against the openEMS tutorial's stage_b_fine).**

| rfx notch (GHz) | reference notch (GHz) | distance (%) | bar (%) | max abs dB | bar (dB) | bins compared | verdict |
|---|---|---|---|---|---|---|---|
| 3.71663 | 3.67436 | 1.1507 | 1 | 2.7874 | 2 | 1080 of 1144 | FIRED |

**W3 cross-ladder consistency.**

| fitted order in the substrate cell | graded limit (GHz) | uniform ladder limit (GHz) | distance (%) | bar (%) | verdict |
|---|---|---|---|---|---|
| 0.9225 | 3.68229 | 3.67000 | 0.3349 | 1 | HELD |

**W4 arm-length witness (10.00 mm against 15.08 mm arms).**

| max abs delta S21 (dB) | bar (dB) | worst at (GHz) | bins compared | max abs delta S11 (dB), reported | verdict |
|---|---|---|---|---|---|
| 0.1346 | 0.5 | 4.8053 | 302 of 317 | 1.0607 | HELD |

**Reported, no window: what the 0.35-cell edge offset is worth.**

| A_on notch (GHz) | A_off notch (GHz) | difference (GHz) | difference (%) |
|---|---|---|---|
| 3.73232 | 3.74739 | -0.01507 | -0.4022 |

### R.5 Provenance

| arm | VESSL run | commit | dirty | jax | backend | started (UTC) |
|---|---|---|---|---|---|---|
| A_on | 369367263265 | 827d5ecf6021 | None | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T04:59:25+00:00 |
| A_off | 369367263264 | 827d5ecf6021 | None | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T04:59:23+00:00 |
| B_off | 369367263266 | 827d5ecf6021 | None | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T05:02:52+00:00 |
| C_off | 369367263283 | 827d5ecf6021 | None | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T05:15:49+00:00 |
| A_off_longarms | 369367263267 | 827d5ecf6021 | None | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T05:02:10+00:00 |

### Conclusions (leader, after reading the record, the five figures and R.0)

**The physics.** On a mesh that is fine only across the metal edges and through
the substrate, the stub notch converges from above along the substrate-cell
ladder 42.3 → 31.75 → 21.2 µm: 3.7474 → 3.7322 → 3.7166 GHz, the last two
rungs 0.42 % apart (W1 held). The convergence is FIRST order in the cell
(fitted 0.92), not second, and its limit is 3.682 GHz — 0.22 % above the
openEMS tutorial's 3.6744 GHz and 0.33 % from the uniform ladder's own
extrapolated 3.67 GHz (W3 held). The two ladders were not run on one board:
the graded board's line sits 2.159 mm from the y_lo absorber instead of the
case's 1.55 mm (R.0 item 1), and nothing here measures what that 609 µm does
to the notch; what the record supports is that the two ladders extrapolate
to limits 0.33 % (12 MHz) apart. What the finest rung actually reads is
still 1.15 % high, so the case's 1 % bar fired (W2). Its 2 dB magnitude
excess is only on the notch's two skirts (11 bins in 3.509–3.553 GHz and 13
in 3.837–3.890 GHz, every one 121–216 MHz from the reference notch, passband
maximum 0.64 dB — recomputed by the reviewer from the record, correcting the
intervals first written here): the frequency offset counted a second time.
The passband agrees; the notch depth reads −50.8 dB against the reference's
−53.4 dB, a 2.6 dB gap the bar does not judge (both curves are below the
−20 dB null level there).

**What the mesh buys.** At the substrate resolution of the uniform h/6 rung
(42.3 µm) the graded mesh solves in 106 s instead of 661 s (0.16) with the
same class of notch error (3.7474 vs the uniform rung's 3.7537 GHz, both about
2 % high). The graded C rung (21.2 µm substrate cell, 2.4 M interior cells,
1008 s) reaches a resolution the uniform mesh could not finish (h/8, 25 M
cells, > 73 min) and brings the notch to 1.15 %. A microstrip stub notch is
therefore reachable to about 1 % on this solver at 17 min on one RTX 4090,
where the uniform mesh stops at 2 % after 11 min and does not finish the next
rung.

**Where the residual is.** The edge-offset placement (a node 0.35 cell inside
each metal edge) RAISED the notch by 0.40 % at rung A relative to the on-node
placement (3.7474 vs 3.7323 GHz), i.e. away from the reference. The two arms
also differ in fine cell (47.2 vs 50.0 µm), band margin and the width the
port's Laplace feed sees, and every one of those differences works in the
opposite direction, so the sign stands and 0.40 % is a floor on the offset's
own effect (reviewer's reading of the record): the solved strip width is not
what keeps the notch high. The first-order term that
remains sits elsewhere — the sheet on the substrate-top plane (the field
singularity at a strip's edge in the plane normal to the sheet is resolved by
FZ, and every rung cuts FZ and F together, so the ladder cannot separate the
two), the T-junction, or the open end's fringing. Which one is a separate
declaration; nothing here identifies it.

**What this changes for a user.** A `dx_profile`/`dy_profile`/`dz_profile`
mesh built by hand around a microstrip stub notch reproduces an external
solver's notch to 1.15 % at the finest affordable rung and extrapolates to
0.22 %, at 0.16 of the uniform cost at equal substrate resolution. That is
one structure, one port family (MSL, `mode="laplace"`), ratio ≤ 1.3 in plane,
ports and absorbers on uniform runways. The support-matrix sentence for "MSL
S-matrix + nonuniform mesh" is queued on #1171 with this record as its
witness; the row does not change in this PR.

**Two facts about the instrument that a reader of the numbers needs.** The
board is the case's except that the line sits 2.159 mm from the y_lo face
instead of 1.55 mm (the absorber runway did not fit under the fine band;
R.0 item 1). And the arms were run twice: the first attempt could not name
its commit (`git archive` leaves nothing for `git rev-parse`; R.0 item 5) and
the instrument now refuses to record such an arm; the second attempt read
the same notch to every printed digit on all four repeated arms.

**Not re-run.** The finest rung missed the 1 % bar by 0.15 %; by the fitted
order a fourth rung (n_z = 16) would land near 0.9 %. That is a new
declaration with a stated purpose, not a re-run of this one.

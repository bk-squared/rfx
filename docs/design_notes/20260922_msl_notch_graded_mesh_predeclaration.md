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

# cv11 — post-#931 recompute (lattice ownership contract)

Branch `feat/931-crossval-b`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XB-crossval-b`.

This directory exists only to hold this note: cv11 writes its artifacts to
`tests/fixtures/waveguide_broad_e5/` and to its own stdout, not to a
`_11_*_results/` directory like cv06b and cv07.

## What changed, and why the case must be re-run

1. **The port-aperture trim is deleted.** Both waveguide ports carried
   `y_range=(0, A_WG_REALIZED - DX_M)` / `z_range=(0, B_WG_REALIZED - DX_M)`,
   an in-script correction for the port eigenproblem spanning n_nodes columns
   instead of n_cells. #889 fixed that in rfx
   (`_node_span_to_cell_span`), so the trim became the double correction the
   script's own SEQUENCING paragraph predicted. Measured build-only on this
   checkout:

   | | `cfg.f_cutoff` | vs quote-realized 6.517391 GHz |
   |---|---|---|
   | with the trim | 6.807677 GHz | +4.454 % |
   | without the trim | 6.512162 GHz | −0.080 % |

   The untrimmed value is exactly the 6.512162 GHz the docstring credited to
   the trim in 2026-08.

2. **The PEC short's realized walls are asserted** (`assert_realized_short`,
   no solve): 145.000 / 146.000 / 147.000 mm, the drawn 2 mm plug with walls
   on both faces and the interior shorted. The far face at 147 mm is new
   under the contract; it sits BEHIND the reflector. The reflection plane is
   the first wall, 145.000 mm, unchanged. The cell-relative `2 * DX_M` extent
   is replaced by the absolute `PEC_SHORT_T_M = 0.002`.

3. `A_WG_REALIZED` / `F_CUTOFF_TE10` (QUOTE-REALIZED) are NOT touched: they
   read the realized DOMAIN, whose walls are `BoundarySpec` faces, which
   design note §1.8 fences out of the body-ownership contract.

## The run

```
vessl run create -f scripts/vessl_931_xb/post-cv11.yaml        # from a non-git cwd
```

* **VESSL run id 369367259194**, name `rfx-931-post-cv11`, preset
  `gpu-rtx4090`, cluster `remilab-c0`, image
  `ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8`, `JAX_PLATFORMS=cpu` (the
  case pins `JAX_ENABLE_X64=0` for the solver carry and is a CPU case;
  measured 2 m 07 s on a 32-core pod).
* Pre-change baseline: run **369367259004** (`rfx-931-base-cv11`). It exits 1
  — cv11 is a diagnostic reporter and its FAIL against conj(MEEP) on the
  pec-short leg is the known stale state recorded in `manifest.json`
  `cases[11]`. Compare leg by leg, not by exit code.
* Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv11-<ts>/`.
* Expected runtime ~5 min including the pip step.

## Artifacts that change

* the RUN RESULT table in `11_waveguide_port_wr90.py`'s docstring (12 gate
  lines, measured 2026-08-28 WITH the trim — superseded, re-measure);
* `tests/fixtures/waveguide_broad_e5/cv11_wr90_main_baseline_stdout.txt`,
  `cv11_wr90_fresh_stdout.txt`, `cv11_wr90_witness_np400_stdout.txt`;
* `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`
  — already flagged stale in `manifest.json` `cases[11]` (0.0186 rebuilt vs
  0.0707 pinned, a 3.7× delta from a June→August code change). The #931
  re-run forces that decision: when it is refreshed, BOTH deltas must be
  explained (June→August code, August→post-#931 realization), never spliced;
* `validation/crossval/manifest.json` `cases[11].claim_scope` — replacement
  text in `docs/design_notes/931_migration/xb-manifest.md`.

## Pre-declared, before the run

1. The pec-short per-bin `|S11|` envelope returns from [0.9980, 1.0019]
   toward [0.9995, 1.0000], and `max_diff` 0.0020 → ~0.0005. Mechanism:
   deleting the trim restores `u_hi == u_grid_size`, which re-enables the
   PEC-ghost aperture-weight zeroing the 2026-04-27 DROP-weight fix depends
   on. An envelope that does NOT tighten falsifies that mechanism.

   **MEASURED — (1) IS FALSIFIED.** Run 369367259194 against baseline
   369367259004, both against the same committed reference tables:

   | leg | baseline (trim) | post (no trim) |
   |---|---|---|
   | pec-short `|S11|` max_diff | 0.0146 | **0.0560** (gate 0.050) |
   | pec-short round-trip phase | 9.99° max / 6.16° mean | **17.20° / 9.71°** (gate 15°) |
   | pec-short vs conj(MEEP) `∠S` | 27.12° / 22.26° | **10.98° / 6.81°** |
   | slab S11 `∠S` | 13.60° / 11.71° | **8.79° / 6.08°** |
   | slab S11 vs conj(MEEP) `∠S` | 25.37° / 21.67° | **18.58° / 16.03°** |
   | empty S11/S21, slab S21 | — | identical |

   Every phase leg improved, two by more than 2× including the external one;
   the pec-short magnitude leg got worse and crossed its gate, taking the
   internal round-trip phase gate with it.

   **ATTRIBUTED — run 369367259198**
   (`scripts/diagnostics/cv11_aperture_trim_ab.py`, both arms on ONE
   checkout, the trim as the only variable):

   | arm | `cfg.f_cutoff` | `|S11|` envelope | max dev |
   |---|---|---|---|
   | trim | 6.807677 GHz (+4.454 %) | [0.9289, 0.9811] | 0.0711 |
   | no_trim | 6.512162 GHz (−0.080 %) | [0.9440, 0.9888] | **0.0560** |

   Removing the trim IMPROVES the leg by 0.0152. The 0.0146 → 0.0560
   degradation is therefore **not the trim's** — about +0.057 of it belongs
   to the **#931 core**, on the waveguide S-matrix lane, which is where
   stage C (commit 0184d64c) replaced the `sigma = 1e10` cell fill with the
   realized PEC edges. That is the fold the inventory critic flagged as
   owned by no group. **This is a finding for the core owners, not something
   this case compensates for**, and it is the reason cv11's pec-short
   magnitude leg is now outside its 0.050 gate.

   The trim stays deleted on both counts: it solves a 22-cell guide where
   the walls make 23, and it is worse on the magnitude leg as well.
2. The round-trip phase leg stays near 3.26° max / 1.45° mean: its reference
   cutoff error is unchanged at −0.080 %, and the reflection plane did not
   move.
3. The slab legs (mag 0.0141 / 0.0023, phase 8.94°, complex 0.0654) move only
   if something touched dielectric sampling, which §1.8 says nothing did. A
   test pins that the slab still realizes 11 x-nodes, 95.000 → 105.000 mm.
4. No gate is tightened on the new envelope in this pass — "tighten the
   magnitude gates after #729 settles, on a re-measured envelope" is the
   script's own rule, and #729's port-aperture default is exactly what just
   moved.

## SUPERSEDED — run 369367259194 must not be ingested

Everything under "Pre-declared, before the run" above, including the
**MEASURED — (1) IS FALSIFIED** table and the **ATTRIBUTED** paragraph
that filed +0.057 against the #931 core, was measured on run
**369367259194**, whose PEC plug was still drawn to the DECLARED
22.86 x 10.16 mm cross-section. The grid realizes the guide by ceil as
23 x 11 mm and a volume's face rounds to the nearest node, so that plug's
top landed at z = 10.000 mm under a wall at 11.000 mm: a
1 x 22.86 x 2 mm vacuum slot along the top broad wall, a parallel-plate
line for Ez carrying |S21| 0.22-0.33 past the "short". Adjudicated in
`a8d59e86` / merged `1b0866db`; the per-arm evidence is
`scripts/diagnostics/pec_short_lane_ab.py` and its JSON under
`scripts/diagnostics/_artifacts/pec_short_lane_ab/`.

The attribution to the core lane's stage-C sigma fold was therefore
**wrong, and is withdrawn**. It was not a lane defect and no change was
made in `rfx/`. The pre-declaration (1) is not "falsified" after all: the
envelope does tighten once the plug closes the guide.

## MEASURED — run 369367259277, post-fix, three legs in one container

`vessl run create -f scripts/vessl_931_xb/post-cv11-v2.yaml` (from a
non-git cwd). Outputs under
`/root/workspace/claude-workspace/rfx/runs/issue931-post-cv11v2-20260907T191351Z/`.
Three legs back to back in ONE container so no environment term separates
the columns:

1. `main_baseline` — the campaign baseline worktree
   `rfx-baseline-d990e18c` (origin/main d990e18c). Reproduces campaign
   baseline run **369367259004** on every gate line, so the baseline is
   itself reproducible.
2. `fresh` — this branch, `NUM_PERIODS_LONG = 200`.
3. `witness_np400` — the same at 400 (the grep is the fixture's first
   line). All 27 gate lines reproduce leg 2 to the last printed digit
   except `[pec-short S11 vs conj(MEEP)] |S| max_diff` 0.1992 vs 0.1991.

Each leg's stdout is committed as
`tests/fixtures/waveguide_broad_e5/cv11_wr90_{main_baseline,fresh,witness_np400}_stdout.txt`
with a provenance header; the full before/after table is now the case
docstring's RUN RESULT section.

| leg | main d990e18c | this branch | gate |
|---|---|---|---|
| pec-short \|S11\| max_diff | 0.0146 | **0.0020** | 0.050 |
| pec-short per-bin envelope | [0.9854, 0.9938] | **[0.9980, 1.0019]** | band [0.93, 1.07] |
| pec-short round-trip phase | 9.99° / 6.16° | **3.26° / 1.45°** | 15° |
| pec-short vs conj(MEEP) ∠S | 27.12° / 22.26° | **21.68° / 17.55°** | 10° (both fail) |
| slab S11 ∠S | 13.60° / 11.71° | **8.79° / 6.08°** | 60° |
| slab S11 vs conj(MEEP) ∠S | 25.37° / 21.67° | **18.58° / 16.03°** | 10° (both fail) |
| slab S11 \|S\| max_diff | 0.1468 | 0.1487 | 0.100 (**both fail** — see below) |
| slab S21, empty | — | identical | — |

Pre-declaration (1) HOLDS on the fixed drawing: the envelope tightens to
[0.9980, 1.0019] and max_diff falls 0.0146 → 0.0020. (2) holds: the
round-trip phase leg is 3.26° / 1.45°. (3) holds for the slab phase legs,
which improve; the slab MAGNITUDE leg is a separate matter, below. (4)
holds — no gate is tightened on the new envelope.

## THE ONE THING THIS RUN FOUND THAT IS NOT #931's — for the core owners

`[slab S11] |S| max_diff` is **0.1468 on origin/main d990e18c** and 0.1487
on this branch, against a 0.100 gate. **Both fail.** The 2026-08-28
revision of the case docstring measured the same line at 0.0186 on main
(cdc38bc8) and 0.0141 on its branch. So the leg degraded ~8x **on main**
between cdc38bc8 and d990e18c; #931 adds +0.0019, a last-digit move.

Confirmed independently through the external referee — same builder, same
`--reference-column Palace_r_h2`, three inputs, no FDTD:

| input | slab S11 max_mag_abs_diff | status |
|---|---|---|
| committed 2026-08-28 branch stdout | 0.0194 | passed |
| 2026-09-07 origin/main d990e18c | **0.1482** | **FAILED** |
| 2026-09-07 this branch | **0.1500** | **FAILED** |

`slab S21` moves with it (0.0045 → 0.0343 on main) and the empty guide
does not move at all, so it is a dielectric-slab-lane term, not a port or
normalization term. Not diagnosed here: cv11 is a reporter, the
regression predates this branch, and a ten-day bisect of main is not a
guess to make inside an ingest.

## THE BROAD-E4 REFRESH IS BLOCKED, DELIBERATELY

`tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`
is **not** refreshed in this pass and its numbers are untouched. Rebuilding
it from the new stdout would set slab S11 `max_mag_abs_diff` 0.0707 →
0.1500, past its own `max_mag_abs_tol = 0.1`, and flip `status` to
`failed`. That would publish a failing external claim and date-stamp a
main-line regression as a #931 result. The file's own `provenance` rule —
a refresh must EXPLAIN its delta, never re-pin it — cannot be satisfied
while one of the deltas is unattributed. Widening the tolerance to make it
pass is not on the table.

Three deltas now separate the committed 0.0707 from a rebuild, and a
refresh must account for all three:

1. **2026-06-16 → 2026-08-28** (the delta the provenance block already
   names): 0.0707 → 0.0186–0.0194, ~3.7x BETTER. Candidate causes listed
   there, never attributed per commit.
2. **2026-08-28 → 2026-09-07 on main**: 0.0194 → 0.1482, ~7.6x WORSE.
   New, unattributed, and the blocking one.
3. **origin/main → #931**: 0.1482 → 0.1500, +1.2%. Measured here.

Refresh it when (2) is attributed, in the same pass that fixes or accepts
it, and write all three deltas into the `provenance` block.

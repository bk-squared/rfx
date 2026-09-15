# Isolated-patch ring-down under a padded lateral domain and a thin CPML — diagnosis (#801)

Branch `diag/801-patch-ringdown-padding`. Pre-declaration: `PREDECLARATION.md` in this
directory (question, observable, gate and falsifier written before any solve; Addendum A
holds the no-FDTD results, B the GPU arm table, C the follow-on ladder). Every number below
is quoted from an artifact in this directory. Times KST.

## The question that was asked first

#801 records an isolated patch whose ring-down grows when the lateral domain is padded
+10h under a `cpml_layers = 2n` absorber, while the same padding with 16 layers settles.
PR #1047 (CPML ψ-coefficient) and PR #1057 (absorber-pad material continuation) landed on
2026-09-15 and fix a mechanism that also reads as exponential growth in a CPML pad. So:
**is #801 the #1043 class?**

## Answer: no, on three independent witnesses

**1. The mechanism cannot fire on this rig.** #1043 needs the E update's permittivity and
the ψ coefficient's permittivity to disagree, which happens only on the subpixel-smoothed
lane. The rig calls `sim.run(num_periods=...)` and passes no `subpixel_smoothing`; the
parameter defaults to `False` (`rfx/api/_execute.py:421`), so `rfx/simulation.py:1481`
calls `apply_cpml_e(..., materials=materials, inv_eps_r_update=None)` and both halves read
one array. PR #1047's own commit message says the same thing: *"Every other
`apply_cpml_e` caller runs `update_e` with `materials.eps_r` and has nothing to thread."*

**2. rfx's own amplification model puts this arm on the unit circle.**
`amplification_at_this_arm.py` runs the model that owns the #1047 analysis
(`tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py::amplification_rho`)
at this arm's `dx = h/4` and `dt = 3.751186055802254e-13 s`:

| ε_a (Yee half) | ε_b (ψ coefficient) | ρ at 6 / 8 / 16 / 32 layers |
|---|---|---|
| 3.38 | 3.38 — **what this arm assembles** | 1.000000000000 at all four |
| 3.38 | 1.0 — the #1043 pair | 2.2446 / 2.2785 / 2.2499 / 2.0921 |
| 1.0 | 3.38 — the reversed pair | 1.000000000000 at all four |
| 1.0 | 1.0 | 1.000000000000 at all four |

Comparator first: a lossless slice with no absorber reads 1.0000000000000029, a vacuum
CPML slice 1.0. Note the second row: when #1043 *does* fire it is nearly **independent of
layer count**. #801's ladder is 8-layer unstable and 16-layer stable, so that mechanism
could not have produced it even if it had been live.

**3. Pre-#1047 and post-#1047 are bit-identical here.** VESSL 369367261204 ran the arm on
main `fc7f7202` and on `c0435798` = `541f703f^`, the commit immediately before #1047, with
everything else held. The raw probe series compare element-wise at **max|diff| = 0.000e+00**
at 8 layers and at 16 layers (peak amplitude 1.793e-02), and the per-probe settling values
agree to every printed digit. The smoothed lane *did* move (max|diff| 2.112e-04), so the fix
is live and reaches this geometry — it just is not on the path this rig takes.

## What the arms actually read

VESSL 369367261204, `gpu-rtx4090`, jax 0.4.33.dev20241023 on cuda:0, float32, n = 4,
+10h lateral pad, 150 periods = 26659 steps. Full table `gpu_369367261204/summary.txt`.

| arm | tree | cpml | settling dB | settled | last-30 % rate /step |
|---|---|---|---|---|---|
| `main_cpml8` | fc7f7202 | 8 | **−43.37** | yes | −1.79e-4 |
| `main_cpml16` | fc7f7202 | 16 | −45.83 | yes | −1.96e-4 |
| `pre1047_cpml8` | c0435798 | 8 | **−43.37** | yes | −1.79e-4 |
| `pre1047_cpml16` | c0435798 | 16 | −45.83 | yes | −1.96e-4 |
| `main_cpml8_noulp` | fc7f7202 | 8 | −43.30 | yes | −1.93e-4 |
| `main_cpml8_subpixel` | fc7f7202 | 8 | −45.89 | yes | −1.73e-4 |
| `pre1047_cpml8_subpixel` | c0435798 | 8 | −45.02 | yes | −1.56e-4 |

Against the issue's own record on tree `fa3a99bd`, re-scored here by the same function
(`recorded_series_reread.json`, which reproduces all eight published `settling_db` values
exactly):

| arm | cpml | settling dB | last-30 % rate /step |
|---|---|---|---|
| n4 +6h | 8 | −51.39 | −2.18e-4 |
| n4 +8h | 8 | −50.09 | −2.10e-4 |
| **n4 +10h** | **8** | **0.00** | **+4.39e-4** |
| n4 +12h | 8 | −25.54 | +1.75e-4 |
| n4 +10h | 16 | −51.60 | −2.25e-4 |
| n3 +10h | 6 | 0.00 | +1.27e-3 |

Figure: `ringdown_envelopes.png`. Left panel, the recorded arms: every arm decays together
at −2.1e-4/step to about −40 dB by 6 ns, and only then do three of them turn around. The n3
arm falls to −87 dB first and grows back from there, so the growing thing is seeded at
round-off, not excited by the source. The growth rate is shared by all four spatially
separated probes to three significant figures (+4.39e-4 / +4.39e-4 / +4.38e-4 / +4.44e-4),
which is one eigenvalue of the update operator and not a property of any probe's position.
Right panel, the seven arms above: no separation at all.

## The finding that stops this from being "#1057 resolved it"

The pre-declared cheap falsifier was that either tree must reproduce the issue's recorded
stable control, −51.60 dB at 16 layers, within 1 dB. **It failed: −45.83 dB, 5.77 dB
outside the band.** The rig's own raster says why — same driver, same declared geometry,
different realized board:

| | fa3a99bd (recorded) | main fc7f7202 |
|---|---|---|
| cavity | 983.75 µm, 5 cells | **787.00 µm, 4 cells** |
| Σ d/ε | 291.05 µm | **232.84 µm** |
| wall planes | 3935.0 µm | **3935.0, 4131.75, 4918.75, 5115.5 µm** |
| preflight on the ground | *"modelled as a one-cell PEC surface — tangential E is zeroed on it and the normal component survives as surface charge"* | *"realized as a filled slab with walls on BOTH faces (every E edge between the two faces is shorted; lattice ownership contract #931 §1.2)"* |

#931's lattice-ownership contract turned the one-cell PEC ground Box from a sheet into a
filled slab, which shortened the electrical cavity by 20 %. The board that grew and the
board that settles are **not the same board**, and the CPML work is bit-identical here. So
the correct statement is *this fixture no longer excites the instability on main*, not
*the instability is fixed*. `main_cpml8` also clears the −40 dB bar by only 3.4 dB, 2.5 dB
less margin than its own 16-layer sibling.

## The lateral-pad ladder on main's board (H3, pre-declared in Addendum C)

Because "settles on main" was a statement about a different board, the issue's own
discriminator was re-run on main's board. VESSL 369367261205, 8 arms, rc 0, artifacts in
`gpu_369367261205/`.

| lateral pad | cpml 8 (= 2n) | cpml 16 |
|---|---|---|
| +6h | −45.26 | — |
| +8h | −44.22 | — |
| +10h | −43.37 | −45.83 (from 369367261204) |
| +12h | −43.42 | — |
| +14h | −43.37 | −46.00 |
| +16h | **−42.95** | −44.97 |

**Every arm settles.** Residual `max(−42.95 + 40, 0) = 0`, so the growth is **not
reproducible on main's board** anywhere in +6h…+16h — past the +10h and +12h where the old
board grew, and past the range this issue explored. The ladder's own falsifier held: +6h and
+8h were comfortably stable on the old board and are comfortably stable here.

Margin does erode monotonically with clearance (−45.26 → −42.95 dB from +6h to +16h) and the
whole family sits 6–8 dB shallower than the old board's stable arms (−50 … −51.6 dB), which
is the shorter cavity, not the absorber.

## The pre-#931 board restored on main (H4, pre-declared in Addendum D)

If the board changing is what stopped the growth, restoring it should bring the growth back.
`--sheet-conductors` declares the ground and the patch as zero-thickness Boxes — SHEETs, one
node plane each with the normal E edge live — which is what the preflight now advises for foil
and what these declarations realized before #931.

**Reconstruction falsifier, run before any solve and again inside the GPU run: PASSED on every
field.** Recorded fa3a99bd vs main `--sheet-conductors`: walls 3934.99992787838 / 4918.749909847975
vs 3935.0 / 4918.75 µm; cavity 983.75 µm over 5 cells both; `k_gnd` 28, `k_patch` 33 both; patch
raster 44 × 52 both; **`sum(d/eps)` 291.050286003532 µm on both, to all twelve printed digits.**
(`n_pec_sheets = 2`, `has_cell_mask = False` — the conductor is read from tangential E edges.)

VESSL 369367261209, 7 arms, rc 0, artifacts in `gpu_369367261209/`:

| arm | cpml | settling dB | settled | recorded on the same board at fa3a99bd |
|---|---|---|---|---|
| n4 +6h | 8 | −57.28 | yes | −51.39 |
| n4 +8h | 8 | −55.26 | yes | −50.09 |
| **n4 +10h** | **8** | **−52.78** | **yes** | **0.00, +4.39e-4/step** |
| n4 +12h | 8 | −54.25 | yes | −25.54, +1.75e-4/step |
| n4 +10h | 16 | −56.87 | yes | −51.60 |
| n4 +10h | 32 | −58.26 | yes | — |
| **n3 +10h** | **6** | **−47.56** | **yes** | **0.00, +1.27e-3/step** |

**H4 REFUTED**, residual `max(−47.56 + 40, 0) = 0`. Every arm settles, including the n = 3 arm
that grew hardest of all on the old board. Restoring the board does **not** restore the growth.

Addendum D declared what that reading means before it was taken: the reconstruction matched the
raster field for field and the arm still settles, so the change responsible is **neither #931's
board change nor #1047/#1057**, and the next step is a bisect rather than another mechanism
hypothesis. Note also that every sheet-board arm is 4–6 dB deeper than its fa3a99bd twin
(−57.28 vs −51.39 at +6h, where both are comfortably stable), so the dynamics differ across the
whole family and not only at the arm that used to grow.

## A second, separate defect found on the way (not the growth)

At +10h the entire +x absorber pad is solved as **vacuum** while the substrate continues
into the −x, +y and −y pads at ε_r 3.38, and the substrate ends two nodes inside the
interior (`pad_material_map.py`, `padmap_n4_pad10_cpml8.json`).

Cause (`pad_facet_rounding.py`): the rig declares its domain as `(38 + 2·pad)·h` and solves
at `dx = h/4`, and that ratio carries a one-ULP excess for some pad values —
232.00000000000003 cells on x at +10h, 156.00000000000003 on y at +8h, 188.00000000000003
on y at +12h, exact at +6h. rfx allocates one more cell than any declared Box fills, the
extra node is vacuum, and `extend_cpml_pad_materials` replicates **that** node through the
whole absorber on that face. The dielectric then ends in a vacuum facet at the interior/pad
seam — the staircase-lane twin of the smoothed-lane facet #831 was filed for and #1057
fixed, reached through grid sizing instead of through the smoothing rebuild.

Falsifier, run: shrinking the declared domain length by one ULP removes the extra cell and
the facet on +8h, +10h and +12h, and changes nothing at +6h.

**It is not the growth.** `main_cpml8_noulp` (facet removed) reads −43.30 dB against
`main_cpml8`'s −43.37 dB, a 0.07 dB difference inside the pre-declared 3 dB band, so H2 is
refuted. The recorded arms say the same from the other side: the facet is present at +8h
(settled −50.09 dB) and at +10h with 16 layers (settled −51.60 dB). It is still worth its
own ticket: a physically null lateral padding silently changes the solved structure, which
is exactly the invariance the rig was built to test.

## Verdict

* **Same class as #1043? No.** Three witnesses: the lane cannot reach the defect; the
  amplification model reads ρ = 1 for this arm's ε pair and is layer-count-flat when it does
  fire; the two trees are bit-identical.
* **Does the arm still grow on main? No** — and not at any lateral pad from +6h to +16h, and
  not with the pre-#931 board restored, and not on the n = 3 arm that grew hardest.
* **What removed it is still unidentified.** It is not #1047/#1057 (bit-identical) and not
  #931's board change alone (restoring the board field-for-field does not restore the growth).
  22 arms across three GPU jobs decay; the recorded arms grew. Something else in the 734
  commits between `fa3a99bd` and `fc7f7202` is responsible.
* **Rig/solver split.** Nothing here asks for a CPML change. The vacuum-pad facet is a
  solver-side surprise (grid sizing feeding the pad continuation) with a rig-side trigger (a
  domain length that is not an exact multiple of `dx`); either side can own it, as its own
  ticket.

## What to do next — a bisect, pre-sized

The next step is localization, not another mechanism hypothesis, and Addendum D committed to
that reading before the measurement was taken. It is cheap: `fa3a99bd..fc7f7202` is 734
commits, so a `git bisect` is 10 steps, and an arm is 26 s of rtx4090 — about 10 minutes of GPU
plus fetch overhead.

* **Arm**: `--n 4 --pad 10 --periods 150` with the rig's own default absorber (`2n = 8`), the
  board AS DECLARED (no `--sheet-conductors`) — the configuration that grew.
* **Criterion**: `bad` (grows) iff `settling_db > -40 dB`; `good` iff it settles.
* **Known endpoints**: `fa3a99bd` bad (0.00 dB), `fc7f7202` good (−43.37 dB).
* **Portability note for whoever runs it**: `raster()` in this lane's driver spans the #931
  rename (`tangential_edge_masks` → `realized_pec_edge_masks`) and the sheet-collector
  signature change in `_assemble_materials`, but intermediate commits may break other
  reporting calls. Add a `--no-raster` escape before starting rather than discovering it at
  bisect step 6.

Two candidate areas to look at first, from `git log fa3a99bd..fc7f7202 -- rfx/boundaries/
rfx/simulation.py rfx/geometry/` (≈ 37 commits): the #931 lattice-ownership stack (a3e4dba4
through f112f7bb — the board change is only one of its effects; it also rewrote how every lane
realizes and applies PEC edges) and `a65c6626` *"never promote a pole-carrying column's statics
into a hi-face pad"*, which is a CPML-pad materials change on the same face family this lane
found the vacuum facet on.

## R2 accounting

* H1 (is this the #1043 class) — attempt 1, **CLOSED REFUTED**, residual
  `r_ref = max(|S_main − S_pre| − 1.0, 0) = 0` in its strongest form (bit-identical).
* H2 (the vacuum pad facet is the growth) — attempt 1, **CLOSED REFUTED**, residual
  `r_H2 = max(0.07 − 3.0, 0) = 0`.
* H3 (does it still exist on main's board at some clearance) — attempt 1, **CLOSED: not
  reproducible**, residual `max(−42.95 + 40, 0) = 0`.
* H4 (does restoring the pre-#931 board restore it) — attempt 1, **CLOSED REFUTED**, residual
  `max(−47.56 + 40, 0) = 0`.

Four hypotheses, four closures, no repeats. The next action is a bisect, which is localization
under a criterion already fixed, not a fifth mechanism attempt.

## Artifacts

| what | where |
|---|---|
| pre-declaration | `PREDECLARATION.md` |
| driver (one arm per process) | `scripts/diagnostics/patch_pad_cpml_ringdown.py` |
| table printer | `scripts/diagnostics/patch_pad_cpml_ringdown_summary.py` |
| GPU job | `scripts/vessl_patch_pad_cpml_ringdown.yaml`, VESSL 369367261204 |
| GPU arm JSONs + log | `gpu_369367261204/` |
| the issue's own arms, re-scored | `read_recorded_series.py`, `recorded_series_reread.json` |
| amplification model at this arm | `amplification_at_this_arm.py`, `.json` |
| pad material map | `pad_material_map.py`, `padmap_*.json` |
| the ULP falsifier | `pad_facet_rounding.py`, `.json` |
| figure | `.../runs/patch-pad-cpml-ringdown-20260915T131410Z-834a43e7/ringdown_envelopes.png` (see below) |
| ladder job | `scripts/vessl_patch_pad_cpml_ringdown_ladder.yaml`, VESSL 369367261205, arms in `gpu_369367261205/` |
| restored-board job | `scripts/vessl_patch_pad_cpml_ringdown_sheet.yaml`, VESSL 369367261209, arms in `gpu_369367261209/` |

Raw probe series (`*_ts.npz`, 427 KB per arm) stay on NFS at
`/root/workspace/claude-workspace/rfx/runs/patch-pad-cpml-ringdown-20260915T131410Z-834a43e7/`;
the JSONs here carry the decimated envelope traces the figure is drawn from.

`**/*.png` is gitignored repo-wide (`.gitignore:66`), so the rendered figure lives beside
those series in the same NFS run directory rather than in the tree. Everything it is drawn
from IS committed here, so it regenerates byte-for-byte from the repo alone:

```sh
python3 scripts/diagnostics/_artifacts/patch_pad_cpml_ringdown/plot_ringdown_envelopes.py \
  --dir scripts/diagnostics/_artifacts/patch_pad_cpml_ringdown \
  --arms-dir scripts/diagnostics/_artifacts/patch_pad_cpml_ringdown/gpu_369367261204 \
  --arms-title "the same arms on main fc7f7202 and on 541f703f^ (VESSL 369367261204)" \
  --out /tmp/ringdown_envelopes.png
```

(The left panel reads the primary checkout's recorded `.npz` files through
`read_recorded_series.py`; `recorded_series_reread.json` is committed, so the plot needs
only this repo.)

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

## The bisect: it is `a3e4dba4`, #931 stage A (VESSL 369367261218)

Pre-declared in Addendum E before it ran: range `fa3a99bd..fc7f7202` (734 commits), the arm
the issue measured (n4, +10h, cpml 8, board AS DECLARED), criterion `settling_db > -40 dB` =
grows, and the predicate inverted so git's "first bad commit" is the first commit where the
growth STOPPED.

**Endpoint falsifier, run before the bisect: both reproduced.** `fa3a99bd` grows at 0.00 dB
with per-probe rates +4.389e-4 / +4.391e-4 / +4.384e-4 / +4.443e-4 per step — the issue's own
numbers, reproduced directly on this hardware for the first time (until now only its recorded
series had been re-scored). `fc7f7202` settles at -43.37 dB.

**Result: `a3e4dba46fff45d65743b02ef65afb3185d0c15b`** — *"feat(pec): lattice ownership
contract — one realized-edge source for volumes, sheets, wires; two_plane deleted (#931,
stage A)"*, PR **#931**, 2026-09-07. `bisect_run.rc` 0, ten steps, no skips.

Its parent `0ad801dc` is **documentation only** (*"docs(design_notes): lattice ownership
contract for conductors…"*), so the A/B the PI asked for is the bisect's own last two steps
and nothing else moved between them:

| commit | what it is | settling dB | per-probe rate /step |
|---|---|---|---|
| `fa3a99bd` | the issue's tree | 0.00 | +4.389e-4 / +4.391e-4 / +4.384e-4 / +4.443e-4 |
| `0ad801dc` | `a3e4dba4^`, docs only | **0.00** | **the same four values** |
| `a3e4dba4` | #931 stage A | **-43.37** | -1.861e-4 / -2.007e-4 / -1.791e-4 / -2.023e-4 |
| `fc7f7202` | main | -43.37 | the same four values |

The raw series say it more sharply than the summary does: `fa3a99bd` and `0ad801dc` are
**bit-identical** (max|diff| 0.000e+00), `a3e4dba4` and `fc7f7202` are **bit-identical**, and
across the boundary max|diff| is 1.571e-02 on a peak of 1.632e-02. The 734-commit range
collapses to one commit; everything before it reproduces the issue exactly and everything
after it reproduces main exactly.

### Mechanism, one paragraph

`a3e4dba4` replaced the rule that decides which E edges a conductor zeroes. The old
`tangential_edge_masks` selected a component *"iff the body extends >= 2 cells in that
component's direction"*, so **a one-cell-thick PEC Box selected only its in-plane components
and left the normal edge through its own cell live**; the new contract is *"an E component is
PEC iff its own location is inside the closed conductor region"*, which shorts every edge
between a one-cell Box's two faces. This fixture is two one-cell PEC Boxes — a ground plane
and a patch — and the ground spans the entire lateral domain, so the rule change acts along
the whole conductor including where it runs into the absorber pads. Its visible consequence on
this board is the one already recorded above: wall planes 1 -> 4, cavity 983.75 -> 787.00 um.
But that is *not* the whole of it, and the H4 arm is what proves it: declaring the same board
as SHEETs on main reproduces the old raster field-for-field (walls, cavity, `sum(d/eps)` to
twelve digits) and still settles at -52.78 dB. So within `a3e4dba4` the operative difference
for the ring-down is in the **realized edge set**, not in the wall planes it moved — which
edge, exactly, is a diff-level question this lane did not open.

### What this makes #801

Not a CPML-parameterisation ticket, and not #1043. The growth was a property of the pre-#931
PEC edge rule on a board whose conductors reach the absorber, and #931 stage A ended it — as a
side effect of a contract change made for other reasons, with no test pinning the ring-down.
The 8/12/16/24-layer ladder the issue proposed would not have found this: the layer count was
never what moved.

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

Filed as **#1070**, with the guard that should have caught it named: `extend_cpml_pad_materials` already has the **#627a hi-face fallback** for this shape, but its docstring bounds it to *"exactly one column inward — ... never more"*. That bound is right for the half-open Box rule alone; the `ceil` overshoot adds a SECOND empty node, so the fallback inspects the interior edge (vacuum), looks one column in (also vacuum) and gives up. Two correct rules composing into a wrong board — measured per pad value in `pad_facet_why_627a_underreaches.py`. There is no existing lock on "every interior cell a declared Box covers is actually filled"; the nearest family (#802/#807, `test_rasterization_coordinate_exactness.py`) pins node coordinates and cross-lane agreement, and both lanes agree here while both are wrong the same way.

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
* **What removed it: `a3e4dba4`, #931 stage A** (the lattice-ownership contract, PR #931),
  found by bisect and confirmed against a documentation-only parent. It is not #1047/#1057
  (bit-identical) and not #931's cavity change alone (restoring the board field-for-field does
  not restore the growth) — it is that commit's change to the realized PEC edge set.
* **Rig/solver split.** Nothing here asks for a CPML change. The vacuum-pad facet is a
  solver-side surprise (grid sizing feeding the pad continuation) with a rig-side trigger (a
  domain length that is not an exact multiple of `dx`); either side can own it, as its own
  ticket.

## What remains

* Which edge in `a3e4dba4`'s new rule carries the difference — the wall planes it moved are
  demonstrably not it (H4). A diff-level question on the realized edge masks at the pad seam,
  answerable without FDTD by comparing the two rules' edge sets on this fixture.
* Nothing pins this. The growth ended as a side effect of a contract change made for other
  reasons; no test covers "a lossless open-domain ring-down does not grow". A two-sided
  settling gate on a padded isolated-patch arm would be the lock, if the PI wants one.
* **#1070**, the vacuum absorber pad, is independent of all of the above.

## R2 accounting

* H1 (is this the #1043 class) — attempt 1, **CLOSED REFUTED**, residual
  `r_ref = max(|S_main − S_pre| − 1.0, 0) = 0` in its strongest form (bit-identical).
* H2 (the vacuum pad facet is the growth) — attempt 1, **CLOSED REFUTED**, residual
  `r_H2 = max(0.07 − 3.0, 0) = 0`.
* H3 (does it still exist on main's board at some clearance) — attempt 1, **CLOSED: not
  reproducible**, residual `max(−42.95 + 40, 0) = 0`.
* H4 (does restoring the pre-#931 board restore it) — attempt 1, **CLOSED REFUTED**, residual
  `max(−47.56 + 40, 0) = 0`.

Four hypotheses, four closures, no repeats. The bisect (VESSL 369367261218) was localization
under a criterion fixed in Addendum E before it ran, not a fifth mechanism attempt, and it
closed on `a3e4dba4` with a documentation-only parent as its control.

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
| why #627a under-reaches | `pad_facet_why_627a_underreaches.py`, `.json` (filed as #1070) |
| figure | `.../runs/patch-pad-cpml-ringdown-20260915T131410Z-834a43e7/ringdown_envelopes.png` (see below) |
| ladder job | `scripts/vessl_patch_pad_cpml_ringdown_ladder.yaml`, VESSL 369367261205, arms in `gpu_369367261205/` |
| restored-board job | `scripts/vessl_patch_pad_cpml_ringdown_sheet.yaml`, VESSL 369367261209, arms in `gpu_369367261209/` |
| bisect job | `scripts/vessl_patch_pad_cpml_ringdown_bisect.yaml` + `patch_pad_cpml_ringdown_bisect_step.sh`, VESSL 369367261218, trace and landing diff in `gpu_369367261218/` |

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

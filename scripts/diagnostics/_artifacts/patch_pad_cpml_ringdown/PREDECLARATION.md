# Pre-declaration — isolated-patch ring-down under a padded lateral domain and a thin CPML (#801)

Written BEFORE any FDTD run in this lane. KST timestamps.

## Question (attempt 1, one hypothesis)

**H1: the growth recorded in #801 is the #1043 class** — the CPML psi coefficient reading a
different permittivity than the Yee half of the same E update (PR #1047, main 541f703f), with
or without the boundary-touching-dielectric pad continuation of PR #1057 (main 43c35f03).

If H1 holds, the same rig on today's main is stable and #801 is resolved by work already
landed; if it does not, the growth is a different mechanism and the diagnosis continues.

## Rig (the one the issue names, unchanged)

`scripts/diagnostics/_artifacts/patch_close_20260830/harnesses/wt-refnull/refute_nulltf_ladder.py`
(primary checkout, untracked there). `build()` and `raster()` are copied VERBATIM into this
lane's driver; only the reporting is extended (full envelope trace, provenance, dirty flag).

Arm under test: `--n 4 --pad 10 --periods 150`, so `cpml_layers = 2n = 8`, boundary `cpml`,
grid 250 x 189 x 81, 26659 steps, dt 3.751186055802254e-13 s. Control arm: the same with
`--cpml 16`.

Recorded at issue time (tree fa3a99bd, rtx4090, VESSL 369367257192/203/199):

| arm | cpml_layers | settling_db (worst probe) | settled |
|---|---|---|---|
| n4 pad10 | 8 | **0.0** (grew to the peak) | False |
| n4 pad12 | 8 | -25.54 | False |
| n4 pad10 | 16 | -51.60 | True |
| n4 pad8 | 8 | -50.09 | True |

## Observable

`settling_db` = worst over the four parity probes of `20*log10(max|s| over the last 5% of the
run / max|s| over the whole run)` — the harness's own arithmetic, and the same end/peak
arithmetic as rfx's shipped ring-down witness (`rfx/probes/settling.py`, #885). The full
per-probe envelope trace is dumped beside it (R5).

## Gate, committed before the run (nonnegative residual)

Let `S_main` and `S_pre` be `settling_db` from the identical arm on main fc7f7202 and on the
issue's own tree fa3a99bd.

* **H1 CONFIRMED** iff `S_main <= -40 dB` (the harness's own bar) **and** `S_pre > -40 dB`,
  i.e. the arm that grew now settles. Residual `r_conf = max(S_main + 40, 0)`; 0 = confirmed.
* **H1 REFUTED** iff `|S_main - S_pre| <= 1.0 dB` with both `> -40 dB`. Residual
  `r_ref = max(|S_main - S_pre| - 1.0, 0)`; 0 = the two trees agree and the growth is untouched.
* Anything else (a move that is neither a fix nor agreement) is **non-closing**: record it,
  do not reinterpret the gate.

A bit-identical time series between the two trees (`max|ts_main - ts_pre| == 0` exactly) is a
stronger form of REFUTED and is checked explicitly.

## Cheap falsifier for the run itself

Both trees must reproduce the issue's recorded stable control (`--cpml 16`, pad 10) at
`settling_db` within 1 dB of -51.60. A rig that cannot reproduce its own recorded stable arm
is not measuring the issue.

## Read-only prior (R1) — cited before the run

* `docs/agent-memory/rfx-known-issues.md`, "Added 2026-09-15", #1043 stage A: *"`apply_cpml_e`
  built its psi coefficient from the staircase `materials.eps_r` while the Yee half of the same
  step used the smoothed `aniso_eps`; with kappa=1 the curl coefficient is proportional to
  1/eps_a + c/eps_b, negative whenever eps_a > eps_b -> exponential growth"*. Consistent with
  this lane's question; the mechanism is stated as conditional on the SMOOTHED lane.
* Same section, #831: *"`run(subpixel_smoothing=True)` rebuilds the update permittivity from
  `sim._geometry` ... WITHOUT the `extend_cpml_pad_materials` step"*. The rig here never passes
  `subpixel_smoothing` and the parameter defaults to `False` (`rfx/api/_execute.py:421`), which
  predicts H1 is REFUTED. The run is what decides it, not the reading.
* Same file, resolved log: *"CPML guided-mode reflection ~12% at default 8-10 layers ... 10
  layers -> 11.7 %, 20 -> 4.2 %, 40 -> 1.8 % ... `kappa_max > 1` makes guided-mode absorption
  WORSE"*. A substrate-plus-ground stack continued into a thin lateral pad is exactly that
  geometry, which is why the layer count is the issue's own discriminator.

## R2 accounting

This is attempt 1 on mechanism hypothesis H1. Under the RF/EM intensifier a second attempt on
the same hypothesis needs a named new falsifier or an identified defect in attempt 1, in
writing, before it runs.

---

## Addendum A — measured before the FDTD arms ran (no solve involved)

Three no-FDTD results were taken while the timing probe ran. They are recorded here, in
the pre-declaration, because they narrow H1 before the arms report and must not look like
post-hoc reasoning.

1. **`rfx/api/_execute.py`: `subpixel_smoothing: bool | str = False`.** The rig calls
   `sim.run(num_periods=...)` and passes nothing, so it is on the STAIRCASE lane.
   `rfx/simulation.py` passes `materials=materials` to `apply_cpml_e` with
   `inv_eps_r_update=None` there, so the psi coefficient and the Yee half read one array.
   #1047's own commit message says it: *"Every other `apply_cpml_e` caller runs `update_e`
   with `materials.eps_r` and has nothing to thread."*
2. **rfx's own amplification model at this arm's numbers** (`amplification_at_this_arm.py`,
   model owner `tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py`):
   with eps_a = eps_b = 3.38 the spectral radius is 1.000000000000 at 6, 8, 16 and 32
   layers. The #1043 pair (eps_a 3.38, eps_b 1.0) gives 2.24 / 2.28 / 2.25 / 2.09 at the
   same four layer counts -- **almost independent of layer count**, so even a firing #1043
   could not produce #801's 8-unstable / 16-stable ladder.
3. **The lateral pads do not all carry the substrate** (`pad_material_map.py`,
   `pad_facet_rounding.py`). At pad = +10h the +x absorber pad is entirely vacuum and the
   substrate ends two nodes inside the interior; at +8h and +12h the same happens on +y;
   at +6h it does not happen at all. Cause: `(38 + 2*pad)*h / (h/4)` carries a one-ULP
   excess (232.00000000000003 at +10h, 156.00000000000003 on y at +8h, 188.00000000000003
   on y at +12h, exact at +6h), rfx allocates one more cell than any Box fills, and
   `extend_cpml_pad_materials` replicates that vacuum node through the pad. Falsifier RUN:
   shrinking the declared domain by one ULP removes the extra cell and the facet on every
   affected arm, and changes nothing at +6h.

None of the three is a verdict on the growth; (3) is a separate defect and (1)+(2) bear on
H1 only.

## Addendum B — GPU arms, pre-declared before submission

The 20-period CPU timing probe took 7.39 min for 3555 steps (443 s), so the 26659-step arm
is ~55 min of CPU -- past this lane's 30-minute bar, and on different hardware than the
issue's own numbers (rtx4090). The arms go to `gpu-rtx4090`, which also removes the
CPU-versus-GPU confounder from the before/after table.

| arm | tree | cpml | domain | subpixel | what it decides |
|---|---|---|---|---|---|
| `main_cpml8` | fc7f7202 | 8 (=2n) | as declared | off | H1 headline: does the growth survive on main |
| `main_cpml16` | fc7f7202 | 16 | as declared | off | falsifier: must reproduce -51.60 dB within 1 dB |
| `pre1047_cpml8` | c0435798 (541f703f^) | 8 | as declared | off | isolates #1047 + #1057 with all else held |
| `pre1047_cpml16` | c0435798 | 16 | as declared | off | the same control on the other tree |
| `main_cpml8_noulp` | fc7f7202 | 8 | one ULP shorter | off | H2 attempt 1: is the vacuum pad facet the growth |
| `main_cpml8_subpixel` | fc7f7202 | 8 | as declared | ON | positive control: the fixed smoothed lane is bounded |
| `pre1047_cpml8_subpixel` | c0435798 | 8 | as declared | ON | positive control: the same geometry WOULD diverge pre-#1047 |

`c0435798` is `541f703f^`, chosen over the issue's own `fa3a99bd` so the comparison isolates
#1047 + #1057 rather than six weeks of unrelated change. The issue's tree is covered instead
by re-reading its OWN recorded probe series with this lane's arithmetic
(`read_recorded_series.py`), which reproduces all eight recorded `settling_db` values
exactly -- so the "before" column is the issue's measurement, not a re-enactment of it.

**H2 gate, committed before the run.** H2 = the vacuum absorber pad on +x is what makes the
arm grow. CONFIRMED iff `main_cpml8_noulp` settles (`settling_db <= -40 dB`) while
`main_cpml8` does not; REFUTED iff both read the same side of the bar with
`|settling_db(noulp) - settling_db(main_cpml8)| <= 3 dB`. Residual
`r_H2 = max(|delta| - 3, 0)`. Prior from the recorded arms: the facet is present at +8h
(settled -50.09 dB) and at +10h with 16 layers (settled -51.60 dB), so H2 is expected to be
REFUTED; it is run because "expected" is not measured.

`fa3a99bd` is NOT re-run. #931 replaced the PEC realization rule between it and main (a
one-cell PEC Box was a sheet there and is a filled slab with both faces here -- the preflight
text differs on exactly that), so a fa3a99bd-vs-main difference could not be attributed to
the CPML work. That confounder is the reason for the c0435798 baseline.

---

## Addendum C — second job, pre-declared while the first was still running

Written after arms 1-3 of VESSL 369367261204 reported and before anything else was
submitted, so the ladder below is not a reaction to its own result.

Arms 1-3 read: `main_cpml8` -43.37 dB SETTLED, `main_cpml16` -45.83 dB SETTLED,
`pre1047_cpml8` -43.37 dB SETTLED with the same four per-probe values as `main_cpml8`.
Two things follow, and they pull in opposite directions:

* H1 is **REFUTED** on its strongest branch: pre-#1047 and main agree digit for digit, so
  neither #1047 nor #1057 touched this configuration. (Bit-identity of the raw series is
  checked separately from the .npz files.)
* The pre-declared cheap falsifier **FAILED**: the 16-layer control reads -45.83 dB where
  the issue recorded -51.60 dB, 5.77 dB away from its 1 dB band. The realized board is not
  the same board: `cavity_cells` 5 -> 4, `cavity_um` 983.75 -> 787.00, `sum(d/eps)` 291.05
  -> 232.84 um, `walls_um` one plane -> four. That is #931's lattice-ownership contract --
  the one-cell PEC ground was a sheet at fa3a99bd and is a filled slab with walls on both
  faces on main. So "the arm settles on main" does NOT establish that #801's mechanism is
  gone; it establishes that this fixture no longer excites it, on a board that changed for
  an unrelated reason. `main_cpml8` also settles by only 3.4 dB against the bar.

**H3 (attempt 1): does the +10h/2n-cell instability still exist on main's board at some
lateral clearance?** The issue's own discriminator is the pad ladder, so run it on main:
pad = +6h, +8h, +10h, +12h, +14h, +16h at `cpml_layers = 2n = 8`, and +14h/+16h again at
16 layers as the paired control. 150 periods, everything else the rig.

**Gate, committed before the run.** Growth is present on an arm iff its worst-probe
`settling_db > -40 dB` AND its last-30% log rate is positive on all four probes (the
recorded unstable arms share a rate to three significant figures -- +4.39e-4/step at
n4 pad10, +1.27e-3/step at n3 pad10 -- which is one eigenvalue of the update operator, not
a probe-position artefact; a single positive probe is not the signature).

* **STILL PRESENT** iff at least one 8-layer arm shows growth by that definition while its
  16-layer sibling does not. Residual `r = max(-40 - max_settling_db_over_8layer_arms, 0)`;
  0 = still present.
* **NOT REPRODUCIBLE ON MAIN'S BOARD** iff every 8-layer arm settles. Residual
  `r = max(worst settling_db over the 8-layer arms + 40, 0)`; 0 = all settled.

Falsifier for the ladder itself: `+6h` and `+8h` at 8 layers settled on the old board
(-51.39 / -50.09 dB) and must settle here too; an arm that grew where the old board was
comfortably stable would say the ladder is measuring the board change, not the absorber.

---

## Addendum D — H4, pre-declared after the ladder reported and before the reconstruction ran

The Addendum C ladder (VESSL 369367261205) read, on main's board, `cpml_layers = 2n = 8`:
+6h -45.26, +8h -44.22, +10h -43.37, +12h -43.42, +14h -43.37, +16h -42.95 dB; its 16-layer
controls +14h -46.00 and +16h -44.97 dB. **Every arm settled**, so H3's residual
`max(-42.95 + 40, 0) = 0` and the growth is **NOT REPRODUCIBLE ON MAIN'S BOARD** anywhere in
+6h..+16h. The ladder's own falsifier held: +6h and +8h, comfortably stable on the old board,
are comfortably stable here.

That closes H3 and leaves exactly one question worth asking: the growth stopped because the
board changed, so **is the mechanism still live in today's solver, or was it something the
pre-#931 board alone could do?** An issue cannot be dispositioned without that.

**H4: with the pre-#931 board restored on today's main, the +10h / 2n-cell arm grows again.**

The board is restored the way the preflight itself now advises -- declare the ground and the
patch as zero-thickness Boxes (SHEETs, one node plane each with the normal E edge live) rather
than as one-cell volumes, which #931 realizes as filled slabs with walls on both faces.

**Reconstruction falsifier — ALREADY RUN, PASSED.** The restored board must reproduce the
recorded fa3a99bd raster field for field, not merely resemble it. Measured (`--dry`, no solve):

| field | fa3a99bd recorded | main + `--sheet-conductors` |
|---|---|---|
| walls | 3934.99992787838 um | 3935.0 um |
| | 4918.749909847975 um | 4918.75 um |
| cavity | 983.749981969595 um, 5 cells | 983.7500000000002 um, 5 cells |
| sum d/eps | 291.050286003532 um | 291.050286003532 um |
| k_gnd / k_patch | 28 / 33 | 28 / 33 |
| patch raster | 44 x 52 cells | 44 x 52 cells |

Equal on every field, and `sum(d/eps)` to all twelve printed digits. (`n_pec_sheets = 2`,
`has_cell_mask = False` -- the conductor is read from tangential E edges, #931 §1.3.)

**Arms:** n = 4, `--sheet-conductors`, 150 periods, at +6h / +8h / +10h / +12h with
`cpml_layers = 2n = 8`; +10h again at 16 and at 32 layers; and n = 3 +10h at `2n = 6`, the arm
that grew hardest of all (+1.27e-3/step).

**Gate, committed before the run.** Growth is present on an arm iff worst-probe
`settling_db > -40 dB` AND the last-30 % log rate is positive on all four probes.

* **H4 CONFIRMED** iff at least one 8-layer (or the 6-layer n = 3) arm shows growth while its
  16-layer sibling does not. Residual `r = max(-40 - max settling_db over the thin arms, 0)`;
  0 = confirmed.
* **H4 REFUTED** iff every thin-absorber arm settles. Residual
  `r = max(worst thin-arm settling_db + 40, 0)`; 0 = all settled.

**Stronger reading available, and declared now so it is not claimed after the fact:** if
`sheet_pad10_cpml8` reproduces the issue's recorded 0.00 dB with a positive rate near
+4.39e-4/step on all four probes, then the mechanism is live on today's solver and #931 merely
moved this fixture off it. If instead it settles while the reconstruction matched the raster
field for field, then something between fa3a99bd and main other than #931 and other than
#1047/#1057 is responsible, and the next step is a bisect, not another mechanism hypothesis.

This is attempt 1 on H4. Attempts on H1, H2 and H3 are closed above.

---

## Addendum E — the bisect, pre-declared before it ran (PI-approved 2026-09-15 KST)

Four hypotheses are closed above and none of them explains what stopped the growth. This is
**localization under a criterion already fixed**, not a fifth mechanism hypothesis, and the
rule below says so in advance.

**Range.** `fa3a99bd..fc7f7202`, 734 commits, so ~10 bisect steps.

**Arm — the one the issue measured, unchanged.** `--n 4 --pad 10 --periods 150 --cpml 8`
(= `2n`), board **AS DECLARED** (no `--sheet-conductors`, no `--shrink-domain-ulp`),
26659 steps, float32, `gpu-rtx4090`.

**Criterion.** `settling_db > -40 dB` = **grows**; `<= -40 dB` = **settles**. Nothing else is
read, and no per-step judgement is exercised.

**Endpoints, from the runs already on record.**

| endpoint | verdict | settling_db |
|---|---|---|
| `fa3a99bd` | grows | 0.00 (recorded, this issue) |
| `fc7f7202` | settles | -43.37 (VESSL 369367261204) |

**The predicate is inverted on purpose.** git bisect walks toward the first commit where a
property APPEARS. The property that appears here is *settling*, so the job runs
`git bisect start fc7f7202 fa3a99bd`: **bad = settles**, **good = grows**, and git's "first bad
commit" is **the first commit where the growth stopped**. The report must use that wording;
"bad" here does not mean broken.

**Endpoint falsifier, run BEFORE the bisect.** `git bisect run` trusts the two labels and never
tests them, so a bisect between two mislabelled ends returns a confident wrong commit. Both
endpoints are re-measured in the same pod first and must reproduce their recorded verdict; a
disagreement aborts the job with exit 4 and no bisect is run. (`fa3a99bd` has never been run in
this pod — only its recorded series has been re-scored — so this is also the first direct
reproduction of the issue's own measurement on this hardware.)

**Portability.** The step script passes `--no-raster`: `raster()` spans the #931
`tangential_edge_masks` -> `realized_pec_edge_masks` rename and the sheet-collector signature
change in `_assemble_materials`, either of which can fail on an intermediate commit for reasons
that have nothing to do with the ring-down. The solve is untouched. A step whose driver exits
non-zero returns 125 and git **skips** that commit rather than scoring it; 125 is never produced
by a real measurement. `sim.preflight()` raising is recorded as the preflight
(`preflight_status: "RAISED: ..."`) rather than swallowed or allowed to abort.

**Named candidates, written down first so the result cannot be retrofitted to them.** From the
~37 commits in range touching `rfx/boundaries/`, `rfx/simulation.py`, `rfx/geometry/`:

1. the **#931 lattice-ownership stack** (`a3e4dba4` … `f112f7bb`) — the board change is only one
   of its effects; it also rewrote how every lane realizes and applies PEC edges;
2. **`a65c6626`** *"never promote a pole-carrying column's statics into a hi-face pad"* — a CPML
   pad materials change on the same face family this lane found the vacuum facet on.

Landing on neither is a legitimate outcome and is reported as such.

**Reporting rule, fixed now.** The first commit where the arm settles is reported with its PR
number and **one paragraph** reading its diff for a mechanism. If that diff plausibly explains
the change, it gets **one** A/B confirmation at that commit and its parent — a single pre-declared
check, not an investigation. No further hypothesis is opened in this lane.

**Sharding.** Not applicable: a bisect step cannot start until the previous one reports. Serial
in one job, ~12 arms (2 endpoints + ~10 steps) at 26 s of solve each.

---

## Addendum F — one round on the LIVE thin-absorber growth (PI-approved 2026-09-15 KST)

Written before any arm of this round ran. The subject is **not** the #931-fixed `n = 4 /
cpml 8` arm (settled, and gated by PR #1077). It is the growth that is **still live on current
main**, established by the independent reviewer's ladder (imported unchanged with provenance to
`thin_absorber_ladder_review2/`, not re-run): at 150 periods with a `+10h` lateral pad,
`n = 3 / cpml 6` reads **0.00 dB at +8.53e-4 per step** and `n = 2 / cpml 4` reads **0.00 dB at
+2.73e-3**, while `n = 2 / cpml 8`, `n = 2 / cpml 16`, `n = 2 / cpml 4 with pad = 0` and
`n = 4 / cpml 8` all settle. `n = 3 / cpml 6` is one of **#801's own recorded arms**.

What that ladder already settles, so no attempt is spent re-deriving it: it is **not**
under-resolution (dx held at n = 2, only the layer count moves), **not** the physical absorber
thickness on its own (1.574 mm grows at 4 and 6 cells and settles at 8), and **not** #1070 at
n = 4 (the vacuum pad is present in growing and settling arms alike).

**Reference arm for every hypothesis below: `n = 3, +10h pad, cpml_layers = 6`, 150 periods =
19994 steps, board as declared.** One arm per hypothesis, one attempt each, on `gpu-rtx4090`
(~30 s per arm), scored with PR #1077's own metric functions.

**Shared verdict rule, committed now.** An arm **GROWS** iff worst-probe `settling_db > -40 dB`
AND the fitted last-half block-max log rate is positive on **all four** probes. It **SETTLES**
iff `settling_db <= -40 dB` AND that rate is negative on all four. Anything else is
**non-closing** and is recorded as such rather than reinterpreted.

### H1 — the conductor's tangential edges at the absorber seam

The ground spans the full declared domain, so it is flush against the absorber on x-lo/y-lo.
**Test:** inset every conductor 2 cells from every lateral pad, so no conductor edge is inside
or adjacent to the absorber. Nothing else changes.
*CONFIRMED* iff the inset arm SETTLES while the reference arm GROWS — residual
`max(settling_inset + 40, 0)`, 0 = confirmed. *REFUTED* iff it still GROWS.
If confirmed, the mechanism is the conductor-at-absorber seam and the landing candidate is a
preflight refusal plus a documented clearance, not a CPML coefficient change.

### H2 — absorber layer COUNT at fixed dx and fixed geometry

**Test:** the same arm at `cpml_layers` 8 and 12 (dx, geometry and padding all held).
*CONFIRMED* iff 8 and/or 12 SETTLE while 6 GROWS, locating a threshold in cells — residual
`max(settling_12 + 40, 0)`. *REFUTED* iff 12 layers still GROWS, which would mean the layer
count is not the governing variable at this dx and the ladder's correlation is a proxy.
Two rungs, one hypothesis, one attempt.

### H3 — the CPML profile itself at few layers (no FDTD)

`_cpml_profile` sets `sigma_max = -ln(R)(m+1)/(2*eta*d)` with `d = n_layers*dx`, and
`alpha = 0.05*(1-rho)` — an absolute CFS value that does not scale with the layer count.
**Test:** rfx's own one-step amplification operator (the model that owns the #1047 analysis,
`tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py::amplification_rho`),
extended with a PEC termination so the slice is a PEC-backed pad cell adjacent to a conductor
edge, evaluated at 4 / 6 / 8 / 16 layers at this arm's dx and dt.
*CONFIRMED* iff `rho > 1 + 1e-6` appears at 4 and 6 layers and not at 8 or 16 — the ladder's
own boundary. *REFUTED* iff `rho <= 1` at every layer count (the 1-D model does not carry the
mechanism) or `rho > 1` at every count (it does not reproduce the boundary).
**Comparator first:** the PEC-terminated model must return `rho = 1` to 1e-6 for a lossless
slice with no absorber before any absorber row is read.

### H4 — the #1070 vacuum node, re-checked at n = 3

Refuted at n = 4 (0.07 dB). **Test:** the reference arm with `--shrink-domain-ulp 1`, which
removes the extra cell and the vacuum pad.
*CONFIRMED* iff the snapped arm SETTLES while the reference GROWS. *REFUTED* iff both GROW with
`|settling difference| <= 3 dB` — residual `max(|delta| - 3, 0)`.

### Stopping rule, fixed now

If one hypothesis closes and the fix lands in `rfx/boundaries/cpml.py` or `rfx/boundaries/pec.py`,
this lane **STOPS at the diagnosis** and writes a landing pre-declaration for the PI. If none
closes after one attempt each, it STOPS as well, and proposes as interim protection an
append-only preflight advisory (family module + `register_config_check`) reading *"cpml_layers
< 8 with a padded lateral domain and a conductor reaching the absorber boundary: known growth
class (#801)"* — an advisory, not a refusal, because the supported envelope is not yet known.
No second attempt on any hypothesis without a named new falsifier in writing.

---

## Addendum G — round F closed; landing pre-declaration for the PI

VESSL 369367261265, `gpu-rtx4090`, 5 arms, rc 0, n = 3 / +10h / 150 periods = 19994 steps,
scored by the rule fixed in Addendum F before any arm ran.

| arm | cpml | settling dB | last-30 % growth | verdict |
|---|---|---|---|---|
| `ref_n3_cpml6` (control) | 6 | **0.00** | 13.13 | **GROWS** |
| `h1_inset2_n3_cpml6` | 6 | **−42.76** | 0.486 | SETTLES |
| `h2_n3_cpml8` | 8 | −44.21 | 0.477 | SETTLES |
| `h2_n3_cpml12` | 12 | −45.99 | 0.468 | SETTLES |
| `h4_ulp_n3_cpml6` | 6 | **0.00** | 7.011 | GROWS |

The control reproduces the reviewer's growing arm on this lane's own run, so the four test
arms are read against a same-session control rather than an imported number.

* **H1 CONFIRMED**, residual `max(−42.76 + 40, 0) = 0`. Pulling every conductor 2 cells back
  from the lateral pads — substrate untouched — turns 0.00 dB into −42.76 dB. The ground is
  otherwise flush against the absorber on x-lo/y-lo.
* **H2 CONFIRMED**, residual `max(−45.99 + 40, 0) = 0`. At the same dx and the same geometry,
  8 and 12 layers settle where 6 grows. The threshold at this dx sits between 6 and 8 cells.
* **H3 NON-CLOSING** (Addendum F, no FDTD): the 1-D slice model fires at 6 layers but not at
  the n = 2 / 4-layer arm that grows hardest, and at 188× too small a rate.
* **H4 REFUTED**, residual 0. The ULP snap removed the extra cell (`domx/dx` 174.0 against
  174.00000000000003, grid 187 against 188) and the arm still reads 0.00 dB. #1070 is not this.

**The condition is a CONJUNCTION**, and either leg removes it: a conductor edge at or adjacent
to the absorber boundary **AND** few absorber layers. That is the same shape as the #931 finding
on the n = 4 arm — a conductor-edge-at-the-seam effect — which is why the two arms belong to one
class rather than two.

### Landing pre-declaration — PI decision, nothing implemented in this lane

The fix is **not** a CPML coefficient change and **not** `rfx/boundaries/pec.py`: at 8 layers the
same conductor placement is stable, and at 2 cells' clearance the same 6 layers are stable, so
neither the profile nor the edge rule is wrong on its own. What is missing is that **nothing
warns**. Proposed landing, in order of confidence:

1. **A preflight advisory** (append-only; family module under `rfx/preflight/` plus
   `register_config_check`), firing on the measured conjunction: `cpml_layers < 8` **and** a
   laterally padded domain **and** a conductor whose realized edges reach the absorber boundary.
   Advisory, not refusal — the supported envelope is not yet mapped, and `pad = 0` with 4 layers
   is stable, so a blanket `cpml_layers` floor would refuse working configurations. Text to be
   settled by the PI; this lane proposes *"cpml_layers < 8 with a padded lateral domain and a
   conductor reaching the absorber boundary: known growth class (#801)"*.
2. **A documented clearance** in the guides: keep conductors ≥ 2 cells off the absorber, which
   is what H1 measured. Needs its own ladder (1 cell? is 2 the floor or just sufficient?) before
   it is written as a number — **not** measured here, and it should not be stated as if it were.
3. **A gate** at `n = 3 / cpml 6` in the shape of PR #1077's, once (1) or (2) lands.

**What is NOT established, and must not be written as if it were:** the eigenvalue mechanism
itself. H1 and H2 say which two conditions must hold together; they do not say why the operator
amplifies, and H3 showed a 1-D slice does not carry it. Anyone continuing should start from a
3-D operator restricted to the seam, not from another parameter sweep — and this lane's R2 count
is one attempt per hypothesis, all four closed, so a new attempt needs a new hypothesis.

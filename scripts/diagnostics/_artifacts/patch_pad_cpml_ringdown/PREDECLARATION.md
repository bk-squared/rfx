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

# Issue #831 — cv03's far-end return: pre-declared hypotheses, arms and gates

Written 2026-09-15 (KST), **before any measurement in this lane**. Frozen here
so every verdict below is judged against a threshold that existed before the
number it judges. Append-only after the first run: corrections go in a new
section, the text above them is not edited.

Lane branch: `diag/831-cv03-far-end-return`, forked from `origin/main` at
`9866bafd`. No file under `rfx/` is modified in this lane.

---

## 0. What #831 says, and the one thing that has to be explained first

`validation/crossval/03_straight_waveguide_flux.py` — 2-D slab guide,
`eps_wg = 12`, width `1a`, `sx = 16a`, resolution 10, `boundary = upml`,
`cpml_layers = 20` — carries a backward wave `|B/A| ~ 0.52-0.54` at the carrier
bin, and #831's section 8 table reports it getting **worse** as the absorber
deepens:

| configuration | SWR | `|r|` |
|---|---:|---:|
| `upml`, 20 cells | 3.33 | 0.538 |
| `upml`, 40 cells | 3.98 | 0.598 |
| `upml`, 60 cells | 4.38 | 0.628 |
| `cpml`, 40 cells | 4.06 | 0.605 |

PR #1027 (merged `07e4b155`, issue #813) swept cv01's straight-guide self-check
— also 2-D, also `eps = 12` — over `cpml_layers` 10/16/20/40 and its number
**improved** monotonically, `mean_self` 0.749 -> 0.884 -> 0.920 -> 0.947.

### 0.1 The two sweeps do not measure the same quantity (established by reading, before any run)

This is recorded here because it changes what an "opposite sign" licenses one to
conclude, and it is a code fact, not a measurement.

- cv01's swept number is `mean_self = mean(T_self_smooth[above])` where
  `T_self = flux_out/flux_in` between two interior planes
  (`scripts/diagnostics/cv01_cpml_flux_selfcheck.py:339-348`). For a lossless
  section the net flux is `|A|^2 - |B|^2` at **every** plane, so this ratio is
  identically blind to a standing wave of any depth. cv03's own design note
  says exactly this
  (`docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md` section 8,
  item 1). cv01's N-trend therefore measures **net loss between its two
  planes**, not absorber reflection.
- cv03's swept number is `|B/A|` from the two-wave fit on the guide centre line
  (`validation/crossval/03_straight_waveguide_flux.py:353-369`), i.e. the
  reflection itself.

So "same physics class, opposite sign => a rig difference" is not available as
an inference: a net-flux ratio rising with N and a reflection amplitude rising
with N are not contradictory and can coexist in one rig. **The trend still
needs a cause** — dissolving the inference is not explaining the number.
Arm A measures BOTH quantities in cv03's own rig at every layer count, which
is what actually crosses observable against rig.

### 0.2 Two candidate rig differences that are NOT the answer (also by reading)

- **Pad extension is not a cv01/cv03 difference.** Both build the guide `Box`
  spanning the full declared domain and both take
  `_assemble_materials(..., include_cpml_pad_extension=True)` by default
  (`rfx/api/_compile.py:156, 336`); neither passes `False`. cv01
  `scripts/diagnostics/cv01_cpml_flux_selfcheck.py:317-319`, cv03
  `validation/crossval/03_straight_waveguide_flux.py:283`. H1 is still worth
  running as a *mechanism* arm; it is not the rig delta.
- **Boundary family is not the answer either.** #831's own table has
  `cpml`, 40 cells at 0.605 against `upml`, 40 cells at 0.598 — the same
  number. Whatever this is, it is not UPML-specific, so
  `docs/agent-memory/rfx-known-issues.md:1989` ("rfx UPML reflects ~25 dB MORE
  than CPML"; mechanism: the parallel-E component attenuated only indirectly
  through curl coupling) cannot carry it alone. Arm A runs both families at
  all three depths so this is measured rather than argued.

---

## 1. R1 — memory this plan is checked against

Read read-only at the primary checkout
`/root/workspace/byungkwan-workspace/research/rfx/docs/agent-memory/`.

1. `rfx-known-issues.md:4158` — *"in an empty WR-90 waveguide driven from port
   1, `|b1/a1|` ... scales with CPML thickness as 10 layers -> 11.7 %, 20 -> 4.2 %,
   40 -> 1.8 %"*, and the default was raised 8 -> 16 on that evidence.
   **Contradicts #831's trend.** A guided mode's reflection off a CPML fell
   with depth in the one place this repo measured it directly. #831 reports the
   opposite on a different guide. Either the guide (`eps = 12` dielectric slab
   vs hollow metal WR-90) changes the sign, or one of the two measurements is
   an artefact. That is what this lane decides.
2. `rfx-known-issues.md:4247` — *"CPML guided-mode reflection ~-4 dB -> resolved:
   D/B-equivalent UPML + subpixel smoothing. Material-independent PML loss
   (sigma/eps_0) ... Self-T = 0.995."* **Consistent with the code as it stands**:
   `rfx/boundaries/upml.py:232` (`loss_pml = sigma_perp * dt / (2 * eps_0)`)
   and `:247` (`loss = sigma_perp * dt / (2 * eps_0)` for H, eps_0 and not mu_0) are
   the D/B-equivalent form, which is impedance-matched in **any** material.
   The absorber therefore is not, on its face, mis-matched against the
   `eps_r = 12` pad.
3. `rfx-known-issues.md:1989` — *"rfx UPML reflects ~25 dB MORE than CPML —
   measured clean floor -43.0 dB vs -68.3 dB"*. Both floors are ~40 dB below
   #831's measured 0.5 (-6 dB), and the CPML row of #831's table matches its
   UPML rows, so this entry bounds neither the level nor the trend.
   **Scope-limited, not contradicted.**
4. `research/CLAUDE.md`, comparator-first rule, and the case ledger at
   `rfx-known-issues.md:4343`: *every* case recorded in this workspace to date
   has closed as a comparator/extractor bug, not a solver/physics bug. The
   design's own N-scaling makes every absorber-side mechanism predict
   improvement with depth (section 2 below), so an extractor/record cause is the
   prior here, and Arm A is built to falsify it first.

**Consistency statement.** This plan is consistent with entries 2-4 and is
designed to resolve the contradiction with entry 1. No entry is being
contradicted on plausibility; entry 1 is being re-measured because #831's own
number disagrees with it.

---

## 2. Why every absorber-side mechanism predicts the *opposite* trend

Recorded before the measurement so that a confirmed trend is recognisably
surprising rather than retro-fitted. All from `rfx/boundaries/cpml.py:130-166`
and `rfx/boundaries/upml.py:69-92`.

- `d = n_layers * dx` and `sigma_max = -ln(R)*(m+1)/(2*eta*d)`, so `int sigma dx`
  is N-invariant by construction: the designed round-trip optical depth does not
  move with N.
- The continuum attenuation of a mode with index `n_eff` through a stretched-
  coordinate absorber is `exp(-n_eff*eta_0*int sigma dx)` — it *grows* with
  `n_eff`, so the `eps_r = 12` pad is attenuated **more** than vacuum, not less.
  At `R = 1e-15`, `int sigma dx = 0.04584` and `n_eff = 2.84`: one-way amplitude
  `exp(-49)`. The design margin against a 0.5 return is ~43 orders.
- The per-cell grading step scales as `sigma_max/N ~ 1/N^2`, and the first pad
  cell at the hi face carries `sigma_max*(0.5/N)^2 ~ 1/N^3`. Discretisation
  reflection — the term that actually sets a numerical PML's floor — therefore
  *falls* with N. This is the cv01/#1027 direction and the `:4158` direction.

So there is no absorber mechanism in this code that produces #831's sign. Either
the absorber is not delivering its assembled profile (measurable, Arm C), or the
wave never reaches the absorber (measurable, Arm A's own trace), or the
measurement is N-dependent (Arm A).

---

## 3. Arms, and what each one runs

Every FDTD arm applies **exact textual edits to a COPY** of the committed case,
imitating `scripts/diagnostics/cv03_flux/tof_reflection_free.py`. The committed
`validation/crossval/03_straight_waveguide_flux.py` is not modified, and no
gate, tolerance or record of it moves in this lane.

Driver: `scripts/diagnostics/cv03_seam_facet/seam_facet.py`, with the
repo-root provenance assert that
`scripts/diagnostics/cv01_cpml_flux_selfcheck.py:239-262` uses (`import rfx`
must resolve under this worktree or the run aborts).

Every arm records, into the artifact: the full preflight stdout verbatim, every
`warnings.warn` raised during build and run, `result.settling_db` +
`result.settling_witness` (a point probe is added at the fit-window centre so
the witness has a record to score — the committed case has no point probe and
would return `settling_db=None`), the fit-window definition in both Meep and
rfx coordinates and in grid indices, and the full per-bin `|B/A|`, `n_eff`,
residual and `T(f)` traces (R5: no headline without its trace).

### Arm C — the assembled profile (NO FDTD)

Build the cv03 `Simulation` at `(boundary, layers)` in {upml, cpml} x {20, 40,
60} via `sim._build_grid()` and dump, on the guide centre row:

- `grid.shape`, `pad_x_lo/hi`, `pad_y_lo/hi`, `grid.interior`;
- rasterized `eps_r(x)` across the FULL x extent, pad included;
- CPML: `sigma(x)`, `kappa(x)`, `alpha(x)`, `b(x)`, `c(x)` as assembled;
  UPML: assembled `sigma_E(x)`, `sigma_H(x)` from `_axis_sigma_E_H(grid, 'x')`;
- `int sigma dx` (discrete sum x dx) per face, and the designed one-way
  amplitude attenuation `exp(-n_eff*eta_0*int sigma dx)` at `n_eff = 2.8389`
  (`docs/design_notes/issue812_cv03_dispersion_matched_frequency.json::oracle.n_eff_at_carrier_bin`).

This is the "instrument the complete assembled profile" the 2026-09-13 closing
comment asked for.

### Arm C0 — the time-of-flight control (H4)

Two FDTD runs, reproducing #831's own localisation on THIS tree with the
COMMITTED estimator: `sx = 40a`, fit window Meep `[-16, -8]`, DFT window
150 a/c0 (reflection-free) and 400 a/c0 (return admitted). The round trip is
`docs/design_notes/issue812_cv03_dispersion_matched_frequency.json::oracle.tof_round_trip_a_over_c0.src_to_far_end_and_back_64a = 230.93`
a/c0, and `docs/design_notes/issue812_cv03_dispersion_matched_frequency.json::oracle.tof_round_trip_a_over_c0.src_to_far_end_to_fit_window_75a = 270.62` a/c0, at
`docs/design_notes/issue812_cv03_dispersion_matched_frequency.json::oracle.n_group_at_carrier_bin = 3.6083`.

### Arm A — the depth sweep, re-measured with the committed estimator

Six FDTD runs: `boundary` in {upml, cpml} x `cpml_layers` in {20, 40, 60},
cv03's recipe otherwise untouched. Records `|B/A|` at the carrier bin AND the
band-mean `T` from the same run, plus `|Ez(x)|` at the carrier bin across the
full grid (the DFT plane probe already spans the pads).

### Arm B — pad extension OFF (H1)

Three FDTD runs, `upml` x {20, 40, 60}, with
`include_cpml_pad_extension=False` forced by wrapping
`Simulation._assemble_materials` **in the copy** (`run()` does not expose the
flag). The guide then ends at the interior/pad seam in vacuum — a bare facet.

---

## 4. Pre-declared outcomes and nonnegative residual gates

Notation: `bA(fam, N)` = `|B/A|` at the carrier bin, arm `(fam, N)`.
Every residual below is nonnegative and is gated at 0; a sign flip or an
increase is non-closing, per R2.

### G-C0 — the control. If this fails nothing else in the lane is readable.

- true: `bA_150 <= 0.01` AND `bA_400 >= 0.40`.
- residual `r_C0 = max(0, bA_150 - 0.01) + max(0, 0.40 - bA_400)`; gate `r_C0 = 0`.
- **If `r_C0 > 0`**: #831's premise does not reproduce on this tree. STOP the
  lane, report that, and do not read Arms A/B/C as evidence about an absorber.

### G-A — does the depth trend reproduce under the committed estimator?

`Delta(fam) = bA(fam, 60) - bA(fam, 20)`. #831's section 8 measured
`Delta(upml) = +0.090`.

- **A-REPRODUCED** (the trend is real on this tree): `Delta(upml) >= +0.05`.
  residual `r_A = max(0, 0.05 - Delta(upml))`; gate 0.
- **A-ABSENT** (the section 8 trend does not survive the committed estimator):
  `|Delta(upml)| <= 0.02`. residual `r_A' = max(0, |Delta(upml)| - 0.02)`; gate 0.
- `0.02 < |Delta(upml)| < 0.05`, or the two families disagreeing in sign:
  **INCONCLUSIVE**, declared now, and no verdict is issued either way.

Rationale for the two bars: #831's section 8 rows were produced by a driver that
was never committed (section 8.2 of the #812 note: *"Round 1's driver for the
section 8.1 rows was never committed, so its four measured operands carry note
provenance"*). A-ABSENT is therefore a live outcome and is declared as one
before the run, not discovered after it.

Secondary, same runs, no gate: `T_band(fam, N)`. Reported so the observable
cross of section 0.1 is on the record whichever way `bA` goes.

### G-B — H1, the pad extension

`Delta_off = bA(upml-noext, 60) - bA(upml-noext, 20)`.

- **H1 TRUE** (the pad's eps sets the sign): `sign(Delta_off) != sign(Delta(upml))`
  AND `|Delta_off| >= 0.05`. residual
  `r_B = max(0, 0.05 - |Delta_off|) + (0 if the signs differ else 1.0)`; gate 0.
- **H1 FALSE**: `Delta_off` has the same sign as `Delta(upml)` — the trend
  survives the switch, so the pad's permittivity is not what sets it.
- Level check, reported not gated: `bA(upml-noext, 20)` vs `bA(upml, 20)`. If
  they are equal to within 0.02 the extension is not reaching this observable
  at all, which is itself a finding about the default.

### G-C — H2, is the absorber delivering its designed profile?

- **H2 TRUE** (coefficient-assembly bug; #831's own falsifier 2):
  `1 - int sigma(60)/int sigma(20) >= 0.05` — the assembled optical depth falls
  with N. residual `r_C = max(0, 0.05 - (1 - int sigma(60)/int sigma(20)))`;
  gate 0.
- **H2 FALSE**: `|1 - int sigma(60)/int sigma(20)| <= 0.02` — N-invariant as
  designed.

### G-D — H3, the outer wall seen through an under-attenuating pad

From Arm A's `|Ez(x)|` trace at the carrier bin:
`A_meas(fam, N) = |Ez|` at the outermost pad cell / `|Ez|` at the last
interior cell, hi face.

For an outer-wall return to supply `|B/A| = 0.5`, the round trip through the
pad must survive at >= 0.5, i.e. `A_meas >= sqrt(0.5) = 0.707`.

- **H3 TRUE**: `A_meas >= 0.707` in at least one arm. residual
  `r_D = max(0, 0.707 - max_arm A_meas)`; gate 0.
- **H3 FALSE**: `A_meas < 0.10` in every arm — the outer wall cannot supply the
  measured return, and the reflector is at or near the seam.
- `0.10 <= A_meas < 0.707`: **INCONCLUSIVE**, declared now.

### G-E — the complement, declared now so it is not post hoc

If G-D returns H3 FALSE and G-C0 passes, the reflector is within roughly one
pad depth of the interior/pad seam. The witness is the shape of `|Ez(x)|` near
the seam and the value at the last interior cell (#831's own falsifier 4:
*"Amplitude at the last interior cell is already ~0.5 of the incident -> the
wave never enters the pad and the seam, not the pad, is the reflector"*).
Reported with the trace; no gate, because a shape is not a scalar and pinning
one here would be inventing a threshold for a quantity whose scale is unknown
before the run.

---

## 5. R2 accounting

This lane is under the RF/EM intensifier: **N >= 1 non-closing attempt is a
STOP**. Each hypothesis above gets exactly ONE arm. A second run of any arm
requires either a named new falsifier or an identified implementation defect in
the first, in writing, appended below this line before it runs.

Attempts consumed at the time of writing: **0**.

## 6. What this lane will NOT do

- It will not modify anything under `rfx/`. If the identified cause is in the
  absorber, the lane STOPS at the diagnosis and writes the landing
  pre-declaration (which tests and fixtures move, including cv01's #813
  numbers) for the PI to decide.
- It will not change cv03's committed record, gates or tolerances.
- If the cause is a rig/comparator defect, the cv03 rig change is proposed as a
  **separate** commit under the same append-only evidence rules, with no gate
  change.

---

## 7. APPEND (2026-09-15 KST) — Arm B is void, and the arm that replaces it

Append-only. Sections 0-6 stand as written. This section is written BEFORE the
arm it declares runs, and after Arms C, C0, A and B had run.

### 7.1 Why Arm B is void: the switch it flips never reaches the solver

Arm B returned `|B/A|` = 0.5311 / 0.5918 / 0.6221 at 20 / 40 / 60 — **identical
to 16 significant figures** to the Arm A `upml` rows, with `settling_db`
identical too. `pad_extension_forced_off: true` is in every Arm B record and the
wrapper was verified to be reached (one `_assemble_materials` call per run, with
`include_cpml_pad_extension` forced to `False`), and it does change the array
it is supposed to change: the centre row goes from `eps_r = 12` in 201/201 cells
to 160/201.

The defect is that **`run(subpixel_smoothing=True)` does not use that array for
the E update.** `rfx/runners/uniform.py:266-271`:

```
    elif subpixel_smoothing:
        from rfx.geometry.smoothing import compute_smoothed_eps
        shape_eps_pairs = [
            (entry.shape, sim._resolve_material(entry.material_name).eps_r)
            for entry in sim._geometry
        ]
        if shape_eps_pairs:
            aniso_eps = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)
```

`aniso_eps` is rebuilt from `sim._geometry` with `background_eps = 1.0`, so the
CPML pad columns are **vacuum** whatever `_assemble_materials` put there.
Measured on this tree, centre row, `eps_r = 12` cell count:

| `cpml_layers` | `nx` | `_assemble_materials` | `compute_smoothed_eps` |
|---:|---:|---:|---:|
| 20 | 201 | 201 | 159 |
| 40 | 241 | 241 | 159 |
| 60 | 281 | 281 | 159 |

This is an identified implementation defect in Arm B's instrument, which is one
of the two things R2's RF/EM intensifier accepts as licence for a second attempt
on the same hypothesis. It is recorded here before the replacement runs.

It also retires #831's own elimination 3 and section 8's rasterization check:
both read `_assemble_materials`, which is not the array the run solved.

### 7.2 Arm B-prime — continue the guide through the pad in the array the solver uses

Two witnesses, declared together, deliberately NOT sharing the suspect quantity
(one changes the declared geometry extent, one changes which array the update
reads):

- **B1 `guidecont_upml_{20,40,60}`** — `subpixel_smoothing=True` unchanged; the
  guide `Box` is widened to `x in [-8a, domain_x + 8a]`, i.e. past the deepest
  pad (60 cells = 6a). Verified on this tree to put `eps_r = 12` in 201/201,
  241/241 and 281/281 centre-row cells through the subpixel path.
- **B2 `nosub_upml_{20,40,60}`** — geometry unchanged; `subpixel_smoothing=False`,
  which routes the update through `base_materials` and therefore through the pad
  extension. Confounded (subpixel smoothing also sets the interface convergence
  order) and is read as a cross-check on B1's SIGN, never as a level.

### 7.3 Gates, frozen before the run

`bA_cont(N)` = `|B/A|` at the carrier bin, arm B1 at `cpml_layers = N`.
Arm A's `upml` reference: 0.5311 / 0.5918 / 0.6221.

- **FACET TRUE** (the guide's vacuum end facet is the reflector):
  `bA_cont(N) <= 0.25` for every N in {20, 40, 60}.
  residual `r_B1 = max(0, max_N bA_cont(N) - 0.25)`; gate 0.
- **FACET FALSE**: `bA_cont(20) >= 0.45` — the return survives with the guide
  genuinely continued, so the facet is not what reflects.
- `0.25 < max_N bA_cont(N) < 0.45`: **INCONCLUSIVE**, declared now.

Trend, same arm: `Delta_cont = bA_cont(60) - bA_cont(20)`.
- **TREND EXPLAINED BY THE FACET**: `Delta_cont <= +0.02` — removing the facet
  removes the depth trend. residual `r_B1t = max(0, Delta_cont - 0.02)`; gate 0.
- **TREND SURVIVES**: `Delta_cont >= +0.05` — the depth dependence is not the
  facet's and still has no owner.

B2 is judged only on sign agreement with B1: `bA_nosub(20) < 0.45` if B1 says
FACET TRUE. A disagreement between B1 and B2 makes the pair inconclusive and is
reported as such rather than resolved by preferring one.

### 7.4 R2 accounting after this append

Attempts on H1 (the pad-material hypothesis): 1 void (instrument defect,
identified and named above) + 1 licensed replacement = **1 counting attempt**.
Attempts on every other hypothesis: 1 each. No further arm runs in this lane
without another append.

---

## 8. Landing pre-declaration — for the PI, not executed in this lane

Written 2026-09-15 (KST) after the arms closed. Results:
`docs/design_notes/issue831_far_end_return_results.md`. **Nothing below is
implemented here.** The lane stops at the diagnosis because the fix is
solver-side and moves physics for existing committed numbers.

### 8.1 The defect, stated in one sentence

`run(subpixel_smoothing=True)` rebuilds the update's permittivity from
`sim._geometry` with `background_eps = 1.0`
(`rfx/runners/uniform.py:266-271`) and so discards the CPML pad material
extension, leaving every geometry that touches the domain boundary terminated
by a vacuum facet at the interior/pad seam — whatever
`include_cpml_pad_extension` says. The extension exists, in its own words, "so
that guided modes in dielectric waveguides see an impedance-matched absorber"
(`rfx/api/_compile.py:310-313`). This is the right-guard-MIS-GATED shape the
2026-07-08 footgun audit named.

### 8.2 Why the gates did not catch it

`tests/unit/boundaries/test_cpml_pad_material_extension.py` has 11 tests. Nine
assert on the array `_assemble_materials` returns; the two that run the solver
(`:406`, `:563`) both pass `subpixel_smoothing=False`. **No test exercises the
feature on the path every crossval with a dielectric actually uses.** Any fix
should close that first, because a fix landed against the same gates would be
unfalsifiable by them.

### 8.3 What a fix would move, and what has to be re-measured

- `rfx/runners/uniform.py` — `compute_smoothed_eps` would need the pad
  extension applied to its output (or the smoothing to run on the extended
  array). The NU mirror shares the extension through
  `extend_cpml_pad_materials`; check whether `rfx/runners/nonuniform.py` has
  the same gap before choosing where the fix lives.
- **cv03** — `|B/A|` moves 0.53 -> ~0.03 and band-mean `T` moves 0.9657 ->
  ~0.99. G2's gate (`T in [0.95, 1.05]`) still passes; its *value* moves and
  the case's committed record must be re-measured, not edited. G1 and the
  two-wave residual should improve; the two-wave estimator itself was adopted
  BECAUSE `|B/A| ~ 0.53` falsified the single-mode fit (section 7 of the #812
  note), so whether it stays the right estimator is a live question once the
  standing wave is gone. Do not re-open G1's 2.0 % threshold.
- **cv01 / #813** — `validation/crossval/01_waveguide_bend.py:382` builds the
  same boundary-touching `Box` under `subpixel_smoothing=True`, so its straight
  guide carries the same facet. Its committed numbers
  (`SWEEP_BASELINE_CPML_FULL_MEAN_SELF = 0.7488520140093946`,
  `COMMITTED_UPML_MEAN_SELF = 0.9891610388008335`,
  `SWEEP_GATE_MEAN_SELF_AT_40 = 0.90906` in
  `scripts/diagnostics/cv01_cpml_flux_selfcheck.py`) and the artifacts under
  `scripts/diagnostics/_artifacts/cv01_cpml_813/` are all measured on the
  facet-terminated rig and would move. #1027's monotone `mean_self` trend is
  NOT explained by this lane — cv01's observable is blind to the reflection —
  so that trend has to be re-measured after a fix, not reasoned about.
- **cv02** also runs `subpixel_smoothing=True`; whether its ring touches the
  boundary was not checked here.
- `tests/contracts/test_evidence_numeric_provenance.py` and
  `tests/fixtures/waveguide_chain_battery/fixture.json` reference the cv01
  numbers above and would need the same re-measurement pass.

### 8.4 The cheap falsifier for any fix

Re-run this lane's `guidecont` arm against the fixed tree with the **committed**
cv03 geometry. The fix is right when the unmodified case reaches `|B/A| <= 0.03`
at 20 layers, the depth trend points down (60 layers below 20), and
`settling_db` clears -100 dB — the three numbers arm B1 already produced by
widening the `Box` by hand.

### 8.5 Also for the PI, separately: two preflight gaps this exposed

Neither is filed as an issue here; both are decisions, not findings.

1. **Nothing warns that a dielectric ends at the absorber seam.** The preflight
   has a geometry-in-absorber check; the failure mode here is geometry
   ABSENT from the absorber under a flag that claims to put it there, and it is
   silent. The PI's stated order after v1.8 puts preflight work behind the mesh
   rasterization viz; this belongs in that slot.
2. **The committed cv03 has no point probe, so `settling_db` is `None`** and
   its own ring-down witness cannot fire. With one added, every committed arm
   reports the witness FAILING at -37 to -30 dB. A crossval whose DFT-derived
   headline has no settling witness is judging an unsettled record.

---

## 9. APPEND (2026-09-15 KST) — Arm E: does cv01 carry the same facet?

Append-only, written before the arm runs. Sections 0-8 stand.

Section 5 of the results note said cv01 "carries the same facet" from reading
`validation/crossval/01_waveguide_bend.py:382`. That is a reading. This arm
measures it, on cv01's own committed rig, by importing
`scripts/diagnostics/cv01_cpml_flux_selfcheck.py` and calling its `run_arm`
rather than copying its constants.

### 9.1 What runs

cv01 Run 1 only (`run_arm("cpml", None)`), at cv01's committed absorber depth
(`cpml_layers = 20` for the `cpml` boundary), twice:

- **control** — exactly as committed;
- **widened** — the guide `Box` widened from `(0 … sx)` to `(-2a … sx + 2a)`,
  past the 20-cell (2a) pad, with `subpixel_smoothing=True` left alone.

Both arms are instrumented, through one explicit `Simulation` subclass that
adds probes at `run()` and delegates, so cv01's geometry, source, monitors and
flux arithmetic stay the committed ones:

- a DFT plane probe on the guide centre line, read with the SAME estimator cv03
  uses (`validation/crossval/comparators/slab_te_dispersion.py::measure_neff_two_wave`),
  fit window rfx `x in [5a, 13a]` — 8a, between cv01's monitors at 4a and
  14.5a, inset by 1a at each end, the same construction cv03's window uses;
- a point probe at the fit-window centre so `settling_db` has a record.

A no-FDTD instrument check runs first: dump the centre-row permittivity that
`compute_smoothed_eps` hands the solver, for both builds.

### 9.2 Gates, frozen before the run

**Instrument check (dispositive on its own, no threshold needed).** cv01
carries the facet if, on the committed build, the solved centre-row pad columns
are vacuum while `_assemble_materials` carries `eps_r = 12`.

**Reflection — the quantity that can see a facet.**
- **CARRIES THE SAME FACET**: `bA_control >= 0.30` AND `bA_widened <= 0.10`.
  residual `r_E = max(0, 0.30 - bA_control) + max(0, bA_widened - 0.10)`; gate 0.
- **DOES NOT**: `bA_control <= 0.10`.
- in between: INCONCLUSIVE, declared now.

**`mean_self` — the comparison the review asked for, with both outcomes
declared because the evidence already in hand says it may not move.**
- Readability control: the committed arm must reproduce
  `SWEEP_BASELINE_CPML_FULL_MEAN_SELF = 0.7488520140093946`
  (`scripts/diagnostics/cv01_cpml_flux_selfcheck.py:424`) to within 0.005, or
  neither row is readable.
- **MOVES**: `mean_self_widened >= 0.85`, i.e. well above the committed
  0.748852 and toward the UPML control 0.989161
  (`scripts/diagnostics/cv01_cpml_flux_selfcheck.py:410`).
  residual `r_E2 = max(0, 0.85 - mean_self_widened)`; gate 0.
- **DOES NOT MOVE**: `|mean_self_widened - 0.748852| <= 0.02`.
- in between: INCONCLUSIVE, declared now.

**Declared expectation, written down so a null is not misread.** `mean_self` is
a net flux ratio, and net flux in a lossless section is `|A|^2 - |B|^2` at every
plane, so it is blind to a standing wave by construction. Arm B1 already
measured what that means: removing cv03's facet moved `|B/A|` 0.5311 -> 0.0296
while band-mean `T` moved only 0.9657 -> 0.9682. **I therefore expect
MEAN_SELF DOES NOT MOVE, and the `|B/A|` pair to carry the verdict.** A null on
`mean_self` is NOT evidence that cv01 is clean; only `bA_control <= 0.10`
would be.

### 9.3 R2

One attempt on one new hypothesis. No re-run without another append.

### 9.4 APPEND — Arm E's first run is void, for a named instrument defect

Written before the re-run. The first Arm E run returned `mean_self = nan` on
both rows and `|B/A| = nan` on the widened row, with this from cv01's own
driver:

> flux_spectrum returned exactly 0.0 at all 200 frequencies, but the DFT
> accumulators are healthy and a float64 recompute of the same sum gives
> nonzero flux (peak |flux| = 3.937e-56). The per-cell E x H* products
> underflowed the float32 minimum normal (~1.18e-38) and were flushed to zero
> (issue #304).

cv01's rig integrates fluxes around 1e-56 and depends on x64;
`scripts/diagnostics/cv01_cpml_flux_selfcheck.py:150` sets
`JAX_ENABLE_X64=1` on its own first line for exactly this reason. This lane's
driver runs the cv01 stage IN-PROCESS and had imported rfx before setting it,
so JAX was already initialised at float32 — the committed rig, solved by a
different solver. The widened row went further and returned NaN outright: with
the facet removed the field leaves faster, so the DFT plane underflowed too.

That is an identified implementation defect in the instrument, the same licence
R2 accepts as for Arm B. The fix is one line at the top of the driver, before
any import that can pull JAX in. **Every stage is re-run after it**, not just
Arm E: an in-process stage that was solving at the wrong precision makes the
whole artifact set unreadable as a single record, and the re-run also puts
`provenance.commit` on a commit that contains the plan.

No gate in section 9.2 moves. Attempts on Arm E's hypothesis: 1 void with a
named defect + 1 licensed replacement = **1 counting attempt**.

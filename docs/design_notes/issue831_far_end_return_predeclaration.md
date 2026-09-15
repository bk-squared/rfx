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

### 8.1 The defect, stated as the thing to file

**The CPML pad material extension — on by default, and there "so that guided
modes in dielectric waveguides see an impedance-matched absorber"
(`rfx/api/_compile.py:310-313`) — is silently bypassed for the update
permittivity whenever `subpixel_smoothing` is truthy.** Every geometry that
touches the domain boundary is then terminated by a vacuum facet at the
interior/pad seam. This is the right-guard-MIS-GATED shape the 2026-07-08
footgun audit named.

Two things this is NOT, both worth stating so the fix lands in the right place:

- It is **not** `background_eps = 1.0`. That argument is correct: it is the
  background outside the declared geometry, and setting it to the guide's
  permittivity would flood the whole vacuum cladding. The defect is the
  **absence of a pad replication step on the rebuilt array**, after it is
  computed.
- It is **not** "the `include_cpml_pad_extension` flag does not reach the
  solver". That flag is a private `_assemble_materials` keyword with one
  in-tree caller (`rfx/vmap_sweep.py:355`), and it does exactly what it says on
  the array it is given. Filing the flag would produce a fix that changes
  nothing.

**Three sites share the gap**, and a fix that closes one leaves the others:

| site | what it builds |
|---|---|
| `rfx/runners/uniform.py:237-248` | Stage-2 `kottke_pec` inverse-permittivity tensor |
| `rfx/runners/uniform.py:266-273` | Stage-1 smoothed `aniso_eps` |
| `rfx/runners/nonuniform.py:852-859` | the NU mirror |

**Two consumption sites**, and for cv03 it is the second one:

- `update_e_aniso` (`rfx/simulation.py:419-422`), reached only when
  `debye is None and lorentz is None`;
- `init_upml(grid, materials, axes=..., aniso_eps=aniso_eps)`
  (`rfx/simulation.py:914-915`), which builds `eps_abs_ex/ey/ez` from the array
  (`rfx/boundaries/upml.py:213-217`) and feeds `apply_upml_e`. On the UPML
  branch `update_e_aniso` is never reached at all, and cv03 runs UPML.

### 8.2 Why the gates did not catch it

`tests/unit/boundaries/test_cpml_pad_material_extension.py` has 11 tests. Nine
assert on the array `_assemble_materials` returns; the two that run the solver
(`:406`, `:563`) both pass `subpixel_smoothing=False`. **No test exercises the
feature on the path every crossval with a dielectric actually uses.**

**And flipping those two to `subpixel_smoothing=True` is not the remedy** —
measured, not assumed: a copy of the file with both flipped gives *10 passed, 1
deselected*. It cannot fail, because both fixtures carry Lorentz poles and
`rfx/simulation.py:415` gates the anisotropic branch behind
`debye is None and lorentz is None`; the smoothed array is never consulted, so
whether the pad is in it does not matter.

The gate a fix needs is a **NEW test**, and it has to be built from a
**static** dielectric:

1. a static dielectric `Box` that touches the domain boundary, `cpml_layers`
   set, `subpixel_smoothing=True`;
2. assert the **solved** pad column carries the interior material — the
   `compute_smoothed_eps` / `compute_inv_eps_tensor_diag` output, not
   `_assemble_materials`'s array, which is what made this invisible for the
   life of the feature;
3. and/or a round-trip assertion, `|B/A| <= 0.03` on a guided mode, which is
   what arm B1 measured.

One case per site of the three in 8.1, or a parametrisation over them.

### 8.3 What a fix would move, and what has to be re-measured

- `rfx/` — a pad replication step on the rebuilt array, at all three sites in
  8.1. `extend_cpml_pad_materials` already exists and is where the interior
  path does it, so the likely shape is to call it on the smoothing output (or
  to smooth the already-extended array) rather than to write a second copy of
  the replication logic. #627 exists because that logic was hand-duplicated
  once before.
- **cv03** — `|B/A|` moves 0.53 -> ~0.03. Band-mean `T` is **not** expected to
  follow: arm B1 moved it only 0.9657 -> 0.9682 and it still degraded with
  depth, and only B2 (which also turns subpixel smoothing off) reached 0.994.
  See 8.4. G2's gate (`T in [0.95, 1.05]`) still passes; the case's committed
  record must be re-measured, not edited. G1 and the
  two-wave residual should improve; the two-wave estimator itself was adopted
  BECAUSE `|B/A| ~ 0.53` falsified the single-mode fit (section 7 of the #812
  note), so whether it stays the right estimator is a live question once the
  standing wave is gone. Do not re-open G1's 2.0 % threshold.
- **cv01 / #813** — measured in Arm E, not read: its committed rig solves with
  vacuum in 42 of 201 centre-row cells and carries `|B/A| = 0.5218`. A fix
  moves **three** surfaces, and missing any one of them leaves a number with
  nothing behind it:
  1. the driver constants —
     `scripts/diagnostics/cv01_cpml_flux_selfcheck.py:410`
     (`COMMITTED_UPML_MEAN_SELF = 0.9891610388008335`), `:418`
     (`SWEEP_GATE_MEAN_SELF_AT_40 = 0.90906`), `:424`
     (`SWEEP_BASELINE_CPML_FULL_MEAN_SELF = 0.7488520140093946`);
  2. the committed case record —
     `validation/crossval/_01_waveguide_bend_results/crossval.json:2283`,
     `mean_self_smoothed_over_band = 0.9891610388008335`;
  3. the **26** values `tests/contracts/test_evidence_numeric_provenance.py`
     resolves out of
     `scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json` (the
     `("Numeric provenance", 26)` floor), plus the **19** of its residual-split
     section — both floors are reproduced counts, so a re-measurement that
     drops citations reds the contract.

  #1027's monotone `mean_self` trend is NOT explained by this lane — that
  observable is blind to the reflection — so it has to be re-measured after a
  fix, not reasoned about.
- **cv02 is CLEAR, by construction.** Its ring is centred at 6a with outer
  radius `r + w = 2a` in a 12a interior
  (`validation/crossval/02_ring_resonator.py:165-188`): 4a of vacuum to every
  interior edge, no shape touching a pad, nothing for a missing replication to
  drop. It needs no re-measurement.
- **cv05 is off this path by default** —
  `validation/crossval/05_patch_antenna.py:150-151` sets
  `SUBPIXEL_SMOOTHING = _sps_env if _sps_env else False` — but
  `RFX_SUBPIXEL_SMOOTHING=kottke_pec` puts it on the Stage-2 site, which
  carries the same gap.

### 8.4 The cheap falsifier for any fix

Re-run this lane's `guidecont` arm against the fixed tree with the **committed**
cv03 geometry. The fix is right when the unmodified case reaches `|B/A| <= 0.03`
at 20 layers, the depth trend points down (60 layers below 20), and
`settling_db` clears -100 dB — the three numbers arm B1 already produced by
widening the `Box` by hand.

**Report band-mean `T` alongside, and do not expect it to move.** B1 moved it
only 0.9657 -> 0.9682 and it still degraded with depth (0.9682 / 0.9596 /
0.9558); only B2, which also turns subpixel smoothing off, reached 0.994. A fix
that reports `T` recovering to 0.99 has changed something else as well, and a
fix that leaves `T` at 0.96 has not failed. Either way the number belongs in
the report, because it is where the second, unisolated effect shows up.

### 8.5 The fix is not free: the configuration it emulates diverged once

Arm E (section 9, results note section 6.4) widened cv01's guide `Box` past the
pad — which is what the fix emulates — and that run **diverged**: *"result
contains non-finite values in time_series (24598 value(s)) — the FDTD likely
diverged"*. On cv03's UPML rig over 5913 steps the same change was stable in
both arms; on cv01's CPML rig over 25000 steps it was not. The lane did not
isolate which of the three differences carries it (boundary family, record
length, or a `Box` declared past the domain edge) and does not guess.

There is a neighbouring case already in the code:
`rfx/api/_compile.py:319-325` records that extending a high-Q Lorentz pole into
the pad "turns a stable edge-touching simulation into a divergent one ... with
no NaN and no exception, so nothing downstream catches it" (#627b, tried and
reverted). #801 is absorber-depth instability from the other direction.

**So the landing needs a stability gate as well as a correctness gate**: a
long-record run (25000 steps at cv01's scale) on a boundary-touching dielectric
under both absorber families, asserting a finite result and a settling witness
that resolves. A fix that makes the permittivity right and the simulation
divergent is worse than the facet.

### 8.6 If this lands it is the first of its kind in the case ledger

`docs/agent-memory/rfx-known-issues.md:4343` is the comparator-bug case ledger,
currently 13 of 13 closed as comparator/extractor defects, and
`research/CLAUDE.md` states that as an invariant rather than a tally. This one
would close as a **solver-assembly** defect: the comparator (cv03's two-wave
estimator) was right, the extractor was right, the oracle was right, and the
array the solver received was wrong.

The ledger and that invariant sentence live in gitignored memory, which this
lane does not edit. Flagged here for the PI to update after landing.

### 8.7 Also for the PI, separately: two preflight gaps this exposed

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
  uses (`measure_neff_two_wave` in
  `validation/crossval/comparators/slab_te_dispersion.py` -- named with the
  file and the symbol separately on purpose. The double-colon spelling is a
  symbol reference, which the numeric-provenance parser rejects by
  construction, and one of those in a backtick span takes the whole note out
  of the gate),
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

### 9.5 APPEND — the readability control in 9.2 named the wrong committed number

Written after Arm E ran, append-only; 9.2 stands as written and is superseded
on this one point.

9.2 says the committed arm must reproduce
`SWEEP_BASELINE_CPML_FULL_MEAN_SELF = 0.7488520140093946`. That constant is the
**10-layer** arm — `scripts/diagnostics/cv01_cpml_flux_selfcheck.py:420-424`
calls it "the sweep's control" and `run_arm`'s own docstring says
`cpml_layers` "defaults to cv01's `cpml_n` = 10". 9.1 declared the run at
cv01's **committed** depth for the `cpml` boundary, which
`cv01_cpml_flux_selfcheck.py:208` fixes at **20**. So the gate named a number
from a different depth than the one it gated.

The gate's FORM does not move: the committed arm must reproduce the committed
`mean_self` for the depth it runs, to within 0.005. The number that belongs to
20 layers is `scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json`,
`layers.20.full.mean_self = 0.9195301017439319`.

Measured: **0.9195301017439319** — bit-identical to all 16 digits. Residual
0.0 exactly. That also settles a question the arm had to answer about itself:
the instrumented `Simulation` subclass, which attaches a point probe and a DFT
plane probe before delegating, **did not perturb the rig**.

No other gate in 9.2 moves, and the `|B/A|` bars are untouched.

---

## 10. APPEND (2026-09-15 KST) — the divergence IS classified; section 8.5 is superseded

Append-only. Sections 0-9 are left as written; **section 8.5 is superseded by
this section**, which replaces its "the lane did not isolate which of the three
differences carries it" with a measurement.

Four arms on cv01 Run 1 (20 layers, 25000 steps, x64 set before import),
contributed by the independent verification of `dccee459`:

| arm | result |
|---|---|
| CPML + widened `Box`, **+2a flush** (0 partial cells) | diverges at **step 402**, 24,598 non-finite |
| CPML + widened `Box`, **+4a overshoot** (0 partial cells) | diverges at step 402, identical envelope |
| **UPML** + widened `Box` | **stable**, `mean_self` = 0.9912 |
| CPML + committed `Box`, **subpixel OFF** (eps 12 in the pad via `extend_cpml_pad_materials`) | **stable**, `mean_self` = 0.9887 |

What that excludes, one candidate per row:

- **not the Kottke half-cell.** Flush and overshoot both give 0 partial cells at
  the seam and both diverge at the same step with the same envelope.
- **not the record length.** Divergence is at step 402 of 25000, not late.
- **not x64.** Set before import in all four.
- **not "a dielectric in a CPML pad" per se.** The subpixel-OFF row puts
  `eps_r = 12` in the CPML pad through `extend_cpml_pad_materials` and is
  stable at 0.9887.
- **not the absorber family alone.** UPML with the same widened `Box` is
  stable at 0.9912.

**What is left is the combination**: the smoothed / anisotropic array carrying
the dielectric into a **CPML** pad — `update_e_aniso` (or the aniso path
generally) together with CPML. The suspect coefficient site is that the CPML
psi correction and the Yee half of the update disagree about epsilon:
`rfx/simulation.py:1464-1466` passes `materials=materials` into
`apply_cpml_e`, and `rfx/boundaries/cpml.py:621-632` builds
`_ce_full = dt / (materials.eps_r * EPS_0)` from it, while the Yee update is
using `aniso_eps`. Wherever those two arrays differ, the two halves of one
timestep are integrating different permittivities.

**One caveat on the widened-`Box` emulation**, which is why section 11 exists:
widening the `Box` past the domain edge also trips the #61 geometry-in-absorber
preflight check. A real pad-replication fix does not — it changes no declared
geometry. So the widened `Box` is a *proxy* for the fix, not the fix, and the
proxy diverging does not by itself establish that the fix does.

### 10.1 Side finding for #813, recorded here because this lane measured it

Removing the facet moves cv01's `mean_self` **0.9741 -> 0.9912** under UPML
(single variable: only the `Box` changes) and **0.9195 -> 0.9887** under CPML
(confounded: the subpixel-OFF row changes two things). The committed
CPML-vs-UPML gap that #813's layer sweep is built around therefore looks
**facet-dominated** rather than absorber-dominated. Not a verdict on #813 —
that lane owns its own re-measurement — but it is the reason a fix cannot be
landed there without re-running the sweep.

### 10.2 A number corrected in section 8.3

Section 8.3's cv03 bullet said band-mean `T` "moves 0.9657 -> ~0.99", which
contradicts 8.4 and the B1 measurement in the results note. It is corrected in
place rather than left standing with a marker, because a wrong number in a
durable document is the failure class #814 and #829 exist to stop. The
corrected text says what B1 measured: `T` moves to 0.9682 and keeps degrading
with depth; only the subpixel-OFF route reaches 0.994.

---

## 11. APPEND (2026-09-15 KST) — Arm F: is the assembly fix alone landable?

Written before the probe runs. Sections 0-10 stand.

Section 10 leaves one question open that decides how #1043 is written: the
widened `Box` is a *proxy* for the fix (it also trips the #61
geometry-in-absorber check, which a real fix does not), so its divergence does
not by itself establish that the fix diverges. This arm runs the fix itself.

### 11.1 What runs

The pad replication applied to `aniso_eps`, at the site section 8.1 names
(`rfx/runners/uniform.py:266-273`), reusing `extend_cpml_pad_materials`
(`rfx/geometry/rasterize_grid.py:877`) rather than writing a second copy of the
replication logic — the shape a real fix would take, per #627.

**It is applied as a monkeypatch in the probe driver. No file under `rfx/` is
modified or committed on this branch.** The patch wraps
`rfx.geometry.smoothing.compute_smoothed_eps`, which
`rfx/runners/uniform.py:267` imports at call time, and extends each of the
three returned components into every pad face from the grid's own
`pad_*_lo/hi`.

Rig: **cv01 Run 1, CPML, 20 layers, 25000 steps, `subpixel_smoothing=True`,
the COMMITTED `Box`** — no geometry change, so the #61 preflight warning must
NOT appear. `JAX_ENABLE_X64` set before any import that pulls JAX in.

Recorded: first non-finite step if any, the per-decile amplitude envelope, the
preflight banner verbatim, `settling_db` and its witness, `mean_self`, and the
solved centre-row permittivity with and without the patch (so the patch's
effect is shown, not assumed).

### 11.2 Gate, frozen before the run

Reference: the subpixel-OFF CPML row from section 10, `mean_self = 0.9887` —
the same rig with `eps_r = 12` in the pad by the interior route.

- **ASSEMBLY FIX ALONE IS LANDABLE**: the run is finite through all 25000 steps
  **AND** `settling_db <= -40 dB` **AND** `|mean_self - 0.9887| <= 0.01`.
  residual `r_F = (0 if finite else 1) + max(0, settling_db - (-40))
  + max(0, |mean_self - 0.9887| - 0.01)`; gate `r_F = 0`.
- **BLOCKED**: the run goes non-finite. Then the landing needs a CPML+subpixel
  coefficient fix FIRST — start at `rfx/simulation.py:1464-1466` and
  `rfx/boundaries/cpml.py:621-632`, where the psi correction takes epsilon from
  `materials.eps_r` while the Yee half uses `aniso_eps` — and **#1043 must say
  the assembly fix is blocked on it**.
- finite but missing one of the other two legs: **INCONCLUSIVE**, declared now,
  and reported as such.

### 11.3 Declared expectation, written down before the run

**I expect it to diverge.** Section 10's flush-vs-overshoot pair leaves almost
nothing between the widened `Box` and this patch in terms of what the two
arrays hold: both put `eps_r = 12` in the pad of the smoothed array under CPML.
The differences are the seam half-cell (6.5 against a sharp step) and the #61
preflight trip, and section 10 already excluded the half-cell. If it diverges,
the useful content is the confirmation that the fix and its proxy behave the
same, which is what licenses writing "blocked" into #1043. If it does NOT
diverge, then something in the proxy other than the array is doing the damage
and section 10's conclusion needs re-opening — which is exactly why this is
worth one run.

### 11.4 R2

One attempt, one new question. No re-run without another append.

---

## 12. APPEND (2026-09-15 KST) — Arm F result: BLOCKED

Ran against section 11's frozen gate.
Artifacts `scripts/diagnostics/_artifacts/cv03_seam_facet/pad_replication_probe.json`
and `..._shape.json`.

### 12.1 Verdict

**BLOCKED.** `r_F = 3.0`, gate 0.

| leg | measured |
|---|---|
| finite through 25000 steps | **no** — 24,601 non-finite, first at time-series index **399** |
| `settling_db <= -40 dB` | **none** — "settling witness has INVALID RECORDS: probe0(ez): non-finite samples" |
| `\|mean_self - 0.9887\| <= 0.01` | **NaN** |

> [run] result contains non-finite values in time_series (24601 value(s)) — the
> FDTD likely diverged.

First amplitude decile already `2.61e+13`; every later decile NaN.

**And the #61 check did not fire**, as section 11 required: the preflight banner
carries only the four `cells per lambda_eff` / lossless-in-open-domain
advisories, with no geometry-in-absorber line, because no declared geometry
changed. So the divergence is **not** an artefact of the widened-`Box` proxy's
preflight trip, and section 10's "the proxy may not represent the fix" caveat is
now closed in the direction of the proxy being representative: the proxy
diverges at step 402, the fix itself at index 399.

The declared expectation in 11.3 was divergence, and it is recorded as having
been declared before the run rather than discovered after it.

### 12.2 What this means for the landing

**The assembly fix alone is not landable.** #1043 must say it is blocked on a
CPML + subpixel coefficient fix, and the place to start is the epsilon
disagreement between the two halves of one timestep:
`rfx/simulation.py:1464-1466` passes `materials=materials` into
`apply_cpml_e`, and `rfx/boundaries/cpml.py:621-632` builds
`_ce_full = dt / (materials.eps_r * EPS_0)` from it, while the Yee half uses
`aniso_eps`.

One new piece of evidence for that reading: the probe put **6.5** in the pad
(12.1.1 below), the widened-`Box` arms put **12**, and both diverge within three
steps of each other. **The divergence is insensitive to the pad's permittivity
value**, which is what an inconsistent-coefficient mechanism looks like and is
not what a material-mismatch mechanism would look like.

### 12.3 The fix's shape, measured (no FDTD)

The probe reused `extend_cpml_pad_materials` on the smoothing output, which is
the obvious shape and is **wrong**: that function replicates the **interior-edge
slice**, and on the smoothed array that column is the Kottke half-cell, not the
material. cv01's committed build, guide centre row, x-lo seam:

| variant | pad value | row across the seam | cells at `eps_r = 12` |
|---|---:|---|---:|
| committed (no replication) | 1.000 | `1.0, 1.0, 6.5, 12.0, 12.0` | 159 |
| naive `extend_cpml_pad_materials` on the smoothed array | **6.500** | `6.5, 6.5, 6.5, 12.0, 12.0` | 159 |
| sourced one column inward | 12.000 | `12.0, 12.0, 6.5, 12.0, 12.0` | 199 |
| reference: `_assemble_materials` (the interior route) | 12.000 | `12.0, 12.0, 12.0, 12.0, 12.0` | 201 |

Sourcing one column inward gets the pad right but strands the 6.5 half-cell
*inside* the absorber — a one-cell film between the guide and its own matched
pad, which is precisely the #655 failure mode the interior route already had to
repair. **So the fix is to smooth the ALREADY-EXTENDED array**, not to
post-process the smoothing output. That is the only one of the three that
reproduces the reference row.

This does not change 12.1's verdict — the probe's 6.5 and the proxy's 12 both
diverge — but it is the shape #1043 should carry, and it is measured rather
than reasoned.

### 12.4 One reporting defect in the probe, recorded

`probe_pad_replication.py`'s instrument check counted only cells at
`eps_r = 12`, so it printed "159/201 both ways" and read as though the patch had
not taken effect, while `solved_first3` in the same record showed `6.5`. The
run had in fact changed (the committed configuration is stable; this diverged).
The counter is the defect, not the patch. `pad_replication_shape.py` is the
companion that reports the pad's actual value, and it runs no FDTD, so no R2
attempt was spent on the repair.

### 12.5 R2

Arm F: one attempt, closed on its pre-declared gate. Not re-run.

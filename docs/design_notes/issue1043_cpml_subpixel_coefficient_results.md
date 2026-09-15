# #1043 Stage A — results

Run against the gates frozen in
`issue1043_cpml_subpixel_coefficient_predeclaration.md`. Artifacts under
`scripts/diagnostics/_artifacts/cpml_subpixel_stability/`; drivers under
`scripts/diagnostics/cpml_subpixel_stability/`.

Every "main" row was produced against a `git archive` of `origin/main`
(499c8e1e) unpacked to a scratch directory, not against a checkout — a builder
editing the same worktree moves the review target silently.

## 1. The mechanism, in three sentences

`apply_cpml_e` built its psi coefficient from `materials.eps_r` while the Yee
half of the same timestep used the subpixel-smoothed `aniso_eps`, so in every
cell where those two arrays disagree the one E update integrated two different
permittivities. With `kappa_max` at its default 1.0 the whole CPML-E correction
is `ce * psi`, and since `psi = b*psi_prev + c*curl` with `c` in `(-1, 0]`, the
instantaneous curl coefficient of such a cell is proportional to
`1/eps_a + c/eps_b` — non-negative for every `c` when the two agree, negative
once the Yee half sees the HIGHER permittivity. A subpixel-smoothed interface
cell inside a CPML pad is exactly that case: the smoothed array carries the
Kottke interface value while the staircase `materials.eps_r` at the same cell
reads whichever side the sample point fell on.

## 2. Hypothesis table

| | hypothesis | verdict | evidence |
|---|---|---|---|
| **H1** | epsilon disagreement between the Yee half and the psi half, unstable when `eps_a > eps_b` | **SUPPORTED** | G-A `r_A = 0`, G-C `r_C = 0`, G-D `r_D = 0` |
| **H2** | CPML profile derived for vacuum + a high-eps cell in the graded region violates a local CFL bound only on the anisotropic path | **FALSIFIED** | G-C consistent-eps scan: `rho = 1.000000` at eps = 1, 2, 4, 6.5, 12, 30, 80 — an absolute permittivity in the graded region is exactly marginal at every value. The eigenvalue moves only when `eps_b` (which is not a property of the medium the wave travels in) is dropped below `eps_a`. |
| **H3** | `1/kappa` applied on the wrong side of the tensor | **FALSIFIED BY INSPECTION, no run spent** | `kappa_max` defaults to 1.0 (`cpml.py:106`, `init_cpml:451-452`), so `_kappa_correction` returns exactly 0 in every arm of #1043's table. Cannot be the mechanism for a divergence that occurs at `kappa = 1`. |
| **H4** | the `update_e_aniso` + CPML + boundary-touching-dielectric path was never exercised by a test | **CONFIRMED (coverage, not mechanism)** | `tests/unit/boundaries/test_cpml_pad_material_extension.py`'s two solver-running tests (`:406`, `:563`) both pass `subpixel_smoothing=False`, and both fixtures carry Lorentz poles, which `simulation.py:415` routes away from the anisotropic branch anyway. No test in the tree combined the three. |

## 3. G-A — the coefficient map (no FDTD)

`K_eff_norm = 1/eps_a + c/eps_b` over every cell `apply_cpml_e` writes, on the
small 2-D rig (`cpml_subpixel_coefficients.py`, 101x101x1, 10 CPML layers,
a full-span eps 12 guide, cv01 Run 1's topology at quarter size).

| arm | `K_eff_norm` min | negative cells | FDTD on main | FDTD on this branch |
|---|---:|---:|---|---|
| `cpml_baseline` (today, no pad replication) | **+0.007941** | 0 | finite | finite |
| `cpml_padrep` (Stage-B replication on `aniso_eps`) | **−0.838213** | 21 | **848 non-finite, first probe index 352** | **finite** |
| `cpml_subpixel_off` | **+0.000662** | 0 | finite | finite |
| `upml_padrep` | (not defined — `apply_cpml_e` never runs) | — | finite | finite |

`r_A = 0`; `r_D = 0` against the pre-fix table on main and against the
all-finite table on this branch. The `K_eff_norm` column is a property of the
two arrays, so it reads the same on both trees — what the fix changes is which
of the two the solver uses, and the FDTD columns are where that shows.

The UPML arm is excluded from `r_A` rather than scored: `apply_upml_e` builds
its own coefficients from `aniso_eps` (`upml.py:213-217`) and has no second
permittivity to disagree with, which is why it was stable all along.

## 4. G-B — where it breaks first

Bisected to the first step at which any field cell is non-finite: **step 328**,
five components across **three cell positions**, all at `j = 55` — the guide's
transverse interface row — inside the x-hi pad.

| component | cell | `eps_a` (update) | `eps_b` (psi) | `c` | `K_eff_norm` | if consistent |
|---|---|---:|---:|---:|---:|---:|
| ez | 95, 55, 0 | 6.500 | 1.000 | −0.3459 | **−0.1921** | +0.1006 |
| ez | 96, 55, 0 | 6.500 | 1.000 | −0.5636 | **−0.4097** | +0.0671 |
| ez | 97, 55, 0 | 6.500 | 1.000 | −0.7614 | **−0.6075** | +0.0367 |
| ex | 95, 55, 0 | 1.000 | 1.000 | −0.3459 | +0.6541 | +0.6541 |
| ex | 96, 55, 0 | 1.000 | 1.000 | −0.5636 | +0.4364 | +0.4364 |

**Position-resolved `r_B = 0.0`** — all three positions are in the
`K_eff_norm < 0` set. **Component-resolved `r_B = 0.4`, which MISSES its
pre-declared gate of `<= 0`**: the two `ex` rows are the same Yee cells being
dragged along (one cell, shared `Hy`/`Hz`), so the component-resolved measure
undercounts by construction. Both figures are in the artifact; the
pre-declared one is recorded as not met rather than replaced.

## 5. G-C — the amplification factor

1-D leapfrog + psi over 24 cells (10 CPML layers), `amplification.py`.
Comparator checked first: a lossless vacuum slice sits on the unit circle to
2e-15 (`rho` = 1.0000000000000016 without the absorber, 1.0000000000000009
with a consistent CPML). (An earlier version of this matrix took rfx's
`hy -= (dt/mu)*curl_y` literally instead of resolving
`curl_y = dEx/dz - dEz/dx` in the 1-D slice, reported `rho = 1.3 … 2.6` for
every case including vacuum, and was caught by exactly this check.)

| case | `rho` | |
|---|---:|---|
| `eps_a = 6.5`, `eps_b = 1.0` (the measured failing cell) | **2.208604** | unstable |
| `eps_a = eps_b = 6.5` (the fix) | **1.000000** | marginal |
| `eps_a = 1`, `eps_b = 12` (today's cv01 pad) | 1.000000 | marginal — wrong but not amplifying |
| `eps_a = eps_b` at 1, 2, 4, 6.5, 12, 30, 80 | 1.000000 | H2 falsified |

`r_C = 0`. Sweeping `eps_b` at `eps_a = 6.5`: `rho` = 2.2086 (1.0), 1.6762
(2.0), 1.3131 (4.0), 1.0989 (6.0), 1.0580 (6.294), 1.0249 (6.45), 1.0091
(6.49), **1.000000 (6.5 = `eps_a`)**, and 1.000000 at 8, 12, 20. The
instantaneous sign flip (`K_eff_norm = 0` at `eps_b = 6.294`) is the leading
indicator; the eigenvalue crosses 1 only at `eps_b = eps_a` exactly, so the
condition is `eps_b >= eps_a` and consistency is the one choice that is both
correct and marginal.

Courant sweep at the failing cell: defect `rho` = 1.354 / 1.727 / 2.209 / 3.035
at Courant 0.2 / 0.35 / 0.5 / 0.7; the fixed path is `1.000000` at all four.

## 6. The fix

`apply_cpml_e` takes an optional `inv_eps_r_update` — the per-component
INVERSE relative permittivity the E half-step actually used — and builds its
psi coefficient from it. `None` keeps `materials.eps_r` and every byte with it.

(§4 of the pre-declaration wrote this as "`aniso_eps` on Stage 1,
`1/aniso_inv_eps` on Stage 2". The shipped parameter is the reciprocal of that
— one inverse-permittivity argument, `1/aniso_eps` on Stage 1 and
`aniso_inv_eps` as-is on Stage 2 — because the inverse form is the one that
stays finite at a Kottke-frozen PEC cell, where `inv = 0` and the permittivity
does not exist. Same quantity, one fewer division, and the pre-declaration is
left as frozen rather than retro-edited to match.)

Threaded at the two call sites that have such an array:

| site | what it passes |
|---|---|
| `rfx/simulation.py` (uniform scan) | `aniso_inv_eps` as-is on the Stage-2 `kottke_pec` path (`inv = 0` at a frozen PEC cell then correctly receives no psi correction), `1/aniso_eps` on Stage 1, `None` when a dispersion model is active (the E update ignores the anisotropic arrays there, so this must too) |
| `rfx/nonuniform.py` (NU scan) | the same, guarded by the same condition that selects `update_e_nu_aniso` |

Every other `apply_cpml_e` caller (`vmap_sweep`, `subgridding`, `probes`,
`runners/distributed*`) runs `update_e`/`update_e_nu` with `materials.eps_r`
and has no anisotropic array at all — nothing to thread, nothing changes.

One implementation detail is load-bearing and was measured, not reasoned about:
`compute_smoothed_eps` returns **float64** under `JAX_ENABLE_X64` while
`materials.eps_r` stays float32. The first version of the fix let that through,
which promoted the psi-scatter precision and moved the field bytes of
`cpml_interior_sub` — a run whose pads hold nothing but vacuum. The coefficient
is now cast to the material dtype, which is the policy the function already
documented at its `dt` handling (#646, promote-never-pin: follow the ambient
material dtype).

Recorded and NOT fixed: `update_e_aniso` divides its curl coefficient by
`(1 + sigma*dt/(2*eps))` and the psi coefficient does not. Same family, but it
is identically zero wherever the absorber pad is lossless (every configuration
measured for #1043) and no witness for it exists.

## 7. Bit identity

SHA-256 over the raw final field bytes of all six components, 400 steps,
`bit_identity.py`, main-archive vs this branch.

| case | identical | |
|---|---|---|
| `cpml_vacuum_nosub` | yes | |
| `cpml_vacuum_sub` | yes | |
| `cpml_interior_nosub` | yes | |
| `cpml_interior_sub` | yes | subpixel + CPML, dielectric clear of every pad |
| `cpml_touching_nosub` | yes | |
| `cpml_touching_lossy_nosub` | yes | |
| `upml_touching_sub` | yes | |
| `upml_touching_nosub` | yes | |
| `pec_touching_sub` | yes | |
| `cpml_touching_sub` | **NO** | declared before the run |
| `cpml_touching_lossy_sub` | **NO** | declared before the run |

The two that move are the only ones with a subpixel-smoothed interface cell
inside a CPML pad. Declared in `expect_change` before either tree ran, and the
measured change set matches it exactly.

### cv01 Run 1

The committed `mean_self = 0.9195301017439319` (20 layers, 25000 steps,
`subpixel_smoothing=True`, the committed `Box`) **moves**, as declared in §4 of
the pre-declaration. Measured on the same rig, both trees, in
`cv01_control.py`:

| tree | `mean_self` (20 layers) | delta vs committed |
|---|---:|---:|
| `origin/main` (499c8e1e) | 0.9195301017439319 | — |
| this branch | 0.9166511849380675 | **−0.0028789168058643844** (−0.31 %) |

Why, shown rather than argued (dumped from the committed build): in the x pads
the guide's centre row reads `materials.eps_r = 12` (the pad extension put it
there) while `aniso_eps` reads `1` (the declared `Box` stops at the interior
edge and nothing replicates into the rebuilt array — that gap IS #1043's
headline defect). The psi coefficient was therefore twelve times too small in
exactly the cells meant to absorb the guided mode. The fix makes the absorber
match the medium the solver is actually stepping, which is a change of
substance, not of rounding.

**Preflight is part of the result.** Both arms carry the same banner, verbatim
and identical between the two trees (full text in the artifacts'
`preflight_warnings` / `run_warnings`):

> `add_source(..., amplitude_kind=None): on this simulation the waveform
> amplitude is a Cb-normalized field add (E += Cb*w; NOT one of the two named
> kinds …)` — issue #571 deprecation, cv01's own registration, unchanged here.

> `[run] preflight found 4 advisory issue(s)` — `dielectric 'wg' on x / y / z:
> 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20
> cells/λ_eff for phase-accurate propagation …`, plus the lossless-in-an-open-
> domain advisory.

No geometry-in-absorber (#61) line on either arm — no declared geometry
changed, which is what makes the two `mean_self` values comparable. cv01 Run 1
records `settling_db = None` on both trees: that rig registers flux monitors,
not probe records, so no ring-down witness is established. **That is a
pre-existing property of cv01's registration, identical before and after, and
it is not evidence that either number is settled.** The witness this branch
does gate on lives in the new test, which registers a probe and asserts
`settling_verdict(...) == "pass"`.

## 7b. What this branch leaves stale, and deliberately does not re-point

Moving cv01's CPML `mean_self` moves every committed surface that recorded a
CPML number off that rig. **No gate goes red** — the numeric-provenance
contract checks that cited keys resolve in the artifact, not that a re-run
reproduces them, and the driver below is on-demand, not collected by pytest —
but the numbers now describe the pre-fix solver:

| surface | what it holds |
|---|---|
| `scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json` | the CPML `mean_self` ladder: `/control/cpml_full_mean_self` 0.7488520140093946, `/layers/{10,16,20,40}/mean_self_full` = 0.7488520140093946 / 0.8840995730804735 / 0.9195301017439319 / … and their `aperture` siblings |
| `scripts/diagnostics/cv01_cpml_flux_selfcheck.py` | `SWEEP_BASELINE_CPML_FULL_MEAN_SELF = 0.7488520140093946`, `SWEEP_BASELINE_OUTSIDE_FRACTION_PCT = -26.041819705557895`, and the `SWEEP_GATE_MEAN_SELF_AT_40` derived from them |

Unaffected, checked rather than assumed: the committed cv01 case record
(`validation/crossval/_01_waveguide_bend_results/crossval.json`,
`mean_self_smoothed_over_band = 0.9891610388008335`) is a **UPML** run, and
UPML is byte-identical here (§7).

Re-pointing the CPML ladder is #813 / Stage-B work, not this branch's: the
whole ladder has to be re-measured together, and #1027's monotone `mean_self`
trend re-read off the new one. Re-pointing one stored number without
re-measuring its siblings is how a stored primary silently re-points its
readers.

## 8. New tests

`tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py`.

| test | on `origin/main` | on this branch |
|---|---|---|
| `test_boundary_touching_dielectric_in_pad_stays_finite[cpml]` (slow, 5000 steps) | **RED** — 4662 non-finite probe samples | green |
| `test_boundary_touching_dielectric_in_pad_stays_finite[upml]` (slow) | green | green |
| `test_amplification_is_bounded_only_when_the_two_epsilons_agree` | green (pure analysis, tree-independent) | green |
| `test_threading_an_equal_permittivity_is_bit_identical` | RED (`TypeError`, the parameter does not exist) | green |
| `test_threading_a_different_permittivity_changes_the_coefficient` | RED (`TypeError`) | green |
| `..._threaded_exactly_when_the_e_update_is_anisotropic[subpixel]` | **RED** | green |
| `..._threaded_exactly_when_the_e_update_is_anisotropic[plain]` | green | green |
| `..._threaded_exactly_when_the_e_update_is_anisotropic[dispersive]` | green | green |

The last three pin WHICH runs get the new coefficient. The dispersive row is
the one that matters for a future edit: `_update_e_with_optional_dispersion`
ignores `aniso_eps`, so threading it unconditionally would re-open the same
disagreement with the sign reversed, and nothing else here would notice.

The committed-geometry CPML control is deliberately NOT pinned as unchanged —
its pad holds `materials.eps_r = 12` against `aniso_eps = 1`, so its numbers
move by design (cv01 Run 1, §7).

The `[cpml]`/`[upml]` pair is the discriminator: the same pad replication under
the other absorber family must keep passing, or the test is measuring "a
dielectric in a pad is unstable" rather than "these two coefficients disagree".

The pad replication in that test is a **test-local helper**, not a production
call: Stage B is what puts it under `rfx/`. When it lands, replace the helper
with the production path — do not delete the test with it.

**Mutation check.** A third tree carrying ONLY the `rfx/boundaries/cpml.py`
half of the fix — the new parameter exists and behaves, the callers still do
not pass it — leaves exactly two tests red:
`..._stays_finite[cpml]` and `..._threaded_exactly_..._[subpixel]`. So the
physics test is not measuring the API surface, and the `simulation.py` /
`nonuniform.py` threading is load-bearing rather than decorative.

Whole-file counts, each measured by running the same file against that tree
rather than read off the table above: `origin/main` **4 failed, 4 passed**;
`rfx/boundaries/cpml.py` half only **2 failed, 6 passed**; this branch
**8 passed**.

## 8b. Suites

All on this branch, CPU, jax 0.6.2. `tests/unit/boundaries/ tests/oracle/
tests/contracts/` was split into 8 groups with pytest-split (no xdist in this
env, and the repo's own note says xdist loadfile workers accumulate XLA state).

| run | result |
|---|---|
| the three directories, 8 shards, full pass | **3175 passed, 8 skipped, 17 xfailed, 1 failed** |
| the one failure: `test_evidence_numeric_provenance.py::test_every_enumerated_document_is_classified` — that contract enumerates every `docs/design_notes/*.md` and refuses an unclassified one, and it caught both new notes | fixed by classifying both as `NO_ARTIFACT_REFERENCE` (neither carries a `path.json::key` span); **its shard re-run 606 passed / 0 failed**, and the whole contract file re-run **1422 passed** |
| the new test file re-run in full after its last edit | **6 passed** (fast) + **2 passed** (slow) |
| `tests/unit/boundaries/ -m slow` | **7 passed, 211 deselected** — the three `test_cpml_reflectivity_regression` cases, `test_pole_extension_divergence_repro_636` (the #636 pole-pad divergence lock on the neighbouring mechanism), `test_clamped_per_face_absorber_actually_absorbs`, and both new `..._stays_finite` cases |
| `tests/unit/nonuniform/` — the other edited path | **399 passed, 11 deselected, 1 xfailed** |
| `tests/crossval/` cv01/cv03 files: `test_cv03_slab_dispersion_oracle.py`, `test_crossval_comprehensive.py`, `test_crossval_gate_logic.py`, `test_aux_echo_record_invariant.py` | **71 passed, 11 deselected** |
| `ruff check rfx/ tests/ scripts/diagnostics/cpml_subpixel_stability/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402` | **All checks passed** |

Stated precisely so the coverage claim is checkable: the 8-shard pass ran
against this branch's `rfx/` code, which has not changed since. What changed
afterwards is the new test file, the contract file and the design notes — all
three re-run directly and in full, as the rows above record. A confirmatory
clean 8-shard re-run was started on the final tree; shards 4, 6, 7 and 8 came
back green (548 / 905 / 606 / 282 passed) and the remaining four were still
grinding through long `tests/oracle/` cases after 75 minutes, so it was stopped
rather than left to run. It is confirmatory, not additional coverage.

## 9. R2 ledger

One attempt, covering G-A + G-B + G-C + G-D together as pre-declared. H1 closed
on it. H2 was discriminated inside G-C and falsified; H3 was falsified by
inspection with no run spent; H4 is a coverage finding and is now closed by the
new test file. No second attempt was needed and none was taken.

The one pre-declared gate NOT met is the component-resolved leg of G-B
(`r_B = 0.4` against `<= 0`), recorded in §4 with the reason and the
position-resolved figure alongside.

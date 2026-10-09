# Design — tracker 1589: the absorber's auxiliary term uses the E update's own curl coefficient (leader, 2026-10-10)

Status: cause measured (rfx-archive `rfx/records/20261010-1589-lossy-medium-through-cpml/`); design fixed here before any code.

## Physics
Inside an absorber layer the E update is `E += Cb * (curl H + psi-term)`: the absorber's auxiliary term is a correction
to the curl, so it must be multiplied by the same coefficient `Cb` as the curl in that cell. rfx multiplies it by
`dt/eps` instead. In a lossless, non-dispersive cell the two are equal. In a lossy cell `Cb = (dt/eps)/(1 + s)`,
`s = sigma*dt/(2*eps)`, so the correction is `(1 + s)` times too strong; in a Debye cell (and a mixed Debye+Lorentz
cell) `Cb` also contains the pole terms and the correction is too strong by the corresponding factor. A Lorentz/Drude
pole does not enter `Cb` (it enters through the polarization current), so a Lorentz-family cell is wrong only by its
static-conductivity factor `(1 + s)`.
Measured on main (uniform, run and forward): a vacuum-side conductor through a CPML face grows from s of about 1
(about 108 ohm/sq as a sheet, on 0.5 / 1 / 2 mm cells; 10 S/m at 1 mm: 2.3e5 after 6000 steps at 6 layers, 1.5e24 at 16);
a lossy dielectric (eps_r 4) already grows at 0.7 of that conductivity, where s < 1 — the s = 1 rule is a vacuum-side
observation, not the criterion; a Debye slab (eps_inf 2, delta_eps 50, tau 5 ps — water-like) grows to 1e17 and two
stronger Debye slabs go non-finite; a Drude slab (fp 50 GHz) with static conductivity at s = 1.2 grows and at s = 3.2
goes non-finite. With the absorber given the update's coefficient every one of these decays (37 plain lossy arms incl.
the eps_r 4 arm, 3 Debye arms, 2 Drude+sigma arms) and a model without loss is array-identical.
One arm is NOT an absorber case and stays on record as such: a Drude slab with fp 200 GHz, gamma 1e11 at dx 1 mm
(omega_p*dt = 2.4) goes non-finite on main AND patched — the declared falsifier fired — and the control shows why: it
goes non-finite in an all-PEC box with no absorber, and its `Cb` equals the plain coefficient (sigma 0), so the patched
arm could not differ. That is the explicit Drude update's own stability limit, with no refusal or warning that names
it: reported to the materials lane, not part of this fix. Below the growth threshold the same mismatch is a silent error of the
absorber for every lossy or dispersive medium that reaches an open face (size not measured: census item).

## Invariant
For every E component c and every cell of an absorber slab: the coefficient that multiplies the absorber's auxiliary
terms (psi and the kappa term) equals the coefficient that multiplies curl H in that cell's E update of the same step.

## Form (chosen so that lossless results do not move)
- Plain media (no poles): `ce_c = ce_today_c / (1 + loss_c)`, with `loss_c` the E update's own per-component loss
  term (`sigma_c*k/eps_c`, or `sigma_c*k*inv_c` on the inverse-permittivity route; `k = dt/(2*eps0)`), taken from the
  same function the update uses (`rfx.core.yee.e_coeffs_eps_r_units`), per-component conductivity = the edge-mean the
  update used (`materials.components.sigma_update`). Where sigma is 0 this is `x / 1.0`: the coefficient is bitwise
  today's. (Using the update's `Cb` array directly was rejected: on the component route today's coefficient is formed
  from `1/eps` and differs from `Cb(eps)` by 1 ULP on 41 % of random float32 permittivities — measured — which would
  move every open-boundary result with a dielectric in the pad.)
- Lorentz/Drude-only models: the owner's `cb = dt/(eps_inf + sigma*dt/2)` (`rfx/materials/lorentz.py`
  `_lorentz_e_coeffs_si/_eps_r`) is the plain form, so the plain rule above applies unchanged and a Lorentz model
  without static conductivity keeps its bits.
- Debye models and mixed Debye+Lorentz models: `ce_c = cb[c]` of the coefficient the update itself multiplies the curl
  with — `DebyeCoeffs.cb` (`rfx/materials/debye.py`), and for the mixed update the `cb` returned by
  `_mixed_e_coeffs_si/_eps_r` (`rfx/materials/lorentz.py`), NOT `LorentzCoeffs.cb`, which the mixed update does not
  use for the curl. Results with a Debye medium in a pad move; that is the fix.
- Where the value comes from in the step: the absorber takes the coefficient from the same place the E update of that
  step takes it. On a path that forms the E coefficients inside the time loop (the multi-device runners do, per the
  `_lorentz_e_coeffs_si` docstring) the absorber uses that in-loop value (or, for the plain rule, the same in-loop
  per-component sigma and eps), never a copy formed before the loop — a pre-loop copy can differ in the last bit from
  the in-loop value (ledger cross-trace entry, "a coefficient moved out of the compiled loop").
- Coefficient owners, complete list: plain (`rfx.core.yee` `e_update_coeffs` / `e_coeffs_eps_r_units`), Debye, Lorentz,
  mixed. A fifth owner found by the survey stops the work and is reported.
- Derivatives: the coefficient keeps the eps_r-unit derivative convention of #1357 (`si_value_eps_r_grad`); the
  gradient with respect to sigma must be finite and is compared against a finite difference in the contract.

## Interface
`apply_cpml_e` gains one keyword, `e_loss=None` (three per-component loss arrays or None) and one, `e_curl_coeff=None`
(three per-component coefficients, overrides everything else when given — the dispersive route). The step passes
`e_loss` from the same invariants that feed the E update, and `e_curl_coeff` when the model has poles. A caller that
passes `materials` with a non-zero `sigma` and neither keyword still gets the loss factor from `materials.sigma`
(cell value) — no silent lossless coefficient for a lossy model on any entry. The two multi-device copies
(`_apply_cpml_e_distributed`, the graded copy in `distributed_nu.py`) take the same two inputs until they are replaced
by the one update (2.0 step "multi-device paths call the one CPML update"). Public signature change: yes, two
keyword-only arguments on a function in the API reference → regenerate the inventory in the PR.

## Per-path table (to be filled by a code-fact survey, then by builds)
Call sites on origin/main 00a7328dc (leader, by grep; the survey confirms and fills owner columns):
| path | call site | coefficient today |
|---|---|---|
| uniform run / forward | `rfx/stepping/uniform.py:288` (`ctx.apply_cpml_e`, `inv_eps_r_update=cpml_inv_eps_r` from `rfx/simulation.py:1917-1935`) | `dt/eps` per component |
| probe-driven loop | `rfx/stepping/probe_loop.py:16,70` → the uniform step | same |
| vmap sweep | `rfx/vmap_sweep.py:537,590` → the uniform step | same |
| graded | `rfx/nonuniform.py:2875` (`_cpml_inv_eps_r`, `:2745-2755`) | same |
| multi-device uniform | `rfx/runners/_distributed_common.py:1765-1800` (own copy; `_ce_si(eps_face)`) | `dt/eps` per face slice |
| multi-device graded | `rfx/runners/distributed_nu.py:818-910` (own copy) | same |
| subgridded coarse grid | `rfx/subgridding/jit_runner.py:1972`, `rfx/subgridding/runner.py:128` (`materials=mats_c` only) | `dt/eps` from the cell value |
| grid-level eager extractors / `rfx/probes/probes.py:21` | imports `apply_cpml_e` | to be read |
| 2-D TF/SF auxiliary grid | `rfx/sources/tfsf_2d.py:378-396,436,513` (own vacuum grid; update and absorber both `dt/EPS_0`) | not reachable: vacuum only |
| UPML | own update | to be read: it decays on the issue's model; confirm it has no separately formed coefficient |
| ADI | own absorber | to be read; out of scope if it has no psi term |
Each row × each owner (plain lossy, Lorentz+sigma, Debye, mixed) ends as fixed / refuses / not reachable, with the
call site. Any path where the absorber forms its own coefficient is the same defect; none is left after the fix.
The H side: `MaterialArrays` carries eps_r, sigma and mu_r only (read 2026-10-10) — no magnetic-loss input, so the H
update's curl coefficient is `dt/mu` everywhere and `apply_cpml_h` has no analogue.

## Judges
1. Contract, always-on, small grids: a lossy slab (sigma chosen at s = 2, above the measured threshold) through each of
   the six faces in turn, on each path in the table that admits it, run and forward: the probe record's last-eighth
   peak is below its second-eighth peak (decays), and equals the uniform run's record within the cross-path bar where a
   path-equivalence bar exists. Same with a Debye slab (eps_inf 2, delta_eps 50, tau 5 ps), a Drude slab (fp 50 GHz,
   gamma 1e10) with static conductivity at s = 2, and a mixed slab (that Debye pole + that Drude pole) on the paths
   that admit them. The slab is asymmetric by construction: off-centre, thickness not symmetric about the probe, and
   one face at a time open with different layer counts on the two faces of the axis.
2. Invariant test, no time stepping: spy the coefficient the absorber multiplies its auxiliary term with and the
   update's `Cb` for the same cells, on models with lossy, lossless-dielectric, Debye, Lorentz+sigma and mixed regions
   in the pads (the coefficient is compared where the step runs it, not as a stand-alone value):
   equal within 2 float32 ULP per component, and bitwise equal to main's coefficient on the lossless cells (expected
   values for the latter captured from a base build, stored as literals for a handful of cells — not recomputed
   through the helper under test).
3. Bit-identity: the S0 path-equivalence matrix and every lossless committed lock unchanged (census below).
4. Gradient: d(objective)/d(sigma) through forward() on the lossy slab against a central finite difference, 1e-2.
5. Mutations (helper calls kept): loss factor dropped → 1 and 2 red; dispersive route falls back to dt/eps_inf →
   1 (Debye arm) and 2 red; mixed route takes `LorentzCoeffs.cb` instead of the mixed `cb` → 2 red (and 1 on the mixed
   arm); multi-device copy keeps the old coefficient → 1 red on that path; loss taken from the cell value where the
   update used the edge mean → 2 red.

## Census
Before/after on the same base: tests/locks, tests/oracle, tests/contracts, tests/unit/{boundaries,materials,sparams,
runners,autodiff,ports} including slow/slow_physics, on VESSL CPU long. Expectation stated now: lossless tests
unchanged bit for bit; tests with a lossy substrate or sheet reaching an open face (microstrip with loss tangent,
Leontovich/thin-conductor sheets, lossy films) and dispersive media in pads move — each reported with its shift; a
pinned number a user receives is re-pinned in the PR with the cause sentence.

## Stopgap content while the fix is in review
`docs/guides/known_limitations.md` wording (lead, 2026-10-10 — no "s >= 1" criterion is published): 'a conductor or
lossy medium running into a CPML face; measured: vacuum-side sheets at or below about 108 ohm/sq grow, a lossy eps_r 4
medium grows earlier'. Since the fix PR pins the invariant with committed tests, the PR carries no entry (an entry
leaves when a committed test pins the fix); the text above is for a PR that merges before the fix (idm's fallback).
The comment in `apply_cpml_e` that records this inconsistency "without a witness" is removed by the fix; the ledger
gets the witness.

## Not known
The growth threshold for eps_r > 1 and for dispersive media (not needed for the fix). The size of the silent absorber
error below the growth threshold (census reports the shifts). The mixed Debye+Lorentz owner has no pre-fix
measurement (judge 1 and 2 cover it in the PR). All measurements so far are on the uniform path, dx 1 mm unless
stated, 6-16 layers.

## Prior record this is held against (R1)
- Tracker 636 (closed 2026-08-30), "extending dispersion poles into the CPML pad destabilizes edge-touching
  structures": a Lorentz pole with a low-loss negative-permittivity band inside the pad grows (649x over 20,000 steps),
  cause attributed there to the pole in the pad itself. This fix is consistent with that record and does not reopen
  it: a Lorentz pole does not enter `Cb`, so the coefficient change cannot touch that mechanism, and this PR claims
  nothing about it. Consequence for the judges: the Lorentz-family arm is the lossy Drude pole measured here (fp 50 GHz,
  gamma 1e10, plus static conductivity; bounded over 6000 steps patched), never a low-loss resonant pole, so a judge
  cannot go red on 636's mechanism and be read as this one.
- Ledger, #1043 / #1357 notes in `apply_cpml_e`: the absorber's coefficient must take the permittivity the E update
  used (per-component edge mean), with the eps_r-unit derivative. This fix extends the same rule from the permittivity
  to the whole curl coefficient; both conventions are kept.
- Ledger cross-trace entry, "a coefficient moved out of the compiled loop": the reason the absorber takes the in-loop
  value where the update forms it in the loop.

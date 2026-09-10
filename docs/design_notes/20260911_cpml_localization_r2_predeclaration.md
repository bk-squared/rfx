# G5 — CPML localization, replacement gate pre-declaration

2026-09-11. PI-authorized replacement of G4's fired gate. Baseline: `29a92108`,
branch `feat/nu-cost-reduction`; work only in `/Users/byungkwankim/Documents/rfx-nu-cost`.
The G4 note is immutable. No candidate measurement or production edit precedes
this declaration. Results and baseline-only calibration will be appended.

## Frozen experiment and decision

G5-1 EXPRESSION IDENTITY. Reuse `887f8ff7:rfx/boundaries/cpml.py` exactly.
Baseline quotes: `_ch_full = dt / (materials.mu_r * MU_0)` followed by
`ch_xlo = _ch_full[:n_x, :, :]`; candidate:
`ch_xlo = dt / (materials.mu_r[:n_x, :, :] * MU_0)`.
Restriction commutes with pointwise multiplication/division, so each selected
cell evaluates `dt / (mu_r[cell] * MU_0)` in both. The six E coefficients obey
the same argument with eps_r and EPS_0.
Baseline: `hz_shifted_xlo = _shift_bwd(state.hz, 0)[:n_x, :, :]`;
candidate: `hz_shifted_xlo = _shift_slab(state.hz, 0, n_x, False, False)`.
At index i both read H[i-1], or zero for i=0. Forward shifts similarly read
E[i+1], or zero at the last cell. This applies independently to all 24 slab
derivatives, including transverse permutations.
Both retain verbatim:
`new_psi_ey_xlo = b_x_lo * cpml_state.psi_ey_xlo + c_x_lo * curl_hz_dx_xlo`.
Baseline: `ey = ey.at[:n_x, :, :].add(-ce_xlo * new_psi_ey_xlo)`;
candidate: `ey = _slab_add(ey, -ce_xlo * new_psi_ey_xlo, 0, False)`;
helper: `slab = lax.dynamic_slice(field, starts, correction.shape)` then
`return lax.dynamic_update_slice(field, slab + correction, starts)`.
Each slab index is unique, so both evaluate old_cell + correction exactly
once. Each subsequent kappa add stays separate and in the same order.
All psi recurrences, b/c/kappa selections, signs, transpose operations,
axis/face/component order and H -> CPML-H -> PMC -> E -> CPML-E -> PEC ->
source order remain unchanged. No public signatures or Yee edits.

Algebraic zero statement, for finite fields and initialized zero psi:
for c outside union of absorbing slabs, either no update addresses c, or
its padded/no-op profile has b=1, c_profile=0, kappa=1. Inductively
psi[t+1]=1*0+0*D=0 and correction = signed_coeff*0 +
signed_coeff*(1/1-1)*D = 0. Thus delta_E(c)=delta_H(c)=0 in both versions.
This is a source/algebra argument, not an empirical zero claim for arbitrary
nonzero externally supplied psi.

G5-2 CONTRACTION-NEUTRAL BIT-IDENTITY. All seven G4 fixtures, unchanged from
`tests/unit/boundaries/test_cpml_localization.py`: uniform8, graded8, mixed8,
uniform4, uniform16, periodic8, kappa8. Exactly 200 steps, all six fields
and all 24 psi, float32: require `np.array_equal`, zero differing elements.
Fresh-process baseline controls try `XLA_FLAGS=--xla_allow_excess_precision=false`,
`jax_default_matmul_precision=highest`, inspect installed per-op precision
signatures, and installed XLA CPU fast-math/fusion controls. No mechanism is
accepted merely because the option parses. It must change at least one
baseline output from its unflagged self AND codegen must demonstrate no FMA
contraction in the relevant field arithmetic. If a control does not change
baseline, explicitly call it vacuous. If no proven mechanism is found, only
G5-3 can decide numerical acceptance. A proven mechanism with a candidate
inequality fires G5-2; fallback does not excuse that failure.

G5-3 FLOAT64 REFERENCE DISTANCE. Additionally reported regardless of G5-2;
decisive fallback iff G5-2 is vacuous. Independent NumPy float64 full-field
Yee/CPML recurrence, not a float64 recompile of candidate or baseline. Use
identical initialized float32 coefficients/metrics promoted to float64,
float64 physical scalar constants and grid scalars, same zero initial state,
same source and boundary ordering. Evaluate the seven named fixtures at
200 completed steps. For each of six fields and 24 psi require BOTH
`abs(candidate-ref64) <= 1.05*abs(baseline-ref64)` at EVERY element and
`max(abs(candidate-ref64)) <= 1.05*max(abs(baseline-ref64))`.
No additive floor: an exactly correct baseline element requires exact
candidate agreement. Report violating counts and max errors separately.
Float64 is a numerical reference, not a claim of exact real arithmetic.

G5-4 PHYSICS INVARIANTS.
(a) Committed measurement class: `tests/oracle/test_pml_reflectivity_upml.py`
module docstring and `test_upml_reflection_floor_regression` explicitly give
CPML **-68.3 dB**, f0=2e9, freq_max=5e9, 8 layers, 250 steps. Use exactly
`tests/unit/boundaries/test_cpml.py::_reflection_db_vs_clean_reference`,
cpml_domain=0.06, GaussianPulse bandwidth=0.5; reference extent is
1.15*250*dt*C0. Baseline-only calibration: THREE fresh-process repetitions,
spread S=max(dB)-min(dB); freeze window W=2*S dB about baseline median.
If S=0, W=0; no rounding allowance or minimum epsilon. Append numerical
calibration and commit it BEFORE candidate application or comparison.
Candidate uses the same oracle and must satisfy |candidate_dB-median|<=W.
(b) PEC-closed null control: same 12-cell uniform domain, dx=1mm,
cpml_layers=0, 200 steps, same center pulse; all six fields at every step
must be bit-identical, as must the energy history computed from them.
No absorber is expected to produce decay; this is an equality control.
(c) mixed8: x_lo PEC, y_hi PMC; x_hi=4,y_lo=6,z_lo=4,z_hi=8.
All eight psi arrays for x_lo/y_hi must be exactly zero in both at 200
steps; absorbing-face psi must meet the applicable G5-2 or G5-3 rule.

G5-5 DIVERGENCE-GROWTH WITNESS. All seven fixtures, steps 1..200,
unflagged float32. For this sensitivity experiment only, initialize center
ez to 1.0 in both baseline and candidate (all other fields/psi zero), so
one ulp has a normal, nonzero scale. Perturbed baseline changes that one
cell to nextafter(float32(1),+inf)=1+2^-23; delta=1.1920928955078125e-7.
Use the same Gaussian pulse thereafter. For every step and each of six
fields, max|candidate-baseline| <= max|perturbed_baseline-baseline|, with
NO tolerance or floor. Archive all 200-point curves; any excess fires STOP.
This seeded witness is separate from unchanged zero-initialized G4 fixtures.

A fired gate stops candidate development and GPU timing; record failure,
never tune a candidate/window away. Complete requested reporting where
possible, labeling unexecuted gates. Restore production after rejection.

## Performance falsifier and regression

Only after numerical acceptance and G5-4/G5-5 pass: ONE GPU ladder (one
resubmission only for run failure), same G1b YAML/harness/card: bare-slow and
nu-uniform, 300^3 L=0/4/8/16 and 400^3 L=8. Three timing windows, median,
max-minus-min spread. Before: `results/w8b_nu_kernel_ablation_4090.json`.
At 300^3 L=8 BOTH lanes must improve rate by MORE THAN twice the larger
before/after spread; report same predicate for L=4/16. L=0 rate changes
must be <= twice the larger spread. This preserves G4's rate-based test
(the G4 prose's subtraction sign conflicts with its speedup intent; higher
Mcells/s is faster). Failure stops the lane and records whole-array
structure was not demonstrated to be the cost. No missing 400^3 baseline
substitution. Authorized rsync, SHA staging, VESSL submission from home,
polling, mandatory harvest/deletion and JSON destination follow user command.
No push, no new PR. All commits end with the supplied co-author trailer.

Before and after: requested full unit/contracts selection (not gpu, slow,
slow_physics), no cacheprovider, pinned PYTHONPATH/Python; ruff E,F,W with
E501,F401,E741,E731,E701,E702,E402 ignored. PYTHONDONTWRITEBYTECODE=1 and
ruff --no-cache avoid writes in the externally located tool environment.
Do not chase the three named pre-existing slow_physics oracle failures.

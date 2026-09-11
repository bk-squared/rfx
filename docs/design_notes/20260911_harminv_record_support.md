# Harminv: preserve poles through finite-record decimation (#872)

Status: implemented following PI approval on 2026-09-11. The scope is reliable
mode identification and frequency/decay assessment for the existing crossvals,
not calibrated modal energy, radiation power or S-parameters. Joint amplitude
fitting repairs the existing noisy dominant-mode regression exposed by the
preprocessing change. Physics-consumer verification is recorded below.

## Mathematical contract

For a sampled exponential `A z^n`, convolution on the interior of a finite
record gives `A z^n sum_l h_l z^-l`. A nonzero filter response changes the
coefficient, not the pole. Zero padding at either end violates that identity.
The estimator must therefore use only outputs whose full FIR support lies
inside the input record.

Each stage explicitly uses a symmetric FIR of order `20*q`, with half-support
`10*q` input samples. Output `k` is centered at input `k*q`; the valid outputs
are exactly `k = 10 .. ceil(N/q)-11`. Exclude ten outputs at each end, at each
stage. A stage is omitted if fewer than the core's existing ten-sample minimum
would remain or if its Hankel matrix cannot preserve the requested pole
capacity. For `N` samples, `L = max(4, min(int(N*p), N-2))` and `M = N-L`;
a `K`-pole model requires at least `K` rows and columns. Preserve
`min(max_modes+1, L_original, M_original)` singular directions at each stage.
The extra direction allows the relative rank cut to distinguish a requested
rank from a saturated matrix. If the original record is already too small,
preserve its available dimensions; this cannot invent missing information.
The same shape helper is used by planning and the actual SVD.

At the default `p=0.33, max_modes=50`, this requires 155 retained samples when
the original record has sufficient capacity. A rejected factor at most 13
then leaves at most 2,262 inputs for the core. Other requested pencil shapes
and ranks change that bound. The historical four-column/two-row clamp at
extreme fractions remains effective, so an impossible 51-direction request
does not force every such record into a raw SVD. Capacity is a dimensional
safeguard, not a frequency/Q accuracy or noise-identifiability guarantee.

Keep the caller's dimensionless `sv_threshold` relative to the largest singular
value. Remove the implicit `sqrt(target_factor)` multiplier. This preserves
the stated rank-cut parameter, **not** a universal noise-detection confidence
level. Singular values also depend on mode separation, damping, and record
length; filtering and truncation can change their ratios.

The usable pole-fit duration is the time between its first and last retained
sample. `harminv_record_duration` shares the estimator's sampling plan. cv02
uses it for Q admission/window inputs and for its record planner. The planner
adds the discarded interval back when asking for a longer raw run. Q cutoffs,
window formulas, and matching policy are unchanged. A shorter usable interval
can change the numerical window produced by the existing formula. The per-mode
settling flag uses the same retained span; the measured whole-signal end/peak
continues to use its full raw span, printed separately with its original offset.

## Joint amplitudes and checks

Fit the original input as `y = Phi c`, with `Phi[n,k] = exp(s_k*n*dt)` and all
finite candidate poles included before frequency/Q reporting filters. The old
independent projections replaced the full Gram matrix by its diagonal and
therefore included other modes' cross terms in each reported coefficient.

The implementation uses one `lstsq` on normalized columns. Growing exponentials
are anchored at the last sample to bound basis magnitudes; coefficients are
then transformed back to the original input's first sample. Real input keeps
both conjugates in the fit, and reports the positive-frequency member with a
deterministic phase. A real cosine's complex coefficient is half its peak
amplitude. The existing 1% reporting merge is unchanged.

The preliminary preprocessing-only candidate had **24 passed, 1 failed** in
the core tests. On the existing 40 dB peak-SNR fixture, independent projection
reported a spurious 8.367 GHz component at amplitude 0.962, ahead of the actual
9.3 GHz component at 0.547. The initial diagnostic fitted only the returned
candidates and conjugates (0.0116/0.502, condition number 12.1); the implemented
fit includes *all* finite candidates and gives **0.0232/0.5012**. The original
noise test now passes unchanged. Low-amplitude noise candidates still exist;
this is not a universal false-mode rejection or confidence claim.

The synthetic tests cover exact damped poles, a 0.003-amplitude weak component,
complex phase at two input origins, real conjugates, signal scaling, and a
strong unreported pole that must still participate in the fit. The 1e-7 bar
separates arithmetic error from bias on these well-conditioned inputs; it is
not a bound on noisy physical records. A nearly coincident-pole test explicitly
shows that small residuals can coexist with noise-sensitive coefficients.
A growing-pole check catches exponential/norm overflow without changing the
reported origin. Core tests and existing noise tests pass; no old tolerance
was relaxed. cv02 duration wiring and existing judge checks also pass.

## Actual FDTD records

Baseline source: `013fc06a2205a82d88712e19b79bad1f20e6cb29`.
`scripts/diagnostics/harminv_record_capture.py` was run against a disposable
`git archive` checkout, with four CPU cores and one BLAS thread. It saves each
actual estimator input once. Both revisions then read the same input bytes
through `scripts/diagnostics/harminv_record_replay.py`.

Inputs and their hashes/versions are under
`tests/fixtures/harminv_decimation/{cv02,cv24-uniform,cv24-single_band,cv24-metric_defect}/`.
The capture JSON's empty cv24 `measured` fields are intentional: capture skips
the estimator and energy audit; these are **not passing crossval artifacts**.
Original committed crossval evidence was not overwritten.

cv02 runs its existing Meep-absent ladder (x64), producing 5,500, 22,501 and
46,868 input samples; it exits 2, external comparison inconclusive. Three
in-band ring modes persist after preprocessing changes. For example the
35 THz mode's Q on the two longer records changes from 87.25/87.87 to
83.91/83.91. This is improved record stability, not an external Q-accuracy
claim. Existing in-band admission still excludes the unstable 60 THz poles.

cv24 (float32 FDTD): all seven declared modes occur in each full/A/B window.
The final joint fit and the case's existing exact discrete-lattice oracle give:

| Record / estimator | Maximum lattice residual | Maximum A/B scatter |
|---|---:|---:|
| Uniform, baseline automatic | 9.8987 ppm | 22.4657 ppm |
| Uniform, final automatic | 0.4619 ppm | 0.2294 ppm |
| Uniform, final undecimated control | 0.4762 ppm | 0.1154 ppm |
| Graded, baseline automatic | 10.2386 ppm | 23.6949 ppm |
| Graded, final automatic | 0.0171 ppm | 0.2132 ppm |
| Graded, final undecimated control | 0.0788 ppm | 0.1080 ppm |
| Swapped-metric falsifier, final automatic | **5545.7482 ppm** | 0.0699 ppm |

The maximum signed lattice residuals supporting the three final automatic
rows are resolved by the numeric-provenance contract from the retained reports:
`docs/research_notes/issue872/cv24-uniform-joint-adjudication.json::variants.candidate_auto.residual_ppm.TM111 = -0.4619`,
`docs/research_notes/issue872/cv24-single_band-joint-adjudication.json::variants.candidate_auto.residual_ppm.TE201 = -0.0171`,
and `docs/research_notes/issue872/cv24-metric_defect-joint-adjudication.json::variants.candidate_auto.residual_ppm.TE102 = 5545.7482`
(each stored in ppm).

The falsifier is a fresh run of cv24's existing metric swap on the same
single-band profile. It still has seven stationary modes, but **fails the
unchanged 4 ppm lattice gate** against the intended metric. Thus improving the
measurement does not make the wrong FDTD operator pass. The three retained-
record regressions complete in about 3 s without any FDTD solve. The baseline
automatic estimator fails the two healthy-record stationarity checks.

The original preprocessing-only reports remain under
`docs/research_notes/issue872/`; final reports are `*-joint.json`, with
`*-joint-adjudication.json` produced by
`scripts/diagnostics/harminv_cavity_adjudication.py`. Replay reuses baseline
results only after matching source hashes and the full input metadata.
Both automatic and explicitly undecimated paths were evaluated. No lattice,
mode-count or stationarity gate was relaxed. These sub-ppm figures quantify
agreement with the **discrete lattice**, not absolute continuum RF accuracy.

## Integration and external checks

- Core estimator, coefficient/phase and existing noise checks pass. The
  existing consumer/contract selection reports 113 passed (14 slow tests
  initially deselected); the separate CPU physics/convergence selection,
  including those 14, reports **21 passed** in 60.50 s. This includes the
  lossy-cavity Q oracle and live cv14 geometry/permittivity falsifiers.
- The full two-arm patch integration reports **8 passed** in 70.23 s on
  VESSL RTX 4090 run **369367260362**, source
  `e3eb58ab80bb71dcfe54cc62c7e0a3288c3c41a8`. Geometry, 200-period records,
  parity identification and all thresholds are unchanged. The CPU attempt
  was deliberately interrupted after over 30 minutes before a patch verdict;
  it is not counted as a completed CPU patch test. The GPU inputs, source
  receipt, complete pytest log and both estimator replays are retained under
  `patch-gpu-369367260362/`. The older GPU image uses JAX 0.4.33 / NumPy 1.26;
  the CPU and Meep checks use JAX 0.6.2. This is checked compatibility, not
  a same-runtime CPU/GPU bit-identity claim.
- A local Meep 1.17.1 environment failed the reference count gate on **both**
  baseline and candidate. Those refusals are retained in `cv02-meep117/`.
  They were not used as successful external evidence.
- A fresh isolated **Meep 1.34.0**, NumPy 1.26.4, SciPy 1.15.2, JAX 0.6.2
  environment ran the entire cv02 script on baseline `013fc06a` and candidate
  `e3eb58ab`. **Both processes returned 0**, and each newly written record's
  commit matches its tested checkout. All three Meep modes are assigned;
  maximum frequency error is **0.574989% -> 0.566633%**, within the unchanged
  5% gate. The candidate reports an additional 0.202687 c/a pole outside the
  judge's [0.1, 0.2] band; it is not presented as new validated mode coverage.

Q interpretation remains deliberately narrow. Only **two of the three**
reference modes pass the existing record-length admission rule. The first
mode still differs substantially (Meep Q=150.884, candidate Q=83.738) and
passes the existing loose Q window. This change does **not** resolve or
endorse the uncertainty-policy questions tracked by #907/#945. The corrected
usable record is 226.257 Meep units rather than the old raw-input 260.980;
window formulas and constants are unchanged, while their duration input is
now the actual fit span.

Fresh Meep reports, input traces, full process logs, source commits and
package receipts are under `cv02-meep134/`. The existing canonical crossval
artifacts remain their dated historical records; these new measurements are
retained as this change's before/after evidence.

Reproduction helpers:

- `harminv_record_capture.py`: one capture in a disposable pinned checkout.
- `harminv_record_replay.py`: old/new auto and explicit-no-decimation replay;
  optional baseline reuse is guarded by source and input hashes.
- `harminv_cavity_adjudication.py`: the existing cv24 gates applied to a replay.
- `cv02-meep134/driver.py`: execute the full case in each clean checkout and
  retain its actual estimator inputs; use the recorded conda environment
  plus `jax[cpu]==0.6.2` and a fresh output directory.

All three VESSL runs (two bootstrap failures before any solve and the final
successful test) were deleted only after their full provider logs were
backed up and hashed under `docs/research_notes/vessl_logs/`. No running
external experiment was terminated.

## CI follow-up: short-record capacity correction

The PI approved the general capacity-preserving stop condition on 2026-09-11.
CI had exposed a stage1 NU cavity failure: 1,846 raw input samples became 18
after two FIR stages, leaving only five pencil columns. The preceding stage
had ten singular values above the same relative rank cutoff. The old
minimum-sample check therefore allowed a saturated model matrix.

The new plan stops after factor 9, retaining 186 samples and 61 columns.
The unchanged 0.03% continuum gate now passes at
`docs/research_notes/issue872/capacity-impact/report.json::records[52].after.continuum_error_pct = 0.0227`
(percent units), versus 0.0827% with the earlier crop-only plan. A fresh full
stage1 FDTD gate also passes. Its old 3.5% test-side assertion was stale; the
test now names the same existing 0.03% constant already enforced by the script.
The original failing input and controls remain in `stage1-short-record/`.

A synthetic 12-pole check isolates model capacity: every pole lies below
even the earlier plan's Nyquist frequency. The old five-column plan fails;
the new plan recovers the in-band poles and decay rates at the unchanged
1e-7 arithmetic check. Actual SVD-shape spies verify capacity and duration
with multiple non-default pencil/rank settings. Extreme clamped fractions
and originally insufficient records retain their existing dimension limits.

The impact replay checks 53 actual estimator inputs in the same environment:
30 cv02/cv24 inputs, two Meep-version inputs, 20 fed/unfed GPU probe traces,
and the failing stage1 input. The first
`docs/research_notes/issue872/capacity-impact/report.json::unchanged_records = 52`
have identical plans and **exactly identical mode outputs** before/after the
capacity correction; only stage1 changes. Source and input hashes, both mode
lists and a reproduction script are retained in `capacity-impact/`. This
same-input check justifies retaining the earlier Meep/GPU integration evidence
at its recorded commit; those FDTD runs were not repeated for this safeguard.

The cv02 advisory report's Q-observability flag now uses retained analysis
time, while its raw measured end/peak and offset remain unchanged (43
consumer/judge checks passed). Numeric-provenance classification now covers
this note and its cited values. The combined estimator, retained-record,
stage1 and cv02 selection passed 93 checks; the final 32 decimation checks
also include the below-Nyquist synthetic falsifier.

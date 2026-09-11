# Harminv: preserve poles through finite-record decimation (#872)

Status: two preprocessing changes approved by the PI on 2026-09-11 and
implemented locally. **Not ready to merge:** an existing noisy-signal test
exposes the independent amplitude projection's cross terms. Joint amplitude
fitting has been proposed to the PI; it has not been applied.

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
would remain. Since every stage factor is at most 13, stopping leaves fewer
than 390 samples for the core; this fallback does not send an arbitrarily
large raw record into the SVD.

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
can change the numerical window produced by the existing formula.

## Falsifiers and observed results

Clean single real/complex exponentials and a pair with amplitude ratio 0.003
are checked against their known poles, at multiple record lengths and signal
scales. The 1e-7 relative frequency/Q test bar separates arithmetic error on
these well-conditioned fixtures from filter bias; it is not a bound on noisy
physical records. Small-noise pairs additionally test false-mode counts.

The existing estimator/decimation tests plus these cases produced **24 passed,
1 failed**. The failure is the existing 40 dB peak-SNR dominant-mode test:
the new rank cut admits noise poles, and independent projection reports an
8.367 GHz pole at amplitude 0.962, ahead of the actual 9.3 GHz pole at 0.547.
Holding those poles fixed and fitting their amplitudes jointly gives 0.0116
and 0.502 respectively (basis condition number 12.1). This is diagnostic
evidence for a needed amplitude repair, not an implemented repair. The
original test has not been weakened or removed.

cv02's consumer-duration wiring and existing judge tests: **42 passed**.

## Actual FDTD records

Baseline source: `013fc06a2205a82d88712e19b79bad1f20e6cb29`.
`scripts/diagnostics/harminv_record_capture.py` was run against a disposable
`git archive` checkout, with four CPU cores and one BLAS thread. It saves each
actual estimator input once. Both revisions then read the same input bytes
through `scripts/diagnostics/harminv_record_replay.py`.

Inputs and their hashes/versions are under
`tests/fixtures/harminv_decimation/{cv02,cv24-uniform,cv24-single_band}/`.
The capture JSON's empty cv24 `measured` fields are intentional: capture skips
the estimator and energy audit; these are **not passing crossval artifacts**.
Original committed crossval evidence was not overwritten.

cv02 runs its existing Meep-absent ladder (x64), producing 5,500, 22,501 and
46,868 input samples; it exits 2, external comparison inconclusive. Three
in-band ring modes persist after preprocessing changes. For example the
35 THz mode's Q on the two longer records changes from 87.25/87.87 to
83.91/83.91. This is improved record stability, not an external Q-accuracy
claim. Existing in-band admission still excludes the unstable 60 THz poles.

cv24 uniform (float32 FDTD): all seven declared modes occur in each of the
full/A/B windows. Reusing the case's existing mode identification and exact
discrete lattice oracle gives:

| Estimator | Maximum lattice residual | Maximum A/B scatter |
|---|---:|---:|
| Baseline automatic | 9.8987 ppm | 22.4657 ppm |
| Candidate automatic, both preprocessing changes | 0.4430 ppm | 0.2294 ppm |
| Baseline, undecimated control | 0.4762 ppm | 0.1154 ppm |

Detailed preprocessing-only reports and the noisy-amplitude diagnosis are
under `docs/research_notes/issue872/`. The graded single-band arm also preserves all seven modes. Its maximum
lattice residual changes from 10.2386 ppm to 0.0171 ppm, and A/B scatter
from 23.6949 ppm to 0.2132 ppm (undecimated control: 0.0788/0.1080 ppm).
The new retained-record regression passes both arms in 2.32 s. Replacing
its estimator with the baseline makes both tests fail at the existing
stationarity gate; the complete archived replay also exceeds the lattice gate. No lattice, mode-count or stationarity gate was relaxed.

## Remaining work

- Decide/apply the amplitude repair and rerun the unchanged noise test.
- Verify conjugate-pair handling and numerical conditioning of that repair.
- Recheck affected actual records and normal estimator consumers before
  claiming #872 complete. cv24's explicit `decimate=False` remains a control.
- Existing Q-gate policy questions (#907/#945) remain separate; this change
  does not make a noisy, driven, or insufficiently resolved record trustworthy.

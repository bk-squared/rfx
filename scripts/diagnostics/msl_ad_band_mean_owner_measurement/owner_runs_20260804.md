# MSL AD gate objective replacement (#530/#515) — owner-platform runs, 2026-08-04

Tracked evidence for the band-mean `|S21|^2` objective replacement (issue #530)
and the #515 AD smoke rebuild. Producer: `scripts/diagnostics/msl_ad_band_mean_s21_owner_measurement.py`
run via `scripts/vessl_msl_ad_band_mean_owner_measurement.yaml` on gpu-rtx4090
(remilab-c0). Both runs clone the PR branch fresh inside the job (the mounted
primary checkout stays on `main`) — see that YAML's comments for why.

Per the #548 standard: this file exists because the raw run logs live only
under the primary checkout's gitignored `.omx/msl-ad-band-mean-owner/` — this
is the tracked copy the gate's docstring (`tests/test_msl_ad_fd_converged.py`)
points readers at.

## Run 1 — `369367251813`, standalone measurement only

- Branch `msl-ad-band-mean-s21-objective` @ `0acbfb54f16958813bcbd2e413b992cff036cb98`
- Started 2026-08-04T14:10:54Z, completed 2026-08-04T14:16:48Z
- Source log (primary checkout, gitignored): `.omx/msl-ad-band-mean-owner/20260804T141126Z-0acbfb54f16958813bcbd2e413b992cff036cb98/measurement.log`

```
rfx        : /tmp/rfx-clone/rfx/__init__.py
git SHA    : 0acbfb54f16958813bcbd2e413b992cff036cb98
jax        : 0.4.33.dev20241023+e3c6d6430   devices: [CudaDevice(id=0)]
fixture    : num_periods=20 n_freqs=8 (gate's h=0.001); plant_defect=True
grid       : (142, 54, 19)  n_steps=26226  checkpoint_segments=141

--- AD (float32, as shipped) ---
  loss = 0.99787211   g_ad = 1.602933e-03   (71.6s)
  float32 ULP of this loss = 5.9605e-08

--- AD (float32, PLANTED #483-CLASS DEFECT: eps_override frozen) ---
  loss = 0.99787259   g_ad(defect) = 0.000000e+00   (19.1s)
  (expect exactly 0.0 -- alpha never enters the traced computation)

--- central FD, float64 loss (correct function; unaffected by the planted defect above) ---
          h               g_fd   rel_err (correct AD)    rel_err (defect AD)   signal (ULP)      s
    3.0e-04   1.6267607988e-03                 0.0146                 1.0000       8.79e+09   47.4
    1.0e-03   1.6071476933e-03                 0.0026                 1.0000        2.9e+10   41.3
    2.0e-03   1.6034690085e-03                 0.0003                 1.0000       5.78e+10   41.8
    5.0e-03   1.6012954754e-03                 0.0010                 1.0000       1.44e+11   41.9
    1.0e-02   1.6026462683e-03                 0.0002                 1.0000       2.89e+11   42.0

  g_ad (correct)       : 1.6029329272e-03
  g_ad (planted defect): 0.0000000000e+00
  f64 FD spread over h : 1.583%
  rel_err at gate's h=0.001 (correct AD) : 0.00262
  rel_err at gate's h=0.001 (defect AD)  : 1.00000
  FD resolving power at gate's h              : 2.9e+10 ULP

=== SUMMARY (for tests._gate_policy.gate_from_envelope) ===
  measured envelope (rel_err at gate's h, correct AD) = 0.0026
  falsifier rel_err (planted #483-class defect)       = 1.0000
  -> the gate must RED on this with margin over whatever threshold gate_from_envelope derives above.
```

## Run 2 — `369367251827`, measurement + the ACTUAL pytest gate

- Branch `msl-ad-band-mean-s21-objective` @ `526831b30d674880d723990b4521b2f438648199`
  (docstring/threshold-derivation commit, on top of run 1's commit)
- Started 2026-08-04T14:24:12Z, completed 2026-08-04T14:32:30Z
- Source logs (primary checkout, gitignored):
  `.omx/msl-ad-band-mean-owner/20260804T142444Z-526831b30d674880d723990b4521b2f438648199/{measurement,gate_pytest}.log`

The standalone-measurement half reproduced Run 1's numbers exactly (same
fixture, same commit's objective code — the docstring/threshold-derivation
commit did not touch the objective itself):

```
  loss = 0.99787211   g_ad = 1.602933e-03   (71.5s)
  float32 ULP of this loss = 5.9605e-08
  loss = 0.99787259   g_ad(defect) = 0.000000e+00   (18.8s)

          h               g_fd   rel_err (correct AD)    rel_err (defect AD)   signal (ULP)      s
    3.0e-04   1.6267607988e-03                 0.0146                 1.0000       8.79e+09   47.7
    1.0e-03   1.6071476933e-03                 0.0026                 1.0000        2.9e+10   41.6
    2.0e-03   1.6034690085e-03                 0.0003                 1.0000       5.78e+10   41.9
    5.0e-03   1.6012954754e-03                 0.0010                 1.0000       1.44e+11   41.9
    1.0e-02   1.6026462683e-03                 0.0002                 1.0000       2.89e+11   41.5
```

**The authoritative part of this run — the actual pytest gate file**, not the
standalone diagnostic replica, run against the same clone:

```
$ python -u -m pytest tests/test_msl_ad_fd_converged.py -m "" -q -ra -s

[MSL-FD-TIGHT] n_steps=26226, checkpoint_segments=141 (~sqrt=161.9)

[MSL-FD-TIGHT] forward |S| range: [0.0080, 1.0000]
[MSL-FD-TIGHT] loss = 9.978721e-01
[MSL-FD-TIGHT] g_ad = 1.602933e-03  (AD wall-time: 66.4s)
[MSL-FD-TIGHT] g_fd = 1.607148e-03  (FD wall-time: 47.4s, h=0.001)
[MSL-FD-TIGHT] FD reference resolving power: 2.9e+10 ULP of the loss (floor 1e+04)
[MSL-FD-TIGHT] rel_err = 0.0026 (threshold: 0.03, enforced below)
[MSL-FD-TIGHT] sign agreement: g_ad=1.6029e-03 g_fd=1.6071e-03
[MSL-FD-TIGHT] total wall-time: 138.1s
[MSL-FD-TIGHT] num_periods=20, n_freqs=8
[MSL-FD-TIGHT] PASS
...
3 passed in 138.72s (0:02:18)
```

(`3 passed` = `test_msl_ad_fd_converged_tight`,
`test_comparator_floor_rejects_the_f32_reference_that_caused_527`,
`test_fd_ulp_span_is_dtype_sensitive_not_container_sensitive`.)

## Derived threshold

`tests._gate_policy.gate_from_envelope(0.0146, quantum=100) == 0.03` — see
`tests/test_msl_ad_fd_converged.py`'s `_REL_ERR_THRESHOLD` comment and the
"GATE REBUILT" / "THRESHOLD DERIVATION" sections of
`test_msl_ad_fd_converged_tight`'s docstring for the full derivation and why
the envelope input is the h-sweep's worst point (0.0146) rather than only the
gate's own h=1e-3 value (0.0026).

## What these runs do NOT establish

Per adversarial review of PR #559: these runs measure AD-vs-FD numerical
agreement and the planted-defect falsifier's resolving power. They do NOT
measure which physical channel (guided-wavelength/beta shift vs. a
reference-plane mismatch against the wave-split's frozen Hammerstad-Jensen
`z0_hj`) dominates `d(loss)/d(alpha)` — that question was open; see
`tests/_msl_ad_objective.py` and `test_msl_ad_fd_converged_tight`'s
docstring.

**RESOLVED, issue #560, 2026-08-06**: a separate decisive probe
(`scripts/diagnostics/msl_ad_z0_anchor_probe.py`, run log
`scripts/diagnostics/msl_ad_z0_anchor_probe_run_20260806.md`, CPU/float32,
same fixture as this file) re-ran `jax.grad` of the identical objective
with the wave split's frozen analytic `z0_hj` anchor swapped for a frozen
per-port FITTED z0 (measured at alpha=1, held constant): `|g_ad|` dropped
from `1.602236e-03` (bit-identical across 2 repeats) to `6.885110e-05`
(the headline value, from an un-repeated run — the run's 2nd repeat was
killed by a background-task duration limit; the bit-identical-2/2 value
under a CLI-rounded anchor is `6.884444e-05`, agreeing to 4 significant
figures). By issue #560's own QUALITATIVE criterion ("drops toward the
FD-unresolvable floor"): the estimated FD signal for `g_b` at the gate's h
is only ~1.16 ULP of a float32 loss, below the 4.449 ULP issue #527
declared untrustworthy for the retired objective's comparator — g_b is
noise-floor by this repo's own established standard. (As a secondary,
self-declared check, this is ~23.3x — NOT a quote from #560, whose body
has no numeric threshold; see the probe script's docstring.) The
reference-plane mismatch (mechanism 2) is the dominant channel, not
beta/reflection physics. This is a channel-ATTRIBUTION finding, not a
numerical-agreement one: it does not change the `rel_err`/PASS verdict
recorded above, only the physical interpretation of the gradient's
magnitude. Separately, anchor B's own loss exceeded 1 (a passivity
violation, attributed to the raw unprojected `eps_override` channel, not
threatening the ratio) — evidence the fitted anchor is not self-evidently
"more correct," so whether `compute_msl_s_matrix`'s production wave split
should anchor on it is a SEPARATE, undecided design question this probe
does not settle.

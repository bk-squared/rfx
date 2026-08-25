# Phase-1a verdict — first free-form A/B/C on the 4090 (2026-08-25)

Jobs 369367256123/24/25 (B, C, A), 60 Adam iters, 2256-cell region,
J = |S21(6 GHz)|^2 alone. Outputs in `out_vessl/{A,B,C}/`.

| arm | soft J final | binarized J_hard(6 GHz) | fill | design (see png) |
|-----|--------------|--------------------------|------|------------------|
| A kottke-linear | 3.0e-2 (oscillating) | 2.8e-2 (-15.5 dB) | 0.43 | fragmented |
| B kottke+RAMP-sigma | 3.2e-4 (smooth, min 5e-6) | 1.0e-2 (-19.9 dB) | 0.41 | speckled; hard notches land at 5.5/7.0 GHz, NOT 6 |
| C legacy | 4.3e-4 (smooth) | 1.4e-3 (-28.5 dB) | 0.53 | SOLID BLOCK adjacent to the line |
| oracle lambda/4 stub | — | 8.9e-4 (-30.5 dB) | — | selective notch |

## Three findings, in decreasing confidence

1. **Damped gray fixes the soft-model traversal (B vs A), as probe-01
   predicted.** A's trajectory oscillates exactly as the measured rugged
   Kottke landscape implies (0.0425 -> 0.008 -> 0.0156 -> 0.0034 -> 0.0111...);
   B descends smoothly two orders further on the identical geometry, budget,
   and init. The probe-directed remedy works where it was aimed.

2. **The objective was exploitable, and arm C found the exploit.** Minimizing
   |S21(f_t)|^2 alone admits the degenerate "brick on the line" solution:
   C's design is a solid metal/absorber slab spanning the region next to the
   through-line, and its band response is a ~-28 dB BROADBAND blocker with no
   6-GHz selectivity (|S21| flat -28..-30 dB across 4.5-6.5 GHz). Its "win"
   is an artifact of the objective, not evidence for the legacy path.
   Phase-1b must optimize notch SELECTIVITY (target suppression + passband
   preservation, normalized per-frequency by the empty-line reference to
   cancel the uncalibrated extractor scale).

3. **B exposes the second classic failure mode: soft->hard transfer.** Its
   speckled density binarizes into sub-wavelength fragments whose hard
   response is detuned (deep notches at 5.5 and 7.0 GHz, local MAX at 6 GHz).
   Filter r=2 + beta<=32 is too weak; Phase-1b raises the continuation
   (beta to 128) and the filter radius, which is the standard cure.

## Gate status

- sigma-effect gate (B): PASS (rel 0.81).
- AD-vs-FD at init: INCONCLUSIVE as implemented — rel errors ~0.9-1.7 with a
  single FD step (1e-2) on a short-window objective; the descent evidence
  (100x smooth reduction) says the gradients are descent-useful. Phase-1b
  reports a two-step FD sweep instead. Do not quote a gradient-accuracy
  number from phase-1a.

## Infra learned (recorded for reuse)

- No jax.jit around value_and_grad (rfx density filter kernel sizing).
- Concurrent VESSL jobs must pip-install from a container-local copy (pip
  writes egg-info/build into the mounted source; simultaneous jobs race).
- 60-iter arm at this size ~ 35-40 min wall on one 4090.

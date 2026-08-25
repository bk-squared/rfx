# Phase-1c verdict — free-form metal TO BEATS the classical stub (2026-08-25)

Jobs 369367256137-142 (B/C x uniform/low/stub inits, 150 Adam iters,
selectivity objective, W_PB=1). Verdict metric: passband-vs-notch contrast of
the BINARIZED design, hard-PEC Kottke evaluator, 30-period window, 9-freq
band; oracle = the analytic lambda/4 stub on the same evaluator (9.8 dB).

| run | contrast (dB) | J_hard(6 GHz) | fill | reading |
|-----|---------------|---------------|------|---------|
| **B stub-seeded** | **+21.5** | **-43.3 dB** | 0.12 | WINNER — widened-stub topology |
| C low-fill | +18.9 | -42.9 dB | 0.14 | free-form discovery, no seed |
| C stub-seeded | +9.0 | -34.0 dB | 0.13 | ~oracle |
| B uniform | -5.0 | -20.2 dB | 0.46 | trapped (blocking-ish) |
| B low-fill | -0.0 | -13.8 dB | 0.07 | collapsed to empty region |
| C uniform | (infra-killed at it=120, J=0.211 descending — rerun candidate) | | | |

## Headline

**The stub-seeded damped-gray arm produced a binarized, hard-PEC-verified
design with +21.5 dB passband-to-notch contrast — 2.2x the analytic
quarter-wave stub's 9.8 dB on the identical evaluator — and a clean V
response centered exactly on the 6.0 GHz target (-29.5 dB normalized).**
The design is physically interpretable: the optimizer widened the seeded
1-cell stub to a ~5-6-cell-wide, ~7-mm patch stub (lower Z0 -> deeper,
sharper notch) with a small junction foot. Separately, C low-fill reached
+18.9 dB from a random low-fill init with NO seed — free-form discovery
works, not just seed refinement.

This crosses the line the accepted T-MTT paper could not: there the gradient
*recovered* classical performance on dielectric problems and free-form metal
was the open limitation; here a free-form metal design *exceeds* the
classical reference on its own figure of merit, end-to-end through the
production solver path.

## What decided it (consistent with probes 0/1a/1b)

- Init dominates: both uniform-init runs failed (the gray-traversal barriers
  measured in probe-01 are real obstacles from a mid-density start); low-fill
  works for C, collapses for B (the damped gray's own absorption feeds the
  passband penalty, pushing density to zero); stub seeding puts B in the
  right basin where its smooth damped-gray gradients then excel.
- The damped-gray (B) mechanism delivers when started sensibly — its
  +21.5 dB is the best result of the whole campaign.

## Honest caveats (Phase-2 targets)

1. Passband insertion loss is still material: winner t_pb = 0.40 (~-8 dB
   mean over the passband set) vs a practical filter's ~-1 dB. W_PB=1
   underweights the passband; sweep W_PB (3, 10) for practical designs. The
   contrast win over the oracle stands on the defined metric.
2. Mesh transferability NOT yet checked — apply the accepted paper's own
   two-step practice: re-evaluate the binarized winner at dx/2 and
   re-optimize at production resolution if it degrades. (Dogfooding our own
   Sec. V-C recommendation.)
3. Same extractor caveats as before (uncalibrated absolute scale; normalized
   quantities and within-evaluator comparisons only).
4. AD-vs-FD composite-objective check still methodologically inconclusive
   (FD noise); needs a directional-derivative Richardson design.
5. Single geometry, single seed noise draw; C_uniform infra-killed mid-run.

## Phase-2 shortlist

W_PB sweep on B-stub and C-low; dx/2 transferability gate on the winner;
second geometry (e.g., bandpass iris or patch-feed match); gradcheck redesign;
then the upstream rfx core patch proposal (RAMP-sigma option in
`density_to_material_fields` + topology_optimize wiring) via the patch-file
workflow, with this campaign as the evidence pack.

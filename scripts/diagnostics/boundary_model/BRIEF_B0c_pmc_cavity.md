# Brief B0c for Codex — W2, the magnetic-wall cavity (measurement only)

Same rules as `/root/workspace/bk-workspace/.boundary-model/BRIEF_B0_facts.md` (Command safety, no
repository edits, no interpreting sentences). Own worktree `/root/workspace/bk-workspace/rfx-wt-bnd-b0c`
(detached origin/main; STOP if it exists); write only under `/root/workspace/bk-workspace/.boundary-model/B0c/`;
remove the worktree at the end. Reuse B0's VESSL job pattern (`.boundary-model/B0/job_runner.py`,
`prepare_jobs.py`) and its source-export method.

## The cavity (leader's specification; B0 recorded W2 as NOT MEASURED for lack of it)
A vacuum box a × b × h = 24 × 20 × 4 mm. Faces x_lo, x_hi: PMC. Faces y_lo, y_hi, z_lo, z_hi: PEC.
No absorber. Declared with `BoundarySpec(x=Boundary("pmc","pmc"), y="pec", z="pec")`.
z-invariant TM modes (Ez, Hx, Hy): Ez = cos(mπx/a') sin(nπy/b), f_mn = (c/2)·sqrt((m/a')² + (n/b)²),
where a' is the distance between the two REALIZED magnetic walls. With electric walls on x the modes
would be sin(mπx/a) and m = 0 would not exist.
- Mode (0,1): exists only with magnetic x walls; f = c/(2b) = 7.494811 GHz, independent of a'.
- Mode (1,1): locates the walls; f(a' = 24 mm) = 9.7547 GHz, f(a' = 23 mm) = 9.9321 GHz (compute both
  analytic values yourself from the formula and print them; these are the leader's hand numbers).
Source: Ez differentiated Gaussian at (7, 6, 2) mm, f0 = 8.5 GHz, fractional bandwidth 0.8; probe Ez at
(17, 13, 2) mm. Resonances by the repository's own Harminv / ring-down helper, as W1 did.

## Arms
dx ∈ {1, 0.5, 0.25} mm × entry point ∈ {`run()` on CPU, `forward()` on CPU, the vmap sweep on CPU,
the non-uniform lane with a constant `dz_profile`, `run()` on GPU (the baked fast path; confirm from the
trace that it was taken)}. The GPU arm and anything over ~10 CPU minutes on VESSL (remilab-c0,
gpu-rtx4090), each job once. Per arm: realized grid and pads, the wall operators the trace recorded
(as B0's matrix did), preflight text verbatim, f(0,1) and f(1,1) with Harminv's error and Q, and the
realized a' implied by f(1,1) (invert the formula with b = 20 mm; say so as "derived").

## Report
`B0c/REPORT.md` (under 60 lines): the table (entry × dx: f01, f11, derived a', operators); whether f(0,1)
appears in each arm; commands with last lines; run ids. `Conclusion: leader fills.`

## Addendum (leader, after B0c/REPORT.md)
The one-launch rule is waived ONCE for the GPU arm: submit the corrected run-gpu job (energy callback
fixed so the CPU backend is available, or the energy record taken another way that does not change the
kernel path — confirm from the trace that `update_he_fast` is still taken). Same three dx values.
Append the results to `B0c/REPORT.md` as a new section "GPU arm (relaunch)" with its run id; do not
edit the existing lines. Everything else as before.

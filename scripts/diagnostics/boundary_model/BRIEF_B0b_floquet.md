# Brief B0b for Codex — does a Floquet port at a scan angle solve the angle it reports? (measurement only)

Same rules as `/root/workspace/bk-workspace/.boundary-model/BRIEF_B0_facts.md` ("Command safety", no
repository edits, no interpreting sentences, `Conclusion: leader fills.`). A second job runs B0 in
parallel: use your OWN worktree `/root/workspace/bk-workspace/rfx-wt-bnd-b0b` (detached, origin/main;
STOP if it exists) and write only under `/root/workspace/bk-workspace/.boundary-model/B0b/`. Remove the
worktree at the end. Anything over ~10 CPU minutes goes to VESSL as B0 describes.

## FACT (leader read, origin/main)
- `add_floquet_port(scan_theta=…)` (`rfx/api/__init__.py:2641-2760`) sets the transverse axes periodic
  and stores the angle; `run()` injects ONE point soft source at the centre of the transverse plane
  (`rfx/runners/uniform.py:655-700`; `forward()` the same at `rfx/api/_execute.py:1981-1996`) and uses the
  angle in the extractor (`rfx/floquet.py:297-384`, wave impedance η0/cos θ). No Bloch phase reaches the
  boundary on that path (grep: `bloch` appears in the runners only for TFSF).
- `rfx/floquet.py:122-181` `apply_bloch_periodic_x` multiplies by the REAL part of the Bloch phase.
- `rfx/ris.py:301` calls the Floquet port with a scan angle.

## ASSUMPTION (measure)
- F1 At scan_theta ≠ 0 the Floquet-port S11 of a dielectric slab does not follow the oblique Fresnel
  reflection; at 0 it does.
- F2 `apply_bloch_periodic_x` is not called on any run path.

## Measure
1. F2 by grep/call-graph and by instrumenting a Floquet run (record whether the function runs).
2. A lossless dielectric slab (εr = 4, thickness a quarter guide wavelength at f0, or the geometry of
   `tests/oracle/test_oblique_fresnel_magnitude.py`), periodic unit cell, Floquet port, TE, at
   scan_theta ∈ {0, 15, 30, 45}°: |S11| and ∠S11 over the port's band against the analytic TE slab
   reflection AT THE ANGLE THE PORT REPORTS, and against the analytic reflection at normal incidence.
   Record the realized grid, pads, the periodic flags the kernel received, preflight text verbatim, the
   energy witness. Also the same slab through the oblique TFSF path at 30° (the repository's validated
   magnitude method) for comparison.
3. Which public entry points accept scan_theta ≠ 0 without a warning or finding (run, forward,
   `rfx/ris.py`'s entry); quote any warning text verbatim.

Report `B0b/REPORT.md` (under 80 lines): F1, F2 CONFIRMED / CONTRADICTED with the tables; commands and
last lines; run ids.

# Brief for Codex — third arm on the two MSL fixtures: trace through the absorber, NOT touching the outer wall

You measure and report NUMBERS. No interpretation, no cause, no verdict, no recommendation. If something
unexpected happens, stop that item, write down exactly what happened, continue.

## What and why
In `msl/` you ran cv06b and cv20 as committed (`baseline`) and with the trace Box declared past the domain
by (absorber cells + 1) cells on both x ends (`continued`). In `continued` the trace reaches the grid's
outermost cells, i.e. the absorber's outer wall. This third arm, `inset1`, continues the trace through
the absorber but stops it ONE cell short of the outer wall at each end.

## Command safety (hard rules) — same as before
- One state-changing action per command. Never `rm -rf`. Never delete, overwrite or edit an existing file.
  NEW files only, all under a NEW directory `/root/workspace/bk-workspace/.801-measure/msl_inset/`
  (plain `mkdir`; if it exists, STOP). You may import (read-only) from `msl/measure.py`, `msl/reduce.py`
  and the tree `src-main`; do not modify them — copy what you need into new files in `msl_inset/`.
- Do not touch any git repository or worktree, nor `src-main*/`, `dumps/`, `cont/`, `ports/`, `msl/`.
- VESSL yaml rules as before; copy resources/image/mount/env/pip line from `msl/vessl_cv20.yaml`. Launch
  each yaml ONCE, append run ids to `msl_inset/run_id.txt`, wait with short polls.

## FACT (verified by the leader from your own records)
- cv06b: grid (553, 280, 37), dx 63.5 um, x absorber indices [0, 7] and [545, 552]; baseline trace edges
  run from x index 8 to 542/543 (`msl/EDGES.md`).
- cv20: grid (297, 66, 45), dx 50 um (`msl/EDGES.md` has its absorber indices and edge extents).
- In `continued`, edges appear in the absorber cells and `eps_r`, `mu_r`, `sigma` are unchanged
  (`msl/TABLE_display.md`, array comparison blocks).

## ASSUMPTION (unverified — check by read-back before any solve)
- A1: declaring the trace from x = -(absorber cells - 1)*dx to x = domain + (absorber cells - 1)*dx gives
  trace edges in every absorber cell EXCEPT the outermost one at each end (x index 0 and the last index
  carry no trace edge; index 1 and last-1 do). Rasterization may round differently: adjust the declared
  bound in steps of 0.5*dx until the read-back shows exactly that, and record the bound you ended on and
  every read-back along the way. If it cannot be achieved, stop and report.

## Steps
1. Dry read-back (no time stepping) for both fixtures: the x-directed and y-directed PEC edge flags along
   the trace centre row for x indices 0..(absorber+3) and the mirrored range at the hi end, for the
   `inset1` declaration. Save as `msl_inset/EDGES_inset1.md`.
2. One GPU job per fixture: run ONLY the `inset1` arm through the lane's public S-parameter function with
   the committed settings (same as your `msl/measure.py` did). Save the same files per run as in `msl/`.
3. Reduce against BOTH existing arms (`msl/<fixture>/baseline`, `msl/<fixture>/continued`):
   `msl_inset/TABLE.md` with, per fixture: ring-down witness per drive, max |S12 - S21|, raw and corrected
   max column power, max passivity correction for all three arms side by side; per S entry the max and
   mean over the band of the change in dB, linear magnitude and phase for inset1-vs-baseline and
   inset1-vs-continued; cv06b notch frequency (3-point) and depth for the three arms; cv20 fitted beta
   signed deviation (min, max, mean over the 9 gated bins, using the same analytic witness as before) and
   angle(S21) per gated bin for the three arms.
4. `msl_inset/REPORT.md`: commands (or the file holding them), run ids, file listing with sizes,
   `EDGES_inset1.md` and `TABLE.md` inline, whether A1 held and the bound you ended on, anything in FACT
   you found wrong. No interpretation. No recommendation. Do not post anything to GitHub.

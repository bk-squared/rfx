# Brief for Codex — scale the absorber's frequency-shift term (alpha) and measure what follows it

You measure and report NUMBERS. No interpretation, no cause, no verdict, no recommendation. If something
unexpected happens, stop that item, write down exactly what happened, continue.

## What and why
rfx's absorbing boundary (CPML) grades three profiles across its layers: sigma, kappa and alpha. alpha is
hard-coded as `alpha = 0.05 * (1.0 - rho)` in `rfx/boundaries/cpml.py` (line ~158 on main df08175c, in the
function that builds one face's profile). Two earlier observations are to be tested against a scaling of
that term, nothing else changed:
 (1) cv20 with the microstrip trace continued into the absorber ends with a slow low-frequency tail
     (`msl/cv20/continued/`), absent in the committed fixture (`msl/cv20/baseline/`);
 (2) the patch ring-down rig's arm n=2 / pad 10h / 4 layers grows (`dumps/n2_pad10_main/`, `result_main.json`).

## Command safety (hard rules) — same as before
- One state-changing action per command. Never `rm -rf`. Never delete, overwrite or edit an existing file.
  NEW files only, all under a NEW directory `/root/workspace/bk-workspace/.801-measure/alpha/`
  (plain `mkdir`; if it exists, STOP). Copy what you need from `msl/measure.py`, `msl_inset/measure.py`,
  `run_arms.py`, `dump_fields.py` into new files there; do not modify the originals.
- Do not touch any git repository or worktree, nor the exported trees `src-main*/`. The alpha scaling is
  applied from YOUR driver by wrapping/monkeypatching the profile builder at import time, never by editing
  the tree.
- VESSL yaml rules as before; copy resources/image/mount/env/pip line from `msl/vessl_cv20.yaml`. Launch each
  yaml ONCE, append run ids to `alpha/run_id.txt`, wait with short polls.

## FACT (verified by the leader, how in brackets)
- `src-main/rfx/boundaries/cpml.py` builds `sigma`, `kappa`, `alpha` per face from `rho` and returns them in
  a params tuple with fields `.kappa`, `.alpha` (and sigma-derived `b`, `c` coefficients); the literal is
  `alpha = 0.05 * (1.0 - rho)` [leader grepped lines 105–170].
- PR #1012 (tree `src-main-plus-1012`) changes this file; use ONLY `src-main` here.
- cv20 arms and their results exist in `msl/cv20/{baseline,continued}/` with `witness_series_00.npz`
  (`time_series`, 25177 steps × 10 probes, dt 9.53287434766e-14 s) [leader read them].

## ASSUMPTION (unverified — establish by read-back before any solve)
- A1: the update coefficients (`b`, `c` or however they are named) are computed FROM alpha inside the same
  builder, so scaling alpha where it is defined propagates to the coefficients the solve uses. If the
  coefficients are computed elsewhere from a stored alpha, find that place. Prove the scaling took effect
  by reading back, from the arguments the solve receives (or from the CPML state object handed to it), the
  alpha profile AND the coefficients for one face, for each factor; factor 1 must be bit-identical to an
  unpatched run.
- A2: factor 0 (alpha = 0 everywhere) does not divide by zero in the coefficient formula. If it does,
  record the error verbatim and use factor 0.01 instead, saying so.

## Steps
1. `alpha/readback.md`: for factors 0 (or 0.01), 0.25, 1, 4 — the x-lo face's alpha, kappa and
   coefficient arrays as received by the solve on cv20 (no time stepping), and a bit-identity check of
   factor 1 against an unpatched import.
2. GPU, cv20 with the trace continued to the wall (the `continued` declaration from `msl/measure.py`),
   factors 0 (or 0.01), 0.25, 1, 4: the lane's public S-parameter call with committed settings. Save the
   same files per run as `msl/` did, including `witness_series_*.npz`.
3. GPU, patch rig arm n=2 / pad 10h / 4 layers as committed (oracle `_build(n=2, pad_h=10, cpml=4)`, 150
   periods, the oracle's two metrics), same four factors. Save `result.json` and `time_series.npz` per run.
4. Reduce → `alpha/TABLE.md`, numbers only:
   - cv20, per factor: ring-down witness per drive; max |S12 − S21|; raw and corrected max column power;
     max passivity correction; fitted beta signed deviation min/max/mean over the 9 gated bins (same
     analytic witness as before); per S entry max and mean change in dB and phase against factor 1; and
     from `witness_series_00.npz`, probe 0 and probe 9: the envelope decay time over the last 60 % of the
     record (block maxima over 12 equal blocks, log-linear fit; give the fitted rate and its reciprocal in
     ns) and the three strongest spectral lines of that segment (Hann window, mean removed), in GHz.
   - patch arm, per factor: settling dB, the four per-probe late-time log rates, worst rate.
5. `alpha/REPORT.md`: commands (or the file holding them), run ids, file listing with sizes, `readback.md`
   and `TABLE.md` inline, which of A1/A2 held and what you did where they did not, anything in FACT you
   found wrong. No interpretation. No recommendation. Do not post anything to GitHub.

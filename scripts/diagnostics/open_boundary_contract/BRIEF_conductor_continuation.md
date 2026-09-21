# Brief for Codex — #801: continue the conductor through the absorber and re-run the growing arms

You do the measurement and report NUMBERS. You do NOT interpret them, name a cause, say whether a
hypothesis is confirmed, or recommend anything. The leader reads your numbers and decides. If anything
unexpected happens, stop and write down exactly what happened.

## The physics, so you know what you are measuring
A lossless patch on a grounded substrate sits inside absorbing (CPML) boundaries and is struck once.
The field can only ring down; on three arms rfx lets it grow. Field dumps showed the growing wave runs
along the +x and +y absorber faces, on a strip where the substrate (eps_r 3.38) continues into the
absorber but the ground plane does NOT (rfx continues dielectrics into the absorber pad, never
conductors). The test: put the conductor into the absorber cells too, change nothing else, re-run.
The pre-declaration with the reading rule is the last comment of `gh issue view 801 --comments`
(repo bk-squared/rfx). Read it.

## Command safety (hard rules)
- One state-changing action per command. No long `&&` chains mixing writes, moves and launches.
- Never `rm -rf`. Never delete or overwrite an existing file. NEW files only, all under
  `/root/workspace/bk-workspace/.801-measure/` (scripts and yaml at its top level with new names;
  outputs under a NEW directory `cont/`, created with a plain `mkdir`; if `cont/` exists, STOP).
- Do not touch any git repository or worktree. Do not edit anything under `src-main*/`, `src-1fc9e38f/`,
  `dumps/`, or any existing script (`run_arms.py`, `dump_fields.py`, `reduce_dumps.py`, `leader_replot.py`).
- VESSL yaml rules (`/root/workspace/bk-workspace/_configs/.claude/rules/vessl-jobs.md`): the `run:` block
  runs under sh (no bashisms), no heredocs, no pipe that can mask a Python failure, provenance fails loudly
  (print sha256 of your driver and of `src-main/rfx/boundaries/cpml.py`, `src-main/rfx/grid.py`), the
  SUBMITTER appends the run id to `run_id.txt` (append with `>>`, never rewrite the file).
- Copy cluster/preset/image/mount/env from `vessl_801_dump.yaml` (it ran successfully, including its
  pip line with `pytest`). `vessl run create -f <yaml>` prints only `Check your Run at: <url>`; the id is
  the last path element. Launch ONCE. Wait with short `vessl run read <id>` polls and `sleep 60`.
- Python for local work: `/root/workspace/bk-workspace/rfx/.venv/bin/python`.

## FACT (verified by the leader, how in brackets)
- `dump_fields.py` runs an arm on a tree and saves `final_state.npz` + `meta.json` (and three snapshots)
  under `dumps/<label>/`; `run_arms.py` shows the minimal build-run-score loop with the oracle's own
  `_build`, `_late_time_log_rate_per_step`, `_settling_db` [both ran on GPU; results on the issue].
- On `src-main`, arm n=2 / pad 10h / 4 layers, grid (125, 95, 41), absorber 4 cells per face: the assembled
  PEC mask on the ground row (z index 14) reads, along x at y=45, `[0 0 0 0 1 1 1 1]` at the lo end and
  `[1 1 1 0 0 0 0 0]` at the hi end; identical along y. `eps_r` at z=15 is 3.38 in every cell including all
  absorber cells [leader printed both from `sim._assemble_materials(grid)`; index 0 of the tuple is the
  materials (`.eps_r`), index 3 is the PEC mask].
- The oracle's builder declares the ground plane as
  `Box((nudge, nudge, z_gnd), (domx + nudge, domy + nudge, z_sub_lo))` with `nudge = -0.1*dx`, material
  "pec"; the substrate Box has the same lateral extents; the patch is a third Box
  [`tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py`, function `_build`].
- Known verdicts on main for the three growing arms: n2_pad10_cpml4 0.00 dB / +2.55e-3 per step;
  n2_pad0_cpml4 0.00 dB / +7.68e-4; n3_pad10_cpml6 0.00 dB / +5.84e-4 [leader's table on the issue].

## ASSUMPTION (unverified — check and report what you find; do not build on it silently)
- A1: a PEC Box declared beyond the domain (lateral lo = `-(pad+1)*dx`, hi = `dom + (pad+1)*dx`) is
  rasterized into the absorber cells, so the ground-row PEC mask becomes all ones along x and y.
  It may not be: coordinates outside the domain may be clipped, or preflight/assembly may refuse.
- A2: `Simulation.run` re-assembles materials internally, so a PEC mask edited after a manual
  `_assemble_materials` call is NOT what the solve uses. If A1 fails you need to find where the solve's
  PEC mask is produced and whether it can be set from outside without editing rfx.

## Steps
1. Write `cont_build_check.py`: for each of the three growing arms, build variant (a) — a copy of the
   oracle's `_build` in YOUR script (do not edit the oracle) whose ONLY change is the ground-plane Box's
   four lateral bounds, extended to `-(pad_cells+1)*dx` and `dom + (pad_cells+1)*dx` on x and y; z bounds,
   substrate, patch, source, probes, nudge all unchanged. No solve. Print for the baseline and for (a):
   grid shape, the ground-row PEC mask's first 8 and last 8 values along x (at mid y) and along y (at mid
   x), the count of PEC cells in absorber cells per face, `eps_r` first/last 8 at the substrate row, and
   any preflight/assembly exception verbatim. Run it locally on CPU. Save its stdout to
   `cont/build_check.txt` (shell redirection).
2. Decide the variant by this rule only (no judgement): if for all three arms (a)'s ground-row mask is
   all ones along both axes through every absorber cell, use (a) alone. Otherwise ALSO implement (b): make
   the solve itself use a PEC mask in which, on each lateral face, the absorber cells replicate the PEC
   mask of the nearest interior column (the last column that carries the declared conductor), for ALL z,
   exactly as the pad extension replicates `eps_r`. (b) must not edit any file under `src-main/`; do it by
   wrapping/monkeypatching from your driver, and PROVE it took effect by reading back the mask the solve
   used (or, if that is impossible, say so and stop). Record in `cont/variant_note.txt` which variant(s)
   you ran and the read-back evidence.
3. Write `cont_run.py` (may import from `dump_fields.py`/`run_arms.py` but not modify them): for each of
   the three growing arms on `src-main`, run the chosen variant(s) for the oracle's full `NUM_PERIODS`,
   score with the oracle's two metric functions, and save under `cont/<arm>_<variant>/`: `result.json`
   (arm, variant, grid, steps, the four per-probe rates, worst rate, settling dB, the ground-row mask
   first/last 8 along x and y AS USED) and `final_state.npz` (six field arrays, float32).
   Also run the n4_pad10_cpml8 gate-point arm with the same variant(s) as a control.
4. CPU smoke: run `cont_run.py` for ONE arm with a short record (`n_steps` about 200) into
   `cont/_cpu_smoke/`; list the files.
5. Write `vessl_801_cont.yaml`, check its run block with `sh -n`, launch once, append the run id to
   `run_id.txt`, wait, confirm every expected directory and file exists.
6. Write `cont_reduce.py` and run it locally: for each result, the same region energy fractions as
   `reduce_dumps.py` computes (interior; six face slabs; edge overlaps; corner overlaps), the max-|E| cell
   and its three distances, and the normalized 1-D energy profiles' first 8 and last 8 values along x and
   y. When you plot, clip the colour range to [max-8 decades, max] (see `leader_replot.py`); a range
   stretched by zero cells is useless. Write `cont/TABLE.md`: one row per (arm, variant) plus the four
   baseline rows copied from `dumps/TABLE.md` and the issue table, columns = settling dB, worst rate, the
   nine region fractions, max-|E| cell and distances. Numbers only.
7. Write `cont/REPORT.md`: every command run with its output (or the path of the file holding it), the
   run id, the file listing with sizes, `build_check.txt` inline, `TABLE.md` inline, which of A1/A2 held
   and what you did where they did not, anything in FACT you found wrong. No interpretation. No
   recommendation. Do not post anything to GitHub.

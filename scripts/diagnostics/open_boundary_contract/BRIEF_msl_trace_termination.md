# Brief for Codex — how a microstrip TRACE ends at an absorbing face, on two physically meaningful fixtures

You measure and report NUMBERS. No interpretation, no cause, no verdict, no recommendation. If something
unexpected happens, stop that item, write down exactly what happened, continue with the next item.

## The physics, so you know what you are measuring
A microstrip line that leaves the domain through an absorbing (CPML) face must keep its cross-section
inside the absorber: substrate, ground AND trace. rfx adds the absorber cells outside the declared domain
and continues dielectrics into them. In the committed MSL fixtures the ground is the z-lo PEC boundary
face (it spans the whole padded grid), the substrate is continued, and the TRACE is a zero-thickness PEC
Box (a "sheet") declared from x = 0 to x = domain length. Your earlier census (`ports/`) could not count
realized sheet edges, and its before/after MSL pair used a plumbing smoke fixture whose |S11| is 1e-9
(numerical noise), so it carries no information. This brief replaces it with two real fixtures.

## Command safety (hard rules) — same as before
- One state-changing action per command. Never `rm -rf`. Never delete, overwrite or edit an existing file.
  NEW files only, all under a NEW directory `/root/workspace/bk-workspace/.801-measure/msl/`
  (plain `mkdir`; if it exists, STOP). Scripts and yaml go inside `msl/`.
- Do not touch any git repository or worktree, nor `src-main*/`, `dumps/`, `cont/`, `ports/`. You MAY
  import from your own `ports/measure_common.py` etc. (read-only) and from the tree `src-main`.
- VESSL yaml rules as before (`/root/workspace/bk-workspace/_configs/.claude/rules/vessl-jobs.md`); copy
  cluster/preset/image/mount/env and the pip line from `ports/vessl_ports_msl.yaml`. Launch each yaml
  ONCE, append run ids to `msl/run_id.txt`, wait with short polls. Local Python:
  `/root/workspace/bk-workspace/rfx/.venv/bin/python` with `JAX_PLATFORMS=cpu`.

## FACT (verified by the leader, how in brackets)
- cv06b, `validation/crossval/06b_msl_notch_filter_uniform.py::_build_sim`: grid (553, 280, 37), absorber
  8 cells on x and y faces and z-hi, z-lo is a PEC boundary; conductor 1 is a Box with bounds
  x 0 → 34.0 mm (the full domain length), y 1.016 → 1.616 mm, z 0.254 mm (zero thickness), reaches x_lo and
  x_hi; conductor 2 is the stub; two `_msl_ports` [your own `ports/census.json`, read by the leader].
- cv20 is `validation/crossval/20_msl_phase_referee.py`; its rfx leg at dx = 50 um has grid (297, 66, 45)
  [issue #830's record]. It was NOT in your census; find its rfx builder.
- `rfx.simulation.run` receives `pec_mask` and `pec_edge_masks`; you already wrapped it to read them back
  (`cont/variant_note.txt`, `ports/port_run.py`) [leader read your records].

## ASSUMPTION (unverified — check, report, do not build on silently)
- A1: the realized PEC EDGE masks the solve receives (`pec_edge_masks`) contain no trace edges inside the
  x-lo and x-hi absorber cells on these two fixtures.
- A2: declaring the trace Box past the domain on x by (absorber cells + 1) cells puts trace edges into
  the absorber cells, and the MSL port builders / preflight accept it.
- A3: the two scripts expose a way to build and run ONLY the rfx leg (no openEMS). If a script can only be
  run as a whole, reproduce its rfx leg in your own script from its builder, with its committed settings,
  and say exactly which settings you copied.

## Part 1 — what the solve receives today, no time stepping  →  `msl/edges_today.json`, `msl/EDGES.md`
For cv06b and cv20, baseline as committed: read back the `pec_edge_masks` (and `pec_mask`) the solve
would receive (build + assemble exactly as the lane's S-parameter function does; if the only way is to
start the public call and capture the arguments of `rfx.simulation.run`, capture them and abort before
stepping). Report, along the trace's centre row at the trace's z plane: the x-directed and y-directed
PEC edge flags for the first (absorber+4) and last (absorber+4) x indices; the count of trace edges
inside each x absorber; the x index of the first and last trace edge; `eps_r` just below the trace over
the same x indices; and where each MSL port's source plane, reference plane and probes sit in x index.

## Part 2 — before / after on GPU  →  `msl/<fixture>/{baseline,continued}/`, `msl/TABLE.md`
For each fixture run the lane's public S-parameter function with the committed settings twice:
baseline, and `continued` where the ONLY change is the trace Box's x bounds declared past the domain by
(absorber cells + 1) cells on both ends (in a copy of the builder inside your script). Read back the
edge masks the solve received in both runs and record the same numbers as Part 1. If A2 fails (edges do
not appear, or the lane refuses), record it verbatim and do not improvise another way.
Save per run: complex S and frequencies (`s.npz`), every diagnostic on the result object (fitted Z0,
beta, `reliable`, `beta_railed`, `probe_clearance`, passivity/column power, ring-down witness, preflight
text verbatim), wall time. `msl/TABLE.md`, per fixture, numbers only:
  - per S entry: max and mean over the band of the change in |S| in dB AND in linear magnitude, and of the
    phase in degrees; |S11| and |S21| of both runs at every frequency bin (a plain two-column list);
  - cv06b: the notch frequency (bin of min |S21|, and a 3-point parabolic interpolation, say which) and the
    notch depth in dB, both runs, and the shift in percent;
  - cv20: fitted beta per gated bin in 3.0–4.5 GHz in both runs and its signed deviation in percent from
    the Hammerstad-Jensen beta the script itself uses (use the script's own analytic witness function and
    eps_eff; say which key or function), and angle(S21) per bin in both runs;
  - max column power, reciprocity |S12 − S21|, ring-down witness in both runs.

## Part 3 — report  →  `msl/REPORT.md`
Commands (or the file holding them), run ids, file listing with sizes, `EDGES.md` and `TABLE.md` inline,
which of A1–A3 held and what you did where they did not, anything in FACT you found wrong.
No interpretation. No recommendation. Do not post anything to GitHub.

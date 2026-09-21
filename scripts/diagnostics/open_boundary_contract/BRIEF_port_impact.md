# Brief for Codex — which port lanes have a conductor ending at an absorber, and what it does to their results

You do measurements and report NUMBERS. You do NOT interpret, name causes, judge importance, or
recommend. If anything unexpected happens, stop that item, write down exactly what happened, continue
with the next item.

## The physics, so you know what you are measuring
An absorbing boundary (CPML) assumes the structure at its face continues unchanged through it. rfx puts
the absorber cells OUTSIDE the declared domain and continues dielectrics into them, but NOT conductors:
a ground plane or a trace declared out to an absorbing face ends at the absorber's inner boundary (and
one node before it on hi faces). On a patch rig this let a wave run along the absorber face and grow;
with the ground plane continued through the absorber the same rig settles to -44 dB (records in
`/root/workspace/bk-workspace/.801-measure/cont/`). The question now: which OTHER committed structures
have a conductor that reaches an absorbing face, and how much do their port results change when the
conductor is continued through the absorber.

## Command safety (hard rules)
- One state-changing action per command. No long `&&` chains mixing writes, moves and launches.
- Never `rm -rf`. Never delete or overwrite an existing file. NEW files only, all under a NEW directory
  `/root/workspace/bk-workspace/.801-measure/ports/` (create with plain `mkdir`; if it exists, STOP).
  Scripts and yaml go inside `ports/` too.
- Do not touch any git repository or worktree. Do not edit anything under `src-main*/`, `src-1fc9e38f/`,
  `dumps/`, `cont/`, or any existing script. Import from the tree `src-main` (main df08175c) only.
- VESSL yaml rules: `/root/workspace/bk-workspace/_configs/.claude/rules/vessl-jobs.md` (sh not bash, no
  heredocs, no pipe that can mask a Python failure, provenance fails loudly with sha256 of your drivers and
  of `src-main/rfx/boundaries/cpml.py`, the submitter appends the run id to `ports/run_id.txt`). Copy
  cluster/preset/image/mount/env and the pip line from `vessl_801_cont.yaml`. Launch each yaml ONCE.
  Wait with short `vessl run read <id>` polls and `sleep 60`. Anything over ~10 CPU minutes goes to VESSL.
- Python for local work: `/root/workspace/bk-workspace/rfx/.venv/bin/python`, `JAX_PLATFORMS=cpu`.

## FACT (verified by the leader, how in brackets)
- In `cont/`, Codex's earlier run proved: a PEC VOLUME Box declared past the domain by (absorber cells + 1)
  cells is rasterized into the absorber cells, and `rfx.simulation.run` receives that mask
  (`cont/variant_note.txt`, read-back of the `pec_mask` and `pec_edge_masks` arguments) [leader read it and
  recomputed the results].
- `Simulation._assemble_materials(grid, pec_sheets=[], pec_wires=[])` returns a 7-tuple; index 0 has
  `.eps_r`, index 3 is the volume PEC mask; zero-thickness PEC Boxes and thin conductors come back as SHEETS
  in the `pec_sheets` collector and sub-cell wires in `pec_wires`, not in the mask
  [`src-main/rfx/api/_compile.py`, docstring of `_assemble_materials`; `src-main/rfx/fidelity.py` ~line 320
  shows a working call].
- `grid.face_pads` gives absorber cells per face as (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi); a face whose
  boundary is PEC/PMC has pad 0 [`src-main/rfx/grid.py`].
- `scripts/capture_example_fidelity_snapshot.py` in the tree exposes `iter_audited_variants()`, the list
  of 62 audited example/crossval variants with a builder each [used by PR #1136's sweep].

## ASSUMPTION (unverified — check, report what you find, do not build on it silently)
- A1: every audited variant can be built (Simulation object + grid + `_assemble_materials`) on CPU in
  seconds without solving. Some will refuse the uniform grid (non-uniform meshes): record them as such
  and use the non-uniform assembly path if one is obvious, else skip and list them.
- A2: a zero-thickness PEC Box (a SHEET) declared past the domain is also realized inside the absorber
  cells. Unknown. The read-back must look at the realized edge masks the solve receives
  (`pec_edge_masks` in `rfx.simulation.run`), not only the volume mask.
- A3: the MSL, coax and mixed S-parameter lanes build their own port geometry (feeds, reference planes,
  probes) relative to the domain; extending a conductor Box past the domain may collide with port
  construction or preflight. If a lane refuses, record the refusal verbatim; do not work around it.

## Part 1 — census, CPU only, no solve  →  `ports/census.json`, `ports/CENSUS.md`
For every audited variant (and additionally the fixtures of these tests if they are not among them:
`tests/unit/sparams/` MSL two-port fixtures, the cv06b and cv20 MSL builders under `validation/crossval/`,
the coax two-port builder, `compute_coax_msl_transition`'s fixture, one WR-90 waveguide two-port with a
dielectric slab): build it, then report per variant:
  - lane (which S-parameter function or port type it uses; "none" if it has no port), grid shape,
    face_pads, boundary type per face;
  - for every conductor entry (volume PEC Box, PEC sheet, thin conductor, wire): its declared bounds, and
    per lateral/longitudinal absorbing face: does the DECLARED span reach that face (within 1e-9 relative of
    the domain length, or beyond it); how many realized conductor cells/edges lie in the LAST interior
    plane next to that face; how many lie INSIDE that face's absorber cells;
  - for every dielectric entry: the same two counts using `eps_r != 1`.
  A variant is "affected" when some conductor's declared span reaches an absorbing face and it has zero
  realized conductor cells/edges inside that face's absorber. List affected variants per lane.

## Part 2 — before/after on affected lanes  →  `ports/<lane>/<fixture>/{baseline,continued}/…`, `ports/TABLE.md`
For each lane with an affected variant, take the SMALLEST affected fixture of that lane (fewest cells ×
steps), at most one fixture per lane, at most five lanes: MSL two-port, coax two-port, coax-to-MSL
transition or mixed, patch with a lumped/wire port, waveguide (only if the census says it is affected).
Run it twice through the lane's own public S-parameter function with its committed settings:
  - baseline: as committed;
  - continued: the ONLY change is that each affected conductor's bounds on the affected axis are declared
    past the domain by (absorber cells + 1) cells on the affected side(s). Do this in a copy of the
    builder inside your script, never by editing the tree. Print the realized conductor masks/edges in the
    absorber cells before solving and read back what the solve received, as in `cont/variant_note.txt`.
    If the continuation does not realize (A2) or the lane refuses (A3), record that and stop that lane.
Save per run: the complex S matrix and frequencies (`s.npz`), every diagnostic the result object carries
(Z0, beta, reliable flags, passivity or reciprocity figures, ring-down/energy witness, preflight text
verbatim), wall time. Then `ports/TABLE.md`, one row per (lane, fixture, quantity): for each S entry the
max and mean over the band of | |S|_cont − |S|_base | in dB and of the phase difference in degrees, the
frequency of each |S11| minimum and resonance in both runs and their difference in percent, max column
power in both runs, the lane's own witnesses in both runs. Numbers only.

## Part 3 — report  →  `ports/REPORT.md`
Commands run (or the path of a file holding them), run ids, file listing with sizes, `CENSUS.md` and
`TABLE.md` inline, which of A1–A3 held and what you did where they did not, anything in FACT you found
wrong. No interpretation. No recommendation. Do not post anything to GitHub.

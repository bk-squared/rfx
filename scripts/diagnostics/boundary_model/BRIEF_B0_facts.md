# Brief B0 for Codex — boundary model redesign: the facts, measured (no product change)

The leader is redesigning how rfx turns a declared boundary (six faces, `BoundarySpec`) into the
walls and absorbers each time-stepping kernel applies. You do NOT design anything and you do NOT
change repository files. You produce (1) an inventory and (2) measurements that the design note will
cite. Where a measurement contradicts a FACT below, report it; do not explain it.

## Command safety (hard rules)
- One state-changing action per command. Never `rm -rf`. No `git reset`, `git checkout <path>`,
  `git clean`, `git stash`, `--amend`, rebase, force-push, push, PR, or GitHub comment.
- Work in a new detached worktree: `git -C /root/workspace/bk-workspace/rfx fetch origin` then
  `git -C /root/workspace/bk-workspace/rfx worktree add --detach /root/workspace/bk-workspace/rfx-wt-bnd-b0 origin/main`
  (STOP if the path exists). Do not edit files in it; put every script and output under
  `/root/workspace/bk-workspace/.boundary-model/B0/` (create with `mkdir`). Remove the worktree with
  `git worktree remove` at the end.
- Python: `/root/workspace/bk-workspace/rfx/.venv/bin/python`, run with `PYTHONPATH` = the worktree.
  pytest with `-p no:cacheprovider`. Nothing over ~10 CPU minutes on this pod (it is shared, load ~30);
  anything longer goes to VESSL (cluster remilab-c0, preset gpu-rtx4090; rules in
  `/root/workspace/bk-workspace/_configs/.claude/rules/vessl-jobs.md`; do NOT run git inside a job —
  export the source with `git -C /root/workspace/bk-workspace/rfx archive --format=tar origin/main | tar -x --no-same-owner -C <dir>`
  and write the commit hash to a PROVENANCE.txt, as `/root/workspace/bk-workspace/.801-measure/B/`
  did). Launch each job once; record run ids.
- No interpreting sentences. Where a conclusion would go write `Conclusion: leader fills.`

## FACT (leader, file:line on origin/main 798ec64e)
- The declaration is per face (`rfx/boundaries/spec.py`); `Simulation.__init__` flattens it into
  `_boundary` ('cpml'/'upml', or 'pec' whenever no face absorbs), `_cpml_layers`, `_pec_faces`,
  `_periodic_axes` (`rfx/api/__init__.py:563-606, 699-704`).
- The step kernels take AXIS strings: `apply_pec(state, axes)` zeroes tangential E on both array ends
  of each named axis (`rfx/boundaries/pec.py:21-55`); `pec_axes=None` expands to every non-periodic axis
  (`rfx/simulation.py:998-1005`), including axes whose faces are PMC or absorbers. Face-level
  `apply_pec_faces` / `apply_pmc_faces` run as well (`rfx/simulation.py:1863-1867`, `:1765-1767`).
- `pec_axes` / `cpml_axes` are derived separately in: `rfx/runners/uniform.py:322-324, 588-589, 620-628`;
  `rfx/api/_execute.py:1441-1456, 1489-1496`; `rfx/vmap_sweep.py:865-880`; `rfx/sparams/waveguide.py:813, 954`;
  `rfx/rcs.py:508`; the grid builder restricts absorbers to the waveguide-port axes
  (`rfx/api/_compile.py:81-126`) and Floquet defaults to x-y periodic (`:66-79`).
- Independent step loops apply walls themselves: `rfx/simulation.py` (scan and the baked-coefficient
  fast path via `precompute_coeffs(pec_axes=...)`, `:2554`), `rfx/nonuniform.py`, `rfx/vmap_sweep.py:628`,
  `rfx/subgridding/jit_runner.py:1981, 2014`, `rfx/runners/_distributed_common.py:479-560`,
  `rfx/runners/distributed_nu.py`, `rfx/adi.py`, `rfx/probes/probes.py:1567-1624`, `rfx/sources/tfsf_2d.py`.

## ASSUMPTION (measure; do not build on silently)
- A1 The six entry points below do not all realize the same faces for the same declaration.
- A2 A PMC face is realized half a cell inside the declared face (H_tan zeroed at H index 0), not on it.
- A3 A waveguide-port run realizes the transverse faces as PEC whatever the declaration says.
- A4 A TFSF run realizes the transverse faces as periodic whatever the declaration says, and no
  preflight finding names that.
- A5 `run()` puts a PEC wall behind every absorber face; `forward()` puts none.

## Deliverable 1 — inventory (`B0/INVENTORY.md` + `B0/inventory.csv`)
Every site in `rfx/` (not tests) that reads or writes any of: `_boundary`, `_boundary_spec`,
`_cpml_layers`, `_pec_faces`, `_periodic_axes`, `_periodic_flags`, `cpml_axes`, `pec_axes`, `pec_faces`,
`pmc_faces`, `face_layers`, `pad_[xyz]*`, `axis_pads`, `face_pads`, `periodic` (as a boundary flag),
`bloch`, `absorber_type`, `conformal_faces`, and every call of `apply_pec`, `apply_pec_faces`,
`apply_pmc_faces`, `apply_pec_mask`-at-array-edge, `init_cpml`, `apply_cpml_[eh]`, `init_upml`,
`apply_upml_[eh]`, `precompute_coeffs`. One CSV row per site: file, line, symbol, category (exactly one
of DECLARE — builds the spec or a legacy view from input; DERIVE — computes one boundary flag from
others; APPLY — applies a wall/absorber/wrap inside a step loop; DECIDE — reads a flag to decide
something that is not a boundary, e.g. preflight, materials pad continuation, port or probe placement;
PASS — only forwards a flag to a call), and the kernel or lane it belongs to (uniform-scan, fast-path,
forward, nonuniform, vmap-sweep, subgridded, distributed-v1, distributed-v2, distributed-nu, adi, tfsf-aux,
waveguide-lane, rcs, floquet, probes-reference, preflight, materials, ports, probes, farfield, io,
other). `INVENTORY.md`: counts per category × lane, and the list of APPLY sites with what each applies.

## Deliverable 2 — realized-boundary matrix (`B0/MATRIX.md` + JSON)
For each ENTRY POINT: `run()` general path; `run(compute_s_params=True)` with one wire port (the
single-wire fast path, `rfx/runners/uniform.py` ~:840); `forward()`; the vmap sweep
(`rfx/vmap_sweep.py`, its public entry); the non-uniform lane (a constant `dz_profile`); the subgridded
lane if it accepts the case; a distributed lane on 2 emulated CPU devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=2`) if it accepts the case; `solver='adi'` if it
accepts the case — and for each BOUNDARY: all PEC; all CPML (8 layers); all UPML; x PMC / y,z PEC;
x PMC / y,z CPML; z_lo PEC with the rest CPML; x,y periodic with z CPML; a TFSF source with all six
faces declared CPML; a waveguide port along x with y,z declared CPML, then PMC, then PEC; Floquet ports
with nothing declared periodic — record, per face, what the kernel ACTUALLY applies at the array edge:
E_tan zeroed (and where: node index), H_tan zeroed (index), absorber active (nonzero sigma layers, count),
wrap/Bloch. Method: instrument at trace time (wrap the wall/absorber functions and
`precompute_coeffs` to record their arguments; do not change their outputs), on a small grid, one or
two steps. A case an entry point refuses is recorded as REFUSED with the exception text. Output: one
table per boundary (rows entry points, columns faces), then a list of every cell where two entry points
disagree and every cell where the realized kind differs from the declared one.

## Deliverable 3 — physics witnesses (`B0/WITNESS.md`, JSON per arm; VESSL for anything long)
- W1 closed PEC cube 24 mm, dx = 1 mm (and 0.5 mm): TM110 (analytic 8.833 GHz) through `run()`,
  `forward()`, the sweep, the NU lane; resonance by the repository's own Harminv/ring-down method.
- W2 PMC-walled rectangular cavity (magnetic walls on the faces the #1164 issue body used, electric on
  the others; say which), at dx, dx/2, dx/4: resonance through the same entry points, and the two
  analytic values with the walls at the declared faces and at half a cell inside (A2).
- W3 the parallel-plate line of `tests/unit/ports/test_lumped_port_known_load_line.py` (PMC side
  faces, PEC plates, no absorber): |S11| for R = Zc/2, Zc, 2Zc through `run(compute_s_params=True)` and
  `forward(port_s11_freqs=...)`, lumped and wire ports, with the port at x = dx (as committed) and ON the
  PMC face's node plane.
- W4 a point source in an all-CPML box (8 layers): the end-of-run field at a probe through `run()` and
  `forward()`; the difference, and the same with the backing-PEC call removed from `run()` by
  instrumentation (A5).
Each witness: realized grid, pads, preflight text verbatim, energy witness, run id if VESSL.

## Report
`/root/workspace/bk-workspace/.boundary-model/B0/REPORT.md`, under 150 lines: A1–A5 each CONFIRMED /
CONTRADICTED / NOT MEASURED with the evidence path; the three deliverables' summaries; any FACT found
wrong; commands with last lines; run ids.

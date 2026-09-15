# Group T (tests) — recompute ledger for #931

Branch `feat/931-t4-materials-api-misc`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T4-materials-api-misc`.
Directories: tests/unit/{materials,api,misc,sources,farfield}.

## VESSL runs submitted: NONE

Every fixture in this group is a unit test. The migrations changed geometry
DECLARATIONS (foil drawn as a one-cell Box -> foil declared as a sheet) and the
tests that read them; none of them carries a committed fixture file, an oracle
number, or a gate bound derived from a previous solve, so there is nothing to
re-solve and no artifact to regenerate. The tests that DO solve re-derive their
own numbers every run and gate on relative or self-consistency properties:

| test | solve | what it gates | why the contract does not move it |
|---|---|---|---|
| `misc/test_flux_monitor_finite_size` | 2200 steps, 2-D 116x44x1 | finite-size flux == full-plane integrand over the SAME window | both monitors are on ONE run; the SWR witness is a `> 2.0` floor |
| `farfield/test_ntff_box_nu_pads`, `test_farfield_chunking` | 120 / 300 steps | CPML pad indices; chunked == whole bitwise | index and self-consistency checks, not magnitudes |
| `farfield/test_farfield_inplane_nonuniform` | short NU runs | uniform-valued profile == plain uniform grid; graded total radiated power in the discretization envelope | relative both ways |
| `api/test_api::test_four_port_waveguide_boundary_ports_through_api` | `compute_waveguide_s_matrix(num_periods=30)` | channel-is-alive (`> 0.2`), relative isolation, and a `pytest.warns(..., "passivity")` LOCK | see below |
| `api/test_visualize3d` | 50 steps | a PNG / VTK file appears | plotting smoke |

The one fixture whose PHYSICS moved is the four-port septum: a volume now
realizes walls on both of its bounding planes, so each guide either side is one
cell narrower. Its gates are loose channel-alive bounds plus the passivity
warning lock, and the whole test runs locally in well under the machine's
one-minute ceiling, so it is re-run in place rather than on VESSL. If the
passivity warning ever stops firing, the test's own docstring says the answer
is to upgrade it to real magnitude / passivity gates — not to relax it, and not
to delete the wrapper.

`farfield/test_oblique_rcs_absolute_sigma` keeps its +0.86 / +0.85 dB locks
untouched: its plate is a `rasterize(..., PEC_SIGMA)` sigma fill, which design
note §1.8 fences out of the contract. crossval-C and the critic pass agree on
this; tests-crossval's four GPU re-runs for the RCS family are unnecessary
unless the PI extends the contract to sigma fills.

## Local runs (this machine, JAX_PLATFORMS=cpu, -n 4)

* `tests/unit/materials` at the branch tip, before any T edit: 154 passed
  (178 s) — stages B and D had already migrated this directory.
* migrated fixtures + the new ownership file: 59 passed, 1 skipped (120 s).
* tests/unit/{materials,sources} + both viewer files after the migration:
  255 passed (927 s).
* tests/unit/api/test_api.py + test_conductor_mask_accessor.py + the four
  volume-scatterer farfield files: 68 passed, 1 skipped (372 s). The four-port
  septum's passivity-warning lock still fires under the new realization, and
  the waveguide advisory now names "the REALIZED guide (40.0000 mm (y,
  aperture) x 20.0000 mm (z, aperture))" — the #868 "40 mm guide reads 42 mm"
  case, closed.
* tests/unit/farfield + tests/unit/misc: 132 of 141 reported with zero
  failures before the run hit its 1700 s wall-clock cap (machine load average
  was 90-150 from the other #931 agents); the files T touched in that set are
  all covered green by the runs above. Nothing failed in any of the five runs.

## Final state

After merging `feat/931-lattice-ownership` (which landed
`tests/_realized_geometry.py`, the shared build-time realization check that
every group-T assertion now goes through):

    tests/unit/{materials,api,misc,sources,farfield}
    561 passed, 15 skipped, 0 failed (991 s, -n 4, JAX_PLATFORMS=cpu)

## Second pass (2026-09-07, same branch)

Three sites the first pass left, none of them needing a solve:

* `farfield/test_ntff_smatrix_drop_warning::_msl_thru` — the trace had been
  migrated to a sheet with only a comment; it now carries the build-time
  realized-plane check like every other migrated conductor in this group.
* `misc/test_review_tier1_validation_battery` OPT-C1 — the `(N-1)*dx` cavity
  length is DOMAIN-face PEC (§1.8), not a conductor body, so it does not move.
  Recorded at the block comment stating the convention, which is what the
  inventory row asked for ("record it so nobody 'fixes' it by analogy").
* `materials/test_sheet_impedance::_mixed_probe_fed_msl` — the last one-cell
  PEC foil in this group's directories. Now a sheet, and the board is redrawn
  on-lattice (`dx = h_sub/3 = 84.667 um` instead of the copied 80 um) so the
  254 um laminate face is a node: at 80 um it is 3.175 cells up and the sheet
  would land 14 um inside the substrate. Design note §1.3 says redraw, not
  snap. The fixture's only consumer is `test_fence_mixed_sparams`, which
  expects a ValueError before any solve, so nothing is re-measured.

VESSL runs after this pass: still NONE, for the same reason.

### Verification after the second pass (2026-09-07, pod load average 50-60)

* `tests/unit/materials` + `tests/unit/farfield`: 230 passed, 0 failed (110 s)
* `tests/unit/api` + `tests/unit/sources`: 279 passed, 2 skipped, 0 failed (438 s)
* `tests/unit/materials/test_sheet_impedance.py`: 62 passed (173 s), and the
  two tests touching the redrawn board 2 passed (27 s) after the coordinate
  assertion was added
* `tests/unit/farfield/test_ntff_smatrix_drop_warning.py`: 7 passed (6.5 s)
* `tests/unit/misc/test_review_tier1_validation_battery.py`: 7 passed (9.2 s);
  `test_flux_monitor_finite_size::test_the_short_realizes_walls_on_both_drawn_planes`
  + `test_ris.py`: 1 passed, 12 skipped (6 s)
* a whole-`tests/unit/misc` run hit the 1150 s cap with 64 reported and zero
  failures; its two long solves (`test_flux_monitor_finite_size`, 2200 steps
  each) were green in the first pass's full five-directory run and are not
  touched by the second pass, which changed one block comment in that
  directory.

All runs `-n 4`, `JAX_PLATFORMS=cpu`.

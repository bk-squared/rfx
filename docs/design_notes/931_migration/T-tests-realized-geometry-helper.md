# Handoff from group T (tests) — `tests/_realized_geometry.py` (shared, landed by the core branch)

`assert_wall_planes(sim, axis, expected_m=...)` and `assert_sheet_planes(...)`
resolve each metre value to `node_index(grid, axis, p)` — the NEAREST node —
and then compare index lists. So when the declared face is OFF the node line,
the expected value snaps exactly the way the realization snapped, both sides
agree, and the assertion passes while the conductor sits somewhere the board
never declared.

Measured 2026-09-07 on `tests/unit/materials/test_sheet_impedance.py`'s
`_mixed_probe_fed_msl` (254 um laminate, 600 um trace):

| dx | node_index(254 um) | z at that node | `expected_m=(254e-6,)` |
|---|---|---|---|
| 80 um (the fixture as copied) | 3 | 240.000 um | passes |
| 84.667 um = h_sub/3 (redrawn) | 3 | 254.000 um | passes |

Both pass; only one is the declared board. This matters because "realized ==
declared" is the whole claim of the migration, and every group is asserting it
through this helper with `expected_m=`.

What T did in its own file: assert the realized plane's physical COORDINATE
next to the index assertion —

    z = np.asarray(coords_from_uniform_grid(rz.grid).z, dtype=float)
    assert abs(float(z[k]) - h_sub) < 1e-9

Suggested change to the helper, for whoever owns it (one keyword, default off
so no existing call changes meaning):

* `assert_wall_planes(..., expected_m=..., on_lattice=True)` — after the index
  comparison, check `abs(node_line[plane] - declared) <= tol` (tol default
  something like `1e-9`, i.e. exact) and fail with both numbers in the message:
  "realized at 240.000 um, declared 254.000 um — the face is off the node line;
  redraw the fixture on-lattice (§1.3), the contract does not snap silently".
* Same keyword on `assert_sheet_planes`.

Any migration that used `expected_m=` on a face that is not exactly on a node
should be re-checked with it; a fixture that then fails is not a helper bug, it
is the #325/#802 off-lattice-interface class the contract is supposed to
surface.

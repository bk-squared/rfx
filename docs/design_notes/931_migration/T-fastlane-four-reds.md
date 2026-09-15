# Four default-lane regressions at 6b8101de (2026-09-08)

Worktree: `rfx-931-fastlane-reds`, branch `feat/931-fastlane-reds`.
All measurements below used this worktree's imports (`PYTHONPATH=$PWD`),
`JAX_PLATFORMS=cpu`. No binary fixture or numerical tolerance was changed.

The local `docs/agent-memory/` and research notes are absent in this worktree;
no sibling checkout was consulted. The load-bearing existing records are:

* Design §1.3: “an exact half-cell tie resolves to the LOWER plane” and
  “A sheet owns NO cell”. The plane and advisory fixes preserve both facts.
* Design §1.7: `realized_pec_edge_masks` is the only geometry-to-edge owner.
  The junction warning now reads those same edges, rather than expecting
  the retired conductivity fold to describe PEC geometry.
* `T1-RECOMPUTE.md`: “Nothing to recapture” was a result on the original
  80 µm fixture. `T2-capture_msl_replay_fixture.md` subsequently records the
  84.667 µm redraw and requires fields, geometry and goldens to change
  together. The unchanged binary captures are historical assembly locks;
  they cannot be interpreted using the new board's indices.

## 1. Replay float64 equivalence — TEST defect

The live integration fixture changed `DX` from 80 µm to `H_SUB/3` and changed
`LY` with it. The autodiff test imported both constants, while its committed
complex64 replay fields and NumPy golden still came from the 80 µm capture.
The replay runner accepted full-plane arrays from a different mesh without
checking their dimensions. This changes V and I before the wave split; it
is not floating-point associativity and the golden is not stale for its data.

| Quantity | Broken interpretation | Captured interpretation |
|---|---:|---:|
| dx (µm) | 84.6666666667 | 80 |
| Grid `(nx, ny, nz)` | `(183, 53, 30)` | `(192, 54, 31)` |
| Stored planes `(frequency, y, z)` | `(5, 54, 31)` | `(5, 54, 31)` |
| Trace plane | 3 | 4 |
| V `(j_centre, k_lo, k_hi-exclusive)` | `(25, 0, 3)` | `(26, 0, 4)` |
| I `(j_lo, j_hi, k_trace_lo, k_trace_hi)` | `(22, 29, 3, 3)` | `(22, 30, 4, 4)` |
| V, drive 0 / port 0, 0.5 GHz | `5.105270e-13 - 8.270205e-16j` | `6.630924e-13 - 1.077116e-15j` |
| I, drive 0 / port 0, 0.5 GHz | `-1.777166e-20 + 2.290879e-21j` | `1.234892e-14 + 2.047347e-16j` |
| max absolute JAX − stored golden | 0.9998806656893184 | 2.2382978971976774e-8 |
| max absolute NumPy − stored golden | 0.9998806704526475 | **0.0** |

The diagnostic wrapped `msl_modal_voltage` and `msl_loop_current` to inspect
these intermediates, replaying the same fields under the two geometries.
The independent NumPy assembly was
`scripts.capture_msl_replay_fixture._compute_numpy_f64_golden_s1` under the
scoped x64 context, using the historical simulation as its `sim_ref`.
Neither comparison ran FDTD. The original test reports **absolute**, not
relative, S error.

All stored frequency bins were inspected. Complex S values below are rounded
for display; the unchanged 1e-5 gate compares full-precision arrays.

| GHz | Broken S00 | Restored S00 | Broken S10 | Restored S10 |
|---:|---|---|---|---|
| 0.500 | `0.999999 - 0.000012j` | `0.001418 + 0.016536j` | `0.000003 - 0.000013j` | `0.995724 - 0.091120j` |
| 1.625 | `1.000010 + 0.000005j` | `0.014649 + 0.051088j` | `-0.000017 - 0.000001j` | `0.955171 - 0.291577j` |
| 2.750 | `1.000004 + 0.000025j` | `0.039898 + 0.078393j` | `-0.000009 - 0.000042j` | `0.874164 - 0.478583j` |
| 3.875 | `0.999999 + 0.000007j` | `0.073956 + 0.093814j` | `0.000003 - 0.000009j` | `0.756713 - 0.643603j` |
| 5.000 | `1.000000 + 0.000011j` | `0.111079 + 0.094658j` | `0.000010 - 0.000011j` | `0.609367 - 0.779626j` |

Fix: explicitly bind replay to its captured 80 µm mesh and lateral extent;
reject mismatched plane shapes before assembly. A build-only regression pins
the captured grid, sheet and actual tangential wall, live normal E, stored
plane dimensions, and V/I indexing; it also deliberately rejects the current
board's mesh. The current on-lattice board stays in the live AD/geometry checks and
integration fixture. The slow end-to-end test keeps its existing builder;
its capture was not rerun or rebaselined in this default-lane repair. Future
replay capture-script runs must update the fields, golden and these geometry
pins atomically.

## 2. Declared MSL sheet plane — TEST defect

The fixture actually uses `H_SUB = 254 µm`, `dx = 254/3 µm` and declares a
sheet Box from `H_SUB` to `H_SUB + dx`. Its midpoint is 296.333333333 µm,
exactly 3.5 cells: the two candidates are 254 and 338.666666667 µm, so §1.3
selects the lower node, **plane 3 at H_SUB**. The old comment and plane-4
expectation described the historical 80 µm grid (midpoint/dx = 3.675).

Fix: assert the three-cell thickness and half-cell midpoint, then require the
sheet declaration and realized tangential wall at `H_SUB`; normal E stays
live and no volume cell is owned. No production snapping logic changed.

## 3. Zero-thickness PEC sheet advisory — TEST defect

The sheet is exactly on the 5 mm node (padded index 14). Commit `cddff05a`
intentionally suppressed `sheet_plane_realized` for exactly aligned sheets,
so strict preflight accepts a correct declaration. Its absence is not missing
metal: build-only assembly measured PEC edge counts **[72, 72, 0]** with no
volume occupancy and a single z-sheet at index 14.

Fix: assert absence of the structured `mesh_resolution` and
`sheet_plane_realized` findings, and separately pin sheet realization,
nonempty tangential edges and live normal E. No advisory prose is matched.

## 4. Reference-guide junction clearance — CODE defect

`_warn_junction_probe_clearance` compared conductivity arrays after #931
removed the PEC `sigma=1e10` fold. All three device/reference sigma differences
were zero, although the componentwise realized edge differences contained
4,620 / 4,620 / 13,849 entries. Thus the junction vanished from the guard.
The existing absorber “0.5 guide-wavelength” finding is a different check;
it cannot replace a probe-to-junction clearance finding.

Fix: retain the conductivity comparison and include componentwise realized
PEC edge differences from the existing owner, emitting structured
`port_junction_probe_clearance` warnings with port locations. Componentwise
comparison matters: unioning Ex/Ey/Ez first can hide changed orientation.
Keep the declared geometry masks for this check even when the Kottke solver
dispatch relinquishes its binary mask to the tensor; this prevents spurious
junctions for identical device/reference declarations on that path.

Build-only public-entry evidence: port clearances **0 / 2 / 2 mm**, all below
the unchanged **3/alpha = 28.1142286057 mm** at 5.5 GHz for a 40 mm guide.
The advisory recommends 5/alpha = 46.8570476762 mm. The public test now stops
at the extractor boundary after the advisories, with no FDTD, and checks
codes and port IDs. Lightweight cases cover identical/distant geometry,
sheet-only differences, component changes with identical location unions,
conductivity differences and absent PEC carriers.

## Verification

Baseline command (before edits):

```sh
JAX_PLATFORMS=cpu PYTHONPATH=$PWD timeout 1200 python3 -m pytest -n 4 \
  tests/unit/autodiff/test_msl_sparam_ad.py \
  tests/unit/preflight/test_preflight_guards.py \
  tests/unit/sparams/test_waveguide_port_reference_sims.py -q
```

Measured baseline: **4 failed, 59 passed** in 100.41 s. The four failures
were exactly the requested cases. Final verification is recorded below.

Final command (after edits; began only after the active sibling pytest exited):

```sh
JAX_PLATFORMS=cpu PYTHONPATH=$PWD timeout 1200 python3 -m pytest -n 4 \
  tests/unit/autodiff/test_msl_sparam_ad.py \
  tests/unit/preflight/test_preflight_guards.py \
  tests/unit/sparams/test_waveguide_port_reference_sims.py \
  tests/contracts/test_lattice_ownership_contract.py \
  tests/unit/preflight/test_preflight_advisory_emission_contract.py \
  -q --durations=10 --junitxml=output/fastlane931/verification.xml
```

**190 passed, 0 failed, 0 skipped, 0 xfailed**, 296 warnings, 87.25 s.
Counts read from the JUnit testcase records:

| File | Passed | Failed |
|---|---:|---:|
| `tests/unit/autodiff/test_msl_sparam_ad.py` | 5 | 0 |
| `tests/unit/preflight/test_preflight_guards.py` | 50 | 0 |
| `tests/unit/sparams/test_waveguide_port_reference_sims.py` | 18 | 0 |
| `tests/contracts/test_lattice_ownership_contract.py` | 114 | 0 |
| `tests/unit/preflight/test_preflight_advisory_emission_contract.py` | 3 | 0 |

The one `slow` MSL end-to-end case was excluded by the existing default
marker selection; it was not run or modified. No test was deleted or xfailed.
The longest selected test (the compact junction's existing A/B solve witness)
took 53.66 s; the next solve-bearing test took 30.71 s. The replay equivalence
test took 18.90 s and performs no FDTD.

`ruff check` passed on all four changed Python files; `git diff --check`
passed. The only production file changed, `rfx/api/_sparams.py`, is exercised
by the waveguide and MSL files above; this Markdown record has no executable
test cases. All three historical MSL binary fixtures are byte-unchanged from
6b8101de. Local detailed logs and the JUnit report are in the ignored
`output/fastlane931/` directory.

All four requested failures are settled. The slow end-to-end capture and
full repository lane remain untested in this repair; the known VESSL
openEMS-presence failures are outside its scope.

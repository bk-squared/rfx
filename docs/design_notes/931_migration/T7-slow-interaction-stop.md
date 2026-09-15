# Ownership and automatic meshing: audit stopped at numerical locks

Date: 2026-09-08. Source: `17b51196a5cb0eabaaf77d3938912427e5b4ce57`.
Worktree: `rfx-931-fastlane-reds`; branch: `feat/931-slow-regressions`.

**Status: blocked by the user's explicit RE-PIN stop condition. This is an
evidence and design-proposal commit, not an implementation or closure claim.**
No production code, tests, gates, or baselines changed. The general interaction
battery below remains to be implemented. No VESSL run was launched.

## Decision and evidence boundary

The user required: "a bit-identity or historical-base lock that fails because
the branch legitimately changed a number is a RE-PIN needing evidence and a
named run: stop and tell me, do not re-pin it yourself."

Both historical locks now compare different conductor realizations. Restoring
the old operator to satisfy either golden would contradict #931. This is
sufficient to invoke the stop; it does not establish that every part of either
numerical delta has been causally isolated or that a new baseline is validated.

The normative note, §1.1, says "For PEC VOLUMES it is sampled at cell centres";
§1.7 says `realized_pec_edge_masks` "is the only function that turns geometry
into PEC edges". This proposal is consistent with both: retain these owners and
make every consumer evaluate them against one explicit mesh snapshot. The
§6 amendment says an exactly placed sheet "prints nothing". Reviving that
advisory solely to satisfy a stale consumer would contradict the amendment.
No `docs/agent-memory/` directory exists in this worktree; no other worktree
was read or modified. The current normative note and prior audit notes were
read instead of assuming inaccessible private memory describes current code.

All seven Group A fixtures supply explicit `dx`. `_mesh.py:69` returns the
declared inputs directly in that case. Their A/B difference therefore cannot
all be explained by automatic mesh planning. There are independent consumer,
ownership-migration, numerical-lock, and runtime-sizing failures.

## Named run and exact disposition

`local-before-17b51196-20260908`: **40 cases = 27 passed + 6 failed + 7 errors**,
495.09 seconds, four workers, no worker deaths, zero skips/xfails in this
selection. It selected the seven supplied Group A node IDs plus the entire
refplane module, including its 27 passing cases. The controller was already
running at handover (PID 1063816, cwd verified); it was observed through
completion, and no second pytest was started. Original command options were
`-o addopts= -n 4`, with `--junitxml=/tmp/rfx-931-slow-regressions/before.xml`.
Explicit marker override is essential even with explicit node IDs.

[baseline-summary.json](slow-interaction-evidence/baseline-summary.json)
contains every node ID, outcome, error message, the full ten-bin complex MSL
S trace, source SHA, and hashes of the original log/JUnit files. The main A/B
counts supplied by the user were not independently rerun in another checkout.

| Supplied test | Outcome and disposition |
|---|---|
| `test_mu_r_override_fenced_on_nonuniform` | FAILED before `forward`: test line 312 calls the deliberately uniform-only `_build_grid()` with a declared profile. Actual `mu_r_override` refusal remains at `_execute.py:3288`. Future consumer correction: use realized-grid shape; retain the NU override refusal. |
| `test_graded_inplane_mesh_integrates_with_local_cell_sizes` | FAILED: 6.0565159% power discrepancy exceeds the unchanged 5% envelope. Its preceding local-cell-versus-old-scalar discriminator passes. Ownership changes affect the scatterer, but a complete numerical attribution is still owed. No widening or fixture tuning authorized by this finding. |
| `test_compute_msl_s_matrix_end_to_end_matches_historical_base` | FAILED; **RE-PIN stop**. 36/40 complex entries mismatch; max absolute difference 0.1702888. Live fixture explicitly changed 80 µm spacing to 254/3 µm and trace plane 320 µm to 254 µm; replay helper preserves historical geometry only for replay. The live historical-base test uses the redrawn board. |
| `test_v173a_baseline_bit_identity` | FAILED; **RE-PIN stop**. Current 1,914,163,251.7905414 Hz versus 1,994,994,938.1663296 Hz: −80,831,686.3757882 Hz. Harness declares a positive-thickness high-sigma API Box, now a volume. The resonance assertion fails before the dip assertion can adjudicate the second lock. |
| `test_dc_limit_pins_receive_sign` | SETUP ERROR: expects two obsolete `wire_port_dead_extent_cells` findings plus `pec_faces_finite_pec`; actual report has only the latter. Existing fast fixture checks in the same file already use that singleton. DC numerical/sign gate did not execute. |
| `test_ad_smoke_eps_override_grad_finite_with_sheet` | FAILED before differentiation: hardcoded 14 checkpoint segments do not divide 3717 steps on the redrawn explicit mesh. Preserve the no-padding guard; derive segmentation from the prepared grid's actual timestep count. Gradient not evaluated. |
| `test_soft_advisory_real_coarse_pec_short_witness` | FAILED: column power 1.2999108 is outside the unchanged `(2.25,3]` witness interval. The volume's coarse near wall changes under ownership; the source's assertion that this cannot change the advisory is unsupported. Further diagnosis needed; neither gate nor witness should be tuned to manufacture the warning. |
| `test_refplane_thru_s21_tracks_box_referee` | SETUP ERROR; shared exact-plane advisory expectation is stale. Numerical gate unexecuted. |
| `test_refplane_thru_leaves_legacy_lock_band_loudly` | Same shared SETUP ERROR; numerical gate unexecuted. |
| `test_refplane_thru_reciprocity` | Same shared SETUP ERROR; numerical gate unexecuted. |
| `test_refplane_thru_measured_line_constants` | Same shared SETUP ERROR; numerical gate unexecuted. |
| `test_refplane_thru_deembedded_phase_tracks_measured_beta` | Same shared SETUP ERROR; numerical gate unexecuted. |
| `test_refplane_thru_energy_and_passivity_labeled` | Same shared SETUP ERROR; numerical gate unexecuted. |

Refplane setup lines 912 and 917 require the advisory and its text even though
`_preflight.py:8618-8623` intentionally suppresses exact-plane, non-tie notices.
The future shared fixture must independently assert realized sheet planes and
edges, and check diagnostics according to their conditional meaning. Removing
an expectation without replacing its geometry witness would weaken the test.

The sigma route matters: `add(material=...)` with sigma ≥ 1e6 enters
`classify_pec_entry` in `_compile.py:276` and `rasterize_grid.py:808`.
The unchanged §1.8 route is raw `rasterize(..., sigma=...)`, coax conductivity
stamping, or subthreshold DC fold. The existing contract test explicitly pins
this distinction at `test_lattice_ownership_contract.py:1041-1045`. Relabeling
the V173A API volume as a fenced sigma fill would change its declaration.

## Independent structural witnesses

[structural-witnesses.json](slow-interaction-evidence/structural-witnesses.json)
was generated in this worktree with `PYTHONPATH=$PWD JAX_PLATFORMS=cpu`; its
recorded import path confirms the local copy. These are assembly/planning
measurements, not additional pytest results or physics validation.

* V173A: explicit dx = 1 mm; shape `(67,67,29)`; 1008 occupied volume cells,
  zero sheets; Ex/Ey/Ez edge counts `[2072,2088,1073]`. Tangential walls are
  at z = 1 and 2 mm, with normal edges at z = 1 mm. This independently
  confirms the changed operator underlying the lock. It does not validate
  a new resonance baseline.
* `_make_dz_profile([(0.2 mm,1.2 mm,4)], domain_z=4 mm, dx=1 mm)` sums to
  3.8 mm. The leading gap is omitted by `auto_config.py:742-747`; the full
  cumulative node trace is retained.
* Overlapping intervals `[0,2]` and `[1,3]` mm with declared extent 5 mm
  produce a 6 mm profile. `auto_config.py:724-760` appends entire overlapping
  thicknesses rather than partitioning physical intervals.
* A thin z dielectric plus a 0.2 mm x-thick PEC volume resolves to dx =
  0.5 mm and is **correctly refused by name** as sub-cell. The planner's
  z-only optimization discards the global feature-limited dx at
  `auto_config.py:469-481`. This is an automatic-planning limitation, not
  a silent loss of conductor ownership. The declaration `(8,8,20)` mm
  also resolves to a z extent of 77.4481145 mm.

The MSL full per-bin trace is finite, nearly reciprocal, and varies smoothly
through all ten frequencies; it is not an empty or all-zero measurement. The
independent geometry change explains why an equality premise is invalid, but
does not substitute for a new physics referee. No full V173A time trace or
farfield angular trace was captured by this inherited test run; those remain
required for adjudicating new physics numbers.

## Proposed general rule and architecture — not implemented

**A declaration retains its meaning throughout construction. At a consumption
boundary, resolve one mesh snapshot, evaluate every conductor against that
grid, and realize its declared kind or refuse it by name before dispatch.**
Automatic meshing chooses unspecified mesh inputs. It cannot infer a new
conductor kind, discard another axis's resolution obligations, bypass a lane
refusal, or mutate a frozen snapshot.

Retain the declaration/resolution split, `freeze_mesh`, the ADI guard,
`classify_pec_entry`, and `realized_pec_edge_masks`. Introduce one explicit
preparation boundary with a resolved record carrying per-axis provenance,
geometry/material revision, frozen status, grid coordinates, and realization
findings. `run`, `forward`, `optimize`, `preflight`, `vmap`, visualization and
export consume that same record or its inspection projection. Capability
validation consumes the prepared geometry and requested lane before dispatch;
inspection may report a refusal without starting a solver. Registration reads
the declaration explicitly. Descriptors can remain compatibility projections,
but a private field read must no longer decide which semantic phase is active.

The mesh planner must partition absolute-coordinate intervals, including
overlaps and every nonzero gap, then satisfy each axis's independent feature
constraints. Z refinement cannot justify relaxing x/y constraints. Sheet
thickness is not a volume-cell constraint; wire filament/volume choice must be
evaluated against the final local cells. Hard admissibility and preferred
interface alignment are separate: §1.3 permits reported nearest-plane sheet
snapping and tie-to-lower selection. Exact alignment must not be falsely
promised for every auto sheet. Unsupported planning arrangements must yield a
named refusal, never a profile that silently moves the geometry.

Proposed domain decision: a `Simulation`'s explicitly supplied domain remains
authoritative, subject to the documented grid realization convention (including
uniform ceil rounding); standalone `auto_configure` may still propose a domain.
Do not transplant its freely sized z proposal into a fixed-domain Simulation.
This is a **design change** to the resolution policy currently recorded in §6,
not a claim that the current domain replacement was accidental. It needs a
normative amendment and migration evidence when work resumes.

Rejected: thirteen individual fixture edits as the deliverable; changing
ownership back to historical sigma/node behavior; silently freezing on read;
dispatching to Yee instead of requested ADI; splitting mesh propagation out of
#931; replacing the already shared conductor rasterizer; widening either
physics gate; regenerating goldens without the required decision.

## Deliberate cross coverage required before closure

Use equivalent-case factoring with explicit coverage metadata, not a claim
that one example covers the Cartesian product. Each case records route, kind,
placement, declared/resolved mesh, timing, entry, lane, expected realization or
named refusal, and oracle. Collection must reject an unclassified case. A
capability row is either implemented and tested, or an explicit tested refusal;
an unsupported lane never implicitly converts geometry. The following is the
planned coverage inventory, not new passing tests.

| Cross | Required witnesses | Existing anchors / remaining work |
|---|---|---|
| Kind × route | PEC library and custom threshold materials as volumes; zero Box and thin-conductor equivalent sheets; f0 versus PEC footprint; filament versus thick wire; raw sigma and low-sigma DC controls | `test_lattice_ownership_contract.py` covers owner primitives and sigma distinction. Cross each with automatic and explicit mesh preparation. |
| Placement × kind × axis | Node, fractional offset, exact half-cell tie, domain walls, periodic seam, positive sub-cell extent; rotate x/y/z; mirror; body on final cell; zero/one/two degenerate axes | Existing owner tests are anchors. Add explicit expected named refusals and preparation parity across all orientations. Never assume z-only grading handles x/y. |
| Mesh inputs × geometry | dx declared/None; profiles declared/inferred/absent; mixed explicit profiles with automatic dx; uniform/NU; sheet-only; overlapping dielectrics; tiny gaps; thin x/y metal plus thin z dielectric | `test_auto_mesh_resolution.py` covers selected documented boards. New interval/topology and mixed-axis witnesses above currently expose unresolved behavior. |
| Timing × route | Empty preview; preview after first geometry; later additions and material replacements; freeze empty and complete models; subsequent rejected sub-cell addition; copy/sweep; mutable profile ownership | `test_frozen_mesh_construction.py` and auto-mesh tests cover parts. Add order independence, revision identity, and prepared-record reuse assertions. Provisional reads never freeze. |
| Entry × resolved grid | run, forward, optimize, preflight, vmap, visualize, export share coordinates and realized masks; retain declarations on export; assert supported inspection and execution semantics separately | Existing sheet-entry tests and consumer tests are partial. Add the same adversarial mixed model through each boundary, not separate unrelated fixtures. |
| Lane × kind × grid | Uniform/NU Yee equality on identical nodes; distributed boundary/seam mask transport; ADI resolved-NU and interior-PEC refusals; subgrid sheet/wire refusals; conformal/Kottke rules | `test_pec_lane_parity_numeric.py` and `test_declared_solver_and_monitor_domain.py` provide anchors. Preserve §1.8 subpixel fences; curved conformal validity remains limited, not silently promoted. |
| Tracing × timing | First consumption under outer JIT/grad of static model; explicit fixed profiles with traced design arrays; refusal for automatic traced shape selection; nonzero finite AD control; segmentation derived from actual dt | Existing static-auto JIT tests are partial. No traced discrete sheet-plane selection; no checkpoint padding. |
| Diagnostics × realization | Exact sheets silent; offset/tie findings; geometry asserted independently of advisory text; dead port cells derived from driven edge component | Shared refplane/DC setup must use the same semantic witness as fast fixtures. A silent advisory is never proof that geometry was realized. |

Written scope fences to retain: raw sigma/DC-fill models are outside PEC body
ownership; automatic geometry/material planning is static; diagonal filament
segments are refused; subgrid sheet/wire execution is refused; ADI interior
PEC and resolved NU are refused; domain PEC is a boundary, not a body;
Kottke and conformal keep their separately documented responsibilities.
Do not invent support for an unverified cell in this table.

Verification after implementation must collect the entire non-GPU suite with
`-o addopts= -m 'not gpu'`, accounting separately for default, slow,
slow_physics and highmem cases. Run serial controllers; at most `-n 4`, with
highmem selections at `-n 0`. Audit exact collected versus completed node IDs,
including skips and pre-existing xfails, and require zero missing/duplicate
cases and zero worker deaths. Keep default fast tests and the explicit small
cross battery in PR gating; the marker-cleared impact lane must also run when
mesh/ownership/preparation surfaces change. GPU/distributed verification is a
separate named lane. None of this broader verification was performed after the
mandatory stop, and no green closure is claimed.

## VESSL handoff

First obtain named CPU evidence for the two re-pin decisions, not another
worker-losing omnibus slow run. Use separate serial jobs on the source SHA
above or this evidence-only commit. From the checkout root:

```sh
mkdir -p output/931-repin-evidence
export PYTHONPATH=$PWD JAX_PLATFORMS=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Job name: rfx-931-msl-lock-evidence
python -m pytest -o addopts= -n 0 -v -s \
  tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_end_to_end_matches_historical_base \
  --junitxml=output/931-repin-evidence/msl.xml \
  > output/931-repin-evidence/msl.log 2>&1
```

```sh
# Separate job name: rfx-931-v173a-lock-evidence; same environment exports.
mkdir -p output/931-repin-evidence
python scripts/harnesses/v173a_physics_equivalence.py \
  > output/931-repin-evidence/v173a.json 2> output/931-repin-evidence/v173a.log
```

The MSL test is expected to fail its existing lock; the V173A harness records
both extracted values without editing the baseline. Preserve source SHA,
JAX/jaxlib/CPU environment, full logs and artifacts with the actual VESSL run
IDs. Neither command authorizes a re-pin. Do not run
`scripts/capture_msl_e2e_golden.py`: it overwrites the committed golden. Before
accepting new values, retain full time/field or angular traces and the realized
geometry witness; quantify every changed lock and record the PI decision.
The full impact/GPU jobs become meaningful after the architecture is
implemented and the two re-pin decisions are resolved.

## R3 audit

No contradiction with the existing ownership or ADI memory requirements was
found in this worktree's spec/audit-note search. The domain policy above is
explicitly a proposed replacement for §6, not a silently enacted exception.
R2 attempts: zero repair attempts; one observed baseline selection and one
structural assembly/planning diagnostic, no parameter sweeps. Cheap falsifier
run: parse all 40 JUnit cases and assert the exact 27/6/7 census; independently
assemble V173A and inspect its volume/sheet classification. Both agree with
the stated stop. New baseline validity, complete numerical attribution and
the general interaction implementation are not verified.

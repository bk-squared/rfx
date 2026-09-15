# Slow consumer repairs and ownership attribution

Worktree: `rfx-931-fastlane-reds`; branch: `feat/931-slow-regressions`.
Starting source: `5ce0de652d6bb58a70735504dce42da83ddd8b16`.

## Scope and governing evidence

The user authorized repairs to nine tests and numerical attribution of two
others, while explicitly reserving the historical MSL and V173A locks for the
PI. Those two tests, their fixtures and their baselines are unchanged.

No `docs/agent-memory/` directory exists in this worktree. The read-first
substitutes are the normative
[`#931 contract`](../20260906_plan_realign_lattice_ownership.md),
[`T7 stop and evidence`](T7-slow-interaction-stop.md), and the review at
`5ce0de65`. The contract's section 1.7 says `realized_pec_edge_masks`
"is the only function that turns geometry into PEC edges"; section 6 makes
an exactly placed sheet silent. These changes are consistent with both:
geometry assertions read the owner, and conditional notices remain conditional.
T7 says a complete numerical attribution is owed for the graded far-field
comparison and disallows widening or fixture tuning. The controlled
counterfactuals below test a cause without changing either failing test.

## Repairs and their general reach

* **Reference-plane fixture:** all six slow gates share one repaired setup.
  Its existing fast geometry oracle is factored into a helper called by the
  slow fixture too. It asserts sheet kind, no occupied volume cells, the
  realized tangential wall, physical plane equal to the declared height, and
  the full declared footprint and z_lo PEC ground. Preflight findings are
  printed and errors
  still refuse the fixture. There is no advisory-code or wording baseline in
  this physics setup. A cheap mutation retained node index 2 while moving
  the declaration from 1.00 to 1.05 mm; the physical-plane assertion rejected
  it. Thus checking the plane index cannot accidentally certify a snapped
  declaration as exact.
* **Magnetic override consumer:** obtain the override shape from
  `_build_realized_grid()`, then enter `forward()` and require the existing
  `mu_r_override` refusal. The same check now covers x, y and z profiles,
  each with equal-valued and genuinely graded cells. Equal-valued profiles
  still select the nonuniform lane. The graded controls preserve the domain
  extent and matching boundary cells.
* **DC receive-sign consumer:** reuse the realized foil and driven-edge
  oracle from the broadband fixture. It now checks both ports: each has
  exactly two live Ez edges below the foil. The broadband and DC fixtures
  share that oracle and retain error-severity preflight checks. The dedicated
  preflight-message test keeps its existing singleton expectation; diagnostic
  emission has its own test instead of deciding whether physics gates run.
* **AD checkpoint consumers:** use the realized grid's `num_timesteps()` and
  the existing `_suggest_checkpoint_segments()` helper. Pass the exact
  three-period step count to the solve. The same repair reaches the sibling
  `test_msl_s_matrix_ad_end_to_end`, the other hardcoded-14 caller found by
  searching `rfx/`, `tests/`, `scripts/` and `validation/`. The no-padding
  refusal and every finite/nonzero-gradient and finite-difference gate stay.
  Measured budgets: sheet fixture 3,717 steps / 59 segments / 63 steps per
  segment; sibling 3,934 / 14 / 281. The sibling retains its old execution
  segmentation while losing the mesh-specific magic constant.

The general rule is to assert geometry against the realized mesh, size arrays
and runtime budgets from that mesh, and test a conditional diagnostic through
its condition. This uses the branch's existing realization and grid-consumer
boundaries. The broader preparation-record and mesh-planner redesign proposed
in T7 is not implemented or claimed closed here. No production code changes;
the ADI guard, `freeze_mesh`, declaration/resolution split and ownership
contract remain intact.

## Attribution protocol

Both experiments use the original fixture declarations, meshes, sources,
frequencies, run lengths, extractors and numerical gates. Historical source
is read with `git show a3e4dba4^:<path>` in this worktree; no alternate
checkout is used. Each diagnostic verifies its `rfx` import is under the
current directory. Historical operators exist only inside the diagnostic
process and are restored afterwards.

The predeclared falsifier in each script is that restoring only the
historical operator must restore the original interval. Failure to do so
would leave that attribution incomplete. These are attribution experiments,
not validation of historical physics or qualification of a replacement gate.

### Graded far-field power

Diagnostic: `scripts/diagnostics/slow_931_farfield_attribution.py`.
It runs current and historical volume-edge maps on the unchanged uniform and
graded fixtures, four sequential 600-step solves. The actual historical lane
used the tangential map `C & (neighbor_before | neighbor_after)` for each
component. On these fixtures historical node sampling and current center
sampling have **zero differing occupancy entries** on both meshes, so this
intervention isolates the edge map.

| Structural witness | Uniform | Graded |
|---|---:|---:|
| Occupied cells | 216 | 864 |
| Historical Ex/Ey/Ez edges | 216/216/216 | 864/864/864 |
| Current Ex/Ey/Ez edges | 294/294/294 | 1092/1092/1014 |
| Historical upper x/y wall | 11.500 mm | 11.625 mm |
| Current upper x/y wall | 11.750 mm | 11.750 mm |
| Historical/current upper z wall | 11.500/11.750 mm | 11.500/11.750 mm |
| Source physical node | (11,11,9.5) mm | (11,11,9.5) mm |
| dt | 0.476643717383 ps | 0.275190378538 ps |

The operator change acts differently on the two meshes: the historical
transverse size deficit was 250 micrometres versus 125 micrometres. A relative
power gate need not cancel that change. Source spectra, acquisition duration,
source cell volume, NTFF coordinates, full angular traces, scalar-cell
extraction and a NumPy/JAX extractor comparison are recorded to test other
possible explanations. The two meshes' different acquisition durations are
held fixed across the operator comparison.

**Measured disposition: PI decision; leave the 5% gate failing.**

| Edge operator | Uniform power metric | Graded power metric | Local-cell discrepancy | Scalar-cell discrepancy |
|---|---:|---:|---:|---:|
| Current | 6.15680348922e-14 | 6.52969118201e-14 | 6.05651445% | 19.86749754% |
| Historical | 6.49238034674e-14 | 6.63119581578e-14 | 2.13812903% | 15.43087602% |

Restoring only historical edges restores the old envelope. The new operator
changes the uniform power by -5.16878001% and the graded power by
-1.53071387%, increasing their discrepancy by 3.91838542 percentage points.
The local-cell-versus-scalar discriminator passes in both arms. NumPy and JAX
angular intensities agree within 1.13e-6 relative to peak, and all recorded
arrays are finite. This attributes the gate crossing to the ownership/mesh
interaction rather than a broken local-cell integration path.

The old integrated-power agreement did not certify angular equivalence:
normalized pattern L2 discrepancies are 18.73% current and 18.84% historical.
Both source and NTFF metadata were checked, but the original finite windows
and abrupt 2.5-ratio grading are not a convergence study. A redesigned
integration oracle or a qualified physical envelope needs the PI's decision.
See the [full trace analysis](slow-consumer-evidence/farfield-result-note.md)
and [inspected angular figure](slow-consumer-evidence/farfield-attribution.png).

### Coarse-short advisory witness

Diagnostic: `scripts/diagnostics/slow_931_advisory_attribution.py`.
It extracts the unchanged nested fixture builder from the test AST. The
actual historical waveguide path at `a3e4dba4^:rfx/api/_sparams.py:3012-3014`
folded node-sampled PEC into `sigma=1e10`; it did not apply the historical
tangential-mask operator used in the far-field case.

The three coarse arms are current ownership, historical node-sigma fold,
and node-sampled occupancy with the current volume-edge owner. The third
arm separates sampling from the material operator. A fourth arm executes
the original finer false-positive control that the failing coarse assertion
otherwise prevents from running. Input fingerprints check that ports and
background materials are identical across the coarse arms.

On the 2 mm mesh, the current short has tangential walls at 84/86 mm;
the historical sigma fold damped the 86 mm node. The right source is at
90 mm, its reference plane at 84 mm and its probe at 70 mm. That reference
plane now lies on the near wall, across the short from its source. Full
incident/outgoing spectra and probe records are therefore needed to interpret
the change; a lower column power alone cannot certify physical accuracy.

**Measured disposition: PI decision; leave `(2.25,3]` failing.**

| Arm | Maximum column power | Soft column-power advisory |
|---|---:|---|
| Current coarse | 1.2999107838 | Absent |
| Historical node/sigma fold | 2.5279102325 | Present |
| Node sampling with current edge owner | 2.5279037952 | Present |
| Original finer control | 1.0372081995 | Absent |

All coarse port and incoming-material fingerprints match, with 1,312 steps
on the same grid. The node-sampled/current-owner bridge restores the interval
and matches the historical first-column complex S within 8.92e-6. Thus the
sampling-induced near-wall shift is sufficient; retaining the retired sigma
operator is unnecessary to recover this scalar witness.

The complete matrices and records contradict a physical passivity reading.
In the current coarse right-drive run, all 10,496 captured V/I samples are
zero: the measurement planes lie across the shielding short from the source.
The extractor returns a zero column using its existing safe denominator.
Historical finite-sigma leakage instead creates tiny nonzero denominators
and finite ratios. With left drive, the right reference reads a standing
field on/in front of the wall, rather than transmitted power on its far side.
Coarse settling also misses the existing -40 dB criterion. These observations
exclude treating the lower current column power as qualified physical accuracy.

The exact fine control hidden behind the coarse assertion satisfies both
original assertions, including absence of an advisory. That result alone
does not qualify its reference-plane placement. Another discovered limitation
is that `_soft_fired` matches any `ADVISORY`, including reciprocity; the
analysis distinguishes the actual column-power warning. This helper and the
entire physical witness test remain untouched for the PI decision.

See the [full matrices/record analysis](slow-consumer-evidence/advisory-result-note.md)
and [inspected power/trace figure](slow-consumer-evidence/advisory-attribution.png).

## Verification and disposition

All runs use `PYTHONPATH=$PWD JAX_PLATFORMS=cpu`, at most four pytest workers,
and one pytest controller at a time. No VESSL job is launched.

**Nine original tests closed; two attribution cases require additional PI
decisions. The two original PI-blocked historical locks remain untouched.**

| Original test | Final disposition |
|---|---|
| `test_refplane_thru_s21_tracks_box_referee` | PASS; shared realization repair |
| `test_refplane_thru_leaves_legacy_lock_band_loudly` | PASS; same shared repair |
| `test_refplane_thru_reciprocity` | PASS; same shared repair |
| `test_refplane_thru_measured_line_constants` | PASS; same shared repair |
| `test_refplane_thru_deembedded_phase_tracks_measured_beta` | PASS; same shared repair |
| `test_refplane_thru_energy_and_passivity_labeled` | PASS; same shared repair |
| `test_mu_r_override_fenced_on_nonuniform` | PASS, expanded to six axis/profile variants; refusal unchanged |
| `test_dc_limit_pins_receive_sign` | PASS; actual sign gate and sign-flip falsifier executed |
| `test_ad_smoke_eps_override_grad_finite_with_sheet` | PASS; gradient -0.1739357, finite and nonzero |
| `test_graded_inplane_mesh_integrates_with_local_cell_sizes` | FAIL unchanged; operator attribution reaches PI stop |
| `test_soft_advisory_real_coarse_pec_short_witness` | FAIL unchanged; sampling/reference-plane attribution reaches PI stop |
| `test_compute_msl_s_matrix_end_to_end_matches_historical_base` | Original PI block; no edits or re-pin; not rerun |
| `test_v173a_baseline_bit_identity` | Original PI block; no edits or re-pin; not rerun |

| Completed selection | Passed | Failed | Existing xfailed | Total |
|---|---:|---:|---:|---:|
| Geometry/consumers and contracts, slow marks included | 192 | 0 | 1 | 193 |
| AD consumers and checkpoint controls, serial | 12 | 0 | 0 | 12 |
| Final targets and attribution tests, slow marks included | 53 | 2 | 0 | 55 |
| **Distinct tests, latest outcome** | **217** | **2** | **1** | **220** |

There are 260 completed test reports, including 40 deliberate reruns in the
final target selection. No duplicate IDs occur within an invocation, and
each JUnit suite count matches its completed case list. No errors, skips,
worker deaths or timeouts occurred in these final selections. The initial
setup-only development check is accounted for separately below. Exact IDs,
outcomes and JUnit hashes are in the
[test census](slow-consumer-evidence/test-census.json).

Final refplane |S21| is smooth from 0.99846 to 0.98179 across its nine bins;
the referee ratios are 0.99192–0.99744. Both ports' complex Zc and beta traces
were inspected. DC |S21| is approximately 1.0000/0.9998 and phase residuals
are -0.0520/-0.1043 rad; the sign-flipped control fails the same band as
required. The sibling MSL AD test passes its finite-difference checks and
has gradient -0.01666153. The checkpoint suite retains forward/gradient
equivalence and rejects non-divisors. See
[consumer traces](slow-consumer-evidence/consumer-physics-traces.txt).

The initial geometry/consumer selection completed in 354.07 seconds:
**192 passed, one existing xfailed, zero failures/errors/skips**. It includes
the complete reference-plane and lumped two-port batteries (slow_physics
included), the six permeability refusal variants (slow), frozen construction,
declared-solver/monitor-domain contracts, and three preflight modules. The
pre-existing strict xfail is
`test_thru_s11_floor`; no marker is changed by this work.

An initial eight-case setup check had six passes and two failures in the newly
added graded x/y controls: I had given their boundary cells unequal sizes,
which correctly hit the grid's existing matching-endpoint guard before the
intended refusal. Moving that grading to the interior preserves the same
domain and permits the intended test. All six final profile variants pass.
This was a control-construction defect, not an attempt to tune a physics gate.

Ruff comparison against `git show 5ce0de65:<file>` finds 12 existing findings
in the five edited test files, zero added and zero removed. Source-span/code
comparison ignores only shifted line numbers. New diagnostic scripts pass
Ruff. The worktree-local evidence directory is `.validation-931-closures/`.

## Reproduce / VESSL handoff

No VESSL job was launched. Launch the following as **separate CPU jobs** on
this commit; retain the real run IDs and the entire `output/931-closures/`
directory. The physics job is expected to report the two PI-decision failures;
neither job invokes either originally blocked historical lock. Run controllers
serially when using the shared pod. Common environment:

```sh
mkdir -p output/931-closures
export PYTHONPATH=$PWD JAX_PLATFORMS=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

CPU job `rfx-931-consumer-ad` (the measured local selection: 12 passed):

```sh
timeout 1200 python -m pytest -o addopts= -n 0 -q -rP \
  tests/unit/sparams/test_msl_sheet_threading.py::test_ad_smoke_eps_override_grad_finite_with_sheet \
  tests/unit/autodiff/test_sparam_ad_end_to_end.py \
  tests/unit/autodiff/test_scan_segmented_checkpoint.py \
  --junitxml=output/931-closures/ad.xml > output/931-closures/ad.log 2>&1
```

CPU job `rfx-931-consumer-physics` (union of the two non-AD selections above):

```sh
timeout 1200 python -m pytest -o addopts= -m 'not gpu' -n 4 --dist=loadfile -q -rP \
  tests/locks/test_refplane_port_waves.py \
  tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py \
  tests/oracle/test_ram_magnetic_mu_r_design.py::test_mu_r_override_fenced_on_nonuniform \
  tests/contracts/test_frozen_mesh_construction.py \
  tests/contracts/test_declared_solver_and_monitor_domain.py \
  tests/unit/preflight/test_preflight_guards.py \
  tests/unit/preflight/test_preflight_reads_realized_edges.py \
  tests/unit/preflight/test_preflight_rasterization.py \
  tests/unit/farfield/test_farfield_inplane_nonuniform.py \
  tests/unit/sparams/test_sparam_passivity_guard.py \
  --junitxml=output/931-closures/physics.xml > output/931-closures/physics.log 2>&1
```

For independently named attribution evidence, CPU job `rfx-931-attribution`
can run the following sequentially with the same environment. Locally these
processes were confined to four CPUs with `taskset -c 0-3`; use CPUs assigned
to the job if restricting affinity. These reproduce attribution, not a newly
qualified baseline:

```sh
python scripts/diagnostics/slow_931_farfield_attribution.py \
  --output-dir output/931-closures > output/931-closures/farfield.log 2>&1
for task_arm in current legacy_sigma node_volume; do
  python scripts/diagnostics/slow_931_advisory_attribution.py --arm "$task_arm" \
    --output-dir output/931-closures > "output/931-closures/advisory_${task_arm}.log" 2>&1
done
python scripts/diagnostics/slow_931_advisory_attribution.py --arm current --fine \
  --output-dir output/931-closures > output/931-closures/advisory_fine.log 2>&1
```

## R3 and remaining limits

No contradiction with the normative ownership rule or the T7 stop was found:
the historical operators are diagnostic counterfactuals only, and both
attribution gates remain red. R2: one predeclared comparative experiment per
attribution hypothesis, both reaching their falsifier; no parameter sweep or
non-closing physics retry. The erroneous graded setup control had an identified
construction defect and was corrected before the final selection.

Executed falsifiers include the same-node/off-plane declaration rejection,
unchanged checkpoint non-divisor refusal, sign-flipped DC gate, and both
historical-operator comparisons. Protected-file hashes and an empty diff for
all `rfx/` and `tests/fixtures/` content are recorded in
[provenance](slow-consumer-evidence/provenance.json). The two originally blocked
test files and both attribution test files are byte-identical to the base.

GPU/distributed, the complete repository default/slow suites, continuum or
time-window convergence, and qualification of replacement locks are not
claimed. The two original MSL/V173A decisions and the two new attribution
decisions remain with the PI. The committed matrices, port and angular traces,
figures and analysis notes are evidence artifacts, not replacement baselines.
Full raw NTFF arrays and test logs/XML remain in `.validation-931-closures/`.

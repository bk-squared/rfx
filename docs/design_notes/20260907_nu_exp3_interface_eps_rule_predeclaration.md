# E3 — the interface-node eps rule in the PRODUCTION path (opt-in prototype for #931) — pre-declaration

**Status:** pre-declaration. This commit precedes the source change
(`Simulation(..., interface_eps=...)`, `rfx/runners/nonuniform.py`), the
instrument wiring (`validation/research/multiband_nu/w7_accuracy_ad.py
--interface-eps`), the tests and every measurement. Every numeric window
below is frozen from this commit onward; results are appended under
"Results", never edited into the windows. One attempt per arm; a fired
falsifier is a result, not a bug to tune away.
**Class:** observable-vs-analytic on the lane F A1 z-stratified cavity
(family: W7 A1 `w7_accuracy_ad`, note
`20260907_nu_band_accuracy_ad_predeclaration.md`, whose oracle, exact
discrete model, harness, extraction, gates and 'dual' rows are reused, not
re-derived), plus a bit-identity contract on the default path.
**Tree:** worktree `rfx-nu-exp3`, branch `exp/nu-experiments-e3`, based on
`feat/nu-band-accuracy-ad @ fea9f078` (which carries `feat/nu-band-profile`
and `origin/main d990e18c`). E2 (`exp/nu-experiments-e2`, worktree
`rfx-nu-exp2`) is read, cited, and not modified.
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-exp3/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Every JSON the instrument writes records `rfx.__file__`, `git_sha`,
`git_dirty`, `argv` and `started_utc` (the W6/W7/E2 convention).
CPU only (a GPU lane runs elsewhere; the machine is shared).
Date: 2026-09-07 (KST).

**This is a PROTOTYPE for #931 (geometry -> lattice ownership).** It adds
an opt-in flag and measures what the flag does on one fixture family. It
changes no support-matrix row, no default, and no committed number. Whether
the rule becomes a default, and for which lanes, is #931's decision, not
this note's.

## 0. Scope — the question

Lane F (section 1a/1b of its note) found that the production NU material
assembly (`rfx.runners.nonuniform.assemble_materials_nu`) samples each E
node's material by the half-open `[lo, hi)` rule on the node coordinate.
At a node-aligned interface the node's material is decided by the last ulp
of the running cell sum (nine meshes, three assignment patterns), and for a
tangential E component either one-sided choice is a first-order error:
+0.15 % at s = 1 on the uniform A1 mesh, fitted order 1.44 (UC) / 1.78 (MB)
against 2.007 / 2.010 for the dual-cell-average rule the instrument
assembles for itself. E2 measured the law behind the average
(`e_shift = K t (h - t)`, second order at fixed fill) and stated the
normal-E rule without measuring it.

The question here: realise the second-order rule INSIDE the production
assembly, behind an opt-in switch, and measure that the production path
then reaches the instrument's numbers. Two claims, both falsifiable:

1. **Bit-identity of the default.** `interface_eps="sampled"` (the default)
   changes nothing: every existing test in the four batteries and the
   example-fidelity contract passes unchanged, and an explicit
   `interface_eps="sampled"` run is bit-identical to a run without the
   kwarg.
2. **The production path realises the rule.** With
   `interface_eps="dual_average"` the w7 A1 'production' arm reaches the
   'dual' arm's numbers: fitted order >= 1.8 on UC and MB, and the same
   per-scale errors to <= 0.5 MHz.

## 1. Findings made while writing this note (zero FDTD)

### 1a. Where each E component sits, and what the rule therefore is

`materials.eps_r` is ONE co-located scalar field on the node lattice,
consumed by all three E components (`update_e_nu`); per-component eps
exists only on the `aniso_eps` channel (`update_e_nu_aniso`, fed today by
`compute_smoothed_eps_nonuniform` when `subpixel_smoothing=True`). The
stagger the NU update realises (read from `rfx/core/yee.py` and
`rfx/nonuniform.py`, not assumed): the E update differences H BACKWARD
with the dual spacing `inv_d_e[k] = 2/(d[k-1]+d[k])`, the H update
differences E FORWARD with the primal spacing `1/d[k]`. So E_c(i, j, k) is
the edge running from node (i, j, k) to the next node along c: it lies
INSIDE cell index k_c along its own axis (half-open, the cell whose low
corner is the node) and ON the node lines of the other two axes, bordering
the four cells `(i_a - 1 | i_a) x (i_b - 1 | i_b)` there.

The rule that follows, per component, from the dual-cell Ampere balance
(E2 1a: E tangential is continuous across an interface, so the dual-cell
integral of eps E is E times the volume integral of eps):

- **tangential axes** (the two axes `a, b != c`): E_c takes the
  area-weighted ARITHMETIC mean of the four cells sharing its edge,
  weights `d_a[i_a - 1 | i_a] * d_b[i_b - 1 | i_b]`. On a z-stratified stack
  (all cells equal in x and y) this reduces exactly to the instrument's
  `dual_eps_nodes`: `(eps[k-1] d[k-1] + eps[k] d[k]) / (d[k-1] + d[k])`;
- **its own axis** (the normal component of an interface normal to c):
  E_c takes the eps of the ONE cell its edge lies in, `eps_cell[k_c]`. A
  node-aligned interface normal to c does not cross the E_c edge at all —
  the edge starts on the interface and runs into cell k_c — so there is
  nothing to average, and the harmonic (D-continuous) rule E2 1a/1c states
  for a normal component reduces to the single cell's value when the edge
  is inside one material. The harmonic rule is needed only when an
  interface cuts the edge (a sub-cell layer, E2's S1 family on Ez) — not
  this lane's case and not measured here;
- at the first node of an axis there is no lower cell: one-sided (the
  cell's own eps); the trailing bounding node's phantom cell copies the
  last real cell (so the last node is one-sided too). This is exactly the
  instrument's end-node convention (`out[0] = ce[0]`, `out[n] = ce[-1]`).

The cell eps is sampled at the CELL CENTRE (`node + d/2`, exact float64
spine) by the same `rasterize_geometry` the node path uses, so no cell
centre ever lands on a node-aligned interface and the ulp residue of lane
F 1a cannot enter. An interface strictly inside a cell (sub-cell feature)
is staircased at the centre — E2's S0 class, first order; stated, not
measured.

Consequence for the SAMPLED default, recorded: at a node whose two
adjacent cells differ along axis a, the sampled value is right for the
normal component E_a when the ulp landed "upper" (cell k's material) and
wrong for it when it landed "lower"; it is one-sided (first order) for
both tangential components either way. The opt-in rule fixes all three.

### 1b. Why the rule lives on the per-component channel and what the scalar field keeps

The smallest change that gives the tangential components the mean WITHOUT
giving the normal component a smeared value is to emit three arrays on the
existing `aniso_eps` channel and leave `MaterialArrays.eps_r` (the scalar
field) as the sampled column. Putting the mean into the scalar field would
give E_z at a substrate/air node `(4.3 + 1)/2` where the E_z edge sits in
air — a first-order error of the S0 class on exactly the component that
dominates microstrip-class models. Physical fidelity first: the normal
component keeps its cell. So under the opt-in:

- the three E components use `(eps_x, eps_y, eps_z)` from the rule
  (`update_e_nu_aniso`);
- `materials.eps_r` stays sampled and is what source normalisation
  (`make_current_source`), the CPML coefficients (`apply_cpml_e`), the
  fidelity report's materialisation rows and `rfx.visualize` read — the
  same split `subpixel_smoothing=True` already has on this lane;
- `materials.sigma` stays sampled and isotropic (`update_e_nu_aniso`
  applies sigma isotropically, as documented there). A lossy interface is
  outside this prototype's domain.

### 1c. What the prototype refuses (loud, never silent)

`interface_eps="dual_average"` raises `ValueError` / `NotImplementedError`
at run time when combined with any of: Debye/Lorentz materials (the
dispersive scan branch does not consume `aniso_eps`); `subpixel_smoothing`
(two eps rules); thin conductors (sheet resample and the sigma folds are
scalar-field operations, `resample_sheet_node_materials`); lumped RLC
(folds into `eps_r`); `eps_override` (the AD-material channel replaces the
scalar field only); a traced (mesh-as-design-variable) profile (the
cell-centre rasterization is host float64); the distributed NU lane and
the S-parameter NU lane (neither carries `aniso_eps`). The surface-
impedance sheet + `aniso_eps` refusal (#677 v1) already exists and holds.
Every refusal is a test (section 4), not a measurement.

## 2. Fixture — lane F A1, verbatim

The z-stratified PEC cavity of lane F 2.1: core 4.3 | thin 3.0 | core 4.3
| air, interfaces at z = 14, 16, 30 mm, L_z 44 mm, a = 30 mm, b = 3 mm, LSE
m = 1, p = 5, `f_true = 10 561 719 600.896 Hz` (transfer-matrix oracle,
selfcheck (i)/(i')/(i'') to 1e-12). Meshes UC / MB at s = 0.5, 1, 2
(transverse cell 0.25 s mm; UC uniform 0.5 s mm; MB builder bands cap 1.4),
AZ at the same scales for the column check only. Source/probe/waveform/
extraction: lane F 2.1 (15 ns harminv, 10 ns truncation invariance).

Under the opt-in the w7 'production' arm reads the E_y component of the
rule's output at the central (i, j) column instead of the scalar field,
then runs on the instrument harness as before (Ey only; the model of lane
F 3.1 applies unchanged to that column). Nothing else in the arm changes.

Lane F's measured 'dual' rows, the reference for E3-F2 (MHz, `err_hz`
from `results/w7_accuracy_ad.json`, invariance-passing, all valid):

| arm | s = 0.5 | s = 1 | s = 2 |
|---|---|---|---|
| UC dual | -3.9786 | -16.0400 | -64.3221 |
| MB dual | -7.5005 | -30.0934 | -121.6770 |

and its fitted 'dual' orders 2.007 (UC) / 2.010 (MB); 'production'
(sampled) orders 1.438 / 1.780.

## 3. Falsifiers (frozen; tolerances never widened after measurement)

Every window is evaluated by the instrument's own judges
(`judge_a1`: G3 model residual, invariance, `production_orders`) or by
the replay test; none is computed by hand.

- **E3-B (bit-identity of the default).** (a) The batteries
  `tests/unit/nonuniform`, `tests/unit/geometry`, `tests/unit/materials`,
  `tests/oracle` with `-m "not gpu and not slow"` and
  `tests/contracts/test_example_fidelity_contract.py` pass on the changed
  tree with the same pass/skip counts as on `fea9f078` (recorded both
  ways). (b) A 200-step NU run (A1 MB s = 2 mesh, PEC, Ey source) with
  `interface_eps="sampled"` explicit and with the kwarg omitted: time
  series bit-identical, `max |diff| = 0` exactly; and
  `assemble_materials_nu` output arrays bit-identical. (c) The lane F
  section 1a interface table (nine meshes, three nodes each) is
  reproduced by the default path exactly (w7 `--selfcheck`, unchanged
  constants). Window: zero differences, zero test regressions.
- **E3-C (the production rule equals the instrument's rule).** For the
  nine (arm in uc, mb, az) x (s in 0.5, 1, 2) meshes, the E_y column the
  production rule emits at (nx//2, ny//2) equals
  `f32(dual_eps_nodes(prof))` at every node to `<= 5e-7` relative (one
  float32 ulp; the f64 arithmetic is the same formula, the x/y averaging
  is the identity on a transversely uniform stack), and the E_z column
  equals the cell-centre (half-open at the centre) eps of cell k at every
  node exactly. Window: `<= 5e-7` relative on E_y, `0` on E_z.
- **E3-O (oracle and selfcheck).** w7 `--selfcheck` `all_pass` on this
  tree with the default rule: oracle (i)/(i')/(i'') `<= 1e-12`; all 30
  model rows to `1e-7`; orders/ratios to `1e-3`; interface tables
  reproduced. Window: `all_pass = True`, else no arm runs.
- **E3-G3 (model residual).** Every measured unit:
  `|f_meas - f_model| <= 0.15 MHz` with f_model from the emitted column
  (lane F G3, same constant). Window: `<= 0.15 MHz`.
- **E3-V (run-length invariance).** `|f_meas(15 ns) - f_meas(10 ns)|
  <= 0.1 MHz` per unit, else the unit is INCONCLUSIVE and excluded from
  the fit (lane F rule). Window: `<= 0.1 MHz`; `>= 3` fit points per arm
  required for E3-F1, else INCONCLUSIVE.
- **E3-F1 (order).** Fitted order of |err| vs h on the production
  dual_average ladder, three scales, UC and MB: `p_uc >= 1.8` and
  `p_mb >= 1.8` (task window). Lane F's dual arm gave 2.007 / 2.010 and
  its sampled production arm 1.438 / 1.780; the exact discrete model
  predicts 2.004 / 2.009 for the dual column. Reported alongside: the
  lane F oracle-validity window `[1.8, 2.2]` on UC (a UC order above 2.2
  would be an anomaly, not a pass).
- **E3-F2 (per-scale agreement with lane F 'dual').** For each of the
  six (arm, s) units: `|err_hz(E3 production dual_average) -
  err_hz(lane F dual)| <= 0.5 MHz` against the table of section 2. Same
  rule realised in production vs in the instrument, same mesh, same dt,
  same waveform; the only differences are the column's assembly route
  and CPU-vs-lane-F float32 reduction order.
- **E3-R (refusals).** Each combination of section 1c raises before any
  step is taken; the error text names `interface_eps`. Pure tests.
- **E3-P (the report states the rule).** `fidelity_report()` prints
  `interface eps rule (NU lane): ...` on every NU model and the returned
  domain row carries `interface_eps_rule` ONLY under the opt-in (so the
  example-fidelity snapshot is untouched by the default). Pure test.

## 4. Declared commands, outputs, tests

```
cd /Users/byungkwankim/Documents/rfx-nu-exp3
export PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-exp3
PY=/Users/byungkwankim/Documents/rfx/.venv/bin/python
$PY -c "import rfx; print(rfx.__file__)"          # must be this tree
# E3-B(a): batteries, before (fea9f078) and after, counts recorded in Results
$PY -m pytest tests/unit/nonuniform tests/unit/geometry tests/unit/materials tests/oracle \
    -q -o addopts="" -m "not gpu and not slow" -p no:cacheprovider
$PY -m pytest tests/contracts/test_example_fidelity_contract.py -q -o addopts="" -m "not gpu"
# E3-O, then the A1 ladder with the opt-in on (one attempt per unit, its own JSON)
$PY -m validation.research.multiband_nu.w7_accuracy_ad --selfcheck \
    --out validation/research/multiband_nu/results/e3_interface_eps_rule.json
$PY -m validation.research.multiband_nu.w7_accuracy_ad --arms a1 --a1-arms uc,mb \
    --scales 2,1,0.5 --rules production --interface-eps dual_average \
    --out validation/research/multiband_nu/results/e3_interface_eps_rule.json
# E3-B(b,c), E3-C, E3-R, E3-P and the order replay
$PY -m pytest tests/unit/nonuniform/test_interface_eps_rule.py -q -o addopts=""
```

Outputs: `validation/research/multiband_nu/results/e3_interface_eps_rule.json`
(the w7 layout: `runs[]`, `selfcheck`, `a1.units` keyed `arm|s|production`
with `interface_eps_rule` recorded per row, `a1.judge`). Tests:
`tests/unit/nonuniform/test_interface_eps_rule.py` — the JSON-free
contract tests (E3-B(b,c), E3-C, E3-R, E3-P) and the replay (skips while
the JSON is absent; re-fits E3-F1 from the JSON's rows through
`w7.fit_line`, re-checks E3-F2 against `results/w7_accuracy_ad.json`'s
dual rows, E3-G3 and E3-V through the rows' own gates, and pins that every
row's `rfx_file` is this tree and `interface_eps_rule == "dual_average"`).

Source surface (declared before it is written): `Simulation.__init__`
gains `interface_eps: str = "sampled"` (validated against
`("sampled", "dual_average")`, stored as `_interface_eps`);
`rfx/runners/nonuniform.py` gains `INTERFACE_EPS_RULES` and
`assemble_interface_eps_nu(sim, grid, materials) -> (eps_x, eps_y, eps_z)`,
and `run_nonuniform_path` feeds it to the existing `aniso_eps` channel
after the subpixel block; the distributed NU lane (`_execute.py`) and the
S-parameter NU lane (`_sparams.py`) refuse the opt-in; `rfx/fidelity.py`
states the rule. `assemble_materials_nu`, `rasterize_geometry`,
`Box.mask_on_coords`, `coords_from_nonuniform_grid` are not edited.

## 5. Impact-sweep rule and the #931 note

No default moves. If E3-B fires (any count changes, any bit differs) that
is a STOP: the prototype is withdrawn from the branch until the difference
is explained in this note. If E3-C fires the rule is mis-assembled and the
ladder is NOT run (E3-C is checked by the test before the ladder). If
E3-F1 or E3-F2 fires with E3-C held, the difference between "same column
in the instrument" and "same column in production" is the result, and it
is reported with numbers.

For #931: this lane hands over (1) a working per-component realisation of
the tangential-mean / normal-own-cell rule on the NU lane behind a flag,
(2) the measured order and per-scale agreement, (3) the list of paths the
rule does not reach (1c) — which is the list #931 has to decide about
before any default changes. The uniform lane (`rfx/api/_compile.py`
`_build_materials`) has the same node sampling and the same ulp residue
and is NOT touched here.

## 6. Where this note departs from the task as written, and why

- The task named `interface_eps='sampled' | 'dual_average'` on
  `Simulation`; taken as written.
- "Arithmetic mean of the two adjacent cells' eps for the tangential-E
  components" is realised as the area-weighted mean over the FOUR cells
  sharing the edge (which is the two-cell mean whenever the interface is
  planar and the transverse cells match, i.e. on every fixture measured
  here); the four-cell form is the one the dual-cell balance gives in
  3-D and costs nothing extra.
- The normal component is not left at the sampled value: it takes the
  cell its edge lies in (1a). That is a change of the normal component at
  nodes where the ulp landed "lower" — deliberate, stated, and checked by
  E3-C's E_z clause.

## Results

### Resume correction declared before the production patch (2026-09-10)

The previous STOP below is retained as historical evidence, but its absolute-green
interpretation is superseded by the lead's clarification. E3-B is a DELTA gate:
**the after-patch failure SET must equal the before-patch failure SET, and every
pinned value in the passing set must be unchanged.** No frozen window above
Results is edited. The baseline carries these four pre-existing failures:

- `tests/unit/nonuniform/test_band_accuracy_ad_replay.py::test_replay_ad3`
- `tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_envelope_regression_lock`
- `tests/oracle/test_leontovich_alpha_oracle.py::test_o3_model_fits_measured_field`
- `tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_oracle_o3`

The lead verified the middle two on `origin/main d990e18c` by running exactly
those two tests in the main checkout (selector command: `pytest
tests/oracle/test_leontovich_alpha_oracle.py -k "test_alpha_envelope_regression_lock
or test_o3_model_fits_measured_field"`): **"2 failed, 11 deselected"**.
Both are `@pytest.mark.slow_physics`; the normal battery's
`-m "not gpu and not slow and not slow_physics"` excludes them. The lead's
sibling-branch CPU battery under that marker set was 6140 passed / 0 failed.
This is lead-provided evidence, not a new run in another checkout.
AD3 attempt 1 was FIRED and its replay intentionally remains red, as recorded in
`docs/design_notes/20260907_nu_band_accuracy_ad_predeclaration.md`.
The fourth failure is the baseline's O3 model-fit trust assertion; no independent
main-branch verification of that fourth test is claimed.

Before-patch baseline: 4 failed, 1490 passed, 8 skipped, 92 deselected,
19 xfailed. The existing baseline JSON is retained verbatim. After-patch battery
will use the identical baseline command and compare failure sets; new E3 tests
will be accounted for separately when comparing pass counts.

### Historical stopped run (superseded gate interpretation)

2026-09-10: **STOP at E3-B, before the production patch.** No E3 ladder
unit was attempted. The frozen declaration above is unchanged.

Baseline provenance: branch `exp/nu-experiments-e3`, git SHA
`4129591bb7cef863adf0f8249c788d31c9b4337c`, CPU only (`JAX_PLATFORMS=cpu`),
`rfx.__file__ = /Users/byungkwankim/Documents/rfx-nu-exp3/rfx/__init__.py`.
Every Python invocation used the requested venv interpreter with
`PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-exp3`. `git diff
fea9f078 4129591b` contains only this predeclaration, so the tested
production source is exactly the declared baseline source.

Evidence (including verbatim pytest failure reports and invocation
provenance):
`validation/research/multiband_nu/results/e3_interface_eps_rule_baseline.json`.
This is baseline evidence, not the declared A1 measurement artifact;
`e3_interface_eps_rule.json` was not created.

### E3-B: FIRED — a pinned default value has moved

The requested battery, with `-q -o addopts="" -m "not gpu and not slow"
-p no:cacheprovider`, completed once on the unpatched tree:

```text
4 failed, 1490 passed, 8 skipped, 92 deselected, 19 xfailed, 573 warnings in 1765.01s (0:29:25)
```

The example-fidelity contract, with `-q -o addopts="" -m "not gpu"
-p no:cacheprovider`, completed once:

```text
175 passed, 9 warnings in 34.21s
```

The decisive failure is the default-path diagnostic pin in
`tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_envelope_regression_lock`:

| Quantity | Measured | Frozen comparator/window | Outcome |
|---|---:|---:|---|
| Two-plane attenuation | 0.8733294904232025 | recorded 0.72494 | pin moved |
| Absolute relative difference from that pin | 0.20469209924021636 | <= 0.05 | FIRED |

This fires the task's explicit rule, **“If any pinned value moves on the
DEFAULT path, STOP and report which; never adjust it.”** It also fails
section 3's requirement that the batteries pass. The discrepancy already
exists at the starting commit: it is **not attributed to E3**, since no
production patch was applied. Its cause was not investigated or tuned in
this experiment.

The other baseline failures, with verbatim assertion quantities, were:

| Test | Measured | Window |
|---|---:|---:|
| `test_replay_ad3` | worst dominant AD-vs-FD 0.5224044347795822 | <= 0.15 |
| `test_o3_model_fits_measured_field` | relative RMS 0.010769780031860503 at 8 GHz | <= 0.01 |
| `test_alpha_oracle_o3` | model-fit trust relative RMS 0.010769780031860503 at 8 GHz | <= 0.01 |

Lane F's Results already identify `test_replay_ad3` as red by design.
That inherited replay failure alone was initially treated as a possible
zero-regression baseline exception; the later, explicit attenuation-pin
failure requires STOP regardless of that interpretation. The complete
pytest report is authoritative for the failed test names.

**After-patch counts: NOT RUN.** There is no production patch to compare.
The uncommitted E3 test draft was removed when the stop fired. E3-B(b)'s
200-step bit comparison and E3-B(c)'s selfcheck/table replay were not run.
No unchanged-bit or zero-regression claim is made from this stopped run.

### Remaining gates: not evaluated after the mandatory stop

| Gate | Status | Measured versus frozen window |
|---|---|---|
| E3-C | NOT RUN | no columns emitted; Ey <= 5e-7 relative and Ez exactly 0 difference not evaluated |
| E3-O | NOT RUN | no selfcheck; `all_pass = True`, oracle <= 1e-12, model <= 1e-7 and orders/ratios <= 1e-3 not evaluated |
| E3-G3 | NOT RUN | no A1 frequency; <= 0.15 MHz not evaluated |
| E3-V | NOT RUN | no A1 trace; <= 0.1 MHz and >= 3 fit points per arm not evaluated |
| E3-F1 | NOT RUN | UC and MB orders absent; >= 1.8 and UC [1.8, 2.2] not evaluated |
| E3-F2 | NOT RUN | zero of six units attempted; <= 0.5 MHz agreement not evaluated |
| E3-R | NOT RUN | no opt-in implementation; refusal tests not executed |
| E3-P | NOT RUN | no opt-in implementation; report-rule tests not executed |

NOT RUN is deliberately neither HELD nor FIRED: these gates have no
measurement. The normal-component rule and its justification remain the
predeclared proposal in section 1a, not an implemented or validated result.
No source, pinned value, reference JSON, default, or support-matrix row
was changed. No script was added, so no CLASSIFICATION entry is needed.
The required code-and-tests commit before any E3 measurement was not made
because the mandatory baseline stop occurred first. Only the stop record
and its provenance are committed; nothing is pushed and no PR is opened.

### Implementation before the one-attempt ladder

The opt-in emits `(eps_x, eps_y, eps_z)` through `aniso_eps`; scalar eps,
sigma, source normalization and the sampled default remain untouched.
Each component averages the four cells sharing its edge with transverse
area weights and zero PEC-cell weights. The normal component keeps its
own cell-centre value: a node-aligned interface does not cut its edge,
so the normal harmonic rule reduces to that one cell. Sub-cell cuts and
lossy interfaces remain outside this prototype's measured domain.
Bounding nodes copy the last real cell; all-PEC averages use finite sampled
eps as a fallback and leave field enforcement to the existing PEC mask.

The declared JSON-free contracts and measurement replay live together in
`tests/unit/nonuniform/test_interface_eps_rule.py`. No script is added;
only the already-classified w7 instrument is wired, so no new CLASSIFICATION
entry is needed. An initial auxiliary four-cell test had an invalid y-profile
(two unequal boundary cells), rejected by the existing grid constructor
before assembly. Its fixture was corrected to have matching boundary cells;
no E3-C window fired (all nine declared columns passed on that first check).
No A1 ladder unit has yet been attempted.

### One-attempt measured results (2026-09-10)

Production/tests commit: `7b846c0b716c5e2e72e9e7f6c0722987a042cebb`.
Every selfcheck and ladder row records this SHA, `git_dirty=False`, and
`rfx_file=/Users/byungkwankim/Documents/rfx-nu-exp3/rfx/__init__.py`.
The requested PYTHONPATH/interpreter and `JAX_PLATFORMS=cpu` were used.
Six units, one attempt each, no `--force`, no smoke, no retuning.

| Gate | Status | Measured versus frozen window |
|---|---|---|
| E3-B | HELD (completed comparison below) | explicit/omitted sampled: 200-step max difference 0; material arrays byte-identical; all 27 default table entries unchanged; example-fidelity 175 passed before and after |
| E3-C | HELD | 9 meshes; maximum Ey relative difference 9.706317172231138e-8 <= 5e-7; Ez difference exactly 0 |
| E3-O | HELD | all_pass=True; oracle (i)/(i')/(i'') residuals 1.8128774706335932e-16 / 1.8796313675281302e-16 / 1.314136397143784e-16 <= 1e-12; all model/table/order/ratio selfchecks pass their unchanged windows |
| E3-G3 | HELD | max absolute model residual 0.0593851955242157 MHz <= 0.15 MHz |
| E3-V | HELD | max 15/10 ns difference 0.06576602301216125 MHz <= 0.1 MHz; 3 valid fit points on each arm |
| E3-F1 | HELD | UC 2.007492133930114 and MB 2.009959571601154 >= 1.8; UC also in [1.8, 2.2] |
| E3-F2 | HELD | all six differences from lane F's full-precision dual rows exactly 0 MHz <= 0.5 MHz |
| E3-R | HELD | 11 refusal cases raise before the stepper; all error messages name interface_eps |
| E3-P | HELD | 2 report cases pass: text states sampled/dual_average, domain key present only for opt-in |

| Arm | Scale | Error (MHz) | Model residual (MHz) | 15/10 ns difference (MHz) | Difference from lane F dual (MHz) |
|---|---:|---:|---:|---:|---:|
| UC | 2 | -64.3221257513771 | -0.01917725951576233 | 0.036346180366516115 | 0.0 |
| MB | 2 | -121.67696552715682 | -0.0593851955242157 | 0.008576109344482422 | 0.0 |
| UC | 1 | -16.03996319467926 | -0.034505641912460326 | 0.06576602301216125 | 0.0 |
| MB | 1 | -30.09341446830559 | 0.004069466527938843 | 0.023096275793075563 | 0.0 |
| UC | 0.5 | -3.978594629137039 | 0.01844343797492981 | 0.03517198579978943 | 0.0 |
| MB | 0.5 | -7.500533034704208 | 0.004676412994384766 | 0.004772365056991577 | 0.0 |

Instrument stdout, verbatim (the final generic W7 verdict needs both dual
and production arms for its separate G1/G2 judges; this E3 run deliberately
contains only production rows. E3 uses the frozen production orders and
per-row G3/invariance judges, rechecked by its passing replay):

```text
A1 UC s=2 production: f=10.497397 GHz err=-64.3221 MHz model=-64.3029 MHz resid=-0.0192 MHz inv=0.0363 MHz zfrac=0.964 valid=True cells=19215 steps=13627 wall=1s
A1 MB s=2 production: f=10.440043 GHz err=-121.6770 MHz model=-121.6176 MHz resid=-0.0594 MHz inv=0.0086 MHz zfrac=0.980 valid=True cells=14091 steps=13627 wall=1s
A1 UC s=1 production: f=10.545680 GHz err=-16.0400 MHz model=-16.0055 MHz resid=-0.0345 MHz inv=0.0658 MHz zfrac=0.964 valid=True cells=139997 steps=27254 wall=6s
A1 MB s=1 production: f=10.531626 GHz err=-30.0934 MHz model=-30.0975 MHz resid=+0.0041 MHz inv=0.0231 MHz zfrac=0.980 valid=True cells=102245 steps=27254 wall=4s
A1 UC s=0.5 production: f=10.557741 GHz err=-3.9786 MHz model=-3.9970 MHz resid=+0.0184 MHz inv=0.0352 MHz zfrac=0.964 valid=True cells=1066425 steps=54508 wall=83s
A1 MB s=0.5 production: f=10.554219 GHz err=-7.5005 MHz model=-7.5052 MHz resid=+0.0047 MHz inv=0.0048 MHz zfrac=0.980 valid=True cells=777225 steps=54508 wall=50s
A1 judge: FIXTURE-INVALID or incomplete (G1, G2, G3 not passed)
wrote validation/research/multiband_nu/results/e3_interface_eps_rule.json
```

E3 contract/replay command: `-m pytest tests/unit/nonuniform/test_interface_eps_rule.py
-q -s -o addopts="" -p no:cacheprovider`. Verbatim final summary:

```text
27 passed, 14 warnings in 7.29s
```

Example fidelity used the identical baseline command including
`-p no:cacheprovider`. Verbatim after summary:

```text
175 passed, 10 warnings in 36.58s
```

An orchestration mistake is retained explicitly: the first after-battery
started while the ladder JSON was incomplete. The newly added replay raised
`KeyError: 'uc|0.5|production'`; this was not a measured-window failure.
That battery was interrupted after 86.97 seconds (2 failed, 142 passed,
1 skipped, 92 deselected); its other failure was the known AD3 replay.
Evidence: `e3_battery_incomplete_artifact.json` and its log. The source and
replay were not changed, and the ladder was never restarted. Once all six
rows existed, all 27 E3 tests passed and the identical full battery was
started again. Its completed comparison is appended below when available.

### Separate main-branch finding for the lead to file

Lead-verified on `origin/main d990e18c`: the two selected tests produced
"2 failed, 11 deselected". The measured quantities in this worktree's
unpatched baseline were:

- `tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_envelope_regression_lock`:
  two-plane alpha 0.8733294904232025 versus pin 0.72494; relative difference
  0.20469209924021636 exceeds the 0.05 window.
- `tests/oracle/test_leontovich_alpha_oracle.py::test_o3_model_fits_measured_field`:
  8 GHz relative RMS 0.010769780031860503 versus the <= 0.01 window.

Both tests are `slow_physics` only (not `slow`); the normal battery
`-m "not gpu and not slow and not slow_physics"` hides these reds.
The E3 baseline marker set includes them. This report does not claim to
have rerun main, and no other checkout was modified. The related baseline
`test_alpha_oracle_o3` also fails its model-fit trust check at the same RMS.
No upstream issue or message was posted; this is the finding for the lead.

### Completed E3-B delta comparison — HELD (2026-09-10)

The after battery finished with the **identical argv and marker set** as
its baseline, including all four known-red tests (no deselections by name):

```sh
JAX_PLATFORMS=cpu PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-exp3 /Users/byungkwankim/Documents/rfx/.venv/bin/python -m pytest tests/unit/nonuniform tests/unit/geometry tests/unit/materials tests/oracle -q -o addopts="" -m "not gpu and not slow" -p no:cacheprovider
```

Verbatim before summary:

```text
4 failed, 1490 passed, 8 skipped, 92 deselected, 19 xfailed, 573 warnings in 1765.01s (0:29:25)
```

Verbatim completed after summary:

```text
4 failed, 1517 passed, 8 skipped, 92 deselected, 19 xfailed, 586 warnings in 1632.51s (0:27:12)
```

The after count includes **27 new E3 passes**: 1517 - 27 = 1490, exactly
matching the baseline. Skips=8, deselections=92, xfails=19 and failures=4
are unchanged. The example-fidelity contract is 175 passed before and after.

**Before failure SET = after failure SET**, explicitly:

```text
tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_envelope_regression_lock
tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_oracle_o3
tests/oracle/test_leontovich_alpha_oracle.py::test_o3_model_fits_measured_field
tests/unit/nonuniform/test_band_accuracy_ad_replay.py::test_replay_ad3
```

Set difference in both directions is empty. No existing test, pinned
expectation or lane F reference JSON was edited. All previously passing
pins still pass; the default material bytes, 200-step trace (max difference
0) and all 27 interface-table entries are unchanged. The known-red
assertion values also reproduce exactly:

| Quantity | Before | After | Unchanged comparator |
|---|---:|---:|---|
| Two-plane alpha | 0.8733294904232025 | 0.8733294904232025 | pin 0.72494; relative window <= 0.05 |
| Two-plane alpha relative difference | 0.20469209924021636 | 0.20469209924021636 | <= 0.05 |
| O3 field-fit RMS and O3 trust RMS at 8 GHz | 0.010769780031860503 | 0.010769780031860503 | <= 0.01 |
| AD3 worst dominant relative error | 0.5224044347795822 | 0.5224044347795822 | <= 0.15 |

Evidence: `e3_battery_after.json` contains full verbatim output, argv,
import path, starting SHA, completion SHA, exact failure sets and counts.
The battery started on source/tests commit `7b846c0b`; HEAD advanced only
to the documentation/evidence commit `ce5fc0d0` during execution. A git
diff verifies that source, tests and instrument were unchanged throughout.
`e3_gate_results.json` now records every E3 gate as HELD.

All requested work is complete. No E3 numeric window fired; there was no
ladder reroll or window change. The invalid auxiliary test fixture and
premature partial-artifact replay are preserved above, with their evidence.
The frozen text above Results is byte-identical to `7b7dc182`. Nothing was
pushed, no PR was opened, and no other rfx worktree was modified.

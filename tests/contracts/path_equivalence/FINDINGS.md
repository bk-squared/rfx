# S0 findings after comparator review

`U`, `N`, `D` = uniform, constant-profile NU, two-device uniform;
`Ng/Dg` = matching graded single/two-device lanes; `F` = forward.
Lengths are 12 steps unless `12/36` is shown. Numeric bars are relative to
peak: `1e-4` accumulated; per-step uses 9 float32 ULP at peak. `N/A` denotes
schema, missing-record, or execution findings, not a numeric discrepancy.
Conclusions: in the S0 PR body (leader)

| Cause | Cells / record | Witness vs bar | Implementation A / B |
|---|---|---|---|
| record-semantics-declared-length | Once, all U/N geometry declarations | `declared_length_m` = user domain on uniform, profile sum on NU; excluded from realized equality | [uniform :631](../../../rfx/api/__init__.py#L631) / [NU x :523](../../../rfx/api/__init__.py#L523), [NU y :535](../../../rfx/api/__init__.py#L535) |
| flux-dA-shape | Same run cells; dA | `(1,1)` vs `(13,12)`; N/A | [U :2717](../../../rfx/simulation.py#L2717) / [N :3140](../../../rfx/nonuniform.py#L3140) |
| flux-dA2-missing | Same run cells; dA2 | Missing on U; N/A | [U :2717](../../../rfx/simulation.py#L2717) / [N :3140](../../../rfx/nonuniform.py#L3140) |
| forward-flux-record-missing | `_flux_monitors`, FU/FN, 12/36 | Missing on both forward result records; availability finding, not a measured path difference | [FU :2709](../../../rfx/api/_execute.py#L2709) / [FN :2732](../../../rfx/api/_execute.py#L2732) |
| nu-missing-vref | `_ports` lumped/passive FU/FN; wire U/N and FU/FN 12/36 | Per-step V_ref missing on N; wire forward DFT V_ref missing on N; N/A | [U lumped :2621](../../../rfx/simulation.py#L2621), [U wire :2579](../../../rfx/simulation.py#L2579) / [N :3092](../../../rfx/nonuniform.py#L3092) |
| port-time-record-missing | Lumped U/N and U/D, passive U/N; wire U/D 12/36; lumped/passive FU/FN | Lumped/passive run samples absent on both; distributed wire samples absent on D; lumped/passive V_port absent on FU (FN's third column is V_port); N/A | [U :1206](../../../rfx/runners/uniform.py#L1206), [FU :2621](../../../rfx/simulation.py#L2621) / [N :1935](../../../rfx/runners/nonuniform.py#L1935), [D :1614](../../../rfx/runners/distributed_v2.py#L1614) |
| uniform-wire-dft-record-missing | `_ports` wire U/N 12/36; lumped/passive FU/FN | Raw `wire_port_sparams` absent on U/FU; FU lumped data has a different container name; N/A | [U :1212](../../../rfx/runners/uniform.py#L1212), [FU :2719](../../../rfx/api/_execute.py#L2719) / [N :1934](../../../rfx/runners/nonuniform.py#L1934) |
| nu-lumped-dft-record-missing | `_ports` lumped/passive FU/FN | Raw `lumped_port_sparams` absent on FN, which exposes one-cell wire data; container-schema finding; N/A | [FU :2715](../../../rfx/api/_execute.py#L2715) / [FN :2732](../../../rfx/api/_execute.py#L2732), [NU construction :1356](../../../rfx/runners/nonuniform.py#L1356) |
| s-parameter-shape | `_ports` lumped/passive FU/FN; wire FU/FN 12/36 | `(2,)` vs `(1,1,2)`; N/A | [FU :2668](../../../rfx/api/_execute.py#L2668) / [FN :1356](../../../rfx/runners/nonuniform.py#L1356) |
| distributed-mode2d-broadcast | `_mode`, U/D | Broadcasting `float32[10,13,2]` into `[10,13,1]`; N/A | [U :134](../../../rfx/runners/uniform.py#L134) / [D :2573](../../../rfx/runners/_distributed_common.py#L2573) |
| graded-distributed-ports-refused | `_ports` lumped Ng/Dg; wire Ng/Dg 12/36 | Graded distributed port extraction refuses; N/A | [N :713](../../../rfx/runners/nonuniform.py#L713) / [Dg lumped :3175](../../../rfx/api/_execute.py#L3175), [Dg wire :458](../../../rfx/api/_preflight.py#L458) |
| graded-distributed-ntff-refused | `_ntff`, Ng/Dg | Public distributed feature guard refuses NTFF; N/A | [N :3181](../../../rfx/nonuniform.py#L3181) / [Dg :5083](../../../rfx/api/_execute.py#L5083) |

DFT first differing step (zero-based): **2**, component **ez**, normal axis **x**,
plane index **5**, coordinate **0.00390625 m**, frequency **5 GHz**, transverse
sample **(4,4)**. Both fields are `0.10673290491104126`; all 12 plane snapshots
are identical. Both post-update state counters are 3. Uniform stamps at
`5.585668646362896e-12 s`; NU at `3.723779097575264e-12 s`.
Observed accumulators at that sample:
U = `1.9567308114521592e-13 - 3.46933412716504e-14j`;
N = `1.9736659138611112e-13 - 2.3195031642999145e-14j`.
The diagnostic observes the real scan carry; timestamp expressions follow the
implementation lines above. Full step arrays and JSON witnesses are in
untracked `.s0-work/`.

The previous wire V discrepancy at 36 steps did not reappear with the corrected
driven-wire builder. The waveguide comparison passes after explicitly matching
PEC walls and modal aperture dimensions. Constructor conflicts are not retained
as findings. Per-cell measurements and timings live outside tracked files.

Distributed TFSF/waveguide refusal checks exercise the explicit admission gate.
The public API instead deliberately falls back to one device
([routing :5051](../../../rfx/api/_execute.py#L5051)); forced-dispatch recursion
is a test artifact and is not retained as a finding.

Record-check xfails before/after classifying the rebased main assembly errors (execution failures
also fail their dependent record groups):

| Category | 85ad0908 | This review |
|---|---:|---:|
| realized | 5 | 5 |
| probes | 5 | 5 |
| port_samples | 14 | 14 |
| port_dft | 9 | 9 |
| observers | 9 | 9 |
| objective | 0 | 0 |
| gradient | 0 | 0 |
| refusal | 0 | 20 |

Total: **42 → 62 checks**, **24 → 44 cells**, **13 → 14 causes** (excluding the separate
declared-length semantic note). The five realized-group xfails are execution
errors; none is a geometry or kernel-material mismatch.

The preceding comparator review re-measured all 14 generated flux/NTFF cells. Restoring per-leaf peaks
restores four H-channel fingerprints in the existing two run-flux findings:
`h1_dft` / `h2_dft` relative differences are `0.0578886856` / `0.0584336938`
at 12 steps and `0.0534573740` / `0.0546932149` at 36 steps, each versus `1e-4`.
Implementation locations are in the flux row above. Uniform/distributed NTFF
still passes with only its named Kahan residuals excluded: maximum physical-leaf
relative difference `1.06943511e-7` versus `1e-4`. Metadata remain exact.
The full matrix was not re-run in this review.

The rebased refusal review re-ran all 36 cells across the five affected PEC/MSL
rows and `_solver`: 40 record checks passed and 20 matched the assembly-error
finding. Every refusal stopped before scanning. The `_solver` builder now uses
PEC boundaries; its subgridded refusal names `solver='adi'` and passes, so the
former absorbing-boundary constructor conflict is not a finding.

S1 G0: the 20 PEC/MSL refusal cells previously assigned to
`refusal-message-runs-adi-material-gate` all produced strict XPASS once the ADI
material gate assembled with explicit empty PEC collectors (it reads eps/sigma
cells only; sheets and wires are refused by their own rows). Their finding entries
were removed; all 20 passed when re-run without xfail. Remaining findings: 42
checks, 24 cells, 13 causes.

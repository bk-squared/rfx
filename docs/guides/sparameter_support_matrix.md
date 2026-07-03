# S-Parameter Calculation Support Matrix

This matrix is the source-of-truth contract for user-facing S-parameter
calculation support across public rfx port-like APIs. It separates three
questions that are easy to conflate:

1. **Is the primitive a calibrated port?**
2. **Which API computes its S-parameters, if any?**
3. **Is that calculation claims-bearing, shadow, experimental, or unsupported?**

Machine-readable companion: `docs/guides/sparameter_support_matrix.json`.
Physics-evidence terminology follows
`docs/guides/physics_validation_evidence_rule.md`. The raw V/I dump replay
schema for E3 evidence is `docs/guides/sparameter_dump_replay.md`.

## Canonical result convention

Full S-matrices use `S[receiver_port, driven_port, frequency_index]` with
shape `(n_ports, n_ports, n_freqs)`. Frequency axes use shape `(n_freqs,)` in
Hz.

## Canonical user-facing rule

| Port / observable primitive | S-parameter API | Result schema | Evidence | Artifact + metric + envelope |
|---|---|---|---|---|
| `add_port(..., extent=None)` lumped port | `Simulation.run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | E2/E3/E4 partial | M13 analytic extractor oracles (`open` / `short` / `matched` / RLC, `max_abs_diff 7.91e-8 <= 2.20e-6`), M11 real raw V/I replay (`max_abs_diff 1.13e-7 <= 9.84e-7`), M14 three-case replay/passivity/reciprocity sweep (`max_column_power 0.971`, reciprocity diff `3.02e-7`), M33 narrow rfx/openEMS PEC-box magnitude comparator (`S11 max_mag_abs_diff 0.11224`, `S21 max_mag_abs_diff 0.00373`), M47 three-case rfx/openEMS PEC-box position sweep (`max case max_mag_abs_diff 0.11835`, `max case mean_mag_abs_diff 0.06466`), and M56 VESSL-parallel rerun of that sweep (`3/3` jobs completed and aggregate passed); broad calibrated-port E5 remains blocked |
| `add_port(..., extent=None)` lumped port | `Simulation.forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `ForwardResult.freqs` (S11 vectors only) | E0/E1 | AD/schema contract; physics claim inherits the lumped-port envelope |
| `add_port(..., extent=...)` wire port | `Simulation.run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | E2/E3/E4 partial + broad-E4-enabling envelope | `examples/crossval/05_patch_antenna.py` for probe-fed patch resonance, M32 generic patch/OpenEMS magnitude comparator (`max_mag_abs_diff 0.05318`, `mean_mag_abs_diff 0.02750` over 1.5--3.4 GHz), M12 real midpoint V/I replay (`max_abs_diff 7.82e-8 <= 9.80e-7`), M15 three-case replay/passivity/reciprocity sweep (`max_column_power 0.979`, reciprocity diff `1.24e-6`), and **M68 broad mesh/length openEMS envelope** (3 cases over `dx in [1, 2] mm`, wire length `[4, 8] mm`, `0.8--1.8 GHz`, `max_mag_abs_diff_across_cases 0.05212`); absolute S-matrix calibration convention still caveated |
| `add_port(..., extent=...)` wire port | `Simulation.forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `ForwardResult.freqs` (S11 vectors only) | E0/E1 | `tests/test_wire_port_sparams_forward.py`; uniform single-device AD path; nonuniform is shadow only |
| `add_msl_port(...)` | `Simulation.compute_msl_s_matrix(...)` | `MSLSMatrixResult.S`, `.freqs`, `.Z0`, `.beta`, `.port_names` | E5-narrow / eigenmode-blocked | M10 envelope report: thru-line `|S21|` / `Z0` slow gate, cv06b notch error `1.63%` and depth `-34.3 dB`, stored-openEMS smoke S11/S21 mean abs diffs `0.02502` / `0.02661`, M29 generic comparator artifact (`max_mag_abs_diff 0.05225`, `mean_mag_abs_diff 0.02664`, magnitude mode), and real raw 3-probe replay `max_abs_diff=0`; eigenmode is a falsified dead-end (fenced, `NotImplementedError`); the non-uniform graded-mesh lane RUNS via the laplace feed since PR #239 (internally gated — settled-S11 GPU physics gate `tests/test_msl_nu_sparam_gate.py`; external NU cross-solver validation still owed) |
| `add_waveguide_port(...)` | `Simulation.compute_waveguide_s_matrix(...)` | `WaveguideSMatrixResult.s_params`, `.freqs`, `.port_names`, `.port_directions`, `.reference_planes` | E5-broad-magnitude (analytic-Airy oracle; +E4-external WR-90 slab, magnitude-only) — committed + auditor-GREEN on a clean checkout | M4 envelope report: `waveguide_ports` gate `37 passed`; cv11 WR-90 empty/PEC-short/slab analytic + external references; M30 generic WR90/Palace magnitude comparator (`max_mag_abs_diff 0.0709`). **Committed broad-E5 (2026-06-15):** five WR-band flux envelopes vs analytic Airy (`tests/fixtures/waveguide_broad_e5/`, 20/20 cases `max_mag_abs_diff <= 0.0414`, VESSL 369367242914) + an rfx-vs-Palace-FEM broad-E4 comparison across empty/PEC-short/slab (`max 0.0707`); both pass `scripts/diagnostics/check_port_external_references.py` on a clean checkout, gated by `tests/test_waveguide_broad_e5_envelope_gates.py`. **Scope caveat:** the broad-E5 envelope uses an analytic-Airy oracle (magnitude-only) and the external E4 reference covers a single WR-90 band (magnitude-only, no multi-band or phase cross-solver gate). **Phase (T2.1, 2026-06-16):** an approximately convention-free TE10 propagation-phase witness now exists (`tests/test_waveguide_phase_gate.py`) — the measured probe/reference plane phase tracks an INDEPENDENT numpy-double analytic β to ~4° (broken extractors give 58–180°), sidestepping the cv11 143° cross-solver phase-convention problem. The modal-impedance factor cancels in the probe/ref ratio only to first order in the residual reflection (~6–15% CPML floor here), so it is approximately, not strictly, β-free; a companion test covers the `_shift_modal_waves` `step_sign` de-embed path. It is a committed gate but is **not yet wired into the broad-E5 auditor verdict** (the auditor stays magnitude-driven until a later T2 step). R5 note: Meep res-3/4 gives a non-physical PEC-short `|S11|>1`, so the converged Palace FEM reference is used for that geometry; rfx itself nails `|S11|=1.0000`. The multimode / T-junction lineage is **not committed** and remains shadow/deferred (not part of this rectangular single-mode broad-E5 claim). The **nonuniform (graded-dy) flux lane** now carries both committed legs — a broad-E5 analytic-Airy envelope (`tests/fixtures/waveguide_nu_broad_e5/`) and, since 2026-07-03, a broad-E4 external cross-check vs Palace FEM (`tests/fixtures/waveguide_nu_broad_e4/`, empty/PEC-short/slab, `max 0.070`, matching the uniform lane); its promotion from shadow to a claims-bearing NU row is a separate decision. |
| `add_waveguide_port(...)` | `Simulation.run(...)` | `Result.waveguide_sparams[name]` (`WaveguideSParamResult`) | E2 diagnostic | single-port diagnostic/calibrated per-port output, not the full multi-port S-matrix API |
| `add_coaxial_port(...)` | `Simulation.compute_coaxial_line_reflection(...)` | `CoaxialLineReflectionResult.s11`, `.freqs`, `.gamma`, `.status`, `.annulus_cells`, residual diagnostics | broad-E5 physics demonstrated — evidence artifacts NOT committed | M74 broad-E5 physics was demonstrated (analytic Γ envelope over short/open/matched + resistive 25/100 Ω loads, two characteristic impedances, mesh-resolution sweep; max `|Γ|` dev 0.037 ≤ 0.05) plus independent MEEP broad-E4 short/open comparison 4--12 GHz (`max |S11|` diff 0.063). However, all evidence artifacts live in gitignored `.omx/`, not committed to the repo. A clean-checkout audit (`check_port_external_references.py`) reports `coaxial_port` BLOCKED — re-validation is pending the validation-framework rework. Do not cite `broad_e5_passed` for coaxial until artifacts are committed and the auditor returns PASSED. |
| `add_coaxial_port(...)` | `Simulation.compute_coaxial_s_matrix(...)` (deprecated / experimental) | `CoaxialSMatrixResult.s_params`, `.freqs`, `.port_names`, `.port_faces`, `.reference_planes`, `.z_tem_ohm`, `.voltages`, `.currents`, `.status` | E2/E3 development path only | Older single-plane V/I closed-box path retained for compatibility. It can report non-physical `|S11| > 1` for lossless shorts and must not be used as the promoted coaxial claims surface. Route public coaxial reflection claims to `compute_coaxial_line_reflection(...)`. |
| `add_floquet_port(...)` | none promoted | none promoted | E2/E3-modal/slab-analytic / no-promoted-api | M18 synthetic specular-TE modal oracle, M20 real-FDTD Ex/Hy DFT-plane dump replay (`max S diff 2.23e-7`), M38 empty-space analytic-null comparison (`max_abs_diff 0.06067`, `mean_abs_diff 0.05306`), M44 homogeneous-slab analytic oracle (`8` cases x `3` frequencies, max power-balance error `4.44e-16`), and M49 rfx-FDTD slab/analytic magnitude comparison (`max_mag_abs_diff 0.06212`, `mean_mag_abs_diff 0.03209`); no RCWA/external crossval or promoted S-matrix API |
| `add_source(...)`, `add_polarized_source(...)` | none | `Result.time_series`, Harminv and field observables | not a port | field/resonance evidence is separate from impedance-defined S-parameters |
| `add_tfsf_source(...)` | none | field, flux, NTFF/RCS observables where supported | not a port | plane-wave observables, not port calibration |
| probes, DFT plane probes, flux monitors | none | field/flux observables | not a port | no port impedance or reference plane |

## Port selection: usage rule & validation ceiling

**`broad-E5` is not a universal goal.** A port's *appropriate* validation level is
set by its physics, not by a uniform ladder. Single-cell feeds (lumped, wire) are
not transmission lines, so there is no analytic line oracle to sweep a broad-E5
envelope against — **E4 is their natural ceiling, and meeting it means "validated
to ceiling", not "blocked from E5"**. Use each port only where its rule says, and
read its status against its `target_ceiling` (machine-readable in
`scripts/diagnostics/port_external_reference_requirements.json`, surfaced by the
auditor, locked by `tests/test_physics_gate_reporting.py::test_every_family_declares_target_ceiling_and_usage_rule`).

| Port | Use it when… | Target ceiling | Status today |
|---|---|---|---|
| `add_waveguide_port` | rectangular waveguide TE/TM, uniform Cartesian lane | **broad-E5** | ✅ broad-E5 achieved |
| `add_msl_port` | matched / thru-line / notch microstrip (quasi-TEM) | broad-E5-regime-restricted | matched-regime only; strong-reflector \|S11\| has a ~0.16-0.22 staircase-Z0 floor (mesh-conv #183); eigenmode port is a falsified dead-end |
| `add_coaxial_port` | coax line reflection (forward); **not** in an AD/optimization loop | broad-E5-needs-differentiable-api | physics demonstrated; blocked — evidence uncommitted + numpy extractor fails the AD moat |
| `add_port(extent=None)` lumped | sub-cell lumped R/L/C/RLC element | **E4-natural-ceiling** | E4 is the ceiling (not a transmission line); validate to E4, do not chase E5 |
| `add_port(extent=...)` wire | one-cell transverse probe/wire feed (magnitude-only) | **E4-natural-ceiling** | use `add_msl_port` for a real line; E4 is the ceiling |
| `add_floquet_port` | periodic structure, **broadside (θ=0) only** | broad-E5-structural-partial | off-broadside needs a split-field complex-Bloch rewrite (structural ceiling) |
| generalized planar (stripline/CPW/launch) | — | needs-implementation | no API yet; would inherit the MSL-class ceiling once built |

## Lane and validation boundaries

### Lumped `add_port(..., extent=None)`

- **Evidence status:** E2/E3/E4 partial. M13 covers closed-form extractor oracles
  for open, short, matched, resistive, capacitor, inductor, series-RLC, and
  parallel-RLC loads (`max_abs_diff 7.91e-8 <= 2.20e-6`). Existing tests cover
  simple physical invariants such as full reflection in a PEC cavity, and M11
  adds a real two-port raw V/I dump replay of the production extractor/sign
  convention (`max_abs_diff 1.13e-7 <= 9.84e-7`). M14 adds a small
  uniform-Yee replay/passivity/reciprocity sweep over three two-port
  geometries (`max_column_power 0.971`, reciprocity diff `3.02e-7`). M33 adds
  a narrow rfx/openEMS two-port PEC-box magnitude comparison (`S11
  max_mag_abs_diff 0.11224`, `S21 max_mag_abs_diff 0.00373`). M47 extends that
  external lane to three PEC-box port-position cases (`max case
  max_mag_abs_diff 0.11835`, `max case mean_mag_abs_diff 0.06466`). M56
  reruns those three cases as independent VESSL jobs (`3/3` completed) and
  aggregates the remote case artifacts with the same aggregate metrics. This
  proves the slow-sim parallel execution path, but it is still not broad
  calibrated-port E5 evidence because matched/open/short/load external
  coverage and a larger envelope remain missing. M50/M56 provide VESSL case
  YAML generation, submission, and artifact aggregation for the M47
  lumped/openEMS slow sweep; they do not widen the physics envelope beyond the
  narrow PEC-box comparison.
- **API:** `run(compute_s_params=True)` for a full S-matrix;
  `forward(port_s11_freqs=...)` for differentiable S11 vectors.
- **Promotion requirement:** broader external cross-solver evidence and a
  mesh/frequency/geometry envelope report before broad E5 claims. The M13
  analytic oracle covers extractor algebra only; M11 covers one real dump; M14
  is an internal uniform-Yee sweep, and M33/M47/M56 are still narrow external
  PEC-box comparisons.
- **Known limits:** nonuniform lumped-port S-parameter extraction is not
  promoted or wired as a supported calculation path.

### Wire `add_port(..., extent=...)`

- **Evidence status:** E2/E3/E4 partial for probe-fed resonance/field
  workflows, plus a broad-E4-enabling mesh/length envelope. M12 adds a real
  two-port midpoint V/I replay of the current wire extractor convention
  (`max_abs_diff 7.82e-8 <= 9.80e-7`), and M15 adds a small uniform-Yee
  replay/passivity/reciprocity sweep over three two-port geometries
  (`max_column_power 0.979`, reciprocity diff `1.24e-6`). M32 adds a narrow
  crossval05 patch/OpenEMS S11 magnitude comparison artifact
  (`max_mag_abs_diff 0.05318`, `mean_mag_abs_diff 0.02750`, 1.5--3.4 GHz).
  M68 adds a broad mesh/length openEMS envelope across three PEC-cavity
  two-port wire-port cases (`dx in [1, 2] mm`, wire length `[4, 8] mm`,
  `0.8--1.8 GHz`, `max_mag_abs_diff_across_cases 0.05212`, all 3 cases
  passed). Absolute S-matrix calibration convention remains caveated.
- **Shadow lane:** nonuniform wire-port extraction is retained and covered by
  regression tests, but it is not the public physics-validated baseline.
- **API:** `run(compute_s_params=True)` for a full S-matrix;
  `forward(port_s11_freqs=...)` for differentiable S11 vectors on the uniform
  single-device path.
- **Known limits:** a wire port is a one-cell-transverse feed model. Use
  `add_msl_port(...)` for the specialized full-strip microstrip-line
  calculator when that model is appropriate. Promotion to broad calibrated
  S-parameter claims requires external reference gates, broader
  mesh/frequency/geometry sweeps, and a resolved or explicitly bounded wire
  calibration convention. M15 is an internal sweep; M32 is external
  E4-enabling evidence for the narrow patch resonance lane, not broad wire-port
  E5.

### Microstrip-line `add_msl_port(...)`

- **API:** `compute_msl_s_matrix(...)`.
- **Result:** `MSLSMatrixResult` with full `S`, frequency grid, extracted `Z0`,
  extracted `beta`, and port names.
- **Evidence status:** E5-narrow / eigenmode-blocked. The internal thru-line gate in
  `tests/test_msl_port_integration.py` checks `Z0`, `|S11|`, and `|S21|`;
  cv06b exercises an analytic quarter-wave notch (notch error `1.63%`, depth
  `-34.3 dB`, median `Re(Z0)=48.6 Ω` — a **single `dx=h_sub/2` run**; the
  committed cv06b gate is intentionally looser, error `<15%` / depth `<-10 dB`,
  pending an external cross-solver rerun); and the
  stored-openEMS comparison report gives narrow thru-line magnitude smoke
  evidence (`S11` / `S21` mean abs diffs `0.02502` / `0.02661`); and M10 real
  raw 3-probe replay matches production with `max_abs_diff=0`. Broad all-mode
  E5 remains blocked: the `mode="eigenmode"` source/extractor is a falsified
  dead-end (fenced), and the non-uniform lane (running via the laplace feed
  since PR #239) still lacks external cross-solver validation.
- **Hard failures:** non-uniform profiles with `mode="eigenmode"` (the
  laplace/uniform feed runs on non-uniform meshes since PR #239), SBP-SAT
  subgridding, ADI, TFSF, and
  mixed port families are rejected instead of being silently included in
  `Result.s_params`.

### Rectangular `add_waveguide_port(...)`

- **Full S-matrix API:** `compute_waveguide_s_matrix(...)`.
- **Single-port run output:** `run(...)` can populate
  `Result.waveguide_sparams[name]`; use the compute API for multi-port
  scattering matrices.
- **Evidence status:** E5-narrow for the documented uniform-Yee rectangular
  waveguide envelope. Empty-guide, PEC-short, passivity, Airy/reference-plane,
  field-dump, and external-solver artifacts exist. Current exact gates are:
  - `tests/test_waveguide_port_validation_battery.py`: empty-guide max
    `|S11| < 0.02`, passivity max column power `< 1.02`, symmetric-obstacle
    reciprocity mean error `< 0.01`, PEC-short `0.99 <= min(|S11|)` and
    `max(|S11|) < 1.03`.
  - `examples/crossval/11_waveguide_port_wr90.py`: empty guide `|S11|`
    gate `0.02`, empty `|S21|` gate `0.03`, PEC-short phase gate `15°`,
    slab S11/S21 magnitude gates `0.10` / `0.07`, phase gate `60°` with
    `|S_ref| >= 0.30` mask, and complex-S envelope `0.30`.
- **Known limits:** full S-matrix extraction requires at least two waveguide
  ports; nonuniform extraction is restricted to `normalize=True` and
  single-mode ports; `normalize=True` is not implemented for multi-mode.
- **Recommended extraction mode: `normalize="flux"`.** Power-flux
  (`extract_waveguide_s_matrix_flux`, exposed via
  `compute_waveguide_s_matrix(normalize="flux")`) combines a reference run's
  Poynting flux and the device run's modal V/I phase, eliminating both
  the Z_TE impedance-mismatch error of `normalize=False` (~3 % at
  `dx/λ=0.07`) and the round-trip dispersion error in the
  `normalize=True` diagonal formula (±10–20 % |S11| swings). 2026-05-26
  cross-band verification on WR-28 Ka / WR-62 Ku / WR-15 V / WR-340 S /
  WR-10 W slabs (εr=2 + εr=4, dx tuned per recipe below):

  | Band | εr | `|S11|` diff (False) | `|S11|` diff (flux) | flux unitarity |
  |------|----|----------------------|---------------------|-----------------|
  | WR-28 Ka  | 2 | 0.048 | 0.040 | 1.000 – 1.005 |
  | WR-28 Ka  | 4 | 0.047 | 0.012 | 0.999 – 1.000 |
  | WR-62 Ku  | 2 | 0.044 | 0.011 | 1.000 – 1.000 |
  | WR-62 Ku  | 4 | 0.024 | 0.008 | 1.000 – 1.000 |
  | WR-15 V   | 2 | 0.049 | 0.017 | 1.000 – 1.000 |
  | WR-15 V   | 4 | 0.046 | 0.041 | 1.000 – 1.000 |
  | WR-340 S  | 2 | 0.043 | 0.006 | 1.000 – 1.001 |
  | WR-340 S  | 4 | 0.027 | 0.005 | 1.000 – 1.001 |
  | WR-10 W   | 2 | 0.034 | 0.010 | 1.000 – 1.000 |
  | WR-10 W   | 4 | 0.037 | 0.009 | 1.000 – 1.000 |

  flux mode brings `|S11|` to 0.005–0.041 (1/2 to 1/10 of the 0.05
  broad-E5 threshold) and unitarity to ±0.05 %p (lossless slab Yee
  residual only). Cost: 2 × N_ports FDTD runs (same as `normalize=True`).
- **Setup recipe for broad-E5 across rectangular WR bands.** Pair the
  `normalize="flux"` extraction with the following geometry/mesh choices:
  - `CPML_LAYERS >= 20` for clean grading polynomial (cv11 cell-count
    threshold). Layer count drives noise floor monotonically:
    `12 -> 0.118, 24 -> 0.056, 36 -> 0.039, 48 -> 0.031` (empty-guide
    `|S11|` with `normalize=False`, WR-28).
  - `cells/lambda_min >= 60` at the highest validation frequency, and
    `cells/lambda_d_min >= 60` in-dielectric for εr-loaded geometries.
  - port-to-discontinuity distance `>= 1 lambda_g_max` for evanescent
    higher-order mode decay.
  - **slab L chosen so Airy `|S11|` stays above 0.1 across the band**
    (`2 β_d L < π`, no first Fabry-Perot null inside). Pre-compute Airy
    `|S11|` range before launching the sweep; e.g. WR-28 passes with
    L=4 mm but WR-62 with L=10 mm has |S11| min 0.03 (below noise
    floor) — shorten to L=3 mm to keep |S11| > 0.4.
  - dx values that divide L into integer cells (staircase rasterization
    quantization amplified by `sqrt(eps_r)` in the Fabry-Perot phase).
  Investigation trail: `docs/research_notes/20260526_waveguide_flux_silver_bullet.md`.

### Coaxial and Floquet ports

- `add_coaxial_port(...)` has two public-facing calculation paths.  Use
  `Simulation.compute_coaxial_line_reflection(...)` for the coaxial
  transmission-line reflection path.  M74 broad-E5 physics was demonstrated
  (analytic Γ envelope over short/open/matched and resistive loads, two
  characteristic impedances, mesh-resolution sweep) plus an independent MEEP
  broad-E4 short/open comparison over 4--12 GHz.  However, all evidence
  artifacts live in gitignored `.omx/`, not committed; a clean-checkout audit
  (`check_port_external_references.py`) reports `coaxial_port` BLOCKED.
  Re-validation is pending the validation-framework rework.  Do not cite
  `broad_e5_passed` for coaxial until artifacts are committed and the auditor
  returns PASSED.
- `Simulation.compute_coaxial_s_matrix(...)` is retained for backward
  compatibility but is deprecated as the older single-plane V/I path.  Its
  closed-box setup can report non-physical `|S11| > 1` for a lossless short,
  so it is not the promoted coaxial S-parameter claims surface.
- `add_floquet_port(...)` is an experimental Floquet/Bloch excitation surface.
  M18 adds a synthetic specular-TE modal bookkeeping oracle, and M20 adds
  one small broadside real-FDTD Ex/Hy DFT-plane dump replay. M38 compares that
  empty-space broadside diagnostic against the analytic null-reflection oracle
  (`max_abs_diff 0.06067`, `mean_abs_diff 0.05306`). M44 adds a non-empty
  homogeneous dielectric-slab analytic oracle for the broadside zero-order
  Floquet limit (`8` cases x `3` frequencies, max lossless power-balance error
  `4.44e-16`). M49 runs a narrow non-empty rfx-FDTD slab case, decomposes raw
  Ex/Hy DFT-plane phasors with an independent NumPy post-processor, and
  compares S11 magnitude against the analytic slab reflection (`max_mag_abs_diff
  0.06212`, `mean_mag_abs_diff 0.03209` over three frequencies). M49 is still
  only E2/E3-enabling: it is not RCWA/external full-wave evidence, not a
  calibrated reference-plane envelope, and not E5. It is still not a promoted
  S-parameter calculator in the public contract; no RCWA or external full-wave
  reference is claimed.

### Future generalized planar ports

Stripline, coplanar waveguide (CPW), and microstrip-to-coax launch ports are
planned surfaces, not promoted S-parameter APIs. The accepted implementation
paths are either a generalized MSL-style quasi-TEM de-embedding flow or a true
2-D eigenmode source/extractor. M39 adds quasi-TEM analytic planning oracles
for representative microstrip, symmetric stripline, CPW, and
microstrip-to-coax-launch proxy geometries (`Z0` range 54.04--61.94 Ω with
finite positive `beta` checks). M45 extends that planning lane with an
analytic quasi-TEM parameter-sweep envelope template (`8` sweeps / `24` sweep
rows) across width, permittivity, and CPW strip-width trends. Promotion still
requires implementation, independent raw V/I dump replay, external cross-solver
evidence per family, and a stated mesh/frequency/substrate-cell envelope.

The broad-E5 external/reference backlog is also machine-readable in
`scripts/diagnostics/port_external_reference_requirements.json` and audited by
`scripts/diagnostics/check_port_external_references.py`. That audit is a
completion gate: it cross-checks this support matrix so every current or
planned port family is tracked, and internal replay/oracle artifacts do not
satisfy broad E5 until the relevant external/reference shard is present and
marked broad-E5 complete. Each required family also has a VESSL shard YAML for
parallel external-reference validation. The lumped, wire, and coaxial shards
now run their narrow openEMS comparison builders before emitting the family
blocker report; the Floquet shard runs its current modal oracle, real-FDTD
empty-space analytic-null comparison, and M49 rfx-FDTD slab/analytic comparison
before reporting the remaining RCWA/external blocker; the generalized-planar
shard runs the M39 quasi-TEM planning oracles before reporting the remaining
implementation/external blockers. None of these shard specs may be interpreted
as completed broad-E5 evidence until the family manifest status is
`broad_e5_passed`. The result
JSONs from those shards are aggregated by
`scripts/diagnostics/check_port_external_shard_results.py`, which is a separate
execution-evidence gate from YAML/manifest coverage. M58/M60 exercised the
non-lumped shard set on VESSL after the no-editable-install/bootstrap fixes;
all seven required family result JSONs are now present in that execution root,
but every family result is still `blocked` because broad-E4 and broad-E5
envelope evidence is missing. The expected YAML, diagnostic pre-run command,
output directory, and shard result JSON wiring is reported by
`scripts/diagnostics/build_port_external_shard_execution_manifest.py`; that
manifest is orchestration evidence only, not physics evidence. External
solver S-matrices should be compared through `scripts/diagnostics/compare_sparameter_reference.py`
where possible so family-specific shards share the same interpolation, term
selection, tolerance, and report schema before any manifest promotion. The
M47 lumped/openEMS sweep can also be split into per-case VESSL jobs with
`scripts/diagnostics/build_lumped_openems_parallel_plan.py` and then aggregated
with `scripts/diagnostics/build_lumped_openems_sweep_comparison.py
--case-artifact-root`; M56 confirms this path on VESSL for three cases. This is
a scheduling path for slow external simulations, not a physics-evidence
shortcut or a broad-E5 promotion. The remaining broad-E5 blockers are also
decomposed by `scripts/diagnostics/report_rf_e5_blocker_ladder.py`, which
orders missing API, raw-dump replay, solver dependency, external-reference,
broad-E4, and E5-envelope stages per family. That ladder is planning/audit
evidence only; it must not be used to promote a family without the referenced
physics artifacts. The
external-reference manifest has an `external_comparison_artifacts` list per
family; `broad_e5_passed` entries are rejected unless at least one listed
comparison artifact exists and reports `status=passed` with an E4/E4-enabling
evidence level, and at least one listed comparison artifact is a **broad E4**
external comparison rather than an E4-enabling/narrow fixture. The manifest also
has a `broad_e5_envelope_artifacts` list per
family; `broad_e5_passed` is rejected unless at least one listed envelope
artifact exists, reports `status=passed`, uses an unblocked E5 evidence level
not labeled narrow/enabling/partial/experimental/shadow, and states a broad
mesh/frequency/geometry scope. Solver availability is audited separately by
`scripts/diagnostics/check_external_solver_dependencies.py`; the audit checks
both executable discovery and real Python importability so ABI-broken modules
are blockers. Availability reports are blocker diagnostics only and must not be
counted as E4/E5 physics evidence.

### Non-port observables

`add_source(...)`, polarized sources, TFSF, point probes, DFT plane probes, and
flux monitors are valid observables/excitations, but they do not define a port
impedance or S-matrix reference. They must not be documented as substitutes for
ports.

## Loud rejection policy

Explicit S-parameter requests outside the table must fail with actionable
errors. In particular:

- `run(compute_s_params=True)` is only for `add_port(...)` lumped/wire ports.
- `compute_msl_s_matrix(...)` is only for `add_msl_port(...)` simulations.
- `compute_waveguide_s_matrix(...)` is only for waveguide-port simulations.
- `compute_coaxial_line_reflection(...)` is only for the documented one-port
  coaxial transmission-line reflection envelope.
- `forward(port_s11_freqs=...)` is only for `add_port(...)` lumped/wire ports
  on the uniform single-device differentiable path.
- Deprecated `compute_coaxial_s_matrix(...)`, Floquet, sources, TFSF, probes,
  and flux monitors must not return `None` as if S-parameters were optional
  output after the user explicitly requested S-parameters.

Before an expensive run, use the port-family routing preflight:

```python
sim.preflight_sparameters(calculator="run")        # run(compute_s_params=True)
sim.preflight_sparameters(calculator="forward")    # forward(port_s11_freqs=...)
sim.preflight_sparameters(calculator="msl")        # compute_msl_s_matrix(...)
sim.preflight_sparameters(calculator="waveguide")  # compute_waveguide_s_matrix(...)
```

The method returns actionable issue strings without running FDTD. Use
`strict=True` when a setup script should fail immediately instead of collecting
warnings.

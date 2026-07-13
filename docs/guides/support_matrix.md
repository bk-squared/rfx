# rfx Support Matrix

This page describes what can be used now. It does not treat a passing unit test
as RF validation, and it does not infer support for one combination from support
for another.

Status terms:

- **supported** — documented for routine use within the stated limits
- **limited** — documented only for the exact geometry, mode, mesh, or observable
  restrictions listed here
- **experimental** — the code path runs and has regression coverage, but the RF
  evidence is not sufficient for a documented result
- **not documented** — code may exist, but there is no current public workflow
- **unsupported** — the requested combination should raise an actionable error

## Current documented baseline

The general baseline is a uniform Cartesian Yee grid with `pec`, `cpml`, or
`upml` where the selected feature is listed as supported. Documented sources
include point/current sources and the port families listed in the
[S-parameter support matrix](sparameter_support_matrix.md).
Documented observables include time-series probes, flux monitors, Harminv
resonances, benchmarked NTFF cases, and the explicitly listed port results.

| Feature | Status | Current limits |
|---|---|---|
| Uniform Cartesian Yee workflows | **supported** | Primary documented grid; each source and observable still has its own restrictions. |
| Rectangular-waveguide S-matrix | **limited** | Uniform single-mode magnitude results are validated across the documented WR-band tests. Use `compute_waveguide_s_matrix(...)`; phase, junction, and nonuniform results have narrower evidence. |
| Microstrip-line S-matrix | **limited** | Use `compute_msl_s_matrix(...)` with the laplace/quasi-TEM model. The external notch comparison is characterized rather than a tight cross-solver match. Eigenmode is unsupported. |
| Lumped and wire-port S-parameters | **limited** | Use the calculation API and evidence limits for the selected port family. Do not interpret these discrete feed models as calibrated transmission-line modes. |
| Coaxial-line reflection | **limited** | `compute_coaxial_line_reflection(...)` requires float32 precision, a three-dimensional second-order uniform Yee grid, CPML tokens on all six boundary faces with positive thickness on both z faces, `cpml_axes="z"`, no periodic axes, and exactly one `face="top"` coaxial port. It is documented over the stated frequency, termination, impedance, and mesh ranges, not as a general coaxial-network solver. |
| Nonuniform meshes | **limited by feature** | See the table below. Support for one observable does not imply support for another. |
| SBP-SAT/subgridding | **not documented** | No current public workflow. |
| ADI | **not documented** | Not part of the public correctness baseline. |
| Distributed execution | **not documented** | Execution scaling is separate from numerical correctness. |
| Floquet/Bloch S-parameters | **experimental** | Excitation and diagnostic helpers exist, but there is no documented high-level Floquet S-parameter API. |

Materials in the general baseline are isotropic linear materials, conductive
materials, and the dispersive subsets covered by the applicable validation
tests. Optimization examples are evidence for their named objective only; they
do not validate every result produced by the same simulation.

## S-parameter API routing

- Lumped and wire `add_port(...)` simulations use
  `run(compute_s_params=True, s_param_freqs=...)` for full matrices.
- The same port family uses `forward(port_s11_freqs=...)` only for uniform,
  single-device S11 vectors.
- `add_msl_port(...)` uses `compute_msl_s_matrix(...)`.
- `add_waveguide_port(...)` uses `compute_waveguide_s_matrix(...)` for a full
  multi-port matrix. `run()` exposes only per-port `waveguide_sparams`.
- A float32, three-dimensional, second-order uniform Yee simulation with
  CPML tokens on all six boundary faces, positive CPML on both z faces,
  `cpml_axes="z"`, no periodic axes, and exactly one
  `add_coaxial_port(..., face="top")` uses
  `compute_coaxial_line_reflection(...)` for the documented result.
  The method builds its own line, source, DFT planes, and termination; do not
  register separate geometry, RLC elements, monitors, or termination helpers.
  It derives the axial layout internally. Use `feed_impedance=` for the feed and
  `dut_impedance=` only with `termination="matched"`.
  `probe_count` must be an integer of at least three, and every requested plane
  must fit between the DUT and source.
  `compute_coaxial_s_matrix(...)` is deprecated and
  experimental.
- Sources, TFSF, probes, and flux monitors do not define a port impedance or
  reference plane and therefore do not produce S-parameters.

The exact schemas, metrics, and RF evidence are in
`docs/guides/sparameter_support_matrix.md`. Evidence levels are defined in
`docs/guides/physics_validation_evidence_rule.md`.

## Nonuniform-mesh classification

Nonuniform support is determined per combination. In particular, a graded mesh
running successfully is not evidence that an S-parameter, flux, or far-field
result is accurate.

| Combination | Status | What is documented |
|---|---|---|
| Periodic/Floquet port + nonuniform mesh | **unsupported** | Preflight or the calculation must fail. |
| NTFF + graded z | **limited** | A short-dipole directivity case agrees with the 1.76 dBi theory within about 0.05 dB (`tests/test_farfield_nonuniform.py`). Other source and geometry combinations need separate validation. |
| DFT plane or full-plane flux + graded z | **experimental** | The calculation runs, but no general RF-accuracy statement is documented. |
| TFSF + graded z | **experimental** | Only normal incidence along `+x` or `-x` (`angle_deg=0`) runs. Oblique incidence and incidence along `+z` or `-z` raise. |
| Rectangular-waveguide S-matrix + nonuniform transverse mesh | **experimental** | Single-mode `normalize=True` and `normalize="flux"` run. Analytic fixtures cover grading ratios 1--3 and relative permittivity 2 and 4. A passed Palace magnitude comparison covers `normalize="flux"`, a graded-`dy` ratio of 2, WR-90 empty/PEC-short/dielectric-slab cases, and 8.2--12.4 GHz (`max_mag_abs_diff=0.07009`). Other profiles, bands, phase, multimode operation, and arbitrary junctions are not validated. `eps_override` and `sigma_override` differentiation is available only with `normalize="flux"`; only `eps_override` has a nonuniform AD-vs-FD regression test. Neither AD check establishes RF accuracy. |
| Lumped-port S-parameters + nonuniform mesh | **unsupported** | The S-parameter request must fail. |
| MSL S-matrix + nonuniform mesh | **experimental** | `mode="laplace"` and `mode="uniform"` have internal settled-S11 regression coverage only. There is no external nonuniform comparison. `mode="eigenmode"` raises. |
| Coaxial port + nonuniform mesh | **unsupported** | The request must fail. |
| Lumped RLC update + nonuniform mesh | **limited** | R/L/C ADE elements participate in the field update. Nonuniform S-parameters and component-value AD are not documented. |
| Volumetric PEC scatterer + nonuniform waveguide | **experimental** | The device/reference handling is regression-tested, but no RF validation is documented for arbitrary iris, post, septum, branch, or T-junction geometries. |

## Interpreting output and warnings

- A preflight pass checks compatibility; it is not a convergence study or an
  external-solver comparison.
- A warning or reliability mask identifies a result that should be excluded or
  investigated. Its absence is not an accuracy guarantee.
- Touchstone, HDF5, CSV, JSON, and plotting helpers preserve or display a result;
  they do not change its support status.
- Reports should state the API, source or port family, mesh type, frequency and
  geometry range, normalization, git SHA, and the cited comparison metric.
- Before using a result outside a listed range, repeat mesh/time convergence and
  compare against an analytic or independent solver reference suitable for that
  configuration.

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
`upml` where the selected feature is listed as supported. **Boundary caveat
(measured 2026-07-20, #403):** for open-domain absorption prefer `cpml` — rfx's
`upml` reflects ~25 dB more than `cpml` at the same layer count (measured floor
−43 dB vs −68 dB at 2 GHz / 8 layers; the UPML parallel-E component is attenuated
only indirectly via curl coupling, `rfx/boundaries/upml.py`). `upml` still
absorbs (interior energy decays), but use `cpml` where reflection floor matters.
Documented sources
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
| ADI | **experimental** | Public workflow exists for the Zheng–Chen–Zhang ADI lane. A lossless 3D PEC-cavity eigenfrequency is within the documented 2% gate at `adi_cfl_factor=2`; factor 5 is stable but outside that accuracy envelope. Use explicit Yee for claims-bearing 3D physics; no production stiff-mesh speedup is validated. |
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
| `boundary="upml"` + nonuniform mesh | **unsupported** | The run must fail. The non-uniform runner has no UPML implementation — its absorber dispatch keys on the CPML layer count and never reads the boundary type — so this combination was accepted and silently ran CPML while `sim._boundary` still reported `upml` (issue #680). Use `boundary="cpml"` on a graded mesh. |
| NTFF + graded z | **limited** | A short-dipole directivity case agrees with the 1.76 dBi theory within about 0.05 dB (`tests/test_farfield_nonuniform.py`). Other source and geometry combinations need separate validation. |
| DFT plane or full-plane flux + graded z | **experimental** | The calculation runs, but no general RF-accuracy statement is documented. |
| TFSF + graded z | **experimental** | Only normal incidence along `+x` or `-x` (`angle_deg=0`) runs. Oblique incidence and incidence along `+z` or `-z` raise. |
| Rectangular-waveguide S-matrix + nonuniform transverse mesh | **experimental** | Single-mode `normalize=True` and `normalize="flux"` run. Analytic fixtures cover grading ratios 1--3 and relative permittivity 2 and 4. A passed Palace magnitude comparison covers `normalize="flux"`, a graded-`dy` ratio of 2, WR-90 empty/PEC-short/dielectric-slab cases, and 8.2--12.4 GHz (`max_mag_abs_diff=0.008529`, improved from `0.07009` when the lane's absorber stopped being 0.33 of a guide wavelength — #576). Other profiles, bands, phase, multimode operation, and arbitrary junctions are not validated. `eps_override` and `sigma_override` differentiation is available only with `normalize="flux"`; only `eps_override` has a nonuniform AD-vs-FD regression test. Neither AD check establishes RF accuracy. |
| Lumped-port S-parameters + nonuniform mesh | **unsupported** | The S-parameter request must fail. |
| MSL S-matrix + nonuniform mesh | **experimental** | `mode="laplace"` and `mode="uniform"` have internal settled-S11 regression coverage only. There is no external nonuniform comparison. `mode="eigenmode"` raises. |
| Coaxial port + nonuniform mesh | **unsupported** | The request must fail. |
| Lumped RLC update + nonuniform mesh | **limited** | R/L/C ADE elements participate in the field update. Nonuniform S-parameters and component-value AD are not documented. |
| Multi-band graded mesh (N fine bands per axis, ratio <= 1.4) | **limited** | The MESH ITSELF is documented — not any observable computed on it. Explicit per-axis profiles with any number of fine bands (small-large-small-large included) and every adjacent cell ratio <= 1.4 are covered by the witness battery below. Read the scope statement in "Multi-band graded mesh" before quoting this row: absorber-adjacent grading is EXCLUDED, and `dt` is unchanged (global min-cell CFL). |
| Volumetric PEC scatterer + nonuniform waveguide | **experimental** | The device/reference handling is regression-tested, but no RF validation is documented for arbitrary iris, post, septum, branch, or T-junction geometries. |

### Multi-band graded mesh

**What this row covers.** An explicit `dx_profile` / `dy_profile` /
`dz_profile` vector holding N fine bands per axis, in any order —
fine-coarse-fine-coarse-fine and other small-large-small-large patterns
included — with **every adjacent cell ratio <= 1.4**, abrupt (a single step
at the cap) or smoothly ramped. Before this row, only a single fine band
was documented.

**The cap is 1.4.** Ratios above it still construct and run; what is lost
is the accuracy class below, not stability. `Simulation(...)` warns above
1.4 and preflight adds `nu_grading_ratio_beyond_validated_cap`, both
advisory. 1.4 matches the commercial default (Tidy3D `max_scale`).

**Evidence** (SPEC-01 lane, tracker #780; pre-declaration
`docs/design_notes/20260829_spec01_multiband_predeclaration.md` — every
window below was committed before its measurement; harness and raw results
in `validation/research/multiband_nu/`; regression packaging in
`tests/test_multiband_nu_envelope.py`):

| Witness | What was measured | Result |
|---|---|---|
| F-S1 stability/energy (`results/w1_pa_1d.json`) | Remis-class dual-cell discrete energy over 10^6 steps, 1-D lossless multi-band, r in {1.0 control, 1.1, 1.2, 1.4, 1.5, 2.0} abrupt + {1.4, 2.0} smooth | drift <= 2.9e-6 relative on every arm, no growth trend, against the pre-declared float-accumulation envelope 1.19e-3 at 10^6 steps |
| F-S1 3-D (`results/w1_pb_full_gpu.json`) | Same functional, 3-D PEC box, 10^6 steps on GPU (VESSL run 369367256892) | end drift +4.7e-9 (r=1.4) and +1.3e-7 (r=2.0), bounded random walk |
| F-S2 per-transition reflection (`results/w2_w3.json`) | Gated two-run differencing against an exact discrete scattering-chain model, 10 GHz, 30 cells/wavelength in the fine band | r=1.4 abrupt -53.9 dB measured vs -54.0 dB modelled (r=1.1 -67.1 dB, r=1.2 -60.7 dB); every in-envelope arm inside its pre-declared window |
| F-S3 round-trip amplitude (`results/w2_w3.json`) | Net amplitude drift across a symmetric rise-and-fall traversal (the Christ amplification/attenuation asymmetry) | \|T\|-deviation <= 7.5e-6 for r <= 1.4, against a 3e-4 floor; r=1.0 null control 2.4e-6 |
| F-S4 convergence order (`results/w4r2_analytic_cavity.json`) | Resonance error vs the ANALYTIC TE101 of an empty PEC cavity, multi-band z profile at the cap ratio r=1.4 vs a uniform-fine control, four scales | p_multiband = p_uniform = 1.95 — 2nd-order supraconvergence preserved (Monk & Suli 1994; Li & Shields 2016); multi-band adds <= 1e-6 relative at matched scale |
| F-S5 differentiability (`results/w5_ad.json`) | `jax.grad` of a multi-band profile observable vs central FD, dominant cells | worst dominant-cell error 1.1e-4 relative (f32 path) and 1.7e-4 (x64 context), inside the existing NU AD convention (15 %) |
| Revert-proof (`results/revert_proof.json`) | Two deliberate defect injections (one witness weight, one solver transition coefficient) | baseline f64 drift 2.1e-16; corrupted witness 5.8e-3, corrupted solver transition coefficient 7.9e-3 — the witness does fire on the defect family it guards |

**Exclusions — these are NOT covered by the row.**

- **Grading must not reach the absorber.** Every witness above ran
  PEC-closed with `cpml_layers = 0`. Nothing here says how a grading
  transition interacts with a CPML/UPML face, and the absorber pad
  replicates the outermost interior cell, so a transition in that runway
  changes the discrete medium in the boundary-normal direction exactly
  where the absorber begins — the documented PML breakdown class.
  Preflight emits `nu_grading_reaches_absorber` when an axis grades within
  the absorbing face's own layer count. Keep that runway uniform, or close
  the face with PEC/PMC.
- **`dt` is unchanged: still the global min-cell CFL** (0.99 of it,
  `rfx.nonuniform.make_nonuniform_grid`). Adding fine bands buys cell
  count, not time steps — the cost is the finest cell everywhere in time.
  Recovering that is not in this row and is not implemented; domain-wise
  `dt` and local time stepping are explicitly not pursued (late-time
  interface instability, Xiao et al. TAP 55(7):1981, 2007).
- **Ratios above 1.4** are advisory-flagged, not validated: r=1.5 measured
  -51.6 dB and r=2.0 -43.9 dB per transition on the same fixture, both
  consistent with the chain model but outside the claimed envelope.
- **Grading in-plane and in z simultaneously** was exercised only in the
  3-D energy witness; no observable-accuracy statement covers it.

**Honest scope — what the witnesses do and do NOT establish.** They are
statements about the mesh and the solver on it: the scheme conserves its
discrete energy, each transition reflects at the modelled level, a
symmetric traversal does not drift in amplitude, the global order stays 2,
and the gradient path is intact. They are NOT a statement that any
S-parameter, flux, far-field, or port result computed on a multi-band mesh
is accurate — each of those keeps the status its own row in the table
above gives it. In particular the F-S4 order result was obtained on an
EMPTY PEC cavity against an analytic eigenfrequency, precisely because the
theorem it tests assumes smooth fields; on a rasterized dielectric-loaded
microstrip-class fixture the same ladder was inconclusive, floored at a
~20 MHz (~4e-3) absolute error present identically in uniform-mesh arms —
a geometry-realization limit of that fixture class, recorded in the design
note, not a grading effect. Convergence on your own geometry still has to
be demonstrated on your own geometry.

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

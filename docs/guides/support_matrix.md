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
| Uniform precision modes | **limited by observable** | `precision="float32"` is the default. `"float64"` requires JAX x64 and is uniform-single-device only. `"mixed"` is also uniform-single-device only; with CPML its float16 field storage raises the absorber residual floor, so it is not a substitute for float32 near low-reflection or S-parameter floors. |
| Rectangular-waveguide S-matrix | **limited** | Uniform single-mode magnitude results are validated across the documented WR-band tests. Use `compute_waveguide_s_matrix(...)`; phase, junction, and nonuniform results have narrower evidence. Differentiation through `eps_override` / `sigma_override` runs on `normalize=False` and `normalize="flux"` only; `normalize=True` is host-assembled and raises `NotImplementedError` for a traced override. Multimode (`n_modes>1`) is host-assembled and outside the differentiable chain. S is a modal voltage-wave ratio referenced to each port's own discrete modal impedance, so it is a power-wave S only when all ports share one cross-section. The `normalize="flux"` result dtype follows `JAX_ENABLE_X64` (complex64 by default, complex128 under x64), as `normalize=False` does. **Chain battery (v1.8): measured, gates red — this family is NOT chain-closed.** One pre-declared WR-90 measurement (VESSL run 369367257823, whole-battery solve wall 1157.6 s) evaluated every criterion of [`chain_closure_contract.md`](../design_notes/chain_closure_contract.md) on rungs `a/9`, `a/18` and `a/36` across the `normalize=False` and `normalize="flux"` lanes. Of 185 stored verdicts, 24 are red — 21 failures plus 3 dx ladders the pre-declared guard reports as *not interpretable* — in four families. Criterion 1 (forward identity) is red on all eight `normalize="flux"` legs: float32 reassociation of the Poynting DFT under the reverse-mode tape moves S by up to 1.09e-5 against `rtol=1e-5, atol=1e-7`, while the same traced call under a scoped x64 context agrees with the untraced call to 1.5e-15; the `normalize=False` lane is bit-identical. Criterion 2 is green. The settling witness was the fifth red family at measurement time: `settling_db` read 0.00 dB at the PEC-short fine rung because the far-port records had underflowed float32 to exactly zero, while the records that stay in the float32 normal range ring down to -99.98 dB and doubling the record to 80 periods moves S by at most 7.3e-6. The witness now skips a record whose peak amplitude is below the storage format's smallest normal scaled so the -40 dB decision is itself representable (1.1754944e-36 for float32) and names the records it skipped, which also removes the mid-rung -40.85 dB pass carried by float32 subnormals (#869); re-derived from the same stored records the PEC-short fine cell reads -113.91 / -114.48 dB per drive on normalize=False and -106.14 / -106.03 dB on normalize="flux", with the -40 dB bar unchanged and the battery not re-run. Criterion 3(a) is red on one leg, the PEC-short `\|S11\|**2` derivative under `eps_override` on the flux lane, which is a zero-derivative objective the pre-declaration expected to skip. Criterion 3(b) is red on the reference-plane rotation: 6.602 degrees against the Yee-discrete beta (gate 3 degrees) and 6.565 degrees against the continuous beta (gate 6 degrees), because the port's discrete TE10 cutoff is solved on an aperture one cell wider than the guide (#868). Criterion 3(c) leaves 3 of 10 dx ladders not interpretable; the next admissible rung is `a/72`. Criterion 3(d) is green at the fine rung, and its referee set is the battery's analytic Airy slab and PEC-short plus the five broad-E5 replay bands (WR-340, WR-62, WR-28, WR-15, WR-10). The phase referee is analytic Airy only; junctions and nonuniform meshes are outside this evidence, and the `normalize=False` extractor's 1.8e-2 column power on an empty guide is open as #873. Criterion 4 of the contract states that failure of any single pass condition means the family is not chain-closed. Power closure carries a witness that is **independent in the plane index only (both routes integrate the same transverse window with the same uniform dA through the same flux kernel, and neither sees the reference-plane de-embedding, which is phase-only)** and that **bounds** rather than resolves the disagreement: at the coarse rung two interior `add_flux_monitor` planes reproduce `1 - \|S11\|**2 - \|S21\|**2` to within 2.146e-05 of the port route against a 0.02 gate, but both routes sit at about 1e-05, the float32 field-noise floor of that rung, so the measurement caps the disagreement instead of demonstrating closure. That witness landed on `main` in PR #870 (`1bccdfba`). Artifacts: `tests/oracle/test_waveguide_chain_battery.py`, `tests/unit/geometry/test_waveguide_chain_battery_geometry.py`, `tests/fixtures/waveguide_chain_battery/fixture.json`, [`waveguide_chain_battery_predeclaration.md`](../design_notes/waveguide_chain_battery_predeclaration.md), `scripts/diagnostics/waveguide_chain_battery_measure.py`. |
| Microstrip-line S-matrix | **limited** | Use `compute_msl_s_matrix(...)` with the laplace/quasi-TEM model. `add_msl_port` accepts `+x`, `-x`, `+y`, and `-y`; the x-only eigenmode path remains outside the public support envelope. The external notch comparison is characterized rather than a tight cross-solver match. |
| Surface-impedance sheets | **limited** | `add_thin_conductor(..., surface_impedance_f0=...)` supports patterned shapes that rasterize to exactly one cell layer and at least one cell on its documented uniform and graded-mesh lanes. It is not a general dispersive or multi-runner conductor model. |
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
| NTFF + graded z | **limited** | A short-dipole directivity case agrees with the 1.76 dBi theory within about 0.05 dB (`tests/unit/farfield/test_farfield_nonuniform.py`). Other source and geometry combinations need separate validation. |
| DFT plane or full-plane flux + graded z | **experimental** | The calculation runs, but no general RF-accuracy statement is documented. |
| Finite-region `add_flux_monitor(size=...)` + graded (`dz_profile`/`dy_profile`/`dx_profile`) mesh | **experimental** | Runs (previously raised `NotImplementedError` on the nonuniform lane). The physical `size`/`center` resolves to a CELL window against the realized graded cumulative cell edges (`_nu_flux_tangential_bounds` in `rfx/runners/nonuniform.py`; NODE→CELL narrowing per #868), and the per-cell area weight is the realized `dA = d1[lo1:hi1] ⊗ d2[lo2:hi2]` — no cubic-cell `dx²`. Correctness is pinned by `tests/unit/nonuniform/test_nu_flux_monitor_finite_size.py`: the finite window equals the full-plane integrand over the same window to machine precision; the selected cell count and cumulative `dA` equal a span recomputed independently from the realized edge array (this pins the pad offset and the node→cell narrowing, and is exact by construction once the window is fixed, so it is an oracle for the WINDOW, not for the requested size); one further check compares the selected physical extent against the REQUESTED `size` within the edge-snapping bound; and both track the mesh across two gradings while the total aperture area stays grading-invariant. Snapping/clamping: each requested endpoint snaps to the nearest cumulative cell edge (up to half a local cell), and a `size`/`center` that reaches outside the interior is CLAMPED to the interior — same as the uniform lane; only a degenerate (<1 cell) window raises. A `UserWarning` is emitted when a requested endpoint falls outside the interior by more than half the end cell (a per-endpoint clamp test), or when the realized extent differs from the requested `size` by more than the largest cell touching the window (a snap past an adjacent edge). A clamp of half an end cell or less is silent, because it is not distinguishable from the ordinary sub-cell snapping every endpoint undergoes. As with the full-plane graded-z flux above, NO general RF-accuracy statement is documented; the graded-normal-axis H co-location (0.5/0.5) limitation is shared with the full-plane path, not introduced here. |
| TFSF + graded z | **experimental** | Only normal incidence along `+x` or `-x` (`angle_deg=0`) runs. Oblique incidence and incidence along `+z` or `-z` raise. |
| Rectangular-waveguide S-matrix + nonuniform mesh (`dx_profile`/`dy_profile`/`dz_profile`) | **experimental** | Single-mode `normalize=True` and `normalize="flux"` run. Analytic fixtures cover grading ratios 1--3 and relative permittivity 2 and 4. A passed Palace magnitude comparison covers `normalize="flux"`, a graded-`dy` ratio of 2, WR-90 empty/PEC-short/dielectric-slab cases, and 8.2--12.4 GHz (`max_mag_abs_diff=0.008529`, improved from `0.07009` when the lane's absorber stopped being 0.33 of a guide wavelength — #576). Other profiles, bands, phase, multimode operation, and arbitrary junctions are not validated. `eps_override` and `sigma_override` differentiation is available only with `normalize="flux"`; only `eps_override` has a nonuniform AD-vs-FD regression test. Neither AD check establishes RF accuracy. Dispatch history: until #811 (fixed 2026-09-01) this API reached the nonuniform lane only for `dx_profile`/`dy_profile` — a `dz_profile`-only simulation was silently solved on the uniform grid built from the scalar `dx` while preflight described the graded mesh (dz-only meshes with different `dt` returned bit-identical S in the falsifier baseline, `scripts/diagnostics/wr90_dz_dispatch_falsifier.py`). A `dz_profile` now dispatches here under the same restrictions, but NO dz-graded accuracy evidence exists yet (#810): dz-graded waveguide S-parameters are dispatch-correct and unvalidated. The lane emits the same energy-based ring-down settling witness (`settling_db`, -40 dB aggregate warning) as the uniform single-mode lane (#827 waveguide instance). Phase envelope on this lane (derived arithmetic, not a measurement): the reference-plane shift `exp(-/+ jβΔ)` and the modal impedance evaluate β at the grid's BOUNDARY cell (`NonUniformGrid.dx`), not the cell the plane sits in. `Z_TE` does not depend on the cell size at all (the discrete `sin(β·dx/2)` equals `s_x·dx/2`), and the β difference is the second-order Yee correction `(β·dx)²/24` at the two sizes: for the committed WR-90 fixture (boundary 1.5 mm, fine 0.75 mm, discrete cutoff 6.650 GHz; `tests/fixtures/waveguide_nu_beta_cell_size_envelope.json`, replayed by `tests/unit/sparams/test_waveguide_nu_beta_cell_size_envelope.py`) Δβ = 0.057 / 0.271 / 0.652 rad/m at 8 / 10 / 12 GHz, i.e. 0.07° / 0.31° / 0.75° over a 20 mm plane offset (0.86° worst at 12.4 GHz) and 0.02° over that fixture's own 0.5 mm applied shift. Both of that fixture's port-to-reference spans lie in uniform 1.5 mm cells, so the two evaluations coincide there and the fixture cannot exercise the difference. A span that crosses cells of more than one size now raises `ValueError` (`tests/unit/sparams/test_waveguide_nu_grading_zone.py`) instead of applying a single β; a fixture with a reference plane inside the graded region, and β integration over the span, are deferred to #854 item 1. This bounds one mechanism; it is not a phase validation of the lane. |
| Lumped-port S-parameters + nonuniform mesh | **unsupported** | The S-parameter request must fail. |
| MSL S-matrix + nonuniform mesh | **experimental** | `mode="laplace"` and `mode="uniform"` have internal settled-S11 regression coverage only. There is no external nonuniform comparison. `mode="eigenmode"` raises. |
| Coaxial port + nonuniform mesh | **unsupported** | The request must fail. |
| Lumped RLC update + nonuniform mesh | **limited** | R/L/C ADE elements participate in the field update. Nonuniform S-parameters and component-value AD are not documented. |
| Multi-band graded mesh (N fine bands along **z**, ratio <= 1.4) | **limited** | The MESH ITSELF is documented — not any observable computed on it — and only for grading along **z**. Explicit `dz_profile` vectors with **up to 3 fine bands / 4 transitions** (small-large-small-large included) and every adjacent cell ratio <= 1.4 are covered by the witness battery below — that is the widest profile any witness actually exercises. Read the scope statement in "Multi-band graded mesh" before quoting this row: in-plane (`dx_profile` / `dy_profile`) grading is UNCOVERED, absorber-adjacent grading is EXCLUDED, and `dt` is unchanged (global min-cell CFL). |
| Volumetric PEC scatterer + nonuniform waveguide | **experimental** | The device/reference handling is regression-tested, but no RF validation is documented for arbitrary iris, post, septum, branch, or T-junction geometries. |

### Multi-band graded mesh

**What this row covers.** An explicit **`dz_profile`** vector holding
fine bands along z, in any order — fine-coarse-fine-coarse-fine and other
small-large-small-large patterns included — with **every adjacent cell
ratio <= 1.4**, abrupt (a single step at the cap) or smoothly ramped.
**Band count is witnessed only up to 3 fine bands / 4 transitions**, the
widest profile in the battery (`fixtures.py`); more bands are expected to
behave the same way — each transition is local and the per-transition
reflection is what was measured — but they are not witnessed, and the
expectation is an argument, not evidence.
Before this row, only a single fine band was documented.

**The z axis is the whole of it.** Every witness below grades z and holds
the transverse mesh uniform — the witness harness takes a scalar transverse
cell size — so this row says nothing about `dx_profile` / `dy_profile`
grading, with or without z grading at the same time. See the exclusions.

**The cap is 1.4 — on z.** Ratios above it still construct and run; what
is lost is the accuracy class below, not stability. On a `dz_profile`,
`Simulation(...)` warns above 1.4 and preflight adds
`nu_grading_ratio_beyond_validated_cap`, both advisory. 1.4 matches the
commercial default (Tidy3D `max_scale`). **On `dx_profile` / `dy_profile`
the threshold is the pre-existing 1.3**, deliberately NOT moved to 1.4:
no witness in this row grades an in-plane axis, so there is no in-plane
provenance for moving an in-plane lock (SPEC-00 §0.2-4). Both the
constructor warning and the preflight advisory carry the per-axis value,
and both are advisory on either axis.

**Evidence** (SPEC-01 lane, tracker #780; pre-declaration
`docs/design_notes/20260829_spec01_multiband_predeclaration.md` — every
window below was committed before its measurement; harness and raw results
in `validation/research/multiband_nu/`; regression packaging in
`tests/unit/nonuniform/test_multiband_nu_envelope.py`):

| Witness | What was measured | Result |
|---|---|---|
| F-S1 stability/energy (`results/w1_pa_1d.json`) | Remis-class dual-cell discrete energy over 10^6 steps, 1-D lossless multi-band, r in {1.0 control, 1.1, 1.2, 1.4, 1.5, 2.0} abrupt + {1.4, 2.0} smooth | drift <= 2.94e-6 relative on every arm, bounded inside the pre-declared float-accumulation envelope (1.19e-3 at 10^6 steps) at every sample. The judge's growth-trend clause did NOT evaluate on any arm: it is gated at a declared trend floor of 50u = 2.980e-6 below which a slope is quantization noise, and the largest drift reached 2.931e-6. What is established is boundedness within the envelope, not the absence of a trend |
| F-S1 3-D (`results/w1_pb_full_gpu.json`) | Same functional, 3-D PEC box, 10^6 steps on GPU (VESSL run 369367256892) | end drift +4.7e-9 (r=1.4) and +1.3e-7 (r=2.0), max 7.7e-8 / 2.0e-7, bounded inside the same envelope; the trend clause did not evaluate here either (both maxima are >1 order below the 2.980e-6 trend floor) |
| F-S2 per-transition reflection (`results/w2_w3.json`) | Gated two-run differencing against an exact discrete scattering-chain model, 10 GHz, 30 cells/wavelength in the fine band | r=1.4 abrupt -53.9 dB measured vs -54.0 dB modelled (r=1.1 -67.1 dB, r=1.2 -60.7 dB); every in-envelope arm inside its pre-declared window. **These dB figures are resolution-specific — quote the law, not the number.** Per-transition reflection scales as (dz/lambda)^2, i.e. -12.0 dB per doubling of fine-band resolution; the same chain model gives, for r=1.4 at 15 / 20 / 30 / 60 / 120 fine cells per free-space wavelength, -41.7 / -46.8 / -54.0 / -66.1 / -78.2 dB (the N^-2 law is asymptotic and under-predicts the reflection below ~30 cells/wavelength) |
| F-S3 round-trip amplitude (`results/w2_w3.json`) | Net amplitude drift across a symmetric rise-and-fall traversal (the Christ amplification/attenuation asymmetry) | \|T\|-deviation <= 7.6e-6 for r <= 1.4, against a 3e-4 floor; r=1.0 null control 2.4e-6 |
| F-S4 convergence order (`results/w4r3_zdominant_cavity.json`) | Resonance error vs the ANALYTIC TE_{1,0,4} of an empty PEC cavity (60 x 3 x 64 mm), multi-band z profile at the cap ratio r=1.4 vs a uniform-fine control, four scales, on a fixture DESIGNED so the graded axis carries the error budget: an exact discrete-dispersion decomposition (`analytic_dispersion.py`, certified to reproduce every arm to 0.033 MHz) puts 89 % (uniform) to 92 % (multi-band) of the modelled error on the z axis, with 36 % of the multi-band total in the grading-specific term | p_multiband = 2.01, p_uniform = 2.02 — 2nd-order supraconvergence preserved at the cap ratio (Monk & Suli 1994; Li & Shields 2016). The multi-band mesh's extra error at matched scale is measured, not inferred: 28.4 MHz (2.9e-3 relative) at the coarsest scale falling to 0.44 MHz (4.6e-5) at the finest — about 1.56x the uniform-fine error amplitude at equal fine cell size, at the same ORDER |
| F-S5 differentiability (`results/w5_ad.json`) | `jax.grad` of a multi-band profile observable vs central FD, dominant cells. **The only witness here that is not PEC-closed:** its grid is built with `cpml_layers = 4` on all six faces, and its profile's boundary runway is non-uniform, so this fixture draws the row's own `nu_grading_reaches_absorber` advisory. It is a gradient-consistency check, not an accuracy measurement — see the absorber exclusion | worst dominant-cell error 1.1e-4 relative (f32 path) and 1.7e-4 (x64 context), inside the existing NU AD convention (15 %) |
| Revert-proof (`results/revert_proof.json`) | Two deliberate defect injections (one witness weight, one solver transition coefficient) | baseline f64 drift 2.1e-16; corrupted witness 5.8e-3, corrupted solver transition coefficient 7.9e-3 — the witness does fire on the defect family it guards. **Scope limit:** the witness builds its weights from the same `grid.inv_*` arrays the solver steps with (`remis_energy.energy_weights`), so a defect INSIDE those arrays — a wrong dual spacing computed by `make_nonuniform_grid` itself — would corrupt witness and solver identically and stay invisible here. It guards the update path against a correct metric, not the metric |
| Order-witness revert-proof (`results/w4r3_revert_proof.json`) | One transition coefficient of the multi-band arm deliberately corrupted (E-update dual metric replaced by the primal cell width at a single coarse->fine node — an error that is identically zero on a uniform mesh) | the resonance moves -47.3 / -24.0 / -12.0 / -6.0 MHz at the four scales (20-160x the fit floor) and the fitted order drops to 1.38, so the committed F-S4 judge FIRES. The order gate above can fail for a grading reason |

**Exclusions — these are NOT covered by the row.**

- **Grading must not reach the absorber.** Every witness above that
  measures an accuracy observable — F-S1 (1-D and 3-D), F-S2, F-S3, F-S4
  and both revert-proofs — is PEC-closed with `cpml_layers = 0`; each
  builds its grid through `harness.build_pec_fixture`. **F-S5 is the
  exception and does NOT run PEC-closed:** `w5_ad_consistency.py` builds
  its grid with `cpml_layers = 4`, i.e. a 4-layer CPML on all six faces,
  so one witness in this row does grade z with an absorber present. Its
  own profile's boundary runway is not uniform either (the anti-tie
  jitter, ~1 % and ~1.5 % on the two z faces), so that fixture would
  itself draw the `nu_grading_reaches_absorber` advisory described here.
  What F-S5 measures is `jax.grad` against central FD on the profile
  vector; it compares no field, reflection, energy or resonance against
  any reference, so what it establishes is that a graded mesh beside an
  absorber constructs, runs and differentiates consistently — not that
  anything computed on it is accurate. **The exclusion therefore rests on
  the narrower true statement: no witness in this row measures an
  accuracy observable with an absorber present.** Nothing here says how a
  grading transition interacts with a CPML/UPML face, and the absorber pad
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
  -51.6 dB and r=2.0 -43.9 dB per transition at the F-S2 fixture's own
  30 cells/wavelength (the same resolution caveat and (dz/lambda)^2 law as
  the F-S2 row above), both consistent with the chain model but outside the
  claimed envelope.
- **In-plane grading is UNCOVERED — not "limited", not "exercised".** No
  witness in this row grades `dx_profile` or `dy_profile`. The witness
  harness takes a SCALAR transverse cell size, so every witness above,
  including the 3-D energy witness (whose transverse mesh is uniform
  1.5 mm), grades z alone. Nothing here covers in-plane grading, nor
  in-plane and z grading simultaneously. `make_nonuniform_grid` accepts
  those profiles and they run; what does not exist is evidence. (The
  solver's per-axis code is structurally symmetric — an argument, not a
  witness, and it is not offered as one.)

**Honest scope — what the witnesses do and do NOT establish.** They are
statements about the mesh and the solver on it, **for grading along z**:
the scheme conserves its discrete energy, each transition reflects at the
modelled level, a symmetric traversal does not drift in amplitude, the
global order stays 2 at ~1.56x the uniform-fine error amplitude, and the
gradient path is intact. They are NOT a statement that any S-parameter,
flux, far-field, or port result computed on a multi-band mesh is accurate —
each of those keeps the status its own row in the table above gives it.

In particular the F-S4 order result was obtained on an EMPTY PEC cavity
against an analytic eigenfrequency, precisely because the theorem it tests
assumes smooth fields. Its fixture is deliberately proportioned so the
graded axis dominates the error budget (89-92 % of it), which is what makes
the order gate capable of failing for a grading reason — the earlier
`w4r2_analytic_cavity` fixture put only ~1 % of its error on that axis and
is retained in the design note as a recorded diagnostic, not as evidence
for this row. On a rasterized dielectric-loaded microstrip-class fixture
the same ladder was inconclusive. What THIS lane measured there is one
thing only: the ~20 MHz (~4e-3) spread was present identically in
uniform-mesh arms, so it is not a grading effect. The reading of that
spread in an earlier draft of this row (a "floor", and a
"geometry-realization limit") is WITHDRAWN — it was an interpretation this
lane never measured, and it is withdrawn on that ground alone.

The evidence that also EXPLAINS the spread was taken in the #786 lane and
ships in **PR #788, which is not merged**, so the following is a pointer,
not something a reader can reach from `main`: measured in the #786 lane
(PR #788, not yet merged) — three lattice-valid rungs this ladder had
skipped, an `f(h)` reported **non-monotone** on that fixture with a turn
at `dz_fine = 0.125 mm` (which would put this ladder's anchor past the
turn, so the fitted quantity was never an error sequence); geometry
quantization exonerated (realized-vs-declared 6.8e-6 cells, exact integer
cell counts, bit-identical material maps); port loading exonerated
(<= 3.5 kHz over a 10^4 drive span). Read those four as unmerged-lane
results pending #788, not as settled facts of this row. Whether that
fixture's `f(h)` converges to the physical answer is NOT settled by either
lane — see #786 and the ledger's dielectric-interface staircasing entry.
Convergence on your own geometry still has to be demonstrated on your own
geometry, and #786's `ladder_guard` precondition (also in PR #788) is
there to catch a ladder read across a turn.

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

# SBP-SAT z-slab true reflection/transmission benchmark specification

## Status

Deferred implementation specification for the SBP-SAT z-slab research lane.
This document is the Milestone 3 benchmark-credibility artifact: it defines the
true R/T benchmark that must exist before public support promotion, and explains
why the current Phase-1 lane intentionally does **not** claim that benchmark yet.

## Current evidence level

Current executable evidence is a **proxy numerical-equivalence benchmark**:

- file: `tests/test_subgrid_crossval.py`
- comparison: subgridded all-PEC z-slab run vs uniform-fine reference run
- signal: one point-probe time series transformed by a single-frequency DFT
- tolerance: relative amplitude error `<= 5%`, phase error `<= 5°`
- claim level: verifies that the current subgridded implementation stays close
  to a uniform-fine run for the selected PEC-cavity proxy fixture

This proxy does **not** establish calibrated physical reflection coefficient
`R(f)`, transmission coefficient `T(f)`, energy balance, S-parameters, or
open-boundary behavior.

## bounded-CPML feasibility result

The bounded CPML coexistence subset now has an executable point-probe
feasibility probe:

- file: `tests/test_sbp_sat_true_rt_feasibility.py`
- fixture: guarded all-CPML quasi-1D vacuum/dielectric-slab/vacuum setup
  inside an axis-aligned refinement box
- source / observables: one soft `Ez` point source and two `Ez` point probes
- comparison: subgridded point-probe extraction vs a uniform-fine reference at
  `1.5`, `2.0`, and `2.5 GHz`
- classification: **inconclusive**
- claim level: internal measurement-contract feasibility only; **not** public
  true R/T, S-parameter, flux, DFT-plane, or port evidence

The first run is useful because it separates the open-boundary question from
the broader ports/observables question.  It is still not claims-bearing:

- `|R|` magnitude and phase matched the uniform-fine reference within the
  provisional gates;
- `|T|` magnitude missed the provisional `<= 5%` gate (`~6.57%`);
- the one-cell probe-shift perturbation missed the provisional transmitted
  stability gate (`max |ΔT| ~= 0.0749`);
- energy balance is recorded only as an advisory diagnostic because point
  probes are not flux-normalized observables.

Therefore the support matrix remains:

- SBP-SAT subgridding status: `experimental`
- benchmark evidence: proxy numerical equivalence plus bounded-CPML
  point-probe feasibility evidence
- true R/T benchmark: deferred
- feasibility probe gate: inconclusive
- public support promotion: blocked

If future work needs this gate to pass, the next separate plan should add a
minimal benchmark-only flux/DFT observable contract rather than promoting
point-probe evidence as calibrated true R/T.

## Private flux/DFT benchmark gate

A private fine-owned flux/DFT accumulator and private analytic sheet/source now
exist for internal SBP-SAT benchmarking:

- file: `tests/test_sbp_sat_true_rt_flux_dft_benchmark.py`
- fixture: translational x/y, guarded all-CPML
  vacuum/dielectric-slab/vacuum setup using z-normal flux planes fully inside
  the fine grid
- source contract: private analytic sheet/source accepted only by
  `run_subgridded_benchmark_flux(...)`; it is not public `Simulation.add_source`,
  not public TFSF, and not exposed through `Simulation.run()` or `Result`
- normalization: vacuum/device two-run incident-normalized comparison
- observable contract: private raw DFT accumulators only; not public
  `add_dft_plane_probe`, not public `add_flux_monitor`, and not
  `Result.dft_planes` / `Result.flux_monitors`
- placement contract: local normal index must satisfy `1 <= idx <= n-2`, and
  sheet/plane tangential extents must remain strictly inside the fine region
- classification: **inconclusive**
- claim level: internal benchmark-only evidence; **not** public true R/T,
  S-parameter, DFT-plane, flux-monitor, TFSF, or port support

The current private gate is useful because it removes the most fragile
point-probe extraction path and replaces the earlier point-source
finite-aperture diagnostic with an explicit sheet/source fixture.  It still
keeps the public claim blocked:

- public DFT-plane and flux-monitor APIs still hard-fail under SBP-SAT
  refinement;
- public TFSF remains unsupported for SBP-SAT and is not used by this fixture;
- mixed periodic+CPML is rejected for this plan, so the fixture remains a
  bounded all-CPML domain rather than a transverse-periodic shortcut;
- synthetic accumulator semantics match the uniform scan-kernel formula,
  including multi-step, all-axis, windowed DFT accumulation;
- private sheet/source lowering and injection are tested as benchmark-only
  implementation details;
- runtime scoring now requires at least two non-floor passband bins, transverse
  magnitude/phase uniformity, plane-location robustness, vacuum stability
  against a uniform-fine reference, and incident-normalized R/T gates;
- the current recorded status is therefore **inconclusive**, not a public
  support promotion and not a reason to reinterpret thresholds.

If this analytic-sheet fixture remains below threshold, the next prerequisite
is a separate private TFSF or boundary-expanded plane-wave fixture plan.  Until
that exists and passes, the private flux/DFT gate remains internal diagnostic
evidence only, and the support matrix continues to mark true R/T as deferred.

## Why true R/T is deferred

A meaningful true R/T benchmark needs incident/reflected/transmitted separation.
The current Phase-1 support surface deliberately keeps most of the surfaces
that would make that benchmark clean and claims-bearing outside public support:

- UPML/open-boundary variants beyond the guarded all-CPML subset
- TFSF plane-wave excitation with subgridding
- DFT plane probes and flux monitors with subgridding
- waveguide/Floquet/S-parameter style ports with subgridding
- impedance/wire/coaxial ports inside the refined region

A time-gated all-PEC cavity fixture could be used as a research diagnostic, but
it would still conflate interface error, source spectrum, finite-domain echo
control, and point-probe normalization. It is therefore not sufficient as the
public true R/T benchmark for SBP-SAT subgridding.

## Required future fixture

The first claims-bearing true R/T benchmark should use a 1D-normal-incidence
slab geometry embedded inside the refined z slab:

- Boundary model: open z boundaries, or a documented equivalent that prevents
  boundary echoes from contaminating the measurement window.
- Excitation: an incident field with a defined spectrum and normalization.
- Material fixture: at least vacuum -> dielectric slab -> vacuum, with the slab
  fully inside the fine region and with a uniform-fine reference at the same
  effective resolution.
- Observables: independent incident, reflected, and transmitted measurements;
  flux or S-parameter style extraction is preferred over a single point probe.
- Frequency band: a documented passband where source energy, mesh dispersion,
  and absorber quality are all measurable.

## Metrics

The benchmark report must compute and store:

1. `R(f)` magnitude and phase against a reference solver or analytic transfer
   matrix where applicable.
2. `T(f)` magnitude and phase against the same reference.
3. Energy-balance residual `|R|^2 + |T|^2 + loss - 1` for lossless and lossy
   fixtures as appropriate.
4. Time-window definition, including exclusion of source turn-on and boundary
   echo windows.
5. Mesh-convergence trend across at least two coarse/fine resolutions.

## Initial tolerance targets

These are target gates for the future benchmark, not claims about current code:

| Metric | Initial target | Notes |
|---|---:|---|
| `|R|` relative error | `<= 5%` where `|R_ref| >= 1e-3` | use absolute error near nulls |
| `|T|` relative error | `<= 5%` where `|T_ref| >= 1e-3` | use absolute error near nulls |
| phase error | `<= 5°` in the usable passband | unwrap before comparison |
| energy residual | `<= 5%` for lossless fixtures | after time-window and absorber corrections |
| convergence | non-increasing error under refinement | investigate any regression |

These tolerances mirror the current proxy amplitude/phase gates only as a
starting point. They must be revisited after a real fixture exposes absorber,
source-normalization, and flux/S-parameter error separately.

## Acceptance gate for implementation

True R/T implementation may begin only after both prerequisites are available
for the subgridded path:

1. An echo-control strategy, either:
   - open-boundary coexistence with documented CPML/UPML interaction; or
   - a validated time-gated finite-domain method with echo rejection.
2. A source/observable contract that separates incident, reflected, and
   transmitted fields.

Until then, the support matrix must say:

- SBP-SAT subgridding status: `experimental`
- benchmark evidence: proxy numerical equivalence plus bounded-CPML
  point-probe feasibility evidence plus private flux/DFT benchmark evidence
- true R/T benchmark: deferred
- feasibility probe gate: inconclusive
- private flux/DFT benchmark gate: inconclusive under the current private
  analytic-sheet bounded-CPML fixture-quality gates
- public support promotion: blocked

## Deferred issue record

**Title:** Implement claims-bearing true R/T benchmark for SBP-SAT z-slab
subgridding.

**Milestone:** after boundary/source/observable contracts are available for the
subgridded path, expected no earlier than the boundary-coexistence and
ports/observables roadmap milestones.

**Problem:** current proxy crossval only compares one point-probe DFT against a
uniform-fine reference. It cannot separate incident, reflected, and transmitted
fields, so it cannot justify public R/T, S-parameter, or energy-balance claims.

**Required work:**

1. Choose and document the true R/T fixture.
2. Implement incident/reflected/transmitted separation.
3. Add amplitude, phase, energy-balance, and convergence assertions.
4. Store fixture metadata and usable frequency band in the test output or docs.
5. Update `docs/guides/support_matrix.json` from `deferred` to `implemented`
   only after the benchmark passes.

**Exit criteria:** all metrics in this document pass, the support matrix is
updated, and public docs cite the new benchmark without broadening beyond the
tested support surface.

## Regression hooks

The support-matrix contract is locked by `tests/test_support_matrix_sbp_sat.py`.
The proxy benchmark naming is locked by `tests/test_subgrid_crossval.py` and by
support-matrix metadata pointing to that file.

# SBP-SAT support promotion proposal

## Status

Milestone 9 proposal for the SBP-SAT subgridding lane.

## Recommendation

**Do not promote SBP-SAT / subgridding beyond `experimental` at this time.**

Keep the support-matrix lane as:

- `status: experimental`
- `boundary: all_pec_plus_selected_reflector_periodic_cpml_subset`
- `geometry: axis_aligned_arbitrary_box_with_cpml_guard_for_absorbing_faces`
- `sources: soft_point_source`
- `observables: point_probe`
- `claim_level: experimental_proxy_validated_only`

This is a promotion proposal in the sense of a formal decision record. The
current recommendation is **retain experimental status**, not public promotion to
supported or shadow.

## Approved public claim set

The current public/docs-safe claim set is:

1. SBP-SAT subgridding exists as an **experimental research lane**.
2. The retained visible surface is an **axis-aligned refinement box** case
   together with selected **reflector/periodic boundary subsets** and a bounded
   **CPML absorbing subset** for boxes outside the active absorber pad plus one
   coarse-cell guard. The supported subset does not mix PMC with periodic axes,
   or CPML with reflector/periodic faces, in one configuration.
3. The shipped runtime supports **soft point sources** and **point probes** only.
4. The current executable benchmark evidence is **proxy numerical equivalence**
   against a uniform-fine reference, not physical reflection/transmission.
5. A bounded-CPML point-probe feasibility probe exists, but it is
   **inconclusive** and remains internal/non-promotional.
6. Unsupported combinations hard-fail instead of degrading silently.

## Rejected public claims

Do **not** claim any of the following today:

- claims-bearing arbitrary 3-D box support beyond the current experimental subset
- UPML absorbing coexistence
- CPML coexistence outside the guarded interior-box subset
- mixed CPML+reflector or CPML+periodic coexistence
- broader PMC / periodic coexistence beyond the currently implemented subset
- true R/T, S-parameter, or calibrated open-boundary validation
- impedance point ports, wire/extent ports, coaxial ports, waveguide ports,
  or Floquet ports inside refined regions
- DFT planes, flux monitors, or NTFF support inside refined regions
- material-scaled SAT support for lossy, magnetic, dispersive, anisotropic, or
  nonlinear materials
- multi-rate or sub-stepped SBP-SAT time integration

## Promotion blockers

Promotion beyond `experimental` is blocked by the following current facts:

1. `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md` still records the true
   R/T benchmark as **deferred**.  The bounded-CPML point-probe feasibility
   probe in `tests/test_sbp_sat_true_rt_feasibility.py` is inconclusive: it
   matched the provisional reflection and phase gates, but missed the
   provisional transmission-magnitude and probe-shift gates.
2. Milestones 7-8 still remain **RFC/spec gates**, not widened runtime
   implementations:
   - ports and observables inside refined regions
   - materials / dispersion / time integration
   The arbitrary-box runtime, selected reflector/periodic subset, and bounded
   CPML absorbing subset now exist, but broader promotion still depends on the
   remaining gates and evidence.
3. The support matrix still correctly records the lane as an experimental
   arbitrary-box lane with selected reflector/periodic/CPML subsets, proxy-only
   evidence, and many unsupported combinations still hard-failing.
4. Public docs already use appropriately narrow wording; broadening them now
   would outrun the evidence.

## Evidence used for this decision

### Public-surface evidence

- `docs/guides/support_matrix.md`
- `docs/guides/support_matrix.json`
- `docs/public/guide/subgridding.mdx`
- `docs/public/api/support-boundaries.mdx`
- `README.md`

### Executable verification evidence

- `tests/test_support_matrix_sbp_sat.py`
- `tests/test_public_subgridding_docs_contract.py`
- `tests/test_subgrid_crossval.py`
- `tests/test_sbp_sat_api_guards.py`
- `tests/test_sbp_sat_box_refinement_spec_contract.py`
- `tests/test_sbp_sat_boundary_coexistence_spec_contract.py`
- `tests/test_sbp_sat_absorbing_crossval.py`
- `tests/test_sbp_sat_true_rt_feasibility.py`
- `tests/test_sbp_sat_ports_observables_spec_contract.py`
- `tests/test_sbp_sat_materials_time_integration_spec_contract.py`

## Promotion trigger for a future revision

A future promotion proposal may recommend widening the lane only after:

1. true R/T benchmark moves from deferred/inconclusive point-probe feasibility
   to implemented claims-bearing evidence;
2. the relevant Milestones 5-8 RFCs have corresponding runtime implementations;
3. support-matrix status and public docs can be updated without contradiction;
4. the final verifier report can tie every widened claim to passing tests and
   benchmark evidence.

## Decision summary

**Current decision:** retain SBP-SAT subgridding as an experimental, proxy-only,
arbitrary-box lane with selected reflector/periodic and bounded CPML boundary subsets.
The new bounded-CPML point-probe feasibility probe does not change that public
decision.

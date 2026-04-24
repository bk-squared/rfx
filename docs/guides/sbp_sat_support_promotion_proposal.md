# SBP-SAT support promotion proposal

## Status

Milestone 9 proposal for the SBP-SAT subgridding lane.

## Recommendation

**Do not promote SBP-SAT / subgridding beyond `experimental` at this time.**

Keep the support-matrix lane as:

- `status: experimental`
- `boundary: all_pec_only`
- `geometry: full_span_xy_z_slab_only`
- `sources: soft_point_source`
- `observables: point_probe`
- `claim_level: experimental_proxy_validated_only`

This is a promotion proposal in the sense of a formal decision record. The
current recommendation is **retain experimental status**, not public promotion to
supported or shadow.

## Approved public claim set

The current public/docs-safe claim set is:

1. SBP-SAT subgridding exists as an **experimental research lane**.
2. The retained visible surface is an **all-PEC, full-span x/y, refined z-slab**
   case only.
3. The shipped runtime supports **soft point sources** and **point probes** only.
4. The current executable benchmark evidence is **proxy numerical equivalence**
   against a uniform-fine reference, not physical reflection/transmission.
5. Unsupported combinations hard-fail instead of degrading silently.

## Rejected public claims

Do **not** claim any of the following today:

- arbitrary 3-D box refinement support
- PMC / periodic / CPML / UPML coexistence
- true R/T, S-parameter, or open-boundary validation
- impedance point ports, wire/extent ports, coaxial ports, waveguide ports,
  or Floquet ports inside refined regions
- DFT planes, flux monitors, or NTFF support inside refined regions
- material-scaled SAT support for lossy, magnetic, dispersive, anisotropic, or
  nonlinear materials
- multi-rate or sub-stepped SBP-SAT time integration

## Promotion blockers

Promotion beyond `experimental` is blocked by the following current facts:

1. `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md` still records the true
   R/T benchmark as **deferred**.
2. Milestones 5-8 produced **RFC/spec gates**, not runtime implementations:
   - all-PEC arbitrary 6-face box refinement
   - boundary coexistence
   - ports and observables inside refined regions
   - materials / dispersion / time integration
3. The support matrix still correctly records the lane as all-PEC-only,
   full-span z-slab only, proxy-only.
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
- `tests/test_sbp_sat_ports_observables_spec_contract.py`
- `tests/test_sbp_sat_materials_time_integration_spec_contract.py`

## Promotion trigger for a future revision

A future promotion proposal may recommend widening the lane only after:

1. true R/T benchmark moves from deferred to implemented;
2. the relevant Milestones 5-8 RFCs have corresponding runtime implementations;
3. support-matrix status and public docs can be updated without contradiction;
4. the final verifier report can tie every widened claim to passing tests and
   benchmark evidence.

## Decision summary

**Current decision:** retain SBP-SAT subgridding as an experimental, proxy-only,
all-PEC z-slab lane.

# rfx roadmap to v2.0

*Status: plan (targets and issue references, not validated results). Tracking issue:
[#825](https://github.com/bk-squared/rfx/issues/825). Last updated: 2026-09-02.*

## The v2.0 criterion

rfx reaches **v2.0** when the full chain

**θ (design parameters) → FDTD → calibrated complex S_ij(f) → objective**

is closed inside the differentiable graph for all four core port families —
rectangular waveguide, lumped/wire, microstrip (MSL), and coax.

The motivating observation: a gradient that matches finite differences is not the
same thing as a physically right objective. Frequency-domain objectives exist today
(`minimize_s11_at_freq`, wave decomposition, flux-path AD), but the calibrated
full S matrix — reference-plane shift and impedance normalization included — is not
yet the default differentiable observable. v2.0 closes that gap, family by family.

## "Chain closed" — the per-family contract

One definition, reused four times:

1. **In-graph S:** modal-amplitude or V–I extraction → reference-plane shift →
   impedance normalization → complex S_ij(f), with no host round-trip.
2. **Physics gates:** passivity, reciprocity, and power closure inside documented
   envelopes.
3. **Falsifier battery:** AD-vs-FD agreement on an S-native objective,
   reference-plane-shift invariance, mesh-refinement consistency, and at least one
   external or analytic referee.
4. **Artifacts:** pinned envelopes, fidelity-snapshot entries, and a ledger record.

## Phases

| Milestone | Scope | Rationale |
|---|---|---|
| **v1.7.x — chain blockers** | #811, #802, #807, #808, #782 + a one-page chain-closure contract | Chain validation on a broken base proves nothing |
| **v1.8 — waveguide + lumped/wire** | Waveguide first; its battery becomes the template. Then lumped/wire: #819 mechanism identified or envelope bounded, #683 implemented | Waveguide is the most mature family (analytic β/Z_TE, flux-path AD) |
| **v1.9 — MSL + coax** | MSL: settle the Z0 definition (#726) first; acceptance fixtures = notch filter + edge-fed patch S11 (#715). Coax: the validated axisymmetric family; #822, #823 | MSL is the physics-riskiest leg — it goes after the template exists |
| **v2.0 — all-family chain closed** | All four batteries green; support matrix redefined as *supported = chain-closed* | Feature-complete point |

The per-family contract is written out, with decidable pass conditions and the artifact each
must leave, in [`docs/design_notes/chain_closure_contract.md`](docs/design_notes/chain_closure_contract.md)
(2026-09-02). The v1.8 waveguide work plan, its pre-declared falsifiers and the gap table against
main 1c38b0d7 are in
[`docs/design_notes/v18_waveguide_s_chain_plan.md`](docs/design_notes/v18_waveguide_s_chain_plan.md).

## Explicitly outside the v2.0 gate

- **Mixed (coax↔MSL) and Floquet lanes** remain experimental, post-2.0 research
  tracks. Gating the release on them would tie it to open research outcomes.
- **Sub-cell conductor thickness (#504)** is a modelling-fidelity limit, documented
  as an envelope rather than a chain-closure requirement.

## After v2.0

v2.0 is the feature-complete milestone. The repository then shifts to maintenance;
follow-up research (S-parameter-native inverse design, sub-cell/conformal metal
modelling) proceeds on separate tracks. This ordering follows the project's
standing rule: correctness over feature sprawl.

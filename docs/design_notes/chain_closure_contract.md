# Chain-closure contract (v2.0 per-family definition)

Status: contract document, the first v1.8 deliverable
(`docs/agent-memory/rfx-known-issues.md:196-197`). Criteria source: `ROADMAP.md:25-32`. Audited
base: main 1c38b0d7, 2026-09-02. The seven questions raised here were decided by the PI on the same
day (the plan's "Decisions"); deferred items are #854.

**Revised 2026-09-21 (PI decision).** The waveguide battery stays the standard. For the three
remaining families the battery is run in the REDUCED form of the section "The v2.0 battery for
lumped/wire, MSL and coax" below: same tests, the project's v2 accuracy bar instead of per-bin
measured envelopes, one short pre-declaration, one run. Criteria 1-4 below remain the definition
of what each test is. File:line citations in the body refer to the audited base `1c38b0d7`; the
S-parameter code has since moved from `rfx/api/_sparams.py` to `rfx/sparams/{msl,coax,waveguide,
mixed,dispatch}.py`, the wire/lumped driver is `rfx/probes/sparam_driver.py`, and the MSL
passivity projection is at `rfx/sparams/msl.py` (`enforce_passivity`). Read a citation as a
pointer to the named function, not to the line.

## The chain

θ → FDTD → S_ij(f) → objective. A family is chain-closed when a design variable θ (today
`eps_override` / `sigma_override`) enters the solver on the JAX tape, the extractor returns complex
S_ij(f) without leaving the tape, and a scalar objective built from that S has a gradient matching
FD, on a fixture whose S also passes the family's physics gates and at least one referee. A gradient
matching FD on an uncalibrated S is not closure; a calibrated S with no gradient is not closure either.

## Criterion 1 — in-graph S

Three tests decide it per lane. (1) **Trace:** `jax.value_and_grad` of a scalar of
`compute_*(..., eps_override=θ).s_params` returns finite values with no `TracerArrayConversionError`.
(2) **Forward identity:** S under a no-op traced override equals the untraced call to
`rtol=1e-5, atol=1e-7` (the bound at `tests/unit/autodiff/test_waveguide_flux_ad.py:104`). (3) **Unsupported
lane:** any lane that cannot trace raises `NotImplementedError` at public dispatch, naming the lane —
the shape at `rfx/api/_sparams.py:2332-2341`, locked by `tests/unit/sparams/test_waveguide_nu_sparam.py:377-390`.

A `np.*` call on the S path is admissible only behind an explicit tracer guard: an `is_tracer(...)`
or `isinstance(..., jax.core.Tracer)` branch that returns before the call, the form at
`rfx/sources/waveguide_port.py:1826-1827` and `rfx/probes/probes.py:745-764`. There is no grep-based
pass condition; tests 1 and 3 decide this criterion. **Dtype:** all three waveguide lanes follow
`JAX_ENABLE_X64`; none hard-casts its assembled column. Artifacts: one trace test and one
unsupported-lane test per lane; a docstring sentence naming the traced inputs and the
reference-impedance convention.

## Criterion 2 — physics gates

On a reflecting DUT, never an empty guide (#395; the empty-guide identity is vacuous, see
`tests/unit/sparams/test_waveguide_twoport_contract_v1.py:131-136`): max column power ≤ 1 + tol_p; complex
reciprocity `max_f |S_ij − S_ji| / max|S| ≤ tol_r`; power closure `|1 − Σ_i |S_ij|²| ≤ tol_c` on a
lossless DUT. Tolerances are derived from a measured envelope by `gate_from_envelope`
(`tests/_gate_policy.py:89`, `ENVELOPE_GATE_MULTIPLIER = 1.5` at `:81`), never chosen.

**Settling witness.** Energy-based `settling_db ≤ −40 dB` is required where the lane emits one, as the
waveguide lanes do (`rfx/api/_sparams.py:7758-7763`). Where no energy monitor exists, one substitute
is admissible: **record-length invariance** in the form of
`tests/crossval/test_waveguide_nu_broad_e5_envelope_gates.py:170-199` — double the record window at a fixed
absorber, require the max|S11| shift below one tenth of the magnitude gate and column power within
1e-3 of unity on a lossless structure. Reason at `:175-176`: rfx has no total-energy monitor, so
truncation shows first as non-passive column power. A two-window Harminv comparison is not a witness.
Artifacts: the fast-lane gate test, the docstring's measured envelope, and `settling_db` or the named
substitute in the fixture JSON.

## Criterion 3 — falsifier battery

One common fixture set (thru, PEC-short, dielectric slab) across the differentiable lanes.

**(a) AD vs central FD.** FD legs in float64 with a ULP-span validity assert of the form
`_MIN_FD_ULP_SPAN = 1.0e4` (`tests/unit/autodiff/test_msl_ad_fd_converged.py:136`; gate `:556`, bidirectional
falsifier `:629-634`), evaluated **before** the accuracy gate. `rel ≤ 0.05` on |S11|², |S21|² and
one complex-S objective. An FD leg below the span floor skips with the span printed.

**(b) Reference-plane invariance.** Under a plane change: |S| invariant to `rtol=1e-3, atol=1e-4` (the
pinned form at `tests/unit/sparams/test_waveguide_twoport_contract_v1.py:270`); ∠S11 rotates by 2βL within a
pre-declared angle; d(objective)/dθ invariant. The shift is post-processing by a unit-modulus
`exp(∓jβΔ)` (`waveguide_port.py:1681-1682`) whose β is a property of the port cross-section, not of θ.
A **magnitude** objective (|S11|², |S21|²) is therefore gradient-invariant up to rounding, ~1e-6; a
**complex** objective is rotation-covariant instead, `d(S21·e^{jφ})/dθ = e^{jφ}·dS21/dθ`. The leg
catches a β that reaches the tape, or a non-unit-modulus shift factor. Never measured here, so
**report-only on its first run** against a pre-declared 1e-2, the same PR pinning
`gate_from_envelope(measured, quantum=1000)`; without that step criterion 3 is open.

**(c) Mesh refinement.** A 3-point dx ladder whose fine-minus-finest delta stays within the coarse
delta plus a stated floor, on |S11|, |S21| and ∠S21. **Stated limitation:** a non-increase test, not a
convergence test — a lane stuck at the wrong value passes it. Two report-first witnesses narrow it:
monotonicity with the successive-delta ratio, and Richardson `2*S_fine - S_coarse` vs the oracle on
adjacent pairs (cv18: current envelope 0.0046 → unchanged gate 0.01,
`validation/crossval/_18_wr90_iris_results/rfx.json::gates.richardson_measured_envelope_abs`;
the pre-#931 envelope was 0.0051). Three
guards: rungs are `dx = a/N` at integer N, so all realize one guide; every bin is evaluated, the worst
reported, the ladder uninterpretable when the ratio-2 successive-delta ratio is far from 0.5 (first
order) or 0.25 (second); each rung asserts rasterized cell counts scale with 1/dx.

**(d) Referee.** One analytic or external referee inside a pinned tolerance, conventions recorded
(Yee half-step; time-convention conjugation, `rfx-known-issues.md:4093-4112`). A magnitude-only flux
gate satisfies (d) alone and can never support criterion 1 or 3(a), having no AD leg. Artifacts:
`tests/test_<family>_chain_battery.py` (fast lane when ≤ 30 s, else slow with the shard named), the
fixture JSON with measured values, and a design note pre-declaring every tolerance, position and
drive setting **before** the first run.

## Criterion 4 — artifacts

Pinned envelope JSONs under replay gates with a bounded-margin lock; a fidelity-snapshot entry for
the fixture host; a per-lane row in `docs/agent-memory/rfx-known-issues.md` giving status against
criteria 1–3 with file:line; re-capture commands that either run from a clean checkout or name a
tracked VESSL YAML. Failure of any single pass condition means the family is not chain-closed. Gates
are never loosened to reach closure; a red gate needs a written root cause first.

## Explicitly not required

- Mixed (coax↔MSL) and Floquet lanes (`ROADMAP.md:45-46`) and #504 (`:47`). Multimode
waveguide (`n_modes > 1`): the host-side assembly at `rfx/sources/waveguide_port.py:2836`, `:3026`,
`:3036` is documented, not fixed. Tracing `freqs` or the plane position as θ; both are static
(`rfx/api/__init__.py:2472`, `waveguide_port.py:1679`). Phase agreement with external solvers (Airy
suffices). `normalize=True` for reflection of strong reflectors (`rfx-known-issues.md:3384-3395`).
Runtime wiring of the reciprocity warning (`rfx/validation.py:468-486`, off at `:342`) — sequenced
after WP2 measures the complex envelope, #854 item 4.
- **The #812 artifact lane.** v1.8 consumes `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`
as it stands, keeps its `STALE` label, and edits neither it nor the crossval gates
cv02/03/04/09/10/14/20/21 or cv06b. That decision is #812 Phase 0 item 2
(`docs/design_notes/20260831_cv11_broad_e4_artifact_provenance.md:5`) and the lane is owned
elsewhere: `rfx-known-issues.md:107-108`, "The #812 lane belongs to the Mac-side session — not fixed
here."

## How a family is declared chain-closed

One PR that (1) links the four artifacts by path, (2) adds the ledger row, (3) updates
`docs/guides/support_matrix.md` and `sparameter_support_matrix.{md,json}` in one diff, (4) carries the
R3 line, (5) is signed off by a verifier that did not author it. Until `ROADMAP.md:41` redefines the
matrix, wording stays "limited"/"experimental" plus "chain-closed (v1.8)"; "supported" is a v2.0 word.

## The v2.0 battery for lumped/wire, MSL and coax (PI, 2026-09-21)

rfx v2 is an open-source simulator with basic feature support. The waveguide family was closed
with 185 stored verdicts over three pre-declared runs; repeating that three more times is not the
goal. The remaining families run the same TESTS with this reduced PROCEDURE.

**What stays.** Criterion 1 (1) trace and (3) unsupported-lane refusal; criterion 1 (2) forward
identity where the traced and untraced call are the same function (it is cheap); criterion 2 on a
REFLECTING DUT with the settling witness; criterion 3 (a) AD against a float64 central FD with
the ULP-span assert evaluated first, (b) reference-plane invariance, (c) a 3-rung dx ladder,
(d) one analytic or external referee.

**What changes.**

| | waveguide battery | reduced battery |
|---|---|---|
| verdict numbers | per item and bin, `gate_from_envelope(measured) x 1.5` | the v2 accuracy bar, the same for every family: magnitude within 2 dB of the referee, frequencies (resonance, notch, cutoff, band edge) within 1 %, the curve's trend over the band agreeing, never a single-bin verdict. Passivity: max column power <= 1 + 0.02. Reciprocity: `max_f |S_ij - S_ji| / max|S| <= 0.02`. AD vs FD: `rel <= 0.05`. Forward identity: `rtol=1e-5, atol=1e-7`. |
| the dx ladder | a non-increase test | the same three rungs ALSO set the cell size the support matrix recommends to users: the coarsest rung from which every quantity above stays inside the bar against the finest rung, stated as cells per wavelength and cells per smallest dimension |
| pre-declaration | every tolerance, position and drive setting | one short note per family: fixture dimensions, the three rungs, the DUTs, the referee. The tolerances are the bar above and are not re-declared |
| size | 3 DUTs x 3 rungs x 2 lanes, three runs | 2-3 DUTs x 3 rungs x 1 lane, one run; a second run only after a written root cause |
| artifacts | envelope JSONs under a bounded-margin lock, snapshot entry, ledger row, re-capture YAML | one result JSON, one replay test reading it, one support-matrix line (with the recommended cell size), one ledger line, the VESSL YAML that produced it |

The passivity and reciprocity numbers are the waveguide battery's own gates (1.02 column power;
its reciprocity gate is 0.01, doubled here for V-I extractors), not new measurements; a family that
cannot meet them is reported with its curve, not re-gated.

A case that cannot be judged by the bar because of what it is (a deep null, a high-Q resonance, a
phase-only quantity) is taken to the PI before another criterion is chosen.

**Per family.**

| family | chain scope | DUTs | dropped, and why | referee |
|---|---|---|---|---|
| lumped / wire | the differentiable S11 of `forward(port_s11_freqs=...)`. The S matrix of `run(compute_s_params=True)` is a numpy post-process and is NOT in the v2.0 chain; a design that differentiates a transmission uses an MSL, coax or waveguide port | short, open, resistive load | reciprocity, power closure, reference-plane invariance: a one-port at a feed point has no second port and no propagation plane | analytic `(R - Z0)/(R + Z0)` at low frequency; the recorded openEMS magnitude comparison, committed as a fixture |
| MSL | `compute_msl_s_matrix`, RAW S. The passivity projection stops being the default result and becomes an explicit post-process, so that the value a user reads and the value a gradient differentiates are one function | open-stub notch (reflecting), thru (control) | none | analytic quarter-wave notch; the committed openEMS / Palace notch records |
| coax | `compute_coaxial_line_reflection` and `compute_coaxial_two_port`, `eps_scale` channel | one-port short / open / resistive (exist); two-port dielectric bead (new, reflecting), thru (control) | none | analytic TEM line; committed Meep and openEMS records. The ladder's observable is |S| and the phase of S, not `beta` |
| waveguide | closed (v1.8), uniform single-mode, `normalize=False` and `"flux"` | -- | non-uniform meshes, multimode and `normalize=True` stay outside in v2.0 | -- |

## Status (main 92bba965, 2026-09-21)

O present, P partial, X absent. Evidence by file:line, the CI lane that runs each test, and the
DUT each gate is measured on, are in the ledger's dated note `20260921_chain_closure_inventory.md`
(the inventory quotes recorded values, so it lives with the ledger rather than in this tree).

| sub-criterion | waveguide | lumped / wire | MSL | coax |
|---|---|---|---|---|
| 1 trace | O | P -- S11 of `forward` only | O | O |
| 1 forward identity | O | X | X -- `run()` returns the projected S, the traced call the raw one | P -- magnitude at 1e-3, weekly lane |
| 1 unsupported-lane refusal | O | O | O | O |
| 2 passivity, reflecting DUT | O | X -- thru only, weekly lane | X -- raw S ungated | P -- one-port short/open/resistive, weekly lane |
| 2 reciprocity | O | dropped | X -- synthetic matrices only | P -- thru only |
| 2 power closure | P -- bounding witness | dropped | X | X |
| 2 settling witness | O | O | P -- asserted only inside a `gpu` test | O |
| 3a AD vs f64 FD, ULP assert | O | P -- reflecting DUT, no ULP assert, float32 FD | P -- assert present; the gate is `gpu`+`slow`, which no CI lane runs; thru | P -- two-port has the assert (thru, weekly); one-port is on a short without it |
| 3b plane invariance | O | dropped | X | X |
| 3c three-rung ladder | O | P -- observable is `sv_max`; the post-fix rungs are read by no test | X | P -- two points, observable `beta` |
| 3d referee | O | P -- openEMS comparison in prose, no fixture | P -- notch vs openEMS characterized, Palace, phase referee; cv06b's own pre-declaration not discharged | O |
| 4 artifacts | O | X | X | P |


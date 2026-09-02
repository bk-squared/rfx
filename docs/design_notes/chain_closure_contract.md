# Chain-closure contract (v2.0 per-family definition)

Status: contract document, the first v1.8 deliverable
(`docs/agent-memory/rfx-known-issues.md:196-197`). Criteria source: `ROADMAP.md:25-32`. Audited
base: main 1c38b0d7, 2026-09-02. The seven questions raised here were decided by the PI on the same
day (the plan's "Decisions"); deferred items are #854.

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
adjacent pairs (cv18: envelope 0.0051 → gate 0.01, `crossval/18_wr90_iris_modematch.py:162`). Three
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

## Status today (main 1c38b0d7)

| Family | 1 In-graph | 2 Gates | 3 Battery | 4 Artifacts |
|---|---|---|---|---|
| Waveguide, uniform | PARTIAL — `False`/`flux` on tape; `normalize=True` and multimode host-side; normalization implicit (`waveguide_port.py:1955`) | VALIDATED magnitude (battery `:307` 1.0005 < 1.02; reciprocity `:340` 0.0005 < 0.01); complex reciprocity ABSENT; independent closure witness ABSENT | PARTIAL — AD-vs-FD on \|S\|² only, float32 FD, no ULP guard; plane invariance on `normalize=True` only; one \|S21\| ladder; referees green | PARTIAL — envelopes gated, re-capture chains off-tree, snapshot "NOT AUDITED" for ports, `index.md:249` stale |
| Waveguide, non-uniform | VALIDATED (slow) for `flux`; β/Z_TE read the boundary cell (`waveguide_port.py:603`) | VALIDATED at 0.02 in the slow lane only (`nu_nontrivial:491-500`) | PARTIAL — flux AD-vs-FD only; no plane test, no dx ladder; referees green on graded-dy | PARTIAL — no dz-graded evidence (#810 OPEN); #827 open for the general lane |
| Lumped / wire | ABSENT on the reference-plane lane — extraction is numpy complex128 → complex64 (`rfx/probes/refplane.py:410-411`, `:539-542`; `sparam_driver.py:268-270`) | thru gates exist but `slow_physics` only (`tests/locks/test_refplane_port_waves.py:779-860`); #819 sv_max 1.003227, mechanism unidentified | ABSENT; #683 decided POST-injection, not implemented | ledger rows present (`rfx-known-issues.md:4278-4306`) |
| MSL | PARTIAL — `extract_msl_nprobe` is jnp (`rfx/probes/msl_wave_decomp.py:601`), but passivity projection is default-ON except on the AD channel (`_sparams.py:2981-2986`), so `run()` and AD see different S | gates run through the projection; raw passivity ungated | ABSENT; Z0 definition #726 open | partial |
| Coax | VALIDATED for the reflection step (`rfx/sources/coaxial_port.py:1601-1610`; `tests/unit/autodiff/test_ad_surface_contract.py:271`); end-to-end AD is `slow_physics` **and** `highmem` (`tests/unit/autodiff/test_coax_end_to_end_ad.py:59-60`, +10.8 GB RSS), so only the weekly highmem job can run it | per-fixture | ABSENT | #822/#823 CLOSED on GitHub and at `rfx-known-issues.md:120`; the open list at `:27-28` is stale |

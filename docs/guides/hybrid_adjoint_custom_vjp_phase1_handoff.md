# Hybrid adjoint custom_vjp — Phase 1 handoff

Status: execution handoff

## Phase 1A

Goal:
- extract and document the canonical `simulation.py` step/replay seam

Required outputs:
- seam map
- carry inventory
- replay contract

## Phase 1B

Goal:
- implement Strategy A POC on the extracted seam

Required fixture:
- uniform
- lossless
- PEC-only
- no CPML
- no Debye/Lorentz
- no NTFF
- simple `time_series` objective

Required acceptance:
- pure AD vs hybrid relative gradient error `<= 1e-4`
- deterministic replay
- explicit fallback to pure AD for unsupported paths

## Stop conditions

Stop and revise the plan if:
- the canonical seam cannot be identified cleanly
- the POC requires CPML/dispersion to make progress
- the replay contract is too entangled to keep Phase 1 narrow

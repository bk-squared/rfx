# post_897 — the same dx ladder re-read on the CORRECTED wire-lane extractor

The three rungs of `../` (VESSL 369367257803, rfx `08828189`) were measured BEFORE
PR #897 (`0225b397`, "Yee half-step phase correction for the H-derived port current
DFT on the wire lane"). That correction moves the very number the ladder reads:
#897 records the battery THRU going `sv_max 1.003227 -> 0.998896`, and 1.003227 is
the dx rung's own `sv_max` to the digit. The ladder therefore had to be re-read
before issue #819's verdict C could be closed.

**Nothing in `../` is overwritten.** The pre-#897 rungs, `../verdict.json` and the
replay gate `tests/unit/sparams/test_thru_singular_value_dx_ladder_replay.py` are
untouched and still green — they lock the historical record. This directory is the
post-correction record alongside it.

## The run

| item | value |
|---|---|
| VESSL run | 369367258720 (`rfx-thru-sv-dx-ladder-post897`, remilab-c0 gpu-rtx4090, image nvcr.io/nvidia/jax:24.10-py3), all rungs rc=0, `thru_sv_dx_ladder_post897_failed_rungs=0` |
| job YAML | `docs/research_notes/audit-2026-09-02/followup/i2_819_ladder_post897.yaml` (slim lane: fresh SHA-guarded clone into /tmp; NFS mount used for OUTPUT only) |
| rfx commit | `f5ee3b59c3e83832f7df1682f0ad30abb260a42e` — SHA guard in the log: `HEAD=f5ee3b59… EXPECT=f5ee3b59… / sha guard passed` |
| producer | `scripts/diagnostics/thru_singular_value_dx_ladder.py`, md5 `21b54da5ddeca8359789b7aa1391d218` at f5ee3b59 — the pre-#897 run's producer differs only by a docstring path, so the measurement code is unchanged |
| half-step present | `rfx/core/dft_utils.py:54 def half_step_current_phase(...)`, 3 uses in `rfx/probes/probes.py`, 2 in `rfx/simulation.py` (quoted in the job log) |
| run dir (originals) | `claude-workspace/rfx/runs/thru_sv_dx_ladder/20260906T071525Z-f5ee3b59-post897/` on the personal-workspaces NFS mount |
| job + per-rung logs | `docs/research_notes/audit-2026-09-02/followup/i2_819_logs/` |
| stack | jax 0.4.33.dev20241023+e3c6d6430, gpu (CudaDevice 0), `JAX_ENABLE_X64=0`, default field dtype, python 3.10.12, rfx 1.7.0 |
| wall time (total) | 9.1 s / 15 s / 162 s |

## Files

| file | sha256 (byte-identical to the NFS original) |
|---|---|
| `rung_dx_over_1.json` | `2d58aa8d0c2210955ce583d9b2e7ed25fcb4099dd03c9fd5d5ffdd191ddc8c04` |
| `rung_dx_over_2.json` | `da212fe4a5aade156a59b02f05d30011e4403bff876cc3975e7b10c30e81fbb8` |
| `rung_dx_over_4.json` | `c9b06e6684cd0ad16fad52467a008159990a66601f8b0fb1a59b71e8612d6361` |
| `verdict.json` | the pre-declared outcome table re-applied by `docs/research_notes/audit-2026-09-02/followup/i2_819_adjudicate.py` |

The adjudicator's thresholds are the pre-declaration's section-3 literals, and it was
proved before this run's numbers were read (`--selftest`) to reproduce `../verdict.json`
on every computed key. It was not re-aimed at this result.

## Headline

| rung | sv_max at 3 GHz, pre-#897 | post-#897 | e = sv_max − 1, pre | post | settling_db per drive |
|---|---|---|---|---|---|
| dx | 1.0032274715 | 0.9988960135 | +3.2274715e-3 | −1.1039865e-3 | −138.3 / −141.8 |
| dx/2 | 1.0003216975 | 0.9981161938 | +3.2169749e-4 | −1.8838062e-3 | −134.7 / −134.8 |
| dx/4 | 0.9991541765 | 0.9980352168 | −8.4582352e-4 | −1.9647832e-3 | −129.2 / −126.2 |

- The over-unity excess is **gone**: every one of the 9 bins at every rung is strictly
  below unity (`unity_crossing_hz` empty at all three rungs). Passivity holds without
  a Yee/near-cutoff excuse.
- The dx rung lands on #897's own recorded corrected number: `0.9988960135` vs the
  quoted `0.998896`, delta `1.35e-8`.
- The residual is a **converging** negative deficit, not a halving one:
  |e2|/|e4| = 0.959. It is a loss floor, not a dt/2 artefact.
- What #897 removed IS a dt/2 artefact: `max|ΔS|` per rung halves (`1.880`, `1.880`)
  and `Δe(3 GHz)/θ(3 GHz)` = −0.482 / −0.491 / −0.498 → −1/2 with θ = ω·dt/2.
- `|S11|` at 7 GHz still grows 0.2782 → 0.3861 → 0.4825 across the ladder, moved from
  its pre-#897 values by only −0.0113 / −0.0060 / −0.0032. #897 neither causes nor
  removes that growth.

Verdict by the pre-declared table, applied literally: **C** again — see
`verdict.json` (`verdict_text`, `outcome_table_scope_note`) and
`docs/research_notes/audit-2026-09-02/followup/i2_819.md`.

## Gates

Nothing moved. `_THRU_MAX_SINGULAR_VALUE` is 1.01 and `_THRU_PASSIVITY_PHYSICAL_BOUND`
is 1.0, both untouched; every post-#897 `sv_max` sits below both.

`verdict.json`'s `validity_gates.G1_pass` reads `false` because the pre-declaration's
G1 constant (1.003227) is the pre-correction battery number. That constant is left
untouched in the producer, the adjudicator and the replay test; the restated gate
against #897's corrected reference (0.998896) is in `verdict.json`
under `G1_restated_post897` and passes at 1.35e-8. G2–G5 pass as written.

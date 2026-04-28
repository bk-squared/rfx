# Refuted Codex 2026-04-28 source/probe attempts — archive

This folder preserves the implementations Codex produced on 2026-04-28 for
WR-90 PEC-short |S11| residual candidates **#1** (continuous-coordinate
soft-current source) and **#2** (continuous-coordinate box-integrated
probe).  Both were refuted as fixes — see verdict table below.

The full diagnostic write-up (commands, gates, OpenEMS/Meep sanity numbers)
lives in `docs/research_notes/2026-04-28_codex_arch_attempts.md`.

## Why archived, not committed to `rfx/`

Per CLAUDE.md "no patch sprawl / no half-finished implementations":
refuted opt-in features should not pollute the public `Simulation` API.
But the implementations are non-trivial and may be useful as comparison
baselines if and when the next architectural candidate (axis-aligned PEC
subpixel handling, FDTD-core change, multi-month) is attempted.  Keeping
them here gives future agents a runnable diagnostic surface without
exposing refuted features in `rfx.api.Simulation.add_waveguide_port`.

## Files

| file | origin | purpose |
|---|---|---|
| `_codex_2026-04-28_source_probe_attempts.patch` | combined diff against `rfx/api.py`, `rfx/runners/nonuniform.py`, `rfx/sources/waveguide_port.py`, `rfx/probes/__init__.py` | reapplyable patch that reintroduces `source_type="soft_current"` plumbing into the rfx package |
| `waveguide_box_probe.py` | was `rfx/probes/waveguide_box.py` | standalone `WaveguideBoxProbe` + `s11_from_box_fields(...)`; H-mesh-aware continuous-coordinate trapezoidal integration with TE10 weighting |
| `_test_waveguide_box_probe.py` | was `tests/test_waveguide_box_probe.py` | regression test for the box probe on a synthetic TE10 field; renamed with leading underscore so pytest does not collect it from the main test run |

## Refutation summary (dump-derived `|S11|` at `mon_left`)

| R | case | spread | gate | verdict |
|---:|---|---:|---:|---|
| 1 | baseline (TFSF + cell) | 0.13258 | — | baseline |
| 1 | #1 only (`soft_current`) | 0.13265 | ≤0.020 | FAIL |
| 1 | #2 only (`box` probe) | 0.13213 | ≤0.020 | FAIL |
| 1 | #1 + #2 | 0.13145 | ≤0.020 | FAIL |
| 1 | OpenEMS reference | 0.0036 | ≤0.005 | PASS |
| 1 | Meep reference | 0.0152 | ≤0.020 | PASS |

Conclusion: source-side spatial weighting and probe-side spatial
weighting are **both ruled out** as primary causes of the rfx
WR-90 PEC-short per-frequency oscillation.  The remaining concrete
candidate is FDTD-core axis-aligned PEC subpixel handling
(memory `project_wr90_architectural_candidates.md` item #3, ~1-2 weeks).

## Reapplying for diagnostics

1. From repo root: `git apply scripts/spikes/2026-04-28/refuted_codex_archive/_codex_2026-04-28_source_probe_attempts.patch`
2. Place `waveguide_box_probe.py` back at `rfx/probes/waveguide_box.py` and
   re-export from `rfx/probes/__init__.py` (the patch handles the export
   line).
3. Place `_test_waveguide_box_probe.py` back at
   `tests/test_waveguide_box_probe.py` (drop the leading underscore).
4. Re-run the diagnostic via the historical command from
   `docs/research_notes/2026-04-28_codex_arch_attempts.md`.

Do **not** reapply as a feature commit unless and until the residual is
re-evaluated against a new architectural candidate that motivates
keeping the source/probe knobs.

# Refuted Codex 2026-04-28 source/probe attempts — archive

This folder preserves the implementations Codex produced on 2026-04-28 for
WR-90 PEC-short |S11| residual candidates **#1** (continuous-coordinate
soft-current source) and **#2** (continuous-coordinate box-integrated
probe).  Both were refuted as fixes — see verdict table below.

This README preserves the recorded commands-independent verdict and
OpenEMS/Meep sanity numbers; the detailed historical write-up and commands
are unavailable in a clean clone.

## Why archived, not committed to `rfx/`

Refuted opt-in features should not pollute the public `Simulation` API.
The implementations are retained only as comparison provenance for the
refutation below. They are incompatible with current HEAD and do not expose
refuted features through `rfx.api.Simulation.add_waveguide_port`.

## Files

| file | origin | purpose |
|---|---|---|
| `_codex_2026-04-28_source_probe_attempts.patch` | combined diff against `rfx/api.py`, `rfx/runners/nonuniform.py`, `rfx/sources/waveguide_port.py`, `rfx/probes/__init__.py` | historical, non-applicable patch provenance for the refuted `source_type="soft_current"` experiment |
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
WR-90 PEC-short per-frequency oscillation. A historical next candidate was
FDTD-core axis-aligned PEC subpixel handling; this archive establishes no
outcome or timeline for that separate investigation.

## Historical patch status

`_codex_2026-04-28_source_probe_attempts.patch` is provenance for the refuted
experiment, not a patch for current HEAD. It targets the former `rfx/api.py`
layout and must not be applied directly to the current `rfx/api/` package.
The archived source and test files likewise document the old experiment;
reusing them would require a manual port, a new predeclared hypothesis, and
new validation evidence. The historical diagnostic command is unavailable in
a clean clone and must not be inferred from this archive.

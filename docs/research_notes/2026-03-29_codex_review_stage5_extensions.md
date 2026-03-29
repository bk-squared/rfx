# Codex Implementation Review: Stage 5+ Extensions

**Date:** 2026-03-29
**Reviewer:** Claude Code (Opus)
**Changeset:** +1,532 / -74 lines across 10 files (uncommitted)

## Summary

Codex made substantial external modifications to rfx while the Stage 5 session
(2D TE/TM, snapshots, HDF5 checkpoint) was running. The changes extend the
simulator with:

1. **TFSF enhancements** — split leapfrog (1D H/E), direction (+x/-x),
   Ey polarization, oblique incidence (analytic fields)
2. **Per-axis boundary generalization** — auto-derive PEC from non-periodic,
   strip CPML from periodic axes
3. **DFT plane probes** — running frequency-domain field monitors in scan body
4. **Waveguide port integration** — modal injection + DFT probe in scan body
5. **High-level API** — TFSF, waveguide port, DFT plane, periodic-axis builders
   with mutual-exclusion validation
6. **CSG fix** — `grid.axis_pads` replaces uniform `cpml_layers` offset

## Files Modified

| File | Delta | Category |
|------|-------|----------|
| `rfx/sources/tfsf.py` | +193 | TFSF split leapfrog, direction, oblique |
| `rfx/simulation.py` | +188 | Boundary generalization, DFT/WG in scan |
| `rfx/api.py` | +410 | Builder methods, validation, wiring |
| `rfx/geometry/csg.py` | -24 | axis_pads fix |
| `rfx/grid.py` | +41 | Already done in Stage 5 |
| `rfx/__init__.py` | +5 | New exports |
| `rfx/sources/__init__.py` | +9 | New package init |
| `tests/test_simulation.py` | +396 | 7 compiled-vs-manual tests |
| `tests/test_api.py` | +333 | 14 API tests |
| `tests/test_grid.py` | +7 | Per-axis CPML test |

## Review Verdict: ACCEPT

### Correctness
- Leapfrog ordering matches Taflove Ch. 5 (H corr → 1D H → E update → E corr → 1D E)
- Boundary auto-derivation is sound
- DFT accumulation uses standard running DFT formula
- Waveguide scan body correctly separates static injection from DFT-tracked probing

### Test Coverage
- Every new feature has compiled-vs-manual-loop validation (gold standard)
- API tests cover 4 TFSF variants, DFT planes, 3 waveguide scenarios, periodic axes, validation
- All tests use `max_err < 1e-5` tolerance

### Minor Notes
- Oblique TFSF uses analytic waveform (no 2D aux grid) — correct for vacuum TFSF scope
- `_build_grid()` and `run()` both set `cpml_axes="x"` for waveguide — consistent but duplicated
- Full test suite results pending at time of review

## Final Test Results
- **132/132 passed** (full suite, 40 min including Meep/openEMS cross-validation)
- Previous: 112 tests (Stage 5)
- Codex additions: 20 new tests
- Zero failures, zero regressions

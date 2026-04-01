# rfx Patch Antenna Simulation Report

**Date:** 2026-04-01
**Author:** Claude Opus 4.6 + Codex (AI Development)
**Platform:** VESSL remilab-c0, RTX 4090 GPU

---

## 1. Design Specification

| Parameter | Value |
|-----------|-------|
| Target frequency | 2.4 GHz (Wi-Fi) |
| Substrate | FR4 (ε_r = 4.4, tan δ = 0.02) |
| Substrate thickness | 1.6 mm |
| Patch width (W) | 38.0 mm (analytical) |
| Patch length (L) | 29.4 mm (analytical) |
| Effective ε_r | 4.09 |
| Edge extension (ΔL) | 0.74 mm |
| Feed type | Probe feed (WirePort) at L/3 from edge |

### Analytical Formulas Used

```
W = c / (2·f₀) · √(2/(ε_r+1)) = 38.0 mm
ε_eff = (ε_r+1)/2 + (ε_r-1)/2 · (1+12h/W)^(-0.5) = 4.09
ΔL = 0.412·h · (ε_eff+0.3)(W/h+0.264) / ((ε_eff-0.258)(W/h+0.8)) = 0.74 mm
L = c / (2·f₀·√ε_eff) - 2·ΔL = 29.4 mm
```

## 2. Simulation Setup

| Parameter | Value |
|-----------|-------|
| Domain | 69 × 78 × 17 mm |
| Cell size (dx) | 0.5 mm |
| Grid | ~139 × 156 × 34 = ~736K cells |
| Timesteps | 4,000 |
| Boundary | CPML (8 layers) |
| Feed | WirePort (multi-cell, extent = 1.6 mm through substrate) |
| Runtime (GPU) | ~7 seconds (RTX 4090) |

### Geometry

```
z ↑
  │  ┌─────────────┐ ← PEC patch (z = h)
  │  │  FR4 (ε=4.4)│ ← Substrate (z = 0..h)
  │  └─────────────┘ ← PEC ground (z = 0)
  │     ↑ WirePort
  └─────────────────→ x
```

## 3. Results

### WirePort Validation

| Test | Result |
|------|--------|
| Distributed impedance | ✅ 6+ cells modified along wire |
| Vertical excitation | ✅ 11 z-cells excited simultaneously |
| Cavity resonance | ✅ 2.03 GHz peak in PEC box |
| API extent parameter | ✅ High-level API works (|Ez| = 12.7) |

**WirePort successfully excites the antenna structure.** The multi-cell feed correctly injects a vertical current through the substrate from ground to patch.

### S11 Extraction: NOT YET AVAILABLE

The time-domain probe at the feed shows field activity, but **proper S11 requires incident/reflected wave separation** which is not yet implemented for WirePort. The FFT of the raw probe signal shows the combined source + cavity response, not the calibrated reflection coefficient.

| What works | What's missing |
|------------|----------------|
| WirePort field excitation | WirePort ↔ SParamProbe integration |
| Correct geometry + materials | Incident wave subtraction |
| GPU execution (7 sec) | Calibrated S11/S21 extraction |

## 4. GPU Performance

| Metric | CPU (this server) | GPU (RTX 4090) |
|--------|-------------------|----------------|
| Patch antenna (736K cells, 4000 steps) | ~15 min (estimated) | **~7 sec** |
| Speedup | — | **~130x** |
| JIT compilation | included | included |

## 5. Identified Gaps

### Gap 1: WirePort S-param Extraction (v1.1 MUST)
WirePort needs integration with `SParamProbe` for proper incident/reflected wave separation. Current `LumpedPort` has this via `setup_lumped_port` + `apply_lumped_port` + `update_sparam_probe`. WirePort needs the same machinery distributed across N cells.

### Gap 2: WirePort in Compiled Runner (DONE)
Codex implemented `make_wire_port_sources()` which creates per-cell `SourceSpec` for the `jax.lax.scan` compiled runner. The high-level API `add_port(..., extent=...)` now works end-to-end.

### Gap 3: PEC via High-Sigma (Stability Issue)
Using `sigma=1e10` for PEC causes NaN instability with CPML. The example uses the `material="pec"` path which delegates to `apply_pec()` (works correctly). Users should NOT use high-sigma for PEC approximation.

## 6. Next Steps

1. **WirePort S-param probe integration** — distribute `SParamProbe` across wire cells, aggregate V/I DFTs
2. **Re-run patch antenna with proper S11** — validate resonance at ~2.4 GHz
3. **Compare with OpenEMS/Meep** — cross-validate patch antenna S11
4. **Add to automated test suite** — `test_patch_antenna_resonance()`

## 7. Conclusion

The WirePort implementation successfully addresses the single-cell lumped port limitation for probe-fed antennas. The antenna geometry and feed excitation work correctly on GPU (7 sec for 736K cells). The remaining gap is S-param extraction, which requires extending the existing `SParamProbe` machinery to multi-cell ports.

**rfx can now model probe-fed patch antennas structurally.** S11 calibration is the last step for a complete antenna simulation workflow.

# 2026-04-02 Next Steps Roadmap

## Commit: 7d9403c (main)
## Tests: 247 passed + 11 new = 258 total

## Validated Baseline
- PEC box cavity: 7.1 GHz (2.9% error vs analytical 7.3 GHz) ✓
- Patch antenna: 2.318 GHz (3.4% error vs design 2.4 GHz) ✓
- Convergence: monotonic, Richardson → 2.341 GHz (2.5%) ✓

## Implementation Roadmap (user-specified order)

### Phase 1: Auto-Configuration (`rfx/auto_config.py`)
User provides: geometry + `freq_range=(f_min, f_max)`
System derives: dx, domain, CPML, n_steps, source config

Rules (from Meep/OpenEMS research + our convergence data):
- dx = C0 / (f_max · √ε_max · 20)
- margin = 0.5 · λ_max  (Meep recommendation)
- cpml = 0.4 · λ_max  (OpenEMS: 8 cells × λ/20)
- n_steps = auto from field decay (Phase 3)
- Feature detection: warn if any geometry < 4·dx

### Phase 2: Harminv Integration
- Filter Diagonalization Method (FDM) for resonance extraction
- scipy.linalg based (generalized eigenvalue problem)
- Extracts: f_res, Q, amplitude, error per mode
- Much more accurate than FFT for short time series

### Phase 3: Run-Until-Converged
- Monitor field energy decay post-source
- Stop when energy < peak × EndCriteria (default 1e-5)
- Also check DFT accumulator stability for broadband S-params
- Like Meep's `stop_when_fields_decayed` + OpenEMS `EndCriteria`

### Phase 4: Port Fix
Two approaches (from Meep/OpenEMS analysis):
1. **Flux monitors** (Meep-style): area-integrated power flow, no port loading
2. **Wave separation** (OpenEMS-style): `uf_inc = 0.5*(U + I·Z0)`, proper normalization
   - Already have the math, but need proper V/I from non-loading probes
   - Separate excitation source from measurement plane

### Phase 5: PML Update
- Increase default: 0.4-0.5·λ_max (current: λ/10 = 0.1·λ)
- Add frequency-dependent profile tuning
- Validate: reflectivity < -40 dB across freq_range

### Phase 6: Validation Examples
1. **Patch antenna** — existing, 3.4% baseline
2. **Dielectric resonator** — high-Q, tests Harminv + convergence
3. **Waveguide coupler** — multi-port S-params, tests port model
4. **Cavity filter** — narrowband, tests frequency resolution
5. Field animation for each

### Phase 7: Field Animation
- `save_field_animation(sim, result, filename)` → MP4/GIF
- Options: component, slice axis, frame interval
- Use matplotlib.animation or imageio
- Export: snapshots → ffmpeg or imageio.mimwrite

## Meep/OpenEMS Research Report
Saved: `.omc/scientist/reports/20260402_fdtd_autoconfig_research.md`

Key findings:
- Both use mask-based PEC (not sigma) ✓ (rfx aligned)
- Meep EigenModeSource has ZERO port loading (ideal for resonance)
- OpenEMS calcPort wave separation is the universal S-param method
- Harminv (FDM) is dramatically better than FFT for resonance/Q
- Both auto-stop on field decay (not fixed n_steps)

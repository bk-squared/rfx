# rfx Simulation Methodology

## 1. Core Problem

FDTD simulation accuracy depends on ~7 configuration parameters that interact nonlinearly. Incorrect settings cause errors from 3% to 60%+, and the failure modes are silent (no crash, just wrong frequencies). The simulator must auto-derive these from geometry + frequency specification.

## 2. Validated Configuration Rules

Based on systematic convergence testing (VESSL runs #369367231424, #369367231427):

### 2.1 Cell Size (dx)

```
dx = min(lambda_min / 20, min_feature_size / 4)
```

- `lambda_min = C0 / freq_max` — shortest wavelength in simulation
- `min_feature_size` — thinnest geometry (substrate thickness, trace width, etc.)
- At least 4 cells per feature for geometric fidelity
- Grid dispersion error ∝ (dx/λ)² — at λ/20, dispersion < 0.1%

**Convergence data (2.4 GHz patch, h=1.6mm):**

| dx | h/dx | f_res | error |
|----|------|-------|-------|
| 1.0mm | 1.6 | 2.248 GHz | 6.3% |
| 0.5mm | 3.2 | 2.318 GHz | 3.4% |
| Richardson (dx→0) | ∞ | 2.341 GHz | 2.5% |

Extrapolated limit ~2.5% is from finite geometry effects (not grid error).

### 2.2 Domain Size

```
margin = max(lambda_max / 4, 8 * dx)
```

- `lambda_max = C0 / freq_min` — longest wavelength
- **Critical**: small margins cause domain resonance that dominates the spectrum
- At margin = 0.12λ, domain resonance at ~1.35 GHz masked the 2.4 GHz patch resonance
- At margin = 0.25λ, domain resonance eliminated

### 2.3 CPML Absorbing Boundary

```
cpml_layers = max(ceil(lambda_max / (10 * dx)), 8)
```

- At least λ/10 physical thickness
- Below this, low-frequency reflections corrupt the spectrum
- CFS-CPML with kappa_max scaling for wideband absorption

### 2.4 PEC Handling

**Must use true PEC mask (E-field zeroing), NOT high-sigma approximation.**

- High-sigma (σ=1e10): Ca ≈ -1 causes oscillation, shifts resonance by 20%+
- True PEC mask: component-specific tangential zeroing
  - Only zero E-components where PEC extends ≥2 cells in that direction
  - Preserves normal E at thin PEC surfaces (surface charge)
  - Uses neighbor analysis: `mask_ez = pec(i,j,k) AND (pec(i,j,k-1) OR pec(i,j,k+1))`

### 2.5 Timesteps

```
n_steps = ceil((T_source + T_ringdown) / dt)
T_source = 6 * tau = 6 / (f0 * bandwidth * pi)
T_ringdown = Q / (pi * f_min)  # Q ≈ 1/tan_delta for dielectric loss
```

- The source must fully decay before ring-down analysis
- Ring-down must be captured for spectral resolution
- For FR4 (tan_d=0.02): Q ≈ 50, T_ringdown ≈ 6.6ns at 2.4 GHz
- Total T_sim ≈ 10ns minimum for patch antennas

### 2.6 Source Model

| Use case | Source type | Rationale |
|----------|------------|-----------|
| Resonance detection (PEC) | `add_source()` → raw field source | No impedance loading; broadband; 0.00% on cavity |
| Resonance detection (CPML) | `add_source()` → J source (Cb/dx) | Prevents DC accumulation from PEC charge; 3.78% on patch |
| S-parameter extraction | Lumped/wire port | Needs V/I decomposition; requires calibration |
| Plane wave scattering | TFSF | Clean incident/scattered separation |

**Auto-selection**: `add_source()` automatically selects the right source type:
- `boundary='pec'` (closed cavity) → raw field add (broadband, exact cavity modes)
- `boundary='cpml'` (open structure) → J source with Cb/dx normalization

**Waveforms**:
- `ModulatedGaussian(f0, bw)` — sin(2πf₀t)·Gaussian envelope. **Zero DC** (∫s=0 exactly). Default for `add_source()`. Same as Meep's `GaussianSource`.
- `GaussianPulse(f0, bw)` — differentiated Gaussian. Near-zero DC (∫s=exp(-9)). Broader bandwidth. Used by ports.

**Port impedance loads the cavity.** A 50Ω port in a high-Q cavity damps the resonance so heavily that the spectral peak becomes a broad hump indistinguishable from the source spectrum. Use `add_source()` for resonance characterization.

### 2.7 Spectral Analysis

**For cavity-interior probes:**
1. **Window**: use only ring-down portion (after source decays): `start = ceil(2 * t0 / dt)`
2. **DC removal**: subtract mean of windowed signal
3. **Hann window**: reduce spectral leakage
4. **Peak finding**: resonance = spectral PEAK (not minimum — cavity amplifies at resonance)

**Common pitfalls:**
- `argmin` of normalized spectrum finds anti-resonance (off by ~2x)
- Unwindowed FFT includes DC from static PEC surface charge
- Short T_sim gives source-dominated spectrum, not cavity spectrum

## 3. Comparison with Meep / OpenEMS

| Feature | Meep | OpenEMS | rfx (current) | rfx (target) |
|---------|------|---------|---------------|--------------|
| Auto mesh | Manual `resolution` | `AutoMesh` with feature detection | Manual `dx` | Auto from geometry |
| PEC | Native material | Native | True PEC mask | Same (validated) |
| Domain sizing | Manual | Manual + recommendations | Manual | Auto from λ |
| CPML/PML | Built-in with auto thickness | Manual `AddPML` | Manual | Auto from λ |
| Port model | `EigenModeSource` + flux | `AddLumpedPort` + `CalcPort` | WirePort + DFT | Flux monitors needed |
| Convergence | `run_until_dft_decay` | Manual | Manual | Auto decay detection |
| Resonance | Harminv (harmonic inversion) | FFT | Windowed FFT | Harminv integration |
| S-params | Flux plane monitors | Port V/I + FFT | Wave decomposition | Need flux monitors |

### Key Meep advantages to adopt:
1. **Harminv**: exponential fitting of time series → much more accurate than FFT for finding resonance frequencies and Q factors from short time series
2. **run_until_dft_decay**: monitors DFT accumulator convergence and stops automatically
3. **Flux monitors**: area-integrated power flow, more robust than single-point probes

### Key OpenEMS advantages to adopt:
1. **AutoMesh**: analyzes geometry and places mesh lines at material boundaries
2. **CalcPort**: proper lumped port with V/I extraction and automatic impedance normalization
3. **NF2FF**: near-to-far-field as built-in post-processing (rfx already has this)

## 4. Auto-Configuration Architecture

### Proposed API

```python
sim = Simulation.from_geometry(
    geometry=[ground, substrate, patch],
    freq_range=(1e9, 4e9),      # simulation frequency band
    accuracy="standard",         # "draft" / "standard" / "high"
)
# Auto-derives: dx, domain, CPML, n_steps, source, spectral method

result = sim.run()
result.resonances      # Harminv-detected modes with Q factors
result.s_params        # Flux-monitor-based S-parameters
result.convergence     # Convergence metric (run two resolutions internally)
```

### Auto-config module (`rfx/auto_config.py`)

```python
def auto_configure(geometry, freq_range, accuracy="standard"):
    """Derive all simulation parameters from geometry + frequency."""

    f_min, f_max = freq_range
    lambda_min = C0 / f_max
    lambda_max = C0 / f_min

    # 1. Analyze geometry
    features = analyze_features(geometry)  # min thickness, extent, materials
    min_feature = features.min_thickness

    # 2. Cell size
    cells_per_lambda = {"draft": 10, "standard": 20, "high": 40}[accuracy]
    cells_per_feature = {"draft": 2, "standard": 4, "high": 8}[accuracy]
    dx = min(lambda_min / cells_per_lambda, min_feature / cells_per_feature)

    # 3. Domain
    margin_factor = {"draft": 0.15, "standard": 0.25, "high": 0.5}[accuracy]
    margin = lambda_max * margin_factor

    # 4. CPML
    cpml_thickness = lambda_max / 10
    cpml_layers = max(ceil(cpml_thickness / dx), 8)

    # 5. Timesteps
    dt = 0.99 * dx / (C0 * sqrt(3))
    T_source = 6 / (f_center * bandwidth * pi)
    Q_est = estimate_Q(features)  # from material loss tangent
    T_ringdown = Q_est / (pi * f_min)
    n_steps = ceil((T_source + T_ringdown) / dt)

    # 6. Convergence check
    if accuracy == "high":
        # Run at two resolutions, compare
        warn_if_not_converged = True

    return SimConfig(dx=dx, margin=margin, cpml_layers=cpml_layers,
                     n_steps=n_steps, ...)
```

### Feature analysis

```python
def analyze_features(geometry):
    """Extract critical dimensions from geometry shapes."""
    thicknesses = []
    for shape, material in geometry:
        if isinstance(shape, Box):
            dims = [abs(c2-c1) for c1,c2 in zip(shape.corner1, shape.corner2)]
            thicknesses.append(min(dims))
    return FeatureInfo(
        min_thickness=min(thicknesses),
        max_extent=max(max(dims) for dims in all_dims),
        has_pec=any(is_pec(m) for _, m in geometry),
        max_eps_r=max(m.eps_r for _, m in geometry),
        max_loss_tangent=...,
    )
```

## 5. Validation Requirements

For rfx to be credible as a simulator:

1. **PEC cavity**: <1% error vs analytical (verified: 2.9% at dx=1mm, converges)
2. **Patch antenna**: <3% error vs analytical with proper configuration (verified: 3.4% at dx=0.5mm)
3. **Cross-validation**: <5% vs Meep/OpenEMS for same geometry (test exists: test_meep_crossval.py)
4. **Convergence**: monotonic convergence with mesh refinement (verified)
5. **Auto-config**: user provides geometry + freq_range, simulator handles the rest

## 6. Next Steps

1. **Implement `auto_config.py`** — derive all params from geometry + freq
2. **Add Harminv** — exponential fitting for resonance/Q extraction (scipy.linalg based)
3. **Add flux monitors** — area-integrated S-parameter extraction
4. **Add `run_until_converged`** — auto-stop when DFT/spectrum stabilizes
5. **Convergence test suite** — automated multi-resolution testing for key benchmarks

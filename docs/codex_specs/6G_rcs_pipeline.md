# Codex Spec 6G: RCS Pipeline (TFSF + NTFF)

## Goal
Add a radar cross section (RCS) computation pipeline that combines TFSF
illumination with NTFF near-to-far-field transform.

## Background
RCS = far-field scattered power density normalized by incident power density.
The standard FDTD approach:
1. TFSF illuminates the target with a plane wave
2. Scattered field (outside TFSF box) is captured by NTFF box
3. NTFF computes far-field pattern
4. RCS(θ,φ) = 4πr² |E_scat|² / |E_inc|²

## Current State
- TFSF: working for normal incidence (+x/-x, Ez/Ey polarization)
- NTFF: `rfx/farfield.py` has `NTFFBox`, `compute_far_field()`,
  `radiation_pattern()`, `directivity()`
- No integration between the two exists
- No RCS computation function

## Deliverable

### 1. `compute_rcs()` in `rfx/rcs.py` (new file)

```python
def compute_rcs(
    grid,
    materials,
    n_steps: int,
    *,
    f0: float,
    bandwidth: float = 0.5,
    theta_inc: float = 0.0,     # incident angle (degrees), 0 = +x
    phi_inc: float = 0.0,       # azimuth (future, currently only x-axis)
    polarization: str = "ez",
    theta_obs: jnp.ndarray | None = None,  # observation angles
    phi_obs: jnp.ndarray | None = None,
    freqs: jnp.ndarray | None = None,
    boundary: str = "cpml",
    cpml_layers: int = 8,
) -> RCSResult:
    """Compute radar cross section of the scatterer defined in materials.

    Returns
    -------
    RCSResult
        freqs, theta, phi, rcs_dbsm (dBsm), rcs_linear (m²)
    """
```

The function should:
1. Set up TFSF source enclosing the scatterer
2. Set up NTFF box just outside the TFSF box (in scattered-field region)
3. Run the simulation
4. Extract DFT fields on NTFF surfaces
5. Compute far-field via existing `compute_far_field()`
6. Normalize by incident field amplitude to get RCS

### 2. `RCSResult` named tuple

```python
class RCSResult(NamedTuple):
    freqs: np.ndarray        # (n_freqs,)
    theta: np.ndarray        # (n_theta,)
    phi: np.ndarray          # (n_phi,)
    rcs_dbsm: np.ndarray     # (n_freqs, n_theta, n_phi) in dBsm
    rcs_linear: np.ndarray   # (n_freqs, n_theta, n_phi) in m²
    monostatic_rcs: np.ndarray | None  # (n_freqs,) backscatter RCS
```

### 3. High-level API: `Simulation.compute_rcs()`

Convenience method on the Simulation class.

### 4. `plot_rcs()` in `rfx/visualize.py`

Add polar/rectangular plot function for RCS patterns.

### 5. Tests in `tests/test_rcs.py`

**Test 1: `test_rcs_pec_plate_normal_incidence`**
- PEC plate (thin box) normal to +x TFSF
- At normal incidence, RCS of a plate = 4π·A²/λ² (physical optics approx)
- For a plate of size a×b at wavelength λ: RCS ≈ 4π(ab)²/λ²
- Assert: computed RCS within 3 dB of analytical at center frequency

**Test 2: `test_rcs_pec_sphere_mie`** (if sphere geometry works well)
- PEC sphere, compare with Mie series (exact analytical solution)
- At ka ≈ 1-3 (resonance region), Mie gives exact RCS
- Assert: monostatic RCS within 3 dB of Mie at center frequency
- NOTE: if Mie computation is complex, use a simpler validation

**Test 3: `test_rcs_result_structure`**
- Verify RCSResult has correct shapes and units
- dBsm = 10*log10(rcs_m²)
- monostatic_rcs corresponds to backscatter direction

## Constraints
- Use existing TFSF (normal incidence only — oblique awaits 6F fix)
- Use existing NTFF infrastructure
- Use existing DFT plane probes for NTFF surface accumulation
- Each test < 120 seconds
- Focus on normal incidence for now (θ_inc=0)

## Verification
Run: `pytest -xvs tests/test_rcs.py`
All tests must pass.

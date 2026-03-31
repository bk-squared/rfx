# Codex Spec 6E4: Overlap Integral Modal Extraction

## Goal
Replace the current V/I line-integral modal extraction with a full 2D
cross-sectional overlap integral for more accurate waveguide S-parameters.

## Background
Current approach in `modal_voltage()` / `modal_current()`:
- `V = ∫ E_t · e_mode dA` (single inner product)
- `I = ∫ H_t · h_mode dA` (single inner product with H averaged to E plane)

Meep's approach (overlap integral):
- `α± = ∫∫ (E_sim × H*_mode ± E*_mode × H_sim) · n̂ dA`
- Directly gives forward/backward modal amplitudes without intermediate V/I

The overlap integral is more accurate because it uses the full vector
cross-product rather than scalar projections, and naturally handles
the Yee stagger correctly.

## Deliverable

### 1. `overlap_modal_amplitude()` in `rfx/sources/waveguide_port.py`

```python
def overlap_modal_amplitude(
    state,
    cfg: WaveguidePortConfig,
    x_idx: int,
    dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute forward and backward modal amplitudes via overlap integral.

    Returns (a_forward, a_backward) as scalars.

    The overlap integral for a mode with profile (e_mode, h_mode):
      P_forward  = ∫∫ (E_sim × H*_mode) · n̂ dA
      P_backward = ∫∫ (E*_mode × H_sim) · n̂ dA
      a_forward  = 0.5 * (P_forward + P_backward) / C_mode
      a_backward = 0.5 * (P_forward - P_backward) / C_mode

    where C_mode = ∫∫ (e_mode × h*_mode) · n̂ dA (mode normalization).

    For x-normal port: n̂ = x̂
      (E × H) · x̂ = Ey*Hz - Ez*Hy
    """
```

The implementation must:
1. Extract E_sim (ey, ez) and H_sim (hy, hz) on the aperture plane at x_idx
2. H fields must be averaged to the E plane (existing pattern in `modal_current`)
3. Compute cross products with mode profiles
4. Integrate over the aperture (sum * dx²)
5. Normalize by mode self-overlap C_mode
6. Generalize for y/z-normal ports using cfg.e_u_component etc.

### 2. `update_waveguide_port_probe_overlap()` — DFT accumulation variant

A new probe update function that accumulates overlap-integral-based forward/backward
wave DFTs instead of the current V/I DFTs. Add new fields to WaveguidePortConfig:
- `a_fwd_ref_dft`, `a_bwd_ref_dft` (at reference plane)
- `a_fwd_probe_dft`, `a_bwd_probe_dft` (at probe plane)

### 3. `extract_waveguide_sparams_overlap()` — extraction from overlap DFTs

Compute S11 = a_bwd_ref / a_fwd_ref, S21 = a_fwd_probe / a_fwd_ref.

### 4. Tests in `tests/test_overlap_extraction.py`

**Test 1: `test_overlap_vs_vi_straight_waveguide`**
- Empty waveguide, compare overlap S21 vs V/I S21
- Overlap should have |S21| closer to 1.0 than V/I

**Test 2: `test_overlap_mode_normalization`**
- Verify C_mode = ∫(e×h*)·n̂ dA is real and positive for TE10

**Test 3: `test_overlap_passivity`**
- With dielectric obstacle, overlap S-matrix should satisfy passivity better
- Σ|S|² < 1.05

## Constraints
- Add new functions alongside existing ones — do NOT remove modal_voltage/modal_current
- Keep backward compatibility: existing extract_waveguide_sparams() unchanged
- Each test < 120 seconds
- Read existing waveguide_port.py very carefully, especially _plane_field, _plane_h_field

## Verification
Run: `pytest -xvs tests/test_overlap_extraction.py`

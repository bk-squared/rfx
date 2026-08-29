"""SPEC-01 multiband NU envelope — explicit-profile fixtures.

Every fixture returns an EXPLICIT ``dz_profile`` numpy vector. The
``rfx.auto_config`` builders (``_make_dz_profile`` / ``smooth_grading`` /
``apply_thirds_rule``) are deliberately NOT used anywhere in this lane —
the witnesses probe SOLVER properties and must not be confounded by
builder behaviour (SPEC-01 §3; #775 merged its preserve_regions fix but
the separation stands regardless).

Conventions
-----------
* All lengths in metres, float64.
* "abrupt" variant: band-to-band cell-size step is a single jump of
  exactly the band ratio ``r`` (the ratio cap applied in one step).
* "smooth" variant: a geometric ramp with per-step ratio <= SMOOTH_STEP
  (1.3) is INSERTED between bands; the band cells themselves are kept.
  Built by a plain loop right here (explicit vector, not a builder).
"""

from __future__ import annotations

import numpy as np

SMOOTH_STEP = 1.3

# ---------------------------------------------------------------------------
# P-A / W1: symmetric fine-coarse-fine-coarse-fine multiband z profile
# ---------------------------------------------------------------------------

DZ_FINE = 1.0e-3          # fine cell, 1 mm  (lambda0/30 at 10 GHz)
N_FINE = 40               # cells per fine band
N_COARSE = 30             # cells per coarse band
DXY = 1.5e-3              # uniform transverse cell (x and y)
B_Y = 30e-3               # y extent (TE10 'b'); 20 cells of 1.5 mm
A_X = 4.5e-3              # x extent; 3 cells (x-invariant fixtures)


def _ramp(d_from: float, d_to: float) -> list[float]:
    """Geometric ramp cells strictly between two band cell sizes with
    per-step ratio <= SMOOTH_STEP. Returns [] when the direct step is
    already within SMOOTH_STEP."""
    hi, lo = max(d_from, d_to), min(d_from, d_to)
    ratio = hi / lo
    if ratio <= SMOOTH_STEP * (1 + 1e-12):
        return []
    n = int(np.ceil(np.log(ratio) / np.log(SMOOTH_STEP)))
    rho = ratio ** (1.0 / n)
    cells = [lo * rho ** i for i in range(1, n)]
    if d_from > d_to:
        cells = cells[::-1]
    return cells


def pa_profile(r: float, variant: str = "abrupt",
               dz_fine: float = DZ_FINE,
               n_fine: int = N_FINE, n_coarse: int = N_COARSE) -> np.ndarray:
    """Symmetric 5-band fine/coarse/fine/coarse/fine z profile (P-A)."""
    f = [dz_fine] * n_fine
    c = [r * dz_fine] * n_coarse
    if variant == "abrupt":
        parts = f + c + f + c + f
    elif variant == "smooth":
        up = _ramp(dz_fine, r * dz_fine)
        dn = _ramp(r * dz_fine, dz_fine)
        parts = f + up + c + dn + f + up + c + dn + f
    else:
        raise ValueError(variant)
    return np.asarray(parts, dtype=np.float64)


# ---------------------------------------------------------------------------
# W2: single-transition profile (fine runway | transition | coarse runway)
# ---------------------------------------------------------------------------

N_FINE_RUNWAY = 140       # fine cells before the transition
N_COARSE_RUNWAY = 150     # coarse cells after it
K_SRC = 40                # source plane (fine region, cells from z-lo wall)
K_PRB = 80                # probe plane  (fine region)


def single_transition_profile(r: float, variant: str = "abrupt",
                              dz_fine: float = DZ_FINE) -> np.ndarray:
    f = [dz_fine] * N_FINE_RUNWAY
    c = [r * dz_fine] * N_COARSE_RUNWAY
    ramp = _ramp(dz_fine, r * dz_fine) if variant == "smooth" else []
    return np.asarray(f + ramp + c, dtype=np.float64)


def uniform_reference_profile(n_extra_fine: int = 260,
                              dz_fine: float = DZ_FINE) -> np.ndarray:
    """Uniform-fine B-run profile for the 2-run differencing: identical
    fine runway (source+probe cells bit-identical), then fine cells
    continuing far enough that the far wall's return arrives later than
    the A-run gate."""
    return np.asarray([dz_fine] * (N_FINE_RUNWAY + n_extra_fine),
                      dtype=np.float64)


# ---------------------------------------------------------------------------
# P-B / W1-3D: PEC box with the same symmetric multiband z profile
# ---------------------------------------------------------------------------

PB_NXY = 32               # transverse cells (uniform 1.5 mm)


def pb_domain_xy() -> tuple[float, float]:
    return (PB_NXY * DXY, PB_NXY * DXY)


# ---------------------------------------------------------------------------
# P-C / W4: microstrip-type resonator, 3 fine bands, explicit dz vector
# ---------------------------------------------------------------------------
# Physical layout (z, from board bottom):
#   [0.0, 1.5mm]   substrate  eps_r=4.3   FINE band
#   [1.5, 3.0mm]   trace level (PEC trace at z=1.5mm)  FINE band
#   [3.0, 7.5mm]   air        COARSE (capped transitions)
#   [7.5, 9.0mm]   upper dielectric eps_r=2.2  FINE band
#   [9.0, 13.5mm]  air to PEC lid  COARSE
# Transverse: a=27mm x b=22.5mm PEC box. PEC trace 13.5mm x 4.5mm at
# z=1.5mm, x in [6.75,20.25]mm, y in [9.0,13.5]mm. All trace edges,
# source and probe positions sit on multiples of 2.25mm so every
# refinement scale s in {1,1.5,2,3} rasterizes the SAME geometry
# (no staircase-alignment confound in the convergence fit).

PC_A = 27e-3
PC_B = 22.5e-3
PC_H_SUB = 1.5e-3
PC_H_TRACE_BAND = 1.5e-3
PC_AIR1 = 4.5e-3
PC_H_UPPER = 1.5e-3
PC_AIR2 = 4.5e-3
PC_EPS_SUB = 4.3
PC_EPS_UPPER = 2.2
PC_DX0 = 0.75e-3          # base transverse cell (s=1)
PC_DZF0 = 0.25e-3         # base fine z cell (s=1)
PC_SCALES = (1.0, 1.5, 2.0, 3.0)
RATIO_CAP = 1.4


def _sym_air_band(length: float, dzf: float, cap: float = RATIO_CAP) -> list[float]:
    """Air band that starts AND ends at fine size dzf, ramping up by
    <=cap to a plateau and back down, summing exactly to `length`.
    Explicit construction: choose plateau size and counts numerically."""
    up = []
    d = dzf
    # ramp up to at most 4*dzf
    while d * cap <= 4 * dzf + 1e-15:
        d = d * cap
        up.append(d)
    ramp_len = 2 * sum(up)
    plateau_d = up[-1] if up else dzf
    rem = length - ramp_len
    if rem < 0:
        # band too short for full ramp: shrink ramp
        while up and rem < 0:
            up = up[:-1]
            ramp_len = 2 * sum(up)
            plateau_d = up[-1] if up else dzf
            rem = length - ramp_len
    n_plateau = max(0, int(np.floor(rem / plateau_d)))
    residual = rem - n_plateau * plateau_d
    cells = up + [plateau_d] * n_plateau + up[::-1]
    if residual > 1e-12:
        # distribute residual evenly over the plateau+ramp cells,
        # keeping every neighbour ratio within the cap: scale the
        # plateau cells up uniformly (ratio change < cap slack).
        n_scale = max(1, n_plateau)
        add = residual / n_scale
        if n_plateau:
            cells = up + [plateau_d + add] * n_plateau + up[::-1]
        else:
            # no plateau: scale the top ramp cell pair
            cells = up[:-1] + [up[-1] + residual / 2] * 1 + [up[-1] + residual / 2] + up[::-1][1:] if up else [length]
    s = sum(cells)
    # final exactness nudge on the middle cell (float dust only)
    mid = len(cells) // 2
    cells[mid] += length - s
    assert abs(sum(cells) - length) < 1e-9
    ratios = [cells[i + 1] / cells[i] for i in range(len(cells) - 1)]
    assert all(1 / (cap * 1.0001) <= q <= cap * 1.0001 for q in ratios), ratios
    assert cells[0] / dzf <= cap * 1.0001 and cells[-1] / dzf <= cap * 1.0001
    return cells


def pc_dz_profile_sym(scale: float) -> np.ndarray:
    """P-C dz vector with cap-respecting SYMMETRIC air bands (the one
    the W4 arms actually use — every neighbour ratio <= 1.4 everywhere,
    including re-entry into fine bands)."""
    dzf = PC_DZF0 * scale
    prof: list[float] = []
    for band_len in (PC_H_SUB, PC_H_TRACE_BAND):
        n = int(round(band_len / dzf))
        assert abs(n * dzf - band_len) < 1e-9, (band_len, dzf)
        prof += [dzf] * n
    prof += _sym_air_band(PC_AIR1, dzf)
    prof += [dzf] * int(round(PC_H_UPPER / dzf))
    prof += _sym_air_band(PC_AIR2, dzf)
    out = np.asarray(prof, dtype=np.float64)
    assert abs(out.sum() - (PC_H_SUB + PC_H_TRACE_BAND + PC_AIR1
                            + PC_H_UPPER + PC_AIR2)) < 1e-9
    return out


def pc_uniform_profile(scale: float) -> np.ndarray:
    """Uniform-fine control dz vector at the same scale (dzf everywhere)."""
    dzf = PC_DZF0 * scale
    total = PC_H_SUB + PC_H_TRACE_BAND + PC_AIR1 + PC_H_UPPER + PC_AIR2
    n = int(round(total / dzf))
    assert abs(n * dzf - total) < 1e-9
    return np.full(n, dzf, dtype=np.float64)


# ---------------------------------------------------------------------------
# W5: small multiband profile for the AD witness
# ---------------------------------------------------------------------------

def w5_profile() -> np.ndarray:
    """Small fine-coarse-fine multiband dz vector (distinct per-cell
    perturbations added so jnp.min(dz) has no ties — the ledger's
    min-tie subgradient caveat)."""
    base = [0.30e-3] * 5 + [0.42e-3] * 4 + [0.30e-3] * 5
    rng = np.random.default_rng(20260829)
    jitter = 1.0 + 0.01 * rng.standard_normal(len(base))
    return np.asarray(base, dtype=np.float64) * jitter

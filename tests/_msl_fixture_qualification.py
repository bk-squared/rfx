"""Physical checks for the #931 aligned live MSL coupon, not snapshot fitting.

Budgets are specified before its first solve: existing integration magnitude
bounds, the repository's -40 dB settling and 1e3 conditioning screens, plus
10% quasi-static Z0/beta model-and-grid budgets and 5 degree electrical-length
phase budget. All ten 0.5--5 GHz bins are inspected; the independently modelled
physics band remains the pre-existing 3--4.5 GHz band. A capture also needs the
same fixed drawing on a refined mesh and an independent confirmation run.
"""
from __future__ import annotations

import numpy as np

# Production S uses probe-0 wave amplitudes, not the launch positions.
# These physical reference planes are checked against compiled port configs.
REFERENCE_PLANES_M = (5e-3, 9e-3)


def quasi_static_line(freqs, *, width=600e-6, height=254e-6,
                      eps_r=3.66, length=REFERENCE_PLANES_M[1] - REFERENCE_PLANES_M[0]):
    """Closed-form thin-strip quasi-TEM approximation (Pozar section 3.7).

    This evaluates the drawing directly, without the simulation raster,
    source's fitted impedance, de-embedding or passivity projection. It is
    an approximate model, not an external full-wave accuracy certification.
    The output waves use the line's own Z0, as the production MSL API does.
    """
    u = width / height
    eeff = (eps_r + 1) / 2 + (eps_r - 1) / (2 * np.sqrt(1 + 12 / u))
    z0 = 120 * np.pi / (np.sqrt(eeff) * (u + 1.393 + .667 * np.log(u + 1.444)))
    beta = 2 * np.pi * np.asarray(freqs) * np.sqrt(eeff) / 299792458.
    return float(z0), beta, np.exp(-1j * beta * length)


def qualify_result(result):
    """Return all failures and raw per-bin witnesses; never modify a result."""
    freqs = np.asarray(result.freqs, dtype=float)
    projected = np.asarray(result.S)
    raw = projected if result.S_raw is None else np.asarray(result.S_raw)
    z0 = np.asarray(result.Z0)
    beta = np.asarray(result.beta)
    oracle_z0, oracle_beta, oracle_s21 = quasi_static_line(freqs)
    # jnp.linspace(float32) represents the intended4.5 GHz bin as
    # 4,500,000,256 Hz. Include endpoint rounding by one float32 ULP;
    # this restores that bin to the checks, not a wider numerical gate.
    lower = float(np.nextafter(np.float32(3e9), np.float32(-np.inf)))
    upper = float(np.nextafter(np.float32(4.5e9), np.float32(np.inf)))
    band = (freqs >= lower) & (freqs <= upper)
    sv = np.full(freqs.shape, np.inf)
    finite_slices = np.all(np.isfinite(raw), axis=(0, 1))
    if np.any(finite_slices):
        sv[finite_slices] = np.linalg.svd(
            np.moveaxis(raw[..., finite_slices], -1, 0), compute_uv=False)[:, 0]
    phase_error = np.angle(raw[1, 0] / oracle_s21, deg=True)
    failures = []

    def require(ok, message):
        if not bool(ok):
            failures.append(message)

    require(np.all(np.isfinite(raw)), "non-finite raw S")
    require(np.all(np.isfinite(projected)), "non-finite projected S")
    require(np.any(band), "missing predeclared 3--4.5 GHz physics band")
    require(result.assembly == "multi_drive_solve", "fallback or missing multi-drive solve")
    reliable = getattr(result, "reliable", None)
    require(reliable is not None and np.all(reliable), "unexcited/unreliable full-spectrum bins")
    settling = getattr(result, "settling_db", None)
    require(settling is not None and np.all(np.isfinite(settling))
            and np.all(np.asarray(settling) <= -40), "drive ringdown exceeds -40 dB")
    cond = getattr(result, "cond_a", None)
    require(cond is not None and np.all(np.isfinite(cond))
            and np.all(np.asarray(cond) <= 1e3), "drive conditioning exceeds 1e3")
    railed = getattr(result, "beta_railed", None)
    require(railed is not None and not np.any(np.asarray(railed)[..., band]),
            "beta fit is railed or missing in physical band")
    if np.any(band):
        require(np.all(np.abs(raw[[0, 1], [0, 1]][:, band]) < .15),
                "raw physical-band reflection exceeds existing .15 bound")
        transmission = np.abs(raw[[1, 0], [0, 1]][:, band])
        require(np.all((transmission > .90) & (transmission < 1.05)),
                "raw physical-band transmission misses existing (.90,1.05) bounds")
        require(np.all(sv[band] <= 1.05), "raw physical-band singular value exceeds 1.05")
        require(np.all(np.abs(raw[0, 1, band] - raw[1, 0, band]) < .01),
                "raw complex reciprocity exceeds .01")
        require(np.all(np.abs(z0[..., band].real / oracle_z0 - 1) < .10),
                "fitted physical-band Z0 differs from drawing model by >=10%")
        require(np.all(np.abs(beta[band].real / oracle_beta[band] - 1) < .10),
                "fitted physical-band beta differs from drawing model by >=10%")
        require(np.all(np.abs(phase_error[band]) < 5),
                "raw S21 electrical-length phase differs from drawing by >=5 degrees")
    return {
        "failures": failures,
        "physics_band_hz": [3e9, 4.5e9],
        "physics_band_indices": np.flatnonzero(band).tolist(),
        "reference_planes_m": list(REFERENCE_PLANES_M),
        "oracle_z0_ohm": oracle_z0,
        "oracle_beta_rad_per_m": oracle_beta.tolist(),
        "raw_sigma_max": sv.tolist(),
        "raw_s21_phase_error_deg": phase_error.tolist(),
        "raw_complex_reciprocity_abs": np.abs(raw[0, 1] - raw[1, 0]).tolist(),
    }

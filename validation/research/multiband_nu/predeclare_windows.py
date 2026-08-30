"""Compute the pre-declared falsifier windows for SPEC-01 W1..W3.

RUN BEFORE ANY FDTD MEASUREMENT; the printed table is frozen into
docs/design_notes/20260829_spec01_multiband_predeclaration.md and the
committed JSON. Window sources are model-class only (float-accumulation
model for F-S1; the exact discrete frequency-domain chain model for
F-S2/F-S3) — no FDTD time-stepping output enters this script.

Usage:
    PYTHONPATH=. .venv/bin/python -m validation.research.multiband_nu.predeclare_windows
"""

from __future__ import annotations

import json
import numpy as np

from . import fixtures as fx
from .chain_model import bloch_kz, power_rt, s0_sy
from .harness import build_pec_fixture

F0 = 10e9
R_VALUES = (1.1, 1.2, 1.4, 1.5, 2.0)

# --- F-S1 float-accumulation window (derivation in the note) -------------
U32 = 2.0 ** -24          # float32 unit roundoff
FS1_K = 20.0              # 2 (quadratic) x sqrt(10) (rounding count) x ~3 safety
FS1_SLOPE_MAX = 0.75      # log-log growth-trend cap (0.5 = random walk)
FS1_FLOOR = 50.0 * U32    # trend fit only evaluated above this drift


def main():
    out = {"f0_hz": F0, "fs1": {
        "envelope": "d(n) <= K*u*sqrt(n), K=20, u=2^-24",
        "cap_at_1e6": FS1_K * U32 * np.sqrt(1e6),
        "slope_max": FS1_SLOPE_MAX,
        "trend_floor": FS1_FLOOR,
    }}

    # dt is set by the fixture geometry (dz_min = fine cell for every r)
    grid, _ = build_pec_fixture(
        fx.single_transition_profile(1.4), (fx.A_X, fx.B_Y), fx.DXY)
    dt = float(grid.dt)
    dy = float(grid.dy)
    b = fx.B_Y
    out["dt"] = dt
    s0, sy = s0_sy(F0, dt, dy, b)
    kz_fine = bloch_kz(F0, dt, dy, b, fx.DZ_FINE)
    out["kz_fine"] = kz_fine
    out["cells_per_guided_wavelength_fine"] = 2 * np.pi / kz_fine / fx.DZ_FINE

    # model self-check: uniform profile must not scatter
    prof_u = np.full(80, fx.DZ_FINE)
    Ru, Tu, _ = power_rt(prof_u, 10, 10, F0, dt, dy, b)
    assert Ru < 1e-10 and abs(Tu - 1) < 1e-9, (Ru, Tu)
    out["model_selfcheck_uniform_R"] = Ru

    # --- F-S2: single-transition reflection windows ----------------------
    fs2 = {}
    for r in R_VALUES:
        row = {}
        for variant in ("abrupt", "smooth"):
            prof = fx.single_transition_profile(r, variant)
            R, T, Tpw = power_rt(prof, 20, 20, F0, dt, dy, b)
            row[variant] = {
                "R_model": R,
                "R_model_db": 20 * np.log10(max(R, 1e-300)),
                "window": max(3.0 * R, 3e-5),
                "window_db": 20 * np.log10(max(3.0 * R, 3e-5)),
                "T_model": T, "T_power_amp": Tpw,
            }
        fs2[str(r)] = row
    out["fs2"] = fs2
    out["fs2_rule"] = ("fires (r<=1.4 claims only) if R_meas(f0) > "
                       "max(3*R_model, 3e-5); r>1.4 arms are reference "
                       "points outside the claimed envelope")

    # --- F-S3: P-A round-trip amplitude drift windows --------------------
    # W3 fixture profile: fine 120 | coarse 30 | fine 40 | coarse 30 | fine 120
    fs3 = {}
    for r in R_VALUES:
        row = {}
        for variant in ("abrupt", "smooth"):
            c = [r * fx.DZ_FINE] * fx.N_COARSE
            f_mid = [fx.DZ_FINE] * fx.N_FINE
            lead = [fx.DZ_FINE] * 120
            ramp_u = fx._ramp(fx.DZ_FINE, r * fx.DZ_FINE) if variant == "smooth" else []
            ramp_d = fx._ramp(r * fx.DZ_FINE, fx.DZ_FINE) if variant == "smooth" else []
            prof = np.asarray(lead + ramp_u + c + ramp_d + f_mid
                              + ramp_u + c + ramp_d + lead, np.float64)
            R, T, _ = power_rt(prof, 40, 40, F0, dt, dy, b)
            row[variant] = {
                "T_model": T,
                "one_minus_T_model": 1 - T,
                "window_halfwidth": max(3e-4, 0.5 * abs(1 - T)),
            }
        fs3[str(r)] = row
    out["fs3"] = fs3
    out["fs3_rule"] = ("fires if | |T_meas(f0)| - |T_model(f0)| | > "
                       "max(3e-4, 0.5*|1-|T_model||)")

    print(json.dumps(out, indent=2, default=float))
    with open("validation/research/multiband_nu/results/predeclared_windows.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)


if __name__ == "__main__":
    main()

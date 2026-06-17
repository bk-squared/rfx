"""Measure + commit the waveguide broad-E5 magnitude NOISE FLOOR (T2.4).

The broad-E5 envelope reported ``noise_floor_baseline: 0.0021`` as a bare
constant. T2.4 (option A) replaces that with a CLEAN-CHECKOUT-VERIFIABLE
measurement: on a matched empty guide the analytic truth is |S11|=0, |S21|=1, so
the residual ``max(|S11|, |1-|S21||)`` is the irreducible |S| extraction floor —
the smallest disagreement the gate could ever demand. This is NOT a fitted
tolerance; it is a measured property of the extractor on a known-exact geometry.

Writes ``tests/fixtures/waveguide_broad_e5/noise_floor_measurement.json``
(git-tracked). Re-run after any change to the extractor / flux path:

    PYTHONPATH=. python scripts/diagnostics/measure_waveguide_noise_floor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "fixtures" / "waveguide_broad_e5" / "noise_floor_measurement.json"

DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
# Single-mode band of the 40x20 mm guide (fc_TE10 = 3.75 GHz); a couple of mesh
# points so the reported floor is not a single-config fluke.
BAND_HZ = (4.5e9, 6.5e9)
N_FREQS = 8
DX_VALUES_M = (0.0015, 0.002)  # two explicit single-mode-band meshes (~30, ~23 cells/lambda)


def _empty_guide_residual(dx):
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    f0 = float(freqs.mean())
    bw = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(freq_max=float(freqs[-1]), domain=DOMAIN, boundary="cpml",
                     cpml_layers=10, dx=dx)
    for x, d, name in ((PORT_LEFT_X, "+x", "left"), (PORT_RIGHT_X, "-x", "right")):
        sim.add_waveguide_port(
            x, direction=d, mode=(1, 0), mode_type="TE", freqs=jnp.asarray(freqs),
            f0=f0, bandwidth=bw, waveform="modulated_gaussian", n_modes=1, name=name,
        )
    r = sim.compute_waveguide_s_matrix(num_periods=40, normalize="flux")
    s = np.asarray(r.s_params)
    idx = {n: i for i, n in enumerate(r.port_names)}
    s11 = np.abs(s[idx["left"], idx["left"], :])
    s21 = np.abs(s[idx["right"], idx["left"], :])
    return {
        "dx_m": float(dx),
        "max_s11_residual": float(s11.max()),          # ideal 0
        "max_s21_residual": float(np.abs(1.0 - s21).max()),  # ideal 0
        "noise_floor": float(max(s11.max(), np.abs(1.0 - s21).max())),
    }


def main():
    cases = [_empty_guide_residual(dx) for dx in DX_VALUES_M]
    noise_floor = max(c["noise_floor"] for c in cases)
    payload = {
        "schema": "waveguide_broad_e5_noise_floor",
        "claim": "irreducible |S| magnitude floor on a matched empty guide "
                 "(analytic |S11|=0, |S21|=1), normalize='flux'",
        "geometry": {"domain_m": list(DOMAIN), "fc_te10_hz": 3.75e9,
                     "band_hz": list(BAND_HZ), "n_freqs": N_FREQS},
        "cases": cases,
        "noise_floor": noise_floor,
        "note": "Replaces the bare noise_floor_baseline constant with a "
                "clean-checkout-verifiable measurement (T2.4). NOT a fitted "
                "tolerance — a measured extractor property on a known-exact case.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"noise_floor = {noise_floor:.5f}")
    for c in cases:
        print(f"  dx={c['dx_m']*1e3:.3f} mm  |S11|res={c['max_s11_residual']:.5f}  "
              f"|1-S21|res={c['max_s21_residual']:.5f}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

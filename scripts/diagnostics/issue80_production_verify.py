# ruff: noqa: E402, E702, E401
"""Issue #80 — verify the PRODUCTION compute_msl_s_matrix V·I split gives a PASSIVE patch S11 once
probe 0 clears the source near-field. The production diagonal S11 is already the local V·I split
(_sparams.py:1015-1019, a=(V+Z0·I)/2, b=(V-Z0·I)/2, closed Ampère-loop I, HJ Z0) — the same thing
validated offline. The historical |S11|=8.94 came from the default n_probe_offset (~5 cells ≈ 1mm)
placing probe 0 INSIDE the ~13-cell source transient. Sweep n_probe_offset and confirm:
  small offset → non-passive (reproduce 8.94 family);  cleared offset → PASSIVE, dip toward 9.2.
Run: python issue80_production_verify.py <n_probe_offset>   (cells; e.g. 5 vs 18)."""
from __future__ import annotations

import os, sys, warnings
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402
from rfx import Box, Simulation  # noqa: E402
from rfx.sources import GaussianPulse  # noqa: E402

EPS_R = 3.38; H_SUB = 0.787e-3; W = 10.129e-3; L = 8.595e-3; W_MSL = 1.8e-3
Z_GND = 4e-3; PORT_MARGIN = 5e-3; FEED_LEN = 8.0e-3
DOM = (29.747e-3, 18.130e-3, 12.787e-3)
FREQS = np.linspace(6e9, 14e9, 41)
NUM_PERIODS = 120.0


def build(dx, n_probe_offset):
    sim = Simulation(freq_max=15e9, domain=DOM, dx=dx, cpml_layers=8, boundary="cpml")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    z_gnd_hi = Z_GND + dx; z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + dx
    x_patch0 = PORT_MARGIN + FEED_LEN; y_c = DOM[1] / 2.0
    sim.add(Box((0, 0, Z_GND), (DOM[0], DOM[1], z_gnd_hi)), material="pec")
    sim.add(Box((0, 0, z_sub_lo), (DOM[0], DOM[1], z_sub_hi)), material="ro4003c")
    sim.add(Box((0, y_c - W_MSL / 2, z_tr_lo), (x_patch0, y_c + W_MSL / 2, z_tr_hi)), material="pec")
    sim.add(Box((x_patch0, y_c - W / 2, z_tr_lo), (x_patch0 + L, y_c + W / 2, z_tr_hi)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, y_c, z_sub_lo), width=W_MSL, height=H_SUB,
                     direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6), eps_r_sub=EPS_R, mode="laplace",
                     n_probes=4, n_probe_offset=n_probe_offset)
    return sim


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "d"
    offs = None if arg in ("d", "default") else int(arg)   # None -> floored default
    dx = H_SUB / 4.0
    sim = build(dx, offs)
    resolved = sim._msl_ports[0].n_probe_offset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msgs = sim.preflight()
        res = sim.compute_msl_s_matrix(freqs=jnp.asarray(FREQS), num_periods=NUM_PERIODS)
    nf_warn = [m for m in (msgs or []) if "fringing transient" in m]
    S11 = np.asarray(res.S)[0, 0, :]
    mag = np.abs(S11); db = 20 * np.log10(mag + 1e-30); fr = FREQS / 1e9; i = int(np.argmin(mag))
    print(f"n_probe_offset arg={arg} -> RESOLVED={resolved} cells (~{resolved*dx*1e3:.1f}mm)  dx={dx*1e6:.0f}µm")
    print(f"  preflight near-field warning fired: {bool(nf_warn)}")
    print(f"  max|S11|={mag.max():.3f}  PASSIVE={mag.max()<=1.05}  dip @ {fr[i]:.2f} GHz ({db[i]:+.2f} dB)")
    print("  |S11| curve: " + " ".join(f"{fr[k]:.0f}:{mag[k]:.2f}" for k in range(0, 41, 3)))


if __name__ == "__main__":
    main()

"""Issue #40 physics check: n_warmup forward bit-match + basic stability."""

from __future__ import annotations

import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse


def build():
    dx = 1e-3
    dz_sub = 0.25e-3
    trans = np.geomspace(dx, dz_sub, 7)[1:-1]
    dz = np.concatenate([
        np.full(12, dx), trans, np.full(8, dz_sub), trans[::-1],
        np.full(24, dx),
    ])
    sim = Simulation(freq_max=4e9, domain=(0.08, 0.075, 0), dx=dx,
                     dz_profile=dz, boundary="cpml", cpml_layers=8)
    eps0 = 8.8541878128e-12
    sigma_fr4 = 2 * np.pi * 2.4e9 * eps0 * 4.3 * 0.02
    sim.add_material("fr4", eps_r=4.3, sigma=sigma_fr4)
    z0 = float(np.sum(dz[:12 + len(trans)]))
    sim.add(Box((0.010, 0.010, z0 - dz_sub), (0.070, 0.065, z0)), material="pec")
    sim.add(Box((0.010, 0.010, z0), (0.070, 0.065, z0 + 8 * dz_sub)),
            material="fr4")
    sim.add(Box((0.0253, 0.0185, z0 + 8 * dz_sub),
                (0.0548, 0.0565, z0 + 9 * dz_sub)), material="pec")
    sim.add_source((0.033, 0.0375, z0 + dz_sub * 2.5), "ez",
                   waveform=GaussianPulse(f0=2.4e9, bandwidth=1.2))
    sim.add_probe((0.045, 0.0425, z0 + dz_sub * 2.5), "ez")
    return sim


def main():
    sim = build()
    for n_warm in (0, 500, 2000):
        fr = sim.forward(n_steps=8000, n_warmup=n_warm,
                         emit_time_series=True, checkpoint_every=200)
        ts = np.asarray(fr.time_series).ravel()
        print(f"[n_warmup={n_warm:5d}] shape={ts.shape} "
              f"max|Ez|={np.max(np.abs(ts)):.3e}  finite="
              f"{bool(np.all(np.isfinite(ts)))}")

    ts0 = np.asarray(sim.forward(n_steps=2000).time_series)
    ts_w = np.asarray(sim.forward(n_steps=2000, n_warmup=500).time_series)
    rel = float(
        np.max(np.abs(ts0 - ts_w)) / max(np.max(np.abs(ts0)), 1e-30)
    )
    print(f"[bit-match] n_warmup=500 vs plain rel_err = {rel:.3e}")
    assert rel < 1e-5, (
        f"n_warmup must not change physics, but rel err = {rel}"
    )
    print("PASS")


if __name__ == "__main__":
    main()

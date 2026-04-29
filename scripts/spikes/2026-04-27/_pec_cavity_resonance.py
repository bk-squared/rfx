"""FDTD core sanity check: PEC rectangular cavity resonance frequencies.

Closed PEC cavity, broadband dipole source, harminv on a probe time
series. Compare extracted resonance frequencies to the analytic
formula for a lossless rectangular cavity:

    f_mnp = (c/2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)

Tests both PEC boundary handling (the suspect locus from follow-ups
#1 and #2) and Yee dispersion at well-resolved frequencies. If
relative errors are < 1% across the 5 lowest modes, the FDTD core's
PEC handling is solid and the slab S11 ~7% floor is a port-extractor
limit, not a simulator-core defect. If errors are > 5%, the FDTD core
itself has a real PEC-boundary issue.

Cavity: a=50 mm, b=40 mm, d=30 mm. Mode dimensions chosen so the 5
lowest modes are non-degenerate.

Lowest 5 analytic modes:
  TM110 ≈ 4.80 GHz
  TE101 ≈ 5.83 GHz
  TE011 ≈ 6.25 GHz
  TM111 / TE111 ≈ 6.93 GHz (degenerate)
  TE201 / TM210 ≈ 7.81 GHz
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp


def main():
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box
    from rfx.harminv import harminv
    from rfx.sources.sources import GaussianPulse

    C0 = 2.998e8
    a, b, d = 0.05, 0.04, 0.03
    DX = 0.0015  # 1.5 mm

    sim = Simulation(
        freq_max=12.0e9,
        domain=(a, b, d),
        boundary=BoundarySpec(
            x=Boundary(lo="pec", hi="pec"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        dx=DX,
    )

    # Dipole-like source: broadband Gaussian pulse on E_z at a non-symmetric
    # point inside the cavity.
    src_pos = (a / 3.0, b / 3.0, d / 3.0)
    pulse = GaussianPulse(f0=6.5e9, bandwidth=0.8)
    sim.add_source(
        position=src_pos,
        component="ez",
        waveform=pulse,
    )

    # Time-domain probe far from the source — also at a non-nodal point
    probe_pos = (2 * a / 5.0, 2 * b / 5.0, 2 * d / 5.0)
    sim.add_probe(position=probe_pos, component="ez")

    # Run long enough for good harminv resolution. Need T such that
    # 1/T < freq spacing. Lowest-mode separations ~0.5 GHz → T > 2 ns.
    # 8 ns gives 0.125 GHz resolution.
    n_steps = 30_000
    print(f"Running {n_steps} steps; estimated T = {n_steps * 0.5 * DX/(C0*np.sqrt(3))*1e9:.2f} ns")
    result = sim.run(n_steps=n_steps, compute_s_params=False)

    ts = np.asarray(result.time_series)
    print(f"time_series shape: {ts.shape}")
    signal = ts[:, 0]
    dt_sim = float(result.dt) if hasattr(result, 'dt') else float(getattr(result, 'time_step', 0))
    if dt_sim == 0:
        # Estimate from CFL: dt = dx / (c · sqrt(3))
        dt_sim = DX / (C0 * np.sqrt(3.0))
    times = np.arange(len(signal)) * dt_sim
    dt = dt_sim
    print(f"signal len: {len(signal)}, dt={dt*1e12:.3f} ps, T_total={times[-1]*1e9:.2f} ns")

    # harminv: extract resonances in 4-9 GHz band
    f_min, f_max = 4.0e9, 9.0e9
    # Skip the source pulse onset (first ns) so we see free decay
    skip_steps = int(1.0e-9 / dt)
    decay_signal = signal[skip_steps:]
    print(f"harminv on {len(decay_signal)} samples after {skip_steps} skip")
    modes = harminv(decay_signal, dt, f_min, f_max)
    print(f"harminv found {len(modes)} modes in [{f_min/1e9:.1f}, {f_max/1e9:.1f}] GHz")

    # Analytic reference table: enumerate (m,n,p) ≤ 3 each
    analytic_modes = []
    for m in range(0, 4):
        for n in range(0, 4):
            for p in range(0, 4):
                # TE: requires (m,n) not both zero AND p≥1; or some other constraints
                # TM: requires m≥1 AND n≥1 (can have p=0)
                # Compute frequency only if at least 2 of {m,n,p} ≥ 1
                count_nonzero = (m > 0) + (n > 0) + (p > 0)
                if count_nonzero < 2:
                    continue
                f = 0.5 * C0 * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
                if f_min <= f <= f_max:
                    # Disambiguate TE/TM — for the test we just match by frequency
                    analytic_modes.append((f, (m, n, p)))
    analytic_modes.sort()
    print(f"\nAnalytic modes in [{f_min/1e9:.1f}, {f_max/1e9:.1f}] GHz:")
    for f_an, (m, n, p) in analytic_modes:
        print(f"  ({m},{n},{p}) -> {f_an/1e9:.4f} GHz")

    # Match each extracted mode to the closest analytic mode
    print(f"\nExtracted vs analytic:")
    print(f"{'extracted_GHz':<15} {'closest_analytic_GHz':<22} {'(m,n,p)':<10} {'rel_err_%':<12} {'Q':<10}")
    print("-" * 80)
    for mode in modes:
        f_ext = float(mode.freq)
        if not np.isfinite(f_ext) or f_ext < f_min or f_ext > f_max:
            continue
        deltas = [(abs(f_ext - f_an), f_an, mnp) for f_an, mnp in analytic_modes]
        deltas.sort()
        if not deltas:
            continue
        _, f_an, mnp = deltas[0]
        rel_err = abs(f_ext - f_an) / f_an * 100.0
        q = float(mode.Q) if hasattr(mode, "Q") else float("nan")
        print(f"{f_ext/1e9:<15.4f} {f_an/1e9:<22.4f} {str(mnp):<10} {rel_err:<+12.4f} {q:<10.1f}")


if __name__ == "__main__":
    main()

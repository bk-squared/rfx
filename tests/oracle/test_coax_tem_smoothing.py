"""Shorted homogeneous coax: TEM frequencies are independent of pin staircase."""

import numpy as np

from rfx import Box, Cylinder, GaussianPulse, Simulation
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.grid import C0


def test_coax_tem_smoothing_accuracy():
    width, length, dx = 8e-3, 24e-3, .25e-3  # a/dx = 4
    sim = Simulation(freq_max=10e9, domain=(width, width, length),
                     boundary="pec", dx=dx)
    sim.add_material("fill", eps_r=4.)
    sim.add(Box((0., 0., 0.), (width, width, length)), material="fill")
    sim.add(Cylinder((3.7e-3, 4.2e-3, length / 2), radius=1e-3,
                     height=length, axis="z"), material="pec")
    sim.add_source((5.6e-3, 4.9e-3, .30 * length), "ex",
                   waveform=GaussianPulse(f0=4.7e9, bandwidth=.8))
    sim.add_probe((2.3e-3, 6.1e-3, .41 * length), "ey")

    grid = sim._build_realized_grid()
    z = coords_from_uniform_grid(grid).z
    np.testing.assert_allclose(z[[0, -1]], [0., length], rtol=0., atol=1e-12 * length)
    realized_length = float(z[-1] - z[0])
    targets = np.arange(1, 3) * C0 / (2 * realized_length * np.sqrt(4.))
    result = sim.run(n_steps=int(np.ceil(6e-9 / grid.dt)), subpixel_smoothing=True)
    modes = result.find_resonances(freq_range=(.6 * targets[0], 1.3 * targets[1]))

    # R5: full extracted spectrum, Q and amplitude, then both analytic matches.
    ordered = sorted(modes, key=lambda mode: mode.freq)
    print(f"\n[coax-TEM] short planes (m): {z[[0, -1]].tolist()}, dt={grid.dt:.16g}")
    print(f"[coax-TEM] analytic (Hz): {targets.tolist()}")
    print(f"[coax-TEM] sim modes (Hz): {[m.freq for m in ordered]}")
    print(f"[coax-TEM] sim Q: {[m.Q for m in ordered]}")
    print(f"[coax-TEM] sim amplitude: {[m.amplitude for m in ordered]}")
    assert modes, "no resonances found in the TEM band"
    errors = []
    for n, target in enumerate(targets, 1):
        by_dist = sorted(modes, key=lambda mode: abs(mode.freq - target))
        mode = by_dist[0]
        d_near = abs(mode.freq - target)
        d_second = abs(by_dist[1].freq - target) if len(by_dist) > 1 else float("inf")
        error = (mode.freq - target) / target
        print(f"[coax-TEM] n={n}: f={mode.freq:.16g}, signed error={error:.8g}, "
              f"nearest delta={d_near:.16g}, 2nd-nearest delta={d_second:.16g}")
        assert d_second >= 5 * d_near, f"TEM{n} mode-ID ambiguous"
        errors.append(error)
    assert np.all(np.abs(errors) < 1e-3), f"TEM signed relative errors: {errors}"

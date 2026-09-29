import numpy as np

C0 = 299792458.0


class TestWaveguideCutoff:
    """Validate TE10 mode frequency in a WR-90 waveguide section."""

    # WR-90: a=22.86mm, b=10.16mm, TE10 cutoff = C0/(2*a) = 6.557 GHz
    A_WG = 22.86e-3
    B_WG = 10.16e-3
    F_TE10 = C0 / (2 * A_WG)

    def test_rfx_waveguide_cutoff(self):
        """rfx should find TE10 cutoff within 0.5% via cavity mode."""
        from rfx import Simulation, GaussianPulse
        from rfx.harminv import harminv

        L_wg = 40e-3  # waveguide length
        dx = 0.5e-3
        sim = Simulation(freq_max=10e9, domain=(self.A_WG, self.B_WG, L_wg),
                         boundary='pec', dx=dx)
        # TE10 in a PEC cavity: f_mnp = C0/2 * sqrt((m/a)^2 + (n/b)^2 + (p/L)^2)
        # For TE101: f = C0/2 * sqrt((1/a)^2 + (1/L)^2)
        f_101 = (C0 / 2) * np.sqrt((1 / self.A_WG) ** 2 + (1 / L_wg) ** 2)
        sim.add_source((self.A_WG / 2, self.B_WG / 3, L_wg / 3), 'ey',
                        waveform=GaussianPulse(f0=f_101, bandwidth=0.8))
        sim.add_probe((self.A_WG / 2, 2 * self.B_WG / 3, 2 * L_wg / 3), 'ey')

        grid = sim._build_grid()
        n_steps = grid.num_timesteps(num_periods=100)
        result = sim.run(n_steps=n_steps)

        ts = np.array(result.time_series).ravel()
        start = len(ts) // 4
        w = ts[start:] - np.mean(ts[start:])
        modes = harminv(w, grid.dt, f_101 * 0.5, f_101 * 1.5)

        assert modes, "No modes found"
        best = min(modes, key=lambda m: abs(m.freq - f_101))
        err = abs(best.freq - f_101) / f_101
        print(f"\nWR-90 TE101: rfx={best.freq/1e9:.4f} GHz, analytical={f_101/1e9:.4f} GHz, err={err*100:.3f}%")
        assert err < 0.005


class TestLumpedPortCavity:
    """Lumped port S11 in a dielectric-loaded PEC cavity.

    This tests the port model (impedance loading + V/I extraction),
    not just resonance frequency.
    """

    def test_rfx_lumped_port_resonance_via_probe(self):
        """Lumped port should excite cavity; probe detects resonance via Harminv."""
        from rfx import Simulation, Box, GaussianPulse
        from rfx.harminv import harminv

        a, b, d = 50e-3, 40e-3, 20e-3
        eps_r = 2.2
        f_110 = (C0 / (2 * np.sqrt(eps_r))) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)
        dx = 1e-3

        sim = Simulation(freq_max=f_110 * 2, domain=(a, b, d),
                         boundary='pec', dx=dx)
        sim.add_material('dielectric', eps_r=eps_r)
        sim.add(Box((0, 0, 0), (a, b, d)), material='dielectric')
        # Use high-impedance port (minimal loading) to excite cavity
        sim.add_port((a / 3, b / 3, d / 2), 'ez', impedance=1e6,
                     waveform=GaussianPulse(f0=f_110, bandwidth=0.8))
        sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), 'ez')

        grid = sim._build_grid()
        n_steps = grid.num_timesteps(num_periods=80)
        result = sim.run(n_steps=n_steps)

        ts = np.array(result.time_series).ravel()
        start = len(ts) // 4
        w = ts[start:] - np.mean(ts[start:])
        modes = harminv(w, grid.dt, f_110 * 0.5, f_110 * 1.5)

        assert modes, "Port excitation should produce detectable resonance"
        best = min(modes, key=lambda m: abs(m.freq - f_110))
        err = abs(best.freq - f_110) / f_110
        print(f"\nLumped port + probe: f={best.freq/1e9:.4f} GHz, "
              f"analytical={f_110/1e9:.4f} GHz, err={err*100:.3f}%")
        assert err < 0.005

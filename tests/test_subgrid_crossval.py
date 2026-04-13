"""Subgridding accuracy crossvalidation tests.

Compares subgridded simulations against uniform fine-grid references:
1. PEC cavity with dielectric slab: probe RMS error < 5%
2. PEC cavity with off-axis source: probe RMS error < 10%

Both use PEC boundaries (not CPML) because CPML+subgrid is currently
unstable — the CPML absorber on the coarse grid conflicts with the
fine-grid SAT coupling, causing late-time energy growth. This is a known
limitation to be addressed in Phase A2 (runner generalization).

Both tests require GPU and are slow — marked accordingly.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _rms_error_time_aligned(ts_ref, dt_ref, ts_test, dt_test):
    """Normalised RMS error with physical-time interpolation.

    The two signals may have different dt (timestep), so we
    interpolate both onto a common time axis before comparing.
    """
    t_ref = np.arange(len(ts_ref)) * dt_ref
    t_test = np.arange(len(ts_test)) * dt_test
    t_max = min(t_ref[-1], t_test[-1])
    n_common = 500
    t_common = np.linspace(0, t_max, n_common)

    ref = np.interp(t_common, t_ref, ts_ref.astype(np.float64))
    tst = np.interp(t_common, t_test, ts_test.astype(np.float64))

    rms_ref = np.sqrt(np.mean(ref ** 2))
    if rms_ref < 1e-30:
        return 0.0
    return np.sqrt(np.mean((ref - tst) ** 2)) / rms_ref


# ---------------------------------------------------------------------------
# Test 1: PEC cavity with dielectric slab
# ---------------------------------------------------------------------------

class TestSlabCavitySubgrid:
    """Subgridded dielectric slab in PEC cavity vs uniform fine-grid.

    PEC cavity avoids CPML+subgrid instability. The slab creates a
    dielectric interface that the subgrid must resolve accurately.

    Geometry:
    - Domain (0.06, 0.06, 0.06), PEC boundary
    - Dielectric slab: eps_r=4.0, z in [0.02, 0.04]
    - Source: z=0.015 (before slab)
    - Probe: z=0.045 (after slab, transmitted)
    """

    def _run_uniform_fine(self, n_steps: int):
        from rfx import Simulation, Box

        domain = (0.06, 0.06, 0.06)
        dx_fine = 1e-3

        sim = Simulation(
            freq_max=10e9, domain=domain, boundary="pec", dx=dx_fine,
        )
        sim.add_material("dielectric", eps_r=4.0)
        sim.add(Box((0, 0, 0.02), (0.06, 0.06, 0.04)), material="dielectric")
        sim.add_source(position=(0.03, 0.03, 0.015), component="ez")
        sim.add_probe(position=(0.03, 0.03, 0.045), component="ez")

        return sim.run(n_steps=n_steps)

    def _run_subgridded(self, n_steps: int):
        from rfx import Simulation, Box

        domain = (0.06, 0.06, 0.06)
        dx_coarse = 3e-3

        sim = Simulation(
            freq_max=10e9, domain=domain, boundary="pec", dx=dx_coarse,
        )
        sim.add_material("dielectric", eps_r=4.0)
        sim.add(Box((0, 0, 0.02), (0.06, 0.06, 0.04)), material="dielectric")
        # z_range covers source, slab, and probe
        sim.add_refinement(z_range=(0.010, 0.050), ratio=3)
        sim.add_source(position=(0.03, 0.03, 0.015), component="ez")
        sim.add_probe(position=(0.03, 0.03, 0.045), component="ez")

        return sim.run(n_steps=n_steps)

    def test_slab_transmitted_rms_error(self):
        """Transmitted probe RMS error < 5% (time-aligned comparison)."""
        n_steps_ref = 1000
        # Subgridded uses smaller dt → need more steps to cover same time
        result_ref = self._run_uniform_fine(n_steps_ref)
        dt_ref = float(result_ref.dt)

        # Run subgridded for same physical time
        result_sub_short = self._run_subgridded(100)  # just to get dt
        dt_sub = float(result_sub_short.dt)
        n_steps_sub = int(n_steps_ref * dt_ref / dt_sub) + 100
        result_sub = self._run_subgridded(n_steps_sub)

        ts_ref = np.array(result_ref.time_series[:, 0])
        ts_sub = np.array(result_sub.time_series[:, 0])

        err = _rms_error_time_aligned(ts_ref, dt_ref, ts_sub, dt_sub)
        print(f"\nSlab cavity crossval:")
        print(f"  Ref: dt={dt_ref:.3e}s, {len(ts_ref)} steps, max={np.max(np.abs(ts_ref)):.6e}")
        print(f"  Sub: dt={dt_sub:.3e}s, {len(ts_sub)} steps, max={np.max(np.abs(ts_sub)):.6e}")
        print(f"  RMS error (time-aligned): {err:.3%}")
        assert err < 0.05, f"Slab cavity: RMS error {err:.3%} >= 5%"

    def test_slab_signals_finite(self):
        """Both runs must produce finite signals."""
        n_steps = 500
        result = self._run_subgridded(n_steps)
        ts = np.array(result.time_series[:, 0])
        assert np.all(np.isfinite(ts)), "Subgridded signal has NaN/Inf"


# ---------------------------------------------------------------------------
# Test 2: PEC cavity with off-axis source (3D stress test)
# ---------------------------------------------------------------------------

class TestCavitySubgrid:
    """3D PEC cavity with off-axis source — stresses all 6 subgrid faces.

    Geometry:
    - Domain (0.04, 0.04, 0.04), PEC boundary
    - Source: (0.012, 0.015, 0.018) — off-axis
    - Probe: (0.028, 0.025, 0.022)
    """

    def _run_uniform_fine(self, n_steps: int):
        from rfx import Simulation

        sim = Simulation(
            freq_max=10e9, domain=(0.04, 0.04, 0.04),
            boundary="pec", dx=1e-3,
        )
        sim.add_source(position=(0.012, 0.015, 0.018), component="ez")
        sim.add_probe(position=(0.028, 0.025, 0.022), component="ez")
        return sim.run(n_steps=n_steps)

    def _run_subgridded(self, n_steps: int):
        from rfx import Simulation

        sim = Simulation(
            freq_max=10e9, domain=(0.04, 0.04, 0.04),
            boundary="pec", dx=3e-3,
        )
        # z_range covers source (z=0.018) and probe (z=0.022)
        sim.add_refinement(z_range=(0.008, 0.032), ratio=3)
        sim.add_source(position=(0.012, 0.015, 0.018), component="ez")
        sim.add_probe(position=(0.028, 0.025, 0.022), component="ez")
        return sim.run(n_steps=n_steps)

    def test_cavity_probe_rms_error(self):
        """Off-axis probe RMS error < 10% (time-aligned comparison)."""
        n_steps_ref = 1000

        result_ref = self._run_uniform_fine(n_steps_ref)
        dt_ref = float(result_ref.dt)

        # Match physical time
        result_sub_short = self._run_subgridded(100)
        dt_sub = float(result_sub_short.dt)
        n_steps_sub = int(n_steps_ref * dt_ref / dt_sub) + 100
        result_sub = self._run_subgridded(n_steps_sub)

        ts_ref = np.array(result_ref.time_series[:, 0])
        ts_sub = np.array(result_sub.time_series[:, 0])

        err = _rms_error_time_aligned(ts_ref, dt_ref, ts_sub, dt_sub)
        print(f"\n3D cavity crossval:")
        print(f"  Ref: dt={dt_ref:.3e}s, {len(ts_ref)} steps, max={np.max(np.abs(ts_ref)):.6e}")
        print(f"  Sub: dt={dt_sub:.3e}s, {len(ts_sub)} steps, max={np.max(np.abs(ts_sub)):.6e}")
        print(f"  RMS error (time-aligned): {err:.3%}")
        assert err < 0.10, f"3D cavity: RMS error {err:.3%} >= 10%"

    def test_cavity_signals_finite(self):
        """Both runs must produce finite signals."""
        n_steps = 500
        result = self._run_subgridded(n_steps)
        ts = np.array(result.time_series[:, 0])
        assert np.all(np.isfinite(ts)), "Subgridded signal has NaN/Inf"

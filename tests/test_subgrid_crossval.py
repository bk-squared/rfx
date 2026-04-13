"""Subgridding accuracy crossvalidation tests.

Compares subgridded simulations against uniform fine-grid references:
1. Fresnel slab: transmitted probe RMS error < 5%
2. 3D PEC cavity: off-axis source/probe RMS error < 10%

Both tests require GPU and are slow — marked accordingly.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rms_error(sig_ref, sig_test):
    """Normalised RMS error between two 1-D arrays of the same length.

    Error is relative to the RMS of the reference signal.
    Both arrays are trimmed/padded to the shorter length before comparison.
    """
    n = min(len(sig_ref), len(sig_test))
    ref = sig_ref[:n]
    tst = sig_test[:n]
    rms_ref = np.sqrt(np.mean(ref ** 2))
    if rms_ref < 1e-30:
        return 0.0
    rms_err = np.sqrt(np.mean((ref - tst) ** 2))
    return rms_err / rms_ref


# ---------------------------------------------------------------------------
# Task 5: Fresnel slab accuracy crossval
# ---------------------------------------------------------------------------

class TestFresnelSlabSubgrid:
    """Subgridded Fresnel slab vs uniform fine-grid reference.

    Geometry:
    - Domain (0.04, 0.04, 0.08), CPML, 8 layers
    - Dielectric slab: eps_r=4.0, z in [0.03, 0.05]
    - Source: z=0.015, probe (transmitted): z=0.065

    Uniform fine grid:  dx = 2e-3/3  (equivalent to coarse/ratio)
    Subgridded sim:     dx_coarse = 2e-3, ratio=3, z_range=(0.025, 0.055)
    """

    def _run_uniform_fine(self, n_steps: int):
        """Uniform fine-grid simulation (reference)."""
        from rfx import Simulation, Box

        domain = (0.04, 0.04, 0.08)
        dx_fine = 2e-3 / 3

        sim = Simulation(
            freq_max=5e9,
            domain=domain,
            boundary="cpml",
            cpml_layers=8,
            dx=dx_fine,
        )
        sim.add_material("dielectric", eps_r=4.0)
        sim.add(
            Box((0.0, 0.0, 0.03), (0.04, 0.04, 0.05)),
            material="dielectric",
        )
        # Transmitted probe
        sim.add_source(position=(0.02, 0.02, 0.015), component="ez")
        sim.add_probe(position=(0.02, 0.02, 0.065), component="ez")
        # Reflected probe
        sim.add_probe(position=(0.02, 0.02, 0.025), component="ez")

        result = sim.run(n_steps=n_steps)
        return result

    def _run_subgridded(self, n_steps: int):
        """Coarse grid + subgridded slab region simulation."""
        from rfx import Simulation, Box

        domain = (0.04, 0.04, 0.08)
        dx_coarse = 2e-3

        sim = Simulation(
            freq_max=5e9,
            domain=domain,
            boundary="cpml",
            cpml_layers=8,
            dx=dx_coarse,
        )
        sim.add_material("dielectric", eps_r=4.0)
        sim.add(
            Box((0.0, 0.0, 0.03), (0.04, 0.04, 0.05)),
            material="dielectric",
        )
        # z_range must cover source (z=0.015) and probes (z=0.025, z=0.065)
        # since the runner only injects sources/reads probes on the fine grid
        sim.add_refinement(z_range=(0.010, 0.070), ratio=3)
        # Transmitted probe
        sim.add_source(position=(0.02, 0.02, 0.015), component="ez")
        sim.add_probe(position=(0.02, 0.02, 0.065), component="ez")
        # Reflected probe
        sim.add_probe(position=(0.02, 0.02, 0.025), component="ez")

        result = sim.run(n_steps=n_steps)
        return result

    def test_fresnel_transmitted_rms_error(self):
        """Transmitted probe RMS error between subgridded and fine reference < 5%."""
        n_steps = 2000

        result_ref = self._run_uniform_fine(n_steps)
        result_sub = self._run_subgridded(n_steps)

        # Probe 0 = transmitted (z=0.065)
        ts_ref = np.array(result_ref.time_series[:, 0])
        ts_sub = np.array(result_sub.time_series[:, 0])

        err = _rms_error(ts_ref, ts_sub)
        assert err < 0.05, (
            f"Fresnel slab: transmitted probe RMS error {err:.3%} >= 5%"
        )

    def test_fresnel_fields_finite(self):
        """Both uniform and subgridded sims must produce finite fields."""
        n_steps = 200

        result_ref = self._run_uniform_fine(n_steps)
        result_sub = self._run_subgridded(n_steps)

        for label, result in [("uniform_fine", result_ref), ("subgridded", result_sub)]:
            ts = np.array(result.time_series)
            assert not np.any(np.isnan(ts)), f"{label}: NaN in time series"
            assert np.all(np.isfinite(ts)), f"{label}: Inf in time series"

    def test_fresnel_subgrid_produces_nonzero_transmission(self):
        """Subgridded sim must show non-trivial transmitted signal."""
        n_steps = 2000

        result_sub = self._run_subgridded(n_steps)
        ts_trans = np.array(result_sub.time_series[:, 0])

        assert np.max(np.abs(ts_trans)) > 1e-10, (
            "Subgridded slab: transmitted probe near-zero"
        )


# ---------------------------------------------------------------------------
# Task 6: 3D cavity stress test
# ---------------------------------------------------------------------------

class TestCavitySubgrid:
    """3D PEC cavity with off-axis source — subgridded vs uniform fine-grid.

    Geometry:
    - Domain (0.04, 0.04, 0.04), PEC boundary
    - Source: (0.012, 0.015, 0.018) — off-axis
    - Probe: (0.028, 0.025, 0.022)
    - Subgrid z_range: (0.012, 0.028)
    """

    def _run_uniform_fine(self, n_steps: int):
        """Uniform fine-grid cavity simulation (reference)."""
        from rfx import Simulation

        domain = (0.04, 0.04, 0.04)
        dx_fine = 2e-3 / 3

        sim = Simulation(
            freq_max=5e9,
            domain=domain,
            boundary="pec",
            dx=dx_fine,
        )
        sim.add_source(position=(0.012, 0.015, 0.018), component="ez")
        sim.add_probe(position=(0.028, 0.025, 0.022), component="ez")

        result = sim.run(n_steps=n_steps)
        return result

    def _run_subgridded(self, n_steps: int):
        """Coarse grid + subgridded cavity simulation."""
        from rfx import Simulation

        domain = (0.04, 0.04, 0.04)
        dx_coarse = 2e-3

        sim = Simulation(
            freq_max=5e9,
            domain=domain,
            boundary="pec",
            dx=dx_coarse,
        )
        # z_range must cover source (z=0.018) and probe (z=0.022) with margin
        sim.add_refinement(z_range=(0.008, 0.032), ratio=3)
        sim.add_source(position=(0.012, 0.015, 0.018), component="ez")
        sim.add_probe(position=(0.028, 0.025, 0.022), component="ez")

        result = sim.run(n_steps=n_steps)
        return result

    def test_cavity_probe_rms_error(self):
        """Off-axis probe RMS error between subgridded and fine reference < 10%."""
        n_steps = 1500

        result_ref = self._run_uniform_fine(n_steps)
        result_sub = self._run_subgridded(n_steps)

        ts_ref = np.array(result_ref.time_series[:, 0])
        ts_sub = np.array(result_sub.time_series[:, 0])

        err = _rms_error(ts_ref, ts_sub)
        assert err < 0.10, (
            f"3D cavity: probe RMS error {err:.3%} >= 10%"
        )

    def test_cavity_fields_finite(self):
        """Both uniform and subgridded cavity sims must produce finite fields."""
        n_steps = 200

        result_ref = self._run_uniform_fine(n_steps)
        result_sub = self._run_subgridded(n_steps)

        for label, result in [("uniform_fine", result_ref), ("subgridded", result_sub)]:
            ts = np.array(result.time_series)
            assert not np.any(np.isnan(ts)), f"{label}: NaN in time series"
            assert np.all(np.isfinite(ts)), f"{label}: Inf in time series"

    def test_cavity_subgrid_nonzero_signal(self):
        """Off-axis probe must record a non-trivial signal in the subgridded cavity."""
        n_steps = 1500

        result_sub = self._run_subgridded(n_steps)
        ts = np.array(result_sub.time_series[:, 0])

        assert np.max(np.abs(ts)) > 1e-10, (
            "3D cavity subgrid: probe near-zero — source injection may have failed"
        )

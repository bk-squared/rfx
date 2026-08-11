"""Tests for jax.vmap batched material-parameter sweep."""

import numpy as np
import numpy.testing as npt
import pytest

from rfx import Simulation, GaussianPulse, Box
from rfx.vmap_sweep import vmap_material_sweep, VmapSweepResult

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dielectric_sim(eps_r: float = 4.0):
    """Create a simple PEC-bounded sim with a dielectric slab and probe."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        boundary="pec",
        dx=0.002,
    )
    sim.add_material("substrate", eps_r=eps_r)
    sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
    sim.add_source((0.01, 0.01, 0.01), "ez",
                    waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.005, 0.01, 0.01), "ez")
    return sim


def _make_cpml_sim(eps_r: float = 4.0, thin_conductor: bool = False):
    """Create a CPML-bounded sim with a dielectric slab and probe.

    ``thin_conductor`` adds a NON-PEC sheet spanning the full x extent
    (issue #642). The full x span is load-bearing, not decoration: #642 is
    a pad-EXTENSION leak, so the conductor has to sit on the very column
    the x pads replicate from. An interior-only conductor
    (x 0.005..0.015, the first draft of this fixture) measured a
    diff/budget ratio of 7.5e-3 at n_steps=200 -- blind by GEOMETRY, a
    third blindness shape on top of the two issue #642 names.
    """
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        boundary="cpml",
        cpml_layers=6,
        dx=0.002,
    )
    sim.add_material("substrate", eps_r=eps_r)
    sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
    if thin_conductor:
        sim.add_thin_conductor(
            Box((0.0, 0.008, 0.008), (0.02, 0.012, 0.012)),
            sigma_bulk=1.0e4, thickness=35e-6)
    sim.add_source((0.01, 0.01, 0.01), "ez",
                    waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.005, 0.01, 0.01), "ez")
    return sim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVmapSweepBasic:
    """Basic functionality tests."""

    def test_vmap_sweep_produces_different_results(self):
        """Different eps_r values must produce different probe time series."""
        sim = _make_dielectric_sim()
        eps_values = np.array([1.0, 2.0, 4.0, 8.0])
        n_steps = 50

        result = vmap_material_sweep(
            sim, "substrate.eps_r", eps_values, n_steps=n_steps,
        )

        assert isinstance(result, VmapSweepResult)
        assert result.time_series.shape == (4, n_steps, 1)
        assert len(result.param_values) == 4
        npt.assert_array_equal(result.param_values, eps_values)

        # Each eps_r should produce a distinct time series
        for i in range(len(eps_values)):
            for j in range(i + 1, len(eps_values)):
                diff = np.max(np.abs(
                    result.time_series[i] - result.time_series[j]
                ))
                assert diff > 0, (
                    f"eps_r={eps_values[i]} and eps_r={eps_values[j]} "
                    f"produced identical time series"
                )

    def test_vmap_matches_sequential(self):
        """Vmap results must match sequential simulation within tolerance."""
        sim = _make_dielectric_sim()
        eps_values = np.array([2.0, 6.0])
        n_steps = 40

        # Vmap path
        vmap_result = vmap_material_sweep(
            sim, "substrate.eps_r", eps_values, n_steps=n_steps,
        )

        # Sequential path: run each simulation individually
        for idx, eps_val in enumerate(eps_values):
            sim_single = _make_dielectric_sim(eps_r=float(eps_val))
            single_result = sim_single.run(n_steps=n_steps)
            single_ts = np.asarray(single_result.time_series)

            npt.assert_allclose(
                vmap_result.time_series[idx],
                single_ts,
                atol=1e-5,
                rtol=1e-4,
                err_msg=f"Mismatch at eps_r={eps_val}",
            )

    def test_vmap_batch_size(self):
        """Handle 10+ batch values without error."""
        sim = _make_dielectric_sim()
        eps_values = np.linspace(1.5, 10.0, 12)
        n_steps = 30

        result = vmap_material_sweep(
            sim, "substrate.eps_r", eps_values, n_steps=n_steps,
        )

        assert result.time_series.shape == (12, n_steps, 1)
        # Verify monotonic change: higher eps_r -> slower wave -> different peak
        peaks = result.peak_field()
        assert peaks.shape == (12,)
        # All peaks should be positive (source is active)
        assert np.all(peaks > 0)


class TestVmapSweepParams:
    """Test different parameter types and naming conventions."""

    def test_global_eps_r_sweep(self):
        """Global eps_r sweep (no material prefix)."""
        sim = _make_dielectric_sim()
        result = vmap_material_sweep(
            sim, "eps_r", np.array([2.0, 4.0]), n_steps=30,
        )
        assert result.time_series.shape[0] == 2
        assert result.param_name == "eps_r"

    def test_sigma_sweep(self):
        """Sweep conductivity."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            dx=0.002,
        )
        sim.add_material("lossy", eps_r=4.0, sigma=0.1)
        sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="lossy")
        sim.add_source((0.01, 0.01, 0.01), "ez",
                        waveform=GaussianPulse(f0=3e9))
        sim.add_probe((0.005, 0.01, 0.01), "ez")

        sigma_values = np.array([0.01, 0.1, 1.0])
        result = vmap_material_sweep(
            sim, "lossy.sigma", sigma_values, n_steps=40,
        )

        assert result.time_series.shape == (3, 40, 1)
        # Higher sigma -> more loss -> lower peak field
        peaks = result.peak_field()
        # The trend should be decreasing (more loss = smaller peak)
        # Allow some tolerance for the very first steps
        assert peaks[0] > peaks[2] or np.isclose(peaks[0], peaks[2], rtol=0.5)

    def test_invalid_param_name(self):
        """Invalid parameter name should raise ValueError."""
        sim = _make_dielectric_sim()
        with pytest.raises(ValueError, match="param_name field must be one of"):
            vmap_material_sweep(sim, "invalid_param", np.array([1.0]), n_steps=10)

    def test_empty_param_values(self):
        """Empty param_values should raise ValueError."""
        sim = _make_dielectric_sim()
        with pytest.raises(ValueError, match="must not be empty"):
            vmap_material_sweep(sim, "eps_r", np.array([]), n_steps=10)


class TestVmapSweepCPML:
    """Test vmap sweep with CPML boundaries."""

    def test_cpml_vmap_produces_results(self):
        """CPML-bounded simulation should work with vmap."""
        sim = _make_cpml_sim()
        eps_values = np.array([2.0, 4.0])
        n_steps = 40

        result = vmap_material_sweep(
            sim, "substrate.eps_r", eps_values, n_steps=n_steps,
        )

        assert result.time_series.shape == (2, n_steps, 1)
        # Results should differ
        diff = np.max(np.abs(
            result.time_series[0] - result.time_series[1]
        ))
        assert diff > 0

    @pytest.mark.parametrize("thin_conductor", [False, True],
                             ids=["plain", "thin_conductor"])
    def test_cpml_vmap_matches_sequential(self, thin_conductor):
        """CPML vmap results must match sequential runs.

        ``n_steps`` and the tolerance are BOTH measured, not chosen
        (issue #642(b)). This test ran at ``n_steps=30``,
        ``atol=1e-5, rtol=1e-4`` and stayed green through the entire
        pre-#637 era for a reason unrelated to the physics it names: the
        pulse had not meaningfully reached the absorber yet, so a pad
        defect could not show. Sensitivity was re-established by
        INJECTING each defect and sweeping ``n_steps``, reporting
        ``max|diff| / (atol + rtol*|reference|)`` -- i.e. >1 means this
        assertion fails. Clean measures exactly ``0.0`` at every row, so
        neither knob costs anything:

        ``_extend_batched_cpml_pad`` disabled (the pre-#637 defect,
        ``plain`` row), under the OLD gate:

            n_steps :   30      45      60      90     120     200
            ratio   : 0.098   0.337   1.010   7.218   26.77   60.77

        60 is where it first crosses -- by 1%, which cross-machine float
        jitter can erase, so 60 is not a safe pick even though it
        nominally "sees it".

        Pre-#642 pad order restored (``thin_conductor`` row), under the
        NEW gate: 0.937 at 120 AND at 150, 5.10e2 at 200, 1.07e9 at 300.
        That is not drift, it is divergence onset -- the R5 decile dump
        at 400 steps shows the injected path's probe envelope growing
        monotonically (last/mid decile 3.9e13) while the sequential
        reference decays (9.1e-3). Leaking a 175 S/m sheet into the CPML
        pad destabilises the run; #642 was never only an accuracy leak.

        Hence ``n_steps=200``: the only value at which BOTH rows are
        sensitive (1.47e4x and 5.10e2x over the gate). And hence the
        tighter gate, which is an independent axis rather than a
        cosmetic tightening -- it catches the ``plain`` defect at 97x
        even at the old ``n_steps=30``, where the old gate saw 0.098x.
        Both are moved because run length is what lets the physics reach
        the absorber, while the tolerance is what a later flake-chase is
        most likely to loosen; either one alone leaves the other blind.
        """
        sim = _make_cpml_sim(thin_conductor=thin_conductor)
        eps_values = np.array([2.0, 6.0])
        n_steps = 200

        vmap_result = vmap_material_sweep(
            sim, "substrate.eps_r", eps_values, n_steps=n_steps,
        )

        for idx, eps_val in enumerate(eps_values):
            sim_single = _make_cpml_sim(eps_r=float(eps_val),
                                        thin_conductor=thin_conductor)
            single_result = sim_single.run(n_steps=n_steps)
            single_ts = np.asarray(single_result.time_series)

            npt.assert_allclose(
                vmap_result.time_series[idx],
                single_ts,
                atol=1e-8,
                rtol=1e-6,
                err_msg=f"CPML mismatch at eps_r={eps_val}",
            )


class TestVmapSweepAutoSteps:
    """Test auto n_steps computation."""

    def test_auto_n_steps(self):
        """When n_steps is None, should auto-compute from num_periods."""
        sim = _make_dielectric_sim()
        result = vmap_material_sweep(
            sim, "substrate.eps_r", np.array([2.0, 4.0]),
            num_periods=5.0,
        )
        assert result.time_series.shape[0] == 2
        assert result.time_series.shape[1] > 0

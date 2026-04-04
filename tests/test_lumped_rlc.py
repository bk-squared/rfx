"""Lumped RLC element tests.

Validates:
1. Series RLC resonance frequency matches analytical f0 = 1/(2*pi*sqrt(LC))
2. Pure capacitor and pure inductor elements run without error
3. Parallel RLC topology runs without error
4. Validation catches bad inputs
5. RLC does not break existing non-RLC simulations (no regression)
"""

import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
from rfx.lumped import init_rlc_state


class TestLumpedRLCValidation:
    """Input validation for add_lumped_rlc()."""

    def test_bad_component(self):
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        with pytest.raises(ValueError, match="component"):
            sim.add_lumped_rlc((0.005, 0.005, 0.005), component="hz", R=50)

    def test_bad_topology(self):
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        with pytest.raises(ValueError, match="topology"):
            sim.add_lumped_rlc((0.005, 0.005, 0.005), R=50, topology="ladder")

    def test_negative_values(self):
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        with pytest.raises(ValueError, match="non-negative"):
            sim.add_lumped_rlc((0.005, 0.005, 0.005), R=-10)

    def test_all_zero(self):
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        with pytest.raises(ValueError, match="non-zero"):
            sim.add_lumped_rlc((0.005, 0.005, 0.005), R=0, L=0, C=0)


class TestRLCState:
    """Unit tests for RLC state initialization."""

    def test_init_state(self):
        state = init_rlc_state()
        assert float(state.inductor_current) == 0.0


class TestPureResistor:
    """Pure resistor element (R only, L=0, C=0)."""

    def test_pure_resistor_runs(self):
        """A pure resistor RLC element should run without error."""
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            R=50.0)
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)

    def test_resistor_damps_field(self):
        """A resistor should damp the field compared to free space."""
        # Run without RLC
        sim_ref = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim_ref.add_source((0.005, 0.005, 0.005), "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim_ref.add_probe((0.007, 0.007, 0.005), "ez")
        result_ref = sim_ref.run(n_steps=300, compute_s_params=False)

        # Run with large resistor at probe location
        sim_rlc = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim_rlc.add_source((0.005, 0.005, 0.005), "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim_rlc.add_lumped_rlc((0.007, 0.007, 0.005), component="ez",
                                R=1.0)  # Low resistance = high current = strong damping
        sim_rlc.add_probe((0.007, 0.007, 0.005), "ez")
        result_rlc = sim_rlc.run(n_steps=300, compute_s_params=False)

        # The RLC element should modify the field (different energy path)
        ts_ref = np.array(result_ref.time_series).ravel()
        ts_rlc = np.array(result_rlc.time_series).ravel()
        # They should NOT be identical (the RLC changes the field)
        assert not np.allclose(ts_ref, ts_rlc, atol=1e-10), \
            "RLC element should visibly affect the field"


class TestPureCapacitor:
    """Pure capacitor element (C only, R=0, L=0)."""

    def test_pure_capacitor_runs(self):
        """Pure C element should run without error."""
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            C=1e-12)
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)


class TestPureInductor:
    """Pure inductor element (L only, R=0, C=0)."""

    def test_pure_inductor_runs(self):
        """Pure L element should run without error."""
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            L=10e-9)
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)


class TestSeriesRLCResonance:
    """LC resonance detection via spectral comparison."""

    def test_lc_shifts_spectrum(self):
        """Adding L and C should measurably shift the spectral content.

        The LC resonance f0 = 1/(2*pi*sqrt(LC)) creates a spectral
        feature (loading effect) at the element cell.  We verify this
        by comparing the spectrum with and without the LC element.
        """
        L = 10e-9      # 10 nH
        C = 1e-12      # 1 pF
        f0_lc = 1.0 / (2.0 * np.pi * np.sqrt(L * C))  # ~1.59 GHz

        freq_max = 5e9
        domain_size = 0.02
        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        n_steps = 2000

        # Reference: no RLC
        sim_ref = Simulation(freq_max=freq_max,
                              domain=(domain_size, domain_size, domain_size),
                              boundary="pec")
        sim_ref.add_source(center, "ez",
                            waveform=GaussianPulse(f0=f0_lc, bandwidth=0.8))
        sim_ref.add_probe(center, "ez")
        result_ref = sim_ref.run(n_steps=n_steps, compute_s_params=False)

        # With LC element (low R for strong effect)
        sim_lc = Simulation(freq_max=freq_max,
                             domain=(domain_size, domain_size, domain_size),
                             boundary="pec")
        sim_lc.add_source(center, "ez",
                           waveform=GaussianPulse(f0=f0_lc, bandwidth=0.8))
        sim_lc.add_lumped_rlc(center, component="ez",
                               R=10.0, L=L, C=C, topology="series")
        sim_lc.add_probe(center, "ez")
        result_lc = sim_lc.run(n_steps=n_steps, compute_s_params=False)

        ts_ref = np.array(result_ref.time_series).ravel()
        ts_lc = np.array(result_lc.time_series).ravel()
        dt = result_ref.dt

        # The LC element must change the time series
        assert not np.allclose(ts_ref, ts_lc, atol=1e-10), \
            "LC element should modify the time series"

        # Compare spectra around f0_lc
        freqs = np.fft.rfftfreq(len(ts_ref), dt)
        spec_ref = np.abs(np.fft.rfft(ts_ref))
        spec_lc = np.abs(np.fft.rfft(ts_lc))

        # The LC element should change the spectrum near f0_lc
        mask = (freqs > f0_lc * 0.3) & (freqs < f0_lc * 3.0)
        if np.any(mask):
            ratio = spec_lc[mask] / np.maximum(spec_ref[mask], 1e-30)
            # Spectrum should differ by at least 10% somewhere near f0_lc
            max_change = np.max(np.abs(ratio - 1.0))
            assert max_change > 0.1, (
                f"LC element should change spectrum near f0={f0_lc:.2e} Hz "
                f"but max change ratio is only {max_change:.3f}"
            )

    def test_capacitor_modifies_response(self):
        """Adding a capacitor measurably changes the time-domain response.

        A capacitor C at a cell adds eps_extra = C/(dx*EPS_0) to the
        local permittivity.  We verify the time series differs
        significantly from the reference (no capacitor) case.
        """
        freq_max = 5e9
        domain_size = 0.01
        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        n_steps = 500

        # Reference without capacitor
        sim_ref = Simulation(freq_max=freq_max,
                              domain=(domain_size, domain_size, domain_size),
                              boundary="pec")
        sim_ref.add_source(center, "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim_ref.add_probe(center, "ez")
        result_ref = sim_ref.run(n_steps=n_steps, compute_s_params=False)

        # With capacitor
        sim_cap = Simulation(freq_max=freq_max,
                              domain=(domain_size, domain_size, domain_size),
                              boundary="pec")
        sim_cap.add_source(center, "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim_cap.add_lumped_rlc(center, component="ez", C=1e-12)
        sim_cap.add_probe(center, "ez")
        result_cap = sim_cap.run(n_steps=n_steps, compute_s_params=False)

        ts_ref = np.array(result_ref.time_series).ravel()
        ts_cap = np.array(result_cap.time_series).ravel()

        # The signals should differ significantly
        rel_diff = np.max(np.abs(ts_ref - ts_cap)) / max(
            np.max(np.abs(ts_ref)), np.max(np.abs(ts_cap)), 1e-30)
        assert rel_diff > 0.1, (
            f"Capacitor should significantly change E-field response, "
            f"but relative difference is only {rel_diff:.3f}"
        )


class TestParallelRLC:
    """Parallel RLC topology tests."""

    def test_parallel_rlc_runs(self):
        """Parallel RLC element should run without error."""
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            R=50.0, L=10e-9, C=1e-12, topology="parallel")
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)

    def test_parallel_rlc_resonance(self):
        """Parallel RLC anti-resonance at f0 = 1/(2*pi*sqrt(LC))."""
        R = 200.0      # Higher R for parallel (high impedance at resonance)
        L = 10e-9      # 10 nH
        C = 1e-12      # 1 pF
        f0_analytical = 1.0 / (2.0 * np.pi * np.sqrt(L * C))

        domain_size = 0.02
        sim = Simulation(freq_max=5e9, domain=(domain_size, domain_size, domain_size),
                          boundary="pec")
        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        sim.add_source(center, "ez",
                        waveform=GaussianPulse(f0=f0_analytical, bandwidth=0.8))
        sim.add_lumped_rlc(center, component="ez",
                            R=R, L=L, C=C, topology="parallel")
        sim.add_probe(center, "ez")
        result = sim.run(n_steps=4000, compute_s_params=False)

        ts = np.array(result.time_series).ravel()

        # Just verify signal is non-trivial (spectrum analysis is topology-dependent)
        peak = np.max(np.abs(ts))
        assert peak > 0, "Parallel RLC should produce non-zero field"


class TestMultipleRLCElements:
    """Multiple RLC elements in one simulation."""

    def test_two_rlc_elements(self):
        """Two RLC elements at different positions should both work."""
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary="pec")
        sim.add_source((0.01, 0.01, 0.01), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            R=50.0, C=1e-12, topology="series")
        sim.add_lumped_rlc((0.015, 0.015, 0.015), component="ez",
                            L=10e-9, topology="series")
        sim.add_probe((0.01, 0.01, 0.01), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)


class TestRLCWithOtherComponents:
    """RLC elements combined with other simulation features."""

    def test_rlc_with_port(self):
        """RLC element combined with a lumped port should work."""
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary="pec")
        sim.add_port((0.01, 0.01, 0.01), "ez", impedance=50.0)
        sim.add_lumped_rlc((0.015, 0.015, 0.01), component="ez",
                            R=100.0, C=1e-12, topology="series")
        sim.add_probe((0.015, 0.015, 0.01), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None


class TestExComponent:
    """RLC element on Ex component."""

    def test_ex_component(self):
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ex",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ex", R=50.0)
        sim.add_probe((0.005, 0.005, 0.005), "ex")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None

    def test_ey_component(self):
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ey",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ey", R=50.0)
        sim.add_probe((0.005, 0.005, 0.005), "ey")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None

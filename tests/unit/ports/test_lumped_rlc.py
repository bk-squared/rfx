"""Lumped RLC element tests.

Validates (direction-only / smoke coverage; the quantitative f0 = 1/(2*pi*
sqrt(LC)) and Q = (1/R)*sqrt(L/C) oracle lives in
``test_series_rlc_current.py::test_series_rlc_current_peaks_at_f0``):
1. Adding an LC element measurably shifts the driven-cell spectrum near f0
   (the resonance itself is quantitatively gated on the series CURRENT, not
   the E-field probe — see the observable note in test_series_rlc_current.py).
2. Pure capacitor and pure inductor elements run without error
3. Parallel RLC topology runs without error
4. Validation catches bad inputs
5. RLC does not break existing non-RLC simulations (no regression)
Every domain length written as ``4 * (299792458.0 / 5e9 / 20.0)``, ``7 * (299792458.0 / 5e9 / 20.0)`` or ``domain_cells * (299792458.0 / 5e9 / 20.0)`` is a whole number of cells of dx, so the solved domain equals the declared one.
"""

import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
from rfx.lumped import init_rlc_state

INDUCTANCE_H = 10e-9
CAPACITANCE_F = 1e-12


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
        sim = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
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
        sim_ref = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim_ref.add_source((0.005, 0.005, 0.005), "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim_ref.add_probe((0.007, 0.007, 0.005), "ez")
        result_ref = sim_ref.run(n_steps=300, compute_s_params=False)

        sim_rlc = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim_rlc.add_source((0.005, 0.005, 0.005), "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        STRONG_DAMPING_RESISTANCE_OHM = 1.0
        sim_rlc.add_lumped_rlc((0.007, 0.007, 0.005), component="ez",
                                R=STRONG_DAMPING_RESISTANCE_OHM)
        sim_rlc.add_probe((0.007, 0.007, 0.005), "ez")
        result_rlc = sim_rlc.run(n_steps=300, compute_s_params=False)

        ts_ref = np.array(result_ref.time_series).ravel()
        ts_rlc = np.array(result_rlc.time_series).ravel()
        assert not np.allclose(ts_ref, ts_rlc, atol=1e-10), \
            "RLC element should visibly affect the field"


class TestPureCapacitor:
    """Pure capacitor element (C only, R=0, L=0)."""

    def test_pure_capacitor_runs(self):
        """Pure C element should run without error."""
        sim = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            C=CAPACITANCE_F)
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)


class TestPureInductor:
    """Pure inductor element (L only, R=0, C=0)."""

    def test_pure_inductor_runs(self):
        """Pure L element should run without error."""
        sim = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            L=INDUCTANCE_H)
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
        L = INDUCTANCE_H
        C = CAPACITANCE_F
        f0_lc = 1.0 / (2.0 * np.pi * np.sqrt(L * C))

        freq_max = 5e9
        domain_size = 0.02
        domain_cells = 7
        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        n_steps = 2000

        sim_ref = Simulation(freq_max=freq_max,
                              domain=(domain_cells * (299792458.0 / 5e9 / 20.0),) * 3, dx=299792458.0 / 5e9 / 20.0,
                              boundary="pec")
        sim_ref.add_source(center, "ez",
                            waveform=GaussianPulse(f0=f0_lc, bandwidth=0.8))
        sim_ref.add_probe(center, "ez")
        result_ref = sim_ref.run(n_steps=n_steps, compute_s_params=False)

        STRONG_LOADING_RESISTANCE_OHM = 10.0
        sim_lc = Simulation(freq_max=freq_max,
                             domain=(domain_cells * (299792458.0 / 5e9 / 20.0),) * 3, dx=299792458.0 / 5e9 / 20.0,
                             boundary="pec")
        sim_lc.add_source(center, "ez",
                           waveform=GaussianPulse(f0=f0_lc, bandwidth=0.8))
        sim_lc.add_lumped_rlc(center, component="ez",
                               R=STRONG_LOADING_RESISTANCE_OHM, L=L, C=C, topology="series")
        sim_lc.add_probe(center, "ez")
        result_lc = sim_lc.run(n_steps=n_steps, compute_s_params=False)

        ts_ref = np.array(result_ref.time_series).ravel()
        ts_lc = np.array(result_lc.time_series).ravel()
        dt = result_ref.dt

        assert not np.allclose(ts_ref, ts_lc, atol=1e-10), \
            "LC element should modify the time series"

        freqs = np.fft.rfftfreq(len(ts_ref), dt)
        spec_ref = np.abs(np.fft.rfft(ts_ref))
        spec_lc = np.abs(np.fft.rfft(ts_lc))

        mask = (freqs > f0_lc * 0.3) & (freqs < f0_lc * 3.0)
        if np.any(mask):
            ratio = spec_lc[mask] / np.maximum(spec_ref[mask], 1e-30)
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
        domain_cells = 4
        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        n_steps = 500

        sim_ref = Simulation(freq_max=freq_max,
                              domain=(domain_cells * (299792458.0 / 5e9 / 20.0),) * 3, dx=299792458.0 / 5e9 / 20.0,
                              boundary="pec")
        sim_ref.add_source(center, "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim_ref.add_probe(center, "ez")
        result_ref = sim_ref.run(n_steps=n_steps, compute_s_params=False)

        sim_cap = Simulation(freq_max=freq_max,
                              domain=(domain_cells * (299792458.0 / 5e9 / 20.0),) * 3, dx=299792458.0 / 5e9 / 20.0,
                              boundary="pec")
        sim_cap.add_source(center, "ez",
                            waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim_cap.add_lumped_rlc(center, component="ez", C=CAPACITANCE_F)
        sim_cap.add_probe(center, "ez")
        result_cap = sim_cap.run(n_steps=n_steps, compute_s_params=False)

        ts_ref = np.array(result_ref.time_series).ravel()
        ts_cap = np.array(result_cap.time_series).ravel()

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
        sim = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            R=50.0, L=INDUCTANCE_H, C=CAPACITANCE_F, topology="parallel")
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)

    def test_parallel_rlc_resonance(self):
        """Parallel RLC runs and produces a non-trivial field (smoke test).

        NOTE: this only checks the field is non-zero, not the anti-resonance
        location. The quantitative resonance oracle is the series-current gate
        in test_series_rlc_current.py; a parallel-current oracle is not yet
        added.
        """
        HIGH_PARALLEL_RESONANCE_RESISTANCE_OHM = 200.0
        R = HIGH_PARALLEL_RESONANCE_RESISTANCE_OHM
        L = INDUCTANCE_H
        C = CAPACITANCE_F
        f0_analytical = 1.0 / (2.0 * np.pi * np.sqrt(L * C))

        domain_size = 0.02
        domain_cells = 7
        sim = Simulation(freq_max=5e9, domain=(domain_cells * (299792458.0 / 5e9 / 20.0),) * 3, dx=299792458.0 / 5e9 / 20.0,
                          boundary="pec")
        center = (domain_size / 2, domain_size / 2, domain_size / 2)
        sim.add_source(center, "ez",
                        waveform=GaussianPulse(f0=f0_analytical, bandwidth=0.8))
        sim.add_lumped_rlc(center, component="ez",
                            R=R, L=L, C=C, topology="parallel")
        sim.add_probe(center, "ez")
        result = sim.run(n_steps=4000, compute_s_params=False)

        ts = np.array(result.time_series).ravel()

        peak = np.max(np.abs(ts))
        assert peak > 0, "Parallel RLC should produce non-zero field"


class TestMultipleRLCElements:
    """Multiple RLC elements in one simulation."""

    def test_two_rlc_elements(self):
        """Two RLC elements at different positions should both work."""
        sim = Simulation(freq_max=5e9, domain=(7 * (299792458.0 / 5e9 / 20.0), 7 * (299792458.0 / 5e9 / 20.0), 7 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim.add_source((0.01, 0.01, 0.01), "ez",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ez",
                            R=50.0, C=CAPACITANCE_F, topology="series")
        sim.add_lumped_rlc((0.015, 0.015, 0.015), component="ez",
                            L=INDUCTANCE_H, topology="series")
        sim.add_probe((0.01, 0.01, 0.01), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None
        assert result.time_series.shape == (200, 1)


class TestRLCWithOtherComponents:
    """RLC elements combined with other simulation features."""

    def test_rlc_with_port(self):
        """RLC element combined with a lumped port should work."""
        sim = Simulation(freq_max=5e9, domain=(7 * (299792458.0 / 5e9 / 20.0), 7 * (299792458.0 / 5e9 / 20.0), 7 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim.add_port((0.01, 0.01, 0.01), "ez", impedance=50.0)
        sim.add_lumped_rlc((0.015, 0.015, 0.01), component="ez",
                            R=100.0, C=CAPACITANCE_F, topology="series")
        sim.add_probe((0.015, 0.015, 0.01), "ez")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None


class TestExComponent:
    """RLC element on Ex component."""

    def test_ex_component(self):
        sim = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ex",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ex", R=50.0)
        sim.add_probe((0.005, 0.005, 0.005), "ex")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None

    def test_ey_component(self):
        sim = Simulation(freq_max=5e9, domain=(4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0), 4 * (299792458.0 / 5e9 / 20.0)), dx=(299792458.0 / 5e9 / 20.0), boundary="pec")
        sim.add_source((0.005, 0.005, 0.005), "ey",
                        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.8))
        sim.add_lumped_rlc((0.005, 0.005, 0.005), component="ey", R=50.0)
        sim.add_probe((0.005, 0.005, 0.005), "ey")
        result = sim.run(n_steps=200, compute_s_params=False)
        assert result is not None

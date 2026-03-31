"""Tests for the field-decay-based stopping criterion (spec 6E2).

Test 1: Decay stops after pulse exits (PEC cavity ring-down)
Test 2: Decay produces better DFT than fixed short run
Test 3: min_steps and max_steps are honored
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import (
    run_until_decay, run, make_source, make_probe, SourceSpec, ProbeSpec,
)


def _make_cavity(freq_max=5e9, domain=(0.05, 0.05, 0.025)):
    """Build a small PEC cavity grid + materials."""
    grid = Grid(freq_max=freq_max, domain=domain)
    materials = init_materials(grid.shape)
    return grid, materials


def _make_sources_and_probes(grid, n_steps):
    """Create a Gaussian pulse source and center probe for *n_steps*."""
    pulse = GaussianPulse(f0=2.5e9, bandwidth=2.5e9)
    src_pos = (0.01, 0.01, 0.01)
    source = make_source(grid, src_pos, "ez", pulse, n_steps)
    probe_pos = (0.025, 0.025, 0.0125)
    probe = make_probe(grid, probe_pos, "ez")
    return [source], [probe], probe_pos


class TestDecayStopsAfterPulseExits:
    """Test 1: decay-based stopping terminates when pulse rings down."""

    def test_decay_stops_after_pulse_exits(self):
        grid, materials = _make_cavity()
        max_steps = 10000
        sources, probes, probe_pos = _make_sources_and_probes(grid, max_steps)

        result = run_until_decay(
            grid, materials,
            decay_by=1e-2,
            check_interval=50,
            min_steps=100,
            max_steps=max_steps,
            monitor_component="ez",
            monitor_position=probe_pos,
            sources=sources,
            probes=probes,
        )

        actual_steps = result.time_series.shape[0]

        # Should have stopped before max_steps
        assert actual_steps < max_steps, (
            f"Expected early stop but ran {actual_steps} steps"
        )

        # Final |field|^2 / peak should be < decay_by
        ts = np.array(result.time_series[:, 0])
        sq = ts ** 2
        peak_sq = np.max(sq)
        final_sq = sq[-1]
        assert peak_sq > 0, "Peak should be nonzero"
        assert final_sq < 1e-2 * peak_sq, (
            f"final_sq/peak_sq = {final_sq/peak_sq:.6e}, expected < 1e-2"
        )


class TestDecayProducesBetterDFT:
    """Test 2: decay run has less truncation ripple than short fixed run."""

    def test_decay_produces_better_dft_than_fixed_short(self):
        grid, materials = _make_cavity()

        # Short fixed run (100 steps) — source waveform sized to 100
        n_short = 100
        sources_short, probes_short, probe_pos = _make_sources_and_probes(
            grid, n_short,
        )
        result_short = run(
            grid, materials, n_short,
            sources=sources_short,
            probes=probes_short,
        )

        # Decay-based run — source waveform sized to max_steps
        max_steps = 10000
        sources_decay, probes_decay, _ = _make_sources_and_probes(
            grid, max_steps,
        )
        result_decay = run_until_decay(
            grid, materials,
            decay_by=1e-3,
            check_interval=50,
            min_steps=100,
            max_steps=max_steps,
            monitor_component="ez",
            monitor_position=probe_pos,
            sources=sources_decay,
            probes=probes_decay,
        )

        def spectral_variance(ts_arr):
            """Compute variance of normalised power spectrum."""
            sig = np.array(ts_arr).flatten()
            spectrum = np.abs(np.fft.rfft(sig)) ** 2
            if spectrum.max() > 0:
                spectrum = spectrum / spectrum.max()
            return float(np.var(spectrum))

        var_short = spectral_variance(result_short.time_series[:, 0])
        var_decay = spectral_variance(result_decay.time_series[:, 0])

        # Decay run ran longer and should have smoother spectrum
        assert result_decay.time_series.shape[0] > result_short.time_series.shape[0], (
            "Decay run should be longer than the 100-step fixed run"
        )
        # The longer, converged signal should have lower spectral variance
        assert var_decay < var_short, (
            f"Expected decay spectral var ({var_decay:.6e}) < "
            f"short spectral var ({var_short:.6e})"
        )


class TestMinMaxStepsHonored:
    """Test 3: min_steps and max_steps constraints are respected."""

    def test_min_steps_honored(self):
        """Even if decayed early, should run at least min_steps."""
        grid, materials = _make_cavity()
        max_steps = 5000
        sources, probes, probe_pos = _make_sources_and_probes(grid, max_steps)

        result = run_until_decay(
            grid, materials,
            decay_by=0.99,  # Very generous — would stop immediately
            check_interval=10,
            min_steps=200,
            max_steps=max_steps,
            monitor_component="ez",
            monitor_position=probe_pos,
            sources=sources,
            probes=probes,
        )

        actual_steps = result.time_series.shape[0]
        assert actual_steps >= 200, (
            f"Expected at least 200 steps (min_steps), got {actual_steps}"
        )

    def test_max_steps_honored(self):
        """Should stop at max_steps even if not decayed."""
        grid, materials = _make_cavity()
        max_steps = 50
        sources, probes, probe_pos = _make_sources_and_probes(grid, max_steps)

        result = run_until_decay(
            grid, materials,
            decay_by=1e-30,  # Impossibly tight — will never converge
            check_interval=10,
            min_steps=10,
            max_steps=max_steps,
            monitor_component="ez",
            monitor_position=probe_pos,
            sources=sources,
            probes=probes,
        )

        actual_steps = result.time_series.shape[0]
        assert actual_steps == max_steps, (
            f"Expected exactly {max_steps} steps (max_steps), got {actual_steps}"
        )

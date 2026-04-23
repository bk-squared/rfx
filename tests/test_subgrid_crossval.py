"""Phase-1 z-slab benchmark comparisons against a uniform-fine reference.

These tests are intentionally narrower than the previous broad 3D cavity
cross-checks.  They target the approved Phase-1 lane:

- z-slab only
- single canonical stepper
- PEC boundary only

The benchmarks use a normal-incidence proxy geometry and compare a
subgridded run against a uniform-fine reference at a fixed evaluation
frequency.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _dft_amplitude_phase(signal: np.ndarray, dt: float, freq_hz: float) -> tuple[float, float]:
    t = np.arange(len(signal)) * dt
    coeff = np.sum(signal.astype(np.float64) * np.exp(-1j * 2.0 * np.pi * freq_hz * t)) * dt
    return abs(coeff), float(np.angle(coeff, deg=True))


def _phase_error_deg(phase_a: float, phase_b: float) -> float:
    return abs((phase_a - phase_b + 180.0) % 360.0 - 180.0)


def _benchmark_errors(result_ref, result_sub, freq_hz: float) -> tuple[float, float]:
    amp_ref, phase_ref = _dft_amplitude_phase(
        np.asarray(result_ref.time_series[:, 0]), float(result_ref.dt), freq_hz
    )
    amp_sub, phase_sub = _dft_amplitude_phase(
        np.asarray(result_sub.time_series[:, 0]), float(result_sub.dt), freq_hz
    )
    amp_error = abs(amp_sub - amp_ref) / max(amp_ref, 1e-30)
    phase_error = _phase_error_deg(phase_sub, phase_ref)
    return amp_error, phase_error


def _assert_benchmark(label: str, result_ref, result_sub, freq_hz: float) -> None:
    amp_error, phase_error = _benchmark_errors(result_ref, result_sub, freq_hz)
    print(
        f"\n{label}: amp_error={amp_error:.3%}, "
        f"phase_error={phase_error:.3f}°"
    )
    assert amp_error <= 0.05
    assert phase_error <= 5.0


class _ThinSlabFixture:
    freq_max = 10e9
    uniform_dx = 1e-3
    subgrid_dx = 2e-3
    ratio = 2
    tau = 1.0
    domain = (0.04, 0.04, 0.04)
    eps_r = 1.5
    slab = (0.019, 0.021)
    source_z = 0.014
    freq_eval = 2.0e9
    n_steps = 800

    def _run_case(
        self,
        *,
        domain: tuple[float, float, float],
        dx: float,
        slab: tuple[float, float],
        source_z: float,
        probe_z: float,
        eps_r: float,
        refinement: tuple[float, float] | None = None,
    ):
        from rfx import Box, Simulation

        sim = Simulation(freq_max=self.freq_max, domain=domain, boundary="pec", dx=dx)
        sim.add_material("dielectric", eps_r=eps_r)
        sim.add(Box((0, 0, slab[0]), (domain[0], domain[1], slab[1])), material="dielectric")
        if refinement is not None:
            sim.add_refinement(z_range=refinement, ratio=self.ratio, tau=self.tau)
        sim.add_source(position=(domain[0] / 2, domain[1] / 2, source_z), component="ez")
        sim.add_probe(position=(domain[0] / 2, domain[1] / 2, probe_z), component="ez")
        return sim.run(n_steps=self.n_steps)


class TestPhase1SubgridBenchmarks(_ThinSlabFixture):
    def test_zslab_plane_wave_reflection_vs_uniform_fine(self):
        """Reflection-side proxy benchmark against a uniform-fine reference."""

        probe_z = 0.0165
        result_ref = self._run_case(
            domain=self.domain,
            dx=self.uniform_dx,
            slab=self.slab,
            source_z=self.source_z,
            probe_z=probe_z,
            eps_r=self.eps_r,
        )
        result_sub = self._run_case(
            domain=self.domain,
            dx=self.subgrid_dx,
            slab=self.slab,
            source_z=self.source_z,
            probe_z=probe_z,
            eps_r=self.eps_r,
            refinement=(0.010, 0.030),
        )

        _assert_benchmark("Reflection proxy benchmark", result_ref, result_sub, self.freq_eval)

    def test_zslab_dielectric_transmission_vs_uniform_fine(self):
        """Transmission benchmark against a uniform-fine reference."""
        from rfx import Box, Simulation

        domain = (0.06, 0.06, 0.06)
        slab = (0.018, 0.022)
        source_z = 0.015
        # Keep the same transmission-side physical question, but sample one
        # effective cell closer to the slab to reduce downstream PEC-cavity
        # contamination in the raw benchmark comparison.
        probe_z = 0.024

        result_ref = self._run_case(
            domain=domain,
            dx=self.uniform_dx,
            slab=slab,
            source_z=source_z,
            probe_z=probe_z,
            eps_r=1.5,
        )
        result_sub = self._run_case(
            domain=domain,
            dx=self.subgrid_dx,
            slab=slab,
            source_z=source_z,
            probe_z=probe_z,
            eps_r=1.5,
            refinement=(0.010, 0.050),
        )

        _assert_benchmark("Transmission benchmark", result_ref, result_sub, self.freq_eval)

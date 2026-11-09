"""Public material fits and frozen models retain their declared mesh."""
import sys

import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.differentiable_material_fit import differentiable_material_fit
from rfx.material_fit import DebyeFitResult


def _conformal_walls():
    return BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec", conformal=True), z="cpml")


@pytest.mark.parametrize("conformal", [False], ids=["default-boundary"])
@pytest.mark.xfail(
    strict=True,
    reason="#2535: the conformal-floor admission (rfx/runners/_admission.py:876) "
    "resolves the auto mesh on the fit's traced per-iteration simulation, so an "
    "auto-mesh differentiable_material_fit is refused before the first step",
)
def test_auto_mesh_material_fit_returns_finite_s_params(conformal):
    def factory(eps_inf, debye_poles, lorentz_poles):
        kwargs = {"boundary": _conformal_walls()} if conformal else {}
        sim = Simulation(freq_max=5e9, domain=(0.024, 0.009, 0.009), **kwargs)
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles,
                         lorentz_poles=lorentz_poles)
        sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)), material="dut")
        sim.add_port((0.003, 0.004, 0.003), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
        sim.add_probe((0.020, 0.004, 0.003), "ez")
        sim.add_probe((0.007, 0.004, 0.003), "ez")
        return sim

    truth = DebyeFitResult(eps_inf=2.0, poles=[DebyePole(delta_eps=3.0, tau=50e-12)], fit_error=0.0)
    # This independent template reports the realized mesh; the fit's factory
    # always returns a fresh, unresolved model, including while tracing.
    grid = factory(truth.eps_inf, truth.poles, []).freeze_mesh()
    print(f"fit conformal={conformal}: mesh={grid.shape}, dx={grid.dx:.12g}, "
          f"steps={grid.num_timesteps(num_periods=20.0)}, expected=finite final_s_params",
          file=sys.stderr)
    fit = differentiable_material_fit(
        factory, np.zeros((1, 1, 5), complex), np.linspace(2e9, 4e9, 5),
        n_debye_poles=1, n_iterations=1, learning_rate=0.0,
        initial_guess=truth, verbose=False,
        normalization="reference_probe", reference_probe=1,
    )
    assert fit.final_s_params is not None, "material fit returned no final_s_params"
    values = np.asarray(fit.final_s_params)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    print(f"fit conformal={conformal}: nonfinite={nonfinite}, bar=0, "
          f"abs_s={np.abs(values).tolist()}", file=sys.stderr)
    assert values.shape == (1, 1, 5)
    assert nonfinite == 0, f"material fit returned {nonfinite} nonfinite S values"


def _frozen_model(domain):
    sim = Simulation(freq_max=5e9, domain=domain, dx=1.5e-3, boundary=_conformal_walls())
    sim.add_material("dut", eps_r=2.0)
    sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)), material="dut")
    sim.add_source((0.006, 0.003, 0.003), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.0075, 0.003, 0.003), "ez")
    return sim


def test_frozen_domain_mutation_preserves_probe_record():
    domain = [0.024, 0.009, 0.009]
    sim = _frozen_model(domain)
    sim.freeze_mesh()
    domain[2] = 0.050
    expected = _frozen_model((0.024, 0.009, 0.009)).run(n_steps=16)
    actual = sim.run(n_steps=16)
    a, b = np.asarray(actual.time_series), np.asarray(expected.time_series)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.any(b), "frozen-domain reference probe record is zero"
    difference = float(np.max(np.abs(a - b)))
    print(f"frozen-domain: max_abs={difference:.12g}, bar=0 (byte equality), "
          f"peak={np.max(np.abs(b)):.12g}, steps=16, "
          f"expected_mesh={expected.grid.shape}, measured_mesh={actual.grid.shape}", file=sys.stderr)
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), (
        f"caller domain mutation changed the frozen probe record: max_abs={difference}, bar=0")

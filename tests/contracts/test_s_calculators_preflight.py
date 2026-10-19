"""Direct S calculators share run()'s admission; run() admits only once."""
import sys

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from rfx.sparams.waveguide import compute_waveguide_s_matrix


class CalculatorGridReached(Exception):
    """Stop at the calculator's own build, after its admission block."""


def _guide():
    sim = Simulation(
        freq_max=8e9, domain=(0.06, 0.024, 0.012), dx=0.003,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")), cpml_layers=6)
    for x, direction in ((0.009, "+x"), (0.051, "-x")):
        sim.add_waveguide_port(x, direction=direction, mode=(1, 0),
                              mode_type="TE", freqs=np.array([7e9]),
                              f0=7e9, bandwidth=0.4)
    # Declared spacing 5 mm, realized plane 6 mm on the 3 mm lattice.
    sim.add(Box((0.021, 0, 0.005), (0.039, 0.024, 0.005)), material="pec")
    return sim


def _lumped(*, sheet=False):
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.012),
                     dx=0.001, boundary="pec")
    for x in (0.004, 0.008):
        sim.add_port((x, 0.006, 0.003), "ez", impedance=50,
                     waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    if sheet:
        # Declared spacing 5.4 mm, realized plane 5 mm on the 1 mm lattice.
        sim.add(Box((0, 0, 0.0054), (0.012, 0.012, 0.0054)), material="pec")
    return sim


def _intercept_calculator_grid(monkeypatch, calculator):
    original = Simulation._build_grid

    def build(self, *args, **kwargs):
        # Admission and preflight keep their real builds. Intercept only
        # the direct call in the calculator, so a missing preflight goes red.
        if sys._getframe(1).f_code is calculator.__code__:
            raise CalculatorGridReached
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Simulation, "_build_grid", build)


@pytest.mark.parametrize("dispatch", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_waveguide_shared_preflight(monkeypatch, dispatch, skip):
    sim = _guide()
    _intercept_calculator_grid(monkeypatch, compute_waveguide_s_matrix)
    call = sim.compute_s_matrix if dispatch else sim.compute_waveguide_s_matrix
    if skip:
        with pytest.raises(CalculatorGridReached):
            call(n_steps=4, skip_preflight=True)
    else:
        # Main's sheet_cavity_spacing finding: the drawn cavity must be honored.
        with pytest.raises(ValueError, match="parallel PEC cavity faces.*drawn spacing"):
            call(n_steps=4)


@pytest.mark.parametrize("skip", [False, True])
def test_lumped_scan_shared_preflight(monkeypatch, skip):
    sim = _lumped(sheet=True)
    _intercept_calculator_grid(monkeypatch, compute_lumped_wire_s_matrix_via_scan)
    if skip:
        with pytest.raises(CalculatorGridReached):
            compute_lumped_wire_s_matrix_via_scan(
                sim, np.array([5e9]), n_steps=4, skip_preflight=True)
    else:
        with pytest.raises(ValueError, match="parallel PEC cavity faces.*drawn spacing"):
            compute_lumped_wire_s_matrix_via_scan(sim, np.array([5e9]), n_steps=4)


def test_run_two_lumped_ports_preflights_once(monkeypatch):
    sim = _lumped()
    original = Simulation._auto_preflight
    calls = []

    def observe(self, *args, **kwargs):
        calls.append(kwargs.get("context"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Simulation, "_auto_preflight", observe)
    result = sim.run(n_steps=12, s_param_n_steps=12, s_param_freqs=np.array([5e9]))
    assert result.s_params.shape == (2, 2, 1)
    assert calls == ["run"]

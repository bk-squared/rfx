"""Calculator declarations are refused before grid construction or a scan."""

import jax
import pytest

from rfx import Box

from tests.unit.sparams.test_waveguide_solver_precision import _guide


@pytest.mark.parametrize("feature", ["solver", "kerr"])
def test_waveguide_calculator_refuses_before_grid_or_scan(monkeypatch, feature):
    sim = _guide(solver="adi" if feature == "solver" else "yee")
    if feature == "kerr":
        sim.add_material("nonlinear", chi3=1e-20)
        sim.add(Box((0.02, 0.01, 0.005), (0.06, 0.03, 0.015)), material="nonlinear")

    def forbidden(*args, **kwargs):
        pytest.fail("calculator reached grid construction or lax.scan")

    monkeypatch.setattr(type(sim), "_build_grid", forbidden)
    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError) as caught:
        sim.compute_waveguide_s_matrix(n_steps=4, normalize=True)
    message = str(caught.value)
    assert "compute_waveguide_s_matrix() does not carry" in message
    assert ("solver='adi'" if feature == "solver" else "Kerr") in message
    assert "run() / forward()" in message


@pytest.mark.parametrize("feature", ["source", "tfsf", "periodic", "graded_interface"])
def test_existing_waveguide_value_errors_precede_admission(feature, monkeypatch):
    from rfx import GaussianPulse
    sim = _guide(nonuniform=feature == "graded_interface")
    if feature == "source":
        sim.add_source((.04, .02, .01), "ez", waveform=GaussianPulse(f0=8e9))
    elif feature == "tfsf":
        sim.add_tfsf_source(f0=8e9)
    elif feature == "periodic":
        # The public setter already refuses this combination; exercise
        # the calculator's existing check on a stored declaration.
        sim._periodic_axes = "z"
    else:
        sim._interface_eps = "dual_average"

    def forbidden(*args, **kwargs):
        pytest.fail("existing ValueError moved behind calculator admission")

    monkeypatch.setattr("rfx.runners._admission.admit", forbidden)
    with pytest.raises(ValueError):
        sim.compute_waveguide_s_matrix(n_steps=4)


def test_waveguide_reference_kerr_refused_before_scan(monkeypatch):
    from tests.unit.runners.test_calculator_admission import declare_physics
    reference = _guide()
    declare_physics(reference, "kerr")

    def forbidden(*args, **kwargs):
        pytest.fail("waveguide stepped before admitting its reference model")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError, match="Kerr"):
        _guide().compute_waveguide_s_matrix(n_steps=4, normalize="flux",
                                           port_reference_sims=[reference, _guide()])

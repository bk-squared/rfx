"""Calculator declarations are refused before grid construction or a scan."""

import jax
import pytest

from tests.unit.sparams.test_waveguide_solver_precision import _guide


@pytest.mark.parametrize("feature", ["solver", "ntff"])
def test_waveguide_calculator_refuses_before_grid_or_scan(monkeypatch, feature):
    sim = _guide(solver="adi" if feature == "solver" else "yee")
    if feature == "ntff":
        sim.add_ntff_box((0.02, 0.01, 0.005), (0.06, 0.03, 0.015), n_freqs=5)

    def forbidden(*args, **kwargs):
        pytest.fail("calculator reached grid construction or lax.scan")

    monkeypatch.setattr(type(sim), "_build_grid", forbidden)
    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError) as caught:
        sim.compute_waveguide_s_matrix(n_steps=4, normalize=True)
    message = str(caught.value)
    assert "compute_waveguide_s_matrix() does not carry" in message
    assert ("solver='adi'" if feature == "solver" else "an NTFF box") in message
    assert "run() / forward()" in message

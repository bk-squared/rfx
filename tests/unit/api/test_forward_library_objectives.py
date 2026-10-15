"""Every objective in rfx.optimize_objectives runs on a real forward() result.

Two ports with port 0 driven, on the uniform mesh (lumped ports) and the graded
mesh (wire ports; graded lumped S needs a reference channel one record lacks).
forward() returns port S as (n_ports, n_ports, n_freqs) with the undriven column
NaN (#1373 decision C); the S objectives read S[0, 0] and S[1, 0], the driven
column. Before that layout they raised IndexError on forward()'s (F,) / (N, F).
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx.optimize_objectives as objectives
from tests.contracts.run_forward_census import BUILDERS, build_p_wire

FREQS = np.array([4., 8., 12., 16.]) * 1e9
SECOND_PROBE = (.016, .006, .005)

OBJECTIVES = {
    "minimize_s11": lambda: objectives.minimize_s11(FREQS, target_db=-80.0),
    "maximize_s21": lambda: objectives.maximize_s21(FREQS),
    "target_impedance": lambda: objectives.target_impedance(8e9),
    "maximize_bandwidth": lambda: objectives.maximize_bandwidth(10e9, 8e9),
    "minimize_s11_at_freq_wave_decomp": lambda: objectives.minimize_s11_at_freq_wave_decomp(8e9),
    "minimize_reflected_energy": lambda: objectives.minimize_reflected_energy(0),
    "maximize_transmitted_energy": lambda: objectives.maximize_transmitted_energy(-1),
    "steer_probe_array": lambda: objectives.steer_probe_array(1, 0),
    "maximize_directivity": lambda: objectives.maximize_directivity(
        np.pi / 2, 0.0, n_theta=9, n_phi=17),
    "maximize_directivity_logratio": lambda: objectives.maximize_directivity_logratio(
        np.pi / 2, 0.0, n_theta=9, n_phi=17),
}
RETIRED = {"minimize_s11_at_freq"}


def _two_port(mesh):
    if mesh == "uniform":
        sim = BUILDERS["P-lumped2"]("uniform")
    else:
        sim = build_p_wire("graded")
        # The port S request refuses a plain source beside the ports.
        sim._ports = [pe for pe in sim._ports if pe.impedance != 0]
        sim.add_port((.020, .012, .003), "ez", impedance=50., extent=.003, excite=False)
    sim.add_probe(SECOND_PROBE, "ez")
    # The census far-field box (tests/contracts/run_forward_census.py, build_o_ntff).
    sim.add_ntff_box((.006, .004, .002), (.025, .016, .010), freqs=np.array([10e9]))
    return sim


def test_every_library_objective_is_listed():
    public = {name: value for name, value in vars(objectives).items()
              if inspect.isfunction(value) and value.__module__ == objectives.__name__
              and not name.startswith("_")}
    # An alias (maximize_directivity_ratio) is covered by the function it names.
    assert {value.__name__ for value in public.values()} == set(OBJECTIVES) | RETIRED


@pytest.mark.parametrize("name", sorted(OBJECTIVES))
@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_objective_value_and_gradient_on_forward(mesh, name):
    sim = _two_port(mesh)
    grid = sim._build_nonuniform_grid() if mesh == "graded" else sim._build_grid()
    objective = OBJECTIVES[name]()

    def loss(eps_override):
        result = sim.forward(eps_override=eps_override, n_steps=150, port_s11_freqs=FREQS,
                             skip_preflight=True, checkpoint=False)
        assert result.s_params.shape == (2, 2, len(FREQS))
        return objective(result)

    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    value, gradient = jax.value_and_grad(loss)(eps)
    assert np.isfinite(float(value)), name
    assert np.isfinite(np.asarray(gradient)).all(), name


def test_s21_on_one_port_forward_names_the_shape():
    result = BUILDERS["P-lumped"]("uniform").forward(
        n_steps=60, port_s11_freqs=FREQS, skip_preflight=True)
    assert result.s_params.shape == (1, 1, len(FREQS))
    with pytest.raises(ValueError, match=r"S\[1, 0\].*shape \(1, 1, 4\)"):
        objectives.maximize_s21(FREQS)(result)


def test_retired_objective_names_its_replacement():
    with pytest.raises(AttributeError, match="minimize_s11_at_freq_wave_decomp"):
        objectives.minimize_s11_at_freq(8e9)

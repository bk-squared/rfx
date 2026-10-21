"""One record's driven column agrees with run's independent drive solves."""
from dataclasses import replace
from types import SimpleNamespace
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import ForwardResult
from rfx.measurement import port_s_from_record
from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp
from rfx.validation import normalize_port_smatrix
from tests.contracts.run_forward_census import BUILDERS
from tests.unit.nonuniform.test_nu_wire_port_lane_parity import (
    _build, FREQS, PORT_Y, PORT_Z,
)


@pytest.mark.parametrize('mesh,ports', [('uniform', 1), ('uniform', 2),
                                       ('graded', 1), ('graded', 2)])
def test_driven_column_matches_run(mesh, ports):
    if mesh == 'graded':
        sim = _build(True, excite=True)
        if ports == 2:
            sim.add_port((12e-3, PORT_Y, PORT_Z), 'ez', impedance=50,
                         extent=3e-3, excite=False)
        run = sim.run(n_steps=150, skip_preflight=True,
                      s_param_freqs=FREQS, compute_s_params=True)
    else:
        sim = BUILDERS['P-lumped' if ports == 1 else 'P-lumped2'](mesh)
        run = sim.run(n_steps=150, skip_preflight=True)
    forward = sim.forward(n_steps=150, skip_preflight=True, port_s11_freqs=run.freqs)
    assert forward.s_params.shape == (ports, ports, len(run.freqs))
    expected, actual = np.asarray(run.s_params), np.asarray(forward.s_params)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0,
                               atol=1e-4 * np.max(np.abs(expected)))
    assert np.isfinite(actual[:, 0]).all()
    if ports == 2:
        assert np.isnan(actual[:, 1]).all()


def test_objective_refuses_vector():
    result = SimpleNamespace(s_params=jnp.ones(3, dtype=jnp.complex64),
                             freqs=jnp.array([1e9, 2e9, 3e9]))
    with pytest.raises(ValueError, match=r'\(n_ports, n_ports, n_freqs\)'):
        minimize_s11_at_freq_wave_decomp(2e9)(result)


def test_validation_refuses_forward_rows():
    result = ForwardResult(time_series=jnp.zeros((1, 1)),
                           s_params=jnp.ones((2, 3)), freqs=jnp.arange(3))
    with pytest.raises(ValueError, match=r'\(n_ports, n_ports, n_freqs\)'):
        normalize_port_smatrix(result)


def test_all_passive_two_port_lane_refusal():
    messages = []
    for graded in (False, True):
        sim = _build(graded).add_port((12e-3, PORT_Y, PORT_Z), 'ez',
                                     impedance=50, extent=3e-3, excite=False)
        with pytest.raises(ValueError, match='needs a driven port.*lumped_port_sparams') as error:
            sim.forward(n_steps=800, skip_preflight=True, port_s11_freqs=FREQS)
        messages.append(str(error.value))
    assert messages[0] == messages[1]


def test_graded_wire_thru_run_checks_only_the_driven_column():
    from tests._interior_tem_line import build

    sim, _ = build('wire', two=True, profile=np.array(
        [1, .8, 1.2, .9, 1.1, .8, 1.2, 1, 1]) * .25e-3)
    sim._ports[1] = replace(sim._ports[1], excite=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sim.run(n_steps=1000, skip_preflight=True,
                         compute_s_params=True, s_param_freqs=[3e9, 5e9, 7e9])
    # run() drives each port in turn: both columns are measured.
    S = np.asarray(result.s_params)
    assert np.isfinite(S).all() and np.all(np.abs(S) <= 1)
    assert not [w for w in caught if 'non-passive' in str(w.message)
                or 'non-finite' in str(w.message)]


def test_graded_lumped_requires_a_reference_channel():
    with pytest.raises(ValueError, match='pre-injection reference'): port_s_from_record([(jnp.ones(3),) * 3], [(None,) * 4 + (50, None, None, True)], wire_mode=False)

"""D9 PR1: a pre-injection reference on both single-device port paths."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from tests._x64_compat import enable_x64


def constant_scene(graded, kind):
    dx = 1 / 1024
    sim = Simulation(freq_max=20e9, domain=(10*dx, 8*dx, 6*dx), dx=dx,
                     boundary='pec',
                     **({'dz_profile': np.full(6, dx)} if graded else {}))
    sim.add_port((3*dx, 3*dx, 2*dx), 'ez', impedance=50.,
                 extent=2*dx if kind == 'wire' else None,
                 waveform=lambda t: jnp.cos(t * 2e10))
    sim.add_probe((7*dx, 4*dx, 3*dx), 'ez')
    return sim


@pytest.mark.parametrize('kind', ['wire', 'lumped'])
def test_constant_profile_reference_identity(kind):
    """T1: exercise the actual hooks, time records, and in-scan spectrum."""
    with enable_x64(False):
        kwargs = dict(n_steps=32, skip_preflight=True, checkpoint=False,
                      port_s11_freqs=np.array([3e9, 7e9, 13e9], np.float32))
        uniform, graded = [constant_scene(nu, kind).forward(**kwargs) for nu in (False, True)]
    a, b = uniform.sparam_time_records[0], graded.sparam_time_records[0]
    assert a['V_ref'].dtype == b['V_ref'].dtype == np.float32
    assert np.any(np.asarray(b['V_ref']) != 0)
    assert not np.array_equal(b['V_ref'], b['V'])  # A driven pre/post swap must be visible.
    np.testing.assert_array_equal(a['V_ref'], b['V_ref'])
    assert np.asarray(a['V_ref']).tobytes() == np.asarray(b['V_ref']).tobytes()
    ga = graded.wire_port_sparams[0][1]
    ua = (uniform.wire_port_sparams[0][1][4] if kind == 'wire'
          else uniform.lumped_port_sparams[0][1][2])
    assert len(ga) == 5
    assert ua.dtype == ga[4].dtype == np.complex64
    np.testing.assert_array_equal(ua, ga[4])
    assert np.asarray(ua).tobytes() == np.asarray(ga[4]).tobytes()


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('kind', ['wire', 'lumped'])
def test_reference_channel_names_match_both_paths(entry, kind):
    """T2: record-only requests also retain the reference channel."""
    expected = ('V', 'I', 'V_port', 'V_ref') if kind == 'wire' else ('V', 'I', 'V_ref')
    for graded in (False, True):
        kwargs = dict(n_steps=16, skip_preflight=True)
        if entry == 'run':
            kwargs['compute_s_params'] = False
        result = getattr(constant_scene(graded, kind), entry)(**kwargs)
        record = result.sparam_time_records[0]
        assert record.channels == expected
        assert record.samples.shape == (16, len(expected))


def test_graded_reference_is_not_an_s_settling_input():
    from rfx.sparams._record_selection import s_read_records
    result = constant_scene(True, 'wire').forward(
        n_steps=32, skip_preflight=True, checkpoint=False,
        port_s11_freqs=np.array([3e9, 7e9], np.float32))
    public = result.sparam_time_records[0]
    selected, = s_read_records(result)
    assert public.channels == ('V', 'I', 'V_port', 'V_ref')
    assert selected.channels == ('V', 'I', 'V_port')
    np.testing.assert_array_equal(selected.samples, public.samples[:, :3])

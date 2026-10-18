"""D9 PR2: publication preserves the port kind and the measured spectra."""
import numpy as np
import pytest
import jax

from rfx.simulation import LumpedPortSParamSpec
from tests._x64_compat import enable_x64
from tests.contracts.test_graded_port_reference import constant_scene


BINS = np.array([3e9, 7e9, 13e9], np.float32)


def test_t3_lumped_spectra_match_constant_profile():
    with enable_x64(False):
        results = []
        for graded in (False, True):
            sim = constant_scene(graded, 'lumped')
            dx = 1 / 1024
            sim.add_port((7*dx, 4*dx, 2*dx), 'ez', impedance=75., excite=False)
            results.append(sim.forward(n_steps=32, skip_preflight=True, checkpoint=False,
                                       port_s11_freqs=BINS))
    uniform, graded = results
    assert uniform.wire_port_sparams is graded.wire_port_sparams is None
    assert len(uniform.lumped_port_sparams) == len(graded.lumped_port_sparams) == 2
    for (us, ua), (gs, ga) in zip(uniform.lumped_port_sparams, graded.lumped_port_sparams, strict=True):
        assert type(us) is type(gs) is LumpedPortSParamSpec
        for name in us._fields:
            a, b = np.asarray(getattr(us, name)), np.asarray(getattr(gs, name))
            assert a.dtype == b.dtype and a.shape == b.shape
            assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)
            assert a.tobytes() == b.tobytes()
        assert len(ua) == len(ga) == 3
        for a, b in zip(ua, ga, strict=True):
            a, b = np.asarray(a), np.asarray(b)
            assert a.dtype == b.dtype == np.complex64
            assert a.shape == b.shape == (len(BINS),)
            assert np.array_equal(a, b, equal_nan=True)
            assert a.tobytes() == b.tobytes()
    assert not np.array_equal(graded.lumped_port_sparams[0][1][0],
                              graded.lumped_port_sparams[0][1][2])


@pytest.mark.parametrize('graded,distributed', [(False, False), (True, False), (True, True)])
def test_t4_mixed_partition_and_declaration_order(graded, distributed):
    devices = jax.devices('cpu')[:2] if distributed else None
    if distributed and len(devices) < 2:
        pytest.skip('requires two CPU devices')
    with enable_x64(False):
        sim = constant_scene(graded, 'wire')
        dx = 1 / 1024
        # Interleave kinds; declaration order within each kind must survive.
        sim.add_port((7*dx, 4*dx, 2*dx), 'ez', impedance=65., excite=False)
        sim.add_port((5*dx, 3*dx, 2*dx), 'ez', impedance=75., extent=2*dx, excite=False)
        sim.add_port((6*dx, 4*dx, 2*dx), 'ez', impedance=85., excite=False)
        result = sim.forward(n_steps=32, skip_preflight=True, checkpoint=False,
                             port_s11_freqs=BINS, distributed=distributed, devices=devices)
    assert len(result.wire_port_sparams or ()) == 2
    assert len(result.lumped_port_sparams or ()) == 2
    assert [s[4] if graded else s.impedance for s, _ in result.wire_port_sparams] == [50., 75.]
    assert [s.impedance for s, _ in result.lumped_port_sparams] == [65., 85.]
    assert all(len(a) == 5 for _, a in result.wire_port_sparams)
    assert all(type(s) is LumpedPortSParamSpec and len(a) == 3 for s, a in result.lumped_port_sparams)

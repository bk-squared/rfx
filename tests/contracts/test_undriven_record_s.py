"""S read from one run's record needs something that excites it.

Two 50 ohm lumped ports, both excite=False, in a small CPML box. With nothing
else exciting the structure the record has no incident wave, and forward()
returned S = 0 at every bin. It now refuses; a plain source that lights the
passive ports keeps the termination diagnostic (#1420). run() on one device
drives each port in turn on either mesh, so it is not a record route and forms S.
"""
import warnings

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec

FREQS = np.linspace(3e9, 9e9, 7)
UNDRIVEN = "needs a driven port"


def _box(mesh, *, source=False, wire=False):
    kw = ({'dx_profile': np.r_[np.full(6, 1e-3), np.full(4, 0.6e-3), np.full(6, 1e-3)]}
          if mesh == 'graded' else {})
    sim = Simulation(freq_max=12e9, domain=(16e-3 if mesh == 'uniform' else 0, 12e-3, 10e-3),
                     dx=1e-3, boundary='cpml', cpml_layers=6, **kw)
    extent = 2e-3 if wire else None
    for x in (5e-3, 11e-3):
        sim.add_port(position=(x, 6e-3, 4e-3), component='ez', impedance=50.0,
                     extent=extent, excite=False)
    if source:
        sim.add_source((8e-3, 4e-3, 5e-3), 'ez', waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
    return sim


def _call(sim, entry):
    if entry == 'forward':
        return sim.forward(n_steps=200, skip_preflight=True, port_s11_freqs=FREQS).s_params
    return sim.run(n_steps=200, skip_preflight=True, compute_s_params=True,
                   s_param_freqs=FREQS).s_params


@pytest.mark.parametrize('mesh,entry,wire', [
    ('uniform', 'forward', False), ('graded', 'forward', False), ('graded', 'forward', True)])
def test_undriven_record_refuses(mesh, entry, wire):
    with pytest.raises(ValueError, match=UNDRIVEN):
        _call(_box(mesh, wire=wire), entry)


@pytest.mark.parametrize('mesh,entry', [('uniform', 'forward'), ('graded', 'forward')])
def test_passive_ports_lit_by_a_plain_source_keep_the_diagnostic(mesh, entry):
    s = np.asarray(_call(_box(mesh, source=True), entry))
    assert s.shape == (2, 2, FREQS.size) and np.isfinite(s).all() and np.abs(s).max() > 0


@pytest.mark.parametrize('mesh,wire', [('uniform', False), ('graded', False), ('graded', True)])
def test_run_drives_each_port(mesh, wire):
    s = np.asarray(_call(_box(mesh, wire=wire), 'run'))
    assert s.shape == (2, 2, FREQS.size) and np.isfinite(s).all() and np.abs(s).max() > 0


def test_default_wire_request_warns_and_returns_no_s():
    sim = Simulation(freq_max=12e9, domain=(16e-3, 12e-3, 10e-3), dx=1e-3,
                     boundary='cpml', cpml_layers=6)
    sim.add_port(position=(5e-3, 6e-3, 4e-3), component='ez', impedance=50.0,
                 extent=2e-3, excite=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sim.run(n_steps=200, skip_preflight=True)
    assert result.s_params is None
    assert any(UNDRIVEN in str(w.message) for w in caught)


def test_a_floquet_port_counts_as_a_drive():
    """A Floquet port injects a plane wave: the default run keeps its S."""
    sim = Simulation(freq_max=12e9, domain=(12e-3, 12e-3, 30e-3), dx=1e-3, cpml_layers=6,
                     boundary=BoundarySpec(x='periodic', y='periodic', z='cpml'))
    sim.add_port(position=(6e-3, 6e-3, 12e-3), component='ex', impedance=50.0,
                 extent=2e-3, excite=False)
    sim.add_floquet_port(20e-3, axis='z', f0=6e9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sim.run(n_steps=200, skip_preflight=True)
    assert not any(UNDRIVEN in str(w.message) for w in caught)
    assert result.s_params is not None and np.abs(np.asarray(result.s_params)).max() > 0

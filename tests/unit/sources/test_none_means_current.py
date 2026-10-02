"""Declaration defaults and ADI field-only admission (#1373)."""
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.api._source_semantics import resolve_amplitude_kind


def simulation(**kwargs):
    return Simulation(freq_max=1e10, domain=(.01, .01, .01), dx=.001,
                      boundary='pec', **kwargs)


def test_declaration_resolves_once_and_warns_once():
    sim = simulation()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        sim.add_source((.005, .005, .005))
        sim.add_source((.006, .005, .005), amplitude_kind=None)
    assert [p.amplitude_kind for p in sim._ports] == ['current', 'current']
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert "now means 'current'" in str(caught[0].message)
    assert 'on every path' in str(caught[0].message)
    assert resolve_amplitude_kind('field') == 'field'
    assert resolve_amplitude_kind('current') == 'current'
    with pytest.raises(ValueError):
        sim.add_source((.005, .005, .005), amplitude_kind='bad')
    assert len(sim._ports) == 2


def test_config_interop_and_polarized_declarations_resolve():
    from rfx.config.loader import simulation_from_dict
    from rfx.interop import design_to_dict, simulation_from_design

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        sim = simulation_from_dict({
            'frequency': {'freq_max': 1e10}, 'domain': [.01, .01, .01],
            'dx': .001, 'boundary': 'pec',
            'sources': [{'type': 'source', 'position': [.005, .005, .005]}],
        })
        assert sim._ports[0].amplitude_kind == 'current'
        document = design_to_dict(sim)
        document['excitations']['soft_sources'][0]['amplitude_kind'] = None
        restored = simulation_from_design(document)
        assert restored._ports[0].amplitude_kind == 'current'
        assert document['excitations']['soft_sources'][0]['amplitude_kind'] is None
        polarized = simulation()
        polarized.add_polarized_source((.005, .005, .005), polarization='slant45')
        assert [p.amplitude_kind for p in polarized._ports] == ['current', 'current']


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_adi_requires_explicit_field_before_stepping(mode, entry, monkeypatch):
    import rfx.adi as adi
    import rfx.api._execute as execute

    def make(kind):
        sim = Simulation(freq_max=1e10,
                         domain=(.01, .01, .01 if mode == "3d" else .001),
                         dx=.001, boundary="pec", mode=mode, solver="adi")
        z = .005 if mode == "3d" else 0.
        kwargs = {} if kind is None else {"amplitude_kind": kind}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim.add_source((.005, .005, z), "ez", waveform=lambda t: 1., **kwargs)
        sim.add_probe((.005, .005, z), "ez")
        return sim

    def premature_step(*args, **kwargs):
        pytest.fail("ADI entered stepping before source admission")

    for kind in (None, "current"):
        sim = make(kind)
        with monkeypatch.context() as patch:
            patch.setattr(adi, "run_adi_3d", premature_step)
            patch.setattr(adi, "run_adi_2d", premature_step)
            patch.setattr(execute, "run_adi_2d", premature_step)
            with pytest.raises(NotImplementedError) as caught:
                getattr(sim, entry)(n_steps=4, skip_preflight=True)
        assert ("ADI implements only amplitude_kind='field'; 'current' is the default "
                "when amplitude_kind is not given (2.0); declare amplitude_kind='field' "
                "to run on ADI (the earlier ADI behaviour).") in str(caught.value)
    result = getattr(make("field"), entry)(n_steps=4, skip_preflight=True)
    assert np.isfinite(result.time_series).all()
    assert np.max(np.abs(result.time_series)) > 0

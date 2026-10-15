"""A run with conformal PEC says that its model enlarges a metal body (#1651)."""

import warnings

from rfx.api import Simulation
from rfx.geometry.csg import Sphere


def _sim():
    sim = Simulation(freq_max=10e9, domain=(0.012,) * 3, dx=0.0015,
                     boundary="cpml", cpml_layers=4)
    sim.add(Sphere(center=(0.006,) * 3, radius=0.002), material="pec")
    sim.add_source((0.003, 0.006, 0.006), component="ez", amplitude_kind="current")
    sim.add_probe((0.009, 0.006, 0.006), component="ez")
    return sim


def _advisories(conformal):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sim().run(n_steps=4, conformal_pec=conformal, compute_s_params=False,
                   skip_preflight=True)
    return [w for w in caught if "#1651" in str(w.message)]


def test_conformal_run_warns_once_and_names_the_limitation():
    found = _advisories(True)
    assert len(found) == 1
    text = str(found[0].message)
    assert "eps/w" in text and "known_limitations" in text


def test_staircase_run_does_not_warn():
    assert _advisories(False) == []

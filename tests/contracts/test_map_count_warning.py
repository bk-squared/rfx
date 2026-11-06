"""Low host mapping limits warn once before time stepping."""

import warnings

import pytest

from rfx import Simulation
from rfx import _host_limits as limits


@pytest.fixture(autouse=True)
def reset_check(monkeypatch):
    monkeypatch.setattr(limits, "_checked", False)


def test_low_map_count_warns_once(monkeypatch):
    monkeypatch.setattr(limits, "_read_map_count", lambda: 65530)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limits.check_map_count()
    assert len(caught) == 1
    assert caught[0].category is limits.MapCountWarning
    text = str(caught[0].message)
    for expected in ("65530", "262,144", "262144", "sysctl", "#2421"):
        assert expected in text
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limits.check_map_count()
    assert caught == []


def test_advised_map_count_is_silent(monkeypatch):
    monkeypatch.setattr(limits, "_read_map_count", lambda: 262144)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limits.check_map_count()
        monkeypatch.setattr(limits, "_read_map_count", lambda: 65530)
        limits.check_map_count()
    assert caught == []


def test_unknown_map_count_is_silent(monkeypatch):
    monkeypatch.setattr(limits, "_read_map_count", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limits.check_map_count()
        monkeypatch.setattr(limits, "_read_map_count", lambda: 65530)
        limits.check_map_count()
    assert caught == []


def test_public_run_warns_once(monkeypatch):
    monkeypatch.setattr(limits, "_read_map_count", lambda: 65530)
    sim = Simulation(freq_max=10e9, domain=(.006, .006, .006), dx=.001,
                     boundary="pec", cpml_layers=2)
    sim.add_source((.003, .003, .003), "ez", amplitude_kind="current")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=2, compute_s_params=False, skip_preflight=True)
    assert len([w for w in caught if w.category is limits.MapCountWarning]) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=2, compute_s_params=False, skip_preflight=True)
    assert [w for w in caught if w.category is limits.MapCountWarning] == []


def test_public_until_decay_warns_once(monkeypatch):
    monkeypatch.setattr(limits, "_read_map_count", lambda: 65530)
    sim = Simulation(freq_max=10e9, domain=(.006, .006, .006), dx=.001,
                     boundary="cpml", cpml_layers=2)
    sim.add_source((.003, .003, .003), "ez", amplitude_kind="current")
    kwargs = dict(n_steps=2, until_decay=.001, decay_min_steps=0,
                  decay_max_steps=2, decay_check_interval=1,
                  compute_s_params=False, skip_preflight=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(**kwargs)
    assert len([w for w in caught if w.category is limits.MapCountWarning]) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(**kwargs)
    assert [w for w in caught if w.category is limits.MapCountWarning] == []


@pytest.mark.parametrize("platform, contents, expected", [
    ("darwin", "65530\n", None),
    ("linux", "not an integer\n", None),
    ("linux", "65530\n", 65530),
])
def test_read_map_count(monkeypatch, platform, contents, expected):
    monkeypatch.setattr(limits.sys, "platform", platform)
    monkeypatch.setattr(limits.Path, "read_text", lambda self: contents)
    assert limits._read_map_count() == expected

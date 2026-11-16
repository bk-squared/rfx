"""Omitted waveguide source centers agree across builders and preflight."""

import re
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.preflight.waveguide import (
    resolve_shared_waveguide_port_freqs,
    resolve_waveguide_port_freqs,
)
from rfx.runners.nonuniform import _build_waveguide_port_config_nu


def _scene(graded, *, n_modes=1, freq_max=12.4e9, freqs=None):
    sim = Simulation(
        freq_max=freq_max, domain=(.1016, .02286, .01016), dx=.00127,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8,
        dz_profile=np.array([.00127] * 4 + [.000635] * 8) if graded else None,
    )
    sim.add_waveguide_port(
        .02, direction="+x", name="launch", n_modes=n_modes, freqs=freqs,
        ref_offset=2, probe_offset=4,
    )
    return sim


def _configs(sim, graded, freqs):
    entry = sim._waveguide_ports[0]
    assert entry.f0 is None
    if graded:
        cfg = _build_waveguide_port_config_nu(
            sim, entry, sim._build_nonuniform_grid(), freqs, n_steps=8,
        )
    else:
        cfg = sim._build_waveguide_port_config(
            entry, sim._build_grid(), freqs, n_steps=8,
        )
    return cfg if isinstance(cfg, list) else [cfg]


@pytest.mark.parametrize("n_modes", [1, 2], ids=["single", "multimode"])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_omitted_f0_is_measurement_band_center(graded, n_modes):
    freqs = np.linspace(8.2e9, 12.4e9, 7)
    sim = _scene(graded, n_modes=n_modes, freqs=freqs)
    cfgs = _configs(sim, graded, freqs)
    assert len(cfgs) == n_modes
    centers = [cfg.src_fcen for cfg in cfgs]
    assert centers == [10.3e9] * n_modes
    other = _scene(not graded, n_modes=n_modes, freqs=freqs)
    assert centers == [cfg.src_fcen for cfg in _configs(other, not graded, freqs)]


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_preflight_judges_built_center_when_port_freqs_omitted(graded):
    # Keep the resolved center below cutoff so the diagnostic states its value.
    sim = _scene(graded, freq_max=10e9)
    entry = sim._waveguide_ports[0]
    assert entry.freqs is None
    freqs = (resolve_shared_waveguide_port_freqs(sim, route="graded config")
             if graded else resolve_waveguide_port_freqs(sim, entry))
    cfg, = _configs(sim, graded, freqs)
    assert cfg.src_fcen == 5.5e9
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim._check_waveguide_port_evanescent()
    source_findings = [
        str(w.message) for w in caught
        if getattr(w.message, "code", None) == "port_source_below_cutoff"
    ]
    assert len(source_findings) == 1
    match = re.search(r"f0=([\d.]+) GHz", source_findings[0])
    assert match is not None
    assert float(match.group(1)) * 1e9 == cfg.src_fcen


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_unresolved_pulse_judges_built_waveguide_center(graded, monkeypatch):
    import rfx.sources.sources as sources

    freqs = np.linspace(8.2e9, 12.4e9, 7)
    sim = _scene(graded, freqs=freqs)
    recorded = []
    modulated_gaussian = sources.ModulatedGaussian

    def record_pulse(*args, **kwargs):
        recorded.append(kwargs["f0"])
        return modulated_gaussian(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sources, "ModulatedGaussian", record_pulse)
        sim._validate_cfg_unresolved_pulse(warnings, sim._dx)

    cfg, = _configs(sim, graded, freqs)
    f0, = recorded
    assert f0 == cfg.src_fcen == 10.3e9

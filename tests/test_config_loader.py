"""Regression tests for the config-driven loader / CLI.

The core gate is *construction equivalence*: a Simulation built from a YAML
config must match one built by calling the builder API directly with the
same parameters (same domain, grid shape, material/port/probe counts, and
material values). Construction equivalence is fast and deterministic; a full
FDTD run is gated behind ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.config import (
    execution_to_run_kwargs,
    shape_from_config,
    simulation_from_dict,
    simulation_from_yaml,
    waveform_from_config,
)
from rfx.config._waveforms import WAVEFORM_REGISTRY

# A small microstrip-thru config (mirrors examples/config/microstrip_thru.yaml
# but smaller / shorter so construction is instant).
_CFG = {
    "frequency": {"freq_max": 12e9},
    "domain": {"x": 0.020, "y": 0.012, "z": 0.006},
    "boundary": "cpml",
    "cpml_layers": 8,
    "dx": 0.0005,
    "materials": {"fr4": {"eps_r": 4.4, "sigma": 0.02}},
    "geometry": [
        {"shape": "box",
         "bounds": [[0.0, 0.0, 0.0015], [0.020, 0.012, 0.002]],
         "material": "pec"},
        {"shape": "box",
         "bounds": [[0.0, 0.0, 0.002], [0.020, 0.012, 0.003]],
         "material": "fr4"},
        {"shape": "box",
         "bounds": [[0.006, 0.0055, 0.003], [0.014, 0.0065, 0.0035]],
         "material": "pec"},
    ],
    "sources": [
        {"type": "port", "position": [0.006, 0.006, 0.002], "component": "ez",
         "impedance": 50.0, "extent": 0.001,
         "waveform": {"type": "gaussian_pulse", "f0": 6e9,
                      "bandwidth": 0.9, "amplitude": 1.0}},
        {"type": "port", "position": [0.014, 0.006, 0.002], "component": "ez",
         "impedance": 50.0, "extent": 0.001, "excite": False},
    ],
    "probes": [{"position": [0.010, 0.006, 0.0025], "component": "ez"}],
    "execution": {
        "n_steps": 1200, "compute_s_params": True,
        "s_param_freq_start": 2e9, "s_param_freq_end": 10e9,
        "s_param_n_freqs": 81, "s_param_n_steps": 1200,
    },
}


def _build_direct() -> Simulation:
    """Build the equivalent Simulation via the builder API directly."""
    sim = Simulation(
        freq_max=12e9,
        domain=(0.020, 0.012, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.0005,
    )
    sim.add_material("fr4", eps_r=4.4, sigma=0.02)
    sim.add(Box((0.0, 0.0, 0.0015), (0.020, 0.012, 0.002)), material="pec")
    sim.add(Box((0.0, 0.0, 0.002), (0.020, 0.012, 0.003)), material="fr4")
    sim.add(Box((0.006, 0.0055, 0.003), (0.014, 0.0065, 0.0035)), material="pec")
    sim.add_port(
        (0.006, 0.006, 0.002), "ez", impedance=50.0, extent=0.001,
        waveform=GaussianPulse(f0=6e9, bandwidth=0.9, amplitude=1.0),
    )
    sim.add_port(
        (0.014, 0.006, 0.002), "ez", impedance=50.0, extent=0.001, excite=False,
    )
    sim.add_probe((0.010, 0.006, 0.0025), "ez")
    return sim


# --------------------------------------------------------------------------
# Construction equivalence — the primary gate.
# --------------------------------------------------------------------------

def test_dict_vs_direct_equivalence():
    sim_cfg = simulation_from_dict(_CFG)
    sim_dir = _build_direct()

    # Top-level scalars
    assert sim_cfg._freq_max == sim_dir._freq_max
    assert sim_cfg._domain == sim_dir._domain
    assert sim_cfg._boundary == sim_dir._boundary
    assert sim_cfg._cpml_layers == sim_dir._cpml_layers
    assert sim_cfg._dx == sim_dir._dx

    # Counts
    assert set(sim_cfg._materials) == set(sim_dir._materials)
    assert len(sim_cfg._geometry) == len(sim_dir._geometry)
    assert len(sim_cfg._ports) == len(sim_dir._ports)
    assert len(sim_cfg._probes) == len(sim_dir._probes)

    # Material values match
    for name in sim_dir._materials:
        m_cfg, m_dir = sim_cfg._materials[name], sim_dir._materials[name]
        assert m_cfg.eps_r == m_dir.eps_r
        assert m_cfg.sigma == m_dir.sigma
        assert m_cfg.mu_r == m_dir.mu_r

    # Geometry bounding boxes + material names match in order
    for g_cfg, g_dir in zip(sim_cfg._geometry, sim_dir._geometry):
        assert g_cfg.material_name == g_dir.material_name
        assert g_cfg.shape.bounding_box() == g_dir.shape.bounding_box()

    # Port positions / components / impedance / excite match in order
    for p_cfg, p_dir in zip(sim_cfg._ports, sim_dir._ports):
        assert p_cfg.position == p_dir.position
        assert p_cfg.component == p_dir.component
        assert p_cfg.impedance == p_dir.impedance
        assert p_cfg.extent == p_dir.extent
        assert p_cfg.excite == p_dir.excite

    # Grid shape is identical
    assert sim_cfg._build_grid().shape == sim_dir._build_grid().shape


def test_yaml_matches_dict(tmp_path):
    import yaml

    yaml_path = tmp_path / "sim.yaml"
    yaml_path.write_text(yaml.safe_dump(_CFG))
    sim_yaml = simulation_from_yaml(yaml_path)
    sim_dict = simulation_from_dict(_CFG)

    assert sim_yaml._domain == sim_dict._domain
    assert sim_yaml._build_grid().shape == sim_dict._build_grid().shape
    assert len(sim_yaml._ports) == len(sim_dict._ports)
    assert len(sim_yaml._probes) == len(sim_dict._probes)
    assert set(sim_yaml._materials) == set(sim_dict._materials)


def test_example_yaml_loads():
    """The shipped example config must build and produce a sane grid."""
    from pathlib import Path

    example = (
        Path(__file__).resolve().parents[1]
        / "examples" / "config" / "microstrip_thru.yaml"
    )
    sim = simulation_from_yaml(example)
    shape = sim._build_grid().shape
    assert all(n > 0 for n in shape)
    assert len(sim._ports) == 2
    assert len(sim._probes) == 1


# --------------------------------------------------------------------------
# execution block -> run kwargs
# --------------------------------------------------------------------------

def test_execution_to_run_kwargs():
    rk = execution_to_run_kwargs(_CFG["execution"])
    assert rk["n_steps"] == 1200
    assert rk["compute_s_params"] is True
    assert rk["s_param_n_steps"] == 1200
    freqs = rk["s_param_freqs"]
    assert isinstance(freqs, np.ndarray)
    assert freqs.shape == (81,)
    assert freqs[0] == pytest.approx(2e9)
    assert freqs[-1] == pytest.approx(10e9)


def test_execution_partial_freq_sweep_errors():
    with pytest.raises(KeyError):
        execution_to_run_kwargs({"s_param_freq_start": 1e9})  # missing end / n


# --------------------------------------------------------------------------
# Sub-builders + loud failure on unsupported / malformed config.
# --------------------------------------------------------------------------

def test_waveform_registry_and_builder():
    assert set(WAVEFORM_REGISTRY) == {"gaussian_pulse", "modulated_gaussian"}
    wf = waveform_from_config(
        {"type": "gaussian_pulse", "f0": 5e9, "bandwidth": 0.8}
    )
    assert isinstance(wf, GaussianPulse)
    assert wf.f0 == 5e9
    assert wf.bandwidth == 0.8


def test_waveform_unknown_type_raises():
    with pytest.raises(NotImplementedError):
        waveform_from_config({"type": "sawtooth", "f0": 1e9})


def test_waveform_gaussian_pulse_accepts_cutoff():
    """The #388 DC-floor warning's primary remedy (cutoff=4.5) must be
    reachable from the config/YAML lane for both pulse families."""
    wf = waveform_from_config(
        {"type": "gaussian_pulse", "f0": 2.2e9, "bandwidth": 1.2,
         "cutoff": 4.5}
    )
    assert isinstance(wf, GaussianPulse)
    assert wf.cutoff == 4.5
    # Default stays at the byte-identical historical value.
    wf_default = waveform_from_config(
        {"type": "gaussian_pulse", "f0": 2.2e9, "bandwidth": 1.2}
    )
    assert wf_default.cutoff == 3.0


def test_shape_box_and_unsupported():
    box = shape_from_config(
        {"shape": "box", "bounds": [[0, 0, 0], [1, 2, 3]]}
    )
    assert isinstance(box, Box)
    assert box.bounding_box() == ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    with pytest.raises(NotImplementedError):
        shape_from_config({"shape": "sphere", "radius": 1.0})


def test_unknown_top_level_key_raises():
    bad = dict(_CFG)
    bad["boundry"] = "cpml"  # typo
    with pytest.raises(KeyError):
        simulation_from_dict(bad)


def test_frequency_unknown_key_raises():
    bad = dict(_CFG)
    bad["frequency"] = {"freq_max": 12e9, "freq_min": 1e9}  # freq_min not accepted
    with pytest.raises(KeyError):
        simulation_from_dict(bad)


def test_domain_unknown_key_raises():
    bad = dict(_CFG)
    bad["domain"] = {"x": 0.020, "y": 0.012, "z": 0.006, "w": 0.001}  # typo axis
    with pytest.raises(KeyError):
        simulation_from_dict(bad)


def test_domain_missing_axis_raises():
    bad = dict(_CFG)
    bad["domain"] = {"x": 0.020, "z": 0.006}  # missing 'y'
    with pytest.raises(KeyError):
        simulation_from_dict(bad)


def test_deferred_waveguide_port_raises():
    bad = {
        "frequency": {"freq_max": 1e10},
        "domain": {"x": 0.01, "y": 0.01, "z": 0.01},
        "dx": 0.001,
        "sources": [{"type": "waveguide", "position": [0, 0, 0]}],
    }
    with pytest.raises(NotImplementedError):
        simulation_from_dict(bad)


def test_missing_required_key_raises():
    with pytest.raises(KeyError):
        simulation_from_dict({"domain": {"x": 0.01, "y": 0.01, "z": 0.01}})


# --------------------------------------------------------------------------
# Full run — slow, opt-in. Confirms YAML run == direct-build run.
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_full_run_matches_direct():
    run_kwargs = execution_to_run_kwargs(_CFG["execution"])
    sim_cfg = simulation_from_dict(_CFG)
    sim_dir = _build_direct()
    res_cfg = sim_cfg.run(**run_kwargs)
    res_dir = sim_dir.run(**run_kwargs)
    assert res_cfg.s_params.shape == res_dir.s_params.shape
    np.testing.assert_allclose(
        np.asarray(res_cfg.s_params), np.asarray(res_dir.s_params),
        rtol=1e-6, atol=1e-8,
    )

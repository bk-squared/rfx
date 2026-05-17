from __future__ import annotations

import warnings

import numpy as np
import jax.numpy as jnp

from rfx import GaussianPulse, Simulation


def _research_port_subgrid_sim(*, passive_waveform: bool = False) -> Simulation:
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002, cpml_layers=0)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="research")
    sim._refinement["use_boundary_terminated_exterior_z_interfaces"] = True
    sim.add_port(
        (0.04 / 3.0, 0.04 / 3.0, 0.002 + 0.45 * 0.022),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3.5e9, bandwidth=0.8),
        excite=True,
    )
    sim.add_port(
        (2.0 * 0.04 / 3.0, 2.0 * 0.04 / 3.0, 0.002 + 0.55 * 0.022),
        "ez",
        impedance=50.0,
        waveform=(
            GaussianPulse(f0=3.5e9, bandwidth=0.8)
            if passive_waveform else None
        ),
        excite=False,
    )
    sim.add_probe((0.02, 0.02, 0.002 + 0.55 * 0.022), "ez")
    return sim


def test_research_subgrid_impedance_ports_do_not_crash_when_no_pec_mask():
    sim = _research_port_subgrid_sim()

    result = sim.run(n_steps=20, compute_s_params=False)

    assert result.s_params is None
    assert result.time_series.shape == (20, 1)
    assert np.all(np.isfinite(np.asarray(result.time_series)))


def test_production_subgrid_public_lumped_sparameter_path_populates_matrix():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002, cpml_layers=0)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    sim.add_port(
        (0.04 / 3.0, 0.04 / 3.0, 0.002 + 0.45 * 0.022),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3.5e9, bandwidth=0.8),
    )
    sim.add_probe((0.02, 0.02, 0.002 + 0.55 * 0.022), "ez")
    sim.add_port(
        (2.0 * 0.04 / 3.0, 2.0 * 0.04 / 3.0, 0.002 + 0.55 * 0.022),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3.5e9, bandwidth=0.8),
        excite=False,
    )
    freqs = jnp.asarray([3.0e9, 3.5e9], dtype=jnp.float32)

    result = sim.run(
        n_steps=10,
        compute_s_params=True,
        s_param_freqs=freqs,
        s_param_n_steps=30,
    )

    assert result.freqs is not None
    assert result.s_params is not None
    assert result.s_params.shape == (2, 2, 2)
    assert np.all(np.isfinite(np.asarray(result.s_params)))


def test_production_subgrid_public_wire_sparameter_path_populates_matrix():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002, cpml_layers=0)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="production")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="waveform is ignored when excite=False")
        sim.add_port(
            (0.04 / 3.0, 0.04 / 3.0, 0.002 + 0.40 * 0.022),
            "ez",
            impedance=50.0,
            extent=0.004,
            waveform=GaussianPulse(f0=3.5e9, bandwidth=0.8),
        )
        sim.add_port(
            (2.0 * 0.04 / 3.0, 2.0 * 0.04 / 3.0, 0.002 + 0.50 * 0.022),
            "ez",
            impedance=50.0,
            extent=0.004,
            waveform=GaussianPulse(f0=3.5e9, bandwidth=0.8),
            excite=False,
        )
    sim.add_probe((0.02, 0.02, 0.002 + 0.55 * 0.022), "ez")
    freqs = jnp.asarray([3.0e9, 3.5e9], dtype=jnp.float32)

    report = sim.validate_subgrid()
    result = sim.run(
        n_steps=10,
        compute_s_params=True,
        s_param_freqs=freqs,
        s_param_n_steps=30,
    )

    assert report.supported, report.format()
    assert result.freqs is not None
    assert result.s_params is not None
    assert result.s_params.shape == (2, 2, 2)
    assert np.all(np.isfinite(np.asarray(result.s_params)))


def test_production_subgrid_sparameters_reject_unguarded_centered_slab():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=0.002, cpml_layers=0)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2, validation="production")
    sim.add_port(
        (0.02, 0.02, 0.020),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3.5e9, bandwidth=0.8),
    )
    sim.add_probe((0.024, 0.024, 0.020), "ez")

    try:
        sim.run(n_steps=5, compute_s_params=True, s_param_n_steps=20)
    except ValueError as exc:
        assert "impedance_port_unvalidated" in str(exc)
    else:  # pragma: no cover - must not silently claim broad port support
        raise AssertionError("unguarded production subgrid S-parameter request unexpectedly succeeded")


def test_research_subgrid_public_lumped_sparameter_path_populates_matrix():
    sim = _research_port_subgrid_sim(passive_waveform=True)
    freqs = jnp.asarray([3.0e9, 3.5e9], dtype=jnp.float32)

    result = sim.run(
        n_steps=10,
        compute_s_params=True,
        s_param_freqs=freqs,
        s_param_n_steps=30,
    )

    assert result.freqs is not None
    assert result.s_params is not None
    assert result.s_params.shape == (2, 2, 2)
    assert np.all(np.isfinite(np.asarray(result.s_params)))
    assert np.any(np.abs(np.asarray(result.s_params[:, 0, :])) > 0.0)
    assert np.any(np.abs(np.asarray(result.s_params[:, 1, :])) > 0.0)


def test_research_subgrid_private_lumped_sparameter_replay_populates_column():
    sim = _research_port_subgrid_sim()
    freqs = jnp.asarray([3.0e9, 3.5e9], dtype=jnp.float32)
    sim._refinement["diagnostic_lumped_sparam_freqs"] = freqs
    sim._refinement["diagnostic_lumped_sparam_driven_index"] = 0

    result = sim.run(n_steps=30, compute_s_params=False)

    assert result.freqs is not None
    assert result.s_params is not None
    assert result.s_params.shape == (2, 2, 2)
    assert np.all(np.isfinite(np.asarray(result.s_params)))
    assert np.any(np.abs(np.asarray(result.s_params[:, 0, :])) > 0.0)
    assert np.allclose(np.asarray(result.s_params[:, 1, :]), 0.0)

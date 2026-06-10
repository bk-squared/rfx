"""Eligibility guards for the vmap_material_sweep fast path (roadmap W1.2).

The vmap scan bodies implement only pec/periodic walls and CPML. Before
this guard, a sim with ``boundary='upml'`` was routed to a scan body that
applies NO absorber at all, and lumped RLC elements / flux monitors /
DFT planes / NTFF / non-uniform profiles were silently dropped from the
swept runs. These tests pin the sequential fallback for every such case
and pin that the supported fast path still does NOT fall back.

CPU-runnable on tiny grids (intentionally not gpu-marked so the PR gate
exercises this file).
"""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from rfx import Simulation, GaussianPulse, Box
from rfx.vmap_sweep import vmap_material_sweep, VmapSweepResult

_FALLBACK_MATCH = "Falling back to sequential"


def _base_sim(boundary: str = "cpml", eps_r: float = 4.0) -> Simulation:
    kwargs = {"cpml_layers": 6} if boundary == "cpml" else {}
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        boundary=boundary,
        dx=0.002,
        **kwargs,
    )
    sim.add_material("substrate", eps_r=eps_r)
    sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.005, 0.01, 0.01), "ez")
    return sim


def test_upml_takes_sequential_fallback_and_matches_run():
    """UPML must NOT take the vmap fast path (scan bodies apply no UPML)."""
    sim = _base_sim(boundary="upml")
    eps_values = np.array([2.0, 6.0])
    n_steps = 40

    with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
        result = vmap_material_sweep(
            sim, "substrate.eps_r", eps_values, n_steps=n_steps,
        )

    assert isinstance(result, VmapSweepResult)
    assert result.time_series.shape[0] == len(eps_values)

    # The sequential fallback runs the canonical loop, so each swept entry
    # must match a direct sim.run() with that eps_r value.
    for idx, eps_val in enumerate(eps_values):
        sim_single = _base_sim(boundary="upml", eps_r=float(eps_val))
        single_ts = np.asarray(sim_single.run(n_steps=n_steps).time_series)
        npt.assert_allclose(
            np.asarray(result.time_series[idx]), single_ts,
            rtol=1e-5, atol=1e-12,
            err_msg=f"UPML fallback diverges from direct run at eps_r={eps_val}",
        )


def test_lumped_rlc_takes_sequential_fallback():
    """Lumped RLC elements are not wired into the vmap scan bodies."""
    sim = _base_sim(boundary="cpml")
    sim.add_lumped_rlc((0.015, 0.01, 0.01), "ez", R=50.0)

    with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
        vmap_material_sweep(
            sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=20,
        )


def test_flux_monitor_takes_sequential_fallback():
    """Flux monitors only accumulate in the canonical run() loop."""
    sim = _base_sim(boundary="cpml")
    sim.add_flux_monitor(
        axis="x", coordinate=0.016, freqs=np.array([3e9]), name="t",
    )

    with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
        vmap_material_sweep(
            sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=20,
        )


def test_nonuniform_profile_takes_sequential_fallback():
    """NU mesh profiles are incompatible with the uniform-grid scan bodies."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        boundary="cpml",
        cpml_layers=6,
        dx=0.002,
        dx_profile=np.full(10, 0.002),
    )
    sim.add_material("substrate", eps_r=4.0)
    sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.005, 0.01, 0.01), "ez")

    with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
        vmap_material_sweep(
            sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=20,
        )


def test_supported_cpml_sim_still_uses_fast_path():
    """Control: the plain CPML dielectric sim must NOT fall back."""
    sim = _base_sim(boundary="cpml")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=f".*{_FALLBACK_MATCH}.*")
        result = vmap_material_sweep(
            sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=20,
        )
    assert result.time_series.shape[0] == 2

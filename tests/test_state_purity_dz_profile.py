"""Regression locks for dz-profile state purity (roadmap W1.3).

forward()/run()/the distributed runner used to PERSIST a synthesized dz
profile onto ``sim._dz_profile`` as a side effect of execution, while the
NU S-matrix extractor save/restored the same attribute — hidden state
divergence depending on which entry point ran first. The synthesis now
happens locally inside ``_build_nonuniform_grid()`` and sim state must
stay untouched.
"""

import numpy as np

from rfx import Simulation, GaussianPulse


def _dx_only_nu_sim() -> Simulation:
    """Minimal runnable sim with ONLY dx_profile set (dz left None)."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.01, 0.004),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
        dx_profile=np.full(20, 1e-3),
    )
    sim.add_source((0.01, 0.005, 0.002), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.012, 0.005, 0.002), "ez")
    return sim


def test_build_nonuniform_grid_does_not_persist_dz():
    sim = _dx_only_nu_sim()
    assert sim._dz_profile is None
    grid = sim._build_nonuniform_grid()
    assert grid is not None
    assert sim._dz_profile is None, (
        "_build_nonuniform_grid() persisted the synthesized dz profile "
        "onto sim state — synthesis must stay local to the grid build"
    )


def test_run_does_not_persist_synthesized_dz():
    sim = _dx_only_nu_sim()
    res = sim.run(n_steps=5)
    assert res is not None
    assert sim._dz_profile is None, (
        "run() persisted a synthesized dz profile onto sim state — the "
        "NU dispatch must not mutate the simulation as a side effect"
    )

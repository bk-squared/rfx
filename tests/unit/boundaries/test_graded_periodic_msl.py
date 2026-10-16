"""A periodic MSL feed sees the conductor immediately across its seam."""
import numpy as np
import pytest

from rfx import Box, Simulation


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_msl_seam_geometry_reaches_step_builder(graded, monkeypatch):
    from rfx import nonuniform, simulation

    profile = {"dz_profile": np.array([.001] * 4 + [.0005] * 8 + [.0015] * 12)} if graded else {}
    sim = Simulation(freq_max=5e9, domain=(.024, .024, .026), dx=.001,
                     cpml_layers=0, boundary={"x": "periodic", "y": "cpml", "z": "cpml"},
                     **profile)
    sim.add(Box((0, 0, .004), (.024, .024, .004)), material="pec")
    sim.add_material("substrate", eps_r=3)
    sim.add(Box((0, 0, .004), (.024, .024, .008)), material="substrate")
    sim.add(Box((.016, .010, .008), (.024, .014, .008)), material="pec")
    sim.add_msl_port(position=(0, .012, .004), width=.004, height=.004, direction="-x")

    class StepBuilderReached(Exception):
        pass

    def reached(*args, **kwargs):
        raise StepBuilderReached

    owner, name = ((nonuniform, "_build_nu_scan") if graded
                   else (simulation, "_build_step_setup"))
    monkeypatch.setattr(owner, name, reached)
    with pytest.raises(StepBuilderReached):
        sim.run(n_steps=1, skip_preflight=True)

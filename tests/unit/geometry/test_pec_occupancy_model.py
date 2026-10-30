"""Constructor-owned physics selection and unsupported-path admission."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.errors import RfxRefusal, RfxUnsupported


@pytest.mark.parametrize("model", ["edge_keep", "kottke"])
def test_uniform_design_occupancy_without_pec_override_admission(monkeypatch, model):
    """A design-only box must refuse Kottke before entering the solve."""
    import rfx.simulation as simulation

    class SolveReached(Exception):
        pass

    def solve(*args, **kwargs):
        raise SolveReached

    monkeypatch.setattr(simulation, "run", solve)
    sim = Simulation(freq_max=10e9, domain=(.008,) * 3, dx=.001,
                     boundary="pec", cpml_layers=0, pec_occupancy_model=model)
    expected = RfxUnsupported if model == "kottke" else SolveReached
    match = "kottke.*pec_occupancy_override only, not to a design occupancy box" if model == "kottke" else None
    with pytest.raises(expected, match=match):
        sim.forward(n_steps=1, design_box=((.002,) * 3, (.004,) * 3),
                    design_occupancy_override=jnp.full((3,) * 3, .5),
                    skip_preflight=True, checkpoint=False)


def test_topology_occupancy_reaches_uniform_solve_with_kottke_tensor(monkeypatch):
    """Topology embeds its design in a full-grid occupancy, which is supported."""
    import rfx.simulation as simulation
    from rfx.model.occupancy import occupancy_from_design

    class SolveReached(Exception):
        pass

    def solve(*args, **kwargs):
        assert kwargs["pec_occupancy"] is None
        assert kwargs["aniso_inv_eps"] is not None
        assert kwargs["design_occupancy"] is None
        raise SolveReached

    monkeypatch.setattr(simulation, "run", solve)
    sim = Simulation(freq_max=10e9, domain=(.008,) * 3, dx=.001,
                     boundary="pec", cpml_layers=0, pec_occupancy_model="kottke")
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid)[0]
    occupancy = occupancy_from_design(
        grid.shape, (slice(2, 5),) * 3, jnp.full((3,) * 3, .5))
    with pytest.raises(SolveReached):
        sim._forward_from_materials(grid, materials, None, None,
                                    n_steps=1, pec_occupancy=occupancy)


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_graded_forward_refuses_kottke_occupancy(distributed, axis):
    widths = np.array([.001] * 2 + [.0012] * 4 + [.001] * 2)
    domain = [.008] * 3
    domain["xyz".index(axis)] = float(widths.sum())
    sim = Simulation(freq_max=10e9, domain=tuple(domain), dx=.001,
                     boundary="pec", cpml_layers=0, pec_occupancy_model="kottke",
                     **{f"d{axis}_profile": widths})
    occupancy = jnp.zeros(sim._build_nonuniform_grid().shape, jnp.float32)
    with pytest.raises(RfxUnsupported, match="single-device uniform forward"):
        sim.forward(n_steps=1, pec_occupancy_override=occupancy,
                    distributed=distributed, skip_preflight=True, checkpoint=False)


@pytest.mark.parametrize("value", ["1", "0", "", "false"])
@pytest.mark.parametrize("model", ["edge_keep", "kottke"])
def test_removed_environment_variable_refuses_at_construction(monkeypatch, value, model):
    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", value)
    with pytest.raises(RfxRefusal, match="RFX_PEC_OCC_KOTTKE.*pec_occupancy_model"):
        Simulation(freq_max=10e9, domain=(.008,) * 3, dx=.001,
                   pec_occupancy_model=model)


@pytest.mark.parametrize("model", ["edge_keep", "kottke", "invalid", None])
def test_constructor_validates_and_stores_model(monkeypatch, model):
    monkeypatch.delenv("RFX_PEC_OCC_KOTTKE", raising=False)
    kwargs = dict(freq_max=10e9, domain=(.008,) * 3, dx=.001)
    assert Simulation(**kwargs)._pec_occupancy_model == "edge_keep"
    if model in ("edge_keep", "kottke"):
        assert Simulation(**kwargs, pec_occupancy_model=model)._pec_occupancy_model == model
    else:
        with pytest.raises(RfxRefusal, match="pec_occupancy_model"):
            Simulation(**kwargs, pec_occupancy_model=model)

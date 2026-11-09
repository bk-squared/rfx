"""Wall-loss declarations survive solve copies; incomplete exports refuse them."""
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._declarations import declaration_variant
from rfx.errors import RfxRefusal
from rfx.model.wall_loss import WallLoss


def model():
    sim = Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.0015, boundary="pec")
    slab = Box((0., 0., 0.), (.012, .003, .012))
    sim.add(slab, material="pec")
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=("y_lo", "y_hi"))
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, shape=slab)
    return sim, slab


@pytest.mark.parametrize("route", ["quick-convergence", "variant", "replace", "shallow", "deep", "pickle", "vmap-fallback"])
def test_copy_carries_wall_loss(route, monkeypatch):
    sim, slab = model()
    expected = [WallLoss(5.8e5, 8e9, None, ("y_lo", "y_hi")), WallLoss(5.8e5, 8e9, slab, ())]
    if route == "quick-convergence":
        import rfx.convergence as convergence
        monkeypatch.setattr(convergence, "convergence_study", lambda sim_factory, dx_values, **kw: sim_factory(dx_values[0]))
        clone = convergence.quick_convergence(sim, n_steps=1)
    elif route == "variant":
        clone = declaration_variant(sim)
    elif route == "replace":
        clone = declaration_variant(sim, _probes=[])
    elif route == "shallow":
        clone = copy.copy(sim)
    elif route == "deep":
        clone = copy.deepcopy(sim)
    elif route == "vmap-fallback":
        from rfx.vmap_sweep import _sequential_fallback
        seen = []

        def run(current, **kwargs):
            seen.append(current)
            return SimpleNamespace(time_series=np.zeros((1, 1)), settling_db=0.,
                                   settling_witness=None, dt=1., dft_planes=None)

        monkeypatch.setattr(Simulation, "run", run)
        _sequential_fallback(sim, "eps_r", np.array([2.]), n_steps=1)
        (clone,) = seen
    else:
        clone = pickle.loads(pickle.dumps(sim))
    assert clone._wall_losses == expected
    assert clone._wall_losses[1].shape is clone._geometry[0].shape
    if route not in ("deep", "pickle", "vmap-fallback"):
        assert all(a is b for a, b in zip(clone._wall_losses, sim._wall_losses))
    if route in ("quick-convergence", "variant", "replace"):
        assert clone._wall_losses is not sim._wall_losses


@pytest.mark.parametrize("route", ["design-dict", "design-json", "scene", "sdf", "geometry-json", "experiment", "dataset"])
def test_export_refuses_wall_loss_by_name(route, tmp_path):
    from rfx.artifacts import build_scene_artifact
    from rfx.interop import design_to_dict, design_to_json, UnsupportedDesignFeature
    from rfx.io import export_geometry_json, save_experiment_report, save_simulation_dataset
    from rfx.surrogate import export_geometry_sdf
    sim, _ = model()
    path = tmp_path / "export"
    calls = {
        "design-dict": lambda: design_to_dict(sim),
        "design-json": lambda: design_to_json(sim),
        "scene": lambda: build_scene_artifact(sim),
        "sdf": lambda: export_geometry_sdf(sim),
        "geometry-json": lambda: export_geometry_json(path, sim),
        "experiment": lambda: save_experiment_report(path, sim, None),
        "dataset": lambda: save_simulation_dataset(path, sim, None),
    }
    with pytest.raises((RfxRefusal, UnsupportedDesignFeature), match="add_wall_loss"):
        calls[route]()
    assert not path.exists()

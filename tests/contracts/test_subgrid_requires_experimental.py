"""#1465: production never enters the unstable subgridded runner."""
import warnings

import pytest

from rfx import Simulation
from rfx.subgridding._notice import ExperimentalSubgridWarning, SUBGRID_NOTICE


def _sim(mode=None):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .016), dx=.002,
                     boundary="pec")
    options = {} if mode is None else {"validation": mode}
    sim.add_refinement((0., .012), ratio=2, **options)
    sim.add_source((.004, .004, .004), "ez", amplitude_kind="field")
    sim.add_probe((.004, .004, .006), "ez")
    return sim


@pytest.mark.parametrize("skip", [False, True])
def test_default_refused_before_runner(monkeypatch, skip):
    from rfx.runners import subgridded
    def entered(*args, **kwargs):
        pytest.fail("subgridded runner entered")
    monkeypatch.setattr(subgridded, "run_subgridded_path", entered)
    with pytest.raises(NotImplementedError) as exc:
        _sim().run(n_steps=2, skip_preflight=skip)
    assert str(exc.value) == SUBGRID_NOTICE


def test_sequential_material_sweep_refused_before_runner(monkeypatch):
    from rfx.runners import subgridded
    from rfx.vmap_sweep import vmap_material_sweep
    from tests.unit.nonuniform.test_refinement_refused_on_graded_mesh import _lumped_port_box
    sim = _lumped_port_box(True)
    sim._refinement.pop("validation")  # exercise the default for restored declarations too
    def entered(*args, **kwargs):
        pytest.fail("subgridded runner entered")
    monkeypatch.setattr(subgridded, "run_subgridded_path", entered)
    with pytest.raises(NotImplementedError, match="unstable and unverified"):
        vmap_material_sweep(sim, "diel.eps_r", [3.0], n_steps=2)


@pytest.mark.parametrize("entry", ["_run_subgridded", "run_subgridded_path", "_run_subgridded_once"])
def test_direct_entries_refuse_before_reading_grid(entry):
    from rfx.runners import subgridded
    sim = _sim()
    fn = getattr(sim, entry, None)
    args = (None, None, None, 2)
    if fn is None:
        fn = getattr(subgridded, entry)
        args = (sim, *args)
    with pytest.raises(NotImplementedError, match="unstable and unverified"):
        fn(*args)


@pytest.mark.parametrize("mode", ["research", "off"])
def test_opt_in_runs_with_one_warning(mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _sim(mode).run(n_steps=2, skip_preflight=True, compute_s_params=False)
    notices = [w for w in caught if issubclass(w.category, ExperimentalSubgridWarning)]
    assert len(notices) == 1
    assert str(notices[0].message) == SUBGRID_NOTICE
    assert result.time_series.shape[0] == 2


def test_validation_reports_unsupported():
    from rfx.subgridding.validation import validate_subgrid_setup
    sim = _sim()
    grid = sim._build_grid()
    materials, _, _, pec, *_ = sim._assemble_materials(grid)
    for report in (sim.validate_subgrid(), validate_subgrid_setup(sim, grid, materials, pec)):
        assert not report.supported
        assert report.support_level == "unsupported-unstable-unverified"
        assert any(i.code == "subgrid_unstable_unverified" for i in report.errors)

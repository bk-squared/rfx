"""#1465: production never enters the unstable subgridded runner."""
import warnings
from pathlib import Path

import numpy as np

import pytest

from rfx import Simulation, GaussianPulse
from rfx.subgridding._notice import ExperimentalSubgridWarning, SUBGRID_NOTICE, SUBGRID_WARNING


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
def test_opt_in_runs_with_one_warning(mode, monkeypatch):
    from rfx.subgridding import jit_runner
    original = jit_runner.run_subgridded_jit
    closures = []
    def observe(*args, **kwargs):
        closures.append(kwargs["opts"].use_boundary_terminated_exterior_z_interfaces)
        return original(*args, **kwargs)
    monkeypatch.setattr(jit_runner, "run_subgridded_jit", observe)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _sim(mode).run(n_steps=2, skip_preflight=True, compute_s_params=False)
    notices = [w for w in caught if issubclass(w.category, ExperimentalSubgridWarning)]
    assert len(notices) == 1
    assert str(notices[0].message) == SUBGRID_WARNING
    assert 'validation="research"' not in str(notices[0].message)
    assert notices[0].filename == __file__
    assert result.time_series.shape[0] == 2
    assert closures == [True]  # #1465: same closure as the former production default


@pytest.mark.parametrize("mode", ["production", "research", "off"])
def test_validation_reports_unsupported(mode):
    from rfx.subgridding.validation import validate_subgrid_setup
    sim = _sim(mode)
    grid = sim._build_grid()
    materials, _, _, pec, *_ = sim._assemble_materials(grid)
    for report in (sim.validate_subgrid(), _sim().validate_subgrid(mode=mode),
                   validate_subgrid_setup(sim, grid, materials, pec, mode=mode)):
        assert not report.supported
        assert report.support_level == (
            "unsupported-unstable-unverified" if mode == "production" else f"{mode}-unverified")
        assert any(i.code == "subgrid_unstable_unverified" for i in report.issues)
        assert not any(i.code == "support_envelope" for i in report.issues)
        assert "validated guarded" not in report.format()
        if mode != "production":
            assert 'pass validation="research"' not in report.format()


def test_direct_disjoint_refused_before_reading_grid():
    from rfx.runners.disjoint import run_disjoint_stage2_path
    sim = _sim()
    sim._refinement["topology"] = "stage2_disjoint_3d"
    with pytest.raises(NotImplementedError, match="unstable and unverified"):
        run_disjoint_stage2_path(sim, None, 2)


@pytest.mark.parametrize("mode", ["research", "off"])
@pytest.mark.parametrize("n_ports", [1, 2])
def test_s_matrix_replays_emit_one_warning(mode, n_ports):
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .020), dx=.002, boundary="pec")
    sim.add_refinement((0., .014), ratio=2, validation=mode)
    for x in (.004, .008)[:n_ports]:
        sim.add_port((x, .006, .004), "ez", impedance=50.,
                     waveform=GaussianPulse(f0=5e9))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(n_steps=4, skip_preflight=True, compute_s_params=True,
                         s_param_freqs=np.array([5e9]), s_param_n_steps=4)
    notices = [w for w in caught if issubclass(w.category, ExperimentalSubgridWarning)]
    assert len(notices) == 1
    assert str(notices[0].message) == SUBGRID_WARNING
    assert notices[0].filename == __file__
    assert result.s_params.shape == (n_ports, n_ports, 1)


@pytest.mark.parametrize("mode", ["research", "off"])
def test_warning_points_to_one_line_user_script(mode):
    sim = _sim(mode)
    script = str(Path(__file__).with_name("user_subgrid_call.py"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(compile("sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)",
                     script, "exec"), {"sim": sim, "__name__": "__main__"})
    notices = [w for w in caught if issubclass(w.category, ExperimentalSubgridWarning)]
    assert len(notices) == 1
    assert notices[0].filename == script
    assert notices[0].lineno == 1

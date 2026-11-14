"""Progress on eager and refined paths; slow rows record complete byte hashes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.progress import events
from tests.contracts._progress_scenes import add_records, measure, scene, require_current_moment_determinism


@pytest.mark.slow
@pytest.mark.parametrize("n_steps", [101, 650])
@pytest.mark.parametrize("mode", ["checkpoint", "plain", "s11"])
def test_eager_uniform(mode, n_steps, tmp_path, monkeypatch):
    sim = scene(source=mode != "s11")
    sim.add_dft_plane_probe(axis="y", coordinate=.004, component="ez", freqs=[3e9, 7e9])
    options = {"checkpoint": False} if mode == "plain" else {}
    if mode == "s11":
        options["port_s11_freqs"] = jnp.array([3e9, 7e9])
    measure(lambda: sim.forward(n_steps=n_steps, skip_preflight=True, **options),
            path="uniform", tmp_path=tmp_path, monkeypatch=monkeypatch,
            metadata=dict(mode=mode, n_steps=n_steps, cells=sim._build_grid().shape))


@pytest.mark.slow
@pytest.mark.parametrize("entry", ["forward", "run"])
def test_eager_uniform_all_records(entry, tmp_path, monkeypatch):
    require_current_moment_determinism()
    sim = scene()
    add_records(sim)
    options = {"compute_s_params": False} if entry == "run" else {}
    measure(lambda: getattr(sim, entry)(n_steps=101, skip_preflight=True, **options),
            path="uniform", tmp_path=tmp_path, monkeypatch=monkeypatch,
            metadata=dict(entry=entry, n_steps=101, cells=sim._build_grid().shape))


@pytest.mark.slow
@pytest.mark.parametrize("n_steps", [37, 101, 650])
def test_eager_pec(n_steps, tmp_path, monkeypatch):
    sim = scene(boundary="pec")
    sim.add_flux_monitor(axis="y", coordinate=.004, freqs=[3e9, 7e9])
    measure(lambda: sim.forward(n_steps=n_steps, skip_preflight=True),
            path="uniform", tmp_path=tmp_path, monkeypatch=monkeypatch,
            metadata=dict(n_steps=n_steps, boundary="pec", cells=sim._build_grid().shape))


@pytest.mark.slow
@pytest.mark.parametrize("n_steps", [101, 650])
def test_eager_graded(n_steps, tmp_path, monkeypatch):
    sim = scene(graded=True)
    sim.add_dft_plane_probe(axis="y", coordinate=.004, component="ez", freqs=[3e9, 7e9])
    measure(lambda: sim.forward(n_steps=n_steps, skip_preflight=True),
            path="graded", ticks=False, tmp_path=tmp_path, monkeypatch=monkeypatch,
            metadata=dict(n_steps=n_steps, cells=sim._build_nonuniform_grid().shape))


@pytest.mark.parametrize("graded", [False, True])
def test_run_forward_phase_sequence(graded, monkeypatch):
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    sim = scene(graded=graded)
    rows = [], []
    with events(rows[0].append):
        sim.run(n_steps=4, compute_s_params=False, skip_preflight=True)
    with events(rows[1].append):
        sim.forward(n_steps=4, skip_preflight=True)
    excluded = {"t", "run_id", "entry", "rate_steps_per_s", "eta_s"}
    normalized = [[{k: v for k, v in row.items() if k not in excluded}
                   for row in batch if row["kind"] != "progress"] for batch in rows]
    assert normalized[0] == normalized[1]


def test_forward_before_step_keeps_original_error(monkeypatch):
    from rfx.runners.progress_events import entry_scope
    class Unresolved:
        _refinement = None
        _ports = ()
        @property
        def _declared_mesh(self):
            raise ValueError("unresolved declaration")
    rows = []
    original = RuntimeError("assembly failed")
    with events(rows.append), pytest.raises(RuntimeError) as caught:
        with entry_scope(Unresolved(), "forward", {}, deferred=True):
            assert rows == []
            raise original
    assert caught.value is original
    assert {r["path"] for r in rows} == {"unresolved"}
    assert [r["reason_code"] for r in rows if r["kind"] == "not_reported"] == [
        "path_not_wired", "compile_not_separated", "path_not_wired", "path_not_wired"]


def test_held_timestamp_and_captured_rlc_tracer(monkeypatch):
    from dataclasses import dataclass
    from rfx.stepping import events as transport
    monkeypatch.setattr(transport, "_TRACED", set())
    clock = [10.]
    monkeypatch.setattr(transport.time, "time", lambda: clock[0])
    rows = []
    with events(rows.append), transport.scope("forward", "uniform", refusal="path_not_wired"):
        assert rows == []
        clock[0] = 20.
        transport.begin_step(None, total=4, ticks=True)
        assert [r["t"] for r in rows[:2]] == [10., 10.]
        transport.end_step(None, 4)
    @dataclass
    class Context:
        rlc_metas: tuple
    rows.clear()
    def objective(inductance):
        with events(rows.append), transport.scope("forward", "uniform", refusal="path_not_wired"):
            assert not transport.begin_step(None, total=4, ticks=True,
                                            probes=(Context((inductance,)),))
        return inductance ** 2
    jax.grad(objective)(1.)
    assert [r["reason_code"] for r in rows if r["kind"] == "not_reported"] == [
        "traced", "compile_not_separated", "traced", "traced"]


@pytest.mark.slow
def test_eager_checkpoint_segments(monkeypatch, tmp_path):
    sim = scene()
    measure(lambda: sim.forward(n_steps=8, checkpoint_segments=2, skip_preflight=True),
            path="uniform", ticks=False, tmp_path=tmp_path, monkeypatch=monkeypatch)


@pytest.mark.slow
def test_traced_auto_geometry_preserves_exception(monkeypatch, record_property):
    from rfx.geometry import Box
    from rfx.stepping import events as transport
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    monkeypatch.setattr(transport, "_TRACED", set())
    def objective(width):
        sim = Simulation(freq_max=10e9, domain=(.012, .009, .007))
        sim.add(Box((0., 0., 0.), (width, .006, .004)), material="pec")
        return jnp.sum(sim.forward(n_steps=2, skip_preflight=True).time_series)
    traced_objective = jax.jit(objective)
    with pytest.raises(Exception) as reference:
        traced_objective(jnp.array(.008))
    rows = []
    with events(rows.append), pytest.raises(type(reference.value)) as actual:
        traced_objective(jnp.array(.008))
    assert type(actual.value) is type(reference.value)
    assert str(actual.value) == str(reference.value)
    assert rows
    record_property("paths", sorted({r["path"] for r in rows}))
    assert {r["path"] for r in rows} == {"unresolved"}


@pytest.mark.slow
def test_grad_inductance_uses_single_scan(monkeypatch):
    from rfx.stepping import events as transport, uniform_loop
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    monkeypatch.setattr(transport, "_TRACED", set())
    calls = []
    original = uniform_loop.scan_with_progress
    def observe(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(uniform_loop, "scan_with_progress", observe)
    def objective(inductance):
        sim = scene(ports=0)
        sim.add_lumped_rlc((.006, .003, .002), "ez", L=1e-9)
        return jnp.sum(sim.forward(n_steps=6, skip_preflight=True,
                                  rlc_values_override={0: {"L": inductance}}).time_series ** 2)
    reference = jax.grad(objective)(jnp.array(1e-9))
    rows = []
    with events(rows.append):
        actual = jax.grad(objective)(jnp.array(1e-9))
    np.testing.assert_array_equal(actual, reference)
    assert np.isfinite(actual)
    assert calls == []
    assert [r["reason_code"] for r in rows if r["kind"] == "not_reported"] == [
        "traced", "compile_not_separated", "traced", "traced"]


@pytest.mark.slow
def test_declared_inductance_grad(monkeypatch):
    from rfx.stepping import events as transport, uniform_loop
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    monkeypatch.setattr(transport, "_TRACED", set())
    calls = []
    original = uniform_loop.scan_with_progress
    def observe(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(uniform_loop, "scan_with_progress", observe)
    def objective(inductance):
        sim = scene(ports=0)
        sim.add_lumped_rlc((.006, .003, .002), "ez", R=50., L=inductance)
        return jnp.sum(sim.forward(n_steps=6, skip_preflight=True).time_series ** 2)
    reference = jax.grad(objective)(1e-9)
    rows = []
    with events(rows.append):
        actual = jax.grad(objective)(1e-9)
    np.testing.assert_array_equal(actual, reference)
    assert calls == []
    assert [r["reason_code"] for r in rows if r["kind"] == "not_reported"] == [
        "traced", "compile_not_separated", "traced", "traced"]


def overlap_scene():
    sim = Simulation(freq_max=10e9, domain=(.012, .010, .020), dx=.002,
                     boundary="pec", cpml_layers=0)
    sim.add_refinement((0., .014), ratio=2, validation="research")
    for index, x in enumerate((.004, .008)):
        sim.add_port((x, .006, .004), "ez", impedance=50., excite=index == 0,
                     waveform=GaussianPulse(5e9))
    sim.add_probe((.008, .004, .006), "ez")
    return sim


@pytest.mark.slow
def test_overlap_ignored_report_every(monkeypatch):
    from itertools import count
    from rfx.errors import ConfigurationWarning
    from rfx.stepping import events as transport
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    clock = count(0., 2.)
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(clock))
    rows = []
    with events(rows.append), pytest.warns(ConfigurationWarning, match="report_every=10 is ignored"):
        overlap_scene().run(n_steps=101, report_every=10, compute_s_params=False, skip_preflight=True)
    boundaries = [r["steps_done"] for r in rows if r["kind"] == "progress"]
    assert boundaries == list(range(2, 100, 2)) + [101]


@pytest.mark.slow
def test_overlap_phase_accounting():
    from tests.contracts._progress_scenes import sequence
    rows = []
    with events(rows.append):
        overlap_scene().run(n_steps=4, compute_s_params=False, skip_preflight=True)
    sequence(rows, "subgridded")


@pytest.mark.slow
@pytest.mark.parametrize("n_steps", [101, 650])
def test_overlap_s_passes(n_steps, tmp_path, monkeypatch):
    sim = overlap_scene()
    rows = measure(lambda: sim.run(n_steps=n_steps, compute_s_params=True,
                                    s_param_freqs=np.array([3e9, 7e9]), skip_preflight=True),
                   path="subgridded", passes=3, tmp_path=tmp_path, monkeypatch=monkeypatch,
                   metadata=dict(n_steps=n_steps, cells=sim._build_grid().shape))
    assert rows[0]["passes"] == ["main", "p1", "p2"]


@pytest.mark.slow
@pytest.mark.parametrize("n_steps", [37, 101, 650])
def test_disjoint(n_steps, tmp_path, monkeypatch):
    sim = Simulation(freq_max=10e9, domain=(.008, .010, .012), dx=.001,
                     boundary="pec", cpml_layers=0)
    sim.add_refinement((.002, .008), ratio=2, validation="off", topology="stage2_disjoint_3d")
    sim.add_source((.003, .004, .004), "ez", amplitude_kind="field")
    sim.add_probe((.005, .006, .005), "ez")
    measure(lambda: sim.run(n_steps=n_steps, compute_s_params=False, skip_preflight=True),
            path="subgridded", tmp_path=tmp_path, monkeypatch=monkeypatch,
            metadata=dict(n_steps=n_steps, cells=sim._build_grid().shape, boundary="pec"))


@pytest.mark.parametrize("entry", ["compute_coaxial_line_reflection", "compute_coaxial_two_port", "compute_coax_msl_transition"])
def test_coax_scope_still_refuses_at_open(entry):
    from rfx.runners.progress_events import entry_scope
    sim = scene()
    rows = []
    with events(rows.append), entry_scope(sim, entry, {}, deferred=True):
        assert rows and rows[0]["kind"] == "plan"
        assert all(r["kind"] == "not_reported" for r in rows[1:])
    assert {r["reason_code"] for r in rows[1:]} == {"path_not_wired", "compile_not_separated"}

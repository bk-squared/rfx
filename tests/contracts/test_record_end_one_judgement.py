"""One source end for every reader: the tables the time step applies (#2030 S1).

Each scene is built through run() (or the coaxial driver) and stopped at the
loop entry, before the first step; the tables the loop would apply are read
from the call. The expected end is computed here with numpy from those tables:
last sample above 1e-6 of its own peak, plus one, plus the free-space transit
ceil(d / (C0 dt)) from the farthest injecting point to the farthest record
(0 for the whole-domain reader of until_decay). A waveguide port's end is its
applied E table held at its last entry, never the analytic v_inc_t. A lane
that refuses the declaration is a "not reachable" row, its refusal pinned.
"""
import ast
import dataclasses
import math
import pathlib
import re
import warnings

import numpy as np
import pytest

from rfx.grid import C0
from rfx.sources.sources import GaussianPulse
from tests.contracts.path_equivalence import builders

REPO = pathlib.Path(__file__).resolve().parents[2]
N = 6000
PULSE = GaussianPulse(f0=5e9, bandwidth=0.5)
ROWS = {
    "lumped": ("_ports", "lumped_port"),
    "wire": ("_ports", "wire_port"),
    "msl": ("_msl_ports", "msl_port"),
    "waveguide": ("_waveguide_ports", "waveguide_port"),
    "floquet": ("_floquet_ports", "floquet_port"),
    "tfsf": ("_tfsf", "plane_wave"),
    "tfsf_closed_box": ("_tfsf", "closed_box"),
}
NOT_REACHABLE = {
    ("tfsf_closed_box", True): "CustomWaveform and closed_box TFSF require a uniform",
    ("floquet", True): "Floquet ports do not support non-uniform z mesh",
    ("coax", True): "compute_coaxial_two_port() supports only a uniform Yee grid",
}


class _Loop(Exception):
    def __init__(self, args, kwargs):
        super().__init__("loop entry")
        self.call = (args, kwargs)


def _scene(kind, graded):
    if kind == "coax":
        from rfx import Simulation
        mesh = {"dz_profile": np.full(60, .001)} if graded else {}
        sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary="cpml", **mesh)
        sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                             waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
        return sim
    sim = builders.build(ROWS[kind], "run_nonuniform" if graded else "run_uniform", graded=graded)
    if kind == "tfsf" and graded:
        # A plane wave along x needs uniform x cells (#2078); y and z stay graded.
        sim._dx_profile = np.full_like(np.asarray(sim._dx_profile), builders.DX)
    # A pulse that ends inside the record (the builders' waveform never ends).
    sim._ports = [dataclasses.replace(p, waveform=PULSE) for p in sim._ports]
    sim._msl_ports = [dataclasses.replace(p, waveform=PULSE) for p in sim._msl_ports]
    return sim


def _loop_call(monkeypatch, sim, kind):
    import rfx.runners.nonuniform as graded_runner
    import rfx.simulation as simulation

    def stop(*args, **kwargs):
        raise _Loop(args, kwargs)

    monkeypatch.setattr(simulation, "run", stop)
    monkeypatch.setattr(graded_runner, "run_nonuniform", stop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if kind == "coax":
                sim.compute_coaxial_two_port(n_steps=N, freqs=np.array([8e9]))
            else:
                sim.run(n_steps=N, skip_preflight=True)
        except _Loop as stopped:
            return stopped.call
    raise AssertionError("the loop entry was not reached")


def _off(table, edge=False):
    """Last sample above 1e-6 of its own peak, plus one; None if still on."""
    w = np.asarray(table)[:N]
    w = np.pad(w, (0, N - len(w)), mode="edge" if edge else "constant")
    w = np.abs(w.astype(np.complex128)) if np.iscomplexobj(w) else np.abs(w.astype(np.float64))
    on = np.flatnonzero(w > 1e-6 * w.max())
    off = int(on[-1]) + 1 if on.size else 0
    return None if off >= N and w.max() > 0 else off


def _transit(distance, dt):
    return math.ceil(distance / (C0 * dt))


def _table(src):
    return np.asarray(src.waveform if hasattr(src, "waveform") else src[4])


def _node(grid, entry):
    idx = (entry.i, entry.j, entry.k) if hasattr(entry, "i") else entry[:3]
    return np.array([grid.node_of(axis, int(i)) for axis, i in zip("xyz", idx)])


def _analytic_v_inc(cfg, dt):
    """The analytic incident voltage the modal record stores (rfx/measurement/modal.py)."""
    t = np.arange(1, N + 1, dtype=np.float64) * dt
    arg = (t - float(cfg.src_t0)) / float(cfg.src_tau)
    return float(cfg.src_amp) * (-2.0 * arg) * np.exp(-arg ** 2)


def _expected(kind, sim, grid, call):
    """(whole-domain end, end with transit to sim's probes), from the applied tables."""
    from rfx.measurement.settling import tfsf_source_drive

    _, kw = call
    dt = float(grid.dt)
    probes = [np.asarray(p.position, dtype=float) for p in sim._probes]
    whole, probed = [], []
    for src in (*(kw.get("sources") or ()), *(kw.get("mag_sources") or ())):
        whole.append(_off(_table(src)))
    if kind in ("lumped", "wire", "msl"):
        for pe in (*sim._ports, *sim._msl_ports):
            starts = [np.asarray(pe.position, dtype=float)]
            if getattr(pe, "extent", None):
                end = starts[0].copy()
                end["xyz".index(pe.component[1])] += pe.extent
                starts.append(end)
            reach = max(np.linalg.norm(s - p) for s in starts for p in probes)
            probed += [None if o is None else o + _transit(reach, dt) for o in whole]
    if kind == "floquet":
        (entry,) = sim._floquet_ports
        center = np.asarray(sim._domain, dtype=float) / 2
        center["xyz".index(entry.axis)] = entry.position
        reach = max(np.linalg.norm(center - p) for p in probes)
        probed = [o + _transit(reach, dt) for o in whole]
    for cfg in kw.get("waveguide_ports") or ():
        off = _off(cfg.e_inc_table, edge=True)
        whole.append(off)
        axis = "xyz".index(cfg.normal_axis)
        reach = max(abs(p[axis] - float(cfg.source_x_m)) for p in probes)
        probed.append(off + _transit(reach, dt))
    if kw.get("tfsf") is not None:
        import jax
        import jax.numpy as jnp
        waveform, diagonal = tfsf_source_drive(kw["tfsf"][0], grid)
        with jax.ensure_compile_time_eval():
            samples = np.asarray(jax.vmap(waveform)(jnp.arange(N, dtype=jnp.float32) * dt))
        whole.append(_off(samples) + _transit(diagonal, dt))
        probed = list(whole)
    assert None not in whole and None not in probed, (whole, probed)
    return max(whole), max(probed)


CASES = [(kind, graded) for kind in (*ROWS, "coax") for graded in (False, True)]


@pytest.mark.parametrize("kind,graded", CASES,
                         ids=[f"{k}-{'graded' if g else 'uniform'}" for k, g in CASES])
def test_end_is_the_latest_end_of_the_applied_tables(kind, graded, monkeypatch):
    from rfx.measurement.record_end import drive_end, injection_end

    if (kind, graded) in NOT_REACHABLE:
        error = NotImplementedError if kind == "tfsf_closed_box" else ValueError
        with pytest.raises(error, match=re.escape(NOT_REACHABLE[kind, graded])):
            _loop_call(monkeypatch, _scene(kind, graded), kind)
        return
    sim = _scene(kind, graded)
    call = _loop_call(monkeypatch, sim, kind)
    args, kw = call
    grid = args[0]
    dt = float(grid.dt)
    if kind == "coax":
        # The coaxial drivers read every built E and H table with transit to
        # their witness probes; run() and forward() refuse a coaxial port.
        sources, mags, probes = kw["sources"], kw["mag_sources"], kw["probes"]
        assert sources and mags and probes
        ends = []
        for src in (*sources, *mags):
            reach = max(np.linalg.norm(_node(grid, src) - _node(grid, p)) for p in probes)
            ends.append(_off(_table(src)) + _transit(reach, dt))
        assert injection_end(N, dt, records=probes, grid=grid, sources=sources,
                             mag_sources=mags) == max(ends)
        with pytest.raises(NotImplementedError, match="not wired into Simulation.run"):
            _loop_call(monkeypatch, _scene(kind, graded), "run")
        return
    whole, probed = _expected(kind, sim, grid, call)
    tfsf = kw.get("tfsf")
    configs = list(kw.get("waveguide_ports") or ())
    # until_decay's reader: distance zero (TF/SF keeps its box diagonal).
    assert drive_end(grid, N, sources=kw.get("sources") or (), mag_sources=kw.get("mag_sources") or (),
                     tfsf=tfsf, waveguide_ports=configs) == whole
    if configs:
        # The record a run returns holds the analytic incident voltage; the
        # scene is chosen so that it ends elsewhere than the applied E table.
        configs = [cfg._replace(v_inc_t=_analytic_v_inc(cfg, dt)) for cfg in configs]
        assert _off(configs[0].v_inc_t) != _off(configs[0].e_inc_table, edge=True)
    # The run()/forward() settling reader: the declarations plus the built
    # waveguide configs, transit to the user probes.
    assert injection_end(N, dt, records=sim._probes, grid=grid, sim=sim,
                         waveguide_ports=configs) == probed


def test_traced_and_still_on_read_one_text_on_every_path():
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace

    from rfx.measurement.record_end import STILL_ON, TRACED, injection_end

    on = SimpleNamespace(src_amp=1., e_inc_table=np.ones(50), source_x_m=0., normal_axis="x")
    step, reason, _ = injection_end(40, 1., waveguide_ports=[on], return_detail=True)
    assert step is None and reason == STILL_ON.format(source="waveguide port[0]")

    @jax.jit
    def traced(x):
        cfg = SimpleNamespace(src_amp=1., e_inc_table=x, source_x_m=0., normal_axis="x")
        assert injection_end(40, 1., waveguide_ports=[cfg], return_detail=True)[:2] == (None, TRACED)
        return x

    traced(jnp.ones(50))


SOURCE_END_NAMES = {"source_end_step", "simulation_source_end_step", "sampled_source_end_step",
                    "drive_end", "injection_end"}
CALLERS = ("rfx/api/_settling.py", "rfx/probes/sparam_driver.py", "rfx/sparams/msl.py",
           "rfx/sparams/mixed.py", "rfx/sparams/coax.py", "rfx/runners/uniform.py",
           "rfx/runners/nonuniform.py", "rfx/measurement/rcs.py", "rfx/measurement/decay.py",
           "rfx/measurement/settling.py", "rfx/ringdown.py")
# An off step from a drive table: a threshold relative to the table's peak.
OFF_STEP = re.compile(r"nonzero\([^\n]*>[^\n]*peak")


def test_no_module_outside_record_end_computes_a_source_end():
    home = REPO / "rfx/measurement/record_end.py"
    found = []
    for path in sorted((REPO / "rfx").rglob("*.py")):
        if path == home:
            continue
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SOURCE_END_NAMES:
                found.append(f"{path.relative_to(REPO)}:{node.lineno} defines {node.name}")
        found += [f"{path.relative_to(REPO)}: off step '{m.group(0)}'" for m in OFF_STEP.finditer(text)]
    assert not found, found
    for caller in CALLERS:
        text = (REPO / caller).read_text()
        assert "from rfx.measurement.record_end import" in text, caller
        assert not re.search(r"probes\.settling import[^\n]*(source_end_step|sampled_source_end_step)", text), caller


def test_floquet_end_is_the_same_under_an_outer_jit():
    """The Floquet table is built from the declared pulse; under a trace it
    must still be concrete, or the collector raised instead of judging."""
    import jax
    import jax.numpy as jnp

    from rfx.measurement.record_end import injection_end

    sim = _scene("floquet", False)
    grid = sim._build_grid()
    host = injection_end(N, grid.dt, records=sim._probes, grid=grid, sim=sim)
    seen = []

    @jax.jit
    def traced(x):
        seen.append(injection_end(N, grid.dt, records=sim._probes, grid=grid, sim=sim))
        return x

    traced(jnp.ones(2))
    assert host is not None and seen == [host]


# S2a: flux and point start at the same applied-drive end as energy.
S2_BOX = ((.012, .012, .012), (.028, .028, .028))


def _s2_scene(graded=False, boundary="cpml"):
    from rfx import Simulation

    mesh = {"dz_profile": np.full(25, .0016)} if graded else {}
    sim = Simulation(domain=(.04, .04, .04), freq_max=10e9, dx=.0016,
                     boundary=boundary, **mesh)
    sim.add_source((.0193, .0207, .0211), "ez", amplitude_kind="current")
    sim.add_probe((.0251, .0172, .0222), "ez")
    return sim


@pytest.mark.parametrize("kind", ["energy", "flux", "point"])
@pytest.mark.parametrize("min_steps,n_d", [(50, 0), (300, 0), (50, 320)])
def test_s2_rule_first_eligible_check(kind, min_steps, n_d, monkeypatch):
    from rfx.measurement.decay import DecayStop, FluxStop, PointStop, finish_decay

    # Realized default pulse; the expected end uses this file's independent sum-free
    # table inspection, never record_end or a decay constructor/helper.
    with monkeypatch.context() as patch:
        _, kw = _loop_call(patch, _s2_scene(), "source")
    n_end = max(_off(_table(src)) for src in kw["sources"])
    assert n_end == 183
    interval, consecutive = 50, 2
    # thr=2 makes every positive level below threshold from the first check,
    # isolating the start gate from the physical ringdown time.
    common = dict(thr=2., min_steps=min_steps, interval=interval, n_end=n_end, n_d=n_d)
    if kind == "energy":
        rule = DecayStop(**common, consecutive=consecutive)
    elif kind == "flux":
        rule = FluxStop(**common, consecutive=consecutive)
    else:
        rule = PointStop(**common)
    ceil_check = lambda n: math.ceil(n / interval) * interval
    first = ceil_check(max(min_steps, n_end, n_d))
    expected = (max(ceil_check(n_end) + (consecutive - 1) * interval,
                    ceil_check(max(min_steps, n_d))) if kind == "energy"
                else first + (consecutive - 1) * interval if kind == "flux" else first)
    for count in range(interval, 501, interval):
        stopped = rule.check(count, np.ones(interval) if kind == "point" else lambda: 1.)
        if count < expected:
            assert stopped is None
        else:
            assert stopped == expected
            break
    else:
        pytest.fail(f"{kind} never stopped at its first eligible check {expected}")
    diagnostic = finish_decay(rule, stopped)
    assert diagnostic.values["drive_end_step"] == n_end
    assert diagnostic.values["stop_step"] == expected


@pytest.mark.parametrize("graded,boundary", [(False, "cpml"), (True, "cpml"), (False, "pec")])
def test_s2_run_waits_for_applied_drive(graded, boundary, monkeypatch, record_property):
    import rfx.runners.nonuniform as graded_runner
    import rfx.simulation as simulation

    owner = graded_runner if graded else simulation
    name = "run_nonuniform_until_decay" if graded else "run_until_decay"
    original, ends = getattr(owner, name), []

    def observed(*args, **kwargs):
        ends.append(max(_off(_table(src)) for src in kwargs["sources"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, observed)
    flux = boundary == "cpml"
    result = _s2_scene(graded, boundary).run(
        until_decay=2., decay_min_steps=50, decay_max_steps=350,
        **({"radiated_flux_box": S2_BOX} if flux else {}), skip_preflight=True)
    (diagnostic,) = [d for d in result.diagnostics if d.code == "until_decay.stop"]
    values = diagnostic.values
    assert ends == [183]
    assert values["stop_step"] >= ends[0]
    assert values["drive_end_step"] == ends[0]
    assert values["criterion"] == "threshold"
    assert values["quantity"] == ("radiated_flux" if flux else "point_field")
    assert values["stop_step"] == (250 if flux else 200)
    record_property("s2_stop", dict(values))


@pytest.mark.parametrize("graded,boundary,decay", [
    (False, "pec", 1e-3), (False, "cpml", None), (True, "cpml", None),
    (False, "pec", None), (True, "pec", None), (False, "cpml", 0.),
])
def test_s2_refuses_a_flux_box_without_a_flux_stop(graded, boundary, decay):
    with pytest.raises(ValueError, match="radiated_flux_box selects the radiated-flux until_decay stop"):
        _s2_scene(graded, boundary).run(n_steps=20, until_decay=decay,
                                      radiated_flux_box=S2_BOX, skip_preflight=True)


def test_s2_direct_loop_refuses_a_closed_box_flux_box(monkeypatch):
    """The public loop called directly, not through run(): the box was dropped there too."""
    import rfx.simulation as simulation

    original = simulation.run_until_decay

    def stop(*args, **kwargs):
        raise _Loop(args, kwargs)

    monkeypatch.setattr(simulation, "run_until_decay", stop)
    with pytest.raises(_Loop) as entry:
        _s2_scene(False, "pec").run(until_decay=1e-3, decay_max_steps=60, skip_preflight=True)
    args, kwargs = entry.value.call
    assert kwargs["boundary"] == "pec"
    with pytest.raises(ValueError, match="radiated_flux_box selects the radiated-flux until_decay stop"):
        original(*args, **{**kwargs, "radiated_flux_box": S2_BOX})


@pytest.mark.parametrize("change", [{"decay_by": 0.0}, {"cpml_axes": ""}], ids=["no_stop", "closed"])
def test_s2_direct_graded_loop_refuses_an_unused_flux_box(change, monkeypatch):
    """The public graded loop called directly: a box with no flux stop, or with no absorber, is refused."""
    import rfx.runners.nonuniform as graded_runner

    original = graded_runner.run_nonuniform_until_decay

    def stop(*args, **kwargs):
        raise _Loop(args, kwargs)

    monkeypatch.setattr(graded_runner, "run_nonuniform_until_decay", stop)
    with pytest.raises(_Loop) as entry:
        _s2_scene(True).run(until_decay=1e-3, decay_max_steps=60, radiated_flux_box=S2_BOX,
                            skip_preflight=True)
    args, kwargs = entry.value.call
    assert kwargs["radiated_flux_box"] == S2_BOX and kwargs["decay_by"] > 0
    with pytest.raises(ValueError, match="radiated_flux_box selects the radiated-flux until_decay stop"):
        original(*args, **{**kwargs, **change})


@pytest.mark.parametrize("flux", [False, True])
def test_s2_direct_constructor_and_still_on_cap(flux, monkeypatch):
    from rfx.measurement.decay import make_field_decay_stop, finish_decay

    with monkeypatch.context() as patch:
        args, kw = _loop_call(patch, _s2_scene(), "source")
    grid, sources = args[0], kw["sources"]
    expected = max(_off(_table(src)) for src in sources)
    rule = make_field_decay_stop(grid, thr=2., min_steps=50, interval=50,
                                max_steps=350, flux=flux, sources=sources)
    assert rule.n_end == expected
    # A shorter cap cuts the pulse while it is still injecting. Peak tracking
    # remains live, but neither rule may stop; the diagnostic names the drive.
    capped = make_field_decay_stop(grid, thr=2., min_steps=0, interval=50,
                                  max_steps=100, flux=flux, sources=sources)
    assert capped.n_end is None
    for count in (50, 100):
        assert capped.check(count, (lambda: 1.) if flux else np.ones(50)) is None
    diagnostic = finish_decay(capped, 100)
    assert diagnostic.values["criterion"] == "cap"
    assert math.isnan(diagnostic.values["drive_end_step"])
    assert "source[0]" in diagnostic.message and "still injecting" in diagnostic.message


def test_s2_peaks_keep_drive_checks_and_between_check_samples():
    from rfx.measurement.decay import FluxStop, PointStop

    flux = FluxStop(.1, 0, 10, 2, 40, env_checks=2)
    for count, value in [(10, 100.), (20, 1.), (30, 1.), (40, 1.)]:
        assert flux.check(count, lambda: value) is None
    assert flux.peak == 100. and flux.below == 1
    assert flux.check(45, lambda: pytest.fail("off-interval flux read")) is None
    assert flux.check(50, lambda: 1.) == 50
    point = PointStop(.1, 0, 10, 40)
    assert point.check(10, [10., 1.]) is None
    assert point.check(30, [1., 1.]) is None
    assert point.check(39, [1.]) is None
    assert point.peak == 100.
    assert point.check(40, [1., 1.]) == 40

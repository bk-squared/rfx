"""Every modal S assembler applies both ports' strict propagation domain."""
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement import waveguide_s as ws, waveguide_modal as wm
from rfx.measurement.waveguide_power import WaveguideCutoffWarning
from rfx.model.waveguide_modes import PortMedium


class Record(SimpleNamespace):
    def _replace(self, **kw):
        return Record(**(vars(self) | kw))


def record(cutoff=6e9, **kw):
    z = jnp.zeros(4)
    return Record(**(dict(
        freqs=jnp.array([5e9, 6e9, 7e9, 9e9]), f_cutoff=cutoff,
        mode_type="TE", mode_indices=(1, 0), medium=PortMedium(),
        direction="+x", dt=0., dx=.001, a=.025, b=.012,
        normal_axis="x", probe_x=3, ref_x=2, x_index=1,
        u_lo=0, u_hi=3, v_lo=0, v_hi=3, src_amp=1.,
        v_ref_t=z, i_ref_t=z, v_probe_t=z, i_probe_t=z, v_inc_t=z,
        n_steps_recorded=4, probe_medium_matches=True,
    ) | kw))


def check_warning(caught, ports=2):
    notices = [w for w in caught if issubclass(w.category, WaveguideCutoffWarning)]
    assert len(notices) == 1
    assert notices[0].filename == __file__
    msg = str(notices[0].message)
    assert "below cutoff" in msg and "no power wave" in msg
    assert "TE10" in msg and "bins [0, 1]" in msg
    if ports == 2:
        assert "port 1" in msg and "bins [0, 1, 2]" in msg


@pytest.fixture
def manufactured(monkeypatch):
    import rfx.stepping.uniform_loop as loop
    import rfx.measurement as measurement
    import rfx.probes.probes as probes
    import rfx.boundaries.pec as pec

    def run(grid, materials, n_steps, *, waveguide_ports, **kw):
        ref = materials == "reference"
        cfgs = [c._replace(v_ref_t=jnp.full(4, .5 if ref else 1.)) for c in waveguide_ports]
        mons = [Record(e1_dft=jnp.full(4, 4. if ref else 1.)) for c in cfgs]
        return Record(waveguide_ports=cfgs, flux_monitors=mons)

    monkeypatch.setattr(loop, "run", run)
    monkeypatch.setattr(ws, "resolve_cpml_axes", lambda grid, axes: axes)
    monkeypatch.setattr(ws, "_s_matrix_port_waves", lambda cfg, **kw: (jnp.full(4, 2.), cfg.v_ref_t))
    monkeypatch.setattr(ws, "_modal_net_power", lambda cfg: np.full(4, 4. if float(cfg.v_ref_t[0]) == .5 else 1.))
    monkeypatch.setattr(pec, "resolve_wall_faces", lambda *a: ((), ()))
    monkeypatch.setattr(measurement, "grid_flux_monitor", lambda *a, **kw: Record(
        **{k: jnp.zeros(4, complex) for k in ("e1_dft", "e2_dft", "h1_dft", "h2_dft")}))
    monkeypatch.setattr(probes, "flux_spectrum", lambda mon: mon.e1_dft)
    return [record(), record(7e9)]


@pytest.mark.parametrize("lane", ["single", "unequal", "multi", "flux", "multi_flux", "normalized"])
def test_matrix_assemblers(lane, manufactured):
    cfgs = manufactured
    if lane == "unequal":
        cfgs[1] = cfgs[1]._replace(medium=PortMedium(2., 1.))
    grid = Record(shape=(5, 4, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if lane in ("single", "unequal"):
            got = ws.extract_waveguide_s_matrix(grid, "device", cfgs, 4, cpml_axes="x")
        elif lane == "multi":
            got, _ = ws.extract_multimode_s_matrix(grid, "device", [[c] for c in cfgs], 4, cpml_axes="x")
        elif lane == "multi_flux":
            got, _ = ws.extract_multimode_s_matrix_flux(grid, "device", "reference", [[c] for c in cfgs], 4, cpml_axes="x")
        else:
            fn = ws.extract_waveguide_s_matrix_flux if lane == "flux" else ws.extract_waveguide_s_params_normalized
            got = fn(grid, "device", "reference", cfgs, 4, cpml_axes="x")
    got = np.asarray(got)
    check_warning(caught)
    for i in range(2):
        for j in range(2):
            defined = np.array([False, False, i == j == 0, True])
            assert np.isnan(got[i, j, ~defined]).all()
            expected = .25 if lane == "normalized" and i == j else .5
            if lane in ("flux", "multi_flux") and i == j:
                expected = np.sqrt(.75)
            if lane == "unequal" and i != j:
                f = 9e9
                zi = np.sqrt(1.25663706212e-6 / (8.8541878128e-12 * cfgs[i].medium.eps_r)) / np.sqrt(1-(cfgs[i].f_cutoff/f)**2)
                zj = np.sqrt(1.25663706212e-6 / (8.8541878128e-12 * cfgs[j].medium.eps_r)) / np.sqrt(1-(cfgs[j].f_cutoff/f)**2)
                expected *= np.sqrt(zj/zi)
            np.testing.assert_allclose(got[i, j, defined], expected, rtol=2e-6)


@pytest.mark.parametrize("lane", ["pair", "s11", "s21", "overlap"])
def test_scalar_extractors(lane, monkeypatch):
    cfg = record()
    monkeypatch.setattr(ws, "_compute_beta", lambda *a, **kw: jnp.zeros(4))
    monkeypatch.setattr(wm, "_compute_beta", lambda *a, **kw: jnp.zeros(4))
    monkeypatch.setattr(ws, "_extract_port_waves_from_time_series", lambda *a: (jnp.full(4, 2.), jnp.ones(4)))
    monkeypatch.setattr(wm, "_compute_mode_impedance", lambda *a, **kw: jnp.ones(4))
    monkeypatch.setattr(wm, "mode_self_overlap", lambda *a: 1.)
    acc = Record(p1_ref_dft=jnp.full(4, 3.), p2_ref_dft=jnp.ones(4),
                 p1_probe_dft=jnp.full(4, 3.), p2_probe_dft=jnp.ones(4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = {"pair": lambda: ws.extract_waveguide_sparams(cfg),
               "s11": lambda: (ws.extract_waveguide_s11(cfg),),
               "s21": lambda: (ws.extract_waveguide_s21(cfg),),
               "overlap": lambda: wm.extract_waveguide_sparams_overlap(acc, cfg)}[lane]()
    check_warning(caught, ports=1)
    for k, value in enumerate(got):
        assert np.isnan(np.asarray(value)[:2]).all()
        expected = 1. if lane == "s21" or k == 1 else .5
        np.testing.assert_array_equal(np.asarray(value)[2:], expected)


@pytest.mark.parametrize("lane", ["pair", "s11"])
@pytest.mark.parametrize("policy", ["mps", "gpu"])
def test_traced_cutoff_notice_without_host_callbacks(lane, policy, monkeypatch):
    from rfx import Simulation, _traced_check
    from rfx.backend import BackendClass

    sim = Simulation(freq_max=12e9, domain=(.12, .024, .012), dx=.002,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8)
    sim.add_waveguide_port(.02, direction="+x", name="launch", f0=9e9,
                          freqs=np.array([5e9, 8e9, 10e9]), ref_offset=3, probe_offset=10)
    cfg = sim.forward(n_steps=2, skip_preflight=True).waveguide_ports["launch"]
    extract = (lambda c: ws.extract_waveguide_sparams(c)[0]) if lane == "pair" else ws.extract_waveguide_s11
    below = np.asarray(cfg.freqs) <= cfg.f_cutoff
    assert below.any() and (~below).any()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reference = np.asarray(extract(cfg))
    np.testing.assert_array_equal(np.isnan(reference), below)
    assert np.isfinite(reference[~below]).all()
    notices = [w for w in caught if issubclass(w.category, WaveguideCutoffWarning)]
    assert len(notices) == 1
    assert "TE10" in str(notices[0].message) and "bins [0]" in str(notices[0].message)

    backend = (BackendClass("mps", True, "slab", "memory_stats_if_available", False)
               if policy == "mps" else BackendClass("gpu", True, "padded", "memory_stats", True))
    monkeypatch.setattr(_traced_check, "backend_class", lambda: backend)
    if policy == "gpu":
        def no_cpu():
            raise RuntimeError("No CPU device")
        monkeypatch.setattr(_traced_check, "host_cpu_device", no_cpu)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = np.asarray(jax.jit(lambda f: extract(cfg._replace(freqs=f)))(cfg.freqs))
        jax.effects_barrier()
    np.testing.assert_array_equal(np.isnan(got), below)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(reference))
    assert np.isfinite(got[~below]).all()
    np.testing.assert_allclose(got[~below], reference[~below], rtol=1e-5)
    notices = [w for w in caught if issubclass(w.category, WaveguideCutoffWarning)]
    assert len(notices) == 1
    assert notices[0].filename == __file__
    assert str(notices[0].message) == (
        "Waveguide S: an entry whose mode is at or below its own cutoff is NaN; this backend "
        "cannot list those ports and bins inside compiled code")


def test_defined_objective_gradient(manufactured, monkeypatch):
    cfgs = manufactured

    def objective(x):
        # The invalid denominator is tiny enough that its squared reciprocal
        # overflows in float32 reverse AD unless inputs are guarded first.
        a = jnp.array([1e-25, 1e-25, 2., 2.]) * x
        monkeypatch.setattr(ws, "_s_matrix_port_waves", lambda cfg, **kw: (a, jnp.ones(4)))
        s = ws.extract_waveguide_s_matrix(Record(shape=(5, 4, 4)), "device", cfgs, 4, cpml_axes="x")
        return jnp.real(s[:, :, -1]).sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = jax.grad(objective)(jnp.float32(1.))
    assert np.isfinite(got)
    np.testing.assert_allclose(got, -2., rtol=1e-6)


@pytest.mark.parametrize("normalize", [True, "flux"])
def test_nonuniform_assemblers(normalize, manufactured, monkeypatch):
    from rfx import Simulation
    import rfx.sparams.waveguide as calculator
    import rfx.runners.nonuniform as runner
    import rfx.preflight.waveguide as preflight
    import rfx.sources.waveguide_port as facade

    sim = Simulation(freq_max=9e9, domain=(.12, .024, .012), dx=.002,
                     dx_profile=np.full(60, .002), cpml_layers=8,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"})
    for name, x, direction in [("left", .02, "+x"), ("right", .10, "-x")]:
        sim.add_waveguide_port(x, direction=direction, name=name, f0=8e9,
                              freqs=manufactured[0].freqs)
    monkeypatch.setattr(runner, "_build_waveguide_port_config_nu",
                        lambda sim, entry, *a: manufactured[entry.name == "right"])
    monkeypatch.setattr(preflight, "resolve_preflight_port_configs", lambda sim, grid, cfgs, **kw: cfgs)
    monkeypatch.setattr(calculator, "_empty_waveguide_reference", lambda *a: sim)
    monkeypatch.setattr(calculator, "_warn_if_ringdown_truncated", lambda *a, **kw: None)
    monkeypatch.setattr(calculator, "_waveguide_s21_phase_residual", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(calculator, "resolve_reported_planes", lambda *a, **kw: Record(reference=0.))
    monkeypatch.setattr(facade, "waveguide_plane_positions", lambda cfg: {"reference": 0.})
    monkeypatch.setattr(facade, "settling_db_from_port_records", lambda *a, **kw: (-60., {}))
    monkeypatch.setattr("rfx.sparams._tail_witness.combine_witness_details", lambda *a: {})
    monkeypatch.setattr(facade, "extract_waveguide_port_waves", lambda cfg, **kw: (jnp.full(4, 2.), cfg.v_ref_t))
    calls = []

    def run(*a, **kw):
        ref = bool(len(calls) % 2)
        calls.append(ref)
        cfgs = {name: cfg._replace(v_ref_t=jnp.full(4, .5 if ref else 1.))
                for name, cfg in zip(("left", "right"), manufactured)}
        return Record(waveguide_ports=cfgs, waveguide_port_flux=[jnp.full(4, 4. if ref else 1.)]*2)

    monkeypatch.setattr(runner, "run_nonuniform_path", run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = calculator._compute_waveguide_s_matrix_nu(sim, n_steps=4, num_periods=1., normalize=normalize)
    check_warning(caught)
    s = np.asarray(result.s_params)
    for i in range(2):
        for j in range(2):
            defined = np.array([False, False, i == j == 0, True])
            assert np.isnan(s[i, j, ~defined]).all()
            expected = .5 if i != j else (.25 if normalize is True else np.sqrt(.75))
            np.testing.assert_allclose(s[i, j, defined], expected, rtol=2e-6)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_declared_run_forward_records(entry, monkeypatch):
    from rfx.measurement.declared import package_declared_measurements
    import rfx.sparams._waveguide_planes as planes

    cfg = record()
    sim = Record(_flux_monitors=[], _waveguide_ports=[Record(name="launch")])
    monkeypatch.setattr(ws, "_compute_beta", lambda *a, **kw: jnp.zeros(4))
    monkeypatch.setattr(ws, "_extract_port_waves_from_time_series", lambda *a: (jnp.full(4, 2.), jnp.ones(4)))
    monkeypatch.setattr(ws, "waveguide_plane_positions", lambda cfg: dict(source=0., reference=0., probe=.01))
    monkeypatch.setattr(planes, "resolve_reported_planes", lambda *a, **kw: (0., .01, "measured"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = package_declared_measurements(sim, Record(), entry=entry, flux_monitors=[], waveguide_cfgs=[cfg])
    check_warning(caught, ports=1)
    for key, expected in [("s11", .5), ("s21", 1.)]:
        value = getattr(got["waveguide_sparams"]["launch"], key)
        assert np.isnan(value[:2]).all()
        np.testing.assert_array_equal(value[2:], expected)

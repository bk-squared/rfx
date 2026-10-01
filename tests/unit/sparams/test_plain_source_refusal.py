"""Plain excitations cannot be part of a single-port-drive S experiment."""
import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from tests.contracts.path_disposition import PLAIN_SOURCE_S_REQUEST, REFUSES


def _model(wire=False, graded=False, source=True, boundary="pec"):
    profile = {"dz_profile": np.array([.001] * 4 + [.0005] * 8)} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     boundary=boundary, cpml_layers=2, **profile)
    sim.add_port((.004, .004, .003), "ez", **({"extent": .002} if wire else {}))
    sim.add_probe((.004, .004, .003), "ez")
    if source:
        sim.add_source((.002, .004, .003), "ez", amplitude_kind="field")
    return sim


def _before_step(*args, **kwargs):
    pytest.fail("reached field stepping with a plain-source S request")


@pytest.mark.parametrize("path,wire,scan_devices", [
    ("run_uniform", False, False), ("run_uniform", True, False),
    ("run_nonuniform", True, False), ("run_distributed", False, False),
    ("s_matrix_scan", False, False), ("s_matrix_scan", True, False),
    ("s_matrix_scan", False, True), ("s_matrix_scan", True, True),
])
@pytest.mark.parametrize("s_request", [None, True])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_plain_source_s_request_refused(
        monkeypatch, path, wire, s_request, scan_devices, boundary):
    assert PLAIN_SOURCE_S_REQUEST[path].kind == REFUSES
    sim = _model(wire, graded=path == "run_nonuniform", boundary=boundary)
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    message = (r"plain sources.*add_source.*Remove the plain source.*"
               r"run\(compute_s_params=False\).*raw port waves")
    with pytest.raises(NotImplementedError, match=message):
        if path == "s_matrix_scan":
            compute_lumped_wire_s_matrix_via_scan(
                sim, [5e9], n_steps=4,
                devices=jax.devices("cpu")[:2] if scan_devices else None)
        else:
            sim.run(n_steps=4, compute_s_params=s_request, skip_preflight=True,
                    devices=jax.devices("cpu")[:2] if path == "run_distributed" else None)


@pytest.mark.parametrize("graded,wire,distributed", [(False, False, False), (False, True, False),
                                                     (True, True, False), (False, False, True)])
def test_plain_source_without_s_request_runs(graded, wire, distributed):
    sim = _model(wire, graded)
    result = sim.run(n_steps=12, compute_s_params=False, skip_preflight=True,
                     devices=jax.devices("cpu")[:2] if distributed else None)
    assert result.s_params is None
    assert np.any(np.asarray(result.time_series))


@pytest.mark.parametrize("s_request", [None, False])
def test_source_only_run_is_unchanged(s_request):
    sim = _model()
    sim._ports = [pe for pe in sim._ports if pe.impedance == 0.0]
    result = sim.run(n_steps=12, compute_s_params=s_request, skip_preflight=True)
    assert result.s_params is None
    assert np.any(np.asarray(result.time_series))


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("s_request", [None, True])
def test_direct_runner_s_request_refused(monkeypatch, graded, s_request):
    from rfx.runners.nonuniform import run_nonuniform_path
    from rfx.runners.uniform import run_uniform

    sim = _model(wire=True, graded=graded)
    kwargs = {}
    if not graded:
        grid = sim._build_grid()
        mats, debye, lorentz, pec, shapes, _, kerr = sim._assemble_materials(grid)
        kwargs = dict(grid=grid, base_materials=mats, debye_spec=debye,
                      lorentz_spec=lorentz, pec_mask=pec, pec_shapes=shapes,
                      kerr_chi3=kerr)
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    runner = run_nonuniform_path if graded else run_uniform
    with pytest.raises(NotImplementedError, match="plain sources.*compute_s_params=False"):
        runner(sim, n_steps=4, compute_s_params=s_request, **kwargs)


def _subgrid_model(source=True):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .012), dx=.001, boundary="pec")
    sim.add_refinement(z_range=(.004, .008), ratio=2, validation="research")
    sim.add_port((.004, .004, .006), "ez", waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.004, .004, .006), "ez")
    if source:
        sim.add_source((.002, .004, .006), "ez", amplitude_kind="field")
    return sim


def test_subgridded_plain_source_s_request_refused(monkeypatch):
    assert PLAIN_SOURCE_S_REQUEST["run_subgridded"].kind == REFUSES
    sim = _subgrid_model()
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    with pytest.raises(NotImplementedError, match="plain sources"):
        sim.run(n_steps=4, compute_s_params=True, skip_preflight=True)


@pytest.mark.parametrize("graded,wire", [(False, False), (False, True), (True, True)])
def test_forward_port_s11_with_plain_source_refused(monkeypatch, graded, wire):
    path = "fwd_nonuniform" if graded else "fwd_uniform"
    assert PLAIN_SOURCE_S_REQUEST[path].kind == REFUSES
    sim = _model(wire, graded=graded)
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    with pytest.raises(NotImplementedError, match="plain sources"):
        sim.forward(n_steps=4, port_s11_freqs=[5e9], skip_preflight=True)


@pytest.mark.parametrize("graded,wire", [(False, False), (True, True)])
def test_forward_port_s11_without_plain_source_runs(graded, wire):
    sim = _model(wire, graded=graded, source=False)
    out = sim.forward(n_steps=12, port_s11_freqs=[5e9], skip_preflight=True)
    assert out.s_params is not None


def test_passive_port_illuminated_by_a_plain_source_is_admitted():
    """No port is driven, so the plain source contaminates no drive: the
    passive termination's reflection is a diagnostic and still runs (#1420)."""
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     boundary="cpml", cpml_layers=2)
    sim.add_port((.004, .004, .003), "ez", impedance=50.0, extent=.002, excite=False)
    sim.add_probe((.004, .004, .003), "ez")
    sim.add_source((.002, .004, .003), "ez", amplitude_kind="field")
    result = sim.run(n_steps=12, compute_s_params=True, s_param_freqs=[5e9],
                     skip_preflight=True)
    assert result.s_params is not None


def _two_passive_lumped(source=True):
    sim = Simulation(freq_max=10e9, domain=(.012, .008, .008), dx=.001,
                     boundary="cpml", cpml_layers=2)
    for x in (.004, .008):
        sim.add_port((x, .004, .004), "ez", impedance=50.0, excite=False)
    sim.add_probe((.006, .004, .004), "ez")
    if source:
        sim.add_source((.006, .004, .004), "ez", amplitude_kind="field")
    return sim


@pytest.mark.parametrize("route", ["run", "run_default", "scan", "run_devices"])
def test_passive_ports_on_drive_by_drive_routes_refused(monkeypatch, route):
    """Drive-by-drive S routes drive every impedance port, passive ones too, so
    a plain source fires in every drive even when no port is excite=True."""
    sim = _two_passive_lumped()
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    with pytest.raises(NotImplementedError, match="plain sources"):
        if route == "scan":
            compute_lumped_wire_s_matrix_via_scan(sim, [5e9], n_steps=4)
        else:
            sim.run(n_steps=4, skip_preflight=True,
                    compute_s_params=None if route == "run_default" else True,
                    devices=jax.devices("cpu")[:2] if route == "run_devices" else None)


def test_subgridded_passive_port_with_plain_source_refused(monkeypatch):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .012), dx=.001, boundary="pec")
    sim.add_refinement(z_range=(.004, .008), ratio=2, validation="research")
    sim.add_port((.004, .004, .006), "ez", excite=False)
    sim.add_probe((.004, .004, .006), "ez")
    sim.add_source((.002, .004, .006), "ez", amplitude_kind="field")
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    with pytest.raises(NotImplementedError, match="plain sources"):
        sim.run(n_steps=4, compute_s_params=True, skip_preflight=True)

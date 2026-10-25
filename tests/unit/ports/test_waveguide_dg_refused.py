"""#1942: refuse circular DG incident tables before stepping."""
from dataclasses import fields, replace

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._spec import _WaveguidePortEntry
from rfx.interop import design_to_dict, simulation_from_design

# Expected text is quoted from DESIGN_refuse.md, not read from implementation.
DG_REFUSAL = (
    "waveform='differentiated_gaussian' is refused on waveguide ports: its "
    "band-limited incident table is built with a zero-phase filter whose "
    "precursor wraps circularly to the end of the record, so the port "
    "re-injects at the record end (an empty matched WR-90 guide read S21 off "
    "by up to 11 dB). Use waveform='modulated_gaussian' (the default). "
    "Tracked in #1942."
)


def _guide(graded=False, internal=False):
    extra = {"dz_profile": np.full(5, .002032)} if graded else {}
    sim = Simulation(
        12e9, (.04, .02286, .01016), dx=.002032, cpml_layers=4,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, **extra,
        snap="declared")  # off-node WR-90 width is this fixture's aperture case (#1887)
    for x, direction in ((.008, "+x"), (.030, "-x")):
        sim.add_waveguide_port(x, direction=direction, f0=10e9,
                               freqs=np.array([9e9, 10e9, 11e9]),
                               probe_offset=2, ref_offset=1)
    if internal:
        # Build precisely the declaration record, omitting its waveform.
        sim._waveguide_ports = [
            _WaveguidePortEntry(**{
                f.name: getattr(e, f.name) for f in fields(e)
                if f.name != "waveform"})
            for e in sim._waveguide_ports]
    return sim


def _enter(sim, route):
    if route in ("run", "run_distributed", "forward", "forward_distributed"):
        sim._waveguide_ports = sim._waveguide_ports[:1]
    if route == "run":
        return sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    if route == "run_distributed":
        return sim.run(n_steps=2, devices=jax.devices(), skip_preflight=True,
                       compute_s_params=False)
    if route == "forward":
        return sim.forward(n_steps=2, skip_preflight=True)
    if route == "forward_distributed":
        return sim.forward(n_steps=2, distributed=True, devices=jax.devices(),
                           skip_preflight=True)
    if route == "preflight":
        return sim.preflight()
    return getattr(sim, route)(n_steps=2, normalize=True)


ROUTES = ["run", "run_distributed", "forward", "forward_distributed",
          "compute_waveguide_s_matrix", "compute_s_matrix"]


def _no_steps(*args, **kwargs):
    pytest.fail("refused waveguide reached stepping")


def test_declaration_refuses_dg(monkeypatch):
    monkeypatch.setattr(jax.lax, "scan", _no_steps)
    sim = _guide()
    with pytest.raises(ValueError) as caught:
        sim.add_waveguide_port(.008, waveform="differentiated_gaussian")
    assert str(caught.value) == DG_REFUSAL
    assert "modulated_gaussian" in str(caught.value)
    assert "#1942" in str(caught.value)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("route", ROUTES)
def test_internal_dg_refused_before_step(graded, route, monkeypatch):
    sim = _guide(graded, internal=True)
    sim._waveguide_ports = [
        replace(e, waveform="differentiated_gaussian")
        for e in sim._waveguide_ports]
    monkeypatch.setattr(jax.lax, "scan", _no_steps)
    with pytest.raises(ValueError) as caught:
        _enter(sim, route)
    assert DG_REFUSAL in str(caught.value)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("route", [*ROUTES, "preflight"])
def test_default_reaches_modulated_config(graded, internal, route, monkeypatch):
    sim = _guide(graded, internal)
    import rfx.api._compile as compile_module
    import rfx.model.waveguide_modes as modes
    import rfx.sources.waveguide_port as facade
    original = modes.init_waveguide_port
    seen = []

    class Built(BaseException):
        pass

    def capture(*args, **kwargs):
        cfg = original(*args, **kwargs)
        seen.append(cfg.waveform)
        raise Built

    for module in (compile_module, modes, facade):
        monkeypatch.setattr(module, "init_waveguide_port", capture)
    monkeypatch.setattr(jax.lax, "scan", _no_steps)
    if route == "preflight" and graded:
        _enter(sim, route)
        # An undrawn vacuum domain needs no section build; graded general
        # preflight retains its declared-geometry path in this case.
        assert seen == []
        assert all(e.waveform == "modulated_gaussian" for e in sim._waveguide_ports)
        return
    with pytest.raises(Built):
        _enter(sim, route)
    assert seen == ["modulated_gaussian"]


@pytest.mark.parametrize("graded", [False, True])
def test_preflight_dg_table_refused_without_steps(graded, monkeypatch):
    sim = _guide(graded)
    sim._waveguide_ports = [replace(e, waveform="differentiated_gaussian")
                            for e in sim._waveguide_ports]
    monkeypatch.setattr(jax.lax, "scan", _no_steps)
    report = sim.preflight()
    # Preflight catches failed advisory builds and reports them as unchecked.
    if not graded:
        assert any(DG_REFUSAL in str(issue) for issue in report)
    else:
        assert all(DG_REFUSAL not in str(issue) for issue in report)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("n_modes", [1, 2])
def test_direct_config_builder_refuses_dg_and_defaults_to_modulated(graded, n_modes):
    from rfx.runners.nonuniform import _build_waveguide_port_config_nu
    sim = _guide(graded, internal=True)
    entry = replace(sim._waveguide_ports[0], n_modes=n_modes)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()

    def build(e):
        if graded:
            return _build_waveguide_port_config_nu(sim, e, grid, e.freqs, 2)
        return sim._build_waveguide_port_config(e, grid, e.freqs, 2)

    with pytest.raises(ValueError) as caught:
        build(replace(entry, waveform="differentiated_gaussian"))
    assert str(caught.value) == DG_REFUSAL
    configs = build(entry)
    configs = configs if isinstance(configs, list) else [configs]
    assert len(configs) == n_modes
    assert all(c.waveform == "modulated_gaussian" for c in configs)


def test_interop_load_refuses_dg_before_step(monkeypatch):
    document = design_to_dict(_guide())
    document["excitations"]["waveguide_ports"][0]["waveform"] = "differentiated_gaussian"
    monkeypatch.setattr(jax.lax, "scan", _no_steps)
    with pytest.raises(ValueError) as caught:
        simulation_from_design(document)
    assert str(caught.value) == DG_REFUSAL


def test_interop_default_realizes_modulated():
    sim = simulation_from_design(design_to_dict(_guide()))
    grid = sim._build_grid()
    for entry in sim._waveguide_ports:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
        assert cfg.waveform == "modulated_gaussian"


@pytest.mark.parametrize("mode_profile", ["analytic", "discrete"])
@pytest.mark.parametrize("n_modes", [1, 2])
def test_graded_config_preserves_declared_mode_profile(mode_profile, n_modes, monkeypatch):
    from rfx.runners.nonuniform import _build_waveguide_port_config_nu

    sim = Simulation(
        12e9, (.04, .02286, .01016), dx=.002032, cpml_layers=4,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        dz_profile=np.array([.0015, .002, .003, .002, .00166]), snap="declared")  # (#1887)
    freqs = np.array([10e9])
    sim.add_waveguide_port(.008, f0=10e9, freqs=freqs,
                           probe_offset=2, ref_offset=1,
                           mode_profile=mode_profile, n_modes=n_modes)
    monkeypatch.setattr(jax.lax, "scan", _no_steps)
    cfg = _build_waveguide_port_config_nu(
        sim, sim._waveguide_ports[0], sim._build_nonuniform_grid(), freqs, 2)
    configs = cfg if isinstance(cfg, list) else [cfg]
    assert len(configs) == n_modes
    assert all(c.mode_profile == mode_profile for c in configs)


@pytest.mark.parametrize("waveform", ["modulated_gaussian", "differentiated_gaussian"])
def test_mixed_calculator_already_refuses_waveguides(waveform, monkeypatch):
    sim = _guide()
    sim._waveguide_ports = [replace(e, waveform=waveform)
                            for e in sim._waveguide_ports]
    # Existing family guard precedes assembly; declarations are borrowed from
    # their APIs so this is the same registration shape the calculator reads.
    other = Simulation(12e9, (.04, .02286, .01016), dx=.002032)
    other.add_port((.008, .010, .004), "ez", impedance=50.)
    other.add_msl_port(position=(.008, .010, .004), width=.004,
                       height=.002, impedance=50., eps_r_sub=2.)
    sim._ports, sim._msl_ports = other._ports, other._msl_ports
    monkeypatch.setattr(jax.lax, "scan", _no_steps)
    with pytest.raises(NotImplementedError, match="waveguide/Floquet"):
        sim.compute_mixed_s_matrix(n_steps=2, skip_preflight=True)

"""Every admitted plane wave launches the pulse recorded in its declaration."""

import numpy as np
import pytest

from rfx import Simulation
from rfx.interop import design_to_dict, simulation_from_design
from rfx.sources.tfsf import init_tfsf, tfsf_from_declaration
from rfx.sources.tfsf_2d import init_tfsf_2d
from rfx.sources.tfsf_discrete_plane_wave import init_plane_wave
from rfx.sources.tfsf_oblique_open import init_tfsf_methodB
from rfx.measurement.incident import measure_incident_spectrum


DG = "differentiated_gaussian"
MG = "modulated_gaussian"
TAU = 7.957747154594768e-11
DG_T0 = 2.3873241463784304e-10
MG_T0 = 3.978873577297384e-10


def _build(path, **kwargs):
    options = dict(f0=8e9, bandwidth=0.5, cpml_layers=4, tfsf_margin=4)
    options.update(kwargs)
    if path == "normal":
        return init_tfsf(61, 1e-3, 1e-12, **options)
    if path == "closed":
        return init_plane_wave(61, 1e-3, 1e-12, ny=49, nz=41, n_steps=16, **options)
    builder = init_tfsf_methodB if path == "methodB" else init_tfsf_2d
    return builder(61, 49, 1e-3, 1e-12, nz=3, theta_deg=30, **options)


def _scene(path, **kwargs):
    sim = Simulation(freq_max=16e9, domain=(.036, .024, .024 if path == "closed" else .003),
                     dx=.001, boundary="cpml", cpml_layers=8)
    options = dict(f0=8e9, bandwidth=.5, margin=4)
    if path in ("methodB", "bloch"):
        options.update(angle_deg=30, method=path)
    if path == "closed":
        options["closed_box"] = True
    options.update(kwargs)
    sim.add_tfsf_source(**options)
    return sim


@pytest.mark.parametrize("path,waveform,tau,t0", [
    ("normal", DG, TAU, DG_T0), ("normal", MG, TAU, MG_T0),
    ("normal", "continuous_wave", 1e-9, 1e-9),
    ("closed", DG, TAU, DG_T0), ("closed", MG, TAU, MG_T0),
    ("methodB", DG, TAU, DG_T0), ("methodB", MG, TAU, MG_T0),
    ("bloch", MG, TAU, MG_T0),
])
def test_p1_launched_source_fields(path, waveform, tau, t0):
    """Pulse width sets spectral width and its delay sets phase.

    Bloch launches the analytic MG with the same 5τ delay as every other MG path (#1831).
    """
    cfg, _ = _build(path, waveform=waveform)
    assert cfg.src_tau == tau
    assert cfg.src_t0 == t0
    if path == "bloch":
        assert not hasattr(cfg, "src_waveform")
    else:
        assert cfg.src_waveform == waveform


TAU_BW02 = 1.989436788648692e-10
DG_T0_BW02 = 5.968310365946076e-10
MG_T0_BW02 = 9.947183943243459e-10


@pytest.mark.parametrize("path,waveform,t0", [
    ("normal", DG, DG_T0_BW02), ("normal", MG, MG_T0_BW02),
    ("closed", DG, DG_T0_BW02), ("closed", MG, MG_T0_BW02),
    ("methodB", DG, DG_T0_BW02), ("methodB", MG, MG_T0_BW02),
    ("bloch", MG, MG_T0_BW02),
])
def test_p1_declared_bandwidth_sets_pulse_width(path, waveform, t0):
    """A narrower declared band means a longer pulse: τ = 1/(π f0 bandwidth) at bandwidth 0.2.

    The other pins use the builders' default bandwidth 0.5, which cannot tell a declared
    bandwidth from a dropped one.
    """
    cfg, _ = _build(path, waveform=waveform, bandwidth=0.2)
    assert cfg.src_tau == TAU_BW02
    assert cfg.src_t0 == t0


def test_p1_declared_bandwidth_reaches_methodb_through_the_declaration():
    """The declaration's bandwidth reaches method B's auxiliary line through the one builder."""
    sim = _scene("methodB", bandwidth=0.2)
    cfg, _ = tfsf_from_declaration(sim._tfsf, sim._build_grid(), sim._freq_max, n_steps=16)
    assert cfg.src_waveform == DG
    assert cfg.src_tau == TAU_BW02
    assert cfg.src_t0 == DG_T0_BW02


@pytest.mark.parametrize("path,expected,t0", [
    ("normal", DG, DG_T0), ("closed", DG, DG_T0),
    ("methodB", DG, DG_T0), ("bloch", MG, MG_T0),
])
def test_p2_default_is_declared_and_launched(path, expected, t0):
    """The default is DG except on Bloch's single transverse-wavenumber path.

    A serialized Bloch declaration therefore records the MG it launches.
    """
    sim = _scene(path)
    assert sim._tfsf.waveform == expected
    cfg, _ = tfsf_from_declaration(sim._tfsf, sim._build_grid(), sim._freq_max, n_steps=16)
    assert cfg.src_tau == TAU
    assert cfg.src_t0 == t0
    assert getattr(cfg, "src_waveform", MG) == expected
    cfg, _ = init_tfsf(61, .001, 1e-12, ny=49, nz=41, f0=8e9,
                       closed_box=path == "closed", n_steps=16,
                       angle_deg=30 if path in ("methodB", "bloch") else 0,
                       method="methodB" if path == "methodB" else "bloch")
    assert getattr(cfg, "src_waveform", MG) == expected
    assert cfg.src_tau == TAU
    assert cfg.src_t0 == t0
    if path == "bloch":
        document = design_to_dict(sim)
        assert document["excitations"]["tfsf"]["waveform"] == MG
        assert design_to_dict(simulation_from_design(document)) == document
        print('Bloch default round trip: waveform="modulated_gaussian"')


@pytest.mark.parametrize("entry", ["declaration", "init", "direct_bloch", "interop"])
def test_p3_explicit_bloch_dg_refused(entry):
    """A fixed k_y = k0(f0) sin θ carries an MG around f0, not baseband DG."""
    with pytest.raises(ValueError) as caught:
        if entry == "declaration":
            _scene("bloch", waveform=DG)
        elif entry == "init":
            init_tfsf(61, .001, 1e-12, ny=49, angle_deg=30, waveform=DG)
        elif entry == "direct_bloch":
            _build("bloch", waveform=DG)
        else:
            document = design_to_dict(_scene("bloch"))
            document["excitations"]["tfsf"]["waveform"] = DG
            simulation_from_design(document)
    message = str(caught.value)
    assert "k_y = k0(f0)·sin θ" in message
    assert 'waveform="modulated_gaussian"' in message
    assert 'method="methodB"' in message
    if entry == "interop":
        print(f"Bloch DG document refused: {message}")


def test_p4_measured_incident_spectrum_ratio():
    """E_DG(f)/E_MG(f) equals DFT[DG(n·dt)]/DFT[MG(n·dt)] over the same samples t = n·dt.

    The linear auxiliary-line launch factor cancels in this ratio.
    """
    n_steps, dt = 2000, 1e-12
    freqs = np.array([6e9, 7e9, 8e9, 9e9, 10e9])
    t = np.arange(n_steps) * dt
    dg_arg = (t - DG_T0) / TAU
    dg = -2 * dg_arg * np.exp(-dg_arg**2)
    mg = np.cos(2 * np.pi * 8e9 * (t - MG_T0)) * np.exp(-((t - MG_T0) / TAU)**2)
    assert abs(dg[-1]) < 1e-6 * np.max(abs(dg))
    assert abs(mg[-1]) < 1e-6 * np.max(abs(mg))
    kernel = np.exp(-2j * np.pi * freqs[:, None] * t)
    expected = (kernel @ dg) / (kernel @ mg)
    measured = []
    for waveform in (DG, MG):
        cfg, state = _build("methodB", waveform=waveform)
        measured.append(measure_incident_spectrum(cfg, state, n_steps=n_steps, freqs=freqs, dx=.001))
    actual = measured[0] / measured[1]
    error = np.abs(actual / expected - 1)
    for f, e in zip(freqs, error):
        print(f"P4 {f / 1e9:g} GHz: relative complex ratio error = {e:.12g}")
    assert np.all(error < 1e-3), dict(zip(freqs, error))


def test_p5_methodb_run_records_differ():
    """DG and carrier-modulated MG drive different fields in the issue's vacuum scene."""
    records = []
    for waveform in (DG, MG):
        sim = _scene("methodB", waveform=waveform)
        sim.add_probe((.018, .012, .0015), component="ez")
        records.append(np.asarray(sim.run(n_steps=1000).time_series))
    assert np.max(np.abs(records[0])) > 0
    assert np.max(np.abs(records[1])) > 0
    assert not np.array_equal(*records)


def test_p1_custom_waveform_identity_and_traced_timing():
    """A custom real drive retains its callable, while traced pulse width follows f0 and bandwidth."""
    import jax
    import jax.numpy as jnp
    from rfx.sources.sources import CustomWaveform
    from rfx.sources.tfsf import pulse_timing, resolve_waveform

    waveform = CustomWaveform(lambda t: jnp.exp(-t**2))
    assert resolve_waveform(waveform, 0, "bloch") is waveform
    timing = jax.jit(lambda f0, bw: jnp.stack(pulse_timing(waveform, f0, bw)))
    np.testing.assert_allclose(timing(8e9, .5), [TAU, DG_T0], rtol=1e-7, atol=0)

"""Name unsupported electric updates before an oblique Bloch wave can step."""
import gc

import jax
import numpy as np
import pytest

from rfx import Box, DebyePole, LorentzPole, Simulation
from rfx.errors import RfxUnsupported
from rfx.stepping import uniform_loop


INPUTS = ("diagonal", "full_tensor", "kottke_pec", "debye", "lorentz", "conformal", "occupancy")
CASES = [pytest.param(kind, entry, polarization, skip,
                      id=f"{kind}-{entry}-{polarization}-skip{skip}")
         for kind in INPUTS
         for entry in ("run", "forward")
         if kind != "occupancy" or entry == "forward"
         if kind != "conformal" or entry == "run"
         for polarization in ("ez", "ey")
         for skip in (False, True)]


@pytest.fixture(autouse=True)
def release_arrays():
    yield
    gc.collect()


def slab(kind, entry, polarization, skip, angle, *, steps=1):
    boundary = {"x": "cpml", "y": "periodic", "z": "periodic"}
    sim = Simulation(freq_max=10e9, domain=(.30, .016, .008), dx=.002,
                     mode="3d", cpml_layers=16, boundary=boundary,
                     **({"pec_occupancy_model": "kottke"} if kind == "occupancy" else {}))
    sim.add_tfsf_source(f0=5e9, bandwidth=.15, direction="+x", angle_deg=angle,
                        waveform="modulated_gaussian", polarization=polarization, method="bloch")
    grid = sim._build_grid()
    front = grid.node_of(0, 19) + .040
    props = {}
    if kind == "debye":
        props["debye_poles"] = [DebyePole(delta_eps=1., tau=1e-10)]
    if kind == "lorentz":
        props["lorentz_poles"] = [LorentzPole(omega_0=2*np.pi*8e9, delta=1e9, kappa=1e20)]
    sim.add_material("slab", eps_r=4., **props)
    sim.add(Box((front, 0, 0), (front+.024, .016, .008)),
            material="pec" if kind == "conformal" else "slab")
    sim.add_probe((front+.045, .3*.016, .004), component=polarization)
    smoothing = {"diagonal": True, "full_tensor": "full_tensor", "kottke_pec": "kottke_pec"}.get(kind, False)
    options = dict(n_steps=steps, skip_preflight=skip, subpixel_smoothing=smoothing)
    if kind == "conformal" and entry == "run":
        options["conformal_pec"] = True
    if kind == "occupancy":
        occupancy = np.zeros(grid.shape, dtype=np.float32)
        occupancy[39:51, :, :] = .5
        options["pec_occupancy_override"] = occupancy
    return sim, options


def expected_refusal(kind):
    clauses = {
        "diagonal": "subpixel smoothing (subpixel_smoothing=True)",
        "full_tensor": "subpixel smoothing (subpixel_smoothing='full_tensor')",
        "kottke_pec": "subpixel smoothing (subpixel_smoothing='kottke_pec')",
        "conformal": "conformal PEC",
        "occupancy": "a PEC occupancy override with pec_occupancy_model='kottke'",
        "debye": "a dispersive material (Debye or Lorentz poles)",
        "lorentz": "a dispersive material (Debye or Lorentz poles)",
    }
    routes = {
        "diagonal": ", or subpixel_smoothing=False",
        "full_tensor": ", or subpixel_smoothing=False",
        "kottke_pec": ", or subpixel_smoothing=False",
        "conformal": ", or conformal PEC off",
        "occupancy": ", or no pec_occupancy_override",
        "debye": ", or a material without poles",
        "lorentz": ", or a material without poles",
    }
    return (
        "An oblique Bloch plane wave (angle_deg=30.0, method='bloch') cannot be combined with "
        f"{clauses[kind]}: the electric-field update for it does not apply the Bloch phase, "
        f"so the fields would be wrong. Use normal incidence{routes[kind]}.")


@pytest.mark.parametrize("kind,entry,polarization,skip", CASES)
def test_oblique_material_refused_before_scan(kind, entry, polarization, skip, monkeypatch):
    calls = []

    def no_scan(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unsupported electric update reached a scan")

    monkeypatch.setattr(jax.lax, "scan", no_scan)
    sim, options = slab(kind, entry, polarization, skip, 30)
    with pytest.raises(RfxUnsupported) as caught:
        getattr(sim, entry)(**options)
    assert str(caught.value) == expected_refusal(kind)
    assert calls == []


class StepAdmitted(Exception):
    pass


@pytest.mark.parametrize("kind,entry,polarization,skip", CASES + [
    pytest.param("plain", entry, polarization, skip,
                 id=f"plain-{entry}-{polarization}-skip{skip}")
    for entry in ("run", "forward") for polarization in ("ez", "ey") for skip in (False, True)])
def test_supported_material_reaches_step(kind, entry, polarization, skip, monkeypatch):
    original = uniform_loop._build_step_setup
    completed = []

    def setup(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(True)
        return result

    def stop(*args, **kwargs):
        raise StepAdmitted

    monkeypatch.setattr(uniform_loop, "_build_step_setup", setup)
    monkeypatch.setattr(jax.lax, "scan", stop)
    sim, options = slab(kind, entry, polarization, skip, 30 if kind == "plain" else 0)
    with pytest.raises(StepAdmitted):
        getattr(sim, entry)(**options)
    assert completed == [True]


def test_low_level_oblique_tensor_refused_before_scan(monkeypatch):
    import rfx
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.sources.tfsf import init_tfsf

    grid = Grid(freq_max=5e9, domain=(0.06, 0.06, 0.006), dx=0.004,
                cpml_layers=6)
    materials = init_materials(grid.shape)
    cfg, state = init_tfsf(grid.nx, grid.dx, grid.dt,
                           cpml_layers=grid.cpml_layers, tfsf_margin=3,
                           f0=5e9, bandwidth=0.2, amplitude=1.0,
                           polarization="ez", angle_deg=30.0, ny=grid.ny)
    eps_x = materials.eps_r.at[cfg.x_lo + 2:cfg.x_hi - 1, :, :].set(2.0)
    tensor = (eps_x, materials.eps_r, materials.eps_r)
    calls = []

    def no_scan(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unsupported electric update reached a scan")

    monkeypatch.setattr(jax.lax, "scan", no_scan)
    with pytest.raises(RfxUnsupported) as caught:
        rfx.run(grid, materials, 8, boundary="cpml", cpml_axes="x",
                periodic=(False, True, True), tfsf=(cfg, state), aniso_eps=tensor)
    assert str(caught.value) == (
        "An oblique Bloch plane wave (angle_deg=30.0, method='bloch') cannot be combined with "
        "an anisotropic permittivity tensor (aniso_eps or aniso_inv_eps): the electric-field "
        "update for it does not apply the Bloch phase, so the fields would be wrong. "
        "Use normal incidence, or no anisotropic permittivity tensor.")
    assert calls == []

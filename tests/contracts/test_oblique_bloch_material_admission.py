"""Admit phase-aware electric updates and name the remaining Bloch restrictions."""
import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, LorentzPole, Simulation
from rfx.errors import RfxUnsupported
from rfx.stepping import uniform_loop


INPUTS = ("diagonal", "full_tensor", "kottke_pec", "debye", "lorentz", "conformal", "occupancy", "kottke_shape")
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
    sim = Simulation(freq_max=10e9, domain=((.30, .016, .008) if polarization == "ez" else (.30, .008, .016)), dx=.002,
                     mode="3d", cpml_layers=16, boundary=boundary,
                     **({"pec_occupancy_model": "kottke"} if kind == "occupancy" else {}))
    sim.add_tfsf_source(f0=5e9, bandwidth=.15, direction="+x", angle_deg=angle,
                        waveform="modulated_gaussian", polarization=polarization, method="bloch")
    grid = sim._build_grid()
    front = grid.node_of(0, 19) + .040
    props = {}
    if kind in ("debye", "mixed"):
        props["debye_poles"] = [DebyePole(delta_eps=1., tau=1e-10)]
    if kind in ("lorentz", "mixed"):
        props["lorentz_poles"] = [LorentzPole(omega_0=2*np.pi*8e9, delta=1e9, kappa=1e20)]
    sim.add_material("slab", eps_r=4., **props)
    domain = (.30, .016, .008) if polarization == "ez" else (.30, .008, .016)
    sim.add(Box((front, 0, 0), (front+.024, domain[1], domain[2])),
            material="pec" if kind in ("conformal", "kottke_shape") else "slab")
    sim.add_probe((front+.045, .3*.016, .004), component=polarization)
    smoothing = {"diagonal": True, "full_tensor": "full_tensor", "kottke_pec": "kottke_pec",
                 "kottke_shape": "kottke_pec"}.get(kind, False)
    options = dict(n_steps=steps, skip_preflight=skip, subpixel_smoothing=smoothing)
    if kind == "conformal" and entry == "run":
        options["conformal_pec"] = True
    if kind == "occupancy":
        occupancy = np.zeros(grid.shape, dtype=np.float32)
        occupancy[39:51, :, :] = .5
        options["pec_occupancy_override"] = occupancy
    return sim, options


REFUSALS = {
    "full_tensor": (
        "An oblique Bloch plane wave (angle_deg=30.0, method='bloch') cannot be combined with "
        "off-diagonal node coupling (node_smoothing): the off-diagonal node coupling "
        "does not apply the Bloch phase, so the fields would be wrong. "
        "Use normal incidence, or no off-diagonal node coupling."),
    "conformal": (
        "An oblique Bloch plane wave (angle_deg=30.0, method='bloch') cannot be combined with "
        "conformal PEC: the electric-field update for it does not apply the Bloch phase, "
        "so the fields would be wrong. Use normal incidence, or conformal PEC off."),
    "occupancy": (
        "An oblique Bloch plane wave (angle_deg=30.0, method='bloch') cannot be combined with "
        "a PEC occupancy override with pec_occupancy_model='kottke': the electric-field "
        "update for it does not apply the Bloch phase, so the fields would be wrong. "
        "Use normal incidence, or no pec_occupancy_override."),
    "kottke_shape": (
        "An oblique Bloch plane wave (angle_deg=30.0, method='bloch') cannot be combined with "
        "subpixel smoothing with a PEC shape: the electric-field update for it does not apply "
        "the Bloch phase, so the fields would be wrong. Use normal incidence, or subpixel_smoothing=False."),
}


class StepAdmitted(Exception):
    pass


@pytest.mark.parametrize("kind,entry,polarization,skip", CASES)
def test_oblique_material_admission_before_scan(kind, entry, polarization, skip, monkeypatch):
    calls = []

    def no_scan(*args, **kwargs):
        calls.append(True)
        raise StepAdmitted

    monkeypatch.setattr(jax.lax, "scan", no_scan)
    sim, options = slab(kind, entry, polarization, skip, 30)
    if kind in REFUSALS:
        with pytest.raises(RfxUnsupported) as caught:
            getattr(sim, entry)(**options)
        assert str(caught.value) == REFUSALS[kind]
        assert calls == []
    else:
        with pytest.raises(StepAdmitted):
            getattr(sim, entry)(**options)
        assert calls == [True]


@pytest.mark.parametrize("kind,skip,override", [
    pytest.param(kind, skip, override, id=f"{kind}-override-{override}-skip{skip}")
    for kind in ("diagonal", "kottke_pec", "debye", "full_tensor", "occupancy")
    for skip in (False, True)
    for override in ("concrete", "traced")])
def test_forward_angle_override_material_admission(kind, skip, override, monkeypatch):
    """Concrete and traced angle overrides retain the same material admission."""
    calls = []

    def no_scan(*args, **kwargs):
        calls.append(True)
        raise StepAdmitted

    monkeypatch.setattr(jax.lax, "scan", no_scan)
    sim, options = slab(kind, "forward", "ez", skip, 30)
    with pytest.raises(RfxUnsupported if kind in REFUSALS else StepAdmitted) as caught:
        if override == "concrete":
            sim.forward(**options, tfsf_angle_override=20.0)
        else:
            jax.grad(lambda a: jnp.sum(sim.forward(**options, tfsf_angle_override=a).time_series))(20.0)
    if kind in REFUSALS:
        assert str(caught.value) == REFUSALS[kind]
        assert calls == []
    else:
        assert calls == [True]


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



def low_level(angle=30.):
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.sources.tfsf import init_tfsf

    grid = Grid(freq_max=5e9, domain=(.06, .06, .006), dx=.004, cpml_layers=6)
    materials = init_materials(grid.shape)
    cfg, state = init_tfsf(grid.nx, grid.dx, grid.dt,
                           cpml_layers=grid.cpml_layers, tfsf_margin=3,
                           f0=5e9, bandwidth=.2, amplitude=1.,
                           polarization="ez", angle_deg=angle, ny=grid.ny)
    eps_x = materials.eps_r.at[cfg.x_lo + 2:cfg.x_hi - 1, :, :].set(2.)
    tensor = (eps_x, materials.eps_r, materials.eps_r)
    options = dict(boundary="cpml", cpml_axes="x", periodic=(False, True, True), tfsf=(cfg, state))
    return grid, materials, tensor, options


@pytest.mark.parametrize("inverse", [False, True])
def test_low_level_oblique_diagonal_tensor_admitted(inverse, monkeypatch):
    import rfx

    grid, materials, tensor, options = low_level()
    options["aniso_inv_eps" if inverse else "aniso_eps"] = tuple(1/e for e in tensor) if inverse else tensor
    def stop(*args, **kwargs):
        raise StepAdmitted
    monkeypatch.setattr(jax.lax, "scan", stop)
    with pytest.raises(StepAdmitted):
        rfx.run(grid, materials, 2, **options)


def test_low_level_node_tensor_refused(monkeypatch):
    import rfx
    from rfx.geometry.node_tensor import compute_node_tensor_eps

    grid, materials, _, options = low_level()
    front = grid.node_of(0, 12)
    tensor = compute_node_tensor_eps(grid, [(Box((front, 0, 0), (front+.008, .06, .006)), 2.)])
    def stop(*args, **kwargs):
        raise AssertionError("node coupling reached a scan")
    monkeypatch.setattr(jax.lax, "scan", stop)
    with pytest.raises(RfxUnsupported) as caught:
        rfx.run(grid, materials, 2, aniso_eps=tensor, **options)
    assert str(caught.value) == REFUSALS["full_tensor"]


@pytest.mark.parametrize("angle", [0., 30.])
@pytest.mark.parametrize("inverse", [False, True])
def test_low_level_poles_with_tensor_refused(angle, inverse, monkeypatch):
    import rfx
    from rfx.materials.debye import init_debye

    grid, materials, tensor, options = low_level(angle)
    poles = init_debye([DebyePole(1., 1e-10)], materials, grid.dt,
                       periodic=(False, True, True), grid=grid)
    options["aniso_inv_eps" if inverse else "aniso_eps"] = tuple(1/e for e in tensor) if inverse else tensor
    def stop(*args, **kwargs):
        raise AssertionError("poles with a tensor reached a scan")
    monkeypatch.setattr(jax.lax, "scan", stop)
    with pytest.raises(RfxUnsupported) as caught:
        rfx.run(grid, materials, 2, debye=poles, **options)
    assert str(caught.value) == (
        "Debye or Lorentz poles cannot be combined with a permittivity tensor: "
        "the dispersive electric-field update does not use the tensor. "
        "Use a material without poles, or no permittivity tensor.")


@pytest.mark.parametrize("pec_input", ["mask", "occupancy"])
def test_low_level_kottke_with_pec_refused(pec_input, monkeypatch):
    import rfx

    grid, materials, tensor, options = low_level()
    mask = jnp.zeros(grid.shape, dtype=bool).at[12:14, :, :].set(True)
    options["pec_mask" if pec_input == "mask" else "pec_occupancy"] = mask if pec_input == "mask" else mask.astype(jnp.float32)
    def stop(*args, **kwargs):
        raise AssertionError("PEC smoothing reached a scan")
    monkeypatch.setattr(jax.lax, "scan", stop)
    with pytest.raises(RfxUnsupported) as caught:
        rfx.run(grid, materials, 2, aniso_inv_eps=tuple(1/e for e in tensor),
                subpixel_smoothing="kottke_pec", **options)
    assert str(caught.value) == REFUSALS["kottke_shape"]


def test_smoothed_dielectric_with_conformal_pec_keeps_only_pec_refusal(monkeypatch):
    sim, options = slab("conformal", "run", "ez", False, 30)
    sim.add(Box((.09, 0, 0), (.11, .016, .008)), material="slab")
    options["subpixel_smoothing"] = True
    def stop(*args, **kwargs):
        raise AssertionError("conformal PEC reached a scan")
    monkeypatch.setattr(jax.lax, "scan", stop)
    with pytest.raises(RfxUnsupported) as caught:
        sim.run(**options)
    assert str(caught.value) == REFUSALS["conformal"]


@pytest.mark.parametrize("kind", ["debye", "lorentz", "mixed"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_oblique_pole_carry_survives_real_scan(kind, entry, monkeypatch):
    from rfx.stepping import prepared

    original = uniform_loop._build_step_setup
    original_advance = prepared.advance
    carries = []
    fields = []
    def observe(*args, **kwargs):
        setup = original(*args, **kwargs)
        carries.append(setup.carry_init)
        return setup
    def observe_fields(product, stepper, *args, **kwargs):
        def recording(*a, **kw):
            raw = stepper(*a, **dict(kw, return_state=True))
            fields.append(raw.state)
            return raw
        return original_advance(product, recording, *args, **kwargs)
    monkeypatch.setattr(uniform_loop, "_build_step_setup", observe)
    monkeypatch.setattr(prepared, "advance", observe_fields)
    sim, options = slab(kind, entry, "ez", False, 30, steps=2)
    result = getattr(sim, entry)(**options)
    assert np.isfinite(np.asarray(result.time_series)).all()
    assert len(fields) == 1
    assert all(np.isfinite(np.asarray(getattr(fields[0], c))).all()
               for c in ("ex", "ey", "ez", "hx", "hy", "hz"))
    assert len(carries) == 1
    names = ("debye", "lorentz") if kind == "mixed" else (kind,)
    for name in names:
        assert all(jnp.issubdtype(a.dtype, jnp.complexfloating) for a in jax.tree.leaves(carries[0][name]))


@pytest.mark.parametrize("kind", ["diagonal", "kottke_pec"])
def test_low_level_realized_tensor_matches_high_level_probe_record(kind, monkeypatch):
    import rfx
    from rfx.stepping import prepared

    original = prepared.advance
    captured = []
    def observe(product, stepper, *args, **kwargs):
        def recording(*a, **kw):
            result = stepper(*a, **kw)
            captured.append((a, kw, result))
            return result
        return original(product, recording, *args, **kwargs)
    monkeypatch.setattr(prepared, "advance", observe)
    sim, options = slab(kind, "run", "ez", False, 30, steps=512)
    sim.run(**options)
    assert len(captured) == 1
    args, kwargs, reference = captured[0]
    kwargs = dict(kwargs, subpixel_smoothing=False)
    actual = rfx.run(*args, **kwargs)
    assert np.asarray(actual.time_series).tobytes() == np.asarray(reference.time_series).tobytes()


@pytest.mark.parametrize("kind,pol,angle,digest", [
    ('plain', 'ez', 30, '5c0b0fa223c360448be858d0298a30f4bb0e6a645b9f947433fea5d850e30f27'),
    ('plain', 'ey', 30, '5c0b0fa223c360448be858d0298a30f4bb0e6a645b9f947433fea5d850e30f27'),
    ('diagonal', 'ez', 0, 'a537be2179a2fd5d1f44280478427dea6ab14625e2a7b463df98021cc56c64b8'),
    ('debye', 'ez', 0, 'f09bc39f1e0095e1e585cdfe5dd05e4afa593b5c8cad3c5547022f0b93b73efe'),
])
def test_supported_probe_record_bytes_unchanged(kind, pol, angle, digest):
    """Hash every byte of the complete float32 record, including its settled tail."""
    import hashlib

    sim, options = slab(kind, "run", pol, False, angle, steps=1798)
    data = np.asarray(sim.run(**options).time_series)
    assert data.dtype == np.float32
    assert data.shape == (1798, 1)
    assert hashlib.sha256(data.tobytes()).hexdigest() == digest

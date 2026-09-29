"""Off-grid PEC walls must be carried or refused before fields advance.

The 12.3 x 13.1 x 12.7 mm box has 1 mm cells and an off-centre PEC
sphere. Identity comparisons use zero tolerance on raw field arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners.uniform import run_uniform
from rfx.topology import TopologyDesignRegion, topology_optimize
from rfx.vmap_sweep import vmap_material_sweep

N_STEPS = 96


def _sim(
    conformal=False, *, ports=None, solver="yee", tfsf=False, absorber=None, kappa=1
):
    walls = BoundarySpec(
        x=Boundary(lo="pec", hi="pec", conformal=conformal), y="pec", z="pec"
    )
    if tfsf:
        walls = BoundarySpec(
            x="cpml", y=Boundary(lo="pec", hi="pec", conformal=conformal), z="cpml"
        )
    sim = Simulation(
        freq_max=10e9,
        domain=(0.0123, 0.0131, 0.0127),
        dx=0.001,
        boundary=absorber or walls,
        solver=solver,
        cpml_layers=4,
        cpml_kappa_max=kappa,
    )
    if solver != "adi":
        sim.add(Sphere(center=(0.0061, 0.0062, 0.0064), radius=0.0012), material="pec")
    sim.add_material("design", eps_r=2.0)
    if not tfsf:
        sim.add(Box((0.002, 0.002, 0.002), (0.003, 0.004, 0.004)), material="design")
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    if tfsf:
        sim.add_tfsf_source(f0=5e9, bandwidth=0.8, margin=3)
    elif ports:
        for x in (0.004, 0.008):
            sim.add_port(
                (x, 0.006, 0.006),
                "ez",
                impedance=50.0,
                waveform=pulse,
                **({"extent": 0.002} if ports == "wire" else {}),
            )
    else:
        sim.add_source(
            (0.004, 0.006, 0.006), "ez", waveform=pulse, amplitude_kind="field"
        )
    sim.add_probe((0.008, 0.006, 0.006), "ez")
    return sim


def _direct(sim, **kwargs):
    grid = sim._build_grid()
    mats, debye, lorentz, pec, shapes, _, kerr = sim._assemble_materials(grid)
    return run_uniform(
        sim,
        n_steps=N_STEPS,
        grid=grid,
        base_materials=mats,
        debye_spec=debye,
        lorentz_spec=lorentz,
        pec_mask=pec,
        pec_shapes=shapes,
        kerr_chi3=kerr,
        **kwargs,
    )


def _mixed(on):
    dx = 0.0005
    sim = Simulation(
        freq_max=10e9,
        domain=(0.0123, 0.0061, 0.0047),
        dx=dx,
        cpml_layers=4,
        boundary=BoundarySpec(
            x="cpml", y=Boundary(lo="pec", hi="pec", conformal=on), z="cpml"
        ),
    )
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0.001), (0.0123, 0.0061, 0.002)), material="sub")
    sim.add(Box((0, 0, 0.001), (0.0123, 0.0061, 0.001)), material="pec")
    sim.add(Box((0, 0.0025, 0.002), (0.0123, 0.0035, 0.002)), material="pec")
    sim.add_port(
        (0.003, 0.003, 0.001),
        "ez",
        impedance=50.0,
        extent=0.001,
        terminates=1,
        waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
    )
    sim.add_msl_port(
        position=(0.009, 0.003, 0.001),
        width=0.001,
        height=0.001,
        direction="-x",
        impedance=50.0,
        eps_r_sub=2.2,
        n_probe_offset=3,
        n_probe_spacing=2,
        waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
    )
    return sim


@pytest.fixture
def no_steps(monkeypatch):
    calls = []

    def scan(*args, **kwargs):
        calls.append(1)
        raise AssertionError("A time-step scan started before refusal")

    monkeypatch.setattr(jax.lax, "scan", scan)
    yield calls
    assert len(calls) == 0


@pytest.mark.parametrize("route", ["topology", "mixed", "vmap"])
def test_calculator_refuses_conformal_before_steps(route, no_steps):
    sim = _mixed(True) if route == "mixed" else _sim(True)
    with pytest.raises(NotImplementedError):
        if route == "topology":
            region = TopologyDesignRegion(
                corner_lo=(0.002, 0.002, 0.002),
                corner_hi=(0.003, 0.004, 0.004),
                material_bg="air",
                material_fg="design",
            )
            topology_optimize(
                sim,
                region,
                lambda r: jnp.sum(r.time_series**2),
                n_iterations=1,
                verbose=False,
                skip_preflight=True,
            )
        elif route == "mixed":
            sim.compute_mixed_s_matrix(
                n_steps=N_STEPS, skip_preflight=True, magnitude_channel="wave"
            )
        else:
            vmap_material_sweep(sim, "design.eps_r", [2.0, 3.0], n_steps=N_STEPS)


def test_adi_forward_refuses_mu_override_before_steps(no_steps):
    sim = _sim(solver="adi")
    with pytest.raises(NotImplementedError):
        sim.forward(
            n_steps=N_STEPS,
            skip_preflight=True,
            mu_r_override=jnp.full(sim._build_grid().shape, 4.0),
        )


@pytest.mark.parametrize("route", ["run", "forward", "direct", "low_level"])
def test_upml_refuses_kappa_before_steps(route, no_steps):
    sim = _sim(absorber="upml", kappa=5)
    with pytest.raises(NotImplementedError):
        if route in ("run", "forward"):
            getattr(sim, route)(n_steps=N_STEPS, skip_preflight=True)
        elif route == "direct":
            _direct(sim)
        else:
            from rfx.boundaries.upml import init_upml

            grid = sim._build_grid()
            init_upml(grid, sim._assemble_materials(grid)[0])


@pytest.mark.parametrize("ports", ["lumped", "wire"])
@pytest.mark.parametrize("declaration,argument", [(False, True), (True, None)])
@pytest.mark.parametrize("compute", [None, True])
def test_run_refuses_conformal_s_matrix_before_steps(
    ports, declaration, argument, compute, no_steps
):
    sim = _sim(declaration, ports=ports)
    with pytest.raises(NotImplementedError):
        sim.run(
            n_steps=N_STEPS,
            skip_preflight=True,
            conformal_pec=argument,
            compute_s_params=compute,
            s_param_n_steps=N_STEPS,
            s_param_freqs=jnp.array([4e9, 5e9, 6e9]),
        )


def _identical(a, b):
    np.testing.assert_array_equal(a.time_series, b.time_series)
    assert float(np.max(np.abs(a.time_series))) > 0
    for field in ("ex", "ey", "ez", "hx", "hy", "hz"):
        np.testing.assert_array_equal(getattr(a.state, field), getattr(b.state, field))


def test_direct_uniform_default_reads_conformal_declaration():
    sim = _sim(True)
    declared = sim.run(n_steps=N_STEPS, skip_preflight=True, compute_s_params=False)
    direct = _direct(sim, compute_s_params=False)
    staircase = _direct(sim, conformal_pec=False, compute_s_params=False)
    _identical(declared, direct)
    # This geometry must resolve the difference; zero is the rounding floor
    # for identity on the same host and interpreter.
    assert np.max(np.abs(declared.time_series - staircase.time_series)) > 0
    _identical(
        staircase,
        sim.run(
            n_steps=N_STEPS,
            skip_preflight=True,
            compute_s_params=False,
            conformal_pec=False,
        ),
    )


@pytest.mark.parametrize("ports", ["lumped", "wire"])
def test_explicit_staircase_s_matrix_is_allowed(ports):
    sim = _sim(True, ports=ports)
    kw = dict(
        n_steps=N_STEPS,
        skip_preflight=True,
        conformal_pec=False,
        s_param_n_steps=N_STEPS,
        s_param_freqs=jnp.array([4e9, 5e9, 6e9]),
    )
    a = sim.run(**kw)
    # The declaration also supplies off-grid boundary shapes to assembly;
    # compare the same declared structure through the direct entry point.
    d = _direct(
        sim,
        conformal_pec=False,
        s_param_n_steps=N_STEPS,
        s_param_freqs=kw["s_param_freqs"],
    )
    _identical(a, d)
    np.testing.assert_array_equal(a.s_params, d.s_params)


@pytest.mark.parametrize("conformal", [False, True])
def test_device_fallback_preserves_explicit_conformal(conformal, two_devices):
    sim = _sim(True, tfsf=True)
    expected = sim.run(n_steps=N_STEPS, skip_preflight=True, conformal_pec=conformal)
    got = sim.run(
        n_steps=N_STEPS,
        skip_preflight=True,
        conformal_pec=conformal,
        devices=two_devices,
    )
    _identical(expected, got)


@pytest.mark.parametrize("ports", ["lumped", "wire"])
def test_conformal_probe_fields_without_s_matrix_are_allowed(ports):
    sim = _sim(True, ports=ports)
    got = sim.run(n_steps=N_STEPS, skip_preflight=True, compute_s_params=False)
    direct = _direct(sim, compute_s_params=False)
    _identical(got, direct)
    assert got.s_params is None


def test_waveguide_device_fallback_preserves_staircase(two_devices):
    sim = Simulation(
        freq_max=20e9,
        domain=(0.0243, 0.0121, 0.0067),
        dx=0.001,
        cpml_layers=4,
        boundary=BoundarySpec(
            x="cpml", y=Boundary(lo="pec", hi="pec", conformal=True), z="pec"
        ),
    )
    sim.add_waveguide_port(
        0.004,
        direction="+x",
        freqs=jnp.array([15e9, 18e9]),
        f0=16e9,
        probe_offset=2,
        ref_offset=1,
    )
    sim.add_probe((0.016, 0.006, 0.003), "ez")
    expected = sim.run(n_steps=N_STEPS, skip_preflight=True, conformal_pec=False)
    got = sim.run(
        n_steps=N_STEPS, skip_preflight=True, conformal_pec=False, devices=two_devices
    )
    _identical(expected, got)


@pytest.mark.parametrize(
    "lane", ["run_nonuniform", "run_adi", "run_subgridded", "run_distributed"]
)
def test_explicit_staircase_is_allowed_on_every_run_lane(lane, two_devices):
    # Reuse the established 12 mm box (20 mm tall with subgridding).
    from tests.unit.runners.test_path_disposition_cells import _base

    results = []
    for conformal in (False, True):
        sim = _base(
            lane,
            boundary=BoundarySpec(
                x=Boundary(lo="pec", hi="pec", conformal=conformal), y="pec", z="pec"
            ),
        )
        kwargs = dict(
            n_steps=48, skip_preflight=True, conformal_pec=False, compute_s_params=False
        )
        if lane == "run_distributed":
            kwargs["devices"] = two_devices
        results.append(sim.run(**kwargs))
    _identical(*results)

"""Radius ports admit remote pole volumes and refuse their electric stencil."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx import Box, DebyePole, GaussianPulse, LorentzPole, Simulation, Sphere
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.materials.debye import init_debye
from rfx.materials.lorentz import init_lorentz
from rfx.materials.lorentz import drude_pole
from rfx.errors import RfxRefusal
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from rfx.sources.sources import WirePort, setup_wire_port
from rfx.stepping.graded_loop import run_nonuniform


D = 1e-3
STEPS = 160
FREQS = np.array([4e9, 6e9, 8e9])
LANES = (
    "run_uniform", "fwd_uniform", "run_nonuniform", "fwd_nonuniform",
    "run_distributed", "fwd_distributed", "run_distributed_nu", "fwd_distributed_nu",
    "s_matrix_scan", "s_matrix_scan_nu", "topology_optimize",
)


def scene(kind, *, graded=False, near=False, component="ez"):
    profiles = {}
    if graded:
        widths = np.full(12, D)
        widths[9:11] = [.9*D, 1.1*D]
        profiles["dx_profile"] = widths
    sim = Simulation(freq_max=15e9, domain=(12*D, 10*D, 8*D), dx=D,
                     boundary="pec", cpml_layers=0, **profiles)
    poles = {}
    if kind == "drude":
        poles["lorentz_poles"] = [drude_pole(omega_p=0., gamma=1e9)]
    elif kind == "lorentz":
        poles["lorentz_poles"] = [LorentzPole(omega_0=2*np.pi*12e9, delta=1e9, kappa=0.)]
    elif kind == "debye":
        poles["debye_poles"] = [DebyePole(delta_eps=0., tau=1e-11)]
    sim.add_material("block", eps_r=2., **poles)
    lo, hi = ((4*D, 3*D, 3*D), (5*D, 4*D, 4*D)) if near else (
        (8*D, 6*D, 2*D), (11*D, 8*D, 6*D))
    sim.add(Box(lo, hi), material="block")
    sim.add_port((3*D, 3*D, 3*D), component, extent=2*D, impedance=50.,
                 radius=.1*D, waveform=GaussianPulse(f0=6e9, bandwidth=.9))
    sim.add_probe((4*D, 3*D, 4*D), component)
    return sim


def records(result):
    return tuple(np.stack([np.asarray(record[name]) for name in ("V", "I", "V_port")], axis=1)
                 for record in result.sparam_time_records)


def solve(sim, lane, monkeypatch):
    if lane == "topology_optimize":
        from rfx.topology import TopologyDesignRegion, topology_optimize
        captured = []

        def objective(result):
            for record in result.sparam_time_records:
                jax.debug.callback(lambda a: captured.append(np.asarray(a)),
                                   jnp.stack([record[name] for name in ("V", "I", "V_port")], axis=1))
            return jnp.mean(result.time_series**2)

        region = TopologyDesignRegion((8*D, D, D), (9*D, 2*D, 2*D),
                                      material_bg="air", material_fg="air", filter_radius=None)
        topology_optimize(sim, region, objective, n_iterations=1, jit=False,
                          verbose=False, skip_preflight=True)
        jax.effects_barrier()
        assert captured
        return tuple(captured)
    if lane.startswith("s_matrix_scan"):
        from rfx.runners import nonuniform
        captured = []
        owner = nonuniform if lane.endswith("_nu") else sim
        name = "run_nonuniform_path" if lane.endswith("_nu") else "_forward_from_materials"
        original = getattr(owner, name)

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            rs = result["sparam_time_records"] if isinstance(result, dict) else result.sparam_time_records
            captured.extend(np.stack([np.asarray(r[n]) for n in ("V", "I", "V_port")], axis=1) for r in rs)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(owner, name, observe)
            compute_lumped_wire_s_matrix_via_scan(sim, FREQS, n_steps=STEPS, skip_preflight=True)
        assert captured
        return tuple(captured)
    distributed = "distributed" in lane
    options = dict(n_steps=STEPS, skip_preflight=True)
    if distributed:
        options["devices"] = jax.devices("cpu")[:2]
        assert len(options["devices"]) == 2
    if lane.startswith("run"):
        result = sim.run(**options, compute_s_params=False)
    else:
        result = sim.forward(**options, checkpoint=False, distributed=distributed)
    return records(result)


def assert_records(expected, actual):
    assert len(expected) == len(actual) > 0
    for reference, value in zip(expected, actual, strict=True):
        assert reference.shape == value.shape
        assert np.isfinite(reference).all() and np.isfinite(value).all()
        for channel in range(reference.shape[1]):
            a, b = reference[:, channel].astype(float), value[:, channel].astype(float)
            peak = np.abs(a)
            assert np.max(peak) > 0, ("unexcited V/I channel", channel)
            allowance = 1.2e-7*np.cumsum(peak)
            error = np.abs(a-b)
            assert np.all(error <= allowance), (
                "pole-media accumulated V/I law", channel,
                int(np.argmax(error-allowance)), float(np.max(error-allowance)))
            fa, fb = np.fft.rfft(a), np.fft.rfft(b)
            assert np.max(np.abs(fa-fb)) <= 1e-4*np.max(np.abs(fa)), (
                "pole-media V/I DFT law", channel)


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("kind", ["drude", "lorentz", "debye"])
def test_remote_zero_strength_pole_matches_eps_infinity(lane, kind, monkeypatch):
    graded = "nonuniform" in lane or lane.endswith("_nu")
    expected = solve(scene("plain", graded=graded), lane, monkeypatch)
    actual = solve(scene(kind, graded=graded), lane, monkeypatch)
    assert_records(expected, actual)


@pytest.mark.parametrize("lane", LANES[:8])
@pytest.mark.parametrize("kind", ["debye", "drude", "lorentz"])
def test_pole_one_cell_from_pin_is_refused(lane, kind):
    sim = scene(kind, graded="nonuniform" in lane or lane.endswith("_nu"), near=True)
    options = dict(n_steps=1, skip_preflight=True)
    if "distributed" in lane:
        options["devices"] = jax.devices("cpu")[:2]
    with pytest.raises(NotImplementedError, match=(
            r"port\[0\].*Debye/Lorentz.*'block'.*stencil cell \(4, 3, 3\).*"
            r"ε∞/4.*ε\(ω\)")):
        if lane.startswith("run"):
            sim.run(**options, compute_s_params=False)
        else:
            sim.forward(**options, distributed="distributed" in lane, checkpoint=False)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("kind", ["debye", "drude", "lorentz"])
@pytest.mark.parametrize("near", [True, False])
def test_lowlevel_radius_pole_clearance(graded, kind, near):
    grid = (scene("plain", graded=True)._build_nonuniform_grid() if graded else
            Grid(freq_max=30e9, domain=(.008,) * 3, dx=D, cpml_layers=0))
    port = WirePort((.004, .004, .002), (.004, .004, .005), "ez", radius=.0001)
    materials = setup_wire_port(grid, port, init_materials(grid.shape))
    assert materials.mu_r_wire is not None
    materials = jax.tree.map(lambda value: value, materials)
    mask = jnp.zeros(grid.shape, bool)
    mask = mask.at[2:6, 2:6, :].set(True) if near else mask.at[6:8, 6:8, :].set(True)
    if kind == "debye":
        poles = dict(debye=init_debye(
            [DebyePole(delta_eps=1., tau=1e-11)], materials, grid.dt, mask=mask, grid=grid))
    else:
        pole = (drude_pole(omega_p=2*np.pi*6e9, gamma=1e9) if kind == "drude" else
                LorentzPole(omega_0=2*np.pi*6e9, delta=1e9, kappa=(2*np.pi*3e9)**2))
        poles = dict(lorentz=init_lorentz([pole], materials, grid.dt, mask=mask, grid=grid))

    def run():
        if graded:
            return run_nonuniform(grid, materials, 2, **poles)
        return rfx.run(grid, materials, 2, boundary="pec", **poles)

    if near:
        with pytest.raises(NotImplementedError, match=r"radius.*Debye/Lorentz.*edge.*ε∞/4.*ε\(ω\)"):
            run()
    else:
        run()


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("offset", [(-2, -1), (-1, -2), (1, 0), (0, 1)])
def test_edge_mean_cells_are_in_the_stencil(component, offset):
    sim = scene("debye", component=component)
    axis = ("ex", "ey", "ez").index(component)
    transverse = [a for a in range(3) if a != axis]
    cell = [3, 3, 3]
    for a, delta in zip(transverse, offset):
        cell[a] += delta
    sim._geometry.clear()
    sim.add(Box(tuple(i*D for i in cell), tuple((i+1)*D for i in cell)), material="block")
    with pytest.raises(NotImplementedError, match=r"port\[0\].*stencil cell"):
        sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)


def test_traced_pole_geometry_is_refused_by_name():
    def forward(radius):
        sim = scene("debye")
        sim._geometry.clear()
        sim.add(Sphere((9*D, 7*D, 4*D), radius), material="block")
        return sim.forward(n_steps=1, skip_preflight=True).time_series

    with pytest.raises(NotImplementedError, match=r"radius.*port\[0\].*traced Debye/Lorentz"):
        jax.jit(forward)(2*D)


def test_mixed_s_matrix_cannot_build_the_wire_only_scene():
    with pytest.raises(RfxRefusal, match="needs at least one add_msl_port"):
        scene("drude").compute_mixed_s_matrix(freqs=FREQS, n_steps=STEPS, skip_preflight=True)

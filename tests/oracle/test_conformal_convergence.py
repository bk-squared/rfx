"""Hollow PEC cylinder TM010 resonance oracle for the conformal update.

An off-axis Ez pulse excites the hollow cylinder; two ringdown windows must
identify the dominant TM010 pole consistently without significant growing
poles. Staircase errors are positive, pinned, and fall under mesh refinement.
Independent radial-distance weights check the cavity complement. Conformal
frequency accuracy and convergence remain strict expected failures until
#1651 replaces the eps/w skin with the Dey-Mittra update.
"""

from rfx.geometry.rasterize_grid import unpadded_high_walls

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.grid import Grid, C0, cells_spanning
from rfx.core.yee import (
    init_state, init_materials, update_e, update_e_aniso, update_h,
)
from rfx.geometry.csg import Cylinder
from rfx.geometry.conformal import (
    compute_conformal_weights_sdf,
    clamp_conformal_weights,
    apply_conformal_pec,
    conformal_eps_correction,
)
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.ringdown import identify
from rfx.sources.sources import GaussianPulse

_X01 = 2.4048255577
_PI = np.pi


def _tm010_freq(radius):
    """Analytical TM010 resonant frequency for a closed cylindrical cavity."""
    return _X01 * C0 / (2.0 * _PI * radius)


_RADIUS = 0.015
_F0 = 2.4048255577 * C0 / (2 * np.pi * _RADIUS)
_N_STEPS = 6000
_PPW = (12, 16, 24)


def _cavity_geometry(ppw):
    dx = (C0 / _F0) / ppw
    n_xy = round((2 * _RADIUS + 4 * dx) / dx)
    length = n_xy * dx
    length_z = 4 * dx
    grid = Grid(freq_max=2 * _F0, domain=(length, length, length_z),
                dx=dx, cpml_layers=0)
    assert grid.dx == dx, (grid.dx, dx)
    assert grid.shape == (n_xy + 1, n_xy + 1, 5), grid.shape
    assert grid.face_pads == (0, 0, 0, 0, 0, 0), grid.face_pads
    center = (length / 2 + 0.17 * dx, length / 2 - 0.23 * dx, length_z / 2)
    cylinder = Cylinder(center=center, radius=_RADIUS, height=10 * length, axis="z")
    return grid, cylinder


def _cavity_trace(ppw, conformal):
    grid, cylinder = _cavity_geometry(ppw)
    cx, cy, cz = cylinder.center
    source = grid.position_to_index((cx + 0.31 * _RADIUS, cy + 0.12 * _RADIUS, cz))
    probe = grid.position_to_index((cx - 0.22 * _RADIUS, cy + 0.37 * _RADIUS, cz))
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    if conformal:
        weights = tuple(1 - w for w in compute_conformal_weights_sdf(grid, [cylinder]))
        weights = clamp_conformal_weights(*weights, 0.1)
        eps = conformal_eps_correction(materials.eps_r, *weights)
    else:
        pec_mask = ~cylinder.mask(grid)
    pulse = GaussianPulse(f0=_F0, bandwidth=1.0)

    def step(state, n):
        state = state._replace(ez=state.ez.at[source].add(pulse(n * grid.dt)))
        state = update_h(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        if conformal:
            state = update_e_aniso(state, materials, *eps, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        else:
            state = update_e(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        state = apply_pec(state)
        if conformal:
            state = apply_conformal_pec(state, *weights)
        else:
            state = apply_pec_mask(state, pec_mask)
        return state, state.ez[probe]

    _, trace = jax.lax.scan(step, state, jnp.arange(_N_STEPS))
    trace = np.asarray(trace)
    assert trace.dtype == np.float32, trace.dtype
    return grid, trace


def _read_cavity_poles(trace, dt):
    windows = []
    for start in (int(0.3 * _N_STEPS), int(0.5 * _N_STEPS)):
        model = identify(trace[:, None], dt, start, _N_STEPS, freq_max=2 * _F0)
        frequencies = model.s.imag / (2 * np.pi)
        near = np.flatnonzero(np.abs(frequencies / _F0 - 1) <= 0.25)
        broad = np.flatnonzero((frequencies >= 0.5 * _F0) & (frequencies <= 1.5 * _F0))
        tm = int(near[np.argmax(model.amplitude[near])]) if near.size else None
        largest = int(broad[np.argmax(model.amplitude[broad])]) if broad.size else None
        windows.append({
            "n_start": start,
            "tm010_hz": float(frequencies[tm]) if tm is not None else None,
            "tm010_amplitude": float(model.amplitude[tm]) if tm is not None else None,
            "largest_hz": float(frequencies[largest]) if largest is not None else None,
            "largest_amplitude": float(model.amplitude[largest]) if largest is not None else None,
            "tm010_is_largest": tm is not None and tm == largest,
            "n_kept": int(model.s.size),
            "n_growing_discarded": int(model.n_growing_discarded),
            "max_growing_amplitude": float(np.max(model.growing_amplitude, initial=0)),
        })
    frequencies = [w["tm010_hz"] for w in windows]
    relative_window_difference = (
        abs(frequencies[1] / frequencies[0] - 1)
        if all(f is not None for f in frequencies) else None
    )
    identification_ok = (
        relative_window_difference is not None
        and relative_window_difference <= 1e-4
        and all(w["max_growing_amplitude"] <= 1e-3 and w["tm010_is_largest"]
                for w in windows)
    )
    return {
        "err": frequencies[0] / _F0 - 1 if frequencies[0] is not None else None,
        "windows": windows,
        "relative_window_difference": relative_window_difference,
        "identification_ok": identification_ok,
    }


def _measure_cavity(ppw, conformal):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    grid, trace = _cavity_trace(ppw, conformal)
    result = _read_cavity_poles(trace, grid.dt)
    result.update(ppw=ppw, arm="conformal" if conformal else "staircase",
                  shape=grid.shape, dx=grid.dx, dt=grid.dt,
                  wall_seconds=time.perf_counter() - wall_start,
                  cpu_seconds=time.process_time() - cpu_start)
    return result


# Step A, 6000 float32 steps, W1=[1800,6000), W2=[3000,6000).
_E_PIN = {12: 0.0496, 16: 0.0424, 24: 0.0352}
_CONFORMAL_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=("#1651 conformal eps/w skin: TM010 accuracy and order fail; hand-built "
            "kernel (update_h, update_e_aniso, eps/w, apply_conformal_pec; no "
            "apply_pec_edges) must use the production kernel before removing xfail"),
)


@pytest.fixture(scope="module")
def cavity_measurements():
    return {(ppw, conformal): _measure_cavity(ppw, conformal)
            for ppw in _PPW for conformal in (False, True)}


def _assert_identified(result):
    for window in result["windows"]:
        assert window["tm010_hz"] is not None, window
        assert window["max_growing_amplitude"] <= 1e-3, window
        assert window["tm010_is_largest"], window
    assert result["relative_window_difference"] <= 1e-4, result


def test_t0_complement_weights():
    """The cavity's open-edge weights follow an independent radial formula."""
    grid, cylinder = _cavity_geometry(16)
    weights = tuple(1 - w for w in compute_conformal_weights_sdf(grid, [cylinder]))
    clamped = clamp_conformal_weights(*weights, 0.1)
    cx, cy, _ = cylinder.center
    # Independent float64 Yee edge midpoints; no rfx SDF/coordinate helper.
    for component, (weight, bounded) in enumerate(zip(weights, clamped)):
        x = (np.arange(grid.nx, dtype=float) + (0.5 if component == 0 else 0)) * grid.dx
        y = (np.arange(grid.ny, dtype=float) + (0.5 if component == 1 else 0)) * grid.dx
        radial_sdf = np.sqrt((x[:, None, None] - cx)**2
                             + (y[None, :, None] - cy)**2) - _RADIUS
        expected = np.broadcast_to(np.clip(0.5 - radial_sdf / grid.dx, 0, 1), grid.shape)
        actual, bounded = np.asarray(weight), np.asarray(bounded)
        # Float32 Yee coordinates near 40 mm have one ulp ~1.5e-6 of dx
        # per coordinate; the complement error guarded here is O(1).
        assert np.max(np.abs(actual - expected)) <= 1e-5
        assert np.any(actual == 0) and np.any(actual == 1)
        np.testing.assert_array_equal(bounded[actual == 0], 0)
        np.testing.assert_array_equal(bounded[actual == 1], 1)


@pytest.mark.parametrize("ppw", _PPW)
def test_t1_staircase_pin(cavity_measurements, ppw):
    result = cavity_measurements[ppw, False]
    _assert_identified(result)
    assert result["err"] > 0, result
    # 3e-3 is below a 1% radius shift and above measured float32 frequency noise,
    # but the staircase wall moves only in whole cells: at 12 ppw a 1% radius
    # change may not move it.
    assert abs(result["err"] - _E_PIN[ppw]) <= 3e-3, result


def test_t1_staircase_refinement(cavity_measurements):
    errors = [cavity_measurements[ppw, False]["err"] for ppw in _PPW]
    assert errors[0] > errors[1] > errors[2] > 0, errors


@_CONFORMAL_XFAIL
# At 12 ppw, Courant 0.572, Yee dispersion is -7.7e-3 (axis) to -2.0e-3
# (diagonal), ~-4.9e-3 angle-averaged: only 1e-4 inside the 5e-3 bar.
# At 16 and 24 ppw it is -2.7e-3 and -1.2e-3, respectively.
@pytest.mark.parametrize("ppw", (16, 24))
def test_t2_conformal_accuracy(cavity_measurements, ppw):
    result = cavity_measurements[ppw, True]
    _assert_identified(result)
    # 5e-3 is the sub-1% oracle bar: a 1% radius error must be red.
    assert abs(result["err"]) <= 5e-3, result


@_CONFORMAL_XFAIL
def test_t2_conformal_order(cavity_measurements):
    results = [cavity_measurements[ppw, True] for ppw in _PPW]
    for result in results:
        _assert_identified(result)
    errors = np.abs([result["err"] for result in results])
    assert np.all(errors > 0), errors
    slope = np.polyfit(np.log([result["dx"] for result in results]), np.log(errors), 1)[0]
    assert slope >= 1.5, slope


class TestConformalConvergence:
    """Retained GPU stability and API checks."""

    @pytest.mark.gpu
    def test_conformal_stability_long_run(self):
        """Conformal PEC cavity should be stable for many timesteps."""
        radius = 0.012
        height = 0.04
        dx = 0.003
        margin = 6 * dx  # larger margin so source is well outside PEC
        Lx = 2 * radius + 2 * margin
        Ly = Lx
        Lz = height + 2 * margin
        center = (Lx / 2, Ly / 2, Lz / 2)

        # Placement uses the declared lengths; the grid gets main's solved lengths (#1887).
        grid = Grid(freq_max=_tm010_freq(radius) * 3,
                    domain=tuple(cells_spanning(L, dx) * dx for L in (Lx, Ly, Lz)),
                    dx=dx, cpml_layers=0)
        cyl = Cylinder(center=center, radius=radius, height=height, axis="z")

        state = init_state(grid.shape)
        materials = init_materials(grid.shape)

        w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [cyl])
        w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)
        eps_ex, eps_ey, eps_ez = conformal_eps_correction(
            materials.eps_r, w_ex, w_ey, w_ez)

        # Place source OUTSIDE the cylinder but inside the domain
        # Source at corner of domain (well outside the cylinder)
        src_pos = (margin, margin, Lz / 2)
        src_idx = grid.position_to_index(src_pos)

        f_target = _tm010_freq(radius)
        pulse = GaussianPulse(f0=f_target, bandwidth=1.5)

        energies = []
        for n in range(5000):
            t = n * grid.dt
            if n < 200:
                state = state._replace(ez=state.ez.at[src_idx].add(pulse(t)))

            state = update_h(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
            state = update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez,
                                   grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
            state = apply_pec(state)
            state = apply_conformal_pec(state, w_ex, w_ey, w_ez)

            if n % 500 == 0 and n > 200:
                energy = float(jnp.sum(state.ex**2 + state.ey**2 + state.ez**2))
                energies.append(energy)

        assert len(energies) >= 3, "Should have collected energy samples"
        assert max(energies) > 0, "Should have nonzero energy"

        # Energy should not grow exponentially (bounded in PEC cavity)
        max_energy = max(energies)
        final_energy = energies[-1]
        assert final_energy < max_energy * 5, \
            f"Energy grew: final={final_energy:.4e}, max={max_energy:.4e}"

    @pytest.mark.gpu
    def test_api_conformal_convergence(self):
        """End-to-end API test: conformal should give different results than staircase."""
        from rfx.api import Simulation

        sim = Simulation(freq_max=5e9, domain=(0.06, 0.06, 0.06))
        sim.add(Cylinder((0.03, 0.03, 0.03), 0.012, 0.04), material="pec")
        sim.add_port((0.03, 0.03, 0.005), "ez", waveform=GaussianPulse(f0=3e9))
        sim.add_probe((0.03, 0.045, 0.03), "ez")

        # The port only excites the model; a conformal lumped S-matrix is refused (#1299).
        result_conf = sim.run(n_steps=200, conformal_pec=True, compute_s_params=False)
        result_stair = sim.run(n_steps=200)

        ts_conf = np.array(result_conf.time_series).ravel()
        ts_stair = np.array(result_stair.time_series).ravel()

        diff = np.max(np.abs(ts_conf - ts_stair))
        assert diff > 0, "Conformal and staircase should produce different results"

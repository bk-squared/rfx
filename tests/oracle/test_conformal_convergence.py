"""Hollow PEC cylinder TM010 resonance oracle for the conformal update.

An off-axis Ez pulse excites the hollow cylinder; two ringdown windows must
identify the dominant TM010 pole consistently without significant growing
poles. Staircase errors are positive, pinned, and fall under mesh refinement.
Independent radial-distance lengths check the cavity complement. With the
cut-cell operands (#1651) the TM010 error of this z-invariant cylinder (E along
the wall) is -0.302 / -0.155 / -0.064 %
at 12 / 16 / 24 ppw, slope 2.2; the two conformal tests stay strict expected
failures because the ring-down identification keeps growing poles from a slow
beat (the envelope oscillates by about 10 % with a period near 4e5 steps and does
not grow over 1e6 steps at 24 ppw), so the identification bar is not met. A
3-D PEC sphere converges only first order with the same operands (#1651).
"""

from rfx.geometry.rasterize_grid import unpadded_high_walls

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.grid import Grid, C0, cells_spanning
from rfx.core.yee import init_state, init_materials
from rfx.stepping.yee import update_e, update_e_aniso, update_h
from rfx.geometry.csg import Cylinder
from rfx.geometry.conformal import (
    compute_conformal_fractions,
    apply_conformal_pec,
    conformal_eps_correction,
)
from rfx.geometry.conformal_cells import sc_clamp
from rfx.model.tensor_operands import finish_conformal_mu
from rfx.model.materials import with_components
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
        lengths, areas = compute_conformal_fractions(grid, [cylinder], clamp=False)
        areas = tuple(1 - a for a in areas)
        weights = tuple(jnp.asarray(w) for w in sc_clamp(
            tuple(1 - np.asarray(w) for w in lengths), tuple(np.asarray(a) for a in areas)))
        materials = finish_conformal_mu(with_components(
            materials, grid, periodic=(False,) * 3), areas)
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
    reason=("#1651: with the cut-cell operands TM010 reads -0.155 % / -0.064 % at 16 / 24 "
            "ppw (slope 2.2), but identification keeps growing poles (up to 0.19 of TM010 at "
            "24 ppw) from a slow beat; 1e6 steps show no growth (2026-10-29)"),
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
    lengths, _ = compute_conformal_fractions(grid, [cylinder], clamp=False)
    cx, cy, _ = cylinder.center
    # Independent end-node radial distances, before complementing and SC.
    for component, length in enumerate(lengths):
        x = np.arange(grid.nx, dtype=float)[:, None, None] * grid.dx
        y = np.arange(grid.ny, dtype=float)[None, :, None] * grid.dx
        d0 = np.sqrt((x - cx)**2 + (y - cy)**2) - _RADIUS
        d1 = np.sqrt((x + (component == 0) * grid.dx - cx)**2
                     + (y + (component == 1) * grid.dx - cy)**2) - _RADIUS
        crossing = ((d0 < 0) & (d1 > 0)) | ((d1 < 0) & (d0 > 0))
        exterior = np.where((d0 >= 0) & (d1 >= 0), 1.,
                            np.maximum(d0, d1) / np.where(crossing, abs(d1 - d0), 1.))
        exterior = np.where((d0 <= 0) & (d1 <= 0), 0., exterior)
        expected = np.broadcast_to(1 - exterior, grid.shape)
        actual = 1 - np.asarray(length)
        assert np.max(np.abs(actual - expected)) <= 1e-7
        assert np.any(actual == 0) and np.any(actual == 1)


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

        (w_ex, w_ey, w_ez), areas = compute_conformal_fractions(grid, [cyl])
        materials = finish_conformal_mu(with_components(
            materials, grid, periodic=(False,) * 3), areas)
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

"""Discrete source invariants; these are algebra gates, not mesh accuracy claims."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.core.yee import EPS_0, MU_0, init_state
from rfx.sources.sources import CustomWaveform
from rfx.sources.tfsf import (
    apply_tfsf_e, apply_tfsf_h, init_tfsf, tfsf_injection_planes,
    update_tfsf_1d, measure_normal_incident_spectrum,
)


def _peak_ulp(actual, expected, budget=9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    peak = np.float32(np.max(np.abs(expected)))
    assert np.max(np.abs(actual - expected)) <= budget * np.spacing(peak)


def _curl(fields, forward):
    """Independent full-array curl (test source commutator only)."""
    def diff(a, axis):
        if forward:
            shifted = np.roll(a, -1, axis=axis)
            idx = [slice(None)] * 3
            idx[axis] = -1
            shifted[tuple(idx)] = 0
            return shifted - a
        shifted = np.roll(a, 1, axis=axis)
        idx = [slice(None)] * 3
        idx[axis] = 0
        shifted[tuple(idx)] = 0
        return a - shifted
    x, y, z = fields
    return np.stack((diff(z, 1) - diff(y, 2),
                     diff(x, 2) - diff(z, 0),
                     diff(y, 0) - diff(x, 1)))


@pytest.mark.parametrize("polarization,direction", list(itertools.product(("ez", "ey"), ("+x", "-x"))))
def test_six_face_corrections_equal_mask_curl_commutator(polarization, direction):
    shape = (17, 16, 15)
    dx, dt = 0.001, 1e-12
    cfg, aux = init_tfsf(shape[0], dx, dt, ny=shape[1], nz=shape[2],
                         cpml_layers=2, tfsf_margin=3, closed_box=True,
                         polarization=polarization, direction=direction)
    rng = np.random.default_rng(891)
    e1d = rng.normal(size=aux.e1d.shape).astype(np.float32)
    h1d = rng.normal(size=aux.h1d.shape).astype(np.float32)
    aux = aux._replace(e1d=jnp.asarray(e1d), h1d=jnp.asarray(h1d))
    mask = np.zeros(shape)
    mask[cfg.x_lo:cfg.x_hi + 1, cfg.y_lo:cfg.y_hi + 1,
         cfg.z_lo:cfg.z_hi + 1] = 1
    ix = np.arange(shape[0]) + cfg.i0 - cfg.x_lo
    e = np.zeros((3,) + shape)
    h = np.zeros_like(e)
    e["xyz".index(polarization[-1])] = e1d[ix, None, None]
    h["xyz".index(cfg.magnetic_component[-1])] = h1d[ix, None, None]
    want_h = -(dt / (MU_0 * dx)) * (mask * _curl(e, True) - _curl(mask * e, True))
    want_e = (dt / (EPS_0 * dx)) * (mask * _curl(h, False) - _curl(mask * h, False))
    state = init_state(shape)
    got_h = apply_tfsf_h(state, cfg, aux, dx, dt)
    got_e = apply_tfsf_e(state, cfg, aux, dx, dt)
    _peak_ulp(np.stack([getattr(got_h, c) for c in ("hx", "hy", "hz")]), want_h)
    _peak_ulp(np.stack([getattr(got_e, c) for c in ("ex", "ey", "ez")]), want_e)
    assert set(tfsf_injection_planes(cfg)) == set("xyz")


def _sim(*, polarization="ez", direction="+x", closed_box=True, waveform=None):
    sim = Simulation(freq_max=30e9, domain=(0.018, 0.016, 0.014), dx=0.001,
                     boundary="cpml", cpml_layers=4)
    if waveform is None:
        dt = sim._build_grid().dt
        # Both quadratures, a finite record and no reliance on Gaussian metadata.
        u = jnp.arange(72, dtype=jnp.float32)
        samples = jnp.sin(jnp.pi * u / 71) ** 2 * (
            0.7 * jnp.cos(0.31 * u) - 0.4 * jnp.sin(0.31 * u))
        def drive(t):
            i = jnp.rint(t / dt).astype(jnp.int32)
            return jnp.where((i >= 0) & (i < len(samples)),
                             samples[jnp.clip(i, 0, len(samples) - 1)], 0.0)
        waveform = CustomWaveform(drive)
    sim.add_tfsf_source(f0=15e9, bandwidth=0.8, polarization=polarization,
                        direction=direction, margin=3,
                        waveform=waveform, closed_box=closed_box)
    return sim


def _aux(sim):
    grid = sim._build_grid()
    entry = sim._tfsf
    return init_tfsf(grid.nx, grid.dx, grid.dt, ny=grid.ny, nz=grid.nz,
                     cpml_layers=grid.cpml_layers, tfsf_margin=entry.margin,
                     polarization=entry.polarization, direction=entry.direction,
                     f0=entry.f0, bandwidth=entry.bandwidth,
                     waveform=entry.waveform, closed_box=entry.closed_box)


@pytest.mark.parametrize("polarization,direction", list(itertools.product(("ez", "ey"), ("+x", "-x"))))
def test_closed_vacuum_replays_real_waveform_inside_and_cancels_all_faces(polarization, direction):
    sim = _sim(polarization=polarization, direction=direction)
    grid = sim._build_grid()
    cfg, aux = _aux(sim)
    bounds = sim.tfsf_box_indices()
    assert bounds == {a: (getattr(cfg, a + "_lo"), getattr(cfg, a + "_hi")) for a in "xyz"}
    # Two inside nodes, each outside face, and all eight exterior corners.
    centre = tuple((lo + hi) // 2 for lo, hi in bounds.values())
    points = [centre, (cfg.x_lo, cfg.y_lo, cfg.z_lo)]
    for axis, (lo, hi) in enumerate(bounds.values()):
        for idx in (lo - 1, hi + 1):
            p = list(centre)
            p[axis] = idx
            points.append(tuple(p))
    points.extend(itertools.product(*[(lo - 1, hi + 1) for lo, hi in bounds.values()]))
    for p in points:
        pos = tuple((i - grid.cpml_layers) * grid.dx for i in p)
        assert grid.position_to_index(pos) == p
        sim.add_probe(pos, component=polarization)
    result = sim.run(n_steps=128)
    trace = np.asarray(result.time_series)
    indices = np.array([cfg.i0 + p[0] - cfg.x_lo for p in points[:2]])
    def step(st, n):
        st = update_tfsf_1d(cfg, st, grid.dx, grid.dt, n * grid.dt)
        return st, st.e1d[indices]
    _, reference = jax.lax.scan(step, aux, jnp.arange(128))
    _peak_ulp(trace[:, :2], reference)
    peak = np.float32(np.max(np.abs(reference)))
    assert peak > 0.1
    assert np.max(np.abs(trace[:, 2:])) <= 9 * np.spacing(peak)
    freqs = np.linspace(8e9, 22e9, 7)
    times = (np.arange(128) + 1) * grid.dt
    spectrum = np.exp(-2j * np.pi * freqs[:, None] * times) @ trace[:, 0] * grid.dt
    expected = measure_normal_incident_spectrum(cfg, aux, 128, freqs, grid.dt,
                                                reference_index=centre[0])
    assert np.max(np.abs(spectrum - expected)) <= 1e-4 * np.max(np.abs(expected))

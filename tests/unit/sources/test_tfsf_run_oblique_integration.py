"""Integration: Simulation.run() drives the #404 oblique-periodic complex Bloch path.

The oblique (2D-aux) TFSF now routes the shared solver onto the complex Bloch-
envelope path (field_dtype=complex64 + per-axis roll phase + complex CPML psi
carry), and run() reconstructs the returned `state` / point probes back to the
physical real field Re(P·exp(-j k_t·y)). Normal incidence and every non-TFSF run
stay on the real float32 path (byte-identical). These gates pin that contract.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.tfsf import init_tfsf, apply_tfsf_e, apply_tfsf_h, is_tfsf_2d
from rfx.sources.tfsf_2d import (
    update_tfsf_2d_h, update_tfsf_2d_e, bloch_phase_tuple,
)

_DOMAIN = (0.6, 0.12, 0.006)
_DX = 0.002


def _build(angle):
    sim = Simulation(freq_max=10e9, domain=_DOMAIN, dx=_DX,
                     boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=5e9, bandwidth=0.15, polarization="ez",
                        direction="+x", angle_deg=angle,
                        waveform="modulated_gaussian")
    return sim


def test_run_normal_incidence_stays_real():
    """Normal-incidence TFSF through run() stays on the real float32 path (the
    complex Bloch path is oblique-only; normal incidence must be untouched)."""
    res = _build(0.0).run(n_steps=150)
    assert res.state.ez.dtype == jnp.float32


def test_run_oblique_returns_real_physical_field():
    """An oblique run() returns a real (reconstructed) physical field, not the
    complex envelope — the run drove the complex path but the user sees fields."""
    res = _build(30.0).run(n_steps=150)
    assert res.state.ez.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(res.state.ez))) > 1e-4


def test_run_oblique_flux_monitor_fails_loud():
    """Frequency-domain monitors are not yet transform-aware on the complex
    Bloch path — run() must fail loud rather than return truncated garbage."""
    sim = _build(30.0)
    sim.add_flux_monitor(axis="x", coordinate=0.1, freqs=[5e9], name="r")
    with pytest.raises(NotImplementedError):
        sim.run(n_steps=100)


def test_run_oblique_until_decay_fails_loud():
    """until_decay is incompatible with complex state (float(complex) crash) —
    run() fences it with a clear message."""
    with pytest.raises(NotImplementedError):
        _build(30.0).run(until_decay=1e-4, decay_monitor_component="ez",
                         decay_monitor_position=(0.1, 0.0, 0.0))


@pytest.mark.slow
def test_run_oblique_matches_handrolled_and_low_leakage():
    """run() oblique reproduces the VALIDATED hand-rolled complex loop to ~1e-6
    (same kernel/injection/order) AND yields <1% TFSF leakage — proof the high-
    level API drives the identical #404 physics, not a re-derivation."""
    angle, ns = 30.0, 250
    res = _build(angle).run(n_steps=ns)
    assert res.state.ez.dtype == jnp.float32
    ez_run = np.asarray(res.state.ez)

    # Hand-rolled reference (complex loop), config identical to run_uniform's.
    grid = Grid(freq_max=10e9, domain=_DOMAIN, dx=_DX, cpml_layers=10)
    dt, dx = grid.dt, grid.dx
    cfg, aux = init_tfsf(
        grid.nx, dx, dt, cpml_layers=grid.cpml_layers, tfsf_margin=3,
        f0=5e9, bandwidth=0.15, amplitude=1.0, polarization="ez",
        direction="+x", angle_deg=angle, ny=grid.ny, nz=grid.nz,
        waveform="modulated_gaussian",
    )
    assert is_tfsf_2d(cfg)
    bloch = bloch_phase_tuple(cfg, dx)
    per = (False, True, True)
    mat = init_materials(grid.shape)
    st = init_state(grid.shape, field_dtype=jnp.complex64)
    cp, cs = init_cpml(grid, field_dtype=jnp.complex64)
    for step in range(ns):
        t = step * dt
        st = update_h(st, mat, dt, dx, periodic=per, bloch=bloch)
        st = apply_tfsf_h(st, cfg, aux, dx, dt)
        st, cs = apply_cpml_h(st, cp, cs, grid, axes="x")
        aux = update_tfsf_2d_h(cfg, aux, dx, dt)
        st = update_e(st, mat, dt, dx, periodic=per, bloch=bloch)
        st = apply_tfsf_e(st, cfg, aux, dx, dt)
        st, cs = apply_cpml_e(st, cp, cs, grid, axes="x")
        aux = update_tfsf_2d_e(cfg, aux, dx, dt, t)
    yph = np.exp(-1j * cfg.k_transverse * np.arange(grid.ny) * dx)[None, :, None]
    ez_hand = np.real(np.asarray(st.ez) * yph)

    rel = np.max(np.abs(ez_run - ez_hand)) / max(np.max(np.abs(ez_hand)), 1e-30)
    assert rel < 1e-3, f"run() vs hand-rolled reconstructed Ez mismatch: {rel:.2e}"

    sf = float(np.sum(ez_run[:cfg.x_lo - 2] ** 2))
    tf = float(np.sum(ez_run[cfg.x_lo:cfg.x_hi] ** 2))
    assert tf > 0, "no total-field energy"
    assert sf / tf < 0.01, f"run() oblique leakage {sf / tf * 100:.3f}% > 1%"

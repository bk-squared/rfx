"""Differentiable reflection coefficient Gamma(f0) objective for TFSF forward.

Gap #1 for PEFM RIS / FSS / metasurface inverse design (see
docs/research_notes/2026-07-21_diffplanewave_pefm_utilization.md): the merged #415
forward optimizes only raw sum(|time_series|^2); RIS/FSS design optimizes the
physical complex REFLECTION coefficient ‖Gamma(f,theta) - Gamma_target‖ (amplitude
AND phase). This builds a PHYSICALLY-VALID + AD-traceable Gamma — NOT the
ris.py::_extract_reflection FFT-of-probe placeholder (which its own docstring warns
is not a real S11).

Method (repo-blessed two-run reference subtraction, "never FFT-of-probe"):
  probe in the total-field region between the TFSF -x boundary and the scatterer;
  incident I(f0) from a vacuum forward (eps-independent CONSTANT, computed once);
  total T(f0, eps) from the scatterer forward (differentiable); reflected = T - I;
  Gamma = (T - I) / I. f0 phasor via the standard -j DFT kernel (real field).
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid


def _adam(v_and_g, x0, steps, lr=0.2, lo=1.0, hi=8.0):
    """Minimal Adam (optax is an optional extra not in [dev] — avoid importing it
    at module level so the fast-suite collection never breaks)."""
    x, m, v = float(x0), 0.0, 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for i in range(1, steps + 1):
        val, g = v_and_g(x)
        hist.append(float(val))
        g = float(g)
        assert np.isfinite(g), "P0: non-finite gradient during optimization"
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** i), v / (1 - b2 ** i)
        x = float(np.clip(x - lr * mh / (np.sqrt(vh) + eps), lo, hi))
    return x, hist

_F0 = 5e9
_NS = 700
_DOMAIN = (0.24, 0.08, 0.006)
_DX = 0.002
_PROBE = (0.05, 0.04, 0.003)   # TF region, on the reflection (-x) side of the slab
_SLAB_X = (0.14, 0.17)         # scatterer slab (corner-origin coords, in [0, 0.24])


def _setup():
    grid = Grid(freq_max=10e9, domain=_DOMAIN, dx=_DX, cpml_layers=8)
    x0 = grid.position_to_index((_SLAB_X[0], 0.04, 0.003))[0]
    x1 = grid.position_to_index((_SLAB_X[1], 0.04, 0.003))[0]
    sim = Simulation(freq_max=10e9, domain=_DOMAIN, dx=_DX, boundary="cpml",
                     cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=_F0, bandwidth=0.5, polarization="ez", direction="+x",
                        waveform="differentiated_gaussian")
    sim.add_probe(_PROBE, component="ez")
    shp = sim.run(n_steps=1).state.ez.shape
    t = jnp.arange(_NS) * grid.dt
    kern = jnp.exp(-1j * 2 * jnp.pi * _F0 * t) * grid.dt   # f0 phasor kernel
    return sim, shp, x0, x1, kern


def _phasor(sim, shp, x0, x1, kern, eps_val):
    eps = jnp.ones(shp, jnp.float32).at[x0:x1, :, :].set(eps_val)
    fr = sim.forward(eps_override=eps, n_steps=_NS, checkpoint=True, skip_preflight=True)
    return jnp.sum(fr.time_series[:, 0].astype(jnp.complex64) * kern)


@pytest.mark.slow
def test_forward_tfsf_reflection_coefficient_is_differentiable_and_physical():
    """A physically-valid, AD-traceable reflection coefficient Gamma(f0): passive
    (|Gamma|<~1), FD-matched gradient of |Gamma|, and a phase that tunes with the
    scatterer permittivity (the RIS phase knob)."""
    sim, shp, x0, x1, kern = _setup()
    I0 = complex(_phasor(sim, shp, x0, x1, kern, 1.0))   # vacuum incident, constant
    assert abs(I0) > 0

    def gamma_mag(eps_val):
        g = (_phasor(sim, shp, x0, x1, kern, eps_val) - I0) / I0
        return jnp.sqrt((g * jnp.conj(g)).real + 1e-30)

    # passive: |Gamma| <~ 1 across the tested range
    mags = {e: float(gamma_mag(e)) for e in (2.0, 4.0, 6.0)}
    assert all(m < 1.05 for m in mags.values()), f"non-passive |Gamma|: {mags}"

    # AD gradient of |Gamma| matches central FD (correct differentiable objective)
    g_ad = float(jax.grad(gamma_mag)(4.0))
    d = 0.02
    fd = (float(gamma_mag(4.0 + d)) - float(gamma_mag(4.0 - d))) / (2 * d)
    assert np.isfinite(g_ad)
    assert abs(g_ad - fd) / max(abs(fd), 1e-30) < 0.05, f"AD {g_ad} vs FD {fd}"

    # reflection PHASE tunes with eps (RIS/metasurface phase design knob)
    def phase(eps_val):
        g = complex(_phasor(sim, shp, x0, x1, kern, eps_val)) - I0
        return np.degrees(np.angle(g))
    phases = [phase(e) for e in (2.0, 3.0, 4.0, 5.0, 6.0)]
    span = max(phases) - min(phases)
    assert span > 90.0, f"reflection phase barely tunes ({span:.0f}deg): {phases}"


@pytest.mark.slow
def test_forward_tfsf_ris_phase_inverse_design():
    """RIS-style inverse design: optimize the scatterer permittivity so its complex
    reflection coefficient Gamma(f0) hits a TARGET (amplitude+phase) — the PEFM yr3
    RIS-역설계 objective — via Adam through forward(). The complex objective drops."""
    sim, shp, x0, x1, kern = _setup()
    I0 = complex(_phasor(sim, shp, x0, x1, kern, 1.0))

    def gamma(eps_val):
        return (_phasor(sim, shp, x0, x1, kern, eps_val) - I0) / I0

    g_target = gamma(3.0)                       # target reflection (from eps=3)
    gr, gi = float(g_target.real), float(g_target.imag)

    def loss(eps_val):
        g = gamma(eps_val)
        return (g.real - gr) ** 2 + (g.imag - gi) ** 2

    v_and_g = jax.value_and_grad(loss)
    eps, hist = _adam(lambda e: v_and_g(e), 5.0, steps=8, lr=0.2)
    hist.append(float(loss(eps)))
    assert hist[-1] < hist[0] * 0.5, f"RIS reflection-target loss did not converge: {hist}"

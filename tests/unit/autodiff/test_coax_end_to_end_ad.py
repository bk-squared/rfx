"""End-to-end AD gate for a mismatched resistive coax termination (#1356).

The full path is eps_scale -> Yee update -> DFT -> modal voltage -> fitted
complex S11 and gamma. A fixed 3 mm reference-plane shift makes propagation
phase physical in the objective as well as the resistive mismatch magnitude.
Float32 fields/AD are checked against scoped float64 DFT/extraction FD.
The 0.05% gate rounds up 1.5 times the earlier measured record envelope 0.000269018689727 to include record drift.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from tests._gate_policy import gate_from_envelope

N_STEPS = 1500
FREQ = jnp.asarray([8.0e9], dtype=jnp.float32)
PROBE_COUNT = 9
DUT_IMPEDANCE = 25.0
REFERENCE_OFFSET_M = 0.003


def _build_sim():
    """The roughly 3.8-cell annulus resolves the nominal 50-ohm line for its mismatched 25-ohm termination."""
    sim = Simulation(domain=(0.008, 0.008, 0.020), freq_max=40e9, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.010), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


def _reflection(eps_scale=None):
    return _build_sim().compute_coaxial_line_reflection(
        termination="matched", dut_impedance=DUT_IMPEDANCE,
        n_steps=N_STEPS, freqs=FREQ, eps_scale=eps_scale,
        probe_count=PROBE_COUNT,
    )


def _s11_objective(deps):
    """Re(S) - Im(S), S referred 3 mm toward the source from the DUT.

    Scale the whole dielectric by e=1+deps, so Z=Z_base/sqrt(e),
    beta=beta_base*sqrt(e), Gamma=(R-Z)/(R+Z), and S=Gamma*exp(-2j*beta*L).
    At e=1, Gamma'=R*Z/(R+Z)**2 and theta'=beta*L for theta=2*beta*L.
    Thus f'=Gamma'*(cos(theta)+sin(theta))
             + Gamma*beta*L*(cos(theta)-sin(theta)).
    For R=25, Z=50 ohm, PTFE at 8 GHz and L=3 mm, f' is about +0.46:
    both the impedance and phase terms are positive and nonzero. The actual
    SMA radii give Z=48.5914 ohm and f'=+0.4540 in the continuum model;
    this predicts sign/scale, not exact finite-grid agreement. The lane
    returns S11 at the DUT; using its *fitted* gamma for the shift retains
    the traced d beta path (an analytic phase would bypass that path).
    """
    eps_scale = jnp.float32(1) + jnp.asarray(deps, dtype=jnp.float32)
    res = _reflection(eps_scale)
    s = res.s11[0] * jnp.exp(-2 * res.gamma[0] * REFERENCE_OFFSET_M)
    return jnp.real(s) - jnp.imag(s)


_FD_H = 0.0025
_FD_REL_ENVELOPE = 0.00018212009579344417
_RECORD_REL_ENVELOPE = 0.00026901868972665496
_REL_ERR_THRESHOLD = gate_from_envelope(
    max(_FD_REL_ENVELOPE, _RECORD_REL_ENVELOPE), quantum=10000,
)


@pytest.mark.slow_physics
@pytest.mark.highmem
def test_coax_reflection_grad_finite_and_fd_consistent():
    """The resistive complex objective has a physical nonzero dielectric slope.
    The finite-difference step 0.0025 lies inside the earlier measured 0.0003–0.01 convergence window.
    """
    try:
        from jax import enable_x64
    except ImportError:
        from tests._x64_compat import enable_x64

    val, g = jax.value_and_grad(_s11_objective)(0.0)
    assert np.isfinite(float(val)), f"objective is not finite: {val}"
    assert np.isfinite(float(g)), f"gradient is not finite: {g}"
    with enable_x64():
        fp, fm = _s11_objective(_FD_H), _s11_objective(-_FD_H)
        assert fp.dtype == fm.dtype == jnp.float64, f"FD not float64: {fp.dtype}"
        fd = (float(fp) - float(fm)) / (2 * _FD_H)
    assert np.isfinite(fd) and fd > 0, "FD slope not finite/positive — rebuild fixture"
    rel = abs(float(g) - fd) / abs(fd)
    assert rel <= _REL_ERR_THRESHOLD, (
        f"AD={float(g):+.9e} vs FD={fd:+.9e} "
        f"(rel diff {rel:.6g} > {_REL_ERR_THRESHOLD:.6g})"
    )


def test_coax_eps_scale_unity_matches_concrete_path():
    """Uniform eps_scale=1 preserves the resistive DUT's complex reflection."""
    a, b = _reflection(), _reflection(jnp.float32(1))
    sa, sb = np.asarray(a.s11), np.asarray(b.s11)
    assert np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))
    np.testing.assert_allclose(sb, sa, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-m", "slow_physics"])

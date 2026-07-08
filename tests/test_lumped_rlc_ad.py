"""WP 4-E: lumped R/L/C component values as a differentiable design variable.

The scan physics is differentiable w.r.t. a component value (the spike note
``docs/design_notes/wp4e_lumped_component_value_ad_spike.md`` established the
pure-L positive witness, AD-vs-FD 8.3e-6).  What was missing was the plumbing:

  * the traced meta builder in ``rfx/lumped.py`` (``build_rlc_meta_traced`` /
    ``setup_rlc_materials_traced``) that drops the ``float()`` coercions and
    keeps topology decisions static, and
  * ``Simulation.forward()`` iterating ``self._lumped_rlc`` and threading the
    metas into the differentiable ``_run(...)`` driver, plus a
    ``rlc_values_override`` injection surface so a component value can enter the
    AD tape AS a tracer (``LumpedRLCSpec`` stores plain floats).

These tests are the gate:

1. ``test_dS11_dR_dC_ad_matches_fd`` — the FEATURE proof: ``grad(|S11|^2)``
   w.r.t. R and w.r.t. C on a stable lumped-port fixture is finite, nonzero and
   FD-consistent (rel < 5%), under SCOPED x64 (never module-level — see repo
   memory ``feedback_jax_x64_module_level_tests``).
2. ``test_run_series_rlc_byte_identity`` — the concrete ``run()`` RLC path is
   unchanged (golden byte-identical to main).
3. ``test_forward_no_rlc_byte_identity`` — a sim with NO lumped RLC is
   unchanged through ``forward()`` (golden byte-identical to main).  This is the
   load-bearing falsifier: all inverse-design / TAP examples ride on it.
4. ``test_forward_with_rlc_is_not_noop`` — a registered RLC element now
   correctly affects ``forward()``; previously it was a SILENT no-op.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:  # modern JAX: scoped x64 promoted to top-level (experimental removed v0.8.0)
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX (< ~0.4.31)
    from jax.experimental import enable_x64 as _enable_x64

from rfx import GaussianPulse, Simulation

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Lumped-port + co-located series R+C fixture.  CPML keeps the port V/I
# well-conditioned (see tests/test_run_forward_s11_contract.py); the co-located
# series R+C exercises the ADE carry (capacitor charge updates each step through
# the traced meta.R / meta.dt_over_C_dx) so the scoped-x64 dtype threading is
# genuinely tested.  Freqs are pinned in-band (source f0=5 GHz, bw=0.9) where
# the incident wave is strong — below-cutoff / weak-incident bins would NaN the
# gradient silently.
_POS = (0.0093, 0.0093, 0.0093)
_F0 = 5e9
_FREQS = np.array([4.5, 5.0, 5.5]) * 1e9
_R0 = 50.0
_C0 = 0.20e-12
_N_STEPS = 1600


def _fixture_sim():
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
        boundary="cpml", cpml_layers=6,
    )
    sim.add_port(position=_POS, component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=_F0, bandwidth=0.9))
    sim.add_lumped_rlc(position=_POS, component="ez",
                       R=_R0, C=_C0, topology="series")
    return sim


def _s11_sq_sum(R, C):
    """Scalar port objective: sum_f |S11(f)|^2 from forward(port_s11_freqs=...)."""
    fr = _fixture_sim().forward(
        port_s11_freqs=_FREQS, n_steps=_N_STEPS,
        rlc_values_override={0: {"R": R, "C": C}},
    )
    s = fr.s_params.reshape(-1)
    return jnp.sum(jnp.abs(s) ** 2)


def test_dS11_dR_dC_ad_matches_fd():
    """grad(|S11|^2) w.r.t. R and C is finite, nonzero and FD-consistent."""
    with _enable_x64(True):
        R = jnp.asarray(_R0, dtype=jnp.float64)
        C = jnp.asarray(_C0, dtype=jnp.float64)

        val, (gR, gC) = jax.value_and_grad(_s11_sq_sum, argnums=(0, 1))(R, C)
        val = float(val)
        gR = float(gR)
        gC = float(gC)

        assert np.isfinite(val) and val > 0.0
        for name, g in (("dR", gR), ("dC", gC)):
            assert np.isfinite(g), f"grad {name} not finite: {g}"
            assert g != 0.0, f"grad {name} is exactly zero (no path to the DoF)"

        # Central finite differences (scoped x64 makes the FD arithmetic clean).
        hR = _R0 * 1e-4
        hC = _C0 * 1e-4
        fdR = (float(_s11_sq_sum(R + hR, C)) - float(_s11_sq_sum(R - hR, C))) / (2 * hR)
        fdC = (float(_s11_sq_sum(R, C + hC)) - float(_s11_sq_sum(R, C - hC))) / (2 * hC)

        rel_R = abs(gR - fdR) / (abs(fdR) + 1e-30)
        rel_C = abs(gC - fdC) / (abs(fdC) + 1e-30)
        assert rel_R < 0.05, f"dS11^2/dR AD {gR} vs FD {fdR} rel {rel_R:.3%}"
        assert rel_C < 0.05, f"dS11^2/dC AD {gC} vs FD {fdC} rel {rel_C:.3%}"


def test_run_series_rlc_byte_identity():
    """Concrete run() series-RLC path is byte-identical to main (golden).

    The traced lane is a SEPARATE code path; the concrete build_rlc_meta /
    setup_rlc_materials / run(lumped_rlc=...) numerics must be untouched.
    (On-platform this is exact array_equal; the golden was proven byte-identical
    to main in the WP 4-E session.  A tight tolerance guards against real
    numeric regressions while tolerating cross-machine float32 noise.)
    """
    golden = np.load(os.path.join(_FIXTURE_DIR, "golden_lumped_rlc_run_series.npy"))
    R, L, C = 50.0, 10e-9, 1e-12
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
    sim.add_source(position=(0.005, 0.005, 0.005), component="ez",
                   waveform=GaussianPulse(f0=f0, bandwidth=f0 * 0.5))
    sim.add_lumped_rlc(position=(0.005, 0.005, 0.005), component="ez",
                       R=R, L=L, C=C, topology="series")
    sim.add_probe(position=(0.005, 0.005, 0.005), component="ez")
    got = np.asarray(sim.run(n_steps=1500).time_series)
    assert got.shape == golden.shape
    np.testing.assert_allclose(got, golden, rtol=1e-5, atol=1e-2)


def test_forward_no_rlc_byte_identity():
    """A sim with NO lumped RLC is unchanged through forward() (golden).

    Load-bearing falsifier: the entire differentiable capability rides on
    forward(); the new self._lumped_rlc branch must be a pure skip when there
    is no RLC element.
    """
    golden = np.load(os.path.join(_FIXTURE_DIR, "golden_forward_no_rlc_s11.npy"))
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
                     boundary="cpml", cpml_layers=6)
    sim.add_port(position=(0.0093, 0.0093, 0.0093), component="ez",
                 impedance=50.0, waveform=GaussianPulse(f0=5e9, bandwidth=0.9))
    got = np.asarray(sim.forward(
        port_s11_freqs=np.array([3., 4., 5., 6., 7.]) * 1e9, n_steps=1200,
    ).s_params)
    assert got.shape == golden.shape
    np.testing.assert_allclose(got, golden, rtol=1e-5, atol=1e-6)


def test_forward_with_rlc_is_not_noop():
    """A registered RLC element now affects forward() (was a silent no-op).

    GREP note: no pre-existing test combined add_lumped_rlc with forward(), so
    no test locked the old no-op behaviour — this is a pure fix, not a
    contract change.
    """
    freqs = np.array([4.5, 5.0, 5.5]) * 1e9  # in-band, well-conditioned
    with_rlc = np.abs(np.asarray(
        _fixture_sim().forward(port_s11_freqs=freqs, n_steps=1200).s_params
    ).reshape(-1))

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
                     boundary="cpml", cpml_layers=6)
    sim.add_port(position=_POS, component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=_F0, bandwidth=0.9))
    without = np.abs(np.asarray(
        sim.forward(port_s11_freqs=freqs, n_steps=1200).s_params
    ).reshape(-1))

    assert np.max(np.abs(with_rlc - without)) > 1e-2, (
        "add_lumped_rlc had ~zero effect through forward() — the silent no-op "
        "was not fixed"
    )

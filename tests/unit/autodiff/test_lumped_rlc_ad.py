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
    from tests._x64_compat import enable_x64 as _enable_x64

from rfx import GaussianPulse, Simulation

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")

# Lumped-port + series R+C fixture, the element ONE CELL from the port.  CPML
# keeps the port V/I well-conditioned (see
# tests/unit/sparams/test_run_forward_s11_contract.py); the series R+C exercises
# the ADE carry (capacitor charge updates each step through the traced meta.R /
# meta.dt_over_C_dx) so the scoped-x64 dtype threading is genuinely tested.
# Freqs are pinned in-band (source f0=5 GHz, bw=0.9) where the incident wave is
# strong — below-cutoff / weak-incident bins would NaN the gradient silently.
#
# WHY THE ELEMENT IS NOT AT THE PORT CELL (2026-09-21): a driven port reads its
# S11 from the terminal V/I pair, and the Ampere-loop current I is the current
# leaving the port cell into the surrounding field.  An element sitting INSIDE
# that cell is in parallel with the source, not in the external network, so it
# changes the drive level and not the external impedance.  Measured on this
# fixture: co-located, R = 50 and R = 500 move |S11| by 1.8e-07 and 2.4e-07 —
# the float32 floor, R-independent — and dS11^2/dR collapses to -1.15e-10.  One
# cell away at R = 500 the same quantities are 8.7e-03 and -8.16e-05.  The old
# co-located fixture could only see the element because the pre-2026-09-21
# extractor read a pre-injection sample at the driven cell, which carries the
# drive level; that sample is not the terminal voltage of the circuit
# (scripts/diagnostics/lumped_port_known_load_line.py).
#
# R0 is 500, not 50: a series R+C off the port cell with R = 50 NaNs the run,
# on this commit and on the commit before it — a pre-existing behaviour of
# add_lumped_rlc away from a port cell, unrelated to the extractor.
_POS = (0.0093, 0.0093, 0.0093)
_F0 = 5e9
_FREQS = np.array([4.5, 5.0, 5.5]) * 1e9
_R0 = 500.0
_C0 = 0.20e-12
_N_STEPS = 1600
_RLC_OFFSET_CELLS = 1

# Central-difference relative step, DERIVED — not tuned to this fixture.
# The total error of (f(x+h)-f(x-h))/2h is (h^2/6)|f'''| + eps_solve*|f|/h,
# minimised at h/x0 ~ (3*eps_solve)^(1/3) for an O(1)-scaled objective. The
# FDTD solve runs its fields in float32 even under scoped x64 (JAX truncates
# the requested f64 arrays — see the dtype warnings this file emits), so
# eps_solve is the float32 epsilon and the cube-root rule gives ~7.1e-3.
#
# ROOT CAUSE this replaces (item B2 review, 2026-09-05): the previous 1e-4
# sat deep in the ROUND-OFF-dominated branch. Measured on this fixture, the
# ABSOLUTE AD-vs-FD discrepancy at h/x0 = 1e-4 is ~5.2e-6 with the port-current
# phase correction disabled and ~5.4e-6 with it enabled — i.e. a fixed noise
# floor, unchanged by the primal edit — while |dS11^2/dR| itself moved 6.4x
# (2.55e-4 -> 3.99e-5). The RELATIVE error therefore crossed the 5% gate
# (1.98% -> 11.9%) purely because the signal shrank, with no AD/primal
# inconsistency anywhere. Measured rel error vs step, correction off/on:
#   h/x0   1e-4      3e-4     1e-3     3e-3     1e-2
#   off    1.98%     0.79%    0.06%    0.09%    0.003%
#   on    11.9%      5.29%    0.41%    0.61%    0.016%
# The derived 7.1e-3 lands inside the converged plateau in BOTH columns, so
# the 5% gate below now tests AD-vs-FD agreement rather than FD round-off.
# The gate itself is UNCHANGED at 5%.
_FD_REL_STEP = float((3.0 * np.finfo(np.float32).eps) ** (1.0 / 3.0))


_DX = 0.02 / 15
_RLC_POS = (_POS[0] + _RLC_OFFSET_CELLS * _DX, _POS[1], _POS[2])


def _fixture_sim():
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=_DX,
        boundary="cpml", cpml_layers=6,
    )
    sim.add_port(position=_POS, component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=_F0, bandwidth=0.9))
    sim.add_lumped_rlc(position=_RLC_POS, component="ez",
                       R=_R0, C=_C0, topology="series")
    return sim


def _bare_port_sim():
    """The same fixture with no lumped element registered."""
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=_DX,
        boundary="cpml", cpml_layers=6,
    )
    sim.add_port(position=_POS, component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=_F0, bandwidth=0.9))
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
    """grad(|S11|^2) w.r.t. R and C is finite, nonzero and FD-consistent.

    Measured on the committed fixture 2026-09-21: dR AD -8.156e-05 vs FD
    -8.155e-05 (rel 0.005%), dC AD -7.3377e+10 vs FD -7.3375e+10 (rel 0.002%),
    against the unchanged 5% gate. With the element at the port cell instead,
    dS11^2/dR is -1.15e-10 — no signal for either side to agree on — which is
    why the fixture moves it one cell out; see the fixture note above.
    """
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

        # Central finite differences at the DERIVED step (see _FD_REL_STEP:
        # the FDTD fields are float32 regardless of the scoped x64, so the FD
        # round-off floor is set by the float32 epsilon, not by x64).
        hR = _R0 * _FD_REL_STEP
        hC = _C0 * _FD_REL_STEP
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
    (On-platform this is exact array_equal; a tight tolerance guards against
    real numeric regressions while tolerating cross-machine float32 noise.)

    Re-pinned for #1163. The golden recorded before it GREW: the series
    50 ohm + 10 nH + 1 pF element in this closed, lossless PEC box multiplied
    the probe field by ~5.5 every 150 steps, from 1.1e-3 to 3.7e3 by step 1500
    -- the explicit coupling's resistance R - d/(D0*A) was negative. A passive
    element in a lossless cavity cannot add energy; with the element solved
    together with its edge field the probe stays below 7.4e-4 for all 1500
    steps. There is no closed form for this cavity; the element's impedance is
    gated against one in tests/oracle/test_series_rlc_load_on_line.py. atol was
    1e-2 against a 3.7e3 peak (2.7e-6 of it); it is now 1e-6 against a 7.4e-4
    peak, so the golden still has something to hold. Not tighter: the same run
    in float64 fields differs from this float32 golden by 1.1e-7 (#1163
    review), so a cross-platform float32 reorder can reach that scale; the
    replaced update departs from this golden by more than 1e-5 from step 1
    and by more than 1e-3 from step 108.
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
    np.testing.assert_allclose(got, golden, rtol=1e-5, atol=1e-6)


def test_forward_no_rlc_byte_identity():
    """A sim with NO lumped RLC is unchanged through forward() (golden).

    Load-bearing falsifier: the entire differentiable capability rides on
    forward(); the new self._lumped_rlc branch must be a pure skip when there
    is no RLC element.
    """
    # #1012 + 965b2db8 re-pin, CPU float32, regenerate_forward_no_rlc_s11_golden.py:
    # max|S11 - main's golden| = 1.773586155e-5;
    # max|S11 - #1012's previous golden| = 5.139599368e-2.
    # Both changes are in this golden; tolerances unchanged.
    # WHY: the port sits in a six-layer CPML box, and #1012 samples the magnetic CPML profile at the Yee half cell,
    # and since 965b2db8 the port loads only its own edge. The test still guards what it
    # was written for: forward() without an RLC element is a pure skip.
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
    """A registered RLC element reaches forward() (was a silent no-op).

    GREP note: no pre-existing test combined add_lumped_rlc with forward(), so
    no test locked the old no-op behaviour — this is a pure fix, not a
    contract change.

    The gate is 1e-3 against a measured 8.68e-3; a silent no-op reads 1.8e-7,
    four orders below it, so noise cannot pass this. The element must sit off
    the port cell to be visible at all — see the fixture note above.
    """
    freqs = np.array([4.5, 5.0, 5.5]) * 1e9  # in-band, well-conditioned
    with_rlc = np.abs(np.asarray(
        _fixture_sim().forward(port_s11_freqs=freqs, n_steps=1200).s_params
    ).reshape(-1))
    without = np.abs(np.asarray(
        _bare_port_sim().forward(port_s11_freqs=freqs, n_steps=1200).s_params
    ).reshape(-1))

    assert np.max(np.abs(with_rlc - without)) > 1e-3, (
        "add_lumped_rlc had ~zero effect through forward() — the silent no-op "
        f"was not fixed (with {with_rlc}, without {without})"
    )


def test_run_compute_s_params_reflects_rlc():
    """run(compute_s_params=True) now also reflects a registered RLC (bonus fix).

    The S-param extraction shares ``_forward_from_materials`` with ``forward()``,
    so threading ``lumped_rlc`` fixes both. Before WP 4-E, run(compute_s_params=
    True) was byte-identical with/without a co-located RLC (max|Δ|=0.0 on main);
    now it differs, locking the broader no-op fix so it can't silently regress.

    Same gate and same measured separation as the forward() sibling above
    (8.68e-3 measured, 1e-3 gate, 1.8e-7 no-op floor).
    """
    freqs = np.array([4.5, 5.0, 5.5]) * 1e9  # in-band, well-conditioned
    with_rlc = np.abs(np.asarray(
        _fixture_sim().run(n_steps=1200, compute_s_params=True,
                           s_param_freqs=freqs).s_params
    ).reshape(-1))
    without = np.abs(np.asarray(
        _bare_port_sim().run(n_steps=1200, compute_s_params=True,
                             s_param_freqs=freqs).s_params
    ).reshape(-1))

    assert np.max(np.abs(with_rlc - without)) > 1e-3, (
        "add_lumped_rlc had ~zero effect through run(compute_s_params=True) — "
        "the S-param extraction path still drops the RLC"
    )

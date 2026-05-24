"""G-AD: End-to-end autodiff smoke test for forward → S-parameter path.

Goal: determine whether jax.grad flows through the *real* FDTD forward
→ compute_msl_s_matrix / compute_waveguide_s_matrix → scalar objective,
and if not, precisely locate where the tape breaks.

WI-1/WI-2 proved the S-matrix *assembly* is jnp-native (AD-traceable
when called on synthetic accumulators).  This file proves — or diagnoses
— whether the *forward scan* (real FDTD) feeds into that assembly as a
differentiable input.

Findings (2026-05-24)
---------------------
Both ``compute_msl_s_matrix`` and ``compute_waveguide_s_matrix`` break
the JAX tape.  They are IMPERATIVE, STATEFUL workflows that are NOT
differentiable end-to-end.  Specific break points:

MSL tape break — two-level:
  1. PRIMARY (architectural): ``compute_msl_s_matrix`` calls
     ``self.run()`` (rfx/api/_sparams.py line ~850) which has NO
     ``eps_override`` parameter.  There is no differentiable input
     channel into the FDTD forward scan; jax.grad has nothing to
     trace through.

  2. SECONDARY (even if (1) were fixed): after the scan, field
     planes are concretised via ``hy_plane = np.asarray(...)`` and
     ``hz_plane = np.asarray(...)`` (rfx/api/_sparams.py lines 870-871).
     This would break any tracer that survived the run.

Waveguide tape break — two-level:
  1. PRIMARY (architectural): ``compute_waveguide_s_matrix`` accepts
     no ``eps_override``; ``extract_waveguide_s_matrix`` (called at
     rfx/api/_sparams.py ~line 490) builds a PLAIN NumPy accumulator
     ``s_matrix = np.zeros(...)`` (rfx/sources/waveguide_port.py
     line 1952) and writes results with ``s_matrix[...] = np.array(...)``
     (line 2000), concretising every JAX array.

  2. SECONDARY: even if the accumulator were jnp-native, there is no
     differentiable input channel into ``extract_waveguide_s_matrix``
     (``materials`` is passed as a concrete assembled struct with no
     tracer path from a user-controlled scalar).

Correct differentiable path (already validated):
  ``sim.forward(eps_override=eps0 * alpha)`` → ``result.time_series``
  → scalar loss → ``jax.grad``.  This works and is demonstrated in
  ``examples/inverse_design/ad_gradient_demo.py`` and
  ``tests/test_waveguide_forward.py::test_forward_with_waveguide_port_is_differentiable``.

The tests below are marked ``xfail`` with a precise tape-break reason.
They do NOT fake a pass; a genuine end-to-end gradient is not reachable
through the current public compute_*_s_matrix API.

A future "G-AD-WIRE" task could connect the differentiable path by:
  - Adding an ``eps_override`` kwarg to ``compute_msl_s_matrix`` and
    forwarding it into ``self.forward()`` (replacing ``self.run()``),
    then converting the MSL DFT-plane assembly to use ``result.dft_planes``
    from ``forward()`` instead of the stateful ``self.run()`` result.
  - Replacing the ``np.zeros`` / ``np.array`` accumulator in
    ``extract_waveguide_s_matrix`` with a jnp-native accumulator and
    passing ``eps_override`` through ``run_simulation``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

# ---------------------------------------------------------------------------
# Tiny MSL thru-line geometry (CPU-fast for AD diagnosis)
# ---------------------------------------------------------------------------
# Deliberately small: 3-cell substrate, minimal line length and lateral extent.
# Geometry mirrors the validated pattern from test_msl_port_integration.py
# but drastically reduced to keep per-grad time < 60 s on CPU.

_MSL_EPS_R = 3.66       # RO4350B
_MSL_H_SUB = 254e-6     # substrate thickness (m)
_MSL_W_TRACE = 600e-6   # trace width (m)
_MSL_DX = 80e-6         # cell size (m) → ~3 substrate cells
_MSL_L_LINE = 4e-3      # short line length (vs 10 mm in integration test)
_MSL_PORT_MARGIN = 1e-3 # smaller margin for speed
_MSL_F_MAX = 5e9


def _build_msl_sim() -> Simulation:
    """Tiny MSL thru-line sim (2 ports, minimal domain)."""
    lx = _MSL_L_LINE + 2 * _MSL_PORT_MARGIN
    ly = _MSL_W_TRACE + 2 * (2 * _MSL_H_SUB + 8 * _MSL_DX)
    lz = _MSL_H_SUB + 0.5e-3  # thin air layer above substrate

    sim = Simulation(
        freq_max=_MSL_F_MAX,
        domain=(lx, ly, lz),
        dx=_MSL_DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )

    sim.add_material("ro4350b", eps_r=_MSL_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _MSL_H_SUB)), material="ro4350b")

    y_centre = ly / 2.0
    trace_y_lo = y_centre - _MSL_W_TRACE / 2.0
    trace_y_hi = y_centre + _MSL_W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, _MSL_H_SUB), (lx, trace_y_hi, _MSL_H_SUB + _MSL_DX)),
        material="pec",
    )

    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="+x",
        impedance=50.0,
    )
    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN + _MSL_L_LINE, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="-x",
        impedance=50.0,
    )
    return sim


# ---------------------------------------------------------------------------
# Tiny WR-90 2-port waveguide geometry (CPU-fast for AD diagnosis)
# ---------------------------------------------------------------------------

_WR90_A = 22.86e-3   # WR-90 broad wall (m)
_WR90_B = 10.16e-3   # WR-90 narrow wall (m)
_WR90_DX = 2e-3      # cell size
_WR90_LX = 0.05      # domain length


def _build_wg_sim() -> Simulation:
    """Tiny 2-port WR-90 sim (minimal domain, CPU-fast)."""
    sim = Simulation(
        freq_max=12e9,
        domain=(_WR90_LX, _WR90_A, _WR90_B),
        dx=_WR90_DX,
        boundary="cpml",
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        direction="+x",
        x_position=0.01,
        y_range=(0.0, _WR90_A),
        z_range=(0.0, _WR90_B),
        n_modes=1,
    )
    sim.add_waveguide_port(
        direction="-x",
        x_position=_WR90_LX - 0.01,
        y_range=(0.0, _WR90_A),
        z_range=(0.0, _WR90_B),
        n_modes=1,
    )
    return sim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    raises=(TypeError, AttributeError, ValueError, Exception),
    reason=(
        "TAPE BREAK (architectural): compute_msl_s_matrix calls self.run() "
        "(rfx/api/_sparams.py ~line 850) which has NO eps_override parameter. "
        "There is no differentiable input channel into the FDTD scan. "
        "Even if one were added, np.asarray() calls on lines 870-871 "
        "of rfx/api/_sparams.py would concretise field planes and break "
        "any surviving tracer. jax.grad returns None/zero or raises a "
        "concretisation error. "
        "Fix: route compute_msl_s_matrix through forward(eps_override=...) "
        "and replace np.asarray field reads with jnp-native DFT accumulation."
    ),
)
def test_msl_s_matrix_ad_end_to_end():
    """Attempt jax.grad through real forward → compute_msl_s_matrix.

    EXPECTED OUTCOME: xfail — the tape breaks because compute_msl_s_matrix
    uses self.run() (no differentiable input) and np.asarray() concretisation.
    See module docstring for precise tape-break evidence.
    """
    sim = _build_msl_sim()

    # We need to inject a differentiable scalar into compute_msl_s_matrix.
    # The public API has no eps_override parameter, so we attempt the only
    # feasible approach: monkeypatch the material assembly to use a traced
    # eps_r.  This is deliberately fragile — it tests whether the tape
    # survives from a traced eps all the way to the S-matrix scalar.

    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        # alpha is a JAX tracer scalar; try to inject it via the material
        # assembly into compute_msl_s_matrix.
        # NOTE: compute_msl_s_matrix calls self._assemble_materials(grid)
        # internally, not accepting eps_override.  There is no public way
        # to pass a tracer in, so we attempt to override the internal
        # assembly path.
        orig_assemble = sim._assemble_materials

        def patched_assemble(g):
            result = orig_assemble(g)
            mats = result[0]
            from rfx.core.yee import MaterialArrays
            new_eps = mats.eps_r * alpha  # inject tracer
            new_mats = MaterialArrays(
                eps_r=new_eps, sigma=mats.sigma, mu_r=mats.mu_r
            )
            return (new_mats,) + result[1:]

        sim._assemble_materials = patched_assemble
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sim.compute_msl_s_matrix(n_freqs=10, num_periods=3)
            S = result.S
            k0 = S.shape[-1] // 2
            loss = jnp.abs(S[0, 0, k0]) ** 2
        finally:
            sim._assemble_materials = orig_assemble
        return loss

    alpha0 = jnp.float32(1.0)
    # If this reaches here without error, check the gradient is non-trivial.
    # We expect this to either raise a concretisation error or return a
    # zero/None gradient because self.run() breaks the tape.
    grad_fn = jax.grad(objective)
    g = grad_fn(alpha0)

    # If we somehow reach here, assert the gradient is meaningful.
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert float(jnp.abs(g)) > 1e-10, (
        f"Gradient is effectively zero ({g}): tape is broken silently. "
        "compute_msl_s_matrix uses self.run() (no eps_override) so no "
        "gradient can flow from alpha through the FDTD scan to the S-matrix."
    )


@pytest.mark.xfail(
    strict=True,
    raises=(TypeError, AttributeError, ValueError, Exception),
    reason=(
        "TAPE BREAK (two-level): (1) compute_waveguide_s_matrix accepts no "
        "eps_override, so there is no differentiable input into the FDTD scan; "
        "(2) extract_waveguide_s_matrix (rfx/sources/waveguide_port.py line 1952) "
        "allocates s_matrix = np.zeros(...) and fills it with "
        "s_matrix[...] = np.array(b_recv / safe_a) (line 2000), concretising "
        "every JAX array produced by the run. "
        "Fix: replace the np.zeros/np.array accumulator with a jnp-native "
        "accumulator and add eps_override forwarding through "
        "compute_waveguide_s_matrix → extract_waveguide_s_matrix → "
        "run_simulation."
    ),
)
def test_waveguide_s_matrix_ad_end_to_end():
    """Attempt jax.grad through real forward → compute_waveguide_s_matrix.

    EXPECTED OUTCOME: xfail — the tape breaks because extract_waveguide_s_matrix
    uses a np.zeros accumulator and np.array() concretisation.
    See module docstring for precise tape-break evidence.
    """
    sim = _build_wg_sim()

    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        orig_assemble = sim._assemble_materials

        def patched_assemble(g):
            result = orig_assemble(g)
            mats = result[0]
            from rfx.core.yee import MaterialArrays
            new_eps = mats.eps_r * alpha  # inject tracer
            new_mats = MaterialArrays(
                eps_r=new_eps, sigma=mats.sigma, mu_r=mats.mu_r
            )
            return (new_mats,) + result[1:]

        sim._assemble_materials = patched_assemble
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sim.compute_waveguide_s_matrix(
                    n_steps=40,
                    normalize=False,
                )
            S = result.S
            k0 = S.shape[-1] // 2
            loss = jnp.abs(S[0, 0, k0]) ** 2
        finally:
            sim._assemble_materials = orig_assemble
        return loss

    alpha0 = jnp.float32(1.0)
    grad_fn = jax.grad(objective)
    g = grad_fn(alpha0)

    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert float(jnp.abs(g)) > 1e-10, (
        f"Gradient is effectively zero ({g}): tape is broken silently. "
        "compute_waveguide_s_matrix uses np.zeros/np.array accumulators "
        "so no gradient can flow from alpha through the FDTD scan to S."
    )


# ---------------------------------------------------------------------------
# Positive control: confirm the FORWARD-only path IS differentiable
# (mirrors tests/test_waveguide_forward.py — here for completeness)
# ---------------------------------------------------------------------------

def test_forward_eps_override_is_differentiable_msl():
    """Positive control: jax.grad through forward(eps_override=) + probe loss.

    This does NOT go through compute_msl_s_matrix — it uses the validated
    forward() path to confirm that the non-S-matrix differentiable path
    still works on the MSL geometry.
    """
    sim = Simulation(
        freq_max=_MSL_F_MAX,
        domain=(6e-3, 3e-3, 2e-3),
        dx=_MSL_DX,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
        cpml_layers=4,
    )
    # Tiny probe-based forward (no MSL port — just tests the diff path exists)
    sim.add_port((3e-3, 1.5e-3, 1e-3), "ez", impedance=50.0)
    sim.add_probe((3e-3, 1.5e-3, 1e-3), "ez")

    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss(alpha):
        r = sim.forward(eps_override=eps_base * alpha, n_steps=20)
        return jnp.sum(jnp.abs(r.time_series) ** 2)

    alpha0 = jnp.float32(1.0)
    val = float(loss(alpha0))
    assert np.isfinite(val), f"forward() loss is not finite: {val}"

    g = float(jax.grad(loss)(alpha0))
    assert np.isfinite(g), f"forward() gradient is not finite: {g}"
    # Gradient may be zero on a trivial geometry but the tape must be intact
    # (finite is the minimal bar; non-zero is expected for any non-trivial sim)


def test_forward_eps_override_is_differentiable_waveguide():
    """Positive control: jax.grad through forward(eps_override=) on WR-90 sim.

    Mirrors tests/test_waveguide_forward.py::test_forward_with_waveguide_port_is_differentiable.
    Confirms the forward() tape is intact on the waveguide geometry used in
    the xfail test above.
    """
    sim = _build_wg_sim()
    # Add a probe so forward() returns non-trivial time_series
    sim.add_probe(position=(_WR90_LX / 2, _WR90_A / 2, _WR90_B / 2), component="ey")
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss(alpha):
        r = sim.forward(eps_override=eps_base * alpha, n_steps=20)
        return jnp.sum(jnp.abs(r.time_series) ** 2)

    alpha0 = jnp.float32(1.0)
    val = float(loss(alpha0))
    assert np.isfinite(val), f"forward() loss is not finite: {val}"

    g = float(jax.grad(loss)(alpha0))
    assert np.isfinite(g), f"forward() gradient is not finite: {g}"

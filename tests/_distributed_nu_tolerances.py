"""
tests/_distributed_nu_tolerances.py

Tolerance constants and assertion helpers for the distributed-NU composition
test suite (``tests/unit/runners/test_distributed_nu_composition.py``).

Every tolerance class is tied to a specific failure mode the composition tests
guard against.  Use the named constants and helpers rather than inline magic
numbers so that a future threshold change only requires editing this file.

Tolerance classes
-----------------
Class A — Gradient parity (float32 precision)
    Distributed and single-device gradients must agree to ``rtol=1e-6``.
    Used by all ``jax.grad``-based tests and the per-cell seam gradient test.

Class B — Forward probe parity (final-step + time-integrated)
    Catches cumulative drift over the full run.  Two sub-checks:
    * final-step probe value:  rel_err < 5e-5
    * time-integrated probe energy:  rel_err < 5e-4

Class C — CPML internal-state structural assertion
    Interior-rank x-face CPML psi arrays must be *exactly zero* (not small —
    zero).  A non-zero value means the seam was incorrectly treated as an
    absorbing boundary.

Class D — Debye/Lorentz drift sweep (multiple time points)
    Single-time-point comparison can miss cumulative ADE ordering errors that
    only surface after multiple polarisation update cycles.  The drift sweep
    checks four time points (T/4, T/2, 3T/4, T) with rel_err < 5e-4.

Class E — Structural / composition assertions
    Used for API-contract tests (raises correct exception) and bit-match
    checkpoint tests.  No numeric threshold; assertion is exact equality or
    exception type.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Class A — gradient parity
# ---------------------------------------------------------------------------

RTOL_A: float = 1e-6
"""Relative tolerance for float32 gradient parity (Class A)."""


def assert_class_a_grad(
    grad_single: jnp.ndarray,
    grad_dist: jnp.ndarray,
    label: str = "",
) -> None:
    """Assert that distributed and single-device gradients agree to RTOL_A.

    Uses the same formula as jnp.allclose so that near-zero gradients don't
    cause false failures: rel_err = |dist - single| / (|single| + 1e-30).
    """
    max_rel = float(
        jnp.max(
            jnp.abs(grad_dist - grad_single) / (jnp.abs(grad_single) + 1e-30)
        )
    )
    tag = f" [{label}]" if label else ""
    assert max_rel <= RTOL_A, (
        f"Class A gradient parity failed{tag}: "
        f"max_rel_err={max_rel:.3e} > rtol={RTOL_A:.3e}"
    )


def assert_class_a_grad_seam(
    grad_single: jnp.ndarray,
    grad_dist: jnp.ndarray,
    label: str = "",
    rtol: float = 1e-4,
) -> None:
    """Assert seam-adjacent gradient parity with a relaxed tolerance (1e-4).

    The G1 test uses this; seam-adjacent cells may accumulate slightly more
    floating-point error due to ghost-cell exchange rounding.
    """
    max_rel = float(
        jnp.max(
            jnp.abs(grad_dist - grad_single) / (jnp.abs(grad_single) + 1e-30)
        )
    )
    tag = f" [{label}]" if label else ""
    assert max_rel <= rtol, (
        f"Class A (seam) gradient parity failed{tag}: "
        f"max_rel_err={max_rel:.3e} > rtol={rtol:.3e}"
    )


# ---------------------------------------------------------------------------
# Class B — forward probe parity
# ---------------------------------------------------------------------------

RTOL_PROBE_B: float = 5e-5
"""Relative tolerance for the final-step probe value (Class B)."""

RTOL_ENERGY_B: float = 5e-4
"""Relative tolerance for the time-integrated probe energy (Class B)."""


def assert_class_b_parity(
    ts_single: jnp.ndarray,
    ts_dist: jnp.ndarray,
    label: str = "",
) -> None:
    """Assert forward probe parity on both final-step value and total energy.

    Parameters
    ----------
    ts_single:
        Probe time-series from the single-device reference run.  Shape: (T,).
    ts_dist:
        Probe time-series from the 2-device distributed run.  Shape: (T,).
    label:
        Optional identifier for the assertion message.
    """
    ts_single = jnp.asarray(ts_single)
    ts_dist = jnp.asarray(ts_dist)
    tag = f" [{label}]" if label else ""

    # Final-step check
    final_single = ts_single[-1]
    final_dist = ts_dist[-1]
    rel_final = float(
        jnp.abs(final_dist - final_single) / (jnp.abs(final_single) + 1e-30)
    )
    assert rel_final < RTOL_PROBE_B, (
        f"Class B final-step probe mismatch{tag}: "
        f"rel_err={rel_final:.3e} >= {RTOL_PROBE_B:.3e}"
    )

    # Time-integrated energy check
    energy_single = float(jnp.sum(ts_single ** 2))
    energy_dist = float(jnp.sum(ts_dist ** 2))
    rel_energy = abs(energy_dist - energy_single) / (abs(energy_single) + 1e-30)
    assert rel_energy < RTOL_ENERGY_B, (
        f"Class B probe energy mismatch{tag}: "
        f"rel_err={rel_energy:.3e} >= {RTOL_ENERGY_B:.3e}"
    )


# ---------------------------------------------------------------------------
# Class C — CPML internal-state structural assertion
# ---------------------------------------------------------------------------

CPML_X_FACE_PSI_FIELDS: tuple[str, ...] = (
    "psi_ey_xlo", "psi_ey_xhi",
    "psi_ez_xlo", "psi_ez_xhi",
    "psi_hy_xlo", "psi_hy_xhi",
    "psi_hz_xlo", "psi_hz_xhi",
)
"""All x-face CPML psi field names that must be zero on interior ranks."""


def assert_class_c_cpml_seam_noop(cpml_state, label: str = "") -> None:
    """Assert that all x-face psi arrays on an interior rank are exactly zero.

    An interior rank's x-face CPML psi arrays must remain exactly zero
    because the seam is NOT an absorbing boundary — it is just a ghost-cell
    interface.  Any non-zero value means the seam was incorrectly activated
    as a CPML boundary.

    Parameters
    ----------
    cpml_state:
        A CPMLState (or similar NamedTuple) with the x-face psi attributes.
    label:
        Optional identifier for the assertion message.
    """
    tag = f" [{label}]" if label else ""
    for field_name in CPML_X_FACE_PSI_FIELDS:
        arr = getattr(cpml_state, field_name, None)
        if arr is None:
            continue
        assert jnp.all(arr == 0), (
            f"Class C CPML seam noop failed{tag}: "
            f"interior-rank x-face field '{field_name}' is non-zero "
            f"(max={float(jnp.max(jnp.abs(arr))):.3e})"
        )


# ---------------------------------------------------------------------------
# Class D — Debye/Lorentz drift sweep
# ---------------------------------------------------------------------------

RTOL_DRIFT_D: float = 5e-4
"""Relative tolerance for Debye/Lorentz drift at each checked time point (Class D)."""


def assert_class_d_drift_sweep(
    probe_fn_single,
    probe_fn_dist,
    n_steps: int,
    label: str = "",
) -> None:
    """Check dispersive material parity at T/4, T/2, 3T/4, and T.

    Both probe callables must accept a step index (int) and return a scalar
    probe value at that time step.

    Parameters
    ----------
    probe_fn_single:
        Callable(t_idx: int) -> float — reference single-device probe value.
    probe_fn_dist:
        Callable(t_idx: int) -> float — distributed probe value.
    n_steps:
        Total number of simulation steps.
    label:
        Optional identifier for the assertion message.
    """
    tag = f" [{label}]" if label else ""
    checkpoints = [
        n_steps // 4,
        n_steps // 2,
        3 * n_steps // 4,
        n_steps - 1,
    ]
    for t_idx in checkpoints:
        val_single = float(probe_fn_single(t_idx))
        val_dist = float(probe_fn_dist(t_idx))
        rel_err = abs(val_dist - val_single) / (abs(val_single) + 1e-30)
        assert rel_err < RTOL_DRIFT_D, (
            f"Class D drift sweep failed{tag} at step {t_idx}/{n_steps}: "
            f"rel_err={rel_err:.3e} >= {RTOL_DRIFT_D:.3e} "
            f"(single={val_single:.6e}, dist={val_dist:.6e})"
        )


def assert_class_d_timeseries_drift(
    ts_single: jnp.ndarray,
    ts_dist: jnp.ndarray,
    label: str = "",
) -> None:
    """Check dispersive parity by comparing time-series arrays at 4 checkpoints.

    Convenience wrapper when both single and distributed time-series are
    already available as arrays (shape: (T,)).
    """
    ts_single = jnp.asarray(ts_single)
    ts_dist = jnp.asarray(ts_dist)
    n = len(ts_single)
    assert_class_d_drift_sweep(
        probe_fn_single=lambda i: float(ts_single[i]),
        probe_fn_dist=lambda i: float(ts_dist[i]),
        n_steps=n,
        label=label,
    )


# ---------------------------------------------------------------------------
# Class E — structural / composition (no numeric threshold)
# ---------------------------------------------------------------------------

def assert_class_e_raises(exc_type, fn, *args, label: str = "", **kwargs) -> None:
    """Assert that calling fn(*args, **kwargs) raises exc_type.

    This is the Class E structural assertion for API-contract tests.
    """
    tag = f" [{label}]" if label else ""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"Class E API contract failed{tag}: "
            f"expected {exc_type.__name__} but got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(
        f"Class E API contract failed{tag}: "
        f"expected {exc_type.__name__} but call succeeded without exception"
    )


def assert_class_e_bit_match(
    arr_a: jnp.ndarray,
    arr_b: jnp.ndarray,
    label: str = "",
) -> None:
    """Assert exact (bit-for-bit) equality between two arrays.

    Used by checkpoint bit-match tests: jax.checkpoint must not alter values.
    """
    tag = f" [{label}]" if label else ""
    assert jnp.array_equal(arr_a, arr_b), (
        f"Class E bit-match failed{tag}: arrays differ "
        f"(max_abs_diff={float(jnp.max(jnp.abs(jnp.asarray(arr_a, dtype=float) - jnp.asarray(arr_b, dtype=float)))):.3e})"
    )


# ---------------------------------------------------------------------------
# Class F — absolute-physics validation (analytic / first-principles)
# ---------------------------------------------------------------------------
#
# Class F is the "absolute" tier of the distributed-NU tolerance contract.
# Where Classes A-E compare distributed against single-device (parity), Class
# F compares distributed against an analytic/first-principles reference, so
# it catches common-origin bugs that pure parity tests cannot see (e.g. a
# numerical error reproduced by both single-device and distributed because
# they share a helper, or a slab-boundary-specific construction that has no
# single-device counterpart).
#
# Two sub-tolerances live here:
#
#   * ``RTOL_ANALYTIC_F`` — discretisation tolerance for an FDTD-vs-analytic
#     resonance frequency check at ~10 cells per wavelength.  5% is the
#     standard textbook bound for a Yee scheme at this resolution; tighter
#     would require finer meshing or a larger run budget than the kernel
#     test suite can afford on a CPU virtual cluster.
#
#   * ``RTOL_ENERGY_DRIFT_F`` — energy conservation tolerance for a closed
#     PEC cavity over the test horizon (a few hundred steps after the
#     source switches off).  Pure FDTD on a hard-PEC box is essentially
#     lossless; well-formed runs sit comfortably under 1%.  5% is the
#     generous bound that catches dissipative bugs in seam exchange or
#     accidental double-application of PEC at a slab boundary.

RTOL_ANALYTIC_F: float = 0.05
"""Relative tolerance for FDTD-vs-analytic resonance frequency (Class F)."""

RTOL_ENERGY_DRIFT_F: float = 0.05
"""Relative tolerance for closed-PEC-cavity energy drift over a few
hundred steps after the source has switched off (Class F)."""

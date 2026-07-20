"""CPU coverage for the nonuniform _compute_waveguide_s_matrix_nu jnp assembly.

G-NU acceptance test.  Approach: (a) small real nonuniform-grid WR-90 2-port
run with a graded dx_profile (which triggers the _nu dispatch path via the
dx_profile != None guard in compute_waveguide_s_matrix).

Why approach (a) over mock/replay:
  - The dispatch condition is ``self._dx_profile is not None``.  A real sim
    exercises the full chain: build_nonuniform_grid -> assemble_materials_nu
    -> run_nonuniform_path (x2) -> extract_waveguide_port_waves -> jnp.stack
    assembly.  A replay would only cover the last two steps.
  - The existing preflight test (test_compute_waveguide_s_matrix_dispatches_nu
    _when_normalized) already confirmed end-to-end runs complete; this file
    adds the physics gate (passivity) + AD check + per-freq trace (R5).

Domain: WR-90 air-filled waveguide, a=22.86mm, b=10.16mm.
Mesh: graded dx_profile (coarse/fine/coarse along x), uniform dy/dz.

FIXED (NU-DRIVE-FIX 2026-05-25):
  _compute_waveguide_s_matrix_nu now returns physical S-parameters (|S21|≈1,
  |S11|≈0) for a WR-90 air thru with dx_profile grading. Two bugs were fixed:
  1. rfx/runners/nonuniform.py _build_waveguide_port_config_nu: missing
     dt=float(grid.dt) in init_waveguide_port call — e_inc_table/h_inc_table
     remained size-1 sentinels (dt=0 path) so _rect_dft used dt=0 → all DFT
     outputs zero, producing all-zero S.
  2. rfx/api/_sparams.py _compute_waveguide_s_matrix_nu: off-diagonal safe_b
     guard threshold 1e-30 too aggressive — NU TFSF operates at float32 signal
     level ~1e-31 (dt/dx scaling), causing the guard to substitute 1.0 for
     b_ref and breaking the b_dev/b_ref≈1 cancellation. Lowered to 1e-60.
"""
from __future__ import annotations

import warnings

import pytest

import jax
import jax.numpy as jnp
import numpy as np

# NOTE: do NOT set module-level jax.config.update("jax_enable_x64", True) here —
# it permanently flips x64 ON for the whole pytest process and leaks into
# downstream same-process tests (test_wire_*/test_verification then fail with
# lax.scan carry-dtype TypeErrors). These NU tests need only float32 (they check
# finite/shape/liveness, no f64-precision assertion). If a future test here needs
# x64, scope it with `with jax.experimental.enable_x64(True):` like
# tests/test_msl_sparam_ad.py — never module-level.

# ---------------------------------------------------------------------------
# WR-90 dimensions
# ---------------------------------------------------------------------------
_A_WG = 0.02286   # m  (broad wall)
_B_WG = 0.01016   # m  (narrow wall)
_F_MAX = 12e9
_FREQS = jnp.linspace(8.2e9, 12.4e9, 5)


def _make_wr90_nu_sim():
    """Minimal 2-port WR-90 sim with a graded dx_profile (triggers _nu path).

    Uses a coarse/fine/coarse dx_profile so dx_profile is not None and the
    dispatch in compute_waveguide_s_matrix routes to _compute_waveguide_s_matrix_nu.
    """
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary
    from rfx.auto_config import smooth_grading

    dx_coarse = 1.5e-3
    dx_fine = 0.75e-3

    # Build a graded profile: coarse - fine (centre 20mm) - coarse
    n_pre = int(round(0.030 / dx_coarse))
    n_fine = int(round(0.040 / dx_fine))
    n_post = int(round(0.030 / dx_coarse))
    raw = np.concatenate([
        np.full(n_pre, dx_coarse),
        np.full(n_fine, dx_fine),
        np.full(n_post, dx_coarse),
    ])
    dx_profile = smooth_grading(raw, max_ratio=1.3)
    domain_x = float(np.sum(dx_profile))

    sim = Simulation(
        freq_max=_F_MAX,
        domain=(domain_x, _A_WG, _B_WG),
        dx=dx_coarse,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
        dx_profile=dx_profile,
    )
    sim.add_waveguide_port(
        0.015, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.020, name="left",
    )
    sim.add_waveguide_port(
        domain_x - 0.015, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=domain_x - 0.020, name="right",
    )
    return sim


# ---------------------------------------------------------------------------
# Helper: run the NU sim, return result (reused across tests via module-level
# cache to avoid running the sim twice).
# ---------------------------------------------------------------------------
_cached_result = None


def _get_result():
    global _cached_result
    if _cached_result is None:
        sim = _make_wr90_nu_sim()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cached_result = sim.compute_waveguide_s_matrix(
                num_periods=2, normalize=True,
            )
    return _cached_result


# ---------------------------------------------------------------------------
# Test 1: dispatch sanity — result has correct shape and is a JAX array
# ---------------------------------------------------------------------------

def test_nu_dispatch_returns_jax_array():
    """_compute_waveguide_s_matrix_nu must return s_params as a jax.Array."""
    res = _get_result()
    assert isinstance(res.s_params, jax.Array), (
        f"s_params type is {type(res.s_params)}; expected jax.Array. "
        "The _nu assembly may be wrapping in numpy instead of jnp.stack."
    )
    assert isinstance(res.freqs, jax.Array), (
        f"freqs type is {type(res.freqs)}; expected jax.Array."
    )
    assert res.s_params.shape == (2, 2, 5), (
        f"Expected shape (2, 2, 5) for 2-port 5-freq result; got {res.s_params.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: finite S — no NaN/Inf anywhere in s_params
# ---------------------------------------------------------------------------

def test_nu_s_params_finite():
    """All entries of s_params must be finite (no NaN/Inf).

    R5: per-frequency dump — full trace, not a bare scalar.
    NOTE: with the NU-DRIVE-FIX applied, S is now physically correct (|S21|≈1,
    |S11|≈0). Finite check still passes. Liveness/identity gate in
    test_nu_s_params_dispatch_and_dft_liveness (real NU physics:
    test_waveguide_nu_nontrivial.py).
    """
    res = _get_result()
    s = np.array(res.s_params)

    # R5: per-frequency dump — full trace, not a bare scalar
    print("\n[G-NU] Per-frequency S-parameter trace (2-port, 5 freqs):")
    freqs_ghz = np.array(res.freqs) / 1e9
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            vals = s[i, j, :]
            mags = np.abs(vals)
            print(
                f"  S[{i},{j},:] |S| = {np.array2string(mags, precision=4)}  "
                f"max|S|={mags.max():.4f}  "
                f"freqs(GHz)={np.array2string(freqs_ghz, precision=2)}"
            )

    assert np.all(np.isfinite(s)), (
        "s_params contains NaN or Inf — _compute_waveguide_s_matrix_nu "
        "assembly failure. Per-freq trace printed above."
    )


# ---------------------------------------------------------------------------
# Test 3: NU dispatch / DFT-liveness gate on the air-thru identity
# (NOT a physical passivity gate — issue #395)
# ---------------------------------------------------------------------------



def test_nu_s_params_dispatch_and_dft_liveness():
    """NU air-thru returns the two-run identity (dispatch + DFT are live).

    ISSUE #395 — what this actually binds: this is an air-filled WR-90 THRU
    with ``normalize=True``, so the device run *is* the reference run and
    ``|S21|≈1`` / ``|S11|≈0`` hold BY CONSTRUCTION (device==reference
    cancellation), independent of NU physics. Verified 2026-07-20: |S11|
    range [0.0000, 0.0000], |S21| range [1.0000, 1.0000]. This is therefore
    a DISPATCH + DFT-liveness gate — it catches the NU-DRIVE-FIX failure mode
    (all-zero S, i.e. |S21| collapses to 0 when the DFT runs with dt=0), NOT
    a physical passivity/reflection measurement. Real NU S-physics on a
    device != reference geometry lives in ``test_waveguide_nu_nontrivial.py``
    (live invariants) and the ``nu_broad_e4`` Palace replay.

    FIXED (NU-DRIVE-FIX 2026-05-25): _compute_waveguide_s_matrix_nu now
    produces the live identity instead of all-zero S. Two bugs were fixed:
      1. rfx/runners/nonuniform.py _build_waveguide_port_config_nu: missing
         dt=float(grid.dt) in init_waveguide_port call — e_inc_table/h_inc_table
         remained size-1 sentinels (dt=0 path), so _rect_dft used dt=0 → all
         DFT outputs zero (|S21| would read 0, tripping the liveness assert).
      2. rfx/api/_sparams.py _compute_waveguide_s_matrix_nu: off-diagonal
         safe_b guard threshold 1e-30 too aggressive — NU TFSF operates at
         float32 signal level ~1e-31 (dt/dx scaling), causing the guard to
         substitute 1.0 for b_ref, breaking the b_dev/b_ref≈1 cancellation.
         Lowered to 1e-60.

    Gates (liveness/identity bounds, not physics):
      - max |S_ij| < 2.0  (no assembly explosion)
      - |S11| < 0.50  (air-thru identity keeps the diagonal near 0)
      - |S21| > 0.05  (DFT is live — the all-zero-S regression drops it to 0)
    """
    res = _get_result()
    s = np.array(res.s_params)

    s11_mag = np.abs(s[0, 0, :])
    s21_mag = np.abs(s[1, 0, :])

    print(f"\n[G-NU] |S11| range: [{s11_mag.min():.4f}, {s11_mag.max():.4f}]")
    print(f"[G-NU] |S21| range: [{s21_mag.min():.4f}, {s21_mag.max():.4f}]")

    assert np.all(np.abs(s) < 2.0), (
        f"max|S|={np.abs(s).max():.4f} >= 2.0 — energy explosion in _nu run."
    )
    assert s11_mag.max() < 0.50, (
        f"|S11| max = {s11_mag.max():.4f}; air-thru two-run identity should "
        "keep the diagonal near 0."
    )
    assert s21_mag.max() > 0.05, (
        f"|S21| max = {s21_mag.max():.4f}; expected > 0.05 — DFT not live. "
        "BUG: NU path returns zero S21 (dt=0 sentinel) while the live identity "
        "gives ~1.0."
    )


# ---------------------------------------------------------------------------
# Test 4: uniform vs nonuniform comparison — documents the divergence (R5)
# ---------------------------------------------------------------------------

def test_nu_vs_uniform_finite_and_comparable():
    """Side-by-side NU vs uniform S-parameter dump (R5 mandate).

    Post NU-DRIVE-FIX (2026-05-25) the NU and uniform air-thru paths AGREE:
    both return the two-run identity |S21|=|S12|≈1, |S11|≈0, so the
    per-freq deviation is ~0 (measured max|delta|=0.0000 on 2026-07-20 —
    NOT the ~1.0 the pre-fix all-zero bug produced; the stale "NU returns
    all-zero" note was removed in issue #395). The loose numeric gate
    (max|delta| < 1.5) still passes and guards against an assembly-level
    NaN/explosion on either path; it is not a physics-accuracy gate (both
    sides are the device==reference identity, so agreement is expected by
    construction — real NU physics is in test_waveguide_nu_nontrivial.py).
    """
    # Nonuniform result (already cached)
    res_nu = _get_result()
    s_nu = np.array(res_nu.s_params)

    # Uniform counterpart: same WR-90, same ports, no dx_profile
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    dom_x = 0.100
    sim_u = Simulation(
        freq_max=_F_MAX,
        domain=(dom_x, _A_WG, _B_WG),
        dx=1.5e-3,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    sim_u.add_waveguide_port(
        0.015, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.020, name="left",
    )
    sim_u.add_waveguide_port(
        dom_x - 0.015, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=dom_x - 0.020, name="right",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_u = sim_u.compute_waveguide_s_matrix(num_periods=2, normalize=True)
    s_u = np.array(res_u.s_params)

    # Shape parity
    assert s_nu.shape == s_u.shape, (
        f"Shape mismatch: NU={s_nu.shape}, uniform={s_u.shape}"
    )

    # Per-frequency overlay (R5)
    print("\n[G-NU] NU vs Uniform per-freq |S| comparison:")
    for i in range(s_nu.shape[0]):
        for j in range(s_nu.shape[1]):
            nu_mags = np.abs(s_nu[i, j, :])
            u_mags = np.abs(s_u[i, j, :])
            diff = np.abs(nu_mags - u_mags)
            print(
                f"  S[{i},{j}] |NU|={np.array2string(nu_mags, precision=4)}  "
                f"|U|={np.array2string(u_mags, precision=4)}  "
                f"diff={np.array2string(diff, precision=4)}"
            )

    # Both must be finite
    assert np.all(np.isfinite(s_u)), "Uniform path s_params contains NaN/Inf."
    assert np.all(np.isfinite(s_nu)), "NU path s_params contains NaN/Inf (rechecked)."

    max_dev = np.abs(s_nu - s_u).max()
    print(f"\n[G-NU] max |S_nu - S_uniform| = {max_dev:.4f}")
    # Post NU-DRIVE-FIX: max_dev ≈ 0 (both paths give the air-thru identity
    # |S21|≈1, |S11|≈0). Gate is loose (< 1.5) — it verifies both are finite
    # and assembly-comparable, not that either is physically correct.
    assert max_dev < 1.5, (
        f"NU vs uniform max deviation {max_dev:.4f} >= 1.5 — "
        "suggests assembly-level NaN/explosion (the all-zero-S regression "
        "would show max_dev≈1.0)."
    )


# ---------------------------------------------------------------------------
# Test 5: AD smoke — jax.grad through post-assembly scalar returns finite grad
# ---------------------------------------------------------------------------

def test_nu_assembly_ad_traceable():
    """jax.grad through a post-assembly scalar objective returns finite grad.

    The WI-2 acceptance required AD traceability for BOTH uniform and
    nonuniform paths.  This test verifies the _nu path specifically.

    Note: the upstream port builder has a known tape break at
    np.asarray(freqs_arr) in add_waveguide_port — that is a pre-existing
    issue out of scope of WI-2/G-NU.  We verify AD traceability of the
    ASSEMBLY OUTPUT (s_params is a jax.Array whose downstream ops are
    differentiable) by differentiating a post-assembly scale parameter.

    NOTE: post NU-DRIVE-FIX the assembled S is the live air-thru identity
    (|S21|≈1), so grad(sum|S*scale|^2)|_{scale=1} = 2*sum|S|^2 ≈ 20 — finite
    and non-zero (measured 2.0e+01 on 2026-07-20), confirming AD traceability
    of the assembly output. (The stale "all-zero bug => grad = 0" note was
    removed in issue #395.)
    """
    res = _get_result()
    s = res.s_params  # jax.Array

    def objective(scale: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(s * scale) ** 2).real

    grad = jax.grad(objective)(jnp.array(1.0))

    print(f"\n[G-NU] AD grad(sum|S*scale|^2)|_{{scale=1}} = {float(grad):.6e}")

    assert jnp.isfinite(grad), (
        f"jax.grad returned non-finite value {grad}. "
        "_compute_waveguide_s_matrix_nu assembly output is not AD-traceable — "
        "WI-2 acceptance condition violated for the NU path."
    )
    assert float(grad) >= 0.0, (
        f"AD grad = {float(grad):.6e} < 0; |sum(|S|^2)| gradient must be non-negative."
    )


# ---------------------------------------------------------------------------
# W1.1 (roadmap 2026-06-10): the NU dispatch must REJECT kwargs it cannot
# forward. eps_override/sigma_override are the documented differentiable
# channels (G-AD-WIRE-WG2); silently dropping them produced silently-wrong
# gradients on graded meshes.
# ---------------------------------------------------------------------------

def test_nu_dispatch_rejects_eps_override():
    sim = _make_wr90_nu_sim()
    with pytest.raises(NotImplementedError, match="eps_override"):
        sim.compute_waveguide_s_matrix(
            normalize=True, eps_override=jnp.ones((2, 2, 2)),
        )


def test_nu_dispatch_rejects_sigma_override():
    sim = _make_wr90_nu_sim()
    with pytest.raises(NotImplementedError, match="sigma_override"):
        sim.compute_waveguide_s_matrix(
            normalize=True, sigma_override=jnp.zeros((2, 2, 2)),
        )


def test_nu_dispatch_rejects_subpixel_smoothing():
    sim = _make_wr90_nu_sim()
    with pytest.raises(NotImplementedError, match="subpixel_smoothing"):
        sim.compute_waveguide_s_matrix(normalize=True, subpixel_smoothing=True)

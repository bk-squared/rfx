"""Fast pin of the E4 differentiable-stackup prototype
(``validation/research/multiband_nu/e4_diff_stackup.py``), pre-declared in
``docs/design_notes/20260907_nu_exp4_diff_stackup_predeclaration.md``
(note section 4). Written and committed BEFORE the instrument's first run;
tolerances are the note's numbers and are never widened here.

(a) E4-M map invariants, no FDTD: column length, four bitwise-equal thin
    cells, interface nodes, the declared Jacobian of dz w.r.t. h_thin, the
    dual-cell node eps formula.
(b) one LIVE gradient check of L1 at N_STEPS = 40 on the (6 mm, 3 mm)
    transverse box: finite gradient; h_thin and eps_thin resolved
    (quanta >= 50, asserted) and within 15 % of central FD with sign
    agreement; eps_core within 15 % with sign IF resolved, else its quanta
    are printed and it is not compared. Same FD steps as note 2.5.
Budget <= 30 s.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from validation.research.multiband_nu import e4_diff_stackup as e4

LIVE_N_STEPS = 40
LIVE_DOMAIN_XY = (6e-3, 3e-3)
LIVE_TOL = 0.15
LIVE_FLOOR_QUANTA = 50.0
FD_REL = {"h_thin": 1e-3, "eps_thin": 1e-2, "eps_core": 1e-2}


def test_map_invariants():
    """E4-M under the scoped x64 context: the 1e-9 m windows are float64-class
    statements (0.27 f32 ulp of the 44 mm column) — bring-up fix, see the
    note's Results. The map itself is dtype-generic."""
    from tests._x64_compat import enable_x64
    with enable_x64():
        jax.clear_caches()
        _map_invariants()
    jax.clear_caches()


def _map_invariants():
    jac = np.asarray(jax.jacfwd(lambda p: e4.stackup(p)[0])(jnp.asarray(e4.PARAMS0))[:, 0], np.float64)
    want = e4.declared_jacobian_h()
    assert np.allclose(jac, want, rtol=1e-6, atol=1e-9), "Jacobian of dz w.r.t. h_thin is not the declared one"
    assert abs(jac.sum()) < 1e-9, "the column length must be invariant in h_thin"
    for h in (1.8e-3, 2.0e-3, 2.2e-3):
        dz, eps = e4.stackup(jnp.asarray((h, 3.0, 4.3)))
        dz = np.asarray(dz, np.float64)
        eps = np.asarray(eps, np.float64)
        assert len(dz) == e4.N_CELLS and len(eps) == e4.N_CELLS + 1
        assert abs(dz.sum() - e4.L_Z) <= 1e-9, (h, dz.sum())
        thin = dz[e4.N_CORE:e4.N_CORE + e4.N_THIN]
        assert np.all(thin == thin[0]), "the four thin cells must be bitwise equal (the tie is by construction)"
        assert np.sum(dz == dz.min()) == e4.N_THIN, "only the thin cells attain the minimum"
        zn = np.concatenate([[0.0], np.cumsum(dz)])
        for k, z in zip(e4.K_IFACE, (15e-3 - h / 2, 15e-3 + h / 2, 30e-3)):
            assert abs(zn[k] - z) <= 1e-9, (k, zn[k], z)
        assert abs(eps[20] - (4.3 * dz[19] + 3.0 * dz[20]) / (dz[19] + dz[20])) <= 1e-6
        assert abs(eps[24] - (3.0 * dz[23] + 4.3 * dz[24]) / (dz[23] + dz[24])) <= 1e-6
        assert abs(eps[44] - (4.3 * dz[43] + 1.0 * dz[44]) / (dz[43] + dz[44])) <= 1e-6
        # interior nodes: (e d + e d) / (d + d) is a computed quotient, not a
        # literal — 1e-12, not bitwise (first-run test fix, no window involved)
        assert np.allclose(eps[21:24], 3.0, rtol=0, atol=1e-12)
        assert np.allclose(eps[45:], 1.0, rtol=0, atol=1e-12) and np.allclose(eps[:20], 4.3, rtol=0, atol=1e-12)


def test_live_l1_gradient_small_box():
    loss = e4.l1_factory(LIVE_N_STEPS, LIVE_DOMAIN_XY, e4.DXY)
    loss_j = jax.jit(loss)
    grad_j = jax.jit(jax.grad(loss))
    p0 = np.asarray(e4.PARAMS0, np.float64)
    g_ad = np.asarray(grad_j(jnp.asarray(e4.PARAMS0)), np.float64)
    assert g_ad.shape == (3,) and np.all(np.isfinite(g_ad)), g_ad
    l0 = float(loss_j(jnp.asarray(p0)))
    ulp = float(np.spacing(np.float32(abs(l0))))
    report = {}
    for i, name in enumerate(e4.PARAM_NAMES):
        step = FD_REL[name] * p0[i]
        pp, pm = p0.copy(), p0.copy()
        pp[i] += step
        pm[i] -= step
        lp, lm = float(loss_j(jnp.asarray(pp))), float(loss_j(jnp.asarray(pm)))
        g_fd = (lp - lm) / (2 * step)
        quanta = abs(lp - lm) / ulp
        rel = abs(g_ad[i] - g_fd) / max(abs(g_fd), 1e-300)
        report[name] = (g_ad[i], g_fd, rel, quanta)
        resolved = quanta >= LIVE_FLOOR_QUANTA
        if name in ("h_thin", "eps_thin"):
            assert resolved, f"{name}: FD reference unresolved ({quanta:.0f} quanta < {LIVE_FLOOR_QUANTA})"
        if resolved:
            assert np.sign(g_ad[i]) == np.sign(g_fd), f"{name}: sign disagreement {report[name]}"
            assert rel <= LIVE_TOL, f"{name}: AD {g_ad[i]:.4e} vs FD {g_fd:.4e} rel {rel:.3e} > {LIVE_TOL}"
        else:
            print(f"{name}: unresolved reference ({quanta:.0f} quanta), not compared; AD {g_ad[i]:.4e} FD {g_fd:.4e}")
    print(report)

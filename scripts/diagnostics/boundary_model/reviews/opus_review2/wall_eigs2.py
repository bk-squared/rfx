"""Reviewer check, second pass (clean monkeypatch, jitted).

wall_eigs.py left the image operator installed after its first image arm, so
its dx = 0.5 / 0.25 mm "free" rows and every "poststep_vacuum_cb" row ran on
the image operator. This script redoes those arms with the original operator
saved at import, and adds:
  * the (2,4) stencil with the image rule (ribbon reverts to 2nd order);
  * a graded axis: the NU E/H formulas with rfx's _profile_to_inv_arrays,
    image at both ends vs the half-cell zeroing.
Same method: eigenvalues of the one-step leapfrog Jacobian.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # scratch script, not a test module
import jax.numpy as jnp

import rfx.core.yee as Y
from rfx.nonuniform import _profile_to_inv_arrays

C0 = 299_792_458.0
PER = (False, True, True)
ORIG = Y._diff_bwd_o


def image_bwd(lo=True, hi=True):
    def f(arr, axis, periodic, order, bloch=None):
        out = ORIG(arr, axis, periodic, order, bloch)
        if axis == 0 and not periodic[0]:
            if lo:
                out = out.at[0].set((_C4 if False else 1.0) * 2.0 * arr[0]) if order == 2 else out
            if hi and order == 2:
                out = out.at[-1].set(-2.0 * arr[-2])
        return out
    return f


def image_bwd_any_order(lo=True, hi=True):
    """order 2 or 4: the ribbon (first/last two slices) is 2nd order, so the
    image replaces only the 2nd-order near difference at index 0 and N-1."""
    def f(arr, axis, periodic, order, bloch=None):
        out = ORIG(arr, axis, periodic, order, bloch)
        if axis == 0 and not periodic[0]:
            if lo:
                out = out.at[0].set(2.0 * arr[0])
            if hi:
                out = out.at[-1].set(-2.0 * arr[-2])
        return out
    return f


def eig_freqs(f, n, dt):
    J = np.asarray(jax.jit(jax.jacfwd(f))(jnp.zeros(2 * n)))
    lam = np.linalg.eigvals(J)
    w = np.angle(lam) / dt
    fs = np.sort(w[w > 1e3]) / (2 * np.pi)
    out = []
    for x in fs:
        if not out or abs(x - out[-1]) > 1e-9 * x:
            out.append(x)
    return np.array(out), float(np.max(np.abs(lam)))


def uniform_map(n, dx, dt, eps, *, bwd=None, e_kind="plain", post=None, so=2):
    mat = Y.MaterialArrays(eps_r=jnp.asarray(eps).reshape(n, 1, 1),
                           sigma=jnp.zeros((n, 1, 1)), mu_r=jnp.ones((n, 1, 1)))
    cb_vac = dt / (Y.EPS_0 * dx)

    def f(v):
        E, H = v[:n], v[n:]
        z = jnp.zeros((n, 1, 1))
        st = Y.FDTDState(ex=z, ey=z, ez=E.reshape(n, 1, 1), hx=z,
                         hy=H.reshape(n, 1, 1), hz=z, step=jnp.array(0))
        Y._diff_bwd_o = ORIG
        st = Y.update_h.__wrapped__(st, mat, dt, dx, PER, so, None)
        Y._diff_bwd_o = bwd or ORIG
        st = Y.update_e.__wrapped__(st, mat, dt, dx, PER, so, None)
        Y._diff_bwd_o = ORIG
        if post == "poststep_vacuum_cb_lo":
            st = st._replace(ez=st.ez.at[0].add(cb_vac * st.hy[0]))
        if post in ("pec_hi", "poststep_vacuum_cb_lo"):
            st = st._replace(ez=st.ez.at[-1].set(0.0))
        return jnp.concatenate([st.ez[:, 0, 0], st.hy[:, 0, 0]])
    return f


def k_from_f(f, dx, dt):
    return 2.0 / dx * np.arcsin(dx / (C0 * dt) * np.sin(np.pi * f * dt))


if __name__ == "__main__":
    L = 30e-3
    print("== free termination (no wall op), vacuum, L = 30 mm")
    for nc in (30, 60, 120):
        dx = L / nc; n = nc + 1; dt = 0.5 * dx / C0
        fs, rho = eig_freqs(uniform_map(n, dx, dt, np.ones(n)), n, dt)
        k1, k2 = k_from_f(fs[0], dx, dt), k_from_f(fs[1], dx, dt)
        Leff = np.pi / (k2 - k1)
        print(f"dx={dx*1e3:.3f} free: f1={fs[0]/1e9:.5f} GHz L_eff={Leff*1e3:.4f} mm = L{(Leff-L)/dx:+.3f} dx, "
              f"k1 L_eff/pi={k1*Leff/np.pi:.3f} (0.5 = one magnetic + one electric end) |lam|={rho:.12f}")

    print("\n== two-layer eps_r=4 on [0,10mm) at the magnetic face, PEC at 30 mm; lo side: R2-style post-step with vacuum cb")
    from scipy.optimize import brentq
    e1, a = 4.0, 10e-3
    g = lambda k0: np.sqrt(e1) * np.tan(np.sqrt(e1) * k0 * a) - 1.0 / np.tan(k0 * (L - a))
    ks = np.linspace(1.0, 400.0, 400000); vals = g(ks); roots = []
    for i in range(len(ks) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]) and abs(vals[i]) < 50 and abs(vals[i + 1]) < 50:
            roots.append(brentq(g, ks[i], ks[i + 1]))
    f_an = roots[0] * C0 / (2 * np.pi)
    for nc in (30, 60, 120, 240):
        dx = L / nc; n = nc + 1; dt = 0.5 * dx / C0
        x = np.arange(n) * dx
        eps = np.where(x < a - 1e-12, e1, 1.0); eps[np.isclose(x, a)] = 0.5 * (e1 + 1.0)
        fs_op, _ = eig_freqs(uniform_map(n, dx, dt, eps, bwd=image_bwd_any_order(True, False), post="pec_hi"), n, dt)
        fs_ps, rho = eig_freqs(uniform_map(n, dx, dt, eps, post="poststep_vacuum_cb_lo"), n, dt)
        print(f"dx={dx*1e3:.3f} operator f1 err={(fs_op[0]-f_an)/f_an*100:+.4f} % | post-step vacuum cb f1 err={(fs_ps[0]-f_an)/f_an*100:+.4f} % |lam|={rho:.9f}")

    print("\n== (2,4) stencil, vacuum, image at both ends (ribbon 2nd order) vs PEC; exact c/2L = %.6f GHz" % (C0/(2*L)/1e9))
    for nc in (30, 60, 120):
        dx = L / nc; n = nc + 1; dt = 0.3 * dx / C0
        fs_i, rho_i = eig_freqs(uniform_map(n, dx, dt, np.ones(n), bwd=image_bwd_any_order(), so=4), n, dt)
        # PEC reference with order 4: zero E at both ends
        mat_pec = None
        f_pec = uniform_map(n, dx, dt, np.ones(n), so=4, post="pec_hi")
        def f_pec2(v, f_pec=f_pec, n=n):
            out = f_pec(v)
            return out.at[0].set(0.0)
        fs_p, rho_p = eig_freqs(f_pec2, n, dt)
        fa = C0 / (2 * L)
        print(f"dx={dx*1e3:.3f} order4 image f1 err={(fs_i[0]-fa)/fa*100:+.5f} % f3 err={(fs_i[2]-3*fa)/(3*fa)*100:+.5f} % |lam|={rho_i:.12f}"
              f" | order4 PEC f1 err={(fs_p[0]-fa)/fa*100:+.5f} % f3 err={(fs_p[2]-3*fa)/(3*fa)*100:+.5f} %")

    print("\n== graded axis (NU formulas, rfx _profile_to_inv_arrays), vacuum PMC-PMC, declared L = 30 mm; exact f_m = m c / 2L")
    for scale in (1.0, 0.5, 0.25):
        # fine 0.4*h cells in the first 6 mm next to the lo face, coarse h beyond, h = 1 mm * scale
        h = 1e-3 * scale
        fine = [0.4 * h] * int(round(6e-3 / (0.4 * h)))
        coarse = [h] * int(round(24e-3 / h))
        cells = np.array(fine + coarse)
        Ldecl = cells.sum()
        full = np.concatenate([cells, cells[-1:]])     # NU _append_bounding_node duplicate
        n = len(full)
        ie, ih = (np.asarray(a, dtype=np.float64) for a in _profile_to_inv_arrays(full))
        dt = 0.5 * cells.min() / C0

        def nu_map(kind):
            def f(v):
                E, H = v[:n], v[n:]
                dE = jnp.concatenate([E[1:], jnp.zeros(1)]) - E
                Hn = H - dt / Y.MU_0 * (-(dE * ih))          # curl_y = -dEz/dx ; H -= dt/mu curl
                if kind == "half":
                    Hn = Hn.at[0].set(0.0).at[-2].set(0.0)
                prv = jnp.concatenate([jnp.zeros(1), Hn[:-1]])
                d = Hn - prv
                if kind == "image":
                    d = d.at[0].set(2.0 * Hn[0]).at[-1].set(-2.0 * Hn[-2])
                En = E + dt / Y.EPS_0 * d * ie
                return jnp.concatenate([En, Hn])
            return f
        fa = C0 / (2 * Ldecl)
        row = f"h={h*1e3:.3f} mm (fine 0.4h at the lo face) L={Ldecl*1e3:.4f}"
        for kind in ("image", "half"):
            fs, rho = eig_freqs(nu_map(kind), n, dt)
            f1 = fs[0]
            Leff = C0 / (2 * f1)
            row += f" | {kind}: f1 err={(f1-fa)/fa*100:+.5f} % L_eff={Leff*1e3:.4f} mm |lam|={rho:.12f}"
        print(row)

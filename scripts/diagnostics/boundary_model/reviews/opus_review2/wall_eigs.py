"""Reviewer check (design v2, D1/D5/root cause): which wall does each
termination realize?  1-D TEM line (Ez, Hy) along x built from rfx's own
uniform kernels (update_h / update_e, y and z periodic with one cell), on
origin/main.  The one-step leapfrog map is linear; its Jacobian's eigenvalues
give the realized resonances exactly (no time series, no fit).

Terminations:
  free    : no wall operation at all (zero-padded shifts only)
  pec     : E_t zeroed at node 0 and N-1 (the backing / PEC face)
  half    : H_t zeroed at Yee index 0 and N-2 (#1205 / apply_pmc_faces)
  image   : odd image of H_t in the backward difference (D1 recommendation):
            lo  d[0]   = H[0]   - (-H[0])   = 2 H[0]
            hi  d[N-1] = (-H[N-2]) - H[N-2] = -2 H[N-2]
  image_aniso : same monkeypatched _diff_bwd_o, but the E update goes through
            update_e_aniso (the subpixel-smoothing branch, own curl)
Declared cavity: nodes 0..N-1, L = (N-1) dx.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # scratch script, not a test module
import jax.numpy as jnp

import rfx.core.yee as Y

C0 = 299_792_458.0
PER = (False, True, True)


def make_ops(conv, n, so=2):
    orig = Y._diff_bwd_o

    def diff_bwd_image(arr, axis, periodic, order, bloch=None):
        out = orig(arr, axis, periodic, order, bloch)
        if axis == 0 and not periodic[0]:
            out = out.at[0].set(2.0 * arr[0])
            out = out.at[-1].set(-2.0 * arr[-2])
        return out
    return diff_bwd_image if conv.startswith("image") else orig


def step_map(conv, n, dx, dt, eps, so=2):
    mat = Y.MaterialArrays(eps_r=jnp.asarray(eps).reshape(n, 1, 1),
                           sigma=jnp.zeros((n, 1, 1)), mu_r=jnp.ones((n, 1, 1)))
    Y._diff_bwd_o = make_ops(conv, n, so)

    def f(v):
        E, H = v[:n], v[n:]
        z = jnp.zeros((n, 1, 1))
        st = Y.FDTDState(ex=z, ey=z, ez=E.reshape(n, 1, 1), hx=z,
                         hy=H.reshape(n, 1, 1), hz=z, step=jnp.array(0))
        st = Y.update_h.__wrapped__(st, mat, dt, dx, PER, so, None)
        if conv == "half":
            st = st._replace(hy=st.hy.at[0].set(0.0).at[-2].set(0.0))
        if conv == "image_aniso":
            e = mat.eps_r
            st = Y.update_e_aniso(st, mat, e, e, e, dt, dx, periodic=PER)
        else:
            st = Y.update_e.__wrapped__(st, mat, dt, dx, PER, so, None)
        if conv == "pec":
            st = st._replace(ez=st.ez.at[0].set(0.0).at[-1].set(0.0))
        return jnp.concatenate([st.ez[:, 0, 0], st.hy[:, 0, 0]])

    J = np.asarray(jax.jacfwd(f)(jnp.zeros(2 * n)))
    Y._diff_bwd_o = make_ops("orig", n)
    lam = np.linalg.eigvals(J)
    w = np.angle(lam) / dt
    f_hz = np.sort(w[w > 1e3]) / (2 * np.pi)
    return f_hz, np.max(np.abs(lam))


def k_from_f(f, dx, dt):
    return 2.0 / dx * np.arcsin(dx / (C0 * dt) * np.sin(np.pi * f * dt))


def uniq(fs, tol=1e-9):
    out = []
    for x in fs:
        if not out or abs(x - out[-1]) > tol * x:
            out.append(x)
    return out


if __name__ == "__main__":
    L = 30e-3
    print("== vacuum line, L = 30 mm declared (nodes 0..N-1)")
    for nc in (30, 60, 120):
        dx = L / nc
        n = nc + 1
        dt = 0.5 * dx / C0
        eps = np.ones(n)
        for conv in ("free", "pec", "half", "image", "image_aniso"):
            fs, rho = step_map(conv, n, dx, dt, eps)
            fs = uniq(fs)
            k1 = k_from_f(fs[0], dx, dt)
            k2 = k_from_f(fs[1], dx, dt)
            # spacing of consecutive modes = pi / L_eff regardless of mode family
            Leff = np.pi / (k2 - k1)
            # quarter-wave family if k1 * Leff / pi ~ 0.5
            fam = k1 * Leff / np.pi
            print(f"dx={dx*1e3:.3f} mm {conv:12s} f1={fs[0]/1e9:9.5f} f2={fs[1]/1e9:9.5f} GHz "
                  f"L_eff={Leff*1e3:8.4f} mm (L+{(Leff-L)/dx:+.3f} dx)  k1*L_eff/pi={fam:.3f}  |lam|max={rho:.12f}")

    print("\n== two-layer: eps_r=4 on [0, 10 mm) touching the magnetic face, vacuum to a PEC at 30 mm")
    # analytic: sqrt(e1) tan(sqrt(e1) k0 a) = cot(k0 (L-a))
    from scipy.optimize import brentq
    e1, a = 4.0, 10e-3
    g = lambda k0: np.sqrt(e1) * np.tan(np.sqrt(e1) * k0 * a) - 1.0 / np.tan(k0 * (L - a))
    ks = np.linspace(1.0, 400.0, 400000)
    vals = g(ks)
    roots = []
    for i in range(len(ks) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]) and abs(vals[i]) < 50 and abs(vals[i + 1]) < 50:
            roots.append(brentq(g, ks[i], ks[i + 1]))
    f_an = np.array(roots[:3]) * C0 / (2 * np.pi)
    print("analytic f (GHz):", np.round(f_an / 1e9, 6))
    for nc in (30, 60, 120, 240):
        dx = L / nc
        n = nc + 1
        dt = 0.5 * dx / C0
        x = np.arange(n) * dx
        eps = np.where(x < a - 1e-12, e1, 1.0)
        eps[np.isclose(x, a)] = 0.5 * (e1 + 1.0)
        rows = []
        for conv in ("image", "image_aniso"):
            # PEC at hi: zero E at N-1 after update; image at lo only
            fs, _ = step_map_two(conv, n, dx, dt, eps) if False else (None, None)
        # inline variant: image at lo, PEC at hi
        Y_orig = Y._diff_bwd_o

        def lo_image(arr, axis, periodic, order, bloch=None):
            out = Y_orig(arr, axis, periodic, order, bloch)
            if axis == 0 and not periodic[0]:
                out = out.at[0].set(2.0 * arr[0])
            return out

        mat = Y.MaterialArrays(eps_r=jnp.asarray(eps).reshape(n, 1, 1),
                               sigma=jnp.zeros((n, 1, 1)), mu_r=jnp.ones((n, 1, 1)))
        cb_vac = dt / (Y.EPS_0 * dx)

        def f_factory(kind):
            def f(v):
                E, H = v[:n], v[n:]
                z = jnp.zeros((n, 1, 1))
                st = Y.FDTDState(ex=z, ey=z, ez=E.reshape(n, 1, 1), hx=z,
                                 hy=H.reshape(n, 1, 1), hz=z, step=jnp.array(0))
                st = Y.update_h.__wrapped__(st, mat, dt, dx, PER, 2, None)
                if kind == "operator":
                    Y._diff_bwd_o = lo_image
                    st = Y.update_e.__wrapped__(st, mat, dt, dx, PER, 2, None)
                    Y._diff_bwd_o = Y_orig
                elif kind == "poststep_vacuum_cb":   # the R2 prototype's lo side
                    st = Y.update_e.__wrapped__(st, mat, dt, dx, PER, 2, None)
                    st = st._replace(ez=st.ez.at[0].add(cb_vac * st.hy[0]))
                elif kind == "half":
                    st = st._replace(hy=st.hy.at[0].set(0.0))
                    st = Y.update_e.__wrapped__(st, mat, dt, dx, PER, 2, None)
                st = st._replace(ez=st.ez.at[-1].set(0.0))
                return jnp.concatenate([st.ez[:, 0, 0], st.hy[:, 0, 0]])
            return f

        line = f"dx={dx*1e3:6.3f} mm"
        for kind in ("operator", "poststep_vacuum_cb", "half"):
            J = np.asarray(jax.jacfwd(f_factory(kind))(jnp.zeros(2 * n)))
            lam = np.linalg.eigvals(J)
            w = np.angle(lam) / dt
            fs = uniq(np.sort(w[w > 1e3]) / (2 * np.pi))
            err = (fs[0] - f_an[0]) / f_an[0]
            line += f" | {kind}: f1={fs[0]/1e9:.6f} GHz err={err*100:+.4f} % |lam|max={np.max(np.abs(lam)):.9f}"
        print(line)

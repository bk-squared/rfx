"""Clean, logged rerun of R1 (all five terminations) for a witness file.
Same method as wall_eigs2.py (original _diff_bwd_o saved at import, jitted
Jacobian). Declared L = 30 mm; 1-D TEM line (Ez, Hy) on rfx's update_h/update_e."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # scratch script, not a test module
import jax.numpy as jnp
import rfx.core.yee as Y
from wall_eigs2 import ORIG, image_bwd_any_order, eig_freqs, k_from_f, C0, PER

def term_map(conv, n, dx, dt):
    mat = Y.MaterialArrays(eps_r=jnp.ones((n, 1, 1)), sigma=jnp.zeros((n, 1, 1)), mu_r=jnp.ones((n, 1, 1)))
    img = image_bwd_any_order()
    def f(v):
        E, H = v[:n], v[n:]
        z = jnp.zeros((n, 1, 1))
        st = Y.FDTDState(ex=z, ey=z, ez=E.reshape(n, 1, 1), hx=z, hy=H.reshape(n, 1, 1), hz=z, step=jnp.array(0))
        Y._diff_bwd_o = ORIG
        st = Y.update_h.__wrapped__(st, mat, dt, dx, PER, 2, None)
        if conv == "half":
            st = st._replace(hy=st.hy.at[0].set(0.0).at[-2].set(0.0))
        Y._diff_bwd_o = img if conv.startswith("image") else ORIG
        if conv == "image_aniso":
            e = mat.eps_r
            st = Y.update_e_aniso(st, mat, e, e, e, dt, dx, periodic=PER)
        else:
            st = Y.update_e.__wrapped__(st, mat, dt, dx, PER, 2, None)
        Y._diff_bwd_o = ORIG
        if conv == "pec":
            st = st._replace(ez=st.ez.at[0].set(0.0).at[-1].set(0.0))
        return jnp.concatenate([st.ez[:, 0, 0], st.hy[:, 0, 0]])
    return f

L = 30e-3
for nc in (30, 60, 120):
    dx = L / nc; n = nc + 1; dt = 0.5 * dx / C0
    for conv in ("free", "pec", "half", "image", "image_aniso"):
        fs, rho = eig_freqs(term_map(conv, n, dx, dt), n, dt)
        k1, k2 = k_from_f(fs[0], dx, dt), k_from_f(fs[1], dx, dt)
        Leff = np.pi / (k2 - k1)
        print(f"dx={dx*1e3:.3f} mm {conv:12s} f1={fs[0]/1e9:9.5f} f2={fs[1]/1e9:9.5f} GHz "
              f"L_eff={Leff*1e3:8.4f} mm (L{(Leff-L)/dx:+.3f} dx) k1*L_eff/pi={k1*Leff/np.pi:.3f} |lam|max={rho:.12f}", flush=True)

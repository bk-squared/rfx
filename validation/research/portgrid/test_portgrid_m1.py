"""M1 fast falsifier battery: r=1 equivalence, short energy audit,
F-M1-grad (jax.grad vs central FD across the interface) and F-M1-vjp
(P-adjoint structure of the reverse pass).

The >=1e6-step F-M1a audit and the F-M1b reflection sweep are the
measurement scripts ``m1_energy_audit.py`` / ``m1_reflection.py`` in this
directory (runtimes beyond unit-test budget).

Run:
  PYTHONPATH=<worktree> .venv/bin/python -m pytest \
      validation/research/portgrid/test_portgrid_m1.py -o addopts="" -q
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._x64_compat import enable_x64


def _paper_spec(r: int, sim2d):
    """Paper Sec. V-A / Fig. 4 fixture: 60 x 40 mm PEC cavity, dx=1mm, dy=2mm,
    centered 40 x 20 mm fine region."""
    nx, ny = 60, 20
    i0, i1 = 10, 50   # x in [10, 50] mm
    j0, j1 = 5, 15    # y in [10, 30] mm (dy = 2 mm)
    spec = sim2d.TwoRegionSpec(
        nx=nx, ny=ny, dx=1e-3, dy=2e-3, i0=i0, i1=i1, j0=j0, j1=j1, r=r,
        dt=np.nan, src_ij=(3, 2), probe_ij=(55, 17),
    )
    spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
    return spec


def test_r1_island_reduces_to_uniform_grid():
    """r=1 'island' must reproduce the plain Yee grid exactly (update (61)
    reduces to the standard equation for r=1, paper Sec. IV-C)."""
    with enable_x64():
        import jax
        from validation.research.portgrid import sim2d

        spec = _paper_spec(1, sim2d)
        n_steps = 3000
        wf = sim2d.gaussian_modulated(n_steps, spec.dt, 3.75e9, 0.74e9)
        step, init, _ = sim2d.make_stepper(spec)
        _, e_sub, p_sub = jax.jit(lambda s, w: sim2d.run_scan(step, s, w))(init(), wf)

        sm = np.zeros((spec.nx, spec.ny))
        sm[spec.src_ij] = 1.0
        ustep, uinit = sim2d.make_uniform_stepper(
            spec.nx, spec.ny, spec.dx, spec.dy, spec.dt, sm, spec.probe_ij)
        _, e_uni, p_uni = jax.jit(lambda s, w: sim2d.run_scan(ustep, s, w))(uinit(), wf)

        p_sub, p_uni = np.asarray(p_sub), np.asarray(p_uni)
        scale = np.max(np.abs(p_uni))
        assert np.max(np.abs(p_sub - p_uni)) <= 1e-12 * scale
        e_sub, e_uni = np.asarray(e_sub), np.asarray(e_uni)
        assert np.max(np.abs(e_sub - e_uni)) <= 1e-12 * np.max(e_uni)


@pytest.mark.parametrize("r", (3, 4, 5))
def test_short_energy_non_growth(r):
    """20k-step preview of F-M1a (same window class, 50x shorter run)."""
    with enable_x64():
        import jax
        from validation.research.portgrid import sim2d

        spec = _paper_spec(r, sim2d)
        n_steps = 20_000
        wf = sim2d.gaussian_modulated(n_steps, spec.dt, 3.75e9, 0.74e9)
        n_off = int(np.max(np.nonzero(wf)[0])) + 1
        step, init, _ = sim2d.make_stepper(spec)
        _, energies, probe = jax.jit(lambda s, w: sim2d.run_scan(step, s, w))(init(), wf)
        energies = np.asarray(energies)
        probe = np.asarray(probe)
        assert np.all(np.isfinite(energies)) and np.all(np.isfinite(probe))
        e_ref = energies[n_off + 1]
        assert e_ref > 0.0
        drift = (energies[n_off + 1:] - e_ref) / e_ref
        assert np.max(drift) <= 1e-8, (r, float(np.max(drift)))
        # conservation is exact in exact arithmetic -> also check the tighter
        # roundoff class actually achieved (recorded, not a falsifier)
        assert np.max(np.abs(drift)) <= 1e-8, (r, float(np.max(np.abs(drift))))


def _grad_fixture(sim2d, theta):
    """Interface-crossing objective: theta = eps_r of a fine-island block whose
    south face lies ON the island's south interface row."""
    import jax.numpy as jnp

    nx, ny = 30, 16
    spec = sim2d.TwoRegionSpec(
        nx=nx, ny=ny, dx=1e-3, dy=1e-3, i0=12, i1=18, j0=5, j1=11, r=3,
        dt=np.nan, src_ij=(4, 8), probe_ij=(26, 8),
    )
    spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
    nfx, nfy = spec.nfx, spec.nfy  # 18 x 18

    # block: fine x-index [3, 15), fine y-index [0, 9) -- includes j=0 row
    ex_mask = np.zeros((nfx, nfy + 1))
    ex_mask[3:15, 0:9] = 1.0
    ey_mask = np.zeros((nfx + 1, nfy))
    ey_mask[3:16, 0:9] = 1.0
    eps_fx = sim2d.EPS0 * (1.0 + (theta - 1.0) * jnp.asarray(ex_mask))
    eps_fy = sim2d.EPS0 * (1.0 + (theta - 1.0) * jnp.asarray(ey_mask))
    return spec, eps_fx, eps_fy


def test_grad_ad_vs_central_fd():
    """F-M1-grad: min over h-sweep of relative AD/FD mismatch <= 1e-6."""
    with enable_x64():
        import jax
        import jax.numpy as jnp
        from validation.research.portgrid import sim2d

        n_steps = 2000

        def loss(theta):
            spec, eps_fx, eps_fy = _grad_fixture(sim2d, theta)
            wf = jnp.asarray(sim2d.gaussian_modulated(n_steps, spec.dt, 10e9, 4e9))
            step, init, _ = sim2d.make_stepper(spec, eps_fx=eps_fx, eps_fy=eps_fy)
            _, _, probe = sim2d.run_scan(step, init(), wf)
            return jnp.sum(probe**2)

        loss_j = jax.jit(loss)
        theta0 = 2.0
        g_ad = float(jax.jit(jax.grad(loss))(theta0))
        assert np.isfinite(g_ad) and g_ad != 0.0

        rels = []
        for h in (1e-4 * theta0, 1e-5 * theta0, 1e-6 * theta0):
            g_fd = (float(loss_j(theta0 + h)) - float(loss_j(theta0 - h))) / (2.0 * h)
            rels.append(abs(g_ad - g_fd) / abs(g_fd))
        assert min(rels) <= 1e-6, (g_ad, rels)


def test_vjp_p_adjoint_structure_of_step():
    """F-M1-vjp: on a tiny fixture the reverse-mode Jacobian of one full step
    equals the forward-mode one, and the interface blocks carry the
    averaging/replication P-adjoint structure with the analytic cb/r weights."""
    with enable_x64():
        import jax
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree
        from validation.research.portgrid import sim2d

        spec = sim2d.TwoRegionSpec(
            nx=6, ny=6, dx=1e-3, dy=1e-3, i0=2, i1=4, j0=2, j1=4, r=3,
            dt=np.nan, src_ij=(0, 0), probe_ij=(5, 5),
        )
        spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
        step, init, aux = sim2d.make_stepper(spec)

        s0 = init()
        v0, unravel = ravel_pytree(s0)

        def f(v):
            new_state, _ = step(unravel(v), 0.0)
            out, _ = ravel_pytree(new_state)
            return out

        jf = jax.jacfwd(f)(v0)          # JVP-built Jacobian
        jr = jax.jacrev(f)(v0)          # VJP-built Jacobian (transpose path)
        scale = float(np.max(np.abs(np.asarray(jf))))
        assert np.max(np.abs(np.asarray(jf) - np.asarray(jr))) <= 1e-12 * scale

        # --- structural audit of the interface exchange block ---
        # d(coarse south-interface Ex)/d(fine boundary-row Hz) must be the
        # segment-MEAN scaled by cb (eq. (61)); its transpose in the reverse
        # Jacobian is the replication pattern scaled by cb / r.
        # Locate flat indices via one-hot probes.
        r = spec.r
        ca_s, cb_s = aux["ifc"]["s"]
        cb_s = np.asarray(cb_s)

        # forward: perturb fhz[k, 0], watch ex[i0 + k//r, j0]
        for k in range(spec.nfx):
            s_pert = jax.tree_util.tree_map(jnp.zeros_like, s0)
            s_pert["fhz"] = s_pert["fhz"].at[k, 0].set(1.0)
            vp, _ = ravel_pytree(s_pert)
            out = np.asarray(f(v0 + vp) - f(v0))
            out_state = unravel(jnp.asarray(out))
            got = float(out_state["ex"][spec.i0 + k // r, spec.j0])
            # H update leaves fhz[k,0] unchanged (it is state at n-1/2 -> the
            # new fhz includes the old value), so the coefficient seen by the
            # interface update is cb * (1/r).
            want = float(cb_s[k // r]) / r
            assert abs(got - want) <= 1e-12 * abs(want), (k, got, want)

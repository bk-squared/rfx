"""Wiring falsifiers for the F-M1b retry additions (pre-declaration §4).

Covers the eq. (58)/(61) material path (sigma-hat + non-vacuum coarse host),
the PML-terminated steppers, and the rod sigma-map generator.

Run:
  PYTHONPATH=<worktree> .venv/bin/python -m pytest \
      validation/research/portgrid/test_portgrid_m1b_retry.py -o addopts="" -q
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from tests._x64_compat import enable_x64


def _uniform_lossy_pec_stepper(sim2d, nx, ny, dx, dy, dt, eps_x, eps_y,
                               sigma_x, sigma_y, src_ij, probe_ij):
    """Reference: plain uniform lossy Yee grid, PEC box, per-edge eps/sigma,
    magnetic-current point source (mirrors the two-region PEC stepper)."""
    import jax.numpy as jnp

    ca_x, cb_x = sim2d._lossy_e_coeffs(jnp.asarray(eps_x), jnp.asarray(sigma_x), dt)
    ca_y, cb_y = sim2d._lossy_e_coeffs(jnp.asarray(eps_y), jnp.asarray(sigma_y), dt)
    ch = dt / sim2d.MU0
    sm = np.zeros((nx, ny))
    sm[src_ij] = 1.0
    src = jnp.asarray(sm)
    pi, pj = probe_ij

    def step(state, src_val):
        ex, ey, hz = state["ex"], state["ey"], state["hz"]
        curl_e = (ex[:, 1:] - ex[:, :-1]) / dy - (ey[1:, :] - ey[:-1, :]) / dx
        hz_new = hz + ch * curl_e + src_val * src
        ex_new = ex.at[:, 1:-1].set(
            ca_x[:, 1:-1] * ex[:, 1:-1]
            + cb_x[:, 1:-1] * (hz_new[:, 1:] - hz_new[:, :-1]) / dy)
        ey_new = ey.at[1:-1, :].set(
            ca_y[1:-1, :] * ey[1:-1, :]
            + cb_y[1:-1, :] * (hz_new[:-1, :] - hz_new[1:, :]) / dx)
        return dict(ex=ex_new, ey=ey_new, hz=hz_new), hz_new[pi, pj]

    def init_state():
        z = partial(jnp.zeros, dtype=jnp.float64)
        return dict(ex=z((nx, ny + 1)), ey=z((nx + 1, ny)), hz=z((nx, ny)))

    return step, init_state


def _material_maps(sim2d, nx, ny, r, island, eps_r, sigma, box):
    """Coarse-host + fine maps for one lossy dielectric box (SI absolute
    coords, box = (x0, x1, y0, y1) in meters), on a dx=dy=d grid."""
    i0, i1, j0, j1 = island

    def maps(n1, n2, d, ox, oy, kind):
        if kind == "x":  # Ex edges at (ox+(i+1/2)d, oy+j*d)
            xs = ox + (np.arange(n1) + 0.5) * d
            ys = oy + np.arange(n2 + 1) * d
            shape = (n1, n2 + 1)
        else:            # Ey edges at (ox+i*d, oy+(j+1/2)d)
            xs = ox + np.arange(n1 + 1) * d
            ys = oy + (np.arange(n2) + 0.5) * d
            shape = (n1 + 1, n2)
        inside = ((xs[:, None] >= box[0]) & (xs[:, None] < box[1])
                  & (ys[None, :] >= box[2]) & (ys[None, :] < box[3]))
        eps = np.full(shape, sim2d.EPS0)
        eps[inside] = eps_r * sim2d.EPS0
        sig = np.zeros(shape)
        sig[inside] = sigma
        return eps, sig

    d = 1e-3
    eps_cx, sig_cx = maps(nx, ny, d, 0.0, 0.0, "x")
    eps_cy, sig_cy = maps(nx, ny, d, 0.0, 0.0, "y")
    df = d / r
    nfx, nfy = (i1 - i0) * r, (j1 - j0) * r
    eps_fx, sig_fx = maps(nfx, nfy, df, i0 * d, j0 * d, "x")
    eps_fy, sig_fy = maps(nfx, nfy, df, i0 * d, j0 * d, "y")
    return dict(eps_cx=eps_cx, sigma_cx=sig_cx, eps_cy=eps_cy, sigma_cy=sig_cy,
                eps_fx=eps_fx, sigma_fx=sig_fx, eps_fy=eps_fy, sigma_fy=sig_fy)


def test_vacuum_maps_reproduce_default_stepper_exactly():
    """Explicit vacuum/lossless maps must give the SAME trajectory as the
    default (map-free) stepper — the material wiring is exact, not approx."""
    with enable_x64():
        import jax
        from validation.research.portgrid import sim2d

        spec = sim2d.TwoRegionSpec(nx=20, ny=14, dx=1e-3, dy=1e-3,
                                   i0=8, i1=12, j0=5, j1=9, r=3,
                                   dt=np.nan, src_ij=(3, 7), probe_ij=(16, 7))
        spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
        wf = sim2d.gaussian_modulated(600, spec.dt, 3.75e9, 0.74e9)

        step0, init0, _ = sim2d.make_stepper(spec)
        _, (e0, p0) = jax.lax.scan(step0, init0(), wf)

        nfx, nfy = spec.nfx, spec.nfy
        step1, init1, _ = sim2d.make_stepper(
            spec,
            sigma_fx=np.zeros((nfx, nfy + 1)), sigma_fy=np.zeros((nfx + 1, nfy)),
            eps_cx=np.full((20, 15), sim2d.EPS0),
            eps_cy=np.full((21, 14), sim2d.EPS0),
            sigma_cx=np.zeros((20, 15)), sigma_cy=np.zeros((21, 14)))
        _, (e1, p1) = jax.lax.scan(step1, init1(), wf)

        p0, p1 = np.asarray(p0), np.asarray(p1)
        e0, e1 = np.asarray(e0), np.asarray(e1)
        assert np.max(np.abs(p1 - p0)) <= 1e-14 * max(np.max(np.abs(p0)), 1e-300)
        assert np.max(np.abs(e1 - e0)) <= 1e-14 * max(np.max(e0), 1e-300)


def test_r1_lossy_island_reduces_to_uniform_lossy_yee():
    """r = 1 with a lossy dielectric spanning the interface: eq. (61) must
    reduce algebraically to the standard lossy Yee edge coefficient, so the
    two-region stepper must reproduce a uniform lossy grid to roundoff."""
    with enable_x64():
        import jax
        from validation.research.portgrid import sim2d

        nx, ny = 24, 16
        island = (9, 15, 6, 12)
        spec = sim2d.TwoRegionSpec(nx=nx, ny=ny, dx=1e-3, dy=1e-3,
                                   i0=island[0], i1=island[1], j0=island[2],
                                   j1=island[3], r=1, dt=np.nan,
                                   src_ij=(3, 8), probe_ij=(20, 8))
        spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
        # lossy box CROSSING the south and west interfaces
        box = (7e-3, 12e-3, 4e-3, 9e-3)
        mm = _material_maps(sim2d, nx, ny, 1, island, eps_r=2.0, sigma=5.0,
                            box=box)
        wf = sim2d.gaussian_modulated(900, spec.dt, 3.75e9, 0.74e9)

        step, init, _ = sim2d.make_stepper(
            spec, eps_fx=mm["eps_fx"], eps_fy=mm["eps_fy"],
            sigma_fx=mm["sigma_fx"], sigma_fy=mm["sigma_fy"],
            eps_cx=mm["eps_cx"], eps_cy=mm["eps_cy"],
            sigma_cx=mm["sigma_cx"], sigma_cy=mm["sigma_cy"])
        _, (_, p_sub) = jax.lax.scan(step, init(), wf)

        ustep, uinit = _uniform_lossy_pec_stepper(
            sim2d, nx, ny, 1e-3, 1e-3, spec.dt,
            mm["eps_cx"], mm["eps_cy"], mm["sigma_cx"], mm["sigma_cy"],
            spec.src_ij, spec.probe_ij)
        _, p_uni = jax.lax.scan(ustep, uinit(), wf)

        p_sub, p_uni = np.asarray(p_sub), np.asarray(p_uni)
        scale = np.max(np.abs(p_uni))
        assert np.max(np.abs(p_sub - p_uni)) <= 1e-12 * scale


@pytest.mark.parametrize("r", (2, 3))
def test_lossy_traverse_energy_monotone_nonincreasing(r):
    """Sec. V-B class: a lossy dielectric slab traversing the interface (the
    sigma-hat terms of (61) active).  After source-off the staggered storage
    (25) must be non-increasing within roundoff (dissipativity)."""
    with enable_x64():
        import jax
        from validation.research.portgrid import sim2d

        nx, ny = 30, 20
        island = (11, 19, 7, 13)
        spec = sim2d.TwoRegionSpec(nx=nx, ny=ny, dx=1e-3, dy=1e-3,
                                   i0=island[0], i1=island[1], j0=island[2],
                                   j1=island[3], r=r, dt=np.nan,
                                   src_ij=(4, 10), probe_ij=(26, 10))
        spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
        # slab straddles the south interface row (y in [5,9) mm crosses j0=7)
        box = (12e-3, 18e-3, 5e-3, 9e-3)
        mm = _material_maps(sim2d, nx, ny, r, island, eps_r=2.0, sigma=5.0,
                            box=box)
        n_steps = 9000
        wf = sim2d.gaussian_modulated(n_steps, spec.dt, 3.75e9, 0.74e9)
        n_off = int(np.max(np.nonzero(wf)[0])) + 1
        assert n_off < n_steps - 2000

        step, init, _ = sim2d.make_stepper(
            spec, eps_fx=mm["eps_fx"], eps_fy=mm["eps_fy"],
            sigma_fx=mm["sigma_fx"], sigma_fy=mm["sigma_fy"],
            eps_cx=mm["eps_cx"], eps_cy=mm["eps_cy"],
            sigma_cx=mm["sigma_cx"], sigma_cy=mm["sigma_cy"])
        _, (energies, _) = jax.lax.scan(step, init(), wf)
        energies = np.asarray(energies)
        assert np.all(np.isfinite(energies))
        e = energies[n_off + 1:]
        e_ref = e[0]
        assert e_ref > 0.0
        # monotone non-increasing within roundoff slack
        assert np.max(np.diff(e)) <= 1e-13 * e_ref
        # and the slab actually dissipates (not a vacuous check)
        assert e[-1] < 0.99 * e_ref


def test_pml_chain_null_r1():
    """r = 1 island through the full retry chain (PML termination, Jy column
    source, column-mean Ey probe) must match the uniform-PML reference to
    roundoff — the retry measurement pipeline contributes nothing."""
    with enable_x64():
        import jax
        from validation.research.portgrid import sim2d

        nx, ny = 66, 40
        spec = sim2d.TwoRegionSpec(nx=nx, ny=ny, dx=1e-3, dy=1e-3,
                                   i0=39, i1=47, j0=16, j1=24, r=1, dt=np.nan)
        spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
        wf = sim2d.gaussian_modulated(1400, spec.dt, 16e9, 10e9)

        step, init, _ = sim2d.make_stepper_pml(spec, src_col=17, probe_col=19)
        _, p_sub = jax.lax.scan(step, init(), wf)
        ustep, uinit, _ = sim2d.make_uniform_pml(
            nx, ny, 1e-3, 1e-3, spec.dt, src_col=17, probe_col=19)
        _, p_uni = jax.lax.scan(ustep, uinit(), wf)

        p_sub, p_uni = np.asarray(p_sub), np.asarray(p_uni)
        scale = np.max(np.abs(p_uni))
        assert scale > 0.0
        assert np.max(np.abs(p_sub - p_uni)) <= 1e-12 * scale


def test_pml_actually_absorbs():
    """Sanity: the guide field decays after source-off (the PML is not inert)."""
    with enable_x64():
        import jax
        import jax.numpy as jnp
        from validation.research.portgrid import sim2d

        nx, ny = 66, 40
        dt = 0.99 * np.sqrt(sim2d.EPS0 * sim2d.MU0) / np.sqrt(2.0) * 1e-3
        n_steps = 3000
        wf = sim2d.gaussian_modulated(n_steps, dt, 16e9, 10e9)
        step, init, _ = sim2d.make_uniform_pml(
            nx, ny, 1e-3, 1e-3, dt, src_col=17, probe_col=19)

        def step_with_energy(state, src_val):
            new, probe = step(state, src_val)
            e = sum(jnp.sum(v * v) for v in new.values())
            return new, (probe, e)

        _, (_, e) = jax.lax.scan(step_with_energy, init(), wf)
        e = np.asarray(e)
        assert np.all(np.isfinite(e))
        assert e[-1] < 1e-6 * np.max(e)  # >= 60 dB field-energy decay


def test_disk_sigma_maps_geometry():
    from validation.research.portgrid import sim2d

    r = 6
    d = 1e-3 / r
    nfx = nfy = 8 * r
    centers = [(41e-3, 18e-3), (41e-3, 22e-3), (45e-3, 18e-3), (45e-3, 22e-3)]
    sx, sy = sim2d.disk_sigma_maps(nfx, nfy, d, d, (39e-3, 16e-3),
                                   centers, 1e-3, 5.8e7)
    assert sx.shape == (nfx, nfy + 1) and sy.shape == (nfx + 1, nfy)
    assert set(np.unique(sx)) <= {0.0, 5.8e7}
    # each rod pi*r^2 vs counted edge area (within staircase tolerance)
    area = np.pi * 1e-3**2 * len(centers)
    for m in (sx, sy):
        frac = np.count_nonzero(m) * d * d / area
        assert 0.9 <= frac <= 1.1, frac
    # four-rod layout is symmetric about y = 20 mm -> maps mirror in y
    assert np.array_equal(sx, sx[:, ::-1])
    assert np.array_equal(sy, sy[:, ::-1])
    # nothing outside the rod bounding box [40,46] x [17,23] mm
    xs = 39e-3 + (np.arange(nfx) + 0.5) * d
    outside = (xs < 40e-3 - 1e-12) | (xs > 46e-3 + 1e-12)
    assert np.all(sx[outside, :] == 0.0)


def test_pml_floor_class_gate():
    """Battery-level gate on the PML termination's reflection FLOOR.

    ``test_pml_actually_absorbs`` only demands 60 dB of energy decay, which a
    badly mismatched PML still satisfies: the reviewer's mutation (halving the
    matched magnetic conductivity sigma* wired at sim2d.py pml_profiles)
    degrades the measured floor from -94 dB to -15.8 dB while leaving that test
    green.  This gate locks the CLASS instead -- the same echo/direct ratio the
    F-M1b-abc floor arm measures, at dt(r=2), against a deliberately loose
    -40 dB window.  It is not the frozen -50 dB falsifier (that verdict lives in
    m1b_retry.py --arm floor + its committed JSON); it exists so a termination
    regression cannot reach M2 unnoticed.

    Gates are the Correction R1 rays; per Correction R2 the direct gate
    truncates the direct pulse near its peak, which biases the measured floor
    UPWARD (pessimistic) -- harmless for a class gate.
    """
    with enable_x64():
        import jax

        from validation.research.portgrid import m1b_retry, sim2d

        dt, _ = m1b_retry._dt_for(sim2d, 2)
        nx, ny, src, prb = 400, 40, 100, 150
        n_steps = int(np.ceil(2.2e-9 / dt))
        wf = sim2d.gaussian_modulated(n_steps, dt, m1b_retry.F0, m1b_retry.HWHM)
        step, init, _ = sim2d.make_uniform_pml(
            nx, ny, m1b_retry.DX, m1b_retry.DY, dt,
            src_col=src, probe_col=prb, npml=m1b_retry.NPML)
        p = np.asarray(jax.jit(
            lambda s, w: jax.lax.scan(step, s, w))(init(), wf)[1])

        t = np.arange(n_steps) * dt
        direct = np.where(t <= 0.30e-9, p, 0.0)
        echo = np.where((t >= 0.55e-9) & (t <= 1.15e-9), p, 0.0)
        assert np.max(np.abs(direct)) > 0.0, "direct gate captured nothing"

        nfft = 1 << int(np.ceil(np.log2(4 * n_steps)))
        f = np.fft.rfftfreq(nfft, dt)
        rr = np.abs(np.fft.rfft(echo, nfft)) / np.maximum(
            np.abs(np.fft.rfft(direct, nfft)), 1e-300)
        floor_db = m1b_retry._band_max_db(f, rr, 2e9, 30e9)
        assert floor_db <= -40.0, (
            f"PML reflection floor {floor_db:.1f} dB exceeds the -40 dB class "
            "gate -- the termination regressed (check the matched sigma* "
            "wiring in sim2d.pml_profiles)"
        )

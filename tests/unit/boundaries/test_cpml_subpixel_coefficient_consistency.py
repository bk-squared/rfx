"""CPML's psi coefficient must use the permittivity the E update used (#1043).

The E half-step in a CPML cell is assembled from two halves::

    E^{n+1} = E^n + (dt/(eps_a*EPS_0)) * curl      <- update_e / update_e_aniso
                  + (dt/(eps_b*EPS_0)) * psi       <- apply_cpml_e

``kappa_max`` defaults to 1.0, so the kappa term is exactly zero and
``ce * psi`` is the whole CPML-E correction. With ``psi = b*psi_prev + c*curl``
and ``c in (-1, 0]``, the instantaneous curl coefficient is proportional to
``1/eps_a + c/eps_b``: non-negative for every ``c`` when ``eps_a == eps_b``,
and negative once ``|c| * eps_a / eps_b > 1``.

Without subpixel smoothing the two halves read the same array and the question
does not arise. With it, ``update_e_aniso`` uses the Kottke-smoothed
per-component array while ``apply_cpml_e`` used the staircase
``materials.eps_r`` — and at a smoothed interface cell inside a CPML pad those
disagree by construction.

#203/#205 are the same inequality reached from the other side (``materials=``
omitted, so ``eps_b = 1`` inside an ``eps_a = eps_r`` dielectric) and are
recorded as "exponential divergence -> all-NaN s11".
"""
from __future__ import annotations

import numpy as np
import pytest

import rfx
import rfx.geometry.smoothing
from rfx import Simulation
from rfx.boundaries.cpml import apply_cpml_e, init_cpml, _cpml_profile
from rfx.core.yee import EPS_0, MU_0
from rfx.geometry.rasterize_grid import extend_cpml_pad_materials
from rfx.sources import GaussianPulse
from rfx.sparams._common import settling_verdict

C0 = 2.998e8
A = 1.0e-6
DX = A / 10


# ---------------------------------------------------------------------------
# The Stage-B assembly fix, as a test-local helper.
#
# #1043's landing continues the declared geometry into the absorber pad on the
# SMOOTHED array, not just on ``materials``. That is Stage B and does not live
# under ``rfx/`` yet; these tests install it here so the stability question it
# raises is gated before it lands. Replace this helper with the production call
# when Stage B does land -- do not delete the tests with it.
# ---------------------------------------------------------------------------

def _install_pad_replication(monkeypatch):
    """Replicate the smoothed permittivity into every absorber pad."""
    import jax.numpy as jnp

    orig = rfx.geometry.smoothing.compute_smoothed_eps

    def patched(grid, shape_eps_pairs, background_eps=1.0, **kw):
        comps = orig(grid, shape_eps_pairs, background_eps=background_eps, **kw)
        pads = (int(grid.pad_x_lo), int(grid.pad_x_hi),
                int(grid.pad_y_lo), int(grid.pad_y_hi),
                int(grid.pad_z_lo), int(grid.pad_z_hi))
        out = []
        for c in comps:
            c = jnp.asarray(c)
            e2, _s, _m = extend_cpml_pad_materials(
                c, jnp.zeros_like(c), jnp.ones_like(c), *pads)
            out.append(e2)
        return tuple(out)

    monkeypatch.setattr(rfx.geometry.smoothing, "compute_smoothed_eps",
                        patched)


def _guide_sim(boundary: str, sx=8.0 * A, sy=8.0 * A, cpml_layers=10):
    """A dielectric guide spanning the full x extent, so it reaches both pads."""
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(sx, sy, DX), dx=DX,
                     boundary=boundary, cpml_layers=cpml_layers, mode="2d_tmz")
    sim.add_material("wg", eps_r=12.0)
    sim.add(rfx.Box((0, sy / 2 - 0.5 * A, 0), (sx, sy / 2 + 0.5 * A, DX)),
            material="wg")
    sim.add_source(position=(11 * DX, sy / 2, 0), component="ez",
                   waveform=GaussianPulse(f0=0.15 * C0 / A, bandwidth=0.667),
                   amplitude_kind="field")
    sim.add_probe(position=(sx - 15 * DX, sy / 2, 0), component="ez")
    return sim


# ---------------------------------------------------------------------------
# (a) the physics gate: a boundary-touching dielectric must stay finite
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("boundary", ["cpml", "upml"])
def test_boundary_touching_dielectric_in_pad_stays_finite(monkeypatch,
                                                          boundary):
    """A dielectric continued into the absorber pad must not blow the run up.

    RED on the pre-#1043 tree under ``cpml`` (measured on the same rig at
    1200 steps: 848 non-finite probe samples, the field itself first breaking
    at step 328 at the guide's transverse interface row inside the x pads);
    ``upml`` passes there and must keep passing, because ``apply_upml_e``
    builds its own coefficients from ``aniso_eps`` and never had two
    permittivities to disagree about.
    """
    _install_pad_replication(monkeypatch)
    sim = _guide_sim(boundary)
    result = sim.run(n_steps=5000, subpixel_smoothing=True,
                     skip_preflight=True)
    ts = np.asarray(result.time_series, dtype=float)

    n_bad = int((~np.isfinite(ts)).sum())
    assert n_bad == 0, (
        f"{boundary}: {n_bad} non-finite probe samples -- a dielectric carried "
        f"into the absorber pad diverged. If the CPML psi coefficient stopped "
        f"reading the E update's own permittivity, this is the witness.")

    verdict = settling_verdict(result.settling_db)
    assert verdict == "pass", (
        f"{boundary}: settling witness {verdict} "
        f"(settling_db={result.settling_db}); the record ends while the guide "
        f"is still ringing, so no DFT-derived number off this run is readable.")


# ---------------------------------------------------------------------------
# (b) the mechanism gate: the one-cell amplification factor
# ---------------------------------------------------------------------------

def amplification_rho(n_layers, dx, dt, eps_a, eps_b, n=24, absorber=True):
    """Spectral radius of the 1-D leapfrog + psi update over ``n`` cells.

        H^{n+1/2} = H^{n-1/2} + (dt/mu0) * dEz/dx
        psi^{n+1} = b*psi^n + c*dHy/dx          (new psi, used the same step)
        E^{n+1}   = E^n + (dt/(eps_a*eps0))*dHy/dx + (dt/(eps_b*eps0))*psi^{n+1}

    ``apply_cpml_e`` writes the new psi and uses it immediately, so this does
    too. ``eps_a`` is the E update's permittivity, ``eps_b`` the psi
    coefficient's. ``absorber=False`` gives b = 1, c = 0 everywhere — a plain
    lossless slice, which is the model's own comparator.

    This is the ONE copy of this model.
    ``scripts/diagnostics/cpml_subpixel_stability/amplification.py`` imports it
    rather than keeping a second; two hand-maintained copies of one derivation
    is how the repo's grep-map drifted.
    """
    b = np.ones(n)
    c = np.zeros(n)
    if absorber:
        p = _cpml_profile(n_layers, dt, dx)
        b[:n_layers] = np.asarray(p.b, dtype=float)
        c[:n_layers] = np.asarray(p.c, dtype=float)
    ea = np.full(n, float(eps_a))
    eb = np.full(n, float(eps_b))

    DE = np.zeros((n, n))
    for i in range(n - 1):
        DE[i, i + 1] = 1.0 / dx
        DE[i, i] = -1.0 / dx
    DH = np.zeros((n, n))
    for i in range(n):
        DH[i, i] = 1.0 / dx
        if i > 0:
            DH[i, i - 1] = -1.0 / dx

    m = 3 * n
    iE, iH, iP = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
    H_new = np.zeros((n, m))
    H_new[:, iH] = np.eye(n)
    H_new[:, iE] = (dt / MU_0) * DE
    P_new = np.zeros((n, m))
    P_new[:, iP] = np.diag(b)
    P_new += np.diag(c) @ DH @ H_new
    E_new = np.zeros((n, m))
    E_new[:, iE] = np.eye(n)
    E_new += np.diag(dt / (ea * EPS_0)) @ DH @ H_new
    E_new += np.diag(dt / (eb * EPS_0)) @ P_new
    Aop = np.zeros((m, m))
    Aop[iE, :] = E_new
    Aop[iH, :] = H_new
    Aop[iP, :] = P_new
    return float(np.max(np.abs(np.linalg.eigvals(Aop))))


def test_amplification_is_bounded_only_when_the_two_epsilons_agree():
    """rho <= 1 iff the psi coefficient's eps is not BELOW the update's.

    This is the mechanism behind the divergence in the test above, stated as
    an eigenvalue rather than as a symptom. The comparator is checked first:
    a lossless vacuum slice must sit exactly on the unit circle, which is what
    caught a sign error in an earlier version of this matrix.
    """
    dt = 0.5 * DX / C0
    tol = 1e-6

    # Comparator first -- a model that cannot reproduce the trivial case
    # cannot be read on the interesting one. Both trivial cases: a plain
    # lossless slice, and the same slice with a consistent absorber.
    rho_bare = amplification_rho(10, DX, dt, 1.0, 1.0, absorber=False)
    assert abs(rho_bare - 1.0) <= tol, (
        f"the amplification model is wrong: a lossless slice with NO absorber "
        f"gives rho={rho_bare}, and it must be 1")
    rho_vac = amplification_rho(10, DX, dt, 1.0, 1.0)
    assert abs(rho_vac - 1.0) <= tol, (
        f"the amplification model is wrong: a lossless vacuum CPML slice gives "
        f"rho={rho_vac}, and it must be 1")

    # Consistent: marginal, at every absolute permittivity. This is what
    # falsifies a "high eps in the graded region breaks the CFL bound"
    # reading -- eps 1 through 80 are all exactly marginal.
    for eps in (1.0, 2.0, 6.5, 12.0, 80.0):
        rho = amplification_rho(10, DX, dt, eps, eps)
        assert rho <= 1.0 + tol, f"consistent eps={eps} gives rho={rho} > 1"

    # eps_b ABOVE eps_a: under-damped absorber, still not amplifying. This is
    # the pre-#1043 state of a boundary-touching guide (aniso_eps = 1 in the
    # pad, materials.eps_r = 12 there) and it is why that case was wrong but
    # stable.
    assert amplification_rho(10, DX, dt, 1.0, 12.0) <= 1.0 + tol

    # eps_b BELOW eps_a: amplifying. 6.5 against 1.0 is the measured failing
    # cell -- a Kottke interface value carried into the pad beside a staircase
    # array that reads vacuum at the same cell.
    rho_defect = amplification_rho(10, DX, dt, 6.5, 1.0)
    assert rho_defect > 1.0 + tol, (
        f"expected an amplifying update at eps_a=6.5 / eps_b=1.0, got "
        f"rho={rho_defect}")
    assert rho_defect > 2.0, (
        f"the defect's growth rate moved: rho={rho_defect}, was 2.2086 when "
        f"this was measured. Re-derive before relaxing.")


# ---------------------------------------------------------------------------
# (c) the regression pin: threading an equal permittivity changes nothing
# ---------------------------------------------------------------------------

def _seeded_cpml_call(inv_eps_r_update, eps_r_value=4.0):
    """Run one ``apply_cpml_e`` on a seeded psi state and return the E fields.

    H is left at zero so ``curl == 0`` and ``psi_new == b * psi_old``; the E
    increment is then ``ce * b * psi_old`` and reads the coefficient directly.
    """
    import jax.numpy as jnp
    from rfx.core.yee import FDTDState, MaterialArrays

    sim = Simulation(freq_max=0.25 * C0 / A, domain=(3.0 * A, 3.0 * A, DX),
                     dx=DX, boundary="cpml", cpml_layers=6, mode="2d_tmz")
    grid = sim._build_grid()
    shape = grid.shape
    params, state0 = init_cpml(grid)

    rng = np.random.default_rng(1043)
    psi_seed = {k: jnp.asarray(rng.standard_normal(v.shape), dtype=v.dtype)
                for k, v in state0._asdict().items()}
    cpml_state = state0._replace(**psi_seed)

    z = jnp.zeros(shape, dtype=jnp.float32)
    fields = FDTDState(ex=z, ey=z, ez=z, hx=z, hy=z, hz=z, step=0)
    mats = MaterialArrays(
        eps_r=jnp.full(shape, eps_r_value, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
    )
    out, _ = apply_cpml_e(fields, params, cpml_state, grid, "xyz",
                          materials=mats,
                          inv_eps_r_update=inv_eps_r_update)
    return np.asarray(out.ex), np.asarray(out.ey), np.asarray(out.ez)


def test_threading_an_equal_permittivity_is_bit_identical():
    """``inv_eps_r_update = 1/materials.eps_r`` must reproduce ``None`` exactly.

    This is the whole byte-identity claim in one assertion: the new parameter
    changes numbers only where the two permittivities actually differ. It also
    pins the dtype, which is where the first version of the fix went wrong --
    ``compute_smoothed_eps`` returns float64 under ``JAX_ENABLE_X64`` while
    ``materials.eps_r`` is float32, and letting that through moved the field
    bytes of runs whose pads hold nothing but vacuum.
    """
    eps_r = 4.0
    base = _seeded_cpml_call(None, eps_r)
    # Hand it a float64 numpy array -- what ``compute_smoothed_eps`` returns
    # under JAX_ENABLE_X64. ``apply_cpml_e`` must bring it back to the
    # material dtype before the coefficient is formed; under x64-off JAX
    # truncates anyway, so this assertion holds in both environments and the
    # x64 lane is where it actually bites.
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(3.0 * A, 3.0 * A, DX),
                     dx=DX, boundary="cpml", cpml_layers=6, mode="2d_tmz")
    shape = sim._build_grid().shape
    inv = tuple(np.full(shape, 1.0 / eps_r, dtype=np.float64)
                for _ in range(3))
    threaded = _seeded_cpml_call(inv, eps_r)

    for name, a, b in zip(("ex", "ey", "ez"), base, threaded):
        assert a.dtype == b.dtype, f"{name}: dtype moved {a.dtype} -> {b.dtype}"
        assert np.array_equal(a, b), (
            f"{name}: threading an EQUAL permittivity changed the field bytes "
            f"(max |delta| = {np.max(np.abs(a - b))}). The parameter must be a "
            f"no-op wherever the two epsilons agree.")


@pytest.mark.parametrize("subpixel,dispersive,expect_threaded", [
    (False, False, False),   # no anisotropic array -> materials.eps_r, as before
    (True, False, True),     # Stage 1 -> 1/aniso_eps
    (True, True, False),     # dispersion wins the E update, so it wins here too
])
def test_the_coefficient_is_threaded_exactly_when_the_e_update_is_anisotropic(
        monkeypatch, subpixel, dispersive, expect_threaded):
    """Pin WHICH runs get the new coefficient, not just what it computes.

    The three stable configurations of #1043's table are stable because their
    two permittivities agree wherever ``apply_cpml_e`` writes — the
    subpixel-OFF run by taking the ``materials`` branch outright, the
    dispersive run the same way. A future edit that threads the array
    unconditionally would break the dispersive run silently, because
    ``_update_e_with_optional_dispersion`` ignores ``aniso_eps`` and the two
    halves would disagree again with the sign reversed.

    (The committed-geometry CPML control is NOT pinned as unchanged: its pad
    holds ``materials.eps_r = 12`` against ``aniso_eps = 1``, so its numbers
    move by design. cv01 Run 1 goes 0.9195301017439319 -> 0.9166511849380675;
    the measurement lives in
    `scripts/diagnostics/cpml_subpixel_stability/cv01_control.py`, removed
    with cv01 on 2026-09-21 and kept at commit 66ed61c2.)
    """
    import rfx.boundaries.cpml as _cpml
    from rfx.materials.lorentz import LorentzPole

    seen = []
    orig = _cpml.apply_cpml_e

    def spy(*a, **kw):
        seen.append(kw.get("inv_eps_r_update"))
        return orig(*a, **kw)

    monkeypatch.setattr(_cpml, "apply_cpml_e", spy)

    sy = 8.0 * A
    sim = _guide_sim("cpml")
    if dispersive:
        w0 = 2 * np.pi * (0.15 * C0 / A)
        sim.add_material("disp", eps_r=4.0,
                         lorentz_poles=[LorentzPole(omega_0=w0, delta=w0 / 100.0,
                                                    kappa=w0 ** 2)])
        sim.add(rfx.Box((2.0 * A, sy / 2 - 0.2 * A, 0),
                        (4.0 * A, sy / 2 + 0.2 * A, DX)), material="disp")
    # Long enough for the pulse to actually couple; a handful of steps leaves
    # "no field energy was recorded" (#336) on the record, which makes a
    # passing spy test look like it measured nothing.
    sim.run(n_steps=120, subpixel_smoothing=subpixel, skip_preflight=True)

    assert seen, "apply_cpml_e was never called -- the spy missed the call site"
    threaded = [v for v in seen if v is not None]
    if expect_threaded:
        assert len(threaded) == len(seen), (
            f"expected every CPML-E call to carry inv_eps_r_update, "
            f"{len(seen) - len(threaded)} of {len(seen)} did not")
    else:
        assert not threaded, (
            f"inv_eps_r_update was passed on a run whose E update does not use "
            f"an anisotropic array (subpixel={subpixel}, dispersive={dispersive})")


def test_threading_a_different_permittivity_changes_the_coefficient():
    """The mutation control for the test above: an unequal eps must move it.

    Without this, a parameter that was silently ignored would pass the
    bit-identity pin and nothing else here would notice.
    """
    import jax.numpy as jnp

    sim = Simulation(freq_max=0.25 * C0 / A, domain=(3.0 * A, 3.0 * A, DX),
                     dx=DX, boundary="cpml", cpml_layers=6, mode="2d_tmz")
    shape = sim._build_grid().shape
    base = _seeded_cpml_call(None, 4.0)
    inv = tuple(jnp.full(shape, 1.0 / 9.0, dtype=jnp.float32) for _ in range(3))
    other = _seeded_cpml_call(inv, 4.0)

    deltas = [float(np.max(np.abs(a - b))) for a, b in zip(base, other)]
    assert max(deltas) > 0.0, (
        "inv_eps_r_update had no effect at all -- the psi coefficient is not "
        f"reading it (per-component max |delta| = {deltas})")
    # And it must be the RATIO of the two permittivities, cell for cell.
    # Restricted to cells well clear of float32 cancellation: the increment is
    # ce * b * psi_seed, and psi_seed is standard-normal, so a handful of
    # cells land near zero where a ratio is meaningless.
    for a, b in zip(base, other):
        scale = np.max(np.abs(a))
        nz = np.abs(a) > 1e-4 * scale
        if not nz.any():
            continue
        ratio = b[nz] / a[nz]
        assert np.allclose(ratio, 4.0 / 9.0, rtol=1e-4), (
            "the coefficient did not scale as 1/eps: expected the field to "
            f"scale by {4.0 / 9.0}, saw min {ratio.min()} max {ratio.max()}")

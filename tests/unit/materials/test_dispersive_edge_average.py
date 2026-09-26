"""A Debye or Lorentz material keeps every E edge on the four-cell mean (#1260).

A Yee E component lies on an edge shared by four cells, and #1210 gave it the
mean of their permittivity and conductivity: tangential E is continuous across
the cell faces that meet at the edge, so the four cells act in parallel. The
dispersive updates did not follow. With a Debye or Lorentz pole anywhere in a
model they replace the E update over the whole grid, and their coefficients
were one value per CELL for all three components. A half-filled 24 mm PEC cube
then read its TE(1,0)-to-z mode 1.7 % high at dx = 1 mm, halving with the mesh
(first order), and adding a 1 mm Debye cube of delta_eps = 1e-6 in a far corner
moved a plain eps_r 4 fill from 5.2162 to 5.3069 GHz.

A pole's susceptibility is parallel along the edge exactly as a conductivity
is, and a pole entry has one tau (one omega_0, delta) wherever it is present,
so the edge is itself a Debye (Lorentz) medium with eps_inf, sigma and each
pole's delta_eps averaged over the four cells. That is what is checked:

1. at interface edges, the coefficient the dispersive update applies encodes
   eps_inf, sigma and delta_eps equal to the mean over the edge's four cells,
   indexed here by hand (not through the helper that builds them);
2. a negligible Debye block far away leaves an eps 1|4 interface edge on the
   plain lane's coefficient (was: the air cell's);
3. a lumped stamp loads its own component only, with a dispersive material in
   the model (#1236 on the dispersive lanes; they used to warn instead);
4. in a homogeneous region, with or without a pole, the coefficients are the
   one-per-cell formula's bit for bit;
5. the distributed slab body's combined Debye + Lorentz update is the
   single-device one (it used 1/gamma_Lorentz for the Lorentz dP, where the
   combined update needs 1/(gamma_Lorentz + sum beta));
6. one mesh of the half-filled cube with a Debye and with a Lorentz filling,
   against the transverse resonance solved with the complex eps(omega):
   |error| <= 5e-3 at dx = 1 mm (the cell-owned rule reads 1.7e-2, the edge
   mean 2.9e-4 on the TE line read as below).

   Which line is read: the TM(1,1)-to-z mode sits 29 MHz below TE(1,0)-to-z
   here, and with a Q ~ 66 filling one linewidth (f/Q ~ 79 MHz) covers both,
   so a single Ex probe can report the two merged. The probe used is the sum
   of Ex at x = L/2 - dx/2 and x = L/2 + dx/2: TE(0,1)-to-z's Ex does not
   depend on x and survives the sum, TM(1,1)-to-z's Ex ~ cos(pi x / a) cancels
   by the cavity's mirror symmetry.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import (
    EPS_0, FDTDState, MaterialArrays, e_component_coeffs, init_materials,
)
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import init_lorentz, lorentz_pole
from rfx.sources.sources import stamp_lumped_sigma

DT = 1.0e-12
SHAPE = (7, 8, 9)
TAU = 1.0e-11
DE = 2.0
W0, DELTA = 2 * np.pi * 60e9, 2 * np.pi * 5e9


def _background(seed=1260):
    rng = np.random.default_rng(seed)
    eps = jnp.asarray(1.0 + 5.0 * rng.random(SHAPE).astype(np.float32))
    sigma = jnp.asarray(0.5 * rng.random(SHAPE).astype(np.float32))
    return MaterialArrays(eps, sigma, jnp.ones(SHAPE, jnp.float32))


def _pole_mask(seed=7):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random(SHAPE) > 0.5)


def _four_cells(c, i, j, k):
    """The four cells incident to E component c at node (i, j, k)."""
    t1, t2 = [a for a in range(3) if a != c]
    out = []
    for d1 in (0, 1):
        for d2 in (0, 1):
            idx = [i, j, k]
            idx[t1] -= d1
            idx[t2] -= d2
            out.append(tuple(idx))
    return out


INTERIOR = [(i, j, k) for i in range(1, SHAPE[0]) for j in range(1, SHAPE[1])
            for k in range(1, SHAPE[2])]


def _edge_means(mats, mask, c, node):
    cells = _four_cells(c, *node)
    eps = np.mean([float(mats.eps_r[x]) for x in cells])
    sig = np.mean([float(mats.sigma[x]) for x in cells])
    frac = np.mean([float(mask[x]) for x in cells])
    return eps, sig, frac


@pytest.mark.parametrize("c", [0, 1, 2])
def test_debye_edge_carries_the_four_cell_mean_of_every_parameter(c):
    mats, mask = _background(), _pole_mask()
    coeffs, _ = init_debye([DebyePole(delta_eps=DE, tau=TAU)], mats, DT,
                           mask=mask)
    b_full = EPS_0 * DE * DT / (2 * TAU + DT)
    cb = np.asarray(coeffs.cb[c])
    ca = np.asarray(coeffs.ca[c])
    beta = np.asarray(coeffs.beta[c])[0]
    for node in INTERIOR:
        eps, sig, frac = _edge_means(mats, mask, c, node)
        # the pole's delta_eps on this edge, read off beta
        assert beta[node] / b_full == pytest.approx(frac, abs=1e-6), node
        # gamma = eps0*eps_inf + beta + sigma*dt/2, from cb and from ca
        gamma = DT / cb[node]
        g_expect = EPS_0 * eps + b_full * frac + sig * DT / 2
        assert gamma == pytest.approx(g_expect, rel=2e-6), node
        numer = EPS_0 * eps - b_full * frac - sig * DT / 2
        assert ca[node] == pytest.approx(numer / g_expect, rel=2e-5, abs=1e-6)


@pytest.mark.parametrize("c", [0, 1, 2])
def test_lorentz_edge_carries_the_four_cell_mean_of_every_parameter(c):
    mats, mask = _background(), _pole_mask()
    coeffs, _ = init_lorentz([lorentz_pole(DE, W0, DELTA)], mats, DT,
                             mask=mask)
    c_full = EPS_0 * DE * W0 ** 2 * DT ** 2 / (1 + DELTA * DT)
    cb = np.asarray(coeffs.cb[c])
    cvals = np.asarray(coeffs.c[c])[0]
    for node in INTERIOR:
        eps, sig, frac = _edge_means(mats, mask, c, node)
        assert cvals[node] / c_full == pytest.approx(frac, abs=1e-6), node
        assert DT / cb[node] == pytest.approx(EPS_0 * eps + sig * DT / 2,
                                              rel=2e-6), node
        # the recurrence coefficients follow the pole on every edge it reaches
        if frac > 0:
            assert float(coeffs.a[0][node]) == pytest.approx(
                (2.0 - W0 ** 2 * DT ** 2) / (1 + DELTA * DT), rel=1e-6)


def _half_filled(shape=(10, 10, 10), k_if=5, eps_fill=4.0):
    eps = np.ones(shape, np.float32)
    eps[:, :, :k_if] = eps_fill
    return MaterialArrays(jnp.asarray(eps), jnp.zeros(shape, jnp.float32),
                          jnp.ones(shape, jnp.float32))


@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_a_negligible_pole_block_leaves_the_interface_on_the_plain_coefficient(kind):
    """The #1260 fixture at coefficient level: an eps 4 | 1 interface (Ex on
    the interface plane touches two eps 4 and two eps 1 cells), a one-cell
    pole of delta_eps 1e-6 in a far corner. The dispersive update's Ex
    coefficient on the interface must be the plain lane's (mean 2.5), not the
    air cell's (1.0)."""
    shape, k_if = (10, 10, 10), 5
    mats = _half_filled(shape, k_if)
    mask = np.zeros(shape, bool)
    mask[8, 8, 8] = True
    if kind == "debye":
        coeffs, _ = init_debye([DebyePole(delta_eps=1e-6, tau=1e-11)], mats,
                               DT, mask=jnp.asarray(mask))
    else:
        coeffs, _ = init_lorentz([lorentz_pole(1e-6, W0, DELTA)], mats, DT,
                                 mask=jnp.asarray(mask))
    _, cb_plain = e_component_coeffs(mats, DT)
    node = (4, 4, k_if)
    for c in range(3):
        got = float(coeffs.cb[c][node])
        assert got == pytest.approx(float(cb_plain[c][node]), rel=1e-6), c
    eps_ex = DT / float(coeffs.cb[0][node]) / EPS_0
    assert eps_ex == pytest.approx(2.5, rel=1e-6), (
        f"Ex on the eps 4|1 plane carries eps {eps_ex:.4f}; 1.0 is the air "
        f"cell alone (the cell-owned rule, #1260)")


@pytest.mark.parametrize("kind", ["debye", "lorentz"])
@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_a_lumped_stamp_loads_only_its_own_edge_on_the_dispersive_lane(
        kind, component):
    """50 ohm on one edge (sigma 20 S/m at the node's cell) in a model with a
    dispersive block elsewhere: that component's coefficient changes at that
    node; the other two are bit-identical to no stamp. Until #1260 the stamp
    loaded all three (and the builder warned)."""
    import warnings
    base = _background()
    mask = np.zeros(SHAPE, bool)
    mask[5:, 5:, 6:] = True
    node, axis = (2, 3, 4), ("ex", "ey", "ez").index(component)
    stamped = stamp_lumped_sigma(base, node, 20.0, component)

    def build(m):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if kind == "debye":
                return init_debye([DebyePole(delta_eps=DE, tau=TAU)], m, DT,
                                  mask=jnp.asarray(mask))[0]
            return init_lorentz([lorentz_pole(DE, W0, DELTA)], m, DT,
                                mask=jnp.asarray(mask))[0]

    ref, got = build(base), build(stamped)
    for c in range(3):
        a, b = np.asarray(got.cb[c]), np.asarray(ref.cb[c])
        if c != axis:
            np.testing.assert_array_equal(a, b, err_msg=f"component {c}")
        else:
            diff = np.argwhere(a != b)
            assert [tuple(d) for d in diff] == [node], diff
            # the stamp enters sigma on its own edge: gamma grows by 20*dt/2
            assert DT / a[node] - DT / b[node] == pytest.approx(
                20.0 * DT / 2, rel=1e-4)


def _legacy_debye(eps_r, sigma, mask, dt):
    """The pre-#1260 one-per-cell formula, spelled out."""
    a = (2.0 * TAU - dt) / (2.0 * TAU + dt)
    b = EPS_0 * DE * dt / (2.0 * TAU + dt)
    alpha = jnp.where(mask, a, 0.0)
    beta = jnp.where(mask, b, 0.0)
    eps_inf = eps_r * EPS_0
    gamma = eps_inf + beta + sigma * dt / 2.0
    safe = jnp.maximum(gamma, EPS_0 * 1e-10)
    return ((eps_inf - beta - sigma * dt / 2.0) / safe, dt / safe,
            (1.0 - alpha) / safe)


def test_homogeneous_regions_keep_the_one_per_cell_bits():
    """Inside a block of one material (with the pole, and without it) the four
    cells of every edge agree, and the coefficients are the one-per-cell
    formula's bit for bit -- the property that keeps homogeneous dispersive
    fixtures and locks where they were."""
    shape = (12, 12, 12)
    eps = np.full(shape, 1.0, np.float32)
    sig = np.zeros(shape, np.float32)
    eps[:6] = 2.2
    sig[:6] = 0.03
    mask = np.zeros(shape, bool)
    mask[:6] = True
    mats = MaterialArrays(jnp.asarray(eps), jnp.asarray(sig),
                          jnp.ones(shape, jnp.float32))
    coeffs, _ = init_debye([DebyePole(delta_eps=DE, tau=TAU)], mats, DT,
                           mask=jnp.asarray(mask))
    ca0, cb0, cc0 = (np.asarray(v) for v in
                     _legacy_debye(mats.eps_r, mats.sigma, jnp.asarray(mask), DT))
    # x in 1..4: inside the pole block; x in 7..11: outside; x = 5, 6 are the
    # interface edges (x = 6 reads cells 5 and 6 for Ey and Ez).
    for sl in (slice(0, 5), slice(7, 12)):
        for c in range(3):
            np.testing.assert_array_equal(np.asarray(coeffs.ca[c])[sl], ca0[sl])
            np.testing.assert_array_equal(np.asarray(coeffs.cb[c])[sl], cb0[sl])
            np.testing.assert_array_equal(np.asarray(coeffs.cc[c])[0][sl],
                                          cc0[sl])


def test_the_distributed_mixed_update_is_the_single_device_one():
    """A cell holding both a Debye and a Lorentz pole: the slab body used the
    Lorentz build's 1/gamma_Lorentz for the Lorentz dP, the single-device
    combined update 1/(gamma_Lorentz + sum beta). With the old body restored
    this test's random state gives E up to 4.2 % apart (330 of 336 values)."""
    from rfx.runners._distributed_common import _update_e_local_with_dispersion
    from rfx.simulation import _update_e_with_optional_dispersion

    rng = np.random.default_rng(12601)
    mats = _background()
    mask = jnp.asarray(np.ones(SHAPE, bool))
    dco, dst = init_debye([DebyePole(delta_eps=DE, tau=TAU)], mats, DT, mask=mask)
    lco, lst = init_lorentz([lorentz_pole(DE, W0, DELTA)], mats, DT, mask=mask)

    def rnd(*s):
        return jnp.asarray(rng.standard_normal(s).astype(np.float32))

    st = FDTDState(*(rnd(*SHAPE) for _ in range(6)), step=jnp.array(0, jnp.int32))
    dst = dst._replace(px=rnd(1, *SHAPE), py=rnd(1, *SHAPE), pz=rnd(1, *SHAPE))
    lst = lst._replace(px=rnd(1, *SHAPE) * 1e-3, py=rnd(1, *SHAPE) * 1e-3,
                       pz=rnd(1, *SHAPE) * 1e-3, px_prev=rnd(1, *SHAPE) * 1e-3,
                       py_prev=rnd(1, *SHAPE) * 1e-3,
                       pz_prev=rnd(1, *SHAPE) * 1e-3)
    dx = 1e-3
    a, _, _ = _update_e_local_with_dispersion(st, mats, DT, dx,
                                              debye=(dco, dst), lorentz=(lco, lst))
    b, _, _ = _update_e_with_optional_dispersion(st, mats, DT, dx,
                                                 debye=(dco, dst),
                                                 lorentz=(lco, lst))
    for comp in ("ex", "ey", "ez"):
        x, y = np.asarray(getattr(a, comp)), np.asarray(getattr(b, comp))
        # away from the low faces, where the slab body's curl pads the same
        np.testing.assert_allclose(x[1:, 1:, 1:], y[1:, 1:, 1:], rtol=1e-5,
                                   atol=1e-6 * float(np.max(np.abs(y))),
                                   err_msg=comp)


# ---------------------------------------------------------------------------
# 6. one mesh of the half-filled cube against the complex-eps transverse
#    resonance (the full two-mesh measurement is in the PR; this is the
#    always-on bound)
# ---------------------------------------------------------------------------

C0 = 299792458.0
L, H = 24e-3, 12e-3
KT = np.pi / L
F_STATIC = 5.217729982e9              # the static eps 4 root (#1213's oracle)
BOUND = 5e-3                          # cell-owned reads 1.7e-2, edge mean 3.4e-4


def _complex_root(epsfun, f=5.2e9):
    """Newton on k1 cot(k1 H) + k2 cot(k2 (d-H)) = 0 over complex f; k cot(kl)
    is even in k, so the sqrt branch is irrelevant (and is +alpha coth for an
    evanescent half). rfx's e^{+jwt}: a decaying mode has Im f > 0."""
    def g(q, l):
        s = np.sqrt(q + 0j)
        return s / np.tan(s * l)

    def F(fr):
        w = 2 * np.pi * fr
        k0 = w / C0
        return g(epsfun(w) * k0 ** 2 - KT ** 2, H) + g(k0 ** 2 - KT ** 2, L - H)

    f = complex(f)
    for _ in range(100):
        d = (F(f + 1e3) - F(f - 1e3)) / 2e3
        step = F(f) / d
        f -= step
        if abs(step) < 1e-2:
            return f
    raise AssertionError("no root")


FILLINGS = {
    "debye": (lambda w: 2.0 + 2.0 / (1 + 1j * w * 1e-12),
              dict(eps_r=2.0, debye_poles=[DebyePole(delta_eps=2.0, tau=1e-12)])),
    "lorentz": (lambda w: 2.0 + 2.0 * W0 ** 2 / (W0 ** 2 - w ** 2 + 2j * DELTA * w),
                dict(eps_r=2.0, lorentz_poles=[lorentz_pole(2.0, W0, DELTA)])),
}


@pytest.mark.parametrize("kind", sorted(FILLINGS))
def test_the_half_filled_cube_with_a_dispersive_filling_reads_its_analytic_mode(kind):
    from rfx import Box, GaussianPulse, Simulation
    from rfx.harminv import harminv

    # the reference, validated against its own limit first
    assert _complex_root(lambda w: 4.0 + 0 * w).real == pytest.approx(
        F_STATIC, rel=1e-9)
    epsfun, material = FILLINGS[kind]
    f_an = _complex_root(epsfun).real

    window = (F_STATIC * 0.94, F_STATIC * 1.10)
    f0 = 0.5 * (window[0] + window[1])
    sim = Simulation(freq_max=3 * f0, domain=(L, L, L), dx=1e-3, boundary="pec")
    sim.add_material("fill", **material)
    sim.add(Box((0, 0, 0), (L, L, H)), material="fill")
    sim.add_source((L * 0.23, L * 0.31, L * 0.41), "ex", amplitude_kind="field",
                   waveform=GaussianPulse(f0=f0, bandwidth=1.0))
    dx = 1e-3
    for x in (L / 2 - dx / 2, L / 2 + dx / 2):   # the x-mirror pair (TE only)
        sim.add_probe((x, L * 0.64, L * 0.29), "ex")
    steps = 6000
    r = sim.run(n_steps=steps, skip_preflight=True)
    ts = np.asarray(r.time_series)
    modes = [m for m in harminv((ts[:, 0] + ts[:, 1])[steps // 4:],
                                float(r.grid.dt), *window) if m.Q > 30]
    f = max(modes, key=lambda m: m.amplitude).freq
    err = abs(f - f_an) / f_an
    assert err < BOUND, (
        f"{kind}-filled half cube: TE(1,0)-to-z reads {f / 1e9:.4f} GHz against "
        f"the complex-eps root {f_an / 1e9:.4f} GHz ({err:.2e}); +1.7e-2 is the "
        f"interface edge taking the air cell alone (#1260)")

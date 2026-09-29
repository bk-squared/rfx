"""The eps gradient is finite and scale-invariant at any objective scale (#1357).

A one-device graded-mesh eps gradient had 16 non-finite cells next to the
source and probes. The PEC wall in the issue's title was a confound: CPML and
PMC gave the same 16 cells. The cause is that every E-coefficient builder
wrote ``eps = eps_r*EPS_0`` and divided by it. The VJP of ``x/eps``
multiplies the cotangent by ``eps**-2`` (about 1.3e22), so once a float32
cotangent on Cb passed about 2.6e16 it overflowed, and ``0*inf`` gave NaN on
the lossless cells. The #1357 objective, a sum of squared probe fields on the
current-driven graded lane, is about 2e17, above that threshold. The fix is
``rfx.core.yee.si_value_eps_r_grad``, which keeps the SI value bit for bit
and differentiates the same coefficients written in eps_r units.

The invariant pinned here is not a value. The objective is quadratic in its
scale, so for a power-of-two ``s``, ``grad(s**2 * L) == s**2 * grad(L)``
exactly in float32 unless something overflows or underflows on the way. Each
row compares the gradient at two cotangent scales on either side of the
threshold that ``origin/main`` crossed. Measured on these models before the
fix: the uniform lane is finite at ``2**60`` and has 804 non-finite cells
of 3375 at ``2**80``. The graded lane is finite at ``2**0`` and has 552 at
``2**20``. The 2-D ADI lane has all 441 cells non-finite at every scale. The
#1357 fixture has 16 at cotangent 1.

Rows that still overflow are strict xfails naming #1357 and #1325. The
Debye/Lorentz ADE builders and the distributed slab updates are rewritten by
open PR #1325, and are routed through the helper after it merges. The named
builders themselves are listed in
``tests/contracts/test_eps_r_unit_coefficient_builders.py``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx.adi
import rfx.boundaries.cpml
import rfx.core.yee
import rfx.lumped
import rfx.materials.thin_conductor
from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import MaterialArrays, init_state, si_value_eps_r_grad
from rfx.geometry.csg import Box
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole

# The fixtures keep the drive conventions the #1357 report used; the per-path
# amplitude_kind deprecation (#571) and the distributed lane's opt-in notice
# are not what this file is about.
pytestmark = [
    pytest.mark.filterwarnings("ignore:add_source\\(\\.\\.\\., amplitude_kind=None\\)"),
    pytest.mark.filterwarnings("ignore:Simulation.forward\\(distributed=True\\) is opt-in"),
]

DX = 1e-3
REL_TOL = 1e-5

# log2 of the cotangent on L, per lane: (small, large). The large one is past
# where origin/main overflowed; on the 2-D ADI lane both were.
SCALES = {"uniform": (0, 90), "nu": (-40, 40), "distributed_nu": (-40, 40),
          "adi_2d": (0, 60)}

DEFERRED_1325 = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#1357: this lane/material still divides by SI eps in the reverse "
           "pass; its builders are rewritten by open PR #1325 and are routed "
           "through si_value_eps_r_grad after it merges")


def _model(lane, material):
    kw = {}
    if lane in ("nu", "distributed_nu"):
        kw["dx_profile"] = np.array([1.0, 0.9, 0.8, 0.9, 1.0, 1.0]) * DX
    if lane.startswith("adi"):
        three_d = lane == "adi_3d"
        sim = Simulation(freq_max=10e9, domain=(12e-3, 12e-3, 12e-3 if three_d else DX),
                         dx=DX, boundary="cpml", cpml_layers=4, solver="adi",
                         mode="3d" if three_d else "2d_tmz")
        z = 6e-3 if three_d else 0.0
        if material == "sigma":
            sim.add_material("m", eps_r=2.0, sigma=0.2)
            sim.add(Box((7e-3, 2e-3, 0.0), (10e-3, 10e-3, 12e-3 if three_d else DX)),
                    material="m")
        sim.add_source((5e-3, 6e-3, z), "ez")
        sim.add_probe((8e-3, 6e-3, z), "ez")
        return sim
    sim = Simulation(freq_max=15e9, domain=(6e-3, 6e-3, 6e-3), dx=DX,
                     cpml_layers=4, boundary="cpml", **kw)
    if material != "vacuum":
        mat = dict(eps_r=3.0, sigma=0.5 if material == "sigma" else 0.0)
        if material in ("debye", "debye+lorentz"):
            mat["debye_poles"] = [DebyePole(delta_eps=2.0, tau=8e-12)]
        if material in ("lorentz", "debye+lorentz"):
            mat["lorentz_poles"] = [lorentz_pole(2.0, 2 * np.pi * 10e9, 1e9)]
        sim.add_material("m", **mat)
        sim.add(Box((3.2e-3, 0.0, 0.0), (6e-3, 6e-3, 6e-3)), material="m")
    sim.add_source((2e-3, 3e-3, 3e-3), "ez")
    sim.add_probe((4e-3, 3e-3, 3e-3), "ez")
    return sim


def _issue_1357_fixture():
    """The fixture of #1357: 7 graded x cells, PEC x_lo, CPML x_hi."""
    profile = np.full(7, DX)
    profile[1:4] = np.array([0.95, 0.95, 1.1]) * DX
    sim = Simulation(freq_max=15e9, domain=(float(profile.sum()), 24 * DX, 24 * DX),
                     dx=DX, cpml_layers=8, dx_profile=profile,
                     boundary=BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                                           y="cpml", z="cpml"))
    sim.add_source((4 * DX, 9 * DX, 10 * DX), "ez")
    sim.add_probe((2 * DX, 9 * DX, 10 * DX), "ez")
    sim.add_probe((5 * DX, 15 * DX, 10 * DX), "ez")
    return sim


def _shape(sim):
    if sim._dx_profile is not None:
        g = sim._build_nonuniform_grid()
        return (g.nx, g.ny, g.nz)
    return tuple(sim._build_grid().shape)


def _gradients(factory, exps, n_steps, **forward_kw):
    """``[grad(2**k * L) for k in exps]`` from one forward pass, with ``L``
    the sum of the squared probe records and eps = 1 everywhere."""
    eps = jnp.ones(_shape(factory()), jnp.float32)

    def loss(e):
        ts = factory().forward(n_steps=n_steps, eps_override=e,
                               skip_preflight=True, **forward_kw).time_series
        return jnp.sum(ts ** 2)

    _, vjp = jax.vjp(loss, eps)
    return [np.asarray(vjp(jnp.float32(2.0 ** k))[0]) for k in exps]


def _assert_finite_and_scale_invariant(grads, exps):
    for g, k in zip(grads, exps):
        n_bad = int(np.sum(~np.isfinite(g)))
        assert n_bad == 0, f"cotangent 2**{k}: {n_bad} non-finite gradient cells"
    ref = grads[0] * 2.0 ** -exps[0]
    peak = float(np.max(np.abs(ref)))
    assert peak > 0.0
    for g, k in zip(grads[1:], exps[1:]):
        err = float(np.max(np.abs(g * 2.0 ** -k - ref))) / peak
        assert err <= REL_TOL, f"cotangent 2**{k}: {err:.3e} of the peak"


def test_issue_1357_fixture_gradient_is_finite():
    """The issue's own objective (cotangent 1) and a 2**-40 copy of it."""
    grads = _gradients(_issue_1357_fixture, (0, -40), n_steps=120)
    _assert_finite_and_scale_invariant(grads, (0, -40))


# Each model is half vacuum (sigma = 0, where 0*inf became NaN) and half a
# lossy or dispersive slab, inside CPML.
@pytest.mark.parametrize("lane, material", [
    ("uniform", "sigma"),
    ("nu", "sigma"),
    ("adi_2d", "sigma"),
    pytest.param("uniform", "debye+lorentz", marks=DEFERRED_1325),
    pytest.param("distributed_nu", "sigma", marks=DEFERRED_1325),
])
def test_gradient_is_scale_invariant(lane, material, request):
    kw = {}
    if lane == "distributed_nu":
        kw = dict(distributed=True, devices=request.getfixturevalue("two_devices"))
    exps = SCALES[lane]
    grads = _gradients(lambda: _model(lane, material), exps,
                       n_steps=24 if lane.startswith("adi") else 40, **kw)
    _assert_finite_and_scale_invariant(grads, exps)


# --- the named builders one by one: VJP at a cotangent of 1e30 -------------

def _materials(rng, shape):
    return MaterialArrays(
        eps_r=jnp.asarray(rng.uniform(1.0, 6.0, shape), jnp.float32),
        sigma=jnp.asarray(np.where(rng.uniform(size=shape) > 0.5,
                                   rng.uniform(0.0, 20.0, shape), 0.0), jnp.float32),
        mu_r=jnp.ones(shape, jnp.float32))


def _builder_cases():
    """``name -> (fn(eps_r), eps_r, cotangent)``. Each cotangent is above the
    one where the builder's SI spelling overflowed (about 2.6e16 on a Cb; any
    cotangent on the ADI couplings; about 1e27 through the inverse-eps,
    ``1/eps`` and ``eps/dt`` forms, where ``dt`` multiplies the cotangent
    first) and below where field-unit arithmetic (a curl's ``1/dx``) would
    overflow on its own."""
    rng = np.random.default_rng(1357)
    shape = (5, 6, 7)
    mats = _materials(rng, shape)
    dt, dx = 1.9e-12, DX
    st = init_state(shape)
    st = st._replace(**{c: jnp.asarray(rng.standard_normal(shape), jnp.float32)
                        for c in ("ex", "ey", "ez", "hx", "hy", "hz")})
    m2 = _materials(rng, (12, 12, 1))
    eps2, sig2 = m2.eps_r[:, :, 0], m2.sigma[:, :, 0]
    f2 = [jnp.asarray(rng.standard_normal((12, 12)), jnp.float32) for _ in range(3)]
    params_2d, cpml_2d = rfx.adi.init_adi_cpml_2d(3, dt, dx, dx, 12, 12)
    cpml_2d = jax.tree.map(
        lambda a: jnp.asarray(rng.standard_normal(a.shape), jnp.float32), cpml_2d)
    sheet = jnp.full(shape, 4e4, jnp.float32).at[0].set(0.0)
    return {
        "e_update_coeffs": (
            lambda e: rfx.core.yee.e_update_coeffs(e, mats.sigma, dt), mats.eps_r, 1e24),
        "precompute_coeffs": (
            lambda e: rfx.core.yee.precompute_coeffs(mats._replace(eps_r=e), dt, dx),
            mats.eps_r, 1e24),
        "update_e_aniso": (
            lambda e: rfx.core.yee.update_e_aniso(st, mats, e, e, e, dt, dx).ex,
            mats.eps_r, 1e24),
        "update_e_aniso_inv": (
            lambda e: rfx.core.yee.update_e_aniso_inv(
                st, mats, 1.0 / e, 1.0 / e, 1.0 / e, dt, dx).ez, mats.eps_r, 1e28),
        "adi_step_2d": (
            lambda e: rfx.adi.adi_step_2d(*f2, e, sig2, 5 * dt, dx, dx)[0], eps2, 1e20),
        "adi_step_3d": (
            lambda e: rfx.adi.adi_step_3d(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
                                          e, mats.sigma, 5 * dt, dx, dx, dx)[2],
            mats.eps_r, 1e20),
        "apply_adi_cpml_2d": (
            lambda e: rfx.adi.apply_adi_cpml_2d(*f2, params_2d, cpml_2d, e, dt, dx, dx)[0],
            eps2, 1e31),
        "sheet_update_coeffs": (
            lambda e: rfx.materials.thin_conductor.sheet_update_coeffs(
                sheet, mats._replace(eps_r=e), dt), mats.eps_r, 1e24),
        "edge_update_denominator": (
            lambda e: rfx.lumped.edge_update_denominator(
                mats._replace(eps_r=e), (2, 3, 3), "ez", dt), mats.eps_r, 1e30),
    }


@pytest.mark.parametrize("name", [
    "e_update_coeffs", "precompute_coeffs", "update_e_aniso", "update_e_aniso_inv",
    "adi_step_2d", "adi_step_3d", "apply_adi_cpml_2d", "sheet_update_coeffs",
    "edge_update_denominator"])
def test_builder_vjp_is_finite_at_any_cotangent_scale(name):
    """The builder's eps VJP at a cotangent past its SI overflow is finite
    and is exactly ``2**100`` times the VJP at ``2**-100`` of it."""
    fn, eps, big = _builder_cases()[name]

    def at(scale):
        out, vjp = jax.vjp(fn, eps)
        return vjp(jax.tree.map(lambda o: jnp.full_like(o, scale), out))[0]

    # Jitted, except the lumped D0: its SI ``eps_r*EPS_0/dt`` overflows as an
    # un-jitted ``jax.grad`` runs it (op by op, outside the time loop), and
    # under jit XLA folds ``(ct / dt) * EPS_0`` into ``ct * (EPS_0/dt)``.
    if name != "edge_update_denominator":
        at = jax.jit(at)
    g_big = np.asarray(at(jnp.float32(big)))
    g_small = np.asarray(at(jnp.float32(big * 2.0 ** -100)))
    assert np.all(np.isfinite(g_big)), f"{name}: {int(np.sum(~np.isfinite(g_big)))} non-finite"
    peak = float(np.max(np.abs(g_big)))
    assert peak > 0.0
    assert float(np.max(np.abs(g_small * 2.0 ** 100 - g_big))) <= REL_TOL * peak


# --- forward bits: the helper is transparent -----------------------------

_IMPORTERS = (rfx.core.yee, rfx.boundaries.cpml, rfx.adi,
              rfx.materials.thin_conductor, rfx.lumped)


def _si_only(si_fn, eps_r_fn, *args):
    return si_fn(*args)


@pytest.mark.parametrize("lane", ["uniform", "nu", "adi_3d"])
def test_forward_bits_are_the_si_spelling(lane, monkeypatch):
    """A forward run is bit-identical to the same run with the helper replaced
    by its SI function, i.e. to the arithmetic before #1357. On the 3-D ADI
    lane a value-level combination (``stop_gradient(si) + (r - stop_gradient(r))``,
    exactly ``si`` in isolation) moved the trace by 1.6e-6 of its peak,
    because XLA's algebraic simplifier rewrote the SI arithmetic differently
    once the graph had changed."""
    def run():
        # No override: the materials are compile-time constants, which is
        # where the simplifier acts (a traced eps_override leaves it nothing
        # to fold).
        return np.asarray(_model(lane, "sigma").forward(
            n_steps=40, skip_preflight=True).time_series)

    got = run()
    for module in _IMPORTERS:
        assert module.si_value_eps_r_grad is si_value_eps_r_grad
        monkeypatch.setattr(module, "si_value_eps_r_grad", _si_only)
    jax.clear_caches()  # no compiled kernel may carry the helper across
    ref = run()
    jax.clear_caches()
    assert np.max(np.abs(ref)) > 0.0
    assert got.tobytes() == ref.tobytes()


def test_helper_returns_si_bits_and_the_eps_r_derivative():
    """The value is ``si_fn``'s, eager and under jit and value_and_grad; the
    derivative is ``eps_r_fn``'s; host inputs stay host arithmetic."""
    def si(e, s, dt):
        return (dt / (e * 8.8541878128e-12)) / (1.0 + s * dt / (2.0 * e * 8.8541878128e-12))

    def r(e, s, dt):
        return rfx.core.yee.e_coeffs_eps_r_units(e, s, dt)[1]

    rng = np.random.default_rng(7)
    e = jnp.asarray(rng.uniform(1.0, 9.0, 64), jnp.float32)
    s = jnp.asarray(rng.uniform(0.0, 50.0, 64), jnp.float32)
    dt = 1.9e-12
    want = np.asarray(si(e, s, dt)).tobytes()
    assert np.asarray(si_value_eps_r_grad(si, r, e, s, dt)).tobytes() == want
    f = jax.jit(lambda e, s: si_value_eps_r_grad(si, r, e, s, dt))
    assert np.asarray(f(e, s)).tobytes() == np.asarray(jax.jit(si, static_argnums=2)(e, s, dt)).tobytes()
    v, g = jax.value_and_grad(lambda e: jnp.sum(si_value_eps_r_grad(si, r, e, s, dt)))(e)
    g_r = jax.grad(lambda e: jnp.sum(r(e, s, dt)))(e)
    assert np.asarray(g).tobytes() == np.asarray(g_r).tobytes()
    assert np.isfinite(float(v))
    host = si_value_eps_r_grad(si, r, np.float64(2.0), 0.5, dt)
    assert isinstance(host, float) and host == si(np.float64(2.0), 0.5, dt)

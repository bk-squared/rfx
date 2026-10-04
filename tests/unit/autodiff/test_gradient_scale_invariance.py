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
threshold that ``origin/main`` crossed. Measured on these models on
``origin/main`` (0d20666a, JAX 0.10.2), non-finite cells: the uniform lane
0 at ``2**60``, 804 of 3375 at ``2**80``, 2522 at ``2**90``; the same with
UPML 1407 at ``2**90``, with PEC/PMC faces 1342 of 1815; the graded lane 0
at ``2**0``, 552 at ``2**20``, 2710 at ``2**40``, and 2712 on
``forward(distributed=True)``; the 2-D ADI lane all 441 at every scale; the
3-D ADI lane 0 at ``2**0``, the first 5 of 9261 at ``2**8`` (an objective
of 3.2e-3, so about 1 after scaling), 4251 at ``2**30``, 7942 at ``2**60``.
The #1357 fixture has 16 at cotangent 1.

Finite and scale-exact is not the same as right, so each builder's float32
tangent is also compared with its SI spelling's derivative in float64
(``test_builder_tangent_is_the_si_derivative``).

Debye/Lorentz ADE builders and the mixed update share the same routing,
including the distributed in-loop builders introduced by #1325. Every reader
of EPS_0 is accounted for in the coefficient-builder contract.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx.adi
import rfx.boundaries.cpml
import rfx.boundaries.upml
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
# where origin/main overflowed; on the 2-D ADI lane both were. "upml" and
# "walls" are the uniform lane with other boundaries: UPML's own E
# coefficients (1407 non-finite cells at 2**90 before they were routed), and
# PEC/PMC faces, which have no coefficient builder of their own.
SCALES = {"uniform": (0, 90), "upml": (0, 90), "walls": (0, 90),
          "nu": (-40, 40), "distributed_nu": (-40, 40),
          "adi_2d": (0, 60), "adi_3d": (0, 60)}

def _model(lane, material):
    kw = {"boundary": "cpml"}
    if lane in ("nu", "distributed_nu"):
        kw["dx_profile"] = np.array([1.0, 0.9, 0.8, 0.9, 1.0, 1.0]) * DX
    if lane == "upml":
        kw["boundary"] = "upml"
    if lane == "walls":
        kw["boundary"] = BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                                      y=Boundary(lo="pmc", hi="cpml"), z="cpml")
    if lane.startswith("adi"):
        three_d = lane == "adi_3d"
        sim = Simulation(freq_max=10e9, domain=(12e-3, 12e-3, 12e-3 if three_d else DX),
                         dx=DX, boundary="pec", solver="adi",
                         mode="3d" if three_d else "2d_tmz")
        z = 6e-3 if three_d else 0.0
        if material == "sigma":
            # origin/main's half-vacuum lossy slab: ADI refuses it in 2.0
            # (#1373), so its rows are strict xfails that XPASS in 2.1.
            sim.add_material("m", eps_r=2.0, sigma=0.2)
            sim.add(Box((7e-3, 2e-3, 0.0), (10e-3, 10e-3, 12e-3 if three_d else DX)),
                    material="m")
        elif material == "fill":
            # The same lossy material over the whole domain: the homogeneous
            # fill ADI carries in 2.0.
            sim.add_material("m", eps_r=2.0, sigma=0.2)
            sim.add(Box((0.0, 0.0, 0.0), (12e-3, 12e-3, 12e-3 if three_d else DX)),
                    material="m")
        sim.add_source((5e-3, 6e-3, z), "ez", amplitude_kind="field")
        sim.add_probe((8e-3, 6e-3, z), "ez")
        return sim
    sim = Simulation(freq_max=15e9, domain=(6e-3, 6e-3, 6e-3), dx=DX,
                     cpml_layers=4, **kw)
    if material != "vacuum":
        mat = dict(eps_r=3.0, sigma=0.5 if material == "sigma" else 0.0)
        if material in ("debye", "debye+lorentz"):
            mat["debye_poles"] = [DebyePole(delta_eps=2.0, tau=8e-12)]
        if material in ("lorentz", "debye+lorentz"):
            mat["lorentz_poles"] = [lorentz_pole(2.0, 2 * np.pi * 10e9, 1e9)]
        sim.add_material("m", **mat)
        sim.add(Box((3.2e-3, 0.0, 0.0), (6e-3, 6e-3, 6e-3)), material="m")
    sim.add_source((2e-3, 3e-3, 3e-3), "ez",
                   amplitude_kind="current" if lane in ("nu", "distributed_nu") else "field")
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


@pytest.mark.parametrize("lane", ["adi_2d", "adi_3d"])
def test_adi_scalar_override_gradient_is_scale_invariant(lane):
    """The ADI lane-level overflow check, always on in 2.0: the derivative with
    respect to a scalar (homogeneous) eps override of a lossy fill inside
    CPML, at cotangents 2**0 and 2**60. The SI spelling's
    ``(MU_0*eps*dx**2)**-2`` overflowed here (#1357)."""
    exps = (0, 60)
    n_steps = {"adi_2d": 24, "adi_3d": 16}[lane]

    def loss(e):
        ts = _model(lane, "fill").forward(n_steps=n_steps, eps_override=e,
                                          skip_preflight=True).time_series
        return jnp.sum(ts ** 2)

    _, vjp = jax.vjp(loss, jnp.float32(1.0))
    grads = [np.asarray(vjp(jnp.float32(2.0 ** k))[0]) for k in exps]
    _assert_finite_and_scale_invariant(grads, exps)


def test_issue_1357_fixture_gradient_is_finite():
    """The issue's own objective (cotangent 1) and a 2**-40 copy of it."""
    grads = _gradients(_issue_1357_fixture, (0, -40), n_steps=120)
    _assert_finite_and_scale_invariant(grads, (0, -40))


# Each model is half vacuum (sigma = 0, where 0*inf became NaN) and half a
# lossy or dispersive slab, inside CPML (or UPML, or PEC/PMC and CPML faces).
# ADI refuses that slab and a traced array override (#1373, 2.0): its rows keep
# origin/main's slab as strict xfails, which XPASS once ADI's interface
# permittivity is fixed (2.1). The always-on ADI check meanwhile is
# test_adi_scalar_override_gradient_is_scale_invariant below.
_ADI_REFUSED = pytest.mark.xfail(
    strict=True, raises=NotImplementedError,
    reason="#1373: ADI refuses a traced array override in 2.0")
# The dispersive rows are split by pole kind to identify each builder: a Debye-only or Lorentz-only model uses that builder's
# own ca/cb, a model with both uses the mixed update (dt/gamma_total).
@pytest.mark.parametrize("lane, material", [
    ("uniform", "sigma"),
    ("upml", "sigma"),
    ("walls", "sigma"),
    ("nu", "sigma"),
    pytest.param("adi_2d", "sigma", marks=_ADI_REFUSED),
    pytest.param("adi_3d", "sigma", marks=_ADI_REFUSED),
    ("distributed_nu", "sigma"),
    pytest.param("uniform", "debye", marks=pytest.mark.slow),
    pytest.param("uniform", "lorentz", marks=pytest.mark.slow),
    pytest.param("uniform", "debye+lorentz", marks=pytest.mark.slow),
    pytest.param("nu", "debye+lorentz", marks=pytest.mark.slow),
    pytest.param("distributed_nu", "debye", marks=pytest.mark.slow),
    pytest.param("distributed_nu", "lorentz", marks=pytest.mark.slow),
    pytest.param("distributed_nu", "debye+lorentz", marks=pytest.mark.slow),
])
def test_gradient_is_scale_invariant(lane, material, request):
    kw = {}
    if lane == "distributed_nu":
        kw = dict(distributed=True, devices=request.getfixturevalue("two_devices"))
    exps = SCALES[lane]
    n_steps = {"adi_2d": 24, "adi_3d": 16}.get(lane, 40)
    grads = _gradients(lambda: _model(lane, material), exps, n_steps=n_steps, **kw)
    _assert_finite_and_scale_invariant(grads, exps)


# --- the named builders one by one: VJP at a cotangent of 1e30 -------------

def _materials(rng, shape):
    return MaterialArrays(
        eps_r=jnp.asarray(rng.uniform(1.0, 6.0, shape), jnp.float32),
        sigma=jnp.asarray(np.where(rng.uniform(size=shape) > 0.5,
                                   rng.uniform(0.0, 20.0, shape), 0.0), jnp.float32),
        mu_r=jnp.ones(shape, jnp.float32))


DISPERSION_BUILDERS = (
    "init_debye", "init_lorentz", "debye_e_coeffs", "lorentz_e_coeffs",
    "mixed_e_component_coeffs",
)


def _dispersion_cases(dtype):
    """Two masked poles, component means, and the mixed update's SI input.

    Float32-rounded inputs are shared by the float32 and float64 rows.
    Initialization and in-loop builders are both exercised: the distributed
    runners call the latter directly instead of repeating initialization.
    """
    rng = np.random.default_rng(13251357)
    shape, dt = (5, 6, 7), 1.9e-12

    def arr(a):
        return jnp.asarray(np.asarray(a, np.float32), dtype)

    m = MaterialArrays(arr(rng.uniform(1.0, 6.0, shape)),
                       arr(np.where(rng.uniform(size=shape) > 0.5,
                                    rng.uniform(0.0, 20.0, shape), 0.0)),
                       arr(np.ones(shape)))
    masks = [jnp.asarray(rng.uniform(size=shape) > .5) for _ in range(2)]
    debye, lorentz = rfx.materials.debye, rfx.materials.lorentz
    dp = [DebyePole(2.0, 8e-12), DebyePole(.7, 20e-12)]
    lp = [lorentz_pole(2.0, 2 * np.pi * 10e9, 1e9),
          lorentz_pole(.7, 2 * np.pi * 15e9, 2e9)]

    def debye_coeffs(e, s):
        return debye.init_debye(dp, m._replace(eps_r=e, sigma=s), dt, masks)[0]

    def lorentz_coeffs(e, s):
        return lorentz.init_lorentz(lp, m._replace(eps_r=e, sigma=s), dt, masks)[0]

    dc, lc = debye_coeffs(m.eps_r, m.sigma), lorentz_coeffs(m.eps_r, m.sigma)

    def components(e, s):
        return rfx.core.yee.component_e_materials(m._replace(eps_r=e, sigma=s))

    def mixed(e, s):
        dc, lc = debye_coeffs(e, s), lorentz_coeffs(e, s)
        return tuple(lorentz.mixed_e_component_coeffs(dc, lc, c, dt) for c in range(3))

    fns = {
        "init_debye": debye_coeffs,
        "init_lorentz": lorentz_coeffs,
        "debye_e_coeffs": lambda e, s: debye.debye_e_coeffs(
            components(e, s), dt, dc.alpha, dc.beta),
        "lorentz_e_coeffs": lambda e, s: lorentz.lorentz_e_coeffs(
            components(e, s), dt, lc.a, lc.b, lc.c),
        "mixed_e_component_coeffs": mixed,
    }
    return {name: (fn, m.eps_r, m.sigma) for name, fn in fns.items()}


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
    upml_grid = Simulation(freq_max=15e9, domain=(3e-3, 3e-3, 3e-3), dx=DX,
                           boundary="upml", cpml_layers=3)._build_grid()
    m3 = _materials(rng, upml_grid.shape)
    return {
        **{name: (lambda e, fn=fn, sigma=sigma: fn(e, sigma), eps, 1e24)
           for name, (fn, eps, sigma) in _dispersion_cases(jnp.float32).items()},
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
        "init_upml": (
            lambda e: rfx.boundaries.upml.init_upml(
                upml_grid, m3._replace(eps_r=e))[:6], m3.eps_r, 1e24),
    }


@pytest.mark.parametrize("name", [
    "e_update_coeffs", "precompute_coeffs", "update_e_aniso", "update_e_aniso_inv",
    "adi_step_2d", "adi_step_3d", "apply_adi_cpml_2d", "sheet_update_coeffs",
    "edge_update_denominator", "init_upml", *DISPERSION_BUILDERS])
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

# The modules that import the helper by name. The distributed runners import
# it inside their CPML functions, from rfx.core.yee, at call time.
_IMPORTERS = (rfx.core.yee, rfx.boundaries.cpml, rfx.boundaries.upml, rfx.adi,
              rfx.materials.thin_conductor, rfx.lumped,
              rfx.materials.debye, rfx.materials.lorentz)


def _si_only(si_fn, eps_r_fn, *args):
    return si_fn(*args)


# Always on: 3-D ADI, the lane where a value-level combination was measured to
# move the bits. The other lanes (about 35 s on CPU together) run in the slow
# lane; test_helper_returns_si_bits_and_the_eps_r_derivative pins the value
# bitwise at the helper itself on every run.
@pytest.mark.parametrize("lane", [
    *(pytest.param(lane, marks=pytest.mark.slow)
      for lane in ("uniform", "upml", "nu", "distributed_nu", "adi_2d")),
    "adi_3d"])
def test_forward_bits_are_the_si_spelling(lane, monkeypatch, request):
    """A forward run is bit-identical to the same run with the helper replaced
    by its SI function: the helper adds nothing to the forward graph. On the
    3-D ADI lane a value-level combination (``stop_gradient(si) + (r -
    stop_gradient(r))``, exactly ``si`` in isolation) moved the trace by
    1.6e-6 of its peak, because XLA's algebraic simplifier rewrote the SI
    arithmetic differently once the graph had changed.

    What this does not pin: the builders' own refactors (the SI arithmetic
    moved into ``*_si`` functions, ``update_e_aniso`` through
    ``e_update_coeffs``, the ADI 2-D ``ce`` hoist, UPML's ``eps_0`` moved into
    its spellings) against the code before #1357. Those were compared with
    ``origin/main`` once, bitwise, when #1357 was reviewed."""
    kw = {}
    if lane == "distributed_nu":
        kw = dict(distributed=True, devices=request.getfixturevalue("two_devices"))

    def run():
        # No override: the materials are compile-time constants, which is
        # where the simplifier acts (a traced eps_override leaves it nothing
        # to fold).
        # ADI carries only a homogeneous fill (#1373).
        material = "fill" if lane.startswith("adi") else "sigma"
        return np.asarray(_model(lane, material).forward(
            n_steps=40, skip_preflight=True, **kw).time_series)

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


# --- the tangent is the SI derivative -------------------------------------

# Rows above pin that each builder's eps gradient is finite and scales
# exactly; neither says it is the RIGHT derivative. A wrong eps_r spelling
# (a term dropped from UPML's Cb denominator, ADI's ce written over eps_r
# instead of eps_plus_r) keeps the forward bits and a finite, scale-exact
# gradient, and was green in every gate. Here each builder's float32 JVP,
# which goes through the eps_r spelling, is compared with the JVP of the same
# builder with the helper replaced by its SI function (``_si_only``) and run
# in float64 on the same float32-rounded inputs, where the SI spelling's
# derivative is exact to ~1e-16 and cannot overflow.
#
# Tolerance: 9 float32 ULP of the tangent's peak (per output leaf), the
# repo's cross-trace budget. A tangent is a short chain of float32 ops
# (a division, a sum or two), each rounded to 0.5 ULP of its own value, so
# the error is a few ULP of the peak: measured at most 5.4 ULP (3-D ADI, through
# its tridiagonal solves), 2.6 on every other row.
#
# The sheet row's inputs are chosen well-conditioned. Its B =
# -expm1(-x2)/sigma_tot, x2 = sigma_tot*dt/(EPS_0*eps_r), has
# dB/dsigma = (dt/EPS_0/eps_r) * (exp(-x2) - (1 - exp(-x2))/x2) / sigma_tot,
# a difference that cancels as x2 -> 0: its float32 error relative to the
# peak grows like 1/x2 in EITHER spelling (over 40 draws with background
# sigma in [0, 20] S/m: up to 423 ULP for both the eps_r and the float32 SI
# spelling). So the row's lossy background cells carry sigma in [10, 40] S/m
# (x2 >= 0.36; at most 3.4 ULP over 40 draws) and the sheet cells 4e4 S/m.
JVP_TOL_ULP = 9.0


def _jvp_cases(dtype):
    """``name -> (fn(eps_r, sigma), eps_r, sigma)``, every array in ``dtype``
    built from one float32 draw, so the float32 and float64 rows see the same
    numbers."""
    rng = np.random.default_rng(13570)

    def arr(a):
        return jnp.asarray(np.asarray(a, np.float32), dtype)

    def mats(shape):
        return MaterialArrays(
            eps_r=arr(rng.uniform(1.0, 6.0, shape)),
            sigma=arr(np.where(rng.uniform(size=shape) > 0.5,
                               rng.uniform(0.0, 20.0, shape), 0.0)),
            mu_r=arr(np.ones(shape)))

    def fields(shape):
        st = init_state(shape, field_dtype=dtype)
        return st._replace(**{c: arr(rng.standard_normal(shape))
                              for c in ("ex", "ey", "ez", "hx", "hy", "hz")})

    dt, dx = 1.9e-12, DX
    shape = (5, 6, 7)
    m, st = mats(shape), fields(shape)
    m2 = mats((12, 12, 1))
    f2 = [arr(rng.standard_normal((12, 12))) for _ in range(3)]
    params_2d, cpml_2d = rfx.adi.init_adi_cpml_2d(3, dt, dx, dx, 12, 12)
    cpml_2d = jax.tree.map(lambda a: arr(rng.standard_normal(a.shape)), cpml_2d)
    sheet = arr(np.where(rng.uniform(size=shape) > 0.5, 4e4, 0.0))
    sheet_sigma = arr(np.where(rng.uniform(size=shape) > 0.5,
                               rng.uniform(10.0, 40.0, shape), 0.0))
    upml_grid = Simulation(freq_max=15e9, domain=(3e-3, 3e-3, 3e-3), dx=DX,
                           boundary="upml", cpml_layers=3)._build_grid()
    mu = mats(upml_grid.shape)
    cpml_grid = Simulation(freq_max=15e9, domain=(4e-3, 4e-3, 4e-3), dx=DX,
                           cpml_layers=3)._build_grid()
    mc, stc = mats(cpml_grid.shape), fields(cpml_grid.shape)
    cparams, cstate = rfx.boundaries.cpml.init_cpml(cpml_grid)
    cstate = jax.tree.map(lambda a: arr(rng.standard_normal(a.shape)), cstate)
    yee = rfx.core.yee
    return {
        **_dispersion_cases(dtype),
        "e_update_coeffs": (
            lambda e, s: yee.e_update_coeffs(e, s, dt), m.eps_r, m.sigma),
        "precompute_coeffs": (
            lambda e, s: yee.precompute_coeffs(m._replace(eps_r=e, sigma=s), dt, dx),
            m.eps_r, m.sigma),
        "update_e_aniso": (
            lambda e, s: yee.update_e_aniso(st, m._replace(sigma=s), e, 0.5 * e + 1.0,
                                            2.0 * e, dt, dx)[:3],
            m.eps_r, m.sigma),
        "update_e_aniso_inv": (
            lambda e, s: yee.update_e_aniso_inv(st, m._replace(sigma=s), 1.0 / e,
                                                0.5 / e, 1.0 / (e + 1.0), dt, dx)[:3],
            m.eps_r, m.sigma),
        "adi_step_2d": (
            lambda e, s: rfx.adi.adi_step_2d(*f2, e, s, 5 * dt, dx, dx),
            m2.eps_r[:, :, 0], m2.sigma[:, :, 0]),
        "adi_step_3d": (
            lambda e, s: rfx.adi.adi_step_3d(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
                                             e, s, 5 * dt, dx, dx, dx),
            m.eps_r, m.sigma),
        "apply_adi_cpml_2d": (
            lambda e, s: rfx.adi.apply_adi_cpml_2d(*f2, params_2d, cpml_2d, e, dt,
                                                   dx, dx)[:3],
            m2.eps_r[:, :, 0], m2.sigma[:, :, 0]),
        "apply_cpml_e": (
            lambda e, s: rfx.boundaries.cpml.apply_cpml_e(
                stc, cparams, cstate, cpml_grid,
                materials=mc._replace(eps_r=e, sigma=s))[0][:3],
            mc.eps_r, mc.sigma),
        "apply_cpml_e[inv_eps_r_update]": (
            lambda e, s: rfx.boundaries.cpml.apply_cpml_e(
                stc, cparams, cstate, cpml_grid, materials=mc._replace(sigma=s),
                inv_eps_r_update=(1.0 / e, 0.5 / e, 1.0 / (e + 1.0)))[0][:3],
            mc.eps_r, mc.sigma),
        "sheet_update_coeffs": (
            lambda e, s: rfx.materials.thin_conductor.sheet_update_coeffs(
                sheet, m._replace(eps_r=e, sigma=s), dt), m.eps_r, sheet_sigma),
        "edge_update_denominator": (
            lambda e, s: rfx.lumped.edge_update_denominator(
                m._replace(eps_r=e, sigma=s), (2, 3, 3), "ez", dt),
            m.eps_r, m.sigma),
        "init_upml": (
            lambda e, s: rfx.boundaries.upml.init_upml(
                upml_grid, mu._replace(eps_r=e, sigma=s))[:6], mu.eps_r, mu.sigma),
    }


def _jvp(name, dtype):
    fn, eps, sigma = _jvp_cases(dtype)[name]
    rng = np.random.default_rng(1)
    t_e = jnp.asarray(rng.standard_normal(eps.shape).astype(np.float32), dtype)
    t_s = jnp.asarray(rng.standard_normal(sigma.shape).astype(np.float32), dtype)
    # Jitted: the ADI rows' eager tridiagonal solves cost seconds op by op.
    t = jax.jit(lambda *p: jax.jvp(fn, p[:2], p[2:])[1])(eps, sigma, t_e, t_s)
    return [np.asarray(leaf, np.float64) for leaf in jax.tree.leaves(t)]


# The builders above that are jax.jit functions themselves.
_JITTED_BUILDERS = (rfx.core.yee.update_e_aniso_inv,)


@pytest.fixture
def si_float64_jvp(monkeypatch):
    """``name -> [leaves]``: the builder's JVP through its SI spelling, in
    float64 (x64 scoped to the call, never the module)."""
    from tests._x64_compat import enable_x64

    def clear():
        # The jitted builders' traces: none made with the helper may serve
        # the SI row, and none made with ``_si_only`` may outlive it. (A
        # process-wide jax.clear_caches() here cost ~30 s over the rows.)
        for fn in _JITTED_BUILDERS:
            fn.clear_cache()

    def run(name):
        with monkeypatch.context() as mp, enable_x64():
            for module in _IMPORTERS:
                mp.setattr(module, "si_value_eps_r_grad", _si_only)
            clear()
            try:
                return _jvp(name, jnp.float64)
            finally:
                clear()
    return run


@pytest.mark.parametrize("name", [
    "e_update_coeffs", "precompute_coeffs", "update_e_aniso", "update_e_aniso_inv",
    "adi_step_2d", "adi_step_3d", "apply_adi_cpml_2d", "apply_cpml_e",
    "apply_cpml_e[inv_eps_r_update]", "sheet_update_coeffs",
    "edge_update_denominator", "init_upml", *DISPERSION_BUILDERS])
def test_builder_tangent_is_the_si_derivative(name, si_float64_jvp):
    """float32 JVP (eps_r spelling) == float64 JVP of the SI spelling, to
    9 float32 ULP of each output's tangent peak."""
    got = _jvp(name, jnp.float32)
    want = si_float64_jvp(name)
    assert len(got) == len(want)
    assert any(np.any(w != 0.0) for w in want), f"{name}: the SI tangent is zero"
    for i, (g, w) in enumerate(zip(got, want)):
        assert np.all(np.isfinite(g)), f"{name}[{i}]: non-finite tangent"
        peak = float(np.max(np.abs(w)))
        if peak == 0.0:  # an output eps does not reach (an H coefficient)
            assert not np.any(g), f"{name}[{i}]: tangent where SI has none"
            continue
        ulp = float(np.max(np.abs(g - w))) / (peak * 2.0 ** -23)
        assert ulp <= JVP_TOL_ULP, f"{name}[{i}]: {ulp:.1f} float32 ULP of peak"

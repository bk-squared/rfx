"""Gate tests for `rfx.observables.jacobian_fwd` (issue #577).

`jacobian_fwd` is a thin PACKAGING layer over stock `jax.vmap(jax.jvp(...))`
-- forward mode already runs end-to-end through `sim.forward(eps_override=
...)` -> `rfx.observables.dft_field`/`field_energy` today with zero solver
changes. This module gates the wrapper's own contract: tangent-basis
construction (`tangents="identity"`), primal de-duplication, batched-vs-
sequential agreement, the complex-Jacobian (unconjugated) convention, the
primal-sharing structural claim, and the four documented fences.

G1-G7 map onto the design contract (`gh issue view 577`):
    G1 FD accuracy (material + geometry legs, many-output tensor leg)
    G2 conjugation guard
    G3 primal-sharing STRUCTURAL witness (load-bearing; deterministic)
    G4 value identity (bit-identical to a plain sim_fn(params) call)
    G5 batch_tangents=True vs False agreement
    G6 fence regressions
    G7 FALSIFIER (severed tape -> Jacobian exactly 0; witnessed red first)

FENCE TAXONOMY (G6) -- read before editing G6: `jacobian_fwd` is generic
over `sim_fn` (signature pinned to `sim_fn, params, tangents,
batch_tangents` by `scripts/check_api_reference.py`'s inventory) and never
sees `Simulation.forward()`'s own keyword arguments -- they live inside the
caller's closure. Two of the three documented fences are RUNTIME RAISES
that `jacobian_fwd` INHERITS for free (the underlying `forward()` call
raises before `jacobian_fwd`'s own machinery runs, so the raise just
propagates up through `jax.jvp`): non-uniform+distributed with a
registered DFT-plane probe (the #619 fence), and `n_warmup` on the
uniform lane (issue #626's own fence -- it used to be a measured SILENT
NO-OP; the uniform lane now raises instead of silently ignoring it). The
remaining one -- `checkpoint`/`checkpoint_segments` (measured EXACTLY
NEUTRAL under forward mode) -- has NO raise to inherit and cannot be
intercepted from outside the closure without breaking the pinned generic
signature, so it is a DOCUMENTED trap rather than a runtime check; G6
pins the docstring text for that one instead of a raise.

(A fourth candidate, `design_mask`, was removed with the parameter itself
in issue #625: the mask was measured to corrupt the gradient rather than
merely fail to save memory, so it was deleted from every public surface
including `forward()`. Passing `design_mask=` anywhere now raises a plain
`TypeError` for an unrecognised keyword -- the same as any misspelled
kwarg -- which is not a `jacobian_fwd`-specific fence worth documenting
here.)

FALSIFIER ARTIFACT: `test_g7_falsifier_severed_tape_reads_exactly_zero` below
PRINTS its captured red-then-green transcript (visible under `pytest -s`
or in CI logs) rather than writing to a fixed path -- no hard-coded
session/scratch path belongs in a committed public-repo test. Copy the
printed transcript into the PR body.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr as _ClosedJaxpr, Jaxpr as _Jaxpr

from rfx import Simulation
from rfx.observables import dft_field, field_energy, jacobian_fwd
from rfx.topology import density_to_material_fields

try:
    from jax import enable_x64
except ImportError:  # older JAX (< ~0.4.31) -- see tests/_x64_compat.py
    from tests._x64_compat import enable_x64


# ---------------------------------------------------------------------------
# Shared fixture -- same geometry family as tests/test_observables_dft_field.py
# (a proven, currently-green fixture), with the geometry-leg design pixel
# moved OFF the material-leg slab (x=24mm vs slab 12-18mm) so a JOINT
# {alpha, rho} params pytree can be gated in one jacobian_fwd call without
# the two legs' design regions overlapping (the #483-class confound the
# design contract warns about: keep design regions off each other's and the
# source's cells so FD and AD differentiate the same single channel).
# ---------------------------------------------------------------------------

_DX = 1.0e-3
_NX, _NY, _NZ = 30, 20, 20
_PLANE = "probe_plane"
_FREQS = jnp.asarray([5.0e9])
_SRC_POS_M = (8 * _DX, 10 * _DX, 10 * _DX)
_SLAB_LO_M = (12 * _DX, 0.0, 0.0)
_SLAB_HI_M = (18 * _DX, _NY * _DX, _NZ * _DX)
_PIXEL_POS_M = (24 * _DX, 10 * _DX, 10 * _DX)  # OFF the slab (12-18mm)
_N_STEPS = 160
# Material leg has a strong signal (rel_err 0.007-0.038% across this range).
_FD_H_VALUES_MATERIAL = (0.01, 0.02, 0.04)
# Geometry leg is a single-pixel perturbation with a much smaller signal
# (~1e-11 vs material's ~1e-9): MEASURED h=0.01 rel_err 7.4% is float32
# rounding-lattice noise on that tiny signal, not a real derivative
# mismatch -- h=0.02/0.04/0.08 all cluster tightly at 0.5-0.6% (the
# convergent-derivative signature vs. a coincidental-h false green), so
# the sweep starts at 0.02 for this leg specifically. This IS the h-drift
# check the design contract asks for: a real derivative holds steady as h
# grows past the noise floor, while a rounding-lattice coincidence would
# not. "float32" here is this gate's storage AND compute dtype (both
# default float32, unaffected by issue #630's storage-vs-arithmetic fix);
# it is not a claim that widening field STORAGE alone would raise this
# floor -- pre-#630 the Yee compute dtype was pinned to float32
# regardless of storage, so it wouldn't have.
_FD_H_VALUES_GEOMETRY = (0.02, 0.04, 0.08)


def _build_sim() -> Simulation:
    sim = Simulation(
        freq_max=8e9, domain=(_NX * _DX, _NY * _DX, _NZ * _DX), dx=_DX,
        boundary="cpml", cpml_layers=5,
    )
    sim.add_source(
        position=_SRC_POS_M, component="ez", amplitude_kind="current",
    )
    sim.add_dft_plane_probe(
        axis="x", coordinate=20 * _DX, component="ez",
        freqs=_FREQS, name=_PLANE,
    )
    return sim


def _grid_and_eps_base(sim: Simulation):
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    return grid, eps_base


def _slab_and_pixel_indices(sim: Simulation):
    grid, eps_base = _grid_and_eps_base(sim)
    lo = grid.position_to_index(_SLAB_LO_M)
    hi = grid.position_to_index(_SLAB_HI_M)
    slab = (slice(lo[0], hi[0] + 1), slice(None), slice(None))
    pix = grid.position_to_index(_PIXEL_POS_M)
    return grid, eps_base, slab, pix


def _make_eps(eps_base, slab, pix, alpha, rho, *, eps_bg=1.0, eps_fg=4.0,
              stop_gradient_tape: bool = False):
    eps = eps_base.at[slab].set(alpha)
    rho_arr = jnp.reshape(rho, (1, 1, 1))
    fields = density_to_material_fields(rho_arr, eps_bg, eps_fg)
    eps = eps.at[pix[0], pix[1], pix[2]].set(fields.eps[0, 0, 0])
    if stop_gradient_tape:
        eps = jax.lax.stop_gradient(eps)
    return eps


def _joint_objective(sim, functional, *, stop_gradient_tape: bool = False):
    """params = {'alpha': material slab scale, 'rho': geometry density
    pixel}, functional applied to the forward() result."""
    grid, eps_base, slab, pix = _slab_and_pixel_indices(sim)

    def objective(params):
        eps = _make_eps(
            eps_base, slab, pix, params["alpha"], params["rho"],
            stop_gradient_tape=stop_gradient_tape,
        )
        result = sim.forward(
            eps_override=eps, n_steps=_N_STEPS, checkpoint=False,
            skip_preflight=True,
        )
        return functional(result)

    return objective


def _material_only_objective(sim, functional, *, stop_gradient_tape: bool = False):
    """params = scalar alpha (material leg only) -- used for the
    many-output tensor leg (G1/G2), which needs a single tangent
    direction, not the joint {alpha, rho} pytree."""
    grid, eps_base, slab, _pix = _slab_and_pixel_indices(sim)

    def objective(alpha):
        eps = eps_base.at[slab].set(alpha)
        if stop_gradient_tape:
            eps = jax.lax.stop_gradient(eps)
        result = sim.forward(
            eps_override=eps, n_steps=_N_STEPS, checkpoint=False,
            skip_preflight=True,
        )
        return functional(result)

    return objective


def _central_fd_f64(objective, x0: float, h: float):
    """Central FD, comparison arithmetic in float64 (#527/#579 discipline):
    on this gate's production config, fields stay float32 STORAGE; only
    the FD/comparison math gains precision. "Fields stay float32" is a
    storage statement, not an arithmetic one (issue #630): before #630 the
    Yee update's compute dtype was hard-pinned to float32 regardless of
    storage, so it would not have helped to widen fields here; #630 makes
    compute dtype follow storage, which is a no-op at this gate's float32
    default and only takes effect under ``precision="float64"``."""
    with enable_x64():
        fp = objective(jnp.asarray(x0 + h, dtype=jnp.float32))
        fm = objective(jnp.asarray(x0 - h, dtype=jnp.float32))
        return (fp - fm) / (2.0 * h)


_ALPHA0 = 2.0
_RHO0 = 0.5
_PARAMS0 = {"alpha": jnp.asarray(_ALPHA0, dtype=jnp.float32),
            "rho": jnp.asarray(_RHO0, dtype=jnp.float32)}


# ---------------------------------------------------------------------------
# G1 -- FD accuracy: material leg, geometry leg (joint pytree, tangents=
# "identity", n_t=2), and the many-output tensor leg.
# ---------------------------------------------------------------------------

def test_g1_material_and_geometry_legs_ad_matches_fd():
    sim = _build_sim()
    functional = field_energy(_PLANE)
    objective = _joint_objective(sim, functional)

    val, jac = jacobian_fwd(objective, _PARAMS0)
    assert float(val) > 0.0, "field_energy did not couple to the source"
    # tangents='identity' flattens {'alpha','rho'} in tree_flatten order
    # (dict keys sorted): 'alpha' first, 'rho' second.
    g_ad_alpha = float(jac[0])
    g_ad_rho = float(jac[1])
    assert jac.shape == (2,)

    for h in _FD_H_VALUES_MATERIAL:
        g_fd_alpha = float(_central_fd_f64(
            lambda a: objective({"alpha": a, "rho": jnp.asarray(_RHO0, dtype=jnp.float32)}),
            _ALPHA0, h,
        ))
        rel = abs(g_ad_alpha - g_fd_alpha) / max(abs(g_fd_alpha), 1e-30)
        assert rel < 0.05, (
            f"material leg h={h}: AD {g_ad_alpha:.6e} vs FD {g_fd_alpha:.6e}, "
            f"rel_err {rel * 100:.2f}%"
        )

    for h in _FD_H_VALUES_GEOMETRY:
        g_fd_rho = float(_central_fd_f64(
            lambda r: objective({"alpha": jnp.asarray(_ALPHA0, dtype=jnp.float32), "rho": r}),
            _RHO0, h,
        ))
        rel = abs(g_ad_rho - g_fd_rho) / max(abs(g_fd_rho), 1e-30)
        assert rel < 0.05, (
            f"geometry leg h={h}: AD {g_ad_rho:.6e} vs FD {g_fd_rho:.6e}, "
            f"rel_err {rel * 100:.2f}%"
        )


def test_g1_many_output_tensor_leg_ad_matches_fd():
    """The (n_freqs, n1, n2) complex dft_field tensor, not a scalar
    reduction -- exercises jacobian_fwd on a many-output sim_fn."""
    sim = _build_sim()
    accessor = dft_field(_PLANE)
    objective = _material_only_objective(sim, accessor)

    alpha0 = jnp.asarray(_ALPHA0, dtype=jnp.float32)
    val, jac = jacobian_fwd(objective, alpha0)
    assert jac.shape == (1,) + val.shape

    h = 0.02
    with enable_x64():
        fp = objective(jnp.asarray(_ALPHA0 + h, dtype=jnp.float32))
        fm = objective(jnp.asarray(_ALPHA0 - h, dtype=jnp.float32))
        j_fd = (fp - fm) / (2.0 * h)

    num = float(jnp.linalg.norm((jac[0] - j_fd).ravel()))
    den = float(jnp.linalg.norm(j_fd.ravel()))
    rel = num / den
    assert rel < 0.05, f"tensor leg ||AD-FD||/||FD|| = {rel * 100:.3f}%"
    assert den > 0.0, "FD tensor Jacobian is exactly zero -- objective did not couple"


# ---------------------------------------------------------------------------
# G2 -- conjugation guard: AD matches the UNCONJUGATED FD, not conj(FD).
# ---------------------------------------------------------------------------

def test_g2_jacobian_is_unconjugated_dy_dx():
    sim = _build_sim()
    accessor = dft_field(_PLANE)
    objective = _material_only_objective(sim, accessor)

    alpha0 = jnp.asarray(_ALPHA0, dtype=jnp.float32)
    _val, jac = jacobian_fwd(objective, alpha0)
    j_ad = jac[0]

    h = 0.02
    with enable_x64():
        fp = objective(jnp.asarray(_ALPHA0 + h, dtype=jnp.float32))
        fm = objective(jnp.asarray(_ALPHA0 - h, dtype=jnp.float32))
        j_fd = (fp - fm) / (2.0 * h)

    rel_unconj = float(jnp.linalg.norm((j_ad - j_fd).ravel())) / float(jnp.linalg.norm(j_fd.ravel()))
    rel_conj = float(jnp.linalg.norm((j_ad - jnp.conj(j_fd)).ravel())) / float(jnp.linalg.norm(j_fd.ravel()))

    assert rel_unconj < 0.05, f"AD should match UNCONJUGATED FD: rel_err={rel_unconj * 100:.3f}%"
    assert rel_conj > 0.5, (
        f"AD unexpectedly matches conj(FD) (rel_err={rel_conj * 100:.3f}%) -- "
        "the discriminator that catches a silent conjugation bug did not fire"
    )


# ---------------------------------------------------------------------------
# G3 -- PRIMAL-SHARING STRUCTURAL WITNESS (load-bearing; deterministic).
#
# Walks the jaxpr of jax.jit(lambda p: jacobian_fwd(sim_fn, p)) and finds
# the top-level lax.scan (the FDTD time loop). If the primal sweep is
# genuinely shared across tangent directions, the scan's carry holds
# exactly ONE unbatched copy of the field state (byte count INDEPENDENT of
# n_t) plus n_t batched tangent copies -- not a wall-clock proxy.
# ---------------------------------------------------------------------------

class _jcore:
    ClosedJaxpr = _ClosedJaxpr
    Jaxpr = _Jaxpr


def _find_shallowest_scan(jaxpr):
    """Depth-first search for the SHALLOWEST `lax.scan` equation, recursing
    through nested (Closed)Jaxpr params -- `jax.jit(...)` wraps the traced
    program in a `pjit` equation whose own params carry the inner jaxpr, so
    a plain top-level scan of `jaxpr.eqns` misses it one level down.

    Returns ``(eqn, inner_jaxpr, total_scan_count)``. ``total_scan_count``
    matters: `batch_tangents=False` (independent sequential `jax.jvp` calls,
    the OPPOSITE of a shared primal sweep) produces N_T separate top-level
    scan equations, not one -- MEASURED. Examining only the shallowest
    scan's carry without also checking the total count would silently
    accept that regression (each independent scan happens to carry the
    same unbatched byte count as the elision case, so the primal-bytes-
    equal check alone does not discriminate "shared" from "not shared").
    """
    by_depth: list[tuple[int, object]] = []

    def walk(jx, depth):
        for eqn in jx.eqns:
            if eqn.primitive.name == "scan":
                by_depth.append((depth, eqn))
            for v in eqn.params.values():
                if isinstance(v, _jcore.ClosedJaxpr):
                    walk(v.jaxpr, depth + 1)
                elif isinstance(v, _jcore.Jaxpr):
                    walk(v, depth + 1)
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, _jcore.ClosedJaxpr):
                            walk(item.jaxpr, depth + 1)
                        elif isinstance(item, _jcore.Jaxpr):
                            walk(item, depth + 1)

    walk(jaxpr, 0)
    if not by_depth:
        raise AssertionError("no lax.scan found anywhere in jaxpr (FDTD time loop)")
    by_depth.sort(key=lambda pair: pair[0])
    eqn = by_depth[0][1]
    return eqn, eqn.params["jaxpr"].jaxpr, len(by_depth)


def _carry_grid_bytes(eqn, spatial: tuple[int, int, int], n_t: int):
    num_carry = eqn.params["num_carry"]
    num_consts = eqn.params["num_consts"]
    carry_avals = [
        v.aval for v in eqn.invars[num_consts:num_consts + num_carry]
    ]
    primal_bytes, batched_bytes = 0, 0
    for a in carry_avals:
        shp = tuple(getattr(a, "shape", ()))
        if len(shp) < 3 or shp[-3:] != spatial:
            continue
        n = 1
        for d in shp:
            n *= d
        nbytes = n * int(jnp.dtype(a.dtype).itemsize)
        if len(shp) >= 4 and shp[0] == n_t and n_t > 1:
            batched_bytes += nbytes
        else:
            primal_bytes += nbytes
    return primal_bytes, batched_bytes


def _make_design_sim_fn(sim, n_p: int):
    """params = tuple of n_p independent scalar leaves, each scaling one
    1-cell-thick x-slice -- gives tangents='identity' exactly n_t=n_p."""
    grid, eps_base = _grid_and_eps_base(sim)
    lo = grid.position_to_index(_SLAB_LO_M)[0]
    xi = [lo + i for i in range(n_p)]

    def sim_fn(params):
        eps = eps_base
        for i, a in enumerate(params):
            eps = eps.at[xi[i], :, :].set(a)
        result = sim.forward(
            eps_override=eps, n_steps=_N_STEPS, checkpoint=False,
            skip_preflight=True,
        )
        return dft_field(_PLANE)(result)

    return grid, sim_fn


def _measure_carry_bytes(n_t: int, *, batch_tangents: bool = True):
    sim = _build_sim()
    grid, sim_fn = _make_design_sim_fn(sim, n_t)
    params = tuple(jnp.asarray(1.0, dtype=jnp.float32) for _ in range(n_t))

    jitted = jax.jit(lambda p: jacobian_fwd(sim_fn, p, batch_tangents=batch_tangents))
    jaxpr = jax.make_jaxpr(jitted)(params).jaxpr
    eqn, _inner, scan_count = _find_shallowest_scan(jaxpr)
    primal_bytes, batched_bytes = _carry_grid_bytes(eqn, tuple(grid.shape), n_t)
    return primal_bytes, batched_bytes, scan_count


def test_g3_primal_carry_is_independent_of_n_t():
    """Deterministic structural witness (no wall-clock): build the
    compiled jaxpr at n_t=2 and n_t=4 independently within one test (not
    across separately-ordered parametrized runs, so this cannot silently
    depend on pytest execution order) and compare the scan's carry byte
    counts directly.

    n_t=1 is deliberately EXCLUDED, not just an arbitrary choice of two
    points: MEASURED, at n_t=1 `jax.vmap` elides the size-1 batch axis, so
    the tangent carry collapses to the SAME shape as the primal carry and
    this classifier's shape-based primal/batched split cannot tell them
    apart (both get bucketed as "primal", doubling the count vs n_t=2..4).
    n_t=2, 3, and 4 all show primal=945,624 B identically while batched
    bytes scale exactly linearly (945,624 x n_t) -- the decisive, elision-
    free evidence -- so n_t=2 and n_t=4 are the comparison pair.

    Two assertions below exist specifically because a degenerate case was
    caught in review: running this SAME classifier against
    `jacobian_fwd(..., batch_tangents=False)` -- genuinely n_t independent
    sequential jvp calls, the OPPOSITE of a shared primal sweep -- passed
    the original primal/batched-ratio checks, because `batched_2 ==
    batched_4 == 0` makes `batched_4 == approx(2 * batched_2)` degenerate
    to `0 == approx(0)`, and each independent scan happens to carry the
    same per-scan byte count as the elision case, so `primal_2 ==
    primal_4` ALSO holds without any real sharing. `batched_2 > 0` /
    `batched_4 > 0` and `scan_count == 1` close that hole: flipping the
    default to `batch_tangents=False` now fails both (MEASURED:
    batched_bytes=0 and scan_count=n_t there, not 1).
    """
    primal_2, batched_2, scan_count_2 = _measure_carry_bytes(2)
    primal_4, batched_4, scan_count_4 = _measure_carry_bytes(4)

    assert scan_count_2 == 1, (
        f"expected exactly ONE lax.scan (the FDTD time loop, shared across "
        f"tangent directions) at n_t=2, found {scan_count_2} -- primal "
        "sharing requires one shared scan, not n_t independent ones"
    )
    assert scan_count_4 == 1, (
        f"expected exactly ONE lax.scan at n_t=4, found {scan_count_4}"
    )
    assert primal_2 > 0 and primal_4 > 0, "no primal-shaped (unbatched) carry found"
    assert batched_2 > 0 and batched_4 > 0, (
        f"no batched (tangent) carry found (n_t=2 -> {batched_2}, n_t=4 -> "
        f"{batched_4}) -- without this check the ratio assertion below "
        "degenerates to 0 == approx(0) and would not catch batching being "
        "broken entirely"
    )
    assert primal_2 == primal_4, (
        f"primal carry bytes depend on n_t: n_t=2 -> {primal_2}, n_t=4 -> "
        f"{primal_4} -- the primal sweep is NOT shared across tangent directions"
    )
    assert batched_4 == pytest.approx(2 * batched_2, rel=0.05), (
        f"batched carry bytes did not scale ~linearly with n_t: "
        f"n_t=2 -> {batched_2}, n_t=4 -> {batched_4}"
    )


# ---------------------------------------------------------------------------
# G4 -- value identity: returned value bit-identical to a plain sim_fn(params).
# ---------------------------------------------------------------------------

def test_g4_value_bit_identical_to_plain_call():
    sim = _build_sim()
    objective = _joint_objective(sim, field_energy(_PLANE))

    plain = objective(_PARAMS0)
    val_batched, _jac = jacobian_fwd(objective, _PARAMS0, batch_tangents=True)
    val_seq, _jac2 = jacobian_fwd(objective, _PARAMS0, batch_tangents=False)

    assert val_batched == plain, f"batched value {val_batched} != plain {plain}"
    assert val_seq == plain, f"sequential value {val_seq} != plain {plain}"


def test_g4_value_bit_identical_many_output():
    sim = _build_sim()
    objective = _material_only_objective(sim, dft_field(_PLANE))
    alpha0 = jnp.asarray(_ALPHA0, dtype=jnp.float32)

    plain = objective(alpha0)
    val, _jac = jacobian_fwd(objective, alpha0)
    np.testing.assert_array_equal(np.asarray(val), np.asarray(plain))


# ---------------------------------------------------------------------------
# G5 -- batch_tangents=True vs False agreement (different XLA programs;
# float32 reassociation tolerance, not bit-identity).
# ---------------------------------------------------------------------------

def test_g5_batched_and_sequential_jacobians_agree():
    sim = _build_sim()
    objective = _joint_objective(sim, field_energy(_PLANE))

    _val_b, jac_batched = jacobian_fwd(objective, _PARAMS0, batch_tangents=True)
    _val_s, jac_seq = jacobian_fwd(objective, _PARAMS0, batch_tangents=False)

    np.testing.assert_allclose(
        np.asarray(jac_batched), np.asarray(jac_seq), rtol=1e-4, atol=1e-12,
    )


def test_g5_batched_and_sequential_agree_many_output():
    sim = _build_sim()
    objective = _material_only_objective(sim, dft_field(_PLANE))
    alpha0 = jnp.asarray(_ALPHA0, dtype=jnp.float32)

    _val_b, jac_batched = jacobian_fwd(objective, alpha0, batch_tangents=True)
    _val_s, jac_seq = jacobian_fwd(objective, alpha0, batch_tangents=False)

    np.testing.assert_allclose(
        np.asarray(jac_batched), np.asarray(jac_seq), rtol=1e-4, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# G6 -- fence regressions. See the module docstring's FENCE TAXONOMY for
# why G6a is a runtime-raise gate and G6b/G6c are docstring-content gates.
# (design_mask's former G6b fence was removed with the parameter itself,
# issue #625 -- see the module docstring's trailing note.)
# ---------------------------------------------------------------------------

def test_g6a_uniform_distributed_raises_a_different_earlier_check():
    """A uniform sim + distributed=True DOES raise NotImplementedError, but
    via forward()'s general "distributed=True is NU-only" dispatch guard
    (_execute.py's "only for non-uniform meshes" check) -- NOT the
    DFT-plane-specific fence jacobian_fwd's own docstring names as fence
    #1 (the #619 fence, which needs a non-uniform mesh to even reach). This
    test pins that this EARLIER, more general check also propagates
    through jax.jvp/jacobian_fwd unchanged; the NAMED fence itself is
    exercised separately below by
    test_g6a_distributed_nu_dft_plane_fence_raises."""
    sim = _build_sim()

    def sim_fn(alpha):
        eps = jnp.ones(sim._build_grid().shape, dtype=jnp.float32) * alpha
        result = sim.forward(
            eps_override=eps, n_steps=2, distributed=True, skip_preflight=True,
        )
        return dft_field(_PLANE)(result)

    with pytest.raises(NotImplementedError, match="non-uniform meshes"):
        jacobian_fwd(sim_fn, jnp.asarray(1.0, dtype=jnp.float32))


def test_g6a_distributed_nu_dft_plane_fence_raises():
    """The fence jacobian_fwd's own docstring actually names as fence #1
    (issue #619, inherited from forward()): non-uniform + distributed=True
    WITH a registered DFT-plane probe raises NotImplementedError naming
    add_dft_plane_probe, and that raise propagates through
    jax.jvp/jacobian_fwd unchanged -- no new code in jacobian_fwd needed.
    Same NU-sim construction as
    tests/test_observables_dft_field.py::test_forward_distributed_nu_dft_planes_fence_raises."""
    nx, ny, nz = 16, 12, 12
    dx = 1e-3
    sim = Simulation(
        freq_max=8e9, domain=(nx * dx, ny * dx, nz * dx), dx=dx,
        boundary="cpml", cpml_layers=4,
    )
    sim._dx_profile = np.full(nx, dx)
    sim._dy_profile = np.full(ny, dx)
    sim._dz_profile = np.full(nz, dx)
    sim.add_source(
        position=(4 * dx, 6 * dx, 6 * dx), component="ez", amplitude_kind="current",
    )
    sim.add_dft_plane_probe(
        axis="x", coordinate=10 * dx, component="ez", freqs=_FREQS,
    )

    def sim_fn(alpha):
        # alpha is unused: the fence fires during forward()'s Python-level
        # dispatch, before any traced computation touches eps/alpha at all.
        return sim.forward(distributed=True, n_steps=2, skip_preflight=True)

    with pytest.raises(NotImplementedError, match="add_dft_plane_probe"):
        jacobian_fwd(sim_fn, jnp.asarray(1.0, dtype=jnp.float32))


def test_g6_design_mask_kwarg_raises_type_error_not_a_fence():
    """issue #625: design_mask was removed from every public surface
    (measured to corrupt the gradient, not merely fail to save memory),
    so it is no longer a jacobian_fwd-specific fence -- passing it now
    raises a plain TypeError for an unrecognised forward() keyword, the
    same as any misspelled kwarg would. Pinned here (rather than deleted
    outright) so a future reintroduction of the parameter is a deliberate
    act, matching the removal's own regression-test convention."""
    sim = _build_sim()
    grid = sim._build_grid()
    mask = jnp.zeros(grid.shape, dtype=bool).at[15, 10, 10].set(True)

    def sim_fn(alpha):
        eps = jnp.ones(grid.shape, dtype=jnp.float32) * alpha
        result = sim.forward(
            eps_override=eps, n_steps=2, design_mask=mask, skip_preflight=True,
        )
        return dft_field(_PLANE)(result)

    with pytest.raises(TypeError, match="design_mask"):
        jacobian_fwd(sim_fn, jnp.asarray(1.0, dtype=jnp.float32))


def test_g6b_n_warmup_fence_raises():
    """Inherited from forward()'s own uniform-lane n_warmup fence (issue
    #626): a nonzero n_warmup on the uniform lane raises
    NotImplementedError regardless of AD mode. (Prior to the #626 fix this
    was a measured SILENT NO-OP, not a raise -- see
    tests/test_n_warmup.py::test_uniform_forward_rejects_n_warmup for the
    forward()-level pin and docs/agent-memory/rfx-known-issues.md for the
    measured bit-identical-gradient evidence that motivated the fence.)"""
    sim = _build_sim()
    grid = sim._build_grid()

    def sim_fn(alpha):
        eps = jnp.ones(grid.shape, dtype=jnp.float32) * alpha
        result = sim.forward(
            eps_override=eps, n_steps=2, n_warmup=1, skip_preflight=True,
        )
        return dft_field(_PLANE)(result)

    with pytest.raises(NotImplementedError, match="n_warmup"):
        jacobian_fwd(sim_fn, jnp.asarray(1.0, dtype=jnp.float32))


def test_g6c_checkpoint_knobs_are_documented_inert_not_a_raise():
    """No forward()-level raise exists to inherit (checkpoint/
    checkpoint_segments are measured EXACTLY NEUTRAL under forward mode) --
    pin the docstring warning instead."""
    doc = jacobian_fwd.__doc__
    assert "checkpoint" in doc
    assert "checkpoint_segments" in doc
    assert "EXACTLY NEUTRAL" in doc
    assert "reverse-mode knobs" in doc


# ---------------------------------------------------------------------------
# G7 -- FALSIFIER: sever the tape with stop_gradient on eps_override ->
# Jacobian must go EXACTLY 0, and (witnessed separately, transcript
# printed for the PR body -- see the module docstring) G1's own assertion
# form must go RED against the severed objective before this gate is
# trusted.
# ---------------------------------------------------------------------------

def test_g7_falsifier_severed_tape_reads_exactly_zero():
    sim = _build_sim()
    functional = field_energy(_PLANE)
    severed = _joint_objective(sim, functional, stop_gradient_tape=True)
    unsevered = _joint_objective(sim, functional, stop_gradient_tape=False)

    val_severed, jac_severed = jacobian_fwd(severed, _PARAMS0)
    val_unsevered, jac_unsevered = jacobian_fwd(unsevered, _PARAMS0)

    # Values are unaffected by stop_gradient (forward physics unchanged).
    assert val_severed == val_unsevered

    assert jac_severed[0] == 0.0, f"severed material tangent not exactly 0: {jac_severed[0]!r}"
    assert jac_severed[1] == 0.0, f"severed geometry tangent not exactly 0: {jac_severed[1]!r}"
    assert jac_unsevered[0] != 0.0, "unsevered material tangent is exactly 0 -- objective did not couple"
    assert jac_unsevered[1] != 0.0, "unsevered geometry tangent is exactly 0 -- objective did not couple"

    # Witness that G1's own tolerance check would go RED against the
    # severed objective (AD=0 exactly, FD finite -> the gate the falsifier
    # is meant to catch really would fire). Capture the transcript.
    h = 0.02
    g_fd_alpha = float(_central_fd_f64(
        lambda a: severed({"alpha": a, "rho": jnp.asarray(_RHO0, dtype=jnp.float32)}),
        _ALPHA0, h,
    ))
    rel = abs(float(jac_severed[0]) - g_fd_alpha) / max(abs(g_fd_alpha), 1e-30)

    transcript = (
        "#577 G7 falsifier transcript\n"
        "============================\n"
        f"severed AD tangent (alpha) = {float(jac_severed[0])!r} (must be exactly 0.0)\n"
        f"severed FD reference (alpha, h={h}) = {g_fd_alpha!r} (finite, nonzero)\n"
        f"rel_err AD-vs-FD on the SEVERED objective = {rel * 100:.2f}%\n"
        f"G1 tolerance is 5% -- this rel_err is >= 100% (0 vs finite), so G1's\n"
        f"own assertion form goes RED against the severed objective, confirming\n"
        f"the gate is witnessed before being trusted green on the unsevered one.\n\n"
        f"unsevered AD tangent (alpha) = {float(jac_unsevered[0])!r} (finite, matches G1)\n"
        f"unsevered geometry tangent (rho) = {float(jac_unsevered[1])!r} (finite)\n"
        f"severed geometry tangent (rho) = {float(jac_severed[1])!r} (must be exactly 0.0)\n"
    )
    assert rel >= 1.0, "falsifier did not actually diverge from the FD reference"
    # Print (not write to a fixed path -- no hard-coded session/scratch path
    # belongs in a committed public-repo test) so `pytest -s` / CI logs carry
    # the transcript for a PR body excerpt; the assertions above are the
    # real gate.
    print(transcript)


# ---------------------------------------------------------------------------
# G8: batch_tangents=False's own out_axes=(None, 0) guard (arc-audit
# follow-up item 4b). The batched path inherits a fail-loud check for free
# from jax.vmap(..., out_axes=(None, 0)) -- see the API TRAP note in
# jacobian_fwd's docstring. The sequential path has no vmap of its own, so
# without an explicit guard a sim_fn that violates the "primal independent
# of tangent" invariant would silently return one arbitrary tangent row's
# primal instead of raising. `leaky`/`leaky_jvp` below is a deliberately
# broken custom_jvp rule -- jax.jvp does not itself enforce primal purity,
# so this is the only way to construct a genuine violation: the primal
# output is defined to depend on the tangent argument.
# ---------------------------------------------------------------------------

@jax.custom_jvp
def _leaky(x):
    return x


@_leaky.defjvp
def _leaky_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return x + jnp.sum(t), t  # BUG (deliberate): primal leaks the tangent


def _leaky_sim_fn(params):
    return _leaky(params)


def _pure_sim_fn(params):
    return jnp.sum(params ** 2)


def test_g8_sequential_guard_raises_on_tangent_dependent_primal():
    """The sequential path must raise -- the SAME class of error the
    batched path raises for free -- not silently return a wrong primal.
    Guard 1 (jax.eval_shape) catches THIS construction: `_leaky`'s primal/
    tangent coupling is visible to vmap's batching interpreter even in a
    single abstract trace, so it fires before the real sequential loop
    (and before guard 2, which never gets a chance to run) -- the
    "out_axes" message below IS guard 1 firing, not a fallback to the
    default Python TypeError. See
    test_g8_sequential_guard_catches_cross_call_impurity_guard1_cannot_see
    below for the complementary case guard 1 structurally cannot detect."""
    params = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    tangents = jnp.eye(3, dtype=jnp.float32)  # explicit escape hatch: identity
    # requires scalar leaves, and params here is one flat (3,) leaf.

    with pytest.raises(ValueError, match="out_axes"):
        jacobian_fwd(_leaky_sim_fn, params, tangents=tangents, batch_tangents=False)

    # Confirm the batched path independently raises too (both paths agree
    # this sim_fn is unsafe -- neither one is the "safe" escape hatch).
    with pytest.raises(ValueError, match="out_axes"):
        jacobian_fwd(_leaky_sim_fn, params, tangents=tangents, batch_tangents=True)


def test_g8_sequential_guard_catches_cross_call_impurity_guard1_cannot_see():
    """Guard 1 (jax.eval_shape) traces sim_fn's body exactly ONCE (vmap
    batches rather than looping), so a side effect that only manifests
    ACROSS repeated independent Python-level calls -- not visible as a
    structural batching dependency inside one trace -- is invisible to
    it. The real sequential loop (`batch_tangents=False`, no jit) DOES
    call sim_fn n_t separate times in eager Python, so it genuinely can
    observe such a thing. This is guard 2's reason to exist: an exact
    pairwise comparison of the n_t actually-computed primal values."""
    call_count = {"n": 0}

    def side_effecting_sim_fn(params):
        call_count["n"] += 1
        # A real (if contrived) impurity: the primal depends on how many
        # times this Python function has been called so far, which a
        # single abstract trace cannot see, but n_t independent eager
        # jax.jvp calls can.
        return jnp.sum(params ** 2) + float(call_count["n"])

    params = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    tangents = jnp.eye(3, dtype=jnp.float32)

    with pytest.raises(ValueError, match="depends on which tangent"):
        jacobian_fwd(side_effecting_sim_fn, params, tangents=tangents, batch_tangents=False)


def test_g8_sequential_guard_silent_for_well_behaved_sim_fn():
    """No false positives: a normal, pure sim_fn must not trip the guard
    on either path, and both paths must still agree on the actual value."""
    params = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    tangents = jnp.eye(3, dtype=jnp.float32)

    value_seq, jac_seq = jacobian_fwd(_pure_sim_fn, params, tangents=tangents, batch_tangents=False)
    value_batch, jac_batch = jacobian_fwd(_pure_sim_fn, params, tangents=tangents, batch_tangents=True)

    assert value_seq == value_batch == jnp.sum(params ** 2)
    np.testing.assert_allclose(np.asarray(jac_seq), np.asarray(jac_batch))


def test_g8_sequential_guard_trace_time_check_survives_outer_jit():
    """Guard 1 (jax.eval_shape) must fire even when jacobian_fwd itself is
    called from inside an outer jax.jit -- exactly how
    scripts/benchmark_jacobian_fwd.py and this file's own compiled-jaxpr
    measurements (G3) invoke the sequential path. Guard 2 (the eager
    numeric cross-check) cannot run under trace -- params are tracers --
    so this specifically isolates guard 1."""
    params = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    tangents = jnp.eye(3, dtype=jnp.float32)

    @jax.jit
    def outer(p):
        value, jacobian = jacobian_fwd(_leaky_sim_fn, p, tangents=tangents, batch_tangents=False)
        return value

    with pytest.raises(ValueError, match="out_axes"):
        outer(params)

    # And a well-behaved sim_fn must still compile and run fine under jit.
    @jax.jit
    def outer_ok(p):
        value, jacobian = jacobian_fwd(_pure_sim_fn, p, tangents=tangents, batch_tangents=False)
        return value, jacobian

    value, jacobian = outer_ok(params)
    assert float(value) == float(jnp.sum(params ** 2))


# ---------------------------------------------------------------------------
# G8 follow-up (arc-audit verification round, items B3/B5 + "guard 1 too
# strict"): three defects found in the G8 guards themselves.
# ---------------------------------------------------------------------------

def test_g8_guard2_nan_divergence_not_misdiagnosed_as_tangent_dependence():
    """B3: jnp.array_equal(nan, nan) is False by default (IEEE-754), so a
    PURE, deterministic sim_fn whose computation diverged (returns the
    SAME nan on every tangent row -- divergence is a property of params,
    not of which tangent direction is evaluated) used to trip guard 2's
    mismatch check and get told its primal "depends on which tangent
    direction is being evaluated" -- wrong diagnosis. batch_tangents=True
    just returns the nan silently (measured); the sequential path must
    match that, not raise."""
    def nan_sim_fn(p):
        return jnp.nan + 0.0 * jnp.sum(p)  # same nan regardless of p or tangent

    params = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    tangents = jnp.eye(3, dtype=jnp.float32)

    value_seq, jac_seq = jacobian_fwd(nan_sim_fn, params, tangents=tangents, batch_tangents=False)
    value_batch, jac_batch = jacobian_fwd(nan_sim_fn, params, tangents=tangents, batch_tangents=True)
    assert jnp.isnan(value_seq) and jnp.isnan(value_batch)
    np.testing.assert_array_equal(np.asarray(jac_seq), np.asarray(jac_batch))


def test_g8_guard2_checks_all_params_leaves_for_tracer_ness():
    """B5: guard 2 used to test only params_leaves[0] for tracer-ness.
    A MIXED params pytree -- one leaf a jit-traced argument, one leaf a
    plain Python/closure-captured CONCRETE value (not itself passed
    through jit's argument list, so it never becomes a tracer) -- used to
    crash guard 2's bool(jnp.array_equal(...)) with
    TracerBoolConversionError whenever the traced leaf sorted before the
    concrete one in pytree-flatten order, since checking only leaf 0
    would wrongly conclude "not tracing" and try to concretely compare a
    tracer. Pinned both orderings (traced-leaf-first and
    concrete-leaf-first) so this cannot silently regress by luck of leaf
    order."""
    def sim_fn(params_tuple):
        a, b = params_tuple
        return jnp.sum(a ** 2) + jnp.sum(b ** 2)

    tangents = (jnp.eye(2, dtype=jnp.float32), jnp.zeros((2, 1), dtype=jnp.float32))
    b0_concrete = jnp.asarray([3.0], dtype=jnp.float32)

    @jax.jit
    def outer_traced_first(a):
        # a: jit-traced; b0_concrete: closure-captured, stays concrete.
        value, jacobian = jacobian_fwd(
            sim_fn, (a, b0_concrete), tangents=tangents, batch_tangents=False,
        )
        return value

    a0_concrete = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    @jax.jit
    def outer_concrete_first(b):
        # a0_concrete: closure-captured, stays concrete; b: jit-traced.
        value, jacobian = jacobian_fwd(
            sim_fn, (a0_concrete, b), tangents=tangents, batch_tangents=False,
        )
        return value

    out1 = outer_traced_first(jnp.asarray([1.0, 2.0], dtype=jnp.float32))
    out2 = outer_concrete_first(jnp.asarray([3.0], dtype=jnp.float32))
    expected = float(jnp.sum(jnp.asarray([1.0, 2.0]) ** 2) + jnp.sum(jnp.asarray([3.0]) ** 2))
    assert float(out1) == expected
    assert float(out2) == expected


def test_g8_guard1_data_dependent_branch_on_params_not_misdiagnosed():
    """Guard 1 too strict: jax.eval_shape abstracts EVERY value inside
    the traced function, including `params` -- which stays CONCRETE
    under a real jax.vmap(..., out_axes=(None, 0)) call (params is not
    the vmapped axis) or under plain jax.jvp. A sim_fn with a
    data-dependent Python branch on `params` (not the tangent) is
    therefore fine under plain jax.jvp AND the batched path, but used to
    crash guard 1 with TracerBoolConversionError purely because
    eval_shape's abstract mode cannot decide the branch -- an eval_shape
    LIMITATION, not a real out_axes violation. Guard 1 now catches
    ConcretizationTypeError and falls through to guard 2 instead of
    propagating it, so the sequential path is no longer STRICTER than
    every other path for this class of sim_fn."""
    def branchy_sim_fn(p):
        return jnp.where(jnp.sum(p) > 0, jnp.sum(p ** 2), -jnp.sum(p ** 2))

    params = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    tangents = jnp.eye(3, dtype=jnp.float32)

    # A genuine Python `if` (not jnp.where) is the sharper reproduction
    # of the reported failure -- jnp.where alone is traceable, but an
    # `if` on a traced boolean is what raises TracerBoolConversionError
    # under eval_shape's full abstraction.
    def branchy_sim_fn_python_if(p):
        if jnp.sum(p) > 0:
            return jnp.sum(p ** 2)
        return -jnp.sum(p ** 2)

    for fn in (branchy_sim_fn, branchy_sim_fn_python_if):
        value_seq, jac_seq = jacobian_fwd(fn, params, tangents=tangents, batch_tangents=False)
        value_batch, jac_batch = jacobian_fwd(fn, params, tangents=tangents, batch_tangents=True)
        assert float(value_seq) == float(value_batch) == 14.0
        np.testing.assert_allclose(np.asarray(jac_seq), np.asarray(jac_batch))

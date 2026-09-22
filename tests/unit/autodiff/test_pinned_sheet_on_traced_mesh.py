"""A patch edge as a design variable: the conductor is pinned to node lines.

The length an antenna engineer moves is the distance between the patch's two
metal edges. Here the edges stay ON node lines and the CELLS between them
are stretched, so the metal gets longer without any metal inside a cell and
without anything to snap. What makes that differentiable is the
DECLARATION: a sheet drawn in metres has to be re-read on whatever node line
the grid happens to have — which on a traced mesh cannot be read at all, and
on a concrete deformed mesh names different node lines than it did before —
while a sheet drawn by NODE INDEX names the same nodes on every mesh and
lets the cells carry the length.

So there are two semantics and this file pins both. A metric sheet keeps its
metres and stays REFUSED on a traced mesh. A node-pinned sheet
(``Simulation.add_pinned_sheet``) keeps its nodes, is accepted on the uniform
and the non-uniform lane, traced or not, and gives one footprint for every
deformation — which is what makes ``jax.grad`` of a probe observable with
respect to the deformation a derivative of the physics rather than of a
raster decision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.pec import SheetSpec, realized_pec_edge_masks
from rfx.geometry import Box
from rfx.materials.thin_conductor import (
    PinnedSheet, pinned_sheet_realized, pinned_sheet_spec)
from rfx.nonuniform import make_nonuniform_grid

DX = 0.5e-3
N_X, N_Y, N_Z = 24, 16, 20
CPML = 4
I_LO, I_HI = 8, 16          # the patch's two x node lines (interior indices)
J_LO, J_HI = 5, 11          # the patch's two y node lines
N_W = 4                     # cells stretched inside / shrunk outside the edge
K_GND, K_PATCH = 6, 12      # ground and patch node planes on z
K_MID = 9                   # a node plane inside the substrate
Z_GND, Z_PATCH = K_GND * DX, K_PATCH * DX


def _is_traced(v):
    return isinstance(v, jnp.ndarray) or hasattr(v, "aval")


def _stretch(delta, n, lo, hi, sign=+1.0):
    """Grow cells ``[lo, hi)`` by ``sign*delta/(hi-lo)``, shrink the same
    number just outside, so the domain keeps its length and the first and
    last cells keep the boundary size."""
    nw = hi - lo
    if _is_traced(delta):
        d = jnp.full((n,), DX)
        d = d.at[lo:hi].add(sign * delta / nw)
        d = d.at[hi:hi + nw].add(-sign * delta / nw)
        return d.astype(jnp.float32)
    d = np.full((n,), DX, dtype=np.float64)
    d[lo:hi] += sign * float(delta) / nw
    d[hi:hi + nw] -= sign * float(delta) / nw
    return d


def _prof_x(delta):
    return _stretch(delta, N_X, I_HI - N_W, I_HI)


def _prof_x_low(delta):
    """The LOW-side edge moves: cells inside the patch shrink, the ones
    below grow back, so node ``I_HI`` does not move."""
    if _is_traced(delta):
        d = jnp.full((N_X,), DX)
        d = d.at[I_LO - N_W:I_LO].add(delta / N_W)
        d = d.at[I_LO:I_LO + N_W].add(-delta / N_W)
        return d.astype(jnp.float32)
    d = np.full((N_X,), DX, dtype=np.float64)
    d[I_LO - N_W:I_LO] += float(delta) / N_W
    d[I_LO:I_LO + N_W] -= float(delta) / N_W
    return d


def _prof_y(delta):
    return _stretch(delta, N_Y, J_HI - N_W, J_HI)


def _grid(dx_profile=None, dy_profile=None, dz_profile=None, **kw):
    return make_nonuniform_grid(
        domain_xy=(N_X * DX, N_Y * DX),
        dz_profile=(np.full((N_Z,), DX, dtype=np.float64)
                    if dz_profile is None else dz_profile),
        dx=DX, cpml_layers=CPML,
        dx_profile=dx_profile, dy_profile=dy_profile, **kw)


PATCH = PinnedSheet(normal_axis=2, plane_index=K_PATCH,
                    i_range=(I_LO, I_HI), j_range=(J_LO, J_HI), name="patch")


def _spec_under_trace(make_grid, profile_fn, delta, sheet=PATCH):
    """The spec the traced lane builds, captured out of a grad trace."""
    captured = {}

    def _go(d):
        captured["spec"] = pinned_sheet_spec(make_grid(d), sheet)
        return jnp.sum(d)

    jax.grad(_go)(jnp.asarray(profile_fn(jnp.float32(delta)),
                              dtype=jnp.float32))
    return captured["spec"]


# --------------------------------------------------------------------------
# 1. the metric semantic: metres stay metres, and a traced mesh still refuses
# --------------------------------------------------------------------------

def test_a_metric_sheet_on_a_traced_mesh_is_still_refused():
    """A sheet drawn in metres has corners that must be located on the node
    line; a traced node line has no host position, so it is refused rather
    than placed somewhere."""
    def _loss(delta):
        sim = Simulation(freq_max=40e9,
                         domain=(N_X * DX, N_Y * DX, N_Z * DX), dx=DX,
                         dx_profile=_prof_x(delta), cpml_layers=CPML,
                         boundary="cpml")
        sim.add_thin_conductor(Box((I_LO * DX, J_LO * DX, Z_PATCH),
                                   (I_HI * DX, J_HI * DX, Z_PATCH)))
        sim.add_source((3 * DX, 8 * DX, K_MID * DX), "ez",
                       amplitude_kind="current")
        sim.add_probe((20 * DX, 8 * DX, K_MID * DX), "ez")
        return jnp.sum(sim.forward(n_steps=4, skip_preflight=True)
                       .time_series ** 2)

    with pytest.raises(ValueError, match="a sheet's plane is a static integer"):
        jax.grad(_loss)(jnp.float32(0.5 * DX))


# --------------------------------------------------------------------------
# 2. the pinned semantic: one declaration, two routes, every deformation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0.0, 0.25, 0.5, 0.9, 1.0, 1.25, 1.5])
def test_one_declaration_two_routes_identical_footprint(k):
    """The same pinned declaration realizes the same nodes on the concrete
    and the traced route, at every deformation — including past the point
    where a metric declaration's +x corner walks off its node line.

    The metal's LENGTH does move: it is the sum of the cells between the two
    pinned node lines, read back here from the realized profile.
    """
    delta = k * DX
    concrete = pinned_sheet_spec(_grid(dx_profile=_prof_x(delta)), PATCH)
    traced = _spec_under_trace(lambda d: _grid(dx_profile=d), _prof_x, delta)

    fp_c = np.asarray(concrete.footprint, dtype=bool)
    fp_t = np.asarray(traced.footprint, dtype=bool)
    assert int(traced.plane) == int(concrete.plane) == K_PATCH + CPML
    assert np.array_equal(fp_t, fp_c)

    xs = np.flatnonzero(fp_t.any(axis=(1, 2)))
    ys = np.flatnonzero(fp_t.any(axis=(0, 2)))
    assert (int(xs[0]), int(xs[-1])) == (I_LO + CPML, I_HI + CPML)
    assert (int(ys[0]), int(ys[-1])) == (J_LO + CPML, J_HI + CPML)

    # Assert REALIZED, not declared: the length the cells actually add up to.
    realized = pinned_sheet_realized(_grid(dx_profile=_prof_x(delta)), PATCH)
    assert realized["plane_m"] == pytest.approx(Z_PATCH)
    assert realized["x_span_m"] == pytest.approx((I_HI - I_LO) * DX + delta)
    assert realized["y_span_m"] == pytest.approx((J_HI - J_LO) * DX)


def test_pinned_footprint_survives_a_traced_z_profile():
    """A traced z mesh moves the sheet's own plane. The plane INDEX does not
    move, and the plane's physical height is whatever the z cells below it
    add up to — which is the quantity a z-deformation design variable moves.
    """
    def _dz(delta):
        return _stretch(delta, N_Z, K_PATCH - N_W, K_PATCH)

    delta = 0.5 * DX
    concrete = pinned_sheet_spec(_grid(dz_profile=_dz(delta)), PATCH)
    traced = _spec_under_trace(lambda d: _grid(dz_profile=d), _dz, delta)
    assert int(traced.plane) == int(concrete.plane) == K_PATCH + CPML
    assert np.array_equal(np.asarray(traced.footprint, dtype=bool),
                          np.asarray(concrete.footprint, dtype=bool))
    realized = pinned_sheet_realized(_grid(dz_profile=_dz(delta)), PATCH)
    assert realized["plane_m"] == pytest.approx(Z_PATCH + delta)


def test_pinned_footprint_survives_a_joint_xy_traced_mesh():
    """Both in-plane axes traced at once."""
    delta = 0.5 * DX

    def _build(d):
        return _grid(dx_profile=_prof_x(d), dy_profile=_prof_y(d))

    concrete = pinned_sheet_spec(_build(delta), PATCH)
    captured = {}

    def _go(d):
        captured["spec"] = pinned_sheet_spec(
            _grid(dx_profile=_prof_x(d), dy_profile=_prof_y(d)), PATCH)
        return jnp.asarray(d, dtype=jnp.float32)

    jax.grad(_go)(jnp.float32(delta))
    assert np.array_equal(
        np.asarray(captured["spec"].footprint, dtype=bool),
        np.asarray(concrete.footprint, dtype=bool))
    realized = pinned_sheet_realized(_build(delta), PATCH)
    assert realized["x_span_m"] == pytest.approx((I_HI - I_LO) * DX + delta)
    assert realized["y_span_m"] == pytest.approx((J_HI - J_LO) * DX + delta)


def test_uniform_and_nonuniform_lanes_realize_the_same_pinned_nodes():
    """The declaration is mesh-independent, so the two lanes cannot disagree."""
    sim = Simulation(freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
                     dx=DX, cpml_layers=CPML, boundary="cpml")
    uniform = pinned_sheet_spec(sim._build_grid(), PATCH)
    nu = pinned_sheet_spec(_grid(), PATCH)
    assert int(uniform.plane) == int(nu.plane)
    assert np.array_equal(np.asarray(uniform.footprint, dtype=bool),
                          np.asarray(nu.footprint, dtype=bool))


# --------------------------------------------------------------------------
# 3. what the declaration refuses, and where the pad goes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(i_range=(I_LO, I_LO), j_range=(J_LO, J_HI)),
    dict(i_range=(I_LO, I_HI), j_range=(J_LO, J_LO)),
])
def test_a_single_node_line_carries_no_current_and_is_refused(bad):
    """One node line has no E edge between two adjacent nodes, so the metal
    would carry no current and vanish silently (#369 class)."""
    sheet = PinnedSheet(normal_axis=2, plane_index=K_PATCH, name="strip", **bad)
    with pytest.raises(ValueError, match="single node line"):
        pinned_sheet_spec(_grid(), sheet)


def test_a_range_outside_the_grid_is_refused():
    sheet = PinnedSheet(normal_axis=2, plane_index=K_PATCH,
                        i_range=(I_LO, N_X + CPML + 5), j_range=(J_LO, J_HI),
                        name="patch")
    with pytest.raises(ValueError, match="outside the grid"):
        pinned_sheet_spec(_grid(), sheet)


def test_interior_indices_carry_each_axis_own_leading_pad():
    """Indices are INTERIOR: interior node ``i`` is array index ``i +
    pad_lo`` on THAT axis. A PEC face gives that axis pad 0 while the others
    keep the absorber, so a mapping that used one pad for every axis, or the
    trailing pad instead of the leading one, lands somewhere else.
    """
    grid = _grid(pec_faces={"x_lo"})
    assert (grid.pad_x_lo, grid.pad_x_hi) == (0, CPML)
    assert (grid.pad_y_lo, grid.pad_z_lo) == (CPML, CPML)
    spec = pinned_sheet_spec(grid, PATCH)
    fp = np.asarray(spec.footprint, dtype=bool)
    xs = np.flatnonzero(fp.any(axis=(1, 2)))
    ys = np.flatnonzero(fp.any(axis=(0, 2)))
    assert (int(xs[0]), int(xs[-1])) == (I_LO, I_HI)                 # pad 0
    assert (int(ys[0]), int(ys[-1])) == (J_LO + CPML, J_HI + CPML)   # pad 4
    assert int(spec.plane) == K_PATCH + CPML
    realized = pinned_sheet_realized(grid, PATCH)
    assert realized["x_edges_m"] == pytest.approx((I_LO * DX, I_HI * DX))
    assert realized["y_edges_m"] == pytest.approx((J_LO * DX, J_HI * DX))
    assert realized["plane_m"] == pytest.approx(Z_PATCH)


def test_from_node_ranges_refuses_a_plane_outside_the_grid():
    with pytest.raises(ValueError, match="outside the grid"):
        SheetSpec.from_node_ranges((10, 10, 10), normal_axis=2, plane=99,
                                   in_plane_ranges=((1, 4), (1, 4)),
                                   name="patch")


# --------------------------------------------------------------------------
# 4. the sheet reaches the solve, and the solve differentiates
# --------------------------------------------------------------------------

def _build_board(delta, *, probe, dx_profile=None, dy_profile=None,
                 pinned=True):
    sim = Simulation(freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
                     dx=DX,
                     dx_profile=_prof_x(delta) if dx_profile is None else dx_profile(delta),
                     dy_profile=None if dy_profile is None else dy_profile(delta),
                     cpml_layers=CPML, boundary="cpml")
    sim.add_material("sub", eps_r=3.38)
    sim.add(Box((2 * DX, 2 * DX, Z_GND), (22 * DX, 14 * DX, Z_PATCH)),
            material="sub")
    sim.add_pinned_sheet(plane_index=K_GND, i_range=(2, 22), j_range=(2, 14),
                         name="ground")
    if pinned:
        sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                             j_range=(J_LO, J_HI), name="patch")
    sim.add_source((3 * DX, 8 * DX, K_MID * DX), "ez", amplitude_kind="current")
    sim.add_probe(probe, "ez")
    return sim


def _assert_patch_edges(sheets):
    names = sorted(str(s.name) for s in sheets)
    assert names == ["ground", "patch"], names
    patch = next(s for s in sheets if s.name == "patch")
    edges = realized_pec_edge_masks(None, sheets=(patch,))
    n_x = int(np.asarray(edges[0]).sum())
    n_y = int(np.asarray(edges[1]).sum())
    # 9 x nodes x 7 y nodes on one plane: 8*7 x-edges and 9*6 y-edges.
    assert (n_x, n_y) == ((I_HI - I_LO) * (J_HI - J_LO + 1),
                          (I_HI - I_LO + 1) * (J_HI - J_LO))
    assert int(np.asarray(edges[2]).sum()) == 0   # a sheet shorts no normal E


def test_a_pinned_sheet_reaches_the_solve_on_the_nonuniform_lane():
    """The declared patch must end up as PEC edges the stepper zeroes — not
    merely as a spec nobody collected. Counted through the one function that
    turns conductor geometry into PEC edges, and confirmed against the same
    board with the patch removed: metal that reaches the solve changes what
    the probe sees.
    """
    from rfx.runners.nonuniform import assemble_materials_nu
    sim = _build_board(0.5 * DX, probe=(20 * DX, 8 * DX, K_MID * DX))
    grid = sim._build_nonuniform_grid()
    sheets, wires = [], []
    assemble_materials_nu(sim, grid, None, sheets, wires)
    _assert_patch_edges(sheets)

    with_patch = float(_loss(0.5 * DX, probe=(20 * DX, 8 * DX, K_MID * DX),
                             n_steps=30))
    bare = _build_board(0.5 * DX, probe=(20 * DX, 8 * DX, K_MID * DX),
                        pinned=False)
    without = float(jnp.sum(bare.forward(n_steps=30, skip_preflight=True)
                            .time_series ** 2))
    assert abs(with_patch - without) / max(without, 1e-30) > 1e-3, (
        f"the patch changes nothing at the probe ({with_patch:.6e} vs "
        f"{without:.6e}) — it is not reaching the solve")


def test_a_pinned_sheet_reaches_the_solve_on_the_uniform_lane():
    """The same declaration, assembled by the uniform lane."""
    sim = Simulation(freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
                     dx=DX, cpml_layers=CPML, boundary="cpml")
    sim.add_pinned_sheet(plane_index=K_GND, i_range=(2, 22), j_range=(2, 14),
                         name="ground")
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    sheets: list = []
    sim._assemble_materials(sim._build_grid(), pec_sheets=sheets,
                            pec_wires=[])
    _assert_patch_edges(sheets)


def _loss(delta, *, probe, dx_profile=None, dy_profile=None, n_steps=40):
    sim = _build_board(delta, probe=probe, dx_profile=dx_profile,
                       dy_profile=dy_profile)
    return jnp.sum(sim.forward(n_steps=n_steps, skip_preflight=True)
                   .time_series ** 2)


#: The four deformations the gradient is checked on. A: the +x edge moves,
#: probe downstream in the substrate. B: the same, probe on the patch's own
#: plane just past the +x edge, where the edge field is strongest. C: the +x
#: and +y edges move together. D: the LOW-side edge moves instead.
AD_CASES = {
    "A_plus_x_edge": dict(probe=(20 * DX, 8 * DX, K_MID * DX)),
    "B_probe_past_the_edge": dict(probe=(18 * DX, 8 * DX, Z_PATCH)),
    "C_joint_xy": dict(probe=(20 * DX, 8 * DX, K_MID * DX), dy_profile=_prof_y),
    "D_low_side_edge": dict(probe=(20 * DX, 8 * DX, K_MID * DX),
                            dx_profile=_prof_x_low),
}


@pytest.mark.parametrize("case", sorted(AD_CASES))
def test_patch_edge_gradient_matches_a_central_difference(case):
    """``jax.grad`` through ``delta -> dx_profile -> forward()`` against a
    central difference through the SAME function.

    The two routes must be solving the same problem before the derivative
    can be compared, so the traced primal is checked against the concrete
    one first — with a pinned declaration they agree to float32 round-off,
    because the footprint does not depend on the route at all.
    """
    kw = AD_CASES[case]
    d0 = 0.5 * DX
    value, grad_ad = jax.value_and_grad(lambda d: _loss(d, **kw))(
        jnp.float32(d0))
    grad_ad, value = float(grad_ad), float(value)
    concrete = float(_loss(d0, **kw))
    assert abs(value - concrete) / abs(concrete) < 1e-4, (
        f"{case}: traced primal {value:.9e} vs concrete {concrete:.9e}")

    h = DX / 64.0
    grad_fd = (float(_loss(d0 + h, **kw)) - float(_loss(d0 - h, **kw))) / (2 * h)
    rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-30)
    assert np.isfinite(grad_ad) and grad_ad != 0.0
    # float32 fields, centred difference at h = dx/64. The residual is the
    # DIFFERENCE's own truncation, not the gradient: laddering h over dx/8,
    # dx/16, dx/32, dx/64 it falls by ~3.9x per halving (second order) on
    # every case -- A 4.9e-2/1.3e-2/3.2e-3/7.9e-4, B 1.8e-3/4.6e-4/1.1e-4/
    # 3.6e-5, C 2.2e-1/6.4e-2/1.7e-2/4.2e-3, D 5.0e-2/1.3e-2/3.2e-3/8.0e-4.
    assert rel < 1.5e-2, (
        f"{case}: AD {grad_ad:+.6e} vs central FD {grad_fd:+.6e} — "
        f"relative {rel:.3e}")

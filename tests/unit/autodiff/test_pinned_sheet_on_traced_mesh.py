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
The patch x and y node lines are interior indices.
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

C0 = 299792458.0
DX = 0.5e-3
N_X, N_Y, N_Z = 24, 16, 20
CPML = 4
I_LO, I_HI = 8, 16
J_LO, J_HI = 5, 11
DEFORMATION_CELLS_PER_EDGE = 4
N_W = DEFORMATION_CELLS_PER_EDGE
D_MIN = DX * (1 - 0.6 / N_W)
DT_PIN = 0.9 / (C0 * np.sqrt(2 / D_MIN**2 + 1 / DX**2))
K_GND, K_PATCH = 6, 12
K_MID = 9
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



def test_a_metric_sheet_on_a_traced_mesh_is_still_refused():
    """A sheet drawn in metres has corners that must be located on the node
    line; a traced node line has no host position, so it is refused rather
    than placed somewhere."""
    def _loss(delta):
        sim = Simulation(freq_max=40e9,
                         domain=(N_X * DX, N_Y * DX, N_Z * DX), dx=DX,
                         dx_profile=_prof_x(delta), cpml_layers=CPML,
                         boundary="cpml",
                         nominal_profiles={"x": _prof_x(0.5 * DX)})
        sim.add_thin_conductor(Box((I_LO * DX, J_LO * DX, Z_PATCH),
                                   (I_HI * DX, J_HI * DX, Z_PATCH)))
        sim.add_source((3 * DX, 8 * DX, K_MID * DX), "ez",
                       amplitude_kind="current")
        sim.add_probe((20 * DX, 8 * DX, K_MID * DX), "ez")
        return jnp.sum(sim.forward(n_steps=4, skip_preflight=True)
                       .time_series ** 2)

    with pytest.raises(ValueError, match="a sheet's plane is a static integer"):
        jax.grad(_loss)(jnp.float32(0.5 * DX))



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
    assert (int(xs[0]), int(xs[-1])) == (I_LO, I_HI)
    assert (int(ys[0]), int(ys[-1])) == (J_LO + CPML, J_HI + CPML)
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



def _build_board(delta, *, probe, dx_profile=None, dy_profile=None,
                 pinned=True, pin_dt=True):
    """When pin_dt is true, the time step is pinned because a derived dt follows the smallest cell, which changes sides at delta = 0."""
    dt_kw = dict(dt=float(DT_PIN), dt_min_cell=float(D_MIN)) if pin_dt else {}
    sim = Simulation(freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
                     dx=DX,
                     dx_profile=_prof_x(delta) if dx_profile is None else dx_profile(delta),
                     dy_profile=None if dy_profile is None else dy_profile(delta),
                     cpml_layers=CPML, boundary="cpml", **dt_kw)
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
    """A sheet shorts no normal E."""
    names = sorted(str(s.name) for s in sheets)
    assert names == ["ground", "patch"], names
    patch = next(s for s in sheets if s.name == "patch")
    edges = realized_pec_edge_masks(None, sheets=(patch,))
    n_x = int(np.asarray(edges[0]).sum())
    n_y = int(np.asarray(edges[1]).sum())
    assert (n_x, n_y) == ((I_HI - I_LO) * (J_HI - J_LO + 1),
                          (I_HI - I_LO + 1) * (J_HI - J_LO))
    assert int(np.asarray(edges[2]).sum()) == 0


def test_a_pinned_sheet_reaches_the_solve_on_the_nonuniform_lane():
    """The declared patch must end up as PEC edges the stepper zeroes — not
    merely as a spec nobody collected. Counted through the one function that
    turns conductor geometry into PEC edges, and confirmed against the same
    board with the patch removed: metal that reaches the solve changes what
    the probe sees.
    The patched and bare boards derive dt, so a dt difference would pass as the patch reaching the
    solve.
    """
    from rfx.runners.nonuniform import assemble_materials_nu
    sim = _build_board(0.5 * DX, probe=(20 * DX, 8 * DX, K_MID * DX), pin_dt=False)
    grid = sim._build_nonuniform_grid()
    sheets, wires = [], []
    assemble_materials_nu(sim, grid, None, sheets, wires)
    _assert_patch_edges(sheets)

    with_patch = float(_loss(0.5 * DX, probe=(20 * DX, 8 * DX, K_MID * DX),
                             n_steps=30, pin_dt=False))
    bare = _build_board(0.5 * DX, probe=(20 * DX, 8 * DX, K_MID * DX),
                        pinned=False, pin_dt=False)
    bare_grid = bare._build_nonuniform_grid()
    for axis in ("dx_arr", "dy_arr", "dz"):
        np.testing.assert_array_equal(np.asarray(getattr(grid, axis)), np.asarray(getattr(bare_grid, axis)))
    assert float(grid.dt) == float(bare_grid.dt), (grid.dt, bare_grid.dt)
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


def _loss(delta, *, probe, dx_profile=None, dy_profile=None, n_steps=40,
          pin_dt=True):
    sim = _build_board(delta, probe=probe, dx_profile=dx_profile,
                       dy_profile=dy_profile, pin_dt=pin_dt)
    return jnp.sum(sim.forward(n_steps=n_steps, skip_preflight=True)
                   .time_series ** 2)


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
    central difference through the SAME function, with dt pinned on both routes.

    The two routes must be solving the same problem before the derivative
    can be compared, so the traced primal is checked against the concrete
    one first — with a pinned declaration they agree to float32 round-off,
    because the footprint does not depend on the route at all.
    An earlier measurement with float32 fields and pinned dt, using centred differences at dx/8,
    dx/16, dx/32, dx/64, gave relative errors A 7.9960e-3 / 2.0170e-3 / 5.4590e-4 / 7.0143e-6, B
    3.8693e-3 / 9.6618e-4 / 2.2386e-4 / 9.0866e-5, C 8.2931e-3 / 2.0895e-3 / 5.4981e-4 / 3.6687e-5,
    D 8.5021e-3 / 2.1354e-3 / 5.7214e-4 / 9.3833e-6.
    """
    kw = AD_CASES[case]
    d0 = 0.5 * DX
    board = _build_board(d0, **{k: v for k, v in kw.items() if k != "probe"},
                         probe=kw["probe"])
    patch = pinned_sheet_spec(board._build_nonuniform_grid(),
                              board._pinned_sheets[-1])
    n_edges = sum(int(np.asarray(m).sum())
                  for m in realized_pec_edge_masks(None, sheets=(patch,)))
    assert n_edges >= 1, f"{case}: the pinned patch realizes no PEC edge"

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
    assert rel < 1.5e-2, (
        f"{case}: AD {grad_ad:+.6e} vs central FD {grad_fd:+.6e} — "
        f"relative {rel:.3e}")



def _pinned_board():
    sim = Simulation(freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
                     dx=DX, cpml_layers=CPML, boundary="cpml")
    sim.add_material("sub", eps_r=3.38)
    sim.add(Box((2 * DX, 2 * DX, Z_GND), (22 * DX, 14 * DX, Z_PATCH)),
            material="sub")
    sim.add_source((3 * DX, 8 * DX, K_MID * DX), "ez",
                   amplitude_kind="current")
    sim.add_probe((20 * DX, 8 * DX, K_MID * DX), "ez")
    return sim


def test_preflight_sees_a_pinned_sheet_added_after_an_earlier_preflight():
    """The campaign-statics cache is keyed on the identity of every
    declaration that reaches the assembly. Without a term for the pinned
    sheets, a patch added after one preflight stayed invisible to the next
    while the solve realized it — preflight then audits a different model
    from the one that runs.
    """
    sim = _pinned_board()
    before = sim._campaign_ctx()
    assert [int(sp.plane) for sp in before.realized().sheets] == []

    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    after = sim._campaign_ctx()
    assert after is not before, "the cache key has no term for _pinned_sheets"
    assert [int(sp.plane) for sp in after.realized().sheets] == [K_PATCH + CPML]


def test_quick_convergence_refuses_a_node_pinned_conductor():
    """A dx sweep rescales node-pinned geometry, so each rung is a different
    antenna. Refused by name rather than run."""
    from rfx.convergence import quick_convergence
    sim = _pinned_board()
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    with pytest.raises(ValueError, match="node-pinned sheet"):
        quick_convergence(sim, n_steps=4)


def test_fidelity_report_carries_a_row_per_pinned_sheet():
    """The report's §1.7 cross-check compares the planes it names against the
    planes the assembly built. A pinned board used to name none, so the
    report told the reader to trust neither realization on a model where the
    two agreed.
    A sheet owns no cell.
    """
    sim = _pinned_board()
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    report = sim.fidelity_report(print_report=False)
    kinds = [f["kind"] for it in report for f in it["findings"]]
    assert "sheet-report-assembly-drift" not in kinds, kinds
    rows = [it for it in report if it["entity"].startswith("pinned_sheet")]
    assert len(rows) == 1
    row = rows[0]
    assert row["realized_plane"]["index"] == K_PATCH + CPML
    assert row["realized_plane"]["coordinate"] == pytest.approx(Z_PATCH)
    assert row["n_cells"] == 0
    spans = {a["axis"]: a["realized_extent_um"] for a in row["axes"]}
    assert spans["x"] == pytest.approx((I_HI - I_LO) * DX * 1e6)
    assert spans["y"] == pytest.approx((J_HI - J_LO) * DX * 1e6)


def test_fidelity_report_cross_checks_a_pinned_and_a_metric_sheet_together():
    sim = _pinned_board()
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    sim.add_thin_conductor(Box((2 * DX, 2 * DX, Z_GND),
                               (22 * DX, 14 * DX, Z_GND)))
    report = sim.fidelity_report(print_report=False)
    kinds = [f["kind"] for it in report for f in it["findings"]]
    assert "sheet-report-assembly-drift" not in kinds, kinds
    planes = sorted(it["realized_plane"]["index"] for it in report
                    if "realized_plane" in it)
    assert planes == [K_GND + CPML, K_PATCH + CPML]


def test_pinned_sheets_reach_the_viewer_and_the_run_record():
    from rfx.artifacts import build_scene_artifact
    from rfx.visualize import _declared_entries
    sim = _pinned_board()
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    assert any(isinstance(e, PinnedSheet) for e in _declared_entries(sim))
    scene = build_scene_artifact(sim, include_private=True)
    assert scene["private_summary"]["pinned_sheets"]["count"] == 1


def test_a_pinned_sheet_round_trips_through_the_design_document():
    from rfx.interop import design_to_dict, simulation_from_design
    sim = _pinned_board()
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    doc = design_to_dict(sim)
    assert len(doc["pinned_sheets"]) == 1
    rebuilt = simulation_from_design(doc)
    assert rebuilt._pinned_sheets == sim._pinned_sheets
    assert np.array_equal(
        np.asarray(pinned_sheet_spec(rebuilt._build_grid(),
                                     rebuilt._pinned_sheets[0]).footprint),
        np.asarray(pinned_sheet_spec(sim._build_grid(),
                                     sim._pinned_sheets[0]).footprint))

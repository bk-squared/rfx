"""Issue #691 — every lumped fold on the non-uniform lane must realize its
nominal value through the E node's DUAL face.

What binds here
---------------
Anything stamped into ``materials.sigma`` / ``materials.eps_r`` at an E node
realizes its lumped value through that node's discrete Ampere control volume::

    G = sigma * dual_b * dual_c / d_parallel_primal
    C = (eps_r_extra * EPS_0) * dual_b * dual_c / d_parallel_primal

with ``(b, c)`` the two axes TRANSVERSE to the component and ``dual`` the
E-node spacing ``(d[k-1] + d[k]) / 2``.  PR #694 fixed the wire/lumped port
termination in ``rfx/runners/nonuniform.py``; the three sites gated here were
left behind and their error is larger.

The ORACLE below is deliberately NOT ``e_node_dual_spacing_at`` (the helper
the production code now calls).  It is ``1 / grid.inv_d*`` — the metric the
non-uniform E update literally divides its curl by — so this file cannot pass
by agreeing with the implementation it is checking.

Fixture discipline
------------------
A uniform-VALUED profile exercises the non-uniform code path but none of its
metrics (primal == dual everywhere), and equal grading ratios on every axis
leave ``dual_x == dual_y == dual_z`` at the node, so an axis permutation
passes.  Every grid below is graded with a DIFFERENT ratio on each axis
(x 2:1, y 3:1, z 4:1) and every gate is evaluated at a fine->coarse node, a
coarse->fine node AND a locally uniform node.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import EPS_0
from rfx.grid import Grid
from rfx.lumped import (LumpedRLCSpec, build_rlc_meta, setup_rlc_materials,
                        setup_rlc_materials_traced)
from rfx.nonuniform import make_nonuniform_grid
from rfx.simulation import MaterialArrays
from rfx.sources.msl_port import MSLPort, msl_cross_section_span, setup_msl_port
from rfx.sources.sources import port_d_parallel, port_sigma

MM = 1e-3
AXIS = {"ex": 0, "ey": 1, "ez": 2}

# Node index -> what the mesh does there, identical on all three axes.
NODE_KINDS = {3: "fine->coarse", 7: "coarse->fine", 5: "locally-uniform"}


def _profile(fine, coarse):
    """3 fine | 4 coarse | 3 fine.  Node 3 is fine->coarse, node 7 is
    coarse->fine, node 5 sits inside the coarse run (locally uniform)."""
    return np.array([fine] * 3 + [coarse] * 4 + [fine] * 3, dtype=np.float64)


DX_PROF = _profile(0.5 * MM, 1.0 * MM)   # ratio 2
DY_PROF = _profile(0.3 * MM, 0.9 * MM)   # ratio 3
DZ_PROF = _profile(0.2 * MM, 0.8 * MM)   # ratio 4


def _graded_grid():
    return make_nonuniform_grid(
        (0.0, 0.0), DZ_PROF, float(DX_PROF[0]),
        dx_profile=DX_PROF, dy_profile=DY_PROF, cpml_layers=0,
        pec_faces={"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"},
    )


def _mats(grid):
    return MaterialArrays(
        eps_r=jnp.ones(grid.shape), sigma=jnp.zeros(grid.shape),
        mu_r=jnp.ones(grid.shape),
    )


def _oracle_duals(grid, idx):
    """Dual spacings read back from the solver's OWN E-update metrics.

    ``inv_d*[k]`` is what ``update_e_nu`` multiplies the curl by, so
    ``1/inv_d*[k]`` is the control-volume width of the E node at ``k`` by
    construction — an oracle independent of ``e_node_dual_spacing_at``.
    """
    invs = (grid.inv_dx, grid.inv_dy, grid.inv_dz)
    return tuple(1.0 / float(np.asarray(invs[ax])[idx[ax]]) for ax in range(3))


def _oracle_primal(grid, idx):
    arrs = (grid.dx_arr, grid.dy_arr, grid.dz)
    return tuple(float(np.asarray(arrs[ax])[idx[ax]]) for ax in range(3))


def _realized_conductance(grid, sigma, idx, component):
    ax = AXIS[component]
    duals = _oracle_duals(grid, idx)
    d_par = _oracle_primal(grid, idx)[ax]
    b, c = [duals[t] for t in range(3) if t != ax]
    return float(sigma[idx]) * b * c / d_par


def _node_position(profile, index):
    """Physical coordinate of the E node at ``index`` (pad-free grids only).

    ``_axis_position_to_index`` resolves against cell EDGES, and the E node at
    index ``i`` sits on edge ``i``.
    """
    return float(np.sum(np.asarray(profile)[:index]))


# ---------------------------------------------------------------------------
# Site 1 — add_lumped_rlc on the NU lane (port_sigma / _axis_cell_sizes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("node", sorted(NODE_KINDS))
def test_rlc_parallel_resistor_realizes_its_ohms_on_a_graded_mesh(component, node):
    """A nominal R stamped by ``setup_rlc_materials`` must realize G*R == 1.

    Pre-#691 measurement on the issue's 0.5/1.0 mm doubly-graded fixture:
    G*R = 2.250000 at both grading steps and 4.000000 at a LOCALLY UNIFORM
    node — a 50 ohm resistor realized as 22.2 / 22.2 / 12.5 ohm.  The
    locally-uniform failure is the one that shows this was never only a
    grading-step defect: ``_axis_cell_sizes`` read the scalar BOUNDARY
    ``grid.dx`` / ``grid.dy`` for x and y, i.e. the wrong cell everywhere.
    """
    grid = _graded_grid()
    idx = (node, node, node)
    pos = (_node_position(DX_PROF, node), _node_position(DY_PROF, node),
           _node_position(DZ_PROF, node))
    R = 50.0
    spec = LumpedRLCSpec(R=R, L=0.0, C=0.0, topology="parallel",
                         position=pos, component=component)
    from rfx.lumped import _resolve_position_to_index
    assert tuple(_resolve_position_to_index(grid, pos)) == idx

    out = setup_rlc_materials(grid, spec, _mats(grid))
    g_r = _realized_conductance(grid, out.sigma, idx, component) * R
    assert g_r == pytest.approx(1.0, rel=2e-6), (
        f"{NODE_KINDS[node]} node {idx}, {component}: realized G*R={g_r!r} "
        f"(realized R = {R / g_r} ohm for a nominal {R} ohm)"
    )


def test_rlc_resistor_gate_sees_a_mixed_node_too():
    """One node that is a DIFFERENT kind on each axis, so a per-axis mix-up
    cannot hide behind three identical indices."""
    grid = _graded_grid()
    idx = (3, 7, 5)
    pos = (_node_position(DX_PROF, 3), _node_position(DY_PROF, 7),
           _node_position(DZ_PROF, 5))
    R = 75.0
    for component in ("ex", "ey", "ez"):
        spec = LumpedRLCSpec(R=R, L=0.0, C=0.0, topology="parallel",
                             position=pos, component=component)
        out = setup_rlc_materials(grid, spec, _mats(grid))
        g_r = _realized_conductance(grid, out.sigma, idx, component) * R
        assert g_r == pytest.approx(1.0, rel=2e-6), (component, g_r)


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("node", sorted(NODE_KINDS))
def test_port_sigma_matches_the_solver_metric_directly(component, node):
    """``port_sigma`` is the shared primitive under the RLC fold and the coax
    gap stamp, so gate it on its own against the inv_d* oracle."""
    grid = _graded_grid()
    idx = (node, node, node)
    Z0 = 50.0
    sigma = port_sigma(grid, idx, component, Z0)
    duals = _oracle_duals(grid, idx)
    ax = AXIS[component]
    b, c = [duals[t] for t in range(3) if t != ax]
    d_par = _oracle_primal(grid, idx)[ax]
    assert sigma * b * c / d_par * Z0 == pytest.approx(1.0, rel=2e-6)
    # d_parallel must be the node's OWN primal cell, not the boundary scalar.
    assert port_d_parallel(grid, idx, component) == pytest.approx(
        d_par, rel=1e-6)


# ---------------------------------------------------------------------------
# Site 3 — the capacitance fold's cubic-cell assumption
# ---------------------------------------------------------------------------

def _realized_capacitance(grid, eps_r, idx, component):
    ax = AXIS[component]
    duals = _oracle_duals(grid, idx)
    d_par = _oracle_primal(grid, idx)[ax]
    b, c = [duals[t] for t in range(3) if t != ax]
    return (float(eps_r[idx]) - 1.0) * EPS_0 * b * c / d_par


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_capacitance_fold_is_wrong_on_an_anisotropic_UNIFORM_grid(component):
    """The shipped fold spelled ``C / (d_par * EPS_0)``, which is
    ``C * d_par / (EPS_0 * d_par**2)`` — valid only on a CUBIC cell.

    This grid is UNIFORM on every axis (no grading at all) and merely
    anisotropic, 1.0 / 0.5 / 0.25 mm.  Pre-#691 the realized capacitance was
    0.125x nominal as ``ex`` and 8.0x as ``ez``.  Kept as a separate case
    from the graded ones because it proves the defect is not a grading
    defect.
    """
    prof_x = np.full(10, 1.0 * MM)
    prof_y = np.full(10, 0.5 * MM)
    prof_z = np.full(10, 0.25 * MM)
    grid = make_nonuniform_grid(
        (0.0, 0.0), prof_z, float(prof_x[0]),
        dx_profile=prof_x, dy_profile=prof_y, cpml_layers=0,
        pec_faces={"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"},
    )
    idx = (5, 5, 5)
    pos = (_node_position(prof_x, 5), _node_position(prof_y, 5),
           _node_position(prof_z, 5))
    C = 1e-12
    spec = LumpedRLCSpec(R=0.0, L=0.0, C=C, topology="parallel",
                         position=pos, component=component)
    out = setup_rlc_materials(grid, spec, _mats(grid))
    ratio = _realized_capacitance(grid, out.eps_r, idx, component) / C
    assert ratio == pytest.approx(1.0, rel=2e-6), (component, ratio)


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("node", sorted(NODE_KINDS))
def test_capacitance_fold_on_a_graded_mesh(component, node):
    grid = _graded_grid()
    idx = (node, node, node)
    pos = (_node_position(DX_PROF, node), _node_position(DY_PROF, node),
           _node_position(DZ_PROF, node))
    C = 2.5e-13
    spec = LumpedRLCSpec(R=0.0, L=0.0, C=C, topology="parallel",
                         position=pos, component=component)
    out = setup_rlc_materials(grid, spec, _mats(grid))
    ratio = _realized_capacitance(grid, out.eps_r, idx, component) / C
    assert ratio == pytest.approx(1.0, rel=2e-6), (
        f"{NODE_KINDS[node]} node {idx}, {component}: C_realized/C = {ratio}")


def test_traced_rlc_setup_folds_the_same_numbers_as_the_concrete_one():
    """``forward()``'s traced twin must not drift from ``run()``'s fold."""
    grid = _graded_grid()
    node = 3
    pos = (_node_position(DX_PROF, node), _node_position(DY_PROF, node),
           _node_position(DZ_PROF, node))
    spec = LumpedRLCSpec(R=50.0, L=0.0, C=1e-12, topology="parallel",
                         position=pos, component="ez")
    concrete = setup_rlc_materials(grid, spec, _mats(grid))
    traced = setup_rlc_materials_traced(grid, spec, _mats(grid))
    idx = (node, node, node)
    assert float(traced.sigma[idx]) == pytest.approx(
        float(concrete.sigma[idx]), rel=1e-12)
    assert float(traced.eps_r[idx]) == pytest.approx(
        float(concrete.eps_r[idx]), rel=1e-12)


# ---------------------------------------------------------------------------
# Site 2 — setup_msl_port
# ---------------------------------------------------------------------------

def _msl_realized_admittance(grid, sigma, span):
    """Total port admittance from the same control volume: cells stacked on
    the substrate NORMAL are in series, cells across the trace WIDTH are in
    parallel — the topology ``setup_msl_port``'s own formula assumes."""
    ip, iw, inr = span["prop_idx"], span["width_idx"], span["normal_idx"]
    columns = {}
    for cell in span["cells"]:
        duals = _oracle_duals(grid, cell)
        d_norm = _oracle_primal(grid, cell)[inr]
        g_cell = float(sigma[cell]) * duals[ip] * duals[iw] / d_norm
        columns.setdefault(cell[iw], []).append(g_cell)
    return sum(1.0 / sum(1.0 / g for g in gs) for gs in columns.values())


@pytest.mark.parametrize("feed_node", sorted(NODE_KINDS))
def test_msl_uniform_termination_realizes_1_over_z0(feed_node):
    """Measured pre-#691 with the feed on an x transition node and a 2:1
    ``dx_profile``: Y*Z0 = 0.750 fine->coarse, 1.500 coarse->fine, 1.000 on a
    uniform control — exactly ``dual_x / primal_x``.

    The fence a previous commit put on this site ("its V/I route is Gwarek,
    not the #672 Ampere loop, so the consistency argument does not transfer")
    is wrong and must not be reinstated: the realized conductance of a
    stamped sigma is fixed by the E node's discrete Ampere / Joule control
    volume, not by which extractor later reads the port.
    """
    grid = _graded_grid()
    x_edges = np.concatenate([[0.0], np.cumsum(DX_PROF)])
    y_edges = np.concatenate([[0.0], np.cumsum(DY_PROF)])
    z_edges = np.concatenate([[0.0], np.cumsum(DZ_PROF)])
    Z0 = 50.0
    port = MSLPort(
        feed_x=float(x_edges[feed_node]),
        y_lo=float(y_edges[3]), y_hi=float(y_edges[6]),
        z_lo=float(z_edges[2]), z_hi=float(z_edges[5]),
        direction="+x", impedance=Z0,
    )
    span = msl_cross_section_span(grid, port)
    assert span["i_feed"] == feed_node
    assert span["w_hi"] > span["w_lo"] and span["n_hi"] > span["n_lo"]
    out = setup_msl_port(grid, port, _mats(grid))
    y_z0 = _msl_realized_admittance(grid, out.sigma, span) * Z0
    assert y_z0 == pytest.approx(1.0, rel=2e-6), (
        f"feed on a {NODE_KINDS[feed_node]} node: Y*Z0 = {y_z0}")


@pytest.mark.parametrize("feed_node", sorted(NODE_KINDS))
def test_msl_eigenmode_termination_dissipates_v_squared_over_z0(feed_node):
    """Eigenmode branch: sigma is uniform over the cross-section and sized by
    ``1 / (Z0 * dx_feed * integral|ez|^2)``.  ``dx_feed`` was the PRIMAL
    propagation cell and the integral used one scalar (dy, dz) pair for the
    whole box, so the matched-load power was wrong by the same dual/primal
    factor.  Gate the realized dissipation directly::

        P = sigma * sum |ez|^2 * d_norm * dual_width * dual_prop  ==  1/Z0

    (``ez_profile`` is normalised to 1 V, so ``V^2/Z0 = 1/Z0``.)  The mode
    profile here is a hand-built dict rather than a Laplace solve: the
    quantity under test is the metric the dict is integrated against, and a
    synthetic profile keeps the gate independent of the solver's shape.
    """
    grid = _graded_grid()
    x_edges = np.concatenate([[0.0], np.cumsum(DX_PROF)])
    y_edges = np.concatenate([[0.0], np.cumsum(DY_PROF)])
    z_edges = np.concatenate([[0.0], np.cumsum(DZ_PROF)])
    Z0 = 50.0
    port = MSLPort(
        feed_x=float(x_edges[feed_node]),
        y_lo=float(y_edges[3]), y_hi=float(y_edges[6]),
        z_lo=float(z_edges[2]), z_hi=float(z_edges[5]),
        direction="+x", impedance=Z0,
    )
    span = msl_cross_section_span(grid, port)
    j_lo, j_hi = span["w_lo"], span["w_hi"]
    k_lo, k_hi = span["n_lo"], span["n_hi"]
    n_w = j_hi - j_lo + 1
    n_n = k_hi - k_lo + 1
    rng = np.random.default_rng(691)
    ez_profile = 0.5 + rng.random((n_w, n_n))
    cells = [(feed_node, j, k)
             for j in range(j_lo, j_hi + 1) for k in range(k_lo, k_hi + 1)]
    mode_profile = dict(
        ez_profile=ez_profile, cell_indices=cells,
        j_grid_lo=j_lo, k_grid_lo=k_lo, n_z_sub=n_n,
        dy=float(DY_PROF[j_lo]), dz=float(DZ_PROF[k_lo]),
        prop_idx=0, width_idx=1, normal_idx=2,
        prop_axis="x", width_axis="y", normal_axis="z",
    )
    out = setup_msl_port(grid, port, _mats(grid), mode_profile=mode_profile)

    power = 0.0
    for cell in cells:
        duals = _oracle_duals(grid, cell)
        d_norm = _oracle_primal(grid, cell)[2]
        ez = ez_profile[cell[1] - j_lo, cell[2] - k_lo]
        power += float(out.sigma[cell]) * ez * ez * d_norm * duals[0] * duals[1]
    assert power * Z0 == pytest.approx(1.0, rel=2e-6), (
        f"feed on a {NODE_KINDS[feed_node]} node: P*Z0 = {power * Z0}")


# ---------------------------------------------------------------------------
# Uniform lane must not move
# ---------------------------------------------------------------------------

def test_uniform_grid_port_metrics_are_byte_identical_to_the_pre_691_form():
    """A uniform ``Grid`` is cubic, so dual == primal and every expression
    above must reduce BIT-FOR-BIT to what the pre-#691 code computed.

    The pre-#691 spelling was ``d_par / (Z0 * d_perp1 * d_perp2)`` with all
    three sizes ``float(grid.dx)``; that is NOT bit-equal to ``1/(Z0*dx)``
    (measured 99.99999999999999 against 100.0 at dx = 0.2 mm), so the pinned
    reference below is the expression, not the algebraic simplification.
    """
    grid = Grid(freq_max=1e10, domain=(2e-3, 2e-3, 2e-3), dx=0.2 * MM,
                cpml_layers=4)
    idx = (7, 7, 7)
    d = float(grid.dx)
    for component in ("ex", "ey", "ez"):
        assert port_sigma(grid, idx, component, 50.0) == d / (50.0 * d * d)
        assert port_d_parallel(grid, idx, component) == d


def test_uniform_grid_capacitance_fold_is_byte_identical_to_the_pre_691_form():
    """On a cubic cell ``C * d_par / (EPS_0 * d * d)`` must land on the same
    ``eps_r`` the old ``C / (d_par * EPS_0)`` did, or every uniform-lane RLC
    number moves for no physical reason.

    The two spellings differ by 1 ULP in float64 (564.7045336865094 against
    564.7045336865095 at dx = 0.2 mm, C = 1 pF), so the pin is on the value
    as STORED — ``materials.eps_r`` is float32 and rounds them together.
    """
    grid = Grid(freq_max=1e10, domain=(2e-3, 2e-3, 2e-3), dx=0.2 * MM,
                cpml_layers=4)
    idx = tuple(grid.position_to_index((1e-3, 1e-3, 1e-3)))
    d = float(grid.dx)
    C = 1e-12
    old_form = np.float32(1.0 + C / (d * EPS_0))
    for component in ("ex", "ey", "ez"):
        spec = LumpedRLCSpec(R=0.0, L=0.0, C=C, topology="parallel",
                             position=(1e-3, 1e-3, 1e-3), component=component)
        out = setup_rlc_materials(grid, spec, _mats(grid))
        assert np.float32(out.eps_r[idx]) == old_form


# ---------------------------------------------------------------------------
# Site 2b — compute_msl_mode_profile's 1 V normalisation
#
# The two gates above hand-build ``mode_profile`` so the metric under test is
# isolated from the Laplace solve.  That is why they could not see the
# regression this section closes: the profile's OWN normalisation still
# divided by a single scalar ``dz`` while #691 made the sigma-sizing integral
# per-cell, so the two scalar-dz errors stopped cancelling.  Everything below
# therefore runs the REAL ``compute_msl_mode_profile``.
# ---------------------------------------------------------------------------

MSL_DX_PROF = np.array([0.2 * MM] * 4 + [0.4 * MM] * 6 + [0.2 * MM] * 4)   # 2:1
MSL_DY_PROF = np.array([0.15 * MM] * 4 + [0.45 * MM] * 6 + [0.15 * MM] * 4)  # 3:1


def _msl_graded_fixture(z_ratio):
    """MSL on a substrate graded ``z_ratio`` along the substrate NORMAL.

    x is graded 2:1 and y 3:1, so ``dual_x != dual_y != dual_z`` at every node
    and an axis permutation cannot pass.
    """
    fine = 0.1 * MM
    dz_prof = np.array([fine] * 2 + [fine * z_ratio] * 2 + [0.2 * MM] * 10,
                       dtype=np.float64)
    grid = make_nonuniform_grid(
        (0.0, 0.0), dz_prof, float(MSL_DX_PROF[0]),
        dx_profile=MSL_DX_PROF, dy_profile=MSL_DY_PROF, cpml_layers=0,
        pec_faces={"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"},
    )
    x_edges = np.concatenate([[0.0], np.cumsum(MSL_DX_PROF)])
    y_edges = np.concatenate([[0.0], np.cumsum(MSL_DY_PROF)])
    z_edges = np.concatenate([[0.0], np.cumsum(dz_prof)])
    port = MSLPort(
        feed_x=float(x_edges[5]),
        y_lo=float(y_edges[5]), y_hi=float(y_edges[9]),
        z_lo=0.0, z_hi=float(z_edges[4]),
        direction="+x", impedance=50.0,
    )
    return grid, port


@pytest.mark.parametrize("z_ratio", [1.0, 2.0, 4.0])
def test_msl_mode_profile_carries_one_volt_through_the_extractors_own_reader(
        z_ratio):
    """``compute_msl_mode_profile`` documents its output as "normalised so
    that integrating Ez·dz along z at the trace centre yields 1 V".

    ``v_centre`` is a MODAL VOLTAGE — a line integral of Ez along the
    substrate normal — and on the Yee grid the Ez node at normal index k sits
    on the PRIMAL edge of length ``dz_arr[k]``, so the discrete integral is
    ``Σ_k Ez[k]·dz_arr[k]``.  The oracle here is not that statement but the
    READER: ``rfx.api._sparams.msl_modal_voltage``, the extractor that turns
    a recorded Ez plane into V, written for a different lane.  If the two
    disagree the injected "1 V" mode is not the volt the S-matrix reads back.

    Measured with the pre-fix scalar-``dz`` spelling on this fixture:
    V = 1.263197 (z_ratio 1), 1.631995 (2), 2.369591 (4).
    """
    from rfx.api._sparams import msl_modal_voltage
    from rfx.sources.msl_port import compute_msl_mode_profile

    grid, port = _msl_graded_fixture(z_ratio)
    mp = compute_msl_mode_profile(grid, port, 4.4)
    ez = np.asarray(mp["ez_profile"])
    j_lo, k_lo, n_z = int(mp["j_grid_lo"]), int(mp["k_grid_lo"]), int(mp["n_z_sub"])
    j_centre = (int(mp["trace_j_lo"]) + int(mp["trace_j_hi"])) // 2

    # Register the profile onto a full (n_freqs, ny, nz) plane the way the
    # source does, then let the extractor integrate it.  ``k_hi`` is
    # exclusive: here it is the top of the profile's own span, which is the
    # span the source injects over.
    ez_plane = np.zeros((1, grid.shape[1], grid.shape[2]), dtype=np.float64)
    ez_plane[0, j_lo:j_lo + ez.shape[0], k_lo:k_lo + ez.shape[1]] = ez
    v = msl_modal_voltage(
        jnp.asarray(ez_plane), j_centre=j_centre,
        k_lo=k_lo, k_hi=k_lo + n_z, dz_arr=np.asarray(grid.dz),
    )
    assert float(v[0]) == pytest.approx(1.0, rel=1e-9), (
        f"z_ratio {z_ratio}: profile carries V = {float(v[0])} V, not 1 V — "
        f"the normalisation and the extractor disagree about ∫Ez·dz"
    )


@pytest.mark.parametrize("z_ratio", [1.0, 2.0, 4.0])
def test_msl_eigenmode_realizes_z0_end_to_end_on_a_graded_substrate(z_ratio):
    """End-to-end: profile solve -> sigma sizing -> realized impedance.

    ``Z_realized = V_port² / P_diss`` with both halves read from the solver's
    own metrics (``1/grid.inv_d*`` and ``grid.d*_arr``), never from
    ``e_node_dual_spacing_at``.  #691 made ``P_diss`` exactly ``1/Z0`` but
    left ``V_port`` scalar-normalised, so the realized impedance moved by
    ``V_port²`` and got WORSE than before that commit.

    Measured on this fixture at HEAD 856ac9b: 79.78 Ω (z_ratio 1),
    133.17 Ω (2), 280.75 Ω (4), against a nominal 50 Ω.
    """
    from rfx.sources.msl_port import compute_msl_mode_profile

    grid, port = _msl_graded_fixture(z_ratio)
    mp = compute_msl_mode_profile(grid, port, 4.4)
    out = setup_msl_port(grid, port, _mats(grid), mode_profile=mp)

    ez = np.asarray(mp["ez_profile"])
    j_lo, k_lo = int(mp["j_grid_lo"]), int(mp["k_grid_lo"])
    j_c = (int(mp["trace_j_lo"]) + int(mp["trace_j_hi"])) // 2 - j_lo
    dz_prim = np.array([_oracle_primal(grid, (0, 0, k_lo + k))[2]
                        for k in range(ez.shape[1])])
    v_port = float(np.sum(ez[j_c, :] * dz_prim))

    power = 0.0
    for cell in mp["cell_indices"]:
        duals = _oracle_duals(grid, cell)
        d_norm = _oracle_primal(grid, cell)[2]
        e = ez[cell[1] - j_lo, cell[2] - k_lo]
        power += float(out.sigma[cell]) * e * e * d_norm * duals[0] * duals[1]

    z_realized = v_port * v_port / power
    assert z_realized == pytest.approx(float(port.impedance), rel=1e-6), (
        f"z_ratio {z_ratio}: realized {z_realized:.2f} ohm against a nominal "
        f"{float(port.impedance):.2f} ohm (V_port = {v_port:.6f}, "
        f"P*Z0 = {power * float(port.impedance):.6f})"
    )


def test_msl_mode_profile_on_a_uniform_grid_is_bit_identical_to_the_scalar_form():
    """Per-cell weighting must not move the uniform lane.

    ``np.sum(ez * dz_span)`` and ``np.sum(ez) * dz`` are NOT algebraically
    guaranteed to agree bit-for-bit even when every ``dz_span`` entry equals
    ``dz`` — each product rounds separately and ``dz`` is not a power of two
    — so this is a MEASURED pin on this fixture, not an identity.  If it ever
    reds, check the delta before touching it: a last-bit move is acceptable,
    a visible one is not.
    """
    from rfx.sources.msl_port import compute_msl_mode_profile

    grid = Grid(freq_max=2e10, domain=(4e-3, 4e-3, 2e-3), dx=0.1 * MM,
                cpml_layers=4)
    port = MSLPort(feed_x=1.0 * MM, y_lo=1.8 * MM, y_hi=2.2 * MM,
                   z_lo=0.0, z_hi=0.3 * MM, direction="+x", impedance=50.0)
    ez = np.asarray(compute_msl_mode_profile(grid, port, 4.4)["ez_profile"])
    assert float(ez.sum()) == 73361.79312530009
    assert float((ez * ez).sum()) == 169878467.3801139
    assert float(ez.max()) == 4556.98256846321


# ---------------------------------------------------------------------------
# Site 3 — the lumped RLC ADE itself (build_rlc_meta + _update_parallel /
# _update_series)
#
# 1d3cc9c corrected `port_d_parallel`, which `build_rlc_meta` reads, and so
# MOVED the realized inductance of `add_lumped_rlc(L=...)` on the NU lane
# without a gate.  Nothing here covered an NU inductor at all.
#
# The element current has to become a current DENSITY through the node's DUAL
# face, and the inductor's implicit self-coupling `gamma` has to use the same
# area:
#
#     E^{n+1} = (D0*E_std - I^n / dual_area) / (D0 + gamma)
#     I^{n+1} = I^n + (dt*d_par/L) * E^{n+1}
#     gamma   = dt * d_par / (L * dual_area)
#
# The old pair spelled both with `d_par**2`.  That is self-consistent — it is
# the correct scheme for a CUBIC cell — which is why the error is a clean
# scale factor: the field sees an inductor of `L * d_par**2 / dual_area`.
# ---------------------------------------------------------------------------

#: ``e_new`` is read back out of a float32 field array, so any quantity
#: derived from it carries float32 storage rounding.  MEASURED worst case over
#: the 9 (component, node) cells below: |ratio - 1| = 1.353e-07, which is
#: 1.13x ``np.finfo(np.float32).eps``.  The gates use 8x that eps -- a margin
#: over the storage bound, and still four orders tighter than the SMALLEST
#: defect being gated (ey at a locally-uniform node read 1.0125, i.e. 1.25 %).
#: Do not tighten below the float32 bound; do not loosen without recomputing
#: it.
_F32_METRIC_REL = 8.0 * float(np.finfo(np.float32).eps)


def _rlc_state_with_current(i_L):
    from rfx.lumped import init_rlc_state
    return init_rlc_state()._replace(inductor_current=jnp.asarray(i_L))


def _single_node_state(grid, component, idx, e_value):
    from rfx.core.yee import FDTDState
    z = jnp.zeros(grid.shape)
    fields = {n: z for n in ("ex", "ey", "ez", "hx", "hy", "hz")}
    fields[component] = z.at[idx].set(e_value)
    return FDTDState(**fields, step=jnp.asarray(0))


def _oracle_dual_area(grid, idx, component):
    """Area of the dual face the element current pierces, from the solver's
    own E-update metrics — never from ``port_dual_transverse``."""
    duals = _oracle_duals(grid, idx)
    ax = AXIS[component]
    b, c = [duals[t] for t in range(3) if t != ax]
    return b * c


def _graded_node(component, node):
    """(grid, spec position, idx) for an element at ``node`` on all 3 axes."""
    grid = _graded_grid()
    pos = (_node_position(DX_PROF, node),
           _node_position(DY_PROF, node),
           _node_position(DZ_PROF, node))
    return grid, pos, (node, node, node)


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("node", sorted(NODE_KINDS))
def test_nu_parallel_inductor_realizes_its_henries_on_a_graded_mesh(
        component, node):
    """``L_realized / L`` must be 1 at every node kind on every axis.

    OPERATIONAL definition, so this cannot pass by restating the rule.  The
    discrete Ampere law at the node says the field is loaded with
    ``I_implied = -dual_area * D0 * (E_new - E_std)``; the element's own ODE
    (``L dI/dt = E*d_par``) produced ``i_L_new``.  The inductance the FIELD
    sees is therefore ``L * i_L_new / I_implied``.  ``dual_area`` comes from
    ``1/grid.inv_d*``.

    Measured before this gate existed (x2/y3/z4 fixture, target 1.0)::

        node 3 fine->coarse     ex 3.333333  ey 2.160000  ez 1.422222
        node 5 locally-uniform  ex 1.388889  ey 1.012500  ez 0.711111
        node 7 coarse->fine     ex 0.833333  ey 0.240000  ez 0.088889

    and at the commit BEFORE 1d3cc9c, ex node 3 read 0.833333 — that commit
    moved it from 20 % low to 233 % high without a test noticing.
    """
    from rfx.lumped import _update_parallel

    grid, pos, idx = _graded_node(component, node)
    L = 1e-9
    spec = LumpedRLCSpec(R=0.0, L=L, C=0.0, topology="parallel",
                         position=pos, component=component)
    mats = setup_rlc_materials(grid, spec, _mats(grid))
    meta = build_rlc_meta(grid, spec, mats)
    assert (meta.i, meta.j, meta.k) == idx

    e_std, i_L = 3.7, 2.5e-3
    state = _single_node_state(grid, component, idx, e_std)
    new_state, rlc_new = _update_parallel(
        state, _rlc_state_with_current(i_L), meta)
    e_new = float(getattr(new_state, component)[idx])
    i_new = float(rlc_new.inductor_current)

    area = _oracle_dual_area(grid, idx, component)
    i_implied = -area * meta.D0 * (e_new - e_std)
    ratio = i_new / i_implied
    assert ratio == pytest.approx(1.0, rel=_F32_METRIC_REL), (
        f"{component} at a {NODE_KINDS[node]} node: L_realized/L = {ratio}"
    )


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("node", sorted(NODE_KINDS))
def test_nu_series_element_loads_the_node_through_its_dual_face(
        component, node):
    """``_update_series`` has its own current-density conversion.

    Read the area back out of the update rather than trusting the field:
    ``(D0 + gamma)*E_new = D0*E_std - I_new/area`` inverts to
    ``area = -I_new / ((D0 + gamma)*E_new - D0*E_std)``, which holds whatever
    ``gamma`` is, so this gate isolates the area alone.
    """
    from rfx.lumped import _update_series

    grid, pos, idx = _graded_node(component, node)
    spec = LumpedRLCSpec(R=25.0, L=2e-9, C=0.0, topology="series",
                         position=pos, component=component)
    mats = setup_rlc_materials(grid, spec, _mats(grid))
    meta = build_rlc_meta(grid, spec, mats)
    assert meta.is_series, "fixture must exercise the series ADE"

    e_std, i_s = 3.7, 2.5e-3
    state = _single_node_state(grid, component, idx, e_std)
    new_state, rlc_new = _update_series(
        state, _rlc_state_with_current(i_s), meta)
    e_new = float(getattr(new_state, component)[idx])
    i_new = float(rlc_new.inductor_current)

    denom = (meta.D0 + meta.gamma) * e_new - meta.D0 * e_std
    area_implied = -i_new / denom
    area_oracle = _oracle_dual_area(grid, idx, component)
    assert area_implied == pytest.approx(area_oracle, rel=_F32_METRIC_REL), (
        f"{component} at a {NODE_KINDS[node]} node: the series ADE converts "
        f"its current through {area_implied} m^2, the node's dual face is "
        f"{area_oracle} m^2 (primal d_par**2 would be "
        f"{_oracle_primal(grid, idx)[AXIS[component]] ** 2})"
    )


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("node", sorted(NODE_KINDS))
def test_nu_traced_rlc_meta_carries_the_same_fold_as_the_concrete_twin(
        component, node):
    """``forward()`` uses ``build_rlc_meta_traced``.  If only the concrete
    builder were fixed the differentiable lane would keep the cubic-cell
    inductance and the two lanes would disagree — the failure this repo has
    hit twice recently."""
    from rfx.lumped import build_rlc_meta_traced, setup_rlc_materials_traced

    grid, pos, idx = _graded_node(component, node)
    spec = LumpedRLCSpec(R=25.0, L=2e-9, C=1e-12, topology="parallel",
                         position=pos, component=component)
    concrete = build_rlc_meta(
        grid, spec, setup_rlc_materials(grid, spec, _mats(grid)))
    traced = build_rlc_meta_traced(
        grid, spec, setup_rlc_materials_traced(grid, spec, _mats(grid)))
    assert float(traced.dual_area) == pytest.approx(
        float(concrete.dual_area), rel=1e-12)
    assert float(traced.gamma) == pytest.approx(float(concrete.gamma), rel=1e-12)
    assert float(concrete.dual_area) == pytest.approx(
        _oracle_dual_area(grid, idx, component), rel=1e-12)


def test_uniform_lane_rlc_ade_output_does_not_move_under_the_dual_area_fold():
    """On a cubic ``Grid`` ``dual_area == d_par**2``, so every ADE observable
    must be unchanged.

    ``gamma`` itself moves by exactly 1 ULP in float64 — ``dt*d/(L*(d*d))``
    is 1.9065748695310059 where ``dt/(L*d)`` was 1.9065748695310056 — because
    the two algebraically identical spellings round differently and ``d`` is
    not a power of two.  The pins below are the OBSERVABLES after 200 driven
    steps, which are float32 and bit-identical across that move; they are the
    thing that must not change.
    """
    from rfx.core.yee import FDTDState
    from rfx.lumped import _update_parallel, _update_series, init_rlc_state

    grid = Grid(freq_max=1e10, domain=(2e-3, 2e-3, 2e-3), dx=0.2 * MM,
                cpml_layers=4)
    pos = (1e-3, 1e-3, 1e-3)
    idx = tuple(int(v) for v in grid.position_to_index(pos))

    def drive(topology, R, L, C, component, steps=200):
        spec = LumpedRLCSpec(R=R, L=L, C=C, topology=topology,
                             position=pos, component=component)
        mats = setup_rlc_materials(grid, spec, _mats(grid))
        meta = build_rlc_meta(grid, spec, mats)
        z = jnp.zeros(grid.shape)
        state = FDTDState(**{n: z for n in ("ex", "ey", "ez", "hx", "hy", "hz")},
                          step=jnp.asarray(0))
        rlc = init_rlc_state()
        step_fn = _update_series if meta.is_series else _update_parallel
        for n in range(steps):
            amp = np.sin(2.0 * np.pi * 5e9 * n * float(grid.dt))
            state = state._replace(
                **{component: getattr(state, component).at[idx].add(amp)})
            state, rlc = step_fn(state, rlc, meta)
        return (float(getattr(state, component)[idx]),
                float(rlc.inductor_current))

    assert drive("parallel", 0.0, 1e-9, 0.0, "ez") == (
        -0.10559134930372238, 6.394681690835569e-07)
    assert drive("parallel", 50.0, 1e-9, 1e-12, "ez") == (
        66.87545013427734, 0.0006415600073523819)
    assert drive("series", 50.0, 1e-9, 0.0, "ex") == (
        -0.6224570274353027, 6.808482453379838e-07)
    assert drive("series", 50.0, 1e-9, 1e-12, "ey") == (
        -0.3608303666114807, 6.59729153085209e-07)


# ---------------------------------------------------------------------------
# Site 4 — a TRACED mesh must fail LOUDLY, and that is a behaviour change
# ---------------------------------------------------------------------------

def test_a_traced_mesh_raises_instead_of_returning_a_zero_gradient():
    """INTENTIONAL behaviour change, gated so it is not "fixed" back.

    ``_axis_cell_sizes`` used to read the SCALAR ``grid.dx`` / ``grid.dy`` for
    the two transverse axes.  Those stay plain floats when the profile is
    traced, so a traced ``dx_profile`` with a concrete ``dz`` did not raise:
    measured at d62d0e0 (the commit before #691), ``jax.grad`` RETURNED, with
    a gradient of exactly 0.0 — while the value it returned was sized from
    the BOUNDARY cell rather than the cell at the port node.

    A wrong number carrying a silently zero gradient is worse than a crash:
    an optimiser reads it as a converged, insensitive design variable.  That
    is the #294 empty-window-gradient class.  So the limitation is unchanged
    — this path has always needed concrete spacings — but it is now loud, and
    the error says so rather than surfacing a raw
    ``TracerArrayConversionError`` from inside numpy.
    """
    import jax

    dz = np.array([0.2 * MM] * 10)
    base_dx = np.array([0.5 * MM] * 3 + [1.0 * MM] * 4 + [0.5 * MM] * 3)

    def objective(scale):
        profile = jnp.asarray(base_dx) * scale
        grid = make_nonuniform_grid(
            (0.0, 0.0), dz, float(base_dx[0]),
            dx_profile=profile, dy_profile=profile, cpml_layers=0,
            pec_faces={"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"},
        )
        return port_sigma(grid, (5, 5, 5), "ez", 50.0)

    with pytest.raises(NotImplementedError, match="CONCRETE per-cell mesh"):
        jax.grad(objective)(1.0)


def test_a_concrete_mesh_still_reaches_the_port_metric():
    """The guard must key on TRACING, not on the grid being non-uniform —
    otherwise it would break every concrete NU port."""
    grid = _graded_grid()
    assert port_sigma(grid, (5, 5, 5), "ez", 50.0) > 0.0
    assert port_d_parallel(grid, (5, 5, 5), "ez") > 0.0

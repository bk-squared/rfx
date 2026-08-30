"""MSL port axis generality — issue #661.

``add_msl_port`` used to accept ``direction`` of ``"+x"``/``"-x"`` only, so
a CAD-imported board whose feed enters along y had to be rotated. A
``MeshShape`` cannot be rotated (the import path takes no rotation
argument), so the workaround was unavailable exactly where it was needed.

What this file pins, and why each part exists
---------------------------------------------

**The measured axis inventory.** The MSL lane has three axis roles, not
one. Instrumenting the live extractor (rather than reading it) split the
code into:

* stages bound to the **propagation** axis -- the probe ladder, the DFT
  plane normal, the ``dx_feed`` in the port conductance, the port-to-CPML
  and probe-span preflight checks. All of these are relabelings.
* stages bound to the **substrate normal**, which is welded to z -- the
  static-Laplace cross-section solve and its ``ez_profile``, the ``"ez"``
  source component, the modal voltage ``V = sum(Ez*dz)``, and the
  trace-conductor PEC scan that walks up from the substrate top. The port
  geometry contract (``position`` + scalar ``height``) cannot name a
  different normal.
* stages that are genuinely axis-free -- the N-probe fit (it consumes
  probe coordinates and only their differences), the Hammerstad-Jensen
  anchor, and the multi-drive ``S = B A^-1`` solve.

Hence ``"+x"/"-x"/"+y"/"-y"`` are supported (a rotation about the
substrate normal) and ``"+z"/"-z"`` are REJECTED rather than
half-generalised -- accepting them would return a z-normal answer for a
board that is not oriented that way.

**The handedness.** The closed Ampere loop needs the right-handed
transverse pair ``(a, b)`` with ``a_hat x b_hat = p_hat``. Deriving it as
a plain ``x <-> y`` rename is a reflection, not a rotation, and it is the
dangerous failure here because it is SILENT. Measured on the committed
thru fixture's own recorded H planes:

* cyclic pair reproduces the x-port current to 9.2e-8 relative;
  naive swap returns exactly ``-I`` (ratio ``-1.00000009``);
* that flip exchanges ``a`` and ``b``, mapping ``S = B A^-1`` to
  ``A B^-1 = S^-1``; for the low-loss matched line every MSL fixture uses,
  ``S`` is nearly unitary so ``S^-1 ~ S^dagger``;
* consequence: ``|S11|`` moves 0.17905 -> 0.17875 and max
  ``||S| - |S_swapped||`` is 1.3e-3, column power stays below 1, and
  ``cond(A)`` is 1.32 -- **no guard in the lane fires**;
* but ``arg(S21)`` is exactly NEGATED (the two angles sum to <= 0.02 deg
  across the band -- a negative group delay), and the complex error is
  ``max |S - S_swapped| = 1.912``.

So the equivalence test below compares COMPLEX S. A magnitude-only
comparison would pass on the very bug it exists to catch.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.sources.msl_port import (
    MSL_SUPPORTED_DIRECTIONS,
    MSLPort,
    _msl_yz_cells,
    msl_ampere_pair,
    msl_axis_roles,
    msl_cross_section_span,
    msl_physical_point,
    msl_port_from_entry,
    msl_probe_x_coords_n,
    setup_msl_port,
)

EPS_R, H_SUB, W_TRACE = 3.66, 254e-6, 600e-6
L_LINE, PORT_MARGIN, DX, F_MAX = 10e-3, 2e-3, 80e-6, 5e9
L_PROP = L_LINE + 2 * PORT_MARGIN
L_LAT = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.5e-3


# ---------------------------------------------------------------------------
# 1. The direction contract
# ---------------------------------------------------------------------------


def test_supported_directions_resolve_to_three_distinct_axis_roles():
    assert MSL_SUPPORTED_DIRECTIONS == ("+x", "-x", "+y", "-y")
    assert msl_axis_roles("+x") == ("x", "y", "z", +1.0)
    assert msl_axis_roles("-x") == ("x", "y", "z", -1.0)
    assert msl_axis_roles("+y") == ("y", "x", "z", +1.0)
    assert msl_axis_roles("-y") == ("y", "x", "z", -1.0)
    # The substrate normal is z for EVERY supported direction. That is the
    # invariant which makes "+z"/"-z" inexpressible.
    for d in MSL_SUPPORTED_DIRECTIONS:
        assert msl_axis_roles(d)[2] == "z"


@pytest.mark.parametrize("direction", ["+z", "-z"])
def test_z_directions_are_rejected_and_say_why(direction):
    """A z-propagating microstrip needs a non-z substrate normal.

    Rejected rather than half-supported: the Laplace solve, the "ez"
    source, ``V = sum(Ez*dz)`` and the trace-PEC scan all reference the
    normal axis, so accepting "+z" would quietly answer for a board lying
    in a different plane than the user's.
    """
    with pytest.raises(ValueError) as ei:
        msl_axis_roles(direction)
    msg = str(ei.value)
    assert "substrate normal" in msg
    assert "+x" in msg and "+y" in msg


@pytest.mark.parametrize("direction", ["x", "y", "+w", "", None, 0])
def test_garbage_directions_raise_the_plain_domain_error(direction):
    with pytest.raises(ValueError, match="direction must be one of"):
        msl_axis_roles(direction)


# ---------------------------------------------------------------------------
# 2. Ampere-loop handedness — the silent-failure guard
# ---------------------------------------------------------------------------


def test_ampere_pair_is_the_right_handed_cyclic_pair_not_an_axis_swap():
    """``a_hat x b_hat`` must equal ``p_hat`` for every direction.

    Pinned as a cross-product identity rather than a literal table so the
    property, not the spelling, is what is protected. The reflection this
    forbids (``(x, z)`` for a y-propagating port) flips the sign of the
    extracted current and inverts S with no guard firing -- see the module
    docstring for the measured numbers.
    """
    basis = {"x": np.array([1.0, 0, 0]),
             "y": np.array([0, 1.0, 0]),
             "z": np.array([0, 0, 1.0])}
    for d in MSL_SUPPORTED_DIRECTIONS:
        prop, _w, _n, _s = msl_axis_roles(d)
        a_ax, b_ax = msl_ampere_pair(d)
        cross = np.cross(basis[a_ax], basis[b_ax])
        assert np.allclose(cross, basis[prop]), (
            f"direction {d!r}: ({a_ax},{b_ax}) is not right-handed about "
            f"{prop} -- cross = {cross}, expected {basis[prop]}. A "
            f"left-handed pair negates the Ampere-loop current and "
            f"inverts the S-matrix silently."
        )
    # And specifically: the y-port pair is NOT the naive (x, z) swap.
    assert msl_ampere_pair("+y") == ("z", "x")
    assert msl_ampere_pair("+y") != ("x", "z")


def test_loop_current_negates_on_the_direction_sign_only():
    """The raw contour is direction-INDEPENDENT; only the negation flips.

    Pins the corrected ``msl_loop_current`` docstring (issue #524). The
    contour is traversed right-handedly about ``+p_hat`` whatever the
    direction SIGN is, so the raw integral is the current along
    ``+p_hat`` for both signs and the only difference is the #140
    negation applied to positive-going ports.

    The docstring previously claimed without qualification that "the
    returned I is positive for a forward quasi-TEM wave"; measured on the
    committed thru fixture, each port on its own drive, that holds for
    ``"+x"`` (``Re((alpha-gamma)/I1) = +57.52 ohm``) and fails for
    ``"-x"`` (``-57.56 ohm``, same magnitude to 0.08%). The lane is
    self-consistent about it -- ``dir_sign`` restores both REPORTED Z0 to
    positive while the wave split consumes the un-normalised current --
    so this is a prose defect, not a code one, and the assertion below is
    what the corrected prose actually says.
    """
    from rfx.sources.msl_port import msl_loop_current

    rng = np.random.default_rng(6610)
    n_f, n_a, n_b = 3, 12, 9
    ha = rng.normal(size=(n_f, n_a, n_b)) + 1j * rng.normal(size=(n_f, n_a, n_b))
    hb = rng.normal(size=(n_f, n_a, n_b)) + 1j * rng.normal(size=(n_f, n_a, n_b))
    kw = dict(j_lo=3, j_hi=8, k_trace_lo=2, k_trace_hi=5,
              dy_arr=np.full(n_a, 8e-5), dz_arr=np.full(n_b, 8e-5))

    for pos, neg in (("+x", "-x"), ("+y", "-y")):
        i_pos = np.asarray(msl_loop_current(ha, hb, direction=pos, **kw))
        i_neg = np.asarray(msl_loop_current(ha, hb, direction=neg, **kw))
        assert np.allclose(i_pos, -i_neg, rtol=0, atol=0), (
            f"{pos}/{neg} must differ by exactly the #140 negation and "
            f"nothing else; got {i_pos} vs {i_neg}"
        )
        assert not np.allclose(i_pos, i_neg), "the negation must actually fire"

    # And the negation follows the SIGN, not the axis letter: the two
    # positive-going directions agree with each other on identical input,
    # as do the two negative-going ones.
    assert np.allclose(
        np.asarray(msl_loop_current(ha, hb, direction="+x", **kw)),
        np.asarray(msl_loop_current(ha, hb, direction="+y", **kw)),
    )


# ---------------------------------------------------------------------------
# 3. Geometry projection
# ---------------------------------------------------------------------------


def test_physical_point_places_each_role_on_its_own_axis():
    assert msl_physical_point("+x", 1.0, 2.0, 3.0) == (1.0, 2.0, 3.0)
    assert msl_physical_point("-x", 1.0, 2.0, 3.0) == (1.0, 2.0, 3.0)
    # For a y port the feed coordinate lands in the y slot and the trace
    # width centre in the x slot; the normal stays z.
    assert msl_physical_point("+y", 1.0, 2.0, 3.0) == (2.0, 1.0, 3.0)
    assert msl_physical_point("-y", 1.0, 2.0, 3.0) == (2.0, 1.0, 3.0)


def _grid(domain):
    sim = Simulation(freq_max=F_MAX, domain=domain, dx=DX, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    return sim._build_grid()


def test_cross_section_cells_span_width_and_normal_at_the_feed_plane():
    """The y-port cross-section is the x<->y image of the x-port one."""
    gx = _grid((L_PROP, L_LAT, LZ))
    gy = _grid((L_LAT, L_PROP, LZ))
    lat_c = L_LAT / 2.0
    px = MSLPort(feed_x=PORT_MARGIN, y_lo=lat_c - W_TRACE / 2,
                 y_hi=lat_c + W_TRACE / 2, z_lo=0.0, z_hi=H_SUB,
                 direction="+x", impedance=50.0)
    py = MSLPort(feed_x=PORT_MARGIN, y_lo=lat_c - W_TRACE / 2,
                 y_hi=lat_c + W_TRACE / 2, z_lo=0.0, z_hi=H_SUB,
                 direction="+y", impedance=50.0)
    cx = _msl_yz_cells(gx, px)
    cy = _msl_yz_cells(gy, py)
    assert len(cx) == len(cy) > 0
    # Same cells, with the first two index slots exchanged.
    assert [(j, i, k) for (i, j, k) in cx] == list(cy)
    sx = msl_cross_section_span(gx, px)
    sy = msl_cross_section_span(gy, py)
    for key in ("i_feed", "w_lo", "w_hi", "w_centre", "n_lo", "n_hi"):
        assert sx[key] == sy[key], key
    assert (sx["prop_idx"], sx["width_idx"]) == (0, 1)
    assert (sy["prop_idx"], sy["width_idx"]) == (1, 0)


def test_probe_ladder_steps_along_the_propagation_axis():
    gx = _grid((L_PROP, L_LAT, LZ))
    gy = _grid((L_LAT, L_PROP, LZ))
    lat_c = L_LAT / 2.0
    kw = dict(n_probes=5, n_offset_cells=16, n_spacing_cells=3)
    for d_fwd, d_rev in (("+x", "-x"), ("+y", "-y")):
        g = gx if d_fwd == "+x" else gy
        fwd = msl_probe_x_coords_n(
            g, MSLPort(PORT_MARGIN, lat_c - W_TRACE / 2, lat_c + W_TRACE / 2,
                       0.0, H_SUB, d_fwd, 50.0), **kw)
        rev = msl_probe_x_coords_n(
            g, MSLPort(PORT_MARGIN + L_LINE, lat_c - W_TRACE / 2,
                       lat_c + W_TRACE / 2, 0.0, H_SUB, d_rev, 50.0), **kw)
        assert all(np.diff(fwd) > 0), f"{d_fwd} ladder must increase"
        assert all(np.diff(rev) < 0), f"{d_rev} ladder must decrease"
    # The y ladder is the x ladder (same 1-D geometry along the feed axis).
    lx = msl_probe_x_coords_n(
        gx, MSLPort(PORT_MARGIN, lat_c - W_TRACE / 2, lat_c + W_TRACE / 2,
                    0.0, H_SUB, "+x", 50.0), **kw)
    ly = msl_probe_x_coords_n(
        gy, MSLPort(PORT_MARGIN, lat_c - W_TRACE / 2, lat_c + W_TRACE / 2,
                    0.0, H_SUB, "+y", 50.0), **kw)
    assert np.allclose(lx, ly)


def test_graded_mesh_probe_ladder_reads_its_own_axis_profile():
    """On a NonUniformGrid the y ladder must read ``dy_arr``, not ``dx_arr``.

    This is the y-axis instance of a bug class this repo has already paid
    for once: the transverse cell-profile helper used to fall through to
    the SCALAR boundary ``grid.dx`` for every cell, which is both the
    wrong axis and not per-cell. Built with DIFFERENT grading on x and y
    so reading the wrong array changes the answer -- on a mesh graded
    identically on both axes the two ladders coincide and the test would
    be vacuous.
    """
    from rfx.nonuniform import make_nonuniform_grid
    from rfx.api._sparams import _msl_cell_profile

    # make_nonuniform_grid requires the boundary cell to equal ``dx``, so
    # the refinement sits in the interior.
    dxp = np.concatenate([np.full(20, 5e-4), np.full(40, 2.5e-4),
                          np.full(20, 5e-4)])
    dyp = np.concatenate([np.full(15, 5e-4), np.full(30, 2.5e-4),
                          np.full(15, 5e-4)])
    g = make_nonuniform_grid(
        domain_xy=(float(np.sum(dxp)), float(np.sum(dyp))),
        dz_profile=np.full(24, 5e-4), dx=5e-4, cpml_layers=8,
        dx_profile=dxp, dy_profile=dyp,
        pec_faces={"z_lo"}, cpml_axes="xy",
    )
    # Per-axis profiles keep their own lengths (the NU branch is
    # authoritative; a scalar fallback would be the defect).
    for ax, n in (("x", g.nx), ("y", g.ny), ("z", g.nz)):
        assert _msl_cell_profile(g, ax, n).shape == (n,)

    kw = dict(n_probes=5, n_offset_cells=6, n_spacing_cells=3)
    mk = lambda d: MSLPort(feed_x=0.006, y_lo=0.008, y_hi=0.010, z_lo=0.0,
                           z_hi=0.002, direction=d, impedance=50.0)
    lad_x = np.asarray(msl_probe_x_coords_n(g, mk("+x"), **kw))
    lad_y = np.asarray(msl_probe_x_coords_n(g, mk("+y"), **kw))

    # Both are strictly increasing and in-domain...
    for tag, lad in (("+x", lad_x), ("+y", lad_y)):
        assert np.all(np.diff(lad) > 0), f"{tag} ladder not monotonic: {lad}"
    # ...and they DIFFER, because the two axes are graded differently.
    # Equality here would mean the y ladder walked the x profile.
    assert not np.allclose(lad_x, lad_y), (
        f"the y ladder reproduced the x ladder on a mesh graded differently "
        f"on the two axes ({lad_x} vs {lad_y}) -- it is reading dx_arr "
        f"instead of dy_arr"
    )


# ---------------------------------------------------------------------------
# 4. The anisotropic sigma witness
# ---------------------------------------------------------------------------


class _AnisoGrid:
    """Uniform-Grid duck type with independently settable per-axis spacing.

    Exists because a CUBIC grid cannot detect a propagation-axis mix-up:
    sigma scales as ``1/d_prop``, so on ``dx == dy`` an x-flavoured sigma
    applied to a y-port is bit-identical to the correct one. The
    end-to-end rotation-equivalence test below runs on a cubic mesh and is
    therefore BLIND to this stage -- this test is the one that sees it.
    """

    def __init__(self, dx, dy, dz, shape):
        self.dx = dx
        self.dx_profile = np.full(shape[0], dx)
        self.dy_profile = np.full(shape[1], dy)
        self.dz_profile = np.full(shape[2], dz)
        self.shape = shape
        self.nx, self.ny, self.nz = shape
        self.dt = 1e-13

    def position_to_index(self, pos):
        return (int(round(pos[0] / self.dx_profile[0])),
                int(round(pos[1] / self.dy_profile[0])),
                int(round(pos[2] / self.dz_profile[0])))


class _Mat:
    def __init__(self, shape):
        import jax.numpy as jnp
        self.sigma = jnp.zeros(shape)
        self.eps_r = jnp.ones(shape)

    def _replace(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self


def _sigma_sum(direction, d_prop, d_width, d_norm):
    """Total added sigma with cell sizes assigned BY ROLE."""
    shape = (48, 48, 20)
    sizes = {"z": d_norm}
    prop, width, _n, _s = msl_axis_roles(direction)
    sizes[prop] = d_prop
    sizes[width] = d_width
    g = _AnisoGrid(sizes["x"], sizes["y"], sizes["z"], shape)
    port = MSLPort(feed_x=10 * d_prop, y_lo=10 * d_width, y_hi=16 * d_width,
                   z_lo=0.0, z_hi=3 * d_norm, direction=direction,
                   impedance=50.0)
    out = setup_msl_port(g, port, _Mat(shape))
    return float(np.sum(np.asarray(out.sigma)))


def test_port_conductance_uses_the_propagation_axis_cell_size():
    """sigma must scale as 1/d_prop for a y port exactly as for an x port.

    If the y lane still read ``dx`` where it should read ``dy``, these two
    would disagree by ``d_width/d_prop`` = 2. On a cubic grid they agree
    either way -- which is precisely why this test is not cubic.
    """
    base_x = _sigma_sum("+x", 80e-6, 80e-6, 80e-6)
    base_y = _sigma_sum("+y", 80e-6, 80e-6, 80e-6)
    assert base_x == pytest.approx(base_y, rel=1e-12)

    # Halve the PROPAGATION cell only: sigma must double, both lanes.
    for d in ("+x", "-x", "+y", "-y"):
        got = _sigma_sum(d, 40e-6, 80e-6, 80e-6)
        assert got == pytest.approx(2.0 * base_x, rel=1e-12), (
            f"{d}: sigma did not scale as 1/d_prop (got {got}, expected "
            f"{2.0 * base_x}) -- the port conductance is reading the wrong "
            f"axis's cell size."
        )
    # Halve the WIDTH cell only: sigma must double as well (it is the
    # parallel-cell count that changes), and NORMAL halving must halve it.
    for d in ("+x", "+y"):
        assert _sigma_sum(d, 80e-6, 40e-6, 80e-6) == pytest.approx(
            2.0 * base_x, rel=1e-12)
        assert _sigma_sum(d, 80e-6, 80e-6, 40e-6) == pytest.approx(
            0.5 * base_x, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Preflight fires on the correct axis per direction
# ---------------------------------------------------------------------------


def _msl_warnings(sim):
    """MSL-port advisories from ``preflight()`` (same helper shape as
    ``tests/test_msl_port_preflight.py``: the report is a list of str)."""
    return [m for m in sim.preflight() if "MSL port" in m]


def _board(domain, direction, feed, lat_c, *, trace_len_axis, dx=DX):
    """Thru-line board with one port; ``feed``/``lat_c`` are port-frame.
    ``dx`` defaults to the module DX (80 um); the substrate-cell test passes
    100 um, where the run grid really has fewer than 4 substrate cells."""
    sim = Simulation(freq_max=F_MAX, domain=domain, dx=dx, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (domain[0], domain[1], H_SUB)),
            material="ro4350b")
    if trace_len_axis == "x":
        lo = (0.0, lat_c - W_TRACE / 2, H_SUB)
        hi = (domain[0], lat_c + W_TRACE / 2, H_SUB + dx)
    else:
        lo = (lat_c - W_TRACE / 2, 0.0, H_SUB)
        hi = (lat_c + W_TRACE / 2, domain[1], H_SUB + dx)
    sim.add(Box(lo, hi), material="pec")
    sim.add_msl_port(
        position=msl_physical_point(direction, feed, lat_c, 0.0),
        width=W_TRACE, height=H_SUB, direction=direction, impedance=50.0,
        n_probe_offset=16,
    )
    return sim


def test_lateral_clearance_check_fires_on_the_width_axis_for_a_y_port():
    """Check 1 measures across the TRACE WIDTH axis — x for a y port.

    Built so an axis mix-up is visible: the width axis is deliberately
    starved while the propagation axis is generous, so a check still
    looking at ``domain[1]`` would stay silent.
    """
    tight = W_TRACE + 2 * (0.2 * H_SUB)      # far under the 2*h_sub rule
    sim = _board((tight, L_PROP, LZ), "+y", PORT_MARGIN, tight / 2.0,
                 trace_len_axis="y")
    msgs = [m for m in _msl_warnings(sim) if "lateral clearance" in m]
    assert msgs, "lateral-clearance check did not fire on the width axis"
    joined = " ".join(msgs)
    assert "−x" in joined or "+x" in joined, (
        f"clearance warning must name the x (width) sides for a y port: "
        f"{joined}"
    )
    assert "domain x-extent" in joined
    # And it must NOT fire when the width axis is generous, even though
    # the propagation axis is now the narrow one.
    ok = _board((L_LAT, L_PROP, LZ), "+y", PORT_MARGIN, L_LAT / 2.0,
                trace_len_axis="y")
    assert not [m for m in _msl_warnings(ok) if "lateral clearance" in m]


def test_port_to_cpml_check_fires_on_the_propagation_axis_for_a_y_port():
    """Check 3 measures along the PROPAGATION axis — y for a y port."""
    # Feed pressed right up against the y_lo wall; the x extent is fine.
    sim = _board((L_LAT, L_PROP, LZ), "+y", 0.2 * H_SUB, L_LAT / 2.0,
                 trace_len_axis="y")
    msgs = [m for m in _msl_warnings(sim) if "distance to nearest" in m]
    assert msgs, "port-to-CPML check did not fire on the propagation axis"
    joined = " ".join(msgs)
    assert "y-CPML" in joined, f"must name the y CPML for a +y port: {joined}"
    assert "domain y-extent" in joined
    # A "-y" port is bounded by the OPPOSITE wall: the same tight feed
    # coordinate is now comfortably clear.
    ok = _board((L_LAT, L_PROP, LZ), "-y", 0.2 * H_SUB, L_LAT / 2.0,
                trace_len_axis="y")
    assert not [m for m in _msl_warnings(ok) if "distance to nearest" in m]


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
def test_h_sub_alignment_checks_fire_for_every_direction(direction):
    """Checks 2 and 2b (substrate cell count, mixed-cell danger zone).

    These are the one preflight family that must NOT move axis: h_sub is
    measured along the substrate normal, and the normal is welded to z for
    every supported direction. So the correct behaviour is that both fire
    identically regardless of ``direction`` -- an implementation that
    "generalised" them onto the propagation axis would be wrong, and would
    stop reporting the substrate resolution for a y port.

    Issue #752 / #766: the checks now count the substrate cells the RUN
    GRID has, read off the assembled permittivity under the port. At
    dx = 80 um the 254 um substrate REALIZES 4 cells (320 um), so check 2
    ("< 4 cells") is correctly silent there and only check 2b fires (the
    declared top sits 0.175 of a cell above a node). The genuine < 4-cell
    case is dx = 100 um (3 cells, 300 um). Both are exercised, on every
    direction, and the realized numbers must be identical across
    directions -- h_sub does not depend on the propagation axis.
    """
    prop, width, _n, _s = msl_axis_roles(direction)
    domain = [0.0, 0.0, LZ]
    domain[{"x": 0, "y": 1}[prop]] = L_PROP
    domain[{"x": 0, "y": 1}[width]] = L_LAT

    # dx = 80 um: 2b fires, 2 must not (the run grid has 4 substrate cells).
    sim80 = _board(tuple(domain), direction, PORT_MARGIN, L_LAT / 2.0,
                   trace_len_axis=prop)
    msgs80 = _msl_warnings(sim80)
    cells80 = [m for m in msgs80 if "substrate cell(s) in z" in m]
    frac80 = [m for m in msgs80 if "mixed-cell danger zone" in m]
    assert cells80 == [], f"4 realized cells must not trip check 2 for {direction}: {cells80}"
    assert frac80, f"mixed-cell check silent for {direction}: {msgs80}"
    assert "sits 0.175 of a cell above the nearest mesh node" in frac80[0], frac80[0]
    assert "4 cell(s) of substrate = 320µm" in frac80[0], frac80[0]

    # dx = 100 um: the run grid has 3 substrate cells -> check 2 fires.
    sim100 = _board(tuple(domain), direction, PORT_MARGIN, L_LAT / 2.0,
                    trace_len_axis=prop, dx=100e-6)
    msgs100 = _msl_warnings(sim100)
    cells100 = [m for m in msgs100 if "substrate cell(s) in z" in m]
    assert cells100, f"substrate-resolution check silent for {direction}: {msgs100}"
    # Same numbers on every axis -- h_sub does not depend on direction.
    assert "only 3 substrate cell(s) in z" in cells100[0], cells100[0]
    assert "actually realizes 3 cell(s) = 300µm" in cells100[0], cells100[0]


def test_probe_span_absorber_check_fires_on_the_propagation_axis():
    """Check 4a walks the probe ladder along the propagation axis."""
    sim = Simulation(freq_max=F_MAX, domain=(L_LAT, L_PROP, LZ), dx=DX,
                     cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (L_LAT, L_PROP, H_SUB)), material="ro4350b")
    sim.add(Box((L_LAT / 2 - W_TRACE / 2, 0.0, H_SUB),
                (L_LAT / 2 + W_TRACE / 2, L_PROP, H_SUB + DX)), material="pec")
    # Ladder deliberately long enough to run off the far y edge.
    sim.add_msl_port(position=(L_LAT / 2, L_PROP - 1e-3, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+y", impedance=50.0,
                     n_probe_offset=20, n_probe_spacing=6)
    msgs = [m for m in _msl_warnings(sim)
            if "deepest" in m and "domain" in m]
    assert msgs, "probe-span absorber check did not fire"
    assert "domain y-extent" in " ".join(msgs)


# ---------------------------------------------------------------------------
# 6. The falsifier: rotation equivalence end to end
# ---------------------------------------------------------------------------


def _thru(axis, n_freqs, num_periods):
    if axis == "x":
        domain, sub = (L_PROP, L_LAT, LZ), (L_PROP, L_LAT, H_SUB)
        lat_c = L_LAT / 2.0
        tlo = (0.0, lat_c - W_TRACE / 2, H_SUB)
        thi = (L_PROP, lat_c + W_TRACE / 2, H_SUB + DX)
        p0, p1, d0, d1 = ((PORT_MARGIN, lat_c, 0.0),
                          (PORT_MARGIN + L_LINE, lat_c, 0.0), "+x", "-x")
    else:
        domain, sub = (L_LAT, L_PROP, LZ), (L_LAT, L_PROP, H_SUB)
        lat_c = L_LAT / 2.0
        tlo = (lat_c - W_TRACE / 2, 0.0, H_SUB)
        thi = (lat_c + W_TRACE / 2, L_PROP, H_SUB + DX)
        p0, p1, d0, d1 = ((lat_c, PORT_MARGIN, 0.0),
                          (lat_c, PORT_MARGIN + L_LINE, 0.0), "+y", "-y")
    sim = Simulation(freq_max=F_MAX, domain=domain, dx=DX, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), sub), material="ro4350b")
    sim.add(Box(tlo, thi), material="pec")
    sim.add_msl_port(position=p0, width=W_TRACE, height=H_SUB,
                     direction=d0, impedance=50.0)
    sim.add_msl_port(position=p1, width=W_TRACE, height=H_SUB,
                     direction=d1, impedance=50.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(n_freqs=n_freqs,
                                       num_periods=num_periods)
    return res, [str(c.message) for c in caught]


@pytest.mark.slow
def test_y_directed_thru_reproduces_the_x_directed_thru():
    """THE FALSIFIER for issue #661.

    Identical physical thru line, built once with the feed along x and
    once on the x<->y mirrored fixture with the feed along y. Same
    structure, same mesh pitch, so every extracted quantity must agree to
    the extraction floor. If it does not, some part of the extractor was
    x-specific after all -- and that finding matters more than the
    feature.

    Compared on COMPLEX S: see the module docstring for why a
    magnitude-only comparison passes on the exact bug this guards.

    Measured (12 bins, num_periods=12, dx=80um, CPU float32)::

        max |S_x - S_y|        3.86e-06   (rel 3.86e-06)
        rel max |Z0_x - Z0_y|  2.14e-04
        rel max |beta_x-beta_y| 3.09e-05
        settling_db  x = [-98.47, -102.96]   y = [-98.49, -102.91]

    The three bounds below are NOT one number, because the three
    quantities do not share a conditioning class -- the measured spread
    is a ladder, and the ladder has a cause:

    * ``S`` is a V*I split at one plane plus a 2x2 ``B A^-1`` solve --
      well conditioned, lands at the float32 reassociation floor;
    * ``beta`` comes from the N-probe scan, which
      ``rfx/probes/msl_wave_decomp.py`` runs in float32 by explicit cast
      (``_estimate_beta``'s own comment records that the residual curve
      is "numerically flat in float32");
    * ``Z0 = (alpha - gamma) / I`` compounds that fit's residual through a
      DIFFERENCE of two fitted amplitudes, so it sits an order above
      beta.

    Each bound is the measured value with roughly an order of margin, and
    all three are >= 3 orders below the failure signature they exist to
    catch (the swap measures ``max|dS| = 1.912``). ``test_equivalence_
    harness_can_move`` is the negative control for that claim.
    """
    rx, wx = _thru("x", n_freqs=12, num_periods=12)
    ry, wy = _thru("y", n_freqs=12, num_periods=12)

    # Preflight output is part of the result (repo rule): the two lanes
    # must raise the SAME advisories. Normalised for (a) axis letters,
    # (b) numeric literals -- the reported Z0 differs in its last printed
    # digit (57.61 vs 57.60 ohm) purely from the float32 fit above, and
    # (c) one-shot import-time DeprecationWarnings, which fire only for
    # whichever lane happens to run first in the process.
    import re

    def _canon(ms):
        out = []
        for m in ms:
            if "deprecated" in m.lower():
                continue
            m = (m.replace("y-CPML", "@-CPML").replace("x-CPML", "@-CPML")
                  .replace("domain x-extent", "domain @-extent")
                  .replace("domain y-extent", "domain @-extent"))
            out.append(re.sub(r"[-+]?\d[\d.,]*(?:[eE][-+]?\d+)?", "#", m))
        return sorted(out)

    cx, cy = _canon(wx), _canon(wy)
    assert cx == cy, (
        "the two lanes produced different advisories:\n"
        f"only in x: {[m for m in cx if m not in cy]}\n"
        f"only in y: {[m for m in cy if m not in cx]}"
    )

    Sx, Sy = np.asarray(rx.S), np.asarray(ry.S)
    Zx, Zy = np.asarray(rx.Z0), np.asarray(ry.Z0)
    Bx, By = np.asarray(rx.beta), np.asarray(ry.beta)
    assert np.allclose(rx.freqs, ry.freqs)

    d_s = float(np.max(np.abs(Sx - Sy)))
    d_z = float(np.max(np.abs(Zx - Zy)) / max(np.max(np.abs(Zx)), 1e-30))
    d_b = float(np.max(np.abs(Bx - By)) / max(np.max(np.abs(Bx)), 1e-30))
    print(f"\n[#661 equivalence] max|dS|={d_s:.3e} rel dZ0={d_z:.3e} "
          f"rel dbeta={d_b:.3e}")
    print(f"[#661 equivalence] settling_db x={np.asarray(rx.settling_db)} "
          f"y={np.asarray(ry.settling_db)}")

    # Ring-down settling witness: fixed-length open-domain records must
    # quote end/peak energy before any claims-bearing number.
    for tag, r in (("x", rx), ("y", ry)):
        sd = np.asarray(r.settling_db, dtype=float)
        assert np.all(sd < -40.0), (
            f"{tag} lane under-settled (settling_db={sd}); the equivalence "
            f"comparison would be reading truncation, not physics"
        )

    assert d_s < 1e-4, f"complex S disagreement {d_s:.3e} exceeds the floor"
    assert d_z < 3e-3, f"Z0 disagreement {d_z:.3e} exceeds the fit floor"
    assert d_b < 1e-3, f"beta disagreement {d_b:.3e} exceeds the fit floor"


@pytest.mark.slow
def test_equivalence_harness_can_move():
    """Negative control for the falsifier above.

    A rotation-equivalence assertion that compares two runs is worthless
    if the comparison cannot register a difference. Perturb the y lane's
    trace width by one mesh cell and confirm the same statistic moves far
    past the 1e-4 bound -- so a PASS above is evidence, not a tautology.
    """
    rx, _ = _thru("x", n_freqs=6, num_periods=8)

    lat_c = L_LAT / 2.0
    w_bad = W_TRACE + DX
    sim = Simulation(freq_max=F_MAX, domain=(L_LAT, L_PROP, LZ), dx=DX,
                     cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (L_LAT, L_PROP, H_SUB)), material="ro4350b")
    sim.add(Box((lat_c - w_bad / 2, 0.0, H_SUB),
                (lat_c + w_bad / 2, L_PROP, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(lat_c, PORT_MARGIN, 0.0), width=w_bad,
                     height=H_SUB, direction="+y", impedance=50.0)
    sim.add_msl_port(position=(lat_c, PORT_MARGIN + L_LINE, 0.0), width=w_bad,
                     height=H_SUB, direction="-y", impedance=50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_bad = sim.compute_msl_s_matrix(n_freqs=6, num_periods=8)

    d = float(np.max(np.abs(np.asarray(rx.S) - np.asarray(r_bad.S))))
    print(f"\n[#661 negative control] one-cell wider trace -> max|dS|={d:.3e}")
    assert d > 1e-3, (
        f"the equivalence statistic moved only {d:.3e} for a one-cell "
        f"geometry change -- it is too blunt to be evidence"
    )

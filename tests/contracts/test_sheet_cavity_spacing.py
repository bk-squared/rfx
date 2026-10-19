"""Build-only #1800: spacing, not absolute sheet position, decides refusal."""
from tests._wall_axis_declaration import assert_declared_wall_warning
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.preflight._common import _fmt_len
from tests.contracts.test_sheet_size_verdict import _ReachedRunner, _stub_runners

CODE = "sheet_cavity_spacing"
MM = 1e-3
DOMAIN = (20 * MM, 18 * MM, 20 * MM)
# Both meshes have nodes at 4 and 6 mm. The graded mesh replaces [4,5,6]
# with [4,4.8,6]; all other cells are 1 mm. These are input coordinates,
# independent of the cavity validator.
NODES = {
    "uniform": np.arange(21) * MM,
    "graded": np.array([0, 1, 2, 3, 4, 4.8, *range(6, 21)]) * MM,
}
DRAWINGS = {
    "exact": (4.0, 6.0),
    "over": (4.0, 6.1),       # nearest nodes 4,6: 2/2.1 - 1 = -4.7619%.
    "under": (4.0, 6.01),     # nearest nodes 4,6: 2/2.01 - 1 = -0.4975%.
    "shifted": (4.04, 6.04),  # both shift -0.04; 2/2 - 1 = 0%.
}


def _build(mesh, case="over", *, snap="strict", thickness=False):
    lower, upper = np.array(DRAWINGS[case]) * MM
    profiles = {"dz_profile": np.diff(NODES[mesh])} if mesh == "graded" else {}
    sim = Simulation(10e9, DOMAIN, dx=MM, boundary="pec", snap=snap, **profiles)
    sim.add_material("lower", sigma=1e10)
    sim.add_material("upper", sigma=1e10)
    # Free in-plane edges lie 0.35 cell beyond their outer covered nodes,
    # so the existing in-plane size verdict accepts these unequal sheets.
    lo = Box((2.65 * MM, 2.65 * MM, lower - (0.04 * MM if thickness else 0)),
             (16.35 * MM, 14.35 * MM, lower + (0.04 * MM if thickness else 0)))
    hi = Box((4.65 * MM, 4.65 * MM, upper),
             (12.35 * MM, 10.35 * MM, upper))
    if thickness:
        sim.add_thin_conductor(lo, thickness=0.08 * MM)
    else:
        sim.add(lo, material="lower")
    sim.add(hi, material="upper")
    sim.add_source((MM, MM, MM), component="ez")
    return sim


def _spacing(mesh, case):
    drawn_planes = np.array(DRAWINGS[case]) * MM
    nodes = NODES[mesh]
    nearest = nodes[np.abs(nodes[:, None] - drawn_planes).argmin(axis=0)]
    drawn = drawn_planes[1] - drawn_planes[0]
    solved = nearest[1] - nearest[0]
    return drawn, solved, nearest


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_strict_refuses_spacing_before_runner(monkeypatch, mesh, entry):
    calls = _stub_runners(monkeypatch)
    sim = _build(mesh)
    report = sim.preflight()
    row, = report.by_code(CODE)
    drawn, solved, nearest = _spacing(mesh, "over")
    assert solved / drawn - 1 < -0.01
    assert row.severity == "error"
    assert not report.ok
    message = str(row)
    assert "'lower'" in message and "'upper'" in message
    assert f"drawn spacing {_fmt_len(drawn)}" in message
    assert f"solved spacing {_fmt_len(solved)}" in message
    for z in nearest:
        assert f"plane at {_fmt_len(z)}" in message
    assert 'Simulation(..., snap="declared")' in message
    assert "put both planes on mesh nodes" in message
    assert "dz_profile=" in message
    kwargs = {"compute_s_params": False} if entry == "run" else {}
    with pytest.raises(ValueError, match="parallel PEC cavity faces"):
        getattr(sim, entry)(n_steps=1, **kwargs)
    assert not calls


@pytest.mark.parametrize("mesh", NODES)
def test_declared_warns_and_reaches_runner(monkeypatch, mesh):
    calls = _stub_runners(monkeypatch)
    sim = _build(mesh, snap="declared")
    report = sim.preflight()
    row, = report.by_code(CODE)
    assert row.severity == "warning"
    assert not any(r.severity == "error" for r in report.by_code(CODE))
    assert report.ok
    assert report.by_code("sheet_cavity_electrical_thickness")
    with pytest.raises(_ReachedRunner):
        sim.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("case", ["exact", "under", "shifted"])
def test_strict_accepts_spacing(monkeypatch, mesh, case):
    calls = _stub_runners(monkeypatch)
    sim = _build(mesh, case)
    drawn, solved, nearest = _spacing(mesh, case)
    assert abs(solved / drawn - 1) < 0.01
    if case == "shifted":
        shifts = np.abs(nearest - np.array(DRAWINGS[case]) * MM)
        assert np.all(shifts / drawn > 0.01)
        assert np.all(shifts / DOMAIN[2] < 0.005)
    report = sim.preflight()
    assert not report.by_code(CODE)
    assert report.ok
    with pytest.raises(_ReachedRunner):
        sim.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


@pytest.mark.parametrize("mesh", NODES)
def test_drawn_thickness_names_interface_remedy(mesh):
    sim = _build(mesh, "exact", thickness=True)
    entries = sim._campaign_ctx().interior_pec_entries()
    assert [e.kind for e in entries] == ["sheet", "sheet"]
    row, = sim.preflight().by_code(CODE)
    # Lower declared upper face 4.04 mm, upper lower face 6 mm;
    # the thin conductor's midpoint is 4 mm, so solved spacing is 2 mm.
    assert "drawn spacing 1.96mm, solved spacing 2mm (+2.04%)" in str(row)
    assert "its drawn thickness counts into the spacing" in str(row)
    assert "zero-thickness sheet on the interface it bounds" in str(row)


@pytest.mark.parametrize("mesh", NODES)
def test_spacing_is_checked_when_electrical_measures_agree(mesh):
    """Two sampled permittivities hide the geometric error in BOTH measures.

    In mm, the mesh intervals have lengths d0,d1 and eps=1,4. Thus
    C=d0+d1/4 and P=d0+2*d1. A homogeneous physical length L=(C*P²)^(1/3)
    at eps=(P/L)² has exactly the same capacitance and phase measures.
    Retain 10% of each interval's material in narrow boxes on its sample
    node; fill the remaining physical length with that homogeneous eps.
    Drawn spacing is 0.1*(d0+d1)+0.9*L, while solved spacing stays 2 mm.
    """
    nodes_mm = NODES[mesh] / MM
    between = nodes_mm[(nodes_mm >= 4) & (nodes_mm <= 6)]
    d0, d1 = np.diff(between)
    cap, phase = d0 + d1 / 4, d0 + 2 * d1
    length = (cap * phase**2) ** (1 / 3)
    eps = (phase / length)**2
    drawn = 0.1 * (d0 + d1) + 0.9 * length
    assert abs(2 / drawn - 1) > 0.01
    profiles = {"dz_profile": np.diff(NODES[mesh])} if mesh == "graded" else {}
    sim = Simulation(10e9, DOMAIN, dx=MM, boundary="pec", **profiles)
    sim.add_material("background", eps_r=eps)
    sim.add_material("sample_low", eps_r=1)
    sim.add_material("sample_high", eps_r=4)
    for lo, hi, material in (
        (4, 4 + drawn, "background"),
        (4, 4 + 0.1 * d0, "sample_low"),
        (4 + d0, 4 + d0 + 0.1 * d1, "sample_high"),
    ):
        sim.add(Box((0, 0, lo * MM), (DOMAIN[0], DOMAIN[1], hi * MM)),
                material=material)
    sim.add(Box((2.65 * MM, 2.65 * MM, 4 * MM),
                (16.35 * MM, 14.35 * MM, 4 * MM)), material="pec")
    sim.add(Box((4.65 * MM, 4.65 * MM, (4 + drawn) * MM),
                (12.35 * MM, 10.35 * MM, (4 + drawn) * MM)), material="pec")
    report = sim.preflight()
    assert not report.by_code("sheet_cavity_electrical_thickness")
    row, = report.by_code(CODE)
    assert row.severity == "error"
    assert f"drawn spacing {_fmt_len(drawn * MM)}, solved spacing 2mm" in str(row)


# Review probes 2/3: keep the drawings explicit and independent of the census.
_LO_XY = (2.65 * MM, 2.65 * MM, 16.35 * MM, 14.35 * MM)
_HI_XY = (4.65 * MM, 4.65 * MM, 12.35 * MM, 10.35 * MM)


def _review_base(mesh, snap):
    profiles = {"dz_profile": np.diff(NODES[mesh])} if mesh == "graded" else {}
    sim = Simulation(10e9, DOMAIN, dx=MM, boundary="pec", snap=snap, **profiles)
    for name in ("lower", "upper", "via"):
        sim.add_material(name, sigma=1e10)
    sim.add_source((MM, MM, MM), component="ez")
    return sim


def _review_sheet(sim, xy, lo, hi, name):
    shape = Box((xy[0], xy[1], lo * MM), (xy[2], xy[3], hi * MM))
    if lo == hi:
        sim.add(shape, material=name)
    else:
        sim.add_thin_conductor(shape)


def _assert_review_verdict(monkeypatch, sim, entry, snap, fragment):
    calls = _stub_runners(monkeypatch)
    report = sim.preflight()
    row, = report.by_code(CODE)
    assert row.severity == ("error" if snap == "strict" else "warning")
    assert report.ok == (snap == "declared")
    assert fragment in str(row)
    kwargs = {"compute_s_params": False} if entry == "run" else {}
    if snap == "strict":
        with pytest.raises(ValueError, match="parallel PEC cavity faces"):
            getattr(sim, entry)(n_steps=1, **kwargs)
        assert not calls
    else:
        with pytest.raises(_ReachedRunner):
            getattr(sim, entry)(n_steps=1, **kwargs)
        assert calls == [True]
    return str(row)


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("heights", [(4.6, 5.4), (4.85, 5.15)])
@pytest.mark.parametrize("snap", ["strict", "declared"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_collapsed_sheets_are_not_skipped(monkeypatch, mesh, heights, snap, entry):
    """Probe2: both drawings snap to 5 mm, or 4.8 mm on the graded mesh.

    At 5.4 mm the graded 4.8/6 mm tie resolves to the lower node. Add the
    upper drawing first to also guard drawn-order, rather than entry-order,
    adjacency when the solved planes coincide.
    """
    sim = _review_base(mesh, snap)
    lo, hi = heights
    _review_sheet(sim, _HI_XY, hi, hi, "upper")
    _review_sheet(sim, _LO_XY, lo, lo, "lower")
    expected_node = (5 if mesh == "uniform" else 4.8) * MM
    drawn = (hi - lo) * MM
    message = _assert_review_verdict(
        monkeypatch, sim, entry, snap,
        f"drawn spacing {_fmt_len(drawn)}, solved spacing 0mm (-100.00%)")
    assert message.split(" | ")[0].count(f"plane at {_fmt_len(expected_node)}") == 2
    assert "domain wall" in message
    assert message.index("'lower'") < message.index("'upper'")


def _thin_substrate(mesh, snap):
    """Probe3's h=0.2 mm, dx=0.5 mm patch over a ground at z=2 mm."""
    nodes = np.arange(13, dtype=float) * 0.5 * MM
    if mesh == "graded":
        nodes[9] = 4.4 * MM  # Nonuniform away from the two drawing planes.
    profiles = {"dz_profile": np.diff(nodes)} if mesh == "graded" else {}
    sim = Simulation(10e9, (30 * MM, 30 * MM, 6 * MM), dx=0.5 * MM,
                     boundary="pec", snap=snap, **profiles)
    sim.add_material("sub", eps_r=3.5)
    sim.add(Box((0, 0, 2 * MM), (30 * MM, 30 * MM, 2.2 * MM)), material="sub")
    # 0.35 * 0.5 mm beyond the outer covered in-plane nodes, as in probe3.
    _review_sheet(sim, (2.825 * MM, 2.825 * MM, 27.175 * MM, 27.175 * MM),
                  2, 2, "pec")
    _review_sheet(sim, (7.825 * MM, 7.825 * MM, 22.175 * MM, 22.175 * MM),
                  2.2, 2.2, "pec")
    sim.add_source((15 * MM, 15 * MM, 2 * MM), component="ez")
    return sim


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("snap", ["strict", "declared"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_thin_substrate_collapses_to_ground(monkeypatch, mesh, snap, entry):
    # 2.0 and 2.2 are both nearest to the 2.0 mm node; gap 0.2 -> 0 mm.
    sim = _thin_substrate(mesh, snap)
    message = _assert_review_verdict(monkeypatch, sim, entry, snap,
                                    "drawn spacing 200µm, solved spacing 0mm (-100.00%)")
    assert message.split(" | ")[0].count("plane at 2mm") == 2
    assert "domain wall z_hi" in message


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("order", ["patch_first", "post_first"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_partial_post_does_not_hide_cavity(monkeypatch, mesh, order, entry):
    """Probe2: the 8..9 x 7..8 mm post blocks the representative column only."""
    sim = _review_base(mesh, "strict")
    _review_sheet(sim, _LO_XY, 4, 4, "lower")
    post = Box((8 * MM, 7 * MM, 4 * MM), (9 * MM, 8 * MM, 6.1 * MM))
    if order == "post_first":
        sim.add(post, material="via")
    _review_sheet(sim, _HI_XY, 6.1, 6.1, "upper")
    if order == "patch_first":
        sim.add(post, material="via")
    # Both node arrays contain 4 and 6 mm. 2/2.1 - 1 = -4.7619%.
    _assert_review_verdict(monkeypatch, sim, entry, "strict",
                           "drawn spacing 2.1mm, solved spacing 2mm (-4.76%)")


@pytest.mark.parametrize("mesh", NODES)
def test_every_overlap_column_blocked_is_skipped(mesh):
    sim = _review_base(mesh, "strict")
    _review_sheet(sim, _LO_XY, 4, 4, "lower")
    _review_sheet(sim, _HI_XY, 6.1, 6.1, "upper")
    # Upper footprint nodes are x=5..12, y=5..10. Every corresponding
    # column has a PEC cell between 4 and 6 mm, unlike the single post.
    sim.add(Box((4 * MM, 4 * MM, 4 * MM), (13 * MM, 11 * MM, 6 * MM)),
            material="via")
    # The filled sheet-sheet columns stay skipped; the block's upper face
    # now bounds a separate 0.1 mm cavity that collapses onto the upper sheet.
    row, = sim.preflight().by_code(CODE)
    assert row.severity == "error"
    assert "1 pair(s)" in str(row)
    assert "'via' (PEC Box face upper)" in str(row)
    assert "drawn spacing 100µm, solved spacing 0mm (-100.00%)" in str(row)


@pytest.mark.parametrize("mesh", NODES)
def test_every_overlap_column_blocked_by_a_non_box_post_is_skipped(mesh):
    # Same sheets (4 and 6.1 mm, solved 2 mm apart: -4.76 % if judged). A
    # Cylinder (r 5.5 mm about (8.5, 7.5) mm, z 4..6 mm) fills every overlap
    # column x=5..12, y=5..10 (farthest cell centre 5 mm off its axis) and,
    # not being a Box, adds no judged face: the pair is skipped, no row.
    from rfx.geometry.csg import Cylinder

    sim = _review_base(mesh, "strict")
    _review_sheet(sim, _LO_XY, 4, 4, "lower")
    _review_sheet(sim, _HI_XY, 6.1, 6.1, "upper")
    sim.add(Cylinder((8.5 * MM, 7.5 * MM, 5 * MM), 5.5 * MM, 2 * MM), material="via")
    assert not sim.preflight().by_code(CODE)


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("upper_lo", [4.4, 4.3], ids=["touching", "overlapping"])
@pytest.mark.parametrize("snap", ["strict", "declared"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_continuous_foils_solved_apart(monkeypatch, mesh, upper_lo, snap, entry):
    """Probe2 touching foil drawing, plus its negative-drawn-gap neighbour."""
    sim = _review_base(mesh, snap)
    _review_sheet(sim, _LO_XY, 4.0, 4.4, "lower")
    # Midpoint 4.6 mm in both cases: 4.4..4.8 or 4.3..4.9.
    _review_sheet(sim, _HI_XY, upper_lo, 9.2 - upper_lo, "upper")
    # Lower midpoint 4.2 -> node 4; upper 4.6 -> 5 (uniform), 4.8 (graded).
    solved = (1 if mesh == "uniform" else 0.8) * MM
    message = _assert_review_verdict(monkeypatch, sim, entry, snap,
        f"drawn touching/overlapping, solved spacing {_fmt_len(solved)}")
    assert "zero-thickness sheet on the interface it bounds" in message


@pytest.mark.parametrize("mesh", NODES)
def test_overlapping_foils_on_same_node_have_no_cavity(mesh):
    sim = _review_base(mesh, "strict")
    _review_sheet(sim, _LO_XY, 4.0, 4.2, "lower")
    _review_sheet(sim, _HI_XY, 4.1, 4.3, "upper")
    # Midpoints 4.1 and 4.2 both resolve to 4; drawn gap=-0.1, solved gap=0.
    # Only their off-node wall spacings are refused, not the merged foils.
    row, = sim.preflight().by_code(CODE)
    assert row.severity == "error"
    assert all("domain wall" in pair for pair in str(row).split(". WHY:")[0].split(" | "))


def test_named_xyz_profile_remedy_preserves_thin_substrate_planes(monkeypatch):
    from rfx.mesh_edges import edge_aware_profiles

    sim = _thin_substrate("uniform", "strict")
    row, = sim.preflight().by_code(CODE)
    assert 'rfx.mesh_edges.edge_aware_profiles(domain, dx, sheets=[...], solids=[...], axes="xyz")' in str(row)
    sheets = [g.shape for g in sim._geometry if g.material_name == "pec"]
    profiles = edge_aware_profiles(sim._domain, 0.5 * MM, sheets=sheets, axes="xyz")
    nodes = np.r_[0., np.cumsum(profiles["dz_profile"])]
    for plane in (2 * MM, 2.2 * MM):
        assert np.min(np.abs(nodes - plane)) < 1e-12
    aligned = Simulation(10e9, sim._domain, dx=0.5 * MM, boundary="pec", **profiles)
    for shape in sheets:
        aligned.add(shape, material="pec")
    aligned.add_source((MM, MM, MM), component="ez")
    calls = _stub_runners(monkeypatch)
    assert not aligned.preflight().by_code(CODE)
    with pytest.raises(_ReachedRunner):
        aligned.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


def _periodic_x_walls(far):
    from rfx.boundaries.spec import BoundarySpec

    sim = Simulation(10e9, DOMAIN, dx=MM, snap="strict",
                     boundary=BoundarySpec(x="periodic", y="pec", z="pec"))
    sim.add_material("lower", sigma=1e10)
    sim.add_material("upper", sigma=1e10)
    sim.add(Box((5 * MM, 2.65 * MM, 2.65 * MM), (5 * MM, 14.35 * MM, 16.35 * MM)), material="lower")
    sim.add(Box((far * MM, 4.65 * MM, 4.65 * MM), (far * MM, 10.35 * MM, 12.35 * MM)), material="upper")
    sim.add_source((MM, MM, MM), component="ez")
    return sim


@pytest.mark.parametrize("far", [19.99, 20.0])
def test_periodic_wall_at_the_far_boundary_is_judged_on_its_own_copy(far):
    # Periodic x, 20 mm cell, 1 mm cells: a wall drawn at 19.99 or 20 mm is
    # solved on the node at 20 mm (the copy of x = 0); from the wall at 5 mm
    # that is 15 mm against 14.99 or 15 mm drawn, within 1 %.
    assert not _periodic_x_walls(far).preflight().by_code(CODE)


def test_periodic_wall_off_its_node_reports_the_unwrapped_spacing():
    # Drawn at 19.6 mm, solved on the node at 20 mm: 15 mm against 14.6 mm.
    row, = _periodic_x_walls(19.6).preflight().by_code(CODE)
    assert row.severity == "error"
    assert f"drawn spacing {_fmt_len(14.6 * MM)}, solved spacing {_fmt_len(15 * MM)}" in str(row)


@pytest.mark.parametrize("order", ["ground_first", "pad_first"])
def test_a_foil_merged_with_the_ground_plane_still_bounds_the_cavity(order):
    # Ground (zero thickness) at 4 mm; a 0.8 mm thin-conductor pad centred on
    # it (3.6..4.4 mm, solved on the 4 mm plane too) under a patch at 6 mm.
    # Ground and pad are one plane of metal, not a cavity; the pad's top face
    # to the patch is drawn 1.6 mm and solved 2 mm.
    sim = Simulation(10e9, DOMAIN, dx=MM, boundary="pec")
    sim.add_material("ground", sigma=1e10)
    sim.add_material("patch", sigma=1e10)
    ground = Box((2.65 * MM, 2.65 * MM, 4 * MM), (16.35 * MM, 14.35 * MM, 4 * MM))
    pad = Box((5.65 * MM, 5.65 * MM, 3.6 * MM), (11.35 * MM, 9.35 * MM, 4.4 * MM))
    if order == "ground_first":
        sim.add(ground, material="ground")
        sim.add_thin_conductor(pad)
    else:
        sim.add_thin_conductor(pad)
        sim.add(ground, material="ground")
    sim.add(Box((4.65 * MM, 4.65 * MM, 6 * MM), (12.35 * MM, 10.35 * MM, 6 * MM)), material="patch")
    sim.add_source((MM, MM, MM), component="ez")
    row, = sim.preflight().by_code(CODE)
    assert row.severity == "error"
    assert "drawn spacing 1.6mm, solved spacing 2mm (+25.00%)" in str(row)


# Box-face expectations use the drawing and explicit nodes, never the census.
BOX_GAPS = [
    ("exact", 2.0, False),
    ("negative_inside", 2 / (1 - 0.0099), False),
    ("negative_outside", 2 / (1 - 0.0101), True),
    ("positive_inside", 2 / (1 + 0.0099), False),
    ("positive_outside", 2 / (1 + 0.0101), True),
    ("off", 2.1, True),
]


def _sheet_over_block(mesh, gap=2.1, shift=0.0, *, snap="strict", cylinder=False):
    sim = _review_base(mesh, snap)
    top = 4 + shift
    if cylinder:
        from rfx.geometry.csg import Cylinder
        ground = Cylinder((9 * MM, 8 * MM, (1 + top) / 2 * MM),
                          5 * MM, (top - 1) * MM)
    else:
        ground = Box((2 * MM, 2 * MM, MM), (17 * MM, 15 * MM, top * MM))
    sim.add(ground, material="lower")
    _review_sheet(sim, _HI_XY, top + gap, top + gap, "upper")
    return sim


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("case,gap,refused", BOX_GAPS, ids=[r[0] for r in BOX_GAPS])
def test_sheet_box_spacing_boundary(monkeypatch, mesh, case, gap, refused):
    # Block upper face at 4, sheet at 4+gap; both meshes solve at 4 and 6.
    # Gaps 2/(1 +/- .0099/.0101) straddle BOTH signs of the 1% rule.
    sim = _sheet_over_block(mesh, gap)
    calls = _stub_runners(monkeypatch)
    report = sim.preflight()
    assert bool(report.by_code(CODE)) == refused
    assert report.ok == (not refused)
    if refused:
        row, = report.by_code(CODE)
        assert row.severity == "error"
        assert "'lower' (PEC Box face upper) plane at 4mm" in str(row)
        assert "'upper' (sheet) plane at 6mm" in str(row)
        assert f"drawn spacing {_fmt_len(gap * MM)}, solved spacing 2mm" in str(row)
        assert f"({2 / gap - 1:+.2%})" in str(row)
        with pytest.raises(ValueError, match="parallel PEC cavity faces"):
            sim.run(n_steps=1, compute_s_params=False)
        assert not calls
    else:
        with pytest.raises(_ReachedRunner):
            sim.run(n_steps=1, compute_s_params=False)
        assert calls == [True]


@pytest.mark.parametrize("mesh", NODES)
def test_sheet_box_equal_shifts_preserve_spacing(monkeypatch, mesh):
    # 4.04 -> 4 and 6.04 -> 6: each shift/gap is 2%, but spacing error is 0.
    sim = _sheet_over_block(mesh, 2, 0.04)
    calls = _stub_runners(monkeypatch)
    assert not sim.preflight().by_code(CODE)
    with pytest.raises(_ReachedRunner):
        sim.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


@pytest.mark.parametrize("mesh", NODES)
def test_sheet_box_declared_warns(monkeypatch, mesh):
    sim = _sheet_over_block(mesh, snap="declared")
    message = _assert_review_verdict(monkeypatch, sim, "run", "declared",
                                    "drawn spacing 2.1mm, solved spacing 2mm (-4.76%)")
    assert "PEC Box face upper" in message
    assert "its drawn thickness" not in message


@pytest.mark.parametrize("mesh", NODES)
def test_lone_box_has_no_cavity(mesh):
    sim = _review_base(mesh, "strict")
    sim.add(Box((2 * MM, 2 * MM, 1.1 * MM), (17 * MM, 15 * MM, 4.3 * MM)),
            material="lower")
    # A second conductor beside it (no shared footprint) so the pairing runs.
    sim.add(Box((18 * MM, 16 * MM, 10 * MM), (19.5 * MM, 17.5 * MM, 10 * MM)),
            material="upper")
    # Its own 3.2 mm drawn thickness solves to 3 mm, but is metal, not a gap.
    # The off-node Box faces do bound cavities against the floor and lid.
    row, = sim.preflight().by_code(CODE)
    assert row.severity == "error"
    assert all("domain wall" in pair for pair in str(row).split(". WHY:")[0].split(" | "))


@pytest.mark.parametrize("mesh", NODES)
def test_cylinder_extreme_stays_electrical_warning(mesh):
    # Cylinder top 4.1 -> 4, sheet 6.2 -> 6: 2/2.1 - 1 = -4.7619%.
    # Even its planar end is excluded: this rule judges only Box volumes.
    report = _sheet_over_block(mesh, 2.1, 0.1, cylinder=True).preflight()
    spacing, = report.by_code(CODE)
    assert spacing.severity == "error"
    assert "domain wall z_hi" in str(spacing)
    assert "PEC Box face" not in str(spacing)
    row, = report.by_code("sheet_cavity_electrical_thickness")
    assert row.severity == "warning"
    assert "plane-to-plane 2mm vs face-to-face 2.1mm" in str(row)
    assert not report.ok


BOX_DOMAIN = (30 * MM, 30 * MM, 5 * MM)
BOX_DRAWINGS = {
    "patch": (Box((0, 0, 0), (30 * MM, 30 * MM, 0.97 * MM)),
              Box((10.13 * MM, 10.37 * MM, 1.61 * MM),
                  (20.13 * MM, 20.37 * MM, 2.11 * MM))),
    "x_gap": (Box((2 * MM, 5 * MM, MM), (8 * MM, 25 * MM, 4 * MM)),
              Box((9.52 * MM, 5 * MM, MM), (15 * MM, 25 * MM, 4 * MM))),
}


def _box_pair(mesh, case, *, profiles=None):
    if profiles is None:
        nodes = np.arange(26) * 0.2 * MM
        nodes[20] = 4.02 * MM  # Grading away from the cavity planes.
        profiles = {"dz_profile": np.diff(nodes)} if mesh == "graded" else {}
    sim = Simulation(10e9, BOX_DOMAIN, dx=0.2 * MM, boundary="pec", **profiles)
    for name, box in zip(("lower", "upper"), BOX_DRAWINGS[case]):
        sim.add_material(name, sigma=1e10)
        sim.add(box, material=name)
    sim.add_source((MM, MM, 3 * MM), component="ez")
    return sim


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("case,fragment", [
    # z=.97 -> 1, 1.61 -> 1.6: .6/.64 - 1 = -6.25%.
    ("patch", "drawn spacing 640µm, solved spacing 600µm (-6.25%)"),
    # x=8 -> 8, 9.52 -> 9.6: 1.6/1.52 - 1 = +5.2632%.
    ("x_gap", "drawn spacing 1.52mm, solved spacing 1.6mm (+5.26%)"),
])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_box_box_off_node_refuses(monkeypatch, mesh, case, fragment, entry):
    message = _assert_review_verdict(monkeypatch, _box_pair(mesh, case), entry,
                                    "strict", fragment)
    assert "'lower' (PEC Box face upper)" in message
    assert "'upper' (PEC Box face lower)" in message
    assert "its drawn thickness" not in message
    assert "solids=[...]" in message and "use a smaller dx" in message


def test_box_solids_profile_remedy_reaches_runner_unchanged(monkeypatch):
    from rfx.mesh_edges import edge_aware_profiles

    plain = _box_pair("uniform", "patch")
    assert plain.preflight().by_code(CODE)[0].severity == "error"
    profiles = edge_aware_profiles(BOX_DOMAIN, 0.2 * MM,
                                   solids=BOX_DRAWINGS["patch"], axes="xyz")
    for a, axis in enumerate("xyz"):
        nodes = np.r_[0., np.cumsum(profiles[f"d{axis}_profile"])]
        for box in BOX_DRAWINGS["patch"]:
            for face in (box.corner_lo[a], box.corner_hi[a]):
                assert np.min(abs(nodes - face)) < 1e-12
    aligned = _box_pair("graded", "patch", profiles=profiles)
    assert not aligned.preflight().by_code(CODE)
    assert aligned.preflight().ok
    seen = []

    def runner(self, *args, **kwargs):
        for axis in "xyz":
            np.testing.assert_array_equal(getattr(self, f"_d{axis}_profile"),
                                          profiles[f"d{axis}_profile"])
        seen.append(True)
        raise _ReachedRunner

    monkeypatch.setattr(Simulation, "_run_nonuniform", runner)
    with pytest.raises(_ReachedRunner):
        aligned.run(n_steps=1, compute_s_params=False)
    assert seen == [True]


def _stacked_blocks(mesh, upper_lo):
    # Lower block top at 4 mm (a node on both meshes); upper block from upper_lo.
    sim = _review_base(mesh, "strict")
    sim.add(Box((2 * MM, 2 * MM, MM), (17 * MM, 15 * MM, 4 * MM)), material="lower")
    # Incidental transverse Box faces are on their existing nearest nodes.
    sim.add(Box((5 * MM, 5 * MM, upper_lo), (12 * MM, 10 * MM, 7 * MM)),
            material="upper")
    return sim


@pytest.mark.parametrize("mesh", NODES)
def test_box_faces_one_ulp_apart_are_touching(monkeypatch, mesh):
    # Float dust from the user's arithmetic (one float64 step, ~1e-18 m) is
    # far inside the rasterizer's on-lattice band: both faces sit on node 4,
    # no cavity exists, and the run proceeds.
    sim = _stacked_blocks(mesh, np.nextafter(4 * MM, 1.0))
    calls = _stub_runners(monkeypatch)
    assert not sim.preflight().by_code(CODE)
    with pytest.raises(_ReachedRunner):
        sim.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


@pytest.mark.parametrize("mesh", NODES)
def test_box_faces_a_micron_apart_are_a_collapsed_cavity(monkeypatch, mesh):
    # 1 um = 1e-3 of a 1 mm cell, far above float dust: a drawn gap that
    # both faces' node (4 mm) solves as 0, so -100% and refused.
    _assert_review_verdict(monkeypatch, _stacked_blocks(mesh, 4.001 * MM), "run",
                           "strict", "drawn spacing 1µm, solved spacing 0mm (-100.00%)")


def _periodic_x_blocks(*spans):
    from rfx.boundaries.spec import BoundarySpec

    sim = Simulation(10e9, DOMAIN, dx=MM, snap="strict",
                     boundary=BoundarySpec(x="periodic", y="pec", z="pec"))
    for i, (xlo, xhi) in enumerate(spans):
        sim.add_material(f"m{i}", sigma=1e10)
        # Keep the periodic-x witness; transverse Box faces use their existing nodes.
        sim.add(Box((xlo * MM, 5 * MM, 5 * MM), (xhi * MM, 10 * MM, 12 * MM)),
                material=f"m{i}")
    sim.add_source((MM, MM, MM), component="ez")
    return sim


@pytest.mark.parametrize("spans", [((4, 8), (16, 20)), ((0, 3), (16, 20)), ((4, 8), (16, 22))],
                         ids=["face_on_seam", "split_strip", "crossing_seam"])
def test_periodic_box_at_or_across_the_seam_is_judged_on_its_drawn_faces(monkeypatch, spans):
    # Periodic x, L = 20 mm, 1 mm cells, every face on a node. A face at 20 or
    # 22 mm is stored on node 0 or 2; read by index it would be the LOWER face.
    # Drawn gaps 16 - 8 = 8 and 16 - 3 = 13 mm, solved the same: accepted.
    sim = _periodic_x_blocks(*spans)
    calls = _stub_runners(monkeypatch)
    assert not sim.preflight().by_code(CODE)
    with pytest.raises(_ReachedRunner):
        sim.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


def test_periodic_box_on_the_seam_with_an_off_node_face_is_refused():
    # Same as face_on_seam with the far block's lower face at 16.4 mm: solved
    # on 16, so 8 mm against 8.4 mm drawn, 8/8.4 - 1 = -4.76 %.
    row, = _periodic_x_blocks((4, 8), (16.4, 20)).preflight().by_code(CODE)
    assert row.severity == "error"
    assert f"drawn spacing {_fmt_len(8.4 * MM)}, solved spacing {_fmt_len(8 * MM)} (-4.76%)" in str(row)


def test_periodic_box_face_at_a_half_cell_tie_is_its_realized_plane():
    # Periodic x, 1 mm cells: a block drawn from 19.5 mm fills cells 19, 0, 1
    # (a tie resolves lower), so its lower face is the node at 19 mm, equally
    # far from 19.5 mm as the interior plane at 20 mm (node 0). From the block
    # [4, 8]: 11 mm solved against 11.5 mm drawn, 11/11.5 - 1 = -4.35 %.
    row, = _periodic_x_blocks((4, 8), (19.5, 22)).preflight().by_code(CODE)
    assert f"plane at {_fmt_len(19 * MM)} (drawn {_fmt_len(19.5 * MM)})" in str(row)
    assert f"drawn spacing {_fmt_len(11.5 * MM)}, solved spacing {_fmt_len(11 * MM)} (-4.35%)" in str(row)


# Domain-wall witnesses: all expected coordinates below are from the drawing
# and the explicit 1 mm node arrays, never from the geometry census.
def _wall_sheet(mesh="uniform", height=2.1, extent=20.0, *, snap="strict", z="pec", dx=MM):
    from rfx.boundaries.spec import BoundarySpec

    profiles = {"dz_profile": np.diff(NODES[mesh])} if mesh == "graded" else {}
    sim = Simulation(10e9, (20 * MM, 18 * MM, extent * MM), dx=dx,
                     boundary=BoundarySpec(x="pec", y="pec", z=z), cpml_layers=3,
                     snap=snap, **profiles)
    sim.add(Box((0, 0, height * MM), (20 * MM, 18 * MM, height * MM)), material="pec")
    sim.add_source((MM, MM, MM), component="ez")
    return sim


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("snap", ["strict", "declared"])
def test_wall_snap_verdict_before_runner(monkeypatch, entry, snap):
    # 5.6 / .4 = 14 cells: the lid is on-node. Sheet 2.1 -> 2.0 mm.
    # Floor gap: 2 / 2.1 - 1 = -4.76%; lid gap: 3.6 / 3.5 - 1 = +2.86%.
    sim = _wall_sheet(height=2.1, extent=5.6, dx=.4 * MM, snap=snap)
    message = _assert_review_verdict(monkeypatch, sim, entry, snap,
                                    "drawn spacing 3.5mm, solved spacing 3.6mm (+2.86%)")
    assert "domain wall z_hi (electric) plane at 5.6mm (drawn 5.6mm)" in message
    assert "divides the heights and the domain extent" in message


def test_on_node_lid_has_no_wall_cavity_error():
    # 6-2 = 4 mm in both the drawing and the explicit uniform nodes.
    assert not _wall_sheet(height=2, extent=6).preflight().by_code(CODE)


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_sheet_to_floor_refuses_before_runner(monkeypatch, mesh, entry):
    # Floor 0 -> 0, sheet 2.1 -> 2: 2/2.1-1 = -4.76%.
    # Lid gap 20-2.1 = 17.9 -> 18: 18/17.9-1 = +0.56%, below 1%.
    message = _assert_review_verdict(monkeypatch, _wall_sheet(mesh), entry, "strict",
                                    "drawn spacing 2.1mm, solved spacing 2mm (-4.76%)")
    assert "domain wall z_lo (electric) plane at 0mm (drawn 0mm)" in message
    assert "domain wall z_hi" not in message
    assert "1 pair(s)" in message


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("height,refused", [
    (2.0, False), (2 / (1 - 0.0099), False), (2 / (1 + 0.0099), False),
    (2 / (1 - 0.0101), True), (2 / (1 + 0.0101), True),
])
def test_floor_spacing_one_percent_boundary(mesh, height, refused):
    # Node 2 mm / [2/(1 +/- p)] - 1 = +/- p, either side of 1%.
    rows = _wall_sheet(mesh, height).preflight().by_code(CODE)
    assert bool(rows) == refused
    if refused:
        assert rows[0].severity == "error"
        assert "domain wall z_lo" in str(rows[0])


def test_box_touching_off_node_lid_is_solved_open(monkeypatch):
    sim = Simulation(10e9, (20 * MM, 18 * MM, 5.5 * MM), dx=MM, boundary="pec", snap="declared")
    assert_declared_wall_warning(sim)
    sim.add(Box((2 * MM, 2 * MM, 0), (17 * MM, 15 * MM, 5.5 * MM)), material="pec")
    sim.add_source((MM, MM, MM), component="ez")
    # Box upper tie 5.5 -> 5; ceil domain lid 5.5 -> 6; touching -> 1 mm.
    message = _assert_review_verdict(monkeypatch, sim, "run", "declared",
                                    "drawn touching/overlapping, solved spacing 1mm")
    assert "domain wall z_hi (electric) plane at 6mm (drawn 5.5mm)" in message
    assert "domain wall z_lo" not in message  # Box bottom 0 -> 0, touching shut.


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("boundary", ["cpml", "pmc", "periodic"])
def test_non_electric_or_absorbing_faces_are_not_judged(mesh, boundary):
    # Same 2.1 -> 2 mm face; no electric wall cavity exists along z.
    assert not _wall_sheet(mesh, z=boundary).preflight().by_code(CODE)


@pytest.mark.parametrize("mesh", NODES)
@pytest.mark.parametrize("full", [True, False], ids=["full_box", "partial_post"])
def test_box_blocks_all_or_only_some_floor_columns(mesh, full):
    sim = _wall_sheet(mesh)
    lo, hi = ((0, 0, 0), (20 * MM, 18 * MM, MM)) if full else (
        (8 * MM, 7 * MM, 0), (9 * MM, 8 * MM, MM))
    sim.add(Box(lo, hi), material="pec")
    # Box top 1 -> 1; sheet 2.1 -> 2. Full box covers the floor-sheet
    # cavity, leaving 1/1.1-1 = -9.09% above the box. A post leaves floor
    # columns open, so the 2/2.1-1 = -4.76% floor cavity is still judged.
    row, = sim.preflight().by_code(CODE)
    assert ("domain wall z_lo" in str(row)) == (not full)
    assert "drawn spacing 1.1mm, solved spacing 1mm (-9.09%)" in str(row)


@pytest.mark.parametrize("absorbing_side", ["lo", "hi"])
def test_graded_axis_with_one_absorbing_side_keeps_its_electric_wall(absorbing_side):
    from rfx.boundaries.spec import Boundary

    z = Boundary("cpml", "pec") if absorbing_side == "lo" else Boundary("pec", "cpml")
    height = 17.9 if absorbing_side == "lo" else 2.1
    # Non-absorbing endpoint is 20 or 0 despite padding on the other end.
    # Both drawings leave 2.1 mm to that endpoint; explicit nodes leave 2 mm.
    row, = _wall_sheet("graded", height, z=z).preflight().by_code(CODE)
    assert "drawn spacing 2.1mm, solved spacing 2mm (-4.76%)" in str(row)
    face = "z_hi" if absorbing_side == "lo" else "z_lo"
    assert f"domain wall {face} (electric)" in str(row)
    assert f"domain wall z_{absorbing_side}" not in str(row)


@pytest.mark.parametrize("height,extent", [(2.1, 20), (2, 5.5)])
@pytest.mark.parametrize("dx", [0.5 * MM, 0.25 * MM])
def test_wall_cavity_edge_aware_remedy_is_flush(height, extent, dx):
    from rfx.mesh_edges import edge_aware_profiles

    domain = (20 * MM, 18 * MM, extent * MM)
    sheet = Box((0, 0, height * MM), (domain[0], domain[1], height * MM))
    profiles = edge_aware_profiles(domain, dx, sheets=[sheet], solids=[], axes="xyz")
    sim = Simulation(10e9, domain, dx=dx, boundary="pec", **profiles)
    sim.add(sheet, material="pec")
    # Drawn sheet and both domain endpoints must be actual nodes. Expected
    # values are the input drawing: 2.1/20 mm or 2/5.5 mm, not helper output.
    for a, axis in enumerate("xyz"):
        assert sum(profiles[f"d{axis}_profile"]) == pytest.approx(domain[a], abs=1e-15)
    nodes = sim._campaign_ctx().nodes[2]
    assert min(abs(nodes - height * MM)) < 1e-15
    assert nodes[0] == pytest.approx(0.0, abs=1e-15)
    assert nodes[-1] == pytest.approx(extent * MM, abs=1e-15)
    assert not sim.preflight().by_code(CODE)


def test_uniform_cell_dividing_wall_extent_is_flush():
    # 5.5/.5 = 11 cells, sheet 2/.5 = node 4: gap 3.5 mm on both drawings.
    assert not _wall_sheet(height=2, extent=5.5, dx=0.5 * MM).preflight().by_code(CODE)


@pytest.mark.parametrize("boundary", ["pec", "pmc", "cpml", "periodic"])
def test_spacing_guard_does_not_assemble_without_a_possible_pair(monkeypatch, boundary):
    from rfx.preflight.pec_geometry import _validate_cfg_sheet_cavity_thickness

    sim = _wall_sheet(z=boundary)
    if boundary == "pec":
        sim._geometry.clear()  # Only two walls: they are not a cavity pair.
    ctx = sim._campaign_ctx()

    # The shared census already assembles; spy only on this validator's
    # additional realized() access after that census is available.
    ctx.interior_pec_entries()

    def unexpected(**kwargs):
        pytest.fail("no possible conductor pair: realized() must not be called")

    monkeypatch.setattr(ctx, "realized", unexpected)
    _validate_cfg_sheet_cavity_thickness(sim, None, ctx)


def test_uniform_wall_kinds_match_the_grid_resolver():
    from rfx.boundaries.pec import resolve_wall_faces
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.preflight.pec_geometry import _cavity_wall_faces

    sim = Simulation(10e9, DOMAIN, dx=MM, boundary=BoundarySpec(
        x="periodic", y=Boundary("pec", "pmc"), z=Boundary("pmc", "pec")))
    ctx = sim._campaign_ctx()
    electric, magnetic = resolve_wall_faces(ctx.grid, ctx.periodic)
    # Drawing: periodic x, electric y_lo/z_hi, magnetic y_hi/z_lo.
    assert _cavity_wall_faces(ctx) == electric == {"y_lo", "z_hi"}
    assert sim._boundary_spec.pmc_faces() == magnetic == {"y_hi", "z_lo"}


@pytest.mark.parametrize("conformal", [False, True])
def test_conformal_wall_is_not_judged_at_its_node(conformal):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    # Lid declared 5.5 mm on 1 mm cells: node 6 mm. Sheet 2 -> 2 mm.
    # Plain PEC: 4/3.5-1 = +14.29% refused. Conformal z: the lid is a
    # Dey-Mittra half-space at 5.5 mm, so the node is not the wall.
    sim = Simulation(10e9, (20 * MM, 18 * MM, 5.5 * MM), dx=MM,
                     boundary=BoundarySpec(x="pec", y="pec",
                                           z=Boundary("pec", "pec", conformal=conformal)), snap="declared")
    assert_declared_wall_warning(sim)
    sim.add(Box((0, 0, 2 * MM), (20 * MM, 18 * MM, 2 * MM)), material="pec")
    sim.add_source((MM, MM, MM), component="ez")
    rows = sim.preflight().by_code(CODE)
    assert bool(rows) == (not conformal)
    if rows:
        assert "drawn spacing 3.5mm, solved spacing 4mm (+14.29%)" in str(rows[0])

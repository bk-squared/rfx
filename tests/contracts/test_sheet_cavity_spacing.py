"""Build-only #1800: spacing, not absolute sheet position, decides refusal."""
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
    with pytest.raises(ValueError, match="parallel PEC sheets"):
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
        with pytest.raises(ValueError, match="parallel PEC sheets"):
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
    assert message.count(f"plane at {_fmt_len(expected_node)}") == 2
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
    assert message.count("plane at 2mm") == 2


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
    assert not sim.preflight().by_code(CODE)


def test_named_xyz_profile_remedy_preserves_thin_substrate_planes(monkeypatch):
    from rfx.mesh_edges import edge_aware_profiles

    sim = _thin_substrate("uniform", "strict")
    row, = sim.preflight().by_code(CODE)
    assert 'rfx.mesh_edges.edge_aware_profiles(domain, dx, sheets=[...], axes="xyz")' in str(row)
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

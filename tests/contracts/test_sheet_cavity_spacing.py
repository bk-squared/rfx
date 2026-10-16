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

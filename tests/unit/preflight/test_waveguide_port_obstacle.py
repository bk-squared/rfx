"""#1913: a PEC short must be visible from the reflection measurement plane.

Synthetic 42 x 21 mm guide; 3 mm cells; 40 periods (the leader's sweep),
only the two band endpoints, 5 and 6.5 GHz; normalize=True.
The phase oracle is the closed form 180 - 2 beta d at the reported plane.
"""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.model.waveguide_planes import entry_planes

CODE = "waveguide_port_obstacle_before_reference"
DX, A, B = 0.003, 0.042, 0.021
FREQS = np.array([5e9, 6.5e9])


def _guide(cells=None, *, one_port=False, **kwargs):
    sim = Simulation(
        freq_max=10e9, domain=(0.12, A, B), dx=DX,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=10, **kwargs)
    if cells is not None:
        face = 0.090 - cells * DX
        sim.add(Box((face - DX, 0, 0), (face, A, B)), material="pec")
    for name, x, direction in (("left", 0.010, "+x"), ("right", 0.090, "-x")):
        if one_port and name == "left":
            continue
        sim.add_waveguide_port(x, direction=direction, name=name, mode=(1, 0),
                              mode_type="TE", freqs=FREQS, f0=6e9, bandwidth=0.5)
    return sim


def _findings(sim):
    return [i for i in sim.preflight() if i.code == CODE]


@pytest.mark.parametrize("cells,face", [(1, 87), (2, 84)])
@pytest.mark.parametrize("dispatch", [False, True])
def test_hidden_short_refused_by_calculator(cells, face, dispatch):
    sim = _guide(cells)
    findings = _findings(sim)
    assert len(findings) == 1
    issue = findings[0]
    assert issue.severity == "error"
    for text in ("right", "pec", "source plane 90 mm", "at 81 mm",
                 f"face at {face} mm", "ref_offset=3", "minimum 3", "Move the port or the object"):
        assert text in str(issue)
    call = sim.compute_s_matrix if dispatch else sim.compute_waveguide_s_matrix
    with pytest.raises(ValueError, match="right.*face at.*mm between"):
        call(n_steps=4)


@pytest.mark.parametrize("cells", [4, 8])
def test_visible_short_has_closed_form_reflection(cells, record_property):
    sim = _guide(cells)
    assert not _findings(sim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s22 = np.asarray(result.s_params)[1, 1]
    # Default reported reference plane is the declared port, 90 mm.
    assert np.asarray(result.reference_planes)[1] == pytest.approx(0.090)
    beta = np.sqrt((2 * np.pi * FREQS / 299792458.0) ** 2 - (np.pi / A) ** 2)
    expected = np.pi - 2 * beta * (cells * DX)
    phase_error = np.degrees(np.angle(s22 * np.exp(-1j * expected)))
    record_property("s22_magnitude", np.abs(s22).tolist())
    record_property("phase_error_deg", phase_error.tolist())
    assert np.all((np.abs(s22) >= 0.995) & (np.abs(s22) <= 1.005))
    assert np.max(np.abs(phase_error)) < 2


@pytest.mark.parametrize("method", ["run", "forward"])
def test_one_port_entry_refuses(method):
    with pytest.raises(ValueError, match="right.*face at 87 mm"):
        getattr(_guide(1, one_port=True), method)(n_steps=4)


def test_skip_preflight_bypasses_calculator(monkeypatch):
    import rfx.sparams.waveguide as calculator
    sim = _guide(1)
    original = Simulation._build_grid

    class CalculatorReached(Exception):
        pass

    def build(self, *args, **kwargs):
        import sys
        if sys._getframe(1).f_code is calculator.compute_waveguide_s_matrix.__code__:
            raise CalculatorReached
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Simulation, "_build_grid", build)
    with pytest.raises(CalculatorReached):
        sim.compute_waveguide_s_matrix(n_steps=4, skip_preflight=True)


def test_filled_guide_admitted():
    sim = _guide()
    sim.add_material("fill", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.12, A, B)), material="fill")
    report = sim.preflight()
    assert report.ok
    assert not [i for i in report if i.code == CODE]


def test_reference_endpoint_is_open():
    sim = _guide()
    # A sheet ON the measured plane (81 mm) has no face in the open interval.
    sim.add(Box((0.081, 0, 0), (0.081, A, B)), material="pec")
    assert not _findings(sim)


@pytest.mark.parametrize("registry", ["geometry", "thin", "pinned"])
def test_axial_sheets_are_obstacles(registry):
    sim = _guide()
    if registry == "geometry":
        sim.add(Box((0.084, 0, 0), (0.084, A, B)), material="pec")
    elif registry == "thin":
        sim.add_thin_conductor(Box((0.084, 0, 0), (0.084, A, B)),
                               sigma_bulk=5.8e7, thickness=35e-6)
    else:
        sim.add_pinned_sheet(normal_axis=0, plane_index=28,
                             i_range=(0, 14), j_range=(0, 7), name="short")
    findings = _findings(sim)
    assert len(findings) == 1
    assert "face at 84 mm" in str(findings[0])


def test_transverse_disjoint_object_admitted():
    sim = _guide()
    sim.add(Box((0.084, A + DX, 0), (0.087, A + 2 * DX, B)), material="pec")
    assert not _findings(sim)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_graded_plane_matches_realized_mode_config(axis, record_property):
    from tests._x64_compat import enable_x64
    with enable_x64():
        # 20 coarse, 20 fine, 10 coarse cells: length 120 mm. At 90 mm the source
        # is node 40, and the preceding 3 cells total 4.5 mm, not 9 mm.
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        from rfx.preflight.waveguide_port_span import _validate_cfg_waveguide_port_span
        profile = np.r_[np.full(20, DX), np.full(20, DX / 2), np.full(10, DX)]
        a = "xyz".index(axis)
        domain = [A, B, A]
        domain[a] = 0.12
        boundaries = {ax: Boundary(lo="pec", hi="pec") for ax in "xyz"}
        boundaries[axis] = "cpml"
        sim = Simulation(freq_max=10e9, domain=tuple(domain), dx=DX,
                         boundary=BoundarySpec(**boundaries), cpml_layers=10,
                         **{f"d{axis}_profile": profile})
        sim.add_waveguide_port(0.090, direction=f"-{axis}", name="graded",
                              mode=(1, 0), mode_type="TE", freqs=FREQS, f0=6e9)
        lo, hi = np.zeros(3), np.array(domain)
        lo[a], hi[a] = 0.086, 0.087
        sim.add(Box(tuple(lo), tuple(hi)), material="pec")
        grid = sim._build_nonuniform_grid()
        entry = sim._waveguide_ports[0]
        source, reference = entry_planes(entry, grid)
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, FREQS, 4)
        # The mode builder sums the float32 cell widths (85.50005 mm); the
        # rule reads the same cells in float64. 1e-6 m is 1/750 of a cell.
        assert source == pytest.approx(0.090, abs=1e-8)
        assert reference == pytest.approx(0.0855, abs=1e-8)
        assert cfg.reference_x_m == pytest.approx(0.0855, abs=1e-6)
        assert reference == pytest.approx(cfg.reference_x_m, abs=1e-6)
        record_property("planes_m", [source, reference, cfg.reference_x_m])
        # Call the real check directly: unrelated graded material admission is
        # separate from this plane-position pin.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_cfg_waveguide_port_span(sim, warnings)
        findings = [w.message for w in caught if getattr(w.message, "code", None) == CODE]
        assert len(findings) == 1
        assert "at 85.5 mm" in str(findings[0])
        assert "face at 87 mm" in str(findings[0])


def test_uniform_guide_drawn_in_pieces_admitted():
    """#1913 review: abutting same-material boxes and a vacuum box change no medium."""
    from rfx.preflight.waveguide_port_span import _validate_cfg_waveguide_port_span
    for boxes in ((("fill", 0.050, 0.085), ("fill", 0.085, 0.110)),
                  (("vacuum", 0.084, 0.087),)):
        sim = Simulation(freq_max=10e9, domain=(0.12, A, B), dx=DX,
                         boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                                               z=Boundary(lo="pec", hi="pec")), cpml_layers=10)
        sim.add_material("fill", eps_r=2.2)
        for material, lo, hi in boxes:
            sim.add(Box((lo, 0, 0), (hi, A, B)), material=material)
        sim.add_waveguide_port(0.090, direction="-x", name="right",
                               mode=(1, 0), mode_type="TE", freqs=FREQS, f0=6e9)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_cfg_waveguide_port_span(sim, warnings)
        assert not [w for w in caught
                    if "waveguide_port_obstacle_before_reference" in str(getattr(w.message, "code", ""))
                    or "cannot see that object" in str(w.message)], boxes


def test_dielectric_step_inside_the_interval_refused():
    """A real medium change between source and reflection plane is an obstacle."""
    from rfx.preflight.waveguide_port_span import _validate_cfg_waveguide_port_span
    sim = Simulation(freq_max=10e9, domain=(0.12, A, B), dx=DX,
                     boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                                           z=Boundary(lo="pec", hi="pec")), cpml_layers=10)
    sim.add_material("fill", eps_r=2.2)
    sim.add(Box((0.050, 0, 0), (0.085, A, B)), material="fill")
    sim.add_waveguide_port(0.090, direction="-x", name="right",
                           mode=(1, 0), mode_type="TE", freqs=FREQS, f0=6e9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_cfg_waveguide_port_span(sim, warnings)
    assert any("has a face at 85 mm" in str(w.message) for w in caught)


@pytest.mark.parametrize("carve", [
    pytest.param(((0.084, 0.015, 0), (0.087, 0.027, B)), id="a_centre_slot"),
    pytest.param(((0.084, 0.015, 0.006), (0.087, 0.027, 0.015)), id="b_centre_hole"),
    pytest.param(((0.084, A / 2, 0), (0.087, A, B)), id="c_half_removed"),
    pytest.param(((0, 0.015, 0), (0.12, 0.027, B)), id="d_full_length_strip"),
])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded_transverse"])
def test_carved_pec_plate_refused(carve, graded):
    # The centre is vacuum on both sides, but PEC remains elsewhere on the
    # 84/87 mm faces. Graded transverse cells must sample that remaining PEC.
    profiles = (dict(dy_profile=np.array([0.003, 0.006] * 4 + [0.003, 0.003]),
                     dz_profile=np.array([0.003, 0.006, 0.003, 0.006, 0.003]))
                if graded else {})
    sim = _guide(1, **profiles)
    sim.add(Box(*carve), material="vacuum")
    plate = [issue for issue in _findings(sim) if "pec (geometry[0])" in str(issue)]
    assert len(plate) == 1
    assert plate[0].severity == "error"
    for text in ("right", "face at 87 mm", "at 81 mm"):
        assert text in str(plate[0])


def test_two_side_pec_boxes_stay_refused():
    sim = _guide()
    sim.add(Box((0.084, 0, 0), (0.087, 0.015, B)), material="pec")
    sim.add(Box((0.084, 0.027, 0), (0.087, A, B)), material="pec")
    findings = _findings(sim)
    assert len(findings) == 2
    for i in range(2):
        assert any(f"pec (geometry[{i}])" in str(issue) and "face at 87 mm" in str(issue)
                   and issue.severity == "error" for issue in findings)


def test_half_filled_guide_with_full_pec_short_stays_refused():
    sim = _guide()
    sim.add_material("fill", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.12, A / 2, B)), material="fill")
    sim.add(Box((0.084, 0, 0), (0.087, A, B)), material="pec")
    findings = _findings(sim)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "pec (geometry[1])" in str(findings[0])
    assert "face at 87 mm" in str(findings[0])


def test_abutting_boxes_differing_only_in_chi3_refused():
    sim = _guide()
    sim.add_material("linear", eps_r=2.2, chi3=0.0)
    sim.add_material("kerr", eps_r=2.2, chi3=1e-18)
    sim.add(Box((0.050, 0, 0), (0.085, A, B)), material="linear")
    sim.add(Box((0.085, 0, 0), (0.110, A, B)), material="kerr")
    findings = _findings(sim)
    assert len(findings) == 2
    for name in ("linear", "kerr"):
        assert any(name in str(issue) and "face at 85 mm" in str(issue)
                   and issue.severity == "error" for issue in findings)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_face_samples_use_realized_transverse_cell_centres(axis, graded):
    from rfx.preflight.waveguide_port_span import _face_sample_points
    transverse = [a for a in range(3) if a != axis]
    domain = np.zeros(3)
    domain[axis], domain[transverse] = 0.12, [A, B]
    profiles = ({f"d{'xyz'[transverse[0]]}_profile": np.array([0.003, 0.006] * 4 + [0.003, 0.003]),
                 f"d{'xyz'[transverse[1]]}_profile": np.array([0.003, 0.006, 0.003, 0.006, 0.003])}
                if graded else {})
    sim = Simulation(freq_max=10e9, domain=tuple(domain), dx=DX, **profiles)
    lo, hi = np.zeros(3), domain.copy()
    lo[transverse], hi[transverse] = [0.005, 0.004], [0.030, 0.016]
    aperture = {transverse[0]: (0.004, 0.035), transverse[1]: (0, B)}
    points = np.array(list(_face_sample_points(sim._build_realized_grid(), lo, hi, aperture)))
    # Explicit centres from the declared cell widths, clipped to the overlap.
    first = ([6, 10.5, 15, 19.5, 24, 28.5] if graded
             else [7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5])
    second = [6, 10.5, 15] if graded else [4.5, 7.5, 10.5, 13.5]
    expected = [[17.5, 10]] + [[u, v] for u in first for v in second]
    np.testing.assert_allclose(points[:, transverse], np.array(expected) * 1e-3, atol=1e-12)


def test_face_samples_cap_and_include_overlap_centre():
    from rfx.preflight.waveguide_port_span import _face_sample_points
    sim = Simulation(freq_max=10e9, domain=(0.12, A, B), dx=0.0003)
    points = np.array(list(_face_sample_points(
        sim._build_realized_grid(), np.zeros(3), np.array([0.12, A, B]),
        {1: (0, A), 2: (0, B)})))
    assert len(points) == 64 * 64
    np.testing.assert_allclose(points[0], [0, A / 2, B / 2], atol=1e-12)
    # The 140 x 70 cell-centre lattice is subsampled across its full extent.
    for a, count in ((1, 140), (2, 70)):
        indices = np.unique(np.rint(points[1:, a] / 0.0003 - 0.5).astype(int))
        assert len(indices) == 64
        assert indices[0] == 0 and indices[-1] == count - 1
        gaps = np.diff(indices)
        assert gaps.max() - gaps.min() <= 1
        np.testing.assert_allclose(points[1:, a] / 0.0003 - 0.5,
                                   np.rint(points[1:, a] / 0.0003 - 0.5), atol=1e-10)


def test_face_samples_keep_centre_when_overlap_contains_no_cell_centres():
    from rfx.preflight.waveguide_port_span import _face_sample_points
    sim = _guide()
    points = list(_face_sample_points(
        sim._build_realized_grid(), np.array([0.084, 0.0209, 0.0104]),
        np.array([0.087, 0.0211, 0.0106]), {1: (0, A), 2: (0, B)}))
    np.testing.assert_allclose(points, [[0, 0.021, 0.0105]], atol=1e-12)

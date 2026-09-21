"""Solved occupancy and diagnostics at each conductor realization site."""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation, Sphere


def _sim(*, fractional=False, nu=False, **kwargs):
    return Simulation(domain=(8.2 if fractional else 8., 8., 8.), dx=1.,
                      freq_max=1e6, boundary="cpml", cpml_layers=2,
                      **({"dz_profile": np.ones(8)} if nu else {}), **kwargs)


def _row(report, index):
    return next(r for r in report if r["entity"].startswith(f"geometry[{index}]"))


@pytest.mark.parametrize("kind", ["pec", "lossy", "impedance"])
def test_nonuniform_thin_conductor_fills_pad_nodes(kind):
    from rfx.runners.nonuniform import assemble_materials_nu
    sim = _sim(nu=True)
    kwargs = dict(sigma_bulk=5.8e7 if kind == "pec" else 100., thickness=.01)
    if kind == "impedance":
        kwargs["surface_impedance_f0"] = 1e6
    sim.add_thin_conductor(Box((0., 0., 4.), (8., 8., 4.)), **kwargs)
    grid = sim._build_nonuniform_grid()
    sheets, impedances = [], []
    materials, *_ = assemble_materials_nu(sim, grid, pec_sheets=sheets,
                                         pec_wires=[], sheet_specs=impedances)
    if kind == "pec":
        mask = np.asarray(sheets[0].footprint)
    elif kind == "impedance":
        mask = np.asarray(impedances[0].mask)
    else:
        mask = np.asarray(materials.sigma) > 0
    assert mask[:, :, 6].all()
    assert mask.sum() == 169


def test_dual_average_excludes_continued_metal_from_edge_average():
    from rfx.runners.nonuniform import assemble_materials_nu, assemble_interface_eps_nu
    sim = _sim(nu=True, interface_eps="dual_average")
    # eps=4 below z=4, vacuum above; PEC occupies the lower incident cells.
    sim.add_material("dielectric", eps_r=4.)
    sim.add(Box((-4., -4., -4.), (12., 12., 4.)), material="dielectric")
    sim.add(Box((0., 2., 2.), (8., 6., 4.4)), material="pec")
    grid = sim._build_nonuniform_grid()
    materials = assemble_materials_nu(sim, grid, pec_sheets=[], pec_wires=[])[0]
    eps = assemble_interface_eps_nu(sim, grid, materials)
    # Only the upper vacuum contributes; including the lower cell gives 2.5.
    assert float(eps[0][0, 6, 6]) == 1.


@pytest.mark.parametrize("kind", ["volume", "sheet", "thin"])
def test_fractional_face_is_present_in_fidelity_interior(kind):
    sim = _sim(fractional=True)
    shape = Box((0., 2., 4. if kind != "volume" else 2.), (8.2, 6., 4.))
    if kind == "thin":
        sim.add_thin_conductor(shape, sigma_bulk=5.8e7, thickness=.01)
    else:
        sim.add(shape, material="pec")
    report = sim.fidelity_report(print_report=False)
    row = next(r for r in report if r["entity"].startswith(
        "thin_conductor[0]" if kind == "thin" else "geometry[0]"))
    assert row["continued_faces"] == ["x-lo", "x-hi"]
    assert row["n_cells"] == (72 if kind == "volume" else 0)
    if kind != "volume":
        assert row["n_sheet_nodes"] == 50


@pytest.mark.parametrize("thin", [False, True])
def test_preflight_entry_arrays_include_the_absorber(thin):
    sim = _sim()
    shape = Box((0., 0., 4.), (8., 8., 4.))
    if thin:
        sim.add_thin_conductor(shape, sigma_bulk=5.8e7, thickness=.01)
    else:
        sim.add(shape, material="pec")
    entries = sim._campaign_ctx().entry_realizations()
    assert len(entries) == 1
    assert np.asarray(entries[0].sheet.footprint).sum() == 169


def test_refused_sibling_does_not_refuse_the_two_bridged_sheets():
    sim = _sim()
    sim.add(Box((0., 0., 0.), (8., 8., 0.)), material="pec")
    sim.add(Box((0., 3., 3.), (8., 5., 3.)), material="pec")
    sim.add(Box((.1, .1, 7.), (.2, .2, 7.)), material="pec")
    sim.add_port((4., 4., 0.), component="ez", extent=3.)
    report = sim.fidelity_report(print_report=False)
    for i, count in ((0, 81), (1, 27)):
        row = _row(report, i)
        assert row["n_sheet_nodes"] == count
        assert not any(f["kind"] == "refused-by-contract" for f in row["findings"])
    assert any(f["kind"] == "refused-by-contract" for f in _row(report, 2)["findings"])
    entries = sim._campaign_ctx().entry_realizations()
    assert [e.kind for e in entries] == ["sheet", "sheet", "refused"]
    assert "plane z=9" in entries[2].error  # node index: two pad cells + z=7


def test_smoothed_pec_branch_reports_unsupported_face():
    from rfx.geometry.smoothing import smoothed_shape_pairs
    sim = _sim()
    sim.add(Sphere((1., 4., 4.), 1.), material="pec")
    _, findings = smoothed_shape_pairs(sim, sim._build_grid())
    assert [(f.axis, f.side, f.conductor) for f in findings] == [(0, "lo", True)]


@pytest.mark.parametrize("smoothing,conformal", [(False, False), (True, False),
                                                ("kottke_pec", False), (True, True)])
def test_unsupported_conductor_warns_once_per_run(smoothing, conformal):
    sim = _sim()
    sim.add(Sphere((1., 4., 4.), 1.), material="pec")
    sim.add_source((4., 4., 4.), component="ez")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=2, skip_preflight=True, subpixel_smoothing=smoothing,
                conformal_pec=conformal)
    messages = [w for w in caught if "Conducting geometry reaches" in str(w.message)]
    assert len(messages) == 1


def test_zero_extent_dielectric_still_continues_along_its_normal():
    from rfx.geometry.smoothing import smoothed_shape_pairs
    sim = _sim()
    sim.add_material("dielectric", eps_r=4.)
    sim.add(Box((0., 2., 2.), (0., 6., 6.)), material="dielectric")
    pairs, _ = smoothed_shape_pairs(sim, sim._build_grid())
    assert pairs[0][0].corner_lo == (-4., 2., 2.)
    assert pairs[0][0].corner_hi == (0., 6., 6.)


def test_shift_probe_uses_realized_fractional_face_trimmed_to_interior():
    sim = _sim(fractional=True)
    sim.add(Box((0., 2., 2.), (8.2, 6., 4.)), material="pec")
    ctx = sim._campaign_ctx()
    entry = ctx.interior_pec_entries()[0]
    # 9 x 4 x 2 occupied cells: Ex=9*5*3, Ey=10*4*3, Ez=10*5*2.
    assert entry.edge_count_shifted(ctx, 0, .05) == 355


def test_grading_check_reads_the_domain_face_of_a_continued_sheet():
    sim = _sim(nu=True)
    sim.add(Box((0., 0., 4.), (8., 8., 4.)), material="pec")
    ctx = sim._campaign_ctx()
    # The transverse domain wall is at index 2; the full sheet runs to 0.
    # Set the already-built local widths to a literal 2:1 jump at that wall.
    sim._dx_profile = np.ones(8)
    ctx.spacings = (np.asarray(ctx.spacings[0]).copy(), *ctx.spacings[1:])
    ctx.spacings[0][1] = .5
    sim._campaign_ctx = lambda: ctx
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim._validate_thin_metal_on_nu_mesh()
    assert any("node 2" in str(w.message) and "ratio 2.00" in str(w.message) for w in caught)


def test_ntff_overlap_reports_interior_wall_coordinates():
    from rfx.preflight._common import PreflightConfigError
    sim = _sim()
    sim.add(Box((0., 2., 2.), (8., 6., 4.)), material="pec")
    sim._ntff = ((0., 3., 2.5), (7., 5., 3.5), np.asarray([1e6]))
    with pytest.raises(PreflightConfigError, match=r"x walls at \[0.000, 8000.000\]"):
        sim._validate_ntff_inverse_design()

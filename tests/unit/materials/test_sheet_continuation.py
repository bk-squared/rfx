"""Sheets reaching an absorber retain their in-plane metal cross-section."""

import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.geometry.rasterize_grid import (
    coords_from_nonuniform_grid, coords_from_uniform_grid)
from rfx.runners.nonuniform import assemble_materials_nu
from tests.unit.materials.test_sheet_impedance import PlanarSheet


def _sim(nonuniform):
    sim = Simulation(domain=(.008, .008, .008), dx=.001, freq_max=10e9,
                     boundary="cpml", cpml_layers=2)
    if nonuniform:
        sim._dz_profile = np.array([.001, .001, .0005, .0005, .001, .001, .001, .001, .001])
    return sim


def _arrays(sim, nonuniform):
    sheets, impedances = [], []
    grid = sim._build_nonuniform_grid() if nonuniform else sim._build_grid()
    if nonuniform:
        materials, _, _, cells = assemble_materials_nu(
            sim, grid, pec_sheets=sheets, sheet_specs=impedances)
        coords = coords_from_nonuniform_grid(grid)
    else:
        materials, _, _, cells, *_ = sim._assemble_materials(
            grid, pec_sheets=sheets, sheet_specs=impedances)
        coords = coords_from_uniform_grid(grid)
    arrays = {name: np.asarray(getattr(materials, name)) for name in ("eps_r", "sigma", "mu_r")}
    arrays["cells"] = np.zeros(grid.shape, bool) if cells is None else np.asarray(cells)
    for axis, mask in zip("xyz", realized_pec_edge_masks(arrays["cells"], sheets=sheets)):
        arrays["edge_" + axis] = np.asarray(mask)
    if sheets:
        arrays["footprint"] = np.asarray(sheets[0].footprint)
    if impedances:
        from rfx.materials.thin_conductor import build_sheet_impedance_ctx
        arrays["footprint"] = np.asarray(impedances[0].mask)
        arrays["sheet_sigma"] = np.asarray(impedances[0].sigma_sheet)
        ctx = build_sheet_impedance_ctx(impedances)
        for name in ("mask_ex", "mask_ey", "mask_ez"):
            arrays[name] = np.asarray(getattr(ctx, name))
    return grid, coords, arrays


@pytest.mark.parametrize("nonuniform", [False, True], ids=["uniform", "nu"])
@pytest.mark.parametrize("kind", ["pec", "leontovich"])
@pytest.mark.parametrize("normal", range(3), ids=list("xyz"))
def test_sheet_arrays_include_the_same_absorber_for_every_shape(nonuniform, kind, normal):
    """A full plate fills both lateral pads but remains one node thick."""
    lo, hi = [0.] * 3, [.008] * 3
    lo[normal] = hi[normal] = .004
    shapes = [Box(tuple(lo), tuple(hi)), PlanarSheet(normal, .004, (0., 0.), (.008, .008))]
    results = []
    try:
        for shape in shapes:
            sim = _sim(nonuniform)
            kw = {"surface_impedance_f0": 10e9} if kind == "leontovich" else {}
            sim.add_thin_conductor(shape, sigma_bulk=1e4 if kw else 5.8e7, **kw)
            with warnings.catch_warnings(record=True) as caught:
                grid, coords, arrays = _arrays(sim, nonuniform)
            expected = np.zeros(grid.shape, bool)
            plane = np.flatnonzero(np.isclose(coords[normal], .004, rtol=0., atol=1e-12))
            assert len(plane) == 1
            index = [slice(None)] * 3
            index[normal] = int(plane[0])
            expected[tuple(index)] = True
            # Independent full-plane expectation: equality alone could pass
            # if both shapes accidentally stopped at the absorber entrance.
            np.testing.assert_array_equal(arrays["footprint"], expected)
            assert not any("has no pad continuation" in str(w.message) for w in caught)
            results.append(arrays)
        assert results[0].keys() == results[1].keys()
        for name in results[0]:
            np.testing.assert_array_equal(results[0][name], results[1][name], err_msg=name)
    finally:
        jax.clear_caches()


@pytest.mark.parametrize("nonuniform", [False, True], ids=["uniform", "nu"])
def test_pattern_at_the_face_is_replicated_without_filling_its_slot(nonuniform):
    sim = _sim(nonuniform)
    sim.add_thin_conductor(
        PlanarSheet(2, .004, (0., .002), (.008, .006), hole=((-1., .003), (.002, .005))),
        sigma_bulk=5.8e7)
    try:
        grid, coords, arrays = _arrays(sim, nonuniform)
        footprint = arrays["footprint"]
        k = int(np.argmin(abs(coords.z - .004)))
        expected = np.zeros(grid.shape, bool)
        # Five literal y rows; the central row's slot reaches x_lo only.
        expected[:, grid.pad_y_lo + 2:grid.pad_y_lo + 7, k] = True
        expected[:grid.pad_x_lo + 2, grid.pad_y_lo + 4, k] = False
        np.testing.assert_array_equal(footprint, expected)
    finally:
        jax.clear_caches()


@pytest.mark.parametrize("nonuniform,legacy_dc", [(False, False), (True, False), (False, True)],
                         ids=["pec-volume-uniform", "pec-volume-nu", "dc-volume-uniform"])
def test_volumetric_nonbox_conductor_still_stops_and_warns(nonuniform, legacy_dc):
    from rfx.geometry.smoothing import continued_conductor_shape
    sim = _sim(nonuniform)
    shape = Sphere(center=(.001, .004, .004), radius=.002)
    if legacy_dc:
        sim.add_thin_conductor(shape, sigma_bulk=1e4, thickness=35e-6)
    else:
        sim.add(shape, material="pec")
    grid = sim._build_nonuniform_grid() if nonuniform else sim._build_grid()
    try:
        with pytest.warns(UserWarning, match="Sphere has no pad continuation"):
            solved = continued_conductor_shape(sim, grid, shape)
        assert solved is shape
        with pytest.warns(UserWarning, match="Sphere has no pad continuation"):
            _, _, arrays = _arrays(sim, nonuniform)
        # The outermost x pad is beyond the sphere; no copied metal there.
        occupied = arrays["sigma"] > 0 if legacy_dc else arrays["cells"]
        assert not occupied[0].any()
        assert occupied[grid.pad_x_lo].any()
    finally:
        jax.clear_caches()

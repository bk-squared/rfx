"""Sheets reaching an absorber retain their in-plane metal cross-section."""

import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Cylinder, GaussianPulse, Simulation, Sphere
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.geometry.rasterize_grid import (
    coords_from_nonuniform_grid, coords_from_uniform_grid)
from rfx.runners.nonuniform import assemble_materials_nu
from tests.unit.materials.test_sheet_impedance import PlanarSheet


def _sim(nonuniform, domain=(.008, .008, .008)):
    sim = Simulation(domain=domain, dx=.001, freq_max=10e9,
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
@pytest.mark.parametrize("kind", ["pec", "leontovich"])
def test_fractional_face_fills_the_gap_node_and_pad(nonuniform, kind):
    """8.6-cell x/y lengths leave a gap before both rounded absorber faces."""
    results = []
    try:
        for shape in (Box((0., 0., .004), (.0086, .0086, .004)),
                      PlanarSheet(2, .004, (0., 0.), (.0086, .0086))):
            sim = _sim(nonuniform, domain=(.0086, .0086, .008))
            kw = {"surface_impedance_f0": 10e9} if kind == "leontovich" else {}
            sim.add_thin_conductor(shape, sigma_bulk=1e4 if kw else 5.8e7, **kw)
            with warnings.catch_warnings(record=True) as caught:
                grid, coords, arrays = _arrays(sim, nonuniform)
            expected = np.zeros(grid.shape, bool)
            expected[:, :, np.argmin(abs(coords.z - .004))] = True
            np.testing.assert_array_equal(arrays["footprint"], expected)
            assert not any("no geometry continuation" in str(w.message) for w in caught)
            row = next(r for r in sim.fidelity_report(print_report=False)
                       if r["entity"].startswith("thin_conductor[0]"))
            assert row["continued_faces"] == ["x-lo", "x-hi", "y-lo", "y-hi"]
            if kind == "pec":
                assert row["n_sheet_nodes"] == 100
            results.append(arrays)
        for name in results[0]:
            np.testing.assert_array_equal(results[0][name], results[1][name], err_msg=name)
    finally:
        jax.clear_caches()


class _TaperedSheet(PlanarSheet):
    """A widening strip hits x-lo obliquely, with unequal first two rows."""

    def __init__(self):
        super().__init__(2, .004, (0., .002), (.004, .006))

    def mask_on_coords(self, x, y, z):
        mask = super().mask_on_coords(x, y, z)
        return mask & (np.asarray(y)[None, :, None] <= .003 + np.asarray(x)[:, None, None] + 1e-12)


@pytest.mark.parametrize("nonuniform", [False, True], ids=["uniform", "nu"])
@pytest.mark.parametrize("kind", ["pec", "leontovich"])
@pytest.mark.parametrize("drawing", ["disc", "taper"])
def test_touching_or_tapered_face_stays_declared_and_reported(nonuniform, kind, drawing):
    from rfx.geometry.smoothing import continued_conductor_shape
    shape = (Cylinder((.004, .004, .004), radius=.004, height=0., axis="z")
             if drawing == "disc" else _TaperedSheet())
    sim = _sim(nonuniform)
    kw = {"surface_impedance_f0": 10e9} if kind == "leontovich" else {}
    sim.add_thin_conductor(shape, sigma_bulk=1e4 if kw else 5.8e7, **kw)
    grid = sim._build_nonuniform_grid() if nonuniform else sim._build_grid()
    expected_faces = {(0, "lo"), (0, "hi"), (1, "lo"), (1, "hi")} if drawing == "disc" else {(0, "lo")}
    try:
        findings = []
        solved = continued_conductor_shape(sim, grid, shape, unextendable=findings)
        assert solved is shape
        assert {(u.axis, u.side) for u in findings} == expected_faces
        with pytest.warns(UserWarning, match="No continuation is applied"):
            _, _, arrays = _arrays(sim, nonuniform)
        assert not arrays["footprint"][:grid.pad_x_lo].any()
        issues = sim.preflight()
        assert any("no continuation" in str(issue).lower() or "not continued" in str(issue).lower()
                   for issue in issues)
        row = next(r for r in sim.fidelity_report(print_report=False)
                   if r["entity"].startswith("thin_conductor[0]"))
        assert row["continued_faces"] == []
    finally:
        jax.clear_caches()


class _NumpySheet(PlanarSheet):
    """Concrete host sampler, including under an outer JAX transformation."""

    def mask_on_coords(self, x, y, z):
        coords = [np.asarray(c) for c in (x, y, z)]
        masks = []
        for axis, c in enumerate(coords):
            if axis == self.axis:
                mask = np.zeros(c.shape, bool)
                mask[np.argmin(abs(c - self.coord))] = True
            else:
                i = self._plane_axes.index(axis)
                mask = (c >= self.plane_lo[i] - 1e-12) & (c <= self.plane_hi[i] + 1e-12)
            masks.append(mask)
        return masks[0][:, None, None] & masks[1][None, :, None] & masks[2][None, None, :]


@pytest.mark.parametrize("kind", ["pec", "leontovich"])
def test_host_sheet_forward_under_outer_jit_equals_box(kind):
    import jax.numpy as jnp
    losses = []
    try:
        for shape in (Box((0., .006, .010), (.020, .014, .010)),
                      _NumpySheet(2, .010, (0., .006), (.020, .014))):
            sim = Simulation(freq_max=6e9, domain=(.04, .02, .02), boundary="cpml", cpml_layers=4)
            kw = {"surface_impedance_f0": 5e9} if kind == "leontovich" else {}
            sim.add_thin_conductor(shape, sigma_bulk=1e4 if kw else 5.8e7, **kw)
            sim.add_source((.030, .01, .012), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=3e9))
            sim.add_probe((.025, .008, .012), "ez")

            def loss(eps):
                result = sim.forward(eps_override=eps, n_steps=20, checkpoint=True, skip_preflight=True)
                return jnp.sum(result.time_series[:, 0] ** 2)

            eps = jnp.ones(sim._build_grid().shape, dtype=jnp.float32)
            eager, compiled = loss(eps), jax.jit(loss)(eps)
            # Under jit XLA fuses and reorders the float32 updates, so eager and compiled
            # need not agree bit for bit: 3.44e-6 relative on the CI runner (PR #1334,
            # fast-suite 4), 0 locally. A one-cell change to the sheet moves this loss by
            # 1.7-2.3e-2 and a missing sheet by 4.0, so 1e-4 hides no geometry change.
            # The Box == non-Box check below is exact.
            np.testing.assert_allclose(compiled, eager, rtol=1e-4, atol=0.)
            losses.append(np.asarray(compiled))
        assert losses[0] > 0
        np.testing.assert_array_equal(losses[0], losses[1])
    finally:
        jax.clear_caches()


def test_continued_sheet_rejects_partial_mapped_arrays():
    from rfx.geometry.smoothing import continued_conductor_shape
    sim = _sim(False)
    shape = _NumpySheet(2, .004, (0., .002), (.008, .006))
    sim.add_thin_conductor(shape, sigma_bulk=5.8e7)
    grid = sim._build_grid()
    coords = coords_from_uniform_grid(grid)
    solved = continued_conductor_shape(sim, grid, shape)
    with pytest.raises(ValueError, match="full in-plane node arrays"):
        solved.mask_on_coords(coords.x[1:], coords.y, coords.z)


def test_continued_curved_sheet_keeps_its_interior_signed_distance():
    from rfx.geometry.smoothing import _get_sdf_fn, _has_sdf, continued_conductor_shape
    sim = _sim(False)
    # The two first interior rows each cover y=2..6 mm; this disc crosses
    # x-lo with a resolved strip, unlike the tangent disc above.
    shape = Cylinder((-.0005, .004, .004), radius=.0026, height=0., axis="z")
    sim.add_thin_conductor(shape, sigma_bulk=5.8e7)
    solved = continued_conductor_shape(sim, sim._build_grid(), shape)
    assert solved is not shape
    assert _has_sdf(solved)
    x = np.array([.00025, .00075, .00125, .00225])[:, None, None]
    y = np.array([.00225, .00325, .00425, .00525])[None, :, None]
    z = np.array([.00375, .00425])[None, None, :]
    radial = np.sqrt((x + .0005) ** 2 + (y - .004) ** 2) - .0026
    height = np.abs(z - .004)
    expected = np.sqrt(np.maximum(radial, 0.) ** 2 + height ** 2)
    actual = _get_sdf_fn(solved)(x, y, z, solved, xp=np)
    np.testing.assert_array_equal(actual, expected)


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

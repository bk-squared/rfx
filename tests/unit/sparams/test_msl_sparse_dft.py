"""Sparse DFT-region contracts for MSL port extraction."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.probes.probes import init_dft_plane_probe, update_dft_plane_probe
from rfx.sources.msl_port import msl_loop_current as _REAL_LOOP_CURRENT
from rfx.sparams._msl_stages import _prepare_msl_drive as _REAL_PREPARE_MSL_DRIVE


def _state(ez: jnp.ndarray, step: int) -> SimpleNamespace:
    return SimpleNamespace(ez=ez, step=jnp.asarray(step, dtype=jnp.int32))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sparse_monitor_matches_full_plane(axis):
    """A cropped plane must retain exactly the requested DFT samples."""
    shape = (5, 7, 8)
    region = (2, 5, 1, 7)
    freqs = jnp.asarray([0.7e9, 1.1e9, 1.9e9], dtype=jnp.float32)
    full = init_dft_plane_probe(
        axis=axis,
        index=2,
        component="ez",
        freqs=freqs,
        grid_shape=shape,
    )
    sparse = init_dft_plane_probe(
        axis=axis,
        index=2,
        component="ez",
        freqs=freqs,
        grid_shape=shape,
        region=region,
    )

    base = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    for step in range(4):
        state = _state(base * (step + 1), step)
        full = update_dft_plane_probe(full, state, dt=2.5e-12)
        sparse = update_dft_plane_probe(sparse, state, dt=2.5e-12)

    lo1, hi1, lo2, hi2 = region
    expected = full.accumulator[:, lo1:hi1, lo2:hi2]
    np.testing.assert_array_equal(np.asarray(sparse.accumulator), np.asarray(expected))
    assert sparse.accumulator.size < full.accumulator.size


def test_sparse_monitor_gradient_matches_full_plane():
    """Cropping must preserve gradients of objectives over retained samples."""
    shape = (4, 6, 7)
    region = (1, 5, 2, 6)
    freqs = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    base = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 10.0

    def objective(scale: jnp.ndarray, *, sparse: bool) -> jnp.ndarray:
        if sparse:
            probe = init_dft_plane_probe(
                axis=0,
                index=1,
                component="ez",
                freqs=freqs,
                grid_shape=shape,
                region=region,
            )
        else:
            probe = init_dft_plane_probe(
                axis=0,
                index=1,
                component="ez",
                freqs=freqs,
                grid_shape=shape,
            )
        for step in range(3):
            probe = update_dft_plane_probe(
                probe,
                _state(base * scale * (step + 1), step),
                dt=0.1,
            )
        acc = probe.accumulator
        if not sparse:
            lo1, hi1, lo2, hi2 = region
            acc = acc[:, lo1:hi1, lo2:hi2]
        return jnp.sum(jnp.abs(acc) ** 2)

    scale = jnp.asarray(0.8, dtype=jnp.float32)
    grad_full = jax.grad(lambda value: objective(value, sparse=False))(scale)
    grad_sparse = jax.grad(lambda value: objective(value, sparse=True))(scale)

    assert jnp.isfinite(grad_sparse)
    assert abs(float(grad_sparse)) > 1e-3
    np.testing.assert_allclose(
        np.asarray(grad_sparse),
        np.asarray(grad_full),
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_msl_registers_sparse_regions(monkeypatch):
    """The production MSL path must opt every internal plane into cropping."""
    from tests.unit.autodiff.test_msl_sparam_ad import (
        _build_thru_line_sim,
        _fake_run_for_theta,
    )

    sim = _build_thru_line_sim()
    fake_run = _fake_run_for_theta(jnp.asarray(0.7, dtype=jnp.float32))
    observed: list[tuple[int, int, object]] = []

    def audited_run(self, *, n_steps=None, num_periods=1.0,
                    compute_s_params=False):
        grid = self._build_grid()
        for entry in self._dft_planes:
            if entry.axis == "x":
                full_shape = (grid.ny, grid.nz)
            elif entry.axis == "y":
                full_shape = (grid.nx, grid.nz)
            else:
                full_shape = (grid.nx, grid.ny)
            region = self._dft_plane_regions.get(entry.name)
            sparse_cells = (
                int(np.prod(full_shape))
                if region is None
                else (region[1] - region[0]) * (region[3] - region[2])
            )
            observed.append((int(np.prod(full_shape)), sparse_cells, region))
        return fake_run(
            self,
            n_steps=n_steps,
            num_periods=num_periods,
            compute_s_params=compute_s_params,
        )

    monkeypatch.setattr(type(sim), "run", audited_run)
    sim.compute_msl_s_matrix(
        n_steps=1,
        freqs=jnp.asarray([1.0e9], dtype=jnp.float32),
        num_periods=1.0,
        enforce_passivity=False,
    )

    assert observed
    assert all(region is not None for _, _, region in observed)
    assert sum(sparse for _, sparse, _ in observed) < sum(
        full for full, _, _ in observed
    )


from tests.unit.ports.test_msl_port_axis_generality import (  # noqa: E402
    DX as _AG_DX, EPS_R as _AG_EPS_R, F_MAX as _AG_F_MAX, H_SUB as _AG_H_SUB,
    L_LAT as _AG_L_LAT, L_LINE as _AG_L_LINE, L_PROP as _AG_L_PROP,
    LZ as _AG_LZ, PORT_MARGIN as _AG_PORT_MARGIN, W_TRACE as _AG_W_TRACE,
)


def _thru_sim(axis: str, lane: str):
    """The zero-thickness sheet on the laminate face represents 35 um foil."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    assert _AG_H_SUB / _AG_DX == 3, "The substrate face must lie on a node three cells above ground."
    AIR_CELL_SIZE_RATIO = 1.25
    kw = {}
    lz = _AG_LZ
    if lane == "nonuniform":
        nz = int(round(_AG_LZ / _AG_DX))
        dz = np.full(nz, _AG_DX)
        dz[nz // 2:nz - 4] = AIR_CELL_SIZE_RATIO * _AG_DX
        kw["dz_profile"] = dz
        lz = float(dz.sum())
    lat_c = _AG_L_LAT / 2.0
    if axis == "x":
        domain, sub = (_AG_L_PROP, _AG_L_LAT, lz), (_AG_L_PROP, _AG_L_LAT, _AG_H_SUB)
        tlo = (0.0, lat_c - _AG_W_TRACE / 2, _AG_H_SUB)
        thi = (_AG_L_PROP, lat_c + _AG_W_TRACE / 2, _AG_H_SUB)
        p0, p1, d0, d1 = ((_AG_PORT_MARGIN, lat_c, 0.0),
                          (_AG_PORT_MARGIN + _AG_L_LINE, lat_c, 0.0), "+x", "-x")
    else:
        domain, sub = (_AG_L_LAT, _AG_L_PROP, lz), (_AG_L_LAT, _AG_L_PROP, _AG_H_SUB)
        tlo = (lat_c - _AG_W_TRACE / 2, 0.0, _AG_H_SUB)
        thi = (lat_c + _AG_W_TRACE / 2, _AG_L_PROP, _AG_H_SUB)
        p0, p1, d0, d1 = ((lat_c, _AG_PORT_MARGIN, 0.0),
                          (lat_c, _AG_PORT_MARGIN + _AG_L_LINE, 0.0), "+y", "-y")
    sim = Simulation(snap="declared", freq_max=_AG_F_MAX, domain=domain, dx=_AG_DX, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")), **kw)
    sim.add_material("ro4350b", eps_r=_AG_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), sub), material="ro4350b")
    sim.add(Box(tlo, thi), material="pec")
    sim.add_msl_port(position=p0, width=_AG_W_TRACE, height=_AG_H_SUB,
                     direction=d0, impedance=50.0)
    sim.add_msl_port(position=p1, width=_AG_W_TRACE, height=_AG_H_SUB,
                     direction=d1, impedance=50.0)
    return sim


def test_thru_foil_is_one_realized_sheet_plane():
    """Build-time (no solve): the extractor's board is the declared board.

    Both gates below compare the cropped extractor against the full-plane
    one on ONE geometry, so they are realization-neutral by construction —
    which is exactly why the geometry needs its own check. Without it the
    trace could silently go back to being a slab and the equivalence would
    still hold, on the wrong board.
    """
    from tests._realized_geometry import (
        assert_sheet_planes, assert_wall_planes, realized)

    for axis in ("x", "y"):
        sim = _thru_sim(axis, "uniform")
        assert_sheet_planes(sim, 2, [_AG_H_SUB],
                            what=f"MSL {axis}-thru foil")
        assert_wall_planes(sim, 2, [_AG_H_SUB],
                           what=f"MSL {axis}-thru foil")
        rz = realized(sim)
        assert rz.pec_mask is None or not bool(np.asarray(rz.pec_mask).any()), \
            "a sheet owns no cell (#931 §1.3); the foil must not enter pec_mask"
        k = int(round(_AG_H_SUB / _AG_DX))
        ez = np.asarray(rz.edge_masks[2], dtype=bool)
        assert not ez[:, :, k].any(), \
            "the normal Ez edge through a sheet stays live (#931 §1.3)"


def _run_s(axis, lane, monkeypatch, *, crop: bool, seen: list):
    """One thru solve; ``crop=False`` clears the crop at its source, the
    variant's ``_dft_plane_regions``, so the plan, runner probe and extractor
    all take the full plane.

    ``seen`` collects the ``region`` every plane probe was initialised with.
    The returned ``loops`` list carries, per ``msl_loop_current`` call, the
    Ampere-loop window the extractor selected and the shape of the plane it
    selected it from — the index-level record the witness below gates on.
    """
    import warnings
    import rfx.probes.probes as _pp
    import rfx.runners.uniform as _ru
    import rfx.sources.msl_port as _mp
    import rfx.sparams._msl_stages as _stages

    real = init_dft_plane_probe

    def spy(*a, **k):
        seen.append(k.get("region"))
        return real(*a, **k)

    def prepare_drive(*a, **k):
        prepared = _REAL_PREPARE_MSL_DRIVE(*a, **k)
        if not crop:
            prepared[0]._dft_plane_regions.clear()
        return prepared

    loops: list[dict] = []
    real_loop = _REAL_LOOP_CURRENT

    def loop_spy(hy_plane, hz_plane, **k):
        loops.append({
            "ha_shape": tuple(int(v) for v in np.shape(hy_plane)),
            "hb_shape": tuple(int(v) for v in np.shape(hz_plane)),
            "a_lo": int(k["j_lo"]), "a_hi": int(k["j_hi"]),
            "b_lo": int(k["k_trace_lo"]), "b_hi": int(k["k_trace_hi"]),
            "n_da": int(np.size(k["dy_arr"])),
            "n_db": int(np.size(k["dz_arr"])),
        })
        return real_loop(hy_plane, hz_plane, **k)

    monkeypatch.setattr(_pp, "init_dft_plane_probe", spy)
    monkeypatch.setattr(_ru, "init_dft_plane_probe", spy)
    monkeypatch.setattr(_stages, "_prepare_msl_drive", prepare_drive)
    monkeypatch.setattr(_mp, "msl_loop_current", loop_spy)
    sim = _thru_sim(axis, lane)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(n_freqs=3, num_periods=3)
    assert "_dft_plane_regions" not in vars(sim), \
        "the runtime crop dict must not survive compute_msl_s_matrix"
    return np.asarray(res.S), np.asarray(res.Z0), loops


_SOLVES: dict[tuple, dict] = {}


def _solve(axis, lane, monkeypatch, *, crop: bool) -> dict:
    key = (axis, lane, crop)
    if key not in _SOLVES:
        seen: list = []
        s, z, loops = _run_s(axis, lane, monkeypatch, crop=crop, seen=seen)
        _SOLVES[key] = {"s": s, "z": z, "regions": seen, "loops": loops}
    return _SOLVES[key]


_LANES = [("uniform", "x"), ("uniform", "y"), ("nonuniform", "x")]


@pytest.mark.parametrize("lane,axis", _LANES)
def test_cropped_extractor_selects_exactly_the_full_plane_indices(
        monkeypatch, lane, axis):
    """INDEX-LEVEL witness on the cropped j/k arithmetic — bitwise, no tolerance.

    ``rfx/api/_sparams.py`` rebases the Ampere-loop window into the cropped
    plane's frame::

        _j_lo = meta["j_lo"] - _w_lo      _k_lo = k_trace_lo - _n_lo
        _j_hi = meta["j_hi"] - _w_lo      _k_hi = k_trace_hi - _n_lo

    Nothing else in the suite reaches that arithmetic.
    ``test_sparse_monitor_matches_full_plane`` above is bitwise but covers
    ``update_dft_plane_probe`` only — 4 synthetic steps on a (5, 7, 8) grid
    — and never touches these offsets; the end-to-end S check below sees
    them only through V, I and the S assembly, where an off-by-one is a
    small float difference rather than a wrong index.

    So this gates the indices themselves. Every assertion is an integer
    identity: no tolerance is spent, on either JAX version, and the run's
    float output is not read at all.

    The four facts, all read out of the extractor's own
    ``msl_loop_current`` calls (and the ``region`` each plane probe was
    built with), for the same solve in both crop and full-plane mode:

    (a) the window has the same WIDTH in both modes — cropping may only
        rebase the window, never resize it;
    (b) inside the cropped plane the window is inset by exactly ONE cell on
        all four sides, which is where ``_h_crop_region`` puts it: the
        region is built as ``(j_lo - 1, j_hi + 1, k_lo - 1, k_hi + 1)``
        because ``msl_loop_current`` walks the contour one cell OUTSIDE the
        trace block (``hy[..., k_trace_lo - 1]``, ``hz[:, j_lo - 1, :]``);
    (c) the cropped plane is exactly that window plus the two margin cells
        on each axis, so (b) is a statement about the whole rectangle and
        not just its lower corner; and
    (d) the region really is the full-plane window ±1, so (b) and (c) are
        anchored to the ABSOLUTE indices rather than to themselves.

    Together: the cells the extractor integrates over in the cropped plane
    are exactly the cells it integrates over in the full plane.

    Mutation coverage — ``rfx/api/_sparams.py``, ``_j_lo = meta["j_lo"] -
    _w_lo`` -> ``+ 1``, the j/k off-by-one class. Measured in a scratch copy
    of the tree, jax 0.10.2 / numpy 2.4.6:

        this witness                3 failed   uniform-x, uniform-y,
                                               nonuniform-x
            E  AssertionError: uniform-x loop call 0
            E  assert (9 - 2) == (30 - 22)

        end-to-end S check below    3 passed   max |dS| vs its own atol
                                               4.8801e-06:
            uniform-x     4.9958e-07   0.102x
            uniform-y     3.7253e-07   0.076x
            nonuniform-x  5.0479e-07   0.103x

    No constant rescues that gate: the weakest surviving mutant is
    3.7253e-07 while the jax 0.6.2 noise floor on the same fixture is
    3.69e-07 — about 1.01x apart. Hence indices, not values.

    ``msl_loop_current``'s own guard does NOT catch this mutant either: it
    rejects ``j_lo < 1``, and the mutant moves j_lo UP, into a window that
    is still inside the plane and still self-consistent — just one cell off.
    Region and loop-window spans are compared as an unordered pair because the (width, normal) region frame and right-handed (a, b) loop frame swap for +y/-y ports.
    """
    crop = _solve(axis, lane, monkeypatch, crop=True)
    full = _solve(axis, lane, monkeypatch, crop=False)

    loops_c, loops_f = crop["loops"], full["loops"]
    assert loops_c and len(loops_c) == len(loops_f), (len(loops_c), len(loops_f))

    for idx, (lc, lf) in enumerate(zip(loops_c, loops_f)):
        where = f"{lane}-{axis} loop call {idx}"
        assert lc["ha_shape"][1:] < lf["ha_shape"][1:], where
        assert lc["hb_shape"] == lc["ha_shape"], where
        assert lc["n_da"] == lc["ha_shape"][1], where
        assert lc["n_db"] == lc["ha_shape"][2], where

        assert lc["a_hi"] - lc["a_lo"] == lf["a_hi"] - lf["a_lo"], where
        assert lc["b_hi"] - lc["b_lo"] == lf["b_hi"] - lf["b_lo"], where

        assert lc["a_lo"] == 1, where
        assert lc["b_lo"] == 1, where
        assert lc["a_hi"] == lc["ha_shape"][1] - 1, where
        assert lc["b_hi"] == lc["ha_shape"][2] - 1, where

        assert lc["ha_shape"][1] == (lf["a_hi"] - lf["a_lo"]) + 2, where
        assert lc["ha_shape"][2] == (lf["b_hi"] - lf["b_lo"]) + 2, where

    h_regions = {r for r in crop["regions"] if r is not None and r[1] - r[0] > 1}
    assert h_regions, crop["regions"]
    expected = {
        frozenset({(lf["a_lo"] - 1, lf["a_hi"] + 1),
                   (lf["b_lo"] - 1, lf["b_hi"] + 1)})
        for lf in loops_f
    }
    observed = {frozenset({(r[0], r[1]), (r[2], r[3])}) for r in h_regions}
    assert observed == expected, (sorted(h_regions), expected)


@pytest.mark.parametrize("lane,axis", _LANES)
def test_cropped_extractor_reproduces_full_plane_s_end_to_end(monkeypatch, lane, axis):
    """The cropped extractor is the SAME COMPUTATION as the full-plane one.

    A compiled-program identity check across two accumulator SHAPES, not a
    physics tolerance: the tolerance below is derived from the ulp budget
    two shapes of one program are allowed to differ by.

    This is the WEAKER of the two gates on the cropped extractor, and it is
    weaker in a specific way: it does not gate the j/k index class. That is
    ``test_cropped_extractor_selects_exactly_the_full_plane_indices`` above,
    which is bitwise. See the derivation at the assertion.
    The absolute matrix-scale budget of (2*n_cells + 1)*u uses the crop area as a generous summation-length bound to absorb the V,I-to-S amplification measured earlier at 2.12x.
    """
    crop = _solve(axis, lane, monkeypatch, crop=True)
    full = _solve(axis, lane, monkeypatch, crop=False)
    s_crop, z_crop, seen_crop = crop["s"], crop["z"], crop["regions"]
    s_full, z_full, seen_full = full["s"], full["z"], full["regions"]

    regions = [r for r in seen_crop if r is not None]
    assert regions and len(regions) == len(seen_crop), seen_crop
    assert all(len(r) == 4 and r[1] > r[0] and r[3] > r[2] for r in regions)
    assert seen_full and all(r is None for r in seen_full), seen_full

    assert s_crop.shape == s_full.shape == (2, 2, 3)

    n_cells = max((r[1] - r[0]) * (r[3] - r[2]) for r in regions)
    V_AND_I_ERROR_CONTRIBUTIONS = 2
    COHERENT_ACCUMULATION_ULPS = 1
    NEAR_ZERO_MATRIX_ENTRY_RTOL = 0.0
    ulps = V_AND_I_ERROR_CONTRIBUTIONS * n_cells + COHERENT_ACCUMULATION_ULPS
    u = float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        s_crop, s_full, rtol=NEAR_ZERO_MATRIX_ENTRY_RTOL,
        atol=ulps * u * float(np.max(np.abs(s_full))))
    np.testing.assert_allclose(
        z_crop, z_full, rtol=NEAR_ZERO_MATRIX_ENTRY_RTOL,
        atol=ulps * u * float(np.max(np.abs(z_full))))
    assert np.all(np.isfinite(s_crop))

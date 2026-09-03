"""Sparse DFT-region contracts for MSL port extraction."""

from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.probes.probes import init_dft_plane_probe, update_dft_plane_probe
from rfx.sources.msl_port import msl_loop_current as _REAL_LOOP_CURRENT


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


def test_compute_msl_registers_sparse_regions():
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

    sim.run = MethodType(audited_run, sim)
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


# ---------------------------------------------------------------------------
# The cropped extractor arithmetic (rfx/api/_sparams.py). The three tests
# above never reach it — the uniform and nonuniform scans accumulate through
# their own hand-copies of the DFT update (rfx/simulation.py,
# rfx/nonuniform.py) rather than update_dft_plane_probe, and the registration
# test feeds the extractor full-plane fakes so it takes the legacy fallback.
#
# The two gates below both run the real thru on both lanes and both port
# orientations (a_is_width True/False), once with the crop engaged and once
# with it forced off. They cover DIFFERENT mutation classes and are not
# interchangeable:
#
#   test_cropped_extractor_selects_exactly_the_full_plane_indices
#       the j/k INDEX class — the rebasing arithmetic
#       (_j_lo = meta["j_lo"] - _w_lo and its three siblings) and the
#       -1/+1 contour margin. Gated on integers, bitwise, no tolerance.
#
#   test_cropped_extractor_reproduces_full_plane_s_end_to_end
#       the VALUE class — that the two accumulator shapes are one
#       computation. Gated on complex S against a derived
#       two-shapes-one-program ulp budget (#876). It does NOT gate the
#       index class: the j/k off-by-one survives it (measured; see the
#       derivation at the assertion).
#
# The bitwise claim about update_dft_plane_probe itself lives in
# test_sparse_monitor_matches_full_plane, further up.
# ---------------------------------------------------------------------------

from tests.unit.ports.test_msl_port_axis_generality import (  # noqa: E402
    DX as _AG_DX, EPS_R as _AG_EPS_R, F_MAX as _AG_F_MAX, H_SUB as _AG_H_SUB,
    L_LAT as _AG_L_LAT, L_LINE as _AG_L_LINE, L_PROP as _AG_L_PROP,
    LZ as _AG_LZ, PORT_MARGIN as _AG_PORT_MARGIN, W_TRACE as _AG_W_TRACE,
)


def _thru_sim(axis: str, lane: str):
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    kw = {}
    lz = _AG_LZ
    if lane == "nonuniform":
        # graded z: a 1.25x step in the air above the substrate, boundary
        # cells kept at dx (make_nonuniform_grid's constraint)
        nz = int(round(_AG_LZ / _AG_DX))
        dz = np.full(nz, _AG_DX)
        dz[nz // 2:nz - 4] = 1.25 * _AG_DX
        kw["dz_profile"] = dz
        lz = float(dz.sum())
    lat_c = _AG_L_LAT / 2.0
    if axis == "x":
        domain, sub = (_AG_L_PROP, _AG_L_LAT, lz), (_AG_L_PROP, _AG_L_LAT, _AG_H_SUB)
        tlo = (0.0, lat_c - _AG_W_TRACE / 2, _AG_H_SUB)
        thi = (_AG_L_PROP, lat_c + _AG_W_TRACE / 2, _AG_H_SUB + _AG_DX)
        p0, p1, d0, d1 = ((_AG_PORT_MARGIN, lat_c, 0.0),
                          (_AG_PORT_MARGIN + _AG_L_LINE, lat_c, 0.0), "+x", "-x")
    else:
        domain, sub = (_AG_L_LAT, _AG_L_PROP, lz), (_AG_L_LAT, _AG_L_PROP, _AG_H_SUB)
        tlo = (lat_c - _AG_W_TRACE / 2, 0.0, _AG_H_SUB)
        thi = (lat_c + _AG_W_TRACE / 2, _AG_L_PROP, _AG_H_SUB + _AG_DX)
        p0, p1, d0, d1 = ((lat_c, _AG_PORT_MARGIN, 0.0),
                          (lat_c, _AG_PORT_MARGIN + _AG_L_LINE, 0.0), "+y", "-y")
    sim = Simulation(freq_max=_AG_F_MAX, domain=domain, dx=_AG_DX, cpml_layers=8,
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


def _run_s(axis, lane, monkeypatch, *, crop: bool, seen: list):
    """One thru solve; ``crop=False`` forces every internal plane to the
    full transverse rectangle so the extractor takes its legacy path.

    ``seen`` collects the ``region`` every plane probe was initialised with.
    The returned ``loops`` list carries, per ``msl_loop_current`` call, the
    Ampere-loop window the extractor selected and the shape of the plane it
    selected it from — the index-level record the witness below gates on.
    """
    import warnings
    import rfx.probes.probes as _pp
    import rfx.runners.uniform as _ru
    import rfx.sources.msl_port as _mp

    real = init_dft_plane_probe

    def spy(*a, **k):
        seen.append(k.get("region"))
        if not crop:
            k["region"] = None
        return real(*a, **k)

    loops: list[dict] = []
    # the PRISTINE function, captured at import: reading it off the module
    # here would chain this run's spy onto the previous run's spy and send
    # both runs' calls into the first ``loops`` list
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

    # the uniform runner bound the name at import; the NU runner and the
    # forward path import it inside the function
    monkeypatch.setattr(_pp, "init_dft_plane_probe", spy)
    monkeypatch.setattr(_ru, "init_dft_plane_probe", spy)
    # compute_msl_s_matrix imports msl_loop_current from its home module
    # inside the method, so patching the home module is what reaches it
    monkeypatch.setattr(_mp, "msl_loop_current", loop_spy)
    sim = _thru_sim(axis, lane)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(n_freqs=3, num_periods=3)
    assert "_dft_plane_regions" not in vars(sim), \
        "the runtime crop dict must not survive compute_msl_s_matrix"
    return np.asarray(res.S), np.asarray(res.Z0), loops


# One thru solve costs ~20 s, and the two gates below want the same six
# (lane, axis, crop) solves. Cache the extracted numbers — plain numpy /
# python data, no JAX objects — so the file pays for each solve once.
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
    """
    crop = _solve(axis, lane, monkeypatch, crop=True)
    full = _solve(axis, lane, monkeypatch, crop=False)

    loops_c, loops_f = crop["loops"], full["loops"]
    # This must stay strict, and it must not be "fixed" by trimming or
    # zipping the longer list. Every assertion below is a paired comparison
    # between one cropped call and the corresponding full-plane one, so an
    # unequal count means the pairing is wrong and NOTHING below is
    # meaningful. It caught a real defect in this test's own scaffolding
    # during review: reading the pristine ``msl_loop_current`` off the
    # module inside ``_run_s`` chained the second solve's spy onto the
    # first's, and both runs' calls landed in the first ``loops`` list
    # (8 vs 4). A witness that dies here dies with and without the mutant,
    # i.e. it has no discriminating power at all — so if this fires, fix
    # the recording, never the assertion.
    assert loops_c and len(loops_c) == len(loops_f), (len(loops_c), len(loops_f))

    for idx, (lc, lf) in enumerate(zip(loops_c, loops_f)):
        where = f"{lane}-{axis} loop call {idx}"
        # the crop engaged for this call at all
        assert lc["ha_shape"][1:] < lf["ha_shape"][1:], where
        assert lc["hb_shape"] == lc["ha_shape"], where
        # the cell-size arrays were cropped with the plane
        assert lc["n_da"] == lc["ha_shape"][1], where
        assert lc["n_db"] == lc["ha_shape"][2], where

        # (a) same window WIDTH in both modes
        assert lc["a_hi"] - lc["a_lo"] == lf["a_hi"] - lf["a_lo"], where
        assert lc["b_hi"] - lc["b_lo"] == lf["b_hi"] - lf["b_lo"], where

        # (b) inset by exactly one cell on all four sides of the crop
        assert lc["a_lo"] == 1, where
        assert lc["b_lo"] == 1, where
        assert lc["a_hi"] == lc["ha_shape"][1] - 1, where
        assert lc["b_hi"] == lc["ha_shape"][2] - 1, where

        # (c) the cropped plane is the window plus its two margin cells
        assert lc["ha_shape"][1] == (lf["a_hi"] - lf["a_lo"]) + 2, where
        assert lc["ha_shape"][2] == (lf["b_hi"] - lf["b_lo"]) + 2, where

    # (d) and the region those offsets are relative to is the full-plane
    #     window ±1. Compared as an unordered pair of spans because the
    #     region is stated in the (width, normal) frame while the loop
    #     window is stated in the right-handed (a, b) frame, and the two
    #     swap for a "+y"/"-y" port (meta["a_is_width"] is False there).
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
    """
    crop = _solve(axis, lane, monkeypatch, crop=True)
    full = _solve(axis, lane, monkeypatch, crop=False)
    s_crop, z_crop, seen_crop = crop["s"], crop["z"], crop["regions"]
    s_full, z_full, seen_full = full["s"], full["z"], full["regions"]

    # the crop really engaged in the first run: every internal MSL plane got
    # a rectangle, and it is strictly smaller than the plane
    regions = [r for r in seen_crop if r is not None]
    assert regions and len(regions) == len(seen_crop), seen_crop
    assert all(len(r) == 4 and r[1] > r[0] and r[3] > r[2] for r in regions)
    assert all(r is not None for r in seen_full)  # the spy saw the same regions...
    # ...and the second run reset every one of them (so it went full-plane)

    assert s_crop.shape == s_full.shape == (2, 2, 3)

    # ---------------------------------------------------------------
    # The bound is a COMPILED-PROGRAM IDENTITY budget, not a physics
    # tolerance (#876). Both runs solve the same fields with the same
    # extractor; the only difference is the SHAPE of the DFT accumulator
    # — a cropped rectangle vs the full transverse plane — i.e. two
    # compiled programs for one computation.
    #
    # Where those ulps come from: update_dft_plane_probe is elementwise,
    # and the V and I reductions run in a fixed order that does not
    # depend on the accumulator shape (V is a Python-loop sum;
    # I is four `.sum(axis=1)` reductions over the four Ampere-loop legs
    # — rfx/sources/msl_port.py:1440-1444 — one per leg, each over an
    # unchanged index order). What changes is whether the XLA fusion
    # emitter contracts the complex multiply-add of the accumulate, and
    # on arm64 it contracts one shape and not the other (measured on
    # this fixture: 7-19 of 60 cropped Hy/Hz accumulator entries 1 ulp
    # from the full-plane ones; the Ez accumulators and V bitwise
    # identical in all 20 calls).
    #
    # Budget, in float32 ulps u = 2**-23:
    #   * the accumulate: the DFT bin is driven at the source frequency,
    #     so the sum is COHERENT (|acc| grows like N*|term| over the N
    #     steps) and N steps of at-most-1-ulp-per-step disagreement stay
    #     1 ulp RELATIVE, not N ulp. That is what the accumulators
    #     measure: exactly 1 ulp after all N steps.
    #   * V and I each inherit at most (n + 1)*u from a length-n
    #     summation — the standard forward-error bound
    #     gamma_n = n*u/(1 - n*u) ~ n*u.
    #   * S is assembled from both, so the two contributions add.
    #
    # Two honest caveats on that arithmetic, rather than a tighter
    # constant that would not survive them:
    #   * n_cells below is the crop's AREA, while the summations it is
    #     standing in for scale with the crop's PERIMETER — V sums one
    #     column and I sums four legs, all O(w) or O(h), not O(w*h). So
    #     the length term is deliberately generous, by roughly the
    #     aspect ratio.
    #   * the V,I -> S assembly is a division and a bilinear solve, not
    #     a sum, and it AMPLIFIES: measured at 2.12x on this fixture
    #     between the V/I disagreement and the S disagreement. The
    #     derivation carries no condition-number term for it; the
    #     generosity of the area term is what absorbs it. Both of these
    #     are reasons this gate is the weaker one, not reasons to trust
    #     it further.
    #
    # The budget is spent ABSOLUTELY at the scale of the matrix: an S
    # entry near zero is a difference of two O(1) wave amplitudes and
    # carries no relative accuracy of its own, so rtol=0 and
    # atol = ulps * u * max|S|. n_cells is read off the crop rectangles
    # this run actually used, so the bound follows the fixture instead
    # of being a written-in number.
    #
    # Measured (n_cells = 20 -> 41 u -> atol = 4.88e-06 at max|S| =
    # 0.9985):
    #   jax 0.10.2 / numpy 2.4.6, arm64: max |dS| = 1.49e-07 (uniform-x,
    #     the number in the issue), 1.34e-07 (uniform-y), 1.20e-07
    #     (nonuniform-x)
    #   jax 0.6.2 / numpy 2.2.6 (CI's versions), arm64: 3.04e-07
    #     (uniform-x), 2.39e-07 (uniform-y), 3.69e-07 (nonuniform-x)
    # i.e. 13x to 41x inside the budget, on both JAX versions.
    #
    # WHAT THIS GATE DOES AND DOES NOT KILL. It kills the Ez-rectangle
    # class: shifting the Ez crop rectangle by ONE index moves S by
    # 1.56e-05, 3.2x ABOVE the budget. It does NOT kill the j/k INDEX
    # class: mutating rfx/api/_sparams.py's
    # `_j_lo = meta["j_lo"] - _w_lo` to `+ 1` moves S by only 4.9958e-07
    # = 0.102x this atol, and the mutant passes all three lanes here.
    # No constant fixes that — the weakest surviving j/k mutant is
    # 3.73e-07 (uniform-y) while the jax 0.6.2 noise floor on the same
    # fixture is 3.69e-07 (nonuniform-x), about 1.01x apart. The j/k
    # class is gated on INDICES instead, bitwise, by
    # test_cropped_extractor_selects_exactly_the_full_plane_indices.
    # ---------------------------------------------------------------
    n_cells = max((r[1] - r[0]) * (r[3] - r[2]) for r in regions)
    ulps = 2 * n_cells + 1
    u = float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        s_crop, s_full, rtol=0.0,
        atol=ulps * u * float(np.max(np.abs(s_full))))
    np.testing.assert_allclose(
        z_crop, z_full, rtol=0.0,
        atol=ulps * u * float(np.max(np.abs(z_full))))
    assert np.all(np.isfinite(s_crop))

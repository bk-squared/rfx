"""Sparse DFT-region contracts for MSL port extraction."""

from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.probes.probes import init_dft_plane_probe, update_dft_plane_probe


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
# End-to-end: the cropped extractor arithmetic must reproduce the full-plane
# S-matrix. The three tests above never reach it — the uniform
# and nonuniform scans accumulate through their own hand-copies of the DFT
# update (rfx/simulation.py, rfx/nonuniform.py), not update_dft_plane_probe,
# and the registration test feeds the extractor full-plane fakes so it
# takes the legacy fallback. This one runs the real thru on both lanes and
# both port orientations (a_is_width True/False), once with the crop
# engaged and once with the crop forced off, and compares complex S to a
# derived two-shapes-one-program ulp budget (#876; the bitwise claim
# lives in test_sparse_monitor_matches_full_plane).
# Mutation twins: an off-by-one in the cropped j/k arithmetic, or a missing
# -1/+1 contour margin, moves S here and nowhere else in the suite — and
# still does under the derived budget: the off-by-one twin moves S 3x above
# it (measured, see the derivation).
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
    full transverse rectangle so the extractor takes its legacy path."""
    import warnings
    import rfx.probes.probes as _pp
    import rfx.runners.uniform as _ru

    real = init_dft_plane_probe

    def spy(*a, **k):
        seen.append(k.get("region"))
        if not crop:
            k["region"] = None
        return real(*a, **k)

    # the uniform runner bound the name at import; the NU runner and the
    # forward path import it inside the function
    monkeypatch.setattr(_pp, "init_dft_plane_probe", spy)
    monkeypatch.setattr(_ru, "init_dft_plane_probe", spy)
    sim = _thru_sim(axis, lane)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(n_freqs=3, num_periods=3)
    assert "_dft_plane_regions" not in vars(sim), \
        "the runtime crop dict must not survive compute_msl_s_matrix"
    return np.asarray(res.S), np.asarray(res.Z0)


@pytest.mark.parametrize("lane,axis", [("uniform", "x"), ("uniform", "y"),
                                       ("nonuniform", "x")])
def test_cropped_extractor_reproduces_full_plane_s_end_to_end(monkeypatch, lane, axis):
    """The cropped extractor is the SAME COMPUTATION as the full-plane one.

    A compiled-program identity check across two accumulator SHAPES, not a
    physics tolerance: the tolerance below is derived from the ulp budget
    two shapes of one program are allowed to differ by, and the exact
    index-level claim is gated bitwise by
    ``test_sparse_monitor_matches_full_plane``. See the derivation at the
    assertion.
    """
    seen_crop: list = []
    s_crop, z_crop = _run_s(axis, lane, monkeypatch, crop=True, seen=seen_crop)
    seen_full: list = []
    s_full, z_full = _run_s(axis, lane, monkeypatch, crop=False, seen=seen_full)

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
    # compiled programs for one computation. The exact, index-level
    # claim ("the crop retains exactly the requested samples") stays a
    # BITWISE gate in test_sparse_monitor_matches_full_plane above; this
    # test asks the weaker end-to-end question, so it may only spend the
    # ulps that two shapes of one program can differ by.
    #
    # Where those ulps come from: update_dft_plane_probe is elementwise
    # and V/I are Python-loop sums in identical order, so no reduction
    # order changes. What changes is whether the XLA fusion emitter
    # contracts the complex multiply-add of the accumulate, and on arm64
    # it contracts one shape and not the other (measured on this
    # fixture: 7-19 of 60 cropped Hy/Hz accumulator entries 1 ulp from
    # the full-plane ones; the Ez accumulators and V bitwise identical
    # in all 20 calls).
    #
    # Budget, in float32 ulps u = 2**-23:
    #   * the accumulate: the DFT bin is driven at the source frequency,
    #     so the sum is COHERENT (|acc| grows like N*|term| over the N
    #     steps) and N steps of at-most-1-ulp-per-step disagreement stay
    #     1 ulp RELATIVE, not N ulp. That is what the accumulators
    #     measure: exactly 1 ulp after all N steps.
    #   * V and I are each a fixed-order sum over at most n_cells
    #     accumulator entries, so each inherits at most (n_cells + 1)*u
    #     — the standard forward-error bound gamma_n = n*u/(1 - n*u) ~
    #     n*u for a length-n summation.
    #   * S is assembled from both, so the two contributions add:
    #     (2*n_cells + 1)*u.
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
    # i.e. 13x to 41x inside the budget, on both JAX versions. And the
    # mutation twin this test exists for still dies: shifting the crop
    # rectangle by ONE index moves S by 1.56e-05 here, 3.2x ABOVE the
    # budget.
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

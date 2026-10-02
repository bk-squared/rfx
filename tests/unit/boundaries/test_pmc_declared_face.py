"""Declared-plane magnetic image; physical H samples are not wall nodes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, Box, GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import CurlBoundary, h_neighbor


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", ["lo", "hi"])
def test_image_difference_ignores_stored_ghost(axis, side):
    h = jnp.arange(5 * 4 * 3, dtype=jnp.float32).reshape(5, 4, 3) + .37
    boundary = CurlBoundary(pmc_faces=frozenset({f"{'xyz'[axis]}_{side}"}))
    edge = [1, 1, 1]
    edge[axis] = 0 if side == "lo" else h.shape[axis] - 1
    idx = tuple(edge)
    inner = list(edge)
    inner[axis] = 0 if side == "lo" else h.shape[axis] - 2
    expected = (2 if side == "lo" else -2) * h[tuple(inner)]
    whole = h - h_neighbor(h, axis, boundary=boundary)
    point = h[idx] - h_neighbor(h, axis, boundary=boundary, index=idx)
    np.testing.assert_allclose(whole[idx], expected, rtol=4*np.finfo(np.float32).eps)
    np.testing.assert_array_equal(point, whole[idx])


def model(graded=False, high=False, **kwargs):
    profile = np.array([1., .9, 1.1, 1., .9, 1.1, 1., .9, 1.1, .9, 1.1, 1.]) * .001
    return Simulation(freq_max=20e9, domain=(.012, .0097, .0074), dx=.001,
                      boundary=BoundarySpec(x=Boundary("pec", "pmc") if high else Boundary("pmc", "pec"),
                                            y="pec", z="pec"),
                      **({"dx_profile": profile} if graded else {}), **kwargs)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("skip", [False, True])
def test_distributed_refuses_image_before_kernel(graded, entry, skip):
    sim = model(graded)
    kw = dict(n_steps=2, skip_preflight=skip, devices=[jax.devices()[0]] * 2)
    if entry == "forward":
        kw["distributed"] = True
    else:
        kw["compute_s_params"] = False
    kernel = "distributed_nu" if graded else "distributed_v2"
    with pytest.raises(NotImplementedError, match=f"x_lo.*{kernel}"):
        getattr(sim, entry)(**kw)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("skip", [False, True])
def test_fourth_order_refuses_unimaged_far_neighbors(entry, skip):
    sim = model(stencil_order=4)
    with pytest.raises(NotImplementedError, match="x_lo.*stencil_order=4.*far neighbors"):
        getattr(sim, entry)(n_steps=2, skip_preflight=skip)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("end", [0., -.00025, .00025])
@pytest.mark.parametrize("high", [False, True])
def test_smoothing_refuses_normal_interface_near_face(graded, end, high):
    sim = model(graded, high=high)
    sim.add_material("dielectric", eps_r=3.)
    low, upper = ((.004, .012-end) if high else (end, .008))
    sim.add(Box((low, .002, .002), (upper, .007, .005)), material="dielectric")
    face = "x_hi" if high else "x_lo"
    with pytest.raises(NotImplementedError, match=f"subpixel smoothing.*{face}.*half a cell"):
        sim.run(n_steps=2, skip_preflight=True, compute_s_params=False,
                subpixel_smoothing=True)


@pytest.mark.parametrize("graded", [False, True])
def test_smoothing_allows_material_extended_through_face(graded):
    sim = model(graded)
    sim.add_material("dielectric", eps_r=3.)
    sim.add(Box((-.002, -.002, -.002), (.015, .013, .012)), material="dielectric")
    sim.add_source((.002, .003, .002), "ez", waveform=GaussianPulse(f0=12e9),
                   amplitude_kind="field")
    result = sim.run(n_steps=8, skip_preflight=True, compute_s_params=False,
                     subpixel_smoothing=True)
    assert np.isfinite(np.asarray(result.state.ez)).all()


@pytest.mark.parametrize("cells", [4, 8, 16])
def test_crossing_flux_face_has_half_dual_area(cells):
    from rfx.probes.probes import init_flux_monitor, flux_spectrum
    def plane(half):
        n = cells + 1 if half else 2*cells + 1
        mon = init_flux_monitor(1, 1, jnp.array([1e9]), (n, 3, 4), 1./cells, .25,
                                pmc_faces=frozenset({"x_lo"}) if half else frozenset())
        # Even Ez,Hx on a y-normal plane; zero at the two far PEC ends.
        x = np.arange(cells+1) if half else np.abs(np.arange(-cells, cells+1))
        profile = jnp.asarray((1-x/cells)[:, None] * np.ones((1, 4)))[None]
        return mon._replace(e1_dft=profile.astype(complex), h2_dft=profile.astype(complex))
    half = flux_spectrum(plane(True))
    full = flux_spectrum(plane(False))
    np.testing.assert_array_equal(2 * half, full)


def test_collapsed_axis_has_no_high_ghost_to_canonicalize():
    from rfx.boundaries.pmc import apply_pmc_faces
    from rfx.core.yee import FDTDState
    fields = [jnp.full((3, 4, 1), i + 1.) for i in range(6)]
    state = FDTDState(*fields, jnp.array(0))
    result = apply_pmc_faces(state, frozenset({"z_hi"}), image=True)
    for before, after in zip(state, result):
        np.testing.assert_array_equal(before, after)

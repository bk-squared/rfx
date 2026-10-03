"""#1373: conformal PEC must not replace dielectric edge means with cells."""
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, simulation
from rfx.boundaries.pec import apply_pec_faces, resolve_wall_faces
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import (
    component_e_materials, edge_mean_components, init_state, lumped_components,
    permittivity_without_lumped,
)


def _run(conformal, *, cut=False, smooth=False, loaded=False, periodic_y=False,
         cut_capacitor=False):
    # Binary-exact dx and x walls avoid a nearly-one SDF weight caused by
    # coordinate roundoff. y/z lengths are not integral numbers of cells.
    # A periodic axis must hold a whole number of cells (16 along y).
    ny = 16 if periodic_y else 16.7
    sim = Simulation(freq_max=10e9, domain=(13/1024, ny/1024, 19.2/1024),
                     dx=1/1024, boundary=BoundarySpec(
                         x=Boundary(lo="pec", hi="pec", conformal=True),
                         y="periodic" if periodic_y else "pec", z="pec"))
    sim.add_material("diel", eps_r=4.0, sigma=0.03 if loaded else 0.0)
    # Periodic: the block starts at the y=0 seam and does not wrap, so a
    # wrapped and an edge-replicated mean differ on the seam edges.
    ylo, yhi = (0.0, .0042) if periodic_y else (.0032, .0106)
    sim.add(Box((.0041, ylo, .0063), (.0094, yhi, .0148)), material="diel")
    if cut:
        sim.add(Box((-.003,) * 3, (.0057, .021, .024)), material="pec")
    if loaded:
        sim.add_lumped_rlc((.007, .006, .010), component="ez", C=1e-13,
                           R=100, topology="parallel")
    if cut_capacitor:
        # An Ez capacitor on the node just outside the cut x wall (x = 6 dx),
        # where the Ez weight is fractional (0 < w < 1).
        sim.add_lumped_rlc((6/1024, 8/1024, 10.5/1024), component="ez",
                           C=1e-13, topology="parallel")
    sim.add_source((.007, .005, .008), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.009, .007, .012), "ez")
    seen = {}
    original = simulation.run

    def capture(grid, materials, n_steps, **kwargs):
        inverse = kwargs.get("aniso_inv_eps")
        seen["inverse"] = None if inverse is None else np.asarray(inverse)
        periodic = simulation.resolve_periodic(grid, kwargs.get("periodic"))
        plain, sigma = component_e_materials(materials, periodic)
        eps = kwargs.get("aniso_eps") or plain
        state = init_state(grid.shape)._replace(ex=eps[0], ey=eps[1], ez=eps[2])
        faces = resolve_wall_faces(grid, periodic)[0]
        # Wall-fixed tangential edges have no material degree of freedom.
        # Their SDF weights are 1/2 even at a grid-aligned wall; apply the
        # actual wall operator, rather than discarding an arbitrary margin.
        realized = apply_pec_faces(state, faces)
        live = apply_pec_faces(init_state(grid.shape)._replace(
            ex=jnp.ones(grid.shape), ey=jnp.ones(grid.shape),
            ez=jnp.ones(grid.shape)), faces)
        stamps = lumped_components(getattr(materials, "eps_r_lumped", None))
        seen["lumped_eps"] = np.asarray([np.zeros(grid.shape) if st is None
                                         else np.asarray(st) for st in stamps])
        seen["volume_mean"] = np.asarray(edge_mean_components(
            permittivity_without_lumped(materials), periodic))
        seen.update(eps=np.asarray(eps), plain=np.asarray(plain),
                    realized=np.asarray((realized.ex, realized.ey, realized.ez)),
                    live=np.asarray((live.ex, live.ey, live.ez), dtype=bool),
                    sigma=np.asarray(sigma), cell=np.asarray(materials.eps_r),
                    weights=(None if kwargs.get("conformal_weights") is None
                             else np.asarray(kwargs["conformal_weights"])))
        return original(grid, materials, n_steps, **kwargs)

    with patch.object(simulation, "run", capture):
        result = sim.run(n_steps=240, skip_preflight=True, compute_s_params=False,
                         conformal_pec=conformal, subpixel_smoothing=smooth)
    seen["trace"] = np.asarray(result.time_series)
    return seen


def _compare_arrays(actual, reference):
    np.testing.assert_array_equal(actual, reference)


def _compare_trace(actual, reference):
    peak = np.float32(np.max(np.abs(reference)))
    assert peak > 0
    # Cross-trace contract: nine float32 ULP of the reference peak, at
    # EVERY sample, including zero crossings. No relative per-sample bar.
    np.testing.assert_allclose(actual, reference, rtol=0,
                               atol=9 * np.spacing(peak))


@pytest.fixture(scope="module")
def aligned():
    return _run(False), _run(True)


def test_grid_aligned_dielectric_run_matches_plain(aligned):
    plain, conformal = aligned
    live_weights = conformal["weights"][conformal["live"]]
    assert not np.any((live_weights > 0) & (live_weights < 1))
    _compare_arrays(conformal["realized"], plain["realized"])
    _compare_trace(conformal["trace"], plain["trace"])


def test_every_dielectric_edge_away_from_walls_uses_four_cells(aligned):
    plain, conformal = aligned
    for c in range(3):
        interface = ((plain["eps"][c] > 1) & (plain["eps"][c] < 4)
                     & (conformal["weights"][c] == 1))
        assert np.count_nonzero(interface) > 0
        assert np.any(plain["eps"][c][interface] != plain["cell"][interface])
        _compare_arrays(conformal["eps"][c][interface], plain["eps"][c][interface])


@pytest.mark.parametrize("smooth", [False, True])
def test_cut_dielectric_edges_keep_mean_over_weight(smooth):
    result = _run(True, cut=True, smooth=smooth)
    for c in range(3):
        eps, w = result["plain"][c], result["weights"][c]
        cut = (w > 0) & (w < 1) & (eps > 1) & (eps < 4) & result["live"][c]
        assert np.count_nonzero(cut) > 0
        _compare_arrays(result["eps"][c][cut], (eps / np.where(w > 0, w, 1))[cut])
    if smooth:
        smoothed = _run(False, cut=True, smooth=True)
        free = result["weights"] == 1
        _compare_arrays(result["eps"][free], smoothed["eps"][free])


def test_capacitor_on_a_cut_edge_is_added_after_the_weight_division():
    # A lumped C is a device on its edge, not volume: the conformal 1/w
    # scales the volume mean only, and the capacitor is added afterwards.
    result = _run(True, cut=True, cut_capacitor=True)
    eps, w = result["eps"][2], result["weights"][2]
    f32 = np.float32
    mean = result["volume_mean"][2].astype(f32)
    stamp = result["lumped_eps"][2].astype(f32)
    w = w.astype(f32)
    on_cut = (stamp != 0) & (w > 0) & (w < 1) & result["live"][2]
    assert np.count_nonzero(on_cut) == 1
    expected = mean / w + stamp
    _compare_arrays(eps[on_cut], expected[on_cut])
    # The opposite order, (mean + C) / w, is a different number here.
    assert np.all(((mean + stamp) / w)[on_cut] != expected[on_cut])


def test_loss_and_edge_owned_lumped_materials_match_plain():
    plain, conformal = _run(False, loaded=True), _run(True, loaded=True)
    _compare_arrays(conformal["realized"], plain["realized"])
    _compare_arrays(conformal["sigma"], plain["sigma"])
    _compare_trace(conformal["trace"], plain["trace"])


@pytest.mark.parametrize("kind", ["eps", "trace"])
def test_comparison_rejects_corrupted_observation(kind, aligned):
    # Mutation guard: deleting either comparison must itself turn red.
    plain, _ = aligned
    if kind == "eps":
        bad = plain["realized"].copy()
        bad[0, 5, 5, 8] += 1
        with pytest.raises(AssertionError):
            _compare_arrays(bad, plain["realized"])
    else:
        bad = plain["trace"].copy()
        bad[100] += 32 * np.spacing(np.float32(np.max(np.abs(bad))))
        with pytest.raises(AssertionError):
            _compare_trace(bad, plain["trace"])


def test_waveguide_conformal_builder_uses_component_means(monkeypatch):
    from rfx.sparams import waveguide

    sim = Simulation(freq_max=10e9, domain=(.060, .025, .012), dx=.002,
                     cpml_layers=4, boundary=BoundarySpec(
                         x="cpml", y=Boundary(lo="pec", hi="pec", conformal=True),
                         z="pec"))
    sim.add_material("diel", eps_r=4)
    sim.add(Box((.023, .004, .003), (.035, .017, .008)), material="diel")
    sim.add_waveguide_port(.012, direction="+x", f0=8e9, name="left")
    sim.add_waveguide_port(.048, direction="-x", f0=8e9, name="right")

    class Captured(Exception):
        pass

    def capture(grid, materials, ref_materials, cfgs, n_steps, **kwargs):
        plain = component_e_materials(materials)[0]
        weights = kwargs["conformal_weights"]
        for actual, eps, w, ref in zip(kwargs["aniso_eps"], plain, weights,
                                       kwargs["ref_aniso_eps"]):
            safe = jnp.where(w > 0, w, 1)
            _compare_arrays(actual, eps / safe)
            _compare_arrays(ref, 1 / safe)
        raise Captured

    monkeypatch.setattr(waveguide, "extract_waveguide_s_params_normalized", capture)
    with pytest.raises(Captured):
        sim.compute_waveguide_s_matrix(n_steps=2, normalize=True)


def test_kottke_stage2_does_not_take_stage1_correction(monkeypatch):
    from rfx.geometry import conformal

    def forbidden(*args, **kwargs):
        pytest.fail("Stage 2 must use its own inverse tensor")

    monkeypatch.setattr(conformal, "conformal_eps_correction", forbidden)
    plain = _run(False, smooth="kottke_pec")
    enabled = _run(True, smooth="kottke_pec")
    assert enabled["weights"] is None
    assert enabled["inverse"] is not None
    _compare_arrays(enabled["inverse"], plain["inverse"])
    _compare_trace(enabled["trace"], plain["trace"])


def test_periodic_dielectric_seam_uses_the_plain_wrap():
    # The block crosses the periodic y seam, so the plain edge mean wraps
    # there. Only the permittivity is compared: with a periodic axis the
    # conformal and plain runs already differ in vacuum (6e-6 to 8e-6 of the
    # probe peak; cause not measured), which is not this defect.
    plain, conformal = _run(False, periodic_y=True), _run(True, periodic_y=True)
    free = conformal["weights"] == 1
    seam = free.copy()
    seam[:, :, 1:-1, :] = False
    seam &= (plain["eps"] > 1) & (plain["eps"] < 4)
    assert np.count_nonzero(seam) > 0
    _compare_arrays(conformal["eps"][free], plain["eps"][free])


def test_occupancy_kottke_lane_starts_from_the_four_cell_mean(monkeypatch):
    """forward()'s opt-in occupancy-Kottke lane with zero occupancy equals plain forward()."""
    import jax.numpy as jnp
    from rfx import Box, GaussianPulse, Simulation

    def build():
        sim = Simulation(freq_max=10e9, domain=(13 / 1024, 16.7 / 1024, 19.2 / 1024),
                         dx=1 / 1024, boundary="pec")
        sim.add_material("diel", eps_r=4.0)
        sim.add(Box((.0041, .0032, .0063), (.0094, .0106, .0148)), material="diel")
        sim.add_source((.007, .005, .008), "ez", amplitude_kind="field",
                       waveform=GaussianPulse(f0=5e9, bandwidth=.8))
        sim.add_probe((.009, .007, .012), "ez")
        return sim

    plain = np.asarray(build().forward(n_steps=240, skip_preflight=True).time_series)
    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "1")
    sim = build()
    zeros = jnp.zeros(sim._build_grid().shape, jnp.float32)
    lane = np.asarray(sim.forward(n_steps=240, skip_preflight=True,
                                  pec_occupancy_override=zeros).time_series)
    # The lane runs the inverse-permittivity kernel, so it rounds differently
    # from plain forward(): measured 1.2e-5 of the peak. The per-cell baseline
    # this replaces read 4.8e-3.
    peak = np.max(np.abs(plain))
    assert np.max(np.abs(lane - plain)) <= 1e-4 * peak

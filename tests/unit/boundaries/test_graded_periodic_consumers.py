"""Periodic graded consumers use the declared period at the seam (#1221)."""

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.smoothing import compute_smoothed_eps, compute_smoothed_eps_nonuniform


def _sim(graded):
    profiles = dict(dy_profile=np.full(7, .001), dz_profile=np.full(9, .001)) if graded else {}
    return Simulation(30e9, (.012, .007, .009), dx=.001, cpml_layers=0,
                      boundary=BoundarySpec(x="periodic", y="pec", z="pec"), **profiles)


def test_smoothed_seam_matches_uniform():
    block = Box((.0095, .002, .003), (.012, .005, .006))
    uniform = compute_smoothed_eps(_sim(False)._build_grid(), [(block, 4.)])
    graded = compute_smoothed_eps_nonuniform(_sim(True)._build_nonuniform_grid(), [(block, 4.)])
    for actual, expected in zip(graded, uniform, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    for eps in (uniform, graded):
        np.testing.assert_array_equal(eps[1][0, 2:5, 4:6], 2.5)


@pytest.mark.parametrize("progressive", [False, True])
def test_whole_period_design_region(progressive):
    from rfx.optimize import DesignRegion, ProgressiveStage, optimize, progressive_optimize

    region = DesignRegion((0., .001, .002), (.012, .005, .006), eps_range=(1., 4.))
    shapes = []
    for graded in (False, True):
        def factory(dx):
            sim = _sim(graded)
            sim.add_source((.005, .003, .004), "ez")
            sim.add_probe((.007, .003, .004), "ez")
            return sim

        def objective(result):
            return jnp.sum(result.time_series ** 2)

        if progressive:
            result = progressive_optimize(
                factory, region, objective, [ProgressiveStage(dx=.001, n_iters=1, n_steps=4)],
                verbose=False, skip_preflight=True)
            shapes.append(result.final_latent.shape)
        else:
            result = optimize(factory(.001), region, objective, n_iters=1, n_steps=4,
                              verbose=False, skip_preflight=True)
            shapes.append(result.latent.shape)
    assert shapes[0] == shapes[1]
    assert shapes[1][0] == 12


def test_wire_extent_at_period_end():
    from rfx.preflight.ports import _wire_port_realized_cells
    from rfx.realized_geometry import realized_geometry

    cells, records = [], []
    for graded in (False, True):
        sim = _sim(graded)
        sim.add_port((.009, .003, .004), "ex", extent=.003)
        grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
        cells.append(_wire_port_realized_cells(grid, sim._ports[0]))
        records.append(realized_geometry(sim).ports)
    assert cells[0] == cells[1] == [(9, 3, 4), (10, 3, 4), (11, 3, 4)]
    assert records[0] == records[1]


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_periodic_profile_must_span_declared_period(entry):
    kw = dict(freq_max=30e9, domain=(.012, .007, .012), dx=.001, cpml_layers=0,
              boundary=BoundarySpec(x="pec", y="pec", z="periodic"))
    sim = Simulation(**kw, dz_profile=np.full(10, .001))
    with pytest.raises(ValueError, match=r"declared period 0.012 m.*realized period 0.01 m"):
        getattr(sim, entry)(n_steps=1, skip_preflight=True)
    grid = Simulation(**kw, dz_profile=np.full(12, .001))._build_nonuniform_grid()
    assert grid.nz == 12


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("wire", [False, True])
def test_periodic_eager_extractors_refuse(graded, wire):
    from rfx.core.yee import init_materials
    from rfx.probes.probes import extract_s_matrix, extract_s_matrix_wire
    from rfx.sources.sources import GaussianPulse, LumpedPort, WirePort

    sim = _sim(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    port = (WirePort(start=(.004, .003, .004), end=(.004, .003, .006), component="ez")
            if wire else LumpedPort(position=(.004, .003, .004), component="ez",
                                    impedance=50., excitation=GaussianPulse(f0=10e9)))
    extract = extract_s_matrix_wire if wire else extract_s_matrix
    with pytest.raises(NotImplementedError, match="the S-parameter extractors do not wrap periodic axes"):
        extract(grid, init_materials(grid.shape), [port], jnp.array([10e9]), n_steps=4)


def test_periodic_flux_stop_matches_uniform():
    from rfx.sources.sources import GaussianPulse

    lengths = []
    for graded in (False, True):
        profiles = dict(dz_profile=np.full(16, .001)) if graded else {}
        sim = Simulation(14e9, (.012, .016, .016), dx=.001, cpml_layers=8,
                         boundary=BoundarySpec(x="periodic", y="cpml", z="cpml"), **profiles)
        sim.add_source((.0055, .0085, .0085), "ez",
                       waveform=GaussianPulse(f0=10e9, bandwidth=1., cutoff=5),
                       amplitude_kind="current")
        sim.add_probe((.0025, .0065, .0075), "ez")
        result = sim.run(until_decay=1e-3, decay_check_interval=20, decay_min_steps=100,
                         decay_max_steps=3000, skip_preflight=True,
                         radiated_flux_box=((0., .003, .003), (.012, .013, .013)))
        lengths.append(len(result.time_series))
    assert lengths[0] == lengths[1] == 200


@pytest.mark.parametrize("seam", [False, True])
def test_ntff_record_and_runner_share_box(seam, monkeypatch):
    from rfx import farfield

    lo = (0. if seam else .002, .004, .004)
    hi = (.012 if seam else .010, .012, .012)
    from rfx.sources.sources import GaussianPulse

    sims = []
    for graded in (False, True):
        profiles = dict(dz_profile=np.full(16, .001)) if graded else {}
        sim = Simulation(14e9, (.012, .016, .016), dx=.001, cpml_layers=8,
                         boundary=BoundarySpec(x="periodic", y="cpml", z="cpml"), **profiles)
        sim.add_source((.0055, .0085, .0085), "ez",
                       waveform=GaussianPulse(f0=10e9, bandwidth=1., cutoff=5),
                       amplitude_kind="current")
        sim.add_probe((.0025, .0065, .0075), "ez")
        sims.append(sim)
    rows = []
    for sim in sims:
        sim.add_ntff_box(lo, hi, freqs=[10e9])
        rows.append(next(row for row in sim.realized_geometry().probes if row.kind == "ntff"))
    message = "NTFF box requires an interior plane with a sampling margin; move its high face inside the period"
    if seam:
        for row in rows:
            assert row.index_box is None
            assert message in row.error
        with pytest.raises(ValueError, match=message):
            sims[1].run(n_steps=1, skip_preflight=True)
    else:
        expected = ((2, 10), (12, 20), (12, 20))
        assert rows[0].index_box == rows[1].index_box == expected
        captured = []
        original = farfield.make_ntff_box

        def capture(*args, **kwargs):
            import inspect
            box = original(*args, **kwargs)
            if inspect.currentframe().f_back.f_code.co_name == "run_nonuniform_path":
                captured.append(((box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi)))
            return box

        monkeypatch.setattr(farfield, "make_ntff_box", capture)
        sims[1].run(n_steps=1, skip_preflight=True)
        assert captured == [rows[1].index_box]


def test_oriented_box_mask_matches_uniform():
    from rfx.geometry.csg import OrientedBox

    shape = OrientedBox(center=(.012, .003, .004), size=(.004, .002, .002),
                        rotation=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    np.testing.assert_array_equal(shape.mask(_sim(True)._build_nonuniform_grid()),
                                  shape.mask(_sim(False)._build_grid()))


def test_dc_film_crossing_the_seam_matches_uniform():
    """A lossy film y 8-12 mm on a 10 mm period covers y 8-10 and 0-2 mm: ex edges on nodes 8, 9, 0, 1, 2,
    ey edges in cells 8, 9, 0, 1, on both meshes (record 12 addendum; review A's crossing scene)."""
    from rfx.core.yee import lumped_components

    films = []
    for graded in (False, True):
        profile = {"dz_profile": np.full(8, 1e-3)} if graded else {}
        sim = Simulation(20e9, (.008, .010, .008), dx=1e-3, cpml_layers=0,
                         boundary=BoundarySpec(x="pec", y="periodic", z="pec"), **profile)
        sim.add_thin_conductor(Box((.002, .008, .004), (.006, .012, .004)), sigma_bulk=1e3, thickness=1e-4)
        grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
        materials = (sim._assemble_materials_nu(grid) if graded else sim._assemble_materials(grid))[0]
        films.append([None if p is None else np.asarray(p) for p in lumped_components(materials.sigma_film)])
    uniform, graded = films
    rows = lambda p: sorted(set(np.nonzero(p)[1].tolist()))
    assert rows(uniform[0]) == rows(graded[0]) == [0, 1, 2, 8, 9]
    assert rows(uniform[1]) == rows(graded[1]) == [0, 1, 8, 9]
    for u, g in zip(uniform, graded):
        if u is not None:
            np.testing.assert_allclose(g, u, rtol=1e-6, atol=0)


@pytest.mark.parametrize("film", ["box", "cylinder"])
def test_dc_film_on_a_traced_periodic_mesh_refuses(film):
    """A traced graded mesh has no seam-image sampler for a lossy film: refused, never folded without its image
    (review A: a cylinder film on the seam kept a third of its conductance; a Box film crashed)."""
    import jax
    from rfx import Cylinder

    shape = (Box((.002, .005, .004), (.006, .010, .004)) if film == "box" else
             Cylinder(center=(.004, .010, .004), radius=.0015, height=1e-4, axis="z"))
    dz = np.array([1e-3] * 3 + [.5e-3] * 4 + [1e-3] * 3, dtype=np.float32)

    def fold(profile):
        sim = Simulation(20e9, (.008, .010, float(np.sum(dz.astype(np.float64)))), dx=1e-3, cpml_layers=0,
                         boundary=BoundarySpec(x="pec", y="periodic", z="pec"), snap="declared",
                         dz_profile=profile)
        sim.add_thin_conductor(shape, sigma_bulk=1e3, thickness=1e-4)
        grid = sim._build_nonuniform_grid()
        return sim._assemble_materials_nu(grid, pec_sheets=[], pec_wires=[])[0].sigma_film

    with pytest.raises(NotImplementedError, match="lossy thin-conductor film on a traced graded mesh with periodic"):
        jax.jit(fold)(jnp.asarray(dz))

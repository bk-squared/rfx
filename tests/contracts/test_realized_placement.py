"""Realized source/probe placement, time step and mesh scale in the record.

``Simulation.realized_geometry()`` must report the indices the run actually
injects and samples at. The run's own structures are read at the kernel call
(the stepping function is replaced by one that stops before the first step),
so the expected side never calls the helper the record calls. One case is
also checked against a hand computation from the declared node coordinates:
a soft source between two nodes inside a graded 0.75 mm cell.
"""
import warnings

import numpy as np
import pytest

import rfx.runners.nonuniform as nu_runner
import rfx.simulation as uniform_kernel
from rfx import Box, Simulation, _realized

C0 = 299_792_458.0
DZ = np.array([1e-3] * 3 + [0.75e-3] * 6 + [1e-3] * 3)
SOURCE = (0.0047, 0.0051, 0.0050)   # z = 5.0 mm: graded nodes 4.5 / 5.25 mm
PROBE = (0.0061, 0.0049, 0.0057)


class _Stop(Exception):
    pass


def _scene(graded):
    kw = dict(dz_profile=DZ) if graded else {}
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.0 if graded else 0.012),
                     dx=1e-3, boundary="cpml", cpml_layers=4, **kw)
    sim.add_source(SOURCE, "ez", amplitude_kind="current")
    sim.add_port((0.008, 0.006, 0.002), "ez", impedance=50.0, extent=0.004)
    sim.add_probe(PROBE, "ez")
    sim.add_dft_plane_probe(axis="z", coordinate=0.006, component="ez", n_freqs=3, name="pz")
    sim.add_flux_monitor(axis="x", coordinate=0.009, n_freqs=3, size=(0.004, 0.004),
                         center=(0.006, 0.006))
    sim.add_ntff_box((0.002, 0.002, 0.002), (0.010, 0.010, 0.008), n_freqs=3)
    return sim


def _kernel_inputs(sim, graded, monkeypatch):
    """What the runner hands its stepping function, without stepping."""
    seen = {}

    def stop(*args, **kwargs):
        seen.update(kwargs, grid=args[0])
        raise _Stop

    monkeypatch.setattr(nu_runner if graded else uniform_kernel,
                        "run_nonuniform" if graded else "run", stop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_Stop):
            sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    return seen


def _cell(spec):
    return ((spec.i, spec.j, spec.k), spec.component) if hasattr(spec, "i") else \
        (tuple(int(v) for v in spec[:3]), spec[3])


@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
def test_record_places_sources_and_probes_where_the_run_does(graded, monkeypatch):
    sim = _scene(graded)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = sim.realized_geometry()
    run = _kernel_inputs(sim, graded, monkeypatch)

    # Sources: the soft source row plus the wire port's driven edges are
    # exactly the cells the run injects at.
    (soft,) = [s for s in record.sources if s.kind == "soft_point"]
    assert soft.error is None and len(soft.edges) == 1
    wire = next(p for p in record.ports if p.kind == "wire")
    expected = {(tuple(c), "ez") for c in (*soft.edges, *wire.edges)}
    assert {(tuple(int(v) for v in c), comp) for c, comp in map(_cell, run["sources"])} == expected
    assert (tuple(soft.edges[0]), "ez") in {_cell(s) for s in run["sources"]}

    # Probes.
    point = next(p for p in record.probes if p.kind == "point")
    (probe,) = run["probes"]
    assert point.index == _cell(probe)[0]
    plane = next(p for p in record.probes if p.kind == "dft_plane")
    (dft,) = run["dft_planes"]
    assert plane.index_box[dft.axis] == (dft.index, dft.index)
    flux_row = next(p for p in record.probes if p.kind == "flux")
    (flux,) = run["flux_monitors"]
    assert flux_row.index_box[flux.axis] == (flux.index, flux.index)
    t = [a for a in range(3) if a != flux.axis]
    assert flux_row.index_box[t[0]] == (flux.lo1, flux.hi1 - 1)
    assert flux_row.index_box[t[1]] == (flux.lo2, flux.hi2 - 1)
    ntff_row = next(p for p in record.probes if p.kind == "ntff")
    box = run["ntff_box" if graded else "ntff"]
    assert ntff_row.index_box == ((box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi))

    # Time step: the grid object the stepping function receives.
    assert record.dt == float(run["grid"].dt) == float(sim._build_realized_grid().dt)


@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
def test_soft_source_index_by_hand(graded):
    """Nearest node on the declared node line, plus the absorber padding."""
    sim = _scene(graded)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = sim.realized_geometry()
    pads = [row.padding[0] for row in record.domain]
    x_nodes = np.arange(13) * 1e-3
    z_nodes = np.concatenate([[0.0], np.cumsum(DZ)]) if graded else x_nodes
    hand = tuple(int(np.argmin(np.abs(nodes - p))) + pad
                 for nodes, p, pad in zip((x_nodes, x_nodes, z_nodes), SOURCE, pads))
    # z = 5.0 mm: nearest graded node 5.25 mm (index 6); uniform node 5 mm.
    assert hand[2] - pads[2] == (6 if graded else 5)
    (soft,) = [s for s in record.sources if s.kind == "soft_point"]
    assert soft.edges == (hand,)


def test_cells_per_wavelength_by_hand():
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.012), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("d4", eps_r=4.0)
    sim.add(Box((0.003, 0.003, 0.003), (0.008, 0.008, 0.008)), material="d4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scale = sim.realized_geometry().cells_per_wavelength
    vacuum = C0 / 10e9 / 1e-3
    assert scale.vacuum == pytest.approx(vacuum, rel=1e-12)
    assert scale.medium == pytest.approx(vacuum / 2, rel=1e-12)
    assert scale.max_eps_mu == 4.0


@pytest.mark.parametrize("smoothing", (False, True))
def test_record_materials_are_the_stepped_cell_materials(smoothing):
    """The record holds the cell eps_r the stepped MaterialArrays carry.

    With smoothing on the E update consumes a per-component tensor formed
    from these cells; the record does not hold that tensor (measured below).
    """
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.012), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("d4", eps_r=4.0)
    sim.add(Box((0.0035, 0.0042, 0.0031), (0.0081, 0.0077, 0.0068)), material="d4")
    sim.add_source((0.0047, 0.0051, 0.0043), "ez", amplitude_kind="current")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = sim.realized_geometry()
        with _realized.capture() as cap:
            sim.run(n_steps=2, subpixel_smoothing=smoothing, skip_preflight=True)
    (e_site,) = [r for r in cap.records if "eps_e" in r]
    assert e_site["site"] == ("aniso.E" if smoothing else "yee.E")
    np.testing.assert_array_equal(np.asarray(e_site["materials"].eps_r),
                                  np.asarray(record.materials.eps_r))
    if not smoothing:
        from rfx.core.yee import component_e_materials
        means = component_e_materials(record.materials, e_site["periodic"], high_wall=(False, False, False))[0]
        for consumed, mean in zip(e_site["eps_e"], means):
            np.testing.assert_array_equal(np.asarray(consumed), np.asarray(mean))


def test_cache_sees_a_probe_added_after_the_first_record():
    sim = _scene(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = sim.realized_geometry()
        sim.add_probe((0.003, 0.003, 0.003), "ex")
        after = sim.realized_geometry()
    assert len(after.probes) == len(before.probes) + 1


def test_library_probes_are_not_listed_as_the_users():
    """A probe an extractor attaches (MSL / mixed witness, ring-down channel) is no user probe."""
    from types import SimpleNamespace
    from rfx.ringdown import _user_geometry

    sim = _scene(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = sim.realized_geometry()
        n_user = len(sim._probes)
        sim.add_probe((0.003, 0.003, 0.003), "ex")
        sim.add_probe((0.004, 0.003, 0.003), "ey")
        both = sim.realized_geometry()
        sim._internal_probe_indices = {n_user}
        marked = sim.realized_geometry()
    points = lambda rec: [p.label for p in rec.probes if p.kind == "point"]  # noqa: E731
    assert points(both) == points(before) + [f"probe[{n_user}]", f"probe[{n_user + 1}]"]
    assert points(marked) == points(before) + [f"probe[{n_user + 1}]"]

    class _Result(SimpleNamespace):
        def _replace(self, **kw):
            return _Result(**{**vars(self), **kw})

    cut = _user_geometry(_Result(realized_geometry=both), n_user, 2)
    assert points(cut.realized_geometry) == points(before)
    others = lambda rec: [p for p in rec.probes if p.kind != "point"]  # noqa: E731
    assert others(cut.realized_geometry) == others(both)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_tfsf_box_is_the_one_the_run_injects(graded, monkeypatch):
    # TFSF refuses soft sources and lumped ports, so it gets its own scene.
    from rfx.sources.tfsf import tfsf_injection_planes
    kw = dict(dz_profile=DZ) if graded else {}
    sim = Simulation(freq_max=10e9, domain=(0.016, 0.012, 0.0 if graded else 0.012),
                     dx=1e-3, boundary="cpml", cpml_layers=4, **kw)
    sim.add_tfsf_source(f0=5e9, polarization="ez", direction="+x")
    row = next(r for r in sim.realized_geometry().sources if r.kind == "tfsf")
    assert row.error is None, row.error
    cfg = _kernel_inputs(sim, graded, monkeypatch)["tfsf"][0]
    planes = tfsf_injection_planes(cfg)
    assert row.index_box == tuple(planes.get(a) for a in "xyz")

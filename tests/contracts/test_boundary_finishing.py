"""PR3a finishing judges: explicit rings and the retained default wrap."""
import gc

import jax
import pytest

from rfx import Box, Simulation
from rfx.boundaries.depths import Kind


class Admitted(Exception):
    pass


@pytest.fixture(autouse=True)
def no_field_steps(monkeypatch):
    def stop(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr(jax.lax, "scan", stop)
    yield
    jax.clear_caches()
    gc.collect()


def model(*, periodic=False, finite=False):
    kwargs = {"boundary": {"x": "cpml", "y": "periodic", "z": "periodic"}} if periodic else {}
    sim = Simulation(20e9, (.020, .011, .004), dx=.001, cpml_layers=8, **kwargs)
    sim.add_material("glass", eps_r=3.2)
    lo, hi = ((.007, .002, 0.), (.012, .006, .002)) if finite else (
        (.007, -.010, -.010), (.012, .021, .014))
    sim.add(Box(lo, hi), material="glass")
    sim.add_tfsf_source(f0=10e9, margin=3, polarization="ez")
    sim.add_probe((.015, .008, .001), "ez")
    return sim


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_explicit_periodic_keeps_declared_ring(skip, entry, monkeypatch):
    sim = model(periodic=True, finite=True)
    original = sim._build_grid
    observed = []

    def observe(**kwargs):
        grid = original(**kwargs)
        observed.append(grid)
        assert grid.shape == (37, 11, 4)
        assert tuple(f.realized for f in grid.boundary_depths) == (8, 8, 0, 0, 0, 0)
        assert tuple(f.kind for f in grid.boundary_depths)[2:] == (Kind.PERIODIC,) * 4
        assert sim._periodic_flags() == (False, True, True)
        return grid

    monkeypatch.setattr(sim, "_build_grid", observe)
    with pytest.raises(Admitted):
        getattr(sim, entry)(n_steps=1, skip_preflight=skip)
    assert observed


@pytest.mark.parametrize("skip", [False, True])
def test_default_wrap_accepts_slab_and_refuses_original_box(skip):
    slab = model()
    grid = slab._build_grid()
    assert grid.shape == (37, 28, 21)
    assert tuple(f.realized for f in grid.boundary_depths) == (8,) * 6
    finding = [i for i in slab.preflight() if i.code == "tfsf_transverse_periodic"]
    assert len(finding) == 1 and finding[0].severity == "warning"
    with pytest.raises(Admitted):
        slab.run(n_steps=1, skip_preflight=skip)
    with pytest.raises(ValueError, match="y_lo, y_hi.*declare boundary.*closed_box=True"):
        model(finite=True).run(n_steps=1, skip_preflight=skip)

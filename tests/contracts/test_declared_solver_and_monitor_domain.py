"""#931: builder guards read declarations; dispatch validates the resolved mesh."""

import pytest

from rfx import Box, Simulation








def _slabs():
    return [Box((0, 0, 0), (.03, .03, .0016)),
            Box((0, 0, .17), (.03, .03, .1716))]


@pytest.mark.parametrize("method", ["add_dft_plane_probe", "add_flux_monitor"])
@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("preview", [False, True])
def test_plane_registration_is_independent_of_geometry_order(method, stage, preview, monkeypatch):
    sim = Simulation(freq_max=10e9, domain=(.03, .03, .2), boundary="pec")
    slabs = _slabs()
    for slab in slabs[:stage]:
        sim.add(slab, material="fr4")
    if preview:
        # Auto mesh keeps the declared column as geometry is added (#1941).
        grid = sim._build_realized_grid()
        cells = grid.cells("z")[grid.pad_z_lo:grid.nz - grid.pad_z_hi - 1]
        if stage:
            assert sum(cells) == pytest.approx(.2, rel=1e-9)
        else:
            assert sim._domain[2] == pytest.approx(.2, rel=1e-9)

    def forbidden_resolution():
        pytest.fail("Registration resolved an unfinished model")

    with monkeypatch.context() as m:
        m.setattr(sim, "_resolve_mesh", forbidden_resolution)
        assert getattr(sim, method)(axis="z", coordinate=.15, n_freqs=1) is sim
        # Resolved expansion must not legalize a coordinate outside the declaration.
        with pytest.raises(ValueError, match="outside the z-domain.*0.2"):
            getattr(sim, method)(axis="z", coordinate=.21)
    for slab in slabs[stage:]:
        sim.add(slab, material="fr4")
    # Adding a second slab changes the cells, not the declared length (#1941).
    grid = sim._build_realized_grid()
    cells = grid.cells("z")[grid.pad_z_lo:grid.nz - grid.pad_z_hi - 1]
    assert sum(cells) == pytest.approx(.2, rel=1e-9)
    records = sim._dft_planes if method == "add_dft_plane_probe" else sim._flux_monitors
    assert len(records) == 1
    assert records[0].coordinate == .15


@pytest.mark.parametrize("method", ["add_dft_plane_probe", "add_flux_monitor"])
@pytest.mark.parametrize("axis,extent", [("x", .03), ("y", .04), ("z", .2)])
def test_plane_registration_preserves_declared_bounds(method, axis, extent):
    sim = Simulation(freq_max=10e9, domain=(.03, .04, .2), boundary="pec")
    register = getattr(sim, method)
    for coordinate in (0, extent):
        register(axis=axis, coordinate=coordinate)
    for coordinate in (-.001, extent + .001):
        with pytest.raises(ValueError, match=f"outside the {axis}-domain"):
            register(axis=axis, coordinate=coordinate)
    with pytest.raises(ValueError, match="axis must be"):
        register(axis="q", coordinate=0)

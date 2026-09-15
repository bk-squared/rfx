"""#931: builder guards read declarations; dispatch validates the resolved mesh."""

import pytest

from rfx import Box, Simulation


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("conductor", [None, "sheet", "volume"])
@pytest.mark.parametrize("preview", [False, True])
def test_auto_mesh_cannot_substitute_yee_for_declared_adi(entry, conductor, preview):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008),
                     boundary="pec", solver="adi")
    sim.add(Box((0, 0, 0), (.008, .008, .0016)), material="fr4")
    if conductor:
        sim.add(Box((.002, .002, .003),
                    (.006, .006, .003 if conductor == "sheet" else .004)),
                material="pec")
    if preview:
        assert sim._uses_nonuniform_mesh
        if conductor:
            assert "adi_interior_pec_unsupported" in {i.code for i in sim.preflight()}

    # Only the entry point is inside raises: a later uniform grid-builder error
    # must never masquerade as the solver refusing this model.
    with pytest.raises(NotImplementedError, match="solver='adi'.*resolved non-uniform"):
        getattr(sim, entry)(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("dx", [None, .002])
def test_supported_uniform_model_reaches_the_declared_adi_solver(entry, dx, monkeypatch):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008),
                     dx=dx, boundary="pec", solver="adi")
    sim.add_source((.004, .004, .004), "ez")
    sim.add_probe((.004, .004, .004), "ez")
    import rfx.adi

    original = rfx.adi.run_adi_3d
    calls = []

    def record_adi(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rfx.adi, "run_adi_3d", record_adi)
    getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert calls == [1], "A solver declaration must reach that solver or be refused"


@pytest.mark.parametrize("mode", ["run", "forward"])
def test_distributed_dispatch_cannot_substitute_yee_for_adi(mode):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008),
                     boundary="pec", solver="adi")
    sim.add(Box((0, 0, 0), (.008, .008, .0016)), material="fr4")
    with pytest.raises(NotImplementedError, match="solver='adi'.*resolved non-uniform"):
        sim._dispatch_plan(mode=mode, n_steps=1, num_periods=1,
                           distributed=True, devices=[object(), object()])


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
        extent = sim._domain[2]
        assert (extent < .15) if stage == 1 else (extent >= .2)

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
    assert sim._domain[2] > .2
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

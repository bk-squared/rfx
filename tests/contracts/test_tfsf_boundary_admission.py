"""Realized TF/SF transverse invariance, before stepping any fields."""
import gc

import numpy as np
import pytest

from rfx import Box, Simulation


@pytest.fixture(autouse=True)
def release_boards(monkeypatch):
    import jax

    def no_steps(*args, **kwargs):
        raise AssertionError("admission judge reached a field scan")

    monkeypatch.setattr(jax.lax, "scan", no_steps)
    yield
    gc.collect()


def plane(*, finite=False, boundary="cpml", closed_box=False, **mesh):
    sim = Simulation(freq_max=10e9, domain=(.02, .01, .01), dx=.001,
                     cpml_layers=2, boundary=boundary, **mesh)
    sim.add_material("slab", eps_r=2.5)
    sim.add(Box((.008, .003 if finite else -1., .003 if finite else -1.),
                (.012, .007 if finite else 1., .007 if finite else 1.)), material="slab")
    sim.add_tfsf_source(f0=5e9, margin=1, closed_box=closed_box)
    return sim


class Admitted(Exception):
    pass


def stop_at_runner(monkeypatch):
    from rfx.runners import uniform

    def stop(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr(uniform, "run_uniform", stop)


@pytest.mark.parametrize("skip", [False, True])
def test_finite_box_is_refused(skip):
    with pytest.raises(ValueError, match=r"TF/SF y_lo, y_hi: .*eps_r.*periodic.*closed_box=True"):
        plane(finite=True).run(n_steps=1, skip_preflight=skip)


def test_slotted_sheet_edge_mask_refusal_names_the_feature():
    sim = Simulation(freq_max=10e9, domain=(.03, .01, .001), dx=.001,
                     cpml_layers=2, mode="2d_tmz")
    sim.add_thin_conductor(Box((.015, -1., -1.), (.016, .004, 1.)), sigma_bulk=5.8e7)
    sim.add_tfsf_source(f0=5e9, margin=1)
    with pytest.raises(ValueError, match=r"y_lo, y_hi: .*pec_edge_masks"):
        sim._dispatch_plan(mode="run", n_steps=1, num_periods=1)


@pytest.mark.parametrize("skip", [False, True])
def test_invariant_slab_keeps_pads_and_reports_faces(skip, monkeypatch):
    sim = plane()
    grid = sim._build_grid()
    assert grid.shape == (25, 15, 15)
    assert tuple(f.realized for f in grid.boundary_depths) == (2,) * 6
    if not skip:
        findings = [item for item in sim.preflight(check_ntff=False)
                    if item.code == "tfsf_transverse_periodic"]
        assert len(findings) == 1
        assert all(face in str(findings[0]) for face in ("y_lo", "y_hi", "z_lo", "z_hi"))
        assert "solved periodic" in str(findings[0])
    stop_at_runner(monkeypatch)
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("boundary", [{"x": "cpml", "y": "pmc", "z": "pec"},
                                      {"x": "cpml", "y": "periodic", "z": "periodic"}])
def test_declared_admissible_faces_reach_dispatch(skip, boundary, monkeypatch):
    sim = plane(boundary=boundary)
    stop_at_runner(monkeypatch)
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
def test_inadmissible_electric_wall_is_refused(skip):
    sim = plane(boundary={"x": "cpml", "y": "pec", "z": "pec"})
    with pytest.raises(ValueError, match=r"y_lo, y_hi.*PERIODIC, PMC"):
        sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
def test_closed_box_keeps_finite_scatterer(skip, monkeypatch):
    stop_at_runner(monkeypatch)
    with pytest.raises(Admitted):
        plane(finite=True, closed_box=True).run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("axis", [1, 2])
def test_padding_and_all_material_components_are_checked(axis):
    from rfx.boundaries.tfsf import check_arrays
    sim = plane()
    grid = sim._build_grid()
    values = np.ones(grid.shape, dtype=np.float32)
    index = [slice(None)] * 3
    index[axis] = 0
    values[tuple(index)] = 2
    with pytest.raises(ValueError, match=f"{'xyz'[axis]}_lo"):
        check_arrays(sim._tfsf, grid, {"mu": values})


def test_old_s0_finite_box_is_refused():
    from tests.contracts.path_equivalence.builders import build, point
    sim = build(("_tfsf", "plane_wave"), "run_uniform")
    sim._geometry.clear()
    sim.add(Box(point(3.2, 2.1, 1.3), point(6.4, 4.3, 3.2)), material="block")
    with pytest.raises(ValueError, match=r"TF/SF y_lo, y_hi"):
        sim._dispatch_plan(mode="run", n_steps=8, num_periods=1)


@pytest.mark.parametrize("skip", [False, True])
def test_archived_37mm_array_is_refused(skip):
    # rfx-archive f8b3b79a, 20261005-tfsf-transverse-period/measure_period.py,
    # arm A, mesh 0. No archived time stepping is executed here.
    sim = Simulation(freq_max=10e9, domain=(.040, .020, .020), dx=.001,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("cube", eps_r=4, sigma=0)
    sim.add(Box((.014, .004, .004), (.026, .016, .016)), material="cube")
    sim.add_tfsf_source(f0=7e9, bandwidth=.8, margin=4, polarization="ez")
    grid = sim._build_grid()
    assert grid.shape == (57, 37, 37)
    assert grid.ny * grid.dx == .037
    with pytest.raises(ValueError, match=r"TF/SF y_lo, y_hi: .*closed_box=True"):
        sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("path", ["uniform", "nonuniform", "distributed_v2",
                                  "distributed_nu", "subgridded", "adi"])
def test_finite_scatterer_refuses_every_dispatch_path(skip, path, monkeypatch):
    import jax
    mesh = {"dz_profile": np.full(10, .001)} if path in ("nonuniform", "distributed_nu") else {}
    if path == "adi":
        mesh["solver"] = "adi"
        with pytest.raises(ValueError, match='solver="adi" refuses absorbing boundaries'):
            plane(finite=True, **mesh)
        return  # this declaration cannot reach ADI dispatch
    sim = plane(finite=True, **mesh)
    if path in ("nonuniform", "distributed_nu"):
        assert_graded_admitted(sim, skip, path == "distributed_nu", monkeypatch)
        return  # Addendum 5e: these paths install no transverse wrap.
    if path == "subgridded":
        sim.add_refinement(z_range=(0., .004), ratio=2, validation="research")
    kwargs = {"devices": jax.devices()} if path.startswith("distributed") else {}
    with pytest.raises(ValueError, match=r"TF/SF y_lo, y_hi"):
        sim.run(n_steps=1, skip_preflight=skip, **kwargs)


@pytest.mark.parametrize("broadcast", [False, True])
def test_low_level_explicit_wrap_keeps_caller_declaration(monkeypatch, broadcast):
    import jax.numpy as jnp
    import rfx.simulation as kernel
    from rfx.sources.tfsf import init_tfsf
    sim = plane()
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid)[0]
    shape = (1, grid.ny, 1) if broadcast else grid.shape
    materials = materials._replace(mu_r=jnp.ones(shape).at[:, 0, :].set(2.))
    source = init_tfsf(grid.nx, grid.dx, grid.dt, cpml_layers=2,
                       tfsf_margin=1, f0=5e9, bandwidth=.5)

    def stop(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr(kernel, "make_core_step", stop)
    with pytest.raises(Admitted):
        kernel.run(grid, materials, 1, tfsf=source, cpml_axes="x",
                   periodic=(False, True, True), pec_axes="")


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("transform", ["grad", "vmap", "jit"])
def test_transformed_finite_scatterer_is_refused(skip, transform):
    import jax
    import jax.numpy as jnp
    sim = plane(finite=True)

    def loss(value):
        return jnp.sum(sim.forward(n_steps=1, skip_preflight=skip).time_series) * value

    argument = jnp.array([1.]) if transform == "vmap" else jnp.array(1.)
    with pytest.raises(ValueError, match=r"TF/SF y_lo, y_hi"):
        getattr(jax, transform)(loss)(argument)


@pytest.mark.parametrize("axis, face", [(1, "y_lo, y_hi"), (2, "z_lo, z_hi")])
@pytest.mark.parametrize("broadcast", [False, True])
def test_pole_count_equal_to_grid_extent_does_not_shift_spatial_axes(axis, face, broadcast):
    from rfx.boundaries.tfsf import check_arrays

    sim = Simulation(freq_max=10e9, domain=(.01,) * 3, dx=.001, cpml_layers=2)
    sim.add_tfsf_source(f0=5e9, margin=1)
    grid = sim._build_grid()
    assert grid.shape == (15, 15, 15)
    # Fifteen poles, then x/y/z. Variation in propagation x is admissible.
    shape = (15, 1, 15 if axis == 1 else 1, 15 if axis == 2 else 1) if broadcast else (15,) * 4
    coefficients = np.ones(shape)
    if not broadcast:
        coefficients[:, 7, :, :] = 2.
    assert check_arrays(sim._tfsf, grid, {"debye": coefficients}) == ("y", "z")
    pad = [slice(None)] * 4
    pad[axis + 1] = 0
    coefficients[tuple(pad)] = 3.
    with pytest.raises(ValueError, match=face + ".*debye"):
        check_arrays(sim._tfsf, grid, {"debye": coefficients})


def test_traced_singleton_transverse_axis_is_constant():
    import jax
    import jax.numpy as jnp
    from rfx.boundaries.tfsf import invariant

    @jax.jit
    def check(value):
        return jnp.asarray(invariant(value, 1, (15, 15, 15))[0])

    assert bool(check(jnp.arange(15.).reshape(15, 1, 1)))


def assert_graded_admitted(sim, skip, distributed, monkeypatch):
    import jax
    import rfx.nonuniform as kernel

    assert not any(issue.code == "tfsf_transverse_periodic" for issue in sim.preflight(check_ntff=False))
    original = kernel._build_nu_scan

    def stop(grid, *args, **kwargs):
        # Literals measured on main 4acbd7925; the real factory includes
        # the low-level admission and NTFF enclosure check, but no step.
        assert grid.shape == (25, 15, 15)
        assert tuple(getattr(grid, f"pad_{a}_{s}") for a in "xyz" for s in ("lo", "hi")) == (2,) * 6
        original(grid, *args, **kwargs)
        raise Admitted

    monkeypatch.setattr(kernel, "_build_nu_scan", stop)
    kwargs = {}
    if distributed:
        assert len(jax.devices()) >= 2, "run this judge with two CPU devices"
        kwargs["devices"] = jax.devices()
        # Main's dispatch refusal is retained; this is not a boundary rewrite.
        with pytest.raises(ValueError, match=r"Distributed \+ non-uniform does not support TFSF"):
            sim.run(n_steps=1, skip_preflight=skip, **kwargs)
        return
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=skip, **kwargs)


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
def test_graded_plane_wave_keeps_absorbers(finite, skip, distributed, monkeypatch):
    sim = plane(finite=finite, dz_profile=np.full(10, .001))
    assert_graded_admitted(sim, skip, distributed, monkeypatch)


def oblique_plane(finite_axis, polarization):
    sim = Simulation(10e9, (.02, .01, .01), dx=.001, cpml_layers=2)
    lo, hi = [.008, -1., -1.], [.012, 1., 1.]
    lo["xyz".index(finite_axis)], hi["xyz".index(finite_axis)] = .003, .007
    sim.add_material("box", eps_r=2.5)
    sim.add(Box(tuple(lo), tuple(hi)), material="box")
    sim.add_tfsf_source(f0=5e9, margin=1, angle_deg=20, method="bloch", polarization=polarization)
    return sim


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("polarization,non_tilt", [("ez", "z"), ("ey", "y")])
def test_oblique_non_tilt_finite_box_refuses(skip, polarization, non_tilt):
    sim = oblique_plane(non_tilt, polarization)
    with pytest.raises(ValueError, match=rf"TF/SF {non_tilt}_lo, {non_tilt}_hi: .*periodic.*closed_box=True"):
        sim.run(n_steps=1, skip_preflight=skip)


def test_oblique_method_b_finding_names_only_the_wrapped_axis(monkeypatch):
    # Open oblique Method B absorbs on y and wraps z: the finding must not
    # call the y faces periodic (verification recheck R1).
    import rfx.simulation as kernel
    sim = Simulation(freq_max=10e9, domain=(.02, .01, .01), dx=.001, cpml_layers=2)
    sim.add_tfsf_source(f0=5e9, margin=1, angle_deg=20, method="methodB")
    findings = [issue for issue in sim.preflight(check_ntff=False) if issue.code == "tfsf_transverse_periodic"]
    assert len(findings) == 1
    text = str(findings[0])
    assert "z_lo, z_hi" in text and "y_lo" not in text and "not judged" not in text
    seen = []

    def stop(*args, **kwargs):
        seen.append(tuple(kwargs["periodic"]))
        raise Admitted

    monkeypatch.setattr(kernel, "_build_step_setup", stop)
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=True)
    assert seen == [(False, False, True)]


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("polarization,tilt", [("ez", "y"), ("ey", "z")])
def test_oblique_tilt_finite_box_reports_unjudged(skip, polarization, tilt, monkeypatch):
    sim = oblique_plane(tilt, polarization)
    findings = [issue for issue in sim.preflight(check_ntff=False) if issue.code == "tfsf_transverse_periodic"]
    assert len(findings) == 1
    text = str(findings[0])
    assert all(face in text for face in ("y_lo", "y_hi", "z_lo", "z_hi"))
    assert f"not judged for invariance along {tilt} at oblique incidence" in text
    stop_at_runner(monkeypatch)
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=skip)

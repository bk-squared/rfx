"""PR3a full-aperture guide declarations, checked before any time step."""
import copy
import gc
import inspect
from pathlib import Path
import warnings

import pytest

from rfx import Simulation
from rfx.boundaries.depths import Kind
from rfx.boundaries.features import WaveguideBoundaryWarning
from rfx.boundaries.pec import resolve_wall_faces


@pytest.fixture(autouse=True)
def release_boards(monkeypatch):
    import jax

    def no_steps(*args, **kwargs):
        raise AssertionError("admission judge reached a field scan")

    monkeypatch.setattr(jax.lax, "scan", no_steps)
    yield
    gc.collect()


def guide(boundary=None, *, partial=False, **mesh):
    kwargs = {} if boundary is None else {"boundary": boundary}
    sim = Simulation(freq_max=10e9, domain=(.04, .02, .01), dx=.001,
                     cpml_layers=4, **kwargs, **mesh)
    sim.add_waveguide_port(.01, f0=8e9, probe_offset=2, ref_offset=1,
                           y_range=(.003, .017) if partial else None)
    return sim


class Admitted(Exception):
    pass


def stop_after_admission(sim, monkeypatch):
    from rfx.runners import uniform

    def stop(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr(uniform, "run_uniform", stop)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("method", ["run", "forward"])
@pytest.mark.parametrize("boundary", ["cpml", {"x": "cpml", "y": "cpml", "z": "pec"}])
def test_explicit_absorber_is_refused_at_dispatch(skip, method, boundary):
    sim = guide(boundary)
    with pytest.raises(ValueError, match=r"full-aperture waveguide requires PEC on .*y_lo.*declare boundary="):
        getattr(sim, method)(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
def test_refusal_declaration_preserves_port_axis_depths(skip):
    import ast
    from rfx.boundaries.spec import Boundary, BoundarySpec

    sim = guide(BoundarySpec(x=Boundary("cpml", "cpml", 2, 3), y="cpml", z="cpml"))
    with pytest.raises(ValueError) as caught:
        sim.run(n_steps=1, skip_preflight=skip)
    declaration = ast.literal_eval(str(caught.value).split("declare boundary=", 1)[1].split(" (", 1)[0])
    assert declaration == {"x": {"lo": "cpml", "hi": "cpml", "lo_thickness": 2, "hi_thickness": 3},
                           "y": "pec", "z": "pec"}
    for model in (sim, guide(declaration)):
        grid = model._build_grid()
        assert grid.shape == (46, 21, 11)
        assert tuple(face.realized for face in grid.boundary_depths) == (2, 3, 0, 0, 0, 0)


@pytest.mark.parametrize("skip", [False, True])
def test_default_records_pec_and_warns_once_at_caller(skip, monkeypatch):
    sim = guide()
    grid = sim._build_grid()
    assert sim._boundary == "cpml" and sim._boundary_explicit is False
    assert grid.shape == (49, 21, 11)
    assert tuple(face.kind for face in grid.boundary_depths) == (
        Kind.ABSORBER, Kind.ABSORBER, Kind.PEC, Kind.PEC, Kind.PEC, Kind.PEC)
    assert tuple(face.realized for face in grid.boundary_depths) == (4, 4, 0, 0, 0, 0)
    stop_after_admission(sim, monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Admitted):
            line = inspect.currentframe().f_lineno + 1
            sim.run(n_steps=1, skip_preflight=skip)
        with pytest.raises(Admitted):
            sim.run(n_steps=1, skip_preflight=skip)
    defaults = [w for w in caught if issubclass(w.category, WaveguideBoundaryWarning)]
    assert len(defaults) == 1
    assert Path(defaults[0].filename) == Path(__file__)
    assert defaults[0].lineno == line
    assert all(face in str(defaults[0].message) for face in ("y_lo", "y_hi", "z_lo", "z_hi"))


def test_declared_pec_and_legacy_zero_depth_absorber_apply_same_walls():
    legacy = guide("cpml")._build_grid()
    declared = guide({"x": "cpml", "y": "pec", "z": "pec"})._build_grid()
    default = guide()._build_grid()
    expected = frozenset(("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")), frozenset()
    for grid in (legacy, declared, default):
        assert grid.shape == (49, 21, 11)
        assert tuple(face.realized for face in grid.boundary_depths) == (4, 4, 0, 0, 0, 0)
        assert resolve_wall_faces(grid, (False, False, False)) == expected


def test_kind_readers_keep_zero_pad_electric_backings():
    import jax
    import numpy as np
    from rfx.boundaries.axes import hi_face_is_wall
    from rfx.boundaries.cpml import init_cpml
    from rfx.boundaries.depths import distributed_electric_walls
    from rfx.boundaries.model import electric_faces, magnetic_faces
    old = guide("cpml")
    new = guide({"x": "cpml", "y": "pec", "z": "pec"})
    grids = (old._build_grid(), new._build_grid(), guide()._build_grid())
    coefficients = []
    for grid in grids:
        assert distributed_electric_walls(grid) == frozenset(("y_lo", "y_hi", "z_lo", "z_hi"))
        assert tuple(hi_face_is_wall(grid, axis) for axis in "xyz") == (False, True, True)
        assert grid.cpml_axes == "x"
        params = init_cpml(grid)[0]
        # Addendum 2 keeps legacy profiles on excluded axes; those profiles
        # are never applied. Compare every profile this guide actually uses.
        coefficients.append(jax.tree_util.tree_leaves((
            params.x_lo, params.x_hi, params.magnetic.x_lo, params.magnetic.x_hi)))
    for actual in coefficients[1:]:
        assert len(actual) == len(coefficients[0])
        assert all(np.array_equal(a, b) for a, b in zip(coefficients[0], actual))
    for sim in (old, new):
        assert electric_faces(sim.boundary_model()) == frozenset(("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"))
        assert magnetic_faces(sim.boundary_model()) == frozenset()


def test_smaller_aperture_keeps_legacy_rewrite():
    sim = guide("cpml", partial=True)
    assert sim._dispatch_plan(mode="run", n_steps=1, num_periods=1).lane == "run_uniform"
    assert tuple(face.kind for face in sim._build_grid().boundary_depths) == (Kind.ABSORBER,) * 6


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
def test_copy_preserves_omitted_boundary(clone):
    copied = clone(guide())
    assert copied._boundary_explicit is False
    assert copied._build_grid().boundary_depths[2].kind == Kind.PEC


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
def test_copy_has_its_own_default_warning(clone, monkeypatch):
    sim = guide()
    stop_after_admission(sim, monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Admitted):
            sim.run(n_steps=1, skip_preflight=True)
        copied = clone(sim)
        with pytest.raises(Admitted):
            copied.run(n_steps=1, skip_preflight=True)
        with pytest.raises(Admitted):
            sim.run(n_steps=1, skip_preflight=True)
    assert sum(issubclass(w.category, WaveguideBoundaryWarning) for w in caught) == 2


def test_design_document_exports_resolved_default_boundary():
    from rfx.interop import design_to_dict, simulation_from_design
    original = guide()
    rebuilt = simulation_from_design(design_to_dict(original))
    assert rebuilt._boundary_explicit is True
    assert rebuilt._boundary_spec.to_dict() == {
        "x": {"lo": "cpml", "hi": "cpml"},
        "y": {"lo": "pec", "hi": "pec"}, "z": {"lo": "pec", "hi": "pec"}}
    assert rebuilt._build_grid().boundary_depths[2].kind == Kind.PEC


def test_frozen_mesh_keeps_default_provenance():
    sim = guide()
    before = sim.freeze_mesh()
    assert sim._boundary_explicit is False
    assert sim._build_grid().shape == before.shape
    assert sim._build_grid().boundary_depths[2].kind == Kind.PEC


def test_terminal_planes_use_realized_pads():
    from rfx.boundaries.model import realize
    legacy = guide("cpml")
    declared = guide({"x": "cpml", "y": "pec", "z": "pec"})
    expected = (-.004, .044, 0., .02, 0., .01)
    for sim in (legacy, declared):
        planes = realize(sim.boundary_model(), sim._build_grid())
        actual = tuple(face.terminal_m if face.terminal_m is not None else face.plane_m
                       for face in planes.faces)
        assert actual == expected


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("path", ["uniform", "nonuniform", "distributed_v2",
                                  "distributed_nu", "subgridded", "adi"])
def test_explicit_guide_rule_on_every_dispatch_path(skip, path, monkeypatch):
    import jax
    import numpy as np
    mesh = {"dz_profile": np.full(10, .001)} if path in ("nonuniform", "distributed_nu") else {}
    if path == "adi":
        mesh["solver"] = "adi"
        with pytest.raises(ValueError, match='solver="adi" refuses absorbing boundaries'):
            guide("cpml", **mesh)
        return  # this declaration cannot reach ADI dispatch
    sim = guide("cpml", **mesh)
    if path == "subgridded":
        sim.add_refinement(z_range=(0., .004), ratio=2, validation="research")
    kwargs = {"devices": jax.devices()} if path.startswith("distributed") else {}
    if path in ("nonuniform", "distributed_nu"):
        import rfx.runners.nonuniform as kernel
        def stop(*args, **kwargs):
            raise Admitted
        monkeypatch.setattr(kernel, "run_nonuniform", stop)
        with pytest.raises(Admitted):
            sim.run(n_steps=1, skip_preflight=skip, **kwargs)
    else:
        with pytest.raises(ValueError, match=r"full-aperture waveguide requires PEC"):
            sim.run(n_steps=1, skip_preflight=skip, **kwargs)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("transform", ["grad", "vmap", "jit"])
def test_transformed_explicit_guide_is_refused(skip, transform):
    import jax
    import jax.numpy as jnp
    sim = guide("cpml")

    def loss(value):
        return jnp.sum(sim.forward(n_steps=1, skip_preflight=skip).time_series) * value

    argument = jnp.array([1.]) if transform == "vmap" else jnp.array(1.)
    with pytest.raises(ValueError, match="full-aperture waveguide requires PEC"):
        getattr(jax, transform)(loss)(argument)


@pytest.mark.parametrize("include_ports", [False, True])
def test_low_level_run_keeps_simulation_admission(include_ports):
    import jax.numpy as jnp
    from rfx import run
    sim = guide("cpml")
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid)[0]
    cfg = sim._build_waveguide_port_config(sim._waveguide_ports[0], grid,
                                          jnp.asarray([8e9]), 8)
    del sim  # the grid must retain its declaration's refusal
    with pytest.raises(ValueError, match="low-level dispatch: full-aperture waveguide requires PEC"):
        run(grid, materials, 8, waveguide_ports=[cfg] if include_ports else [])

"""One point-object rule; indices are hand-counted, never lookup-derived."""
import numpy as np
import pytest

from rfx import Simulation
from rfx._internal._grid_metric import field_index
from rfx.sources.sources import _wire_port_cells, wire_port_from_entry


def scene(lane):
    # Ten cells total 10 mm; first/last cells are 1 mm on every axis.
    # Five interior cells total 5 mm, even on the graded line.
    widths = np.array([1, 1, .8, 1.2, 1, 1, 1.2, .8, 1, 1]) * .001
    profiles = ({f"d{a}_profile": widths for a in "xyz"} if lane == "graded" else {})
    return Simulation(freq_max=25e9, domain=(.01,) * 3, dx=.001,
                      cpml_layers=8, **profiles)


def add(sim, kind, x, allow):
    kw = dict(allow_in_absorber=allow)
    position = (x, .005, .005)
    if kind == "probe":
        sim.add_probe(position, **kw)
    elif kind == "source":
        sim.add_source(position, amplitude_kind="field", **kw)
    else:
        sim.add_port(position, extent=.002 if kind == "wire" else None, **kw)


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("kind", ["probe", "source", "lumped", "wire"])
@pytest.mark.parametrize("outside", [False, True], ids=["pad1", "beyond_grid"])
@pytest.mark.parametrize("allow", [False, True])
def test_point_placement(lane, kind, outside, allow):
    sim = scene(lane)
    x = -.010 if outside else -.001
    add(sim, kind, x, allow)
    if outside or not allow:
        # Mesh resolution is deferred; preflight and skip-preflight solves
        # must give the same refusal. Ten cells deep is two beyond CPML8.
        depth = "10 cells" if outside else "1 cell"
        message = rf"{depth} into the x-lo CPML pad.*allow_in_absorber=True"
        with pytest.raises(ValueError, match=message):
            sim.preflight(strict=True)
        with pytest.raises(ValueError, match=message):
            sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)
        return

    report = sim.preflight()
    assert report.ok
    assert any(i.code == "absorber_overlap" and
               ("attenuated" in str(i) if kind == "probe" else "absorbed" in str(i))
               for i in report)
    grid = sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()
    # 8 pad cells - 1 mm / (1 mm per cell); y/z: 8 + five interior cells.
    expected = (8 - 1, 8 + 5, 8 + 5)
    entry = (sim._probes if kind == "probe" else sim._ports)[0]
    assert entry.allow_in_absorber is True
    assert field_index(grid, entry.position, entry.component,
                       allow_in_absorber=True) == expected
    if kind == "wire":
        assert _wire_port_cells(grid, wire_port_from_entry(entry)) == [
            expected, (7, 13, 14)]
    else:
        from rfx.model.graded_grid import position_to_index
        index = (position_to_index(grid, entry.position, allow_in_absorber=True)
                 if lane == "graded" else
                 grid.position_to_index(entry.position, allow_in_absorber=True))
        assert index == expected
    sim.run(n_steps=1, compute_s_params=False)


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_opt_in_is_per_object_and_wire_end_is_checked(lane):
    sim = scene(lane)
    add(sim, "probe", -.001, True)
    add(sim, "source", -.001, False)
    with pytest.raises(ValueError, match="1 cell into the x-lo CPML pad"):
        sim.preflight(strict=True)
    sim = scene(lane)
    sim.add_port((.005, .005, .009), extent=.010, allow_in_absorber=True)
    with pytest.raises(ValueError, match="outside the grid.*allow_in_absorber=True"):
        sim.preflight(strict=True)


@pytest.mark.parametrize("kind", ["probe", "source", "lumped", "wire"])
@pytest.mark.parametrize("lane", ["uniform", "graded", "distributed", "distributed_nu", "subgridded"])
@pytest.mark.parametrize("operation", ["run", "forward"])
def test_every_solve_lane_refuses_before_stepping(lane, operation, kind):
    if lane == "subgridded" and operation == "forward":
        pytest.skip("forward has no subgridded solve")
    sim = scene("graded" if lane in ("graded", "distributed_nu") else "uniform")
    add(sim, kind, -.001, False)
    if lane == "subgridded":
        sim.add_refinement((.002, .008), ratio=2, validation="off")
    kwargs = dict(n_steps=1, skip_preflight=True)
    if lane.startswith("distributed"):
        import jax
        devices = jax.devices("cpu")[:2]
        assert len(devices) == 2
        kwargs["devices"] = devices
        if operation == "forward":
            kwargs["distributed"] = True
    if operation == "run":
        kwargs["compute_s_params"] = False
    with pytest.raises(ValueError, match="1 cell into the x-lo CPML pad.*allow_in_absorber=True"):
        getattr(sim, operation)(**kwargs)


@pytest.mark.parametrize("kind", ["probe", "source", "lumped", "wire"])
def test_subgridded_full_cpml_box_cannot_reach_fine_placement(kind, monkeypatch):
    from rfx.runners import subgridded
    sim = scene("uniform")
    sim.add_refinement((.002, .008), ratio=2, validation="off")
    add(sim, kind, -.001, True)
    reached = []
    original = subgridded._run_subgridded_once

    def witness(*args, **kwargs):
        reached.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(subgridded, "_run_subgridded_once", witness)
    # Opt-in clears point admission, but this lane does not carry a full CPML box.
    # Its lane gate runs before the nested fine-grid position lookup.
    with pytest.raises(ValueError, match="a CPML absorber is not carried by the subgridded run"):
        sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)
    assert reached == [True]


@pytest.mark.parametrize("kind", ["probe", "source", "lumped", "wire"])
@pytest.mark.parametrize("allow", [False, True])
def test_subgridded_absorbing_lid_placement(kind, allow):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    boundary = BoundarySpec(x=Boundary(lo="pec", hi="pec"),
                            y=Boundary(lo="pec", hi="pec"),
                            z=Boundary(lo="pec", hi="cpml"))
    sim = Simulation(freq_max=25e9, domain=(.012, .012, .016), dx=.001,
                     cpml_layers=8, boundary=boundary)
    sim.add_refinement((0., .010), ratio=2, validation="off")
    position = (.005, .005, .017)  # One cell into the supported coarse z lid.
    kw = dict(component="ex", allow_in_absorber=allow)
    if kind == "probe":
        sim.add_probe(position, **kw)
    elif kind == "source":
        sim.add_source(position, amplitude_kind="field", **kw)
    else:
        sim.add_port(position, extent=.002 if kind == "wire" else None, **kw)
    if not allow:
        with pytest.raises(ValueError, match="1 cell into the z-hi CPML pad.*allow_in_absorber=True"):
            sim.preflight(strict=True)
    message = ("outside the fine grid shape" if allow else
               "1 cell into the z-hi CPML pad.*allow_in_absorber=True")
    with pytest.raises(ValueError, match=message):
        sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)


@pytest.mark.parametrize("lane", ["vmap_sweep", "material_fit"])
@pytest.mark.parametrize("allow", [False, True])
def test_material_paths_apply_pad_probe_rule(lane, allow, monkeypatch):
    from rfx import Box, simulation, vmap_sweep
    from rfx.differentiable_material_fit import differentiable_material_fit
    from rfx.vmap_sweep import vmap_material_sweep

    placed = []
    make_probe = simulation.make_probe

    def record_probe(*args, **kwargs):
        placed.append(True)
        return make_probe(*args, **kwargs)

    monkeypatch.setattr(simulation, "make_probe", record_probe)
    monkeypatch.setattr(vmap_sweep, "make_probe", record_probe)

    def fixture(eps_inf=2., debye_poles=(), lorentz_poles=()):
        sim = Simulation(freq_max=25e9, domain=(.004,) * 3, dx=.002,
                         cpml_layers=2)
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles,
                         lorentz_poles=lorentz_poles)
        sim.add(Box((0.,) * 3, (.004,) * 3), material="dut")
        if lane == "vmap_sweep":
            sim.add_source((.002,) * 3, amplitude_kind="field")
        else:
            sim.add_port((.002,) * 3)
        kw = {"allow_in_absorber": True} if allow else {}
        sim.add_probe((-.002, .002, .002), **kw)
        return sim

    def run():
        if lane == "vmap_sweep":
            result = vmap_material_sweep(fixture(), "dut.eps_r", np.array([2.]),
                                         n_steps=1)
            return result.time_series
        result = differentiable_material_fit(
            fixture, np.zeros((1, 1, 1), dtype=complex), np.array([10e9]),
            n_debye_poles=0, n_iterations=1, verbose=False)
        return result.final_s_params

    if not allow:
        with pytest.raises(ValueError, match=r"Probe\[0\].*1 cell into the x-lo CPML pad.*allow_in_absorber=True"):
            run()
        # A later kernel admission must not hide placement before refusal.
        assert placed == []
    else:
        output = run()
        assert output.shape == (1, 1, 1)
        assert np.isfinite(output).all()


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("boundary", ["cpml", "pec"])
def test_float32_domain_end_is_admitted(lane, boundary):
    from rfx.model.point_placement import admit_simulation_points

    end = .012  # float32 rounds this endpoint above the declared high face.
    widths = np.array([1., .8, 1.2, 1.] * 3) * .001
    profiles = {"dx_profile": widths} if lane == "graded" else {}
    sim = Simulation(freq_max=25e9, domain=(end,) * 3, dx=.001,
                     boundary=boundary, cpml_layers=8, **profiles)
    sim.add_probe((float(np.float32(end)), .005, .005))
    grid = sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()
    admit_simulation_points(sim, grid)
    assert sim._probes[0].allow_in_absorber is False
    assert field_index(grid, sim._probes[0].position, "ez")[0] == (
        20 if boundary == "cpml" else 12)
    if boundary == "cpml":
        sim.add_probe((end + .25 * .001, .005, .005))
        with pytest.raises(ValueError, match="0.25 cells into the x-hi CPML pad.*allow_in_absorber=True"):
            admit_simulation_points(sim, grid)


def test_ringdown_default_probe_keeps_port_opt_in():
    from rfx import GaussianPulse
    from rfx.ringdown import RingdownSpec

    sim = scene("uniform")
    sim.add_port((-.001, .005, .005), extent=.002, allow_in_absorber=True,
                 waveform=GaussianPulse(f0=25e9, bandwidth=1.))
    result = sim.forward(n_steps=128, skip_preflight=True,
                         port_s11_freqs=np.array([10e9]), ringdown=RingdownSpec())
    assert result.time_series.shape == (128, 1)
    assert sim._probes == []


def test_ringdown_run_keeps_port_opt_in():
    from rfx import GaussianPulse
    from rfx.ringdown import RingdownSpec

    sim = scene("uniform")
    sim.add_port((-.001, .005, .005), extent=.002, allow_in_absorber=True,
                 waveform=GaussianPulse(f0=25e9, bandwidth=1.))
    result = sim.run(n_steps=128, skip_preflight=True,
                     s_param_freqs=np.array([10e9]), ringdown=RingdownSpec())
    assert result.time_series.shape == (128, 0)
    assert sim._probes == []


@pytest.mark.parametrize("operation, columns", [("forward", 1), ("run", 0)])
def test_ringdown_interior_port_with_back_leg_in_pad(operation, columns):
    from rfx import GaussianPulse
    from rfx.ringdown import RingdownSpec

    sim = scene("uniform")
    # x cell 8 is the first interior cell; the Hy back leg is cell 7 (-1 mm).
    sim.add_port((0., .005, .005), extent=.002,
                 waveform=GaussianPulse(f0=25e9, bandwidth=1.))
    assert sim._ports[0].allow_in_absorber is False
    assert field_index(sim._build_grid(), (0., .005, .005), "ez") == (8, 13, 13)
    frequency_arg = "port_s11_freqs" if operation == "forward" else "s_param_freqs"
    result = getattr(sim, operation)(n_steps=128, skip_preflight=True,
                                    ringdown=RingdownSpec(),
                                    **{frequency_arg: np.array([10e9])})
    assert result.time_series.shape == (128, columns)
    assert sim._probes == []


@pytest.mark.parametrize("allow", [False, True])
def test_vector_probe_in_pad(allow):
    sim = scene("uniform")
    sim.add_source((.005,) * 3, amplitude_kind="field")
    kw = {"allow_in_absorber": True} if allow else {}
    sim.add_vector_probe((-.001, .005, .005), **kw)
    if not allow:
        with pytest.raises(ValueError, match="1 cell into the x-lo CPML pad.*allow_in_absorber=True"):
            sim.forward(n_steps=1, skip_preflight=True)
    else:
        result = sim.forward(n_steps=1, skip_preflight=True)
        assert result.time_series.shape == (1, 6)

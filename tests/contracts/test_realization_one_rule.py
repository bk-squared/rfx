"""Independent lattice expectations and single-assembly metadata witnesses."""
from dataclasses import replace

import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation, Sphere


FAMILIES = ("pec_box", "dielectric_box", "sphere", "sheet", "dc_film", "wire", "wall", "probe")
LANES = ("uniform_run", "uniform_forward", "graded_run", "graded_forward")
D = .001
LO = np.array((2.23, 2.37, 2.19)) * D
HI = np.array((6.61, 6.73, 6.57)) * D


def scene(family, lane):
    profiles = {"dz_profile": np.array((1., .8, 1.2, .9, 1.1, .7, 1.3, 1., 1., 1.)) * D} if lane.startswith("graded") else {}
    sim = Simulation(freq_max=1e9, domain=(10.04*D, 10*D, 10*D), dx=D,
                     boundary="pec", snap="declared", **profiles)
    sim.add_material("glass", eps_r=2.)
    if family in ("pec_box", "dielectric_box"):
        sim.add(Box(tuple(LO), tuple(HI)), material="pec" if family == "pec_box" else "glass")
    elif family == "sphere":
        sim.add(Sphere((4.23*D, 4.61*D, 4.17*D), 2.13*D), material="glass")
    elif family in ("sheet", "dc_film"):
        shape = Box((LO[0], LO[1], 4.27*D), (HI[0], HI[1], 4.27*D))
        if family == "sheet":
            sim.add(shape, material="pec")
        else:
            sim.add_thin_conductor(shape, sigma_bulk=100., thickness=35e-6)
    elif family == "wire":
        sim.add(PolylineWire(((2.23*D, 4.27*D, 4.19*D),
                              (6.61*D, 4.27*D, 4.19*D)), radius=0.), material="pec")
    elif family == "probe":
        sim.add_probe((4.23*D, 4.61*D, 4.17*D), component="ez")
    return sim


def expected(family, record):
    """Write the family rules out; use only public node/cell arrays as inputs."""
    nodes = record.nodes
    cells = record.cell_sizes
    centres = [n[:-1] + c/2 for n, c in zip(nodes, cells)]
    if family == "wall":
        # Nearest whole cell, exact half goes up; graded z uses its profile.
        return np.array([(0., nodes[a][int(np.floor(length / D + .5))])
                         for a, length in enumerate((10.04*D, 10*D, 10*D))])
    if family == "probe":
        points = (4.23*D, 4.61*D, 4.17*D)
        return np.array([(n[np.argmin(abs(n - p))],) * 2 for n, p in zip(nodes, points)])
    if family == "wire":
        points = ((2.23*D, 6.61*D), (4.27*D, 4.27*D), (4.19*D, 4.19*D))
        indices = [(int(np.argmin(abs(n-lo))), int(np.argmin(abs(n-hi))))
                   for n, (lo, hi) in zip(nodes, points)]
        # A filament record bounds the occupied edge-start nodes.
        return np.array([(n[lo], n[hi - (a == 0)]) for a, (n, (lo, hi))
                         in enumerate(zip(nodes, indices))])
    if family == "sphere":
        x, y, z = np.meshgrid(*centres, indexing="ij")
        mask = ((x-4.23*D)**2 + (y-4.61*D)**2 + (z-4.17*D)**2 <= (2.13*D)**2)
        occupied = np.where(mask)
        return np.array([(n[i.min()], n[i.max()+1]) for n, i in zip(nodes, occupied)])
    result = []
    for a, n in enumerate(nodes):
        if family in ("sheet", "dc_film") and a == 2:
            k = np.argmin(abs(n - 4.27*D))
            # Both surface families occupy one node plane in the normal direction.
            result.append((n[k], n[k]))
            continue
        sample = n[:-1] if family in ("dielectric_box", "sheet") else centres[a]
        mask = ((sample >= LO[a]) & (sample <= HI[a] if family == "sheet" else sample < HI[a]))
        indices = np.flatnonzero(mask)
        lo, hi = indices[0], indices[-1]
        if family == "sheet":
            result.append((n[lo] - .35*(n[lo]-n[lo-1]), n[hi] + .35*(n[hi+1]-n[hi])))
        else:
            result.append((n[lo], n[hi+1]))
    return np.array(result)


def label(family):
    return {"wall": "domain", "probe": "probe[0]", "dc_film": "thin_conductor[0]"}.get(family, "geometry[0]")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lane", LANES)
def test_spans_and_assembly(family, lane, monkeypatch):
    import rfx.model.realization_table as tables
    sim = scene(family, lane)
    builds, roots, reports = [], [], []
    original_build = tables.realization_table

    def build(prepared):
        builds.append(prepared)
        return original_build(prepared)

    from rfx.stepping import prepared as stepping
    original_step = stepping.advance

    def step(root, runner, *args, **kwargs):
        roots.append(root)
        assert root.realization_table.prepared() is root
        assert builds[0].conductors is root
        return original_step(root, runner, *args, **kwargs)

    original_gate = sim._run_preflight_gate
    def gate(report, **kwargs):
        assert report.realization_table is not None
        reports.append(report)
        return original_gate(report, **kwargs)

    monkeypatch.setattr(tables, "realization_table", build)
    monkeypatch.setattr(stepping, "advance", step)
    monkeypatch.setattr(sim, "_run_preflight_gate", gate)
    if lane.endswith("forward"):
        result = sim.forward(n_steps=1, skip_preflight=False)
    else:
        result = sim.run(n_steps=1, skip_preflight=False, compute_s_params=False)
    record = result.realized_geometry
    assert len(builds) == len(roots) == 1
    assert builds[0].conductors is roots[0]
    table = record.realization_table
    assert roots[0].realization_table is table
    assert reports[0].realization_table is table
    row = table.row(label(family))
    hand = expected(family, record)
    np.testing.assert_allclose([a.bounds_m for a in row.axes], hand, rtol=0, atol=1e-12)
    np.testing.assert_allclose([a.realized_span_m for a in row.axes], hand[:, 1]-hand[:, 0], rtol=0, atol=1e-12)
    for a, axis in enumerate(row.axes):
        cell = independent_cell(family, record, hand, a)
        assert axis.miss_cells == pytest.approx((axis.realized_span_m-axis.declared_span_m)/cell)
        assert axis.centroid_shift_cells == pytest.approx((sum(axis.bounds_m)-sum(axis.declared_bounds_m))/(2*cell))
    if family == "dc_film":
        area = float(np.prod(hand[:2, 1]-hand[:2, 0]))
        declared = float(np.prod((HI-LO)[:2]))
        assert row.realized_area_m2 == pytest.approx(area)
        assert row.declared_area_m2 == pytest.approx(declared)
        assert row.area_equivalent_length_m == pytest.approx(np.sqrt(area))
        assert row.area_miss_percent == pytest.approx(100*(np.sqrt(area/declared)-1))
    # Diagnostic views share one prepared table and keep their historical types.
    preview = sim.realized_geometry()
    assert preview.realization_table is table
    assert len(builds) == 1
    assert sim.realized_geometry().realization_table is preview.realization_table
    report = sim.fidelity_report(print_report=False)
    assert report[0]["realization_table"] is preview.realization_table
    np.testing.assert_allclose([a.bounds_m for a in preview.realization_table.row(label(family)).axes], hand, rtol=0, atol=1e-12)


def test_declared_span_mutation_has_independent_oracle():
    sim = scene("dielectric_box", "uniform_run")
    record = sim.realized_geometry()
    row = record.realization_table.row("geometry[0]")
    corrupt = replace(row, axes=tuple(replace(a, realized_span_m=a.declared_span_m) for a in row.axes))
    hand = expected("dielectric_box", record)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose([a.realized_span_m for a in corrupt.axes], hand[:, 1]-hand[:, 0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("lane", LANES)
def test_prior_preview_preserves_cache_and_reads_kernel_table(lane):
    sim = scene("dielectric_box", lane)
    before = sim.realized_geometry()
    cached = sim._realized_geometry_record
    result = (sim.forward(n_steps=1, skip_preflight=True) if lane.endswith("forward") else
              sim.run(n_steps=1, skip_preflight=True, compute_s_params=False))
    assert sim._realized_geometry_record is cached
    table = result.realized_geometry.realization_table
    assert table is not before.realization_table
    assert sim.realized_geometry().realization_table is table
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is table


@pytest.mark.parametrize("nominal", [False, True], ids=("unresolved", "nominal"))
def test_traced_mesh_reference(nominal, monkeypatch):
    import jax
    import jax.numpy as jnp
    observed = []
    stepped = watch_steps(monkeypatch)
    profile = np.full(10, D)

    def evaluate(delta):
        traced = jnp.asarray(profile).at[3].add(delta).at[4].add(-delta)
        sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D,
            boundary="pec", dz_profile=traced, dt=.2*D/299792458., dt_min_cell=.5*D,
            nominal_profiles={"z": profile} if nominal else None)
        sim.add_material("glass", eps_r=2.)
        sim.add(Box(tuple(LO), tuple(HI)), material="glass")
        result = sim.forward(n_steps=1, skip_preflight=True)
        observed.append(result.realized_geometry)
        return jnp.sum(result.time_series)

    jax.grad(evaluate)(0.)
    record = observed[0]
    assert len(stepped) == 1 and stepped[0] is record.realization_table
    row = record.realization_table.row("geometry[0]")
    assert "mesh.z" in row.traced_inputs
    if nominal:
        assert row.status == "evaluated: nominal_profiles"
        concrete = scene("dielectric_box", "uniform_run").realized_geometry()
        np.testing.assert_allclose([a.bounds_m for a in row.axes], expected("dielectric_box", concrete), atol=1e-12, rtol=0)
    else:
        assert row.status == "not evaluated: traced"
        assert row.axes == ()


@pytest.mark.parametrize("entry", ("run", "forward"))
def test_adi_boundary(entry):
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec", solver="adi")
    kwargs = {"compute_s_params": False} if entry == "run" else {}
    result = getattr(sim, entry)(n_steps=1, skip_preflight=True, **kwargs)
    assert result.realized_geometry.realization_table.lane == ("run_adi" if entry == "run" else "fwd_adi")
    assert result.realized_geometry.dt == result.dt


def test_nominal_reference_does_not_add_a_live_refusal():
    import jax
    import jax.numpy as jnp
    observed = []
    profile = np.full(10, D)

    def evaluate(delta):
        widths = jnp.asarray(profile).at[3].add(delta).at[4].add(-delta)
        sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec",
                         dz_profile=widths, nominal_profiles={"z": profile},
                         dt=.2*D/299792458., dt_min_cell=.5*D)
        sim.add(Box((2*D, 2*D, 2.2*D), (6*D, 6*D, 2.8*D)), material="pec")
        result = sim.forward(n_steps=1, skip_preflight=True)
        observed.append(result.realized_geometry.realization_table.row("geometry[0]"))
        return jnp.sum(result.time_series)

    assert jax.grad(evaluate)(0.) == 0.
    assert observed[0].status == "not evaluated: traced"


@pytest.mark.parametrize("family", FAMILIES)
def test_distributed_run(family, monkeypatch):
    import jax
    import rfx.model.conductors as conductors
    import rfx.model.realization_table as tables
    if len(jax.devices("cpu")) < 2:
        pytest.skip("requires two CPU devices")
    sim = scene(family, "uniform_run")
    observed, builds = [], []
    original_build = tables.realization_table

    def build(prepared):
        builds.append(id(prepared.conductors))
        return original_build(prepared)

    from rfx.stepping import prepared as stepping
    original_step = stepping.advance
    def step(root, runner, *args, **kwargs):
        table = root.realization_table
        assert table.prepared() is root
        assert builds == [id(root)]
        observed.append(table)
        return original_step(root, runner, *args, **kwargs)

    monkeypatch.setattr(tables, "realization_table", build)
    monkeypatch.setattr(stepping, "advance", step)
    result = sim.run(n_steps=1, devices=jax.devices("cpu")[:2],
                     skip_preflight=True, compute_s_params=False)
    record = result.realized_geometry
    assert observed == [record.realization_table]
    row = record.realization_table.row(label(family))
    np.testing.assert_allclose([a.bounds_m for a in row.axes], expected(family, record), atol=1e-12, rtol=0)
    # Dense kernel products have been released; reporting restores their captured
    # diagnostic arrays and must never assemble another set of conductors.
    def reassembled(*args, **kwargs):
        pytest.fail("distributed reporter assembled a second product")

    monkeypatch.setattr(conductors, "realized_conductors", reassembled)
    preview = sim.realized_geometry()
    assert preview.realization_table is record.realization_table
    assert not preview.materials.eps_r.flags.writeable
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is record.realization_table
    assert len(builds) == 1


@pytest.mark.parametrize("lane", ("uniform_forward", "graded_run", "graded_forward"))
def test_other_distributed_reporters_share_the_kernel_table(lane, monkeypatch):
    import jax
    import rfx.model.conductors as conductors
    if len(jax.devices("cpu")) < 2:
        pytest.skip("requires two CPU devices")
    sim = scene("dielectric_box", lane)
    kwargs = dict(n_steps=1, skip_preflight=True, devices=jax.devices("cpu")[:2])
    result = (sim.forward(distributed=True, **kwargs) if lane.endswith("forward") else
              sim.run(compute_s_params=False, **kwargs))
    record = result.realized_geometry
    assert "distributed" in record.lane
    np.testing.assert_allclose([a.bounds_m for a in record.realization_table.row("geometry[0]").axes],
                               expected("dielectric_box", record), atol=1e-12, rtol=0)

    def reassembled(*args, **kwargs):
        pytest.fail("distributed reporter assembled a second product")

    monkeypatch.setattr(conductors, "realized_conductors", reassembled)
    assert sim.realized_geometry().realization_table is record.realization_table
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is record.realization_table


@pytest.mark.parametrize("smoothing", (True, "full_tensor", "kottke_pec"))
@pytest.mark.parametrize("lane", ("uniform_run", "graded_run"))
def test_filled_volume(smoothing, lane):
    if smoothing == "kottke_pec" and lane == "graded_run":
        pytest.skip("the graded lane refuses this smoothing mode")
    sim = scene("dielectric_box", lane)
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False,
                     subpixel_smoothing=smoothing)
    row = result.realized_geometry.realization_table.row("geometry[0]")
    assert row.filled_volume_m3 is not None
    assert row.filled_volume_m3 == pytest.approx(float(np.prod(HI-LO)), rel=.15)
    assert "smoothed fill" in row.model


def test_conformal_open_area():
    sim = scene("wall", "uniform_run")
    sim.add(Sphere((4.23*D, 4.61*D, 4.17*D), 2.13*D), material="pec")
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False, conformal_pec=True)
    row = result.realized_geometry.realization_table.row("geometry[0]")
    assert row.open_area_m2 is not None
    assert all(0 < area < 11 * (10*D)**2 for area in row.open_area_m2)


def test_refined_region_uses_fine_mask():
    sim = scene("dielectric_box", "uniform_run")
    sim.add_refinement(z_range=(2*D, 8*D), ratio=2, validation="research")
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    row = result.realized_geometry.realization_table.row("geometry[0]")
    assert len(row.regions) == 1
    name, axes = row.regions[0]
    assert name == "fine"
    nodes = [np.arange(0, 10.5*D, D/2), np.arange(0, 10.5*D, D/2),
             np.arange(2*D, 8.5*D, D/2)]
    # The fine rasterizer samples dielectric Boxes at cell centres.
    for a, (axis, n) in enumerate(zip(axes, nodes)):
        sample = n[:-1] + D/4
        indices = np.flatnonzero((sample >= LO[a]) & (sample < HI[a]))
        np.testing.assert_allclose(axis.bounds_m, (n[indices[0]], n[indices[-1]+1]), atol=1e-12, rtol=0)


def test_traced_shape_is_metadata_without_new_verdict():
    import jax
    import jax.numpy as jnp
    observed = []

    def evaluate(radius):
        sim = scene("wall", "uniform_run")
        sim.add(Sphere((4.23*D, 4.61*D, 4.17*D), radius), material="glass")
        result = sim.forward(n_steps=1, skip_preflight=True)
        observed.append(result.realized_geometry.realization_table.row("geometry[0]"))
        return jnp.sum(result.time_series)

    assert jax.jit(evaluate)(2.13*D) == 0.
    assert observed[0].status == "not evaluated: traced"
    assert "geometry[0].radius" in observed[0].traced_inputs


def test_graded_jit_result_keeps_host_metadata():
    import jax
    sim = scene("dielectric_box", "graded_forward")
    def evaluate():
        result = sim.forward(n_steps=1, skip_preflight=True)
        return result.time_series, result.realized_geometry
    _, record = jax.jit(evaluate)()
    assert record.realization_table.row("geometry[0]").axes


@pytest.mark.parametrize("graded", (False, True))
@pytest.mark.parametrize("smoothed", (False, True))
def test_material_continuation_reads_actual_pad_copy(graded, smoothed):
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="cpml", cpml_layers=2,
                     **({"dz_profile": np.full(10, D)} if graded else {}))
    sim.add_material("glass", eps_r=2.)
    sim.add(Box((0., 2*D, 2*D), (10*D, 6*D, 6*D)), material="glass")
    if smoothed:
        sim.run(n_steps=1, skip_preflight=True, compute_s_params=False, subpixel_smoothing=True)
    record = sim.realized_geometry()
    assert record.realization_table.row("geometry[0]").continued_faces == ("x-lo", "x-hi")
    # The high boundary node and its pad carry the same non-vacuum column.
    assert np.asarray(record.materials.eps_r)[-1, 5, 5] == 2.


def test_disjoint_research_lane_reports_its_vacuum_model(monkeypatch):
    stepped = watch_steps(monkeypatch)
    sim = scene("dielectric_box", "uniform_run")
    sim.add_refinement(z_range=(2*D, 8*D), ratio=2, validation="research",
                       topology="stage2_disjoint_3d")
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    assert len(stepped) == 1 and stepped[0] is result.realized_geometry.realization_table
    row = result.realized_geometry.realization_table.row("geometry[0]")
    assert row.status == "not represented: disjoint vacuum stepper has no material operands"
    assert row.axes == ()
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is result.realized_geometry.realization_table


@pytest.mark.parametrize("frozen", (False, True))
def test_inplane_profile_preserves_drawn_wall_extent(frozen):
    sim = Simulation(freq_max=1e9, domain=(12*D, 10*D, 10*D), dx=D,
                     dx_profile=np.full(10, D), boundary="pec")
    if frozen:
        sim.freeze_mesh()
    record = sim.realized_geometry()
    axis = record.realization_table.row("domain").axes[0]
    assert axis.declared_span_m == 12*D
    np.testing.assert_allclose(axis.realized_span_m, 10*D)
    np.testing.assert_allclose(axis.miss_cells, -2.)
    # The legacy record retains the existing admitted-domain interpretation.
    assert record.domain[0].declared_length_m == sim._domain[0]
    result = sim.forward(n_steps=1, skip_preflight=True)
    solved = result.realized_geometry.realization_table.row("domain").axes[0]
    assert solved.declared_span_m == 12*D
    np.testing.assert_allclose(solved.realized_span_m, 10*D)


def independent_cell(family, record, hand, axis):
    nodes, cells = record.nodes[axis], record.cell_sizes[axis]
    if family == "wall":
        return float(np.mean(cells[:10]))
    if family == "probe":
        return float(cells[np.argmin(abs(nodes - hand[axis, 0]))])
    if family in ("sheet", "dc_film") and axis == 2:
        return float(cells[np.argmin(abs(nodes - 4.27*D))])
    if family == "sheet":
        indices = np.flatnonzero((nodes[:-1] >= LO[axis]) & (nodes[:-1] <= HI[axis]))
    else:
        lo, hi = hand[axis]
        left = np.argmin(abs(nodes-lo))
        right = np.argmin(abs(nodes-hi)) - (family != "wire")
        indices = np.arange(left, right+1)
    return float(np.mean(cells[indices]))


def watch_steps(monkeypatch, *, direct_replays=False):
    """Observe ownership at take(), then the product and operands at stepping."""
    from rfx.model.conductors import PreparedSolve
    from rfx.stepping import prepared
    import weakref
    taken, stepped = [], []
    original_take, original_step = PreparedSolve.take, prepared.advance

    def take(self, **kwargs):
        root = original_take(self, **kwargs)
        taken.append(weakref.ref(root))
        return root

    def advance(root, runner, *args, **kwargs):
        if not (direct_replays and stepped):
            assert taken and taken[-1]() is root, "stepping must receive the take() product"
        table = root.realization_table
        assert table.prepared() is root, "table was built from a separate assembly"
        stepped.append(table)
        layout = kwargs.get("_operand_layout", "grid")

        def operands(*operands, **options):
            if layout in ("grid", "graded_slab"):
                assert operands[0] is root.grid
                assert operands[1] is root.materials
                if layout == "graded_slab":
                    assert operands[2] is root.pec_edges
                elif "pec_edge_masks" in options:
                    assert options["pec_edge_masks"] is root.pec_edges
            elif layout == "slab":
                assert options["sharded_materials"] is root.materials
                assert options["sharded_pec_mask"] is root.pec_edges
            elif layout == "adi3d":
                assert operands[6] is root.materials.eps_r
                assert operands[7] is root.materials.sigma
                assert options["pec_edge_masks"] is root.pec_edges
            elif layout == "disjoint":
                assert operands[1] is root.kernel_context[0]
            return runner(*operands, **options)

        return original_step(root, operands, *args, **kwargs)

    monkeypatch.setattr(PreparedSolve, "take", take)
    monkeypatch.setattr(prepared, "advance", advance)
    return stepped


@pytest.mark.parametrize("lane", (
    "uniform_run", "uniform_forward", "graded_run", "graded_forward",
    "distributed_v2_run", "distributed_v2_forward", "distributed_nu_run",
    "distributed_nu_forward", "adi_run", "adi_forward", "subgridded"))
def test_prepared_at_step(lane, monkeypatch):
    import jax
    stepped = watch_steps(monkeypatch)
    sim = scene("probe", "graded_run" if "graded" in lane or "_nu_" in lane else "uniform_run")
    if lane.startswith("adi"):
        sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec", solver="adi")
        sim.add_probe((4.23*D, 4.61*D, 4.17*D), component="ez")
    if lane == "subgridded":
        sim.add_refinement(z_range=(2*D, 8*D), ratio=2, validation="research")
    gates = []
    original_gate = sim._run_preflight_gate

    def gate(report, **kwargs):
        assert report.realization_table is not None
        gates.append(report.realization_table)
        return original_gate(report, **kwargs)

    monkeypatch.setattr(sim, "_run_preflight_gate", gate)
    kwargs = dict(n_steps=1, skip_preflight=False)
    if lane.startswith("distributed"):
        devices = jax.devices("cpu")[:2]
        assert len(devices) == 2
        kwargs["devices"] = devices
        if lane.endswith("forward"):
            kwargs["distributed"] = True
    result = (sim.forward(**kwargs) if lane.endswith("forward") else
              sim.run(compute_s_params=False, **kwargs))
    table = result.realized_geometry.realization_table
    assert len(stepped) == 1 and stepped[0] is table
    assert gates and all(t is table for t in gates)
    assert sim.realized_geometry().realization_table is table
    if lane == "subgridded":
        hand = np.array(((4*D,)*2, (4.5*D,)*2, (4*D,)*2))
    else:
        hand = expected("probe", result.realized_geometry)
    np.testing.assert_allclose([a.bounds_m for a in table.row("probe[0]").axes], hand, atol=1e-12, rtol=0)
    # Neither sim nor the result pins the solve's dense product.
    import gc
    gc.collect()
    assert table.prepared() is None


def test_two_port_primary_table(monkeypatch):
    from rfx import GaussianPulse
    import rfx.realized_geometry as records
    calls = []
    original = records.record_from_conductors

    def build(*args, **kwargs):
        assert kwargs.get("compact", True)
        calls.append(kwargs["lane"])
        return original(*args, **kwargs)

    monkeypatch.setattr(records, "record_from_conductors", build)
    stepped = watch_steps(monkeypatch, direct_replays=True)
    sim = Simulation(freq_max=10e9, domain=(12*D, 10*D, 10*D), dx=D, boundary="pec")
    sim.add_material("glass", eps_r=2.)
    sim.add(Box((2.3*D, 2.4*D, 2.2*D), (8.6*D, 6.7*D, 4.6*D)), material="glass")
    for x in (3.5*D, 8.5*D):
        sim.add_port((x, 5.5*D, 6.5*D), "ez", impedance=50., waveform=GaussianPulse(f0=5e9, bandwidth=.9))
    result = sim.run(n_steps=5, compute_s_params=True, s_param_freqs=np.array([4e9, 5e9]), skip_preflight=True)
    table = result.realized_geometry.realization_table
    assert table is stepped[0]
    assert len(stepped) == 3
    assert len(calls) == len({id(t) for t in stepped})  # One compact record per prepared product.
    assert sim.realized_geometry().realization_table is table
    assert table.lane == "run_uniform"
    assert {"port[0]", "port[1]"} <= {row.label for row in table.rows}
    import gc
    gc.collect()
    assert all(t.prepared() is None for t in stepped)


def test_fidelity_comparisons_read_only_table_rows(monkeypatch):
    sim = scene("dielectric_box", "uniform_run")
    sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    record = sim.realized_geometry()
    expected_axes = sim.fidelity_report(print_report=False)[1]["axes"]
    corrupted = replace(record, entities=tuple(replace(e, axes=tuple(
        replace(a, face_residual_m=(999., 999.), declared_bounds_m=(999., 999.),
                bounds_m=(999., 999.), extent_m=999.) for a in e.axes),
        declared_bounds_m=((999.,)*3, (999.,)*3)) for e in record.entities))
    object.__setattr__(corrupted, "_report_grid", record._report_grid)
    monkeypatch.setattr(sim, "realized_geometry", lambda: corrupted)
    assert sim.fidelity_report(print_report=False)[1]["axes"] == expected_axes


def test_conformal_owner_is_declaration_identity():
    from rfx.model.realization_integrals import begin, quantities
    sim = scene("dielectric_box", "uniform_run")
    conductor = Sphere((5*D,)*3, 2*D)
    sim.add(conductor, material="pec")
    grid = sim._build_grid()
    begin(sim, grid, conformal_shapes=(conductor,))
    assert quantities(grid)["owners"][id(conductor)] == 1

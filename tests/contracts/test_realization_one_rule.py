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


@pytest.mark.parametrize("snap", ("strict", "declared"))
def test_skip_preflight_keeps_narrowed_microstrip_verdict(snap):
    import warnings

    # The PEC bottom wall is the ground; the strip rests on the substrate.
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec", snap=snap)
    sim.add_material("substrate", eps_r=2.)
    sim.add(Box((0., 0., 0.), (10*D, 10*D, 4*D)), material="substrate")
    sim.add(Box((2*D, 3.9*D, 4*D), (8*D, 6.1*D, 5*D)), material="pec")
    grid = sim._build_grid()
    cells = np.asarray(grid.cells(1), dtype=float)
    nodes = np.r_[0., np.cumsum(cells)]
    centres = (nodes[:-1] + nodes[1:])/2
    occupied = np.flatnonzero((centres >= 3.9*D) & (centres < 6.1*D))
    # Occupied centres 4.5, 5.5 mm span faces 4, 6 mm: 2.2 -> 2 mm.
    width = nodes[occupied[-1]+1] - nodes[occupied[0]]
    drawn = (6.1-3.9)*D
    np.testing.assert_allclose(nodes[occupied[[0, -1]]] / D, [4., 5.])
    assert width == pytest.approx(2*D)
    assert (width-drawn)/drawn*100 == pytest.approx(-100/11)
    assert (width-drawn)/cells[occupied[0]] == pytest.approx(-.2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if snap == "strict":
            with pytest.raises(ValueError, match="y extent miss"):
                sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
        else:
            result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
            row = result.realized_geometry.realization_table.row("geometry[1]")
            assert row.axes[1].realized_span_m == pytest.approx(width)
    findings = [w.message for w in caught
                if getattr(w.message, "code", None) == "realization"]
    assert len(findings) == (snap == "declared")
    if findings:
        assert "y extent miss -0.2 cells (-9.09090909091 %)" in str(findings[0])


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("operation", ("pickle", "deepcopy"))
def test_result_metadata_roundtrip(lane, operation):
    import copy
    import pickle
    import weakref

    def restore(value):
        return (pickle.loads(pickle.dumps(value)) if operation == "pickle"
                else copy.deepcopy(value))

    sim = scene("dielectric_box", lane)
    result = (sim.forward(n_steps=1, skip_preflight=True) if lane.endswith("forward") else
              sim.run(n_steps=1, skip_preflight=True, compute_s_params=False))
    original = result.realized_geometry.realization_table
    restored = restore(result)
    table = restored.realized_geometry.realization_table
    assert type(restored) is type(result)
    assert table is not original
    assert table.rows == original.rows
    assert table.lane == original.lane
    assert isinstance(table.prepared, weakref.ReferenceType)
    assert table.prepared() is None

    # Walk tuple payloads AND attachment dictionaries/slots, without following
    # weakrefs into the solve. Exercise each reachable realization object too.
    pending, seen, walked = [result], set(), set()
    while pending:
        value = pending.pop()
        if id(value) in seen or isinstance(value, (type, weakref.ReferenceType)):
            continue
        seen.add(id(value))
        module = type(value).__module__
        if module.startswith(("rfx.model.realization", "rfx.realized_geometry")):
            assert type(restore(value)) is type(value)
            walked.add(type(value).__name__)
        if isinstance(value, dict):
            pending.extend((*value.keys(), *value.values()))
        elif isinstance(value, (tuple, list, set)):
            pending.extend(value)
        if hasattr(value, "__dict__") and module.startswith("rfx"):
            pending.extend(vars(value).values())
        for cls in type(value).__mro__:
            slots = getattr(cls, "__slots__", ())
            slots = (slots,) if isinstance(slots, str) else slots
            pending.extend(getattr(value, name) for name in slots
                           if name not in ("__dict__", "__weakref__") and hasattr(value, name))
    assert {"RealizedGeometry", "RealizationTable", "RealizationRow", "RealizationAxis"} <= walked


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
    nodes = tuple(np.asarray(axis) for axis in record.nodes)
    cells = tuple(np.asarray(axis) for axis in record.cell_sizes)
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
        # Sum every occupied edge, from its start node to its end node.
        return np.array([(n[lo], n[lo] + sum(n[k+1]-n[k] for k in range(lo, hi)))
                         for n, (lo, hi) in zip(nodes, indices)])
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
        sample = n[:-1] if family == "sheet" else centres[a]
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
        assert len(builds) == 1
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
    # Run diagnostics share the solved table; forward leaves pre-run reporting.
    count = len(builds)
    preview = sim.realized_geometry()
    assert len(builds) - count <= 1
    if lane.endswith("forward"):
        assert preview.realization_table is not table
        assert preview.realization_table.rows == table.rows
    else:
        assert preview.realization_table is table
    count = len(builds)
    assert sim.realized_geometry().realization_table is preview.realization_table
    assert len(builds) - count <= 1
    count = len(builds)
    report = sim.fidelity_report(print_report=False)
    assert len(builds) - count <= 1
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
def test_prior_preview_reads_kernel_table(lane):
    sim = scene("dielectric_box", lane)
    before = sim.realized_geometry()
    result = (sim.forward(n_steps=1, skip_preflight=True) if lane.endswith("forward") else
              sim.run(n_steps=1, skip_preflight=True, compute_s_params=False))
    table = result.realized_geometry.realization_table
    assert table is not before.realization_table
    preview = sim.realized_geometry().realization_table
    if lane.endswith("forward"):
        assert preview is not table
        assert preview.rows == table.rows
    else:
        assert preview is table
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is preview


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


@pytest.mark.parametrize("nominal", [False, True], ids=("unresolved", "nominal"))
def test_forward_traced_dz_with_live_dt(nominal, monkeypatch):
    import jax
    import jax.numpy as jnp
    profile = np.array((1., .8, 1.2, .9, 1.1, .7, 1.3, 1., 1., 1.)) * D

    def build(widths, reference):
        sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D,
                         boundary="pec", dz_profile=widths,
                         nominal_profiles={"z": profile} if reference else None)
        sim.add_material("glass", eps_r=2.)
        sim.add(Box(tuple(LO), tuple(HI)), material="glass")
        sim.add_source((4*D, 4*D, 4*D), component="ez")
        sim.add_probe((4*D, 4*D, 5*D), component="ez")
        return sim

    concrete = build(profile, False).forward(n_steps=4, skip_preflight=True).realized_geometry
    observed = []
    stepped = watch_steps(monkeypatch)

    def evaluate(delta):
        widths = jnp.asarray(profile).at[3].add(delta).at[4].add(-delta)
        result = build(widths, nominal).forward(n_steps=4, skip_preflight=True)
        observed.append(result.realized_geometry)
        return jnp.sum(result.time_series**2)

    assert np.isfinite(jax.grad(evaluate)(0.))
    record = observed[0]
    assert len(stepped) == 1 and stepped[0] is record.realization_table
    rows = record.realization_table.rows
    assert all("mesh.z" in row.traced_inputs for row in rows)
    if nominal:
        assert all(row.status == "evaluated: nominal_profiles" for row in rows)
        assert tuple(replace(row, status="evaluated", traced_inputs=()) for row in rows) == concrete.realization_table.rows
        assert record.dt == concrete.dt
    else:
        assert all(row.status == "not evaluated: traced" and row.axes == () for row in rows)
        assert record.dt is None


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
    assert len(builds) == 1
    record = result.realized_geometry
    assert observed == [record.realization_table]
    row = record.realization_table.row(label(family))
    np.testing.assert_allclose([a.bounds_m for a in row.axes], expected(family, record), atol=1e-12, rtol=0)
    # A diagnostic call prepares current declarations and checks its rows.
    count = len(builds)
    preview = sim.realized_geometry()
    assert len(builds) - count <= 1
    assert preview.realization_table is record.realization_table
    assert not preview.materials.eps_r.flags.writeable
    count = len(builds)
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is record.realization_table
    assert len(builds) - count <= 1


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

    preview = sim.realized_geometry().realization_table
    if lane.endswith("forward"):
        assert preview is not record.realization_table
        assert preview.rows == record.realization_table.rows
    else:
        assert preview is record.realization_table
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is preview


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


@pytest.mark.parametrize("conformal", (True, "cut_cell_floor"))
def test_conformal_open_area(conformal):
    from rfx.grid import Grid
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec",
                     **({"dt": .5*Grid.courant_dt(D)} if conformal == "cut_cell_floor" else {}))
    sim.add(Sphere((4.23*D, 4.61*D, 4.17*D), 2.13*D), material="pec")
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False, conformal_pec=conformal)
    row = result.realized_geometry.realization_table.row("geometry[0]")
    assert row.open_area_m2 is not None
    assert all(0 < area < 11 * (10*D)**2 for area in row.open_area_m2)
    assert sim.realized_geometry().realization_table is result.realized_geometry.realization_table
    assert sim.fidelity_report(print_report=False)[0]["realization_table"] is result.realized_geometry.realization_table


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
                     snap="declared",
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
    nodes, cells = np.asarray(record.nodes[axis]), np.asarray(record.cell_sizes[axis])
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
    "distributed_nu_forward", "subgridded"))
def test_prepared_at_step(lane, monkeypatch):
    import jax
    stepped = watch_steps(monkeypatch)
    sim = scene("probe", "graded_run" if "graded" in lane or "_nu_" in lane else "uniform_run")
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
    preview = sim.realized_geometry().realization_table
    if lane.endswith("forward"):
        assert preview is not table
        assert preview.rows == table.rows
    else:
        assert preview is table
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


def reachable_arrays(value, seen=None, *, skip_fields=()):
    """Walk owned fields, including metadata slots, without following weakrefs."""
    import types
    import weakref
    import jax
    seen = set() if seen is None else seen
    if id(value) in seen or isinstance(value, (weakref.ReferenceType, type,
            types.FunctionType, types.ModuleType, str, bytes)):
        return
    seen.add(id(value))
    if isinstance(value, (np.ndarray, jax.Array)):
        yield value
        return
    if isinstance(value, dict):
        children = (*value.keys(), *value.values())
    elif isinstance(value, (tuple, list, set)):
        children = value
    else:
        children = [v for k, v in vars(value).items() if k not in skip_fields] if hasattr(value, '__dict__') else []
        for cls in type(value).__mro__:
            slots = getattr(cls, '__slots__', ())
            slots = (slots,) if isinstance(slots, str) else slots
            children.extend(getattr(value, name) for name in slots
                            if name not in ('__dict__', '__weakref__', *skip_fields) and hasattr(value, name))
    for child in children:
        yield from reachable_arrays(child, seen, skip_fields=skip_fields)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('preview_first', (False, True))
def test_no_dense_array_retained_after_solve(lane, preview_first):
    """S1 adds no retained dense arrays beyond main's existing preview cache.

    Main keeps a cache created by a reporter before the solve; its identity is
    covered by the pinned-sheet contract. Exempt only the exact arrays already
    owned by that preview's historical fields. S1 attachments are excluded from
    this baseline inventory, but all fields (including those attachments) are
    traversed after solving, so no S1-added array is exempted.
    """
    import gc
    from rfx.model.realization_reference import _kernel_views
    sim = scene('dielectric_box', lane)
    preview_arrays = ()
    if preview_first:
        sim.realized_geometry()
        s1_fields = ('realization_table', '_realization_tables', '_prepared_owner',
                     '_realization_record', 'refinements', 'kernel_context',
                     '_report_grid', '_quantity_grid', '_prepared_conductors',
                     '_realization_origin')
        preview_arrays = tuple(reachable_arrays((sim._realized_geometry_record,
            sim._pf_campaign_ctx, getattr(sim, '_coax_geometry', None)), skip_fields=s1_fields))
    preview_ids = {id(array) for array in preview_arrays}
    kwargs = dict(n_steps=1, skip_preflight=True)
    result = (sim.forward(**kwargs) if lane.endswith('forward') else
              sim.run(compute_s_params=False, **kwargs))
    axis_limit = max(len(axis) - 1 for axis in result.realized_geometry.nodes)
    gc.collect()
    assert result.realized_geometry.realization_table.prepared() is None
    # Realization metadata owns this contract; measurement fields keep their arrays.
    roots = {
        'result.realized_geometry': result.realized_geometry,
        'sim realization store': _kernel_views.get(sim),
        'sim._realized_geometry_record': getattr(sim, '_realized_geometry_record', None),
        'sim._pf_campaign_ctx': getattr(sim, '_pf_campaign_ctx', None),
        'sim._coax_geometry': getattr(sim, '_coax_geometry', None),
    }
    for name, root in roots.items():
        dense_shapes = [array.shape for array in reachable_arrays(root)
                        if array.size > axis_limit and id(array) not in preview_ids]
        assert not dense_shapes, (name, dense_shapes)
    assert not list(reachable_arrays(_kernel_views.get(sim)))


def test_reporter_refuses_rows_that_changed_at_call_time(monkeypatch):
    from rfx._declarations import declaration_key
    import rfx.model.realization_table as tables
    sim = scene('dielectric_box', 'uniform_run')
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    key = declaration_key(sim)
    original_axis = tables._axis

    def changed_rule(axis, declared, realized, cell):
        return original_axis(axis, declared, (realized[0], realized[1] + cell), cell)

    # Change only the reporter-time rule, keeping the solved declarations intact.
    monkeypatch.setattr(tables, '_axis', changed_rule)
    assert declaration_key(sim) == key
    with pytest.raises(ValueError, match='Realization rows differ from the solved table'):
        sim.realized_geometry()
    with pytest.raises(ValueError, match='Realization rows differ from the solved table'):
        sim.fidelity_report(print_report=False)
    assert len(result.realized_geometry.realization_table.rows) == 2


@pytest.mark.parametrize('solved', (False, True))
@pytest.mark.parametrize('edit', ('add', 'replace'))
@pytest.mark.parametrize('reporter', ('geometry', 'fidelity'))
def test_reporter_reads_box_changed_between_calls(solved, edit, reporter):
    from rfx.model.realization_reference import _kernel_views
    sim = scene('dielectric_box', 'uniform_run')
    result = None
    if solved:
        result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    first = sim.realized_geometry()
    # Keep the previous preparation alive while replacing the declaration.
    entry = sim._geometry[0]
    box = Box(tuple(LO), (8.61*D, HI[1], HI[2]))
    if edit == 'add':
        sim.add(box, material='glass')
    else:
        sim._geometry[0] = replace(entry, shape=box)
    index = 1 if edit == 'add' else 0
    if reporter == 'geometry':
        second = sim.realized_geometry()
        table = second.realization_table
        # The dielectric cell-centre sampler owns cells 2..8.
        assert table.row(f'geometry[{index}]').axes[0].bounds_m == pytest.approx((2*D, 9*D))
    else:
        report = sim.fidelity_report(print_report=False)
        table = report[0]['realization_table']
        assert report[index + 1]['axes'][0]['realized_um'] == pytest.approx((2000., 9000.))
    assert table.rows != first.realization_table.rows
    assert len(table.rows) == (3 if edit == 'add' else 2)
    assert sim not in _kernel_views
    if result is not None:
        assert result.realized_geometry.realization_table is first.realization_table
        assert len(result.realized_geometry.realization_table.rows) == 2


@pytest.mark.parametrize('lane', ('uniform_forward', 'graded_forward'))
def test_forward_leaves_sim_report_on_pre_run_path(lane):
    from rfx.model.realization_reference import _kernel_views
    sim = scene('dielectric_box', lane)
    before = sim.realized_geometry()
    result = sim.forward(n_steps=1, skip_preflight=True)
    assert sim not in _kernel_views
    after = sim.realized_geometry()
    assert after.realization_table.rows == before.realization_table.rows
    assert after.realization_table is not result.realized_geometry.realization_table
    assert sim.fidelity_report(print_report=False)[0]['realization_table'] is after.realization_table


VERDICT_FAMILIES = ("pec_box", "pec_sphere", "dielectric_box", "magnetic_box",
                    "lossy_box", "debye_box", "f0_film", "sheet", "dc_film", "wire", "port", "wall", "profile_domain")


@pytest.mark.parametrize("boundary", ("periodic", "pec", "cpml"))
@pytest.mark.parametrize("material", ("glass", "pec"))
@pytest.mark.parametrize("short", (False, True))
def test_interior_face_comparison(boundary, material, short):
    sim = Simulation(domain=(10*D,)*3, dx=D, freq_max=1e9,
                     boundary={"x": boundary, "y": "pec", "z": "pec"}, cpml_layers=2, snap="strict")
    sim.add_material("glass", eps_r=2.)
    # The occupied x cells are 2..9. A lower face at 2.087042532 mm
    # still selects cell 2; its interior span is 8 mm / 1.011.
    end = 12*D if boundary == "periodic" else 11*D
    start = ((end - 10*D/1.011 if short else 2*D) if boundary == "periodic"
             else 10*D - 8*D/1.011 if short else 2*D)
    sim.add(Box((start, 2*D, 2*D), (end, 8*D, 8*D)), material=material)
    record = sim.realized_geometry()
    row = record.realization_table.row("geometry[0]")
    domain = record.domain[0]
    nodes = np.asarray(record.nodes[0])
    wall = nodes[domain.padding[0] + domain.n_cells]
    if boundary == "periodic":
        period = wall - nodes[0]
        length = end-start
        first = start % period
        intervals = ((0., period),) if length >= period else (
            ((first, first+length),) if first+length <= period else
            ((0., first+length-period), (first, period)))
        declared = sum(hi-lo for lo, hi in intervals)
        # The gap between 2 mm and 2.108803165 mm contains no cell centre.
        physical = sum(np.diff(nodes[:domain.n_cells+1]))
        assert row.axes[0].declared_intervals_m == intervals
        assert row.axes[0].realized_span_m == pytest.approx(physical)
    else:
        physical = wall - nodes[domain.padding[0] + 2]
        declared = wall - start
        assert physical == pytest.approx(8*D)
    assert 100*(physical/declared-1) == pytest.approx(1.1 if short else 0.)
    if boundary != "periodic":
        assert row.axes[0].declared_bounds_m == pytest.approx((start, wall))
        assert row.clipped_faces == (("x+", end, wall),)
    report = sim.preflight()
    hits = [f for f in report if f.code == "realization" and f.loc == "geometry[0]"]
    assert len(hits) == int(short)
    if short:
        assert hits[0].severity == ("warning" if material == "glass" else "error")
        assert ("periodic union" if boundary == "periodic" else "clipped faces x+") in str(hits[0])


@pytest.mark.parametrize("material", ("glass", "pec"))
def test_periodic_split_union_preserves_gap(material):
    sim = Simulation(domain=(10*D,)*3, dx=D, freq_max=1e9,
                     boundary={"x": "periodic", "y": "pec", "z": "pec"})
    sim.add_material("glass", eps_r=2.)
    sim.add(Box((2*D, 2*D, 2*D), (11*D, 8*D, 8*D)), material=material)
    record = sim.realized_geometry()
    n = np.asarray(record.nodes[0])
    period = n[record.domain[0].n_cells] - n[0]
    intervals = ((n[0], 11*D-period), (2*D, period))
    # Projected occupied cells: 0 and 2..9, leaving cell 1 empty.
    occupied_length = (n[1]-n[0]) + (n[10]-n[2])
    row = record.realization_table.row("geometry[0]").axes[0]
    assert row.declared_intervals_m == intervals
    assert row.realized_intervals_m == ((n[0], n[1]), (n[2], n[10]))
    assert row.realized_span_m == pytest.approx(occupied_length)
    assert row.declared_span_m == pytest.approx(sum(hi-lo for lo, hi in intervals))
    assert not [f for f in sim.preflight() if f.code == "realization"]


def test_periodic_wire_union_includes_terminal_edge():
    sim = Simulation(domain=(10*D,)*3, dx=D, freq_max=1e9,
                     boundary={"x": "periodic", "y": "pec", "z": "pec"})
    sim.add(PolylineWire(((9*D, 3*D, 2*D), (10*D, 3*D, 2*D)), radius=0.), material="pec")
    record = sim.realized_geometry()
    n = np.asarray(record.nodes[0])
    row = record.realization_table.row("geometry[0]").axes[0]
    assert row.realized_intervals_m == ((n[9], n[10]),)
    assert row.realized_span_m == pytest.approx(n[10]-n[9])
    assert not [f for f in sim.preflight() if f.code == "realization"]


@pytest.mark.parametrize("graded", (False, True))
def test_probe_point_shift_is_reported_without_finding(graded):
    sim = Simulation(domain=(10*D,)*3, dx=D, freq_max=1e9, boundary="pec",
                     **({"dz_profile": np.full(10, D)} if graded else {}))
    sim.add_probe((5.4*D, 5*D, 5*D), component="ez")
    record = sim.realized_geometry()
    nodes, cells = np.asarray(record.nodes[0]), np.asarray(record.cell_sizes[0])
    row = record.realization_table.row("probe[0]")
    shift = (nodes[5] - 5.4*D)/cells[5]
    assert shift == pytest.approx(-.4)
    assert row.axes[0].centroid_shift_cells == pytest.approx(shift)
    assert not [f for f in sim.preflight() if f.code == "realization"]


@pytest.mark.parametrize("graded", (False, True))
def test_point_source_retains_node_placement(graded):
    from rfx import GaussianPulse
    sim = Simulation(domain=(10*D,)*3, dx=D, freq_max=1e9, boundary="pec",
                     **({"dz_profile": np.full(10, D)} if graded else {}))
    sim.add_source((5*D,)*3, "ez", waveform=GaussianPulse(f0=5e8, bandwidth=.9),
                   amplitude_kind="current")
    record = sim.realized_geometry()
    row = record.realization_table.row("port[0]")
    for a, axis in enumerate(row.axes):
        node = float(record.nodes[a][5])
        assert axis.bounds_m == pytest.approx((node, node))
        assert axis.realized_span_m == 0.
    assert not [f for f in sim.preflight() if f.code == "realization"]


def test_one_cell_graded_port_driven_span():
    from rfx import GaussianPulse
    sim = Simulation(domain=(16*D, 12*D, 12*D), dx=D, freq_max=1e9,
                     dz_profile=np.full(12, D), boundary="pec")
    sim.add(Box((4*D, 5*D, 3*D), (12*D, 7*D, 5*D)), material="pec")
    sim.add(Box((4*D, 5*D, 7*D), (12*D, 7*D, 9*D)), material="pec")
    sim.add_port((5*D, 6*D, 5*D), "ez", extent=D, impedance=50.,
                 waveform=GaussianPulse(f0=5e8, bandwidth=.9))
    record = sim.realized_geometry()
    nodes, cells = np.asarray(record.nodes[2]), np.asarray(record.cell_sizes[2])
    assert nodes[5:7] == pytest.approx((.005, .006))
    assert cells[5] == pytest.approx(.001)
    row = record.realization_table.row("port[0]")
    assert row.axes[2].bounds_m == pytest.approx((nodes[5], nodes[6]))
    assert row.axes[2].realized_span_m/cells[5] == pytest.approx(1.)


@pytest.mark.parametrize("lane", LANES + ("distributed_v2_run", "distributed_v2_forward",
                                         "distributed_nu_run", "distributed_nu_forward", "subgridded"))
@pytest.mark.parametrize("family", VERDICT_FAMILIES)
@pytest.mark.parametrize("miss", (.009, .011))
@pytest.mark.parametrize("snap", ("strict", "declared"))
def test_verdict_boundary(family, lane, miss, snap):
    """Contract (iii): actual scenes straddling the common geometric bar.

    The oracle below uses only node/cell arrays and the family's sampling rule;
    neither a realization helper nor the judge supplies an expected dimension.
    """
    import warnings

    if lane == "subgridded" and family in ("sheet", "f0_film", "dc_film", "wire", "debye_box", "profile_domain"):
        pytest.skip("this declaration is refused by subgrid lane admission")
    if lane.startswith("distributed") and family == "f0_film":
        pytest.skip("surface-impedance sheets are refused by distributed lane admission")
    if "distributed_nu" in lane and family == "dc_film":
        pytest.skip("DC films are refused by distributed graded lane admission")
    graded = lane.startswith("graded") or "_nu_" in lane or family == "profile_domain"
    profile = np.array((.8, 1.2) * 5) * D
    domain = [120*D if family in ("wire", "port") else 10*D, 10*D, 10*D]
    kwargs = {"dz_profile": profile} if graded else {}
    if family in ("wall", "profile_domain"):
        domain[0] = 10*D/(1+miss)
    if family == "profile_domain":
        kwargs["dx_profile"] = np.full(10, D)
    # Build the oracle grid under declared acceptance before exercising strict.
    sim = Simulation(freq_max=1e9, domain=tuple(domain), dx=D, boundary="pec",
                     snap="declared", **kwargs)
    oracle = Simulation(freq_max=1e9, domain=(120*D if family in ("wire", "port") else 10*D, 10*D, 10*D),
                        dx=D, boundary="pec", snap="declared", **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = oracle._build_nonuniform_grid() if graded else oracle._build_grid()
    cells = [np.asarray(grid.cells(a), dtype=float) for a in range(3)]
    nodes = [np.r_[0., np.cumsum(d)] - np.sum(d[:getattr(grid, f"pad_{'xyz'[a]}_lo")])
             for a, d in enumerate(cells)]
    centre = np.array((5., 5., 4.8 if graded else 5.))*D
    length = 4*D/(1+miss)
    low, high = centre-length/2, centre+length/2
    if family in ("pec_box", "dielectric_box", "magnetic_box", "lossy_box", "debye_box"):
        if family == "pec_box":
            material = "pec"
        else:
            from rfx.materials.debye import DebyePole
            sim.add_material("test", eps_r=2., **(
                {"mu_r": 2.} if family == "magnetic_box" else
                {"sigma": .01} if family == "lossy_box" else
                {"debye_poles": [DebyePole(1., 1e-11)]} if family == "debye_box" else {}))
            material = "test"
        sim.add(Box(tuple(low), tuple(high)), material=material)
    elif family == "pec_sphere":
        sim.add(Sphere(tuple(centre), length/2), material="pec")
    elif family in ("sheet", "f0_film"):
        length = 4.7*D/(1+miss)
        low, high = centre-length/2, centre+length/2
        low[2] = high[2] = centre[2]
        if family == "sheet":
            sim.add(Box(tuple(low), tuple(high)), material="pec")
        else:
            sim.add_thin_conductor(Box(tuple(low), tuple(high)), sigma_bulk=1e4,
                                   thickness=35e-6, surface_impedance_f0=1e9)
    elif family == "dc_film":
        low[2] = high[2] = centre[2]
        sim.add_thin_conductor(Box(tuple(low), tuple(high)), sigma_bulk=100., thickness=35e-6)
    elif family in ("wire", "port"):
        if family == "wire":
            # Edges 2..41 span 40 cells, from the first start to the last end.
            # (stepped_span - drawn) / drawn = -miss => drawn = stepped_span / (1-miss).
            stepped_span = sum(nodes[0][k+1]-nodes[0][k] for k in range(2, 42))
            length = stepped_span/(1-miss)
        else:
            length = 40*D/(1+miss)
        low = np.array(((1.66 if family == "wire" else 2.)*D, centre[1], centre[2]))
        high = low + (length, 0., 0.)
        if family == "wire":
            sim.add(PolylineWire((tuple(low), tuple(high)), radius=0.), material="pec")
        else:
            from rfx import GaussianPulse
            sim.add_port(tuple(low), "ex", extent=length, impedance=50.,
                         waveform=GaussianPulse(f0=5e8, bandwidth=.9))

    if family in ("wall", "profile_domain"):
        physical = nodes[0][10]-nodes[0][0]
        expected_miss = physical/domain[0]-1
    elif family in ("wire", "port"):
        a = np.argmin(abs(nodes[0]-low[0]))
        b = np.argmin(abs(nodes[0]-high[0]))
        physical = nodes[0][b]-nodes[0][a]
        expected_miss = physical/length-1
    else:
        spans = []
        for a in range(2):
            sample = nodes[a][:-1] if family in ("sheet", "f0_film") else nodes[a][:-1]+cells[a]/2
            occupied = np.flatnonzero((sample >= low[a]) & (sample < high[a]))
            first, last = occupied[0], occupied[-1]
            physical = (nodes[a][last]-nodes[a][first]+.35*(cells[a][first-1]+cells[a][last])
                        if family in ("sheet", "f0_film") else nodes[a][last+1]-nodes[a][first])
            spans.append(physical)
        expected_miss = (np.sqrt(np.prod(spans))/length-1
                         if family in ("sheet", "f0_film", "dc_film") else spans[0]/length-1)
    assert abs(expected_miss) == pytest.approx(miss, abs=2e-6)
    sim._snap = snap
    if lane == "subgridded":
        sim.add_refinement(z_range=(2*D, 8*D), ratio=2, validation="research")
    solve_options = dict(n_steps=1, skip_preflight=True)
    if lane.startswith("distributed"):
        import jax
        solve_options["devices"] = jax.devices("cpu")[:2]
        assert len(solve_options["devices"]) == 2
        if lane.endswith("forward"):
            solve_options["distributed"] = True

    def solve():
        if lane.endswith("forward"):
            return sim.forward(**solve_options)
        return sim.run(compute_s_params=False, **solve_options)

    dielectric = family in ("dielectric_box", "magnetic_box", "lossy_box", "debye_box")
    refuses = miss > .01 and snap == "strict" and not dielectric and family != "port"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if refuses:
            with pytest.raises(ValueError):
                solve()
        else:
            solve()
    codes = {"realization", "wall_axis_off_node", "dc_film_area", "sheet_effective_size", "sheet_cavity_spacing"}
    messages = [w.message for w in caught if getattr(w.message, "code", None) in codes]
    if miss < .01 or family == "port":
        assert not messages
    elif not refuses:
        assert messages
        if dielectric:
            assert len(messages) == 1
            text = str(messages[0])
            for axis in "xy":
                assert f"{axis}: realized" in text
            assert text.count("cells (") >= 4
            assert text.count(" m)") >= 4
            assert "centroid shift" in text


def test_refined_volume_has_one_realization_warning():
    import warnings
    sim = scene("dielectric_box", "uniform_run")
    sim.add_refinement(z_range=(2*D, 8*D), ratio=2, validation="research")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    messages = [w.message for w in caught if getattr(w.message, "code", None) == "realization"]
    assert len(messages) == 1
    findings = [f for f in result.realized_geometry.realization_table.findings
                if f.code == "realization" and f.loc == "geometry[0]"]
    assert len(findings) == 1
    assert "geometry[0]/fine" in findings[0]


def _radius_at_miss(measure, target, low, high):
    """Choose a drawing using the independent quadrature written in this test."""
    radii = np.linspace(low, high, 161)
    values = [measure(r)[0]-target for r in radii]
    bracket = next(((a, b) for a, b, fa, fb in zip(radii[:-1], radii[1:], values[:-1], values[1:])
                    if fa*fb <= 0), None)
    assert bracket is not None, (target, min(values), max(values))
    low, high = bracket
    for _ in range(40):
        mid = (low+high)/2
        if (measure(low)[0]-target)*(measure(mid)[0]-target) <= 0:
            high = mid
        else:
            low = mid
    return (low+high)/2


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("smoothing", (True, "full_tensor"))
@pytest.mark.parametrize("miss", (.009, .011))
@pytest.mark.parametrize("snap", ("strict", "declared"))
def test_integral_verdict_boundary_smoothed(lane, smoothing, miss, snap):
    import warnings
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec", snap=snap,
                     **({"dz_profile": np.full(10, D)} if lane.startswith("graded") else {}))
    grid = sim._build_nonuniform_grid() if lane.startswith("graded") else sim._build_grid()
    cells = [np.asarray(grid.cells(a)) for a in range(3)]
    nodes = [np.r_[0., np.cumsum(c[:-1])] for c in cells]
    centre = np.array((5.23, 5.17, 4.91))*D
    samples = []
    for component in ((3,) if smoothing == "full_tensor" else range(3)):
        xyz = np.meshgrid(*[n + (cells[a]/2 if a == component else 0.)
                            for a, n in enumerate(nodes)], indexing="ij")
        samples.append(np.sqrt(sum((x-centre[a])**2 for a, x in enumerate(xyz))))

    def measure(radius):
        # All occupied quadrature cells are interior; their primal/dual widths
        # are D in this grid, for both the node tensor and E-edge samples.
        filled = np.mean([np.sum(np.clip(.5-(distance-radius)/D, 0., 1.))*D**3
                          for distance in samples])
        declared = 4*np.pi*radius**3/3
        return (filled/declared)**(1/3)-1, filled

    radius = _radius_at_miss(measure, miss, .8*D, 3.5*D)
    assert measure(radius)[0] == pytest.approx(miss, abs=1e-10)
    sim.add_material("glass", eps_r=2.)
    sim.add(Sphere(tuple(centre), radius), material="glass")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = dict(n_steps=1, skip_preflight=True, subpixel_smoothing=smoothing)
        result = (sim.forward(**options) if lane.endswith("forward") else
                  sim.run(compute_s_params=False, **options))
    row = result.realized_geometry.realization_table.row("geometry[0]")
    assert row.filled_volume_m3 == pytest.approx(measure(radius)[1], rel=2e-6)
    messages = [w.message for w in caught if getattr(w.message, "code", None) == "realization"]
    assert len(messages) == (miss > .01)
    if messages:
        assert "filled volume-equivalent length" in str(messages[0])
        assert "centroid shift" in str(messages[0])


@pytest.mark.parametrize("conformal", (True, "cut_cell_floor"))
@pytest.mark.parametrize("miss", (.009, .011))
@pytest.mark.parametrize("snap", ("strict", "declared"))
def test_integral_verdict_boundary_conformal(conformal, miss, snap):
    import warnings
    from rfx.grid import Grid
    sim = Simulation(freq_max=1e9, domain=(4*D,)*3, dx=D, boundary="pec", snap=snap,
                     **({"dt": .5*Grid.courant_dt(D)} if conformal == "cut_cell_floor" else {}))
    grid = sim._build_grid()
    cells = [np.asarray(grid.cells(a)) for a in range(3)]
    nodes = [np.r_[0., np.cumsum(c[:-1])] for c in cells]
    centre = np.array((2.1, 2.17, 1.91))*D
    corners = []
    for normal in range(3):
        u, v = [a for a in range(3) if a != normal]
        for plane in nodes[normal]:
            for i in range(4):
                for j in range(4):
                    points = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        point = centre.copy()
                        point[normal], point[u], point[v] = plane, nodes[u][i+du], nodes[v][j+dv]
                        points.append(np.linalg.norm(point-centre))
                    corners.append(points)
    distances = np.asarray(corners)
    square = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))

    def measure(radius):
        realized = 0.
        for signed in distances-radius:
            polygon, edges = [], []
            for i in range(4):
                j = (i+1) % 4
                if signed[i] >= 0:
                    polygon.append(square[i])
                if (signed[i] >= 0) != (signed[j] >= 0):
                    polygon.append(square[i]+signed[i]/(signed[i]-signed[j])*(square[j]-square[i]))
                    edges.append(max(signed[i], signed[j])/abs(signed[j]-signed[i]))
                else:
                    edges.append(float(signed[i] >= 0))
            if len(polygon) < 3:
                area = 0.
            else:
                polygon = np.array(polygon)
                area = abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)-
                                  polygon[:, 1]*np.roll(polygon[:, 0], -1)))/2
            if conformal == "cut_cell_floor" and 0 < area < 1:
                area = max(area, .25*max(edges))
            realized += area*D**2
        # Each plane cuts a complete disk, wholly inside the transverse domain.
        drawn = sum(len(n)*(4*D)**2 - np.pi*np.maximum(radius**2-(n-centre[a])**2, 0.).sum()
                    for a, n in enumerate(nodes))
        return np.sqrt(realized/drawn)-1, realized

    radius = _radius_at_miss(measure, miss, .6*D, 1.75*D)
    assert measure(radius)[0] == pytest.approx(miss, abs=1e-10)
    sim.add(Sphere(tuple(centre), radius), material="pec")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if snap == "strict" and miss > .01:
            with pytest.raises(ValueError, match="summed open area-equivalent length"):
                sim.run(n_steps=1, skip_preflight=True, compute_s_params=False, conformal_pec=conformal)
        else:
            result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False, conformal_pec=conformal)
            row = result.realized_geometry.realization_table.row("geometry[0]")
            assert sum(row.open_area_m2) == pytest.approx(measure(radius)[1], rel=2e-6)
    messages = [w.message for w in caught if getattr(w.message, "code", None) == "realization"]
    assert len(messages) == (miss > .01 and snap == "declared")


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("miss", (.009, .011))
@pytest.mark.parametrize("snap", ("strict", "declared"))
def test_aperture_verdict_boundary(lane, miss, snap):
    import warnings
    graded = lane.startswith("graded")
    sim = Simulation(freq_max=100e9, domain=(10*D,)*3, dx=D,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=6, snap=snap,
                     **({"dz_profile": np.array((.8, 1.2)*5)*D} if graded else {}))
    length = 4*D/(1+miss)
    centre = (5*D, (4.8 if graded else 5)*D)
    ranges = [(c-length/2, c+length/2) for c in centre]
    # The aperture's snapped edges are bounded by node-aligned PEC walls.
    zlo, zhi = ((2.8, 6.8) if graded else (3., 7.))
    for lo, hi in (((0, 0, 0), (10, 3, 10)), ((0, 7, 0), (10, 10, 10)),
                   ((0, 0, 0), (10, 10, zlo)), ((0, 0, zhi), (10, 10, 10))):
        sim.add(Box(tuple(np.array(lo)*D), tuple(np.array(hi)*D)), material="pec")
    sim.add_waveguide_port(5*D, direction="+x", y_range=ranges[0], z_range=ranges[1],
                          freqs=np.array([80e9]), f0=80e9, bandwidth=.4,
                          ref_offset=1, probe_offset=2)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    spans = []
    for a, bounds in zip((1, 2), ranges):
        cells = np.asarray(grid.cells(a))
        nodes = np.r_[0., np.cumsum(cells[:-1])]
        lo, hi = [np.argmin(abs(nodes-p)) for p in bounds]
        spans.append(nodes[hi]-nodes[lo])
    np.testing.assert_allclose(np.array(spans)/length-1, miss, atol=2e-6, rtol=0)
    report = sim.preflight()
    aperture = [f for f in report if f.code == "port_aperture_snap" and f.loc == "waveguide_port[0]"]
    assert len(aperture) == (1 if graded else 2) * (miss > .01)
    if aperture:
        assert aperture[0].severity == ("error" if snap == "strict" else "warning")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = dict(n_steps=1, skip_preflight=True)
        def solve():
            return (sim.forward(**options) if lane.endswith("forward") else
                    sim.run(compute_s_params=False, **options))
        if snap == "strict" and miss > .01:
            with pytest.raises(ValueError, match="waveguide_port"):
                solve()
        else:
            result = solve()
            row = result.realized_geometry.realization_table.row("waveguide_port[0]")
            np.testing.assert_allclose([a.realized_span_m for a in row.axes[1:]], spans, atol=1e-12, rtol=0)
    messages = [w.message for w in caught if getattr(w.message, "code", None) == "port_aperture_snap"]
    assert len(messages) == (miss > .01 and snap == "declared")


@pytest.mark.parametrize("operation", ("preflight", "run", "skip"))
def test_dc_film_geometry_reuses_one_outcome_per_film(operation, monkeypatch):
    import warnings
    from rfx.model import thin_conductors
    calls = []
    original = thin_conductors._check_dc_area

    def witness(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(thin_conductors, "_check_dc_area", witness)
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec", snap="declared")
    length = 4*D/1.02
    for z in (3*D, 7*D):
        sim.add_thin_conductor(Box((5*D-length/2, 5*D-length/2, z),
                                   (5*D+length/2, 5*D+length/2, z)),
                               sigma_bulk=100., thickness=35e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if operation == "preflight":
            report = sim.preflight()
            messages = [issue for issue in report if issue.code == "dc_film_area"]
        else:
            sim.run(n_steps=1, compute_s_params=False, skip_preflight=operation == "skip")
            messages = [w.message for w in caught if getattr(w.message, "code", None) == "dc_film_area"]
    assert len(messages) == 2
    assert len(calls) == 2


@pytest.mark.parametrize("snap", ("strict", "declared"))
def test_standalone_preflight_reads_the_verdict(snap, monkeypatch):
    from rfx.model import realization_verdict
    sim = scene("pec_box", "uniform_run")
    sim._snap = snap
    calls = []
    original = realization_verdict.judge_realization

    def witness(table, mode):
        findings = original(table, mode)
        calls.extend(findings)
        return findings

    monkeypatch.setattr(realization_verdict, "judge_realization", witness)
    report = sim.preflight()
    findings = [issue for issue in report if issue.code == "realization"]
    assert findings
    assert all(any(issue is finding for finding in calls) for issue in findings)
    assert any(issue.severity == ("error" if snap == "strict" else "warning") for issue in findings)


@pytest.mark.parametrize("smoothing", (True, "full_tensor"))
def test_integral_without_reference_is_reported(smoothing):
    import warnings
    # Overlapping curved unions have no analytic union-volume reference.
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec")
    sim.add_material("glass", eps_r=2.)
    for x in (4*D, 5*D):
        sim.add(Sphere((x, 5*D, 5*D), 2*D), material="glass")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False,
                         subpixel_smoothing=smoothing)
    table = result.realized_geometry.realization_table
    for i in range(2):
        row = table.row(f"geometry[{i}]")
        assert row.status == "not judged: no reference"
        assert row.filled_volume_m3 > 0
        messages = [w.message for w in caught if getattr(w.message, "loc", None) == row.label
                    and getattr(w.message, "code", None) == "realization"]
        assert len(messages) == 1
        assert "not judged: no reference" in str(messages[0])


@pytest.mark.parametrize("case", ("uniform_z", "uniform_x", "graded_z"))
def test_wire_span_includes_the_last_stepped_edge(case, monkeypatch):
    from rfx.stepping import prepared as stepping

    dx = 1 / 1024 if case == "graded_z" else D
    profiles = {}
    if case == "graded_z":
        # The source-on-PEC fixture's metric; z=3*dx is NOT a node here.
        for axis, count in zip("xyz", (12, 8, 8)):
            profile = np.full(count, dx)
            profile[2:4] = (.9*dx, 1.1*dx)
            profiles[f"d{axis}_profile"] = profile
    axis = 0 if case == "uniform_x" else 2
    domain = (24, 12, 12) if axis == 0 else (12, 8, 8)
    start, end = ((10, 6, 6), (14, 6, 6)) if axis == 0 else (
        (6, 4, 2.9 if profiles else 3), (6, 4, 5))
    captured = []
    original = stepping.advance

    def advance(product, stepper, *args, **kwargs):
        def step(grid, *step_args, **step_kwargs):
            captured.append((grid, tuple(np.asarray(m, dtype=bool)
                                         for m in step_kwargs["pec_edge_masks"])))
            return stepper(grid, *step_args, **step_kwargs)
        return original(product, step, *args, **kwargs)

    monkeypatch.setattr(stepping, "advance", advance)
    for snap in ("declared", "strict"):
        sim = Simulation(freq_max=15e9, domain=tuple(v*dx for v in domain),
                         dx=dx, boundary="pec", snap=snap, **profiles)
        sim.add(PolylineWire((tuple(v*dx for v in start), tuple(v*dx for v in end)),
                             radius=0), material="pec")
        refused = None
        try:
            result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
        except ValueError as exc:
            refused = exc
        assert refused is None, f"{case} {snap}: {refused}"
        grid, masks = captured[-1]
        indices = np.where(masks[axis])[axis]
        assert sorted(indices.tolist()) == ([10, 11, 12, 13] if axis == 0 else [3, 4])
        # Count the actual stepping operand's edges by hand using grid cells.
        cells = np.asarray(grid.cells(axis), dtype=float)
        hand_span = sum(float(cells[k]) for k in indices)
        assert hand_span == pytest.approx((4 if axis == 0 else 2.1 if profiles else 2)*dx)
        row = result.realized_geometry.realization_table.row("geometry[0]")
        assert row.axes[axis].realized_span_m == pytest.approx(hand_span)
        for transverse in set(range(3)) - {axis}:
            assert not masks[transverse].any()
            assert row.axes[transverse].bounds_m == pytest.approx((start[transverse]*dx,)*2)


def test_port_built_conductors_leave_user_wire_verdict_intact(monkeypatch):
    import warnings
    from rfx import GaussianPulse
    from rfx.errors import RfxRefusal
    import rfx.sparams.coax as calculators

    stamped = []
    original = calculators.coax_kernel_conductors
    def capture(*args, **kwargs):
        root = original(*args, **kwargs)
        stamped.append(root._realization_record)
        return root
    monkeypatch.setattr(calculators, "coax_kernel_conductors", capture)

    # All three declarations use the same uniform grid and 0.23-cell offset.
    dx = D / 4
    def uniform():
        return Simulation(freq_max=20e9, domain=(.01, .01, .02), dx=dx,
                          boundary="cpml", cpml_layers=3, snap="strict")

    wire = uniform()
    start = (20.23*dx, 20.23*dx, 20.23*dx)
    wire.add_port(start, component="ez", extent=4.38*dx, impedance=50., excite=False)
    coax = uniform()
    coax.add_coaxial_port((start[0], start[1], .01), face="top", pin_length=.005,
                         pin_radius=2.13*dx, outer_radius=7.75*dx,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wire_result = wire.run(n_steps=1, compute_s_params=False, skip_preflight=True)
        coax.compute_coaxial_line_reflection(
            termination="matched", dut_impedance=50., n_steps=400,
            freqs=np.array([8e9]), probe_count=3)
    tables = (wire_result.realized_geometry.realization_table,
              stamped[0].realization_table)
    port = tables[0].row("port[0]")
    assert port.axes[2].declared_span_m == pytest.approx(4.38*dx)
    assert port.axes[2].realized_span_m == pytest.approx(5*dx)
    # Node samples: pin x/y nodes 19..22 -> cell faces 19..23;
    # shell wall is 1 mm = 4dx: radius 11.75dx, nodes 9..31 -> faces 9..32.
    for name, drawn, realized in (("pin", 4.26*dx, 4*dx),
                                  ("shell", 23.5*dx, 23*dx)):
        row = tables[1].row(f"coaxial_port[0]/{name}")
        for axis in row.axes[:2]:
            assert axis.declared_span_m == pytest.approx(drawn)
            assert axis.realized_span_m == pytest.approx(realized)
            assert abs(axis.miss_percent) > 1.
    for table in tables:
        rows = [r for r in table.rows if r.label.startswith(("port[", "coaxial_port["))]
        assert rows
        for row in rows:
            assert row.label in table.judged
            assert row.status == "not judged here: port-built (judged by the port)"
        assert not [f for f in table.findings if f.code == "realization"]
    assert not [w for w in caught if getattr(w.message, "code", None) == "realization"]

    user = uniform()
    user.add(PolylineWire((start, (start[0], start[1], 24.61*dx)), radius=0.), material="pec")
    with pytest.raises(RfxRefusal, match=r"realization geometry\[0\].*extent miss"):
        user.run(n_steps=1, compute_s_params=False, skip_preflight=True)


def test_conformal_zero_drawn_area_has_no_reference_on_nz_one():
    sim = Simulation(20e9, (.012,) * 3, dx=D, boundary="cpml",
                     cpml_layers=3, mode="2d_tez")
    sim.add(Sphere((.006, .006, 0.), .0023), material="pec")
    result = sim.run(n_steps=1, conformal_pec=True, skip_preflight=True,
                     compute_s_params=False)
    assert len(result.realized_geometry.cell_sizes[2]) == 1
    table = result.realized_geometry.realization_table
    row = table.row("geometry[0]")
    assert row.open_area_m2 == (0., 0., 0.)
    assert row.status == "not judged: no reference"
    assert row.area_miss_percent is None
    assert row.label in table.judged
    findings = [f for f in table.findings if f.code == "realization" and f.loc == row.label]
    assert len(findings) == 1
    assert "not judged: no reference" in str(findings[0])


def test_conformal_waveguide_result_names_unevaluated_realization_once():
    from rfx.boundaries.spec import Boundary, BoundarySpec
    text = ("realization: not evaluated: conformal; geometric comparisons were skipped. "
            "Evaluate on uniform run() to check the realized dimensions.")
    def geometry():
        sim = Simulation(10e9, (.06, .024, .012), dx=.002, cpml_layers=4,
                         boundary=BoundarySpec(x="cpml", y=Boundary("pec", "pec", conformal=True),
                                               z=Boundary("pec", "pec", conformal=True)),
                         snap="declared")
        sim.add(Box((.027, .009, .003), (.033, .015, .009)), material="pec")
        return sim
    sim = geometry()
    sim.add_waveguide_port(.016, direction="+x", f0=8e9, freqs=[8e9], name="left")
    sim.add_waveguide_port(.044, direction="-x", f0=8e9, freqs=[8e9], name="right")
    result = sim.compute_waveguide_s_matrix(n_steps=200, normalize=True, skip_preflight=True)
    table = result.realized_geometry.realization_table
    findings = [f for f in table.findings if f.code == "realization"]
    assert [(str(f), f.code, f.source) for f in findings] == [
        (text, "realization", "judge_realization")]
    assert table.row("geometry[0]").status == "not evaluated: conformal"
    assert "geometry[0]" in table.judged

    uniform = geometry().run(n_steps=1, compute_s_params=False, skip_preflight=True)
    evaluated = uniform.realized_geometry.realization_table
    row = evaluated.row("geometry[0]")
    assert row.status == "evaluated"
    assert row.integral_comparison is not None
    assert row.label in evaluated.judged
    assert not [f for f in evaluated.findings if "not evaluated: conformal" in str(f)]

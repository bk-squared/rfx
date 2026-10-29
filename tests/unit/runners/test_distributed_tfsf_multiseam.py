"""Closed-box TF/SF multi-seam validation on up to four devices.

Design: rfx-archive 4b8992b7e, 20261027-tfsf-multiseam-attempt2-PREDECLARE.md.

CPU only. The scratch driver imports these builders and measurements. Steps
are completed steps (1-based); all observer output goes to stderr.
"""
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Box, Simulation, SnapshotSpec, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.measurement.farfield import compute_far_field
from rfx.farfield import make_ntff_box
from rfx.grid import C0
from rfx.measurement.incident import measure_normal_incident_spectrum
from rfx.sources.tfsf import tfsf_from_declaration
from rfx.stepping.slab import Slab

COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")
Z0 = 376.730313668


@dataclass
class Scene:
    sim: object
    cfg: object
    line: object
    grid: object
    steps: int
    metadata: dict
    freqs: np.ndarray
    mie_dbsm: float | None = None


def describe(sim, name, count, steps, expected_nx, expected_box, expected_seams, freqs):
    grid = sim._build_grid()
    cfg, line = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=steps)
    layout = Slab(grid.nx, count)
    seams = [r * layout.nx_per_rank for r in range(1, count)]
    box_rows = [[getattr(cfg, f"{axis}_lo"), getattr(cfg, f"{axis}_hi")] for axis in "xyz"]
    ntff = make_ntff_box(grid, *sim._ntff)
    ntff_rows = [[ntff.i_lo, ntff.i_hi], [ntff.j_lo, ntff.j_hi], [ntff.k_lo, ntff.k_hi]]
    pads = [[getattr(grid, f"pad_{axis}_lo"), getattr(grid, f"pad_{axis}_hi")] for axis in "xyz"]
    meta = dict(scene=name, devices=count, nx=grid.nx, box_rows=[cfg.x_lo, cfg.x_hi],
                shape=list(grid.shape), box_rows_xyz=box_rows, ntff_rows=ntff_rows,
                cpml_layers=grid.cpml_layers, cpml_face_depths=pads,
                slab_width=layout.nx_per_rank, pad_x=layout.pad_x,
                rank_padding=[0] * (count - 1) + [layout.pad_x],
                physical_rows_per_rank=[hi - lo for lo, hi in layout.owned],
                seams=seams, dx_m=grid.dx, steps=steps,
                box_x_m=[(v - grid.pad_x_lo) * grid.dx for v in (cfg.x_lo, cfg.x_hi)])
    assert grid.nx == expected_nx, meta
    assert (cfg.x_lo, cfg.x_hi) == expected_box, meta
    assert seams == expected_seams, meta
    assert layout.pad_x == (-grid.nx) % count, meta
    for a, ((lo, hi), (tf_lo, tf_hi), (pad_lo, pad_hi)) in enumerate(zip(ntff_rows, box_rows, pads)):
        assert pad_lo < lo < tf_lo < tf_hi < hi < grid.shape[a] - pad_hi - 1, meta
    for probe in sim._probes:
        node = grid.position_to_index(probe.position)
        assert node[0] not in seams
        assert all(2 * node[a] != grid.shape[a] - 1 and node[a] != grid.shape[a] // 2
                   for a in range(3)), (probe, node)
    meta["probe_nodes"] = [list(grid.position_to_index(p.position)) for p in sim._probes]
    meta["probe_components"] = [p.component for p in sim._probes]
    return Scene(sim, cfg, line, grid, steps, meta, np.asarray(freqs))


def build_box(name="B2", count=3, *, scale=1, steps=None):
    """Fixed 39 mm domain; PEC and 1 mm, 2.65 S/m sheet span x.

    Boxes use half-open material node occupancy. At h the occupied x rows
    are 18..30 (B1) or 14..34 (B2/B4), with vacuum at/next to TF faces.
    Finer dimensions use nearest-cell, half-up rounding (leader decision).
    """
    assert name in ("B1", "B2", "B4")
    assert scale in (1, 1.5, 2)
    dx = .001 / scale
    layers = int(4 * scale)
    coordinates = {}

    def snap(label, declared):
        cells = int(np.floor(declared / dx + .5 + 1e-10))
        realized = cells * dx
        coordinates[label] = dict(declared_mm=declared * 1e3, realized_mm=realized * 1e3,
                                  cells=cells, shift_mm=(realized - declared) * 1e3,
                                  shift_cells=(realized - declared) / dx)
        return realized

    domain = snap("domain", .039)
    margin_m = snap("margin", .011 if name == "B1" else .007)
    margin = int(round(margin_m / dx))
    snap("cpml_thickness", .004)

    def corner(label, values):
        return tuple(snap(f"{label}.{axis}", value) for axis, value in zip("xyz", values))
    steps = int(240 * scale) if steps is None else steps
    sim = Simulation(freq_max=15e9, domain=(domain,) * 3, dx=dx,
                     boundary="cpml", cpml_layers=layers)
    sim.add_material("plate", eps_r=1.0, sigma=1e7)
    sim.add_material("sheet", eps_r=1.0, sigma=2.65)
    xlo, xhi = ((.014, .027) if name == "B1" else (.010, .031))
    # Thin in y, extended along x: both objects cross both B2 seams.
    sim.add(Box(corner("plate.lo", (xlo, .015, .014)),
                corner("plate.hi", (xhi, .016, .027))), material="plate")
    sim.add(Box(corner("sheet.lo", (xlo, .023, .014)),
                corner("sheet.hi", (xhi, .024, .027))), material="sheet")
    sim.add_tfsf_source(f0=7.5e9, bandwidth=.8, polarization="ez",
                        direction="+x", closed_box=True, margin=margin)
    for p, position in enumerate(((.015, .016, .017), (.025, .018, .021))):
        sim.add_vector_probe(corner(f"probe{p}", position))
    freqs = np.linspace(5e9, 10e9, 9)
    # Outside all TF correction planes, inside the absorber, fixed metres.
    sim.add_ntff_box(corner("ntff.lo", (.003,) * 3), corner("ntff.hi", (.036,) * 3), freqs=freqs)
    nx = {1: 48, 1.5: 72, 2: 95}[scale]
    lo = layers + margin
    seams = { (1, 2): [24], (1, 3): [16, 32], (1, 4): [12, 24, 36],
              (1.5, 3): [24, 48], (1.5, 4): [18, 36, 54],
              (2, 3): [32, 64], (2, 4): [24, 48, 72]}[scale, count]
    scene = describe(sim, name, count, steps, nx, (lo, nx - lo - 1), seams, freqs)
    declared_lo = .011 if name == "B1" else .007
    for axis in "xyz":
        for side, declared, realized in zip(("lo", "hi"), (declared_lo, .039 - declared_lo),
                                            scene.metadata["box_x_m"]):
            coordinates[f"tfsf.{axis}_{side}"] = dict(
                declared_mm=declared * 1e3, realized_mm=realized * 1e3,
                shift_mm=(realized - declared) * 1e3, shift_cells=(realized - declared) / dx)
    scene.metadata.update(scale=scale, coordinates=coordinates,
                          sheet_declared_thickness_m=.001,
                          sheet_realized_thickness_mm=coordinates["sheet.hi.y"]["realized_mm"] - coordinates["sheet.lo.y"]["realized_mm"],
                          sheet_sigma=2.65,
                          source_f0_hz=7.5e9, source_bandwidth=.8,
                          face_on_seam=[v in seams for v in scene.metadata["box_rows"]])
    emit(scene.metadata)
    return scene


def build_s1(count, *, steps=400):
    """Unequal face depths and a short padded last slab, fixed predeclared geometry."""
    depths = ((6, 4), (4, 6), (5, 4))
    boundary = BoundarySpec(**{
        axis: Boundary(lo="cpml", hi="cpml", lo_thickness=lo, hi_thickness=hi)
        for axis, (lo, hi) in zip("xyz", depths)})
    sim = Simulation(freq_max=15e9, domain=(.039, .035, .033), dx=.001,
                     boundary=boundary, cpml_layers=6)
    sim.add_material("plate", eps_r=1., sigma=1e7)
    sim.add_material("sheet", eps_r=1., sigma=2.65)
    corners = {"plate": ((.007, .009, .016), (.019, .014, .022)),
               "sheet": ((.016, .022, .008), (.029, .023, .020))}
    for material, (lo, hi) in corners.items():
        sim.add(Box(lo, hi), material=material)
    sim.add_tfsf_source(f0=7.5e9, bandwidth=.8, polarization="ez",
                        direction="+x", closed_box=True, margin=4)
    for position in ((.013, .017, .011), (.024, .012, .019), (.001, .030, .025)):
        sim.add_vector_probe(position)
    freqs = np.linspace(5e9, 10e9, 9)
    sim.add_ntff_box((.002, .003, .002), (.035, .033, .030), freqs=freqs)
    scene = describe(sim, "S1", count, steps, 50, (10, 39),
                     {2: [25], 3: [17, 34], 4: [13, 26, 39]}[count], freqs)
    meta = scene.metadata
    assert scene.grid.shape == (50, 46, 43), meta
    assert scene.grid.cpml_layers == 6, meta
    assert meta["cpml_face_depths"] == [list(v) for v in depths], meta
    assert meta["box_rows_xyz"] == [[10, 39], [10, 35], [10, 32]], meta
    assert meta["ntff_rows"] == [[8, 41], [7, 37], [7, 35]], meta
    assert meta["pad_x"] == {2: 0, 3: 1, 4: 2}[count], meta
    assert meta["physical_rows_per_rank"][-1] == {2: 25, 3: 16, 4: 11}[count], meta
    # These are corner rows; volume occupancy is half-open at the high corner.
    scatterer_rows = {name: [list(scene.grid.position_to_index(p)) for p in pair]
                      for name, pair in corners.items()}
    assert [p[0] for p in scatterer_rows["plate"]] == [13, 25], scatterer_rows
    assert [p[0] for p in scatterer_rows["sheet"]] == [22, 35], scatterer_rows
    for lo, hi in scatterer_rows.values():
        for a, (tf_lo, tf_hi) in enumerate(meta["box_rows_xyz"]):
            assert tf_lo + 1 < lo[a] and hi[a] <= tf_hi - 1, scatterer_rows
    assert len(sim._probes) == 18
    meta.update(domain_m=[.039, .035, .033], scatterer_corner_rows=scatterer_rows,
                scatterer_x_rows={name: [p[0] for p in pair] for name, pair in scatterer_rows.items()},
                source_f0_hz=7.5e9, source_bandwidth=.8, source_polarization="ez",
                source_direction="+x", margin=4, sheet_sigma=2.65,
                sheet_declared_thickness_m=.001,
                ntff_corners_m=[list(p) for p in sim._ntff[:2]],
                face_on_seam=[v in meta["seams"] for v in meta["box_rows"]])
    emit(meta)
    return scene


def build_sphere(count, x_extra_cells=0):
    assert x_extra_cells in (0, 6)
    directory = Path(__file__).resolve().parents[2] / "fixtures" / "rcs_sphere_mie"
    geo = json.loads((directory / "fixture.json").read_text())["geometry"]
    spec = importlib.util.spec_from_file_location("multiseam_mie_oracle", directory / "mie_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    oracle.validate_oracle()
    f0, radius = geo["f0_hz"], geo["radius_m"]
    dx = C0 / f0 / geo["resolution_cells_per_lambda"]
    domain = list(geo["domain_m"])
    domain[0] += x_extra_cells * dx
    sim = Simulation(freq_max=f0 * 1.5, domain=tuple(domain), dx=dx,
                     boundary="cpml", cpml_layers=geo["cpml_layers"])
    sim.add_material("PEC", eps_r=1., sigma=geo["pec_sigma_s_per_m"])
    sim.add(Sphere(center=tuple(d / 2 for d in domain), radius=radius), material="PEC")
    sim.add_tfsf_source(f0=f0, bandwidth=geo["bandwidth"], polarization=geo["polarization"],
                        direction="+x", closed_box=True, margin=3)
    sim.add_ntff_box((2 * dx,) * 3, ((40 + x_extra_cells) * dx, 40 * dx, 40 * dx), freqs=[f0])
    scene = describe(sim, "S4" if x_extra_cells else "F4", count, geo["n_steps"],
                     90 + x_extra_cells, (27, 62 + x_extra_cells),
                     {(0, 3): [30, 60], (6, 4): [24, 48, 72]}[x_extra_cells, count], [f0])
    assert scene.grid.shape == (90 + x_extra_cells, 90, 90), scene.metadata
    assert scene.metadata["box_rows_xyz"] == [[27, 62 + x_extra_cells], [27, 62], [27, 62]]
    assert scene.metadata["ntff_rows"] == [[26, 64 + x_extra_cells], [26, 64], [26, 64]]
    scene.metadata.update(domain_m=domain, sphere_center_m=[d / 2 for d in domain],
                          sphere_radius_m=radius, x_extra_cells=x_extra_cells,
                          ntff_corners_m=[list(p) for p in sim._ntff[:2]])
    ka = 2 * np.pi * radius * f0 / C0
    s1, _ = oracle.mie_S1_S2(ka, np.pi, n_max=20)
    scene.mie_dbsm = float(10 * np.log10(4 * np.pi * radius ** 2 * abs(s1) ** 2 / ka ** 2))
    emit(scene.metadata)
    return scene


@contextmanager
def observe_line(records, *, multi):
    """Read actual incident advance outputs, never replay them as expectations.

    Hooks preserve operands/results. Reported stage is observed, not inferred
    from a final grid difference. As with the merged ghost witness, callbacks
    are compiled into the measured run.
    """
    from rfx.sources import tfsf
    from rfx.stepping import slab_tfsf

    def capture(phase, step, *fields):
        records[phase, int(step)] = tuple(np.array(v, copy=True) for v in fields)

    def wrap(original, phase):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            line = result[1] if multi else result
            # H advance leaves step unchanged; E advance increments it.
            step = line.step + (1 if phase == "H" else 0)
            jax.debug.callback(lambda n, *v: capture(phase, n, *v), step,
                               *(getattr(line, c) for c in COMPONENTS))
            return result
        return call

    module = slab_tfsf if multi else tfsf
    hn, en = ("apply_h", "apply_e") if multi else ("update_tfsf_1d_h", "update_tfsf_1d_e")
    with patch.object(module, hn, wrap(getattr(module, hn), "H")), \
         patch.object(module, en, wrap(getattr(module, en), "E")):
        yield


def run_scene(scene, count, *, snapshots=True, line_records=None):
    options = dict(n_steps=scene.steps, devices=jax.devices("cpu")[:count])
    if snapshots:
        options["snapshot"] = SnapshotSpec(interval=1, components=COMPONENTS)
    with redirect_stdout(sys.stderr):
        if line_records is None:
            result = scene.sim.run(**options)
        else:
            with observe_line(line_records, multi=count > 1):
                result = scene.sim.run(**options)
        jax.block_until_ready(result.state)
        jax.effects_barrier()
    return result


def difference(a, b):
    """Float64 subtraction avoids rounding the measured float32 difference."""
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return np.abs(b.astype(np.float64) - a.astype(np.float64))


def first_difference(a, b):
    assert a.dtype == b.dtype == np.float32
    nodes = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
    return nodes[0].tolist() if len(nodes) else None


def far_field(result):
    with redirect_stdout(sys.stderr):
        ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, [np.pi / 2], [np.pi])
    return np.stack([np.asarray(ff.E_theta), np.asarray(ff.E_phi)])


def compare_far_field(one, many):
    a, b = far_field(one), far_field(many)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.abs(b.astype(np.complex128) - a.astype(np.complex128))
    peak = float(np.max(np.abs(a)))
    assert peak > 0, "vacuous far-field reference"
    return dict(delta=delta.tolist(), max_abs=float(delta.max()), peak=peak, bar=1e-4 * peak,
                one_real=a.real.tolist(), one_imag=a.imag.tolist(),
                multi_real=b.real.tolist(), multi_imag=b.imag.tolist(),
                passes=bool(np.all(delta <= 1e-4 * peak)))


def measure_probes(scene, one, many):
    """P: probe difference relative to the one-device record peak."""
    a, b = np.asarray(one.time_series), np.asarray(many.time_series)
    assert a.shape == b.shape == (scene.steps, len(scene.sim._probes))
    delta = difference(a, b)
    peak_by_step = np.abs(a).max(axis=1)
    peak_step = int(np.argmax(peak_by_step)) + 1
    peak = float(peak_by_step.max())
    max_abs = float(delta.max())
    precondition = bool(peak > 0 and 50 < peak_step < scene.steps)
    report = dict(max_abs=max_abs, peak=peak, peak_step=peak_step,
                  peak_by_step=peak_by_step.tolist(), bar=1e-5 * peak,
                  relative_difference=max_abs / peak if peak else None,
                  precondition=precondition,
                  passes=bool(precondition and max_abs <= 1e-5 * peak))
    return report, delta


def measure_box(scene, count):
    lines = [{}, {}]
    one = run_scene(scene, 1, line_records=lines[0])
    many = run_scene(scene, count, line_records=lines[1])
    report = dict(scene.metadata, frequencies_hz=scene.freqs.tolist(), grid_diagnostics={}, probes={})
    first = []
    for c in COMPONENTS:
        a, b = np.asarray(one.snapshots[c]), np.asarray(many.snapshots[c])
        assert a.shape == (scene.steps, *scene.grid.shape)
        np.testing.assert_array_equal(one.snapshot_axes[c].steps, np.arange(1, scene.steps + 1))
        np.testing.assert_array_equal(many.snapshot_axes[c].steps, one.snapshot_axes[c].steps)
        delta = difference(a, b).reshape(scene.steps, -1).max(axis=1)
        peak = np.abs(a).reshape(scene.steps, -1).max(axis=1)
        at = first_difference(a, b)
        if at is not None:
            first.append(dict(step=at[0] + 1, component=c, node=at[1:], stage="completed grid step"))
        report["grid_diagnostics"][c] = dict(delta=delta.tolist(), peak=peak.tolist(), first_difference=at)
    report["P"], delta = measure_probes(scene, one, many)
    for p, probe in enumerate(scene.sim._probes):
        report["probes"][str(p)] = dict(component=probe.component,
                                         delta=delta[:, p].tolist())
    report["first_difference"] = min(first, key=lambda v: v["step"]) if first else None
    report["FF"] = compare_far_field(one, many)
    # Read both real line advances. A mismatch here is evidence of its stage;
    # no automatic claim that it caused a later mismatch in the grid.
    line_first = None
    for step in range(1, scene.steps + 1):
        for phase in ("H", "E"):
            for c, x, y in zip(COMPONENTS, lines[0][phase, step], lines[1][phase, step]):
                at = first_difference(x, y)
                if at is not None and line_first is None:
                    line_first = dict(step=step, component=c, node=at,
                                      stage=f"incident line {phase} advance")
    report["first_incident_difference"] = line_first
    report["law"] = {}
    e_line = np.array([max(np.abs(v).max() for v in lines[0]["E", n][:3])
                       for n in range(1, scene.steps + 1)], dtype=np.float64)
    h_line = np.array([max(np.abs(v).max() for v in lines[0]["E", n][3:])
                       for n in range(1, scene.steps + 1)], dtype=np.float64)
    for family, comps in (("E", COMPONENTS[:3]), ("H", COMPONENTS[3:])):
        d = np.max([report["grid_diagnostics"][c]["delta"] for c in comps], axis=0)
        p = np.max([report["grid_diagnostics"][c]["peak"] for c in comps], axis=0)
        scale = np.maximum.reduce([p, e_line if family == "E" else e_line / Z0,
                                   Z0 * h_line if family == "E" else h_line])
        bound = 1.2e-7 * np.cumsum(scale)
        ratio = np.divide(d, bound, out=np.zeros_like(d), where=bound != 0)
        assert not np.any((bound == 0) & (d != 0))
        report["law"][family] = dict(delta=d.tolist(), grid_peak=p.tolist(),
                                      incident_E_peak=e_line.tolist(), incident_H_peak=h_line.tolist(),
                                      accumulated_bound=bound.tolist(), ratio=ratio.tolist(),
                                      max_ratio=float(ratio.max()))
    emit(report)
    return report


def emit(report):
    print(json.dumps(report, allow_nan=False), file=sys.stderr)


def assert_box_bars(report):
    assert report["P"]["precondition"], report
    assert report["P"]["passes"], report
    assert all(v["max_ratio"] <= 1 for v in report["law"].values()), report
    assert report["FF"]["passes"], report


def measure_box_record(builder, count):
    """The predeclare permits exactly one 1.5x extension if the peak is missing."""
    report = measure_box(builder(steps=400), count)
    if not report["P"]["precondition"]:
        initial = report
        report = measure_box(builder(steps=600), count)
        report["record_extension"] = dict(reason="P peak precondition failed", initial=initial)
    return report


def measure_sphere(scene, count):
    one, many = (run_scene(scene, n, snapshots=False) for n in (1, count))
    for result in (one, many):
        box = result.ntff_box
        assert (box.i_lo, box.i_hi, box.j_lo, box.j_hi, box.k_lo, box.k_hi) == tuple(v for pair in scene.metadata["ntff_rows"] for v in pair)
    incident = measure_normal_incident_spectrum(scene.cfg, scene.line, scene.steps,
                                                scene.freqs, scene.grid.dt)
    levels = []
    for result in (one, many):
        ff = far_field(result)
        power = np.sum(np.abs(ff) ** 2, axis=0)
        sigma = float((4 * np.pi * power / np.abs(incident[:, None, None]) ** 2).item())
        levels.append(float(10 * np.log10(sigma)))
    report = dict(scene.metadata, one_dbsm=levels[0], multi_dbsm=levels[1], mie_dbsm=scene.mie_dbsm,
                  one_error_db=abs(levels[0] - scene.mie_dbsm),
                  multi_error_db=abs(levels[1] - scene.mie_dbsm), mie_bar_db=1.0,
                  far_field=compare_far_field(one, many))
    emit(report)
    return report


def measure_ghost(scene, count=3):
    from rfx.runners._distributed_ntff import SlabNTFF
    layout = Slab(scene.grid.nx, count)
    observations = []
    original = SlabNTFF.update

    def capture(hx, step):
        rows = np.asarray(hx).reshape(count, layout.nx_local, scene.grid.ny, scene.grid.nz)
        for rank in range(count - 1):
            a, b = rows[rank, -1].copy(), rows[rank + 1, 1].copy()
            at = first_difference(a, b)
            observations.append(dict(step=int(step) + 1, seam=(rank + 1) * layout.nx_per_rank,
                                     bit_equal=at is None, first_node=at,
                                     max_abs=float(difference(a, b).max()), peak=float(np.abs(b).max())))

    def update(self, buffer, state, dt, step, **kwargs):
        jax.debug.callback(capture, state.hx, step)
        return original(self, buffer, state, dt, step, **kwargs)

    with patch.object(SlabNTFF, "update", update):
        run_scene(scene, count, snapshots=False)
    assert len(observations) == scene.steps * (count - 1)
    assert len({(v["step"], v["seam"]) for v in observations}) == len(observations)
    assert all(max(v["peak"] for v in observations if v["seam"] == seam) > 0
               for seam in scene.metadata["seams"])
    report = dict(scene.metadata, ghost_bar="bit-equal", ghost_max_abs=max(v["max_abs"] for v in observations),
                  passes=all(v["bit_equal"] for v in observations),
                  observations=sorted(observations, key=lambda v: (v["step"], v["seam"])))
    emit(report)
    return report


def require_devices(count):
    if len(jax.devices("cpu")) < count:
        pytest.skip(f"requires {count} CPU host devices")
    assert not jax.config.jax_enable_x64, "predeclare requires float32"


@pytest.mark.parametrize("name", ["B2", "S1"])
def test_comparator_two_devices(name):
    require_devices(2)
    builder = (lambda **kw: build_box("B2", 2, **kw)) if name == "B2" else (lambda **kw: build_s1(2, **kw))
    assert_box_bars(measure_box_record(builder, 2))


@pytest.mark.parametrize("count", [3, 4])
def test_s1_multiseam(count):
    require_devices(count)
    assert_box_bars(measure_box_record(lambda **kw: build_s1(count, **kw), count))


@pytest.mark.parametrize("count", [3, 4])
def test_s1_forward_parity(count):
    require_devices(count)
    scene = build_s1(count)
    with redirect_stdout(sys.stderr):
        one = scene.sim.forward(n_steps=scene.steps, checkpoint=False)
        many = scene.sim.forward(n_steps=scene.steps, checkpoint=False,
                                 distributed=True, devices=jax.devices("cpu")[:count])
        jax.block_until_ready(many.time_series)
        jax.effects_barrier()
    report, _ = measure_probes(scene, one, many)
    emit(dict(scene="S1", entry="forward", devices=count, steps=scene.steps,
              P={k: v for k, v in report.items() if k != "peak_by_step"}))
    assert report["precondition"], report
    assert report["passes"], report


@pytest.mark.parametrize("count,x_extra_cells", [(3, 0), (4, 6)], ids=["F4-3", "S4-4"])
def test_sphere_multiseam(count, x_extra_cells):
    require_devices(count)
    report = measure_sphere(build_sphere(count, x_extra_cells=x_extra_cells), count)
    assert report["one_error_db"] <= 1.0, report
    assert report["multi_error_db"] <= 1.0, report
    assert report["far_field"]["passes"], report


def test_b2_right_hx_ghost():
    require_devices(3)
    report = measure_ghost(build_box("B2", 3))
    assert report["passes"], report

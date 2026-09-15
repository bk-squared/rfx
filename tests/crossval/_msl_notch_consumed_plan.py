"""Fingerprint concrete uniform-scalar CV06b runner inputs before fields.

Pass the wrapper's ORIGINAL ``rfx.simulation.run`` to bind the call. This
module neither assembles a second model nor advances a field. Comparisons
are per drive within one backend/version. Only the controlled stub's node
footprint and its fully enclosed tangential edges may differ. The allowed
box includes the main-line high node so junction Ey edges are covered.

Final supplied PEC edge masks are authoritative on this restricted lane:
simulation._build_step_setup re-realizes sheets only when those masks are
None. Occupancy, Kottke/conformal, ADE and other physics are refused here.
Sheet footprints outside the allowed box are ALSO retained for provenance.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import operator
from dataclasses import fields

import jax
import numpy as np

from rfx.boundaries.pec import SheetSpec
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.grid import Grid
from rfx.probes.probes import DFTPlaneProbe
from rfx.simulation import ProbeSpec, SourceSpec


_ABSENT = (
    "debye", "lorentz", "tfsf", "flux_monitors", "waveguide_ports", "ntff",
    "snapshot", "aniso_eps", "aniso_inv_eps", "pec_mask", "pec_wires",
    "pec_occupancy", "conformal_weights", "wire_port_sparams",
    "lumped_port_sparams", "wire_refplane_sparams", "lumped_rlc",
    "kerr_chi3", "mag_sources", "sheet_impedance",
)
_OPTIONS = (
    "boundary", "cpml_axes", "pec_axes", "periodic", "checkpoint",
    "checkpoint_segments", "aniso_inv_eps_smooth", "return_state", "stencil_order", "report_every",
)
_KNOWN_RUN = set(_ABSENT + _OPTIONS + (
    "grid", "materials", "n_steps", "sources", "probes", "dft_planes",
    "pec_sheets", "pec_edge_masks", "field_dtype", "report_every", "report_label",
))
_GRID_FIELDS = set((
    "freq_max domain cpml_layers kappa_max pec_faces pmc_faces conformal_faces "
    "face_layers cpml_axes mode is_2d dx dt pad_x_lo pad_x_hi pad_y_lo pad_y_hi "
    "pad_z_lo pad_z_hi pad_x pad_y pad_z axis_pads face_pads nx ny nz shape interior"
).split())


def _plain(value):
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    if isinstance(value, slice):
        return [_plain(value.start), _plain(value.stop), _plain(value.step)]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_plain(v) for v in sorted(value)]
    raise ValueError(f"unsupported scalar metadata: {type(value).__name__}")


def _array(value, name, shape=None):
    if isinstance(value, jax.core.Tracer):
        raise ValueError(f"{name}: consumed-plan checks require concrete arrays")
    array = np.asarray(value)
    if array.dtype.kind not in "bifuc" or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite numeric data")
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"{name}: shape {array.shape} differs from {tuple(shape)}")
    return array


def _record(array):
    array = np.ascontiguousarray(array)
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(memoryview(array).cast("B")).hexdigest())


def _index(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean indices are not scalar Yee indices")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError("expected an integer Yee index") from exc


def stub_box_from_geometry(geometry):
    """Baseline actual node indices, inclusive; includes the junction edge."""
    return (tuple(map(int, geometry["stub_i"])),
            (int(geometry["trace_j"][1]), int(geometry["stub_open_j"])),
            (int(geometry["plane_k"]), int(geometry["plane_k"])))


def _box_slices(box, shape, component=None):
    if len(box) != 3:
        raise ValueError("allowed_stub_box must have three inclusive node intervals")
    slices = []
    for axis, bounds in enumerate(box):
        if len(bounds) != 2:
            raise ValueError("allowed_stub_box requires integer node endpoints")
        lo, hi = map(_index, bounds)
        if not 0 <= lo <= hi < shape[axis]:
            raise ValueError("allowed_stub_box is outside the runner grid")
        # An E edge must have BOTH endpoints inside the permitted node box.
        slices.append(slice(lo, hi if axis == component else hi + 1))
    if box[2][0] != box[2][1] or any(box[a][0] == box[a][1] for a in (0, 1)):
        raise ValueError("CV06b controlled geometry must be an xy sheet")
    return tuple(slices)


def _outside_record(array, allowed):
    outside = array.copy()
    outside[allowed] = False
    return _record(outside)


def _cell(item, shape):
    cell = tuple(_index(getattr(item, name)) for name in ("i", "j", "k"))
    if any(not 0 <= c < n for c, n in zip(cell, shape)):
        raise ValueError(f"source/probe cell {cell} is outside the grid")
    return list(cell)


def fingerprint_run_call(run_function, args, kwargs, *, allowed_stub_box):
    """Return JSON-native invariant hashes plus permitted-variation evidence.

    ``allowed_stub_box`` is the SAME baseline box for every arm/drive. The
    returned ``observed`` payload is not part of equality: it includes full
    changing PEC hashes and progress labels. Progress chunking stays fixed.
    Everything under
    ``invariants`` is compared. New run arguments are refused until reviewed.
    """
    bound = inspect.signature(run_function).bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    if set(values) != _KNOWN_RUN:
        raise ValueError(f"unsupported runner argument schema: {sorted(set(values) ^ _KNOWN_RUN)}")
    for name in _ABSENT:
        value = values[name]
        if value is not None and not (isinstance(value, (list, tuple)) and not value):
            raise ValueError(f"unsupported CV06b physics/channel: {name}")
    grid = values["grid"]
    if type(grid) is not Grid or set(vars(grid)) != _GRID_FIELDS:
        raise ValueError("expected the known concrete uniform Grid schema")
    if grid.mode != "3d" or grid.pmc_faces or grid.conformal_faces:
        raise ValueError("unsupported CV06b grid/boundary physics")
    if values["boundary"] != "cpml" or values["stencil_order"] != 2 or values["aniso_inv_eps_smooth"]:
        raise ValueError("expected scalar second-order Yee with CPML")
    if values["periodic"] not in (None, (False, False, False)):
        raise ValueError("periodic physics is outside this CV06b check")
    steps = _index(values["n_steps"])
    if steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    dtype = np.dtype(values["field_dtype"] if values["field_dtype"] is not None else np.float32)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("expected a real float32/float64 field lane")
    box = tuple(tuple(map(_index, pair)) for pair in allowed_stub_box)
    node_region = _box_slices(allowed_stub_box, grid.shape)
    coordinates = coords_from_uniform_grid(grid)
    invariants = dict(
        allowed_stub_box=_plain(box), n_steps=steps,
        grid={key: _plain(value) for key, value in sorted(vars(grid).items())},
        nodes={axis: _record(np.asarray(getattr(coordinates, axis))) for axis in "xyz"},
        options={key: _plain(values[key]) for key in _OPTIONS},
        field_dtype=dtype.str,
        execution=dict(jax_version=jax.__version__, backend=jax.default_backend(),
                       x64_enabled=bool(jax.config.x64_enabled),
                       matmul_precision=_plain(jax.config.jax_default_matmul_precision)),
        unsupported_channels={name: "absent" for name in _ABSENT},
    )
    mats = values["materials"]
    if getattr(mats, "_fields", ()) != ("eps_r", "sigma", "mu_r"):
        raise ValueError("unknown material-array schema")
    invariants["materials_after_load"] = {}
    for name in mats._fields:
        array = _array(getattr(mats, name), name, grid.shape)
        if array.dtype.kind != "f" or np.any(array < 0) or (name != "sigma" and np.any(array == 0)):
            raise ValueError("expected positive epsilon/mu and nonnegative conductivity")
        invariants["materials_after_load"][name] = _record(array)

    source_records = []
    for source in values["sources"] or ():
        if (type(source) is not SourceSpec or source._fields != ("i", "j", "k", "component", "waveform")
                or source.component != "ez"):
            raise ValueError("expected the CV06b Ez-only SourceSpec channel")
        wave = _array(source.waveform, "source waveform", (steps,))
        if wave.dtype.kind != "f":
            raise ValueError("expected a real source waveform")
        source_records.append(dict(cell=_cell(source, grid.shape), component=source.component,
                                   waveform=_record(wave)))
    if not source_records:
        raise ValueError("CV06b drive has no electric sources")
    invariants["sources"] = source_records
    invariants["point_probes"] = []
    for probe in values["probes"] or ():
        if (type(probe) is not ProbeSpec or probe._fields != ("i", "j", "k", "component")
                or probe.component not in ("ex", "ey", "ez", "hx", "hy", "hz")):
            raise ValueError("unknown point-probe schema/component")
        invariants["point_probes"].append(dict(cell=_cell(probe, grid.shape), component=probe.component))
    invariants["dft_planes"] = []
    for probe in values["dft_planes"] or ():
        if (type(probe) is not DFTPlaneProbe or probe._fields != (
                "accumulator", "freqs", "component", "axis", "index", "total_steps",
                "window", "window_alpha", "region")):
            raise ValueError("unknown DFT-plane schema/axis")
        axis, index = _index(probe.axis), _index(probe.index)
        if axis not in (0, 1, 2):
            raise ValueError("unknown DFT-plane axis")
        if probe.component not in ("ex", "ey", "ez", "hx", "hy", "hz"):
            raise ValueError("unknown DFT component")
        if not 0 <= index < grid.shape[axis] or probe.window != "rect":
            raise ValueError("unsupported DFT plane index/window")
        freqs = _array(probe.freqs, "DFT frequencies")
        if freqs.ndim != 1 or not freqs.size or np.any(freqs <= 0):
            raise ValueError("expected positive DFT frequency vector")
        plane_shape = [n for a, n in enumerate(grid.shape) if a != axis]
        if probe.region is not None:
            region = tuple(map(_index, probe.region))
            if len(region) != 4 or not (0 <= region[0] < region[1] <= plane_shape[0]
                                       and 0 <= region[2] < region[3] <= plane_shape[1]):
                raise ValueError("DFT crop is outside the field plane")
            plane_shape = [region[1] - region[0], region[3] - region[2]]
        accumulator = _array(probe.accumulator, "DFT accumulator", (freqs.size, *plane_shape))
        if accumulator.dtype.kind != "c" or np.any(accumulator != 0):
            raise ValueError("expected a fresh zero complex DFT accumulator")
        invariants["dft_planes"].append(dict(
            component=probe.component, axis=axis, index=index,
            region=_plain(probe.region), freqs=_record(freqs), accumulator=_record(accumulator),
            total_steps=_index(probe.total_steps), window=probe.window, window_alpha=_plain(probe.window_alpha)))

    masks = values["pec_edge_masks"]
    if masks is None or len(masks) != 3:
        raise ValueError("require final supplied PEC edge masks; do not substitute declaration-only masks")
    full_masks = []
    invariants["pec_outside_stub"] = []
    for component, value in enumerate(masks):
        mask = _array(value, "PEC edge mask", grid.shape)
        if mask.dtype.kind != "b":
            raise ValueError("expected boolean PEC edge masks")
        full_masks.append(_record(mask))
        invariants["pec_outside_stub"].append(_outside_record(mask, _box_slices(box, grid.shape, component)))
    sheets = values["pec_sheets"] or ()
    if len(sheets) != 2:
        raise ValueError("CV06b expects exactly two PEC sheets")
    invariants["sheets_outside_stub"] = []
    full_sheets = []
    for sheet in sheets:
        if (type(sheet) is not SheetSpec or {f.name for f in fields(sheet)} != {
                "normal_axis", "plane", "footprint", "name"}
                or sheet.normal_axis != 2 or sheet.plane != box[2][0]):
            raise ValueError("expected the trace and stub on one canonical PEC sheet plane")
        footprint = _array(sheet.footprint, "sheet footprint", grid.shape)
        if footprint.dtype.kind != "b":
            raise ValueError("expected boolean sheet node footprints")
        invariants["sheets_outside_stub"].append(dict(normal_axis=sheet.normal_axis, plane=sheet.plane,
            name=_plain(sheet.name), footprint=_outside_record(footprint, node_region)))
        full_sheets.append(_record(footprint))
    digest = hashlib.sha256(json.dumps(invariants, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(schema_version=1, invariant_sha256=digest, invariants=invariants,
                observed=dict(pec_full=full_masks, sheet_footprints_full=full_sheets,
                              report_every=_plain(values["report_every"]), report_label=_plain(values["report_label"])))


def compare_run_plans(baseline, candidate):
    """Reject the first invariant difference; permitted PEC payloads may vary."""
    if baseline.get("schema_version") != 1 or candidate.get("schema_version") != 1:
        raise ValueError("unsupported consumed-plan schema")

    def difference(a, b, path):
        if type(a) is not type(b):
            return path
        if isinstance(a, dict):
            if a.keys() != b.keys():
                return path
            for key in a:
                found = difference(a[key], b[key], f"{path}.{key}")
                if found:
                    return found
        elif isinstance(a, list):
            if len(a) != len(b):
                return path
            for index, (left, right) in enumerate(zip(a, b)):
                found = difference(left, right, f"{path}[{index}]")
                if found:
                    return found
        elif a != b:
            return path
        return None

    changed = difference(baseline["invariants"], candidate["invariants"], "invariants")
    if changed:
        raise ValueError(f"consumed runner plan changed: {changed}")

#!/usr/bin/env python3
"""Read-only #752 raw-record analysis. Writes only a NEW output directory.

No RFX imports, field calls, reference correction, or production-data replacement.
Scientific verdicts live in summary.json; exit 0 means analysis was written.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from scipy.optimize import minimize_scalar

LABELS = ("h3", "h4", "h5", "h6", "dx80", "dx60")
EXPECTED_DX = dict(
    zip(LABELS, (254e-6 / 3, 254e-6 / 4, 254e-6 / 5, 254e-6 / 6, 80e-6, 60e-6))
)
FLO, FHI, BAR, LIMIT, RESIDUAL_LIMIT = 3e9, 4.5e9, -40.0, 0.004, 0.02
C0 = 299792458.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return [safe(value.real), safe(value.imag)]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def references(w, h, er):
    u = w / h
    ee = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 / u)
    simplified = (
        60 / math.sqrt(ee) * math.log(8 / u + u / 4)
        if u <= 1
        else 120 * math.pi / (math.sqrt(ee) * (u + 1.393 + 0.667 * math.log(u + 1.444)))
    )
    a = (
        1
        + math.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49
        + math.log(1 + (u / 18.1) ** 3) / 18.7
    )
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    ee_hj = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    fu = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    hj = (
        376.730313668
        / (2 * math.pi * math.sqrt(ee_hj))
        * math.log(fu / u + math.sqrt(1 + (2 / u) ** 2))
    )
    return {"repo_simplified": simplified, "HJ1980": hj}, ee


def fixed_fit(v, x, beta, current, direction):
    """Independent complex128 SVD fit, keeping physical probe-0 reference."""
    v = np.asarray(v, np.complex128)
    x = np.asarray(x, np.float64) - float(x[0])
    scale = float(np.linalg.norm(v))
    if (
        not np.isfinite(v).all()
        or not np.isfinite(current)
        or scale == 0
        or current == 0
    ):
        return {
            "valid": False,
            "reason": "nonfinite or zero voltage/current evidence",
            "relative_residual": np.inf,
        }
    matrix = np.column_stack((np.exp(-1j * beta * x), np.exp(1j * beta * x)))
    coeff, _, rank, singular = np.linalg.lstsq(matrix, v / scale, rcond=None)
    resid = float(np.linalg.norm(matrix @ coeff - v / scale))
    coeff = coeff * scale
    cond = float(singular[0] / singular[-1]) if singular[-1] else np.inf
    z0 = direction * (coeff[0] - coeff[1]) / current
    return {
        "valid": bool(rank == 2 and np.isfinite(z0) and np.isfinite(cond)),
        "beta": float(beta),
        "alpha": coeff[0],
        "gamma": coeff[1],
        "z0": z0,
        "relative_residual": resid,
        "design_condition": cond,
        "rank": int(rank),
    }


def wide_fit(v, x, current, direction, beta0, nodes):
    """Dense independent beta0*(.3..1.7) scan; refine every local minimum."""
    ratios = np.linspace(0.3, 1.7, nodes)
    cache = {}

    def evaluate(r):
        key = float(r)
        if key not in cache:
            cache[key] = fixed_fit(v, x, key * beta0, current, direction)
        return cache[key]

    values = np.array([evaluate(r)["relative_residual"] for r in ratios])
    if not np.isfinite(values).any():
        return {"valid": False, "reason": "no finite fit"}
    local = (
        np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])) + 1
    )
    # Group a plateau so floating dust cannot generate thousands of minimizers.
    groups = np.split(local, np.where(np.diff(local) > 1)[0] + 1) if len(local) else []
    local = np.array([g[np.argmin(values[g])] for g in groups if len(g)], dtype=int)
    truncated = len(local) > 32
    if truncated:
        local = local[np.argsort(values[local])[:32]]
    candidates = [(0.3, evaluate(0.3)), (1.7, evaluate(1.7))]
    for index in local:
        solved = minimize_scalar(
            lambda r: evaluate(r)["relative_residual"] ** 2,
            bounds=(ratios[index - 1], ratios[index + 1]),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 100},
        )
        r = float(solved.x)
        candidates.append((r, evaluate(r)))
        candidates.append((float(ratios[index]), evaluate(ratios[index])))
    index = int(np.argmin(values))
    candidates.append((float(ratios[index]), evaluate(ratios[index])))

    def window(lo, hi):
        items = [(r, f) for r, f in candidates if lo <= r <= hi]
        items += [(lo, evaluate(lo)), (hi, evaluate(hi))]
        ratio, best = min(items, key=lambda item: item[1]["relative_residual"])
        best = dict(
            best,
            beta_ratio_to_repo=ratio,
            at_search_limit=bool(
                min(ratio - lo, hi - ratio) <= (ratios[1] - ratios[0]) / 2
            ),
        )
        return best

    best = window(0.3, 1.7)
    best["prior_windows"] = {
        "production_fraction_window": window(0.65, 1.35),
        "minus20percent_prior_same_fraction_window": window(0.52, 1.08),
        "plus20percent_prior_same_fraction_window": window(0.78, 1.62),
    }
    best["local_minima"] = [
        {
            "beta_ratio_to_repo": r,
            "relative_residual": f["relative_residual"],
            "z0": f.get("z0"),
        }
        for r, f in candidates[2::2]
    ]
    best["local_minima_count_truncated"] = truncated
    best["scan_grid_nodes"] = nodes
    return best


def settling(path):
    with np.load(path, allow_pickle=False) as data:
        raw = data["time_series"]
    if (
        raw.ndim != 2
        or len(raw) < 10
        or raw.shape[1] == 0
        or not np.isfinite(raw).all()
    ):
        return {"valid": False, "reason": "unavailable/invalid point records"}
    # Match the existing MSL definition, independently in extended precision.
    power = raw.astype(np.longdouble) ** 2
    peak = power.max(axis=0)
    tail = power[-max(1, len(raw) // 10) :].mean(axis=0)
    tiny = np.longdouble(np.finfo(np.float64).tiny)
    db = 10 * np.log10((tail + tiny) / (peak + tiny))
    return {
        "valid": bool(np.isfinite(db).all()),
        "per_column_db": np.asarray(db, float),
        "worst_db": float(db.max()),
        "shape": list(raw.shape),
        "storage_dtype": str(raw.dtype),
        "zero_peak_columns": np.flatnonzero(peak == 0),
    }


def low_signal(v, current):
    vm, im = np.abs(v[:, 0]), np.abs(current)
    return (vm < 0.1 * np.median(vm)) & (im < 0.1 * np.median(im))


def array_signature(value):
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def _roundoff_equal(actual, expected, magnitude_sum, operations, *stores):
    """Storage-roundoff identity check, not a physical error allowance.

    Positive operation counts conservatively include scalar casts, complex
    products, reductions and final stores. The absolute L1 scale retains
    cancellation error; a relative tolerance on the near-zero result would not.
    """
    eps = max(np.finfo(np.asarray(a).real.dtype).eps for a in (actual, *stores))
    nu = operations * eps / 2
    gamma = nu / (1 - nu)
    bound = gamma * (np.asarray(magnitude_sum) + np.abs(actual))
    return bool(np.all(np.abs(actual - expected) <= bound))


def validate_evidence(
    plan, metadata, raw, result, diagnostics, consumed, drive_records, drive_inputs
):
    """Cross-check the frozen uniform-foil capture before any numerical fit.

    Every drive/port raw record is evidence, including passive records. These
    are schema, provenance and duplicated-data identities, not extra physics
    thresholds. In particular no passive-port model residual is gated here.
    """
    errors = []

    def require(condition, reason):
        if not bool(condition) and reason not in errors:
            errors.append(reason)

    try:
        required_raw = {
            "freqs_hz",
            "raw_v",
            "raw_i1",
            "raw_i1_left",
            "raw_i1_same_index",
            "raw_z0",
            "raw_q",
            "production_smatrix",
            "production_z0",
            "production_beta",
            "driven_port_indices",
        }
        required_result = {
            "freqs",
            "S",
            "Z0",
            "beta",
            "settling_db",
            "reference_impedances",
            "reliable",
            "beta_railed",
            "cond_a",
        }
        for key in sorted(required_raw - raw.keys()):
            errors.append(f"missing_raw_array:{key}")
        for key in sorted(required_result - result.keys()):
            errors.append(f"missing_result_array:{key}")
        if errors:
            return errors
        for group, arrays in [
            ("raw", raw),
            ("result", result),
            *[(f"drive{d}_records", a) for d, a in enumerate(drive_records)],
            *[(f"drive{d}_inputs", a) for d, a in enumerate(drive_inputs)],
        ]:
            for key, array in arrays.items():
                array = np.asarray(array)
                require(array.dtype.kind in "biufc", f"nonnumeric_array:{group}.{key}")
                if array.dtype.kind in "biufc":
                    require(np.isfinite(array).all(), f"nonfinite_array:{group}.{key}")
        if errors:
            return errors
        freqs = np.asarray(raw["freqs_hz"])
        nf = len(freqs)
        shapes = {
            "raw_v": (2, 2, 5, nf),
            "raw_i1": (2, 2, nf),
            "raw_i1_left": (2, 2, nf),
            "raw_i1_same_index": (2, 2, nf),
            "raw_z0": (2, 2, nf),
            "raw_q": (2, 2, nf),
            "production_smatrix": (2, 2, nf),
            "production_z0": (2, nf),
            "production_beta": (nf,),
            "driven_port_indices": (2,),
        }
        for name, shape in shapes.items():
            require(raw[name].shape == shape, f"raw_shape:{name}")
        for name, shape in {
            "S": (2, 2, nf),
            "Z0": (2, nf),
            "beta": (nf,),
            "settling_db": (2,),
            "reference_impedances": (2,),
            "reliable": (2, nf),
            "beta_railed": (2, nf),
            "cond_a": (nf,),
        }.items():
            require(result[name].shape == shape, f"result_shape:{name}")
        require(
            np.array_equal(freqs, plan["frequencies_hz"]), "plan_frequency_mismatch"
        )
        require(np.array_equal(freqs, result["freqs"]), "result_frequency_mismatch")
        require(
            np.array_equal(raw["driven_port_indices"], [0, 1]),
            "raw_drive_identity_mismatch",
        )
        require(
            len(consumed) == len(drive_records) == len(drive_inputs) == 2,
            "consumed_drive_count_mismatch",
        )
        require(
            diagnostics.get("completed_drives") == 2, "reported_drive_count_mismatch"
        )
        if errors:
            return errors
        for name, other in (
            ("production_smatrix", "S"),
            ("production_z0", "Z0"),
            ("production_beta", "beta"),
        ):
            require(
                np.array_equal(raw[name], result[other]),
                f"result_dump_mismatch:{other}",
            )
        own_z0 = np.stack([raw["raw_z0"][p, p] for p in range(2)])
        require(
            np.array_equal(own_z0, raw["production_z0"]), "raw_z0_diagonal_mismatch"
        )
        for key, expected in {
            "schema": "rfx.msl_nprobe_dump",
            "schema_version": 4,
            "s_wave_convention": "power",
            "current_convention": "native_msl_loop_current",
            "current_spatial_alignment": "linear_bracketing_H_to_E_node",
        }.items():
            require(metadata.get(key) == expected, f"convention_mismatch:{key}")
        require(
            metadata.get("production_smatrix_assembly") == diagnostics.get("assembly"),
            "assembly_metadata_mismatch",
        )
        require(
            np.array_equal(
                metadata["s_reference_impedances_ohm"], result["reference_impedances"]
            ),
            "reference_impedance_mismatch",
        )
        dx, dt = float(plan["dx_m"]), float(plan["dt_s"])
        shape = tuple(plan["grid_shape"])
        require(
            tuple(metadata["grid"][f"n{a}"] for a in "xyz") == shape,
            "grid_shape_mismatch",
        )
        require(
            metadata["grid"]["dx_m"] == dx and metadata["grid"]["dt_s"] == dt,
            "grid_metric_mismatch",
        )
        require(metadata["n_probes_per_port"] == [5, 5], "probe_count_mismatch")
        require(
            len(metadata["port_definitions"]) == len(plan["port_definitions"]) == 2,
            "port_count_mismatch",
        )
        require(
            len(metadata["current_plane_stencils"]) == 2,
            "current_stencil_count_mismatch",
        )
        if errors:
            return errors
        # The frozen campaign has eight external x/y pad cells and z_lo PEC.
        px, py, pz = 8, 8, 0
        nh = int(plan["height_intervals"])
        expected_probes, expected_dfts, port_geometry = [], [], []
        for p, (definition, declared) in enumerate(
            zip(metadata["port_definitions"], plan["port_definitions"])
        ):
            for key in (
                "name",
                "direction",
                "mode",
                "n_probe_offset",
                "n_probe_spacing",
                "n_probes",
            ):
                require(
                    definition[key] == declared[key],
                    f"port_definition_mismatch:p{p}.{key}",
                )
            for key, planned in (
                ("position_m", "position"),
                ("width_m", "width"),
                ("height_m", "height"),
                ("impedance_ohm", "impedance"),
            ):
                require(
                    np.array_equal(definition[key], declared[planned]),
                    f"port_definition_mismatch:p{p}.{key}",
                )
            require(
                definition["direction"] == ("+x" if p == 0 else "-x"),
                f"port_direction_mismatch:p{p}",
            )
            sign = 1 if p == 0 else -1
            feed = round(definition["position_m"][0] / dx) + px
            centre = round(definition["position_m"][1] / dx) + py
            jlo = (
                round((definition["position_m"][1] - definition["width_m"] / 2) / dx)
                + py
            )
            jhi = (
                round((definition["position_m"][1] + definition["width_m"] / 2) / dx)
                + py
            )
            indices = feed + sign * (
                definition["n_probe_offset"]
                + np.arange(5) * definition["n_probe_spacing"]
            )
            xs = (indices - px) * dx
            require(
                np.allclose(xs, plan["probe_positions_m"][p], rtol=0, atol=1e-14),
                f"probe_coordinates_mismatch:p{p}",
            )
            stencil = metadata["current_plane_stencils"][p]
            ei = int(indices[0])
            require(
                stencil["axis"] == "x" and stencil["e_index"] == ei,
                f"current_E_reference_mismatch:p{p}",
            )
            require(
                stencil["h_indices"] == [ei - 1, ei], f"current_H_indices_mismatch:p{p}"
            )
            expected_numeric = {
                "voltage_coordinate": float(xs[0]),
                "registration_coordinates": [(ei - 1 - px) * dx, (ei - px) * dx],
                "sample_coordinates": [(ei - 0.5 - px) * dx, (ei + 0.5 - px) * dx],
                "weights": [0.5, 0.5],
            }
            for key, expected in expected_numeric.items():
                # The solver stores interpolation weights after several
                # float64 coordinate subtractions.  A few ulps are storage
                # roundoff, not a geometry change; the dedicated metadata
                # mutations above move the coordinate by a full cell and
                # remain rejected.
                require(
                    np.allclose(stencil[key], expected, rtol=0, atol=1e-12),
                    f"current_stencil_mismatch:p{p}.{key}",
                )
            for index in indices:
                expected_probes.append(
                    {
                        "i": int(index),
                        "j": centre,
                        "k": round(definition["height_m"] / (2 * dx)) + pz,
                        "component": "ez",
                    }
                )
                expected_dfts.append(("ez", int(index), [centre, centre + 1, 0, nh]))
            for component in ("hy", "hz"):
                for index in (ei - 1, ei):
                    expected_dfts.append(
                        (component, index, [jlo - 1, jhi + 1, nh - 1, nh + 1])
                    )
            port_geometry.append((feed, jlo, jhi))
        for d, (call, records, inputs) in enumerate(
            zip(consumed, drive_records, drive_inputs)
        ):
            require(call["drive"] == d, f"consumed_drive_identity_mismatch:drive{d}")
            require(
                call["n_steps"] == plan["n_steps"],
                f"consumed_step_count_mismatch:drive{d}",
            )
            require(
                call["boundary"] == "cpml"
                and call["cpml_axes"] == "xyz"
                and call["pec_axes"] is None
                and call["periodic"] is None,
                f"consumed_boundary_mismatch:drive{d}",
            )
            for material in ("eps_r", "sigma", "mu_r"):
                actual = inputs[f"material_{material}"]
                require(
                    actual.shape == shape, f"input_material_shape:drive{d}.{material}"
                )
                signature = array_signature(actual)
                require(
                    signature == call["material_signatures"][material],
                    f"consumed_material_mismatch:drive{d}.{material}",
                )
                if material != "sigma":
                    require(
                        signature == plan["material_signatures"][material],
                        f"planned_material_mismatch:drive{d}.{material}",
                    )
            for axis in range(3):
                require(
                    array_signature(inputs[f"pec_{axis}"])
                    == plan["pec_edge_signatures"][axis],
                    f"consumed_PEC_mismatch:drive{d}.e{axis}",
                )
            require(
                call["probes"] == expected_probes, f"consumed_probe_mismatch:drive{d}"
            )
            require(
                records["time_series"].shape == (plan["n_steps"], len(expected_probes)),
                f"point_record_shape_mismatch:drive{d}",
            )
            source_locations = []
            feed, jlo, jhi = port_geometry[d]
            for index, source in enumerate(call["sources"]):
                location = (source["i"], source["j"], source["k"])
                source_locations.append(location)
                require(
                    source["component"] == "ez"
                    and source["i"] == feed
                    and max(0, jlo - nh) <= source["j"] <= min(shape[1] - 1, jhi + nh)
                    and 0 <= source["k"] < nh,
                    f"consumed_source_location_mismatch:drive{d}.{index}",
                )
                waveform = inputs[f"source_{index}"]
                require(
                    waveform.shape == (plan["n_steps"],),
                    f"source_waveform_shape:drive{d}.{index}",
                )
                require(
                    array_signature(waveform) == source["waveform"],
                    f"source_waveform_signature:drive{d}.{index}",
                )
            require(
                bool(source_locations)
                and len(source_locations) == len(set(source_locations)),
                f"source_support_invalid:drive{d}",
            )
            expected_input_keys = {f"material_{n}" for n in ("eps_r", "sigma", "mu_r")}
            expected_input_keys |= {f"pec_{a}" for a in range(3)}
            expected_input_keys |= {f"source_{s}" for s in range(len(call["sources"]))}
            require(
                inputs.keys() == expected_input_keys, f"input_array_inventory:drive{d}"
            )
            require(
                len(call["dft_planes"]) == len(expected_dfts),
                f"DFT_count_mismatch:drive{d}",
            )
            require(
                records.keys()
                == {"time_series", *[f"dft_{n}" for n in range(len(expected_dfts))]},
                f"record_array_inventory:drive{d}",
            )
            for index, (actual, expected) in enumerate(
                zip(call["dft_planes"], expected_dfts)
            ):
                component, plane, region = expected
                require(
                    (
                        actual["component"],
                        actual["axis"],
                        actual["index"],
                        actual["region"],
                    )
                    == (component, 0, plane, region),
                    f"DFT_registration_mismatch:drive{d}.{index}",
                )
                require(
                    np.array_equal(actual["freqs_hz"], freqs),
                    f"DFT_frequency_mismatch:drive{d}.{index}",
                )
                require(
                    actual["total_steps"] == plan["n_steps"]
                    and actual["window"] == "rect",
                    f"DFT_window_mismatch:drive{d}.{index}",
                )
                require(
                    records[f"dft_{index}"].shape
                    == (nf, region[1] - region[0], region[3] - region[2]),
                    f"DFT_array_shape:drive{d}.{index}",
                )
        require(
            np.array_equal(
                drive_inputs[0]["material_sigma"], drive_inputs[1]["material_sigma"]
            ),
            "source_load_sigma_differs_between_drives",
        )
        if errors:
            return errors
        for d in range(2):
            for p in range(2):
                offset = p * 9
                for probe in range(5):
                    plane = drive_records[d][f"dft_{offset + probe}"]
                    terms = plane[:, 0, :].astype(np.complex128) * dx
                    require(
                        _roundoff_equal(
                            raw["raw_v"][d, p, probe],
                            terms.sum(axis=1),
                            np.abs(terms).sum(axis=1),
                            8 * (nh + 1),
                            plane,
                        ),
                        f"voltage_DFT_mismatch:drive{d}.port{p}.probe{probe}",
                    )
                weights = metadata["current_plane_stencils"][p]["weights"]
                left, right = raw["raw_i1_left"][d, p], raw["raw_i1_same_index"][d, p]
                expected = weights[0] * left.astype(np.complex128) + weights[
                    1
                ] * right.astype(np.complex128)
                # Side currents have the SAME temporal correction and direction
                # convention as I1. Linear interpolation therefore commutes with
                # the linear Ampere loop, modulo finite-precision reductions.
                planes = [drive_records[d][f"dft_{offset + q}"] for q in range(5, 9)]
                scale = sum(np.abs(h).sum(axis=(1, 2)) * dx for h in planes)
                scale += weights[0] * np.abs(left) + weights[1] * np.abs(right)
                operations = 16 * (sum(h.shape[1] * h.shape[2] for h in planes) + 1)
                require(
                    _roundoff_equal(
                        raw["raw_i1"][d, p],
                        expected,
                        scale,
                        operations,
                        left,
                        right,
                        *planes,
                    ),
                    f"current_interpolation_mismatch:drive{d}.port{p}",
                )
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        ZeroDivisionError,
        AttributeError,
        OverflowError,
    ) as exc:
        errors.append(f"malformed_provenance:{type(exc).__name__}:{exc}")
    return errors


def analyze_case(label, directory, nodes):
    directory = Path(directory)
    needed = [
        "plan.json",
        "raw-vi.npz",
        "result.npz",
        "diagnostics.json",
        "drive0-records.npz",
        "drive1-records.npz",
        "drive0-inputs.npz",
        "drive1-inputs.npz",
        "consumed-plans.json",
    ]
    missing = [name for name in needed if not (directory / name).is_file()]
    if missing:
        return {
            "label": label,
            "status": "incomplete",
            "missing_files": missing,
            "input_directory": str(directory),
        }
    plan = json.loads((directory / "plan.json").read_text())
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    consumed = json.loads((directory / "consumed-plans.json").read_text())
    with np.load(directory / "raw-vi.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        # port_names is the schema's legacy object/string array. Never unpickle
        # it; typed plan/port definitions and driven_port_indices own the mapping.
        raw = {
            name: data[name]
            for name in data.files
            if name not in ("metadata_json", "port_names")
        }
    with np.load(directory / "result.npz", allow_pickle=False) as data:
        result = {name: data[name] for name in data.files}
    drive_records, drive_inputs = [], []
    for drive in range(2):
        with np.load(
            directory / f"drive{drive}-records.npz", allow_pickle=False
        ) as data:
            drive_records.append({name: data[name] for name in data.files})
        with np.load(
            directory / f"drive{drive}-inputs.npz", allow_pickle=False
        ) as data:
            drive_inputs.append({name: data[name] for name in data.files})
    provenance_errors = validate_evidence(
        plan, metadata, raw, result, diagnostics, consumed, drive_records, drive_inputs
    )
    if provenance_errors:
        return {
            "label": label,
            "status": "inconclusive",
            "provenance_valid": False,
            "protocol_failure_reasons": provenance_errors,
            "comparisons": {},
            "input_directory": str(directory),
            "input_sha256": {name: digest(directory / name) for name in needed},
        }
    freqs = np.asarray(raw["freqs_hz"], float)
    nf = len(freqs)
    primary = (freqs >= FLO) & (freqs <= FHI)
    assert plan["label"] == label and primary.any()
    assert np.array_equal(
        freqs, np.linspace(0.5e9, 5e9, 30, dtype=np.float32).astype(float)
    ), "frequency grid changed"
    assert np.array_equal(freqs, np.asarray(plan["frequencies_hz"], float))
    assert (
        abs(plan["dx_m"] - EXPECTED_DX[label]) < 1e-18
        and plan["trace_thickness_m"] == 0
    )
    assert abs(plan["height_m"] - plan["height_intervals"] * plan["dx_m"]) < 1e-17
    assert abs(plan["width_m"] - plan["width_intervals"] * plan["dx_m"]) < 1e-17
    assert plan["eps_r"] == 3.66 and plan["num_periods"] in (20.0, 40.0)
    assert (
        metadata["schema"] == "rfx.msl_nprobe_dump" and metadata["schema_version"] == 4
    )
    assert metadata["s_wave_convention"] == "power"
    assert raw["raw_v"].shape[:2] == (2, 2) and raw["raw_v"].shape[-1] == nf
    assert raw["raw_i1"].shape == (2, 2, nf)
    assert result["Z0"].shape == (2, nf) and result["S"].shape == (2, 2, nf)
    assert np.array_equal(freqs, result["freqs"])
    refs, ee = references(plan["width_m"], plan["height_m"], plan["eps_r"])
    beta0 = 2 * np.pi * freqs * math.sqrt(ee) / C0
    z_independent = np.full((2, 2, nf), np.nan + 1j * np.nan)
    z_native_refit = np.full((2, 2, nf), np.nan + 1j * np.nan)
    native_resid = np.full((2, 2, nf), np.inf)
    independent_resid = np.full((2, 2, nf), np.inf)
    independent_rails = np.ones((2, 2, nf), bool)
    weak = np.zeros((2, 2, nf), bool)
    rows = []
    for driven in range(2):
        for port in range(2):
            definition = metadata["port_definitions"][port]
            sign = 1 if definition["direction"].startswith("+") else -1
            axis = {"x": 0, "y": 1}[definition["direction"][-1]]
            n = int(metadata["n_probes_per_port"][port])
            dx = float(metadata["grid"]["dx_m"])
            source_node = round(definition["position_m"][axis] / dx)
            xs = (
                source_node
                + sign
                * (
                    definition["n_probe_offset"]
                    + np.arange(n) * definition["n_probe_spacing"]
                )
            ) * dx
            np.testing.assert_allclose(
                xs, plan["probe_positions_m"][port], rtol=0, atol=1e-14
            )
            assert n == 5 and np.all(np.diff(xs) * sign > 0)
            assert (
                definition["mode"] == "laplace" and definition["impedance_ohm"] == 50.0
            )
            assert abs(definition["height_m"] - plan["height_m"]) < 1e-17
            assert abs(definition["width_m"] - plan["width_m"]) < 1e-17
            delta = xs[1] - xs[0]
            # q is unambiguous here: even the wide search remains below Nyquist.
            assert np.max(1.7 * beta0 * abs(delta)) < np.pi
            v = np.asarray(raw["raw_v"][driven, port, :n, :].T, np.complex128)
            current = np.asarray(raw["raw_i1"][driven, port], np.complex128)
            weak[driven, port] = low_signal(v, current)
            native_beta = -np.angle(raw["raw_q"][driven, port]) / delta
            for k, f in enumerate(freqs):
                native = fixed_fit(v[k], xs, float(native_beta[k]), current[k], sign)
                independent = wide_fit(
                    v[k], xs, current[k], sign, float(beta0[k]), nodes
                )
                native_resid[driven, port, k] = native["relative_residual"]
                independent_resid[driven, port, k] = independent.get(
                    "relative_residual", np.inf
                )
                independent_rails[driven, port, k] = independent.get(
                    "at_search_limit", True
                )
                z_native_refit[driven, port, k] = native.get("z0", np.nan)
                z_independent[driven, port, k] = independent.get("z0", np.nan)
                rows.append(
                    {
                        "drive": driven,
                        "port": port,
                        "frequency_hz": float(f),
                        "primary_bin": bool(primary[k]),
                        "own_drive": driven == port,
                        "native_beta_inferred_from_q": float(native_beta[k]),
                        "production_raw_z0": raw["raw_z0"][driven, port, k],
                        "relative_low_signal": bool(weak[driven, port, k]),
                        "production_current_regularizer_fraction": float(
                            1e-30 / abs(current[k])
                        )
                        if current[k] != 0
                        else np.inf,
                        "native_beta_float64_refit": native,
                        "independent_wide_fit": independent,
                    }
                )
    settle = [settling(directory / f"drive{d}-records.npz") for d in range(2)]
    failures = []
    if len(consumed) != 2 or diagnostics.get("completed_drives") != 2:
        failures.append("incomplete_drives")
    for name in ("S", "Z0", "beta"):
        if name not in result or not np.isfinite(result[name][..., primary]).all():
            failures.append("nonfinite_primary_" + name)
    if not np.array_equal(result["S"], raw["production_smatrix"], equal_nan=True):
        failures.append("S_differs_from_preprojection_dump")
    sd = result.get("settling_db")
    if (
        sd is None
        or sd.shape != (2,)
        or not np.isfinite(sd).all()
        or not np.all(sd <= BAR)
    ):
        failures.append("reported_settling_not_eligible")
    if not all(s["valid"] and s["worst_db"] <= BAR for s in settle):
        failures.append("recomputed_settling_not_eligible")
    if sd is not None and sd.shape == (2,) and all(s["valid"] for s in settle):
        if not np.allclose(sd, [s["worst_db"] for s in settle], rtol=0, atol=1e-8):
            failures.append("settling_replay_disagrees")
    reliable = result.get("reliable")
    railed = result.get("beta_railed")
    if reliable is None or reliable.shape != (2, nf) or not reliable[:, primary].all():
        failures.append("production_signal_screen")
    if railed is None or railed.shape != (2, nf) or railed[:, primary].any():
        failures.append("production_beta_rail")
    for port in range(2):
        if (
            not np.isfinite(native_resid[port, port, primary]).all()
            or np.max(native_resid[port, port, primary]) > RESIDUAL_LIMIT
        ):
            failures.append(f"own_drive_native_voltage_model_residual_port{port}")
    comparisons = {}
    for name, ref in refs.items():
        comparisons[name] = []
        for port in range(2):
            production = result["Z0"][port, primary]
            own = z_independent[port, port, primary]
            native = z_native_refit[port, port, primary]
            mean = float(np.mean(production.real.astype(np.float64)))
            error = (mean - ref) / ref
            diagonal_rows = [
                r
                for r in rows
                if r["drive"] == port and r["port"] == port and r["primary_bin"]
            ]
            window_means = {}
            for window in (
                "production_fraction_window",
                "minus20percent_prior_same_fraction_window",
                "plus20percent_prior_same_fraction_window",
            ):
                window_means[window] = float(
                    np.mean(
                        [
                            r["independent_wide_fit"]
                            .get("prior_windows", {})
                            .get(window, {})
                            .get("z0", complex(np.nan))
                            .real
                            for r in diagonal_rows
                        ]
                    )
                )
            comparisons[name].append(
                {
                    "port": port,
                    "reference_ohm": ref,
                    "production_band_mean_real_z0": mean,
                    "production_relative_error": error,
                    "within_0p4_percent": bool(
                        np.isfinite(error) and abs(error) <= LIMIT
                    ),
                    "independent_band_mean_real_z0": float(np.mean(own.real)),
                    "independent_relative_error": float(np.mean(own.real) / ref - 1),
                    "native_beta_refit_band_mean_real_z0": float(np.mean(native.real)),
                    "wide_minus_production_band_mean_relative": float(
                        (np.mean(own.real) - mean) / ref
                    ),
                    "native_refit_minus_production_band_mean_relative": float(
                        (np.mean(native.real) - mean) / ref
                    ),
                    "max_independent_vs_production_complex_z0_relative": float(
                        np.max(abs(own - production)) / ref
                    ),
                    "prior_window_band_mean_real_z0": window_means,
                    "prior_window_band_mean_spread_relative": float(
                        (max(window_means.values()) - min(window_means.values())) / ref
                    ),
                    "production_real_spectral_min": float(np.min(production.real)),
                    "production_real_spectral_max": float(np.max(production.real)),
                    "production_max_abs_imag_over_reference": float(
                        np.max(abs(production.imag)) / ref
                    ),
                }
            )
    ss = np.asarray(result["S"])[:, :, primary]
    sv = np.array(
        [
            np.linalg.svd(ss[:, :, k], compute_uv=False)[0]
            if np.isfinite(ss[:, :, k]).all()
            else np.nan
            for k in range(ss.shape[-1])
        ]
    )
    own_rails = np.stack([independent_rails[p, p] for p in range(2)])
    identity = {
        "driver_sha256": plan.get("driver_sha256"),
        "jax": plan.get("jax"),
        "python": plan.get("python"),
        "backend": plan.get("backend"),
        "x64": plan.get("x64"),
        "port_waveforms": [p.get("waveform") for p in plan["port_definitions"]],
        "frequency_sha256": hashlib.sha256(freqs.tobytes()).hexdigest(),
    }
    output = {
        "label": label,
        "status": "usable_under_declared_screens" if not failures else "inconclusive",
        "protocol_failure_reasons": failures,
        "input_directory": str(directory),
        "input_sha256": {n: digest(directory / n) for n in needed},
        "campaign_identity": identity,
        "plan": plan,
        "frequencies_hz": freqs,
        "primary_mask": primary,
        "primary_bin_count": int(primary.sum()),
        "settling_recomputed": settle,
        "comparisons": comparisons,
        "quality_flags": {
            "production_reliable": reliable,
            "production_beta_railed": railed,
            "independent_relative_low_signal_per_drive_port": weak,
            "independent_wide_rails_own_drive": own_rails,
            "independent_wide_primary_rails_present": bool(own_rails[:, primary].any()),
            "native_voltage_residual_own_drive": np.stack(
                [native_resid[p, p] for p in range(2)]
            ),
            "independent_voltage_residual_own_drive": np.stack(
                [independent_resid[p, p] for p in range(2)]
            ),
        },
        "S_diagnostics_not_Z0_accuracy_gates": {
            "assembly": diagnostics.get("assembly"),
            "cond_a": result.get("cond_a"),
            "max_primary_column_power": float(np.max(np.sum(abs(ss) ** 2, axis=0))),
            "max_primary_sigma_max": float(np.max(sv)),
            "max_primary_complex_reciprocity_difference": float(
                np.max(abs(ss[0, 1] - ss[1, 0]))
            ),
        },
        "warnings": diagnostics.get("warnings", []),
        "probe_clearance": diagnostics.get("probe_clearance"),
        "per_record_fit_diagnostics": rows,
    }
    return output


def aggregate(cases):
    result = {}
    identities = {
        json.dumps(safe(c["campaign_identity"]), sort_keys=True)
        for c in cases
        if "campaign_identity" in c
    }
    coherent = len(identities) <= 1
    for reference in ("repo_simplified", "HJ1980"):
        groups = {}
        for ports, name in (
            ((0,), "port0_historical_hypothesis"),
            ((1,), "port1_secondary"),
            ((0, 1), "both_ports_secondary"),
        ):
            eligible = [
                c for c in cases if c.get("status") == "usable_under_declared_screens"
            ]
            bad = [
                {
                    "case": c["label"],
                    "port": p,
                    "relative_error": c["comparisons"][reference][p][
                        "production_relative_error"
                    ],
                }
                for c in eligible
                for p in ports
                if not c["comparisons"][reference][p]["within_0p4_percent"]
            ]
            complete = len(eligible) == len(LABELS) and {
                c["label"] for c in eligible
            } == set(LABELS)
            verdict = (
                "inconclusive"
                if not coherent
                else "refuted"
                if bad
                else "consistent_with_six_point_data"
                if complete
                else "inconclusive"
            )
            groups[name] = {
                "verdict": verdict,
                "coherent_campaign_inputs": coherent,
                "all_six_within_0p4_percent": bool(complete and not bad and coherent),
                "valid_violations": bad,
                "usable_cases": [c["label"] for c in eligible],
            }
        result[reference] = groups
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--case-dir",
        action="append",
        default=[],
        help="Override one case directory, LABEL=PATH",
    )
    parser.add_argument("--scan-nodes", type=int, default=1401)
    args = parser.parse_args()
    if args.scan_nodes < 101:
        parser.error("--scan-nodes must be at least101")
    paths = {label: args.input_root / label for label in LABELS}
    for entry in args.case_dir:
        label, sep, path = entry.partition("=")
        if not sep or label not in paths:
            parser.error("case-dir needs a known LABEL=PATH")
        paths[label] = Path(path)
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for label in LABELS:
        print("ANALYZE", label, str(paths[label]), flush=True)
        try:
            case = analyze_case(label, paths[label], args.scan_nodes)
        except Exception as error:
            case = {
                "label": label,
                "status": "analysis_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "input_directory": str(paths[label]),
            }
        cases.append(case)
        dump(args.out / f"{label}.json", case)
        print(
            label,
            case["status"],
            case.get("protocol_failure_reasons", case.get("error", "")),
            flush=True,
        )
    summary = {
        "analysis_script_sha256": digest(__file__),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "scan_nodes": args.scan_nodes,
        "primary_band_hz": [FLO, FHI],
        "relative_comparison_limit": LIMIT,
        "normal_model_residual_rejection_screen": RESIDUAL_LIMIT,
        "no_field_solves": True,
        "production_values_not_replaced": True,
        "fit_precision": "independent complex128/float64; production extractor casts complex64/float32",
        "limits": [
            "2% voltage residual is not a0.4% uncertainty bound",
            "finite condition does not certify current scale",
            "wide-scan/prior sensitivity is reported, never used to calibrate production",
            "raw S power remains separate; no #726 redesign",
            "agreement is not absolute RF accuracy",
        ],
        "case_status": {c["label"]: c["status"] for c in cases},
        "hypotheses": aggregate(cases),
    }
    dump(args.out / "summary.json", summary)
    print(json.dumps(safe(summary["hypotheses"]), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

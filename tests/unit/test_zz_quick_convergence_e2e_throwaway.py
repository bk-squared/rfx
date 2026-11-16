"""Throwaway measurements for DESIGN-e2e.md (archive ddc882998).

No numerical expectations are assertions; outcomes include failures and warnings.
Touch-point table: none - throwaway measurement file, no product code changed.
Conclusion: 리더가 채움
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
import rfx.convergence as conv

OUT = Path(__file__).resolve().parents[2] / "e2e_out"
# The number measurement computes int(np.ceil(MSL["record_s"] / float(g.dt)))
# on _build_msl()._build_realized_grid(). Reuse its design-fixed result at all dx.
N_STEPS = 6994
RUN_KWARGS = {}


MSL = dict(
    eps_r=3.0,        # a common low-loss laminate value
    h=0.9e-3,         # substrate height; 3 cells
    cell=0.3e-3,      # h/3, the coarsest cells/h the oracle uses (3); keeps ~54k cells
    w=2.1e-3,         # ~2.3 h, near 50 ohm for eps_r 3 (Hammerstad), 7 cells
    length=21.0e-3,   # 70 cells; leaves 16.2 mm between feeds for probes
    air=1.8e-3,       # side and top air, 6 cells (small to stay near 60k cells)
    f_max=10e9,       # 30+ cells per guided wavelength at f_max
    cpml=8,           # half the oracle's 16 to stay small; advisory only
    feed_inset=2.4e-3,  # 8 cells from each line end
    record_s=4e-9,    # ~2.7 transits of the 21 mm line at eps_eff ~2.4, x ~10
)
MSL_FREQS = np.linspace(2e9, 10e9, 33)


def _build_msl(dx=MSL["cell"]):
    p = MSL
    domain = (p["length"], 2 * p["air"] + p["w"], p["h"] + p["air"])
    mesh_kwargs = {} if dx is None else {"dx": dx}
    sim = Simulation(
        freq_max=p["f_max"], domain=domain, cpml_layers=p["cpml"], **mesh_kwargs,
        snap="declared",
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=p["eps_r"])
    sim.add(Box((0, 0, 0), (p["length"], domain[1], p["h"])), material="substrate")
    from rfx.preflight.line_port_coverage import port_node_coordinate
    yc = p["air"] + p["w"] / 2
    x1, x2 = p["feed_inset"], p["length"] - p["feed_inset"]
    ends = [port_node_coordinate(sim, (x, yc, 0)) for x in (x1, x2)]
    sim.add(Box((ends[0], p["air"], p["h"]), (ends[1], p["air"] + p["w"], p["h"])),
            material="pec")
    # Explicit probe layout: the automatic spacing (~lambda_g/4 per step)
    # would need more line than 16 mm. Offset 4 cells, spacing 3 cells.
    for name, x, direction in (("p1", x1, "+x"), ("p2", x2, "-x")):
        sim.add_msl_port(position=(x, yc, 0), width=p["w"], height=p["h"],
                         eps_r_sub=p["eps_r"], impedance=50.0,
                         direction=direction, name=name,
                         n_probes=3, n_probe_offset=4, n_probe_spacing=3)
    # One Ez probe mid-substrate under the strip, off the line's centre
    # (0.55 L), so the run() record has an observable on both sides.
    sim.add_probe((0.55 * p["length"], yc, p["h"] / 2), "ez")
    return sim


def _safe(value):
    """Keep every returned field and array element, including raw Results."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else str(v)
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": _safe(value.real), "imag": _safe(value.imag)}
    if isinstance(value, np.generic):
        return _safe(value.item())
    if hasattr(value, "__array__"):
        return _safe(np.asarray(value).tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _safe(getattr(value, f.name)) for f in fields(value)}
    if hasattr(value, "_asdict"):
        return _safe(value._asdict())
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_safe(v) for v in sorted(value, key=repr)]
    if hasattr(value, "__dict__"):
        return {"type": type(value).__name__, "fields": _safe(vars(value))}
    return repr(value)


def _measure(record, operation, *, serialize=False):
    """Catch each operation independently; preserve any partial observations."""
    start = time.perf_counter()
    value = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = operation()
            record["wall_time_s"] = time.perf_counter() - start
            record["outcome"] = "returned"
            record["returned_type"] = type(value).__name__
            if serialize:
                record["returned"] = _safe(value)
        except Exception as err:
            # A serialization error must not masquerade as a simulation failure.
            key = "serialization_exception" if record.get("outcome") == "returned" else "exception"
            record[key] = {"type": type(err).__name__, "message": str(err)[:300]}
            if key == "exception":
                record["outcome"] = "raised"
        finally:
            record.setdefault("wall_time_s", time.perf_counter() - start)
            record["warnings"] = [
                {"category": w.category.__name__, "message": str(w.message)[:200]}
                for w in caught
            ]
    return value


def _write(part, record):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{part}.json"
    path.write_text(json.dumps(_safe(record), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    # Rebuild from the available parts so either test can run first or alone.
    lines = ["# E2E measurement outcomes", ""]
    for name in ("part_a", "part_b"):
        part_path = OUT / f"{name}.json"
        if not part_path.exists():
            continue
        data = json.loads(part_path.read_text(encoding="utf-8"))
        for label, call in data["calls"].items():
            detail = call.get("exception", call.get("serialization_exception"))
            suffix = "" if detail is None else " — " + detail["type"] + ": " + detail["message"]
            lines.append(
                f"- {name} / {label}: {call['outcome']}; "
                f"{len(call['warnings'])} warnings; {call['wall_time_s']:.3f} s"
                + suffix.replace("\n", " ")
            )
    lines.extend(["", "Touch-point table: none - throwaway measurement file, no product code changed",
                  "", "Conclusion: 리더가 채움", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return path


def _user_metric(metric):
    # Bodies copied verbatim from quick_convergence's string metric mapping.
    if metric == "resonance":
        def metric_fn(result):
            modes = result.find_resonances()
            if not modes:
                return float("nan")
            return float(modes[0].freq)
    elif metric == "peak_field":
        def metric_fn(result):
            ts = np.asarray(result.time_series)
            return float(np.max(np.abs(ts)))
    return metric_fn


def test_e2e_quick_vs_user():
    record = {"scene": "msl_through", "parameters": MSL, "band_hz": MSL_FREQS,
              "n_steps": N_STEPS, "run_kwargs": RUN_KWARGS, "calls": {}}
    for metric in ("peak_field", "resonance"):
        qc = {"arguments": {"metric": metric, "dx_factors": [1.5, 1.0],
                            "base_dx": MSL["cell"], "n_steps": N_STEPS,
                            "until_decay": None, "run_kwargs": RUN_KWARGS}}
        record["calls"][f"QC/{metric}"] = qc
        _measure(qc, lambda: conv.quick_convergence(
            _build_msl(), metric=metric, dx_factors=[1.5, 1.0],
            n_steps=N_STEPS, run_kwargs=dict(RUN_KWARGS)), serialize=True)
        user = {"arguments": {"sim_factory": "_build_msl(dx)",
                              "dx_values": [0.45e-3, 0.3e-3], "metric_fn": metric,
                              "n_steps": N_STEPS, "until_decay": None,
                              "run_kwargs": RUN_KWARGS, "expected_order": 2.0}}
        record["calls"][f"USER/{metric}"] = user
        _measure(user, lambda: conv.convergence_study(
            sim_factory=_build_msl, dx_values=[0.45e-3, 0.3e-3],
            metric_fn=_user_metric(metric), n_steps=N_STEPS,
            run_kwargs=dict(RUN_KWARGS)), serialize=True)
    assert _write("part_a", record).exists()
    assert (OUT / "summary.md").exists()


def _profile(profile):
    if profile is None:
        return {"present": False, "length": 0, "min": None, "max": None, "values": None}
    arr = np.asarray(profile)
    return {"present": True, "length": len(arr),
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None, "values": arr.tolist()}


class _Captured(Exception):
    """Intentional stop before convergence_study runs any simulation."""


def _factory(sim, factor, record):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        raise _Captured()

    original = conv.convergence_study
    conv.convergence_study = fake
    try:
        try:
            conv.quick_convergence(sim, metric="peak_field", dx_factors=[factor])
        except _Captured:
            record["capture_stop"] = {"type": "_Captured", "message": ""}
    finally:
        conv.convergence_study = original
    record["captured_kwargs"] = {
        k: (getattr(v, "__qualname__", repr(v)) if callable(v) else _safe(v))
        for k, v in captured.items()
    }
    if "sim_factory" not in captured:
        raise RuntimeError("quick_convergence returned without calling convergence_study")
    rebuilt = captured["sim_factory"](captured["dx_values"][0])
    passed = rebuilt._declared_mesh["_dz_profile"]
    resolved = sim._dz_profile
    record["passed_dz_profile"] = _profile(passed)
    record["passed_dz_profile_equal_to_source"] = (
        passed is None and resolved is None if passed is None or resolved is None
        else bool(np.array_equal(np.asarray(passed), np.asarray(resolved)))
    )
    record["resolved_dx"] = rebuilt._dx
    record["resolved_domain"] = rebuilt._domain
    record["resolved_dz_profile"] = _profile(rebuilt._dz_profile)
    grid = rebuilt._build_realized_grid()
    record["grid"] = {"type": type(grid).__name__,
                      "shape": [int(v) for v in grid.shape], "dt": float(grid.dt)}
    return record["grid"]


def test_auto_mesh_msl():
    record = {"scene": "msl_through", "parameters": MSL, "dx_omitted": True,
              "calls": {"automatic_scene": {}}}
    source = record["calls"]["automatic_scene"]

    def build():
        sim = _build_msl(dx=None)
        source["resolved_dx"] = sim._dx
        source["resolved_domain"] = sim._domain
        source["resolved_dz_profile"] = _profile(sim._dz_profile)
        source["dz_profile_inferred"] = (
            sim._declared_mesh["_dz_profile"] is None and sim._dz_profile is not None)
        return sim

    sim = _measure(source, build)
    # Even a source construction failure is an outcome; do not change the scene.
    for factor in (1.0, 0.5):
        call = {"arguments": {"metric": "peak_field", "dx_factors": [factor],
                               "n_steps": None, "until_decay": None, "run_kwargs": None}}
        record["calls"][f"factory/{factor}"] = call

        def inspect():
            if sim is None:
                raise RuntimeError("Automatic scene unavailable; factory call could not be attempted")
            return _factory(sim, factor, call)

        _measure(call, inspect)
    record["auto_mesh_warnings"] = [
        w for call in record["calls"].values() for w in call["warnings"]
        if w["category"] == "AutoMeshWarning"
    ]
    assert _write("part_b", record).exists()
    assert (OUT / "summary.md").exists()

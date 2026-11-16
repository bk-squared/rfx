"""Throwaway dx-variant measurement; no expectations are asserted.

Design: rfx-archive 465889c81, records/20261116-declaration-rebuild/
DESIGN-dx-variant.md. No time stepping and no precision configuration changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parents[2] / "variant_out"
FREQ_MAX = 10e9
DOMAIN = (0.020, 0.016, 0.012)
DX = 1e-3
CPML = 8
_DIP = [0.9, 0.8, 0.8, 0.8, 0.8, 0.9]
PZ = np.array([1.0] * 3 + _DIP + [1.0] * 4) * 1e-3
GRID_ATTRS = ("shape", "dt", "dx", "pad_x_lo", "pad_x_hi", "pad_y_lo", "pad_y_hi",
              "pad_z_lo", "pad_z_hi", "cpml_layers", "cpml_axes", "face_layers")
GRID_FIELDS = ("type",) + GRID_ATTRS


def base(dx, **over):
    from rfx.api import Simulation

    kwargs = dict(freq_max=FREQ_MAX, domain=DOMAIN, dx=dx, cpml_layers=CPML)
    kwargs.update(over)
    sim = Simulation(**kwargs)
    sim.add_source((0.007, 0.006, 0.005), "ez")
    sim.add_probe((0.013, 0.009, 0.007), "ez")
    return sim


def floquet_base(dx):
    from rfx.api import Simulation

    return Simulation(freq_max=FREQ_MAX, domain=(0.015, 0.015, 0.03),
                      dx=dx, boundary="cpml", cpml_layers=CPML)


def uniform(dx):
    return base(dx)


def auto_dx(dx):
    from rfx import Box

    sim = base(dx)
    sim.add_material("variant_dielectric", eps_r=3.0)
    sim.add(Box((0.005, 0.004, 0.003), (0.010, 0.008, 0.006)),
            material="variant_dielectric")
    return sim


def floquet_auto(dx):
    sim = floquet_base(dx)
    sim.add_floquet_port(0.008, axis="z", polarization="te", n_modes=1)
    return sim


def graded_z(dx):
    return base(dx, dz_profile=PZ.copy())


def dt_pin(dx):
    from rfx.grid import Grid

    # Scene input, not an expected output. Fresh builds recompute this pin.
    return base(dx, dt=0.5 * Grid.courant_dt(dx, 3))


SCENES = (("uniform", uniform, DX), ("auto_dx", auto_dx, None),
          ("floquet_auto", floquet_auto, None), ("graded_z", graded_z, DX),
          ("dt_pin", dt_pin, DX))


def _exception(exc):
    return {"type": type(exc).__name__, "message": str(exc)[:200]}


def _grid_summary(sim):
    # Same field selection and representation as the declaration census.
    grid = sim._build_realized_grid()
    out = {"type": type(grid).__name__}
    for attr in GRID_ATTRS:
        value = getattr(grid, attr, "<absent>")
        if attr == "shape" and value != "<absent>":
            value = tuple(int(v) for v in value)
        out[attr] = repr(value) if not isinstance(value, (int, float, str, tuple)) else value
    return out


def _identity_checks(source):
    """Check a separate no-replacement variant after the dx build attempt."""
    result = {"key_equal": None, "value_containers": {}, "exception": None}
    try:
        from rfx._declarations import DECLARATIONS, declaration_key, declaration_variant

        variant = declaration_variant(source)
        result["variant_has_frozen_mesh"] = "_frozen_mesh" in variant.__dict__
    except Exception as exc:
        result["exception"] = _exception(exc)
        return result
    # Keep container measurements even when key evaluation fails.
    try:
        result["key_equal"] = bool(declaration_key(variant) == declaration_key(source))
    except Exception as exc:
        result["key_exception"] = _exception(exc)
    for name, form in DECLARATIONS:
        if form != "value":
            continue
        try:
            value = getattr(source, name, None)
            if isinstance(value, (dict, set, list)):
                result["value_containers"][name] = {
                    "type": type(value).__name__,
                    "same_object": getattr(variant, name, None) is value,
                }
        except Exception as exc:
            result["value_containers"][name] = {"exception": _exception(exc)}
    return result


def _build(scene, builder, source_dx, factor, build, radius):
    row = dict(scene=scene, f=factor, build=build, r=radius.get("r"),
               target_dx=None, outcome="exception", stage="reference_dx",
               exception=None, grid=None, variant_has_frozen_mesh=None)
    source = None
    try:
        if radius.get("exception") is not None:
            row["exception"] = radius["exception"]
            return row
        target_dx = factor * radius["r"]
        row["target_dx"] = target_dx
        if build == "F":
            row["stage"] = "fresh_builder"
            sim = builder(dx=target_dx)
        else:
            row["stage"] = "source_builder"
            source = builder(dx=source_dx)
            if build == "E_frozen":
                row["stage"] = "freeze_mesh"
                frozen_grid = source.freeze_mesh()
                row["frozen_source_grid_dx"] = float(frozen_grid.dx)
            row["source_has_frozen_mesh"] = "_frozen_mesh" in source.__dict__
            row["stage"] = "declaration_variant"
            from rfx._declarations import declaration_variant

            sim = declaration_variant(source, _dx=target_dx)
            row["variant_has_frozen_mesh"] = "_frozen_mesh" in sim.__dict__
        row["stage"] = "grid_summary"
        row["grid"] = _grid_summary(sim)
        row["outcome"] = "grid"
        row["stage"] = "complete"
    except Exception as exc:
        row["exception"] = _exception(exc)
    finally:
        if factor == 1.0 and build != "F":
            row["identity_checks"] = (
                _identity_checks(source) if source is not None else
                {"unavailable": "source was not constructed", "exception": row["exception"]}
            )
    return row


def _compare(e_row, fresh_row):
    # Exact census-style representation comparison, with no tolerance or oracle.
    if e_row["grid"] is None or fresh_row["grid"] is None:
        return {field: {"equal": None, "reason": "E or F grid unavailable"}
                for field in GRID_FIELDS}
    return {field: {"equal": repr(e_row["grid"][field]) == repr(fresh_row["grid"][field]),
                    "E": e_row["grid"][field], "F": fresh_row["grid"][field]}
            for field in GRID_FIELDS}


def _cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _markdown(report):
    lines = ["# dx variant probe", "", "Observations only; conclusion: 리더가 채움", "",
             "E vs F uses exact representation equality; unavailable is not a mismatch.", "",
             "| scene | f | build | outcome or grid summary | E vs F per grid field |",
             "| --- | --- | --- | --- | --- |"]
    for row in report["rows"]:
        outcome = (json.dumps(row["grid"], ensure_ascii=False) if row["grid"] is not None
                   else f"{row['stage']}: {json.dumps(row['exception'], ensure_ascii=False)}")
        outcome += f"; variant_has_frozen_mesh={row['variant_has_frozen_mesh']}"
        comparison = "; ".join(
            f"{field}=" + ("unavailable" if item["equal"] is None else
                           "equal" if item["equal"] else
                           f"differs (E={item['E']!r}, F={item['F']!r})")
            for field, item in row.get("E_vs_F", {}).items()
        ) or "reference F"
        lines.append("| " + " | ".join(map(_cell, (
            row["scene"], row["f"], row["build"], outcome, comparison))) + " |")
    lines += ["", "## f = 1 no-replacement variant checks", "",
              "| scene | source state | checks |", "| --- | --- | --- |"]
    for row in report["rows"]:
        if "identity_checks" in row:
            lines.append("| " + " | ".join(map(_cell, (
                row["scene"], row["build"],
                json.dumps(row["identity_checks"], ensure_ascii=False)))) + " |")
    lines += ["", "## Measurement choices", ""]
    lines.extend(f"- {choice}" for choice in report["choices"])
    return "\n".join(lines) + "\n"


def test_dx_variant_probe():
    report = {"design": "rfx-archive 465889c81: DESIGN-dx-variant.md",
              "conclusion": "리더가 채움", "rows": [], "reference_dx": {},
              "choices": [
                  "Dielectric Box: (5,4,3) to (10,8,6) mm, eps_r=3, census interior Box coordinates.",
                  "One realized twin per scene supplies r for both factors and all builds; no fallback dx.",
                  "Each scene/f/E build has a new source; E_unfrozen is never asked for a grid.",
                  "PZ is copied unchanged in every graded_z builder, including fresh builds.",
                  "dt_pin uses Grid.courant_dt only to define scene inputs, never expected outputs.",
                  "f=1 no-replacement checks run on both E source states after their build attempt.",
                  "Grid fields follow the census, including cpml_axes; comparisons are exact repr equality.",
                  "Missing variants and unavailable comparisons are null, never inferred false.",
                  "Twin failure is recorded for all dependent rows; no scene repair or retry.",
              ]}
    for scene, builder, source_dx in SCENES:
        try:
            twin = builder(dx=source_dx)
            radius = {"r": float(twin._build_realized_grid().dx), "exception": None}
        except Exception as exc:
            radius = {"r": None, "exception": _exception(exc)}
        report["reference_dx"][scene] = radius
        for factor in (1.0, 0.5):
            rows = [_build(scene, builder, source_dx, factor, build, radius)
                    for build in ("E_unfrozen", "E_frozen", "F")]
            for row in rows[:2]:
                row["E_vs_F"] = _compare(row, rows[2])
            report["rows"].extend(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "variant.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    (OUT / "variant.md").write_text(_markdown(report), encoding="utf-8")
    assert (OUT / "variant.json").exists() and (OUT / "variant.md").exists()

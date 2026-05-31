"""Runtime artifact/report/bundle export helpers for :mod:`rfx`.

The functions in this module are intentionally host-side utilities: they
inspect live ``Simulation``-like objects, summarize optional result-like
objects, and write JSON/Markdown/bundle files after simulation setup or run
completion.  They do not import ``Simulation`` and they do not run inside JAX
traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as _np


SCHEMA_VERSION = "rfx-runtime-artifact-v1"
SCENE_SCHEMA_VERSION = "rfx-scene-artifact-v1"
MANIFEST_SCHEMA_VERSION = "rfx-artifact-manifest-v1"
ALLOWED_REPRODUCIBILITY_STATUS = {
    "provenance-only",
    "partial",
    "not-replayable",
    "unknown",
}

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "generated_at",
    "rfx_version",
    "scope",
    "simulation",
    "scene",
    "cad_compat",
    "mesh",
    "preflight",
    "result",
    "visualization",
    "bundle",
    "provenance",
    "reproducibility",
    "limitations",
)

REQUIRED_NESTED = {
    "simulation": ("freq_max", "domain", "mode", "boundary", "cpml_layers", "dx", "mesh_profiles"),
    "scene": ("materials", "geometry", "sources", "ports", "probes"),
    "cad_compat": ("source_type", "status", "entities", "limitations"),
    "mesh": ("status", "grid_shape", "cell_count", "grid_type", "nonuniform", "audit"),
    "preflight": ("status", "issues", "warnings_source"),
    "result": ("status", "result_type", "grid", "time_series", "s_params", "freqs", "snapshots"),
    "visualization": ("outputs", "primary_view", "checks"),
    "bundle": ("files", "manifest_path", "validation_status"),
    "provenance": ("python", "platform", "rfx_version", "cwd", "command"),
    "reproducibility": ("status", "repository", "commit", "worktree_status", "command", "inputs", "limitations"),
}


@dataclass(frozen=True)
class ArtifactBundle:
    """Paths written by :func:`export_artifact_bundle`."""

    root: Path
    report_json: Path
    manifest_json: Path
    report_markdown: Path | None = None
    scene_json: Path | None = None
    geometry_json: Path | None = None
    field_vtk: Path | None = None
    files: tuple[Path, ...] = ()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rfx_version() -> str:
    module = sys.modules.get("rfx")
    version = getattr(module, "__version__", None)
    if isinstance(version, str):
        return version
    try:
        return importlib.metadata.version("rfx")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, _np.generic):
        return value.item()
    try:
        if hasattr(value, "item") and callable(value.item):
            item = value.item()
            if item is not value:
                return _scalar(item)
    except Exception:
        pass
    return value


def _safe_float(value: Any) -> float | None:
    value = _scalar(value)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _jsonable(value: Any, *, array_values: bool = False, max_array_values: int = 16) -> Any:
    """Best-effort conversion to JSON-compatible scalars/lists/dicts.

    Arrays default to compact metadata to avoid accidentally dumping large
    fields.  Small explicit coordinate-like arrays can opt into values.
    """

    value = _scalar(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v, array_values=array_values, max_array_values=max_array_values) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, array_values=array_values, max_array_values=max_array_values) for v in value]
    if _is_array_like(value):
        shape = tuple(int(x) for x in getattr(value, "shape", ()))
        size = int(getattr(value, "size", 0) or 0)
        if array_values and size <= max_array_values:
            try:
                return _np.asarray(value).tolist()
            except Exception:
                pass
        return _array_summary(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "_asdict") and callable(value._asdict):
        try:
            return _jsonable(value._asdict())
        except Exception:
            pass
    # Record object identity without chasing arbitrary private state.
    return {"type": type(value).__name__, "repr": repr(value)}


def _array_summary(value: Any, *, include_peak: bool = False) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    summary: dict[str, Any] = {
        "present": True,
        "shape": [int(x) for x in getattr(value, "shape", ())],
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
        "size": int(getattr(value, "size", 0) or 0),
    }
    if include_peak:
        try:
            arr = _np.asarray(value)
            summary["peak"] = float(_np.max(_np.abs(arr))) if arr.size else 0.0
        except Exception as exc:  # pragma: no cover - depends on result backend
            summary["peak_error"] = str(exc)
    return summary


def _sequence_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False, "count": 0}
    try:
        return {"present": True, "count": len(value)}
    except Exception:
        return {"present": True, "count": None, "type": type(value).__name__}


def _profile_summary(profile: Any) -> dict[str, Any]:
    if profile is None:
        return {"present": False, "count": 0}
    out = _array_summary(profile)
    out["count"] = out.get("shape", [0])[0] if out.get("shape") else out.get("size", 0)
    try:
        arr = _np.asarray(profile, dtype=float)
        if arr.size:
            out.update({
                "min": float(_np.min(arr)),
                "max": float(_np.max(arr)),
                "sum": float(_np.sum(arr)),
            })
    except Exception:
        pass
    return out


def _mesh_profiles(sim: Any) -> dict[str, Any]:
    return {
        "dx_profile": _profile_summary(getattr(sim, "_dx_profile", None)),
        "dy_profile": _profile_summary(getattr(sim, "_dy_profile", None)),
        "dz_profile": _profile_summary(getattr(sim, "_dz_profile", None)),
    }


def _simulation_summary(sim: Any) -> dict[str, Any]:
    return {
        "freq_max": _safe_float(getattr(sim, "_freq_max", None)),
        "domain": _jsonable(getattr(sim, "_domain", None), array_values=True),
        "mode": getattr(sim, "_mode", None),
        "boundary": getattr(sim, "_boundary", None),
        "boundary_spec": _jsonable(getattr(sim, "_boundary_spec", None)),
        "cpml_layers": getattr(sim, "_cpml_layers", None),
        "cpml_kappa_max": _safe_float(getattr(sim, "_cpml_kappa_max", None)),
        "dx": _safe_float(getattr(sim, "_dx", None)),
        "mesh_profiles": _mesh_profiles(sim),
        "precision": getattr(sim, "_precision", None),
        "solver": getattr(sim, "_solver", None),
        "adi_cfl_factor": _safe_float(getattr(sim, "_adi_cfl_factor", None)),
        "periodic_axes": getattr(sim, "_periodic_axes", ""),
        "refinement": _jsonable(getattr(sim, "_refinement", None)),
    }


def _material_summary(material: Any) -> dict[str, Any]:
    return {
        "eps_r": _safe_float(getattr(material, "eps_r", None)),
        "sigma": _safe_float(getattr(material, "sigma", None)),
        "mu_r": _safe_float(getattr(material, "mu_r", None)),
        "chi3": _safe_float(getattr(material, "chi3", None)),
        "debye_poles": _sequence_summary(getattr(material, "debye_poles", None)),
        "lorentz_poles": _sequence_summary(getattr(material, "lorentz_poles", None)),
    }


def _bbox(shape: Any) -> list[list[Any]] | None:
    if not hasattr(shape, "bounding_box"):
        return None
    try:
        lo, hi = shape.bounding_box()
        return [_jsonable(lo, array_values=True), _jsonable(hi, array_values=True)]
    except Exception as exc:  # pragma: no cover - shape-dependent
        return [["error", str(exc)]]


def _entry_mapping(entry: Any) -> dict[str, Any]:
    if hasattr(entry, "_asdict") and callable(entry._asdict):
        return dict(entry._asdict())
    if hasattr(entry, "__dict__"):
        return dict(entry.__dict__)
    return {}


def _entry_summary(entry: Any, *, kind: str, index: int) -> dict[str, Any]:
    data = _entry_mapping(entry)
    shape = data.pop("shape", None)
    out: dict[str, Any] = {
        "id": f"{kind}-{index}",
        "kind": kind,
        "type": type(entry).__name__,
    }
    if shape is not None:
        out.update({
            "shape_type": type(shape).__name__,
            "bounding_box": _bbox(shape),
        })
    for key, value in data.items():
        out[key] = _jsonable(value, array_values=True)
    return out


def _geometry_entities(scene: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for key in ("geometry", "sources", "ports", "probes"):
        for item in scene.get(key, []) or []:
            entities.append({
                "entity_type": key.rstrip("s"),
                "id": item.get("id"),
                "kind": item.get("kind"),
                "type": item.get("shape_type") or item.get("type"),
                "material": item.get("material_name") or item.get("material"),
            })
    return entities


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_scene_artifact(sim: Any, *, include_private: bool = False) -> dict[str, Any]:
    """Build a JSON-serializable artifact of the live native rfx scene.

    The returned mapping is intentionally a summary of runtime state, not a CAD
    export.  It uses ``scene`` as the canonical geometry vocabulary and avoids
    dumping large arrays or arbitrary private object state by default.
    """

    simulation = _simulation_summary(sim)
    materials = {
        str(name): _material_summary(material)
        for name, material in getattr(sim, "_materials", {}).items()
    }
    geometry = [
        _entry_summary(entry, kind="geometry", index=i)
        for i, entry in enumerate(getattr(sim, "_geometry", []) or [])
    ]

    raw_ports = list(getattr(sim, "_ports", []) or [])
    sources = [
        _entry_summary(entry, kind="source", index=i)
        for i, entry in enumerate(raw_ports)
        if getattr(entry, "impedance", None) in (None, 0, 0.0)
    ]
    lumped_ports = [
        _entry_summary(entry, kind="lumped-port", index=i)
        for i, entry in enumerate(raw_ports)
        if getattr(entry, "impedance", None) not in (None, 0, 0.0)
    ]
    probes = [
        _entry_summary(entry, kind="probe", index=i)
        for i, entry in enumerate(getattr(sim, "_probes", []) or [])
    ]

    ports = lumped_ports
    ports.extend(
        _entry_summary(entry, kind="waveguide-port", index=i)
        for i, entry in enumerate(getattr(sim, "_waveguide_ports", []) or [])
    )
    ports.extend(
        _entry_summary(entry, kind="msl-port", index=i)
        for i, entry in enumerate(getattr(sim, "_msl_ports", []) or [])
    )
    ports.extend(
        _entry_summary(entry, kind="coaxial-port", index=i)
        for i, entry in enumerate(getattr(sim, "_coaxial_ports", []) or [])
    )
    ports.extend(
        _entry_summary(entry, kind="floquet-port", index=i)
        for i, entry in enumerate(getattr(sim, "_floquet_ports", []) or [])
    )

    scene: dict[str, Any] = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "simulation": simulation,
        # Flat aliases make the scene artifact convenient for grep/review and
        # preserve the older lightweight geometry-export ergonomics.
        "freq_max": simulation["freq_max"],
        "domain": simulation["domain"],
        "mode": simulation["mode"],
        "boundary": simulation["boundary"],
        "cpml_layers": simulation["cpml_layers"],
        "dx": simulation["dx"],
        "mesh_profiles": simulation["mesh_profiles"],
        "materials": materials,
        "geometry": geometry,
        "sources": sources,
        "ports": ports,
        "probes": probes,
        "monitors": {
            "dft_planes": [
                _entry_summary(entry, kind="dft-plane", index=i)
                for i, entry in enumerate(getattr(sim, "_dft_planes", []) or [])
            ],
            "flux_monitors": [
                _entry_summary(entry, kind="flux-monitor", index=i)
                for i, entry in enumerate(getattr(sim, "_flux_monitors", []) or [])
            ],
            "ntff": _jsonable(getattr(sim, "_ntff", None), array_values=True),
            "tfsf": _jsonable(getattr(sim, "_tfsf", None), array_values=True),
        },
    }
    if include_private:
        scene["private_summary"] = {
            "thin_conductors": _sequence_summary(getattr(sim, "_thin_conductors", None)),
            "lumped_rlc": _sequence_summary(getattr(sim, "_lumped_rlc", None)),
            "coaxial_terminations": _sequence_summary(getattr(sim, "_coaxial_terminations", None)),
        }
    return _jsonable(scene)


def _grid_shape(grid: Any) -> list[int] | None:
    shape = getattr(grid, "shape", None)
    if shape is not None:
        try:
            return [int(x) for x in shape]
        except Exception:
            pass
    dims = [getattr(grid, axis, None) for axis in ("nx", "ny", "nz")]
    if all(dim is not None for dim in dims):
        return [int(dim) for dim in dims]
    return None


def _mesh_summary(sim: Any, *, n_steps: int | None, available_memory_gb: float | None) -> dict[str, Any]:
    profiles = _mesh_profiles(sim)
    nonuniform = any(section.get("present") for section in profiles.values())
    mesh: dict[str, Any] = {
        "status": "unavailable",
        "grid_shape": None,
        "cell_count": None,
        "grid_type": "nonuniform" if nonuniform else "uniform",
        "nonuniform": bool(nonuniform),
        "dx": _safe_float(getattr(sim, "_dx", None)),
        "dt": None,
        "profiles": profiles,
        "audit": {"status": "not-run"},
    }

    try:
        grid = sim._build_nonuniform_grid() if nonuniform else sim._build_grid()
        shape = _grid_shape(grid)
        cell_count = None
        if shape:
            cell_count = int(shape[0] * shape[1] * shape[2])
        mesh.update({
            "status": "available",
            "grid_shape": shape,
            "cell_count": cell_count,
            "grid_type": type(grid).__name__,
            "dx": _safe_float(getattr(grid, "dx", getattr(sim, "_dx", None))),
            "dt": _safe_float(getattr(grid, "dt", None)),
        })
    except Exception as exc:
        mesh["error"] = f"{type(exc).__name__}: {exc}"

    if hasattr(sim, "mesh_intelligence_report"):
        try:
            kwargs: dict[str, Any] = {"available_memory_gb": available_memory_gb}
            if n_steps is not None:
                kwargs["n_steps"] = n_steps
            audit = sim.mesh_intelligence_report(**kwargs)
            mesh["audit"] = _jsonable(audit.to_dict() if hasattr(audit, "to_dict") else audit)
            mesh["audit"]["status"] = "available"
        except Exception as exc:
            mesh["audit"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return mesh


def _preflight_summary(sim: Any, *, n_steps: int | None, available_memory_gb: float | None) -> dict[str, Any]:
    if not hasattr(sim, "preflight"):
        return {"status": "unavailable", "issues": [], "warnings_source": "Simulation.preflight missing"}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            issues = sim.preflight(
                strict=False,
                check_ntff=True,
                check_resolution=True,
                check_ad_memory=bool(n_steps),
                n_steps_for_memory=n_steps,
                available_memory_gb=available_memory_gb,
            )
        issues_list = [str(issue) for issue in (issues or [])]
        return {
            "status": "passed" if not issues_list else "issues",
            "issues": issues_list,
            "warnings_source": "Simulation.preflight(strict=False) captured host-side",
            "stdout": [line for line in stdout.getvalue().splitlines() if line.strip()],
        }
    except Exception as exc:
        return {
            "status": "error",
            "issues": [f"{type(exc).__name__}: {exc}"],
            "warnings_source": "Simulation.preflight raised",
            "stdout": [line for line in stdout.getvalue().splitlines() if line.strip()],
        }


def _grid_result_summary(grid: Any) -> dict[str, Any] | None:
    if grid is None:
        return None
    return {
        "type": type(grid).__name__,
        "shape": _grid_shape(grid),
        "dx": _safe_float(getattr(grid, "dx", None)),
        "dt": _safe_float(getattr(grid, "dt", None)),
    }


def _freq_summary(freqs: Any) -> dict[str, Any] | None:
    if freqs is None:
        return None
    out = _array_summary(freqs)
    try:
        arr = _np.asarray(freqs, dtype=float)
        out.update({
            "count": int(arr.size),
            "min": float(_np.min(arr)) if arr.size else None,
            "max": float(_np.max(arr)) if arr.size else None,
        })
    except Exception:
        pass
    return out


def _dict_array_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _array_summary(v) if _is_array_like(v) else _jsonable(v) for k, v in value.items()}
    return _jsonable(value)


def _result_summary(result: Any) -> dict[str, Any]:
    if result is None:
        return {
            "status": "not-provided",
            "result_type": None,
            "grid": None,
            "time_series": None,
            "s_params": None,
            "freqs": None,
            "snapshots": None,
            "ntff_data": None,
            "ntff_box": None,
            "dft_planes": None,
            "flux_monitors": None,
            "waveguide_ports": None,
            "waveguide_sparams": None,
            "dt": None,
            "freq_range": None,
        }

    return {
        "status": "provided",
        "result_type": type(result).__name__,
        "grid": _grid_result_summary(getattr(result, "grid", None)),
        "time_series": _array_summary(getattr(result, "time_series", None), include_peak=True),
        "s_params": _array_summary(getattr(result, "s_params", None), include_peak=True) if getattr(result, "s_params", None) is not None else None,
        "freqs": _freq_summary(getattr(result, "freqs", None)),
        "snapshots": _dict_array_summary(getattr(result, "snapshots", None)),
        "ntff_data": _jsonable(getattr(result, "ntff_data", None)),
        "ntff_box": _jsonable(getattr(result, "ntff_box", None)),
        "dft_planes": _dict_array_summary(getattr(result, "dft_planes", None)),
        "flux_monitors": _dict_array_summary(getattr(result, "flux_monitors", None)),
        "waveguide_ports": _jsonable(getattr(result, "waveguide_ports", None)),
        "waveguide_sparams": _dict_array_summary(getattr(result, "waveguide_sparams", None)),
        "dt": _safe_float(getattr(result, "dt", None)),
        "freq_range": _jsonable(getattr(result, "freq_range", None), array_values=True),
    }


def _git(args: list[str], *, cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _provenance(extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    cwd = Path.cwd()
    version = _rfx_version()
    repo = _git(["rev-parse", "--show-toplevel"], cwd=cwd)
    commit = _git(["rev-parse", "HEAD"], cwd=cwd)
    status = _git(["status", "--short"], cwd=cwd)
    command = list(sys.argv)
    inputs = []
    if extra and isinstance(extra.get("inputs"), list):
        inputs = _jsonable(extra["inputs"])
    prov = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rfx_version": version,
        "cwd": str(cwd),
        "command": command,
        "generated_by": "rfx.artifacts",
    }
    repro = {
        "status": "provenance-only",
        "repository": repo,
        "commit": commit,
        "worktree_status": status if status is not None else "unknown",
        "command": command,
        "inputs": inputs,
        "limitations": [
            "Artifacts do not fully replay a simulation without user code/config",
            "Deterministic replay is not claimed by rfx runtime artifact v1",
        ],
    }
    return prov, repro


def build_runtime_report(
    sim: Any,
    result: Any = None,
    *,
    extra: dict[str, Any] | None = None,
    n_steps: int | None = None,
    available_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Build a complete runtime artifact report for a live simulation."""

    scene = build_scene_artifact(sim)
    provenance, reproducibility = _provenance(extra)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "rfx_version": provenance["rfx_version"],
        "scope": "runtime-export",
        "simulation": scene["simulation"],
        "scene": scene,
        "cad_compat": {
            "source_type": "native-rfx-scene",
            "status": "not-cad-import",
            "entities": _geometry_entities(scene),
            "limitations": ["No STEP/STL/GDS/Gerber import is performed by this feature"],
        },
        "mesh": _mesh_summary(sim, n_steps=n_steps, available_memory_gb=available_memory_gb),
        "preflight": _preflight_summary(sim, n_steps=n_steps, available_memory_gb=available_memory_gb),
        "result": _result_summary(result),
        "visualization": {
            "outputs": [],
            "primary_view": None,
            "checks": ["field VTK is opt-in and requires usable state/grid inputs"],
        },
        "bundle": {
            "files": [],
            "manifest_path": None,
            "validation_status": "not-exported",
        },
        "provenance": provenance,
        "reproducibility": reproducibility,
        "limitations": [
            "No CAD import or external CAD parser is implemented in this feature",
            "No GUI/AppCSXCAD equivalent is implemented",
            "No body-fitted meshing or solver-loop rewrite is implemented",
            "Artifacts are provenance/report bundles and do not provide deterministic replay in v1",
        ],
    }
    if extra:
        report["extra"] = _jsonable(extra)
    return _jsonable(report)


# ---------------------------------------------------------------------------
# Rendering and validation
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True, default=str) + "`"
    return str(value)


def render_artifact_markdown(report: dict[str, Any]) -> str:
    """Render a concise human-readable artifact review report."""

    sim = report.get("simulation", {}) if isinstance(report.get("simulation"), dict) else {}
    mesh = report.get("mesh", {}) if isinstance(report.get("mesh"), dict) else {}
    preflight = report.get("preflight", {}) if isinstance(report.get("preflight"), dict) else {}
    result = report.get("result", {}) if isinstance(report.get("result"), dict) else {}
    cad = report.get("cad_compat", {}) if isinstance(report.get("cad_compat"), dict) else {}
    viz = report.get("visualization", {}) if isinstance(report.get("visualization"), dict) else {}
    bundle = report.get("bundle", {}) if isinstance(report.get("bundle"), dict) else {}
    repro = report.get("reproducibility", {}) if isinstance(report.get("reproducibility"), dict) else {}

    lines = [
        "# rfx Runtime Artifact Report",
        "",
        f"Schema: `{report.get('schema_version', 'unknown')}`",
        f"Generated: `{report.get('generated_at', 'unknown')}`",
        f"Scope: `{report.get('scope', 'unknown')}`",
        "",
        "## Simulation",
        "",
        f"- Frequency max: {_fmt(sim.get('freq_max'))}",
        f"- Domain: {_fmt(sim.get('domain'))}",
        f"- Mode/boundary: {_fmt(sim.get('mode'))} / {_fmt(sim.get('boundary'))}",
        f"- dx / CPML layers: {_fmt(sim.get('dx'))} / {_fmt(sim.get('cpml_layers'))}",
        "",
        "## Native Scene / CAD Bridge",
        "",
        f"- Materials: {len(report.get('scene', {}).get('materials', {}) if isinstance(report.get('scene'), dict) else {})}",
        f"- Geometry entries: {len(report.get('scene', {}).get('geometry', []) if isinstance(report.get('scene'), dict) else [])}",
        f"- CAD compatibility status: `{cad.get('status')}` from `{cad.get('source_type')}`.",
        "- No CAD import is performed; this report summarizes the native rfx runtime scene.",
        "",
        "## Mesh / Preflight",
        "",
        f"- Mesh status: `{mesh.get('status')}`; type: `{mesh.get('grid_type')}`; nonuniform: `{mesh.get('nonuniform')}`",
        f"- Grid shape / cells: {_fmt(mesh.get('grid_shape'))} / {_fmt(mesh.get('cell_count'))}",
        f"- Preflight status: `{preflight.get('status')}`; issues: {len(preflight.get('issues') or [])}",
        "",
        "## Result",
        "",
        f"- Result status/type: `{result.get('status')}` / `{result.get('result_type')}`",
        f"- Time series: {_fmt(result.get('time_series'))}",
        f"- S-parameters: {_fmt(result.get('s_params'))}",
        f"- Frequencies: {_fmt(result.get('freqs'))}",
        "",
        "## Visualization",
        "",
        f"- Primary view: {_fmt(viz.get('primary_view'))}",
        f"- Outputs: {_fmt(viz.get('outputs'))}",
        "",
        "## Bundle",
        "",
        f"- Validation status: `{bundle.get('validation_status')}`",
        f"- Manifest: {_fmt(bundle.get('manifest_path'))}",
        f"- Files: {len(bundle.get('files') or [])}",
        "",
        "## Reproducibility",
        "",
        f"- Status: `{repro.get('status')}`",
        f"- Repository: {_fmt(repro.get('repository'))}",
        f"- Commit: {_fmt(repro.get('commit'))}",
        "- Deterministic replay is not claimed by runtime artifact v1.",
        "",
        "## Limitations",
        "",
    ]
    for item in report.get("limitations", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def validate_artifact_report(report: dict[str, Any], *, bundle_root: str | Path | None = None) -> list[str]:
    """Return validation errors for an artifact report.

    The validator is intentionally lightweight and dependency-free; callers can
    treat any returned message as a report failure.
    """

    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a dict"]

    for field in REQUIRED_TOP_LEVEL:
        if field not in report:
            errors.append(f"missing top-level field: {field}")

    for section, required in REQUIRED_NESTED.items():
        value = report.get(section)
        if not isinstance(value, dict):
            errors.append(f"section {section!r} must be an object")
            continue
        for field in required:
            if field not in value:
                errors.append(f"missing nested field: {section}.{field}")

    if "limitations" in report and not isinstance(report.get("limitations"), list):
        errors.append("limitations must be a list")
    for section in ("visualization", "bundle"):
        value = report.get(section)
        if isinstance(value, dict) and not isinstance(value.get("files" if section == "bundle" else "outputs"), list):
            errors.append(f"{section}.{'files' if section == 'bundle' else 'outputs'} must be a list")

    cad = report.get("cad_compat", {})
    if isinstance(cad, dict):
        if cad.get("source_type") != "native-rfx-scene":
            errors.append("cad_compat.source_type must be 'native-rfx-scene' in v1")
        if cad.get("status") != "not-cad-import":
            errors.append("cad_compat.status must be 'not-cad-import' in v1")
        source_text = f"{cad.get('source_type', '')} {cad.get('status', '')}".lower()
        if any(token in source_text for token in ("step", "stl", "gds", "gerber", "excellon", "imported")):
            errors.append("cad_compat must not imply external CAD import support in v1")

    repro = report.get("reproducibility", {})
    if isinstance(repro, dict):
        status = repro.get("status")
        if status not in ALLOWED_REPRODUCIBILITY_STATUS:
            errors.append(
                "reproducibility.status must be one of "
                + ", ".join(sorted(ALLOWED_REPRODUCIBILITY_STATUS))
            )
        if status == "replayable":
            errors.append("reproducibility.status='replayable' is not allowed in v1")
        if not isinstance(repro.get("inputs"), list):
            errors.append("reproducibility.inputs must be a list")
        if not isinstance(repro.get("limitations"), list):
            errors.append("reproducibility.limitations must be a list")

    if bundle_root is not None:
        root = Path(bundle_root)
        bundle = report.get("bundle", {})
        if isinstance(bundle, dict):
            for item in bundle.get("files", []) or []:
                rel = item.get("path") if isinstance(item, dict) else item
                if not rel:
                    errors.append("bundle.files entry missing path")
                    continue
                rel_path = Path(str(rel))
                if rel_path.is_absolute():
                    errors.append(f"bundle file path must be relative: {rel}")
                    continue
                if not (root / rel_path).exists():
                    errors.append(f"declared bundle file missing: {rel}")
    return errors


# ---------------------------------------------------------------------------
# Bundle export
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(root: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for item in files:
        rel = Path(str(item["path"]))
        path = root / rel
        if not path.exists() or rel.name == "manifest.json":
            continue
        entries.append({
            "path": rel.as_posix(),
            "role": item.get("role"),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "files": entries,
    }


def export_artifact_bundle(
    path: str | Path,
    sim: Any,
    result: Any = None,
    *,
    include_markdown: bool = True,
    include_scene: bool = True,
    include_geometry_json: bool = True,
    include_field_vtk: bool = False,
    field_state: Any = None,
    field_grid: Any = None,
    extra: dict[str, Any] | None = None,
) -> ArtifactBundle:
    """Write a runtime artifact bundle directory and return its paths."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    markdown_path = root / "report.md" if include_markdown else None
    scene_path = root / "scene.json" if include_scene else None
    geometry_path = root / "geometry.json" if include_geometry_json else None
    manifest_path = root / "manifest.json"
    vtk_path: Path | None = None

    report = build_runtime_report(sim, result=result, extra=extra)
    declared_files: list[dict[str, Any]] = [{"path": "report.json", "role": "report-json"}]

    if scene_path is not None:
        _write_json(scene_path, report["scene"])
        declared_files.append({"path": "scene.json", "role": "scene-json"})

    if geometry_path is not None:
        from rfx.io import export_geometry_json  # local import avoids top-level cycles

        export_geometry_json(geometry_path, sim)
        declared_files.append({"path": "geometry.json", "role": "legacy-geometry-json"})

    if include_field_vtk:
        state = field_state if field_state is not None else getattr(result, "state", None)
        grid = field_grid if field_grid is not None else getattr(result, "grid", None)
        if state is not None and grid is not None:
            from rfx.visualize3d import save_field_vtk  # local import; existing suffix behavior

            written = Path(save_field_vtk(state, grid, str(root / "fields")))
            vtk_path = written if written.is_absolute() else root / written.name
            declared_files.append({"path": vtk_path.relative_to(root).as_posix(), "role": "field-vtk"})
            report["visualization"]["outputs"].append({
                "path": vtk_path.relative_to(root).as_posix(),
                "role": "field-vtk",
                "status": "written",
            })
            report["visualization"]["primary_view"] = vtk_path.relative_to(root).as_posix()
        else:
            report["visualization"]["outputs"].append({
                "role": "field-vtk",
                "status": "missing-input",
                "reason": "include_field_vtk=True requires field_state/field_grid or result.state/result.grid",
            })
    else:
        report["visualization"]["outputs"].append({"role": "field-vtk", "status": "not-requested"})

    if markdown_path is not None:
        declared_files.append({"path": "report.md", "role": "report-markdown"})
    declared_files.append({"path": "manifest.json", "role": "manifest-json"})

    report["bundle"] = {
        "files": declared_files,
        "manifest_path": "manifest.json",
        "validation_status": "pending",
    }
    _write_json(report_path, report)
    if markdown_path is not None:
        markdown_path.write_text(render_artifact_markdown(report))
    _write_json(manifest_path, _manifest(root, declared_files))

    validation_errors = validate_artifact_report(report, bundle_root=root)
    report["bundle"]["validation_status"] = "passed" if not validation_errors else "failed"
    if validation_errors:
        report["bundle"]["validation_errors"] = validation_errors
    _write_json(report_path, report)
    if markdown_path is not None:
        markdown_path.write_text(render_artifact_markdown(report))
    _write_json(manifest_path, _manifest(root, declared_files))

    files = tuple(
        root / str(item["path"])
        for item in declared_files
        if (root / str(item["path"])).exists()
    )
    return ArtifactBundle(
        root=root,
        report_json=report_path,
        manifest_json=manifest_path,
        report_markdown=markdown_path if markdown_path and markdown_path.exists() else None,
        scene_json=scene_path if scene_path and scene_path.exists() else None,
        geometry_json=geometry_path if geometry_path and geometry_path.exists() else None,
        field_vtk=vtk_path if vtk_path and vtk_path.exists() else None,
        files=files,
    )

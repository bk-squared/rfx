"""Unified mesh-planning facade for automation and preflight reports.

The planner intentionally reuses existing mesh derivation and validation APIs:
``auto_configure`` for geometry/frequency setup, ``Simulation.preflight`` /
``preflight_sparameters`` for support checks, and
``Simulation.mesh_intelligence_report`` for concrete simulation sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rfx.auto_config import C0, SimConfig, analyze_features, auto_configure
from rfx.api._ad_memory import _format_memory_gb
from rfx.api._spec import MeshIntelligenceReport
from rfx.core.jax_utils import is_tracer

SCHEMA_VERSION = "mesh-plan/v1"


def mesh_intelligence_report(
    sim,
    *,
    n_steps: int | None = None,
    checkpoint_every: int | None = None,
    checkpoint_segments: int | None = None,
    available_memory_gb: float | None = None,
    n_warmup: int = 0,
    check_ntff: bool = True,
    check_resolution: bool = True,
) -> MeshIntelligenceReport:
    """Return a consolidated mesh-quality and memory-planning report.

        The report is intentionally advisory: it reuses ``preflight()``
        for physics/geometry warnings, reuses ``estimate_ad_memory()``
        when ``n_steps`` is provided, and adds a uniform-fine comparator
        that estimates how many cells a globally fine mesh would need if
        it used the minimum cell size present in any non-uniform profile.

        This is the Stage-1 production-near "subgrid-like" planning
        surface: it helps users decide whether existing non-uniform mesh
        plus segmented checkpointing is enough before attempting
        research-only true subgridding.

        ``checkpoint_every`` and ``checkpoint_segments`` are forwarded to the
        same AD estimator used by :meth:`plan_ad_memory`; they are mutually
        exclusive. ``n_warmup`` is forwarded as reverse-mode tape metadata
        only, matching ``estimate_ad_memory``.
        """
    import contextlib
    import io

    if any(
        p is not None and is_tracer(p)
        for p in (sim._dx_profile, sim._dy_profile, sim._dz_profile)
    ):
        raise ValueError(
            "mesh_intelligence_report requires concrete mesh profiles; "
            "tracer-valued mesh-as-design-variable profiles cannot be "
            "summarized host-side."
        )

    dx = sim._dx or (C0 / sim._freq_max / 20.0)

    # #696: one shape source, shared with estimate_ad_memory — the
    # grid the solve will actually build. This method carried its own
    # copy of the same ceil(extent/dx)+1+2*cpml re-derivation, so the
    # report's cell count and its AD estimate could describe two
    # different grids, neither of them the one that runs.
    _accounting = sim._ad_memory_static_accounting()
    grid_shape = (
        _accounting["nx"], _accounting["ny"], _accounting["nz"],
    )
    cells = int(_accounting["cells"])

    axis_min = [
        float(np.min(sim._dx_profile)) if sim._dx_profile is not None else dx,
        float(np.min(sim._dy_profile)) if sim._dy_profile is not None else dx,
        float(np.min(sim._dz_profile)) if sim._dz_profile is not None else dx,
    ]
    min_cell_size = min(axis_min)
    uniform_fine_shape = tuple(
        int(math.ceil(extent / min_cell_size)) + 1 + 2 * sim._cpml_layers
        for extent in sim._domain
    )
    uniform_fine_cells = int(
        uniform_fine_shape[0] * uniform_fine_shape[1] * uniform_fine_shape[2]
    )
    cell_savings_factor = (
        float(uniform_fine_cells / cells) if cells > 0 else float("inf")
    )

    # preflight() prints a summary by design; suppress it so the
    # report method remains a pure information-returning API.
    with contextlib.redirect_stdout(io.StringIO()):
        preflight_issues = tuple(
            sim.preflight(
                strict=False,
                check_ntff=check_ntff,
                check_resolution=check_resolution,
                check_ad_memory=False,
            )
        )

    ad_memory = None
    if n_steps is not None:
        ad_memory = sim.estimate_ad_memory(
            n_steps,
            available_memory_gb=available_memory_gb,
            checkpoint_every=checkpoint_every,
            checkpoint_segments=checkpoint_segments,
            n_warmup=n_warmup,
        )

    uses_nonuniform = any(
        p is not None
        for p in (sim._dx_profile, sim._dy_profile, sim._dz_profile)
    )
    recommendation_parts: list[str] = []
    if preflight_issues:
        recommendation_parts.append(
            f"resolve {len(preflight_issues)} preflight issue(s) before "
            "trusting physics results"
        )
    elif uses_nonuniform and cell_savings_factor >= 2.0:
        recommendation_parts.append(
            f"non-uniform mesh is useful here: ~{cell_savings_factor:.1f}x "
            "fewer cells than a uniform mesh at the finest spacing"
        )
    elif uses_nonuniform:
        recommendation_parts.append(
            "non-uniform mesh gives limited cell savings; verify that "
            "the refinement profile is worth the added validation burden"
        )
    else:
        recommendation_parts.append(
            "uniform mesh: use preflight/memory estimates as baseline; "
            "consider non-uniform profiles before research-only subgrid"
        )

    if ad_memory is not None:
        if ad_memory.warning:
            recommendation_parts.append(ad_memory.warning)
        elif (checkpoint_every or checkpoint_segments) and ad_memory.ad_segmented_gb is not None:
            recommendation_parts.append(
                f"use segmented AD estimate ({_format_memory_gb(ad_memory.ad_segmented_gb)}) "
                "rather than the legacy step-checkpoint heuristic"
            )
        else:
            recommendation_parts.append(
                "estimated full reverse-mode AD memory is "
                f"{_format_memory_gb(ad_memory.ad_full_gb)}"
            )

    return MeshIntelligenceReport(
        grid_shape=grid_shape,
        cells=cells,
        uniform_fine_shape=uniform_fine_shape,
        uniform_fine_cells=uniform_fine_cells,
        cell_savings_factor=cell_savings_factor,
        min_cell_size=float(min_cell_size),
        nominal_dx=float(dx),
        uses_nonuniform=uses_nonuniform,
        preflight_issues=preflight_issues,
        ad_memory=ad_memory,
        recommendation="; ".join(recommendation_parts) + ".",
    )

def _jsonable(value: Any) -> Any:
    """Convert NumPy/scalar containers into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _artifact_declarations(
    artifact_root: str | Path | None,
) -> dict[str, dict[str, str | None]]:
    root = None if artifact_root is None else str(Path(artifact_root))
    return {
        "scene": {
            "path": None,
            "status": "not_claimed",
        },
        "mesh_plan": {
            "path": None if root is None else str(Path(root) / "mesh_plan.json"),
            "status": "not_requested" if root is None else "declared_only",
        },
        "report": {
            "path": None if root is None else str(Path(root) / "report.md"),
            "status": "not_requested" if root is None else "declared_only",
        },
        "replay": {
            "path": None,
            "status": "not_claimed",
        },
    }


def _memory_payload(
    *,
    source: str,
    cells: int | None = None,
    uniform_fine_cells: int | None = None,
    cell_savings_factor: float | None = None,
    estimated_mb: float | None = None,
    max_memory_mb: float | None = None,
    ad_memory: Any | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Return a schema-stable memory block for both planner constructors."""
    return {
        "source": source,
        "cells": None if cells is None else int(cells),
        "uniform_fine_cells": (
            None if uniform_fine_cells is None else int(uniform_fine_cells)
        ),
        "cell_savings_factor": (
            None if cell_savings_factor is None else float(cell_savings_factor)
        ),
        "estimated_mb": None if estimated_mb is None else float(estimated_mb),
        "max_memory_mb": None if max_memory_mb is None else float(max_memory_mb),
        "ad_memory": None if ad_memory is None else ad_memory.to_dict(),
        "status": status,
    }


@dataclass(frozen=True)
class MeshPlan:
    """Inspectable mesh-planning result built from existing rfx facts.

    ``MeshPlan`` is an advisory artifact. It serializes the derived mesh,
    CFL basis, memory estimate, support rows, and intended artifact paths
    without writing files or claiming deterministic solver replay.
    """

    schema_version: str
    plan_source: str
    freq_range: tuple[float | None, float]
    accuracy: str | None
    boundary: str
    cell_sizes: dict[str, Any]
    grid_shape: tuple[int, int, int]
    domain: tuple[float, float, float]
    margin: float | None
    cfl: dict[str, Any]
    absorber: dict[str, Any]
    resolution_basis: list[dict[str, Any]]
    support_checks: list[dict[str, Any]]
    memory: dict[str, Any]
    artifact_declarations: dict[str, dict[str, Any]]
    warnings: tuple[str, ...] = ()
    recommendation: str = ""
    _sim_config: SimConfig | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return _jsonable(
            {
                "schema_version": self.schema_version,
                "plan_source": self.plan_source,
                "freq_range": self.freq_range,
                "accuracy": self.accuracy,
                "boundary": self.boundary,
                "cell_sizes": self.cell_sizes,
                "grid_shape": self.grid_shape,
                "domain": self.domain,
                "margin": self.margin,
                "cfl": self.cfl,
                "absorber": self.absorber,
                "resolution_basis": self.resolution_basis,
                "support_checks": self.support_checks,
                "memory": self.memory,
                "artifact_declarations": self.artifact_declarations,
                "warnings": self.warnings,
                "recommendation": self.recommendation,
            }
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the plan for audit artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)

    def to_markdown(self) -> str:
        """Render a compact human-readable planning report."""
        f_min, f_max = self.freq_range
        freq_text = "unknown" if f_min is None else f"{f_min:.6g} Hz"
        freq_text = f"{freq_text} – {f_max:.6g} Hz"
        lines = [
            "# Mesh Plan",
            "",
            f"- Schema: `{self.schema_version}`",
            f"- Frequency range: {freq_text}",
            f"- Plan source: `{self.plan_source}`",
            f"- Accuracy: `{self.accuracy}`" if self.accuracy is not None else "- Accuracy: unknown",
            f"- Boundary: `{self.boundary}`",
            (
                "- Grid shape: "
                f"{self.grid_shape[0]} × {self.grid_shape[1]} × {self.grid_shape[2]}"
            ),
            (
                "- Domain: "
                f"{self.domain[0]:.6g} × {self.domain[1]:.6g} × "
                f"{self.domain[2]:.6g} m"
            ),
            (
                f"- Base dx: {self.cell_sizes.get('dx'):.6g} m"
                if self.cell_sizes.get("dx") is not None
                else "- Base dx: unavailable"
            ),
            (
                f"- Timestep: {self.cfl.get('dt'):.6g} s"
                if self.cfl.get("dt") is not None
                else "- Timestep: unavailable"
            ),
            (
                f"- Memory estimate: {self.memory.get('estimated_mb'):.3g} MB"
                if self.memory.get("estimated_mb") is not None
                else "- Memory estimate: unavailable"
            ),
            f"- Recommendation: {self.recommendation}",
            "",
            "## Support checks",
        ]
        if self.support_checks:
            for row in self.support_checks:
                lines.append(
                    f"- `{row.get('status')}` {row.get('source')}: "
                    f"{row.get('message')}"
                )
        else:
            lines.append("- No support checks recorded.")
        if self.resolution_basis:
            lines.extend(["", "## Resolution basis"])
            for row in self.resolution_basis:
                lines.append(
                    f"- `{row.get('status')}` {row.get('kind')}: "
                    f"{row.get('message')}"
                )
        if self.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)
        lines.extend(["", "## Artifact declarations"])
        for name, entry in self.artifact_declarations.items():
            path = entry.get("path") or "unassigned"
            lines.append(f"- `{name}`: {entry.get('status')} ({path})")
        return "\n".join(lines)

    def to_sim_config(self) -> SimConfig:
        """Return the underlying ``SimConfig`` when built by ``plan_mesh``."""
        if self._sim_config is None:
            raise ValueError(
                "to_sim_config() is only available for geometry-derived mesh plans"
            )
        return self._sim_config

    def to_sim_kwargs(self) -> dict[str, Any]:
        """Return ``Simulation`` constructor kwargs when built by ``plan_mesh``."""
        return self.to_sim_config().to_sim_kwargs()


def _cell_sizes_from_config(config: SimConfig) -> dict[str, Any]:
    dz_min = (
        float(np.min(config.dz_profile))
        if config.dz_profile is not None
        else config.dx
    )
    dz_max = (
        float(np.max(config.dz_profile))
        if config.dz_profile is not None
        else config.dx
    )
    return {
        "nominal_dx": float(config.dx),
        "dx": float(config.dx),
        "dy": float(config.dx),
        "dx_min": float(config.dx),
        "dx_max": float(config.dx),
        "dy_min": float(config.dx),
        "dy_max": float(config.dx),
        "dz_min": float(dz_min),
        "dz_max": float(dz_max),
        "profiles_present": {
            "x": False,
            "y": False,
            "z": config.dz_profile is not None,
        },
    }


def _cfl_from_config(config: SimConfig) -> dict[str, Any]:
    dz_min = (
        float(np.min(config.dz_profile))
        if config.dz_profile is not None
        else config.dx
    )
    limiting_axis = "z" if dz_min < config.dx else "uniform"
    return {
        "dt": float(config.dt),
        "formula": "0.99 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz_min^2))",
        "basis": "auto_configure",
        "limiting_axis": limiting_axis,
    }


def _absorber_from_config(config: SimConfig) -> dict[str, Any]:
    thickness = float(config.cpml_layers * config.dx)
    return {
        "boundary": config.boundary,
        "cpml_layers": int(config.cpml_layers),
        "physical_thickness": {"x": thickness, "y": thickness, "z": thickness},
        "disabled_faces": (
            []
            if config.cpml_layers
            else ["x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"]
        ),
    }


def _resolution_basis_from_config(
    config: SimConfig,
    geometry: list,
    materials: dict[str, Any],
) -> list[dict[str, Any]]:
    features = analyze_features(geometry, materials)
    lambda_min = C0 / config.freq_range[1]
    lambda_medium = lambda_min / float(np.sqrt(features.max_eps_r))
    rows = [
        {
            "source": "auto_configure",
            "kind": "wavelength",
            "status": "ok",
            "message": "Cell size derived from highest frequency and maximum eps_r.",
            "lambda_min": lambda_min,
            "max_eps_r": float(features.max_eps_r),
            "lambda_min_medium": lambda_medium,
            "cells_per_wavelength": float(config.cells_per_wavelength),
        }
    ]
    if geometry:
        cells = (
            float(features.min_thickness / config.dx)
            if features.min_thickness > 0
            else None
        )
        rows.append(
            {
                "source": "auto_configure",
                "kind": "feature",
                "status": "ok" if cells is None or cells >= 2 else "warn",
                "message": (
                    "Thinnest feature resolution from auto_configure feature analysis."
                ),
                "min_thickness": float(features.min_thickness),
                "cells": cells,
            }
        )
    if config.dz_profile is not None:
        rows.append(
            {
                "source": "auto_configure",
                "kind": "nonuniform_z",
                "status": "info",
                "message": "Thin z-features activated an auto-generated dz_profile.",
                "dz_cells": int(len(config.dz_profile)),
                "dz_min": float(np.min(config.dz_profile)),
                "dz_max": float(np.max(config.dz_profile)),
            }
        )
    return rows


def plan_mesh(
    geometry: list,
    freq_range: tuple[float, float],
    materials: dict | None = None,
    accuracy: str = "standard",
    *,
    boundary: str = "cpml",
    dx_override: float | None = None,
    margin_override: float | None = None,
    n_steps_override: int | None = None,
    max_memory_mb: float | None = None,
    artifact_root: str | Path | None = None,
) -> MeshPlan:
    """Plan a mesh from geometry and frequency inputs using ``auto_configure``.

    This function is a facade around ``auto_configure``; it does not duplicate
    the mesh preset math. The returned ``MeshPlan`` adds structured audit rows,
    artifact declarations, and serialization helpers. ``artifact_root`` only
    declares intended paths; it never creates directories or writes files.
    """
    material_map = {} if materials is None else materials
    config = auto_configure(
        geometry,
        freq_range,
        materials=material_map,
        accuracy=accuracy,
        boundary=boundary,
        dx_override=dx_override,
        margin_override=margin_override,
        n_steps_override=n_steps_override,
        max_memory_mb=max_memory_mb,
    )
    memory_status = "unknown"
    if max_memory_mb is not None:
        memory_status = (
            "ok" if config.estimated_memory_mb <= max_memory_mb * 1.2 else "warn"
        )
    warnings = tuple(config.warnings)
    grid_shape = config.grid_shape
    cells = int(grid_shape[0] * grid_shape[1] * grid_shape[2])
    return MeshPlan(
        schema_version=SCHEMA_VERSION,
        plan_source="auto_configure",
        freq_range=tuple(float(v) for v in config.freq_range),
        accuracy=config.accuracy,
        boundary=config.boundary,
        cell_sizes=_cell_sizes_from_config(config),
        grid_shape=tuple(int(v) for v in grid_shape),
        domain=tuple(float(v) for v in config.domain),
        margin=float(config.margin),
        cfl=_cfl_from_config(config),
        absorber=_absorber_from_config(config),
        resolution_basis=_resolution_basis_from_config(
            config,
            geometry,
            material_map,
        ),
        support_checks=[
            {
                "source": "auto_configure",
                "status": "ok" if not warnings else "warn",
                "message": "Geometry/frequency mesh configuration derived.",
            }
        ],
        memory=_memory_payload(
            source="SimConfig.estimated_memory_mb",
            cells=cells,
            estimated_mb=config.estimated_memory_mb,
            max_memory_mb=max_memory_mb,
            status=memory_status,
        ),
        artifact_declarations=_artifact_declarations(artifact_root),
        warnings=warnings,
        recommendation=config.summary(),
        _sim_config=config,
    )


def _profile_min(profile: Any, fallback: float) -> float:
    return float(np.min(profile)) if profile is not None else float(fallback)


def _profile_max(profile: Any, fallback: float) -> float:
    return float(np.max(profile)) if profile is not None else float(fallback)


def _plan_cell_sizes_from_state(state: dict[str, Any]) -> dict[str, Any]:
    freq_max = float(state["freq_max"])
    dx = state.get("dx")
    nominal_dx = float(dx) if dx is not None else C0 / freq_max / 20.0
    dx_profile = state.get("dx_profile")
    dy_profile = state.get("dy_profile")
    dz_profile = state.get("dz_profile")
    dx_min = _profile_min(dx_profile, nominal_dx)
    dx_max = _profile_max(dx_profile, nominal_dx)
    dy_min = _profile_min(dy_profile, nominal_dx)
    dy_max = _profile_max(dy_profile, nominal_dx)
    return {
        "nominal_dx": nominal_dx,
        "dx": nominal_dx,
        "dy": nominal_dx,
        "dx_min": dx_min,
        "dx_max": dx_max,
        "dy_min": dy_min,
        "dy_max": dy_max,
        "dz_min": _profile_min(dz_profile, nominal_dx),
        "dz_max": _profile_max(dz_profile, nominal_dx),
        "profiles_present": {
            "x": dx_profile is not None,
            "y": dy_profile is not None,
            "z": dz_profile is not None,
        },
    }


def _absorber_from_state(state: dict[str, Any], nominal_dx: float) -> dict[str, Any]:
    cpml_layers = int(state["cpml_layers"])
    thickness = float(cpml_layers * nominal_dx)
    return {
        "boundary": state["boundary"],
        "cpml_layers": cpml_layers,
        "physical_thickness": {"x": thickness, "y": thickness, "z": thickness},
        "disabled_faces": list(state.get("pec_faces", ())),
    }


# rfx.api._preflight.PreflightIssue severity -> support-row status. "info" is
# already a status this schema uses elsewhere (the dispersion and stability
# rows), so an informational finding gets its own row state rather than being
# reported as something to fix.
_SUPPORT_STATUS_BY_SEVERITY = {"error": "fail", "warning": "warn", "info": "info"}


def _support_rows(source: str, issues: Any) -> list[dict[str, Any]]:
    """One row per preflight finding, keyed on the finding's OWN severity.

    ``issues`` holds ``PreflightIssue`` (a ``str`` subclass carrying
    ``.severity``) on every current path. The severity is read off the issue
    when it is there and only falls back to the historical ``"ERROR:"`` text
    prefix for a plain string — which is also a correctness fix, since an
    error-severity issue raised as a ``PreflightErrorWarning`` carries no such
    prefix and used to be filed as ``warn``.
    """
    if not issues:
        return [{"source": source, "status": "ok", "message": "All checks passed."}]
    rows = []
    for issue in issues:
        severity = getattr(issue, "severity", None)
        if severity is None:
            status = "fail" if str(issue).startswith("ERROR:") else "warn"
        else:
            status = _SUPPORT_STATUS_BY_SEVERITY.get(severity, "warn")
        rows.append({"source": source, "status": status, "message": str(issue)})
    return rows


def plan_simulation_mesh(
    sim: Any,
    *,
    n_steps: int | None = None,
    checkpoint_every: int | None = None,
    available_memory_gb: float | None = None,
    sparameter_calculator: str | None = None,
    artifact_root: str | Path | None = None,
) -> MeshPlan:
    """Plan/audit a configured ``Simulation`` using existing preflight APIs.

    ``artifact_root`` only declares intended paths; it never creates directories
    or writes files.

    ``sparameter_calculator="waveguide"`` is the one calculator whose preflight
    is not free: it builds the grid and solves each waveguide port's mode so
    the setup audits can read the runner's own plane indices and cutoffs. It
    runs no FDTD, but on a fine mesh with several ports it is seconds rather
    than milliseconds. Those audits also file an informational row per port
    pair (the known E-plane offset), which is a ``status="info"`` row, not a
    ``"warn"`` one.
    """
    state = dict(sim._mesh_planner_state())
    report = sim.mesh_intelligence_report(
        n_steps=n_steps,
        checkpoint_every=checkpoint_every,
        available_memory_gb=available_memory_gb,
    )
    support_checks = _support_rows("general_preflight", report.preflight_issues)
    if sparameter_calculator is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                sparam_issues = tuple(
                    sim.preflight_sparameters(calculator=sparameter_calculator)
                )
            except (ValueError, NotImplementedError) as exc:
                sparam_issues = (f"{type(exc).__name__}: {exc}",)
        support_checks.extend(_support_rows("sparameter_preflight", sparam_issues))

    cell_sizes = _plan_cell_sizes_from_state(state)
    limiting_axis = min(
        (
            ("x", cell_sizes["dx_min"]),
            ("y", cell_sizes["dy_min"]),
            ("z", cell_sizes["dz_min"]),
        ),
        key=lambda item: item[1],
    )[0]
    nominal_dx = float(cell_sizes["nominal_dx"])
    freq_max = float(state["freq_max"])
    domain = tuple(float(v) for v in state["domain"])
    return MeshPlan(
        schema_version=SCHEMA_VERSION,
        plan_source="configured_simulation",
        freq_range=(None, freq_max),
        accuracy=None,
        boundary=str(state["boundary"]),
        cell_sizes=cell_sizes,
        grid_shape=tuple(int(v) for v in report.grid_shape),
        domain=domain,
        margin=None,
        cfl={
            "dt": float(state["dt"]),
            "formula": "Simulation._build_grid().dt",
            "basis": "configured Simulation mesh",
            "limiting_axis": (
                limiting_axis
                if min(cell_sizes["dx_min"], cell_sizes["dy_min"], cell_sizes["dz_min"])
                < nominal_dx
                else "uniform"
            ),
        },
        absorber=_absorber_from_state(state, nominal_dx),
        resolution_basis=[
            {
                "source": "mesh_intelligence_report",
                "kind": "configured_mesh",
                "status": "info",
                "message": (
                    "Concrete Simulation mesh summarized without rerunning "
                    "geometry derivation."
                ),
                "uses_nonuniform": bool(report.uses_nonuniform),
                "min_cell_size": float(report.min_cell_size),
                "nominal_dx": float(report.nominal_dx),
            }
        ],
        support_checks=support_checks,
        memory=_memory_payload(
            source="Simulation.mesh_intelligence_report",
            cells=report.cells,
            uniform_fine_cells=report.uniform_fine_cells,
            cell_savings_factor=report.cell_savings_factor,
            ad_memory=report.ad_memory,
            status="ok" if not report.preflight_issues else "warn",
        ),
        artifact_declarations=_artifact_declarations(artifact_root),
        warnings=(),
        recommendation=report.recommendation,
    )


__all__ = ["MeshPlan", "plan_mesh", "plan_simulation_mesh"]

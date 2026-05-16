"""Production-envelope validation for the public subgridding API.

The current promoted support surface is deliberately narrow.  The numerical
runner is a z-slab refinement path: the fine grid covers the full x/y interior
and refines a bounded z interval.  The validation below encodes the physical
assumptions required by that SBP-SAT closure instead of relying on smoke tests:

* production support is limited to guarded one-sided PEC/no-CPML full-x/y
  z slabs whose observables stay away from the remaining artificial interface,
* artificial coarse/fine interfaces must not coincide with a material or PEC
  discontinuity,
* all source/probe plus lumped/wire impedance-port observables handled by the
  runner must live on the fine grid,
* NTFF/far-field boxes are limited to the same guarded fine-grid envelope;
  other unsupported RF post-processing/features are rejected before execution,
* dispersive/nonlinear material models are rejected because this runner receives
  only static eps/sigma/mu arrays.

The report is user-facing so examples and CI can show why a case is inside or
outside the production validation envelope.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class SubgridValidationIssue(NamedTuple):
    """One validation issue for a subgridded simulation."""

    severity: str  # "error", "warning", or "info"
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable artifact-friendly representation."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


class SubgridRegion(NamedTuple):
    """Coarse/fine index mapping for the current z-slab subgrid runner."""

    fi_lo: int
    fi_hi: int
    fj_lo: int
    fj_hi: int
    fk_lo: int
    fk_hi: int
    nx_f: int
    ny_f: int
    nz_f: int
    dx_c: float
    dx_f: float
    ratio: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable region mapping."""
        return {
            "fi_lo": self.fi_lo,
            "fi_hi": self.fi_hi,
            "fj_lo": self.fj_lo,
            "fj_hi": self.fj_hi,
            "fk_lo": self.fk_lo,
            "fk_hi": self.fk_hi,
            "nx_f": self.nx_f,
            "ny_f": self.ny_f,
            "nz_f": self.nz_f,
            "dx_c": self.dx_c,
            "dx_f": self.dx_f,
            "ratio": self.ratio,
        }


class SubgridValidationReport(NamedTuple):
    """Validation report for ``Simulation.add_refinement``."""

    supported: bool
    mode: str
    support_level: str
    region: SubgridRegion | None
    issues: tuple[SubgridValidationIssue, ...]

    @property
    def errors(self) -> tuple[SubgridValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[SubgridValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def format(self) -> str:
        """Return a compact multiline report."""
        lines = [
            f"subgrid validation: supported={self.supported} "
            f"mode={self.mode} level={self.support_level}"
        ]
        for issue in self.issues:
            lines.append(f"- {issue.severity.upper()} [{issue.code}] {issue.message}")
        return "\n".join(lines)

    def raise_if_unsupported(self) -> "SubgridValidationReport":
        """Raise ``ValueError`` with the formatted report when unsupported.

        Returning ``self`` on success keeps the method convenient for examples
        and scripts that want both a fail-fast guard and an artifact.
        """
        if not self.supported:
            raise ValueError(self.format())
        return self

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable validation artifact."""
        return {
            "supported": self.supported,
            "mode": self.mode,
            "support_level": self.support_level,
            "region": None if self.region is None else self.region.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the validation report for research-note artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)


def build_subgrid_region(sim, grid) -> SubgridRegion | None:
    """Return the runner's coarse/fine index mapping, or ``None``."""
    ref = getattr(sim, "_refinement", None)
    if ref is None:
        return None
    topology = ref.get("topology", "overlap_z_slab")
    if topology == "stage2_disjoint_3d":
        return build_stage2_disjoint_region(sim, grid)
    ratio = int(ref["ratio"])
    z_lo, z_hi = ref["z_range"]
    dx_c = float(grid.dx)
    dx_f = dx_c / ratio
    pad_z_lo = int(getattr(grid, "pad_z_lo", grid.cpml_layers))
    pad_z_hi = int(getattr(grid, "pad_z_hi", grid.cpml_layers))
    fk_lo = max(int(round(z_lo / dx_c)) + pad_z_lo, pad_z_lo)
    fk_hi = min(int(round(z_hi / dx_c)) + pad_z_lo + 1, grid.nz - pad_z_hi)
    xy_margin = ref.get("xy_margin")
    if xy_margin is None:
        fi_lo = grid.pad_x_lo
        fi_hi = grid.nx - grid.pad_x_hi
        fj_lo = grid.pad_y_lo
        fj_hi = grid.ny - grid.pad_y_hi
    else:
        margin = float(xy_margin)
        fi_lo = max(int(round(margin / dx_c)) + grid.pad_x_lo, grid.pad_x_lo)
        fi_hi = min(
            int(round((sim._domain[0] - margin) / dx_c)) + grid.pad_x_lo + 1,
            grid.nx - grid.pad_x_hi,
        )
        fj_lo = max(int(round(margin / dx_c)) + grid.pad_y_lo, grid.pad_y_lo)
        fj_hi = min(
            int(round((sim._domain[1] - margin) / dx_c)) + grid.pad_y_lo + 1,
            grid.ny - grid.pad_y_hi,
        )
    return SubgridRegion(
        fi_lo=fi_lo,
        fi_hi=fi_hi,
        fj_lo=fj_lo,
        fj_hi=fj_hi,
        fk_lo=fk_lo,
        fk_hi=fk_hi,
        nx_f=(fi_hi - fi_lo - 1) * ratio + 1,
        ny_f=(fj_hi - fj_lo - 1) * ratio + 1,
        nz_f=(fk_hi - fk_lo - 1) * ratio + 1,
        dx_c=dx_c,
        dx_f=dx_f,
        ratio=ratio,
    )


def build_stage2_disjoint_region(sim, grid) -> SubgridRegion:
    """Return the public-contract region for the Stage-2 disjoint topology.

    This currently reuses the existing physical refinement box indexing so
    validation can check source/probe/material placement before the dedicated
    disjoint public runner lands.  The actual disjoint runner will own only the
    coarse exterior plus the fine block instead of the overlap z-slab shadow.
    """
    ratio = int(sim._refinement["ratio"])
    z_lo, z_hi = sim._refinement["z_range"]
    dx_c = float(grid.dx)
    dx_f = dx_c / ratio
    pad_z_lo = int(getattr(grid, "pad_z_lo", grid.cpml_layers))
    pad_z_hi = int(getattr(grid, "pad_z_hi", grid.cpml_layers))
    fk_lo = max(int(round(z_lo / dx_c)) + pad_z_lo, pad_z_lo)
    fk_hi = min(int(round(z_hi / dx_c)) + pad_z_lo + 1, grid.nz - pad_z_hi)
    xy_margin = sim._refinement.get("xy_margin")
    if xy_margin is None:
        fi_lo = grid.pad_x_lo
        fi_hi = grid.nx - grid.pad_x_hi
        fj_lo = grid.pad_y_lo
        fj_hi = grid.ny - grid.pad_y_hi
    else:
        margin = float(xy_margin)
        fi_lo = max(int(round(margin / dx_c)) + grid.pad_x_lo, grid.pad_x_lo)
        fi_hi = min(
            int(round((sim._domain[0] - margin) / dx_c)) + grid.pad_x_lo + 1,
            grid.nx - grid.pad_x_hi,
        )
        fj_lo = max(int(round(margin / dx_c)) + grid.pad_y_lo, grid.pad_y_lo)
        fj_hi = min(
            int(round((sim._domain[1] - margin) / dx_c)) + grid.pad_y_lo + 1,
            grid.ny - grid.pad_y_hi,
        )
    return SubgridRegion(
        fi_lo=fi_lo,
        fi_hi=fi_hi,
        fj_lo=fj_lo,
        fj_hi=fj_hi,
        fk_lo=fk_lo,
        fk_hi=fk_hi,
        nx_f=(fi_hi - fi_lo) * ratio,
        ny_f=(fj_hi - fj_lo) * ratio,
        nz_f=(fk_hi - fk_lo) * ratio,
        dx_c=dx_c,
        dx_f=dx_f,
        ratio=ratio,
    )


def _issue(severity: str, code: str, message: str) -> SubgridValidationIssue:
    return SubgridValidationIssue(severity=severity, code=code, message=message)


def _active_diagnostic_overrides(ref) -> tuple[str, ...]:
    """Return private diagnostic knobs that differ from production defaults."""
    active: list[str] = []
    if ref.get("use_material_sat", True) is not True:
        active.append("use_material_sat")
    if float(ref.get("material_sat_scale", 1.0)) != 1.0:
        active.append("material_sat_scale")
    if float(ref.get("material_sat_coarse_scale", 1.0)) != 1.0:
        active.append("material_sat_coarse_scale")
    if float(ref.get("material_sat_fine_scale", 1.0)) != 1.0:
        active.append("material_sat_fine_scale")
    if float(ref.get("material_sat_e_coarse_scale", 1.0)) != 1.0:
        active.append("material_sat_e_coarse_scale")
    if float(ref.get("material_sat_e_fine_scale", 1.0)) != 1.0:
        active.append("material_sat_e_fine_scale")
    if float(ref.get("material_sat_h_coarse_scale", 1.0)) != 1.0:
        active.append("material_sat_h_coarse_scale")
    if float(ref.get("material_sat_h_fine_scale", 1.0)) != 1.0:
        active.append("material_sat_h_fine_scale")
    if float(ref.get("material_sat_zlo_scale", 1.0)) != 1.0:
        active.append("material_sat_zlo_scale")
    if float(ref.get("material_sat_zhi_scale", 1.0)) != 1.0:
        active.append("material_sat_zhi_scale")
    if float(ref.get("material_sat_e_zlo_scale", 1.0)) != 1.0:
        active.append("material_sat_e_zlo_scale")
    if float(ref.get("material_sat_e_zhi_scale", 1.0)) != 1.0:
        active.append("material_sat_e_zhi_scale")
    if float(ref.get("material_sat_h_zlo_scale", 1.0)) != 1.0:
        active.append("material_sat_h_zlo_scale")
    if float(ref.get("material_sat_h_zhi_scale", 1.0)) != 1.0:
        active.append("material_sat_h_zhi_scale")
    if float(ref.get("material_sat_pair_a_zlo_scale", 1.0)) != 1.0:
        active.append("material_sat_pair_a_zlo_scale")
    if float(ref.get("material_sat_pair_b_zlo_scale", 1.0)) != 1.0:
        active.append("material_sat_pair_b_zlo_scale")
    if ref.get("material_sat_zlo_common_trace_projection", "dual") != "dual":
        active.append("material_sat_zlo_common_trace_projection")
    if ref.get("material_sat_zhi_common_trace_projection", "dual") != "dual":
        active.append("material_sat_zhi_common_trace_projection")
    if float(ref.get("material_sat_e_h_trace_blend", 1.0)) != 1.0:
        active.append("material_sat_e_h_trace_blend")
    if float(ref.get("material_sat_e_h_trace_zlo_blend", 1.0)) != 1.0:
        active.append("material_sat_e_h_trace_zlo_blend")
    if float(ref.get("material_sat_e_h_trace_zhi_blend", 1.0)) != 1.0:
        active.append("material_sat_e_h_trace_zhi_blend")
    if ref.get("material_sat_e_h_trace_zlo_filter", "full") != "full":
        active.append("material_sat_e_h_trace_zlo_filter")
    if ref.get("material_sat_e_h_trace_zlo_components", "all") != "all":
        active.append("material_sat_e_h_trace_zlo_components")
    if ref.get("material_sat_e_h_trace_zlo_vector_mix", "identity") != "identity":
        active.append("material_sat_e_h_trace_zlo_vector_mix")
    if float(ref.get("material_sat_e_h_trace_zlo_residual_limit", 0.0)) != 0.0:
        active.append("material_sat_e_h_trace_zlo_residual_limit")
    if float(ref.get("material_sat_normal_e_scale", 0.0)) != 0.0:
        active.append("material_sat_normal_e_scale")
    if float(ref.get("material_sat_zhi_coarse_eps_blend", 0.0)) != 0.0:
        active.append("material_sat_zhi_coarse_eps_blend")
    if ref.get("defer_material_h_sat_until_after_e", False):
        active.append("defer_material_h_sat_until_after_e")
    if float(ref.get("coarse_shadow_source_scale", 1.0)) != 1.0:
        active.append("coarse_shadow_source_scale")
    if float(ref.get("fine_source_scale", 1.0)) != 1.0:
        active.append("fine_source_scale")
    if ref.get("coarse_shadow_source_projection", "physical_nearest") != "physical_nearest":
        active.append("coarse_shadow_source_projection")
    if ref.get("box_shadow_sync_fields", "all") != "all":
        active.append("box_shadow_sync_fields")
    if ref.get("box_shadow_sync_region", "volume") != "volume":
        active.append("box_shadow_sync_region")
    if float(ref.get("box_shadow_sync_scale", 1.0)) != 1.0:
        active.append("box_shadow_sync_scale")
    if ref.get("box_shadow_sync_face_axes", "all") != "all":
        active.append("box_shadow_sync_face_axes")
    if ref.get("box_shadow_sync_components", "all") != "all":
        active.append("box_shadow_sync_components")
    if ref.get("box_shadow_sync_face_sides", "all") != "all":
        active.append("box_shadow_sync_face_sides")
    if ref.get("box_shadow_sync_timing", "all") != "all":
        active.append("box_shadow_sync_timing")
    if ref.get("material_sat_face_projection", "node_adjoint") != "node_adjoint":
        active.append("material_sat_face_projection")
    for name in (
        "sync_coarse_interface_from_fine",
        "sync_coarse_shadow_from_fine",
        "sync_box_coarse_shadow_from_fine",
        "mask_coarse_shadow_interior",
        "use_exterior_z_interfaces",
        "use_boundary_terminated_exterior_z_interfaces",
        "use_exterior_box_interfaces",
        "ghost_exterior_coarse_shadow_from_fine",
        "inject_sources_before_e_coupling",
        "inject_sources_on_coarse_shadow",
    ):
        if bool(ref.get(name, False)):
            active.append(name)
    if ref.get("diagnostic_lumped_sparam_freqs") is not None:
        active.append("diagnostic_lumped_sparam_freqs")
    return tuple(active)


def _material_plane_delta(arr, a, b, region: SubgridRegion) -> float:
    slab_a = arr[region.fi_lo:region.fi_hi, region.fj_lo:region.fj_hi, a]
    slab_b = arr[region.fi_lo:region.fi_hi, region.fj_lo:region.fj_hi, b]
    return float(jnp.max(jnp.abs(slab_a - slab_b)))


def _material_transition_clearance_violations(
    arr,
    *,
    region: SubgridRegion,
    grid,
    artificial_z_m: float,
    min_clearance_m: float,
    tol: float,
) -> list[tuple[int, int, float, float]]:
    """Return material z-plane transitions too close to an artificial interface.

    Direct jumps exactly across the artificial coarse/fine interface are handled
    by ``material_jump_at_zlo/zhi_interface``. This helper catches the next
    fail-closed case established by diagnostics: a material transition one or a
    few coarse cells away from the remaining artificial interface can still
    break long-window agreement even when the material is continuous across the
    interface itself.
    """
    pad_z_lo = int(getattr(grid, "pad_z_lo", grid.cpml_layers))
    violations: list[tuple[int, int, float, float]] = []
    for k0 in range(0, max(grid.nz - 1, 0)):
        k1 = k0 + 1
        if k1 >= grid.nz:
            continue
        # Direct artificial-interface jumps already have more specific error
        # codes. Keep this rule focused on nearby transitions.
        if (k0, k1) in {
            (region.fk_lo - 1, region.fk_lo),
            (region.fk_hi - 1, region.fk_hi),
        }:
            continue
        delta = _material_plane_delta(arr, k0, k1, region)
        if delta <= tol:
            continue
        transition_z_m = ((k0 + 0.5) - pad_z_lo) * float(grid.dx)
        clearance_m = abs(float(transition_z_m) - float(artificial_z_m))
        if clearance_m + 1e-15 < min_clearance_m:
            violations.append((k0, k1, float(delta), float(clearance_m)))
    return violations


def _position_inside_fine(pos, sim, region: SubgridRegion) -> bool:
    # Use physical extents rather than relying only on rounded indices.
    z_lo, z_hi = sim._refinement["z_range"]
    tol = 0.5 * region.dx_f
    xy_margin = sim._refinement.get("xy_margin")
    if xy_margin is None:
        x_lo, x_hi = 0.0, float(sim._domain[0])
        y_lo, y_hi = 0.0, float(sim._domain[1])
    else:
        margin = float(xy_margin)
        x_lo, x_hi = margin, float(sim._domain[0]) - margin
        y_lo, y_hi = margin, float(sim._domain[1]) - margin
    return (
        x_lo - tol <= pos[0] <= x_hi + tol
        and y_lo - tol <= pos[1] <= y_hi + tol
        and z_lo - tol <= pos[2] <= z_hi + tol
    )


def _one_sided_physical_z_boundary(sim, region: SubgridRegion, grid) -> str | None:
    """Return the physical z boundary touched by the slab, if exactly one.

    Use the requested physical ``z_range`` rather than only rounded coarse-grid
    indices.  Index rounding can make a near-boundary diagnostic look
    boundary-terminated even when its requested slab does not actually touch
    the PEC wall; those cases must stay out of production validation.
    """
    del region, grid
    z_lo, z_hi = sim._refinement["z_range"]
    domain_z = float(sim._domain[2])
    tol = 1e-12 * max(1.0, domain_z)
    touches_zlo = abs(float(z_lo)) <= tol
    touches_zhi = abs(float(z_hi) - domain_z) <= tol
    if touches_zlo == touches_zhi:
        return None
    return "z_lo" if touches_zlo else "z_hi"


def _boundary_face_kind(sim, face: str) -> str:
    """Return the configured boundary token for a physical face."""
    spec = getattr(sim, "_boundary_spec", None)
    if spec is not None:
        axis, side = face.split("_")
        boundary = getattr(spec, axis)
        return str(getattr(boundary, side))
    if face in getattr(sim, "_pec_faces", set()):
        return "pec"
    return str(getattr(sim, "_boundary", "pec"))


def _guarded_boundary_production_allowed(
    sim,
    grid,
    physical_boundary: str | None,
    xy_margin,
) -> bool:
    """Return whether the current one-sided slab has a supported boundary mix.

    Full PEC/no-CPML remains the baseline.  A narrow CPML/UPML-compatible lane
    is also allowed when the refined slab touches a PEC z face, the opposite z
    face may be absorbing, and x/y are closed PEC faces.  This avoids claiming
    fine-grid CPML support on faces touched by the refined region.
    """
    if physical_boundary is None or xy_margin is not None:
        return False
    if _boundary_face_kind(sim, physical_boundary) != "pec":
        return False
    pads = {
        "x_lo": int(getattr(grid, "pad_x_lo", grid.cpml_layers)),
        "x_hi": int(getattr(grid, "pad_x_hi", grid.cpml_layers)),
        "y_lo": int(getattr(grid, "pad_y_lo", grid.cpml_layers)),
        "y_hi": int(getattr(grid, "pad_y_hi", grid.cpml_layers)),
        "z_lo": int(getattr(grid, "pad_z_lo", grid.cpml_layers)),
        "z_hi": int(getattr(grid, "pad_z_hi", grid.cpml_layers)),
    }
    # Fine x/y faces have no fine-grid CPML implementation.  Keep production
    # support to closed x/y faces; z may have an absorber only on the face not
    # touched by the refined slab.
    if any(pads[face] > 0 for face in ("x_lo", "x_hi", "y_lo", "y_hi")):
        return False
    if pads[physical_boundary] > 0:
        return False
    opposite = "z_hi" if physical_boundary == "z_lo" else "z_lo"
    return all(pads[face] == 0 for face in ("z_lo", "z_hi") if face != opposite)


def _artificial_interface_margin_fraction(pos, sim, physical_boundary: str) -> float:
    """Return distance from the remaining artificial z interface as slab fraction."""
    z_lo, z_hi = sim._refinement["z_range"]
    span = max(float(z_hi - z_lo), 1e-30)
    if physical_boundary == "z_lo":
        return float(z_hi - pos[2]) / span
    return float(pos[2] - z_lo) / span


def _xy_window_margin_fraction(pos, sim) -> float:
    """Return min x/y distance from local window faces as a window fraction."""
    xy_margin = sim._refinement.get("xy_margin")
    if xy_margin is None:
        return float("inf")
    margin = float(xy_margin)
    x_lo, x_hi = margin, float(sim._domain[0]) - margin
    y_lo, y_hi = margin, float(sim._domain[1]) - margin
    x_span = max(x_hi - x_lo, 1e-30)
    y_span = max(y_hi - y_lo, 1e-30)
    return float(
        min(
            (pos[0] - x_lo) / x_span,
            (x_hi - pos[0]) / x_span,
            (pos[1] - y_lo) / y_span,
            (y_hi - pos[1]) / y_span,
        )
    )


def _xy_window_axis_fractions(pos, sim) -> tuple[float, float]:
    """Return x/y local-window fractions for a physical position."""
    xy_margin = sim._refinement.get("xy_margin")
    if xy_margin is None:
        return (float("inf"), float("inf"))
    margin = float(xy_margin)
    x_lo, x_hi = margin, float(sim._domain[0]) - margin
    y_lo, y_hi = margin, float(sim._domain[1]) - margin
    x_span = max(x_hi - x_lo, 1e-30)
    y_span = max(y_hi - y_lo, 1e-30)
    return (float((pos[0] - x_lo) / x_span), float((pos[1] - y_lo) / y_span))


def _local_xy_center_source_candidate_allowed(sim) -> bool:
    """Return whether a local x/y window matches the narrow production lane.

    The only claims-bearing local x/y evidence so far is a central soft-source
    lane whose source is co-injected on the overlapping coarse shadow grid by
    the runner default.  Off-centre source coverage still has waveform
    residuals, so production validation must keep those cases closed.
    """
    if sim._refinement.get("xy_margin") is None:
        return False
    if sim._boundary != "pec" or int(getattr(sim, "_cpml_layers", 0)) != 0:
        return False
    sources = [pe for pe in getattr(sim, "_ports", ()) if pe.impedance == 0.0]
    if not sources:
        return False
    center_tol = 1e-12
    for source in sources:
        xf, yf = _xy_window_axis_fractions(source.position, sim)
        if abs(xf - 0.5) > center_tol or abs(yf - 0.5) > center_tol:
            return False
    return True


def validate_subgrid_setup(
    sim,
    grid,
    materials,
    pec_mask=None,
    *,
    mode: str = "production",
) -> SubgridValidationReport:
    """Validate a ``Simulation.add_refinement`` setup against support claims."""
    issues: list[SubgridValidationIssue] = []
    ref = getattr(sim, "_refinement", None)
    if ref is None:
        return SubgridValidationReport(
            supported=False,
            mode=mode,
            support_level="none",
            region=None,
            issues=(_issue("error", "no_refinement", "no refinement is configured"),),
        )

    if mode not in {"production", "research", "off"}:
        issues.append(_issue("error", "bad_validation_mode", f"unknown mode {mode!r}"))

    topology = ref.get("topology", "overlap_z_slab")
    if topology not in {"overlap_z_slab", "stage2_disjoint_3d"}:
        issues.append(
            _issue(
                "error",
                "bad_subgrid_topology",
                f"unknown subgrid topology {topology!r}",
            )
        )
    is_stage2_disjoint_topology = topology == "stage2_disjoint_3d"
    if mode == "production" and is_stage2_disjoint_topology:
        issues.append(
            _issue(
                "error",
                "disjoint_topology_public_runner_unintegrated",
                "Stage-2 disjoint 3-D topology is selected as the next "
                "centered/two-interface integration lane but is not yet wired "
                "to production waveform/crossval gates. Use validation='research' "
                "only for research-runner smoke checks.",
            )
        )

    ratio = ref["ratio"]
    if not isinstance(ratio, int) or ratio < 2:
        issues.append(_issue("error", "bad_ratio", "subgrid ratio must be an integer >= 2"))

    z_lo, z_hi = ref["z_range"]
    if not (np.isfinite(z_lo) and np.isfinite(z_hi) and 0.0 <= z_lo < z_hi <= sim._domain[2]):
        issues.append(
            _issue(
                "error",
                "bad_z_range",
                f"z_range={ref['z_range']!r} must be finite, ordered, and inside the domain",
            )
        )

    xy_margin = ref.get("xy_margin")
    xy_window_margin_min_fraction = 0.45
    local_xy_candidate_allowed = _local_xy_center_source_candidate_allowed(sim)
    if xy_margin is not None:
        margin = float(xy_margin)
        if (
            not np.isfinite(margin)
            or margin < 0.0
            or 2.0 * margin >= min(float(sim._domain[0]), float(sim._domain[1]))
        ):
            issues.append(
                _issue(
                    "error",
                    "bad_xy_margin",
                    "xy_margin must be finite, non-negative, and leave a "
                    "non-empty x/y fine window",
                )
            )
        if mode == "production" and not local_xy_candidate_allowed:
            issues.append(
                _issue(
                    "error",
                    "xy_windowed_external_crossval_blocked",
                    "local x/y windowed subgridding remains research-only: "
                    "only the central soft-source lane has enough waveform "
                    "and external-frequency evidence for production; use a "
                    "uniform/non-uniform mesh or validation='research' for "
                    "off-centre sources, CPML, ports, or broader local x/y "
                    "closures.",
                )
            )

    if mode == "production":
        active_overrides = _active_diagnostic_overrides(ref)
        if active_overrides:
            issues.append(
                _issue(
                    "error",
                    "diagnostic_subgrid_override_unvalidated",
                    "production subgrid validation cannot run diagnostic-only "
                    f"override(s): {', '.join(active_overrides)}. Use "
                    "validation='research' for internal diagnostics; do not "
                    "use these knobs for claims-bearing production results.",
                )
            )

    region = build_subgrid_region(sim, grid)
    if region is None:
        issues.append(_issue("error", "no_region", "could not build subgrid region"))
    elif region.nz_f <= 0 or region.nx_f <= 0 or region.ny_f <= 0:
        issues.append(_issue("error", "empty_fine_grid", f"fine grid is empty: {region}"))

    physical_z_boundary = None
    guarded_boundary_allowed = False
    if region is not None:
        pad_z_lo = int(getattr(grid, "pad_z_lo", grid.cpml_layers))
        pad_z_hi = int(getattr(grid, "pad_z_hi", grid.cpml_layers))
        physical_z_boundary = _one_sided_physical_z_boundary(sim, region, grid)
        guarded_boundary_allowed = _guarded_boundary_production_allowed(
            sim,
            grid,
            physical_z_boundary,
            xy_margin,
        )
        boundary_margin_min_fraction = 0.30
        if sim._boundary in ("cpml", "upml") and (pad_z_lo > 0 or pad_z_hi > 0):
            overlaps_zlo_absorber = pad_z_lo > 0 and region.fk_lo <= pad_z_lo
            overlaps_zhi_absorber = pad_z_hi > 0 and region.fk_hi >= grid.nz - pad_z_hi
            if overlaps_zlo_absorber or overlaps_zhi_absorber:
                sev = "error" if mode == "production" else "warning"
                issues.append(
                    _issue(
                        sev,
                        "subgrid_overlaps_absorber",
                        "production subgrid z interfaces must stay outside CPML/UPML",
                    )
                )

        if mode == "production" and not is_stage2_disjoint_topology:
            if physical_z_boundary is None:
                issues.append(
                    _issue(
                        "error",
                        "z_slab_requires_guarded_boundary",
                        "production subgrid support is limited to a guarded "
                        "one-sided full-x/y z slab that touches exactly one "
                        "physical z boundary; centered/two-interface slabs "
                        "remain research diagnostics until a dedicated "
                        "centered-slab interface closure and validation "
                        "artifact exist.",
                    )
                )
            elif xy_margin is None and not guarded_boundary_allowed:
                issues.append(
                    _issue(
                        "error",
                        "boundary_terminated_requires_pec_no_cpml",
                        "one-sided boundary-terminated production subgrid "
                        "support requires the refined slab to touch a PEC z "
                        "face and avoid any fine-grid face that would need "
                        "CPML/UPML; only the opposite z face may be absorbing",
                    )
                )

        # Artificial z interfaces can use the material-weighted SAT only for a
        # static material load that is continuous across the interface.  A true
        # material discontinuity at an artificial coarse/fine interface remains
        # fail-closed until the material-jump waveform gate passes.
        if 0 < region.fk_lo < grid.nz:
            for name, arr, tol in (
                ("eps_r", materials.eps_r, 1e-6),
                ("sigma", materials.sigma, 1e-9),
                ("mu_r", materials.mu_r, 1e-6),
            ):
                delta = _material_plane_delta(arr, region.fk_lo - 1, region.fk_lo, region)
                if delta > tol:
                    issues.append(
                        _issue(
                            "error",
                            "material_jump_at_zlo_interface",
                            f"{name} changes by {delta:.3e} across the z-lo subgrid interface; "
                            "move material discontinuities inside the fine region or use a uniform mesh",
                        )
                    )
        if 0 <= region.fk_hi - 1 < grid.nz and region.fk_hi < grid.nz:
            for name, arr, tol in (
                ("eps_r", materials.eps_r, 1e-6),
                ("sigma", materials.sigma, 1e-9),
                ("mu_r", materials.mu_r, 1e-6),
            ):
                delta = _material_plane_delta(arr, region.fk_hi - 1, region.fk_hi, region)
                if delta > tol:
                    issues.append(
                        _issue(
                            "error",
                            "material_jump_at_zhi_interface",
                            f"{name} changes by {delta:.3e} across the z-hi subgrid interface; "
                            "move material discontinuities inside the fine region or use a uniform mesh",
                        )
                    )

        if mode == "production" and physical_z_boundary is not None:
            z_lo, z_hi = sim._refinement["z_range"]
            span = float(z_hi) - float(z_lo)
            artificial_z_m = float(z_hi if physical_z_boundary == "z_lo" else z_lo)
            material_clearance_min_m = max(
                4.0 * float(region.dx_c),
                boundary_margin_min_fraction * span,
            )
            for name, arr, tol in (
                ("eps_r", materials.eps_r, 1e-6),
                ("sigma", materials.sigma, 1e-9),
                ("mu_r", materials.mu_r, 1e-6),
            ):
                violations = _material_transition_clearance_violations(
                    arr,
                    region=region,
                    grid=grid,
                    artificial_z_m=artificial_z_m,
                    min_clearance_m=material_clearance_min_m,
                    tol=tol,
                )
                if violations:
                    k0, k1, delta, clearance_m = min(
                        violations,
                        key=lambda item: item[3],
                    )
                    issues.append(
                        _issue(
                            "error",
                            "material_transition_near_artificial_interface",
                            f"{name} changes by {delta:.3e} between z planes "
                            f"{k0} and {k1}, only {clearance_m:.3e} m from "
                            "the remaining artificial subgrid interface; "
                            "production static-material support requires "
                            "nearby material transitions to stay outside the "
                            f"validated clearance buffer ({material_clearance_min_m:.3e} m) "
                            "or use a uniform/non-uniform reference mesh.",
                        )
                    )

        if pec_mask is not None:
            pec = pec_mask.astype(jnp.bool_)
            pec_interface_guarded_allowed = (
                mode == "production"
                and xy_margin is None
                and physical_z_boundary is not None
                and guarded_boundary_allowed
                and str(getattr(sim, "_boundary", "pec")) == "pec"
                and int(getattr(grid, "cpml_layers", 0)) == 0
            )
            if mode == "production" and xy_margin is not None:
                has_local_pec = bool(
                    jnp.any(
                        pec[
                            region.fi_lo:region.fi_hi,
                            region.fj_lo:region.fj_hi,
                            region.fk_lo:region.fk_hi,
                        ]
                    )
                )
                if has_local_pec:
                    issues.append(
                        _issue(
                            "error",
                            "xy_windowed_pec_unvalidated",
                            "PEC geometry inside a local x/y subgrid window is not production-validated",
                        )
                    )
            for code, k0, k1 in (
                ("pec_at_zlo_interface", region.fk_lo - 1, region.fk_lo),
                ("pec_at_zhi_interface", region.fk_hi - 1, region.fk_hi),
            ):
                if 0 <= k0 < grid.nz and 0 <= k1 < grid.nz:
                    has_pec = bool(
                        jnp.any(pec[region.fi_lo:region.fi_hi, region.fj_lo:region.fj_hi, k0])
                        or jnp.any(pec[region.fi_lo:region.fi_hi, region.fj_lo:region.fj_hi, k1])
                    )
                    if has_pec:
                        if not pec_interface_guarded_allowed:
                            issues.append(
                                _issue(
                                    "error",
                                    code,
                                    "PEC geometry touches an artificial subgrid interface; "
                                    "material-weighted/conformal SAT is validated only "
                                    "inside the guarded one-sided PEC/no-CPML slab envelope",
                                )
                            )

        for idx, pe in enumerate(getattr(sim, "_ports", ())):
            label = "source" if pe.impedance == 0.0 else "port"
            if pe.impedance > 0.0:
                port_allowed = (
                    mode == "production"
                    and physical_z_boundary is not None
                    and sim._boundary == "pec"
                    and int(grid.cpml_layers) == 0
                    and xy_margin is None
                )
                if not port_allowed:
                    issues.append(
                        _issue(
                            "error",
                            "impedance_port_unvalidated",
                            "production subgrid validation supports lumped and "
                            "axis-aligned wire impedance ports only inside the guarded "
                            "one-sided PEC/no-CPML boundary envelope",
                        )
                    )
            if not _position_inside_fine(pe.position, sim, region):
                issues.append(
                    _issue(
                        "error",
                        "source_or_port_outside_fine_grid",
                        f"{label} #{idx} at {pe.position} is outside the refined z slab",
                    )
                )
            port_positions = [pe.position]
            if pe.extent is not None:
                axis = {"ex": 0, "ey": 1, "ez": 2}[pe.component]
                end_position = list(pe.position)
                end_position[axis] += float(pe.extent)
                end_position = tuple(end_position)
                port_positions.append(end_position)
                if not _position_inside_fine(end_position, sim, region):
                    issues.append(
                        _issue(
                            "error",
                            "source_or_port_outside_fine_grid",
                            f"{label} #{idx} endpoint at {end_position} is "
                            "outside the refined z slab",
                        )
                    )
            if mode == "production" and physical_z_boundary is not None:
                margin = min(
                    _artificial_interface_margin_fraction(
                        position,
                        sim,
                        physical_z_boundary,
                    )
                    for position in port_positions
                )
                if margin + 1e-12 < boundary_margin_min_fraction:
                    issues.append(
                        _issue(
                            "error",
                            "boundary_terminated_margin_too_close",
                            f"{label} #{idx} at {pe.position} is only "
                            f"{margin:.3f} slab spans from the remaining "
                            "artificial z interface; validated production "
                            "boundary-terminated support requires >= "
                            f"{boundary_margin_min_fraction:.2f}",
                        )
                    )
            if mode == "production" and xy_margin is not None:
                xy_margin_fraction = _xy_window_margin_fraction(pe.position, sim)
                if xy_margin_fraction + 1e-12 < xy_window_margin_min_fraction:
                    issues.append(
                        _issue(
                            "error",
                            "xy_windowed_margin_too_close",
                            f"{label} #{idx} at {pe.position} is only "
                            f"{xy_margin_fraction:.3f} local-window spans "
                            "from an artificial x/y interface; validated "
                            "local x/y window support requires >= "
                            f"{xy_window_margin_min_fraction:.2f}",
                        )
                    )
        for idx, probe in enumerate(getattr(sim, "_probes", ())):
            if not _position_inside_fine(probe.position, sim, region):
                issues.append(
                    _issue(
                        "error",
                        "probe_outside_fine_grid",
                        f"probe #{idx} at {probe.position} is outside the refined z slab",
                    )
                )
            if mode == "production" and physical_z_boundary is not None:
                margin = _artificial_interface_margin_fraction(
                    probe.position,
                    sim,
                    physical_z_boundary,
                )
                if margin + 1e-12 < boundary_margin_min_fraction:
                    issues.append(
                        _issue(
                            "error",
                            "boundary_terminated_margin_too_close",
                            f"probe #{idx} at {probe.position} is only "
                            f"{margin:.3f} slab spans from the remaining "
                            "artificial z interface; validated production "
                            "boundary-terminated support requires >= "
                            f"{boundary_margin_min_fraction:.2f}",
                        )
                    )
            if mode == "production" and xy_margin is not None:
                xy_margin_fraction = _xy_window_margin_fraction(probe.position, sim)
                if xy_margin_fraction + 1e-12 < xy_window_margin_min_fraction:
                    issues.append(
                        _issue(
                            "error",
                            "xy_windowed_margin_too_close",
                            f"probe #{idx} at {probe.position} is only "
                            f"{xy_margin_fraction:.3f} local-window spans "
                            "from an artificial x/y interface; validated "
                            "local x/y window support requires >= "
                            f"{xy_window_margin_min_fraction:.2f}",
                        )
                    )

    if getattr(sim, "_ntff", None) is not None:
        corner_lo, corner_hi, _ = sim._ntff
        # NTFF accumulation itself occurs on the fine-grid surface.  Permit it
        # in the same guarded full-x/y one-sided boundary envelope as the
        # time-domain public path, including the narrow closed-face/opposite-z
        # CPML lane.  This is a finite public-path claim only; external
        # open-boundary radiation accuracy remains a separate crossval gate.
        ntff_allowed = bool(guarded_boundary_allowed)
        if not ntff_allowed or region is None:
            issues.append(
                _issue(
                    "error",
                    "ntff_unvalidated",
                    "production subgrid validation supports NTFF/far-field "
                    "only when the NTFF box is entirely inside the guarded "
                    "one-sided full-x/y fine-grid envelope; CPML/UPML is "
                    "limited to the already-guarded opposite-z absorber lane",
                )
            )
        else:
            ordered = all(
                np.isfinite(lo) and np.isfinite(hi) and float(lo) < float(hi)
                for lo, hi in zip(corner_lo, corner_hi)
            )
            if not ordered:
                issues.append(
                    _issue(
                        "error",
                        "bad_ntff_box",
                        f"NTFF box corners must be finite and ordered, got "
                        f"{corner_lo!r} to {corner_hi!r}",
                    )
                )
            elif not (
                _position_inside_fine(corner_lo, sim, region)
                and _position_inside_fine(corner_hi, sim, region)
            ):
                issues.append(
                    _issue(
                        "error",
                        "ntff_box_outside_fine_grid",
                        f"NTFF box corners {corner_lo!r} to {corner_hi!r} "
                        "must be entirely inside the refined fine grid",
                    )
                )
            else:
                margin = min(
                    _artificial_interface_margin_fraction(
                        position,
                        sim,
                        physical_z_boundary,
                    )
                    for position in (corner_lo, corner_hi)
                )
                if margin + 1e-12 < boundary_margin_min_fraction:
                    issues.append(
                        _issue(
                            "error",
                            "ntff_box_margin_too_close",
                            f"NTFF box is only {margin:.3f} slab spans from "
                            "the remaining artificial z interface; validated "
                            "production boundary-terminated support requires "
                            f">= {boundary_margin_min_fraction:.2f}",
                        )
                    )
    if getattr(sim, "_dft_planes", None):
        issues.append(_issue("error", "dft_plane_unvalidated", "DFT planes are not validated with subgridding"))
    if getattr(sim, "_waveguide_ports", None):
        issues.append(_issue("error", "waveguide_port_unvalidated", "waveguide ports are not validated with subgridding"))
    if getattr(sim, "_floquet_ports", None):
        issues.append(_issue("error", "floquet_port_unvalidated", "Floquet ports are not validated with subgridding"))
    if getattr(sim, "_tfsf", None) is not None:
        issues.append(_issue("error", "tfsf_unvalidated", "TFSF is not validated with subgridding"))
    if getattr(sim, "_lumped_rlc", None):
        issues.append(_issue("error", "rlc_unvalidated", "lumped RLC is not validated with subgridding"))

    nonvacuum_static_material_names: list[str] = []
    static_material_allowed = (
        mode == "production"
        and not is_stage2_disjoint_topology
        and physical_z_boundary is not None
        and sim._boundary == "pec"
        and int(grid.cpml_layers) == 0
        and xy_margin is None
    )
    for name, spec in getattr(sim, "_materials", {}).items():
        is_static_nonvacuum = (
            abs(float(spec.eps_r) - 1.0) > 1e-12
            or abs(float(spec.sigma)) > 1e-18
            or abs(float(spec.mu_r) - 1.0) > 1e-12
        )
        if is_static_nonvacuum:
            nonvacuum_static_material_names.append(name)
            if not static_material_allowed:
                issues.append(
                    _issue(
                        "error",
                        "material_weighted_sat_missing",
                        f"material {name!r} changes eps/sigma/mu; production subgrid "
                        "material support is limited to the one-sided guarded "
                        "PEC/no-CPML boundary envelope. Use a uniform/non-uniform "
                        "mesh reference lane outside that envelope.",
                    )
                )
        if spec.debye_poles or spec.lorentz_poles or spec.chi3:
            issues.append(
                _issue(
                    "error",
                    "dispersive_or_nonlinear_material",
                    f"material {name!r} uses dispersion/nonlinearity not wired into the subgrid runner",
                )
            )

    if is_stage2_disjoint_topology and mode == "research":
        support_level = "research-stage2-disjoint-3d-public-contract"
    elif mode == "research":
        # Research mode reports the same issues but does not make warnings fatal.
        support_level = "research-experimental"
    elif is_stage2_disjoint_topology:
        support_level = "unsupported-production-stage2-disjoint-3d-integration-pending"
    elif nonvacuum_static_material_names and static_material_allowed:
        support_level = "production-z-slab-guarded-boundary-static-material-envelope"
    elif xy_margin is not None and physical_z_boundary is not None and local_xy_candidate_allowed:
        support_level = "production-local-xy-window-central-source-envelope"
    elif xy_margin is not None and physical_z_boundary is not None:
        support_level = "unsupported-production-local-xy-window-external-crossval-blocked"
    elif physical_z_boundary is not None:
        support_level = "production-z-slab-guarded-boundary-vacuum-envelope"
    else:
        support_level = "unsupported-production-z-slab"

    supported = not any(issue.severity == "error" for issue in issues)
    if supported and is_stage2_disjoint_topology:
        issues.append(
            _issue(
                "info",
                "stage2_disjoint_public_contract",
                "Stage-2 disjoint 3-D topology is available as a research "
                "public-contract selector with finite smoke-runner support only; "
                "production waveform gates, material-interface support, and "
                "external crossval are still required before support claims.",
            )
        )
    elif supported:
        issues.append(
            _issue(
                "info",
                "support_envelope",
                "validated guarded one-sided z-slab subgrid: source/probe "
                "lumped/wire impedance-port, and NTFF/far-field observables "
                "for full-x/y slabs, closed PEC touched z-boundary with no "
                "fine-grid CPML requirement, static material loads only when "
                "continuous across artificial interfaces, and no material/PEC "
                "jumps at artificial interfaces; local x/y support is limited "
                "to the central soft-source envelope",
            )
        )
    return SubgridValidationReport(
        supported=supported,
        mode=mode,
        support_level=support_level,
        region=region,
        issues=tuple(issues),
    )

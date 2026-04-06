"""Structured preflight diagnostics for agent-friendly optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import warnings

import jax
import jax.numpy as jnp

from rfx.auto_config import analyze_features


@dataclass(frozen=True)
class PreflightIssue:
    """Machine-readable preflight issue."""

    code: str
    severity: str
    message: str
    suggestions: tuple[str, ...] = ()


@dataclass
class PreflightReport:
    """Structured preflight report for optimization workflows."""

    optimizer_name: str
    objective_name: str
    n_steps: int
    n_steps_auto: bool
    grid_shape: tuple[int, int, int]
    cells: int
    trace_cost: int
    compiled_memory_mb: float | None = None
    compiled_memory_stats: dict | None = None
    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def strict_ok(self) -> bool:
        return self.ok and not any(issue.severity == "warning" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)

    def summary(self) -> str:
        lines = [
            f"{self.optimizer_name} preflight:",
            f"  objective = {self.objective_name}",
            f"  grid = {self.grid_shape} ({self.cells} cells)",
            f"  n_steps = {self.n_steps}" + (" [auto]" if self.n_steps_auto else ""),
            f"  trace_cost = {self.trace_cost:.3e}",
        ]
        if self.compiled_memory_mb is not None and self.compiled_memory_stats is not None:
            lines.append(
                "  compiled_grad_memory = "
                f"{self.compiled_memory_mb:.2f} MB "
                f"(args={self.compiled_memory_stats['argument_mb']:.2f}, "
                f"out={self.compiled_memory_stats['output_mb']:.2f}, "
                f"temp={self.compiled_memory_stats['temp_mb']:.2f})"
            )
        if not self.issues:
            lines.append("  issues = none")
        else:
            lines.append("  issues:")
            for issue in self.issues:
                lines.append(f"    - [{issue.severity}] {issue.code}: {issue.message}")
                for suggestion in issue.suggestions:
                    lines.append(f"      suggestion: {suggestion}")
        return "\n".join(lines)

    def emit_warnings(self) -> None:
        for issue in self.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, stacklevel=3)

    def enforce(self, mode: str = "guided") -> None:
        if mode not in {"guided", "strict"}:
            raise ValueError(f"preflight mode must be 'guided' or 'strict', got {mode!r}")

        errors = [issue for issue in self.issues if issue.severity == "error"]
        if errors:
            raise ValueError(self.summary())

        if mode == "strict" and self.has_warnings:
            raise ValueError(self.summary())

        if mode == "guided":
            self.emit_warnings()


def _objective_name(objective) -> str:
    return getattr(
        objective,
        "_rfx_name",
        getattr(objective, "__name__", type(objective).__name__),
    )


def _objective_accepts_ntff_box(objective) -> bool:
    try:
        return "ntff_box" in inspect.signature(objective).parameters
    except (TypeError, ValueError):
        return False


def _call_objective(objective, result):
    if _objective_accepts_ntff_box(objective):
        return objective(result, ntff_box=result.ntff_box)
    return objective(result)


def _geometry_material_map(sim) -> dict[str, object]:
    names = {entry.material_name for entry in getattr(sim, "_geometry", ())}
    return {name: sim._resolve_material(name) for name in names}


def _critical_dielectric_layer_issues(sim, grid) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    dz_min = float(min(sim._dz_profile)) if getattr(sim, "_dz_profile", None) is not None else float(grid.dx)
    pec_threshold = sim._PEC_SIGMA_THRESHOLD

    for entry in getattr(sim, "_geometry", ()):
        shape = entry.shape
        if type(shape).__name__ != "Box":
            continue
        try:
            corner_lo, corner_hi = shape.bounding_box()
        except NotImplementedError:
            continue

        dims = [abs(corner_hi[i] - corner_lo[i]) for i in range(3)]
        x_dim, y_dim, z_dim = dims
        if z_dim <= 1e-12:
            continue
        planar_dims = [d for d in (x_dim, y_dim) if d > 1e-12]
        if len(planar_dims) < 2:
            continue

        mat = sim._resolve_material(entry.material_name)
        if mat.sigma >= pec_threshold:
            continue

        planar_span = max(x_dim, y_dim)
        if planar_span <= 0.0 or z_dim > min(planar_dims):
            continue

        cells = z_dim / dz_min
        layer_desc = (
            f"Thin dielectric layer {entry.material_name!r} spans {z_dim * 1e3:.3f} mm "
            f"but is resolved by only {cells:.2f} z-cells."
        )
        if cells < 1.0:
            issues.append(
                PreflightIssue(
                    code="CRITICAL_LAYER_UNRESOLVED",
                    severity="error",
                    message=layer_desc,
                    suggestions=(
                        "Refine dz / dx so the layer spans at least one full cell.",
                        "Use non-uniform z meshing or a thicker effective layer model.",
                    ),
                )
            )
        elif cells < 4.0:
            issues.append(
                PreflightIssue(
                    code="CRITICAL_LAYER_POORLY_RESOLVED",
                    severity="warning",
                    message=layer_desc,
                    suggestions=(
                        "Aim for at least 4 z-cells across thin dielectric layers in the supported-safe lane.",
                    ),
                )
            )
    return issues


def _generic_feature_resolution_issues(sim, grid) -> list[PreflightIssue]:
    geometry = [(entry.shape, entry.material_name) for entry in getattr(sim, "_geometry", ())]
    if not geometry:
        return []

    issues: list[PreflightIssue] = []
    try:
        features = analyze_features(
            geometry,
            _geometry_material_map(sim),
            pec_threshold=sim._PEC_SIGMA_THRESHOLD,
        )
    except Exception:
        return []

    min_res = float(grid.dx)
    if getattr(sim, "_dz_profile", None) is not None:
        min_res = min(min_res, float(min(sim._dz_profile)))

    if features.min_thickness > 0:
        cells = features.min_thickness / max(min_res, 1e-30)
        if cells < 2.0:
            issues.append(
                PreflightIssue(
                    code="THIN_FEATURE_RISK",
                    severity="warning",
                    message=(
                        f"The thinnest feature spans only {cells:.2f} cells at the current mesh. "
                        "Generic thin-geometry fidelity may be poor."
                    ),
                    suggestions=(
                        "Refine dx or dz for thin features.",
                        "Treat results as exploratory unless convergence is checked.",
                    ),
                )
            )
    return issues


def _zero_thickness_pec_nonuniform_issue(sim) -> list[PreflightIssue]:
    if getattr(sim, "_dz_profile", None) is None:
        return []

    issues: list[PreflightIssue] = []
    pec_threshold = sim._PEC_SIGMA_THRESHOLD
    for entry in getattr(sim, "_geometry", ()):
        try:
            corner_lo, corner_hi = entry.shape.bounding_box()
        except NotImplementedError:
            continue
        dims = [abs(corner_hi[i] - corner_lo[i]) for i in range(3)]
        if not any(dim <= 1e-12 for dim in dims):
            continue
        mat = sim._resolve_material(entry.material_name)
        if mat.sigma < pec_threshold:
            continue
        issues.append(
            PreflightIssue(
                code="ZERO_THICKNESS_PEC_NONUNIFORM_UNSUPPORTED",
                severity="error",
                message=(
                    f"PEC geometry {entry.material_name!r} has zero thickness on at least one axis "
                    "while a non-uniform z mesh is active."
                ),
                suggestions=(
                    "Model PEC with finite physical thickness on non-uniform meshes.",
                    "Or disable the non-uniform mesh for this setup.",
                ),
            )
        )
        break
    return issues


def _cpml_clearance_issues(sim, grid) -> list[PreflightIssue]:
    if getattr(sim, "_boundary", None) != "cpml" or getattr(sim, "_cpml_layers", 0) <= 0:
        return []

    issues: list[PreflightIssue] = []
    domain = getattr(sim, "_domain", None)
    if domain is None:
        return issues

    axis_names = ("x", "y", "z")
    active_axes = [axis_names[i] for i, pad in enumerate(grid.axis_pads) if pad > 0]

    def _clearance_cells(position):
        clearances = []
        for axis in active_axes:
            idx = axis_names.index(axis)
            pos = position[idx]
            upper = domain[idx]
            if pos < 0.0 or pos > upper:
                return None
            clearances.append(min(pos / grid.dx, (upper - pos) / grid.dx))
        return min(clearances) if clearances else None

    for kind, entries in (("port", getattr(sim, "_ports", ())), ("probe", getattr(sim, "_probes", ()))):
        for entry in entries:
            clearance = _clearance_cells(entry.position)
            if clearance is None:
                issues.append(
                    PreflightIssue(
                        code=f"{kind.upper()}_OUTSIDE_DOMAIN",
                        severity="error",
                        message=(
                            f"{kind.capitalize()} at {entry.position!r} lies outside the physical domain "
                            f"{domain!r}."
                        ),
                        suggestions=("Move the measurement/excitation point inside the physical domain.",),
                    )
                )
                continue
            if clearance < 1.0:
                issues.append(
                    PreflightIssue(
                        code=f"{kind.upper()}_ON_CPML_INTERFACE",
                        severity="error",
                        message=(
                            f"{kind.capitalize()} at {entry.position!r} is effectively on the CPML-facing "
                            "physical boundary."
                        ),
                        suggestions=("Move it at least one full cell inward from the boundary.",),
                    )
                )
            elif clearance <= 4.0:
                issues.append(
                    PreflightIssue(
                        code=f"{kind.upper()}_NEAR_CPML_BOUNDARY",
                        severity="warning",
                        message=(
                            f"{kind.capitalize()} at {entry.position!r} is only {clearance:.1f} cells from a "
                            "CPML-backed boundary."
                        ),
                        suggestions=(
                            "Keep critical probes and excitations several cells away from CPML-facing boundaries.",
                        ),
                    )
                )
    return issues


def _physics_issues(sim, objective, *, grid) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    issues.extend(_cpml_clearance_issues(sim, grid))
    issues.extend(_zero_thickness_pec_nonuniform_issue(sim))
    issues.extend(_critical_dielectric_layer_issues(sim, grid))
    issues.extend(_generic_feature_resolution_issues(sim, grid))

    if getattr(sim, "_boundary", None) == "pec" and (
        getattr(sim, "_ntff", None) is not None or getattr(objective, "_rfx_requires_ntff", False)
    ):
        issues.append(
            PreflightIssue(
                code="PEC_BOUNDARY_WITH_NTFF_RISK",
                severity="warning",
                message=(
                    "This setup combines PEC boundaries with explicit NTFF usage. "
                    "That usually indicates a closed-cavity workflow rather than an open-radiation setup."
                ),
                suggestions=(
                    "Use boundary='cpml' for open-radiation / NTFF workflows.",
                ),
            )
        )

    if getattr(objective, "_rfx_prefers_loaded_port", False):
        ports = getattr(sim, "_ports", ())
        if ports and all(getattr(port, "impedance", 0.0) == 0.0 for port in ports):
            issues.append(
                PreflightIssue(
                    code="SOFT_SOURCE_OBJECTIVE_MISMATCH",
                    severity="warning",
                    message=(
                        f"Objective {_objective_name(objective)!r} is more reliable with impedance-loaded ports, "
                        "but the current setup uses only soft sources."
                    ),
                    suggestions=(
                        "Prefer sim.add_port(...) for port-like reflection/transmission workflows.",
                    ),
                )
            )

    return issues


def _clamped_region_issue(sim, grid, region) -> PreflightIssue | None:
    lo_idx = list(grid.position_to_index(region.corner_lo))
    hi_idx = list(grid.position_to_index(region.corner_hi))
    pads = (grid.pad_x, grid.pad_y, grid.pad_z)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo_idx[d] = max(lo_idx[d], pads[d])
        hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads[d])
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
    if any(size <= 0 for size in design_shape):
        return PreflightIssue(
            code="EMPTY_DESIGN_REGION",
            severity="error",
            message=(
                "Design region becomes empty after clamping to the non-CPML "
                "interior. Move the region away from the domain boundary."
            ),
            suggestions=(
                "Move corner_lo/corner_hi away from CPML.",
                "Increase the physical domain if the region must stay near an edge.",
            ),
        )
    return None


def _objective_issues(sim, objective, *, optimizer_name: str) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    objective_name = _objective_name(objective)

    if getattr(objective, "_rfx_requires_sparams", False):
        issues.append(
            PreflightIssue(
                code="OBJECTIVE_REQUIRES_SPARAMS",
                severity="error",
                message=(
                    f"{optimizer_name}() does not compute S-parameters in its "
                    f"differentiable forward pass, so objective "
                    f"{objective_name!r} is incompatible."
                ),
                suggestions=(
                    "Use Simulation.run(compute_s_params=True) for forward analysis.",
                    "Use time-domain proxy objectives for gradient optimization.",
                ),
            )
        )

    min_probes = getattr(objective, "_rfx_min_probes", None)
    if min_probes is not None:
        n_probes = len(getattr(sim, "_probes", ()))
        if n_probes < min_probes:
            hint = getattr(objective, "_rfx_probe_hint", "")
            issues.append(
                PreflightIssue(
                    code="MISSING_REQUIRED_PROBES",
                    severity="error",
                    message=(
                        f"{optimizer_name}() with objective {objective_name!r} "
                        f"requires at least {min_probes} probe(s), but this "
                        f"Simulation currently has {n_probes}.{hint}"
                    ),
                    suggestions=(
                        "Add the required probes with sim.add_probe(...).",
                    ),
                )
            )

    if getattr(objective, "_rfx_requires_ntff", False) and getattr(sim, "_ntff", None) is None:
        issues.append(
            PreflightIssue(
                code="MISSING_NTFF_BOX",
                severity="error",
                message=(
                    f"{optimizer_name}() with objective {objective_name!r} "
                    "requires an NTFF box."
                ),
                suggestions=("Call sim.add_ntff_box(...) before optimization.",),
            )
        )

    experimental_note = getattr(objective, "_rfx_experimental_note", None)
    if experimental_note:
        issues.append(
            PreflightIssue(
                code="EXPERIMENTAL_OBJECTIVE_LANE",
                severity="warning",
                message=experimental_note,
                suggestions=(
                    "Prefer the supported-safe proxy-objective lane first.",
                    "Use strict mode to block experimental optimizer paths in agentic workflows.",
                ),
            )
        )

    return issues


def _risk_issues(sim, objective, *, optimizer_name: str, grid, n_steps: int, n_steps_auto: bool) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    cells = int(grid.nx * grid.ny * grid.nz)
    trace_cost = int(cells * max(int(n_steps), 1))
    has_cpml = getattr(sim, "_boundary", None) == "cpml" and getattr(sim, "_cpml_layers", 0) > 0
    has_ntff = getattr(sim, "_ntff", None) is not None or getattr(objective, "_rfx_requires_ntff", False)

    if trace_cost >= 500_000_000 and (has_cpml or has_ntff or n_steps_auto):
        risk_factors: list[str] = []
        if n_steps_auto:
            risk_factors.append("auto-derived n_steps")
        if has_cpml:
            risk_factors.append("CPML")
        if has_ntff:
            risk_factors.append("NTFF")
        risk_summary = ", ".join(risk_factors) or "large trace"
        issues.append(
            PreflightIssue(
                code="MEMORY_RISK",
                severity="warning",
                message=(
                    f"{optimizer_name}() preflight detected a high reverse-mode "
                    f"trace cost (cells*n_steps={trace_cost:.3e}) with "
                    f"{risk_summary}. Large JAX/XLA optimizer traces can OOM "
                    "well above setup-time memory estimates."
                ),
                suggestions=(
                    "Reduce n_steps first.",
                    "Then simplify CPML/NTFF or coarsen the grid if needed.",
                ),
            )
        )

    if has_ntff and not jax.config.jax_enable_x64:
        issues.append(
            PreflightIssue(
                code="NTFF_FLOAT32_RISK",
                severity="warning",
                message=(
                    f"{optimizer_name}() is using NTFF while JAX_ENABLE_X64 is "
                    "disabled. Far-field-driven optimization is more reliable "
                    "in float64."
                ),
                suggestions=("Set JAX_ENABLE_X64=1 when NTFF accuracy matters.",),
            )
        )

    return issues


def _latent_to_eps_local(latent: jnp.ndarray, eps_min: float, eps_max: float) -> jnp.ndarray:
    return eps_min + (eps_max - eps_min) * jax.nn.sigmoid(latent)


def _compiled_memory_stats_or_unavailable(compiled) -> tuple[dict | None, str | None]:
    stats = compiled.memory_analysis()
    arg_mb = stats.argument_size_in_bytes / 1e6
    out_mb = stats.output_size_in_bytes / 1e6
    temp_mb = stats.temp_size_in_bytes / 1e6
    total_mb = arg_mb + out_mb + temp_mb

    if jax.default_backend() != "cpu" and total_mb <= 0.0:
        extra = ""
        if hasattr(compiled, "cost_analysis"):
            try:
                cost = compiled.cost_analysis() or {}
                bytes_accessed = float(cost.get("bytes accessed", 0.0))
                if bytes_accessed > 0.0:
                    extra = (
                        f" Cost analysis reported roughly {bytes_accessed / 1e6:.2f} MB "
                        "of bytes accessed, but this is not treated as a hard "
                        "memory gate."
                    )
            except Exception:
                pass
        return None, (
            "The current backend reports zero compile-time memory usage from "
            f"memory_analysis(), so budget enforcement is unavailable.{extra}"
        )

    return {
        "argument_mb": arg_mb,
        "output_mb": out_mb,
        "temp_mb": temp_mb,
        "total_mb": total_mb,
    }, None


def _compile_memory_gate_optimize(sim, region, objective, *, resolved_n_steps: int):
    from rfx.core.yee import MaterialArrays

    if getattr(objective, "_rfx_requires_ntff", False):
        return None, "NTFF objectives are experimental; compile-memory gating is unavailable for them."

    if not getattr(objective, "_rfx_supports_memory_gate", False):
        return None, "Compile-memory gating currently supports only tagged built-in proxy objectives."

    grid = sim._build_grid()
    lo_idx = list(grid.position_to_index(region.corner_lo))
    hi_idx = list(grid.position_to_index(region.corner_hi))
    pads = (grid.pad_x, grid.pad_y, grid.pad_z)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo_idx[d] = max(lo_idx[d], pads[d])
        hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads[d])
    lo_idx = tuple(lo_idx)
    hi_idx = tuple(hi_idx)
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
    if any(size <= 0 for size in design_shape):
        return None, "Design region is empty after interior clamping."

    base_materials, debye_spec, lorentz_spec, base_pec_mask, _, _ = sim._assemble_materials(grid)
    base_eps_r = base_materials.eps_r
    base_sigma = base_materials.sigma
    base_mu_r = base_materials.mu_r
    eps_min, eps_max = region.eps_range
    init_latent = jnp.zeros(design_shape, dtype=jnp.float32)

    def forward(latent):
        eps_design = _latent_to_eps_local(latent, eps_min, eps_max)
        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)
        materials = MaterialArrays(eps_r=eps_r, sigma=base_sigma, mu_r=base_mu_r)
        result = sim._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=resolved_n_steps,
            checkpoint=True,
            pec_mask=base_pec_mask,
        )
        return _call_objective(objective, result)

    compiled = jax.jit(jax.value_and_grad(forward)).lower(init_latent).compile()
    return _compiled_memory_stats_or_unavailable(compiled)


def _compile_memory_gate_topology(sim, design_region, objective, *, resolved_n_steps: int):
    from rfx.core.yee import MaterialArrays
    from rfx.topology import density_to_material_fields

    if getattr(objective, "_rfx_requires_ntff", False):
        return None, "NTFF objectives are experimental; compile-memory gating is unavailable for them."

    if not getattr(objective, "_rfx_supports_memory_gate", False):
        return None, "Compile-memory gating currently supports only tagged built-in proxy objectives."

    grid = sim._build_grid()
    lo_idx = list(grid.position_to_index(design_region.corner_lo))
    hi_idx = list(grid.position_to_index(design_region.corner_hi))
    pads = (grid.pad_x, grid.pad_y, grid.pad_z)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo_idx[d] = max(lo_idx[d], pads[d])
        hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads[d])
    lo_idx = tuple(lo_idx)
    hi_idx = tuple(hi_idx)
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
    if any(size <= 0 for size in design_shape):
        return None, "Design region is empty after interior clamping."

    mat_bg = sim._resolve_material(design_region.material_bg)
    mat_fg = sim._resolve_material(design_region.material_fg)
    eps_bg = mat_bg.eps_r
    eps_fg = mat_fg.eps_r
    sigma_bg = mat_bg.sigma
    sigma_fg = mat_fg.sigma
    pec_threshold = sim._PEC_SIGMA_THRESHOLD
    bg_is_pec = sigma_bg >= pec_threshold
    fg_is_pec = sigma_fg >= pec_threshold
    if bg_is_pec or fg_is_pec:
        return None, "Compile-memory gating is limited to non-PEC topology lanes."

    base_materials, debye_spec, lorentz_spec, base_pec_mask, _, _ = sim._assemble_materials(grid)
    base_eps_r = base_materials.eps_r
    base_sigma = base_materials.sigma
    base_mu_r = base_materials.mu_r
    init_logit = jnp.zeros(design_shape, dtype=jnp.float32)

    filt_r = design_region.effective_filter_radius
    filter_radius_cells = None if filt_r is None else (filt_r / grid.dx)
    if bg_is_pec or fg_is_pec:
        beta = 1.0
    else:
        beta = 1.0

    def forward(logit_param):
        rho = jax.nn.sigmoid(logit_param)
        fields = density_to_material_fields(
            rho,
            eps_bg,
            eps_fg,
            filter_radius_cells=filter_radius_cells,
            beta=beta,
            sigma_bg=sigma_bg,
            sigma_fg=sigma_fg,
            pec_bg=bg_is_pec,
            pec_fg=fg_is_pec,
        )
        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.eps)
        sigma = base_sigma.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.sigma)
        materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=base_mu_r)
        pec_occupancy = None
        if fields.pec_occupancy is not None:
            pec_occupancy = jnp.zeros(grid.shape, dtype=jnp.float32)
            pec_occupancy = pec_occupancy.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.pec_occupancy)
        result = sim._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=resolved_n_steps,
            checkpoint=True,
            pec_mask=base_pec_mask,
            pec_occupancy=pec_occupancy,
        )
        return _call_objective(objective, result)

    compiled = jax.jit(jax.value_and_grad(forward)).lower(init_logit).compile()
    return _compiled_memory_stats_or_unavailable(compiled)


def build_optimize_preflight_report(
    sim,
    region,
    objective,
    *,
    optimizer_name: str = "optimize",
    n_steps: int | None = None,
    num_periods: float = 20.0,
    memory_budget_mb: float | None = None,
) -> PreflightReport:
    grid = sim._build_grid()
    resolved_n_steps = n_steps if n_steps is not None else grid.num_timesteps(num_periods=num_periods)
    issues = _objective_issues(sim, objective, optimizer_name=optimizer_name)
    issues.extend(_physics_issues(sim, objective, grid=grid))
    region_issue = _clamped_region_issue(sim, grid, region)
    if region_issue is not None:
        issues.append(region_issue)
    issues.extend(
        _risk_issues(
            sim,
            objective,
            optimizer_name=optimizer_name,
            grid=grid,
            n_steps=resolved_n_steps,
            n_steps_auto=n_steps is None,
        )
    )
    cells = int(grid.nx * grid.ny * grid.nz)
    compiled_stats = None
    has_hard_errors = any(issue.severity == "error" for issue in issues)
    if memory_budget_mb is not None and not has_hard_errors:
        try:
            compiled_stats, reason = _compile_memory_gate_optimize(
                sim,
                region,
                objective,
                resolved_n_steps=resolved_n_steps,
            )
            if compiled_stats is None:
                issues.append(
                    PreflightIssue(
                        code="COMPILED_MEMORY_CHECK_UNAVAILABLE",
                        severity="error",
                        message=(
                            "A memory budget was requested, but compile-time "
                            f"gradient memory analysis is unavailable: {reason}"
                        ),
                        suggestions=(
                            "Use a tagged built-in supported-safe proxy objective.",
                            "Avoid explicit memory budgets for experimental objective paths.",
                        ),
                    )
                )
            elif compiled_stats["total_mb"] > memory_budget_mb:
                issues.append(
                    PreflightIssue(
                        code="COMPILED_MEMORY_BUDGET_EXCEEDED",
                        severity="error",
                        message=(
                            f"Compiled gradient memory estimate is "
                            f"{compiled_stats['total_mb']:.2f} MB, exceeding "
                            f"the requested budget of {memory_budget_mb:.2f} MB."
                        ),
                        suggestions=(
                            "Reduce n_steps first.",
                            "Then coarsen the grid or simplify the design region.",
                        ),
                    )
                )
        except Exception as exc:
            issues.append(
                PreflightIssue(
                    code="COMPILED_MEMORY_CHECK_FAILED",
                    severity="error",
                    message=(
                        "A memory budget was requested, but compile-time "
                        f"gradient memory analysis failed: {type(exc).__name__}: {exc}"
                    ),
                    suggestions=(
                        "Try a smaller supported-safe problem first to reduce compile pressure.",
                        "If this persists on a supported-safe lane, treat it as a toolchain/backend limitation rather than an objective-lane guarantee.",
                    ),
                )
            )
    return PreflightReport(
        optimizer_name=optimizer_name,
        objective_name=_objective_name(objective),
        n_steps=resolved_n_steps,
        n_steps_auto=n_steps is None,
        grid_shape=grid.shape,
        cells=cells,
        trace_cost=int(cells * max(int(resolved_n_steps), 1)),
        compiled_memory_mb=(compiled_stats["total_mb"] if compiled_stats is not None else None),
        compiled_memory_stats=compiled_stats,
        issues=issues,
    )


def build_topology_preflight_report(
    sim,
    design_region,
    objective,
    *,
    optimizer_name: str = "topology_optimize",
    n_steps: int | None = None,
    num_periods: float = 20.0,
    beta_schedule=None,
    memory_budget_mb: float | None = None,
) -> PreflightReport:
    grid = sim._build_grid()
    resolved_n_steps = n_steps if n_steps is not None else grid.num_timesteps(num_periods=num_periods)
    issues = _objective_issues(sim, objective, optimizer_name=optimizer_name)
    issues.extend(_physics_issues(sim, objective, grid=grid))
    region_issue = _clamped_region_issue(sim, grid, design_region)
    if region_issue is not None:
        issues.append(region_issue)

    try:
        mat_bg = sim._resolve_material(design_region.material_bg)
        mat_fg = sim._resolve_material(design_region.material_fg)
        pec_threshold = sim._PEC_SIGMA_THRESHOLD
        if mat_bg.sigma >= pec_threshold or mat_fg.sigma >= pec_threshold:
            suggestions = ["Start with dielectric topology first if you need a more stable baseline."]
            if beta_schedule is None:
                suggestions.append("A gentler PEC-safe beta schedule will be used by default.")
            issues.append(
                PreflightIssue(
                    code="PEC_TOPOLOGY_EXPERIMENTAL",
                    severity="warning",
                    message=(
                        "PEC participates in the topology design region. PEC "
                        "topology remains experimental and can still show flat-"
                        "loss behavior on difficult problems."
                    ),
                    suggestions=tuple(suggestions),
                )
            )
    except Exception:
        pass

    issues.extend(
        _risk_issues(
            sim,
            objective,
            optimizer_name=optimizer_name,
            grid=grid,
            n_steps=resolved_n_steps,
            n_steps_auto=n_steps is None,
        )
    )
    cells = int(grid.nx * grid.ny * grid.nz)
    compiled_stats = None
    has_hard_errors = any(issue.severity == "error" for issue in issues)
    if memory_budget_mb is not None and not has_hard_errors:
        try:
            compiled_stats, reason = _compile_memory_gate_topology(
                sim,
                design_region,
                objective,
                resolved_n_steps=resolved_n_steps,
            )
            if compiled_stats is None:
                issues.append(
                    PreflightIssue(
                        code="COMPILED_MEMORY_CHECK_UNAVAILABLE",
                        severity="error",
                        message=(
                            "A memory budget was requested, but compile-time "
                            f"gradient memory analysis is unavailable: {reason}"
                        ),
                        suggestions=(
                            "Use a tagged built-in supported-safe proxy objective.",
                            "Avoid explicit memory budgets for experimental objective paths.",
                        ),
                    )
                )
            elif compiled_stats["total_mb"] > memory_budget_mb:
                issues.append(
                    PreflightIssue(
                        code="COMPILED_MEMORY_BUDGET_EXCEEDED",
                        severity="error",
                        message=(
                            f"Compiled gradient memory estimate is "
                            f"{compiled_stats['total_mb']:.2f} MB, exceeding "
                            f"the requested budget of {memory_budget_mb:.2f} MB."
                        ),
                        suggestions=(
                            "Reduce n_steps first.",
                            "Then coarsen the grid or shrink the design region.",
                        ),
                    )
                )
        except Exception as exc:
            issues.append(
                PreflightIssue(
                    code="COMPILED_MEMORY_CHECK_FAILED",
                    severity="error",
                    message=(
                        "A memory budget was requested, but compile-time "
                        f"gradient memory analysis failed: {type(exc).__name__}: {exc}"
                    ),
                    suggestions=(
                        "Try a smaller supported-safe problem first to reduce compile pressure.",
                        "If this persists on a supported-safe lane, treat it as a toolchain/backend limitation rather than an objective-lane guarantee.",
                    ),
                )
            )
    return PreflightReport(
        optimizer_name=optimizer_name,
        objective_name=_objective_name(objective),
        n_steps=resolved_n_steps,
        n_steps_auto=n_steps is None,
        grid_shape=grid.shape,
        cells=cells,
        trace_cost=int(cells * max(int(resolved_n_steps), 1)),
        compiled_memory_mb=(compiled_stats["total_mb"] if compiled_stats is not None else None),
        compiled_memory_stats=compiled_stats,
        issues=issues,
    )

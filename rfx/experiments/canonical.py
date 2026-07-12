"""Canonical multi-workflow ExperimentSpec v2 and deterministic compiler.

The v2 contract is the shared semantic boundary for Studio, generated Python,
MCP tools, and solver workers.  It intentionally supports only the three P0
golden workflow lanes; unsupported variants fail with stable coded diagnostics.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import math
import re
from typing import Any, Callable, Mapping


CANONICAL_SCHEMA_VERSION = "rfx-experiment/v2"
CANONICAL_COMPILED_VERSION = "rfx-compiled-experiment/v2"
SCENE_PREVIEW_VERSION = "rfx-scene-preview/v1"
SUPPORTED_WORKFLOWS = frozenset(
    {"patch_antenna", "wr90_waveguide", "multilayer_fresnel"}
)
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class CompileDiagnostic:
    code: str
    severity: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }


class CanonicalSpecError(ValueError):
    """A coded parse/semantic error in a canonical experiment document."""

    def __init__(self, code: str, path: str, message: str):
        super().__init__(f"[{code}] {path}: {message}")
        self.code = code
        self.path = path
        self.message = message

    def to_diagnostic(self) -> CompileDiagnostic:
        return CompileDiagnostic(self.code, "error", self.path, self.message)


class ExperimentCompileError(ValueError):
    """Compilation failure carrying one or more structured diagnostics."""

    def __init__(self, diagnostics: list[CompileDiagnostic]):
        if not diagnostics:
            raise ValueError("ExperimentCompileError requires diagnostics")
        self.diagnostics = tuple(diagnostics)
        super().__init__(
            "; ".join(
                f"[{item.code}] {item.path}: {item.message}" for item in diagnostics
            )
        )


@dataclass(frozen=True)
class CanonicalExperimentSpec:
    """Validated, JSON-backed immutable v2 experiment value object."""

    _document: dict[str, Any]

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "CanonicalExperimentSpec":
        migrated = migrate_canonical_spec(document)
        normalized = _json_copy(migrated)
        _validate_document(normalized)
        return cls(normalized)

    @classmethod
    def from_json(cls, text: str) -> "CanonicalExperimentSpec":
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CanonicalSpecError("invalid_json", "$", str(exc)) from exc
        if not isinstance(document, dict):
            raise CanonicalSpecError("type_error", "$", "document must be an object")
        return cls.from_dict(document)

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self._document)

    def canonical_json(self) -> str:
        return _canonical_json(self._document)

    @property
    def sha256(self) -> str:
        return _sha256_text(self.canonical_json())

    @property
    def workflow(self) -> str:
        return self._document["kind"]

    @property
    def metadata(self) -> dict[str, Any]:
        return _json_copy(self._document["metadata"])

    @property
    def semantic_plan(self) -> dict[str, Any]:
        return {
            key: _json_copy(self._document[key])
            for key in (
                "kind",
                "simulation",
                "materials",
                "geometry",
                "excitations",
                "observations",
                "boundaries",
                "execution",
                "study",
                "validation",
                "artifacts",
            )
        }

    @property
    def semantic_fingerprint(self) -> str:
        return _sha256_text(_canonical_json(self.semantic_plan))


@dataclass(frozen=True)
class CanonicalCompiledExperiment:
    spec: CanonicalExperimentSpec
    plan: dict[str, Any]
    generated_python: str
    diagnostics: tuple[CompileDiagnostic, ...]
    sha256: str
    semantic_fingerprint: str

    @property
    def config(self) -> dict[str, Any]:
        """Compatibility alias used by the durable service persistence layer."""

        return _json_copy(self.plan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_COMPILED_VERSION,
            "spec_sha256": self.spec.sha256,
            "compiled_sha256": self.sha256,
            "semantic_fingerprint": self.semantic_fingerprint,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "plan": _json_copy(self.plan),
        }

    def build_simulation(self):
        simulation = _build_simulation(self.spec)
        simulation._experiment_spec_sha256 = self.spec.sha256
        simulation._experiment_semantic_fingerprint = self.semantic_fingerprint
        return simulation

    def preflight(self) -> dict[str, Any]:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report = self.build_simulation().preflight(strict=False)
        return report.to_dict()

    def scene_preview(self) -> dict[str, Any]:
        return build_scene_preview(self.spec)

    def execute(self):
        simulation = self.build_simulation()
        document = self.spec._document
        execution = document["execution"]
        if self.spec.workflow == "wr90_waveguide":
            observation = _one(document["observations"], "waveguide_sparameters")
            matrix = simulation.compute_waveguide_s_matrix(
                n_steps=execution["n_steps"],
                normalize=observation["normalize"],
            )
            first_excitation = document["excitations"][0]["id"]
            field_simulation = _build_simulation(
                self.spec, excitation_ids={first_excitation}
            )
            field_result = field_simulation.run(
                n_steps=execution["n_steps"], compute_s_params=False
            )
            return WorkflowExecutionResult(
                workflow=self.spec.workflow,
                state=field_result.state,
                time_series=field_result.time_series,
                s_params=matrix.s_params,
                freqs=matrix.freqs,
                grid=field_result.grid,
                port_names=matrix.port_names,
            )
        if self.spec.workflow == "patch_antenna":
            observation = _one(document["observations"], "sparameters")
            import jax.numpy as jnp

            frequencies = jnp.linspace(
                float(observation["start_hz"]),
                float(observation["stop_hz"]),
                observation["points"],
            )
            return simulation.run(
                n_steps=execution["n_steps"],
                compute_s_params=True,
                s_param_freqs=frequencies,
                s_param_n_steps=execution["s_param_n_steps"],
            )
        device = simulation.run(n_steps=execution["n_steps"], compute_s_params=False)
        reference = _build_simulation(self.spec, include_geometry=False).run(
            n_steps=execution["n_steps"], compute_s_params=False
        )
        return WorkflowExecutionResult(
            workflow=self.spec.workflow,
            state=device.state,
            time_series=device.time_series,
            s_params=None,
            freqs=None,
            grid=device.grid,
            reflection_transmission=_fresnel_reflection_transmission(
                device, reference, document
            ),
        )


@dataclass(frozen=True)
class WorkflowExecutionResult:
    """Workflow-neutral worker result with a field state and primary observable."""

    workflow: str
    state: Any
    time_series: Any
    s_params: Any | None
    freqs: Any | None
    grid: Any
    port_names: tuple[str, ...] | None = None
    reflection_transmission: dict[str, Any] | None = None


def compile_canonical_experiment(
    document: CanonicalExperimentSpec | Mapping[str, Any],
) -> CanonicalCompiledExperiment:
    try:
        spec = (
            document
            if isinstance(document, CanonicalExperimentSpec)
            else CanonicalExperimentSpec.from_dict(document)
        )
    except CanonicalSpecError as exc:
        raise ExperimentCompileError([exc.to_diagnostic()]) from exc

    plan = spec.semantic_plan
    semantic = spec.semantic_fingerprint
    compiled_sha = _sha256_text(_canonical_json(plan))
    diagnostics: list[CompileDiagnostic] = []
    if spec.metadata.get("fidelity") == "structural-cpu-smoke":
        diagnostics.append(
            CompileDiagnostic(
                "structural_smoke_only",
                "warning",
                "$.metadata.fidelity",
                "This fixture proves lifecycle behavior and is not quantitative RF evidence.",
            )
        )
    generated = _generate_python(spec, semantic, compiled_sha)
    return CanonicalCompiledExperiment(
        spec=spec,
        plan=plan,
        generated_python=generated,
        diagnostics=tuple(diagnostics),
        sha256=compiled_sha,
        semantic_fingerprint=semantic,
    )


def build_scene_preview(
    document: CanonicalExperimentSpec | Mapping[str, Any],
) -> dict[str, Any]:
    spec = (
        document
        if isinstance(document, CanonicalExperimentSpec)
        else CanonicalExperimentSpec.from_dict(document)
    )
    source = spec._document
    entities = []
    for item in source["geometry"]:
        entities.append(
            {
                "id": item["id"],
                "role": "geometry",
                "kind": item["kind"],
                "material_id": item["material_id"],
                "bounds_m": _json_copy(item["bounds_m"]),
            }
        )
    overlays = []
    for item in source["excitations"]:
        overlay = {"id": item["id"], "role": "excitation", "kind": item["kind"]}
        for field in ("position_m", "extent_m", "direction", "x_position_m"):
            if field in item:
                overlay[field] = _json_copy(item[field])
        overlays.append(overlay)
    for item in source["observations"]:
        overlay = {"id": item["id"], "role": "observation", "kind": item["kind"]}
        for field in ("position_m", "coordinate_m", "axis"):
            if field in item:
                overlay[field] = _json_copy(item[field])
        overlays.append(overlay)
    return {
        "schema_version": SCENE_PREVIEW_VERSION,
        "spec_sha256": spec.sha256,
        "semantic_fingerprint": spec.semantic_fingerprint,
        "workflow": spec.workflow,
        "domain_m": _json_copy(source["simulation"]["domain_m"]),
        "entities": entities,
        "overlays": overlays,
    }


def simulation_semantic_fingerprint(simulation: Any) -> str:
    value = getattr(simulation, "_experiment_semantic_fingerprint", None)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("simulation has no canonical experiment semantic fingerprint")
    return value


def migrate_canonical_spec(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise CanonicalSpecError("type_error", "$", "document must be an object")
    version = document.get("schema_version")
    if version == CANONICAL_SCHEMA_VERSION:
        return dict(document)
    migration = _MIGRATIONS.get(str(version))
    if migration is None:
        code = (
            "missing_schema_version"
            if version is None
            else "unsupported_schema_version"
        )
        raise CanonicalSpecError(
            code,
            "$.schema_version",
            f"expected {CANONICAL_SCHEMA_VERSION!r}, got {version!r}",
        )
    return migration(document)


def _migrate_v1_patch(document: Mapping[str, Any]) -> dict[str, Any]:
    from .spec import ExperimentSpec

    legacy = ExperimentSpec.from_dict(document)
    model = legacy.model
    margin = (model.cpml_layers + 1) * model.cell_size_m
    z_ground_lo = model.stack_base_z_m or margin
    z_ground_hi = z_ground_lo + model.ground.thickness_m
    z_substrate_hi = z_ground_hi + model.substrate.thickness_m
    z_patch_hi = z_substrate_hi + model.patch.thickness_m
    center_x, center_y = model.domain_m[0] / 2, model.domain_m[1] / 2

    def bounds(
        size: tuple[float, float], z_lo: float, z_hi: float
    ) -> list[list[float]]:
        return [
            [center_x - size[0] / 2, center_y - size[1] / 2, z_lo],
            [center_x + size[0] / 2, center_y + size[1] / 2, z_hi],
        ]

    clearance = min(model.substrate.thickness_m * 0.1, model.cell_size_m * 0.1)
    feed_x = center_x - model.patch.size_m[0] / 2 + model.feed.inset_m
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "kind": "patch_antenna",
        "metadata": {
            "id": _slug(legacy.name),
            "title": legacy.name,
            "description": "Migrated patch-antenna experiment",
            "tags": ["migrated-v1"],
            "author": "unknown",
            "parent_revision": None,
            **legacy.metadata,
        },
        "simulation": {
            "dimensionality": "3d",
            "domain_m": list(model.domain_m),
            "cell_size_m": model.cell_size_m,
            "freq_max_hz": model.frequency_sweep.stop_hz,
            "precision": legacy.execution.precision,
        },
        "materials": [
            {
                "id": "substrate",
                "kind": "isotropic",
                "relative_permittivity": model.substrate.relative_permittivity,
                "conductivity_s_per_m": model.substrate.conductivity_s_per_m,
                "provenance": "migrated-v1",
            }
        ],
        "geometry": [
            {
                "id": "ground",
                "kind": "box",
                "material_id": "pec",
                "bounds_m": bounds(model.ground.size_m, z_ground_lo, z_ground_hi),
            },
            {
                "id": "substrate",
                "kind": "box",
                "material_id": "substrate",
                "bounds_m": bounds(model.substrate.size_m, z_ground_hi, z_substrate_hi),
            },
            {
                "id": "patch",
                "kind": "box",
                "material_id": "pec",
                "bounds_m": bounds(model.patch.size_m, z_substrate_hi, z_patch_hi),
            },
        ],
        "excitations": [
            {
                "id": "feed",
                "kind": "lumped_port",
                "position_m": [feed_x, center_y, z_ground_hi + clearance],
                "component": "ez",
                "impedance_ohm": model.feed.impedance_ohm,
                "extent_m": model.substrate.thickness_m - 2 * clearance,
                "direction": "+x",
                "f0_hz": model.design_frequency_hz,
                "bandwidth": model.feed.pulse_bandwidth,
            }
        ],
        "observations": [
            {
                "id": "s11",
                "kind": "sparameters",
                "start_hz": model.frequency_sweep.start_hz,
                "stop_hz": model.frequency_sweep.stop_hz,
                "points": model.frequency_sweep.points,
            }
        ],
        "boundaries": {
            "x": {"lo": "cpml", "hi": "cpml"},
            "y": {"lo": "cpml", "hi": "cpml"},
            "z": {"lo": "cpml", "hi": "cpml"},
            "cpml_layers": model.cpml_layers,
        },
        "execution": {
            "backend": "cpu",
            "n_steps": legacy.execution.n_steps,
            "s_param_n_steps": legacy.execution.s_param_n_steps,
            "timeout_seconds": 3600,
        },
        "study": {"kind": "single"},
        "validation": {
            "support_lane": "patch-antenna",
            "required_checks": ["preflight"],
            "metrics": [],
        },
        "artifacts": {
            "save": ["spec", "generated-python", "scene", "preflight", "events", "s11"]
        },
    }


_MIGRATIONS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "rfx-experiment/v0": _migrate_v1_patch,
    "rfx-experiment/v1": _migrate_v1_patch,
}


def _build_simulation(
    spec: CanonicalExperimentSpec,
    *,
    excitation_ids: set[str] | None = None,
    include_geometry: bool = True,
):
    import jax.numpy as jnp

    from rfx import Box, GaussianPulse, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    document = spec._document
    simulation_cfg = document["simulation"]
    boundaries = document["boundaries"]
    boundary_spec = BoundarySpec(
        x=Boundary(**boundaries["x"]),
        y=Boundary(**boundaries["y"]),
        z=Boundary(**boundaries["z"]),
    )
    simulation = Simulation(
        freq_max=simulation_cfg["freq_max_hz"],
        domain=tuple(simulation_cfg["domain_m"]),
        dx=simulation_cfg["cell_size_m"],
        mode=simulation_cfg["dimensionality"],
        precision=simulation_cfg["precision"],
        boundary=boundary_spec,
        cpml_layers=boundaries["cpml_layers"],
    )
    for material in document["materials"]:
        simulation.add_material(
            material["id"],
            eps_r=material["relative_permittivity"],
            sigma=material["conductivity_s_per_m"],
        )
    if include_geometry:
        for geometry in document["geometry"]:
            simulation.add(
                Box(tuple(geometry["bounds_m"][0]), tuple(geometry["bounds_m"][1])),
                material=geometry["material_id"],
            )
    for excitation in document["excitations"]:
        if excitation_ids is not None and excitation["id"] not in excitation_ids:
            continue
        kind = excitation["kind"]
        if kind == "lumped_port":
            simulation.add_port(
                tuple(excitation["position_m"]),
                excitation["component"],
                impedance=excitation["impedance_ohm"],
                extent=excitation["extent_m"],
                direction=excitation["direction"],
                waveform=GaussianPulse(
                    f0=excitation["f0_hz"], bandwidth=excitation["bandwidth"]
                ),
            )
        elif kind == "waveguide_port":
            simulation.add_waveguide_port(
                excitation["x_position_m"],
                direction=excitation["direction"],
                mode=tuple(excitation["mode"]),
                mode_type=excitation["mode_type"],
                freqs=jnp.linspace(
                    float(excitation["start_hz"]),
                    float(excitation["stop_hz"]),
                    excitation["points"],
                ),
                f0=excitation["f0_hz"],
                bandwidth=excitation["bandwidth"],
                reference_plane=excitation["reference_plane_m"],
                name=excitation["id"],
            )
        elif kind == "tfsf_plane_wave":
            simulation.add_tfsf_source(
                f0=excitation["f0_hz"],
                bandwidth=excitation["bandwidth"],
                amplitude=excitation["amplitude"],
                margin=excitation["margin_cells"],
                polarization=excitation["polarization"],
                direction=excitation["direction"],
                waveform=excitation["waveform"],
            )
    for observation in document["observations"]:
        kind = observation["kind"]
        if kind == "point_probe":
            simulation.add_probe(
                tuple(observation["position_m"]), observation["component"]
            )
        elif kind in {"dft_plane", "flux_monitor"}:
            frequencies = jnp.linspace(
                float(observation["start_hz"]),
                float(observation["stop_hz"]),
                observation["points"],
            )
            if kind == "dft_plane":
                simulation.add_dft_plane_probe(
                    axis=observation["axis"],
                    coordinate=observation["coordinate_m"],
                    component=observation["component"],
                    freqs=frequencies,
                    name=observation["id"],
                )
            else:
                simulation.add_flux_monitor(
                    axis=observation["axis"],
                    coordinate=observation["coordinate_m"],
                    freqs=frequencies,
                    name=observation["id"],
                )
    return simulation


def _fresnel_reflection_transmission(device, reference, document) -> dict[str, Any]:
    import numpy as np

    from rfx.probes.probes import flux_spectrum

    flux_observations = [
        item for item in document["observations"] if item["kind"] == "flux_monitor"
    ]
    if len(flux_observations) != 2:
        raise ValueError("multilayer Fresnel workflow requires two flux monitors")
    reflection_observation, transmission_observation = flux_observations
    device_reflection = device.flux_monitors[reflection_observation["id"]]
    device_transmission = device.flux_monitors[transmission_observation["id"]]
    reference_reflection = reference.flux_monitors[reflection_observation["id"]]
    reference_transmission = reference.flux_monitors[transmission_observation["id"]]
    incident_left = np.abs(np.asarray(flux_spectrum(reference_reflection), dtype=float))
    incident_right = np.abs(
        np.asarray(flux_spectrum(reference_transmission), dtype=float)
    )
    device_left = np.asarray(flux_spectrum(device_reflection), dtype=float)
    device_right = np.asarray(flux_spectrum(device_transmission), dtype=float)
    scale = max(float(np.max(incident_left)), float(np.max(incident_right)), 1e-30)
    floor = scale * 1e-9
    reflection = np.abs(incident_left - np.abs(device_left)) / np.maximum(
        incident_left, floor
    )
    transmission = np.abs(device_right) / np.maximum(incident_right, floor)
    frequencies = np.asarray(reference_reflection.freqs, dtype=float)

    slab = document["geometry"][0]
    material = next(
        item for item in document["materials"] if item["id"] == slab["material_id"]
    )
    thickness = float(slab["bounds_m"][1][0] - slab["bounds_m"][0][0])
    refractive_index = math.sqrt(float(material["relative_permittivity"]))
    delta = 2 * np.pi * frequencies * refractive_index * thickness / 299_792_458.0
    cosine = np.cos(delta)
    sine = np.sin(delta)
    denominator = (
        cosine + 1j * sine / refractive_index + 1j * refractive_index * sine + cosine
    )
    analytic_reflection_amplitude = (
        cosine + 1j * sine / refractive_index - 1j * refractive_index * sine - cosine
    ) / denominator
    analytic_transmission_amplitude = 2.0 / denominator
    analytic_reflection = np.abs(analytic_reflection_amplitude) ** 2
    analytic_transmission = np.abs(analytic_transmission_amplitude) ** 2
    return {
        "frequencies_hz": frequencies.tolist(),
        "reflection": reflection.tolist(),
        "transmission": transmission.tolist(),
        "analytic_reflection": analytic_reflection.tolist(),
        "analytic_transmission": analytic_transmission.tolist(),
        "signal_valid": ((incident_left > floor) & (incident_right > floor)).tolist(),
    }


def _validate_document(document: dict[str, Any]) -> None:
    _object_keys(
        document,
        "$",
        required={
            "schema_version",
            "kind",
            "metadata",
            "simulation",
            "materials",
            "geometry",
            "excitations",
            "observations",
            "boundaries",
            "execution",
            "study",
            "validation",
            "artifacts",
        },
    )
    if document["schema_version"] != CANONICAL_SCHEMA_VERSION:
        _fail("unsupported_schema_version", "$.schema_version", "unsupported version")
    if document["kind"] not in SUPPORTED_WORKFLOWS:
        _fail(
            "unsupported_workflow",
            "$.kind",
            f"supported: {sorted(SUPPORTED_WORKFLOWS)}",
        )

    metadata = _object_keys(
        document["metadata"],
        "$.metadata",
        required={"id", "title", "description", "tags", "author", "parent_revision"},
        optional={"fidelity", "claims"},
    )
    _identifier(metadata["id"], "$.metadata.id")
    for field in ("title", "description", "author"):
        _nonempty_text(metadata[field], f"$.metadata.{field}")
    _string_list(metadata["tags"], "$.metadata.tags")
    if metadata["parent_revision"] is not None:
        _nonempty_text(metadata["parent_revision"], "$.metadata.parent_revision")

    simulation = _object_keys(
        document["simulation"],
        "$.simulation",
        required={
            "dimensionality",
            "domain_m",
            "cell_size_m",
            "freq_max_hz",
            "precision",
        },
    )
    if simulation["dimensionality"] not in {"3d", "2d_tmz"}:
        _fail(
            "unsupported_variant",
            "$.simulation.dimensionality",
            "expected '3d' or '2d_tmz'",
        )
    domain = _number_list(simulation["domain_m"], "$.simulation.domain_m", length=3)
    if any(value <= 0 for value in domain):
        _fail("range_error", "$.simulation.domain_m", "all dimensions must be > 0")
    _positive(simulation["cell_size_m"], "$.simulation.cell_size_m")
    _positive(simulation["freq_max_hz"], "$.simulation.freq_max_hz")
    if simulation["precision"] not in {"float32", "mixed"}:
        _fail(
            "unsupported_variant",
            "$.simulation.precision",
            "expected 'float32' or 'mixed'",
        )

    material_ids = {"pec"}
    materials = _list(document["materials"], "$.materials")
    for index, raw in enumerate(materials):
        path = f"$.materials[{index}]"
        item = _object_keys(
            raw,
            path,
            required={
                "id",
                "kind",
                "relative_permittivity",
                "conductivity_s_per_m",
                "provenance",
            },
        )
        _identifier(item["id"], f"{path}.id")
        if item["id"] in material_ids:
            _fail("duplicate_id", f"{path}.id", "material id must be unique")
        material_ids.add(item["id"])
        if item["kind"] != "isotropic":
            _fail(
                "unsupported_variant", f"{path}.kind", "P0 supports isotropic materials"
            )
        if (
            _positive(item["relative_permittivity"], f"{path}.relative_permittivity")
            < 1
        ):
            _fail("range_error", f"{path}.relative_permittivity", "must be >= 1")
        if _number(item["conductivity_s_per_m"], f"{path}.conductivity_s_per_m") < 0:
            _fail("range_error", f"{path}.conductivity_s_per_m", "must be >= 0")
        _nonempty_text(item["provenance"], f"{path}.provenance")

    object_ids: set[str] = set()
    for index, raw in enumerate(_list(document["geometry"], "$.geometry")):
        path = f"$.geometry[{index}]"
        item = _object_keys(
            raw, path, required={"id", "kind", "material_id", "bounds_m"}
        )
        _unique_object_id(item["id"], f"{path}.id", object_ids)
        if item["kind"] != "box":
            _fail("unsupported_variant", f"{path}.kind", "P0 supports box geometry")
        if item["material_id"] not in material_ids:
            _fail("unknown_reference", f"{path}.material_id", "unknown material id")
        bounds = _list(item["bounds_m"], f"{path}.bounds_m")
        if len(bounds) != 2:
            _fail("shape_error", f"{path}.bounds_m", "expected [lo, hi]")
        lo = _number_list(bounds[0], f"{path}.bounds_m[0]", length=3)
        hi = _number_list(bounds[1], f"{path}.bounds_m[1]", length=3)
        if any(
            lo[axis] < 0 or hi[axis] > domain[axis] or lo[axis] >= hi[axis]
            for axis in range(3)
        ):
            _fail(
                "geometry_out_of_domain",
                f"{path}.bounds_m",
                "box must fit inside domain with positive extent",
            )

    excitation_kinds = []
    for index, raw in enumerate(_list(document["excitations"], "$.excitations")):
        path = f"$.excitations[{index}]"
        base = _mapping(raw, path)
        kind = base.get("kind")
        excitation_kinds.append(kind)
        if kind == "lumped_port":
            item = _object_keys(
                base,
                path,
                required={
                    "id",
                    "kind",
                    "position_m",
                    "component",
                    "impedance_ohm",
                    "extent_m",
                    "direction",
                    "f0_hz",
                    "bandwidth",
                },
            )
            _number_list(item["position_m"], f"{path}.position_m", length=3)
            if item["component"] not in {"ex", "ey", "ez"}:
                _fail(
                    "unsupported_variant",
                    f"{path}.component",
                    "unsupported field component",
                )
            if item["direction"] not in {"+x", "-x", "+y", "-y"}:
                _fail(
                    "unsupported_variant",
                    f"{path}.direction",
                    "unsupported lumped-port direction",
                )
            for field in ("impedance_ohm", "extent_m", "f0_hz", "bandwidth"):
                _positive(item[field], f"{path}.{field}")
        elif kind == "waveguide_port":
            item = _object_keys(
                base,
                path,
                required={
                    "id",
                    "kind",
                    "x_position_m",
                    "direction",
                    "mode",
                    "mode_type",
                    "start_hz",
                    "stop_hz",
                    "points",
                    "f0_hz",
                    "bandwidth",
                    "reference_plane_m",
                },
            )
            for field in (
                "x_position_m",
                "start_hz",
                "stop_hz",
                "f0_hz",
                "bandwidth",
                "reference_plane_m",
            ):
                _positive(item[field], f"{path}.{field}")
            if (
                item["direction"] not in {"+x", "-x"}
                or item["mode_type"] != "TE"
                or item["mode"] != [1, 0]
            ):
                _fail(
                    "unsupported_variant", path, "P0 WR90 supports x-normal TE10 ports"
                )
            _integer(item["points"], f"{path}.points", minimum=2)
        elif kind == "tfsf_plane_wave":
            item = _object_keys(
                base,
                path,
                required={
                    "id",
                    "kind",
                    "f0_hz",
                    "bandwidth",
                    "amplitude",
                    "margin_cells",
                    "polarization",
                    "direction",
                    "waveform",
                },
            )
            for field in ("f0_hz", "bandwidth", "amplitude"):
                _positive(item[field], f"{path}.{field}")
            _integer(item["margin_cells"], f"{path}.margin_cells", minimum=1)
            if item["polarization"] not in {"ez", "ey"} or item["direction"] not in {
                "+x",
                "-x",
            }:
                _fail(
                    "unsupported_variant",
                    path,
                    "P0 TFSF supports x propagation and ez/ey polarization",
                )
            if item["waveform"] not in {
                "differentiated_gaussian",
                "modulated_gaussian",
            }:
                _fail(
                    "unsupported_variant",
                    f"{path}.waveform",
                    "unsupported TFSF waveform",
                )
        else:
            _fail(
                "unsupported_variant", f"{path}.kind", "unsupported excitation variant"
            )
        _unique_object_id(base["id"], f"{path}.id", object_ids)

    observation_kinds = []
    for index, raw in enumerate(_list(document["observations"], "$.observations")):
        path = f"$.observations[{index}]"
        base = _mapping(raw, path)
        kind = base.get("kind")
        observation_kinds.append(kind)
        if kind == "sparameters":
            item = _object_keys(
                base, path, required={"id", "kind", "start_hz", "stop_hz", "points"}
            )
        elif kind == "waveguide_sparameters":
            item = _object_keys(base, path, required={"id", "kind", "normalize"})
            if item["normalize"] not in {True, False, "flux"}:
                _fail(
                    "unsupported_variant",
                    f"{path}.normalize",
                    "expected boolean or 'flux'",
                )
        elif kind == "point_probe":
            item = _object_keys(
                base, path, required={"id", "kind", "position_m", "component"}
            )
            _number_list(item["position_m"], f"{path}.position_m", length=3)
        elif kind in {"dft_plane", "flux_monitor"}:
            required = {
                "id",
                "kind",
                "axis",
                "coordinate_m",
                "start_hz",
                "stop_hz",
                "points",
            }
            if kind == "dft_plane":
                required.add("component")
            item = _object_keys(base, path, required=required)
            if item["axis"] not in {"x", "y", "z"}:
                _fail("unsupported_variant", f"{path}.axis", "expected x/y/z")
            _positive(item["coordinate_m"], f"{path}.coordinate_m")
        elif kind == "field_snapshot":
            item = _object_keys(
                base, path, required={"id", "kind", "component", "axis", "coordinate_m"}
            )
        else:
            _fail(
                "unsupported_variant", f"{path}.kind", "unsupported observation variant"
            )
        if kind in {"sparameters", "dft_plane", "flux_monitor"}:
            if _positive(item["stop_hz"], f"{path}.stop_hz") <= _positive(
                item["start_hz"], f"{path}.start_hz"
            ):
                _fail("range_error", path, "stop_hz must exceed start_hz")
            _integer(item["points"], f"{path}.points", minimum=2)
        _unique_object_id(base["id"], f"{path}.id", object_ids)

    boundaries = _object_keys(
        document["boundaries"], "$.boundaries", required={"x", "y", "z", "cpml_layers"}
    )
    for axis in "xyz":
        face = _object_keys(
            boundaries[axis], f"$.boundaries.{axis}", required={"lo", "hi"}
        )
        for side in ("lo", "hi"):
            if face[side] not in {"cpml", "pec", "pmc", "periodic"}:
                _fail(
                    "unsupported_variant",
                    f"$.boundaries.{axis}.{side}",
                    "unsupported boundary",
                )
    _integer(boundaries["cpml_layers"], "$.boundaries.cpml_layers", minimum=0)

    execution = _object_keys(
        document["execution"],
        "$.execution",
        required={"backend", "n_steps", "s_param_n_steps", "timeout_seconds"},
    )
    if execution["backend"] != "cpu":
        _fail(
            "unsupported_backend", "$.execution.backend", "G1 CPU lane requires 'cpu'"
        )
    for field in ("n_steps", "s_param_n_steps", "timeout_seconds"):
        _integer(execution[field], f"$.execution.{field}", minimum=1)
    study = _object_keys(document["study"], "$.study", required={"kind"})
    if study["kind"] != "single":
        _fail("unsupported_variant", "$.study.kind", "G1 supports single runs")
    _object_keys(
        document["validation"],
        "$.validation",
        required={"support_lane", "required_checks", "metrics"},
    )
    _object_keys(document["artifacts"], "$.artifacts", required={"save"})

    required_by_workflow = {
        "patch_antenna": ({"lumped_port"}, {"sparameters"}),
        "wr90_waveguide": ({"waveguide_port"}, {"waveguide_sparameters"}),
        "multilayer_fresnel": ({"tfsf_plane_wave"}, {"flux_monitor"}),
    }
    required_exc, required_obs = required_by_workflow[document["kind"]]
    if not required_exc <= set(excitation_kinds) or not required_obs <= set(
        observation_kinds
    ):
        _fail(
            "workflow_contract",
            "$",
            f"{document['kind']} is missing required excitation/observation variants",
        )


def _generate_python(
    spec: CanonicalExperimentSpec, semantic: str, compiled_sha: str
) -> str:
    encoded = json.dumps(spec.canonical_json(), ensure_ascii=True)
    return (
        "# Deterministic rfx ExperimentSpec v2 export; do not edit generated sections.\n"
        f"# spec_sha256={spec.sha256}\n"
        f"# compiled_sha256={compiled_sha}\n"
        f"# semantic_fingerprint={semantic}\n"
        "import json\n"
        "import os\n\n"
        "os.environ['JAX_PLATFORMS'] = 'cpu'\n"
        "os.environ['JAX_PLATFORM_NAME'] = 'cpu'\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
        "os.environ['ROCR_VISIBLE_DEVICES'] = ''\n\n"
        "from rfx.experiments.canonical import compile_canonical_experiment\n\n"
        f"SPEC_JSON = {encoded}\n"
        f"SEMANTIC_FINGERPRINT = '{semantic}'\n\n"
        "def compiled_experiment():\n"
        "    return compile_canonical_experiment(json.loads(SPEC_JSON))\n\n"
        "def build_simulation():\n"
        "    return compiled_experiment().build_simulation()\n\n"
        "def preflight():\n"
        "    return compiled_experiment().preflight()\n\n"
        "def run():\n"
        "    return compiled_experiment().execute()\n\n"
        "if __name__ == '__main__':\n"
        "    print(json.dumps(preflight(), sort_keys=True))\n"
    )


def _one(items: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matches = [item for item in items if item["kind"] == kind]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind!r} observation")
    return matches[0]


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CanonicalSpecError(
            "not_json", "$", "document must be finite JSON data"
        ) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "experiment"


def _fail(code: str, path: str, message: str):
    raise CanonicalSpecError(code, path, message)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("type_error", path, "expected object")
    return value


def _object_keys(
    value: Any, path: str, *, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    mapping = _mapping(value, path)
    optional = optional or set()
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required - optional)
    if missing:
        _fail("missing_field", path, f"missing {missing}")
    if unknown:
        _fail("unknown_field", path, f"unknown {unknown}")
    return mapping


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("type_error", path, "expected array")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("type_error", path, "expected number")
    return float(value)


def _positive(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0:
        _fail("range_error", path, "must be > 0")
    return number


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail("type_error", path, f"expected integer >= {minimum}")
    return value


def _number_list(value: Any, path: str, *, length: int) -> list[float]:
    values = _list(value, path)
    if len(values) != length:
        _fail("shape_error", path, f"expected {length} numbers")
    return [_number(item, f"{path}[{index}]") for index, item in enumerate(values)]


def _nonempty_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("type_error", path, "expected non-empty string")
    return value


def _string_list(value: Any, path: str) -> list[str]:
    values = _list(value, path)
    for index, item in enumerate(values):
        _nonempty_text(item, f"{path}[{index}]")
    return values


def _identifier(value: Any, path: str) -> str:
    text = _nonempty_text(value, path)
    if not _ID_PATTERN.fullmatch(text):
        _fail("invalid_id", path, "expected lowercase stable id")
    return text


def _unique_object_id(value: Any, path: str, seen: set[str]) -> None:
    identifier = _identifier(value, path)
    if identifier in seen:
        _fail("duplicate_id", path, "object id must be unique")
    seen.add(identifier)

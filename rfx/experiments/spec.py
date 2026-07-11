"""Versioned, strict experiment specifications for AI-native rfx runs.

The first contract intentionally supports one claims-bearing workflow:
``patch_antenna_s11`` on a CPU backend.  The schema is narrower than the
general :mod:`rfx.config` surface so automation cannot silently guess units,
port types, or execution hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


CURRENT_SCHEMA_VERSION = "rfx-experiment/v1"
LEGACY_SCHEMA_VERSION = "rfx-experiment/v0"
SUPPORTED_KIND = "patch_antenna_s11"


class ExperimentSpecError(ValueError):
    """Raised when an experiment document violates the versioned contract."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentSpecError(f"{path} must be an object")
    return dict(value)


def _known(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ExperimentSpecError(f"{path} has unknown field(s): {unknown}")


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ExperimentSpecError(f"{path}.{key} is required")
    return mapping[key]


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentSpecError(f"{path} must be a non-empty string")
    return value.strip()


def _number(value: Any, path: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentSpecError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ExperimentSpecError(f"{path} must be finite")
    if positive and result <= 0.0:
        raise ExperimentSpecError(f"{path} must be > 0")
    return result


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ExperimentSpecError(f"{path} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentSpecError(f"{path} must be an integer") from exc
    if result != value or result < minimum:
        raise ExperimentSpecError(f"{path} must be an integer >= {minimum}")
    return result


def _pair(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ExperimentSpecError(f"{path} must contain exactly 2 numbers")
    return (_number(value[0], f"{path}[0]"), _number(value[1], f"{path}[1]"))


def _triple(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ExperimentSpecError(f"{path} must contain exactly 3 numbers")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


@dataclass(frozen=True)
class FrequencySweep:
    start_hz: float
    stop_hz: float
    points: int


@dataclass(frozen=True)
class SubstrateSpec:
    size_m: tuple[float, float]
    thickness_m: float
    relative_permittivity: float
    conductivity_s_per_m: float


@dataclass(frozen=True)
class MetalPlateSpec:
    size_m: tuple[float, float]
    thickness_m: float


@dataclass(frozen=True)
class FeedSpec:
    inset_m: float
    impedance_ohm: float
    pulse_bandwidth: float


@dataclass(frozen=True)
class PatchAntennaSpec:
    design_frequency_hz: float
    frequency_sweep: FrequencySweep
    domain_m: tuple[float, float, float]
    cell_size_m: float
    cpml_layers: int
    stack_base_z_m: float | None
    substrate: SubstrateSpec
    ground: MetalPlateSpec
    patch: MetalPlateSpec
    feed: FeedSpec


@dataclass(frozen=True)
class ExecutionSpec:
    backend: str
    precision: str
    n_steps: int
    s_param_n_steps: int


@dataclass(frozen=True)
class ExperimentSpec:
    """Validated current-version experiment specification."""

    schema_version: str
    name: str
    kind: str
    model: PatchAntennaSpec
    execution: ExecutionSpec
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ExperimentSpec":
        return _parse_current(migrate_spec(document))

    @classmethod
    def from_json(cls, text: str) -> "ExperimentSpec":
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExperimentSpecError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(document)

    def to_dict(self) -> dict[str, Any]:
        model = self.model
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "kind": self.kind,
            "model": {
                "design_frequency_hz": model.design_frequency_hz,
                "frequency_sweep": {
                    "start_hz": model.frequency_sweep.start_hz,
                    "stop_hz": model.frequency_sweep.stop_hz,
                    "points": model.frequency_sweep.points,
                },
                "domain_m": list(model.domain_m),
                "cell_size_m": model.cell_size_m,
                "cpml_layers": model.cpml_layers,
                "stack_base_z_m": model.stack_base_z_m,
                "substrate": {
                    "size_m": list(model.substrate.size_m),
                    "thickness_m": model.substrate.thickness_m,
                    "relative_permittivity": model.substrate.relative_permittivity,
                    "conductivity_s_per_m": model.substrate.conductivity_s_per_m,
                },
                "ground": {
                    "size_m": list(model.ground.size_m),
                    "thickness_m": model.ground.thickness_m,
                },
                "patch": {
                    "size_m": list(model.patch.size_m),
                    "thickness_m": model.patch.thickness_m,
                },
                "feed": {
                    "inset_m": model.feed.inset_m,
                    "impedance_ohm": model.feed.impedance_ohm,
                    "pulse_bandwidth": model.feed.pulse_bandwidth,
                },
            },
            "execution": {
                "backend": self.execution.backend,
                "precision": self.execution.precision,
                "n_steps": self.execution.n_steps,
                "s_param_n_steps": self.execution.s_param_n_steps,
            },
            "metadata": self.metadata,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def migrate_spec(document: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a known older document to the current in-memory schema.

    Versionless documents are deliberately rejected.  The only pre-release
    migration renames the v0 ``patch_antenna``/``run`` blocks to the v1
    ``model``/``execution`` names and makes the experiment kind explicit.
    """

    doc = _mapping(document, "spec")
    version = doc.get("schema_version")
    if version == CURRENT_SCHEMA_VERSION:
        return doc
    if version == LEGACY_SCHEMA_VERSION:
        _known(
            doc,
            {"schema_version", "name", "patch_antenna", "run", "metadata"},
            "spec",
        )
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "name": _required(doc, "name", "spec"),
            "kind": SUPPORTED_KIND,
            "model": _required(doc, "patch_antenna", "spec"),
            "execution": _required(doc, "run", "spec"),
            "metadata": doc.get("metadata", {}),
        }
    if version is None:
        raise ExperimentSpecError("spec.schema_version is required")
    raise ExperimentSpecError(
        f"unsupported schema_version {version!r}; expected {CURRENT_SCHEMA_VERSION!r}"
    )


def _parse_current(doc: Mapping[str, Any]) -> ExperimentSpec:
    _known(
        doc,
        {"schema_version", "name", "kind", "model", "execution", "metadata"},
        "spec",
    )
    name = _text(_required(doc, "name", "spec"), "spec.name")
    kind = _text(_required(doc, "kind", "spec"), "spec.kind")
    if kind != SUPPORTED_KIND:
        raise ExperimentSpecError(f"spec.kind must be {SUPPORTED_KIND!r} for schema v1")

    model_doc = _mapping(_required(doc, "model", "spec"), "spec.model")
    _known(
        model_doc,
        {
            "design_frequency_hz",
            "frequency_sweep",
            "domain_m",
            "cell_size_m",
            "cpml_layers",
            "stack_base_z_m",
            "substrate",
            "ground",
            "patch",
            "feed",
        },
        "spec.model",
    )
    design_frequency = _number(
        _required(model_doc, "design_frequency_hz", "spec.model"),
        "spec.model.design_frequency_hz",
    )

    sweep_doc = _mapping(
        _required(model_doc, "frequency_sweep", "spec.model"),
        "spec.model.frequency_sweep",
    )
    _known(sweep_doc, {"start_hz", "stop_hz", "points"}, "spec.model.frequency_sweep")
    sweep = FrequencySweep(
        start_hz=_number(
            _required(sweep_doc, "start_hz", "frequency_sweep"),
            "frequency_sweep.start_hz",
        ),
        stop_hz=_number(
            _required(sweep_doc, "stop_hz", "frequency_sweep"),
            "frequency_sweep.stop_hz",
        ),
        points=_integer(
            _required(sweep_doc, "points", "frequency_sweep"),
            "frequency_sweep.points",
            minimum=2,
        ),
    )
    if sweep.stop_hz <= sweep.start_hz:
        raise ExperimentSpecError("frequency_sweep.stop_hz must exceed start_hz")
    if not sweep.start_hz <= design_frequency <= sweep.stop_hz:
        raise ExperimentSpecError("design_frequency_hz must lie inside frequency_sweep")

    substrate_doc = _mapping(
        _required(model_doc, "substrate", "spec.model"), "spec.model.substrate"
    )
    _known(
        substrate_doc,
        {"size_m", "thickness_m", "relative_permittivity", "conductivity_s_per_m"},
        "spec.model.substrate",
    )
    substrate = SubstrateSpec(
        size_m=_pair(
            _required(substrate_doc, "size_m", "substrate"), "substrate.size_m"
        ),
        thickness_m=_number(
            _required(substrate_doc, "thickness_m", "substrate"),
            "substrate.thickness_m",
        ),
        relative_permittivity=_number(
            _required(substrate_doc, "relative_permittivity", "substrate"),
            "substrate.relative_permittivity",
        ),
        conductivity_s_per_m=_number(
            substrate_doc.get("conductivity_s_per_m", 0.0),
            "substrate.conductivity_s_per_m",
            positive=False,
        ),
    )
    if substrate.relative_permittivity < 1.0:
        raise ExperimentSpecError("substrate.relative_permittivity must be >= 1")
    if substrate.conductivity_s_per_m < 0.0:
        raise ExperimentSpecError("substrate.conductivity_s_per_m must be >= 0")

    ground = _parse_plate(_required(model_doc, "ground", "spec.model"), "ground")
    patch = _parse_plate(_required(model_doc, "patch", "spec.model"), "patch")

    feed_doc = _mapping(_required(model_doc, "feed", "spec.model"), "spec.model.feed")
    _known(
        feed_doc,
        {"inset_m", "impedance_ohm", "pulse_bandwidth"},
        "spec.model.feed",
    )
    feed = FeedSpec(
        inset_m=_number(_required(feed_doc, "inset_m", "feed"), "feed.inset_m"),
        impedance_ohm=_number(
            feed_doc.get("impedance_ohm", 50.0), "feed.impedance_ohm"
        ),
        pulse_bandwidth=_number(
            feed_doc.get("pulse_bandwidth", 0.8), "feed.pulse_bandwidth"
        ),
    )

    domain = _triple(
        _required(model_doc, "domain_m", "spec.model"), "spec.model.domain_m"
    )
    cell_size = _number(
        _required(model_doc, "cell_size_m", "spec.model"), "spec.model.cell_size_m"
    )
    cpml_layers = _integer(
        model_doc.get("cpml_layers", 8), "spec.model.cpml_layers", minimum=2
    )
    base_raw = model_doc.get("stack_base_z_m")
    stack_base = (
        None if base_raw is None else _number(base_raw, "spec.model.stack_base_z_m")
    )

    execution_doc = _mapping(_required(doc, "execution", "spec"), "spec.execution")
    _known(
        execution_doc,
        {"backend", "precision", "n_steps", "s_param_n_steps"},
        "spec.execution",
    )
    backend = _text(execution_doc.get("backend", "cpu"), "spec.execution.backend")
    if backend != "cpu":
        raise ExperimentSpecError("schema v1 only permits execution.backend='cpu'")
    precision = _text(
        execution_doc.get("precision", "float32"), "spec.execution.precision"
    )
    if precision not in {"float32", "mixed"}:
        raise ExperimentSpecError("execution.precision must be 'float32' or 'mixed'")
    execution = ExecutionSpec(
        backend=backend,
        precision=precision,
        n_steps=_integer(
            _required(execution_doc, "n_steps", "execution"),
            "execution.n_steps",
            minimum=2,
        ),
        s_param_n_steps=_integer(
            execution_doc.get("s_param_n_steps", execution_doc.get("n_steps")),
            "execution.s_param_n_steps",
            minimum=2,
        ),
    )

    metadata = _mapping(doc.get("metadata", {}), "spec.metadata")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExperimentSpecError("spec.metadata must be finite JSON data") from exc

    model = PatchAntennaSpec(
        design_frequency_hz=design_frequency,
        frequency_sweep=sweep,
        domain_m=domain,
        cell_size_m=cell_size,
        cpml_layers=cpml_layers,
        stack_base_z_m=stack_base,
        substrate=substrate,
        ground=ground,
        patch=patch,
        feed=feed,
    )
    _validate_geometry(model)
    return ExperimentSpec(
        schema_version=CURRENT_SCHEMA_VERSION,
        name=name,
        kind=kind,
        model=model,
        execution=execution,
        metadata=metadata,
    )


def _parse_plate(value: Any, path: str) -> MetalPlateSpec:
    doc = _mapping(value, f"spec.model.{path}")
    _known(doc, {"size_m", "thickness_m"}, f"spec.model.{path}")
    return MetalPlateSpec(
        size_m=_pair(_required(doc, "size_m", path), f"{path}.size_m"),
        thickness_m=_number(_required(doc, "thickness_m", path), f"{path}.thickness_m"),
    )


def _validate_geometry(model: PatchAntennaSpec) -> None:
    domain_x, domain_y, domain_z = model.domain_m
    for label, size in (
        ("ground", model.ground.size_m),
        ("substrate", model.substrate.size_m),
        ("patch", model.patch.size_m),
    ):
        if size[0] >= domain_x or size[1] >= domain_y:
            raise ExperimentSpecError(
                f"{label}.size_m must fit strictly inside domain_m x/y"
            )
    if (
        model.patch.size_m[0] > model.substrate.size_m[0]
        or model.patch.size_m[1] > model.substrate.size_m[1]
    ):
        raise ExperimentSpecError("patch.size_m must fit inside substrate.size_m")
    if (
        model.substrate.size_m[0] > model.ground.size_m[0]
        or model.substrate.size_m[1] > model.ground.size_m[1]
    ):
        raise ExperimentSpecError("substrate.size_m must fit inside ground.size_m")
    if model.feed.inset_m >= model.patch.size_m[0]:
        raise ExperimentSpecError("feed.inset_m must be smaller than patch length")

    margin = (model.cpml_layers + 1) * model.cell_size_m
    base = model.stack_base_z_m if model.stack_base_z_m is not None else margin
    top = (
        base
        + model.ground.thickness_m
        + model.substrate.thickness_m
        + model.patch.thickness_m
    )
    if base <= model.cpml_layers * model.cell_size_m:
        raise ExperimentSpecError("stack_base_z_m must be above the lower CPML region")
    if top >= domain_z - model.cpml_layers * model.cell_size_m:
        raise ExperimentSpecError("antenna stack must fit below the upper CPML region")

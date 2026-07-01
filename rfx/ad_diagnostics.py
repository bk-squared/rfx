"""JAX autodiff diagnostic helpers for rfx.

These helpers expose JAX's saved-residual inspection as structured JSON
artifacts. They are diagnostics for AD tape explainability, not runtime memory
profiles or peak-memory certificates.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.ad_checkpoint as ad_checkpoint


_AVAL_RE = re.compile(r"^(?P<dtype>[A-Za-z][A-Za-z0-9]*)(?:\[(?P<shape>[^\]]*)\])?$")
_NAMED_RE = re.compile(r"named '([^']+)'")
_DTYPE_BYTES = {
    "bool": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
    "c64": 8,
    "c128": 16,
}


def _json_safe(value: object) -> object:
    """Return common metadata values as JSON-native containers."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_safe(value.tolist())
    return value


def _snapshot_artifact(value: object) -> object:
    """Snapshot a caller-provided artifact without importing rfx API types."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "to_json") and callable(value.to_json):
        return _json_safe(json.loads(value.to_json()))
    return _json_safe(value)


class ADResidualRecord(NamedTuple):
    """One line from JAX saved-residual inspection.

    ``estimated_bytes`` is derived from the printed abstract value when shape
    and dtype are parseable. It is a static byte count for the residual value,
    not allocator overhead, fragmentation, or measured peak memory.
    """

    aval: str
    source: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    size: int | None = None
    estimated_bytes: int | None = None
    source_kind: str = "unknown"
    name: str | None = None
    raw_line: str = ""
    line_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable residual record."""
        return {
            "aval": self.aval,
            "source": self.source,
            "dtype": self.dtype,
            "shape": None if self.shape is None else list(self.shape),
            "size": self.size,
            "estimated_bytes": self.estimated_bytes,
            "estimated_gb": (
                None if self.estimated_bytes is None else self.estimated_bytes / 1e9
            ),
            "source_kind": self.source_kind,
            "name": self.name,
            "raw_line": self.raw_line,
            "line_index": self.line_index,
        }


class ADResidualInspection(NamedTuple):
    """Structured artifact for JAX saved-residual inspection.

    This artifact is produced from ``jax.ad_checkpoint.print_saved_residuals``.
    It explains what JAX says would be saved for reverse-mode AD under the given
    Python function and sample arguments. It is not a runtime profile, not an
    XLA memory report, and not a peak-memory certificate.
    """

    records: tuple[ADResidualRecord, ...]
    raw_lines: tuple[str, ...]
    total_estimated_bytes: int | None
    total_estimated_gb: float | None
    unknown_estimate_count: int

    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this diagnostic artifact."""
        return "jax_saved_residuals_inspection"

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable saved-residual artifact."""
        return {
            "evidence_class": self.evidence_class,
            "records": [record.to_dict() for record in self.records],
            "raw_lines": list(self.raw_lines),
            "total_estimated_bytes": self.total_estimated_bytes,
            "total_estimated_gb": self.total_estimated_gb,
            "unknown_estimate_count": self.unknown_estimate_count,
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the inspection artifact with non-finite floats rejected."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


class ADResidualGroup(NamedTuple):
    """Grouped summary of saved residual records."""

    key: str
    source_kind: str
    name: str | None
    dtype: str | None
    record_count: int
    known_count: int
    unknown_count: int
    known_estimated_bytes: int
    line_indices: tuple[int, ...]

    @property
    def known_estimated_gb(self) -> float:
        """Known static byte estimate in GB for parseable records in the group."""
        return self.known_estimated_bytes / 1e9

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable group artifact."""
        return {
            "key": self.key,
            "source_kind": self.source_kind,
            "name": self.name,
            "dtype": self.dtype,
            "record_count": self.record_count,
            "known_count": self.known_count,
            "unknown_count": self.unknown_count,
            "known_estimated_bytes": self.known_estimated_bytes,
            "known_estimated_gb": self.known_estimated_gb,
            "line_indices": list(self.line_indices),
        }


class ADParserHealth(NamedTuple):
    """Parser health summary for saved-residual diagnostics."""

    record_count: int
    known_count: int
    unknown_count: int
    unknown_fraction: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable parser-health artifact."""
        return {
            "record_count": self.record_count,
            "known_count": self.known_count,
            "unknown_count": self.unknown_count,
            "unknown_fraction": self.unknown_fraction,
            "warnings": list(self.warnings),
        }


class ADSavedResidualDiagnosticReport(NamedTuple):
    """rfx saved-residual diagnostic report.

    The report summarizes JAX saved-residual evidence for one differentiated
    callable and sample argument set. It is trace-time explainability, not a
    runtime profiler, XLA memory report, peak-memory guarantee, certificate, or
    RF/gradient validation artifact.
    """

    inspection: ADResidualInspection
    top_residuals: tuple[ADResidualRecord, ...]
    groups: tuple[ADResidualGroup, ...]
    parser_health: ADParserHealth
    known_estimated_bytes: int
    known_estimated_gb: float
    total_estimated_bytes: int | None
    total_estimated_gb: float | None
    jax_version: str
    workflow: str | None
    context: Mapping[str, object] | None
    artifact_snapshots: Mapping[str, object]
    recommendations: tuple[str, ...]

    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this rfx diagnostic report."""
        return "rfx_ad_saved_residuals_diagnostic"

    @property
    def known_only_bytes(self) -> int:
        """Known-byte subtotal excluding parser-unknown residual lines."""
        return self.known_estimated_bytes

    @property
    def source_evidence_class(self) -> str:
        """Evidence class of the underlying JAX saved-residual adapter."""
        return self.inspection.evidence_class

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable diagnostic report."""
        return {
            "evidence_class": self.evidence_class,
            "source_evidence_class": self.source_evidence_class,
            "jax_version": self.jax_version,
            "workflow": self.workflow,
            "context": None if self.context is None else _json_safe(self.context),
            "artifact_snapshots": _json_safe(self.artifact_snapshots),
            "inspection": self.inspection.to_dict(),
            "raw_lines": list(self.inspection.raw_lines),
            "records": [record.to_dict() for record in self.inspection.records],
            "top_residuals": [record.to_dict() for record in self.top_residuals],
            "groups": [group.to_dict() for group in self.groups],
            "parser_health": self.parser_health.to_dict(),
            "known_estimated_bytes": self.known_estimated_bytes,
            "known_only_bytes": self.known_only_bytes,
            "known_estimated_gb": self.known_estimated_gb,
            "total_estimated_bytes": self.total_estimated_bytes,
            "total_estimated_gb": self.total_estimated_gb,
            "recommendations": list(self.recommendations),
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the diagnostic report with non-finite floats rejected."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


def _parse_aval(aval: str) -> tuple[str | None, tuple[int, ...] | None, int | None, int | None]:
    match = _AVAL_RE.match(aval)
    if match is None:
        return None, None, None, None

    dtype = match.group("dtype")
    shape_text = match.group("shape")
    if shape_text is None:
        shape: tuple[int, ...] = ()
    elif shape_text.strip() == "":
        shape = ()
    else:
        dims: list[int] = []
        for item in shape_text.split(","):
            item = item.strip()
            if not item.isdigit():
                return dtype, None, None, None
            dims.append(int(item))
        shape = tuple(dims)

    size = math.prod(shape) if shape else 1
    dtype_bytes = _DTYPE_BYTES.get(dtype)
    estimated_bytes = None if dtype_bytes is None else int(size * dtype_bytes)
    return dtype, shape, int(size), estimated_bytes


def _source_kind(source: str) -> str:
    if source.startswith("from the argument"):
        return "argument"
    if _NAMED_RE.search(source):
        return "named"
    if source.startswith("output of"):
        return "intermediate"
    return "unknown"


def parse_saved_residual_line(line: str, *, line_index: int | None = None) -> ADResidualRecord:
    """Parse one line printed by JAX saved-residual inspection.

    Unknown future JAX formats are preserved as raw text with parseable fields
    set to ``None`` instead of being silently discarded.
    """
    stripped = line.strip()
    if not stripped:
        return ADResidualRecord(aval="", source="", raw_line=line, line_index=line_index)

    aval, _, source = stripped.partition(" ")
    dtype, shape, size, estimated_bytes = _parse_aval(aval)
    name_match = _NAMED_RE.search(source)
    return ADResidualRecord(
        aval=aval,
        source=source,
        dtype=dtype,
        shape=shape,
        size=size,
        estimated_bytes=estimated_bytes,
        source_kind=_source_kind(source),
        name=None if name_match is None else name_match.group(1),
        raw_line=line,
        line_index=line_index,
    )


def inspect_ad_saved_residuals(
    fun: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> ADResidualInspection:
    """Inspect JAX saved residuals for ``fun(*args, **kwargs)``.

    The implementation captures ``jax.ad_checkpoint.print_saved_residuals`` and
    parses the printed abstract values into a structured, JSON-serializable
    artifact. Exceptions from tracing ``fun`` propagate to the caller.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ad_checkpoint.print_saved_residuals(fun, *args, **kwargs)

    raw_lines = tuple(
        line for line in buffer.getvalue().splitlines() if line.strip()
    )
    records = tuple(
        parse_saved_residual_line(line, line_index=index)
        for index, line in enumerate(raw_lines)
    )
    known_bytes = [
        record.estimated_bytes
        for record in records
        if record.estimated_bytes is not None
    ]
    unknown_count = len(records) - len(known_bytes)
    total_bytes = None if unknown_count else int(sum(known_bytes))
    return ADResidualInspection(
        records=records,
        raw_lines=raw_lines,
        total_estimated_bytes=total_bytes,
        total_estimated_gb=None if total_bytes is None else total_bytes / 1e9,
        unknown_estimate_count=unknown_count,
    )


def _known_estimated_bytes(records: Sequence[ADResidualRecord]) -> int:
    return int(sum(record.estimated_bytes or 0 for record in records))


def _group_key(record: ADResidualRecord) -> tuple[str, str | None, str | None]:
    return record.source_kind, record.name, record.dtype


def _make_group(records: Sequence[ADResidualRecord]) -> ADResidualGroup:
    first = records[0]
    source_kind, name, dtype = _group_key(first)
    known_count = sum(record.estimated_bytes is not None for record in records)
    unknown_count = len(records) - known_count
    key_parts = [source_kind]
    if name is not None:
        key_parts.append(f"name={name}")
    if dtype is not None:
        key_parts.append(f"dtype={dtype}")
    return ADResidualGroup(
        key="|".join(key_parts),
        source_kind=source_kind,
        name=name,
        dtype=dtype,
        record_count=len(records),
        known_count=known_count,
        unknown_count=unknown_count,
        known_estimated_bytes=_known_estimated_bytes(records),
        line_indices=tuple(
            record.line_index for record in records if record.line_index is not None
        ),
    )


def _build_groups(records: Sequence[ADResidualRecord]) -> tuple[ADResidualGroup, ...]:
    buckets: dict[tuple[str, str | None, str | None], list[ADResidualRecord]] = {}
    for record in records:
        buckets.setdefault(_group_key(record), []).append(record)
    groups = [_make_group(group_records) for group_records in buckets.values()]
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                -group.known_estimated_bytes,
                -group.record_count,
                group.key,
            ),
        )
    )


def _parser_health(inspection: ADResidualInspection) -> ADParserHealth:
    record_count = len(inspection.records)
    unknown_count = inspection.unknown_estimate_count
    known_count = record_count - unknown_count
    unknown_fraction = 0.0 if record_count == 0 else unknown_count / record_count
    warnings: list[str] = []
    if record_count == 0:
        warnings.append("JAX printed no saved residual lines for this callable and sample args.")
    if unknown_count:
        warnings.append(
            "Some saved residual lines could not be byte-estimated; known byte totals are partial and raw_lines must be inspected."
        )
    return ADParserHealth(
        record_count=record_count,
        known_count=known_count,
        unknown_count=unknown_count,
        unknown_fraction=unknown_fraction,
        warnings=tuple(warnings),
    )


def _top_residuals(
    records: Sequence[ADResidualRecord],
    top_n: int,
) -> tuple[ADResidualRecord, ...]:
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    known = [record for record in records if record.estimated_bytes is not None]
    return tuple(
        sorted(
            known,
            key=lambda record: (
                -(record.estimated_bytes or 0),
                record.line_index if record.line_index is not None else 10**9,
                record.raw_line,
            ),
        )[:top_n]
    )


def _recommendations(
    *,
    inspection: ADResidualInspection,
    top_residuals: Sequence[ADResidualRecord],
    groups: Sequence[ADResidualGroup],
    parser_health: ADParserHealth,
    artifact_snapshots: Mapping[str, object],
) -> tuple[str, ...]:
    recs: list[str] = [
        "Treat this report as trace-time JAX saved-residual explainability, not runtime profiling, XLA memory analysis, a peak-memory certificate, or RF validation."
    ]
    if parser_health.unknown_count:
        recs.append(
            "Inspect raw_lines before comparing byte totals because some residual lines were not parseable by the current rfx parser."
        )
    if top_residuals:
        largest = top_residuals[0]
        label = largest.name or largest.source_kind or largest.aval
        recs.append(
            f"Largest parseable saved residual is {label!r} at line {largest.line_index} with {largest.estimated_bytes} known static bytes; consider checkpoint naming/remat policy around that value."
        )
    elif inspection.records:
        recs.append(
            "No parseable residual byte estimates were available; use raw_lines and JAX version metadata for manual inspection."
        )
    else:
        recs.append(
            "No saved residual lines were printed; confirm the inspected callable matches the differentiated objective of interest."
        )
    if groups:
        dominant_group = groups[0]
        recs.append(
            f"Dominant parseable residual group is {dominant_group.key!r}; compare this group with static AD memory planning artifacts before changing checkpoint policy."
        )
    if artifact_snapshots:
        recs.append(
            "Passed memory-planning artifacts are included as snapshots for comparison only; residual bytes do not prove budget fit."
        )
    else:
        recs.append(
            "For grid-level memory planning, compare this report with sim.explain_ad_memory(...), plan_ad_memory(...), or mesh_intelligence_report(...)."
        )
    recs.append(
        "Run separate gradient checks, convergence tests, or RF reference validation before making physics claims."
    )
    return tuple(recs)


def diagnose_ad_saved_residuals(
    fun: Callable[..., Any],
    *args: Any,
    top_n: int = 10,
    workflow: str | None = None,
    context: Mapping[str, object] | None = None,
    artifacts: Mapping[str, object] | None = None,
    **kwargs: Any,
) -> ADSavedResidualDiagnosticReport:
    """Return an rfx diagnostic report for JAX saved residuals.

    The report layers rfx-owned summaries and recommendations on top of
    ``inspect_ad_saved_residuals``. It inspects ``fun(*args, **kwargs)`` and
    preserves exceptions from tracing the callable. Optional ``context`` and
    ``artifacts`` are serialized as snapshots only; they are not used to infer
    runtime memory fit or physics validity.
    """
    inspection = inspect_ad_saved_residuals(fun, *args, **kwargs)
    top = _top_residuals(inspection.records, top_n)
    groups = _build_groups(inspection.records)
    health = _parser_health(inspection)
    snapshots = {
        str(key): _snapshot_artifact(value)
        for key, value in (artifacts or {}).items()
    }
    known_bytes = _known_estimated_bytes(inspection.records)
    return ADSavedResidualDiagnosticReport(
        inspection=inspection,
        top_residuals=top,
        groups=groups,
        parser_health=health,
        known_estimated_bytes=known_bytes,
        known_estimated_gb=known_bytes / 1e9,
        total_estimated_bytes=inspection.total_estimated_bytes,
        total_estimated_gb=inspection.total_estimated_gb,
        jax_version=jax.__version__,
        workflow=workflow,
        context=None if context is None else _json_safe(context),
        artifact_snapshots=snapshots,
        recommendations=_recommendations(
            inspection=inspection,
            top_residuals=top,
            groups=groups,
            parser_health=health,
            artifact_snapshots=snapshots,
        ),
    )


__all__ = [
    "ADParserHealth",
    "ADResidualGroup",
    "ADResidualInspection",
    "ADResidualRecord",
    "ADSavedResidualDiagnosticReport",
    "diagnose_ad_saved_residuals",
    "inspect_ad_saved_residuals",
    "parse_saved_residual_line",
]

"""Opt-in, content-free operational telemetry policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping


ALLOWED_FIELDS = frozenset(
    {
        "event",
        "rfx_version",
        "platform",
        "backend",
        "duration_ms",
        "outcome",
        "error_code",
        "workflow_kind",
        "schema_version",
    }
)
FORBIDDEN_FIELD_TOKENS = (
    "spec",
    "prompt",
    "artifact",
    "content",
    "secret",
    "token",
    "key",
    "path",
    "title",
    "description",
)


@dataclass(frozen=True)
class TelemetryPolicy:
    enabled: bool = False

    @classmethod
    def from_environment(cls) -> "TelemetryPolicy":
        value = os.environ.get("RFX_TELEMETRY", "off").strip().lower()
        if value not in {"off", "on"}:
            raise ValueError("RFX_TELEMETRY must be 'off' or 'on'")
        return cls(enabled=value == "on")

    def sanitize(self, event: Mapping[str, Any]) -> dict[str, Any]:
        unexpected = set(event) - ALLOWED_FIELDS
        if unexpected:
            raise ValueError(
                f"telemetry fields are not allowlisted: {sorted(unexpected)}"
            )
        for key in event:
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_FIELD_TOKENS):
                raise ValueError(
                    f"telemetry field may contain experiment content: {key}"
                )
        encoded = json.dumps(event, allow_nan=False)
        if len(encoded.encode("utf-8")) > 4096:
            raise ValueError("telemetry event exceeds 4096 bytes")
        return json.loads(encoded)


class LocalTelemetrySink:
    """Testable local sink; no network transport exists unless explicitly added."""

    def __init__(self, path: str | Path, policy: TelemetryPolicy | None = None):
        self.path = Path(path).expanduser().resolve()
        self.policy = policy or TelemetryPolicy.from_environment()

    def emit(self, event: Mapping[str, Any]) -> bool:
        if not self.policy.enabled:
            return False
        sanitized = self.policy.sanitize(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitized, sort_keys=True) + "\n")
        return True

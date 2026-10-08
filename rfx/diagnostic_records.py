"""Immutable solver-independent diagnostic records."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Literal, Mapping


Severity = Literal["info", "advisory", "warning", "refusal"]
LEGACY_SEVERITY = {
    "info": "info",
    "advisory": "warning",
    "warning": "warning",
    "refusal": "error",
}


class _ImmutableValues(dict):
    """Scalar mapping compatible with dataclasses.asdict, deepcopy and pickle."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("diagnostic values are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        # Validated values are immutable scalars or strings.
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


@dataclass(frozen=True)
class Diagnostic:
    """One finding, with named SI scalars or explanatory strings.

    Values are copied and made read-only. Path is absent for checks performed
    before execution dispatch. The record contains no solver or array objects.
    """

    code: str
    severity: Severity
    subject: str | None
    message: str
    values: Mapping[str, int | float | str]
    path: str | None = None
    source: str | None = None

    def __post_init__(self):
        for name in ("code", "message", "subject", "path", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) and not (
                value is None and name in ("subject", "path", "source")
            ):
                raise TypeError(
                    f"diagnostic {name} must be a string"
                    + (" or None" if name in ("subject", "path", "source") else "")
                )
        if self.severity not in LEGACY_SEVERITY:
            raise ValueError(f"unknown diagnostic severity: {self.severity!r}")
        values = {}
        for name, value in self.values.items():
            if not isinstance(name, str):
                raise TypeError("diagnostic value names must be strings")
            if isinstance(value, Integral):
                value = int(value)
            elif isinstance(value, Real):
                value = float(value)
            elif not isinstance(value, str):
                raise TypeError(f"diagnostic value {name!r} must be a scalar or string")
            values[name] = value
        object.__setattr__(self, "values", _ImmutableValues(values))

    @property
    def legacy_severity(self) -> str:
        return LEGACY_SEVERITY[self.severity]

    def __hash__(self):
        return hash(
            (
                self.code,
                self.severity,
                self.subject,
                self.message,
                tuple(sorted(self.values.items())),
                self.path,
                self.source,
            )
        )

    def __reduce__(self):
        """Keep legacy report copying/pickling compatible with immutable values."""
        return type(self), (
            self.code, self.severity, self.subject, self.message,
            dict(self.values), self.path, self.source,
        )

    def to_dict(self) -> dict:
        """JSON-safe even for a nonfinite numerical observation."""
        return {
            "code": self.code,
            "severity": self.severity,
            "subject": self.subject,
            "message": self.message,
            "values": {
                name: (
                    str(value)
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for name, value in self.values.items()
            },
            "path": self.path,
            "source": self.source,
        }


def from_legacy(
    message, *, code="uncoded", severity="warning", loc=None, source=None, refusal=False
):
    """Mechanical bridge for families not yet migrated to structured values."""
    return Diagnostic(
        code,
        "refusal"
        if refusal
        else {
            "error": "refusal",
            "warning": "advisory",
            "info": "info",
        }.get(severity, "advisory"),
        loc,
        str(message),
        {},
        source=source,
    )


def for_legacy_issue(diagnostic, message, severity):
    """Match the actual issue: errors block, warnings advise, and info informs.

    Any other severity word advises, as the report's own filter has always
    treated it (only "error" blocks, only "info" is a record).
    """
    level = {"error": "refusal", "warning": "advisory", "info": "info"}.get(severity, "advisory")
    text = str(message)
    if diagnostic.severity == level and diagnostic.message == text:
        return diagnostic
    return replace(
        diagnostic,
        severity=level,
        message=text,
    )

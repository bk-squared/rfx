"""Public bases for rfx exceptions and warnings."""

from typing import Literal as _Literal, TypedDict as _TypedDict


class _EdgeAwareBlockedRun(_TypedDict):
    lo: float
    hi: float
    gap: float
    nearest_below: float | None
    nearest_above: float | None


class RfxError(Exception):
    """Base of every exception rfx defines."""


class RfxRefusal(RfxError, ValueError):
    """rfx refuses a request it cannot compute correctly; the message says why and what to do."""


class EdgeAwareProfileRefusal(RfxRefusal):
    """An edge-aware axis cannot close under the requested cell rule."""

    axis: str
    diagnostic_kind: _Literal["local", "forward"]
    blocked_runs: list[_EdgeAwareBlockedRun]
    route_dx: float | None
    route_boundary_cell: float | None
    _rule: tuple[float, float, float | None, float]
    _route_follows_dx: bool


class RfxUnsupported(RfxRefusal, NotImplementedError):
    """A refusal of a feature combination rfx does not implement on this path."""


class RfxWarning(UserWarning):
    """Base of every rfx warning except Python DeprecationWarning and FutureWarning."""


class AccuracyWarning(RfxWarning):
    """The result may be inaccurate, unconverged, unsettled, unresolved, unjudged or extrapolated."""


class ConfigurationWarning(RfxWarning):
    """rfx adjusted, defaulted or ignored part of the request as the message states."""


class RingdownSecondDerivativeWarning(RfxWarning):
    """A second derivative reached the identification window of ``forward(ringdown=...)``.

    The completion moves its poles by a first-order step, so any second-order
    form of an objective that reads a completed output is NaN; first
    derivatives, and every derivative of an objective that reads no completed
    output, are exact.
    """

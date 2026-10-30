"""Public bases for rfx exceptions and warnings."""


class RfxError(Exception):
    """Base of every exception rfx defines."""


class RfxRefusal(RfxError, ValueError):
    """rfx refuses a request it cannot compute correctly; the message says why and what to do."""


class RfxUnsupported(RfxRefusal, NotImplementedError):
    """A refusal of a feature combination rfx does not implement on this path."""


class RfxWarning(UserWarning):
    """Base of every rfx warning except Python DeprecationWarning and FutureWarning."""


class AccuracyWarning(RfxWarning):
    """The result may be inaccurate, unconverged, unsettled, unresolved, unjudged or extrapolated."""


class ConfigurationWarning(RfxWarning):
    """rfx adjusted, defaulted or ignored part of the request as the message states."""

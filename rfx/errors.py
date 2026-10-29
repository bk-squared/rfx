"""Public bases for rfx exceptions."""


class RfxError(Exception):
    """Base of every exception rfx defines."""


class RfxRefusal(RfxError, ValueError):
    """rfx refuses a request it cannot compute correctly; the message says why and what to do."""


class RfxUnsupported(RfxRefusal, NotImplementedError):
    """A refusal of a feature combination rfx does not implement on this path."""

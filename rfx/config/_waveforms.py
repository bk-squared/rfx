"""Waveform registry for the config-driven CLI.

Maps a small set of ``type`` strings to the corresponding rfx source
waveform classes and builds an instance from a plain config ``dict``.

The mapping is intentionally tiny — only the waveforms reachable from the
MVP YAML schema are registered. Unknown types raise a clear error naming
the offending value and the supported set.
"""

from __future__ import annotations

from rfx.sources.sources import GaussianPulse, ModulatedGaussian

# type-string -> waveform class. Each class takes (f0, bandwidth, amplitude,
# cutoff) as its constructor args (GaussianPulse gained cutoff for the
# issue-#388 deposited-DC mitigation).
WAVEFORM_REGISTRY = {
    "gaussian_pulse": GaussianPulse,
    "modulated_gaussian": ModulatedGaussian,
}

# Per-waveform set of accepted config keys (besides ``type``). Used to give a
# precise error on an unexpected key rather than a bare TypeError from the
# dataclass constructor.
_ALLOWED_KEYS = {
    "gaussian_pulse": {"f0", "bandwidth", "amplitude", "cutoff"},
    "modulated_gaussian": {"f0", "bandwidth", "amplitude", "cutoff"},
}


def waveform_from_config(cfg: dict):
    """Build a waveform instance from a config dict.

    Parameters
    ----------
    cfg : dict
        Must contain ``type`` (one of :data:`WAVEFORM_REGISTRY`) and at
        least ``f0``. Remaining keys are passed through to the waveform
        constructor (``bandwidth``, ``amplitude``, ``cutoff``).

    Returns
    -------
    GaussianPulse | ModulatedGaussian
    """
    if not isinstance(cfg, dict):
        raise TypeError(
            f"waveform config must be a mapping, got {type(cfg).__name__}"
        )
    if "type" not in cfg:
        raise KeyError(
            "waveform config is missing required key 'type' "
            f"(supported: {sorted(WAVEFORM_REGISTRY)})"
        )
    wtype = cfg["type"]
    if wtype not in WAVEFORM_REGISTRY:
        raise NotImplementedError(
            f"Unsupported waveform type {wtype!r}. "
            f"Supported waveforms: {sorted(WAVEFORM_REGISTRY)}."
        )
    cls = WAVEFORM_REGISTRY[wtype]
    allowed = _ALLOWED_KEYS[wtype]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise KeyError(
            f"waveform {wtype!r} got unexpected key(s) {sorted(unknown)}; "
            f"allowed keys: {sorted(allowed)}"
        )
    if "f0" not in kwargs:
        raise KeyError(f"waveform {wtype!r} is missing required key 'f0'")
    return cls(**{k: float(v) for k, v in kwargs.items()})

"""Power-wave domain shared by waveguide scattering assemblers."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx._traced_check import traced_check
from rfx._warning_origin import user_stacklevel
from rfx.errors import RfxWarning


class WaveguideCutoffWarning(RfxWarning):
    """A requested scattering channel has no propagating power wave."""


def power_wave_defined(cfg):
    """A mode carries a power wave strictly above its own medium's cutoff."""
    return cfg.freqs > cfg.f_cutoff


def power_wave_pair(receiving, driving):
    return power_wave_defined(receiving) & power_wave_defined(driving)


def finite_wave_ratio(numerator, denominator, defined, *, xp=jnp):
    """Guard both division operands, including the inactive AD branch."""
    return xp.where(defined, numerator, 0) / xp.where(defined, denominator, 1)


def mask_power_wave(value, defined, *, xp=jnp):
    return xp.where(defined, value, xp.nan)


def power_wave_ratio(numerator, denominator, receiving, driving, *, xp=jnp):
    defined = power_wave_pair(receiving, driving)
    return mask_power_wave(
        finite_wave_ratio(numerator, denominator, defined, xp=xp), defined, xp=xp,
    )


def warn_power_waves(cfgs, *, port_indices=None):
    """One notice per extraction call, listing all undefined channels/bins."""
    from rfx.measurement.waveguide_s import _mode_indices_from_config

    cfgs = tuple(cfgs)
    indices = range(len(cfgs)) if port_indices is None else port_indices
    labels = [
        f"port {i}, {cfg.mode_type}{''.join(map(str, _mode_indices_from_config(cfg)))}"
        for i, cfg in zip(indices, cfgs)
    ]
    masks = tuple(power_wave_defined(cfg) for cfg in cfgs)
    freqs = tuple(cfg.freqs for cfg in cfgs)

    def emit(masks, freqs):
        rows = []
        for label, defined, bins in zip(labels, masks, freqs):
            bad = np.flatnonzero(~np.asarray(defined))
            if bad.size:
                rows.append(f"{label}: bins {bad.tolist()} ({np.asarray(bins)[bad].tolist()} Hz)")
        if rows:
            warnings.warn(
                "Waveguide S entries set to NaN: below cutoff, no power wave; " + "; ".join(rows),
                WaveguideCutoffWarning, stacklevel=user_stacklevel(),
            )

    traced_check("waveguide S", "below-cutoff notice", emit, masks, freqs)

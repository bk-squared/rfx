"""Setup-time spectral coordinates shared by realized echo-bound callers."""
import jax
import jax.numpy as jnp
import numpy as np


def declared_read_bins(sim, freqs=None, *, port_freqs=(), waveguide_ports=None,
                       include_monitors=True):
    """Collect the bins the declared result channels will store.

    Callers supply the S bins resolved by their own port setup, including
    implicit defaults. Concrete bins keep the channel storage precision.
    When any bins are traced, all coordinates enter one float32 device
    vector; duplicates need no dynamic-size unique operation for the max.
    Calculators returning only S channels exclude the field monitors.
    """
    from rfx.measurement.monitors import default_frequency_grid
    from rfx.preflight.waveguide import resolve_waveguide_port_freqs

    values = [value for value in (freqs, *port_freqs) if value is not None]
    if include_monitors:
        for field in ('_dft_planes', '_flux_monitors'):
            for entry in getattr(sim, field, ()):
                values.append(entry.freqs if entry.freqs is not None else
                              default_frequency_grid(sim._freq_max, entry.n_freqs))
        for field, index in (('_ntff', 2), ('_current_moments', 3)):
            spec = getattr(sim, field, None)
            if spec is not None:
                dtype = (np.float64 if field == '_current_moments' and
                         jax.config.x64_enabled else np.float32)
                value = spec[index]
                values.append(jnp.asarray(value, dtype) if any(
                    isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))
                    else np.asarray(value, dtype))
    if waveguide_ports is None:
        values.extend(resolve_waveguide_port_freqs(sim, entry)
                      for entry in getattr(sim, '_waveguide_ports', ()))
    else:
        configs = (waveguide_ports.values() if hasattr(waveguide_ports, 'values')
                   else waveguide_ports)
        values.extend(cfg.freqs for cfg in configs)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(values)):
        return jnp.concatenate([jnp.asarray(value, jnp.float32).reshape(-1)
                                for value in values])
    return np.unique(np.concatenate([np.asarray(value, np.float64).reshape(-1)
                                     for value in values])) if values else np.array([])

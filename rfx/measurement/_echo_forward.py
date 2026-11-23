"""Lossless transport of the live declared index to host-only record reads."""
import numpy as np


def echo_override_declaration(sim, arguments):
    """Retain per-call scalar maxima on a private simulation view."""
    import copy
    import jax
    import jax.numpy as jnp

    names = ('eps_override', 'design_eps_override', 'mu_r_override')
    if not any(arguments.get(name) is not None for name in names):
        return sim
    maxima = tuple(
        1. if arguments.get(name) is None else
        (jnp.max(arguments[name]) if isinstance(arguments[name], (jax.Array, jax.core.Tracer))
         else np.max(np.asarray(arguments[name], dtype=np.float64)))
        for name in names)
    member = copy.copy(sim)
    member._echo_override_maxima = maxima
    return member


def declared_index_reader(sim, freqs):
    """Separate live material leaves from a host-only float64 index function."""
    import jax
    from types import SimpleNamespace
    from rfx.measurement.record_end import material_echo_index

    rows = []
    for name in dict.fromkeys(entry.material_name for entry in sim._geometry):
        spec = sim._resolve_material(name)
        rows.append((spec.eps_r, spec.mu_r,
                     tuple(tuple(p) for p in spec.debye_poles or ()),
                     tuple((p.omega_0, p.delta, getattr(p.kappa, 'si', p.kappa))
                           for p in spec.lorentz_poles or ())))
    leaves, definition = jax.tree_util.tree_flatten(
        (rows, getattr(sim, '_echo_override_maxima', ())))
    dynamic = tuple(isinstance(value, (jax.Array, jax.core.Tracer)) for value in leaves)
    fixed = tuple(None if live else value for value, live in zip(leaves, dynamic))
    operands = tuple(value for value, live in zip(leaves, dynamic) if live)
    bins = np.asarray([0., sim._freq_max] if freqs is None or not len(freqs) else freqs,
                      dtype=np.float64)

    def read(*values):
        stream = iter(values)
        specs = []
        materials, maxima = definition.unflatten(
            [next(stream) if live else value for value, live in zip(fixed, dynamic)])
        for eps, mu, dp, lp in materials:
            specs.append(SimpleNamespace(
                eps_r=eps, mu_r=mu,
                debye_poles=tuple(SimpleNamespace(delta_eps=d, tau=t) for d, t in dp),
                lorentz_poles=tuple(SimpleNamespace(omega_0=w, delta=d, kappa=k)
                                    for w, d, k in lp)))
        return material_echo_index(specs, bins, declared=True, override_maxima=maxima)

    return read, operands


def forward_echo_bound(grid, sim, freqs, waveguide_ports=()):
    """Carry an exact float64 index as two uint32 children in every context."""
    import jax.numpy as jnp
    from rfx.measurement.record_end import echo_group_factor
    from rfx.ringdown._jax import host_echo_index

    configs = tuple(waveguide_ports.values() if hasattr(waveguide_ports, 'values')
                    else waveguide_ports or ())
    bins = freqs
    if (bins is None or not len(bins)) and configs:
        bins = np.unique(np.concatenate([np.asarray(cfg.freqs) for cfg in configs]))
    read, operands = declared_index_reader(sim, bins)
    words = (host_echo_index(read, *operands) if operands else
             jnp.asarray(np.asarray(read(), dtype=np.float64).reshape(1).view(np.uint32)))
    return dict(index_words=words, g=echo_group_factor(configs)), read, operands


def realized_echo_bound(grid, bound):
    """Read one primal result's index and physical widths on the host."""
    from rfx.measurement.record_end import echo_delay_bound_on_grid

    if bound is None:
        return None
    if 'index_words' in bound:
        index = np.asarray(bound['index_words'], dtype=np.uint32).reshape(2).view(np.float64)[0]
        bound = dict(n_max=float(index), g=bound['g'])
    return echo_delay_bound_on_grid(grid, bound)

"""Device echo bounds from the volume material model used by a runner."""
import jax
import jax.numpy as jnp
import numpy as np


def _live(value):
    return any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(value))


def _up(value, steps=1):
    value = jax.lax.stop_gradient(jnp.asarray(value, dtype=jnp.float32))
    for _ in range(steps):
        value = jnp.nextafter(value, jnp.float32(jnp.inf))
    return value


def _poles(spec):
    if spec is None:
        return ()
    poles, masks = spec
    masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
    return tuple(zip(poles, masks))


def _strength(pole):
    return getattr(pole.kappa, 'si', pole.kappa)


def _response(pole, frequency, lorentz):
    values = (pole.omega_0, pole.delta, _strength(pole)) if lorentz else tuple(pole)
    xp = jnp if _live((values, frequency)) else np
    omega = 2 * xp.pi * xp.asarray(frequency)
    if not lorentz:
        return xp.asarray(pole.delta_eps) / (1 + 1j * omega * pole.tau)
    w0, delta, strength = values
    w0, delta, strength = xp.asarray(w0), xp.asarray(delta), xp.asarray(strength)
    active = w0 != 0
    denominator = w0**2 - omega**2 + 2j * delta * omega
    return xp.where(active, strength / xp.where(active, denominator, 1.), 0.)


def realized_echo_bound(grid, materials, *, freqs=(), debye=None, lorentz=None,
                        smoothing_eps=(), waveguide_ports=(), valid=None):
    """Read the largest volume sqrt(abs(eps(f)*mu)) without a callback.

    Capacitor records and wire contour corrections do not describe echo paths.
    Smoothing includes every dielectric shape's eps, never an occupied inverse.
    Concrete susceptibilities are evaluated in host float64, one frequency at
    a time; traced poles use float32 with relative margin eps32*(1+w0/(2*delta))
    per Lorentz pole. Each map cell is sampled at its own in-band peak too.
    An undamped in-band resonance raises concretely and becomes infinity under
    a trace, so the device window judgement cannot pass.

    n_max is stop_gradient and rounded upward by one float32 ULP using
    nextafter(n, +inf). Concrete D and g retain host float64. For traced D,
    gamma_(m+8), m the total number of physical widths, covers the positive
    sums, three squares, their sum and square root; D/(1-gamma) is rounded
    upward once. tau_max is stopped and rounded upward by four ULP to cover
    conversion, two products and division. The host requirement therefore
    cannot exceed the returned requirement; its margin is these stated
    errors plus the one-ULP n_max increment.
    """
    from rfx.core.yee import permittivity_without_lumped
    from rfx.measurement.record_end import echo_group_factor

    eps = jax.lax.stop_gradient(jnp.asarray(permittivity_without_lumped(materials), jnp.float32))
    mu = jax.lax.stop_gradient(jnp.asarray(materials.mu_r, jnp.float32))
    dp, lp = _poles(debye), _poles(lorentz)
    bins = np.asarray(() if freqs is None else freqs, dtype=np.float64).reshape(-1)
    configs = tuple(waveguide_ports.values() if hasattr(waveguide_ports, 'values')
                    else waveguide_ports or ())
    if not bins.size and configs:
        bins = np.unique(np.concatenate([np.asarray(cfg.freqs, np.float64) for cfg in configs]))
    if (dp or lp) and not bins.size:
        raise ValueError('the dispersive material read band is unavailable')
    samples = [0., *bins]
    invalid = jnp.asarray(False)
    margin = jnp.float32(0.)
    for pole, mask in lp:
        xp = jnp if _live((pole.omega_0, pole.delta, _strength(pole))) else np
        w0, delta = xp.asarray(pole.omega_0), xp.asarray(pole.delta)
        peak = xp.sqrt(xp.maximum(w0**2 - 2 * delta**2, 0.)) / (2 * xp.pi)
        in_band = (peak >= bins.min()) & (peak <= bins.max())
        samples.append(xp.where(in_band, peak, 0.))
        resonance = w0 / (2 * xp.pi)
        bad = (w0 > 0) & (delta == 0) & (resonance >= bins.min()) & (resonance <= bins.max())
        if mask is not None:
            bad = bad & mask
        if not _live(bad) and np.any(bad):
            raise ValueError('an undamped Lorentz resonance lies inside the read band')
        invalid = invalid | jnp.any(bad)
        if xp is jnp:
            margin = margin + np.finfo(np.float32).eps * jnp.max(
                jnp.where(w0 > 0, 1 + w0 / jnp.where(delta > 0, 2 * delta, 1.), 0.))
    n = jnp.float32(1.)
    for frequency in samples:
        response = eps.astype(jnp.complex64)
        for rows, is_lorentz in ((dp, False), (lp, True)):
            for pole, mask in rows:
                chi = jnp.asarray(_response(pole, frequency, is_lorentz), jnp.complex64)
                response = response + (chi if mask is None else jnp.where(mask, chi, 0.))
        values = jnp.sqrt(jnp.abs(response * mu))
        if valid is not None:
            values = jnp.where(valid, values, 0.)
        n = jnp.maximum(n, jnp.max(values))
    for shape_eps in smoothing_eps:
        n = jnp.maximum(n, jnp.sqrt(jnp.max(jnp.asarray(shape_eps, jnp.float32)) * jnp.max(mu)))
    n = _up(jnp.where(invalid, jnp.inf, n * (1 + margin)))
    lengths, count = [], 0
    for axis in 'xyz':
        if axis == 'z' and getattr(grid, 'is_2d', False):
            lengths.append(0.)
            continue
        cells = grid.cells(axis)
        if axis not in getattr(grid, 'periodic_axes', ()):
            cells = cells[:-1]
        lo, hi = getattr(grid, f'pad_{axis}_lo'), getattr(grid, f'pad_{axis}_hi')
        cells = cells[lo:len(cells) - hi]
        count += len(cells)
        lengths.append(jnp.sum(cells) if _live(cells) else np.sum(cells, dtype=np.float64))
    if _live(lengths):
        gamma = (count + 8) * np.finfo(np.float32).eps
        gamma = gamma / (1 - gamma)
        distance = _up(jnp.linalg.norm(jnp.asarray(lengths)) / (1 - gamma))
    else:
        distance = np.asarray(np.linalg.norm(lengths), dtype=np.float64)
    group = np.asarray(echo_group_factor(configs), dtype=np.float64)
    tau = _up(2 * distance * n / (299792458.0 * group), 4)
    return dict(D=distance, g=group, n_max=n, tau_max=tau)


def runner_echo_bound(sim, grid, materials, *, freqs=None, debye=None, lorentz=None,
                      smoothing=False, waveguide_ports=(), valid=None):
    """Use precisely the smoothing shape reader and physical pole specs."""
    shapes = ()
    if smoothing:
        from rfx.geometry.smoothing import smoothed_shape_pairs
        shapes = tuple(eps for _, eps in smoothed_shape_pairs(sim, grid)[0])
    if freqs is None or not len(freqs):
        freqs = (0., sim._freq_max)
    return realized_echo_bound(grid, materials, freqs=freqs, debye=debye, lorentz=lorentz,
                              smoothing_eps=shapes, waveguide_ports=waveguide_ports, valid=valid)


def sharded_echo_bound(sim, grid, materials, sharded_grid, mesh, *, freqs=None,
                       debye=None, lorentz=None):
    """Reduce the staged volume arrays collectively, excluding alignment rows."""
    from dataclasses import replace
    from rfx.stepping.slab import cut
    from rfx.stepping.slab_view import uniform_slab_rows, stack_and_place
    wrapped = sharded_grid.step_kind == 'wrapped'
    layout = sharded_grid.layout

    def stage(value, kind):
        if wrapped:
            return stack_and_place(value, uniform_slab_rows(layout), mesh, outside=0.)
        return cut(value, layout, kind, mesh=mesh)

    def spec(source):
        rows = []
        masks = []
        for pole, mask in _poles(source):
            fields = {}
            for name in ('delta', 'kappa'):
                value = getattr(pole, name, None)
                if getattr(value, 'ndim', 0) == 3:
                    fields[name] = stage(value, 'lumped')
            if fields:
                pole = pole._replace(**fields) if hasattr(pole, '_replace') else replace(pole, **fields)
            rows.append(pole)
            masks.append(None if mask is None else stage(mask, 'pole_mask'))
        return (rows, masks) if rows else None

    valid = stage(np.ones(grid.shape, dtype=bool), 'pole_mask').astype(bool)
    return runner_echo_bound(sim, grid, materials, freqs=freqs,
        debye=spec(debye), lorentz=spec(lorentz), valid=valid)

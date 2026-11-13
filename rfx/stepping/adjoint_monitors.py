"""Monitor admission and wavelet sources for the uniform spectral adjoint."""
import numpy as np
import jax.numpy as jnp

from rfx._internal._precision import HIGHEST
from rfx.core.yee import MU_0, component_h_materials
from rfx.geometry.rasterize_grid import unpadded_high_walls
from rfx.measurement.adjoint_targets import ntff_face_slabs
from rfx.measurement.wavelet import _wavelet_basis, _wavelet_coefficients


def plane_slice(meta, shape):
    _, axis, index, _, region = meta
    transverse = [d for d in range(3) if d != axis]
    if region is None:
        region = (0, shape[transverse[0]], 0, shape[transverse[1]])
    out = [slice(region[0], region[1]), slice(region[2], region[3])]
    out.insert(axis, index)
    return tuple(out)


def monitor_frequencies(ctx):
    """Validate identical bins and every sampled edge, including NTFF halos."""
    bins = [np.asarray(m[3]) for m in ctx.dft_meta]
    slots = [plane_slice(m, ctx.grid.shape) for m in ctx.dft_meta]
    if ctx.use_ntff:
        bins.append(np.asarray(ctx.ntff.freqs))
        slots.extend(ntff_face_slabs(ctx.ntff))
    if not bins:
        raise NotImplementedError("gradient='adjoint' requires DFT monitors; time-domain objectives unsupported")
    for f in bins:
        if (f.ndim != 1 or not len(f) or not np.all(np.isfinite(f))
                or np.any(f <= 0) or np.any(f >= 0.5 / ctx.dt)
                or len(np.unique(f)) != len(f)):
            raise ValueError("gradient='adjoint' requires distinct positive sub-Nyquist bins")
    if any(not np.array_equal(f, bins[0]) for f in bins[1:]):
        if ctx.use_ntff:
            raise NotImplementedError(
                "gradient='adjoint' requires identical monitor frequency bins; "
                "NTFF boxes store bins as float32, so under float64 give DFT planes "
                "the float32-rounded values (np.asarray(freqs, np.float32))")
        raise NotImplementedError("gradient='adjoint' requires identical monitor frequency bins")
    margin = ctx.grid.cpml_layers if ctx.use_cpml else 1
    for slot in slots:
        for d, s in enumerate(slot):
            lo, hi = (s.start, s.stop) if isinstance(s, slice) else (s, s+1)
            if lo < margin + 1 or hi > ctx.grid.shape[d] - margin - 1:
                raise NotImplementedError("gradient='adjoint' requires interior monitor region outside CPML/boundaries")
    return bins[0]


def magnetic_wavelet_hook(ctx, targets, freqs, length, dtype):
    """S_H = -Db b_H exp(-i omega dt/2), on the integer-step basis."""
    mu = (ctx.materials.components.mu_update if ctx.materials.components is not None
          else component_h_materials(ctx.materials, ctx.periodic,
                                     high_wall=unpadded_high_walls(ctx.grid)))
    weights = []
    for name, slot, target in targets:
        half_step = jnp.exp(-1j * jnp.pi * freqs * ctx.dt)
        shifted = target * half_step.reshape((-1,) + (1,)*(target.ndim-1))
        w = _wavelet_coefficients(freqs, ctx.dt, length, shifted)
        m = mu[("hx", "hy", "hz").index(name)]
        db = ctx.dt / (MU_0 * (m if jnp.ndim(m) == 0 else m[slot]))
        weights.append((name, slot, w, db))

    def inject(frame):
        basis = _wavelet_basis(freqs, ctx.dt, length, frame.step_idx.astype(dtype))
        for name, slot, w, db in weights:
            value = 2*jnp.real(jnp.einsum('f,f...->...', basis, w, precision=HIGHEST))
            increment = -db * value
            frame.st = frame.st._replace(**{
                name: getattr(frame.st, name).at[slot].add(increment.astype(dtype))})
    return inject

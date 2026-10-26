"""Replay port samples through the time-step accumulators."""

from __future__ import annotations

import warnings
import numpy as np
from rfx.ringdown import _PortMeta, _port_vi


def _emulate_accumulators(lane: str, pm: _PortMeta, e32, h32, dt):
    """rfx's wire-port DFT accumulation re-run on the probe samples (W0).

    The scan body is the lane's wire block with the field reads replaced by
    the probe samples, using the shared physical E/H phase kernel and a
    sequential sum in the run's accumulator dtype.
    Returns ``((v, i, v_port) accumulators as numpy arrays of the run's dtype,
    (v_port, i) per step)`` -- the per-step series are the values the scan
    accumulated, in the dtype it computed them in (float32 on a float32 run).
    """
    import jax
    import jax.numpy as jnp
    from rfx.core.dft_utils import port_dft_phase

    freqs = pm.freqs

    def body(carry, xs):
        step_idx, e, hx, hy, hz = xs
        v_dft, i_dft, vp_dft = carry
        v, v_port, i_val = _port_vi(lane, pm, None, e, hx, hy, hz)
        phase = port_dft_phase(step_idx, freqs, dt, dtype=v_dft.dtype)
        i_phase = port_dft_phase(step_idx, freqs, dt, 'H', dtype=i_dft.dtype)
        return (v_dft + v * phase, i_dft + i_val * i_phase,
                vp_dft + v_port * phase), (v_port, i_val)

    nf = int(freqs.shape[0])
    # The run's own carry dtype: complex64, or complex128 on the uniform lane
    # under x64 (``rfx.simulation``'s ``_sparam_acc_dtype``).
    init = tuple(jnp.zeros(nf, dtype=np.asarray(a).dtype) for a in pm.accs)
    n = e32.shape[0]
    xs = (jnp.arange(n, dtype=jnp.int32), jnp.asarray(e32),
          jnp.asarray(h32["hx"]), jnp.asarray(h32["hy"]), jnp.asarray(h32["hz"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final, per_step = jax.jit(lambda c, x: jax.lax.scan(body, c, x))(init, xs)
    return tuple(np.asarray(a) for a in final), tuple(np.asarray(a) for a in per_step)

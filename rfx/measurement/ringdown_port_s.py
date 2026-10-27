"""Port completion assembly and measured-column witness projections."""
import numpy as np


def _assemble(lane: str, pms, spectra_vp, spectra_i, freqs32, dt, v_mid=None):
    """The lane's own S-matrix assembly on given (V_port, I) spectra per port.

    ``spectra_vp[p]`` and ``spectra_i[p]`` are the accumulator-convention
    spectra (the current already carries its half-step phase). They enter the
    assembly in the dtype of the run's own accumulators, as the run's do.
    Returns a numpy array shaped and typed like ``Result.s_params``.
    """
    import jax.numpy as jnp

    nf = int(np.asarray(freqs32).shape[0])
    acc_dtype = np.asarray(pms[0].accs[2]).dtype

    def as_acc(a):
        return jnp.asarray(np.asarray(a), dtype=acc_dtype)

    S_lane = _assemble_core(
        lane, pms, [as_acc(a) for a in spectra_vp], [as_acc(a) for a in spectra_i],
        freqs32, dt, v_mid=None if v_mid is None else [as_acc(a) for a in v_mid])
    if lane == "uniform":
        # the runner's container (rfx/runners/uniform.py, single-port wire path)
        S = np.zeros((1, 1, nf), dtype=np.complex64)
        S[0, 0, :] = np.array(S_lane)
        return S
    return np.asarray(S_lane)


def _assemble_core(lane: str, pms, vp, ii, freqs32, dt, v_mid=None):
    """The lane's S assembly as JAX operations (traceable).

    ``vp[p]`` and ``ii[p]`` are the port's V_port and I spectra in the dtype
    of the run's accumulators, the current carrying its half-step phase. The
    Both lanes return ``(P, P, nf)``; uniform admission requires one port.
    Uniform completion preserves its measured reflection values.
    """
    import jax.numpy as jnp

    if lane == "uniform":
        from rfx.probes.probes import driven_port_reflection
        (pm,) = pms
        return driven_port_reflection(vp[0], ii[0], pm.z0)[None, None, :]
    from types import SimpleNamespace

    from rfx.measurement.graded_result import assemble_graded_result as _assemble_nu_result
    nf = int(np.asarray(freqs32).shape[0])
    zeros = jnp.zeros(nf, dtype=vp[0].dtype)
    accs = tuple(
        (zeros if v_mid is None else v_mid[p], ii[p], zeros, vp[p])
        for p in range(len(pms)))
    metas = [pm.meta for pm in pms]
    setup = SimpleNamespace(
        dt=float(dt), dft_planes=None, flux_monitors=None,
        wire_ports=[{'kind': pm.owner.kind} for pm in pms],
        # Spectral completion reuses S assembly without a time-step scan.
        step_fn_inv=SimpleNamespace(port_owners=(), port_acc_indices=()),
        waveguide_meta=None, wp_meta=metas,
        sp_freqs=jnp.asarray(np.asarray(freqs32), dtype=jnp.float32),
        use_wire_ports=True, use_dft_planes=False, use_flux_monitors=False,
        use_lumped_rlc=False, use_ntff=False, use_waveguide_ports=False)
    r = _assemble_nu_result(setup, {"fdtd": None, "wire_sparams": accs}, None)
    return r["s_params"]


def _s_observable(lane: str, pms, freqs32, dt):
    """``(f_bins, to_s)``: the float64 bins and the map from completed port
    spectra ``(nf, 2P)`` (``V_port, I`` per port) to the lane's S-matrix.

    The current's half-step phase is applied here, as the run's accumulators
    carry it. :meth:`RingdownRun._complete` and the early stop's check
    (:class:`RingdownStop`) both judge ``WE`` through this one map.
    """
    from rfx.ringdown import _assemble, _cmul
    from rfx.core.dft_utils import half_step_current_phase

    f_bins = np.asarray(freqs32, dtype=np.float32).astype(np.float64)
    half = np.asarray(half_step_current_phase(f_bins, dt)).astype(np.complex128)
    nf = int(f_bins.size)

    def to_s(spectra):
        # ``spectra`` may hold several copies of the bins stacked along its
        # first axis (``_tail_influence``); the S axis then holds them in turn.
        reps = int(np.shape(spectra)[0]) // nf
        h = half if reps == 1 else np.tile(half, reps)
        fq = freqs32 if reps == 1 else np.tile(np.asarray(freqs32), reps)
        vp = [spectra[:, 2 * p] for p in range(len(pms))]
        ii = [_cmul(spectra[:, 2 * p + 1], h) for p in range(len(pms))]
        return _assemble(lane, pms, vp, ii, fq, dt)

    to_s.measured = lambda spectra: measured_s(pms, to_s(spectra))
    return f_bins, to_s


def measured_s(pms, s_params):
    driven = [p for p, pm in enumerate(pms) if pm.excite]
    return np.asarray(s_params)[:, driven, :] if driven else np.asarray(s_params)

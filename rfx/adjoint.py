"""Settled-spectrum reciprocity adjoint of the uniform production scan (#1424)."""
from dataclasses import replace

import jax
import jax.numpy as jnp

from rfx._precision import HIGHEST


def admit_forward_adjoint(sim, *, distributed, ringdown, design_box, design_eps,
                          other_overrides, port_s11_freqs, design_sigma=None):
    """Refuse inputs outside the first implementation before lane dispatch."""
    if design_sigma is not None:
        raise NotImplementedError(
            "gradient='adjoint' refuses design_sigma_override: #1424 "
            "the conductivity derivative is not validated")
    reason = None
    if distributed:
        reason = "distributed execution"
    elif any(p is not None for p in
             (sim._dx_profile, sim._dy_profile, sim._dz_profile)):
        reason = "graded meshes"
    elif ringdown is not None:
        reason = "ring-down"
    elif (any(p.impedance != 0 or p.extent is not None for p in sim._ports)
          or any(getattr(sim, p) for p in
                 ("_waveguide_ports", "_msl_ports", "_coaxial_ports",
                  "_floquet_ports", "_lumped_rlc"))
          or port_s11_freqs is not None):
        reason = "ports and lumped circuits"
    elif sim._tfsf is not None:
        reason = "TFSF sources"
    elif any(m.debye_poles or m.lorentz_poles for m in sim._materials.values()):
        reason = "Debye/Lorentz materials"
    elif any(m.chi3 != 0 for m in sim._materials.values()):
        reason = "Kerr materials"
    elif sim._current_moments is not None:
        reason = "current moments"
    elif sim._solver != "yee":
        reason = "non-Yee solvers"
    elif any(v is not None for v in other_overrides):
        reason = "overrides other than design eps"
    elif design_box is None or design_eps is None:
        reason = "calls without design_box and design_eps_override"
    if reason is not None:
        raise NotImplementedError(f"gradient='adjoint' does not support {reason}")


def _plane_slice(meta, shape):
    _, axis, index, _, region = meta
    transverse = [d for d in range(3) if d != axis]
    if region is None:
        region = (0, shape[transverse[0]], 0, shape[transverse[1]])
    out = [slice(region[0], region[1]), slice(region[2], region[3])]
    out.insert(axis, index)
    return tuple(out)


def _wavelet_basis(freqs, dt, length, n):
    """Compact discrete derivative: telescoping sum has no deposited DC."""
    def primitive(k):
        envelope = jnp.where((k >= 0) & (k < length - 1),
                            jnp.sin(jnp.pi * (k + 1) / length)**4, 0.)
        return envelope[..., None] * jnp.exp(
            2j * jnp.pi * ((k + 1) * dt)[..., None] * freqs)
    return primitive(n) - primitive(n - 1)


def _wavelet_coefficients(freqs, dt, length, target):
    """Real compact wavelet, exact complex bin targets via Nf-square solves."""
    t = (jnp.arange(length, dtype=freqs.dtype) + 1) * dt
    carrier = jnp.exp(2j * jnp.pi * t[:, None] * freqs)
    basis = _wavelet_basis(freqs, dt, length, jnp.arange(length, dtype=freqs.dtype))
    transform = jnp.conj(carrier).T * dt
    a = jnp.matmul(transform, basis, precision=HIGHEST)
    b = jnp.matmul(transform, jnp.conj(basis), precision=HIGHEST)
    flat = target.reshape((len(freqs), -1))
    cross = jnp.linalg.solve(jnp.conj(a), jnp.conj(b))
    rhs = flat - jnp.matmul(b, jnp.linalg.solve(jnp.conj(a), jnp.conj(flat)), precision=HIGHEST)
    return jnp.linalg.solve(a - jnp.matmul(b, cross, precision=HIGHEST), rhs).reshape(target.shape)


def design_adjoint_scan(ctx, initial, xs):
    """Two ordinary forward scans with frequency-sized local residuals.

    Point objectives select a pixel of a registered E-plane DFT. Direct
    records/final-field objectives are refused through symbolic cotangents.
    """
    import numpy as np
    from rfx.core.jax_utils import recorded_scan
    from rfx.core.yee import curl_h, e_component_coeffs
    from rfx.simulation import make_core_step, core_step_invariants

    unsupported = ("use_debye", "use_lorentz", "use_kerr", "use_upml",
                   "use_design_occupancy", "use_current_moments",
                   "use_sheet_impedance", "use_tfsf", "use_waveguide_ports",
                   "use_lumped_rlc", "use_wire_sparams", "use_lumped_sparams",
                   "use_ntff", "use_flux_monitors", "use_aniso_inv",
                   "use_conformal", "use_pec_occupancy", "use_pmc_faces")
    for name in unsupported:
        if getattr(ctx, name):
            raise NotImplementedError(f"gradient='adjoint' does not support {name}")
    if not ctx.use_design_box:
        raise NotImplementedError("gradient='adjoint' requires a design box")
    if ctx.bloch is not None or any(ctx.periodic) or ctx.aniso_eps is not None:
        raise NotImplementedError("gradient='adjoint' does not support Bloch/periodic/anisotropy")
    dtype = initial["fdtd"].ex.dtype
    if dtype not in (jnp.float32, jnp.float64) or ctx.stencil_order != 2:
        raise NotImplementedError("gradient='adjoint' requires float32/float64 and stencil_order=2")
    if not ctx.dft_meta:
        raise NotImplementedError("gradient='adjoint' requires E DFT monitors; time-domain objectives unsupported")
    freqs_host = []
    for meta in ctx.dft_meta:
        if meta[0] not in ("ex", "ey", "ez"):
            raise NotImplementedError("gradient='adjoint' does not support H DFT planes")
        f = np.asarray(meta[3])
        if (not np.all(np.isfinite(f)) or np.any(f <= 0)
                or np.any(f >= 0.5 / ctx.dt) or len(np.unique(f)) != len(f)):
            raise ValueError("gradient='adjoint' requires distinct positive sub-Nyquist bins")
        freqs_host.extend(f.tolist())
        slm = _plane_slice(meta, ctx.grid.shape)
        margin = ctx.grid.cpml_layers if ctx.use_cpml else 1
        for d, s in enumerate(slm):
            lo, hi = (s.start, s.stop) if isinstance(s, slice) else (s, s+1)
            if lo < margin + 1 or hi > ctx.grid.shape[d] - margin - 1:
                raise NotImplementedError("gradient='adjoint' requires interior monitor region outside CPML/boundaries")
    common = np.asarray(ctx.dft_meta[0][3])
    if any(not np.array_equal(np.asarray(m[3]), common) for m in ctx.dft_meta):
        raise NotImplementedError("gradient='adjoint' requires identical monitor frequency bins")
    freqs = jnp.asarray(common, dtype=dtype)
    bounds = ctx.design_box.bounds
    sl = tuple(slice(bounds[2*d], bounds[2*d+1]) for d in range(3))
    ab = (ctx.design_box.ca, ctx.design_box.cb)
    base = replace(ctx, design_box=ctx.design_box._replace(ca=None, cb=None))
    invariants = core_step_invariants(base)
    nf = len(freqs)
    cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    z = tuple(jnp.zeros((nf,) + a.shape, cdtype) for a in ab[0])
    steps = len(xs[0])
    wavelet_length = min(steps, max(16, steps // 4))

    def run(coeffs, targets=None, recording=False):
        local = replace(base, design_box=base.design_box._replace(ca=coeffs[0], cb=coeffs[1]))
        if targets is not None:
            # Suppress primal drives. CPML and ordinary Yee updates are reused.
            local = replace(local, src_meta=(), mag_src_meta=(), use_mag_sources=False,
                            use_dft_planes=False, dft_meta=(), prb_meta=())
            monitor_cb = list(e_component_coeffs(ctx.materials, ctx.dt, ctx.periodic)[1])
            monitor_cb = [a.at[sl].set(b) for a, b in zip(monitor_cb, coeffs[1])]
            weights = [_wavelet_coefficients(jnp.asarray(m[3], dtype=dtype), ctx.dt,
                                             wavelet_length, t)
                       for m, t in zip(ctx.dft_meta, targets)]

        def hook(prev, state):
            e = tuple(getattr(prev, c)[sl] for c in ("ex", "ey", "ez"))
            curl = tuple(v[sl] for v in curl_h(prev.hx, prev.hy, prev.hz,
                                            ctx.dx, ctx.periodic, ctx.stencil_order, ctx.bloch))
            if targets is not None:
                n = prev.step
                for m, w in zip(ctx.dft_meta, weights):
                    c = m[0]
                    slot = _plane_slice(m, ctx.grid.shape)
                    basis = _wavelet_basis(jnp.asarray(m[3], dtype=dtype), ctx.dt,
                                           wavelet_length, n.astype(dtype))
                    value = 2 * jnp.real(jnp.einsum('f,fij->ij', basis, w, precision=HIGHEST))
                    cb = monitor_cb[("ex", "ey", "ez").index(c)][slot]
                    state = state._replace(**{c: getattr(state, c).at[slot].add((cb*value).astype(dtype))})
            return state, (e, curl)

        core = make_core_step(local, {**invariants, "ctx": local}, design_hook=hook if recording else None)
        def step(carry, row):
            state, acc_e, acc_h, peak, last_e = carry
            state, probes, extras = core(state, *row)
            if targets is None:
                # Yee E edges owned by the design box, after the full update.
                last_e = jnp.max(jnp.stack([
                    jnp.max(jnp.abs(getattr(state["fdtd"], c)[sl]))
                    for c in ("ex", "ey", "ez")]))
                peak = jnp.maximum(peak, last_e)
            if recording:
                # Forward Epre and curl use the post-update timestamp. Adjoint
                # E is post-update; its DFT then represents L^-1 b.
                e, h = extras["design_record"]
                if targets is not None:
                    e = tuple(getattr(state["fdtd"], c)[sl] for c in ("ex", "ey", "ez"))
                phase = (jnp.exp(-2j*jnp.pi*freqs*(row[0]+1)*ctx.dt)*ctx.dt).astype(cdtype)
                acc_e = tuple(a + phase[:, None, None, None]*v for a, v in zip(acc_e, e))
                if targets is None:
                    acc_h = tuple(a + phase[:, None, None, None]*v for a, v in zip(acc_h, h))
            return (state, acc_e, acc_h, peak, last_e), (probes,) if targets is None else None
        start = initial if targets is None else {k: v for k, v in initial.items() if k != "dft_planes"}
        (last, e, h, peak, last_e), outputs = recorded_scan(
            step, (start, z, z, jnp.zeros((), dtype), jnp.zeros((), dtype)), xs)
        if targets is None:
            # No excitation gives no settling evidence, rather than a false zero.
            last = {**last, "adjoint_settling": jax.lax.stop_gradient(
                jnp.where(peak > 0, last_e / peak, jnp.nan))}
        return (last, e, h), outputs

    @jax.custom_vjp
    def scan(coeffs):
        (last, _, _), outputs = run(coeffs)
        return last, outputs

    def forward(coeffs):
        coeffs = jax.tree.map(lambda p: p.value, coeffs)
        (last, e, h), outputs = run(coeffs, recording=True)
        return (last, outputs), (coeffs, e, h)

    def backward(residual, cotangents):
        coeffs, e, h = residual
        last_bar, outputs_bar = cotangents
        symbolic = jax.custom_derivatives.SymbolicZero
        for name, bar in last_bar.items():
            if name != "dft_planes" and any(not isinstance(v, symbolic) for v in jax.tree.leaves(bar)):
                raise NotImplementedError("gradient='adjoint' does not support final-field/time-domain objectives")
        if any(not isinstance(v, symbolic) for v in jax.tree.leaves(outputs_bar)):
            raise NotImplementedError("gradient='adjoint' does not support time-domain objectives; register E DFT bins")
        targets = [jnp.zeros_like(a) if isinstance(b, symbolic) else b
                   for a, b in zip(initial["dft_planes"], last_bar["dft_planes"])]
        (_, adj, _), _ = run(coeffs, targets=targets, recording=True)
        grads = tuple(tuple(jnp.real(jnp.sum(a*f, axis=0)/cb).astype(c.dtype)
                            for a, f, cb, c in zip(adj, fields, coeffs[1], cs))
                      for fields, cs in zip((e, h), coeffs))
        return (grads,)

    scan.defvjp(forward, backward, symbolic_zeros=True)
    last, outputs = scan(ab)
    return {**last, "adjoint_settling": jax.lax.stop_gradient(
        last["adjoint_settling"])}, outputs

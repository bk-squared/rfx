"""Settled-spectrum reciprocity adjoint of the uniform production scan (#1424).

Discrete overlap
----------------

Use the production DFT convention `F = dt sum_n f[n] exp(-i w n dt)`.
E is sampled after the update, at `(n+1)dt`; the design hook observes
`E[n]` and `curl H[n+1/2]`. For each bin, eliminating H gives a symmetric
uniform interior electric operator L with `L E = q/Cb`, where q is an
additive E increment with the post-update timestamp. Reciprocity gives
`dY_m = sum_j L^-1[m,j] (dCa_j Epre_j + dCb_j curlH_j)/Cb_j`.
For JAX's complex cotangent b, `dJ = Re sum_m b_m dY_m` (no conjugation).
An ordinary second forward run with `DFT(q_m)=Cb_m b_m` produces A on the
design edges. Therefore

dJ/dCa_j = Re sum_bins A_j Epre_j / Cb_j
dJ/dCb_j = Re sum_bins A_j curlH_j / Cb_j.

Both field factors include dt; the source DFT targets b, not b*dt.
JAX differentiates the production edge averaging and Ca/Cb arithmetic
outside the custom VJP. Public adjoint admission refuses every supplied
`design_sigma_override` (#1424: conductivity derivative not validated).
Fixed material conductivity remains allowed; its dCa/deps reaches gCa.
Finite record endpoint terms are omitted: the result is a settled-spectrum
gradient. No time tape or transposed time sweep is part of F2.

Injection and wavelets
----------------------

Inject the monitor's E component at its exact Yee array index, after the
design update and before the ordinary source/monitor stage. A point is a
single selected pixel of an E DFT plane; a plane injects each of its pixels.
Interior uniform dual volumes are V=dx^3: reciprocal density sources use
b/V and the overlap uses V, so these factors cancel. CPML and boundary
monitor pixels are excluded from admission. The half-step current spectrum
is `exp(+i w dt/2)` times the post-update increment spectrum; equivalently
q uses the E timestamp and no additional phase. Epre uses the same timestamp
as q, giving its explicit one-step delay relative to post-update E.

For each monitor use compact early real wavelets formed from positive and
negative complex exponentials with a smooth envelope. Take the compact
discrete difference of each carrier to enforce zero deposited DC. Their sampled
DFTs form A (positive carriers) and B (negative carriers). Solve
`(A-B conj(A)^-1 conj(B)) c = b-B conj(A)^-1 conj(b)` and inject
`2 Re sum_k c_k difference(envelope exp(+i w_k t))`. These are Nf-by-Nf solves;
including the conjugate carriers makes arbitrary complex targets reachable
with real fields. Monitors must share identical bins. Reject empty, duplicate,
DC/Nyquist/out-of-band bins.
The remaining run lets the wavelet response decay. Storage is design
edges × components × bins, plus monitor bins and ordinary field carries.

Magnetic increments
-------------------
Let z = exp(i w dt), D = dt/mu, and stamp both E and H at (n+1)dt.
The H equation is (1-z^-1) H = -D z^-1 curl E + p. Eliminating H
puts curl^T p/(1-z^-1) on the electric right-hand side. Transposing an
H-sample objective instead gives -curl^T D z^-1 b/(1-z^-1). Therefore
inject the magnetic increment with DFT(p) = -D z^-1 b, before the E
update. The minus sign is mixed reciprocity; z^-1 includes the two
half-step registers for an H plane stamped at the E timestamp.

NTFF surface sources
--------------------
The spatial sampling map is the per-step accumulator increment evaluated
with unit dt and a zero frequency. jax.linear_transpose of this map distributes
face cotangents back to the original Yee samples, including every face-centre
average and repeated edge/corner contribution. The wavelet solve stays on the
six faces; the spatial transpose runs on each instantaneous wavelet, so no
frequency-by-volume source array is stored. NTFF stamps H at (n+1/2)dt,
so its cotangent is multiplied by exp(+i w dt/2) before the H-plane rule
above: DFT(p) = -D exp(-i w dt/2) b. E uses the unchanged electric rule.
Only the six accumulated faces are observables; Kahan compensation outputs
are refused. Flux monitors remain refused.

Gates, measurements and the refusal table: issue #1424 and PR #1430.
"""
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


def _magnetic_target(freqs, dt, target):
    """Mixed reciprocity for H sampled with the post-E timestamp."""
    phase = -jnp.exp(-2j * jnp.pi * freqs * dt)
    return phase.reshape((-1,) + (1,) * (target.ndim - 1)) * target


def _ntff_transpose(ctx, state):
    """Transpose the accumulator's own spatial increment, without a field tape."""
    from rfx.farfield import accumulate_ntff, init_ntff_data
    components = ("ex", "ey", "ez", "hx", "hy", "hz")
    # Unit dt and a zero bin factor out ONLY the temporal DFT multiplier.
    # All spatial indices and face-centre weights remain the accumulator's.
    box = ctx.ntff._replace(freqs=jnp.zeros((1,), dtype=state.ex.dtype))
    zero = init_ntff_data(box, field_dtype=state.ex.dtype)
    fields = tuple(jnp.zeros_like(getattr(state, c)) for c in components)

    def increment(fields):
        st = state._replace(**dict(zip(components, fields)))
        value = accumulate_ntff(zero, st, box, 1., 0)
        return tuple(jnp.real(a[0]) for a in value[:6])

    return jax.linear_transpose(increment, fields)


def design_adjoint_scan(ctx, initial, xs):
    """Two ordinary forward scans with frequency-sized local residuals.

    Point objectives select a pixel of a registered E-plane DFT. Direct
    records/final-field objectives are refused through symbolic cotangents.
    """
    import numpy as np
    from rfx.core.jax_utils import recorded_scan
    from rfx.core.yee import curl_h, e_component_coeffs, component_h_materials, MU_0
    from rfx.simulation import make_core_step, core_step_invariants

    unsupported = ("use_debye", "use_lorentz", "use_kerr", "use_upml",
                   "use_design_occupancy", "use_current_moments",
                   "use_sheet_impedance", "use_tfsf", "use_waveguide_ports",
                   "use_lumped_rlc", "use_wire_sparams", "use_lumped_sparams",
                   "use_flux_monitors", "use_aniso_inv",
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
    bins = [np.asarray(m[3]) for m in ctx.dft_meta]
    if ctx.use_ntff:
        bins.append(np.asarray(ctx.ntff.freqs))
    if not bins:
        raise NotImplementedError("gradient='adjoint' requires DFT monitors; time-domain objectives unsupported")
    for f in bins:
        if (f.ndim != 1 or not len(f) or not np.all(np.isfinite(f)) or np.any(f <= 0)
                or np.any(f >= 0.5 / ctx.dt) or len(np.unique(f)) != len(f)):
            raise ValueError("gradient='adjoint' requires distinct positive sub-Nyquist bins")
    common = bins[0]
    if any(not np.array_equal(f, common) for f in bins):
        raise NotImplementedError("gradient='adjoint' requires identical monitor frequency bins")
    margin = ctx.grid.cpml_layers if ctx.use_cpml else 1
    for meta in ctx.dft_meta:
        slm = _plane_slice(meta, ctx.grid.shape)
        for d, sample in enumerate(slm):
            lo, hi = (sample.start, sample.stop) if isinstance(sample, slice) else (sample, sample+1)
            if lo < margin + 1 or hi > ctx.grid.shape[d] - margin - 1:
                raise NotImplementedError("gradient='adjoint' requires interior monitor region outside CPML/boundaries")
    if ctx.use_ntff:
        box = ctx.ntff
        # Include the lower H neighbour and upper in-plane E neighbour.
        for d, (lo, hi) in enumerate(((box.i_lo, box.i_hi), (box.j_lo, box.j_hi),
                                      (box.k_lo, box.k_hi))):
            if lo - int(box.face_centre) < margin + 1 or hi >= ctx.grid.shape[d] - margin - 1:
                raise NotImplementedError("gradient='adjoint' requires interior NTFF samples outside CPML/boundaries")
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
                            use_dft_planes=False, dft_meta=(), prb_meta=(), use_ntff=False)
            monitor_cb = list(e_component_coeffs(ctx.materials, ctx.dt, ctx.periodic)[1])
            monitor_cb = [a.at[sl].set(b) for a, b in zip(monitor_cb, coeffs[1])]
            monitor_ch = tuple(ctx.dt / (MU_0 * m) for m in component_h_materials(ctx.materials))
            plane_targets, ntff_targets = targets
            plane_targets = [(_magnetic_target(freqs, ctx.dt, t) if m[0].startswith("h") else t)
                       for m, t in zip(ctx.dft_meta, plane_targets)]
            weights = [_wavelet_coefficients(jnp.asarray(m[3], dtype=dtype), ctx.dt,
                                             wavelet_length, t)
                       for m, t in zip(ctx.dft_meta, plane_targets)]
            if ctx.use_ntff:
                transpose_ntff = _ntff_transpose(ctx, initial["fdtd"])
                # NTFF stamps H half a step earlier than the plane monitor.
                half_phase = jnp.exp(1j * jnp.pi * freqs * ctx.dt)[:, None, None, None]
                ntff_targets = [jnp.concatenate((t[..., :2],
                    _magnetic_target(freqs, ctx.dt, t[..., 2:] * half_phase)), axis=-1)
                    for t in ntff_targets]
                ntff_weights = [_wavelet_coefficients(freqs, ctx.dt, wavelet_length, t)
                                for t in ntff_targets]

        def inject(state, magnetic):
            n = state.step if magnetic else state.step - 1
            basis = _wavelet_basis(freqs, ctx.dt, wavelet_length, n.astype(dtype))
            for m, w in zip(ctx.dft_meta, weights):
                c = m[0]
                if c.startswith("h") != magnetic:
                    continue
                slot = _plane_slice(m, ctx.grid.shape)
                value = 2 * jnp.real(jnp.einsum("f,fij->ij", basis, w, precision=HIGHEST))
                components = ("hx", "hy", "hz") if magnetic else ("ex", "ey", "ez")
                coefficient = (monitor_ch if magnetic else monitor_cb)[components.index(c)][slot]
                state = state._replace(**{c: getattr(state, c).at[slot].add((coefficient*value).astype(dtype))})
            if ctx.use_ntff:
                faces = tuple(2 * jnp.real(jnp.einsum("f,fijc->ijc", basis, w, precision=HIGHEST))
                              for w in ntff_weights)
                fields, = transpose_ntff(faces)
                names = ("hx", "hy", "hz") if magnetic else ("ex", "ey", "ez")
                selected = fields[3:] if magnetic else fields[:3]
                coefficients = monitor_ch if magnetic else monitor_cb
                state = state._replace(**{c: getattr(state, c) + (a * v).astype(dtype)
                    for c, a, v in zip(names, coefficients, selected)})
            return state

        def hook(prev, state):
            e = tuple(getattr(prev, c)[sl] for c in ("ex", "ey", "ez"))
            curl = tuple(v[sl] for v in curl_h(prev.hx, prev.hy, prev.hz,
                                            ctx.dx, ctx.periodic, ctx.stencil_order, ctx.bloch))
            if targets is not None:
                state = inject(state, False)
            return state, (e, curl)

        core = make_core_step(local, {**invariants, "ctx": local}, design_hook=hook if recording else None,
                              magnetic_hook=(lambda st: inject(st, True)) if targets is not None else None)
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
        start = initial if targets is None else {k: v for k, v in initial.items() if k not in ("dft_planes", "ntff")}
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
            if name not in ("dft_planes", "ntff") and any(not isinstance(v, symbolic) for v in jax.tree.leaves(bar)):
                raise NotImplementedError("gradient='adjoint' does not support final-field/time-domain objectives")
        if any(not isinstance(v, symbolic) for v in jax.tree.leaves(outputs_bar)):
            raise NotImplementedError("gradient='adjoint' does not support time-domain objectives; register DFT bins")
        def materialize(a, b):
            return jnp.zeros_like(a) if isinstance(b, symbolic) else b
        plane_targets = [materialize(a, b) for a, b in
                         zip(initial.get("dft_planes", ()), last_bar.get("dft_planes", ()))]
        ntff_targets = []
        if ctx.use_ntff:
            # Compensation is bookkeeping, not a frequency-domain observable.
            if any(not isinstance(b, symbolic) for b in last_bar["ntff"][6:]):
                raise NotImplementedError("gradient='adjoint' does not support NTFF compensation objectives")
            ntff_targets = [materialize(a, b) for a, b in
                            zip(initial["ntff"][:6], last_bar["ntff"][:6])]
        targets = (plane_targets, ntff_targets)
        (_, adj, _), _ = run(coeffs, targets=targets, recording=True)
        grads = tuple(tuple(jnp.real(jnp.sum(a*f, axis=0)/cb).astype(c.dtype)
                            for a, f, cb, c in zip(adj, fields, coeffs[1], cs))
                      for fields, cs in zip((e, h), coeffs))
        return (grads,)

    scan.defvjp(forward, backward, symbolic_zeros=True)
    last, outputs = scan(ab)
    return {**last, "adjoint_settling": jax.lax.stop_gradient(
        last["adjoint_settling"])}, outputs

"""Exact local-design discrete adjoint of the uniform production scan (#1424)."""
from dataclasses import replace

import jax
import jax.numpy as jnp


def admit_forward_adjoint(sim, *, distributed, ringdown, design_box, design_eps,
                          other_overrides, port_s11_freqs):
    """Refuse inputs outside the first implementation before lane dispatch."""
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
        reason = "overrides other than design eps and design sigma"
    elif design_box is None or design_eps is None:
        reason = "calls without design_box and design_eps_override"
    if reason is not None:
        raise NotImplementedError(f"gradient='adjoint' does not support {reason}")


def design_adjoint_scan(ctx, initial, xs):
    """Scan with six local field arrays per step and no carry checkpoints.

    Coefficient pullbacks outside this custom VJP retain the production edge
    averaging, lossy coefficient arithmetic and design sigma representation.
    """
    from rfx.core.yee import curl_h
    from rfx.simulation import make_core_step, core_step_invariants

    if not ctx.use_design_box:
        raise NotImplementedError("gradient='adjoint' requires a design box")
    unsupported = ("use_debye", "use_lorentz", "use_kerr", "use_upml",
                   "use_design_occupancy", "use_current_moments",
                   "use_sheet_impedance", "use_tfsf", "use_waveguide_ports",
                   "use_lumped_rlc", "use_wire_sparams", "use_lumped_sparams")
    for name in unsupported:
        if getattr(ctx, name):
            raise NotImplementedError(f"gradient='adjoint' does not support {name}")
    if ctx.bloch is not None or jnp.iscomplexobj(initial["fdtd"].ex):
        raise NotImplementedError("gradient='adjoint' does not support complex/Bloch fields")
    if initial["fdtd"].ex.dtype not in (jnp.float32, jnp.float64):
        raise NotImplementedError("gradient='adjoint' requires float32 or float64 fields")
    bounds = ctx.design_box.bounds
    sl = tuple(slice(bounds[2*d], bounds[2*d+1]) for d in range(3))
    coeffs = (ctx.design_box.ca, ctx.design_box.cb)
    # Strip design tracers from the closed-over context. Only the explicit
    # custom-VJP argument carries their dependence into the time loop.
    base = replace(ctx, design_box=ctx.design_box._replace(ca=None, cb=None))
    invariants = core_step_invariants(base)
    zero = jax.tree.map(jnp.zeros_like, initial)

    def kernel(ab, hook=None):
        local = replace(base, design_box=base.design_box._replace(ca=ab[0], cb=ab[1]))
        return make_core_step(local, {**invariants, "ctx": local}, design_hook=hook)

    def record(prev, state):
        dtype = jnp.promote_types(prev.ex.dtype, jnp.float32)
        e = tuple(getattr(prev, c)[sl].astype(dtype) for c in ("ex", "ey", "ez"))
        h = tuple(getattr(prev, c).astype(dtype) for c in ("hx", "hy", "hz"))
        curl = tuple(c[sl] for c in curl_h(*h, ctx.dx, ctx.periodic,
                                         ctx.stencil_order, ctx.bloch))
        return state, (e, curl)

    def run(ab, carry, recording):
        core = kernel(ab, record if recording else None)
        def step(c, row):
            c, probes, extras = core(c, *row)
            out = (probes,)
            return c, (out, extras["design_record"]) if recording else out
        return jax.lax.scan(step, carry, xs)

    @jax.custom_vjp
    def scan(ab, carry):
        return run(ab, carry, False)

    def forward(ab, carry):
        last, (outputs, tape) = run(ab, carry, True)
        return (last, outputs), (ab, tape)

    def backward(residual, cotangents):
        ab, tape = residual
        last_bar, output_bars = cotangents
        injection = tuple(jnp.zeros_like(a, dtype=initial["fdtd"].ex.dtype)
                          for a in ab[0])
        def reverse(c, row):
            carry_bar, ab_bar = c
            x, output_bar, (e, curl) = row
            # Integer clock is prescribed by the scan index; all differentiable
            # carry entries are zero. The affine source does not affect the VJP.
            primal = {**zero, "fdtd": zero["fdtd"]._replace(step=x[0])}
            def linear_step(state, delta):
                def inject(prev, st):
                    st = st._replace(**{
                        name: getattr(st, name).at[sl].add(v)
                        for name, v in zip(("ex", "ey", "ez"), delta)})
                    return st, ()
                out, probes, _ = kernel(ab, inject)(state, *x)
                return out, (probes,)
            _, pullback = jax.vjp(linear_step, primal, injection)
            prev_bar, local_bar = pullback((carry_bar, output_bar))
            increments = (tuple(g*v for g, v in zip(local_bar, e)),
                          tuple(g*v for g, v in zip(local_bar, curl)))
            ab_bar = jax.tree.map(lambda a, b: a + b.astype(a.dtype), ab_bar, increments)
            return (prev_bar, ab_bar), None
        (first_bar, ab_bar), _ = jax.lax.scan(
            reverse, (last_bar, jax.tree.map(jnp.zeros_like, ab)),
            (xs, output_bars, tape), reverse=True)
        return ab_bar, first_bar

    scan.defvjp(forward, backward)
    return scan(coeffs, initial)

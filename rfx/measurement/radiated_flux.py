"""Shared radiated-flux envelope for the uniform and graded decay loops."""

import jax.numpy as jnp


def flux_box_bounds(grid, box):
    """Resolve closed normal planes and half-open tangential cell windows."""
    from rfx._periodic import interval_indices
    lo, hi = interval_indices(grid, *box)
    return (min(lo[0], hi[0]), max(lo[0], hi[0]),
            min(lo[1], hi[1]), max(lo[1], hi[1]),
            min(lo[2], hi[2]), max(lo[2], hi[2]))


def radiated_power(state, bounds, shape) -> float:
    """Net outgoing Poynting flux P = ∮ (E×H)·n̂ dA over the box (co-located approx; a stop
    criterion needs only the DECAY, so the half-cell Yee stagger is neglected). The dA=dx²
    factor is a constant that cancels in P/peak_P."""
    ex, ey, ez = state.ex, state.ey, state.ez
    hx, hy, hz = state.hx, state.hy, state.hz
    il, ih, jl, jh, kl, kh = bounds
    jj, kk = slice(jl, jh), slice(kl, kh)
    ii = slice(il, ih)
    # Normal planes wrap; the tangential cell slices keep endpoint N.
    il, ih = il % shape[0], ih % shape[0]
    jl, jh = jl % shape[1], jh % shape[1]
    kl, kh = kl % shape[2], kh % shape[2]
    # +x/-x faces: S_x = Ey·Hz - Ez·Hy
    p = jnp.sum(ey[ih, jj, kk] * hz[ih, jj, kk] - ez[ih, jj, kk] * hy[ih, jj, kk])
    p -= jnp.sum(ey[il, jj, kk] * hz[il, jj, kk] - ez[il, jj, kk] * hy[il, jj, kk])
    # +y/-y faces: S_y = Ez·Hx - Ex·Hz
    p += jnp.sum(ez[ii, jh, kk] * hx[ii, jh, kk] - ex[ii, jh, kk] * hz[ii, jh, kk])
    p -= jnp.sum(ez[ii, jl, kk] * hx[ii, jl, kk] - ex[ii, jl, kk] * hz[ii, jl, kk])
    # +z/-z faces: S_z = Ex·Hy - Ey·Hx
    p += jnp.sum(ex[ii, jj, kh] * hy[ii, jj, kh] - ey[ii, jj, kh] * hx[ii, jj, kh])
    p -= jnp.sum(ex[ii, jj, kl] * hy[ii, jj, kl] - ey[ii, jj, kl] * hx[ii, jj, kl])
    return float(p)


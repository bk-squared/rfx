"""Physical-field reconstruction for oblique Bloch measurements."""


def reconstruct_oblique_physical(sim_result, tfsf_cfg, grid, probes):
    """Issue #404: convert the complex Bloch envelope P back to the physical real field
    ``Re(P·exp(-j k_t·x_t))`` for the oblique-periodic path.

    Reconstructs the final ``state`` (per-cell transverse phase ramp along the
    periodic axis) and point-probe ``time_series`` (fixed phase at each probe's
    transverse cell). Other outputs are fenced upstream. The sign convention is
    ``cfg.k_transverse`` — the same one the 2D-aux injection and the hand-rolled
    acceptance tests use (``exp(-1j·k_transverse·y)``).
    """
    import jax.numpy as _jnp
    k_t = float(tfsf_cfg.k_transverse)
    taxis = 1 if tfsf_cfg.transverse_axis == "y" else 2
    d_t = float(grid.cells("y")[0]) if taxis == 1 \
        else float(grid.cells("z")[0])
    n_t = grid.shape[taxis]
    ramp = _jnp.exp(-1j * k_t * _jnp.arange(n_t) * d_t)

    new_state = sim_result.state
    if new_state is not None and _jnp.iscomplexobj(new_state.ex):
        shp = [1, 1, 1]
        shp[taxis] = n_t
        ramp3 = ramp.reshape(shp)
        new_state = new_state._replace(**{
            f: _jnp.real(getattr(new_state, f) * ramp3)
            for f in ("ex", "ey", "ez", "hx", "hy", "hz")
        })

    new_ts = sim_result.time_series
    if new_ts is not None and _jnp.iscomplexobj(new_ts):
        if probes:  # time_series is (n_steps, n_probes) in probe order
            idxs = _jnp.array([(p.j if taxis == 1 else p.k) for p in probes])
            ph = _jnp.exp(-1j * k_t * idxs * d_t)
            new_ts = _jnp.real(new_ts * ph[None, :])
        else:
            new_ts = _jnp.real(new_ts)

    return sim_result._replace(state=new_state, time_series=new_ts)

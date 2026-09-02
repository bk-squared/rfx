"""Passivity self-check for forward()/run() lumped/wire S11 (item-3 follow-up).

The eager single-cell lumped/wire S-parameter extractor can return non-physical
|S11| > 1 where the incident wave is weak (spectral band edges), because the
curl-of-H port current is ill-conditioned there (float32). Measured up to ~1.98
on a high-eps closed cavity. compute_*_s_matrix already runs a passivity
self-check (_warn_if_nonpassive_smatrix); forward() and run(compute_s_params=True)
previously surfaced none, so a researcher (or an optimizer's eager setup call)
could silently trust/chase those bins.

These tests pin that forward()/run() now warn on a gross passivity violation,
stay silent on a passive sweep and on mild float noise (tol=0.10, the repo
standard), and — critically — that the check is tracer-safe (skipped under
jax.grad, so it never perturbs an AD/optimization run).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse

_FREQS = np.linspace(1e9, 12e9, 11)


def _cavity(eps_r):
    """PEC cavity; a high-eps dielectric block drives the band-edge |S11|>1."""
    sim = Simulation(freq_max=12e9, domain=(0.02, 0.02, 0.02), dx=1.0e-3, boundary="pec")
    if eps_r is not None:
        sim.add_material("d", eps_r=eps_r)
        sim.add(Box((0.004, 0.004, 0.004), (0.016, 0.016, 0.016)), material="d")
    sim.add_port(
        position=(0.01, 0.01, 0.01), component="ez", impedance=50.0,
        waveform=GaussianPulse(f0=4e9, bandwidth=0.7),
    )
    return sim


def _warned_nonpassive(recorded):
    return any("non-passive" in str(w.message) for w in recorded)


def test_forward_lumped_s11_passivity_warns_on_gross_violation():
    """forward() must warn when the eager extractor returns |S11| >> 1."""
    sim = _cavity(eps_r=10.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fr = sim.forward(port_s11_freqs=_FREQS)
    assert float(np.abs(np.asarray(fr.s_params).reshape(-1)).max()) > 1.10
    assert _warned_nonpassive(rec), "forward() should warn on non-passive |S11|>1"


def test_forward_passive_does_not_warn():
    """A passive (vacuum) sweep must not trigger the passivity warning."""
    sim = _cavity(eps_r=None)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fr = sim.forward(port_s11_freqs=_FREQS)
    assert float(np.abs(np.asarray(fr.s_params).reshape(-1)).max()) <= 1.10
    assert not _warned_nonpassive(rec), "passive sweep must not warn"


def test_passivity_helper_fires_on_gross_silent_otherwise():
    """Unit-test the shared helper directly (geometry-independent).

    This is the robust proof that the guard FIRES on a gross violation — used
    on both the forward() and run() paths. It is geometry-independent, so it
    pins the helper's firing/silence regardless of which extractor path feeds
    it (since item-5 Stage 3, run()-lumped shares forward()'s scan path and the
    band-edge firing is now witnessable on geometry too — see
    ``test_run_passivity_warns_on_band_edge_after_consolidation``).
    """
    from rfx.probes.probes import warn_if_nonpassive_lumped_s11
    f = np.array([1e9, 5e9, 9e9])
    for label, s, expect in [
        ("gross 1.98", np.array([0.5, 1.98, 0.3]), True),
        ("mild 1.01 (within tol)", np.array([0.5, 0.9, 1.01]), False),
        ("passive", np.array([0.5, 0.9, 0.3]), False),
    ]:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            warn_if_nonpassive_lumped_s11(s, f, extractor="unit")
        assert _warned_nonpassive(rec) is expect, f"helper {label}: wrong warn state"


def test_run_passivity_warns_on_band_edge_after_consolidation():
    """run(compute_s_params=True) warns on the eps-cavity band-edge.

    As of item-5 Stage 3, run()-lumped routes its S-parameter extraction
    through the production-scan driver (the SAME graph as forward()), not the
    old eager extract_s_matrix loop. It therefore shares forward()'s band-edge
    behavior: the lossless eps cavity produces a non-physical |S11| > 1 at a
    spectral band edge, and the #210 passivity guard correctly surfaces it.

    This supersedes the pre-consolidation premise (run() was the better-
    conditioned path and stayed passive on this cavity). The forward() analog
    is ``test_forward_lumped_s11_passivity_warns_on_gross_violation`` above;
    run() now mirrors it because they are one path.
    """
    sim = _cavity(eps_r=10.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sim.run(n_steps=1500, compute_s_params=True, s_param_freqs=_FREQS)
    assert float(np.abs(np.asarray(res.s_params).reshape(-1)).max()) > 1.10
    assert _warned_nonpassive(rec), "run()-lumped should warn on the band-edge |S11|>1"


def test_passivity_check_is_tracer_safe_under_grad():
    """Under jax.grad the check must be skipped (no warn, finite grad, no crash)."""
    sim = _cavity(eps_r=10.0)
    g = sim._build_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)
    cell = (g.nx // 2, g.ny // 2, g.nz // 2)

    def objective(alpha):
        eps = eps_base.at[cell].set(alpha)
        fr = sim.forward(eps_override=eps, port_s11_freqs=_FREQS[:4])
        return jnp.sum(jnp.abs(fr.s_params) ** 2)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        grad = float(jax.grad(objective)(jnp.float32(1.0)))
    assert np.isfinite(grad), "AD grad through forward S11 must be finite"
    assert not _warned_nonpassive(rec), "passivity check must be skipped under jax.grad"

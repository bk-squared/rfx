"""Contract: run(compute_s_params=True) vs forward(port_s11_freqs=) S11 agreement.

Background (ARCH-6 review, 2026-06-21, root-caused + adversarially verified):
A user often *validates* a port design with run(compute_s_params=True) and then
*optimizes* it with forward(port_s11_freqs=...). These are two S11 estimators
for the same single-cell lumped/wire port (run() via the uniform runner's
inline decomposition; forward() via extract_lumped_s11 on a separate scan), and
they can silently disagree:

  - The two go through different code paths that compile to different XLA
    reduction graphs. The single-cell port current is a curl-of-H finite
    difference (a small difference of similar numbers) accumulated in a
    hardcoded-complex64 DFT (rfx/simulation.py:702-704). At spectral band edges
    (weak incident wave) this ratio is ill-conditioned, so the two graphs round
    in a different float32 order and |S11| diverges. This is the
    project_port_e5_program E4-natural-ceiling for single-cell ports
    (magnitude-only, single-cell convention unreliable) — NOT a one-sided bug.
  - NOTE: the exact graph difference that triggers the rounding split was NOT
    isolated. forward()'s auto-added port probe (_execute.py:687-692) was tested
    as a candidate trigger and RULED OUT (removing it leaves the PEC gap
    unchanged). Per R2 the mechanism hunt was stopped rather than bisected
    further; these tests pin the observable contract, not the trigger.

Empirically (verified):
  - On an absorbing (CPML) boundary the V/I stay well-conditioned and the two
    paths are BYTE-IDENTICAL at every frequency -> tight agreement is required.
  - On a lossless PEC cavity the two agree to <=~2% in-band (near the source
    f0) but diverge into the weak-signal tail (up to ~0.2 at 2*f0), where
    forward() additionally becomes non-physical (|S11|>1) and n_steps-unstable.
    run() stays passive and n_steps-stable, so run() is the trustworthy path.

This is the project_port_e5_program E4-natural-ceiling for single-cell ports
(magnitude-only, single-cell convention unreliable); it is NOT a one-sided bug
to be "fixed" by forcing bit-equality (that would launder real float32 noise).

These tests pin the relationship that MUST hold (well-conditioned agreement +
run() passivity + in-band agreement) without blessing the band-edge noise.
"""

import numpy as np

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse

_F0 = 5e9
_FREQS = np.array([1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]) * 1e9
# In-band bins (well-conditioned: near the Gaussian f0=5GHz, bw=0.9 -> ~2.7-7.3GHz).
# 8.5/10 GHz are the weak-signal tail where single-cell float32 conditioning bites.
_INBAND = _FREQS <= 7.0e9


def _wire_sim(boundary):
    kw = {} if boundary == "pec" else {"cpml_layers": 6}
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
        boundary=boundary, **kw,
    )
    sim.add_port(
        position=(0.0093, 0.0093, 0.0093), component="ez", impedance=50.0,
        waveform=GaussianPulse(f0=_F0, bandwidth=0.9), extent=0.004,
    )
    return sim


def _run_s11(boundary):
    r = _wire_sim(boundary).run(n_steps=2000, compute_s_params=True, s_param_freqs=_FREQS)
    return np.abs(np.asarray(r.s_params)[0, 0, :])


def _forward_s11(boundary):
    fr = _wire_sim(boundary).forward(port_s11_freqs=_FREQS)
    return np.abs(np.asarray(fr.s_params).reshape(-1))


def test_run_forward_s11_agree_on_well_conditioned_cpml():
    """On an absorbing boundary the two estimators must agree tightly.

    This is the load-bearing cross-check: a regression that makes run() and
    forward() diverge where the V/I are well-conditioned (e.g. a new graph or
    accumulation change) fails here.
    """
    run_s11 = _run_s11("cpml")
    fwd_s11 = _forward_s11("cpml")
    assert run_s11.shape == fwd_s11.shape == _FREQS.shape
    np.testing.assert_allclose(run_s11, fwd_s11, atol=2e-3, rtol=0,
                               err_msg="run vs forward S11 diverged on a well-conditioned CPML port")


def test_run_s11_is_passive_both_boundaries():
    """run() is the trustworthy estimator: |S11| <= 1 for a passive 1-port."""
    for boundary in ("cpml", "pec"):
        s11 = _run_s11(boundary)
        assert np.all(np.isfinite(s11)), f"run() S11 not finite ({boundary})"
        assert s11.max() <= 1.0 + 1e-3, f"run() S11 not passive ({boundary}): max={s11.max():.4f}"


def test_run_forward_s11_agree_in_band_pec():
    """In-band (well-conditioned) the two estimators agree within the E4 envelope.

    Band-edge bins (>7 GHz here) are intentionally excluded: single-cell float32
    conditioning makes them diverge and forward() non-passive there — a known
    E4-ceiling limit (project_port_e5_program), documented not asserted-equal.
    """
    run_s11 = _run_s11("pec")[_INBAND]
    fwd_s11 = _forward_s11("pec")[_INBAND]
    np.testing.assert_allclose(run_s11, fwd_s11, atol=0.05, rtol=0,
                               err_msg="run vs forward S11 diverged in-band on PEC (beyond the E4 envelope)")

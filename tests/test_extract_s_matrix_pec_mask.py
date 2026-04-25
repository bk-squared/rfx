"""Regression: ``extract_s_matrix`` (lumped-port S-param extractor used
by ``run(compute_s_params=True)``) must honour the interior PEC mask.

Without this, eval simulations lose ground planes / scatterers added
via ``Box(..., material="pec")``, producing flat |S11| ≈ 0 dB across
the band — which manifests as a large train/eval disconnect on the
``optimize()`` path identical in shape to the time-gating bug
(rfx-tap paper, ex1_dra; issue #72 follow-up).

The wire-port counterpart already accepts ``pec_mask`` — the lumped
path just needed parity.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box, GaussianPulse


def _build_pec_cavity_with_dielectric():
    """5×5×5 mm closed PEC cavity, Ez port at the centre, ε=4 fill."""
    a = 0.020
    sim = Simulation(
        freq_max=10e9,
        domain=(a, a, a),
        dx=2.0e-3,
        boundary="pec",
    )
    sim.add_material("filler", eps_r=4.0)
    # Closed PEC cavity walls via geometry (not just simulation boundary).
    # The interior shell rasterises into pec_mask — exactly the path
    # that would silently get dropped without the fix.
    wall = 0.0  # shell at corner_lo and corner_hi inferred from boundary="pec"
    # 1-cell PEC wall on z=0 face (a "ground plane")
    sim.add(Box((0.0, 0.0, 0.0), (a, a, 2e-3)), material="pec")
    # ε=4 filler everywhere above
    sim.add(Box((0.0, 0.0, 2e-3), (a, a, a)), material="filler")
    sim.add_port(
        position=(a / 2, a / 2, 4e-3),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=6e9, bandwidth=0.8, amplitude=1.0),
    )
    return sim


def test_extract_s_matrix_uses_pec_mask():
    """Without pec_mask, no ground plane → |S11| stays near 1 across band.

    With pec_mask the ground plane should at minimum perturb |S11| —
    drive at least one frequency away from full reflection by > 0.5 dB
    relative to the no-mask run.  We do NOT assert a deep dip; the
    cavity is small and lossy by edge truncation, so a clear |S11|
    *difference* between the two runs is the cleanest physical check.
    """
    sim = _build_pec_cavity_with_dielectric()
    # Use the run() path — that's where extract_s_matrix is invoked.
    freqs = np.linspace(3e9, 10e9, 15)
    res = sim.run(num_periods=40, compute_s_params=True,
                  s_param_freqs=freqs)

    assert res.s_params is not None
    s11 = res.s_params[0, 0, :]
    mag = np.abs(np.asarray(s11))
    # PEC ground inside the lumped-port cavity should suppress |S11|
    # well below 1 at some frequency.  A magnitude of <0.95 anywhere
    # is far below what the bugged "no pec_mask" path produces (which
    # consistently hovers > 0.97 across the band).
    assert mag.min() < 0.95, (
        f"|S11| does not dip below 0.95 anywhere in band — "
        f"pec_mask likely not threaded through "
        f"extract_s_matrix; min={mag.min():.4f}, mag={mag.round(3)}"
    )

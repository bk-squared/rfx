"""Regression tests for the probe-aware normalized two-port contract.

These tests lock the core physical properties of the finalized v1
normalized waveguide S-matrix path:
- empty guide remains reciprocal with unity transmission
- PEC short reflects strongly without column-power blow-up
- reference-plane knobs do not affect the normalized result
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.geometry.csg import Box


FREQS = np.linspace(4.5e9, 8.0e9, 20)
F0 = float(FREQS.mean())
BW = max(0.2, min(0.8, (FREQS[-1] - FREQS[0]) / max(F0, 1.0)))


def _build_twoport_sim(*, kind: str, left_ref=None, right_ref=None):
    sim = Simulation(
        freq_max=max(float(FREQS[-1]), F0),
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
    )
    if kind == "pec_short":
        sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
        sim.add(Box((0.05, 0.0, 0.0), (0.055, 0.04, 0.02)), material="pec_like")
    elif kind == "empty":
        pass
    else:
        raise ValueError(kind)

    sim.add_waveguide_port(
        0.01,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(FREQS),
        f0=F0,
        bandwidth=BW,
        ref_offset=3,
        probe_offset=15,
        name="left",
        reference_plane=left_ref,
    )
    sim.add_waveguide_port(
        0.09,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(FREQS),
        f0=F0,
        bandwidth=BW,
        ref_offset=3,
        probe_offset=15,
        name="right",
        reference_plane=right_ref,
    )
    return sim


def test_normalized_twoport_empty_preserves_core_contract():
    sim = _build_twoport_sim(kind="empty")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s = np.asarray(result.s_params)
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    recip = np.abs(np.abs(s[1, 0, :]) - np.abs(s[0, 1, :])) / np.maximum(
        np.maximum(np.abs(s[1, 0, :]), np.abs(s[0, 1, :])),
        1e-12,
    )

    assert float(np.mean(np.abs(s[0, 0, :]))) < 0.15
    assert 0.98 < float(np.mean(np.abs(s[1, 0, :]))) < 1.02
    assert float(np.mean(column_power)) < 1.05
    assert float(np.mean(recip)) < 1e-3


def test_normalized_twoport_pec_short_is_strongly_reflective():
    sim = _build_twoport_sim(kind="pec_short")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s = np.asarray(result.s_params)
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    recip = np.abs(np.abs(s[1, 0, :]) - np.abs(s[0, 1, :])) / np.maximum(
        np.maximum(np.abs(s[1, 0, :]), np.abs(s[0, 1, :])),
        1e-12,
    )

    assert 0.85 < float(np.mean(np.abs(s[0, 0, :]))) < 1.05
    assert float(np.mean(np.abs(s[1, 0, :]))) < 0.10
    assert float(np.mean(column_power)) < 1.10
    assert float(np.mean(recip)) < 1e-3


def test_normalized_twoport_reference_plane_invariant_on_empty():
    base = _build_twoport_sim(kind="empty").compute_waveguide_s_matrix(
        num_periods=40, normalize=True
    )
    shifted = _build_twoport_sim(
        kind="empty", left_ref=0.02, right_ref=0.08
    ).compute_waveguide_s_matrix(num_periods=40, normalize=True)

    s_base = np.asarray(base.s_params)
    s_shift = np.asarray(shifted.s_params)
    assert np.allclose(s_base, s_shift, rtol=1e-5, atol=1e-7)


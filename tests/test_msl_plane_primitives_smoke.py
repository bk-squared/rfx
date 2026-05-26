"""Smoke coverage for the MSL plane-probe primitives.

WI-3 (2026-05-24) deleted `test_msl_plane_extractor_jax.py` along with the
deprecated plane *extractor* (`extract_msl_s_params_jax_plane`). But the
plane-probe PRIMITIVES it transitively exercised —
`register_msl_plane_probes`, `_v_from_plane`, `_i_from_plane` — survive and
are still consumed by `examples/inverse_design/msl_stub_notch_tuning.py`.
This file restores minimal regression coverage for those primitives so a
silent break surfaces here rather than only when the example is run.

Scope: register the 4 plane DFT probes on a real cv06b-class MSL thru-line,
run a short forward, and assert the line-integrated V and closed-loop I
phasors come back finite, correctly-shaped, and non-zero. NOT an accuracy
gate (that lived in the deleted file vs the imperative reference) — purely a
"the primitives still wire up and produce signal" smoke.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.msl_wave_decomp import (
    register_msl_plane_probes,
    _v_from_plane,
    _i_from_plane,
)

EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 127e-6
L_LINE = 30e-3
PORT_MARGIN = 1.6e-3
F_MAX = 9e9


def _build_thru_line():
    """Two-port cv06b-class MSL thru-line (recovered from the deleted test)."""
    LX = L_LINE + 2 * PORT_MARGIN
    L_STUB_MAX = 14e-3
    LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX) + L_STUB_MAX + 2 * (2 * H_SUB + 8 * DX)
    LZ = H_SUB + 1.0e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2
    sim.add(
        Box((0, y_trace - W_TRACE / 2, H_SUB),
            (LX, y_trace + W_TRACE / 2, H_SUB + DX)),
        material="pec",
    )
    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB, direction="-x", impedance=50.0)
    return sim


@pytest.mark.slow
def test_plane_primitives_register_and_produce_signal():
    sim = _build_thru_line()
    freqs = jnp.linspace(F_MAX / 10, F_MAX, 4, dtype=jnp.float32)
    # Drive port 0; register plane probes on port 0 (driven) and port 1.
    p0 = register_msl_plane_probes(sim, port_index=0, freqs=freqs, name_prefix="d")
    p1 = register_msl_plane_probes(sim, port_index=1, freqs=freqs, name_prefix="p")
    fr = sim.forward(num_periods=15.0, skip_preflight=True)

    # V from the 3 Ez planes + closed-loop I from the Hy plane on the driven port.
    v = jnp.stack([_v_from_plane(fr, p0.ez1_name, p0),
                   _v_from_plane(fr, p0.ez2_name, p0),
                   _v_from_plane(fr, p0.ez3_name, p0)], axis=-1)
    i1 = _i_from_plane(fr, p0.hy_name, p0)
    v_p1 = _v_from_plane(fr, p1.ez1_name, p1)

    v_np, i_np, vp1_np = np.asarray(v), np.asarray(i1), np.asarray(v_p1)
    # Shapes: V is (n_freqs, 3); I and the passive-port V are (n_freqs,).
    assert v_np.shape == (4, 3), v_np.shape
    assert i_np.shape == (4,), i_np.shape
    # Finite + non-zero (the primitives wired up and recorded signal).
    for name, arr in (("V_driven", v_np), ("I_driven", i_np), ("V_passive", vp1_np)):
        assert np.all(np.isfinite(arr)), f"{name} not finite: {arr}"
        assert float(np.max(np.abs(arr))) > 0.0, f"{name} is all-zero: {arr}"

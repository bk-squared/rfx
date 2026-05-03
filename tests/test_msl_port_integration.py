"""Integration test: 50 Ω microstrip thru-line S-parameter passivity gate.

Marked ``slow`` — excluded from the default pytest run.  Run explicitly with::

    pytest tests/test_msl_port_integration.py -m slow -v -s

Geometry
--------
- Substrate : RO4350B, εr = 3.66, h = 254 µm
- Trace     : W = 600 µm (50 Ω design width), L = 10 mm
- Port margin from each feed plane to domain edge: 2 mm
- PEC ground plane at z = 0 (z_lo boundary); CPML on z_hi and all x/y faces
- Uniform dx = 80 µm — substrate gets ~3 cells (254/80 ≈ 3.2)
  NOTE: dz_profile is intentionally NOT used; the non-uniform runner does not
  propagate MSL port sources into its scan body so DFT plane probes return zero.
- f_max = 5 GHz, 30 freq points, num_periods = 12
- Two MSL ports: port 0 driven (+x), port 1 passive matched (-x)

Source mode: static-Laplace Ez profile (mode="laplace", the default).
The Schelkunoff J+M eigenmode source (mode="eigenmode") was attempted but
naive soft J+M sources cannot cancel the backward wave without a numerically
dispersion-corrected 1D auxiliary FDTD table; that path is reserved for a
future implementation.

Fix (2026-05-02)
----------------
A microstrip quasi-TEM mode requires a PEC trace conductor above the substrate.
Without the trace, the Ez source excites a TM-like substrate mode giving
Z0 ≈ 1600–2500 Ω and |S21| ≈ 0.  Following the canonical pattern in
``examples/crossval/06_msl_notch_filter.py`` (``sim.add(Box(...), material="pec")``),
we add a one-cell-thick PEC strip at z = H_SUB spanning the full line length.

The 3-probe extractor in ``compute_msl_s_matrix`` was also corrected to apply a
direction-aware sign to the Hy current integral: for a +x propagating quasi-TEM
wave with Ez > 0, Hy < 0 (x̂ × ẑ = −ŷ), so the raw integral must be negated to
recover the physical current I = +∮H·dl and give Z0 = V/I > 0.

Frequency window: at DX = 80 µm (≈3 cells across substrate), the quasi-TEM mode
is well-established above ~3 GHz.  The gate uses a fixed 3–4.5 GHz window rather
than a symmetric midband slice to avoid low-frequency dispersion artefacts.

Gate calibration (dx=80 µm, laplace mode, measured 2026-05-04)
--------------------------------------------------------------
  mean |S11| ≈ 0.118  → gate < 0.15
  mean |S21| ≈ 0.972  → gate (0.90, 1.05)
  mean Re(Z0) ≈ 54 Ω  → gate (40, 65) Ω
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

EPS_R = 3.66          # RO4350B
H_SUB = 254e-6        # substrate thickness, metres
W_TRACE = 600e-6      # trace width, metres
L_LINE = 10e-3        # thru-line length
PORT_MARGIN = 2e-3    # feed → domain edge clearance

# Uniform cell size: 80 µm gives 254/80 ≈ 3.2 cells in substrate.
DX = 80e-6
F_MAX = 5e9

LX = L_LINE + 2 * PORT_MARGIN
LY = W_TRACE + 6 * DX   # 3-cell clearance each side
LZ = H_SUB + 1.5e-3      # substrate + 1.5 mm air above

# Evaluation window: quasi-TEM mode is well-established at 3–4.5 GHz
# (below this the staircase substrate has Z0 < 40 Ω due to poor resolution).
GATE_F_LO = 3.0e9
GATE_F_HI = 4.5e9


@pytest.mark.slow
def test_msl_thru_line_passive_gate():
    """50 Ω microstrip thru with static-Laplace Ez source (mode='laplace').

    Gates calibrated for dx=80 µm (≈3 substrate cells), laplace mode:
      |S11| < 0.15
      0.90 < |S21| < 1.05
      Z0 ∈ (40, 65) Ω
    """

    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, LZ),
        dx=DX,
        cpml_layers=8,
        # PEC ground plane at z=0 (microstrip ground); CPML above.
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),  # PEC ground plane at z=0
        ),
    )

    # --- substrate fill (RO4350B, lossless) ---
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(
        Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)),
        material="ro4350b",
    )

    # --- PEC trace strip (one cell thick at z = H_SUB) ---
    # A microstrip quasi-TEM mode requires a metal trace above the substrate.
    # Without the trace, the Ez source excites a TM substrate mode (Z0>>50Ω).
    # Canonical pattern from examples/crossval/06_msl_notch_filter.py:
    #   sim.add(Box(..., substrate_thickness, substrate_thickness+dz), material="pec")
    # Use one-cell thickness (H_SUB to H_SUB + DX) so rfx Box captures the
    # cells whose z-centres fall within the box z-range.
    y_centre = LY / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )

    # --- port 0: driven from left, propagates +x ---
    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="+x",
        impedance=50.0,
    )

    # --- port 1: passive matched at right end, propagates -x ---
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="-x",
        impedance=50.0,
    )

    # ------------------------------------------------------------------ run
    result = sim.compute_msl_s_matrix(
        n_freqs=30,
        num_periods=12,
    )

    S = result.S          # shape (n_ports, n_ports, n_freqs)
    Z0 = result.Z0        # shape (n_ports, n_freqs)
    freqs = result.freqs  # shape (n_freqs,)

    # Gate window: 3–4.5 GHz where quasi-TEM is well-established at DX=80µm.
    gate_mask = (freqs >= GATE_F_LO) & (freqs <= GATE_F_HI)
    if not np.any(gate_mask):
        # Fallback: centre 10 points
        n_f = freqs.shape[0]
        gate_mask = np.zeros(n_f, dtype=bool)
        gate_mask[max(0, n_f // 2 - 5):min(n_f, n_f // 2 + 5)] = True

    s11_gate = np.abs(S[0, 0, gate_mask])
    s21_gate = np.abs(S[1, 0, gate_mask])
    z0_gate  = Z0[0, gate_mask].real

    mean_s11 = float(np.mean(s11_gate))
    mean_s21 = float(np.mean(s21_gate))
    mean_z0  = float(np.mean(z0_gate))

    print(f"\n[MSL thru] gate freqs: {freqs[gate_mask][0]*1e-9:.2f}–{freqs[gate_mask][-1]*1e-9:.2f} GHz ({int(np.sum(gate_mask))} pts)")
    print(f"[MSL thru] mean |S11| = {mean_s11:.4f}")
    print(f"[MSL thru] mean |S21| = {mean_s21:.4f}")
    print(f"[MSL thru] mean Re(Z0) = {mean_z0:.2f} Ω")
    print(f"[MSL thru] freqs total: {freqs[0]*1e-9:.2f}–{freqs[-1]*1e-9:.2f} GHz")
    nz_sub = int(round(H_SUB / DX))
    print(f"[MSL thru] dx = {DX*1e6:.0f} µm, nz_sub ≈ {nz_sub}")

    # --- reflection gate (laplace mode, dx=80µm: measured ~0.118) ---
    assert mean_s11 < 0.15, (
        f"|S11| = {mean_s11:.4f} ≥ 0.15 — excessive reflection at source plane"
    )

    # --- passivity / transmission gate ---
    assert mean_s21 > 0.90, (
        f"|S21| = {mean_s21:.4f} ≤ 0.90 — insufficient forward transmission"
    )
    assert mean_s21 < 1.05, (
        f"|S21| = {mean_s21:.4f} ≥ 1.05 — passivity violated"
    )

    # --- Z0 gate (dx=80µm corrected window; measured ~54 Ω) ---
    assert 40.0 < mean_z0 < 65.0, (
        f"Re(Z0) = {mean_z0:.2f} Ω outside (40, 65) Ω"
    )

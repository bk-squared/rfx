"""Issue #80 / #118 — patch RESONANCE gate via Harminv ring-down (port-independent).

Companion to ``test_issue80_patch_s11_regression.py``. That gate validates the |S11|
PASSIVITY and the edge-fed match-point physics; THIS gate validates the patch RESONANCE
frequency itself, using a method that does not depend on the MSL port extractor.

WHY A SEPARATE RESONANCE GATE
-----------------------------
For a 50-ohm line butt-coupled directly to the patch radiating edge, the |S11| magnitude
DIP is the off-resonance impedance MATCH point (Re(Zin) -> ~50 ohm), which sits ABOVE the
TM010 resonance — NOT the resonance itself (the patch is badly matched at resonance, where
the edge resistance is hundreds of ohms). So the |S11| dip frequency is the wrong observable
for the resonance (issue #118; witnessed in
``scripts/diagnostics/issue118_match_vs_resonance_witness.py`` — Im(Zin)=0 ~9.5-9.6 GHz vs
the |S11| dip ~11 GHz at coarse mesh).

The radiating resonance is measured here by the cv05-validated Harminv ring-down on a cavity
Ez probe — port-calibration-independent. Three independent methods agree the patch TM010 is
~9.2-9.3 GHz: rfx Harminv 9.32 (Q44), OpenEMS S11 9.20, analytic Balanis 9.19/9.21. This gate
pins that the dominant ring-down mode is the patch TM010 near 9.2, and that it is NOT the
~11.9 GHz feed-line λ/2 mode (the historical wrong-mode reading).

Marked ``slow`` (full patch FDTD ring-down); CPU-feasible at 4 substrate cells (~7 min),
excluded from the fast suite — gives the resonance physics CPU coverage that the gpu+slow
S11 gate does not.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.harminv import harminv
from rfx.sources import GaussianPulse

# issue-#80 reproduction geometry (matches the MSL-port gate)
EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
PORT_MARGIN = 5.0e-3
Z_GND = 4e-3
FEED_LEN = 8.0e-3
DOM_X, DOM_Y, DOM_Z = 29.747e-3, 18.130e-3, 12.787e-3
N_SUB_CELLS = 4
DX = H_SUB / N_SUB_CELLS

TARGET_GHZ = 9.21        # analytic Balanis radiation resonance (== OpenEMS 9.20)
TOL_GHZ = 0.8            # coarse-mesh slack (rfx Harminv lands ~9.3; ~9% band)
SPURIOUS_GHZ = 11.9      # the feed-line λ/2 mode — must NOT be picked as the dominant mode


def _build_patch() -> "Simulation":
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
                     dx=DX, cpml_layers=8, boundary="cpml")
    z_gnd_hi = Z_GND + DX
    z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + DX
    x_patch0 = PORT_MARGIN + FEED_LEN
    y_c = DOM_Y / 2.0
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, Z_GND), (DOM_X, DOM_Y, z_gnd_hi)), material="pec")            # ground
    sim.add(Box((0, 0, z_sub_lo), (DOM_X, DOM_Y, z_sub_hi)), material="ro4003c")     # substrate
    sim.add(Box((0, y_c - W_MSL / 2, z_tr_lo),
                (x_patch0, y_c + W_MSL / 2, z_tr_hi)), material="pec")               # feed
    sim.add(Box((x_patch0, y_c - W / 2, z_tr_lo),
                (x_patch0 + L, y_c + W / 2, z_tr_hi)), material="pec")               # patch
    sim.add_msl_port(
        position=(PORT_MARGIN, y_c, z_sub_lo),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )
    # cavity Ez probe (off-centre) for the Harminv ringdown readout
    sim.add_probe(position=(x_patch0 + 0.7 * L, y_c - 0.2 * W, 0.5 * (z_sub_lo + z_sub_hi)),
                  component="ez")
    return sim, x_patch0, y_c


@pytest.mark.slow
def test_issue80_patch_resonance_harminv():
    """The patch TM010 (Harminv ringdown) is near 9.2 GHz, NOT the spurious ~11.9."""
    sim, _x_patch0, _y_c = _build_patch()
    sim.preflight()
    grid = sim._build_grid()
    n_steps = int(grid.num_timesteps(num_periods=80.0))
    res = sim.run(n_steps=n_steps)

    ts = np.asarray(res.time_series).ravel()
    dt = float(res.dt)
    sig = ts[int(len(ts) * 0.3):]                  # ringdown region (skip drive)
    modes = [m for m in harminv(sig, dt, 6e9, 14e9)
             if m.Q > 2 and abs(m.amplitude) > 1e-9]
    assert modes, "Harminv found no ring-down modes — sim too short or unstable"

    # DOMINANT mode by amplitude (NOT nearest-analytic — avoid selection bias)
    dom = max(modes, key=lambda m: abs(m.amplitude))
    f_dom = dom.freq / 1e9
    spectrum = sorted((m.freq / 1e9, m.Q, float(abs(m.amplitude))) for m in modes)
    print(f"\n[ISSUE80-HARMINV] dominant = {f_dom:.3f} GHz (Q={dom.Q:.1f}); "
          f"spectrum={[f'{f:.2f}/Q{q:.0f}' for f, q, _ in spectrum]}")

    # (1) the dominant ring-down mode IS the patch TM010 near 9.2 (= OpenEMS/analytic)
    assert abs(f_dom - TARGET_GHZ) <= TOL_GHZ, (
        f"dominant patch mode {f_dom:.3f} GHz not within {TARGET_GHZ}±{TOL_GHZ} "
        f"(OpenEMS 9.20, analytic 9.19). Spectrum: {spectrum}"
    )
    # (2) it must NOT be the spurious ~11.9 feed-line λ/2 mode
    assert abs(f_dom - SPURIOUS_GHZ) > 1.0, (
        f"dominant mode {f_dom:.3f} regressed toward the ~{SPURIOUS_GHZ} GHz feed-line "
        "λ/2 mode — the wrong-mode reading, NOT the patch radiation resonance."
    )

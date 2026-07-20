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
pins that the patch TM010 near 9.2 dominates the ring-down, and that it is NOT the ~11.9 GHz
feed-line λ/2 mode (the historical wrong-mode reading).

RING-DOWN SETTLING WITNESS (repo mandatory rule; issue #402)
------------------------------------------------------------
No claims-bearing Harminv number is quotable from a truncated ring-down. The run now goes
through ``sim.run(num_periods=...)`` (n_steps=None) so the framework's #332 truncation
advisory fires, and this test ADDITIONALLY asserts the end-of-run cavity-probe envelope is
below the -40 dB truncation-suspect bar (measured -65.9 dB at num_periods=120, N_SUB=4 — the
radiating patch mode Q~44 drains far past the bar). Previously the gate passed an explicit
``n_steps`` which sets ``fixed_num_periods=False`` and silently SUPPRESSED that advisory — the
right-guard-mis-gated pattern (issue #402 audit lane A).

MODE IDENTIFICATION — WHY NOT THE e4 FAR-FIELD RULE (issue #402 finding)
-----------------------------------------------------------------------
The canonical patch gate ``test_patch_canonical_farfield_e4.py`` identifies the radiating
mode by the far-field rule (broadside beam + radiated-power spectral peak) and bans amplitude
rank / nearest-textbook. That rule is NOT transplantable to THIS fixture, and the reason is
architectural, not incidental:

  * The feed is EDGE-FED: the 50-ohm microstrip enters from the domain boundary at the MSL
    port and runs x = 0 .. 13 mm before reaching the patch. Real power crosses that boundary.
  * The ground plane is FULL-DOMAIN: the ground PEC spans the whole x-y extent and butts into
    the CPML (an effectively infinite ground).

A closed 6-face Huygens NTFF box cannot enclose these radiators — any side face crosses the
feed PEC and the infinite ground PEC, so ``sim.preflight()`` raises "NTFF box must enclose all
radiators with no PEC crossing any face", and a box placed anyway integrates reactive
near-field: on this geometry the measured pattern shows unphysical DOWNWARD radiation (beam
peaks at θ = 140-180°, i.e. below an infinite ground plane) and the radiated-power peak lands
on a non-broadside bin. A far-field verdict from that box would be a corrupted-observable pass
(R5). A doctrine-compliant far-field mode-ID would require a separate FINITE-ground,
probe-fed companion geometry (as in e4) — out of scope for this reproduction gate.

So this gate identifies the patch mode WITHOUT far-field, and WITHOUT single-probe global
amplitude rank, using two geometry-independent physical facts:
  (1) FREQUENCY ORDER: the patch TM010 half-wave resonance is the LOWEST radiating resonance
      of the structure; the feed-line λ/2 mode is HIGHER (~11.9 GHz). The mode is therefore
      selected from the physical patch band [8.0, 10.5] GHz, not from the global amplitude
      argmax (which can land on a feed/spurious mode — the shortcut e4 bans).
  (2) PATCH-DOMINATED RING-DOWN: the patch band must ring LOUDER than the ≥11 GHz feed band —
      the positive anti-spurious witness that the ring-down is the patch resonance, not the
      feed-line λ/2 mode. Amplitude is used only to compare bands / disambiguate near-
      degenerate patch modes, never to pick the physical mode across the whole spectrum.
The ±0.8 GHz acceptance is the recorded external/analytic anchor (OpenEMS 9.20, Balanis 9.21),
honestly labelled as an anchor, not a far-field verdict. Observed full-res ring-down
(N_SUB=4, num_periods=120): 8.78/Q31, 9.32/Q44 (patch TM010), 11.90/Q18 (feed λ/2), 13.72/Q38
— the patch band (9.32, amp 0.019) dominates the feed band (11.90, amp 0.0035).

Marked ``slow`` (full patch FDTD ring-down); CPU-feasible at 4 substrate cells (~15 min at
num_periods=120), excluded from the fast suite — gives the resonance physics CPU coverage that
the gpu+slow S11 gate does not.
"""
from __future__ import annotations

import math

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
TOL_GHZ = 0.8            # coarse-mesh slack (rfx Harminv lands ~9.32; ~9% band)
SPURIOUS_GHZ = 11.9      # the feed-line λ/2 mode — must NOT dominate the ring-down
PATCH_BAND_GHZ = (8.0, 10.5)   # physical patch TM010 radiating band (frequency order)
FEED_BAND_LO_GHZ = 11.0        # feed-line λ/2 and higher-order / spurious band
NUM_PERIODS = 120.0            # settling: cavity probe reads -65.9 dB (bar -40)
SETTLING_BAR_DB = -40.0


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
    """The patch TM010 (Harminv ringdown) is near 9.2 GHz, NOT the spurious ~11.9, and the
    ring-down is settled below the -40 dB truncation bar before any frequency is trusted."""
    sim, _x_patch0, _y_c = _build_patch()

    # R: never ignore preflight — surface any warning before trusting any number.
    advisories = [str(a) for a in sim.preflight()]
    print(f"\n[ISSUE80-HARMINV] preflight advisories ({len(advisories)}) — quoted verbatim:")
    for a in advisories:
        print(f"  ! {a}")

    # num_periods (n_steps=None) so the framework #332 truncation advisory fires.
    res = sim.run(num_periods=NUM_PERIODS)

    ts = np.asarray(res.time_series).ravel()
    dt = float(res.dt)

    # --- ring-down settling witness (repo mandatory rule): no gated number from a
    #     truncated record. Assert the cavity-probe envelope is below the -40 dB bar. ---
    env = np.abs(ts)
    peak = float(np.max(env))
    tail = float(np.max(env[int(len(env) * 0.95):]))
    end_db = 20.0 * math.log10(max(tail, 1e-300) / max(peak, 1e-300))
    print(f"[ISSUE80-HARMINV] settling witness: end-of-run cavity envelope {end_db:.1f} dB "
          f"of peak (bar {SETTLING_BAR_DB} dB)")
    assert end_db < SETTLING_BAR_DB, (
        f"ring-down not settled: end-of-run envelope {end_db:.1f} dB does not clear the "
        f"{SETTLING_BAR_DB} dB truncation bar — raise NUM_PERIODS before trusting the "
        "Harminv resonance (issue #402; framework #332 advisory)"
    )

    sig = ts[int(len(ts) * 0.3):]                  # ringdown region (skip drive)
    modes = [m for m in harminv(sig, dt, 6e9, 14e9)
             if m.Q > 2 and abs(m.amplitude) > 1e-9]
    assert modes, "Harminv found no ring-down modes — sim too short or unstable"

    spectrum = sorted((m.freq / 1e9, m.Q, float(abs(m.amplitude))) for m in modes)
    print("[ISSUE80-HARMINV] ring-down spectrum: "
          f"{[f'{f:.2f}/Q{q:.0f}/a{a:.2g}' for f, q, a in spectrum]}")

    # --- Mode-ID by frequency order (NOT global amplitude rank, NOT far-field: see the
    #     module docstring for why the e4 far-field rule is architecturally unavailable
    #     on this edge-fed, full-domain-ground fixture). ---
    patch_modes = [m for m in modes if PATCH_BAND_GHZ[0] <= m.freq / 1e9 <= PATCH_BAND_GHZ[1]]
    feed_modes = [m for m in modes if m.freq / 1e9 >= FEED_BAND_LO_GHZ]
    assert patch_modes, (
        f"no ring-down mode in the patch radiating band {PATCH_BAND_GHZ} GHz — the patch "
        f"TM010 is missing from the ring-down. Spectrum: {spectrum}"
    )
    patch = max(patch_modes, key=lambda m: abs(m.amplitude))
    f_patch = patch.freq / 1e9
    patch_amp = float(abs(patch.amplitude))
    feed_amp = max((float(abs(m.amplitude)) for m in feed_modes), default=0.0)
    print(f"[ISSUE80-HARMINV] identified patch mode = {f_patch:.3f} GHz (Q={patch.Q:.1f}); "
          f"patch-band amp {patch_amp:.3g} vs feed-band(>={FEED_BAND_LO_GHZ:.0f}GHz) amp {feed_amp:.3g}")

    # (1) the patch-band mode IS the TM010 near 9.2 (= OpenEMS 9.20 / analytic 9.21 anchor)
    assert abs(f_patch - TARGET_GHZ) <= TOL_GHZ, (
        f"patch-band mode {f_patch:.3f} GHz not within {TARGET_GHZ}±{TOL_GHZ} "
        f"(OpenEMS 9.20, analytic 9.19). Spectrum: {spectrum}"
    )
    # (2) frequency-order + anti-spurious: the ring-down is PATCH-dominated, not the
    #     ~11.9 GHz feed-line λ/2 mode (the historical wrong-mode reading).
    assert patch_amp > feed_amp, (
        f"ring-down is dominated by a >={FEED_BAND_LO_GHZ:.0f} GHz feed/spurious mode "
        f"(amp {feed_amp:.3g} >= patch-band amp {patch_amp:.3g}) — regressed toward the "
        f"~{SPURIOUS_GHZ} GHz feed-line λ/2 mode, NOT the patch radiation resonance. "
        f"Spectrum: {spectrum}"
    )

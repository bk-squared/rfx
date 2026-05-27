"""Issue #80 Stage-0e: grounded shorted-stub, LUMPED/WIRE-port feed — architect's gate-1.

Fixes the fidelity gap in Stage-0/0b/0c/0d (those stubs had NO explicit ground
plane). Here: explicit PEC ground + substrate + trace + shorting via, fed by a
wire port (ground→trace, the realistic feed). Tests the architect's lumped-feed
Direction A on a faithful grounded microstrip BEFORE any production wiring.

Reference physics: a LOSSLESS shorted stub is a total reflector → true |S11|=1 at
all f, Zin = jZ0·tan(βL) purely reactive ⇒ Re(Zin)≥0 (on the boundary). A passive
extractor gives |S11|≤1 (≤1.05 with numerical/CPML loss) and must NOT worsen as
dx→0 (the discriminator that killed the MSL ∮H·dl loop, which went 1.39→2.56).

Gate: |S11|≤1.05 band-wide AND Re(Zin)≥0 AND non-worsening at dx=80→50.8µm.
Pass ⇒ lumped feed is a trustworthy passive REFERENCE to fix the MSL extractor
against (user goal: keep the MSL port; lumped is the reference/interim).
"""
from __future__ import annotations

import warnings

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

_EPS_R, _H_SUB, _W_TRACE = 3.66, 254e-6, 600e-6
_F_MAX, _PORT_MARGIN, _STUB_LEN = 12e9, 2e-3, 6e-3
_Z0 = 50.0


def _build_grounded_stub(dx: float) -> Simulation:
    z_gnd_hi = dx                       # ground plane: z in [0, dx]
    z_sub_lo, z_sub_hi = dx, dx + _H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + dx
    lx = _PORT_MARGIN + _STUB_LEN + _PORT_MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * dx)
    lz = z_tr_hi + 0.6e-3
    sim = Simulation(freq_max=_F_MAX, domain=(lx, ly, lz), dx=dx,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("ro4350b", eps_r=_EPS_R)
    y_c = ly / 2.0
    tl, th = y_c - _W_TRACE / 2, y_c + _W_TRACE / 2
    x_end = _PORT_MARGIN + _STUB_LEN
    sim.add(Box((0, 0, 0), (lx, ly, z_gnd_hi)), material="pec")             # GROUND
    sim.add(Box((0, 0, z_sub_lo), (lx, ly, z_sub_hi)), material="ro4350b")  # substrate
    sim.add(Box((0, tl, z_tr_lo), (x_end, th, z_tr_hi)), material="pec")    # trace
    sim.add(Box((x_end, tl, 0), (x_end + dx, th, z_tr_hi)), material="pec")  # shorting via
    # Wire-port feed: ground→trace at the feed plane (the realistic lumped feed)
    sim.add_port(
        position=(_PORT_MARGIN, y_c, z_sub_lo),
        component="ez", impedance=_Z0, extent=_H_SUB,
        waveform=GaussianPulse(f0=6e9, bandwidth=1.6),
    )
    return sim


def main() -> None:
    freqs = np.linspace(1.2e9, 12e9, 81)
    print("\n=== Stage-0e: grounded shorted-stub, wire-port feed (architect gate-1) ===")
    print(f"{'dx[µm]':>7} {'sub_cells':>9} {'max|S11|':>9} {'>1.05 bins':>10} "
          f"{'ReZin<0':>8} {'min|S11|':>9}")
    prev_max = None
    for dx in (80e-6, 50.8e-6):
        sim = _build_grounded_stub(dx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.preflight()
            n_steps = int(sim._build_grid().num_timesteps(num_periods=60.0))
            res = sim.run(compute_s_params=True, s_param_freqs=freqs,
                          s_param_n_steps=n_steps)
        S11 = np.asarray(res.s_params)[0, 0, :]
        Zin = _Z0 * (1 + S11) / (1 - S11 + 1e-30)
        sl = slice(len(freqs) // 8, -len(freqs) // 8)
        mx = float(np.max(np.abs(S11)[sl]))
        print(f"{dx*1e6:7.1f} {_H_SUB/dx:9.2f} {mx:9.3f} "
              f"{int(np.sum(np.abs(S11)[sl] > 1.05)):10d} "
              f"{int(np.sum(Zin.real[sl] < 0)):8d} {float(np.min(np.abs(S11)[sl])):9.3f}")
        prev_max = mx if prev_max is None else prev_max

    print("\n=== READING ===")
    print("PASS (lumped feed is passive REFERENCE) if: max|S11|≤~1.05 band-wide, "
          "0 Re(Zin)<0 bins, and NOT worsening dx=80→50.8 (vs MSL loop 1.39→2.56).")
    print("Then: use this passive reference to fix the MSL-port extractor (user goal),")
    print("comparing the MSL ∮H·dl S11 against this lumped S11 on identical geometry.")


if __name__ == "__main__":
    main()

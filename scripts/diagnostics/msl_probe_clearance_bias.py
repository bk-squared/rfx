"""Does probe-to-reflector clearance bias |S11|, or only Z0? (issue #726)

Two honesty messages disagree about the same condition: the Z0 guard says
"the V*I-split S11/S21 are unaffected", while preflight says standing-wave
content at the probes biases "Z0 extraction AND |S11|@notch — physical
|S11| -> 1 at a quarter-wave open stub may read as -5 to -10 dB instead of
0 dB". This measures which is true on the fixture the preflight text
describes, where the answer is known analytically.

Fixture: the committed cv06b quarter-wave open-stub notch filter. At the
notch the open stub transforms to a short across the line, so a passive
lossless structure must reflect essentially everything: |S11| -> 0 dB. The
ONLY variable between arms is where the port (and therefore its probe
comb) sits relative to the stub:

  clean : port at PORT_MARGIN, 15 mm of line before the stub (~lambda_g/4+)
  near  : port 2 mm before the stub, so the probe comb lands on it

Everything else — geometry, mesh, materials, solve length — is identical.

RECORDED VERDICT (2026-08-27): NOT READ. Both arms failed the settling
witness, so neither |S11| number is quotable and the pre-declared comparison
was never made. The verdict block in main() is not settling-gated — it
prints whichever branch the deltas select — so do not quote it from an
unsettled run. Superseded the next day by
scripts/diagnostics/msl_probe_clearance_shorted_line.py, whose truth is
exact at every frequency and which gates its control arm on settling before
the second arm is spent; the answer to the #726 contradiction is recorded
there and in the issue comment of 2026-08-28. Kept for the record of what
was tried first.
"""
from __future__ import annotations

import argparse
import math
import numpy as np

from rfx import Box, Simulation

C0 = 299792458.0
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
STUB_LEN = 12e-3
L_LINE = 30e-3
PORT_MARGIN = 2e-3
DX = 80e-6

u = W_TRACE / H_SUB
EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5
F_NOTCH_AN = C0 / (4 * STUB_LEN * math.sqrt(EPS_EFF))


def build(port_x: float) -> Simulation:
    LX = L_LINE + 2 * PORT_MARGIN
    msl_clearance = 2 * (2 * H_SUB + 8 * DX)
    LY = W_TRACE + msl_clearance + STUB_LEN + 2 * (2 * H_SUB + 8 * DX)
    LZ = H_SUB + 1.5e-3
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = msl_clearance / 2 + W_TRACE / 2
    trace_y_lo, trace_y_hi = y_trace - W_TRACE / 2, y_trace + W_TRACE / 2
    sim.add(Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
            material="pec")
    sx = LX / 2.0
    sim.add(Box((sx - W_TRACE / 2, trace_y_hi, H_SUB),
                (sx + W_TRACE / 2, trace_y_hi + STUB_LEN, H_SUB + DX)),
            material="pec")
    sim.add_msl_port(position=(port_x, y_trace, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB, direction="-x",
                     impedance=50.0)
    return sim, sx


def run_arm(label: str, port_x: float, num_periods: float):
    sim, stub_x = build(port_x)
    print(f"\n=== ARM {label}: port at x = {port_x * 1e3:.2f} mm, stub at "
          f"{stub_x * 1e3:.2f} mm (clearance {abs(stub_x - port_x) * 1e3:.2f} mm) ===")
    advisories = [str(a) for a in sim.preflight()]
    probe_warn = [a for a in advisories if "probe" in a and "reflector" in a]
    print(f"  preflight: {len(advisories)} advisories; "
          f"{len(probe_warn)} probe-clearance warning(s)")
    for a in probe_warn:
        print(f"  ! {a[:220]}")
    res = sim.compute_msl_s_matrix(n_freqs=100, num_periods=num_periods)
    f = np.asarray(res.freqs, float)
    S11 = np.asarray(res.S[0, 0], np.complex128)
    Z0 = np.asarray(res.Z0[0], np.complex128)
    k = int(np.argmin(np.abs(f - F_NOTCH_AN)))
    band = (f > 0.6 * F_NOTCH_AN) & (f < 1.4 * F_NOTCH_AN)
    kmax = int(np.argmax(np.abs(S11) * band))
    print(f"  |S11| at the analytic notch {F_NOTCH_AN / 1e9:.3f} GHz: "
          f"{20 * np.log10(abs(S11[k])):+.2f} dB")
    print(f"  |S11| max in [0.6, 1.4]*f_notch: "
          f"{20 * np.log10(abs(S11[kmax])):+.2f} dB at {f[kmax] / 1e9:.3f} GHz")
    print(f"  Re(Z0) over that band: {np.real(Z0[band]).min():.2f}.."
          f"{np.real(Z0[band]).max():.2f} ohm (analytic ~"
          f"{87 / math.sqrt(EPS_R + 1.41) * math.log(5.98 * H_SUB / (0.8 * W_TRACE + DX)):.1f})")
    sett = getattr(res, "settling_db", None)
    if sett is not None:
        print(f"  settling witness: {np.max(np.asarray(sett)):.1f} dB")
    return dict(f=f, S11=S11, Z0=Z0, notch_db=20 * np.log10(abs(S11[k])),
                peak_db=20 * np.log10(abs(S11[kmax])), peak_f=f[kmax])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-periods", type=float, default=20.0)
    a = ap.parse_args()
    print(f"analytic notch {F_NOTCH_AN / 1e9:.3f} GHz (eps_eff {EPS_EFF:.4f}); "
          "at the notch a lossless passive structure must reflect: |S11| -> 0 dB")
    clean = run_arm("clean", PORT_MARGIN, a.num_periods)
    near = run_arm("near", (L_LINE + 2 * PORT_MARGIN) / 2 - 2e-3, a.num_periods)
    d_notch = near["notch_db"] - clean["notch_db"]
    d_peak = near["peak_db"] - clean["peak_db"]
    print("\n=== VERDICT (pre-declared) ===")
    print(f"  |S11|@notch  clean {clean['notch_db']:+.2f} dB -> near "
          f"{near['notch_db']:+.2f} dB  (delta {d_notch:+.2f} dB)")
    print(f"  |S11| band peak clean {clean['peak_db']:+.2f} dB -> near "
          f"{near['peak_db']:+.2f} dB  (delta {d_peak:+.2f} dB)")
    if abs(d_peak) >= 2.0:
        print("  -> PREFLIGHT is right: probe clearance biases |S11|, not just "
              "Z0. The guard's 'V*I-split S11/S21 are unaffected' is wrong in "
              "this regime.")
    else:
        print("  -> GUARD is right within 2 dB: |S11| is insulated from the "
              "probe-clearance condition that wrecks Z0; preflight's 5-10 dB "
              "claim does not reproduce here.")


if __name__ == "__main__":
    main()

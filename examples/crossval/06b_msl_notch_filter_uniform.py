"""Cross-validation 06b: MSL Notch Filter — uniform mesh + add_msl_port.

This is a sibling to ``06_msl_notch_filter.py`` (which uses non-uniform mesh
+ wire ``add_port`` + graded-σ absorber + Z_probe=1kΩ workaround). 06b
shows the same notch-filter physics using the **distributed MSL port**
(``add_msl_port``, validated 2026-05-04 to OpenEMS-class accuracy) on a
uniform mesh, with no graded-σ absorber.

Rationale:
  - Wire ``add_port(extent=...)`` covers ONE cell transverse to the trace,
    missing ~3/4 of the quasi-TEM mode's lateral extent. Partial reflection
    at both ports sets up a Fabry-Perot comb that masks the stub notch.
  - ``add_msl_port`` covers the FULL trace cross-section with a Laplace-Ez
    source distribution + distributed-σ matched termination + 3-probe
    de-embedding. F-P ripple is replaced by the simulator noise floor
    (\|S11\|≈0.10 = -20dB, OpenEMS-class).

Scope:
  - Uniform mesh dx=80µm (matches today's validated mesh-conv at 3
    substrate cells). cv06 uses non-uniform; ``add_msl_port`` is
    uniform-mesh-validated — non-uniform support is a separate task.
  - Smaller domain than cv06 (line length 30mm vs 100mm) to keep
    runtime modest.
  - Stub length 12mm (same as cv06) → analytic notch ~3.69 GHz.

Authoritative MSL port correctness gates: the unit + integration tests
under ``tests/test_msl_port*.py``. This crossval is a **physics-level
demo** that the new port API can resolve a stub-notch resonance without
the wire-port + absorber workaround.

Run: ``python examples/crossval/06b_msl_notch_filter_uniform.py``
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8


# Geometry — same as cv06, smaller line length
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
STUB_LEN = 12e-3
L_LINE = 30e-3        # vs cv06's 100mm
PORT_MARGIN = 2e-3
F_MAX = 7e9
DX = 80e-6            # validated mesh-conv at this cell size

# Hammerstad-Jensen ε_eff for analytic notch frequency
u = W_TRACE / H_SUB
EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5
F_NOTCH_AN = C0 / (4 * STUB_LEN * np.sqrt(EPS_EFF))


def _build_sim() -> Simulation:
    """Build the notch-filter simulation with msl_port at both ends."""
    LX = L_LINE + 2 * PORT_MARGIN
    # Lateral box: W + 2·(2·h_sub + 8·dx) on the MSL side, plus stub_length
    # on the +y side to fit the open-circuit stub.
    msl_clearance = 2 * (2 * H_SUB + 8 * DX)
    LY = W_TRACE + msl_clearance + STUB_LEN + 2 * (2 * H_SUB + 8 * DX)
    LZ = H_SUB + 1.5e-3

    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")

    # Place trace at y where there's clearance below + stub above
    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2.0
    trace_y_lo = y_trace - W_TRACE / 2.0
    trace_y_hi = y_trace + W_TRACE / 2.0

    # Main microstrip line (full LX so it goes through CPML — required for
    # MSL port termination, see commit 8882ef1 on msl_port_integration test).
    sim.add(
        Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )

    # Open-circuit stub branching off the main line at x = LX/2
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_TRACE / 2.0
    stub_x_hi = stub_x_centre + W_TRACE / 2.0
    sim.add(
        Box((stub_x_lo, trace_y_hi, H_SUB),
            (stub_x_hi, trace_y_hi + STUB_LEN, H_SUB + DX)),
        material="pec",
    )

    sim.add_msl_port(
        position=(PORT_MARGIN, y_trace, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="+x", impedance=50.0,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_trace, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0,
    )
    return sim


def main() -> int:
    print("=" * 70)
    print("Crossval 06b: MSL Notch Filter (uniform mesh + add_msl_port)")
    print("=" * 70)
    print(f"εr={EPS_R}, h_sub={H_SUB*1e6:.0f}µm, W={W_TRACE*1e6:.0f}µm")
    print(f"line length L={L_LINE*1e3:.0f}mm, stub L_stub={STUB_LEN*1e3:.1f}mm")
    print(f"u={u:.3f}, ε_eff_HJ={EPS_EFF:.3f}, "
          f"analytic notch f={F_NOTCH_AN/1e9:.3f} GHz")
    print(f"mesh: dx={DX*1e6:.0f}µm, n_z_sub={int(round(H_SUB/DX))}")
    print()

    sim = _build_sim()

    print("Preflight:")
    sim.preflight(strict=False)
    print()

    print("Running rfx 2-port S-matrix sweep...")
    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=100, num_periods=20.0)
    dt = time.time() - t0
    print(f"  ... done in {dt:.1f}s")

    f = np.asarray(res.freqs)
    s11 = np.asarray(res.S[0, 0, :])
    s21 = np.asarray(res.S[1, 0, :])
    z0 = np.asarray(res.Z0[0, :])

    # Find S21 minimum (the notch)
    s21_db = 20 * np.log10(np.abs(s21) + 1e-30)
    i_notch = int(np.argmin(s21_db))
    f_notch_rfx = float(f[i_notch])
    s21_notch_db = float(s21_db[i_notch])

    err_pct = abs(f_notch_rfx - F_NOTCH_AN) / F_NOTCH_AN * 100.0

    print()
    print("Result:")
    print(f"  Notch frequency (rfx)      = {f_notch_rfx/1e9:.3f} GHz")
    print(f"  Notch frequency (analytic) = {F_NOTCH_AN/1e9:.3f} GHz")
    print(f"  Notch frequency error      = {err_pct:.2f} %")
    print(f"  Notch depth |S21|          = {s21_notch_db:.1f} dB")
    print(f"  Re(Z0) median              = {float(np.median(z0.real)):.1f} Ω")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(f / 1e9, 20 * np.log10(np.abs(s21) + 1e-30),
                 label="|S21| rfx (msl_port)", color="C0")
    axes[0].plot(f / 1e9, 20 * np.log10(np.abs(s11) + 1e-30),
                 label="|S11| rfx (msl_port)", color="C1")
    axes[0].axvline(F_NOTCH_AN / 1e9, color="k", ls="--", lw=0.8,
                    label=f"analytic notch ({F_NOTCH_AN/1e9:.3f} GHz)")
    axes[0].set_ylabel("|S| [dB]")
    axes[0].set_ylim(-50, 5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_title("MSL notch filter — uniform mesh + add_msl_port")

    axes[1].plot(f / 1e9, np.abs(z0), label="|Z0|")
    axes[1].axhline(50, color="k", ls="--", lw=0.8, label="50 Ω")
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_ylabel("Z0 [Ω]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "06b_msl_notch_filter_uniform.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out_png}")

    # Pass criteria — physics-demo, loose tolerances:
    pass_notch_freq = err_pct < 15.0     # within 15% of analytic
    pass_notch_depth = s21_notch_db < -10  # at least 10 dB notch visible
    pass_z0 = 40 < float(np.median(z0.real)) < 65

    print()
    print("Gates:")
    print(f"  Notch freq vs analytic (< 15 %): "
          f"{'PASS' if pass_notch_freq else 'FAIL'}  ({err_pct:.2f} %)")
    print(f"  Notch depth (< -10 dB):          "
          f"{'PASS' if pass_notch_depth else 'FAIL'}  ({s21_notch_db:.1f} dB)")
    print(f"  Z0 median ∈ (40, 65) Ω:          "
          f"{'PASS' if pass_z0 else 'FAIL'}  ({float(np.median(z0.real)):.1f} Ω)")

    all_ok = pass_notch_freq and pass_notch_depth and pass_z0
    print(f"\n{'PASS' if all_ok else 'FAIL'}: cv06b — "
          f"{'MSL port resolves stub notch' if all_ok else 'gates failed'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

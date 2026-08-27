"""Cross-validation 06b: MSL Notch Filter — uniform mesh + add_msl_port.

This began as a sibling to the RETIRED ``06_msl_notch_filter.py`` (non-uniform
mesh + wire ``add_port`` + graded-σ absorber + Z_probe=1kΩ workaround; removed
as artifact-anchored — issue #339, content in git history). 06b
shows the same notch-filter physics using the **distributed MSL port**
(``add_msl_port``) on a uniform mesh, with no graded-σ absorber. Under the
physics-first evidence taxonomy this is E2-promising: it uses an analytic
quarter-wave notch and internal MSL gates, but broad E5 claims still require
raw V/I replay and an external cross-solver envelope.

Rationale:
  - Wire ``add_port(extent=...)`` covers ONE cell transverse to the trace,
    missing ~3/4 of the quasi-TEM mode's lateral extent. Partial reflection
    at both ports sets up a Fabry-Perot comb that masks the stub notch.
  - ``add_msl_port`` covers the FULL trace cross-section with a Laplace-Ez
    source distribution + distributed-σ matched termination + 3-probe
    de-embedding. F-P ripple is reduced to the current narrow-envelope
    floor (|S11|≈0.10 = -20dB).

External cross-check (openEMS, 2026-07-05):
  A matched-geometry rfx-vs-openEMS comparison is committed under
  ``tests/fixtures/msl_notch_e4/`` (gate:
  ``tests/test_msl_notch_e4_comparison_gates.py``). At a CONVERGED dx=50µm mesh
  (5.08 substrate cells) where both solvers are passive, the off-notch |S21|
  transmission agrees to ~0.1, while the notch frequency agrees to ~6% (rfx
  3.63 GHz vs openEMS 3.43 GHz; fringing-free analytic 3.69 GHz).
  UPDATE (Palace FEM referee, 2026-07-07): an independent
  conformal-tet FEM run on the matched geometry lands at ~3.631 GHz at two mesh
  densities — closest to rfx (+0.1%) — see
  ``tests/fixtures/msl_notch_e4/msl_stub_notch_palace_referee.json``. Our earlier
  working interpretation (open-end fringing as the driver of the split) is
  revised: the FEM value indicates the fringing correction is ~1-2%. The
  "err<15% vs analytic" gate below is rfx-vs-ANALYTIC and is NOT an
  OpenEMS-class number. HISTORICAL: this paragraph describes the dx=80µm
  mesh this script shipped at through 2026-08 — see "Mesh convention"
  below for the dx=63.5µm mesh it ships at now. CAVEAT this fix does NOT
  resolve: the E4 fixture (``tests/fixtures/msl_notch_e4/``) was produced
  by ``scripts/diagnostics/build_msl_notch_rfx_dx50.py`` at DX=50µm, where
  h_sub/dx=5.08 is itself off-lattice and realizes h_sub=300µm (not
  254µm) — that producer carries the SAME #722/#723 defect this script
  just fixed and is EXPLICITLY DEFERRED (see its own docstring), not
  fixed here. So as of this change the E4/Palace comparison above and
  this script solve DIFFERENT boards (300µm vs 254µm) — treat the
  "rfx 3.63 GHz" and "Re(Z0) 57.9 Ω" figures above as belonging to the
  300µm board, not to this script's own output.

Mesh convention (issue #723, 2026-08-27):
  DX = H_SUB / 4 = 63.5µm (was 80µm, h_sub/dx=3.175 — the "IMPORTANT"
  paragraph above already called that mesh UNDER-RESOLVED for
  external-class work). gcd(254, 600) = 2µm, so no single cubic cell size
  makes both H_SUB and W_TRACE land exactly on the lattice at any sane
  cost (63.5µm: 600/63.5=9.449 cells). This script resolves that by
  REALIZING H_SUB exactly (the dimension the port-resolution preflight
  checks and the Z0-bias sweep below both key off) and QUOTING THE
  REALIZED W_TRACE in the analytic reference, rather than re-declaring
  W_TRACE on a lattice multiple. ``_realized_trace_width()`` reads that
  value live from ``sim.fidelity_report()`` after ``_build_sim()`` — NOT
  a ``round(W_TRACE / DX) * DX`` re-derivation, which gives the WRONG
  answer here (571.5µm / 9 cells) because the half-open ``[lo, hi)`` node
  rasterization (``rfx.geometry.csg.Box``) counts the OVERLAPPED node
  span, not the declared extent rounded to the nearest cell.

  Verified via ``sim.fidelity_report()`` at DX=H_SUB/4 (this run, quoted
  verbatim):
    "geometry[0] 'ro4350b' ... z: declared [0.0, 254.0] um -> realized
    [0.0, 254.0] um | face residuals (0.0, 0.0) um | extent 254.0 ->
    254.0 um" — substrate thickness now EXACT (was +25.98% at dx=80µm).
    "geometry[1] 'pec' ... y: declared [1016.0, 1616.0] um -> realized
    [1016.0, 1651.0] um | face residuals (0.0, 35.0) um | extent 600.0 ->
    635.0 um" and "geometry[2] 'pec' ... x: ... extent 600.0 -> 635.0 um"
    — main trace and stub realize the SAME 635.0µm width (10 cells), so
    ``u = W_realized / H_SUB`` = 2.500 describes one consistent board
    (this was NOT true at dx=80µm: 560µm on the trace vs 640µm on the
    stub). Total: 5 entities, 9 findings (was 12 at dx=80µm) — the two
    retired findings are BOTH MSL-port substrate-resolution warnings
    (below); the off-lattice conductor-edge warning is NOT retired (see
    "Preflight honesty" below).

  Reference-formula effect: u 2.362 (declared) -> 2.500 (realized) moves
  ε_eff_HJ 2.869 -> 2.882 and the analytic notch 3.6872 -> 3.6790 GHz
  (-0.22%) — small next to the current 15% gate.

  Z0 anchor: ``scripts/diagnostics/msl_z0_bias_floor_sweep.py`` runs this
  SAME W=600/h=254 RO4350B fixture through a predeclared dx grid that
  includes this exact mesh — committed row (``msl_z0_bias_floor_sweep/
  msl_z0_bias_floor_sweep.json``, label "aligned h_sub/4"): dx_um=63.5,
  z0_measured_ohm=46.098, z0_hj_ohm=47.895 (HJ on the DECLARED 600/254
  board). ``rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff``:
  HJ(635µm, 254µm, 3.66) = 46.18 Ω, 0.18% from the measured 46.10 Ω, vs
  HJ(600µm, 254µm, 3.66) = 47.90 Ω, 3.9% away — comparator-first evidence
  that the realized width, not the declared one, is the right analytic
  anchor on this mesh. Predicted post-fix median Re(Z0) ≈ 46.1 Ω, 15.2%
  above the (40, 65) Ω gate floor below (window unchanged; NOT re-pinned
  by this change — that needs a fresh solve, see "Runtime" below).

  Runtime and the envelope this re-pins: measured grid shape (this run,
  ``sim._build_grid().shape``) dx=80µm -> (442, 232, 31) = 3,178,864
  cells; dx=63.5µm -> (553, 280, 37) = 5,729,080 cells (1.802x). Combined
  with the ~1.260x more timesteps (dt ∝ dx), wall clock scales ~2.271x.
  The committed baseline (``_06b_notch_uniform_logs/20260809T_run.log``)
  measured "... done in 2599.6s" at dx=80µm -> a ~5902s (98 min) CPU
  projection at dx=63.5µm on a GPU-less pod (NOT measured — no solve was
  run to produce this docstring). That same log's PRE-FIX envelope,
  which a fresh run at this mesh must re-pin with written provenance
  (not silently, and not by this script's authors alone):
  "Notch frequency (rfx) = 3.627 GHz", "Notch frequency error = 1.63 %",
  "Notch depth |S21| = -34.2 dB", "Re(Z0) median = 57.9 Ω".

  KNOWN LIMITATION, filed as #729 (NOT folded into #723): at every
  ALIGNED dx, ``add_msl_port``'s own cross-section audit
  (``rfx/sources/msl_port.py::msl_cross_section_span``) independently
  rasterizes the substrate height and overshoots it by one cell — the
  z_hi = h_sub face lands exactly on a node and ``Grid.position_to_index``
  (round-to-nearest) resolves to the cell above. Measured during the
  #723 review: n_z_sub=5 rows / 317.5µm at DX=H_SUB/4 (was n_z_sub=4 /
  320µm at the old dx=80µm, where it coincidentally matched the then
  26%-too-thick FDTD board), so the port's quasi-static Laplace mode
  model solves a ~317.5µm substrate with the trace strip at z=317.5µm
  while the FDTD board is exactly 254µm with the PEC wall at z=254µm
  (z0_static 53.29 -> 56.88 Ω across that same measurement). This is an
  rfx API rasterization behaviour, not a script-convention choice, and
  the extraction reads real FDTD V/I fields rather than the port's
  z0_static, so its effect is bounded by the committed
  msl_z0_bias_floor_sweep row above (gamma_implied=-0.019,
  mean_s11_raw=0.0223) — it does not block this fix, but needs its own
  issue against ``msl_cross_section_span``.

  Preflight honesty: this mesh retires BOTH MSL-port substrate-resolution
  warnings ("only 3 substrate cell(s) in z ... Refine to dx ≤ 64µm" and
  "h_sub/dx = 3.175 ... mixed-cell danger zone", both present at dx=80µm,
  both ABSENT at dx=63.5µm). It does NOT retire, and marginally WORSENS,
  the off-lattice conductor-edge warning — this run's own preflight,
  quoted verbatim: dx=80µm "geometry[1] 'pec' y: extent 600µm, worst face
  residual 28µm (4.67% of the extent)" -> dx=63.5µm "geometry[1] 'pec' y:
  extent 600µm, worst face residual 28.5µm (4.75% of the extent)" — the
  expected price of quoting the realized width instead of re-declaring
  W_TRACE on a lattice multiple.

Scope:
  - Uniform mesh dx=63.5µm = H_SUB/4 (issue #723; was dx=80µm, h_sub/dx=
    3.175, an UNDER-RESOLVED mixed-cell substrate per the "External
    cross-check" paragraph above and this script's own MSL-port
    preflight). The retired cv06 used non-uniform; ``add_msl_port``
    promotion remains uniform-lane only until a separate non-uniform
    evidence ladder exists.
  - Smaller domain than cv06 (line length 30mm vs 100mm) to keep
    runtime modest.
  - Stub length 12mm (same as cv06) → analytic notch ~3.68 GHz (realized
    width; see "Mesh convention").

Authoritative MSL port correctness gates: the unit + integration tests
under ``tests/test_msl_port*.py``. This crossval is a **physics-level
demo** that the new port API can resolve a stub-notch resonance without
the wire-port + absorber workaround.

Run: ``python validation/crossval/06b_msl_notch_filter_uniform.py``
(CPU-only projection ~98 min; see "Runtime" above — not measured by this
change).
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
DX = H_SUB / 4         # 63.5um — REALIZE-DECLARED on z (issue #723); see
                        # "Mesh convention" below.


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


def _realized_trace_width(sim: Simulation) -> float:
    """Main-trace width as the RASTERIZER actually realizes it (metres).

    Read live from ``sim.fidelity_report()`` rather than re-derived with a
    ``round(W_TRACE / DX) * DX`` formula: the half-open ``[lo, hi)`` node
    convention (``rfx.geometry.csg.Box``) counts the OVERLAPPED node span,
    not the rounded declared extent, and the two disagree by a cell at this
    mesh (571.5um / 9 cells from the round() formula vs the true 635.0um /
    10 cells — see the "Mesh convention" docstring section, issue #723
    BLOCKER 1). ``geometry[1]`` is the main trace (added first of the two
    PEC bodies in ``_build_sim``); its y-axis is transverse to propagation.
    """
    report = sim.fidelity_report(print_report=False)
    for item in report:
        if item["entity"] == "geometry[1] 'pec'":
            for ax in item["axes"]:
                if ax["axis"] == "y":
                    return float(ax["realized_extent_um"]) * 1e-6
    raise RuntimeError(
        "_realized_trace_width: could not find geometry[1] 'pec' y-axis in "
        "sim.fidelity_report() — did _build_sim()'s geometry order change?"
    )


def main() -> int:
    print("=" * 70)
    print("Crossval 06b: MSL Notch Filter (uniform mesh + add_msl_port)")
    print("=" * 70)
    print(f"εr={EPS_R}, h_sub={H_SUB*1e6:.0f}µm, W_declared={W_TRACE*1e6:.0f}µm")
    print(f"line length L={L_LINE*1e3:.0f}mm, stub L_stub={STUB_LEN*1e3:.1f}mm")
    print(f"mesh: dx={DX*1e6:.1f}µm, n_z_sub={int(round(H_SUB/DX))}")

    sim = _build_sim()

    # Hammerstad-Jensen ε_eff for the analytic notch — from the REALIZED
    # trace width, not the declared one (issue #723; see "Mesh convention").
    w_realized = _realized_trace_width(sim)
    u = w_realized / H_SUB
    EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5
    F_NOTCH_AN = C0 / (4 * STUB_LEN * np.sqrt(EPS_EFF))
    print(f"W_realized={w_realized*1e6:.1f}µm, u={u:.3f}, ε_eff_HJ={EPS_EFF:.3f}, "
          f"analytic notch f={F_NOTCH_AN/1e9:.3f} GHz")
    print()

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

"""Cross-solver validation 07: Sheen 1990 microstrip low-pass filter.

!!! SUPERSEDED FRAMING — read the Palace referee before trusting the "first null"
    metric below. A conformal-mesh Palace-FEM referee (VESSL, runs 369367248550 /
    369367248558; scripts/diagnostics/build_sheen_lpf_palace_referee.py +
    tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json) showed the stopband
    is a DOUBLE transmission-zero (~7.0 AND ~8.0 GHz), not a single null. openEMS
    resolves BOTH zeros and matches Palace to ~0.7%; rfx's coarse 200um mesh
    DISTORTS the doublet (spurious ~6.6 GHz dip, no clean 8 GHz zero). So the
    single-argmin "first null" comparison here (rfx 7.218 vs openEMS 7.983 = 9.6%)
    is largely a COMPARATOR ARTIFACT — the argmin picks different doublet members
    per solver and flips with mesh (Palace 7.02->8.05 GHz coarse->mid). The honest
    three-way verdict (sides_with = openems) lives in the referee fixture; rfx is
    the less structure-faithful solver on this doublet at this resolution.

Reproduces the classic FDTD-microwave benchmark of
  D. M. Sheen, S. M. Ali, M. D. Abouzahra, J. A. Kong,
  "Application of the three-dimensional finite-difference time-domain method
  to the analysis of planar microwave circuits," IEEE Trans. MTT 38(7):849-857,
  July 1990.
in BOTH rfx (add_msl_port / compute_msl_s_matrix) and openEMS, then cross-checks
the |S21|(f) PASSBAND and the FIRST STOPBAND NULL FREQUENCY.

Geometry (exact): RT/Duroid substrate eps_r=2.2, h=0.794 mm; a 2.413 mm-wide
50-ohm feed on each side (matches the paper's stated 50-ohm width) joined by a
wide 20.320 x 2.540 mm low-impedance section. Exact trace coordinates are taken
from the Elsherbeni & Demir reproduction ("The FDTD Method for Electromagnetics
with MATLAB Simulations", Sec. 6.2), transcribed in the roseengineering/rffdtd
example examples/lowpass.py. See SHEEN geometry block below.

HONEST SCOPE (do NOT overclaim):
  - The rfx MSL lane is documented "limited / E5-narrow" (see
    docs/guides/sparameter_support_matrix.md). Strong-reflector |S11| rides a
    0.16-0.22 staircase-Z0 floor, so the deep-null DEPTH is NOT gated here.
  - We gate the first-null FREQUENCY (few-% envelope) and the passband band-mean
    |S21|. All deltas are stated rfx-centric ("method distance"); openEMS is a
    reference, not ground truth.
  - dB is always recomputed from the raw |S| arrays, never from a producer dB.

Axis convention: MSL propagation is along +x (rfx add_msl_port only supports
+x/-x). The Sheen board is mapped   rfx_x = Sheen_y (propagation),
rfx_y = Sheen_x (transverse), z = z. The two 50-ohm feeds are extended by
EXTEND_FEED of matched 50-ohm line on each side so BOTH solvers' de-embedding
have clean line downstream of the reference plane; this adds only linear phase,
not |S21| structure (openEMS's own MSL tutorial uses 50 mm feeds for the same
reason).

WHAT THE GATES ACTUALLY LOCK (diagnostic-reporter, NOT an accuracy claim)
------------------------------------------------------------------------
Because the Palace referee overturned the single-null accuracy framing, the
default ``compare`` mode does NOT gate rfx-vs-openEMS agreement. It gates:

  A. comparator sanity   -- the openEMS REFERENCE is itself sound (passive
     within the documented |S| <= 1.05 / column-power <= 1.10 envelope,
     median Re(Z0) near 50 ohm, a real passband and a deep stopband zero).
     Comparator-first: an unsound reference makes everything downstream
     unreadable.
  B. comparison shape    -- both result sets are present and cover the
     characterized band with the committed bin counts.
  C. characterization lock -- the committed argmin "first null" numbers and
     the Palace referee's recorded verdict (sides_with = openems) reproduce.
     This locks WHAT WAS CHARACTERIZED so a silent change goes red.
  D. rfx non-passivity FOOTPRINT lock -- see below.

KNOWN DEFECT IN THE COMMITTED rfx ARTIFACT (gate D, quoted not hidden)
---------------------------------------------------------------------
The committed rfx run is NOT passive. Its MSL de-embedding blows up toward the
top of the swept band: worst bin 17.05 GHz with |S11| = 15.3, |S21| = 39.5,
column power |S11|^2+|S21|^2 = 1794.5, and Re(Z0) even goes negative (-1.1
ohm). 58 of 120 bins (4.43-20.0 GHz) exceed the documented passivity envelope
(column power <= 1.10, tests/test_sparam_passivity_guard.py) -- including 29
bins with excursions up to 1.56 INSIDE the 5-15 GHz doublet-search band that
the study reads its result from. The 0.5-3 GHz passband is the only clean
region (max column power 1.08, within envelope).

Per the repo passivity rule this is an EXTRACTION defect, not physics, and it
is NOT gated as a pass. Gate D instead locks the defect's footprint (violating
bin count, worst bin location, worst column power) as a regression lock, so
the case goes red if the defect changes. Consequence for the reader: NO |S|
magnitude from the rfx leg of this study is quotable as physics. The study is
registered for its stopband-STRUCTURE characterization only.

Exit contract (crossval registry, validation/crossval/manifest.json)
-------------------------------------------------------------------
    0 -> every configured gate above passed
    1 -> a gate failed, or the rfx result leg is missing (execution gap)
    2 -> the openEMS reference is unavailable (missing result file, or
         CSXCAD/openEMS not importable for a solver mode): inconclusive

Usage:
    python validation/crossval/07_sheen_lpf.py            # default: compare (gates)
    python validation/crossval/07_sheen_lpf.py tutorial   # comparator-first: openEMS canonical MSL tutorial
    python validation/crossval/07_sheen_lpf.py openems    # Sheen filter in openEMS -> _07_sheen_results/openems.json
    python validation/crossval/07_sheen_lpf.py rfx        # Sheen filter in rfx     -> _07_sheen_results/rfx.json
    python validation/crossval/07_sheen_lpf.py compare    # load both, gate the committed characterization
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(SCRIPT_DIR, "_07_sheen_results")
os.makedirs(RES_DIR, exist_ok=True)
# Palace-FEM referee fixture (three-solver verdict; see the banner above).
REFEREE_JSON = os.path.join(
    SCRIPT_DIR, "..", "..", "tests", "fixtures", "sheen_lpf_e4",
    "sheen_lpf_palace_referee.json",
)
C0 = 2.99792458e8

# --- resolve `import rfx` from THIS checkout (a bare run would otherwise pick
# --- up whatever rfx is installed or first on sys.path, which in a multi-
# --- worktree setup is a DIFFERENT copy of the solver than the one under test).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Documented single-run passivity envelope (tests/test_sparam_passivity_guard.py):
# |S| <= ~1.05  <=>  column power |S11|^2 + |S21|^2 <= 1.10.
PASSIVITY_COLUMN_POWER_MAX = 1.10

# ---------------------------------------------------------------------------
# COMMITTED CHARACTERIZATION (gates C and D lock these; see the banner above).
# Values re-derived from the committed _07_sheen_results/*.json artifacts.
# ---------------------------------------------------------------------------
COMMITTED = dict(
    rfx_null_ghz=7.2185,        # argmin |S21| over 5-15 GHz
    oems_null_ghz=7.9831,
    null_tol_pct=1.0,           # regression tolerance on the locked numbers
    rfx_nbins=120,
    oems_nbins=801,
    rfx_bad_bins=58,            # bins exceeding PASSIVITY_COLUMN_POWER_MAX
    rfx_worst_bad_ghz=17.050,
    rfx_worst_column_power=1794.55,
    rfx_bad_bins_in_null_band=29,   # of those, inside the 5-15 GHz search band
    referee_sides_with="openems",
)

# ---------------------------------------------------------------------------
# SHEEN geometry (metres). Native Sheen frame: x_S (transverse, patch length),
# y_S (propagation, feeds), z (stack). Values from rffdtd examples/lowpass.py.
# ---------------------------------------------------------------------------
EPS_R = 2.2
H_SUB = 0.794e-3
BOARD_XS = 22.320e-3        # Sheen x extent
BOARD_YS = 19.472e-3        # Sheen y extent (propagation)
W_FEED = 2.413e-3           # 50-ohm feed width
# input feed:  Sheen x 6.650-9.063, y 0-8.466   -> centre_xS = 7.8565, len 8.466
# output feed: Sheen x 13.257-15.670, y 11.006-19.472
# wide patch:  Sheen x 1.000-21.320 (20.320), y 8.466-11.006 (2.540)
IN_FEED_XS_C = 0.5 * (6.650e-3 + 9.063e-3)      # 7.8565 mm
OUT_FEED_XS_C = 0.5 * (13.257e-3 + 15.670e-3)   # 14.4635 mm
PATCH_XS_LO, PATCH_XS_HI = 1.000e-3, 21.320e-3  # transverse span of patch
PATCH_YS_LO, PATCH_YS_HI = 8.466e-3, 11.006e-3  # propagation span of patch
IN_FEED_LEN = PATCH_YS_LO                        # 8.466 mm (edge -> patch)
OUT_FEED_LEN = BOARD_YS - PATCH_YS_HI            # 8.466 mm (patch -> edge)

# ---------------------------------------------------------------------------
# Modelling choices shared by BOTH solvers (matched geometry).
# ---------------------------------------------------------------------------
EXTEND_FEED = 4.0e-3        # extra matched 50-ohm feed per side (de-embed room)
Y_CLEAR = 3.0e-3            # transverse clearance from patch edge to boundary
Z_AIR = 3.0e-3             # air above the trace
F_MAX = 20.0e9              # Sheen analysis band top
F_LO = 0.5e9

# derived propagation-axis (rfx/openEMS x) layout, absolute metres:
#   x=0 .............. board input edge (extended feed start)
#   IN region: 0 -> PATCH_X0 = EXTEND + IN_FEED_LEN
#   PATCH:     PATCH_X0 -> PATCH_X1 = PATCH_X0 + (PATCH_YS_HI-PATCH_YS_LO)
#   OUT region:PATCH_X1 -> LX = PATCH_X1 + OUT_FEED_LEN + EXTEND
PATCH_X0 = EXTEND_FEED + IN_FEED_LEN
PATCH_LEN_PROP = PATCH_YS_HI - PATCH_YS_LO        # 2.540 mm
PATCH_X1 = PATCH_X0 + PATCH_LEN_PROP
LX = PATCH_X1 + OUT_FEED_LEN + EXTEND_FEED

# transverse (rfx/openEMS y): map Sheen x -> y with +Y_SHIFT so patch clears y=0
PATCH_TRV_LEN = PATCH_XS_HI - PATCH_XS_LO          # 20.320 mm
Y_SHIFT = Y_CLEAR - PATCH_XS_LO                     # patch_lo -> Y_CLEAR
def yS(x_sheen):                                    # Sheen-x -> domain-y
    return x_sheen + Y_SHIFT
IN_FEED_YC = yS(IN_FEED_XS_C)
OUT_FEED_YC = yS(OUT_FEED_XS_C)
PATCH_Y_LO, PATCH_Y_HI = yS(PATCH_XS_LO), yS(PATCH_XS_HI)
LY = PATCH_Y_HI + Y_CLEAR
LZ = H_SUB + Z_AIR

PORT_MARGIN = 2.5e-3       # port plane distance from x-boundary (>2*h_sub)


def _geom_banner():
    print("Sheen 1990 microstrip LPF   (Elsherbeni-Demir Sec 6.2 / rffdtd lowpass.py)")
    print(f"  substrate eps_r={EPS_R}, h={H_SUB*1e3:.3f} mm")
    print(f"  50-ohm feed width={W_FEED*1e3:.3f} mm; wide section "
          f"{PATCH_TRV_LEN*1e3:.3f} x {PATCH_LEN_PROP*1e3:.3f} mm")
    print(f"  matched-feed extension per side = {EXTEND_FEED*1e3:.1f} mm")
    print(f"  domain (prop x, trv y, z) = {LX*1e3:.2f} x {LY*1e3:.2f} x "
          f"{LZ*1e3:.2f} mm")


# ===========================================================================
# rfx side
# ===========================================================================
def run_rfx(dx, num_periods, n_freqs):
    from rfx import Simulation, Box
    from rfx.boundaries.spec import Boundary, BoundarySpec
    import io
    import contextlib

    n_sub = int(round(H_SUB / dx))
    print("=" * 72)
    print("rfx side")
    print("=" * 72)
    _geom_banner()
    print(f"  mesh: dx={dx*1e6:.1f} um  -> substrate = {n_sub} cells  "
          f"(coarse; >=4 is the documented MSL minimum)")

    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("duroid", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="duroid")

    # metal: input feed (full x 0 -> PATCH_X0), patch, output feed (PATCH_X1 -> LX)
    tz0, tz1 = H_SUB, H_SUB + dx
    sim.add(Box((0.0, IN_FEED_YC - W_FEED / 2, tz0),
                (PATCH_X0, IN_FEED_YC + W_FEED / 2, tz1)), material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz0),
                (PATCH_X1, PATCH_Y_HI, tz1)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz0),
                (LX, OUT_FEED_YC + W_FEED / 2, tz1)), material="pec")

    sim.add_msl_port(position=(PORT_MARGIN, IN_FEED_YC, 0.0),
                     width=W_FEED, height=H_SUB, direction="+x",
                     impedance=50.0, eps_r_sub=EPS_R, name="p1")
    sim.add_msl_port(position=(LX - PORT_MARGIN, OUT_FEED_YC, 0.0),
                     width=W_FEED, height=H_SUB, direction="-x",
                     impedance=50.0, eps_r_sub=EPS_R, name="p2")

    print("\n--- rfx preflight (verbatim) ---")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.preflight(strict=False)
    preflight_txt = buf.getvalue()
    print(preflight_txt if preflight_txt.strip() else "(no preflight output)")
    print("--- end preflight ---\n")

    freqs = np.linspace(F_LO, F_MAX, n_freqs)
    import jax.numpy as jnp
    print(f"running compute_msl_s_matrix (num_periods={num_periods}, "
          f"n_freqs={n_freqs}) ...")
    import time
    t0 = time.time()
    res = sim.compute_msl_s_matrix(freqs=jnp.asarray(freqs),
                                   num_periods=num_periods)
    dt = time.time() - t0
    print(f"  done in {dt:.1f} s")

    f = np.asarray(res.freqs)
    s11 = np.asarray(res.S[0, 0, :])
    s21 = np.asarray(res.S[1, 0, :])
    z0 = np.asarray(res.Z0[0, :])
    esum = np.abs(s11) ** 2 + np.abs(s21) ** 2
    out = dict(
        solver="rfx", dx_um=dx * 1e6, n_sub_cells=n_sub,
        num_periods=num_periods, runtime_s=dt,
        freqs_hz=f.tolist(),
        s11_mag=np.abs(s11).tolist(), s21_mag=np.abs(s21).tolist(),
        re_z0=np.real(z0).tolist(),
        energy_sum=esum.tolist(),
        preflight=preflight_txt,
    )
    with open(os.path.join(RES_DIR, "rfx.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    _quick_report("rfx", f, np.abs(s21), np.abs(s11), np.real(z0), esum)
    print(f"saved {os.path.join(RES_DIR, 'rfx.json')}")


# ===========================================================================
# openEMS side
# ===========================================================================
def _require_openems():
    """Exit 2 (visible SKIP) if the openEMS reference solver is unavailable.

    CSXCAD + openEMS are NOT rfx dependencies, so a machine without them cannot
    produce the reference. That is an inconclusive crossval, never a pass.
    """
    try:
        import CSXCAD  # noqa: F401
        import openEMS  # noqa: F401
    except ImportError as exc:
        print("\n" + "!" * 72)
        print("!! CROSSVAL 07 SKIPPED — CSXCAD / openEMS not installed")
        print(f"!!   reason: {exc}")
        print("!!   the openEMS reference cannot be produced, so this run is NOT")
        print("!!   a crossval. Exiting 2 (inconclusive) so CI does not read it")
        print("!!   as PASS.")
        print("!" * 72)
        raise SystemExit(2)


def _openems_common_setup(f_max):
    _require_openems()
    import numpy as _np
    _np.int = int      # numpy 2.x shim: openEMS ports.py uses removed aliases
    _np.float = float
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    # bound the run: resonant sections otherwise chase the default energy floor
    # for a very long time on CPU. 1e-4 (-40 dB) is an adequate settling floor.
    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f_max / 2, f_max / 2)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    return FDTD, CSX


def run_openems_tutorial():
    """Comparator-first: canonical openEMS MSL_NotchFilter tutorial.
    Known-good: Z0 ~ 50 ohm, S21 notch ~3.43 GHz (repo fixture: openEMS 3.4286 GHz)."""
    import os
    import tempfile
    FDTD, CSX = _openems_common_setup(7e9)
    from openEMS.physical_constants import C0 as _C0
    unit = 1e-6
    MSL_length, MSL_width, sub_t, sub_epr, stub = 50000, 600, 254, 3.66, 12e3
    f_max = 7e9
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    resolution = _C0 / (f_max * np.sqrt(sub_epr)) / unit / 50
    tm = np.array([2 * resolution / 3, -resolution / 3]) / 4
    mesh.AddLine('x', 0); mesh.AddLine('x', MSL_width / 2 + tm)
    mesh.AddLine('x', -MSL_width / 2 - tm); mesh.SmoothMeshLines('x', resolution / 4)
    mesh.AddLine('x', [-MSL_length, MSL_length]); mesh.SmoothMeshLines('x', resolution)
    mesh.AddLine('y', 0); mesh.AddLine('y', MSL_width / 2 + tm)
    mesh.AddLine('y', -MSL_width / 2 - tm); mesh.SmoothMeshLines('y', resolution / 4)
    mesh.AddLine('y', [-15 * MSL_width, 15 * MSL_width + stub])
    mesh.AddLine('y', (MSL_width / 2 + stub) + tm); mesh.SmoothMeshLines('y', resolution)
    mesh.AddLine('z', np.linspace(0, sub_t, 5)); mesh.AddLine('z', 3000)
    mesh.SmoothMeshLines('z', resolution)
    sub = CSX.AddMaterial('RO4350B', epsilon=sub_epr)
    sub.AddBox([-MSL_length, -15 * MSL_width, 0],
               [MSL_length, 15 * MSL_width + stub, sub_t])
    pec = CSX.AddMetal('PEC')
    port = [None, None]
    port[0] = FDTD.AddMSLPort(1, pec, [-MSL_length, -MSL_width / 2, sub_t],
                              [0, MSL_width / 2, 0], 'x', 'z', excite=-1,
                              FeedShift=10 * resolution, MeasPlaneShift=MSL_length / 3,
                              priority=10)
    port[1] = FDTD.AddMSLPort(2, pec, [MSL_length, -MSL_width / 2, sub_t],
                              [0, MSL_width / 2, 0], 'x', 'z',
                              MeasPlaneShift=MSL_length / 3, priority=10)
    pec.AddBox([-MSL_width / 2, MSL_width / 2, sub_t],
               [MSL_width / 2, MSL_width / 2 + stub, sub_t], priority=10)
    sim_path = os.path.join(tempfile.gettempdir(), 'sheen_tutorial_check')
    FDTD.Run(sim_path, cleanup=True, numThreads=8)
    f = np.linspace(1e6, f_max, 1601)
    for p in port:
        p.CalcPort(sim_path, f)
    z0 = np.real(np.asarray(port[0].Z_ref))
    for p in port:
        p.CalcPort(sim_path, f, ref_impedance=50)
    s11 = port[0].uf_ref / port[0].uf_inc
    s21 = port[1].uf_ref / port[0].uf_inc
    band = (f > 1e9) & (f < 3e9)
    s21db = 20 * np.log10(np.abs(s21) + 1e-30)
    i = int(np.argmin(s21db))
    print("=" * 72)
    print("COMPARATOR-FIRST: openEMS canonical tutorial (MSL_NotchFilter.py)")
    print("=" * 72)
    print(f"  Re(Z0) median 1-3 GHz = {np.median(z0[band]):.2f} ohm   "
          f"(known-good ~50 ohm)")
    print(f"  S21 notch = {f[i] / 1e9:.4f} GHz @ {s21db[i]:.1f} dB   "
          f"(repo fixture openEMS: 3.4286 GHz)")
    esum = np.abs(s11) ** 2 + np.abs(s21) ** 2
    print(f"  max |S11|^2+|S21|^2 = {np.max(esum):.4f} (passivity)")


def run_openems(f_max):
    import os
    import tempfile
    import time
    FDTD, CSX = _openems_common_setup(f_max)
    print("=" * 72)
    print("openEMS side")
    print("=" * 72)
    _geom_banner()
    unit = 1.0  # work in metres
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)

    res = C0 / (f_max * np.sqrt(EPS_R)) / 50.0          # ~lambda/50 transverse
    res = min(res, H_SUB / 4.0)                          # >=4 substrate cells
    tm = np.array([2 * res / 3, -res / 3]) / 4
    # x (propagation)
    mesh.AddLine('x', [0.0, LX, PATCH_X0, PATCH_X1])
    mesh.SmoothMeshLines('x', res)
    # y (transverse) - refine both feed edges + patch edges
    for yc in (IN_FEED_YC, OUT_FEED_YC):
        mesh.AddLine('y', yc + W_FEED / 2 + tm)
        mesh.AddLine('y', yc - W_FEED / 2 - tm)
    mesh.AddLine('y', [0.0, LY, PATCH_Y_LO, PATCH_Y_HI])
    mesh.SmoothMeshLines('y', res)
    # z: >=4 substrate cells + air
    mesh.AddLine('z', np.linspace(0, H_SUB, 5))
    mesh.AddLine('z', LZ)
    mesh.SmoothMeshLines('z', res)

    sub = CSX.AddMaterial('duroid', epsilon=EPS_R)
    sub.AddBox([0.0, 0.0, 0.0], [LX, LY, H_SUB])
    pec = CSX.AddMetal('PEC')

    port = [None, None]
    # port 1: excited, propagation +x, feed spans x 0 -> PATCH_X0 at IN_FEED_YC
    port[0] = FDTD.AddMSLPort(
        1, pec, [0.0, IN_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X0, IN_FEED_YC + W_FEED / 2, 0.0], 'x', 'z', excite=-1,
        FeedShift=PORT_MARGIN, MeasPlaneShift=0.45 * PATCH_X0, priority=10)
    # port 2: passive, propagation -x, feed spans x LX -> PATCH_X1 at OUT_FEED_YC
    out_len = LX - PATCH_X1
    port[1] = FDTD.AddMSLPort(
        2, pec, [LX, OUT_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X1, OUT_FEED_YC + W_FEED / 2, 0.0], 'x', 'z',
        MeasPlaneShift=0.45 * out_len, priority=10)
    # wide low-impedance patch (top surface)
    pec.AddBox([PATCH_X0, PATCH_Y_LO, H_SUB], [PATCH_X1, PATCH_Y_HI, H_SUB],
               priority=10)

    sim_path = os.path.join(tempfile.gettempdir(), 'sheen_openems')
    t0 = time.time()
    FDTD.Run(sim_path, cleanup=True, numThreads=8)
    dt = time.time() - t0
    print(f"  openEMS run done in {dt:.1f} s")

    f = np.linspace(F_LO, f_max, 801)
    # pass 1 (no ref_impedance): port.Z_ref = measured line impedance array
    for p in port:
        p.CalcPort(sim_path, f)
    z0 = np.real(np.asarray(port[0].Z_ref))
    # pass 2 (ref=50): S-parameters referenced to the 50-ohm system impedance
    for p in port:
        p.CalcPort(sim_path, f, ref_impedance=50)
    s11 = port[0].uf_ref / port[0].uf_inc
    s21 = port[1].uf_ref / port[0].uf_inc
    esum = np.abs(s11) ** 2 + np.abs(s21) ** 2
    out = dict(
        solver="openems", res_um=res * 1e6, runtime_s=dt,
        freqs_hz=f.tolist(),
        s11_mag=np.abs(s11).tolist(), s21_mag=np.abs(s21).tolist(),
        re_z0=z0.tolist(), energy_sum=esum.tolist(),
    )
    with open(os.path.join(RES_DIR, "openems.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    _quick_report("openems", f, np.abs(s21), np.abs(s11), z0, esum)
    print(f"saved {os.path.join(RES_DIR, 'openems.json')}")


# ===========================================================================
# reporting / comparison
# ===========================================================================
def _quick_report(tag, f, s21, s11, z0, esum):
    s21db = 20 * np.log10(s21 + 1e-30)
    band = (f >= 5e9) & (f <= 15e9)
    if band.any():
        idx = np.where(band)[0]
        j = idx[int(np.argmin(s21db[band]))]
        print(f"[{tag}] Re(Z0) median = {np.median(z0):.1f} ohm | "
              f"first null in 5-15 GHz = {f[j] / 1e9:.3f} GHz @ {s21db[j]:.1f} dB | "
              f"max energy-sum = {np.max(esum):.3f}")


def _find_null(f, s21, lo, hi):
    m = (f >= lo) & (f <= hi)
    idx = np.where(m)[0]
    s21db = 20 * np.log10(s21 + 1e-30)
    j = idx[int(np.argmin(s21db[m]))]
    return f[j], s21db[j]


def _passband_mean(f, s21, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.mean(s21[m])), float(np.mean(20 * np.log10(s21[m] + 1e-30)))


def _load_legs():
    """Load the two committed result legs, or exit with the right status.

    Missing openEMS reference -> 2 (inconclusive, the reference is required for
    a PASS). Missing rfx leg   -> 1 (the rfx side never ran: execution gap).
    """
    rfx_path = os.path.join(RES_DIR, "rfx.json")
    oems_path = os.path.join(RES_DIR, "openems.json")
    if not os.path.isfile(rfx_path):
        print("!! CROSSVAL 07 FAILED — rfx result leg missing: "
              f"{rfx_path}\n!!   run `python {os.path.basename(__file__)} rfx` "
              "first. Exiting 1 (execution gap).")
        raise SystemExit(1)
    if not os.path.isfile(oems_path):
        print("!" * 72)
        print("!! CROSSVAL 07 SKIPPED — openEMS reference leg missing: "
              f"{oems_path}")
        print("!!   the reference is required for a PASS. Exiting 2 "
              "(inconclusive).")
        print("!" * 72)
        raise SystemExit(2)
    with open(rfx_path) as fp:
        R = json.load(fp)
    with open(oems_path) as fp:
        O = json.load(fp)
    return R, O


def _column_power(d):
    """|S11|^2 + |S21|^2 per bin, recomputed from raw magnitudes (never trusted
    from a producer-side 'energy_sum' field)."""
    return np.array(d["s11_mag"]) ** 2 + np.array(d["s21_mag"]) ** 2


def compare(null_lo, null_hi, pass_lo, pass_hi, paper_null_ghz):
    R, O = _load_legs()
    fr, s21r = np.array(R["freqs_hz"]), np.array(R["s21_mag"])
    fo, s21o = np.array(O["freqs_hz"]), np.array(O["s21_mag"])
    s11r, s11o = np.array(R["s11_mag"]), np.array(O["s11_mag"])
    er, eo = np.array(R["energy_sum"]), np.array(O["energy_sum"])

    # passivity / |S21|>1 flags (extraction artifacts, not physics)
    nbadr = int(np.sum(s21r > 1.0)); nbado = int(np.sum(s21o > 1.0))

    fnull_r, dr = _find_null(fr, s21r, null_lo, null_hi)
    fnull_o, do = _find_null(fo, s21o, null_lo, null_hi)
    pm_r, pmdb_r = _passband_mean(fr, s21r, pass_lo, pass_hi)
    pm_o, pmdb_o = _passband_mean(fo, s21o, pass_lo, pass_hi)

    print("=" * 72)
    print("Sheen 1990 LPF  --  rfx vs openEMS cross-validation")
    print("=" * 72)
    print(f"rfx : dx={R['dx_um']:.1f}um ({R['n_sub_cells']} sub-cells), "
          f"num_periods={R['num_periods']}, {R['runtime_s']:.0f}s, "
          f"Re(Z0)~{np.median(R['re_z0']):.1f} ohm")
    print(f"oems: res~{O['res_um']:.0f}um, {O['runtime_s']:.0f}s, "
          f"Re(Z0)~{np.median(O['re_z0']):.1f} ohm")
    print(f"max energy-sum |S11|^2+|S21|^2: rfx={er.max():.3f}  oems={eo.max():.3f}")
    if nbadr or nbado:
        print(f"!! |S21|>1 bins (EXTRACTION/NORMALIZATION artifact, not physics): "
              f"rfx={nbadr}  oems={nbado}")
    print()
    print(f"{'quantity':<34}{'rfx':>12}{'openEMS':>12}{'paper':>10}")
    print("-" * 68)
    pv = f"{paper_null_ghz:.2f}" if paper_null_ghz else "n/a"
    print(f"{'first S21 null freq [GHz]':<34}{fnull_r/1e9:>12.3f}"
          f"{fnull_o/1e9:>12.3f}{pv:>10}")
    print(f"{'  null depth [dB] (NOT gated)':<34}{dr:>12.1f}{do:>12.1f}{'':>10}")
    print(f"{'passband mean |S21| (lin)':<34}{pm_r:>12.3f}{pm_o:>12.3f}{'':>10}")
    print(f"{'passband mean |S21| [dB]':<34}{pmdb_r:>12.2f}{pmdb_o:>12.2f}{'':>10}")
    print("-" * 68)

    # ------------------------------------------------------------------
    # REPORTED, NOT GATED: the rfx-vs-openEMS "method distance".
    # The Palace referee showed the argmin picks different members of a
    # transmission-zero DOUBLET per solver, so this number is largely a
    # comparator artifact. It is printed for continuity with the original
    # study and deliberately carries NO gate.
    # ------------------------------------------------------------------
    dnull_pct = abs(fnull_r - fnull_o) / fnull_o * 100.0
    dpass = abs(pm_r - pm_o)
    print(f"\nREPORTED (not gated) rfx-vs-openEMS null-freq method distance "
          f"= {dnull_pct:.1f} %")
    print(f"REPORTED (not gated) rfx-vs-openEMS passband |S21| mean abs diff "
          f"= {dpass:.3f}")
    print("  ^ NOT an accuracy gate: the Palace referee (tests/fixtures/"
          "sheen_lpf_e4/)")
    print("    showed the stopband is a DOUBLE zero and the argmin picks a "
          "different")
    print("    doublet member per solver. sides_with = openems.")
    if paper_null_ghz:
        print(f"rfx-vs-paper  null-freq distance = "
              f"{abs(fnull_r/1e9 - paper_null_ghz)/paper_null_ghz*100:.1f} %")
        print(f"oems-vs-paper null-freq distance = "
              f"{abs(fnull_o/1e9 - paper_null_ghz)/paper_null_ghz*100:.1f} %")

    # overlay figure (dB recomputed from raw |S| here, not from any producer dB)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fr / 1e9, 20 * np.log10(s21r + 1e-30), "C0-", label="|S21| rfx")
        ax.plot(fo / 1e9, 20 * np.log10(s21o + 1e-30), "C3--", label="|S21| openEMS")
        ax.plot(fr / 1e9, 20 * np.log10(s11r + 1e-30), "C0:", alpha=0.5, label="|S11| rfx")
        ax.plot(fo / 1e9, 20 * np.log10(s11o + 1e-30), "C3:", alpha=0.5, label="|S11| openEMS")
        ax.axvline(fnull_r / 1e9, color="C0", ls="-", lw=0.7, alpha=0.5)
        ax.axvline(fnull_o / 1e9, color="C3", ls="--", lw=0.7, alpha=0.5)
        ax.axvspan(null_lo / 1e9, null_hi / 1e9, color="k", alpha=0.05,
                   label="null search band")
        ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("|S| [dB]")
        ax.set_ylim(-50, 5); ax.set_xlim(fr.min() / 1e9, fr.max() / 1e9)
        ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=8)
        ax.set_title("Sheen 1990 LPF: rfx vs openEMS  (deep-null DEPTH not gated)")
        p = os.path.join(RES_DIR, "sheen_compare.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\nsaved figure {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    return _gates(R, O, fr, fnull_r, fnull_o, pm_o, do, null_lo, null_hi)


def _gates(R, O, fr, fnull_r, fnull_o, pm_o, oems_null_db, null_lo, null_hi):
    """The four configured gates. Returns True iff every one passed.

    Diagnostic-reporter gates: comparator sanity, comparison shape, the
    committed characterization, and the rfx non-passivity FOOTPRINT. None of
    them asserts rfx-vs-openEMS accuracy (see the module banner).
    """
    print("\n" + "=" * 72)
    print("GATES (diagnostic-reporter — characterization locks, NOT accuracy)")
    print("=" * 72)
    ok = True

    def gate(name, passed, detail):
        nonlocal ok
        ok = ok and bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # --- Gate A: comparator sanity (comparator-first) ------------------
    cp_o = _column_power(O)
    z0_o = float(np.median(O["re_z0"]))
    pm_o_ok = pm_o >= 0.85
    gate("A1 openEMS reference passive",
         cp_o.max() <= PASSIVITY_COLUMN_POWER_MAX,
         f"max column power {cp_o.max():.4f} <= {PASSIVITY_COLUMN_POWER_MAX}")
    gate("A2 openEMS median Re(Z0) near 50 ohm",
         40.0 < z0_o < 65.0, f"{z0_o:.2f} ohm in (40, 65)")
    gate("A3 openEMS shows a passband", pm_o_ok,
         f"passband mean |S21| {pm_o:.4f} >= 0.85")
    gate("A4 openEMS shows a deep stopband zero", oems_null_db <= -20.0,
         f"{oems_null_db:.2f} dB <= -20 dB")

    # --- Gate B: comparison shape / presence ---------------------------
    gate("B1 rfx bin count", len(R["freqs_hz"]) == COMMITTED["rfx_nbins"],
         f"{len(R['freqs_hz'])} == {COMMITTED['rfx_nbins']}")
    gate("B2 openEMS bin count", len(O["freqs_hz"]) == COMMITTED["oems_nbins"],
         f"{len(O['freqs_hz'])} == {COMMITTED['oems_nbins']}")
    in_band = ((fr >= null_lo) & (fr <= null_hi)).sum()
    gate("B3 doublet-search band populated in both", in_band >= 10,
         f"{in_band} rfx bins in {null_lo/1e9:.0f}-{null_hi/1e9:.0f} GHz")

    # --- Gate C: committed characterization lock ----------------------
    tol = COMMITTED["null_tol_pct"]
    for tag, got, want in (("rfx", fnull_r / 1e9, COMMITTED["rfx_null_ghz"]),
                           ("openEMS", fnull_o / 1e9, COMMITTED["oems_null_ghz"])):
        d = abs(got - want) / want * 100.0
        gate(f"C1 {tag} argmin null reproduces committed value", d <= tol,
             f"{got:.4f} vs committed {want:.4f} GHz ({d:.2f}% <= {tol}%)")
    sides = _referee_sides_with()
    gate("C2 Palace referee verdict on record",
         sides == COMMITTED["referee_sides_with"],
         f"sides_with = {sides!r} (expected "
         f"{COMMITTED['referee_sides_with']!r})")

    # --- Gate D: rfx non-passivity FOOTPRINT lock ----------------------
    # NOT a passivity pass. The committed rfx MSL extraction is non-physical;
    # this locks the defect so a silent change turns the case red.
    cp_r = _column_power(R)
    bad = np.where(cp_r > PASSIVITY_COLUMN_POWER_MAX)[0]
    worst = int(np.argmax(cp_r))
    print(f"\n  !! rfx leg is NOT PASSIVE — {len(bad)}/{len(cp_r)} bins exceed "
          f"column power {PASSIVITY_COLUMN_POWER_MAX}.")
    print(f"  !! worst bin {fr[worst]/1e9:.3f} GHz: |S11|={R['s11_mag'][worst]:.3f}, "
          f"|S21|={R['s21_mag'][worst]:.3f}, column power={cp_r[worst]:.2f}")
    print(f"  !! min Re(Z0) = {min(R['re_z0']):.1f} ohm. This is an EXTRACTION "
          "defect, not physics:")
    print("  !! NO |S| magnitude from the rfx leg of this study is quotable.")
    n_band = int((cp_r[(fr >= null_lo) & (fr <= null_hi)]
                  > PASSIVITY_COLUMN_POWER_MAX).sum())
    print(f"  !! {n_band} of those bins fall INSIDE the "
          f"{null_lo/1e9:.0f}-{null_hi/1e9:.0f} GHz band this study reads its "
          "result from.")
    gate("D1 non-passive bin count matches committed footprint",
         len(bad) == COMMITTED["rfx_bad_bins"],
         f"{len(bad)} == {COMMITTED['rfx_bad_bins']}")
    gate("D1b in-band non-passive bin count matches committed footprint",
         n_band == COMMITTED["rfx_bad_bins_in_null_band"],
         f"{n_band} == {COMMITTED['rfx_bad_bins_in_null_band']}")
    d_worst = abs(fr[worst] / 1e9 - COMMITTED["rfx_worst_bad_ghz"])
    gate("D2 worst non-passive bin at the committed frequency",
         d_worst <= 0.05,
         f"{fr[worst]/1e9:.3f} GHz vs committed "
         f"{COMMITTED['rfx_worst_bad_ghz']:.3f} GHz")
    d_cp = abs(cp_r[worst] - COMMITTED["rfx_worst_column_power"]) / \
        COMMITTED["rfx_worst_column_power"] * 100.0
    gate("D3 worst column power matches committed value", d_cp <= 5.0,
         f"{cp_r[worst]:.2f} vs committed "
         f"{COMMITTED['rfx_worst_column_power']:.2f} ({d_cp:.2f}% <= 5%)")

    print("\n" + ("ALL GATES PASSED — committed characterization reproduced."
                  if ok else "SOME CHECKS FAILED"))
    print("Scope: stopband STRUCTURE characterization only. This case makes NO "
          "rfx accuracy claim\nand its rfx |S| magnitudes are non-physical "
          "above ~16 GHz (see gate D).")
    return ok


def _referee_sides_with():
    """Read the Palace-FEM referee's recorded three-solver verdict."""
    try:
        with open(REFEREE_JSON) as fp:
            return json.load(fp)["referee"]["sides_with"]
    except (OSError, KeyError, ValueError) as exc:
        print(f"  (referee fixture unreadable: {exc})")
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Sheen 1990 microstrip LPF cross-solver study "
                    "(diagnostic-reporter).")
    # `compare` is the default so a bare `python 07_sheen_lpf.py` (how the
    # crossval runner invokes every case) gates the committed characterization
    # instead of dying in argparse.
    ap.add_argument("mode", nargs="?", default="compare",
                    choices=["tutorial", "openems", "rfx", "compare"])
    ap.add_argument("--dx-um", type=float, default=200.0)
    ap.add_argument("--num-periods", type=float, default=20.0)
    ap.add_argument("--n-freqs", type=int, default=120)
    ap.add_argument("--fmax-ghz", type=float, default=F_MAX / 1e9)
    ap.add_argument("--null-lo-ghz", type=float, default=5.0)
    ap.add_argument("--null-hi-ghz", type=float, default=15.0)
    ap.add_argument("--pass-lo-ghz", type=float, default=0.5)
    ap.add_argument("--pass-hi-ghz", type=float, default=3.0)
    ap.add_argument("--paper-null-ghz", type=float, default=0.0)
    a = ap.parse_args()
    # Solver-producing modes write a result leg and are not themselves gated;
    # only `compare` evaluates the configured gates and can return 1.
    if a.mode == "tutorial":
        run_openems_tutorial()
        return 0
    if a.mode == "openems":
        run_openems(a.fmax_ghz * 1e9)
        return 0
    if a.mode == "rfx":
        run_rfx(a.dx_um * 1e-6, a.num_periods, a.n_freqs)
        return 0
    all_ok = compare(a.null_lo_ghz * 1e9, a.null_hi_ghz * 1e9,
                     a.pass_lo_ghz * 1e9, a.pass_hi_ghz * 1e9,
                     a.paper_null_ghz or None)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

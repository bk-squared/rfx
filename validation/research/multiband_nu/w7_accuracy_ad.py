"""W7 — accuracy and autodiff on band-builder meshes.

Pre-declaration (binding, every window frozen there before this file was
written): docs/design_notes/20260907_nu_band_accuracy_ad_predeclaration.md.
This instrument is committed BEFORE its first measurement run.

Arms
----
A1  z-stratified PEC cavity (core 4.3 | thin 3.0 | core 4.3 | air), LSE/Ey
    family, m = 1, p = 5. Ladder s in {0.5, 1, 2} on three profiles (UC
    uniform df, MB ``make_band_profile``, AZ ``_make_dz_profile``) and two
    eps rules ("dual": instrument-assembled dual-cell average, GATED;
    "production": the ``Simulation`` rasterizer's own column, REPORTED).
    Oracle: 1-D transfer matrix (note 2.1) with its two self-checks (i)/(i').
    Model: exact discrete stratified eigenproblem on the rfx operators (note
    3.1). Gates G1/G2/G3, A1-O(ii), A1-F1, A1-F2 (note 3.2).
A2  in-plane TM110 air cavity, two fine bands per axis, caps 1.3 and 1.4,
    n_steps 8000 and 12000, uniform control (note 2.2 / 3.3).
A3  TM111 with all three axes multi-band, n_steps 8000 and 12000, uniform
    control (note 2.3 / 3.4).
AD1 profile-vector gradient on the A1 MB s = 2 dz vector (W5 pattern),
AD2 material gradient d L / d eps_thin on the same mesh,
AD3 joint (dx, dy, dz) gradient on the A3 mesh,
AD4 twenty seeded builder stacks: finite forward trace and gradient,
AD5 the tied-minimum table (inside AD1/AD3 rows)   (note 2.4-2.8 / 3.5).

Usage (note section 4)::

    PYTHONPATH=. python -m validation.research.multiband_nu.w7_accuracy_ad --selfcheck --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.w7_accuracy_ad --arms a1 --scales 2,1 --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.w7_accuracy_ad --arms a1 --scales 0.5 --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.w7_accuracy_ad --arms a2,a3 --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.w7_accuracy_ad --arms ad1,ad2,ad3,ad4 --out R

``--out`` is merged per unit so a ladder can be split across calls and
resumed; every unit carries ``rfx_file``, ``git_sha``, ``git_dirty``,
``argv``, ``started_utc`` and its wallclock. One attempt per unit: a unit
already in the JSON is skipped. ``--force`` re-measures it but first moves
the earlier row to ``<key>|superseded|<started_utc>`` so both attempts stay
in the file (the note's rule for an instrument defect that forces a
re-run; say so in the note). ``--smoke`` shrinks step counts and FD cell
subsets so the code path executes in seconds; smoke rows are tagged
``smoke: true`` and are not evidence.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import scipy.linalg as sla
from scipy.optimize import brentq

import rfx
from rfx import Box, GaussianPulse, Simulation
from rfx.auto_config import _make_dz_profile
from rfx.core.yee import MaterialArrays
from rfx.harminv import harminv
from rfx.nonuniform import make_band_profile, make_nonuniform_grid, run_nonuniform

from .analytic_dispersion import C0, operator_eigenvalues
from .harness import build_pec_fixture

# ---------------------------------------------------------------------------
# A1 fixture (note 2.1) — every length in metres
# ---------------------------------------------------------------------------
DF0 = 0.5e-3                     # fine z cell at s = 1
DC0 = 0.7e-3                     # coarse z cell at s = 1 (= 1.4 df)
DXY0 = 0.25e-3                   # transverse cell at s = 1 (= df / 2)
R_CAP = 1.4
T_CORE, T_THIN, T_AIR = 14e-3, 2e-3, 14e-3
A1_EDGES = [0.0, T_CORE, T_CORE + T_THIN, 2 * T_CORE + T_THIN,
            2 * T_CORE + T_THIN + T_AIR]
A1_EPS = [4.3, 3.0, 4.3, 1.0]
L_Z = A1_EDGES[-1]                                        # 44 mm
A1_A_X = 30e-3
A1_B_Y = 3e-3
A1_M_X = 1
A1_P_Z = 5
F_TRUE_DECLARED = 10_561_719_600.896                     # Hz, note 2.1
A1_SCALES = (0.5, 1.0, 2.0)
A1_ARMS = ("uc", "mb", "az")
A1_RULES = ("dual", "production")
A1_FEATS = [(0.0, T_CORE, 4.3), (T_CORE, T_CORE + T_THIN, 3.0),
            (T_CORE + T_THIN, 2 * T_CORE + T_THIN, 4.3)]   # AZ input
Z_SRC = 16e-3
Z_PRB = 30e-3
T_TOTAL = 15e-9
T_TRUNC = 10e-9
SIGMA_T = 200e-12
HARMINV_BAND = (7e9, 14e9)
BAND = (9.6e9, 11.5e9)
MATCH_GUARD = 0.03
E_FLOOR_HZ = 0.3e6
INVARIANCE_HZ = 0.1e6            # = E_FLOOR / 3
H1 = DF0                         # matched finest cell for A1-F2
# frozen windows (note 3.2)
ORACLE_REL = 1e-12
SELFCHECK_REL = 1e-7
G1_Z_FRACTION_MIN = 0.80
G2_GRADING_SHARE_MIN = 0.20
G3_MODEL_RESIDUAL_HZ = 0.15e6
P_UC_WINDOW = (1.8, 2.2)
P_MB_MIN = 1.8
P_MB_ANOMALY = 2.4
RHO_MAX = 2.0
# ---------------------------------------------------------------------------
# A2 / A3 fixtures (note 2.2 / 2.3)
# ---------------------------------------------------------------------------
A2_X_EDGES = [0.0, 10e-3, 12e-3, 26e-3, 28e-3, 40e-3]
A2_Y_EDGES = [0.0, 8e-3, 10e-3, 22e-3, 24e-3, 34e-3]
A2_SIZES = [1e-3, 0.25e-3, 1e-3, 0.25e-3, 1e-3]
A2_PROT = [False, True, False, True, False]
A2_CAPS = (1.3, 1.4)
A2_NZ = 10
A2_N_STEPS = (8000, 12000)
A2_ERR_MAX = 0.10e-2
A2_DIFF_MAX = 0.05e-2
A2_INV_MAX = 0.017e-2
A3_EDGES = {"x": [0.0, 19e-3, 21e-3, 40e-3],
            "y": [0.0, 16.5e-3, 18.5e-3, 35e-3],
            "z": [0.0, 17e-3, 19e-3, 36e-3]}
A3_SIZES = [1e-3, 0.25e-3, 1e-3]
A3_PROT = [False, True, False]
A3_CAP = 1.4
A3_N_STEPS = (8000, 12000)
A3_ERR_MAX = 0.20e-2
A3_DIFF_MAX = 0.10e-2
A3_INV_MAX = 0.033e-2
BOUNDARY_CELL = 1e-3
SEP_ABS_HZ = 0.5e9               # committed separation gate
SEP_RATIO = 3.0
ANTI_VACUITY = 2.0
# ---------------------------------------------------------------------------
# AD (note 2.4-2.8 / 3.5)
# ---------------------------------------------------------------------------
FD_REL_H = 1e-3
DOMINANT_FRAC = 0.05
AD1_TOL = 0.15
AD2_TOL = 0.05
AD2_H = 0.015
AD3_TOL = 0.15
AD_N_STEPS = 120
AD1_DOMAIN_XY = (6e-3, 3e-3)
AD1_DX = 0.5e-3
AD1_SRC = (6, 3, 12)
AD1_PRB = (6, 3, 10)
AD3_SRC = (25, 23, 23)
AD3_PRB = (28, 23, 23)
AD4_SEED = 20260907
AD4_N_STACKS = 20
AD4_N_STEPS = 20
AD4_DOMAIN_XY = (3e-3, 3e-3)
AD4_DX = 1e-3
AD5_TIE_REL = 0.0                # exact f32 equality with the axis minimum
# ---------------------------------------------------------------------------
# Second pass (review of the first Results; note "Second pass" section).
# Declared before any second-pass measurement; the first-pass constants above
# are untouched and the first-pass rows stay in the JSON under their keys.
# ---------------------------------------------------------------------------
A2_N_STEPS_EXT = (24000, 48000)  # run-length extension: graded 13.8 / 27.6 ns (dt 5.75e-13)
A3_N_STEPS_EXT = (24000, 48000)  # graded 11.4 / 22.9 ns (dt 4.77e-13); uniform 45.8 / 91.5 ns
AD3B_FD_REL_HS = (1e-3, 3e-3, 1e-2, 3e-2)   # FD steps recorded (1e-3 reproduces attempt 1)
AD3B_FD_REL_H_GATE = 1e-2                   # the gated step of the second attempt
AD_REF_FLOOR_QUANTA = 50.0                  # |g_fd| * 2h / ulp(loss0) below this: reference unresolved
OTHER_FAMILY_BAND = (9e9, 12.5e9)           # where the non-excited families are tabulated
# ---------------------------------------------------------------------------
# Numbers quoted in the note that --selfcheck must reproduce (note 3.1)
# ---------------------------------------------------------------------------
A1_MODEL_TABLE_MHZ = {           # (arm, s): (dual, production)   note 1b
    ("uc", 0.5): (-3.9969, +3.9910), ("uc", 1.0): (-16.0053, +0.3411),
    ("uc", 2.0): (-64.3028, -29.4613),
    ("mb", 0.5): (-7.5050, -18.3505), ("mb", 1.0): (-30.0973, -77.9318),
    ("mb", 2.0): (-121.6174, -216.6348),
    ("az", 0.5): (-7.3851, -12.5552), ("az", 1.0): (-22.7528, -14.9238),
    ("az", 2.0): (-24.3055, -16.5254),
}
A1_MODEL_ORDERS = {"dual_uc": 2.004, "dual_mb": 2.009,
                   "production_uc": 1.442, "production_mb": 1.781}
A1_MODEL_RHO = 1.883
A1_MODEL_RHO_PER_SCALE = {0.5: 1.878, 1.0: 1.881, 2.0: 1.891}
A1_MODEL_G1 = {"uc": 0.964, "mb": 0.980}
A1_MODEL_G2_RANGE = (0.467, 0.471)
A1_NEIGHBOURS_HZ = (8.708377e9, 12.275836e9)
A1_ISOLATION = 0.162
A1_INTERFACE_TABLE = {           # note 1a, eps at the nodes z = 14/16/30 mm
    ("uc", 0.5): (3.0, 4.3, 1.0), ("uc", 1.0): (3.0, 4.3, 1.0),
    ("uc", 2.0): (3.0, 4.3, 1.0),
    ("mb", 0.5): (4.3, 3.0, 4.3), ("mb", 1.0): (3.0, 4.3, 4.3),
    ("mb", 2.0): (3.0, 4.3, 4.3),
    ("az", 0.5): (4.3, 3.0, 4.3), ("az", 1.0): (3.0, 4.3, 1.0),
    ("az", 2.0): (3.0, 4.3, 1.0),
}
A1_NZ = {"uc": {0.5: 176, 1.0: 88, 2.0: 44}, "mb": {0.5: 128, 1.0: 64, 2.0: 32},
         "az": {0.5: 133, 1.0: 75, 2.0: 67}}
A1_AZ_DZ_MIN_UM = {0.5: 111.1, 1.0: 166.7, 2.0: 166.7}
A2_MODEL_ERR_PCT = {"1.3": -0.02146, "1.4": -0.02766, "uniform": -0.01142}
A2_F110_HZ = 5.786173e9
A2_PROFILE_N = {("1.3", "x"): 63, ("1.3", "y"): 57, ("1.4", "x"): 60, ("1.4", "y"): 54}
A2_PROFILE_MAXR = {("1.3", "x"): 1.286476, ("1.3", "y"): 1.286476,
                   ("1.4", "x"): 1.316978, ("1.4", "y"): 1.316357}
A3_MODEL_ERR_PCT = {"graded": -0.03488, "uniform": -0.00098}
A3_MODEL_TERMS_PCT = {"e_x": -0.0088, "e_y": -0.0144, "e_z": -0.0136, "e_t": +0.0019}
A3_F111_HZ = 7.051389e9
A3_PROFILE_N = {"x": 50, "y": 46, "z": 46}
A3_PROFILE_MAXR = {"x": 1.318010, "y": 1.308969, "z": 1.317812}
AD4_NZ_RANGE = (6, 580)
AD4_DZ_MIN_UM = 1.847
AD4_LAYER_COUNTS = [5, 4, 5, 6, 1, 5, 4, 5, 6, 3, 3, 1, 5, 6, 3, 3, 2, 1, 5, 3]
AD1_EPS_PATTERN = [4.3] * 10 + [3.0] * 2 + [4.3] * 11 + [1.0] * 10    # 33 nodes

CELL_TOL = 1e-12                 # m, I1/I3
DEFAULT_OUT = "validation/research/multiband_nu/results/w7_accuracy_ad.json"


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


RESULTS_DIR = "validation/research/multiband_nu/results/"


def _git_modified_tracked() -> list[str]:
    """Tracked files that differ from HEAD (porcelain, untracked excluded)."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
        return [ln[3:].strip() for ln in out.splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, OSError):
        return ["<git unavailable>"]


def _git_dirty() -> bool:
    """True when a TRACKED file outside the instrument's own results directory
    differs from HEAD. Second pass: the first-pass flag was ``git status
    --porcelain != ''``, which counts untracked files, so the results JSON
    being written marked every row dirty and the flag carried no provenance.
    The results JSON is tracked and rewritten after every unit, so it is
    excluded here and the full modified list is recorded alongside
    (``git_modified_tracked``); untracked files are counted in
    ``git_untracked``."""
    return any(not p.startswith(RESULTS_DIR) for p in _git_modified_tracked())


def _git_untracked() -> int:
    try:
        out = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True)
        return len([ln for ln in out.splitlines() if ln.strip()])
    except (subprocess.CalledProcessError, OSError):
        return -1


def provenance() -> dict:
    return {"rfx_file": rfx.__file__, "git_sha": _git_sha(), "git_dirty": _git_dirty(),
            "git_modified_tracked": _git_modified_tracked(), "git_untracked": _git_untracked(),
            "argv": sys.argv[1:], "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Analytic oracle: 1-D transfer matrix, LSE / Ey family (note 2.1)
# ---------------------------------------------------------------------------
def lse_det(f: float, kx2: float, edges, eps) -> float:
    """psi(L) for psi(0) = 0, psi'(0) = 1 propagated through the stack.

    psi'' + (eps_i k0^2 - kx^2) psi = 0 per layer, psi and psi' continuous
    (Ey and Hx tangential, mu = 1), PEC at both ends. The 2 x 2 propagator
    is entire in q^2, so evanescent layers are handled by the complex sqrt
    and psi(L) is real.
    """
    k0 = 2 * np.pi * f / C0
    psi, dpsi = 0.0 + 0j, 1.0 + 0j
    for i in range(len(eps)):
        d = edges[i + 1] - edges[i]
        q = np.sqrt(eps[i] * k0 ** 2 - kx2 + 0j)
        c = np.cos(q * d)
        s = np.sin(q * d) / q if abs(q) > 1e-30 else d
        psi, dpsi = c * psi + s * dpsi, -q * q * s * psi + c * dpsi
    return float(psi.real)


def lse_field(f: float, kx2: float, edges, eps, z: np.ndarray) -> np.ndarray:
    """psi(z) of the LSE mode at f, normalized to max |psi| = 1."""
    k0 = 2 * np.pi * f / C0
    out = np.zeros_like(z, dtype=float)
    psi, dpsi = 0.0 + 0j, 1.0 + 0j
    for i in range(len(eps)):
        d = edges[i + 1] - edges[i]
        q = np.sqrt(eps[i] * k0 ** 2 - kx2 + 0j)
        sel = (z >= edges[i] - 1e-15) & (z <= edges[i + 1] + 1e-15)
        zz = z[sel] - edges[i]
        out[sel] = (np.cos(q * zz) * psi + np.sin(q * zz) / q * dpsi).real
        c, s = np.cos(q * d), np.sin(q * d) / q
        psi, dpsi = c * psi + s * dpsi, -q * q * s * psi + c * dpsi
    return out / np.max(np.abs(out))


def lse_roots(kx2: float, edges, eps, f_lo: float, f_hi: float,
              n_grid: int = 20000) -> np.ndarray:
    """Every root of lse_det in [f_lo, f_hi]: sign scan + Brent."""
    fs = np.linspace(f_lo, f_hi, n_grid)
    vals = np.array([lse_det(f, kx2, edges, eps) for f in fs])
    roots = []
    for i in range(len(fs) - 1):
        if vals[i] == 0.0:
            roots.append(fs[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(lse_det, fs[i], fs[i + 1], args=(kx2, edges, eps),
                                xtol=1e-6, rtol=1e-15, maxiter=200))
    return np.array(roots)


def lsm_det(f: float, kx2: float, edges, eps) -> float:
    """(phi'/eps)(L) for phi(0) = 1, phi'(0) = 0 propagated through the stack.

    The Hy ("LSM") family of the same n = 0 sector: phi'' + (eps_i k0^2 -
    kx^2) phi = 0 per layer, phi and phi'/eps continuous (Hy and Ex
    tangential), phi' = 0 at both PEC walls. Not excited by an Ey source
    (the two n = 0 polarisations decouple); tabulated so the note's
    isolation statement names every eigenmode, not only the excited ones.
    """
    k0 = 2 * np.pi * f / C0
    phi, u = 1.0 + 0j, 0.0 + 0j          # u = phi' / eps, continuous
    for i in range(len(eps)):
        d = edges[i + 1] - edges[i]
        q = np.sqrt(eps[i] * k0 ** 2 - kx2 + 0j)
        c = np.cos(q * d)
        s = np.sin(q * d) / q if abs(q) > 1e-30 else d
        dphi = eps[i] * u
        phi, dphi = c * phi + s * dphi, -q * q * s * phi + c * dphi
        u = dphi / eps[i]
    return float(u.real)


def family_roots(det, kx2: float, edges, eps, f_lo: float, f_hi: float,
                 n_grid: int = 20000) -> np.ndarray:
    """Every root of ``det`` in [f_lo, f_hi]: sign scan + Brent (lse_roots
    generalised to either family)."""
    fs = np.linspace(f_lo, f_hi, n_grid)
    vals = np.array([det(f, kx2, edges, eps) for f in fs])
    roots = []
    for i in range(len(fs) - 1):
        if vals[i] == 0.0:
            roots.append(fs[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(det, fs[i], fs[i + 1], args=(kx2, edges, eps),
                                xtol=1e-6, rtol=1e-15, maxiter=200))
    return np.array(roots)


def other_families(f_true: float) -> dict:
    """Every eigenmode of the A1 box in OTHER_FAMILY_BAND that the fixture
    admits but the source does not excite: Ey with m = 2, 3, 4, 6 (killed by
    the equal-sign pair at a/3, 2a/3: sin(m pi/3) + sin(2m pi/3) = 0) and Hy
    (LSM) with m = 0..4 (not excited by an Ey source in the n = 0 sector).
    Reported, no gate: the isolation figure of note 2.1 is conditional on
    these two suppressions, and this table says by how much."""
    lo, hi = OTHER_FAMILY_BAND
    rows = []
    for m in (2, 3, 4, 6):
        amp = float(np.sin(m * np.pi / 3) + np.sin(2 * m * np.pi / 3))
        for r in family_roots(lse_det, (m * np.pi / A1_A_X) ** 2, A1_EDGES, A1_EPS, lo, hi):
            rows.append({"family": "Ey", "m": m, "f_hz": float(r), "rel_to_f_true": float((r - f_true) / f_true),
                         "source_pair_amplitude": amp})
    for m in range(0, 5):
        for r in family_roots(lsm_det, (m * np.pi / A1_A_X) ** 2, A1_EDGES, A1_EPS, lo, hi):
            rows.append({"family": "Hy", "m": m, "f_hz": float(r), "rel_to_f_true": float((r - f_true) / f_true),
                         "source_pair_amplitude": 0.0})
    rows.sort(key=lambda d: abs(d["rel_to_f_true"]))
    return {"band_hz": list(OTHER_FAMILY_BAND), "rows": rows,
            "nearest_any_family": (rows[0] if rows else None),
            "max_source_pair_amplitude_suppressed": max((abs(d["source_pair_amplitude"]) for d in rows), default=0.0)}


def root_near(fn, f_guess: float, rel_span: float = 0.03, n: int = 400) -> float:
    """The root of fn nearest f_guess inside +-rel_span (sign scan + Brent)."""
    fs = np.linspace(f_guess * (1 - rel_span), f_guess * (1 + rel_span), n)
    v = np.array([fn(f) for f in fs])
    best = None
    for i in range(n - 1):
        if v[i] * v[i + 1] < 0:
            r = brentq(fn, fs[i], fs[i + 1], xtol=1e-6, rtol=1e-15, maxiter=200)
            if best is None or abs(r - f_guess) < abs(best - f_guess):
                best = r
    if best is None:
        raise RuntimeError(f"no root of the determinant within {rel_span:.0%} of {f_guess}")
    return best


def closed_form(m: int, p: int, a: float, L: float, eps: float = 1.0) -> float:
    return (C0 / 2) * np.sqrt((m / a) ** 2 + (p / L) ** 2) / np.sqrt(eps)


def a1_f_true() -> float:
    kx2 = (A1_M_X * np.pi / A1_A_X) ** 2
    return root_near(lambda f: lse_det(f, kx2, A1_EDGES, A1_EPS), F_TRUE_DECLARED)


def oracle_selfcheck() -> dict:
    """Frozen checks (i) and (i') of note 2.1, plus the declared mode data.

    (i): all eps_i = 1 reproduces f_mp = (c0/2) sqrt((m/a)^2 + (p/L)^2) for
    m in {1, 5}, every root in 3-16 GHz, at a = 30 mm (the fixture) and at
    a = 60 mm (where the m = 5 family has in-band roots, so the check is
    not vacuous for m = 5). (i'): all eps_i = 4.3 reproduces the closed
    form / sqrt(4.3), m = 1, 2-8 GHz, both a's.
    """
    out = {"i_worst_rel": 0.0, "i_roots": 0, "ip_worst_rel": 0.0, "ip_roots": 0,
           "i_roots_by_case": {}}
    for a in (A1_A_X, 60e-3):
        for m in (1, 5):
            kx2 = (m * np.pi / a) ** 2
            roots = lse_roots(kx2, A1_EDGES, [1.0] * 4, 3e9, 16e9)
            out["i_roots_by_case"][f"a={a*1e3:.0f}mm,m={m}"] = int(len(roots))
            for r in roots:
                p = int(round(np.sqrt(max((2 * np.pi * r / C0) ** 2 - kx2, 0.0)) * L_Z / np.pi))
                f_cf = closed_form(m, p, a, L_Z)
                out["i_worst_rel"] = max(out["i_worst_rel"], abs(r - f_cf) / f_cf)
                out["i_roots"] += 1
        kx2 = (np.pi / a) ** 2
        for r in lse_roots(kx2, A1_EDGES, [4.3] * 4, 2e9, 8e9):
            p = int(round(np.sqrt(max(4.3 * (2 * np.pi * r / C0) ** 2 - kx2, 0.0)) * L_Z / np.pi))
            f_cf = closed_form(1, p, a, L_Z, 4.3)
            out["ip_worst_rel"] = max(out["ip_worst_rel"], abs(r - f_cf) / f_cf)
            out["ip_roots"] += 1
    out["i_pass"] = bool(out["i_worst_rel"] <= ORACLE_REL and out["i_roots"] >= 10
                         and out["i_roots_by_case"]["a=60mm,m=5"] >= 1)
    out["ip_pass"] = bool(out["ip_worst_rel"] <= ORACLE_REL and out["ip_roots"] >= 3)
    # (i''), second pass: the Hy / LSM determinant reproduces the same closed
    # form (p >= 0 allowed) for all eps_i = 1, m = 1, 3-16 GHz, a = 30 mm.
    out["ipp_worst_rel"], out["ipp_roots"] = 0.0, 0
    kx2 = (np.pi / A1_A_X) ** 2
    for r in family_roots(lsm_det, kx2, A1_EDGES, [1.0] * 4, 3e9, 16e9):
        p = int(round(np.sqrt(max((2 * np.pi * r / C0) ** 2 - kx2, 0.0)) * L_Z / np.pi))
        f_cf = closed_form(1, p, A1_A_X, L_Z)
        out["ipp_worst_rel"] = max(out["ipp_worst_rel"], abs(r - f_cf) / f_cf)
        out["ipp_roots"] += 1
    out["ipp_pass"] = bool(out["ipp_worst_rel"] <= ORACLE_REL and out["ipp_roots"] >= 5)
    # the declared mode and its isolation from EVERY visible family
    f_true = a1_f_true()
    kx2 = (np.pi / A1_A_X) ** 2
    spec = []
    for m in (1, 5, 7):
        for r in lse_roots((m * np.pi / A1_A_X) ** 2, A1_EDGES, A1_EPS, 5e9, 14e9):
            spec.append((float(r), m))
    spec.sort()
    others = [r for r, m in spec if abs(r - f_true) > 1.0]
    iso = min(abs(r - f_true) / f_true for r in others)
    zg = np.linspace(0, L_Z, 8001)
    pg = lse_field(f_true, kx2, A1_EDGES, A1_EPS, zg)
    p_half_waves = int(np.sum(np.diff(np.sign(pg[1:-1])) != 0)) + 1
    psi = lse_field(f_true, kx2, A1_EDGES, A1_EPS, np.array([14e-3, Z_SRC, Z_PRB]))
    below = max([r for r in others if r < f_true])
    above = min([r for r in others if r > f_true])
    out.update({
        "f_true_hz": f_true,
        "f_true_declared_hz": F_TRUE_DECLARED,
        "f_true_matches_declared": bool(abs(f_true - F_TRUE_DECLARED) <= 1e-2),
        "p_half_waves": p_half_waves,
        "p_matches_declared": bool(p_half_waves == A1_P_Z),
        "neighbours_hz": [below, above],
        "neighbours_match_declared": bool(
            abs(below - A1_NEIGHBOURS_HZ[0]) <= 1e-6 * A1_NEIGHBOURS_HZ[0]
            and abs(above - A1_NEIGHBOURS_HZ[1]) <= 1e-6 * A1_NEIGHBOURS_HZ[1]),
        "isolation": float(iso),
        "isolation_matches_declared": bool(abs(iso - A1_ISOLATION) <= 1e-3),
        "psi_at_14_16_30_mm": psi.tolist(),
        "spectrum_5_14_ghz": spec,
        "other_families": other_families(f_true),
    })
    return out


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def a1_uc(s: float) -> np.ndarray:
    df = DF0 * s
    n = int(round(L_Z / df))
    assert abs(n * df - L_Z) < CELL_TOL
    return np.full(n, df, np.float64)


def a1_mb(s: float) -> np.ndarray:
    df, dc = DF0 * s, DC0 * s
    return np.asarray(make_band_profile(A1_EDGES, [dc, df, dc, dc],
                                        protected=[False, True, False, False],
                                        max_ratio=R_CAP), np.float64)


def a1_mb_declared(s: float) -> np.ndarray:
    df, dc = DF0 * s, DC0 * s
    n_c = int(round(T_CORE / dc))
    n_f = int(round(T_THIN / df))
    n_a = int(round(T_AIR / dc))
    return np.asarray([dc] * n_c + [df] * n_f + [dc] * n_c + [dc] * n_a, np.float64)


def a1_az(s: float) -> np.ndarray:
    return np.asarray(_make_dz_profile(A1_FEATS, L_Z, DC0 * s), np.float64)


def a1_profile(arm: str, s: float) -> np.ndarray:
    return {"uc": a1_uc, "mb": a1_mb, "az": a1_az}[arm](s)


def a2_profiles(cap: float):
    px = make_band_profile(A2_X_EDGES, A2_SIZES, protected=A2_PROT, max_ratio=cap,
                           boundary_cell=BOUNDARY_CELL)
    py = make_band_profile(A2_Y_EDGES, A2_SIZES, protected=A2_PROT, max_ratio=cap,
                           boundary_cell=BOUNDARY_CELL)
    pz = np.full(A2_NZ, BOUNDARY_CELL)
    return (np.asarray(px, np.float64), np.asarray(py, np.float64), pz)


def a3_profiles():
    out = {}
    for ax in "xyz":
        out[ax] = np.asarray(make_band_profile(A3_EDGES[ax], A3_SIZES, protected=A3_PROT,
                                               max_ratio=A3_CAP, boundary_cell=BOUNDARY_CELL),
                             np.float64)
    return out["x"], out["y"], out["z"]


def profile_report(prof: np.ndarray, edges) -> dict:
    p = np.asarray(prof, np.float64)
    nodes = np.concatenate([[0.0], np.cumsum(p)])
    rr = np.maximum(p[1:] / p[:-1], p[:-1] / p[1:]) if len(p) > 1 else np.array([1.0])
    return {"n": int(len(p)), "sum_m": float(p.sum()),
            "max_ratio": float(rr.max()), "n_pairs_over_cap": int(np.sum(rr > R_CAP + 1e-9)),
            "d_min_m": float(p.min()), "d_max_m": float(p.max()),
            "interface_err_m": float(max(np.min(np.abs(nodes - e)) for e in edges)),
            "cells_m": p.tolist()}


# ---------------------------------------------------------------------------
# Exact discrete stratified model (note 3.1)
# ---------------------------------------------------------------------------
def cell_eps(prof: np.ndarray, edges, eps) -> np.ndarray:
    """eps of every CELL by its midpoint (no cell straddles an interface)."""
    zn = np.concatenate([[0.0], np.cumsum(prof)])
    out = np.empty(len(prof))
    for k in range(len(prof)):
        zc = 0.5 * (zn[k] + zn[k + 1])
        hit = [eps[i] for i in range(len(eps)) if edges[i] <= zc < edges[i + 1]]
        out[k] = hit[0] if hit else eps[-1]
    return out


def dual_eps_nodes(prof: np.ndarray, edges=A1_EDGES, eps=A1_EPS) -> np.ndarray:
    """Dual-cell average at every interior node (note 1b); end nodes take
    their cell's eps (they are PEC for Ey anyway). Length N + 1."""
    ce = cell_eps(prof, edges, eps)
    n = len(prof)
    out = np.ones(n + 1)
    out[0] = ce[0]
    out[n] = ce[-1]
    for k in range(1, n):
        out[k] = (ce[k - 1] * prof[k - 1] + ce[k] * prof[k]) / (prof[k - 1] + prof[k])
    return out


def halfopen_eps_nodes(prof: np.ndarray, edges, eps) -> np.ndarray:
    """eps at every node by the half-open [lo, hi) rule on the node
    coordinate (the AD4 convention, note 2.7). Length N + 1."""
    zn = np.concatenate([[0.0], np.cumsum(prof)])
    out = np.empty(len(zn))
    for k, z in enumerate(zn):
        hit = [eps[i] for i in range(len(eps)) if edges[i] <= z < edges[i + 1]]
        out[k] = hit[0] if hit else eps[-1]
    return out


def f32_values(arr) -> np.ndarray:
    """The float64 image of the float32 values the solver actually sees."""
    return np.asarray(np.asarray(arr, np.float32), np.float64)


def discrete_lambda(prof: np.ndarray, eps_nodes: np.ndarray, mu_x: float,
                    n_modes: int = 12) -> np.ndarray:
    """Eigenvalues lambda of (-L + mu_x DD) Z = lambda (DD Eps) Z on the
    interior z nodes 1..N-1 of the padded profile (#562 trailing duplicate,
    inv_h[-1] = 0) — the rfx operators exactly, with eps_k at node k."""
    d = np.asarray(prof, np.float64)
    n = len(d)
    dfull = np.concatenate([d, d[-1:]])
    inv_h = 1.0 / dfull
    inv_h[-1] = 0.0
    dd = np.concatenate([[dfull[0]], 0.5 * (dfull[:-1] + dfull[1:])])
    ks = np.arange(1, n)
    diag = inv_h[ks] + inv_h[ks - 1] + mu_x * dd[ks]
    off = -inv_h[ks[:-1]]
    a = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    b = np.diag(dd[ks] * np.asarray(eps_nodes)[ks])
    return np.sort(sla.eigh(a, b, eigvals_only=True))[:n_modes]


def mu_1d(prof: np.ndarray, m: int) -> float:
    return float(operator_eigenvalues(np.asarray(prof, np.float64), m)[m - 1])


def leap(lam: float, dt: float) -> float:
    """sin(w dt/2) = (c0 dt/2) sqrt(lambda)."""
    arg = C0 * dt * np.sqrt(lam) / 2.0
    if not 0.0 < arg < 1.0:
        raise ValueError(f"non-propagating / unstable argument {arg}")
    return float(np.arcsin(arg) / (np.pi * dt))


def a1_model(prof: np.ndarray, eps_nodes: np.ndarray, dx: float, dt: float,
             f_true: float | None = None) -> dict:
    """Predicted frequency and the nested (e_z, e_x, e_t) split (note 3.1)."""
    f_true = a1_f_true() if f_true is None else f_true
    n_x = int(round(A1_A_X / dx))
    mu_x = mu_1d(np.full(n_x, dx), A1_M_X)
    lam = discrete_lambda(prof, eps_nodes, mu_x)
    f_cont_mux = root_near(lambda f: lse_det(f, mu_x, A1_EDGES, A1_EPS), f_true)
    lam_target = (2 * np.pi * f_cont_mux / C0) ** 2
    lam_p = float(lam[np.argmin(np.abs(lam - lam_target))])
    f_full = leap(lam_p, dt)
    f_no_z = leap(lam_target, dt)
    f_no_xz = leap((2 * np.pi * f_true / C0) ** 2, dt)
    e_z, e_x, e_t = f_full - f_no_z, f_no_z - f_no_xz, f_no_xz - f_true
    den = abs(e_z) + abs(e_x) + abs(e_t)
    return {"f_model": f_full, "e_z": e_z, "e_x": e_x, "e_t": e_t,
            "e_total_model": f_full - f_true,
            "z_fraction": abs(e_z) / den if den else float("nan"),
            "mu_x": mu_x, "lambda": lam_p, "dt": float(dt)}


def a1_production_column(prof: np.ndarray, s: float, f_true: float | None = None,
                         interface_eps: str = "sampled") -> dict:
    """The eps column the PRODUCTION path assembles for this profile (note
    1a): ``Simulation(boundary="pec", dz_profile=prof)`` with three ``Box``
    entries filling [0, 14), [14, 16), [16, 30) mm on the A1 transverse box,
    ``_build_nonuniform_grid`` + ``_assemble_materials_nu``, the column read
    at the central (i, j). Reported exactly as assembled (float32 values)."""
    f_true = a1_f_true() if f_true is None else f_true
    dx = DXY0 * s
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=2 * f_true, domain=(A1_A_X, A1_B_Y, L_Z),
                         boundary="pec", dx=dx, dz_profile=np.asarray(prof, np.float64),
                         interface_eps=interface_eps)
        sim.add_material("w7_core", eps_r=4.3)
        sim.add_material("w7_thin", eps_r=3.0)
        sim.add(Box((0.0, 0.0, A1_EDGES[0]), (A1_A_X, A1_B_Y, A1_EDGES[1])), material="w7_core")
        sim.add(Box((0.0, 0.0, A1_EDGES[1]), (A1_A_X, A1_B_Y, A1_EDGES[2])), material="w7_thin")
        sim.add(Box((0.0, 0.0, A1_EDGES[2]), (A1_A_X, A1_B_Y, A1_EDGES[3])), material="w7_core")
        grid = sim._build_nonuniform_grid()
        mats = sim._assemble_materials_nu(grid)[0]
    if interface_eps == "dual_average":
        from rfx.runners.nonuniform import assemble_interface_eps_nu
        eps = np.asarray(assemble_interface_eps_nu(sim, grid, mats)[1], np.float64)
    else:
        eps = np.asarray(mats.eps_r, np.float64)
    col = eps[grid.nx // 2, grid.ny // 2, :]
    zn = np.concatenate([[0.0], np.cumsum(np.asarray(prof, np.float64))])
    table = {}
    for z in (A1_EDGES[1], A1_EDGES[2], A1_EDGES[3]):
        k = int(np.argmin(np.abs(zn - z)))
        table[f"{z*1e3:.0f}mm"] = {"k": k, "node_coord_repr": repr(float(zn[k])),
                                   "eps": float(col[k]), "node_err_m": float(abs(zn[k] - z))}
    return {"column": col.tolist(), "grid_shape": list(grid.shape),
            "interface_table": table,
            "interior_transversely_uniform": bool(np.all(eps[:-1, :-1, :] == col[None, None, :])),
            "warnings": sorted({str(x.message)[:200] for x in w})}


# ---------------------------------------------------------------------------
# Harminv extraction shared by the A1 units
# ---------------------------------------------------------------------------
def extract_a1(ts: np.ndarray, dt: float, n_skip: int, f_true: float) -> tuple:
    sig = ts[n_skip:]
    if len(sig) < 64:
        return float("nan"), [], False
    sig = sig - sig.mean()
    modes = sorted(harminv(sig, dt, *HARMINV_BAND), key=lambda m: m.freq)
    in_band = [m for m in modes if BAND[0] <= m.freq <= BAND[1]]
    if in_band:
        f_meas = float(min(in_band, key=lambda m: abs(m.freq - f_true)).freq)
    else:
        f_meas = float("nan")
    valid = bool(np.isfinite(f_meas) and abs(f_meas - f_true) <= MATCH_GUARD * f_true)
    lines = [(float(m.freq), float(m.Q), float(abs(m.amplitude))) for m in modes[:8]]
    return f_meas, lines, valid


def measure_a1(arm: str, s: float, rule: str, smoke: bool = False,
               f_true: float | None = None, interface_eps: str = "sampled") -> dict:
    """One (arm, scale, rule) unit of the A1 ladder."""
    assert arm in A1_ARMS and rule in A1_RULES
    f_true = a1_f_true() if f_true is None else f_true
    t_unit = time.time()
    prof = a1_profile(arm, s)
    rep = profile_report(prof, A1_EDGES)
    declared_match = None
    if arm == "mb":
        dec = a1_mb_declared(s)
        declared_match = bool(len(dec) == len(prof) and np.max(np.abs(dec - prof)) <= CELL_TOL)
    dx = DXY0 * s
    if rule == "dual":
        col64 = dual_eps_nodes(prof)
        col = f32_values(col64)
        column_info = {"rule": "dual", "column_source": "instrument dual-cell average, float32"}
    else:
        pc = a1_production_column(prof, s, f_true, interface_eps=interface_eps)
        col = np.asarray(pc["column"], np.float64)
        column_info = {"rule": "production",
                       "column_source": "Simulation._assemble_materials_nu column at (nx//2, ny//2)",
                       "interface_table": pc["interface_table"],
                       "interior_transversely_uniform": pc["interior_transversely_uniform"],
                       "assembly_warnings": pc["warnings"]}
    zn = np.concatenate([[0.0], np.cumsum(prof)])
    iface = {f"{z*1e3:.0f}mm": float(col[int(np.argmin(np.abs(zn - z)))])
             for z in (A1_EDGES[1], A1_EDGES[2], A1_EDGES[3])}
    grid, mats = build_pec_fixture(prof, (A1_A_X, A1_B_Y), dx)
    assert len(col) == grid.nz, (len(col), grid.nz)
    eps3 = np.broadcast_to(col[None, None, :], grid.shape)
    mats = mats._replace(eps_r=jnp.asarray(eps3, jnp.float32))
    dt = float(grid.dt)
    n_steps = 4000 if smoke else int(round(T_TOTAL / dt))
    t = np.arange(n_steps) * dt
    t0 = 5 * SIGMA_T
    wf = (np.exp(-((t - t0) / SIGMA_T) ** 2 / 2.0)
          * np.sin(2 * np.pi * f_true * (t - t0))).astype(np.float32)
    k_src = int(np.argmin(np.abs(zn - Z_SRC)))
    k_prb = int(np.argmin(np.abs(zn - Z_PRB)))
    src_err = float(abs(zn[k_src] - Z_SRC))
    prb_err = float(abs(zn[k_prb] - Z_PRB))
    assert src_err < CELL_TOL and prb_err < CELL_TOL, (arm, s, src_err, prb_err)
    i_a = int(round(A1_A_X / 3 / dx))
    i_b = int(round(2 * A1_A_X / 3 / dx))
    assert abs(i_a * dx - A1_A_X / 3) < CELL_TOL and abs(i_b * dx - 2 * A1_A_X / 3) < CELL_TOL
    j_c = grid.ny // 2
    t_run = time.time()
    out = run_nonuniform(grid, mats, n_steps,
                         sources=[(i_a, j_c, k_src, "ey", wf), (i_b, j_c, k_src, "ey", wf)],
                         probes=[(i_a, j_c, k_prb, "ey")])
    ts = np.asarray(out["time_series"][:, 0], np.float64)
    wall_run = time.time() - t_run
    n_skip = int((2 * t0) / dt)
    f_meas, lines, valid = extract_a1(ts, dt, n_skip, f_true)
    n_trunc = min(int(round(T_TRUNC / dt)), n_steps)
    f_meas_10, _, valid_10 = extract_a1(ts[:n_trunc], dt, n_skip, f_true)
    invariance = (abs(f_meas - f_meas_10) if (valid and valid_10) else float("nan"))
    model = a1_model(prof, col, dx, dt, f_true)
    resid = (f_meas - model["f_model"]) if valid else float("nan")
    row = {
        "arm": arm, "scale": s, "rule": rule, "smoke": bool(smoke),
        "interface_eps_rule": interface_eps,
        "profile_m": prof.tolist(), "nz": len(prof), "declared_match": declared_match,
        "profile": {k: v for k, v in rep.items() if k != "cells_m"},
        "eps_column": col.tolist(), "interface_eps": iface, **column_info,
        "dx": dx, "dt": dt, "n_steps": n_steps, "t_total_s": n_steps * dt,
        "cells": int(grid.nx * grid.ny * grid.nz), "grid_shape": list(grid.shape),
        "k_src": k_src, "k_prb": k_prb, "i_a": i_a, "i_b": i_b, "j_c": j_c,
        "n_skip": n_skip, "harminv_lines": lines,
        "f_true": f_true, "f_meas": f_meas, "valid": valid,
        "err_hz": (f_meas - f_true) if valid else float("nan"),
        "abs_err_hz": abs(f_meas - f_true) if valid else float("nan"),
        "f_meas_10ns": f_meas_10, "valid_10ns": valid_10,
        "invariance_hz": invariance,
        "invariance_pass": bool(np.isfinite(invariance) and invariance <= INVARIANCE_HZ),
        "f_model": model["f_model"], "e_z": model["e_z"], "e_x": model["e_x"],
        "e_t": model["e_t"], "e_total_model": model["e_total_model"],
        "z_fraction": model["z_fraction"], "mu_x": model["mu_x"],
        "model_residual_hz": resid,
        "g3_pass": bool(np.isfinite(resid) and abs(resid) <= G3_MODEL_RESIDUAL_HZ),
        "wallclock_run_s": wall_run, "wallclock_s": time.time() - t_unit,
        **provenance(),
    }
    return row


def a1_key(arm: str, s: float, rule: str) -> str:
    return f"{arm}|{s:g}|{rule}"


def fit_line(points):
    """LS fit of log10|e| = p log10 h + b over (h, |e|) points."""
    h = np.log10([p[0] for p in points])
    e = np.log10([p[1] for p in points])
    p, b = np.polyfit(h, e, 1)
    return float(p), float(b)


def _ladder_points(units: dict, arm: str, rule: str, require_invariance: bool):
    pts = []
    for s in A1_SCALES:
        r = units.get(a1_key(arm, s, rule))
        if r is None or not r["valid"] or r.get("smoke"):
            continue
        if require_invariance and not r["invariance_pass"]:
            continue
        if r["abs_err_hz"] < E_FLOOR_HZ:
            continue
        pts.append((DF0 * s, r["abs_err_hz"], s))
    return pts


def judge_a1(units: dict) -> dict:
    """The frozen A1 rules (note 3.2), re-evaluable on any partial ladder."""
    out = {"n_units": len(units), "units_present": sorted(units)}
    dual_rows = [r for k, r in units.items() if r["rule"] == "dual" and r["arm"] in ("uc", "mb")
                 and not r.get("smoke")]
    # G1 / G2 on UC and MB dual (model quantities, present whenever the unit ran)
    g1 = min([r["z_fraction"] for r in dual_rows], default=float("nan"))
    shares = []
    for s in A1_SCALES:
        mb = units.get(a1_key("mb", s, "dual"))
        uc = units.get(a1_key("uc", s, "dual"))
        if mb and uc and not mb.get("smoke") and not uc.get("smoke"):
            shares.append(abs(mb["e_z"] - uc["e_z"]) / abs(mb["e_total_model"]))
    g2 = min(shares, default=float("nan"))
    # G3 on every non-smoke unit that extracted a mode
    resid_dual = [abs(r["model_residual_hz"]) for r in units.values()
                  if r["rule"] == "dual" and r["valid"] and not r.get("smoke")]
    resid_prod = [abs(r["model_residual_hz"]) for r in units.values()
                  if r["rule"] == "production" and r["valid"] and not r.get("smoke")]
    g3_dual = max(resid_dual, default=float("nan"))
    g3_prod = max(resid_prod, default=float("nan"))
    inconclusive = [k for k, r in units.items() if not r.get("smoke")
                    and (not r["valid"] or not r["invariance_pass"])]
    out["fixture_gates"] = {
        "g1_min_z_fraction": g1, "g1_target": G1_Z_FRACTION_MIN,
        "g1_pass": bool(np.isfinite(g1) and g1 >= G1_Z_FRACTION_MIN),
        "g2_min_grading_share": g2, "g2_target": G2_GRADING_SHARE_MIN,
        "g2_pass": bool(np.isfinite(g2) and g2 >= G2_GRADING_SHARE_MIN),
        "g3_max_residual_dual_hz": g3_dual, "g3_max_residual_production_hz": g3_prod,
        "g3_target_hz": G3_MODEL_RESIDUAL_HZ,
        "g3_dual_pass": bool(np.isfinite(g3_dual) and g3_dual <= G3_MODEL_RESIDUAL_HZ),
        "g3_production_pass": bool(np.isfinite(g3_prod) and g3_prod <= G3_MODEL_RESIDUAL_HZ),
        "inconclusive_units": inconclusive,
    }
    # orders and rho on the dual ladders (invariance-passing points only)
    pts_uc = _ladder_points(units, "uc", "dual", True)
    pts_mb = _ladder_points(units, "mb", "dual", True)
    fit = {"uc_points": [(h, e, s) for h, e, s in pts_uc], "mb_points": [(h, e, s) for h, e, s in pts_mb]}
    p_uc = b_uc = p_mb = b_mb = None
    if len(pts_uc) >= 3:
        p_uc, b_uc = fit_line([(h, e) for h, e, _ in pts_uc])
    if len(pts_mb) >= 3:
        p_mb, b_mb = fit_line([(h, e) for h, e, _ in pts_mb])
    fit.update({"p_uc": p_uc, "b_uc": b_uc, "p_mb": p_mb, "b_mb": b_mb})
    rho = None
    if p_uc is not None and p_mb is not None:
        lh = np.log10(H1)
        rho = float(10 ** ((p_mb * lh + b_mb) - (p_uc * lh + b_uc)))
    fit["rho_at_h1"] = rho
    fit["ratio_per_scale"] = {}
    for s in A1_SCALES:
        mb = units.get(a1_key("mb", s, "dual"))
        uc = units.get(a1_key("uc", s, "dual"))
        if mb and uc and mb["valid"] and uc["valid"]:
            fit["ratio_per_scale"][f"{s:g}"] = mb["abs_err_hz"] / uc["abs_err_hz"]
    out["fit"] = fit
    gates = out["fixture_gates"]
    fixture_ok = gates["g1_pass"] and gates["g2_pass"] and gates["g3_dual_pass"]
    out["a1_o_ii"] = {"window": list(P_UC_WINDOW), "p_uc": p_uc,
                      "pass": (None if p_uc is None else bool(P_UC_WINDOW[0] <= p_uc <= P_UC_WINDOW[1]))}
    out["a1_f1"] = {"p_mb_min": P_MB_MIN, "anomaly_above": P_MB_ANOMALY, "p_mb": p_mb,
                    "fired": (None if p_mb is None else bool(p_mb < P_MB_MIN)),
                    "anomaly": (None if p_mb is None else bool(p_mb > P_MB_ANOMALY))}
    out["a1_f2"] = {"rho_max": RHO_MAX, "rho": rho,
                    "fired": (None if rho is None else bool(rho > RHO_MAX))}
    if not fixture_ok:
        failed = [g for g, k in (("G1", "g1_pass"), ("G2", "g2_pass"), ("G3", "g3_dual_pass"))
                  if not gates[k]]
        out["verdict"] = "FIXTURE-INVALID or incomplete (%s not passed)" % ", ".join(failed)
    elif p_uc is None or p_mb is None:
        out["verdict"] = "INCONCLUSIVE (fewer than 3 fit points on a dual ladder)"
    elif not out["a1_o_ii"]["pass"]:
        out["verdict"] = f"STOP: dual UC order {p_uc:.3f} outside {P_UC_WINDOW} (oracle-validity check ii)"
    else:
        out["verdict"] = (f"p_uc={p_uc:.3f} p_mb={p_mb:.3f} rho={rho:.3f} "
                          f"F1_fired={out['a1_f1']['fired']} F2_fired={out['a1_f2']['fired']}"
                          + (" ANOMALY(p_mb>2.4)" if out["a1_f1"]["anomaly"] else ""))
    # reported: production ladders, e_samp, AZ, cost
    rep = {"e_samp_meas_hz": {}, "e_samp_model_hz": {}, "production_orders": {}, "az": {}, "cost": {}}
    for arm in A1_ARMS:
        for s in A1_SCALES:
            d = units.get(a1_key(arm, s, "dual"))
            p = units.get(a1_key(arm, s, "production"))
            if d and p:
                key = f"{arm}|{s:g}"
                rep["e_samp_model_hz"][key] = p["f_model"] - d["f_model"]
                rep["e_samp_meas_hz"][key] = ((p["f_meas"] - d["f_meas"])
                                              if (p["valid"] and d["valid"]) else None)
            for r in (d, p):
                if r:
                    rep["cost"][a1_key(arm, s, r["rule"])] = {
                        "cells": r["cells"], "n_steps": r["n_steps"], "dt": r["dt"],
                        "nz": r["nz"], "dz_min_m": r["profile"]["d_min_m"],
                        "wallclock_run_s": r["wallclock_run_s"]}
    for arm in ("uc", "mb"):
        pts = _ladder_points(units, arm, "production", True)
        rep["production_orders"][arm] = (fit_line([(h, e) for h, e, _ in pts])[0]
                                         if len(pts) >= 3 else None)
    for s in A1_SCALES:
        for rule in A1_RULES:
            r = units.get(a1_key("az", s, rule))
            if r:
                rep["az"][f"{s:g}|{rule}"] = {"f_meas": r["f_meas"], "f_model": r["f_model"],
                                              "model_residual_hz": r["model_residual_hz"],
                                              "err_hz": r["err_hz"], "nz": r["nz"],
                                              "dz_min_m": r["profile"]["d_min_m"], "dt": r["dt"],
                                              "n_steps": r["n_steps"], "cells": r["cells"],
                                              "wallclock_run_s": r["wallclock_run_s"]}
    out["reported"] = rep
    return out


# ---------------------------------------------------------------------------
# A2 / A3 on the Simulation path (the committed oracles' method)
# ---------------------------------------------------------------------------
def _run_cavity_sim(px, py, pz, f_target: float, src_pos, prb_pos, n_steps: int,
                    graded: bool) -> dict:
    """Build the PEC cavity Simulation on the NU path and run it — the
    committed-oracle recipe (Ez GaussianPulse source, Ez probe,
    find_resonances over 0.6-1.5 f, nearest-to-analytic, separation gate)."""
    a, b, d = float(np.sum(px)), float(np.sum(py)), float(np.sum(pz))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if graded:
            sim = Simulation(freq_max=2 * f_target, domain=(0, 0, 0), boundary="pec",
                             dx=BOUNDARY_CELL, dz_profile=np.asarray(pz),
                             dx_profile=np.asarray(px), dy_profile=np.asarray(py))
        else:
            sim = Simulation(freq_max=2 * f_target, domain=(a, b, d), boundary="pec",
                             dx=BOUNDARY_CELL, dz_profile=np.asarray(pz))
        sim.add_source(src_pos, "ez", waveform=GaussianPulse(f0=f_target, bandwidth=0.8))
        sim.add_probe(prb_pos, "ez")
        grid = sim._build_nonuniform_grid()
        src_idx = [int(v) for v in sim._pos_to_nu_index(grid, src_pos)]
        prb_idx = [int(v) for v in sim._pos_to_nu_index(grid, prb_pos)]
        t_run = time.time()
        result = sim.run(n_steps=n_steps)
        wall = time.time() - t_run
        modes = result.find_resonances(freq_range=(0.6 * f_target, 1.5 * f_target))
    freqs = sorted(float(m.freq) for m in modes)
    by_dist = sorted(freqs, key=lambda f: abs(f - f_target))
    if by_dist:
        f_sim = by_dist[0]
        d_near = abs(f_sim - f_target)
        d_second = abs(by_dist[1] - f_target) if len(by_dist) > 1 else float("inf")
        sep_pass = bool(d_second > SEP_RATIO * d_near and d_second > SEP_ABS_HZ)
    else:
        f_sim, d_near, d_second, sep_pass = float("nan"), float("nan"), float("nan"), False
    dt = float(grid.dt)
    return {"a_m": a, "b_m": b, "d_m": d, "grid_shape": list(grid.shape),
            "cells": int(grid.nx * grid.ny * grid.nz), "dt": dt, "n_steps": n_steps,
            "t_total_s": n_steps * dt, "src_idx": src_idx, "prb_idx": prb_idx,
            "modes_ghz_q": [(float(m.freq), float(m.Q)) for m in sorted(modes, key=lambda m: m.freq)],
            "f_sim": f_sim, "d_near_hz": d_near, "d_second_hz": d_second,
            "separation_pass": sep_pass,
            "err": (f_sim - f_target) / f_target if np.isfinite(f_sim) else float("nan"),
            "warnings": sorted({str(x.message)[:240] for x in w}),
            "wallclock_run_s": wall}


def a2_model_err(px, py, dt: float, f110: float) -> float:
    lam = mu_1d(px, 1) + mu_1d(py, 1)
    return (leap(lam, dt) - f110) / f110


def measure_a2(cap: float | None, n_steps: int, smoke: bool = False) -> dict:
    """One A2 unit: cap in {1.3, 1.4} (graded) or None (uniform control)."""
    t_unit = time.time()
    if cap is None:
        a, b = A2_X_EDGES[-1], A2_Y_EDGES[-1]
        px, py = np.full(int(round(a / BOUNDARY_CELL)), BOUNDARY_CELL), np.full(int(round(b / BOUNDARY_CELL)), BOUNDARY_CELL)
        pz = np.full(A2_NZ, BOUNDARY_CELL)
    else:
        px, py, pz = a2_profiles(cap)
    a, b, d = float(px.sum()), float(py.sum()), float(pz.sum())
    f110 = closed_form(1, 1, a, b)   # (c/2) sqrt((1/a)^2 + (1/b)^2)
    f210 = (C0 / 2) * np.sqrt((2 / a) ** 2 + (1 / b) ** 2)
    f120 = (C0 / 2) * np.sqrt((1 / a) ** 2 + (2 / b) ** 2)
    n = 600 if smoke else n_steps
    r = _run_cavity_sim(px, py, pz, f110, (a / 3, b / 3, d / 2), (2 * a / 3, 2 * b / 3, d / 2),
                        n, graded=cap is not None)
    ratio_x, ratio_y = float(px.max() / px.min()), float(py.max() / py.min())
    row = {"cap": cap, "n_steps_declared": n_steps, "smoke": bool(smoke),
           "profile_x": profile_report(px, A2_X_EDGES), "profile_y": profile_report(py, A2_Y_EDGES),
           "ratio_x": ratio_x, "ratio_y": ratio_y,
           "anti_vacuity_pass": (True if cap is None else bool(ratio_x > ANTI_VACUITY and ratio_y > ANTI_VACUITY)),
           "f_analytic": f110, "f_tm210": f210, "f_tm120": f120,
           "model_err": a2_model_err(px, py, r["dt"], f110), **r,
           "wallclock_s": time.time() - t_unit, **provenance()}
    row["model_residual"] = (row["err"] - row["model_err"]) if np.isfinite(row["err"]) else float("nan")
    return row


def a2_key(cap, n_steps: int) -> str:
    return f"{'uniform' if cap is None else cap}|{n_steps}"


def judge_a2(units: dict) -> dict:
    return _judge_inplane(units, A2_CAPS, A2_N_STEPS, A2_ERR_MAX, A2_DIFF_MAX, A2_INV_MAX, a2_key)


def judge_a2_ext(units: dict) -> dict:
    """Second pass: the frozen A2 windows re-evaluated at the extended step
    pair A2_N_STEPS_EXT (same numbers, longer physical run)."""
    return _judge_inplane(units, A2_CAPS, A2_N_STEPS_EXT, A2_ERR_MAX, A2_DIFF_MAX, A2_INV_MAX, a2_key)


def leapfrog_term(row: dict) -> float:
    """e_t of a unit: the leapfrog (time-stepping) part of its error, from
    the exact extents and dt the row records — (leap(k^2, dt) - f) / f with
    k^2 the continuum eigenvalue. Mesh-independent apart from dt, so a
    graded-minus-uniform difference at unequal dt carries the CONTROL's
    e_t; the spatial part is err - e_t (reviewer finding, second pass)."""
    k2 = (np.pi / row["a_m"]) ** 2 + (np.pi / row["b_m"]) ** 2
    if row.get("graded") is not None:            # A3 rows (p = 1 along z)
        k2 += (np.pi / row["d_m"]) ** 2
    f = row["f_analytic"]
    return float((leap(k2, row["dt"]) - f) / f)


def _judge_inplane(units, caps, steps, err_max, diff_max, inv_max, keyfn) -> dict:
    out = {"err_max": err_max, "diff_max": diff_max, "inv_max": inv_max, "arms": {}}
    real = {k: r for k, r in units.items() if not r.get("smoke")}
    for cap in caps:
        for n in steps:
            g = real.get(keyfn(cap, n))
            u = real.get(keyfn(None, n))
            if g is None:
                continue
            e = {"err": g["err"], "separation_pass": g["separation_pass"],
                 "anti_vacuity_pass": g["anti_vacuity_pass"],
                 "f1_pass": bool(np.isfinite(g["err"]) and abs(g["err"]) <= err_max),
                 "err_uniform": (u["err"] if u else None),
                 "diff_pt": ((g["err"] - u["err"]) if (u and np.isfinite(g["err"]) and np.isfinite(u["err"])) else None)}
            e["f2_pass"] = (None if e["diff_pt"] is None else bool(abs(e["diff_pt"]) <= diff_max))
            # reported (second pass): split the difference into its leapfrog and spatial parts
            e["e_t"] = leapfrog_term(g)
            e["spatial_err"] = (g["err"] - e["e_t"]) if np.isfinite(g["err"]) else None
            e["model_spatial_err"] = g["model_err"] - e["e_t"]
            e["t_total_s"], e["dt"] = g["t_total_s"], g["dt"]
            if u is not None:
                e["e_t_uniform"] = leapfrog_term(u)
                e["spatial_err_uniform"] = (u["err"] - e["e_t_uniform"]) if np.isfinite(u["err"]) else None
                e["model_spatial_err_uniform"] = u["model_err"] - e["e_t_uniform"]
                e["spatial_diff_pt"] = ((e["spatial_err"] - e["spatial_err_uniform"])
                                        if (e["spatial_err"] is not None and e["spatial_err_uniform"] is not None) else None)
                e["model_spatial_diff_pt"] = e["model_spatial_err"] - e["model_spatial_err_uniform"]
                e["model_diff_pt"] = g["model_err"] - u["model_err"]
                e["t_total_s_uniform"], e["dt_uniform"] = u["t_total_s"], u["dt"]
            out["arms"][keyfn(cap, n)] = e
        g8, g12 = real.get(keyfn(cap, steps[0])), real.get(keyfn(cap, steps[1]))
        if g8 and g12 and np.isfinite(g8["f_sim"]) and np.isfinite(g12["f_sim"]):
            inv = abs(g8["f_sim"] - g12["f_sim"]) / g8["f_analytic"]
            out["arms"][f"{cap}|invariance"] = {"inv": inv, "pass": bool(inv <= inv_max)}
    verdicts = []
    for cap in caps:
        keys = [keyfn(cap, n) for n in steps]
        rows = [out["arms"].get(k) for k in keys]
        inv = out["arms"].get(f"{cap}|invariance")
        if any(r is None for r in rows) or inv is None:
            verdicts.append(f"cap {cap}: incomplete")
            continue
        if not inv["pass"] or not all(r["separation_pass"] and r["anti_vacuity_pass"] for r in rows):
            verdicts.append(f"cap {cap}: INCONCLUSIVE (extraction / separation / vacuity)")
            continue
        f1 = all(r["f1_pass"] for r in rows)
        f2 = all(r["f2_pass"] for r in rows)
        verdicts.append(f"cap {cap}: F1 {'HELD' if f1 else 'FIRED'}, F2 "
                        f"{'HELD' if f2 else ('FIRED' if f2 is not None else 'no uniform control')}")
    out["verdict"] = "; ".join(verdicts) if verdicts else "no units"
    return out


def a3_model(px, py, pz, dt: float, f111: float) -> dict:
    a, b, d = float(px.sum()), float(py.sum()), float(pz.sum())
    mu = {"x": mu_1d(px, 1), "y": mu_1d(py, 1), "z": mu_1d(pz, 1)}
    k2 = {"x": (np.pi / a) ** 2, "y": (np.pi / b) ** 2, "z": (np.pi / d) ** 2}
    f_full = leap(mu["x"] + mu["y"] + mu["z"], dt)
    f_no_z = leap(mu["x"] + mu["y"] + k2["z"], dt)
    f_no_yz = leap(mu["x"] + k2["y"] + k2["z"], dt)
    f_no_xyz = leap(k2["x"] + k2["y"] + k2["z"], dt)
    return {"model_err": (f_full - f111) / f111,
            "e_x": (f_no_yz - f_no_xyz) / f111, "e_y": (f_no_z - f_no_yz) / f111,
            "e_z": (f_full - f_no_z) / f111, "e_t": (f_no_xyz - f111) / f111}


def measure_a3(graded: bool, n_steps: int, smoke: bool = False) -> dict:
    t_unit = time.time()
    if graded:
        px, py, pz = a3_profiles()
    else:
        px = np.full(int(round(A3_EDGES["x"][-1] / BOUNDARY_CELL)), BOUNDARY_CELL)
        py = np.full(int(round(A3_EDGES["y"][-1] / BOUNDARY_CELL)), BOUNDARY_CELL)
        pz = np.full(int(round(A3_EDGES["z"][-1] / BOUNDARY_CELL)), BOUNDARY_CELL)
    a, b, d = float(px.sum()), float(py.sum()), float(pz.sum())

    def fm(m, n_, p):
        return (C0 / 2) * np.sqrt((m / a) ** 2 + (n_ / b) ** 2 + (p / d) ** 2)
    f111 = fm(1, 1, 1)
    n = 600 if smoke else n_steps
    r = _run_cavity_sim(px, py, pz, f111, (a / 3, b / 3, d / 4), (2 * a / 3, 2 * b / 3, d / 4),
                        n, graded=graded)
    ratios = {ax: float(p.max() / p.min()) for ax, p in (("x", px), ("y", py), ("z", pz))}
    row = {"graded": graded, "n_steps_declared": n_steps, "smoke": bool(smoke),
           "profile_x": profile_report(px, A3_EDGES["x"]), "profile_y": profile_report(py, A3_EDGES["y"]),
           "profile_z": profile_report(pz, A3_EDGES["z"]), "ratios": ratios,
           "anti_vacuity_pass": (True if not graded else bool(all(v > ANTI_VACUITY for v in ratios.values()))),
           "f_analytic": f111, "f_tm110": fm(1, 1, 0), "f_tm211": fm(2, 1, 1),
           **a3_model(px, py, pz, r["dt"], f111), **r,
           "wallclock_s": time.time() - t_unit, **provenance()}
    row["model_residual"] = (row["err"] - row["model_err"]) if np.isfinite(row["err"]) else float("nan")
    return row


def a3_key(graded, n_steps: int) -> str:
    return f"{'graded' if graded else 'uniform'}|{n_steps}"


def judge_a3(units: dict) -> dict:
    return _judge_inplane(units, ("graded",), A3_N_STEPS, A3_ERR_MAX, A3_DIFF_MAX, A3_INV_MAX,
                          lambda g, n: a3_key(g == "graded", n))


def judge_a3_ext(units: dict) -> dict:
    """Second pass: the frozen A3 windows at A3_N_STEPS_EXT."""
    return _judge_inplane(units, ("graded",), A3_N_STEPS_EXT, A3_ERR_MAX, A3_DIFF_MAX, A3_INV_MAX,
                          lambda g, n: a3_key(g == "graded", n))


# ---------------------------------------------------------------------------
# AD arms
# ---------------------------------------------------------------------------
def _ad_waveform(n_steps: int, dt):
    """The W5 pulse: exp(-((t - 15 dt) / (5 dt))^2)."""
    dt32 = jnp.asarray(dt, jnp.float32)
    t = jnp.arange(n_steps, dtype=jnp.float32) * dt32
    return jnp.exp(-(((t - 15.0 * dt32) / (5.0 * dt32)) ** 2)).astype(jnp.float32)


def _vacuum(shape):
    return MaterialArrays(eps_r=jnp.ones(shape, jnp.float32), mu_r=jnp.ones(shape, jnp.float32),
                          sigma=jnp.zeros(shape, jnp.float32))


def ad1_loss_factory(eps_col, n_steps: int):
    eps_col = jnp.asarray(np.asarray(eps_col, np.float64), jnp.float32)

    def loss(dz):
        grid = make_nonuniform_grid(domain_xy=AD1_DOMAIN_XY, dz_profile=dz, dx=AD1_DX, cpml_layers=0)
        shape = grid.shape
        mats = _vacuum(shape)._replace(eps_r=jnp.broadcast_to(eps_col[None, None, :], shape))
        wf = _ad_waveform(n_steps, grid.dt)
        out = run_nonuniform(grid, mats, n_steps,
                             sources=[(*AD1_SRC, "ey", wf)], probes=[(*AD1_PRB, "ey")])
        return jnp.sum(out["time_series"] ** 2)
    return loss


def ad_compare_profile(loss, d0: np.ndarray, label: str, tol: float,
                       fd_cells=None) -> dict:
    """jax.grad vs central FD (h = FD_REL_H d[k]) on a cell-size vector.

    Tied-minimum cells (AD5) are excluded from the dominant-cell comparison
    and reported one-sided (g_ad, FD+, FD-). ``fd_cells`` restricts the FD
    loop (smoke only)."""
    d0 = np.asarray(d0, np.float64)
    loss_j = jax.jit(loss)
    grad_j = jax.jit(jax.grad(loss))
    t0 = time.time()
    x0 = jnp.asarray(d0)
    g_ad = np.asarray(grad_j(x0), np.float64)
    loss0 = float(loss_j(x0))
    d32 = f32_values(d0)
    tied = np.isclose(d32, d32.min(), rtol=AD5_TIE_REL, atol=0.0)
    n = len(d0)
    cells = list(range(n)) if fd_cells is None else list(fd_cells)
    g_fd = np.full(n, np.nan)
    fd_plus = np.full(n, np.nan)
    fd_minus = np.full(n, np.nan)
    for k in cells:
        h = FD_REL_H * d0[k]
        dp = d0.copy()
        dp[k] += h
        dm = d0.copy()
        dm[k] -= h
        lp = float(loss_j(jnp.asarray(dp)))
        lm = float(loss_j(jnp.asarray(dm)))
        g_fd[k] = (lp - lm) / (2 * h)
        fd_plus[k] = (lp - loss0) / h
        fd_minus[k] = (loss0 - lm) / h
    have = np.isfinite(g_fd)
    free = have & ~tied
    if free.any():
        gmax = np.abs(g_fd[free]).max()
        dominant = free & (np.abs(g_fd) > DOMINANT_FRAC * gmax)
    else:
        dominant = np.zeros(n, bool)
    rel = np.abs(g_ad - g_fd) / np.maximum(np.abs(g_fd), 1e-300)
    worst = float(rel[dominant].max()) if dominant.any() else float("nan")
    signs_ok = bool(np.all(np.sign(g_ad[dominant]) == np.sign(g_fd[dominant]))) if dominant.any() else None
    return {
        "label": label, "n_cells": n, "n_fd_cells": len(cells), "loss0": loss0,
        "loss_jitted": True, "fd_rel_h": FD_REL_H, "dominant_frac": DOMINANT_FRAC, "tol": tol,
        "all_finite": bool(np.all(np.isfinite(g_ad))),
        "n_tied_min": int(tied.sum()), "tied_cells": np.nonzero(tied)[0].tolist(),
        "n_dominant": int(dominant.sum()), "dominant_cells": np.nonzero(dominant)[0].tolist(),
        "worst_dominant_rel_err": worst,
        "median_dominant_rel_err": (float(np.median(rel[dominant])) if dominant.any() else float("nan")),
        "sign_agreement": signs_ok,
        "fired": (None if not dominant.any() else bool(worst > tol or not signs_ok)),
        "g_ad": g_ad.tolist(), "g_fd": g_fd.tolist(),
        "tie_table": [{"k": int(k), "d_m": float(d0[k]), "g_ad": float(g_ad[k]),
                       "fd_plus": float(fd_plus[k]), "fd_minus": float(fd_minus[k])}
                      for k in np.nonzero(tied)[0]],
        "wallclock_s": time.time() - t0,
    }


def measure_ad1(smoke: bool = False, eps_col=None) -> dict:
    """AD1: profile gradient on the A1 MB s = 2 dz vector, eps fixed by node
    index to the production column (note 2.4); f32 gated, x64 reported."""
    t_unit = time.time()
    dz0 = a1_mb(2.0)
    if eps_col is None:
        eps_col = np.asarray(a1_production_column(dz0, 2.0)["column"], np.float64)
    pattern_ok = bool(len(eps_col) == len(AD1_EPS_PATTERN)
                      and np.max(np.abs(f32_values(AD1_EPS_PATTERN) - eps_col)) == 0.0)
    n_steps = 12 if smoke else AD_N_STEPS
    fd_cells = [0, 10, 12] if smoke else None
    out = {"dz_m": dz0.tolist(), "eps_column": list(map(float, eps_col)),
           "eps_pattern_matches_declared": pattern_ok, "n_steps": n_steps, "smoke": bool(smoke),
           "src": list(AD1_SRC), "prb": list(AD1_PRB), "domain_xy": list(AD1_DOMAIN_XY), "dx": AD1_DX}
    try:
        out["f32"] = ad_compare_profile(ad1_loss_factory(eps_col, n_steps), dz0, "f32-default",
                                        AD1_TOL, fd_cells)
        out["f32"]["dt"] = float(make_nonuniform_grid(AD1_DOMAIN_XY, dz0, AD1_DX, cpml_layers=0).dt)
    except Exception as exc:  # noqa: BLE001 — a tracer-path failure IS the fired witness
        out["f32"] = {"error": repr(exc), "fired": True, "label": "f32-default"}
    if not smoke:
        try:
            from tests._x64_compat import enable_x64
            with enable_x64():
                jax.clear_caches()
                out["x64_context"] = ad_compare_profile(ad1_loss_factory(eps_col, n_steps), dz0,
                                                        "x64-context", AD1_TOL, fd_cells)
            jax.clear_caches()
        except Exception as exc:  # noqa: BLE001 — reported, not gated (W5 convention)
            out["x64_context"] = {"error": repr(exc)}
    out["wallclock_s"] = time.time() - t_unit
    out.update(provenance())
    return out


def measure_ad2(smoke: bool = False, eps_col=None) -> dict:
    """AD2: d L / d eps_thin on the AD1 mesh vs central FD, h = 0.015."""
    t_unit = time.time()
    dz0 = a1_mb(2.0)
    if eps_col is None:
        eps_col = np.asarray(a1_production_column(dz0, 2.0)["column"], np.float64)
    thin_nodes = np.nonzero(np.isclose(eps_col, 3.0))[0]
    n_steps = 12 if smoke else AD_N_STEPS
    grid = make_nonuniform_grid(domain_xy=AD1_DOMAIN_XY, dz_profile=dz0, dx=AD1_DX, cpml_layers=0)
    shape = grid.shape
    base = jnp.asarray(eps_col, jnp.float32)
    idx = jnp.asarray(thin_nodes)
    wf = _ad_waveform(n_steps, grid.dt)

    def loss(eps_thin):
        col = base.at[idx].set(eps_thin)
        mats = _vacuum(shape)._replace(eps_r=jnp.broadcast_to(col[None, None, :], shape))
        out = run_nonuniform(grid, mats, n_steps,
                             sources=[(*AD1_SRC, "ey", wf)], probes=[(*AD1_PRB, "ey")])
        return jnp.sum(out["time_series"] ** 2)

    row = {"thin_nodes": thin_nodes.tolist(), "eps_thin0": 3.0, "h": AD2_H, "tol": AD2_TOL,
           "n_steps": n_steps, "smoke": bool(smoke), "dt": float(grid.dt), "loss_jitted": True}
    try:
        loss_j = jax.jit(loss)
        g_ad = float(jax.jit(jax.grad(loss))(jnp.float32(3.0)))
        lp = float(loss_j(jnp.float32(3.0 + AD2_H)))
        lm = float(loss_j(jnp.float32(3.0 - AD2_H)))
        g_fd = (lp - lm) / (2 * AD2_H)
        rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-300)
        row.update({"g_ad": g_ad, "g_fd": g_fd, "rel_err": rel,
                    "sign_agreement": bool(np.sign(g_ad) == np.sign(g_fd)),
                    "finite": bool(np.isfinite(g_ad)),
                    "fired": bool(not np.isfinite(g_ad) or rel > AD2_TOL or np.sign(g_ad) != np.sign(g_fd))})
    except Exception as exc:  # noqa: BLE001
        row.update({"error": repr(exc), "fired": True})
    row["wallclock_s"] = time.time() - t_unit
    row.update(provenance())
    return row


def measure_ad3(smoke: bool = False) -> dict:
    """AD3: joint (dx, dy, dz) gradient on the A3 builder vectors."""
    t_unit = time.time()
    px, py, pz = a3_profiles()
    a, b = float(px.sum()), float(py.sum())
    n_steps = 12 if smoke else AD_N_STEPS

    def loss(profiles):
        dx_p, dy_p, dz_p = profiles
        grid = make_nonuniform_grid(domain_xy=(a, b), dz_profile=dz_p, dx=BOUNDARY_CELL,
                                    dx_profile=dx_p, dy_profile=dy_p, cpml_layers=0)
        wf = _ad_waveform(n_steps, grid.dt)
        out = run_nonuniform(grid, _vacuum(grid.shape), n_steps,
                             sources=[(*AD3_SRC, "ez", wf)], probes=[(*AD3_PRB, "ez")])
        return jnp.sum(out["time_series"] ** 2)

    out = {"n_steps": n_steps, "smoke": bool(smoke), "src": list(AD3_SRC), "prb": list(AD3_PRB),
           "dx_m": px.tolist(), "dy_m": py.tolist(), "dz_m": pz.tolist(), "axes": {}}
    try:
        loss_j = jax.jit(loss)
        grad_j = jax.jit(jax.grad(loss))
        x0 = (jnp.asarray(px, jnp.float32), jnp.asarray(py, jnp.float32), jnp.asarray(pz, jnp.float32))
        g_ad = [np.asarray(g, np.float64) for g in grad_j(x0)]
        loss0 = float(loss_j(x0))
        out["loss0"] = loss0
        out["dt"] = float(make_nonuniform_grid((a, b), pz, BOUNDARY_CELL, dx_profile=px,
                                               dy_profile=py, cpml_layers=0).dt)
        base = [np.asarray(px), np.asarray(py), np.asarray(pz)]
        for ai, ax in enumerate("xyz"):
            d0 = base[ai]
            n = len(d0)
            d32 = f32_values(d0)
            tied = np.isclose(d32, d32.min(), rtol=AD5_TIE_REL, atol=0.0)
            cells = ([0, n // 2, n - 1] if smoke else range(n))
            g_fd = np.full(n, np.nan)
            fdp = np.full(n, np.nan)
            fdm = np.full(n, np.nan)
            for k in cells:
                h = FD_REL_H * d0[k]
                vp = d0.copy()
                vp[k] += h
                vm = d0.copy()
                vm[k] -= h
                dp = list(base)
                dm = list(base)
                dp[ai] = vp
                dm[ai] = vm
                lp = float(loss_j(tuple(jnp.asarray(v, jnp.float32) for v in dp)))
                lm = float(loss_j(tuple(jnp.asarray(v, jnp.float32) for v in dm)))
                g_fd[k] = (lp - lm) / (2 * h)
                fdp[k] = (lp - loss0) / h
                fdm[k] = (loss0 - lm) / h
            free = np.isfinite(g_fd) & ~tied
            if free.any():
                gmax = np.abs(g_fd[free]).max()
                dominant = free & (np.abs(g_fd) > DOMINANT_FRAC * gmax)
            else:
                dominant = np.zeros(n, bool)
            rel = np.abs(g_ad[ai] - g_fd) / np.maximum(np.abs(g_fd), 1e-300)
            worst = float(rel[dominant].max()) if dominant.any() else float("nan")
            signs = (bool(np.all(np.sign(g_ad[ai][dominant]) == np.sign(g_fd[dominant])))
                     if dominant.any() else None)
            out["axes"][ax] = {
                "n_cells": n, "all_finite": bool(np.all(np.isfinite(g_ad[ai]))),
                "n_tied_min": int(tied.sum()), "tied_cells": np.nonzero(tied)[0].tolist(),
                "n_dominant": int(dominant.sum()), "dominant_cells": np.nonzero(dominant)[0].tolist(),
                "worst_dominant_rel_err": worst,
                "median_dominant_rel_err": (float(np.median(rel[dominant])) if dominant.any() else float("nan")),
                "sign_agreement": signs, "tol": AD3_TOL,
                "fired": (None if not dominant.any() else bool(worst > AD3_TOL or not signs)),
                "g_ad": g_ad[ai].tolist(), "g_fd": g_fd.tolist(),
                "tie_table": [{"k": int(k), "d_m": float(d0[k]), "g_ad": float(g_ad[ai][k]),
                               "fd_plus": float(fdp[k]), "fd_minus": float(fdm[k])}
                              for k in np.nonzero(tied)[0]],
            }
        fired = [out["axes"][ax]["fired"] for ax in "xyz"]
        out["fired"] = (None if all(f is None for f in fired) else bool(any(f is True for f in fired)))
    except Exception as exc:  # noqa: BLE001
        out.update({"error": repr(exc), "fired": True})
    out["wallclock_s"] = time.time() - t_unit
    out.update(provenance())
    return out


def _ad3_loss(px, py, pz, n_steps: int):
    a, b = float(px.sum()), float(py.sum())

    def loss(profiles):
        dx_p, dy_p, dz_p = profiles
        grid = make_nonuniform_grid(domain_xy=(a, b), dz_profile=dz_p, dx=BOUNDARY_CELL,
                                    dx_profile=dx_p, dy_profile=dy_p, cpml_layers=0)
        wf = _ad_waveform(n_steps, grid.dt)
        out = run_nonuniform(grid, _vacuum(grid.shape), n_steps,
                             sources=[(*AD3_SRC, "ez", wf)], probes=[(*AD3_PRB, "ez")])
        return jnp.sum(out["time_series"] ** 2)
    return loss


AD3B_REASON = (
    "Second attempt of AD3 (instrument defect in the FD reference, not a tolerance edit). "
    "Attempt 1 (key 'ad3', kept, FIRED on y and z) compared jax.grad against a central FD with "
    "h = 1e-3 d_k on an f32 loss (loss0 = 0.1306, ulp 1.49e-8): the FD reference cannot resolve a slope "
    "finer than ulp/(2h) and the non-tied dominant cells of y and z carried only 3-17 such quanta. "
    "This attempt keeps the loss, fixture, dominance rule (5 % of the largest non-tied |g_fd|), "
    "tolerance (0.15) and sign rule, records FD at h in AD3B_FD_REL_HS, gates at h = 1e-2 (quantum 10x "
    "smaller), and declares a reference-resolution floor: a dominant cell whose |g_fd| is below "
    "AD_REF_FLOOR_QUANTA quanta is reported as unresolved by the reference (with its AD, FD and "
    "forward-mode jvp values) and does not enter the gate; an axis with fewer resolved than unresolved "
    "dominant cells is INCONCLUSIVE (reference-limited), not HELD. Forward-mode jvp per cell is "
    "recorded as a second, FD-free reference (reported)."
)


def measure_ad3_second(smoke: bool = False) -> dict:
    """AD3 second attempt (see AD3B_REASON): same loss as measure_ad3;
    reverse grad, forward jvp per cell, central FD at every h of
    AD3B_FD_REL_HS, gate at AD3B_FD_REL_H_GATE on reference-resolved
    dominant cells."""
    t_unit = time.time()
    px, py, pz = a3_profiles()
    n_steps = 12 if smoke else AD_N_STEPS
    hs = (1e-3, 1e-2) if smoke else AD3B_FD_REL_HS
    loss = _ad3_loss(px, py, pz, n_steps)
    out = {"attempt": 2, "first_attempt_key": "ad3", "reason": AD3B_REASON,
           "n_steps": n_steps, "smoke": bool(smoke), "src": list(AD3_SRC), "prb": list(AD3_PRB),
           "dx_m": px.tolist(), "dy_m": py.tolist(), "dz_m": pz.tolist(),
           "fd_rel_hs": list(hs), "fd_rel_h_gate": AD3B_FD_REL_H_GATE, "ref_floor_quanta": AD_REF_FLOOR_QUANTA,
           "dominant_frac": DOMINANT_FRAC, "tol": AD3_TOL, "axes": {}}
    try:
        loss_j = jax.jit(loss)
        grad_j = jax.jit(jax.grad(loss))
        jvp_j = jax.jit(lambda x, t: jax.jvp(loss, (x,), (t,))[1])
        base = [np.asarray(px), np.asarray(py), np.asarray(pz)]
        x0 = tuple(jnp.asarray(v, jnp.float32) for v in base)
        g_ad = [np.asarray(g, np.float64) for g in grad_j(x0)]
        loss0 = float(loss_j(x0))
        ulp = float(np.spacing(np.float32(loss0)))
        out.update({"loss0": loss0, "loss_ulp": ulp,
                    "dt": float(make_nonuniform_grid((float(px.sum()), float(py.sum())), pz, BOUNDARY_CELL,
                                                     dx_profile=px, dy_profile=py, cpml_layers=0).dt)})
        for ai, ax in enumerate("xyz"):
            d0 = base[ai]
            n = len(d0)
            d32 = f32_values(d0)
            tied = np.isclose(d32, d32.min(), rtol=AD5_TIE_REL, atol=0.0)
            cells = ([0, n // 2, n - 1] if smoke else list(range(n)))
            g_jvp = np.full(n, np.nan)
            for k in cells:
                t = [jnp.zeros_like(v) for v in x0]
                t[ai] = t[ai].at[k].set(1.0)
                g_jvp[k] = float(jvp_j(x0, tuple(t)))
            have_j = np.isfinite(g_jvp)
            rel_fr = np.abs(g_ad[ai] - g_jvp) / np.maximum(np.abs(g_jvp), 1e-300)
            axrow = {
                "n_cells": n, "all_finite": bool(np.all(np.isfinite(g_ad[ai]))),
                "n_tied_min": int(tied.sum()), "tied_cells": np.nonzero(tied)[0].tolist(),
                "g_ad": g_ad[ai].tolist(), "g_jvp": g_jvp.tolist(),
                "rev_vs_fwd_worst_rel_all": (float(rel_fr[have_j].max()) if have_j.any() else None),
                "rev_vs_fwd_worst_rel_nontied": (float(rel_fr[have_j & ~tied].max()) if (have_j & ~tied).any() else None),
                "gmax_all_jvp": (float(np.abs(g_jvp[have_j]).max()) if have_j.any() else None),
                "gmax_nontied_jvp": (float(np.abs(g_jvp[have_j & ~tied]).max()) if (have_j & ~tied).any() else None),
                "per_h": {},
            }
            for h_rel in hs:
                g_fd = np.full(n, np.nan)
                fdp = np.full(n, np.nan)
                fdm = np.full(n, np.nan)
                for k in cells:
                    h = h_rel * d0[k]
                    vp = d0.copy()
                    vp[k] += h
                    vm = d0.copy()
                    vm[k] -= h
                    dp = list(base)
                    dm = list(base)
                    dp[ai] = vp
                    dm[ai] = vm
                    lp = float(loss_j(tuple(jnp.asarray(v, jnp.float32) for v in dp)))
                    lm = float(loss_j(tuple(jnp.asarray(v, jnp.float32) for v in dm)))
                    g_fd[k] = (lp - lm) / (2 * h)
                    fdp[k] = (lp - loss0) / h
                    fdm[k] = (loss0 - lm) / h
                quanta = np.abs(g_fd) * 2 * h_rel * d0 / ulp
                free = np.isfinite(g_fd) & ~tied
                if free.any():
                    gmax = np.abs(g_fd[free]).max()
                    dominant = free & (np.abs(g_fd) > DOMINANT_FRAC * gmax)
                else:
                    dominant = np.zeros(n, bool)
                resolved = dominant & (quanta >= AD_REF_FLOOR_QUANTA)
                unresolved = dominant & ~resolved
                rel = np.abs(g_ad[ai] - g_fd) / np.maximum(np.abs(g_fd), 1e-300)
                worst_all = float(rel[dominant].max()) if dominant.any() else float("nan")
                signs_all = (bool(np.all(np.sign(g_ad[ai][dominant]) == np.sign(g_fd[dominant])))
                             if dominant.any() else None)
                worst_res = float(rel[resolved].max()) if resolved.any() else float("nan")
                signs_res = (bool(np.all(np.sign(g_ad[ai][resolved]) == np.sign(g_fd[resolved])))
                             if resolved.any() else None)
                inconclusive = bool(dominant.any() and resolved.sum() <= unresolved.sum())
                axrow["per_h"][f"{h_rel:g}"] = {
                    "fd_rel_h": h_rel, "g_fd": g_fd.tolist(), "fd_plus": fdp.tolist(), "fd_minus": fdm.tolist(),
                    "quanta": quanta.tolist(),
                    "n_dominant": int(dominant.sum()), "dominant_cells": np.nonzero(dominant)[0].tolist(),
                    "worst_dominant_rel_err_attempt1_rule": worst_all, "sign_agreement_attempt1_rule": signs_all,
                    "fired_attempt1_rule": (None if not dominant.any() else bool(worst_all > AD3_TOL or not signs_all)),
                    "n_resolved": int(resolved.sum()), "resolved_cells": np.nonzero(resolved)[0].tolist(),
                    "n_unresolved": int(unresolved.sum()),
                    "worst_resolved_rel_err": worst_res, "sign_agreement_resolved": signs_res,
                    "median_resolved_rel_err": (float(np.median(rel[resolved])) if resolved.any() else float("nan")),
                    "unresolved_dominant": [{"k": int(k), "d_m": float(d0[k]), "g_ad": float(g_ad[ai][k]),
                                             "g_fd": float(g_fd[k]), "g_jvp": float(g_jvp[k]),
                                             "quanta": float(quanta[k]), "rel_fd": float(rel[k]),
                                             "rel_jvp": float(rel_fr[k])}
                                            for k in np.nonzero(unresolved)[0]],
                    "inconclusive": inconclusive,
                    "fired": (None if (not resolved.any() or inconclusive)
                              else bool(worst_res > AD3_TOL or not signs_res)),
                    "tie_table": [{"k": int(k), "d_m": float(d0[k]), "g_ad": float(g_ad[ai][k]),
                                   "g_jvp": float(g_jvp[k]), "fd_plus": float(fdp[k]), "fd_minus": float(fdm[k]),
                                   "split_model": float(fdp[k] + (fdm[k] - fdp[k]) / max(int(tied.sum()), 1))}
                                  for k in np.nonzero(tied)[0]],
                }
            gate = axrow["per_h"][f"{AD3B_FD_REL_H_GATE:g}"]
            axrow.update({"gate_h": AD3B_FD_REL_H_GATE, "n_dominant": gate["n_dominant"],
                          "dominant_cells": gate["dominant_cells"], "n_resolved": gate["n_resolved"],
                          "n_unresolved": gate["n_unresolved"], "worst_resolved_rel_err": gate["worst_resolved_rel_err"],
                          "sign_agreement": gate["sign_agreement_resolved"], "inconclusive": gate["inconclusive"],
                          "fired": gate["fired"], "tol": AD3_TOL})
            out["axes"][ax] = axrow
        fired = [out["axes"][ax]["fired"] for ax in "xyz"]
        out["inconclusive_axes"] = [ax for ax in "xyz" if out["axes"][ax]["inconclusive"]]
        out["fired"] = (None if all(f is None for f in fired) else bool(any(f is True for f in fired)))
    except Exception as exc:  # noqa: BLE001
        out.update({"error": repr(exc), "fired": True})
    out["wallclock_s"] = time.time() - t_unit
    out.update(provenance())
    return out


def ad4_family() -> list[dict]:
    """The 20 seeded stacks of note 2.7, drawn in the declared order."""
    rng = np.random.default_rng(AD4_SEED)
    fam = []
    for i in range(AD4_N_STACKS):
        n_lay = int(rng.integers(1, 7))
        thick = 10 ** rng.uniform(np.log10(20e-6), np.log10(3e-3), size=n_lay)
        frac = 10 ** rng.uniform(np.log10(0.05), np.log10(0.33), size=n_lay)
        targets = thick * frac
        prot = rng.random(n_lay) < 0.5
        cap = float(rng.choice([1.2, 1.3, 1.4]))
        eps = rng.uniform(1.0, 6.0, size=n_lay)
        pin = bool(rng.random() < 0.5)
        edges = np.concatenate([[0.0], np.cumsum(thick)])
        bc = None
        if pin:
            prot[0] = prot[-1] = False
            lim = min(thick[0], thick[-1]) * min(0.25, (cap - 1) / (1.1 * cap))
            bc = float(min(targets[0], targets[-1], lim))
        prof = np.asarray(make_band_profile(edges, targets, protected=prot.tolist(),
                                            max_ratio=cap, boundary_cell=bc), np.float64)
        fam.append({"i": i, "n_layers": n_lay, "cap": cap, "pin": pin, "boundary_cell": bc,
                    "edges_m": edges.tolist(), "eps": eps.tolist(), "protected": prot.tolist(),
                    "targets_m": targets.tolist(), "profile_m": prof.tolist(),
                    "nz": int(len(prof)), "dz_min_m": float(prof.min()), "dz_max_m": float(prof.max())})
    return fam


def measure_ad4(smoke: bool = False) -> dict:
    t_unit = time.time()
    fam = ad4_family()
    stacks = fam[:3] if smoke else fam
    rows = []
    n_nonfinite = 0
    for st in stacks:
        prof = np.asarray(st["profile_m"])
        eps_nodes = halfopen_eps_nodes(prof, st["edges_m"], st["eps"])
        nz = len(prof) + 1
        k_src = nz // 2
        k_prb = k_src - 1 if k_src - 1 >= 1 else k_src + 1
        eps_col = jnp.asarray(eps_nodes, jnp.float32)

        def loss(dz, eps_col=eps_col, k_src=k_src, k_prb=k_prb):
            grid = make_nonuniform_grid(domain_xy=AD4_DOMAIN_XY, dz_profile=dz, dx=AD4_DX, cpml_layers=0)
            mats = _vacuum(grid.shape)._replace(eps_r=jnp.broadcast_to(eps_col[None, None, :], grid.shape))
            wf = _ad_waveform(AD4_N_STEPS, grid.dt)
            out = run_nonuniform(grid, mats, AD4_N_STEPS,
                                 sources=[(1, 1, k_src, "ez", wf)], probes=[(1, 1, k_prb, "ez")])
            return jnp.sum(out["time_series"] ** 2), out["time_series"]

        row = {k: st[k] for k in ("i", "n_layers", "cap", "pin", "nz", "dz_min_m", "dz_max_m")}
        try:
            grid = make_nonuniform_grid(domain_xy=AD4_DOMAIN_XY, dz_profile=prof, dx=AD4_DX, cpml_layers=0)
            row["dt"] = float(grid.dt)
            (l0, ts), g = jax.value_and_grad(loss, has_aux=True)(jnp.asarray(prof, jnp.float32))
            ts = np.asarray(ts, np.float64)
            g = np.asarray(g, np.float64)
            row.update({"trace_finite": bool(np.all(np.isfinite(ts))), "trace_max": float(np.abs(ts).max()),
                        "grad_finite": bool(np.all(np.isfinite(g))), "grad_max_abs": float(np.abs(g).max()),
                        "loss": float(l0), "dt_positive": bool(row["dt"] > 0)})
            ok = row["trace_finite"] and row["grad_finite"] and row["dt_positive"] and np.isfinite(float(l0))
        except Exception as exc:  # noqa: BLE001
            row.update({"error": repr(exc)})
            ok = False
        row["pass"] = bool(ok)
        n_nonfinite += 0 if ok else 1
        rows.append(row)
    return {"seed": AD4_SEED, "n_stacks": len(stacks), "smoke": bool(smoke), "n_steps": AD4_N_STEPS,
            "family_nz_range": [min(s["nz"] for s in fam), max(s["nz"] for s in fam)],
            "family_dz_min_m": min(s["dz_min_m"] for s in fam),
            "family_layer_counts": [s["n_layers"] for s in fam],
            "stacks": rows, "n_failed": n_nonfinite, "fired": bool(n_nonfinite > 0),
            "wallclock_s": time.time() - t_unit, **provenance()}


# ---------------------------------------------------------------------------
# --selfcheck: oracle, builder vectors, model table (no FDTD)
# ---------------------------------------------------------------------------
def _rel(x, y):
    return abs(x - y) / max(abs(y), 1e-300)


def _ad1_loss_dtype_probe(eps_col) -> dict:
    dz0 = a1_mb(2.0)
    loss = ad1_loss_factory(eps_col, AD_N_STEPS)

    def probe():
        x = jnp.asarray(dz0)
        L = jax.eval_shape(loss, x)
        G = jax.eval_shape(jax.grad(loss), x)
        dt = jax.eval_shape(lambda d: make_nonuniform_grid(AD1_DOMAIN_XY, d, AD1_DX, cpml_layers=0).dt, x)
        return {"input": str(x.dtype), "loss": str(L.dtype), "grad": str(G.dtype), "dt": str(dt.dtype)}
    out = {"default": probe()}
    try:
        from tests._x64_compat import enable_x64
        with enable_x64():
            jax.clear_caches()
            out["x64_context"] = probe()
        jax.clear_caches()
    except Exception as exc:  # noqa: BLE001
        out["x64_context"] = {"error": repr(exc)}
    out["x64_context_raises_solver_precision"] = bool(out["x64_context"].get("loss") == "float64")
    return out


def selfcheck(verbose: bool = True) -> dict:
    t0 = time.time()
    sc = {"oracle": oracle_selfcheck(), "checks": {}}
    ch = sc["checks"]
    f_true = sc["oracle"]["f_true_hz"]
    ch["oracle_i"] = sc["oracle"]["i_pass"]
    ch["oracle_ip"] = sc["oracle"]["ip_pass"]
    ch["oracle_ipp_lsm"] = sc["oracle"]["ipp_pass"]
    ch["f_true_declared"] = sc["oracle"]["f_true_matches_declared"]
    ch["mode_p"] = sc["oracle"]["p_matches_declared"]
    ch["neighbours"] = sc["oracle"]["neighbours_match_declared"]
    ch["isolation"] = sc["oracle"]["isolation_matches_declared"]
    # --- A1 profiles, columns and the model table ---------------------------
    sc["a1"] = {"units": {}, "profiles": {}}
    model_rows = {}
    for arm in A1_ARMS:
        for s in A1_SCALES:
            prof = a1_profile(arm, s)
            rep = profile_report(prof, A1_EDGES)
            prof_ok = rep["n"] == A1_NZ[arm][s]
            if arm == "mb":
                dec = a1_mb_declared(s)
                prof_ok = prof_ok and len(dec) == len(prof) and np.max(np.abs(dec - prof)) <= CELL_TOL
                prof_ok = prof_ok and rep["interface_err_m"] <= CELL_TOL and rep["max_ratio"] <= R_CAP + 1e-9
            if arm == "az":
                prof_ok = prof_ok and abs(rep["d_min_m"] * 1e6 - A1_AZ_DZ_MIN_UM[s]) <= 0.05
            zn = np.concatenate([[0.0], np.cumsum(prof)])
            nodes_ok = all(np.min(np.abs(zn - z)) <= CELL_TOL for z in (Z_SRC, Z_PRB))
            ch[f"a1_profile_{arm}_{s:g}"] = bool(prof_ok and nodes_ok)
            sc["a1"]["profiles"][f"{arm}|{s:g}"] = {**{k: v for k, v in rep.items() if k != "cells_m"},
                                                    "src_prb_on_nodes": nodes_ok}
            dx = DXY0 * s
            grid = make_nonuniform_grid((A1_A_X, A1_B_Y), prof, dx, cpml_layers=0)
            dt = float(grid.dt)
            pc = a1_production_column(prof, s, f_true)
            col_prod = np.asarray(pc["column"])
            col_dual = f32_values(dual_eps_nodes(prof))
            m_dual = a1_model(prof, col_dual, dx, dt, f_true)
            m_prod = a1_model(prof, col_prod, dx, dt, f_true)
            q_dual, q_prod = A1_MODEL_TABLE_MHZ[(arm, s)]
            ok_dual = _rel(m_dual["f_model"], f_true + q_dual * 1e6) <= SELFCHECK_REL
            ok_prod = _rel(m_prod["f_model"], f_true + q_prod * 1e6) <= SELFCHECK_REL
            table = tuple(round(pc["interface_table"][k]["eps"], 1) for k in ("14mm", "16mm", "30mm"))
            ok_table = table == A1_INTERFACE_TABLE[(arm, s)]
            ch[f"a1_model_{arm}_{s:g}_dual"] = bool(ok_dual)
            ch[f"a1_model_{arm}_{s:g}_production"] = bool(ok_prod)
            ch[f"a1_interface_table_{arm}_{s:g}"] = bool(ok_table)
            model_rows[(arm, s)] = (m_dual, m_prod)
            sc["a1"]["units"][f"{arm}|{s:g}"] = {
                "dt": dt, "cells": int(grid.nx * grid.ny * grid.nz), "n_steps_15ns": int(round(T_TOTAL / dt)),
                "dual_e_mhz": m_dual["e_total_model"] / 1e6, "production_e_mhz": m_prod["e_total_model"] / 1e6,
                "quoted_dual_mhz": q_dual, "quoted_production_mhz": q_prod,
                "z_fraction_dual": m_dual["z_fraction"], "e_z_dual": m_dual["e_z"],
                "interface_table": pc["interface_table"], "interface_eps": list(table),
                "interior_transversely_uniform": pc["interior_transversely_uniform"],
                "production_column": pc["column"], "dual_column": col_dual.tolist()}
            if verbose:
                print(f"  selfcheck A1 {arm} s={s:g}: dual {m_dual['e_total_model']/1e6:+.4f} "
                      f"(quoted {q_dual:+.4f}) production {m_prod['e_total_model']/1e6:+.4f} "
                      f"(quoted {q_prod:+.4f}) MHz table={table} ok={ok_dual and ok_prod and ok_table}",
                      flush=True)
    # orders / rho / G1 / G2 from the model
    h = np.log10([DF0 * s for s in A1_SCALES])
    fits = {}
    for arm in ("uc", "mb"):
        for j, rule in enumerate(A1_RULES):
            e = np.log10([abs(model_rows[(arm, s)][j]["e_total_model"]) for s in A1_SCALES])
            fits[f"{rule}_{arm}"] = np.polyfit(h, e, 1)
    orders = {k: float(v[0]) for k, v in fits.items()}
    lh = np.log10(H1)
    rho = float(10 ** (np.polyval(fits["dual_mb"], lh) - np.polyval(fits["dual_uc"], lh)))
    rho_scale = {s: abs(model_rows[("mb", s)][0]["e_total_model"]) / abs(model_rows[("uc", s)][0]["e_total_model"])
                 for s in A1_SCALES}
    g1 = {arm: min(model_rows[(arm, s)][0]["z_fraction"] for s in A1_SCALES) for arm in ("uc", "mb")}
    g2 = [abs(model_rows[("mb", s)][0]["e_z"] - model_rows[("uc", s)][0]["e_z"])
          / abs(model_rows[("mb", s)][0]["e_total_model"]) for s in A1_SCALES]
    sc["a1"]["model_orders"] = orders
    sc["a1"]["model_rho"] = rho
    sc["a1"]["model_rho_per_scale"] = {f"{s:g}": v for s, v in rho_scale.items()}
    sc["a1"]["model_g1"] = g1
    sc["a1"]["model_g2"] = g2
    ch["a1_model_orders"] = bool(all(abs(orders[k] - A1_MODEL_ORDERS[k]) <= 1e-3 for k in A1_MODEL_ORDERS))
    ch["a1_model_rho"] = bool(abs(rho - A1_MODEL_RHO) <= 1e-3
                              and all(abs(rho_scale[s] - A1_MODEL_RHO_PER_SCALE[s]) <= 1e-3 for s in A1_SCALES))
    ch["a1_model_g1"] = bool(all(abs(g1[a] - A1_MODEL_G1[a]) <= 1e-3 for a in g1))
    ch["a1_model_g2"] = bool(A1_MODEL_G2_RANGE[0] - 1e-3 <= min(g2) and max(g2) <= A1_MODEL_G2_RANGE[1] + 1e-3)
    # --- A2 -------------------------------------------------------------------
    sc["a2"] = {}
    a, b = A2_X_EDGES[-1], A2_Y_EDGES[-1]
    f110 = closed_form(1, 1, a, b)
    ch["a2_f110"] = bool(abs(f110 - A2_F110_HZ) <= 1e3)
    for cap in A2_CAPS:
        px, py, pz = a2_profiles(cap)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(freq_max=2 * f110, domain=(0, 0, 0), boundary="pec", dx=BOUNDARY_CELL,
                             dz_profile=pz, dx_profile=px, dy_profile=py)
            dt = float(sim._build_nonuniform_grid().dt)
        err = a2_model_err(px, py, dt, f110)
        rx, ry = profile_report(px, A2_X_EDGES), profile_report(py, A2_Y_EDGES)
        ok = (rx["n"] == A2_PROFILE_N[(f"{cap}", "x")] and ry["n"] == A2_PROFILE_N[(f"{cap}", "y")]
              and abs(rx["max_ratio"] - A2_PROFILE_MAXR[(f"{cap}", "x")]) <= 1e-6
              and abs(ry["max_ratio"] - A2_PROFILE_MAXR[(f"{cap}", "y")]) <= 1e-6
              and rx["interface_err_m"] <= CELL_TOL and ry["interface_err_m"] <= CELL_TOL
              and abs(px[0] - BOUNDARY_CELL) <= CELL_TOL and abs(px[-1] - BOUNDARY_CELL) <= CELL_TOL
              and abs(py[0] - BOUNDARY_CELL) <= CELL_TOL and abs(py[-1] - BOUNDARY_CELL) <= CELL_TOL
              and rx["max_ratio"] <= cap + 1e-9 and ry["max_ratio"] <= cap + 1e-9)
        ch[f"a2_profiles_{cap}"] = bool(ok)
        ch[f"a2_model_{cap}"] = bool(abs(err * 100 - A2_MODEL_ERR_PCT[f"{cap}"]) <= 1e-5)
        sc["a2"][f"{cap}"] = {"dt": dt, "model_err_pct": err * 100, "quoted_pct": A2_MODEL_ERR_PCT[f"{cap}"],
                              "x": {k: v for k, v in rx.items() if k != "cells_m"},
                              "y": {k: v for k, v in ry.items() if k != "cells_m"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=2 * f110, domain=(a, b, A2_NZ * BOUNDARY_CELL), boundary="pec",
                         dx=BOUNDARY_CELL, dz_profile=np.full(A2_NZ, BOUNDARY_CELL))
        dt_u = float(sim._build_nonuniform_grid().dt)
    err_u = a2_model_err(np.full(40, 1e-3), np.full(34, 1e-3), dt_u, f110)
    ch["a2_model_uniform"] = bool(abs(err_u * 100 - A2_MODEL_ERR_PCT["uniform"]) <= 1e-5)
    sc["a2"]["uniform"] = {"dt": dt_u, "model_err_pct": err_u * 100, "quoted_pct": A2_MODEL_ERR_PCT["uniform"]}
    # --- A3 -------------------------------------------------------------------
    px, py, pz = a3_profiles()
    a3, b3, d3 = float(px.sum()), float(py.sum()), float(pz.sum())
    f111 = (C0 / 2) * np.sqrt((1 / a3) ** 2 + (1 / b3) ** 2 + (1 / d3) ** 2)
    ch["a3_f111"] = bool(abs(f111 - A3_F111_HZ) <= 1e3)
    reps = {ax: profile_report(p, A3_EDGES[ax]) for ax, p in (("x", px), ("y", py), ("z", pz))}
    ch["a3_profiles"] = bool(all(reps[ax]["n"] == A3_PROFILE_N[ax]
                                 and abs(reps[ax]["max_ratio"] - A3_PROFILE_MAXR[ax]) <= 1e-6
                                 and reps[ax]["interface_err_m"] <= CELL_TOL
                                 and reps[ax]["max_ratio"] <= A3_CAP + 1e-9 for ax in "xyz"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=2 * f111, domain=(0, 0, 0), boundary="pec", dx=BOUNDARY_CELL,
                         dz_profile=pz, dx_profile=px, dy_profile=py)
        dt = float(sim._build_nonuniform_grid().dt)
        sim_u = Simulation(freq_max=2 * f111, domain=(a3, b3, d3), boundary="pec", dx=BOUNDARY_CELL,
                           dz_profile=np.full(36, 1e-3))
        dt_u = float(sim_u._build_nonuniform_grid().dt)
    m3 = a3_model(px, py, pz, dt, f111)
    m3u = a3_model(np.full(40, 1e-3), np.full(35, 1e-3), np.full(36, 1e-3), dt_u, f111)
    ch["a3_model"] = bool(abs(m3["model_err"] * 100 - A3_MODEL_ERR_PCT["graded"]) <= 1e-5
                          and abs(m3u["model_err"] * 100 - A3_MODEL_ERR_PCT["uniform"]) <= 1e-5
                          and all(abs(m3[k] * 100 - A3_MODEL_TERMS_PCT[k]) <= 1e-4 for k in A3_MODEL_TERMS_PCT))
    sc["a3"] = {"dt": dt, "dt_uniform": dt_u, "f111": f111, "extents_m": [a3, b3, d3],
                "model_graded_pct": {k: v * 100 for k, v in m3.items()},
                "model_uniform_pct": m3u["model_err"] * 100,
                "profiles": {ax: {k: v for k, v in reps[ax].items() if k != "cells_m"} for ax in "xyz"}}
    # --- AD4 family and the AD1 column -------------------------------------------
    fam = ad4_family()
    nzs = [s["nz"] for s in fam]
    ch["ad4_family"] = bool((min(nzs), max(nzs)) == AD4_NZ_RANGE
                            and abs(min(s["dz_min_m"] for s in fam) * 1e6 - AD4_DZ_MIN_UM) <= 5e-4
                            and [s["n_layers"] for s in fam] == AD4_LAYER_COUNTS)
    sc["ad4_family"] = {"nz": nzs, "dz_min_um": [s["dz_min_m"] * 1e6 for s in fam],
                        "layers": [s["n_layers"] for s in fam]}
    col = np.asarray(sc["a1"]["units"]["mb|2"]["production_column"])
    ch["ad1_eps_pattern"] = bool(len(col) == 33 and np.max(np.abs(f32_values(AD1_EPS_PATTERN) - col)) == 0.0)
    # second pass, reported: the dtype the AD1 loss is computed in, with and
    # without the x64 context (jax.eval_shape, no FDTD). rfx.nonuniform pins
    # every profile and field to float32, so the "x64 context" arm of AD1 is
    # the same float32 solve — recorded so the note cannot claim otherwise.
    sc["ad1_loss_dtype"] = _ad1_loss_dtype_probe(col)
    sc["all_pass"] = bool(all(ch.values()))
    sc["failed"] = [k for k, v in ch.items() if not v]
    sc["wallclock_s"] = time.time() - t0
    sc.update(provenance())
    if verbose:
        print(f"selfcheck: oracle (i) worst {sc['oracle']['i_worst_rel']:.2e} over {sc['oracle']['i_roots']} roots, "
              f"(i') worst {sc['oracle']['ip_worst_rel']:.2e} over {sc['oracle']['ip_roots']} roots; "
              f"f_true={f_true:.3f} Hz; all_pass={sc['all_pass']} failed={sc['failed']}", flush=True)
    return sc


# ---------------------------------------------------------------------------
# JSON merge + CLI
# ---------------------------------------------------------------------------
def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"runs": [], "a1": {"units": {}}, "a2": {"units": {}}, "a3": {"units": {}}}


def _save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def _print_a1(r: dict) -> None:
    print(f"A1 {r['arm'].upper()} s={r['scale']:g} {r['rule']}: f={r['f_meas']/1e9:.6f} GHz "
          f"err={r['err_hz']/1e6:+.4f} MHz model={r['e_total_model']/1e6:+.4f} MHz "
          f"resid={r['model_residual_hz']/1e6:+.4f} MHz inv={r['invariance_hz']/1e6:.4f} MHz "
          f"zfrac={r['z_fraction']:.3f} valid={r['valid']} cells={r['cells']} steps={r['n_steps']} "
          f"wall={r['wallclock_run_s']:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="",
                    help="comma list from a1,a2,a3,ad1,ad2,ad3,ad4 (second pass: a2ext,a3ext,ad3b)")
    ap.add_argument("--scales", default="2,1,0.5", help="A1 scales")
    ap.add_argument("--interface-eps", choices=("sampled", "dual_average"), default="sampled")
    ap.add_argument("--rules", default="dual,production", help="A1 eps rules")
    ap.add_argument("--a1-arms", default="uc,mb,az", help="A1 profiles")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true", help="oracle + builder + model checks only")
    ap.add_argument("--smoke", action="store_true", help="tiny step counts; rows tagged smoke")
    ap.add_argument("--force", action="store_true",
                    help="re-measure an existing unit, keeping the earlier row under a superseded key")
    args = ap.parse_args(argv)

    def _supersede(container: dict, key: str) -> bool:
        """True when ``key`` is free to measure; with --force the earlier row
        is kept under a tagged key, never overwritten."""
        if key not in container or args.smoke:
            return True
        if not args.force:
            print(f"{key}: already measured (one attempt per unit) - skipping; --force keeps both", flush=True)
            return False
        tag = container[key].get("started_utc", "unknown")
        container[f"{key}|superseded|{tag}"] = container.pop(key)
        print(f"{key}: earlier attempt kept as {key}|superseded|{tag}", flush=True)
        return True

    print("rfx.__file__ =", rfx.__file__, flush=True)
    data = _load(args.out)
    run = {**provenance(), "arms": args.arms, "scales": args.scales, "rules": args.rules,
           "a1_arms": args.a1_arms, "smoke": args.smoke, "selfcheck_only": args.selfcheck,
           "interface_eps_rule": args.interface_eps}
    data.setdefault("runs", []).append(run)
    sc = selfcheck()
    data["selfcheck"] = sc
    _save(args.out, data)
    if not sc["all_pass"]:
        print("SELFCHECK FAILED:", sc["failed"], "- refusing to run any arm", flush=True)
        return 2
    if args.selfcheck:
        print("wrote", args.out)
        return 0

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    f_true = sc["oracle"]["f_true_hz"]
    if "a1" in arms:
        units = data.setdefault("a1", {}).setdefault("units", {})
        for s in [float(x) for x in args.scales.split(",") if x.strip()]:
            for arm in [x.strip() for x in args.a1_arms.split(",") if x.strip()]:
                for rule in [x.strip() for x in args.rules.split(",") if x.strip()]:
                    key = a1_key(arm, s, rule)
                    if not _supersede(units, key):
                        continue
                    row = measure_a1(arm, s, rule, smoke=args.smoke, f_true=f_true,
                                     interface_eps=args.interface_eps)
                    if args.smoke:
                        units[key + "|smoke"] = row
                    else:
                        units[key] = row
                    _print_a1(row)
                    data["a1"]["judge"] = judge_a1({k: v for k, v in units.items() if not v.get("smoke")})
                    _save(args.out, data)
        print("A1 judge:", data["a1"]["judge"]["verdict"], flush=True)
    if "a2" in arms:
        units = data.setdefault("a2", {}).setdefault("units", {})
        for cap in (*A2_CAPS, None):
            for n in A2_N_STEPS:
                key = a2_key(cap, n) + ("|smoke" if args.smoke else "")
                if not _supersede(units, key):
                    continue
                row = measure_a2(cap, n, smoke=args.smoke)
                units[key] = row
                print(f"A2 {key}: f={row['f_sim']/1e9:.6f} GHz err={row['err']*100:+.5f} % "
                      f"model={row['model_err']*100:+.5f} % sep={row['separation_pass']} "
                      f"src={row['src_idx']} prb={row['prb_idx']} wall={row['wallclock_run_s']:.0f}s", flush=True)
                data["a2"]["judge"] = judge_a2({k: v for k, v in units.items() if not v.get("smoke")})
                _save(args.out, data)
        print("A2 judge:", data["a2"]["judge"]["verdict"], flush=True)
    if "a3" in arms:
        units = data.setdefault("a3", {}).setdefault("units", {})
        for graded in (True, False):
            for n in A3_N_STEPS:
                key = a3_key(graded, n) + ("|smoke" if args.smoke else "")
                if not _supersede(units, key):
                    continue
                row = measure_a3(graded, n, smoke=args.smoke)
                units[key] = row
                print(f"A3 {key}: f={row['f_sim']/1e9:.6f} GHz err={row['err']*100:+.5f} % "
                      f"model={row['model_err']*100:+.5f} % sep={row['separation_pass']} "
                      f"wall={row['wallclock_run_s']:.0f}s", flush=True)
                data["a3"]["judge"] = judge_a3({k: v for k, v in units.items() if not v.get("smoke")})
                _save(args.out, data)
        print("A3 judge:", data["a3"]["judge"]["verdict"], flush=True)
    if "a2ext" in arms:
        units = data.setdefault("a2", {}).setdefault("units", {})
        for cap in (*A2_CAPS, None):
            for n in A2_N_STEPS_EXT:
                key = a2_key(cap, n) + ("|smoke" if args.smoke else "")
                if not _supersede(units, key):
                    continue
                row = measure_a2(cap, n, smoke=args.smoke)
                row["second_pass_extension"] = True
                units[key] = row
                print(f"A2ext {key}: f={row['f_sim']/1e9:.6f} GHz err={row['err']*100:+.5f} % "
                      f"model={row['model_err']*100:+.5f} % resid={row['model_residual']:+.2e} "
                      f"T={row['t_total_s']*1e9:.1f} ns sep={row['separation_pass']} "
                      f"wall={row['wallclock_run_s']:.0f}s", flush=True)
                data["a2"]["judge_ext"] = judge_a2_ext({k: v for k, v in units.items() if not v.get("smoke")})
                data["a2"]["judge"] = judge_a2({k: v for k, v in units.items() if not v.get("smoke")})
                _save(args.out, data)
        print("A2 ext judge:", data["a2"]["judge_ext"]["verdict"], flush=True)
    if "a3ext" in arms:
        units = data.setdefault("a3", {}).setdefault("units", {})
        for graded in (True, False):
            for n in A3_N_STEPS_EXT:
                key = a3_key(graded, n) + ("|smoke" if args.smoke else "")
                if not _supersede(units, key):
                    continue
                row = measure_a3(graded, n, smoke=args.smoke)
                row["second_pass_extension"] = True
                units[key] = row
                print(f"A3ext {key}: f={row['f_sim']/1e9:.6f} GHz err={row['err']*100:+.5f} % "
                      f"model={row['model_err']*100:+.5f} % resid={row['model_residual']:+.2e} "
                      f"T={row['t_total_s']*1e9:.1f} ns sep={row['separation_pass']} "
                      f"wall={row['wallclock_run_s']:.0f}s", flush=True)
                data["a3"]["judge_ext"] = judge_a3_ext({k: v for k, v in units.items() if not v.get("smoke")})
                data["a3"]["judge"] = judge_a3({k: v for k, v in units.items() if not v.get("smoke")})
                _save(args.out, data)
        print("A3 ext judge:", data["a3"]["judge_ext"]["verdict"], flush=True)
    eps_col = np.asarray(sc["a1"]["units"]["mb|2"]["production_column"], np.float64)
    for name, fn in (("ad1", lambda: measure_ad1(args.smoke, eps_col)),
                     ("ad2", lambda: measure_ad2(args.smoke, eps_col)),
                     ("ad3", lambda: measure_ad3(args.smoke)),
                     ("ad3b", lambda: measure_ad3_second(args.smoke)),
                     ("ad4", lambda: measure_ad4(args.smoke))):
        if name not in arms:
            continue
        key = ("ad3_second_attempt" if name == "ad3b" else name) + ("_smoke" if args.smoke else "")
        if not _supersede(data, key):
            continue
        row = fn()
        data[key] = row
        _save(args.out, data)
        if name == "ad1":
            f = row["f32"]
            print(f"AD1 f32: worst dominant rel err {f.get('worst_dominant_rel_err')} "
                  f"dominant={f.get('n_dominant')} tied={f.get('n_tied_min')} fired={f.get('fired')} "
                  f"error={f.get('error')}", flush=True)
        elif name == "ad2":
            print(f"AD2: g_ad={row.get('g_ad')} g_fd={row.get('g_fd')} rel={row.get('rel_err')} "
                  f"fired={row.get('fired')} error={row.get('error')}", flush=True)
        elif name == "ad3":
            print("AD3:", {ax: (v["worst_dominant_rel_err"], v["n_dominant"], v["n_tied_min"], v["fired"])
                           for ax, v in row.get("axes", {}).items()}, "fired=", row.get("fired"),
                  "error=", row.get("error"), flush=True)
        elif name == "ad3b":
            for ax, v in row.get("axes", {}).items():
                print(f"AD3 second attempt {ax}: gate h={v['gate_h']:g} dominant={v['n_dominant']} "
                      f"resolved={v['n_resolved']} unresolved={v['n_unresolved']} "
                      f"worst resolved rel={v['worst_resolved_rel_err']:.4g} signs={v['sign_agreement']} "
                      f"rev-vs-jvp all={v['rev_vs_fwd_worst_rel_all']:.3g} inconclusive={v['inconclusive']} "
                      f"fired={v['fired']}; per h attempt-1 rule: "
                      + ", ".join(f"{h}:{p['worst_dominant_rel_err_attempt1_rule']:.3g}" for h, p in v["per_h"].items()),
                      flush=True)
            print("AD3 second attempt fired=", row.get("fired"), "inconclusive=", row.get("inconclusive_axes"),
                  "error=", row.get("error"), flush=True)
        else:
            print(f"AD4: {row['n_stacks']} stacks, failed={row['n_failed']} fired={row['fired']}", flush=True)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

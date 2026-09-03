"""cv26 oblique-slab Fresnel -- the analytic oracle, the realized-angle
convention, the Meep k_point mapping, the Yee-lattice terms, the derived
windows, records and falsifiers.

Pure numpy (scipy for the banded solve). Importing this module does not
import rfx, jax or meep, so the case script, the Meep leg and the tests all
read the same constants. Everything numeric is fixed by
``docs/design_notes/20260902_cv26_oblique_fresnel_predeclaration.md``;
change the note (append-only) before changing a number here.

Conventions
-----------
* ``e^{+j omega t}``; a +x-travelling wave is ``e^{-j k_x x}``; loss is
  ``Im eps < 0``.
* The transverse Bloch wavenumber is FIXED per run: ``k_y = k0(f0) sin
  theta0`` (``rfx/sources/tfsf_2d.py:233``). At any other frequency the
  realized angle is ``theta(f) = asin(k_y c / (2 pi f))``; below the cutoff
  ``f_c = f0 sin theta0`` the incident wave is evanescent. Every oracle here
  is evaluated at the REALIZED angle of the bin, never at theta0.
* Polarizations: the rfx Bloch TFSF injects E perpendicular to the plane of
  incidence for both ``ez`` (tilt in xy) and ``ey`` (tilt in xz) -- Fresnel
  TE (s) in both cases. TM (p) on an eps-slab is reached through the exact
  eps <-> mu duality: an ``ez`` wave on a slab with ``mu_r = eps_slab``,
  ``eps_r = 1`` has the TM Fresnel r, t of the eps-slab (``slab_rt`` takes
  both eps and mu, and the duality is asserted in the tests).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests._gate_policy import gate_from_envelope  # noqa: E402

C0 = 299_792_458.0
EPS_0 = 8.8541878128e-12
MU_0 = 1.0 / (C0 ** 2 * EPS_0)
ETA_0 = math.sqrt(MU_0 / EPS_0)
TWO_PI = 2.0 * math.pi

CASE_ID = "26_oblique_slab_fresnel"
SCHEMA = "cv26-oblique-slab/v1"
RESULTS_DIRNAME = "_26_oblique_results"

# ---------------------------------------------------------------------------
# Rig (note section 1): cv04's slab and mesh, the 2-D TMz grid, the Bloch
# TFSF path (init_tfsf_2d directly, so theta0 = 0 exercises the SAME complex
# path), probes +-30 cells, aux-grid incident reference at the same x.
# ---------------------------------------------------------------------------
DX_M = 1.0e-3
D_SLAB_M = 10.0e-3
EPS_R_SLAB = 4.0
N_CPML = 20                 # cv04's absorber depth
CPML_ORDER = 3              # rfx/boundaries/cpml.py _cpml_profile default
CPML_KAPPA_MAX = 1.0        # cv04 (Grid default)
CPML_R_ASYMPTOTIC = 1e-15   # rfx/boundaries/cpml.py _cpml_profile default
CPML_ALPHA_MAX = 0.05       # alpha = 0.05 (1 - rho)  (S/m)
TFSF_MARGIN = 5
PROBE_OFFSET_CELLS = 30
NY_CELLS = 4                # cv04's 0.004 m transverse extent (envelope is y-uniform)
COURANT_2D = 0.99 / math.sqrt(2.0)   # rfx/grid.py:96 dt = 0.99 dx/(c sqrt 2) in 2-D: c dt/dx = 0.70004
DT_S = COURANT_2D * DX_M / C0
TFSF_F0_HZ = 10.0e9
SRC_T0_OVER_TAU = 3.0       # rfx/sources/tfsf_2d.py:212  t0 = 3 tau
PULSE_END_ARG_40DB = math.sqrt(math.log(100.0))   # exp(-a^2) = 1e-2 -> 2.146
BW_MAX = 0.25               # tfsf_2d docstring: fractional bandwidth <~ 0.3
TAIL_WINDOW = 50
TAIL_PURITY_LIMIT = 1e-3    # cv04's window-purity bar on the incident (04_multilayer_fresnel.py:209)
CUTOFF_INC_AMP = TAIL_PURITY_LIMIT   # incident amplitude at the cutoff f_c = f0 sin theta0: content
                                     # with zero group velocity along x never leaves the probe, so it
                                     # must sit under the purity bar (section 2)
CUTOFF_ARG = math.sqrt(-math.log(CUTOFF_INC_AMP))   # 2.628
NX_INTERIOR = 1500          # primary rig (section 5: the CPML gate at the 60 deg arm)
NX_INTERIOR_GRAZE = 100     # compact box: the absorber echo is INSIDE the record
NFFT_OVERSAMPLE = 8
MASK_F_LO_HZ = 3.0e9
MASK_F_HI_HZ = 15.0e9
MASK_AMP_FRAC = 0.02        # cv04's evaluated-band mask (amplitude)
GATED_INC_AMP_FRAC = 0.10   # gated bins: incident amplitude >= 10 % of peak
THETA_GATE_MAX_DEG = 70.0   # primary rig cap (section 5)
GRAZE_THETA0_DEG = 82.0
GRAZE_THETA_GATE_DEG = (80.0, 85.0)
SETTLING_LIMIT = 1e-2       # -40 dB (cv22 section 13)
RECORD_EXTEND_STEPS = 100
# NX_GROW_CELLS is gone (round 2, note section 13.6): growing the box made the settle grow
# FASTER than round 1's arrival cap, so the loop could not terminate at 45 or 60 degrees.
CONS_MAX_LIMIT = 0.06       # cv04 passivity ceiling R + T <= 1.06
LEAK_BAR = 1e-3             # vacuum-arm witness: |scat/inc| at the refl probe (-60 dB)
PML_REL = 0.5               # grazing absorber gate: relative window on the a-priori 3-D absorber term |R_lat3D - 1|
PML_FLOOR_R = 2.0 * LEAK_BAR + LEAK_BAR ** 2   # = injection_term(1.0): the rig floor in R at R = 1
PML_MIN_TERM = 3.0 * PML_FLOOR_R    # a grazing bin is gated only where the a-priori |R_lat - 1| >= this
CPML_DEPTH_LADDER = (8, 16, 32)

# ---------------------------------------------------------------------------
# Arms (note section 2)
# ---------------------------------------------------------------------------
ARM_ORDER = ("te_00", "te_30", "te_45", "te_60", "tm_00", "tm_45", "tm_60")
ARM_THETA0_DEG = {"te_00": 0.0, "te_30": 30.0, "te_45": 45.0, "te_60": 60.0,
                  "tm_00": 0.0, "tm_45": 45.0, "tm_60": 60.0}
ARM_POL = {a: ("tm" if a.startswith("tm") else "te") for a in ARM_ORDER}
GRAZE_ARMS = ("graze_vac", "graze_pec", "graze_te")
BREWSTER_ARM = "tm_60"
MEEP_ARMS = ("te_00", "te_30", "te_45", "te_60", "tm_45", "tm_60")
# Note section 4.6: an arm whose a-priori lattice term (mean |ideal lattice -
# Fresnel| over its gated band) sits within LATTICE_MARGIN_MIN of its mean
# window at dx runs at dx/2 as its PRIMARY recipe (cv23 section 13.3:
# resolution, not tolerance; the windows are unchanged). Filled by
# ``primary_dx_div`` from the numbers, pinned here so the note can quote it.
LATTICE_MARGIN_MIN = 1.5
ARM_DX_DIV = {"te_00": 1, "te_30": 2, "te_45": 2, "te_60": 2, "tm_00": 1, "tm_45": 2, "tm_60": 2}
MEEP_PRIMARY_RESOLUTION = 40      # px/cm (cv22/cv23's converged rung)
MEEP_A_M = 0.01
MEEP_COURANT = 0.5
MEEP_CENTER_OFFSET_CELLS = 0.5    # cv23 section 14.3: nominal block off-centre by half a pixel
                                  # holds exactly d/(a/res) integer-position nodes
MEEP_FALSIFIER_ARM = "te_45"
MEEP_FALSIFIERS = {"k_2pi": "F4: k_point in rad/a instead of 2 pi/a (k_y x 2 pi)"}
MEEP_FALSIFIER_CASE_NAMES = {f"meep_{MEEP_FALSIFIER_ARM}_{k}": k for k in MEEP_FALSIFIERS}

# --- Meep leg acceptance (note section 14, round 2). A reference leg that
# writes infinities and reports success is worse than no reference at all, so
# the producer must REFUSE to hand the case R, T it cannot vouch for. These
# are VALIDITY bounds, never agreement bounds: the leg never checks itself
# against the Fresnel oracle (that would turn an E4 disagreement into a
# silent SKIP). ---
MEEP_ACCEPT_TOL = CONS_MAX_LIMIT          # 0.06, cv04's passivity ceiling, reused as the
                                          # physicality bound on 0 <= R, T <= 1 and |R + T - 1|
MEEP_FLUX_FLOOR = 1e-9                    # |inc_flux| on a gated bin, relative to its band max:
                                          # below this the normalisation is degenerate and R, T
                                          # are 0/0 (round 1: inc_flux was IDENTICALLY zero)
MEEP_STOP_DT = 50.0                       # Meep time units between decay windows (Meep's own default use)
MEEP_DECAY_BY = 1e-3                      # the decay ratio the leg runs to


def theta_brewster_rad(eps_r: float) -> float:
    return math.atan(math.sqrt(eps_r))


def bandwidth_for(theta0_deg: float) -> float:
    """Per-arm fractional bandwidth (note section 2): the incident amplitude
    at the cutoff f_c = f0 sin theta0 must be <= CUTOFF_INC_AMP = the purity
    bar (energy with zero group velocity along x never leaves the probe and
    would sit in the tail window as "incident"), so
    (1 - sin theta0)/bw >= sqrt(ln 1000); floored to 4 decimals, capped at
    BW_MAX."""
    s = math.sin(math.radians(theta0_deg))
    bw = (1.0 - s) / CUTOFF_ARG
    bw = math.floor(bw * 1e4) / 1e4
    return float(min(BW_MAX, bw))


ARM_BW = {a: bandwidth_for(ARM_THETA0_DEG[a]) for a in ARM_ORDER}
GRAZE_BW = bandwidth_for(GRAZE_THETA0_DEG)


def ky_from(f0_hz: float, theta0_deg: float) -> float:
    """The fixed transverse wavenumber of a run (rad/m), |k_y| = k0(f0) sin theta0."""
    return TWO_PI * f0_hz / C0 * math.sin(math.radians(theta0_deg))


def realized_theta_rad(f_hz, ky: float):
    """theta(f) = asin(k_y c/(2 pi f)); NaN where the wave is evanescent."""
    f = np.asarray(f_hz, dtype=float)
    s = ky * C0 / (TWO_PI * f)
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(s) <= 1.0, np.arcsin(np.clip(s, -1.0, 1.0)), np.nan)


def cutoff_hz(ky: float) -> float:
    return ky * C0 / TWO_PI


def incident_amp_rel(f_hz, f0_hz: float, bw: float):
    """Spectral amplitude of the aux grid's complex modulated Gaussian
    exp(-j 2 pi f0 (t - t0)) exp(-((t - t0)/tau)^2), tau = 1/(pi f0 bw),
    relative to its peak: exp(-((f - f0)/(f0 bw))^2)."""
    f = np.asarray(f_hz, dtype=float)
    return np.exp(-((f - f0_hz) / (f0_hz * bw)) ** 2)


# ---------------------------------------------------------------------------
# Meep k_point mapping (note section 7). Meep: "The k_point vector is
# specified in Cartesian coordinates in units of 2 pi / distance" with the
# Bloch phase exp(i k . r); with a = MEEP_A_M per Meep unit, k_meep = k a/(2 pi).
# ---------------------------------------------------------------------------

def meep_k_point(f0_hz: float, theta0_deg: float, a_m: float = MEEP_A_M) -> tuple[float, float, float]:
    ky = ky_from(f0_hz, theta0_deg)
    return (0.0, ky * a_m / TWO_PI, 0.0)


def ky_from_meep_k_point(k_point, a_m: float = MEEP_A_M) -> float:
    return float(k_point[1]) * TWO_PI / a_m


def meep_k_point_wrong_2pi(f0_hz: float, theta0_deg: float, a_m: float = MEEP_A_M):
    """F4: the k_point taken in rad/a (k a) instead of 2 pi/a units."""
    ky = ky_from(f0_hz, theta0_deg)
    return (0.0, ky * a_m, 0.0)


def meep_fwidth_for(bw: float, f0_hz: float) -> float:
    """Meep GaussianSource(fwidth) whose spectrum equals the rfx aux source's:
    Meep's envelope exp(-(t-t0)^2/(2 w^2)), w = 1/fwidth, has amplitude
    spectrum exp(-2 pi^2 (f - fcen)^2 / fwidth^2); equate to
    exp(-((f - f0)/(bw f0))^2) -> fwidth = sqrt(2) pi bw f0 (Hz)."""
    return math.sqrt(2.0) * math.pi * bw * f0_hz


# ---------------------------------------------------------------------------
# Analytic oracle: single slab at oblique incidence, TE / TM, complex eps, mu
# ---------------------------------------------------------------------------

def _kx(f_hz, ky: float, eps, mu):
    """k_x = sqrt(eps mu k0^2 - k_y^2) on the branch with Im k_x <= 0
    (decaying for e^{-j k_x x})."""
    k0 = TWO_PI * np.asarray(f_hz, dtype=float) / C0
    kx = np.sqrt((eps * mu) * k0 ** 2 - ky ** 2 + 0j)
    return np.where(kx.imag > 0, -kx, kx)


def slab_rt(f_hz, ky: float, eps_slab, d_m: float, pol: str, mu_slab=1.0):
    """Complex amplitude (r, t) of a slab (eps_slab, mu_slab, thickness d) in
    vacuum for a wave with transverse wavenumber k_y (fixed) at frequency f.
    TE (E perpendicular to the plane of incidence): r12 = (mu2 k1 - mu1 k2)/(mu2 k1 + mu1 k2).
    TM (H perpendicular):                            r12 = (eps2 k1 - eps1 k2)/(eps2 k1 + eps1 k2).
    t is referenced to the two slab faces. R = |r|^2, T = |t|^2 (same medium
    on both sides)."""
    k1 = _kx(f_hz, ky, 1.0, 1.0)
    k2 = _kx(f_hz, ky, eps_slab, mu_slab)
    if pol == "te":
        r12 = (mu_slab * k1 - k2) / (mu_slab * k1 + k2)
    elif pol == "tm":
        r12 = (eps_slab * k1 - k2) / (eps_slab * k1 + k2)
    else:
        raise ValueError(pol)
    ph = np.exp(-2j * k2 * d_m)
    den = 1.0 - r12 ** 2 * ph
    r = r12 * (1.0 - ph) / den
    t = (1.0 - r12 ** 2) * np.exp(-1j * k2 * d_m) / den
    return r, t


def slab_RT(f_hz, ky: float, eps_slab, d_m: float, pol: str, mu_slab=1.0):
    r, t = slab_rt(f_hz, ky, eps_slab, d_m, pol, mu_slab)
    return np.abs(r) ** 2, np.abs(t) ** 2


def oracle_RT(f_hz, ky: float, pol: str, eps_slab=EPS_R_SLAB, d_m=D_SLAB_M):
    """The DECLARED oracle of an arm: the eps-slab's TE or TM Fresnel R, T at
    the realized angle of every bin (NaN below cutoff)."""
    R, T = slab_RT(f_hz, ky, eps_slab, d_m, pol)
    ok = np.isfinite(realized_theta_rad(f_hz, ky))
    return np.where(ok, R, np.nan), np.where(ok, T, np.nan)


def rfx_slab_materials(pol: str, eps_slab=EPS_R_SLAB) -> tuple[float, float]:
    """(eps_r, mu_r) rfx puts in the slab cells for an arm: the eps-slab for
    TE; the dual mu-slab for TM (note section 2)."""
    return (eps_slab, 1.0) if pol == "te" else (1.0, eps_slab)


# ---------------------------------------------------------------------------
# The 2-D Yee lattice at fixed k_y (note section 3)
# ---------------------------------------------------------------------------

def yee_omega_hat(f_hz, dt: float):
    return 2.0 * np.sin(TWO_PI * np.asarray(f_hz, dtype=float) * dt / 2.0) / dt


def yee_Ky(ky: float, dx: float) -> float:
    """|(e^{-j k_y dx} - 1)/dx| -- the exact transverse Bloch difference."""
    return 2.0 * math.sin(ky * dx / 2.0) / dx


def yee_kx(f_hz, ky: float, eps, mu, dx: float, dt: float):
    """Numerical k_x on the 2-D Yee lattice at fixed k_y:
    (2 sin(k_x dx/2)/dx)^2 + (2 sin(k_y dx/2)/dx)^2 = eps mu (2 sin(w dt/2)/(c dt))^2.
    Branch Im k_x <= 0."""
    wh = yee_omega_hat(f_hz, dt)
    Ky = yee_Ky(ky, dx)
    arg = (dx / 2.0) * np.sqrt((eps * mu) * (wh / C0) ** 2 - Ky ** 2 + 0j)
    kx = (2.0 / dx) * np.arcsin(arg)
    return np.where(kx.imag > 0, -kx, kx)


def yee_vgx(f_hz, ky: float, eps, mu, dx: float, dt: float):
    """d omega / d k_x on the 2-D Yee lattice at FIXED k_y (m/s).

    Differentiating the lattice dispersion relation K_x^2 + K_y^2 = eps mu W^2
    -- K_x = 2 sin(k_x dx/2)/dx, K_y = 2 sin(k_y dx/2)/dx,
    W = 2 sin(w dt/2)/(c dt) -- at fixed k_y gives

        v_gx = c K_x cos(k_x dx/2) / (eps mu W cos(w dt/2)).

    In the continuum limit K_x -> k_x, W -> w/c and this is c cos(theta)/n:
    the x group velocity VANISHES at the cutoff f_c = k_y c / 2 pi, which is
    why an oblique record is not the normal-incidence record times a constant.
    """
    f = np.asarray(f_hz, dtype=float)
    wh = yee_omega_hat(f, dt)          # = c W, with W = 2 sin(w dt/2)/(c dt)
    W = wh / C0
    Ky = yee_Ky(ky, dx)                # noqa: F841  (named for the relation quoted above)
    kx = yee_kx(f, ky, eps, mu, dx, dt)
    Kx = 2.0 * np.sin(kx * dx / 2.0) / dx
    with np.errstate(divide="ignore", invalid="ignore"):
        v = C0 * (Kx * np.cos(kx * dx / 2.0)) / ((eps * mu) * W * np.cos(TWO_PI * f * dt / 2.0))
    v = np.asarray(v)
    return np.where(np.abs(v.imag) <= 1e-9 * np.maximum(np.abs(v.real), 1e-300), v.real, np.nan)


def slab_rt_with_k(k1, k2, eps_slab, mu_slab, d_m: float, pol: str):
    """The slab transfer matrix with EXPLICIT k_x in each medium (used with the
    lattice k_x to isolate the numerical-dispersion / anisotropy term)."""
    if pol == "te":
        r12 = (mu_slab * k1 - k2) / (mu_slab * k1 + k2)
    else:
        r12 = (eps_slab * k1 - k2) / (eps_slab * k1 + k2)
    ph = np.exp(-2j * k2 * d_m)
    den = 1.0 - r12 ** 2 * ph
    return r12 * (1.0 - ph) / den, (1.0 - r12 ** 2) * np.exp(-1j * k2 * d_m) / den


def dispersion_term(f_hz, ky: float, pol: str, dx: float = DX_M, dt: float = DT_S,
                    eps_slab=EPS_R_SLAB, d_m=D_SLAB_M) -> dict:
    """W_disp(f): |R_TMM(k_num) - R_TMM(k)| and the same for T, plus the slab
    round-trip phase error 2 (k_x2,num - k_x2) d (rad). The lattice k_x of
    each medium is put through the same transfer matrix -- the bulk
    numerical dispersion (anisotropic in theta) without the interface-node
    term, which the exact lattice solution below carries."""
    eps_s, mu_s = rfx_slab_materials(pol, eps_slab)
    k1 = _kx(f_hz, ky, 1.0, 1.0)
    k2 = _kx(f_hz, ky, eps_s, mu_s)
    k1n = yee_kx(f_hz, ky, 1.0, 1.0, dx, dt)
    k2n = yee_kx(f_hz, ky, eps_s, mu_s, dx, dt)
    # the rfx mu-slab is TE on (eps=1, mu=eps_slab); its transfer matrix is the
    # same function with mu in the interface coefficient
    p = "te" if pol == "te" else "te"
    r, t = slab_rt_with_k(k1, k2, eps_s, mu_s, d_m, p)
    rn, tn = slab_rt_with_k(k1n, k2n, eps_s, mu_s, d_m, p)
    return {"W_R": np.abs(np.abs(rn) ** 2 - np.abs(r) ** 2),
            "W_T": np.abs(np.abs(tn) ** 2 - np.abs(t) ** 2),
            "phase_err_rad": 2.0 * (k2n - k2).real * d_m,
            "kx1": k1, "kx2": k2, "kx1_num": k1n, "kx2_num": k2n}


def cpml_profile_np(n_layers: int, dt: float, dx: float, order: int = CPML_ORDER,
                    kappa_max: float = CPML_KAPPA_MAX, R_asymptotic: float = CPML_R_ASYMPTOTIC):
    """rfx/boundaries/cpml.py _cpml_profile, re-derived in numpy (float64;
    rfx rounds to float32 at the CPMLParams boundary). Index 0 is the OUTER
    cell (sigma_max) -- the lo-face order; the hi face is the flip."""
    d = n_layers * dx
    sigma_max = -math.log(R_asymptotic) * (order + 1) / (2.0 * ETA_0 * d) * kappa_max
    rho = 1.0 - np.arange(n_layers, dtype=float) / max(n_layers - 1, 1)
    sigma = sigma_max * rho ** order
    kappa = 1.0 + (kappa_max - 1.0) * rho ** order
    alpha = CPML_ALPHA_MAX * (1.0 - rho)
    denom = sigma * kappa + kappa ** 2 * alpha
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS_0)
    c = np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
    return {"sigma": sigma, "kappa": kappa, "alpha": alpha, "b": b, "c": c}


def cpml_continuum_reflection(theta_rad, R_asymptotic: float = CPML_R_ASYMPTOTIC):
    """Amplitude reflection of the CONTINUUM polynomial-graded PML at angle
    theta: R_asym^{cos theta} (the sigma integral is depth-independent by
    construction of sigma_max)."""
    return R_asymptotic ** np.cos(np.asarray(theta_rad, dtype=float))


def rig_cells(nx_interior: int, n_cpml: int = N_CPML, d_slab_m: float = D_SLAB_M,
              dx: float = DX_M, margin: int = TFSF_MARGIN, probe_off: int = PROBE_OFFSET_CELLS,
              dx_div: int = 1) -> dict:
    """cv04's cell bookkeeping (04_multilayer_fresnel.py PART 1): the grid has
    nx = nx_interior + 2 n_cpml + 1 nodes; TFSF x_lo = n_cpml + margin,
    x_hi = nx - x_lo - 1; slab cells [nx//2 - d/2dx, nx//2 + d/2dx); probes
    +-probe_off cells from the slab faces. ``dx_div = K`` refines the SAME
    rig in cells (dx/K; interior, absorber, margin and probe offsets x K;
    the aux grid's own constants are NOT scaled -- tfsf_2d fixes them)."""
    K = int(dx_div)
    if K != 1:
        nx_interior, n_cpml, dx, margin, probe_off = nx_interior * K, n_cpml * K, dx / K, margin * K, probe_off * K
    nx = int(nx_interior) + 2 * n_cpml + 1
    x_lo = n_cpml + margin
    x_hi = nx - x_lo - 1
    half = int(d_slab_m / (2 * dx))
    slab_lo = nx // 2 - half
    slab_hi = nx // 2 + half
    return {"nx": nx, "n_cpml": n_cpml, "dx": dx, "dx_div": K, "x_lo": x_lo, "x_hi": x_hi, "slab_lo": slab_lo,
            "slab_hi": slab_hi, "probe_refl": slab_lo - probe_off, "probe_trans": slab_hi + probe_off,
            "dist_cpml_hi": nx - n_cpml - (slab_hi + probe_off), "dist_cpml_lo": (slab_lo - probe_off) - n_cpml,
            "aux_src_to_x_lo": 55 - 33}   # tfsf_2d: i0_x = 30 + 25 maps to x_lo; src_x = 30 + 3


AUX_N_CPML = 30          # rfx/sources/tfsf_2d.py:181  n_cpml_2d
AUX_N_MARGIN = 25        # tfsf_2d.py:183  n_margin_x
AUX_SRC_OFFSET = 3       # tfsf_2d.py:193  src_x = n_cpml_2d + 3
AUX_CPML_ORDER = 4       # tfsf_2d.py:198
AUX_CPML_KAPPA_MAX = 7.0  # tfsf_2d.py:199
AUX_SIGMA_FACTOR = 0.8   # tfsf_2d.py:201  sigma_max = 0.8 (m+1)/(eta dx) kappa_max


def aux_cells(cells: dict) -> dict:
    """The 2-D auxiliary grid's x bookkeeping (init_tfsf_2d): n2x = 30 + 25 +
    (x_hi - x_lo + 2) + 25 + 30, i0_x = 55 maps to x_lo, source at 33."""
    n2x = AUX_N_CPML + AUX_N_MARGIN + (cells["x_hi"] - cells["x_lo"] + 2) + AUX_N_MARGIN + AUX_N_CPML
    return {"n2x": n2x, "i0_x": AUX_N_CPML + AUX_N_MARGIN, "src_x": AUX_N_CPML + AUX_SRC_OFFSET, "n_cpml": AUX_N_CPML}


def aux_cpml_profile_np(dt: float, dx: float):
    """tfsf_2d.py:197-208 verbatim in numpy: 4th order, kappa_max 7,
    sigma_max = 0.8 (m+1)/(eta dx) kappa_max, alpha = 0.05 (1 - rho); index 0 outer."""
    n = AUX_N_CPML
    sigma_max = AUX_SIGMA_FACTOR * (AUX_CPML_ORDER + 1) / (ETA_0 * dx) * AUX_CPML_KAPPA_MAX
    rho = 1.0 - np.arange(n, dtype=float) / max(n - 1, 1)
    sigma = sigma_max * rho ** AUX_CPML_ORDER
    kappa = 1.0 + (AUX_CPML_KAPPA_MAX - 1.0) * rho ** AUX_CPML_ORDER
    alpha = CPML_ALPHA_MAX * (1.0 - rho)
    denom = sigma * kappa + kappa ** 2 * alpha
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS_0)
    c = np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
    return {"sigma": sigma, "kappa": kappa, "alpha": alpha, "b": b, "c": c}


def aux_lattice_field(f_hz, ky: float, cells: dict, *, dx: float = DX_M, dt: float = DT_S) -> np.ndarray:
    """EXACT time-harmonic field of the 2-D auxiliary grid (tfsf_2d.py) at
    fixed k_y: the same TMz lattice with its OWN CFS-CPML on both x ends and
    a unit soft source at src_x (shape (n_freq, n2x)). This is the incident
    field the Bloch TFSF actually injects -- the +x plane wave PLUS the aux
    absorber's own echoes (a -x component from its hi end), which the compact
    box does not time-gate out."""
    from scipy.linalg import solve_banded
    ac = aux_cells(cells)
    n2x, n = ac["n2x"], ac["n_cpml"]
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    wh = yee_omega_hat(f, dt)
    Ky = yee_Ky(ky, dx)
    z = np.exp(1j * TWO_PI * f * dt)
    prof = aux_cpml_profile_np(dt, dx)
    kappa = np.ones(n2x); b = np.zeros(n2x); c = np.zeros(n2x)
    kappa[:n], b[:n], c[:n] = prof["kappa"], prof["b"], prof["c"]
    kappa[n2x - n:], b[n2x - n:], c[n2x - n:] = prof["kappa"][::-1], prof["b"][::-1], prof["c"][::-1]
    in_pml = (np.arange(n2x) < n) | (np.arange(n2x) >= n2x - n)
    out = np.empty((f.size, n2x), complex)
    for m in range(f.size):
        Sinv = np.where(in_pml, 1.0 / kappa + c / (1.0 - b / z[m]), 1.0 + 0j)
        jw = 1j * wh[m]
        eps_t = EPS_0 - Ky ** 2 / (wh[m] ** 2 * MU_0)
        a = Sinv / (jw * MU_0 * dx)
        a_prev = np.concatenate([[0.0], a[:-1]])
        diag = jw * eps_t + Sinv * (a + a_prev) / dx
        sub = -Sinv[1:] * a_prev[1:] / dx
        sup = -Sinv[:-1] * a[:-1] / dx
        rhs = np.zeros(n2x, complex); rhs[ac["src_x"]] = 1.0
        ab = np.zeros((3, n2x), complex); ab[0, 1:] = sup; ab[1, :] = diag; ab[2, :-1] = sub
        out[m] = solve_banded((1, 1), ab, rhs)
    return out


def yee_lattice_full(f_hz, ky: float, cells: dict, *, eps_slab: float = 1.0, mu_slab: float = 1.0,
                     dx: float = DX_M, dt: float = DT_S, n_cpml: int | None = None,
                     cpml_kwargs: dict | None = None, ideal_absorber: bool = False,
                     pec: bool = False, aux: str = "model") -> dict:
    """EXACT time-harmonic solution of the rfx 2-D TMz Yee lattice at fixed
    k_y on the cv04 rig -- E nodes 0..nx-1 (E_nx = 0, the zero-padded forward
    difference), Hy links i+1/2 (Hy_{-1/2} = 0), Hx at the nodes, the slab's
    (eps, mu) on nodes [slab_lo, slab_hi), the CPML recursion on both x faces
    exactly as apply_cpml_e/h realize it, and the TFSF face corrections as
    forcing terms with the aux grid's own lattice plane wave as the incident
    field (note section 3).

    With z = e^{j w dt}, w_hat = 2 sin(w dt/2)/dt, K_y = 2 sin(k_y dx/2)/dx and
    the recursive-convolution transfer function S_i^{-1} = 1/kappa_i +
    c_i/(1 - b_i/z) (1 outside the absorber):

        j w_hat mu_i  Hy_{i+1/2} = S_i^{-1} (E_{i+1} - E_i)/dx      (+ TFSF at x_lo-1, x_hi)
        j w_hat mu_i  Hx_i       = -D+ E_i,   D+ = (e^{-j k_y dx} - 1)/dx
        j w_hat eps~_i E_i       = S_i^{-1} (Hy_{i+1/2} - Hy_{i-1/2})/dx   (+ TFSF at x_lo, x_hi+1)
        eps~_i = eps_i - K_y^2 / (w_hat^2 mu_i)

    -- a tridiagonal system per frequency. Returns the total field at the
    two probes, the incident there, and R = |E_scat/E_inc|^2 at the refl
    probe, T = |E_tot/E_inc|^2 at the trans probe -- exactly what the run
    measures once settled. ``ideal_absorber=True`` replaces the CPML by an
    outgoing-wave termination (the march of the semi-infinite lattice; the
    absorber term is then absent) -- the reference the compact-box arms are
    read against. ``pec=True`` pins E = 0 on the slab nodes (the hard PEC
    the graze_pec arm applies after every E update). ``aux="model"`` (the
    default) drives the faces with the aux grid's EXACT field
    (``aux_lattice_field``: the injected incident includes the aux absorber's
    own echoes) and normalizes by that field at the probes, as the run does;
    ``aux="plane"`` uses the ideal unit +x lattice plane wave."""
    from scipy.linalg import solve_banded

    nx = cells["nx"]
    # ``n_cpml`` is the DECLARED depth at dx; a refined rig (cells["dx_div"] = K)
    # runs K times as many absorber cells (rig_cells scales cells["n_cpml"] the same way)
    n = int(cells["n_cpml"]) if n_cpml is None else int(n_cpml) * int(cells.get("dx_div", 1))
    x_lo, x_hi = cells["x_lo"], cells["x_hi"]
    p_r, p_t = cells["probe_refl"], cells["probe_trans"]
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    wh = yee_omega_hat(f, dt)
    Ky = yee_Ky(ky, dx)
    z = np.exp(1j * TWO_PI * f * dt)

    eps = np.full(nx, EPS_0)
    mu = np.full(nx, MU_0)
    eps[cells["slab_lo"]:cells["slab_hi"]] = EPS_0 * eps_slab
    mu[cells["slab_lo"]:cells["slab_hi"]] = MU_0 * mu_slab

    # per-node CPML coefficient arrays (lo face: outer cell first; hi face flipped)
    kappa = np.ones(nx); b = np.zeros(nx); c = np.zeros(nx)
    if not ideal_absorber and n > 0:
        prof = cpml_profile_np(n, dt, dx, **(cpml_kwargs or {}))
        kappa[:n], b[:n], c[:n] = prof["kappa"], prof["b"], prof["c"]
        kappa[nx - n:], b[nx - n:], c[nx - n:] = prof["kappa"][::-1], prof["b"][::-1], prof["c"][::-1]
    in_pml = (np.arange(nx) < n) | (np.arange(nx) >= nx - n) if (not ideal_absorber and n > 0) else np.zeros(nx, bool)

    kx = yee_kx(f, ky, 1.0, 1.0, dx, dt)          # the aux grid's vacuum lattice wavenumber
    if aux == "model":
        E_aux = aux_lattice_field(f, ky, cells, dx=dx, dt=dt)
        ac = aux_cells(cells)
        off = ac["i0_x"] - x_lo                   # aux index of 3-D node i is i + off
    elif aux != "plane":
        raise ValueError(aux)
    R = np.empty(f.size); T = np.empty(f.size)
    E_pr = np.empty(f.size, complex); E_pt = np.empty(f.size, complex)
    Ei_pr = np.empty(f.size, complex); Ei_pt = np.empty(f.size, complex)
    idx = np.arange(nx)
    for m in range(f.size):
        Sinv = np.where(in_pml, 1.0 / kappa + c / (1.0 - b / z[m]), 1.0 + 0j)
        jw = 1j * wh[m]
        eps_t = eps - Ky ** 2 / (wh[m] ** 2 * mu)
        a = Sinv / (jw * mu * dx)                                   # link coefficient, link i -> i+1
        if aux == "model":
            Einc = E_aux[m][idx + off]                             # the aux field at every 3-D node
        else:
            Einc = np.exp(-1j * kx[m] * (idx - x_lo) * dx)         # unit +x lattice plane wave at x_lo
        Hinc = (np.roll(Einc, -1) - Einc) / (jw * MU_0 * dx)       # H_{i+1/2} (aux interior: no CPML there)
        FH = np.zeros(nx, complex)
        FH[x_lo - 1] = -Einc[x_lo] / (jw * MU_0 * dx)
        FH[x_hi] = Einc[x_hi + 1] / (jw * MU_0 * dx)
        FE = np.zeros(nx, complex)
        FE[x_lo] = -Hinc[x_lo - 1] / dx
        FE[x_hi + 1] = Hinc[x_hi] / dx
        a_prev = np.concatenate([[0.0], a[:-1]])
        FH_prev = np.concatenate([[0.0], FH[:-1]])
        diag = jw * eps_t + Sinv * (a + a_prev) / dx
        sub = -Sinv[1:] * a_prev[1:] / dx           # row i, column i-1
        sup = -Sinv[:-1] * a[:-1] / dx              # row i, column i+1
        rhs = FE + Sinv * (FH - FH_prev) / dx
        if ideal_absorber:
            # outgoing-wave termination on the lattice at both ends: beyond node 0 a
            # -x wave, E_{-1} = E_0 e^{-j kx dx}; beyond node nx-1 a +x wave,
            # E_nx = E_{nx-1} e^{-j kx dx}; both fold into the diagonal.
            ph = np.exp(-1j * kx[m] * dx)
            a_m1 = 1.0 / (jw * mu[0] * dx)
            diag = diag.copy()
            diag[0] = jw * eps_t[0] + Sinv[0] * (a[0] + a_m1 * (1.0 - ph)) / dx
            diag[-1] = jw * eps_t[-1] + Sinv[-1] * (a[-1] * (1.0 - ph) + a_prev[-1]) / dx
        if pec:
            lo, hi = cells["slab_lo"], cells["slab_hi"]
            diag = diag.copy(); rhs = rhs.copy(); sub = sub.copy(); sup = sup.copy()
            diag[lo:hi] = 1.0; rhs[lo:hi] = 0.0
            sub[lo - 1:hi - 1] = 0.0      # rows lo..hi-1, column i-1
            sup[lo:hi] = 0.0              # rows lo..hi-1, column i+1
        ab = np.zeros((3, nx), complex)
        ab[0, 1:] = sup
        ab[1, :] = diag
        ab[2, :-1] = sub
        E = solve_banded((1, 1), ab, rhs)
        E_pr[m], E_pt[m] = E[p_r], E[p_t]
        Ei_pr[m], Ei_pt[m] = Einc[p_r], Einc[p_t]
        R[m] = abs((E[p_r] - Einc[p_r]) / Einc[p_r]) ** 2
        T[m] = abs(E[p_t] / Einc[p_t]) ** 2
    return {"R": R, "T": T, "r_amp": np.sqrt(R), "E_probe_refl": E_pr, "E_probe_trans": E_pt,
            "E_inc_refl": Ei_pr, "E_inc_trans": Ei_pt, "kx_lattice": kx}


# ---------------------------------------------------------------------------
# Windows (note section 4): cv04's committed envelope through the shared
# gate policy, plus the two named terms of this rig.
# ---------------------------------------------------------------------------
# tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[...].observed_baseline
# and the per-bin max|R+T-1| pinned at validation/crossval/04_multilayer_fresnel.py:309.
CV04_ENVELOPE = {"mean_dR": 0.0066, "mean_dT": 0.011, "per_bin_max_RT_closure": 0.0487}
W_BIN = gate_from_envelope(CV04_ENVELOPE["per_bin_max_RT_closure"], quantum=1000)   # 0.074
W_MEAN_R = gate_from_envelope(CV04_ENVELOPE["mean_dR"], quantum=1000)               # 0.010
W_MEAN_T = gate_from_envelope(CV04_ENVELOPE["mean_dT"], quantum=1000)               # 0.017


def injection_term(R_or_T, leak_bar: float = LEAK_BAR):
    """W_inj: a leakage field of relative amplitude <= LEAK_BAR adds
    coherently to the scattered / transmitted field: |sqrt(X) + L|^2 - X <=
    2 sqrt(X) L + L^2."""
    X = np.clip(np.asarray(R_or_T, dtype=float), 0.0, None)
    return 2.0 * np.sqrt(X) * leak_bar + leak_bar ** 2


# ---------------------------------------------------------------------------
# Record length (note section 6) and gated bins (section 5)
# ---------------------------------------------------------------------------

def arm_spec(arm: str) -> dict:
    if arm in ARM_ORDER:
        th = ARM_THETA0_DEG[arm]; pol = ARM_POL[arm]; bw = ARM_BW[arm]; nx = NX_INTERIOR
        gate = (0.0, THETA_GATE_MAX_DEG); slab = True
    elif arm in GRAZE_ARMS:
        th = GRAZE_THETA0_DEG; pol = "te"; bw = GRAZE_BW; nx = NX_INTERIOR_GRAZE
        gate = GRAZE_THETA_GATE_DEG; slab = (arm == "graze_te")
    else:
        raise KeyError(arm)
    eps_s, mu_s = rfx_slab_materials(pol) if slab else (1.0, 1.0)
    return {"arm": arm, "theta0_deg": th, "pol": pol, "bw": bw, "nx_interior": nx, "theta_gate_deg": gate,
            "slab": slab, "pec": arm == "graze_pec", "eps_slab_rfx": eps_s, "mu_slab_rfx": mu_s,
            "ky": ky_from(TFSF_F0_HZ, th),
            "f_cutoff_hz": cutoff_hz(ky_from(TFSF_F0_HZ, th)), "f0_hz": TFSF_F0_HZ,
            "compact": arm in GRAZE_ARMS}


def gated_mask(f_hz, spec: dict):
    """Gated bins: incident amplitude >= GATED_INC_AMP_FRAC of peak, the
    realized angle inside the arm's gate range, propagating."""
    th = np.degrees(realized_theta_rad(f_hz, spec["ky"]))
    w = incident_amp_rel(f_hz, spec["f0_hz"], spec["bw"])
    lo, hi = spec["theta_gate_deg"]
    return np.isfinite(th) & (w >= GATED_INC_AMP_FRAC) & (th >= lo - 1e-9) & (th <= hi + 1e-9)


def band_edges(spec: dict) -> dict:
    """The 10 % incident band and the realized angles at its edges (before the gate cap)."""
    f0, bw = spec["f0_hz"], spec["bw"]
    half = f0 * bw * math.sqrt(-math.log(GATED_INC_AMP_FRAC))
    f_lo, f_hi = f0 - half, f0 + half
    th_lo = float(np.degrees(realized_theta_rad(f_hi, spec["ky"])))
    th_hi_raw = realized_theta_rad(max(f_lo, spec["f_cutoff_hz"] * (1 + 1e-12)), spec["ky"])
    th_hi = float(np.degrees(th_hi_raw)) if np.isfinite(th_hi_raw) else 90.0
    return {"f_lo_hz": f_lo, "f_hi_hz": f_hi, "theta_at_f_hi_deg": th_lo, "theta_at_f_lo_deg": th_hi,
            "f_cutoff_hz": spec["f_cutoff_hz"], "inc_amp_at_cutoff": float(incident_amp_rel(spec["f_cutoff_hz"], f0, bw))}


def slab_ringdown_rate(f_hz, spec: dict):
    """Amplitude decay rate (1/s) of the slab etalon at the realized angle:
    -ln|r12|^2 / t_rt with t_rt = 2 d n / (c cos theta_t) -- the round-trip
    group delay along x at FIXED k_y (v_gx = (c/n) cos theta_t)."""
    eps_s, mu_s = spec["eps_slab_rfx"], spec["mu_slab_rfx"]
    k1 = _kx(f_hz, spec["ky"], 1.0, 1.0)
    k2 = _kx(f_hz, spec["ky"], eps_s, mu_s)
    r12 = (mu_s * k1 - k2) / (mu_s * k1 + k2)        # rfx slab is always TE on (eps, mu)
    n2 = math.sqrt(eps_s * mu_s)
    k0 = TWO_PI * np.asarray(f_hz, dtype=float) / C0
    cos_t = (k2 / (n2 * k0)).real
    t_rt = 2.0 * D_SLAB_M * n2 / (C0 * cos_t)
    rho = np.abs(r12) ** 2
    with np.errstate(divide="ignore"):
        return -np.log(rho) / t_rt, t_rt, rho


# ---------------------------------------------------------------------------
# Record length -- DERIVED from the exact lattice (note section 13, round 2)
#
# Round 1 built the record from t_pulse + L / (c cos theta_hi) with theta_hi
# the upper edge of the GATED band, and capped it at the arrival of the first
# absorber echo of the FASTEST gated component.  Both halves are wrong at
# oblique incidence:
#
#   * the witness is a broadband time-domain max over the raw probe samples,
#     so it is not the gated band that has to have cleared the probes but
#     every component above the WITNESS bar -- and at fixed k_y those live
#     arbitrarily close to the cutoff, where v_gx -> 0 (``yee_vgx``).  The
#     measured settling angle is ~65 deg on the 30 deg arm and ~80 deg on the
#     45 deg arm, against the 44.6 / 58.3 deg the round-1 law assumed;
#   * the echo the arrival cap gates out is the echo of the FASTEST gated
#     component, whose CPML reflection at 37 deg is ~1e-10 and could never
#     move a witness.  Gating on its ARRIVAL, not its AMPLITUDE, cut every
#     45 / 60 deg arm off before it had settled, and growing the box does not
#     help: the settle grows with the box at 1/v_slow while the cap grows at
#     2.9/v_fast, and v_slow/v_fast < 1/2.9 on those arms.
#
# The round-2 law computes the record instead of estimating it.  The three
# witnesses of the case (``_witness``) are LINEAR functionals of the aux
# source, so each probe time series is the inverse transform of (source
# spectrum) x (exact lattice transfer function at that probe) -- and the
# lattice transfer function, absorber and aux grid included, is exactly what
# ``yee_lattice_full`` returns.  No FDTD is involved; the bars are unchanged.
# ---------------------------------------------------------------------------
RECORD_NFFT = 1 << 17        # 131072 samples: > 4x the longest predicted record
RECORD_AMP_FLOOR = 1e-9      # source bins kept (relative amplitude); 1e-9 is 6 decades
                             # under the tightest witness bar (TAIL_PURITY_LIMIT = 1e-3)


def record_probe_series(spec: dict, *, nx_interior: int | None = None, dx_div: int = 1,
                        n_cpml: int = N_CPML, nfft: int = RECORD_NFFT,
                        amp_floor: float = RECORD_AMP_FLOOR, ideal_absorber: bool = False) -> dict:
    """The four probe time series the case records (total / incident at both
    probes), computed EXACTLY from the lattice: ifft(S(f) H(f)) with S the aux
    source spectrum and H the lattice transfer function at that probe.

    The case DFTs ``conj(x)`` so that the carrier lands at +f0; everything
    here is built in that conjugated domain, which is why the source is
    ``exp(+j 2 pi f0 (t - t0))`` and only positive-frequency bins are kept."""
    K = int(dx_div)
    dt = DT_S / K
    nx_int = int(spec["nx_interior"] if nx_interior is None else nx_interior)
    cells = rig_cells(nx_int, n_cpml, dx_div=K)
    eps_s, mu_s = rfx_slab_materials(spec["pol"]) if spec["slab"] else (1.0, 1.0)
    tau = 1.0 / (math.pi * spec["f0_hz"] * spec["bw"])
    t0 = SRC_T0_OVER_TAU * tau
    t = np.arange(nfft) * dt
    src = np.exp(1j * TWO_PI * spec["f0_hz"] * (t - t0)) * np.exp(-(((t - t0) / tau) ** 2))
    S = np.fft.fft(src)
    fr = np.fft.fftfreq(nfft, d=dt)
    keep = (fr > 0) & (np.abs(S) > amp_floor * np.abs(S).max())
    lat = yee_lattice_full(fr[keep], spec["ky"], cells, eps_slab=eps_s, mu_slab=mu_s,
                           dx=cells["dx"], dt=dt, n_cpml=n_cpml, pec=bool(spec["pec"]),
                           ideal_absorber=ideal_absorber)
    out = {}
    for name, H in (("tot_r", lat["E_probe_refl"]), ("tot_t", lat["E_probe_trans"]),
                    ("inc_r", lat["E_inc_refl"]), ("inc_t", lat["E_inc_trans"])):
        Y = np.zeros(nfft, complex)
        Y[keep] = S[keep] * H
        out[name] = np.fft.ifft(Y)
    out.update(dt_s=dt, cells=cells, nfft=nfft, n_freq_bins=int(keep.sum()))
    return out


def _trailing_max(env: np.ndarray, tw: int) -> np.ndarray:
    """out[n] = max(env[n - tw : n]) -- the case's tail window, as a function
    of where the record ends (inf until a full window exists)."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(env.size, np.inf)
    out[tw - 1:] = sliding_window_view(env, tw).max(axis=1)
    return out


def record_witnesses(ser: dict, dx_div: int = 1) -> dict:
    """``26_oblique_slab_fresnel.py::_witness`` evaluated on the exact series,
    for every possible record end."""
    tw = TAIL_WINDOW * int(dx_div)
    inc_peak = max(np.abs(ser["inc_r"]).max(), np.abs(ser["inc_t"]).max())
    scat = ser["tot_r"] - ser["inc_r"]
    purity = np.maximum(np.abs(ser["inc_r"]), np.abs(ser["inc_t"])) / inc_peak
    return {"inc_peak": float(inc_peak), "tail_window": tw,
            "purity": _trailing_max(purity, tw),
            "refl": _trailing_max(np.abs(scat) / inc_peak, tw),
            "trans": _trailing_max(np.abs(ser["tot_t"]) / inc_peak, tw),
            "peak_step": int(np.argmax(np.abs(ser["tot_t"])))}


def absorber_window(spec: dict, e_absorber: float) -> dict:
    """The a-priori R / T term of an absorber echo of relative amplitude
    ``e_absorber`` carried inside the record, on the arm's own gated band: the
    same coherent-addition bound the injection term uses,
    |sqrt(X) + e|^2 - X = 2 sqrt(X) e + e^2, with X the DECLARED Fresnel R or
    T of the bin.  Admissible only where it stays inside the declared bin
    window W_BIN -- no window is widened."""
    if not np.isfinite(e_absorber):
        return {"W_absorber_R_max": float("nan"), "W_absorber_T_max": float("nan"), "absorber_ok": False}
    nfft = 1 << 16
    f = np.fft.rfftfreq(nfft, d=DT_S)
    g = gated_mask(f, spec)
    if not g.any():
        return {"W_absorber_R_max": float("nan"), "W_absorber_T_max": float("nan"), "absorber_ok": False}
    R_an, T_an = oracle_RT(f[g], spec["ky"], spec["pol"])
    if not spec["slab"]:
        R_an = np.ones_like(f[g]); T_an = np.ones_like(f[g])       # the PEC / vacuum arms: worst case
    wR = float(np.nanmax(injection_term(R_an, leak_bar=e_absorber)))
    wT = float(np.nanmax(injection_term(T_an, leak_bar=e_absorber)))
    return {"W_absorber_R_max": wR, "W_absorber_T_max": wT,
            "absorber_ok": bool(max(wR, wT) <= W_BIN)}


def predict_settling(spec: dict, *, nx_interior: int | None = None, dx_div: int = 1,
                     n_cpml: int = N_CPML, nfft: int = RECORD_NFFT,
                     with_absorber_term: bool = True) -> dict:
    """The DERIVED record of one arm at one rung: the first step at which all
    three witnesses sit under their (unchanged) bars, plus the a-priori
    absorber-echo term of that record.

    ``e_absorber`` is the largest probe-field difference over the record
    between the rig with its CPML and the same lattice with an outgoing-wave
    termination, relative to the incident peak -- the echo AMPLITUDE the
    record actually contains.  It enters R through the same coherent-addition
    bound the injection term uses (``injection_term``), and the record is
    admissible only where that term stays inside the declared bin window
    W_BIN: no window is widened, the arrival cap is simply replaced by the
    amplitude statement it was standing in for."""
    K = int(dx_div)
    ser = record_probe_series(spec, nx_interior=nx_interior, dx_div=K, n_cpml=n_cpml, nfft=nfft)
    w = record_witnesses(ser, K)
    ok = (w["purity"] < TAIL_PURITY_LIMIT) & (w["refl"] < SETTLING_LIMIT) & (w["trans"] < SETTLING_LIMIT)
    ok[:w["peak_step"]] = False       # before the pulse arrives the probes are trivially quiet
    idx = np.flatnonzero(ok)
    n = int(idx[0]) + 1 if idx.size else None
    out = {"arm": spec["arm"], "dx_div": K, "nx_interior": int(ser["cells"]["nx"]),
           "n_settle": n, "nfft": nfft, "n_freq_bins": ser["n_freq_bins"],
           "tail_window": w["tail_window"], "inc_peak": w["inc_peak"],
           "purity_at_settle": (float(w["purity"][n - 1]) if n else None),
           "refl_at_settle": (float(w["refl"][n - 1]) if n else None),
           "trans_at_settle": (float(w["trans"][n - 1]) if n else None)}
    if n is not None:
        dt = ser["dt_s"]
        cells = ser["cells"]
        tau = 1.0 / (math.pi * spec["f0_hz"] * spec["bw"])
        n_src_end = (SRC_T0_OVER_TAU * tau + PULSE_END_ARG_40DB * tau) / dt
        L = cells["aux_src_to_x_lo"] + (cells["probe_trans"] - cells["x_lo"])
        v_eff = L / max(n - n_src_end - w["tail_window"], 1.0)          # cells/step
        v0 = C0 * dt / cells["dx"]
        out["path_cells"] = int(L)
        out["v_eff_cells_per_step"] = float(v_eff)
        out["theta_eff_deg"] = float(np.degrees(np.arccos(np.clip(v_eff / v0, -1.0, 1.0))))
    if with_absorber_term and n is not None:
        ser_i = record_probe_series(spec, nx_interior=nx_interior, dx_div=K, n_cpml=n_cpml,
                                    nfft=nfft, ideal_absorber=True)
        e = max(float(np.abs(ser[k][:n] - ser_i[k][:n]).max()) / w["inc_peak"] for k in ("tot_r", "tot_t"))
        out["e_absorber"] = e
        out.update(absorber_window(spec, e))
    return out


# The DECLARED record of every arm and rung, keyed (arm, dx_div, n_cpml, nx_interior).
# Each entry is ``predict_settling``'s output on that rig: ``n_settle`` is the first
# step at which the case's three witnesses sit under their UNCHANGED bars
# (TAIL_PURITY_LIMIT on the incident, SETTLING_LIMIT on the scattered and the
# transmitted), and ``e_absorber`` is the largest probe-field difference over that
# record between the rig with its CPML and the same lattice with an outgoing-wave
# termination, relative to the incident peak.  ``theta_eff_deg`` is the realized
# angle whose x group velocity the record implies -- the physics readout: the record
# is set by content far outside the gated band, close to the cutoff.
# Reproduced by tests/crossval/test_cv26_oblique_fresnel_comparator.py (slow marks).
RECORD_DECLARED: dict[tuple[str, int, int, int], dict] = {
    ("te_00", 1, 20, 1500): {"n_settle": 1512, "e_absorber": 1.4737376641852275e-06, "theta_eff_deg": 14.13},
    ("tm_00", 1, 20, 1500): {"n_settle": 1512, "e_absorber": 1.5244516563353113e-06, "theta_eff_deg": 14.13},
    ("te_30", 1, 20, 1500): {"n_settle": 3094, "e_absorber": 6.554616968592558e-06, "theta_eff_deg": 64.64},
    ("te_30", 2, 20, 1500): {"n_settle": 6387, "e_absorber": 4.790797487325276e-06, "theta_eff_deg": 65.97},
    ("te_45", 1, 20, 1500): {"n_settle": 7362, "e_absorber": 0.06663583135586092, "theta_eff_deg": 80.13},
    ("te_45", 2, 20, 1500): {"n_settle": 14283, "e_absorber": 0.029841546179688764, "theta_eff_deg": 79.93},
    ("te_60", 1, 20, 1500): {"n_settle": 12811, "e_absorber": 0.03690936622405856, "theta_eff_deg": 84.22},
    ("te_60", 2, 20, 1500): {"n_settle": 26438, "e_absorber": 0.01788762793520409, "theta_eff_deg": 84.5},
    ("tm_45", 1, 20, 1500): {"n_settle": 7362, "e_absorber": 0.047689691056383224, "theta_eff_deg": 80.13},
    ("tm_45", 2, 20, 1500): {"n_settle": 14283, "e_absorber": 0.02246093995265182, "theta_eff_deg": 79.93},
    ("tm_60", 1, 20, 1500): {"n_settle": 12811, "e_absorber": 0.056103069518629214, "theta_eff_deg": 84.22},
    ("tm_60", 2, 20, 1500): {"n_settle": 26438, "e_absorber": 0.026598550957916182, "theta_eff_deg": 84.5},
    ("graze_vac", 1, 20, 100): {"n_settle": 22008, "e_absorber": 2.743083609229469e-14, "theta_eff_deg": 87.22},
    ("graze_pec", 1, 20, 100): {"n_settle": 22008, "e_absorber": 0.06697031182708263, "theta_eff_deg": 87.22},
    ("graze_te", 1, 20, 100): {"n_settle": 22008, "e_absorber": 0.05644219130054219, "theta_eff_deg": 87.22},
    ("graze_pec", 1, 8, 100): {"n_settle": 22008, "e_absorber": 0.2797267820174598, "theta_eff_deg": 87.22},
    ("graze_pec", 1, 16, 100): {"n_settle": 22008, "e_absorber": 0.09216001043621881, "theta_eff_deg": 87.22},
    ("graze_pec", 1, 32, 100): {"n_settle": 22008, "e_absorber": 0.03550201937209352, "theta_eff_deg": 87.22},
}


def derive_record(spec: dict, dt: float | None = None, *, n_cpml: int = N_CPML, nx_interior: int | None = None,
                  dx_div: int = 1) -> dict:
    """n_steps_min = n_pulse_end + n_ring + TAIL_WINDOW (cv22 section 13
    adapted to the oblique path length):
      n_pulse_end : t0 + a40 tau (the complex Gaussian at -40 dB) plus the
                    propagation from the aux source to the TRANSMISSION probe
                    at the group velocity along x of the SLOWEST gated
                    component, v_gx = c cos(theta_hi);
      n_ring      : max over the gated bins of ln(100 w(f))/rate(f), rate from
                    slab_ringdown_rate at the realized angle (the vacuum arm has
                    no etalon: n_ring covers only the absorber echo's return);
      echo        : for a compact box the absorber echo is INSIDE the record:
                    the trans-probe -> hi-CPML -> refl-probe path at v_gx(theta_hi)
                    is added (the primary rig time-gates it out instead).
    The CPML gate (first echo at the FASTEST gated component, cv04's 0.95
    rule) must exceed n_steps on the primary rig (asserted by the case)."""
    K = int(dx_div)
    dt = DT_S / K if dt is None else float(dt)
    nx_int = spec["nx_interior"] if nx_interior is None else int(nx_interior)
    cells = rig_cells(nx_int, n_cpml, dx_div=K)
    tau = 1.0 / (math.pi * spec["f0_hz"] * spec["bw"])
    t0 = SRC_T0_OVER_TAU * tau
    v0 = C0 * dt / cells["dx"]                            # cells/step at normal incidence
    nfft = 1 << 16
    f = np.fft.rfftfreq(nfft, d=dt)
    g = gated_mask(f, spec)
    if not g.any():
        raise ValueError(f"{spec['arm']}: no gated bins")
    th = realized_theta_rad(f[g], spec["ky"])
    th_hi, th_lo = float(th.max()), float(th.min())
    v_slow = v0 * math.cos(th_hi)
    v_fast = v0 * math.cos(th_lo)
    src_to_trans = cells["aux_src_to_x_lo"] + (cells["probe_trans"] - cells["x_lo"])
    n_pulse_end = int(math.ceil((t0 + PULSE_END_ARG_40DB * tau) / dt + src_to_trans / v_slow))
    w = incident_amp_rel(f[g], spec["f0_hz"], spec["bw"])
    if spec["slab"]:
        rate, t_rt, rho = slab_ringdown_rate(f[g], spec)
        t_ring = np.log(100.0 * w) / rate
        i = int(np.argmax(t_ring))
        ring = {"rate_ring_1_s": float(rate[i]), "f_ring_hz": float(f[g][i]), "theta_ring_deg": float(np.degrees(th[i])),
                "w_ring": float(w[i]), "rho_etalon": float(rho[i]), "t_rt_s": float(t_rt[i]), "t_ring_s": float(t_ring[i])}
        n_ring = int(math.ceil(max(0.0, ring["t_ring_s"]) / dt))
    else:
        ring = {"rate_ring_1_s": None, "f_ring_hz": None, "theta_ring_deg": None, "w_ring": None,
                "rho_etalon": None, "t_rt_s": None, "t_ring_s": 0.0}
        n_ring = 0
    n_echo = 0
    if spec["compact"]:
        # the transmitted wave: trans probe -> hi absorber -> back to the refl probe;
        # the reflected wave: slab front -> lo absorber -> back to the refl probe
        path_hi = (cells["x_hi"] + 1 - cells["probe_trans"]) + 2 * cells["dist_cpml_hi"] \
            + (cells["probe_trans"] - cells["probe_refl"])
        path_lo = (cells["slab_lo"] - cells["n_cpml"]) + (cells["probe_refl"] - cells["n_cpml"])
        n_echo = int(math.ceil(max(path_hi if not spec["pec"] else 0, path_lo) / v_slow))
    tail_window = TAIL_WINDOW * K
    n_closed_form = n_pulse_end + n_echo + n_ring + tail_window
    # cv04's CPML gate at the fastest gated component (first-arrival at the trans
    # probe + the round trip to the nearer absorber); irrelevant for the compact box
    n_arrive_fast = int(math.ceil((t0 - 2.0 * tau) / dt + src_to_trans / v_fast))
    t_safe = n_arrive_fast + int(2 * min(cells["dist_cpml_hi"], cells["dist_cpml_lo"]) / v_fast * 0.95)
    # --- the DECLARED record (note section 13): the exact settling step of this
    # lattice, from ``predict_settling``.  The closed form above is kept only as a
    # diagnostic -- it under-predicts by 2.0x (30 deg) to 2.7x (60 deg) because the
    # witness is broadband and the content that binds it sits near the cutoff, not
    # at the gated band edge.  ``t_safe_cpml_steps`` likewise stays as a reported
    # number; the absorber is gated by ``e_absorber``, its AMPLITUDE over the record. ---
    # the declared record belongs to the DECLARED arm; --smoke re-aims the compact
    # rig at SMOKE_THETA0_DEG and must fall back to the closed form
    _canon = arm_spec(spec["arm"])
    _same = (abs(spec["bw"] - _canon["bw"]) < 1e-12 and abs(spec["ky"] - _canon["ky"]) < 1e-9
             and bool(spec["slab"]) == bool(_canon["slab"]))
    decl = RECORD_DECLARED.get((spec["arm"], K, int(n_cpml), nx_int)) if _same else None
    if decl is not None:
        n_steps, source = int(decl["n_settle"]), "declared (predict_settling)"
        e_abs = float(decl["e_absorber"])
    else:
        n_steps, source = n_closed_form, "closed-form fallback (NOT declared)"
        e_abs = float("nan")
    aw = absorber_window(spec, e_abs)
    return {"arm": spec["arm"], "nx_interior": nx_int, "n_cpml": n_cpml, "dt_s": dt, "bw": spec["bw"],
            "src_tau_s": tau, "src_t0_s": t0, "theta_gate_lo_deg": float(np.degrees(th_lo)),
            "theta_gate_hi_deg": float(np.degrees(th_hi)), "v_cells_slow": v_slow, "v_cells_fast": v_fast,
            "n_pulse_end": n_pulse_end, "n_ring": n_ring, "n_echo": n_echo, "tail_window": tail_window,
            "n_steps": n_steps, "n_closed_form": n_closed_form, "record_source": source,
            "theta_eff_deg": (float(decl["theta_eff_deg"]) if decl else float("nan")),
            "e_absorber": e_abs, **aw,
            "t_safe_cpml_steps": int(t_safe), "settling_limit": SETTLING_LIMIT,
            "n_gated_bins_nfft65536": int(g.sum()), **ring, **cells}


# ---------------------------------------------------------------------------
# Evaluation (E2 / lattice witness / grazing gates / E4)
# ---------------------------------------------------------------------------

def evaluate_e2(freqs_hz, R_rfx, T_rfx, spec: dict, dt: float, *, tail: dict, oracle_pol: str | None = None,
                oracle_ky: float | None = None, oracle_eps: float = EPS_R_SLAB, cells: dict | None = None,
                n_cpml: int = N_CPML) -> dict:
    """The E2 gates of one arm against the DECLARED oracle (the arm's pol and
    k_y unless a falsifier judges against another declared value)."""
    f = np.asarray(freqs_hz, dtype=float)
    R_rfx = np.asarray(R_rfx, dtype=float); T_rfx = np.asarray(T_rfx, dtype=float)
    pol = spec["pol"] if oracle_pol is None else oracle_pol
    ky = spec["ky"] if oracle_ky is None else oracle_ky
    g = gated_mask(f, dict(spec, ky=ky))
    th = realized_theta_rad(f, ky)
    R_an, T_an = oracle_RT(f, ky, pol, eps_slab=oracle_eps)
    dx = DX_M if cells is None else float(cells["dx"])
    disp = dispersion_term(f, ky, pol, dx, dt, eps_slab=oracle_eps)
    w_disp_R, w_disp_T = disp["W_R"], disp["W_T"]
    w_inj_R, w_inj_T = injection_term(R_an), injection_term(T_an)
    window_R = W_BIN + w_disp_R + w_inj_R
    window_T = W_BIN + w_disp_T + w_inj_T
    dR = np.abs(R_rfx - R_an); dT = np.abs(T_rfx - T_an)
    dR_g, dT_g = dR[g], dT[g]
    mean_win_R = W_MEAN_R + float(np.nanmean(w_disp_R[g])) + float(np.nanmean(w_inj_R[g]))
    mean_win_T = W_MEAN_T + float(np.nanmean(w_disp_T[g])) + float(np.nanmean(w_inj_T[g]))
    closure = np.abs(R_rfx + T_rfx - 1.0)
    gates = {
        "G1_R": bool(np.all(dR_g <= window_R[g])), "G1_T": bool(np.all(dT_g <= window_T[g])),
        "G2_R": bool(np.nanmean(dR_g) <= mean_win_R), "G2_T": bool(np.nanmean(dT_g) <= mean_win_T),
        "G3_closure": bool(np.all(closure[g] <= W_BIN + w_disp_R[g] + w_disp_T[g] + w_inj_R[g] + w_inj_T[g])),
        "G3_passivity": bool(np.all((R_rfx + T_rfx)[g] <= 1.0 + CONS_MAX_LIMIT)),
        "G3_tail": bool(tail["ok"]),
    }
    out = {
        "arm": spec["arm"], "pol": spec["pol"], "oracle_pol": pol, "theta0_deg": spec["theta0_deg"],
        "ky_rad_m": spec["ky"], "oracle_ky_rad_m": ky, "bw": spec["bw"], "f_cutoff_hz": cutoff_hz(ky),
        "freqs_hz": f.tolist(), "theta_deg": np.degrees(th).tolist(), "gated": g.tolist(),
        "n_bins_gated": int(g.sum()),
        "R_rfx": R_rfx.tolist(), "T_rfx": T_rfx.tolist(), "R_an": R_an.tolist(), "T_an": T_an.tolist(),
        "dR": dR.tolist(), "dT": dT.tolist(), "w_disp_R": w_disp_R.tolist(), "w_disp_T": w_disp_T.tolist(),
        "w_inj_R": w_inj_R.tolist(), "w_inj_T": w_inj_T.tolist(),
        "window_R": window_R.tolist(), "window_T": window_T.tolist(),
        "phase_err_rad": np.asarray(disp["phase_err_rad"]).tolist(),
        "max_dR_gated": float(np.nanmax(dR_g)), "max_dT_gated": float(np.nanmax(dT_g)),
        "mean_dR_gated": float(np.nanmean(dR_g)), "mean_dT_gated": float(np.nanmean(dT_g)),
        "mean_window_R": mean_win_R, "mean_window_T": mean_win_T,
        "worst_bin_R_hz": float(f[g][int(np.nanargmax(dR_g))]), "worst_bin_T_hz": float(f[g][int(np.nanargmax(dT_g))]),
        "max_closure_gated": float(np.nanmax(closure[g])), "max_RT_gated": float(np.nanmax((R_rfx + T_rfx)[g])),
        "theta_gated_deg": [float(np.degrees(th[g]).min()), float(np.degrees(th[g]).max())],
        "tail": tail, "gates": gates, "e2_ok": bool(all(gates.values())),
        "W_bin": W_BIN, "W_mean_R": W_MEAN_R, "W_mean_T": W_MEAN_T, "leak_bar": LEAK_BAR,
    }
    # exact-lattice witness (reported, never gated): the declared arm's OWN lattice
    if cells is not None:
        lat = yee_lattice_full(f, spec["ky"], cells, eps_slab=spec["eps_slab_rfx"], mu_slab=spec["mu_slab_rfx"],
                               dx=dx, dt=dt, n_cpml=n_cpml, pec=spec.get("pec", False))
        lat_ideal = yee_lattice_full(f, spec["ky"], cells, eps_slab=spec["eps_slab_rfx"], mu_slab=spec["mu_slab_rfx"],
                                     dx=dx, dt=dt, ideal_absorber=True, pec=spec.get("pec", False), aux="plane")
        lat_cpml_only = yee_lattice_full(f, spec["ky"], cells, eps_slab=spec["eps_slab_rfx"], mu_slab=spec["mu_slab_rfx"],
                                         dx=dx, dt=dt, n_cpml=n_cpml, pec=spec.get("pec", False), aux="plane")
        dRl = np.abs(R_rfx - lat["R"]); dTl = np.abs(T_rfx - lat["T"])
        out["lattice"] = {
            "R_lattice": lat["R"].tolist(), "T_lattice": lat["T"].tolist(),
            "R_lattice_ideal_absorber": lat_ideal["R"].tolist(), "T_lattice_ideal_absorber": lat_ideal["T"].tolist(),
            "W_lat_R": np.abs(lat["R"] - np.nan_to_num(R_an)).tolist(), "W_lat_T": np.abs(lat["T"] - np.nan_to_num(T_an)).tolist(),
            "mean_W_lat_R_gated": float(np.nanmean(np.abs(lat["R"] - R_an)[g])),
            "mean_W_lat_T_gated": float(np.nanmean(np.abs(lat["T"] - T_an)[g])),
            "mean_dR_lattice_gated": float(dRl[g].mean()), "mean_dT_lattice_gated": float(dTl[g].mean()),
            "max_dR_lattice_gated": float(dRl[g].max()), "max_dT_lattice_gated": float(dTl[g].max()),
            "absorber_term_R_gated_max": float(np.abs(lat["R"] - lat_ideal["R"])[g].max()),
            "absorber_term_T_gated_max": float(np.abs(lat["T"] - lat_ideal["T"])[g].max()),
            "cpml3d_term_R_gated_max": float(np.abs(lat_cpml_only["R"] - lat_ideal["R"])[g].max()),
            "aux_echo_term_R_gated_max": float(np.abs(lat["R"] - lat_cpml_only["R"])[g].max()),
        }
    return out


def lattice_margin(arm: str, dx_div: int = 1) -> dict:
    """A priori: the arm's mean |ideal lattice - Fresnel| over its gated bins
    (the rfx-vs-Fresnel residual the exact lattice predicts) against its mean
    window at dx/K; margin = window / term (note section 4.6)."""
    spec = arm_spec(arm)
    K = int(dx_div)
    dt = DT_S / K
    rec = derive_record(spec, dt, dx_div=K)
    nfft = int(2 ** math.ceil(math.log2(rec["n_steps"])) * NFFT_OVERSAMPLE)
    f = np.fft.rfftfreq(nfft, d=dt)
    f = f[(f > MASK_F_LO_HZ) & (f < MASK_F_HI_HZ)]
    g = gated_mask(f, spec)
    cells = rig_cells(spec["nx_interior"], N_CPML, dx_div=K)
    lat = yee_lattice_full(f[g], spec["ky"], cells, eps_slab=spec["eps_slab_rfx"], mu_slab=spec["mu_slab_rfx"],
                           dx=cells["dx"], dt=dt, ideal_absorber=True, aux="plane")
    R_an, T_an = oracle_RT(f[g], spec["ky"], spec["pol"])
    disp = dispersion_term(f[g], spec["ky"], spec["pol"], cells["dx"], dt)
    wR = W_MEAN_R + float(np.nanmean(disp["W_R"])) + float(np.nanmean(injection_term(R_an)))
    wT = W_MEAN_T + float(np.nanmean(disp["W_T"])) + float(np.nanmean(injection_term(T_an)))
    tR = float(np.nanmean(np.abs(lat["R"] - R_an))); tT = float(np.nanmean(np.abs(lat["T"] - T_an)))
    return {"arm": arm, "dx_div": K, "n_bins": int(g.sum()), "nfft": nfft, "n_steps": rec["n_steps"],
            "term_R": tR, "term_T": tT, "window_R": wR, "window_T": wT, "margin_R": wR / tR, "margin_T": wT / tT,
            "max_term_R": float(np.nanmax(np.abs(lat["R"] - R_an))),
            "mean_W_disp_R": float(np.nanmean(disp["W_R"])), "mean_W_inj_R": float(np.nanmean(injection_term(R_an)))}


def primary_dx_div(arm: str) -> int:
    """The rule of note section 4.6 evaluated from the numbers: dx/2 iff the
    dx margin (R or T) is below LATTICE_MARGIN_MIN."""
    m = lattice_margin(arm, 1)
    return 2 if min(m["margin_R"], m["margin_T"]) < LATTICE_MARGIN_MIN else 1


def evaluate_brewster(e2: dict) -> dict:
    """The Brewster gate on the TM arm (note section 4.4): at the gated bin
    whose realized angle is nearest theta_B = atan(sqrt eps), R_TM must sit
    under the derived per-bin window (R_TM,Fresnel = 0 there); the angle of
    the measured minimum is REPORTED against theta_B."""
    f = np.asarray(e2["freqs_hz"]); th = np.asarray(e2["theta_deg"]); g = np.asarray(e2["gated"], bool)
    thB = math.degrees(theta_brewster_rad(EPS_R_SLAB))
    i_all = np.where(g)[0]
    iB = i_all[int(np.argmin(np.abs(th[i_all] - thB)))]
    R = np.asarray(e2["R_rfx"]); win = np.asarray(e2["window_R"])
    imin = i_all[int(np.argmin(R[i_all]))]
    return {"theta_brewster_deg": thB, "bin_hz": float(f[iB]), "theta_bin_deg": float(th[iB]),
            "R_rfx_at_brewster": float(R[iB]), "R_an_at_brewster": float(np.asarray(e2["R_an"])[iB]),
            "floor": float(win[iB]), "ok": bool(R[iB] <= win[iB]),
            "theta_of_measured_min_deg": float(th[imin]), "R_measured_min": float(R[imin]),
            "min_within_1deg": bool(abs(th[imin] - thB) <= 1.0)}


def evaluate_leakage(freqs_hz, R_rfx, spec: dict) -> dict:
    """The vacuum arm (note section 4.5): with no scatterer the scattered
    field at the refl probe is the Bloch TFSF's own leakage (the incident
    wave never leaves the total-field region, so the absorber is NOT hit);
    |scat/inc| = sqrt(R) must be <= LEAK_BAR on every gated bin (a rig gate:
    a failure is an injection defect, never a physics verdict)."""
    f = np.asarray(freqs_hz, dtype=float)
    L = np.sqrt(np.clip(np.asarray(R_rfx, dtype=float), 0.0, None))
    g = gated_mask(f, spec)
    return {"leak_amp": L.tolist(), "gated": g.tolist(), "n_bins_gated": int(g.sum()),
            "max_leak_gated": float(L[g].max()), "leak_bar": LEAK_BAR, "G_leak": bool(np.all(L[g] <= LEAK_BAR))}


def evaluate_grazing_pec(freqs_hz, R_rfx, spec: dict, dt: float, cells: dict, *, n_cpml: int,
                         declared_n_cpml: int = N_CPML, declared_cpml_kwargs: dict | None = None) -> dict:
    """G6 (note section 4.5): the PEC compact-box arm. Fresnel gives R = 1 at
    every angle; the unit reflected wave hits the lo absorber and returns, so
    the measured excess R - 1 = |1 + r_pml e^{j phi}|^2 - 1 IS the absorber's
    reflection at the realized grazing angle (first order 2 |r_pml| cos phi).
    Gate: |R_meas - R_lat| <= PML_REL |R_lat3D - 1| + PML_FLOOR_R on the gated
    bins where the 3-D absorber's own term |R_lat3D - 1| >= PML_MIN_TERM (the
    relative part scales with the claim, not with the total whose zero
    crossings would collapse the window to the floor), with R_lat the
    exact lattice of the DECLARED absorber (20 cells, R_asym 1e-15) --
    falsifier arms are built with their defect and judged against it."""
    f = np.asarray(freqs_hz, dtype=float)
    R = np.asarray(R_rfx, dtype=float)
    dx = float(cells["dx"])
    lat = yee_lattice_full(f, spec["ky"], cells, dx=dx, dt=dt, n_cpml=declared_n_cpml,
                           cpml_kwargs=declared_cpml_kwargs, pec=True)                    # full: 3-D CPML + aux echo
    lat3 = yee_lattice_full(f, spec["ky"], cells, dx=dx, dt=dt, n_cpml=declared_n_cpml,
                            cpml_kwargs=declared_cpml_kwargs, pec=True, aux="plane")      # 3-D CPML only
    term = lat["R"] - 1.0                      # the whole a-priori excess over Fresnel (R = 1)
    term3 = lat3["R"] - 1.0                    # the 3-D absorber's own part (the claim)
    term_aux = lat["R"] - lat3["R"]            # the injection path's aux-absorber echo
    th = realized_theta_rad(f, spec["ky"])
    g = gated_mask(f, spec) & (np.abs(term3) >= PML_MIN_TERM)
    cont = cpml_continuum_reflection(np.nan_to_num(th, nan=0.0))
    # the relative part scales with the 3-D absorber term (the claim), never with the total,
    # whose zero crossings would collapse the window to the floor
    win = PML_REL * np.abs(term3) + PML_FLOOR_R
    d = np.abs(R - lat["R"])
    ok = bool(g.any()) and bool(np.all(d[g] <= win[g]))
    gm = gated_mask(f, spec)
    return {"R_lattice": lat["R"].tolist(), "absorber_term_lattice": term.tolist(),
            "cpml3d_term_lattice": term3.tolist(), "aux_echo_term_lattice": term_aux.tolist(),
            "excess_meas": (R - 1.0).tolist(), "r_continuum": cont.tolist(),
            "two_r_continuum": (2.0 * cont).tolist(), "window": win.tolist(), "gated": g.tolist(),
            "gated_band": gm.tolist(), "n_bins_gated": int(g.sum()), "n_bins_band": int(gm.sum()),
            "max_abs_dev_gated": float(d[g].max()) if g.any() else None,
            "max_excess_meas_band": float(np.abs(R - 1.0)[gm].max()),
            "max_absorber_term_band": float(np.abs(term)[gm].max()),
            "max_cpml3d_term_band": float(np.abs(term3)[gm].max()),
            "max_aux_echo_term_band": float(np.abs(term_aux)[gm].max()),
            "theta_gated_deg": [float(np.degrees(th[g]).min()), float(np.degrees(th[g]).max())] if g.any() else None,
            "run_n_cpml": int(n_cpml), "declared_n_cpml": int(declared_n_cpml), "G6_absorber": ok}


def evaluate_grazing_slab(freqs_hz, R_rfx, T_rfx, spec: dict, dt: float, cells: dict, *, n_cpml: int) -> dict:
    """G7 (note section 4.5): the slab compact-box arm's R, T against the
    exact lattice WITH the absorber (the whole discrete system) inside
    W_bin + W_inj; the excess over Fresnel at the realized angle is REPORTED
    per bin next to the a-priori absorber term (lattice with absorber minus
    lattice with an ideal absorber)."""
    f = np.asarray(freqs_hz, dtype=float)
    R = np.asarray(R_rfx, dtype=float); T = np.asarray(T_rfx, dtype=float)
    dx = float(cells["dx"])
    lat = yee_lattice_full(f, spec["ky"], cells, eps_slab=spec["eps_slab_rfx"], mu_slab=spec["mu_slab_rfx"],
                           dx=dx, dt=dt, n_cpml=n_cpml)
    ideal = yee_lattice_full(f, spec["ky"], cells, eps_slab=spec["eps_slab_rfx"], mu_slab=spec["mu_slab_rfx"],
                             dx=dx, dt=dt, ideal_absorber=True, aux="plane")
    lat3 = yee_lattice_full(f, spec["ky"], cells, eps_slab=spec["eps_slab_rfx"], mu_slab=spec["mu_slab_rfx"],
                            dx=dx, dt=dt, n_cpml=n_cpml, aux="plane")
    R_an, T_an = oracle_RT(f, spec["ky"], spec["pol"])
    g = gated_mask(f, spec)
    winR = W_BIN + injection_term(lat["R"]); winT = W_BIN + injection_term(lat["T"])
    dR = np.abs(R - lat["R"]); dT = np.abs(T - lat["T"])
    return {"R_lattice": lat["R"].tolist(), "T_lattice": lat["T"].tolist(),
            "R_lattice_ideal": ideal["R"].tolist(), "T_lattice_ideal": ideal["T"].tolist(),
            "absorber_term_R": (lat["R"] - ideal["R"]).tolist(), "absorber_term_T": (lat["T"] - ideal["T"]).tolist(),
            "excess_over_fresnel_R": (R - R_an).tolist(), "excess_over_fresnel_T": (T - T_an).tolist(),
            "gated": g.tolist(), "n_bins_gated": int(g.sum()),
            "max_dR_lattice_gated": float(dR[g].max()), "max_dT_lattice_gated": float(dT[g].max()),
            "max_absorber_term_R_gated": float(np.abs(lat["R"] - ideal["R"])[g].max()),
            "max_absorber_term_T_gated": float(np.abs(lat["T"] - ideal["T"])[g].max()),
            "max_cpml3d_term_R_gated": float(np.abs(lat3["R"] - ideal["R"])[g].max()),
            "max_aux_echo_term_R_gated": float(np.abs(lat["R"] - lat3["R"])[g].max()),
            "max_lattice_ideal_minus_fresnel_R_gated": float(np.nanmax(np.abs(ideal["R"] - R_an)[g])),
            "max_excess_over_fresnel_R_gated": float(np.abs(R - R_an)[g].max()),
            "G7_R": bool(np.all(dR[g] <= winR[g])), "G7_T": bool(np.all(dT[g] <= winT[g]))}


def evaluate_e4(e2: dict, meep_doc: dict) -> dict:
    """E4 (note section 4.3): Meep at the same fixed k_y (its k_point mapped
    from the DECLARED theta0, verified in the leg's pre-check) against the
    Fresnel oracle at the realized angle, and rfx against Meep, with the
    triangle-inequality windows. Meep's per-bin values are interpolated onto
    the rfx bins inside Meep's flux band."""
    f = np.asarray(e2["freqs_hz"]); g = np.asarray(e2["gated"], bool)
    fm = np.asarray(meep_doc["freqs_hz"], dtype=float)
    Rm = np.interp(f, fm, np.asarray(meep_doc["R"], dtype=float))
    Tm = np.interp(f, fm, np.asarray(meep_doc["T"], dtype=float))
    covered = bool(fm.min() <= f[g].min() and fm.max() >= f[g].max())
    ky_meep = ky_from_meep_k_point(meep_doc["k_point"], meep_doc["a_m"])
    ky_ok = bool(abs(ky_meep - e2["ky_rad_m"]) <= 1e-9 * max(1.0, abs(e2["ky_rad_m"])))
    pre_ok = bool(meep_doc.get("precheck", {}).get("passed", False))
    R_an = np.asarray(e2["R_an"]); T_an = np.asarray(e2["T_an"])
    R_r = np.asarray(e2["R_rfx"]); T_r = np.asarray(e2["T_rfx"])
    wdR = np.asarray(e2["w_disp_R"]); wdT = np.asarray(e2["w_disp_T"])
    wiR = np.asarray(e2["w_inj_R"]); wiT = np.asarray(e2["w_inj_T"])
    win4R = W_BIN + wdR; win4T = W_BIN + wdT
    win5R = 2 * W_BIN + wdR + wiR; win5T = 2 * W_BIN + wdT + wiT
    d4R = np.abs(Rm - R_an); d4T = np.abs(Tm - T_an); d5R = np.abs(R_r - Rm); d5T = np.abs(T_r - Tm)
    m4R = W_MEAN_R + float(np.nanmean(wdR[g])); m4T = W_MEAN_T + float(np.nanmean(wdT[g]))
    m5R = 2 * W_MEAN_R + float(np.nanmean((wdR + wiR)[g])); m5T = 2 * W_MEAN_T + float(np.nanmean((wdT + wiT)[g]))
    gates = {"precheck_passed": pre_ok, "k_point_matches_declared": ky_ok, "band_covered": covered,
             "G4_R": bool(np.all(d4R[g] <= win4R[g])), "G4_T": bool(np.all(d4T[g] <= win4T[g])),
             "G4_mean_R": bool(np.nanmean(d4R[g]) <= m4R), "G4_mean_T": bool(np.nanmean(d4T[g]) <= m4T),
             "G5_R": bool(np.all(d5R[g] <= win5R[g])), "G5_T": bool(np.all(d5T[g] <= win5T[g])),
             "G5_mean_R": bool(np.nanmean(d5R[g]) <= m5R), "G5_mean_T": bool(np.nanmean(d5T[g]) <= m5T)}
    return {"present": True, "source": meep_doc.get("_source"), "resolution": meep_doc.get("resolution"),
            "k_point": list(meep_doc["k_point"]), "ky_meep_rad_m": ky_meep, "precheck": meep_doc.get("precheck"),
            "R_meep": Rm.tolist(), "T_meep": Tm.tolist(), "dR_meep_tmm": d4R.tolist(), "dT_meep_tmm": d4T.tolist(),
            "dR_rfx_meep": d5R.tolist(), "dT_rfx_meep": d5T.tolist(),
            "window4_R": win4R.tolist(), "window4_T": win4T.tolist(), "window5_R": win5R.tolist(), "window5_T": win5T.tolist(),
            "mean_dR_meep_tmm_gated": float(np.nanmean(d4R[g])), "mean_dT_meep_tmm_gated": float(np.nanmean(d4T[g])),
            "max_dR_meep_tmm_gated": float(np.nanmax(d4R[g])), "max_dT_meep_tmm_gated": float(np.nanmax(d4T[g])),
            "mean_dR_rfx_meep_gated": float(np.nanmean(d5R[g])), "mean_dT_rfx_meep_gated": float(np.nanmean(d5T[g])),
            "max_dR_rfx_meep_gated": float(np.nanmax(d5R[g])), "max_dT_rfx_meep_gated": float(np.nanmax(d5T[g])),
            "mean_window4_R": m4R, "mean_window4_T": m4T, "mean_window5_R": m5R, "mean_window5_T": m5T,
            "gates": gates, "e4_ok": bool(all(gates.values()))}


# ---------------------------------------------------------------------------
# Falsifiers (note section 8)
# ---------------------------------------------------------------------------
FALSIFIERS = {
    # name: (arm, description, run-side defect, oracle-side defect)
    "te_60_angle_m5": ("te_60", "F1: TFSF angle mis-set by 5 deg (k_y of 55 deg), judged at the declared 60 deg k_y",
                       {"theta0_deg": 55.0}, {}),
    "te_45_swap_tm": ("te_45", "F2: TE run judged against the TM oracle", {}, {"oracle_pol": "tm"}),
    "tm_60_swap_te": ("tm_60", "F2: TM (dual) run judged against the TE oracle (fails the Brewster bin)", {}, {"oracle_pol": "te"}),
    "te_45_eps_x1p2": ("te_45", "F3: slab eps x 1.2 (4.8), judged against eps = 4", {"eps_scale": 1.2}, {}),
    "graze_pec_depth_half": ("graze_pec", "F5: CPML depth halved (10 cells), judged against the 20-cell prediction",
                             {"n_cpml": 10}, {}),
    "graze_pec_sigma_half": ("graze_pec", "F5b: CPML sigma_max halved (R_asym 1e-15 -> 10^-7.5), judged against the declared profile",
                             {"cpml_kwargs": {"R_asymptotic": 10 ** -7.5}}, {}),
}
FALSIFIER_MUST_EXIT_1 = ("te_60_angle_m5", "te_45_swap_tm", "tm_60_swap_te", "te_45_eps_x1p2",
                         "graze_pec_depth_half", "graze_pec_sigma_half")
# Evaluated and NOT declared (note section 8): +5 deg on the 45 deg arm gives mean|dR| 0.94x its window
# and on the 30 deg arm 1.07x -- coin tosses (cv22's Debye tau x 1.3 rule); +5 deg on the 60 deg arm is
# 4.8x but puts the run's cutoff (9.06 GHz) inside the band with 3.4e-2 of incident there, so the purity
# witness would fire too and the reading would be ambiguous; -5 deg (55 deg, cutoff content 3e-6) is 3.0x.
FALSIFIERS_REJECTED = {"te_45_angle_p5": 0.94, "te_30_angle_p5": 1.07, "te_60_angle_p5": "purity-ambiguous"}


# ---------------------------------------------------------------------------
# Meep leg: minimum run time and artifact acceptance (note section 14, round 2)
# ---------------------------------------------------------------------------

def meep_unavailable_reason(meep_doc, path: str, rel_to: str | None = None) -> str | None:
    """Why the E4 gate must SKIP instead of deciding, or ``None`` if the
    reference may be used.  An ABSENT reference and one the leg REJECTED are
    the same verdict -- "reference unavailable" -- and neither may be read as
    a number.  Round 1 read R = -inf, T = +inf out of a rejected artifact and
    reported E4 FAIL, which says "rfx disagrees with Meep"; it did not."""
    rel = os.path.relpath(path, rel_to) if rel_to else path
    if meep_doc is None:
        return f"no Meep artifact at {rel}"
    if meep_doc.get("accepted") is False:
        if meep_doc.get("falsifier"):
            # A DECLARED defect injection (MEEP_FALSIFIERS) exists to be judged: its whole
            # point is that the E4 gate must FAIL on it, and it fails on `precheck_passed`.
            # Withholding it would turn a falsifier into a SKIP and the lane would stop
            # detecting the defect it is there to detect.  This carve-out is reachable only
            # from --falsifier <declared name> on the declared arm.
            return None
        why = "; ".join(meep_doc.get("rejection_reasons") or ["(no reason recorded)"])
        return f"{rel}: the Meep leg REJECTED its own output -- {why}"
    for k in ("R", "T", "freqs_hz", "k_point"):
        if k not in meep_doc:
            return f"{rel}: the artifact carries no '{k}'"
    R = np.asarray(meep_doc["R"], dtype=float); T = np.asarray(meep_doc["T"], dtype=float)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(T))) and not meep_doc.get("falsifier"):
        # a pre-acceptance (v1) artifact, or one written by hand
        return f"{rel}: non-finite R/T in the artifact and no acceptance record"
    return None


def meep_min_after_sources(spec: dict, src_x: float, trans_x: float, a_m: float = MEEP_A_M) -> float:
    """Meep time units the leg must run AFTER its sources end before
    ``stop_when_fields_decayed`` is allowed to fire.

    Round 1 used Meep's helper unguarded.  Its first decay window closes
    ``MEEP_STOP_DT`` after the sources end; on the two WIDE-bandwidth arms
    (te_00 at bw 0.25, te_30 at 0.1902) the source is short, the transmission
    monitor is 78 a downstream, and at that first check the monitored point
    had seen IDENTICALLY zero field -- so ``old_cur <= max_abs * decay_by``
    read ``0 <= 0``, the run stopped with nothing in the flux monitors, the
    normalisation came out zero and the leg wrote R = -inf, T = +inf for all
    400 bins.  The narrow-bandwidth arms survived only because their longer
    source pushed the first check past first arrival.

    The bound is the same physics as the rfx record: geometric transit from
    the source plane to the far monitor at the x group velocity of the
    SLOWEST gated component (c cos theta_hi; c = 1 in Meep units), plus the
    slab etalon's ring-down at that angle."""
    edges = band_edges(spec)
    th_hi = math.radians(min(edges["theta_at_f_lo_deg"], spec["theta_gate_deg"][1]))
    transit = abs(trans_x - src_x) / max(math.cos(th_hi), 1e-6)
    t_ring = 0.0
    if spec["slab"]:
        nfft = 1 << 16
        f = np.fft.rfftfreq(nfft, d=DT_S)
        g = gated_mask(f, spec)
        rate, _t_rt, _rho = slab_ringdown_rate(f[g], spec)
        w = incident_amp_rel(f[g], spec["f0_hz"], spec["bw"])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ring = float(np.nanmax(np.where(rate > 0, np.log(100.0 * w) / rate, 0.0)))
        t_ring = max(t_ring, 0.0) * C0 / a_m        # seconds -> Meep time units
    return float(transit + t_ring)


def meep_accept(freqs_hz, R, T, inc_flux, spec: dict, *,
                inc_flux_refl=None, tol: float = MEEP_ACCEPT_TOL,
                flux_floor: float = MEEP_FLUX_FLOOR) -> dict:
    """Is this Meep output fit to be a reference at all?  VALIDITY only --
    never agreement with the oracle, which would turn an E4 disagreement into
    a silent SKIP.  Returns ``accepted`` and the named reasons it is not.

    1. every R, T finite over the whole flux band (a diverged run is not a
       reference for any bin);
    2. the flux normalisation is finite, non-zero, and on every gated bin at
       least ``flux_floor`` of its band maximum (round 1's failure);
    3. on the gated band 0 - tol <= R, T <= 1 + tol and |R + T - 1| <= tol,
       tol = MEEP_ACCEPT_TOL (cv04's passivity ceiling);
    4. the flux band covers the gated band;
    5. the EMPTY run's cross-box flux identity, when the leg reports it: with
       no scatterer the x-power through the reflection and transmission
       planes must agree to ``tol``.  This is the leg's vacuum witness -- the
       one form of it the two-pass rig can deliver, since R and T of a
       vacuum arm are 0 and 1 by construction of the subtraction."""
    f = np.asarray(freqs_hz, dtype=float)
    R = np.asarray(R, dtype=float); T = np.asarray(T, dtype=float)
    inc = np.asarray(inc_flux, dtype=float)
    g = gated_mask(f, spec)
    reasons = []
    if not g.any():
        reasons.append("no gated bin lies in the Meep flux band")
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(T))):
        reasons.append(f"non-finite R/T in {int((~np.isfinite(R)).sum())}/{int((~np.isfinite(T)).sum())} "
                       f"of {R.size} flux bins")
    if not np.all(np.isfinite(inc)):
        reasons.append("non-finite flux normalisation")
    inc_max = float(np.nanmax(np.abs(inc))) if inc.size else 0.0
    if not (inc_max > 0.0):
        reasons.append("flux normalisation is identically zero (the reference run accumulated no flux)")
    elif g.any():
        rel = np.abs(inc[g]) / inc_max
        if float(rel.min()) < flux_floor:
            reasons.append(f"flux normalisation degenerate on a gated bin: min |inc|/max |inc| = "
                           f"{float(rel.min()):.3e} < {flux_floor:g}")
    stats = {}
    if g.any() and np.all(np.isfinite(R[g])) and np.all(np.isfinite(T[g])):
        Rg, Tg = R[g], T[g]
        stats = {"R_min": float(Rg.min()), "R_max": float(Rg.max()), "T_min": float(Tg.min()),
                 "T_max": float(Tg.max()), "closure_max": float(np.abs(Rg + Tg - 1.0).max())}
        if stats["R_min"] < -tol or stats["R_max"] > 1.0 + tol:
            reasons.append(f"R outside [0, 1] by more than {tol:g} on a gated bin "
                           f"([{stats['R_min']:.4f}, {stats['R_max']:.4f}])")
        if stats["T_min"] < -tol or stats["T_max"] > 1.0 + tol:
            reasons.append(f"T outside [0, 1] by more than {tol:g} on a gated bin "
                           f"([{stats['T_min']:.4f}, {stats['T_max']:.4f}])")
        if stats["closure_max"] > tol:
            reasons.append(f"|R + T - 1| = {stats['closure_max']:.4f} > {tol:g} on a gated bin")
    if g.any() and not (f.min() <= f[g].min() and f.max() >= f[g].max()):
        reasons.append("Meep flux band does not cover the gated band")
    if inc_flux_refl is not None:
        ir = np.asarray(inc_flux_refl, dtype=float)
        if not np.all(np.isfinite(ir)):
            reasons.append("non-finite reflection-plane flux in the empty run")
        elif g.any() and inc_max > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = ir[g] / inc[g]
            dev = float(np.nanmax(np.abs(ratio - 1.0)))
            stats["vacuum_flux_ratio_dev"] = dev
            if not np.isfinite(dev) or dev > tol:
                reasons.append(f"empty-run cross-box flux identity fails: max |flux_refl/flux_trans - 1| = "
                               f"{dev:.4f} > {tol:g}")
    return {"accepted": not reasons, "reasons": reasons, "tol": tol, "flux_floor": flux_floor,
            "n_gated": int(g.sum()), **stats}


def falsifier_prediction(name: str, dt: float | None = None) -> dict:
    """Analytic margin of a falsifier on the rfx bin grid of its arm's derived
    record: the defective run's oracle-side prediction vs the declared oracle
    (E2 falsifiers), or the lattice absorber prediction with the defect vs the
    declared one (grazing falsifiers)."""
    arm, desc, run_def, or_def = FALSIFIERS[name]
    spec = arm_spec(arm)
    K = ARM_DX_DIV.get(arm, 1)
    dt = DT_S / K if dt is None else float(dt)
    dx = DX_M / K
    rec = derive_record(spec, dt, dx_div=K)
    nfft = int(2 ** math.ceil(math.log2(rec["n_steps"])) * NFFT_OVERSAMPLE)
    f = np.fft.rfftfreq(nfft, d=dt)
    f = f[(f > MASK_F_LO_HZ) & (f < MASK_F_HI_HZ)]
    g = gated_mask(f, spec)
    out = {"name": name, "arm": arm, "description": desc, "n_bins_gated": int(g.sum()), "dx_div": K}
    if arm in GRAZE_ARMS:
        cells = rig_cells(spec["nx_interior"], run_def.get("n_cpml", N_CPML), dx_div=K)
        cells_decl = rig_cells(spec["nx_interior"], N_CPML, dx_div=K)
        lat_def = yee_lattice_full(f, spec["ky"], cells, dx=dx, dt=dt, n_cpml=run_def.get("n_cpml", N_CPML),
                                   cpml_kwargs=run_def.get("cpml_kwargs"), pec=True)
        lat_decl = yee_lattice_full(f, spec["ky"], cells_decl, dx=dx, dt=dt, n_cpml=N_CPML, pec=True)
        lat3_decl = yee_lattice_full(f, spec["ky"], cells_decl, dx=dx, dt=dt, n_cpml=N_CPML, pec=True, aux="plane")
        term_decl = lat_decl["R"] - 1.0
        term3_decl = lat3_decl["R"] - 1.0
        gg = g & (np.abs(term3_decl) >= PML_MIN_TERM)
        win = PML_REL * np.abs(term3_decl) + PML_FLOOR_R
        d = np.abs(lat_def["R"] - lat_decl["R"])
        e_def = np.abs(lat_def["R"] - 1.0); e_decl = np.abs(term_decl)
        out.update({"n_bins_gated": int(gg.sum()), "predicted_fails_G6": bool(np.any(d[gg] > win[gg])),
                    "bins_beyond_window": int(np.sum(d[gg] > win[gg])),
                    "max_abs_dev": float(d[gg].max()) if gg.any() else None,
                    "max_window": float(win[gg].max()) if gg.any() else None,
                    "excess_decl_gated_range": [float(e_decl[gg].min()), float(e_decl[gg].max())] if gg.any() else None,
                    "excess_def_gated_range": [float(e_def[gg].min()), float(e_def[gg].max())] if gg.any() else None,
                    "max_ratio_excess_def_over_decl": float((e_def[gg] / e_decl[gg]).max()) if gg.any() else None})
        return out
    # E2 falsifiers: what the defective run's physics gives (its own oracle) vs the declared oracle
    ky_run = ky_from(TFSF_F0_HZ, run_def.get("theta0_deg", spec["theta0_deg"]))
    eps_run = EPS_R_SLAB * run_def.get("eps_scale", 1.0)
    R_run, T_run = oracle_RT(f, ky_run, spec["pol"], eps_slab=eps_run)
    pol_or = or_def.get("oracle_pol", spec["pol"])
    R_decl, T_decl = oracle_RT(f, spec["ky"], pol_or)
    disp = dispersion_term(f, spec["ky"], pol_or, dx, dt)
    winR = W_BIN + disp["W_R"] + injection_term(R_decl); winT = W_BIN + disp["W_T"] + injection_term(T_decl)
    dR = np.abs(R_run - R_decl); dT = np.abs(T_run - T_decl)
    mR = W_MEAN_R + float(np.nanmean(disp["W_R"][g])) + float(np.nanmean(injection_term(R_decl)[g]))
    mT = W_MEAN_T + float(np.nanmean(disp["W_T"][g])) + float(np.nanmean(injection_term(T_decl)[g]))
    binsR = int(np.nansum(dR[g] > winR[g])); binsT = int(np.nansum(dT[g] > winT[g]))
    iR = int(np.nanargmax(np.where(g, dR, -1))); iT = int(np.nanargmax(np.where(g, dT, -1)))
    th = np.degrees(realized_theta_rad(f, spec["ky"]))
    out.update({"mean_dR": float(np.nanmean(dR[g])), "mean_dT": float(np.nanmean(dT[g])),
                "mean_window_R": mR, "mean_window_T": mT,
                "ratio_mean_R": float(np.nanmean(dR[g]) / mR), "ratio_mean_T": float(np.nanmean(dT[g]) / mT),
                "bins_beyond_W_bin_R": binsR, "bins_beyond_W_bin_T": binsT,
                "worst_R": {"hz": float(f[iR]), "theta_deg": float(th[iR]), "dR": float(dR[iR])},
                "worst_T": {"hz": float(f[iT]), "theta_deg": float(th[iT]), "dT": float(dT[iT])},
                "predicted_fails": bool(binsR + binsT > 0 or np.nanmean(dR[g]) > mR or np.nanmean(dT[g]) > mT)})
    if name == "tm_60_swap_te":
        thB = math.degrees(theta_brewster_rad(EPS_R_SLAB))
        iB = np.where(g)[0][int(np.argmin(np.abs(th[g] - thB)))]
        out["brewster_bin"] = {"hz": float(f[iB]), "theta_deg": float(th[iB]), "R_te_oracle": float(R_decl[iB]),
                               "R_tm_run": float(R_run[iB]), "floor": float(winR[iB])}
    return out


def rfx_json_name(falsifier: str | None) -> str:
    return "rfx.json" if falsifier is None else f"rfx__falsifier_{falsifier}.json"


def meep_json_name(arm: str, falsifier: str | None = None, tag: str | None = None) -> str:
    if tag:
        return f"meep_{arm}__{tag}.json"
    return f"meep_{arm}.json" if falsifier is None else f"meep_{arm}__falsifier_{falsifier}.json"

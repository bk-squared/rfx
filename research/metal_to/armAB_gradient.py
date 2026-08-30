"""Arms A and B — the gradient arm of the Phase-2 dual-band notch benchmark.

WHAT THE TWO ARMS ARE, AND WHY THEY ARE THESE TWO
-------------------------------------------------
The established RF metal-TO line (Hassan/Wadbro/Berggren, T-MTT 68(4):1326,
2020; TAP 2014/2015; Lu et al. EuCAP 2025) interpolates CONDUCTIVITY and uses
NO Heaviside projection: intermediate conductivity is ohmically lossy, so an
energy objective is already SELF-PENALIZING toward binary. Its documented
failure mode is that the self-penalization becomes too aggressive and traps the
optimizer, and the documented cure is FILTER-RADIUS CONTINUATION rather than
projection sharpening. Our own Phase-1 arm B (a re-derivation of that
interpolation, with a linear RAMP) collapsed to an empty design region from a
low-fill start, which is exactly that trap.

So:

  arm A  conductivity interpolation + filter-radius continuation.
         The established scheme. No projection at any point.
  arm B  arm A + Heaviside projection with beta continuation.
         The ABLATION that answers the reviewer's question -- "why project at
         all when gray metal is already self-penalizing?" -- by measurement
         rather than by assumption.

Everything else is identical between the arms by construction: the same
fixture, box, initialisation, frequency grid, window, surrogate, optimizer,
learning rate, continuation schedule, seed and SOLVE BUDGET. The only thing
arm B adds is ``apply_projection``.

THE DIFFERENTIABLE PATHWAY
--------------------------
    theta (2 x nx x ny latent)
      -> sigmoid                          rho in (0, 1)
      -> density filter, radius in METRES, applied PER SIDE
      -> [arm B only] Heaviside projection, beta
      = rho_phys
      -> pec_occupancy_override = rho_phys        (Kottke PEC limit at rho=1)
         sigma_override         = sigma(rho_phys) (exponential, Hassan's range)
      -> sim.forward(...)  -> plane DFT probes -> extract_msl_nprobe -> S21

Three deliberate choices, each of which a review found broken in Phase-1:

  * THE FILTER RADIUS IS IN METRES (PLAN gate 4). Phase-1 specified it in
    cells, which silently changes the design problem under mesh refinement --
    the same script at dx/2 would optimize a different minimum feature size
    under the same name. ``radius_cells = radius_m / dx`` is computed at the
    point of use and BOTH are logged.
  * NO ``jax.jit`` AROUND ``value_and_grad``. ``rfx.topology.apply_density_filter``
    computes its kernel size as ``int(jnp.ceil(radius_cells))``; under jit even
    a Python-float radius becomes abstract and that raises
    ``ConcretizationTypeError``. ``forward()`` jits its own scan internally, so
    the per-iteration cost of not jitting the wrapper is negligible.
  * REALIZED GEOMETRY IS LOGGED, not requested geometry. Every record carries
    the binarised cell counts, column/row spans and box count that the solver
    actually saw, per side, plus whether the design still touches the feed.

WHY BOTH OVERRIDES, WHEN HASSAN USES ONLY CONDUCTIVITY
------------------------------------------------------
Hassan's rho=1 limit is a good conductor (~1e5 S/m), not PEC. Ours must be
PEC, because the number that counts is produced by the FROZEN IMPERATIVE PATH,
which realizes the binarised design as hard ``Box(material="pec")`` geometry.
If the descent optimized a 1e5 S/m structure and the evaluator scored a PEC
one, the two would be different design problems and the divergence would be
invisible from inside the loop -- the precise failure that got the Phase-1
headline retracted (NOTE_xval1_verdict.md, "same-operator evaluation"). So the
occupancy field carries rho into rfx's Kottke PEC limit (identical to the
subpixel machinery a hard PEC Box goes through) and the conductivity map
supplies the ohmic self-penalization on the GRAY cells, where the PEC limit is
not yet active. At rho = 1 the cell is PEC and its sigma is irrelevant; at
rho = 0 it is bare dielectric plus 1e-3 S/m, which is 4e-6 of the displacement
current at 5.5 GHz -- i.e. nothing.

THE OBJECTIVE IS A SURROGATE, AND ONLY A SURROGATE
--------------------------------------------------
``score_dualband.M`` is built from min / max over bands and a
bandwidth-weighted integral over the passband. None of those is what one wants
inside Adam. The surrogate below is M with each non-smooth operator replaced by
its smooth analogue, TERM FOR TERM, at a stated temperature:

    frozen term                          surrogate term
    -----------------------------------  ------------------------------------
    il = min(IL, r_cap=25 dB)            smooth cap, il_c = 25 - kappa*
    (score() clips before aggregating)   softplus((25 - IL)/kappa)
    R_L = min_{B_L} il                   softmin_tau(il_c over B_L)
    S_L = max(0, 20 - R_L)               kappa*softplus((20 - R_L~)/kappa)
    R_U, S_U                             same, over B_U
    G   = max_{gap} il                   softmax_tau(il_c over 5450-5625 MHz)
    S_G = min(max(0, G - 10), 20)        smooth hinge then smooth cap
    S_P = min(bandwidth-weighted mean     same trapezoid, same segments, with
          of max(0, il - 1), 20)         softplus for the hinge and a smooth cap
    M   = S_L + S_U + S_G + S_P          J  = s_L + s_U + s_G + s_P

Temperatures: ``tau = 1.0 dB`` for softmin/softmax, ``kappa = 0.5 dB`` for
every hinge and cap (both settable). Both smoothings are CONSERVATIVE in the
same direction the metric would want: softmin <= min (it under-reports
rejection by at most ``tau*log(N_band)`` ~ 2.2 dB) and softmax >= max (it
over-reports gap blockage by at most ``tau*log(N_gap)``), so the surrogate is
an upper bound on M up to the softplus rounding, and descent on J cannot be
rewarded for a violation the metric would see.

MEASURED against the frozen metric on the 123-point scoring grid, at
``tau = 1.0``, ``kappa = 0.5`` (synthetic IL traces, no solver involved):

    trace                                     M        J
    ---------------------------------------  ------   ------
    ideal flat 25 dB across both bands         0.00     0.09
    Stage-0-like merged 4.9-6.0 GHz block     16.43    16.21
    empty line (IL = 0 everywhere)            40.00    40.06
    solid brick (IL = 60 everywhere)          35.00    35.00

The residual few tenths of a dB are the smoothings' own width: a term sitting
AT a cap reads ``kappa*ln2 = 0.35 dB`` below the hard cap (this is where the
merged case's ``s_G = 14.65`` against ``S_G = 15.00`` comes from), and an
unsaturated hinge reads up to ``kappa*ln2`` above zero. Both are constants, not
a ranking change, and both shrink with kappa.

``--surrogate power`` selects instead the PRE-REGISTERED linear-power hinge
recorded in ``score_dualband`` §5 (weights 1.0 / 2.0 / 1.25). It is kept
runnable so the pre-registration is honoured, but it is not the default: it has
no analogue of S_P's bandwidth weighting and replaces S_G's max by a mean.

**NOTHING FROM THE SURROGATE IS EVER REPORTED AS A RESULT.** Every quoted
number in this file comes from ``phase2_calibrate.score_design`` on the
IMPERATIVE path -- hard-PEC boxes, ``compute_msl_s_matrix``, insertion loss
against the cached imperative empty line, the frozen ``score_dualband`` metric
and its validity block. J and M are logged side by side at every scoring
iteration precisely so surrogate/score divergence is visible from inside the
run.

EMPTY-LINE NORMALISATION INSIDE THE OBJECTIVE
---------------------------------------------
The metric is insertion loss RELATIVE TO THE EMPTY LINE, and the plane
extractor this path uses is documented as diverged in absolute scale
(``rfx.probes.msl_wave_decomp._v_from_plane``: V reads ~12 % low, I under-counts
~1.5x, and the two partially cancel). So the differentiable objective solves
the EMPTY design ONCE through the IDENTICAL pipeline (rho = 0 everywhere, same
overrides, same window, same probes), caches |S21_empty|, and forms

    IL_dB(f) = -20 log10( |S21_dut(f)| / |S21_empty(f)| ).

The extractor's absolute scale cancels exactly. Mixing a differentiable DUT
against the imperative empty reference would leave that scale in the objective
as a frequency-dependent bias, which is why it is not done here. The two empty
lines are both computed and both logged, so the size of that bias is on the
record rather than assumed away.

BUDGET
------
Counted in MAXWELL SOLVES, not iterations, so arm C (binary heuristic) can be
given the same total. One ``value_and_grad`` call = 1 forward + 1 backward = 2
solves. Every history record carries the running totals. Imperative scoring
solves are counted SEPARATELY (each drives both ports) and are monitoring, not
descent: they are not part of the budget arm C must match, and the record says
so.

RUN
---
  python research/metal_to/armAB_gradient.py --arm A --iters 120 --periods 45
  python research/metal_to/armAB_gradient.py --arm B --iters 120 --periods 45
  SMOKE=1 python research/metal_to/armAB_gradient.py --arm A   # CPU API check

``SMOKE=1`` shrinks the window, the iteration count and the frequency grid and
writes to ``out_smoke/armAB``. A smoke record is an unsettled ring-down
artifact and is stamped ``quotable: false``; it proves that the gradient flows
and that the binarise-and-score path executes, and it proves nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "validation" / "tmtt_paper"))

# The Kottke PEC limit is the path a hard PEC Box goes through, and it is the
# path Phase-1 measured as the good one (the legacy E-zeroing path is the
# gradient-starved reference). It is read from the environment inside
# ``forward()``, so it must be set BEFORE the first solve, not after.
os.environ.setdefault("RFX_PEC_OCC_KOTTKE", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402

import phase2_calibrate as cal  # noqa: E402
import phase2_fixture as fx  # noqa: E402
import score_dualband as sd  # noqa: E402
from rfx.probes.msl_wave_decomp import (  # noqa: E402
    _i_from_plane,
    _v_from_plane,
    extract_msl_nprobe,
)
from rfx.topology import apply_density_filter, apply_projection  # noqa: E402

C0 = 2.998e8
SMOKE = os.environ.get("SMOKE", "0") == "1"

OUT = Path(os.environ.get(
    "OUTPUT_DIR",
    HERE / "out_smoke" / "armAB" if SMOKE else HERE / "out_vessl" / "armAB"))
OUT.mkdir(parents=True, exist_ok=True)
# Same gating armD uses: a SMOKE empty-line reference is a ring-down artifact
# and must never sit where a production run would find it.
CACHE = OUT.parent / "empty_ref"
CACHE.mkdir(parents=True, exist_ok=True)

# Hassan's conductivity range for metal TO (T-MTT 2020 / TAP 2015). The map is
# exponential in rho, so d(sigma)/d(rho) is non-zero everywhere including at
# rho = 0 -- a linear map with sigma(0) = 0 has a vanishing gradient exactly
# where a low-fill start lives.
SIGMA_MIN_DEFAULT = 1.0e-3      # S/m at rho = 0
SIGMA_MAX_DEFAULT = 1.0e5       # S/m at rho = 1

# Surrogate smoothing (dB). Stated here, echoed into every record.
TAU_DB_DEFAULT = 1.0            # softmin / softmax temperature
KAPPA_DB_DEFAULT = 0.5          # smooth-hinge / smooth-cap width
T_FLOOR = 1e-12                 # |S21|/|S21_empty| floor -> IL <= 240 dB

# Continuation defaults, in METRES. 6 coarse cells -> 1.5 coarse cells.
FILTER_R_START_MM_DEFAULT = 0.762
FILTER_R_END_MM_DEFAULT = 0.191
BETA_STAGES_DEFAULT = (8.0, 16.0, 32.0, 64.0)


# ---------------------------------------------------------------------------
# 0. Budget -- Maxwell solves, not iterations
# ---------------------------------------------------------------------------
class Budget:
    """Running solve counts. Descent cost is forward + backward.

    ``imperative`` is counted separately and deliberately: each imperative
    ``compute_msl_s_matrix`` call drives BOTH ports, and those solves are
    monitoring (they produce the reported numbers), not descent. Arm C's
    budget must match ``total_descent``, not ``total_all``.
    """

    def __init__(self):
        self.forward = 0
        self.backward = 0
        self.imperative = 0          # 2-port imperative solves
        self.forward_empty_ref = 0   # subset of self.forward

    def fwd(self, n=1):
        self.forward += n

    def grad(self, n=1):
        self.forward += n
        self.backward += n

    def imp(self, n=1):
        self.imperative += n

    @property
    def total_descent(self) -> int:
        return self.forward + self.backward

    def as_dict(self) -> dict:
        return dict(
            solves_forward=self.forward,
            solves_backward=self.backward,
            solves_total_descent=self.total_descent,
            solves_forward_empty_ref=self.forward_empty_ref,
            solves_imperative_2port=self.imperative,
            note=("a gradient iteration = 1 forward + 1 backward = 2 solves; "
                  "imperative solves drive both ports and are monitoring, not "
                  "descent"),
        )


# ---------------------------------------------------------------------------
# 1. Frequency grids
# ---------------------------------------------------------------------------
#: A deliberately tiny descent grid for the CPU smoke. It keeps >= 3 samples in
#: each stopband, in the gap and in each passband segment, which is what the
#: surrogate's trapezoid and the frozen metric both need; it is not a scoring
#: grid and nothing computed on it is quotable.
SMOKE_GRID_MHZ = np.array(
    [3100, 3900, 4700, 5150, 5250, 5350, 5450, 5550, 5625,
     5725, 5775, 5825, 5925, 6700, 7600, 8600], dtype=int)


def descent_grid_hz() -> np.ndarray:
    g = SMOKE_GRID_MHZ if SMOKE else sd.descent_grid_mhz()
    return np.asarray(g, dtype=float) * 1e6


def scoring_grid_hz() -> np.ndarray:
    g = SMOKE_GRID_MHZ if SMOKE else sd.scoring_grid_mhz()
    return np.asarray(g, dtype=float) * 1e6


# ---------------------------------------------------------------------------
# 2. Smooth operators. Every one of these has a named frozen counterpart.
# ---------------------------------------------------------------------------
def _softplus_k(x, kappa: float):
    """Smooth ``max(0, x)``. Overshoots by at most ``kappa*ln2`` at x = 0."""
    return kappa * jnp.logaddexp(0.0, x / kappa)


def _smooth_cap(x, cap: float, kappa: float):
    """Smooth ``min(x, cap)``."""
    return cap - _softplus_k(cap - x, kappa)


def _softmin(x, tau: float):
    """Smooth ``min(x)``. <= min(x), by at most ``tau*log(len(x))``."""
    n = x.shape[0]
    return -tau * (logsumexp(-x / tau) - math.log(n))


def _softmax(x, tau: float):
    """Smooth ``max(x)``. >= max(x), by at most ``tau*log(len(x))``."""
    n = x.shape[0]
    return tau * (logsumexp(x / tau) - math.log(n))


def _trapz(y, x):
    """Trapezoid, written out so it does not depend on numpy>=2 naming."""
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


# ---------------------------------------------------------------------------
# 3. Band bookkeeping -- the SAME segments ``score_dualband.score`` uses
# ---------------------------------------------------------------------------
class Segments:
    """Static index sets over one frequency grid, derived from the frozen file.

    Derived from ``sd.BAND_L_MHZ`` / ``sd.BAND_U_MHZ`` / ``sd.GUARD_MHZ`` /
    ``sd.F_LO_MHZ`` / ``sd.F_HI_MHZ`` rather than re-typed, so a change to the
    frozen metric cannot leave the surrogate pointing at the old bands.
    """

    def __init__(self, f_mhz: np.ndarray):
        f = np.asarray(f_mhz, dtype=int)
        self.f_mhz = f
        self.band_l = sd.BAND_L_MHZ
        self.band_u = sd.BAND_U_MHZ
        self.gap = (sd.BAND_L_MHZ[1] + sd.GUARD_MHZ,
                    sd.BAND_U_MHZ[0] - sd.GUARD_MHZ)
        self.pass_lo = (sd.F_LO_MHZ, sd.BAND_L_MHZ[0] - sd.GUARD_MHZ)
        self.pass_hi = (sd.BAND_U_MHZ[1] + sd.GUARD_MHZ, sd.F_HI_MHZ)

        def idx(seg):
            return np.where((f >= seg[0]) & (f <= seg[1]))[0]

        self.i_l, self.i_u, self.i_g = idx(self.band_l), idx(self.band_u), idx(self.gap)
        self.pass_idx = [idx(self.pass_lo), idx(self.pass_hi)]
        for name, ii, need in (("lower stopband", self.i_l, 2),
                               ("upper stopband", self.i_u, 2),
                               ("inter-band gap", self.i_g, 2)):
            if ii.size < need:
                raise ValueError(
                    f"the descent frequency grid samples the {name} "
                    f"{ii.size} time(s); the surrogate needs >= {need}")
        for seg, ii in zip((self.pass_lo, self.pass_hi), self.pass_idx):
            if ii.size < 2:
                raise ValueError(
                    f"the descent grid samples passband segment {seg} "
                    f"{ii.size} time(s); the trapezoid needs >= 2")
        self.pass_bw = float(sum(f[ii][-1] - f[ii][0] for ii in self.pass_idx))

    def describe(self) -> str:
        return (f"grid {self.f_mhz.size} pts  B_L {self.i_l.size}  "
                f"B_U {self.i_u.size}  gap{self.gap} {self.i_g.size}  "
                f"pass {self.pass_idx[0].size}+{self.pass_idx[1].size} "
                f"({self.pass_bw/1e3:.2f} GHz)")


# ---------------------------------------------------------------------------
# 4. The surrogate
# ---------------------------------------------------------------------------
def il_db_surrogate(s21, s21_empty_mag):
    """IL(f) in dB from the DIFFERENTIABLE path. Positive = attenuation.

    Same sign convention as ``phase2_calibrate.insertion_loss`` (and therefore
    as every threshold in the frozen metric). The magnitude is floored at
    ``T_FLOOR`` so a deep notch cannot send ``d(IL)/d(|S21|)`` to infinity in
    float32; the floor is 240 dB below the empty line, i.e. ten times deeper
    than the frozen metric's 25 dB clip ever looks.
    """
    t = jnp.abs(s21) / (s21_empty_mag + 1e-30)
    t = jnp.maximum(t, T_FLOOR)
    return -20.0 * jnp.log10(t)


def surrogate_metric(il_db, seg: Segments, thr: sd.Thresholds,
                     tau: float, kappa: float, weights=(1.0, 1.0, 1.0, 1.0)):
    """J ~= M, term for term. See the module docstring's mapping table.

    Returns ``(J, terms)`` where ``terms`` names each surrogate term after the
    frozen term it stands for, so a record can be read against ``Result``.
    """
    f = jnp.asarray(seg.f_mhz, dtype=jnp.float32)
    il_c = _smooth_cap(il_db, thr.r_cap_db, kappa)      # score()'s np.minimum

    r_l = _softmin(il_c[seg.i_l], tau)                  # min over B_L
    r_u = _softmin(il_c[seg.i_u], tau)                  # min over B_U
    s_l = _softplus_k(thr.r_req_db - r_l, kappa)        # max(0, 20 - R_L)
    s_u = _softplus_k(thr.r_req_db - r_u, kappa)

    g_max = _softmax(il_c[seg.i_g], tau)                # max over the gap
    s_g = _smooth_cap(_softplus_k(g_max - thr.il_gap_db, kappa),
                      thr.term_cap_db, kappa)

    num = 0.0
    for ii in seg.pass_idx:                             # _seg_mean_excess
        e = _softplus_k(il_c[ii] - thr.il_pass_db, kappa)
        num = num + _trapz(e, f[ii])
    s_p = _smooth_cap(num / seg.pass_bw, thr.term_cap_db, kappa)

    w = weights
    j = w[0] * s_l + w[1] * s_u + w[2] * s_g + w[3] * s_p
    return j, dict(s_L=s_l, s_U=s_u, s_G=s_g, s_P=s_p,
                   R_L_soft=r_l, R_U_soft=r_u, gap_max_soft=g_max)


def surrogate_power(s21, s21_empty_mag, seg: Segments, thr: sd.Thresholds):
    """The PRE-REGISTERED linear-power hinge of ``score_dualband`` §5.

    Kept runnable so the pre-registration is honoured. Not the default: it has
    no analogue of S_P's bandwidth weighting and it replaces S_G's max by a
    mean, so its correspondence with M is looser than the dB-domain surrogate's.
    """
    t2 = (jnp.abs(s21) / (s21_empty_mag + 1e-30)) ** 2       # |S21|^2, normalized
    t_r = 10.0 ** (-thr.r_req_db / 10.0)
    t_g = 10.0 ** (-thr.il_gap_db / 10.0)
    t_p = 10.0 ** (-thr.il_pass_db / 10.0)
    w = sd.SURROGATE_WEIGHTS
    s_l = jnp.mean(jax.nn.relu(t2[seg.i_l] - t_r))
    s_u = jnp.mean(jax.nn.relu(t2[seg.i_u] - t_r))
    s_g = jnp.mean(jax.nn.relu(t_g - t2[seg.i_g]))
    p = jnp.concatenate([t2[ii] for ii in seg.pass_idx])
    s_p = jnp.mean(jax.nn.relu(t_p - p))
    j = (w["band"] * (s_l + s_u) + w["gap"] * s_g + w["passband"] * s_p)
    return j, dict(s_L=s_l, s_U=s_u, s_G=s_g, s_P=s_p,
                   R_L_soft=jnp.nan, R_U_soft=jnp.nan, gap_max_soft=jnp.nan)


# ---------------------------------------------------------------------------
# 5. Continuation schedule -- radius in METRES, logged in both units
# ---------------------------------------------------------------------------
def continuation_schedule(n_iters: int, r_start_m: float, r_end_m: float,
                          n_stages: int, dx: float, arm: str,
                          beta_stages=BETA_STAGES_DEFAULT) -> list:
    """Piecewise-constant filter-radius (and, arm B, beta) stages.

    Piecewise-constant rather than continuous because
    ``apply_density_filter`` sizes its cone kernel as ``ceil(radius_cells)``:
    a continuously shrinking radius would re-trace and re-compile the filter
    every iteration for no modelling benefit. The radius is geometric between
    the endpoints (a length scale, not an additive one) and is specified in
    METRES so the same schedule at dx/2 is the same physical problem.
    """
    n_stages = max(1, int(n_stages))
    if r_start_m <= 0 or r_end_m <= 0:
        raise ValueError("filter radii must be positive lengths in metres")
    out = []
    for k in range(n_stages):
        frac = 0.0 if n_stages == 1 else k / (n_stages - 1)
        r_m = r_start_m * (r_end_m / r_start_m) ** frac
        it0 = int(round(k * n_iters / n_stages))
        it1 = int(round((k + 1) * n_iters / n_stages))
        if arm == "B":
            b = float(beta_stages[min(k, len(beta_stages) - 1)])
        else:
            b = 0.0                     # 0 == no projection at all
        out.append(dict(stage=k, iter_lo=it0, iter_hi=it1,
                        filter_radius_m=float(r_m),
                        filter_radius_mm=float(r_m * 1e3),
                        filter_radius_cells=float(r_m / dx),
                        beta=b))
    return out


def stage_at(schedule: list, it: int) -> dict:
    for s in schedule:
        if s["iter_lo"] <= it < s["iter_hi"]:
            return s
    return schedule[-1]


# ---------------------------------------------------------------------------
# 6. Realized geometry -- what the lattice made, not what was asked for
# ---------------------------------------------------------------------------
def realized(mask) -> dict:
    """Per-side realized cell bookkeeping, armD's convention verbatim."""
    out = {}
    total = 0
    for side in fx.SIDES:
        m = np.asarray(mask[side])
        total += int(m.sum())
        if not m.any():
            out[f"{side}_cells"] = 0
            out[f"{side}_touches_trace"] = False
            continue
        cols = np.where(m.any(axis=1))[0]
        rows = np.where(m.any(axis=0))[0]
        out[f"{side}_cells"] = int(m.sum())
        out[f"{side}_cols"] = int(len(cols))
        out[f"{side}_rows"] = int(len(rows))
        out[f"{side}_col_range"] = [int(cols.min()), int(cols.max())]
        out[f"{side}_row_range"] = [int(rows.min()), int(rows.max())]
        # ``BoxSide.trace_row`` is ny-1 on the lo side and 0 on the hi side,
        # because a mask column ascends in GLOBAL y on both sides.
        trace_row = (m.shape[1] - 1) if side == "lo" else 0
        out[f"{side}_touches_trace"] = bool(m[:, trace_row].any())
    out["cells_total"] = total
    return out


def grayness(rho: np.ndarray) -> float:
    """``mean 4 rho (1-rho)`` -- 0 when binary, 1 when every cell is 0.5.

    The number the projection ablation lives or dies by: arm A's claim is that
    a lossy gray conductor drives this to ~0 on its own.
    """
    r = np.asarray(rho, dtype=float)
    return float(np.mean(4.0 * r * (1.0 - r)))


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("A", "B"), required=True)
    ap.add_argument("--iters", type=int, default=2 if SMOKE else 120)
    ap.add_argument("--periods", type=float, default=1.0 if SMOKE else 45.0,
                    help="descent window, periods of F_MAX (Stage-0: 45)")
    ap.add_argument("--verify-periods", type=float, default=None,
                    help="window for the FINAL frozen score (Stage-0: 90); "
                         "defaults to --periods under SMOKE, 90 otherwise")
    ap.add_argument("--filter-radius-mm", type=str,
                    default=f"{FILTER_R_START_MM_DEFAULT},{FILTER_R_END_MM_DEFAULT}",
                    help="continuation endpoints 'start,end' in MILLIMETRES "
                         "(a LENGTH; converted to cells at the point of use)")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", choices=("uniform", "low", "stub"), default="uniform")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--surrogate", choices=("metric", "power"), default="metric")
    ap.add_argument("--tau-db", type=float, default=TAU_DB_DEFAULT)
    ap.add_argument("--kappa-db", type=float, default=KAPPA_DB_DEFAULT)
    ap.add_argument("--sigma-min", type=float, default=SIGMA_MIN_DEFAULT)
    ap.add_argument("--sigma-max", type=float, default=SIGMA_MAX_DEFAULT)
    ap.add_argument("--score-every", type=int, default=0 if SMOKE else 20,
                    help="binarise and score through the FROZEN IMPERATIVE path "
                         "every N iterations (0 = only at the end)")
    ap.add_argument("--sigma-gate", action="store_true",
                    help="prove sigma_override is not silently ignored (2 solves)")
    ap.add_argument("--gradcheck", action="store_true",
                    help="directional-derivative Richardson check at init "
                         "(4 solves); the PLAN's Stage-2 gate")
    ap.add_argument("--allow-unsettled-empty", action="store_true",
                    help="score against an unsettled empty line; forced on "
                         "under SMOKE and stamped quotable=false")
    args = ap.parse_args()

    arm = args.arm
    tag = f"armAB_{arm}_{args.init}_{args.surrogate}_i{args.iters}_s{args.seed}"
    verify_periods = (args.verify_periods if args.verify_periods is not None
                      else (args.periods if SMOKE else 90.0))
    allow_unsettled = args.allow_unsettled_empty or SMOKE
    try:
        r0_mm, r1_mm = (float(v) for v in args.filter_radius_mm.split(","))
    except Exception:
        raise SystemExit("--filter-radius-mm wants 'start,end' in mm, e.g. 0.762,0.191")
    budget = Budget()
    thr = sd.SCORE

    # ---- fixture, box, windows -------------------------------------------
    f_desc = descent_grid_hz()
    f_score = scoring_grid_hz()
    fixture = fx.build_sim(f_desc)
    sim = fixture.sim
    grid = sim._build_grid()
    box = fx.design_box(grid, fixture.mesh)
    dx = float(box.dx)

    pre = [str(m) for m in sim.preflight()]
    period = 1.0 / float(sim._freq_max)
    n_raw = int(math.ceil(args.periods * period / float(grid.dt)))
    k_seg = max(8, int(math.isqrt(max(n_raw, 1))))
    n_steps = ((n_raw + k_seg - 1) // k_seg) * k_seg

    seg = Segments(np.asarray(f_desc / 1e6, dtype=int))
    schedule = continuation_schedule(args.iters, r0_mm * 1e-3, r1_mm * 1e-3,
                                     args.stages, dx, arm)

    print(f"[armAB:{arm}] {'SMOKE  ' if SMOKE else ''}grid={tuple(grid.shape)} "
          f"({int(np.prod(grid.shape)):,d} cells)  dx={dx*1e6:.1f} um  "
          f"dt={float(grid.dt)*1e12:.4f} ps")
    print(f"[armAB:{arm}] design box lo {box.lo.shape} + hi {box.hi.shape} "
          f"= {box.n_vars} variables   iz={box.iz} nz={box.lo.nz}")
    print(f"[armAB:{arm}] descent {args.periods:g} periods = {n_steps} steps "
          f"(k={k_seg}, {n_steps*float(grid.dt)*1e9:.2f} ns)   "
          f"verification {verify_periods:g} periods")
    print(f"[armAB:{arm}] descent {seg.describe()}")
    print(f"[armAB:{arm}] scoring grid {f_score.size} pts   "
          f"surrogate={args.surrogate} tau={args.tau_db} dB kappa={args.kappa_db} dB")
    print(f"[armAB:{arm}] sigma map exponential {args.sigma_min:g} -> "
          f"{args.sigma_max:g} S/m   KOTTKE={os.environ['RFX_PEC_OCC_KOTTKE']}")
    print(f"[armAB:{arm}] continuation (radius in METRES, cells shown for scale):")
    for s in schedule:
        print(f"    stage {s['stage']}  it {s['iter_lo']:3d}-{s['iter_hi']:3d}  "
              f"r = {s['filter_radius_mm']:.3f} mm = "
              f"{s['filter_radius_cells']:.2f} cells   "
              f"beta = {'none (no projection)' if s['beta'] == 0 else s['beta']}")
    print(f"[armAB:{arm}] preflight: {len(pre)} message(s)")

    # ---- differentiable pipeline ------------------------------------------
    shape3 = tuple(int(v) for v in grid.shape)
    sides = [(name, box.side(name)) for name in fx.SIDES]
    freqs_j = jnp.asarray(f_desc, dtype=jnp.float32)
    beta0 = (2.0 * jnp.pi * freqs_j
             * jnp.sqrt(jnp.asarray(fx.EPS_EFF, dtype=jnp.float32))
             / jnp.asarray(C0, dtype=jnp.float32))
    x_probes = jnp.array([0.0, fixture.d_set.delta, 2.0 * fixture.d_set.delta],
                         dtype=jnp.float32)
    log_ratio = math.log(args.sigma_max / args.sigma_min)

    def rho_phys_from_theta(theta, radius_cells: float, beta: float):
        """theta -> sigmoid -> per-side density filter -> [projection] -> rho.

        The filter is applied PER SIDE. The two sides are separated by the
        5-cell through-line, which is not a design variable: a filter run
        across the whole (2, nx, ny) stack would blur metal through the feed
        and give every cell on one side a gradient path to the other.
        """
        rho = jax.nn.sigmoid(theta)                 # (2, nx, ny)
        out = []
        for k in range(2):
            r = apply_density_filter(rho[k], radius_cells)
            if beta > 0.0:
                r = apply_projection(r, beta)
            out.append(r)
        return jnp.stack(out)

    def fields_from_rho(rho_phys):
        """Scatter the two sides into full-grid occupancy and sigma arrays."""
        occ = jnp.zeros(shape3, dtype=jnp.float32)
        sig = jnp.zeros(shape3, dtype=jnp.float32)
        sig_cells = args.sigma_min * jnp.exp(log_ratio * rho_phys)
        for k, (_, s) in enumerate(sides):
            occ = occ.at[s.ix_lo:s.ix_hi, s.iy_lo:s.iy_hi,
                         s.iz:s.iz + s.nz].set(rho_phys[k][:, :, None])
            sig = sig.at[s.ix_lo:s.ix_hi, s.iy_lo:s.iy_hi,
                         s.iz:s.iz + s.nz].set(sig_cells[k][:, :, None])
        return occ, sig

    def s21_from_fields(occ, sig):
        fr = sim.forward(pec_occupancy_override=occ, sigma_override=sig,
                         n_steps=n_steps, checkpoint_segments=k_seg,
                         skip_preflight=True)
        d, p = fixture.d_set, fixture.p_set
        v_d = jnp.stack([_v_from_plane(fr, d.ez1_name, d),
                         _v_from_plane(fr, d.ez2_name, d),
                         _v_from_plane(fr, d.ez3_name, d)], axis=-1)
        v_p = jnp.stack([_v_from_plane(fr, p.ez1_name, p),
                         _v_from_plane(fr, p.ez2_name, p),
                         _v_from_plane(fr, p.ez3_name, p)], axis=-1)
        r_d = extract_msl_nprobe(v_d, x_probes, _i_from_plane(fr, d.hy_name, d), beta0)
        r_p = extract_msl_nprobe(v_p, x_probes, _i_from_plane(fr, p.hy_name, p), beta0)
        return r_p["alpha"] / (r_d["alpha"] + 1e-30)

    # ---- (3) the empty reference, on the SAME differentiable path ---------
    t0 = time.time()
    rho_empty = jnp.zeros((2,) + box.lo.shape, dtype=jnp.float32)
    s21_empty = s21_from_fields(*fields_from_rho(rho_empty))
    budget.fwd(); budget.forward_empty_ref += 1
    s21_empty_mag = jax.lax.stop_gradient(jnp.abs(s21_empty) + 1e-30)
    emag = np.asarray(s21_empty_mag)
    print(f"[armAB:{arm}] differentiable empty line: |S21| in "
          f"[{emag.min():.3e}, {emag.max():.3e}] "
          f"(ripple {20*np.log10(emag.max()/emag.min()):.2f} dB) "
          f"[{time.time()-t0:.0f}s]")
    # A record shorter than the port-to-port transit has no transmitted wave in
    # it at all, so |S21_empty| collapses to numerical dust and every IL in the
    # objective becomes a ratio of two dusts. Measured here at 1 period: the
    # empty line read |S21| = 3e-5 and the surrogate was meaningless. This is
    # NOT the settling gate (that is -40 dB on the imperative record); it is the
    # much weaker precondition that the wave arrived at all.
    t_transit = fx.L_LINE * math.sqrt(fx.EPS_EFF) / C0
    t_record = n_steps * float(grid.dt)
    print(f"[armAB:{arm}] record {t_record*1e12:.0f} ps vs port-to-port transit "
          f"{t_transit*1e12:.0f} ps ({t_record/t_transit:.2f}x)")
    if t_record < 2.0 * t_transit or float(emag.max()) < 1e-3:
        msg = (f"the descent record ({t_record*1e12:.0f} ps, "
               f"{args.periods:g} periods) is too short for this line "
               f"({t_transit*1e12:.0f} ps transit) and/or the empty-line "
               f"|S21|max = {float(emag.max()):.2e} has collapsed. The IL "
               f"surrogate would be a ratio of numerical dust.")
        if SMOKE:
            print(f"[armAB:{arm}] !! {msg} SMOKE continues anyway; nothing "
                  f"from this run means anything physically.")
        else:
            raise SystemExit(msg + " Use the Stage-0 descent window (45 "
                             "periods).")
    print(f"[armAB:{arm}]   the plane extractor is documented as diverged in "
          f"ABSOLUTE scale; the objective divides it out, which is why the DUT "
          f"and this reference must come from the same path.")

    # ---- initialisation ---------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    noise = 0.01 * jax.random.normal(key, (2,) + box.lo.shape, dtype=jnp.float32)
    if args.init == "uniform":
        theta0 = 0.0 + noise                              # rho ~ 0.50
    elif args.init == "low":
        theta0 = -1.0986 + noise                          # rho ~ 0.25
    else:
        # The classical two-stub design as a soft seed, through the SAME
        # rasterizer arm D uses -- one lambda/4 stub per band, 8 mm apart, on
        # opposite sides of the trace.
        x_c = (0.5 * (box.hi.ix_lo + box.hi.ix_hi) - box.hi.pads[0]) * dx
        stubs = [("lo", x_c - 4.0e-3, fx.W_TRACE, fx.quarter_wave(5.25e9)),
                 ("hi", x_c + 4.0e-3, fx.W_TRACE, fx.quarter_wave(5.775e9))]
        seed = fx.mask_from_stubs(stubs, box)
        seed2 = np.stack([np.asarray(seed[n], dtype=np.float32) for n in fx.SIDES])
        theta0 = jnp.asarray(np.where(seed2 > 0.5, 2.0, -2.0),
                             dtype=jnp.float32) + noise

    # ---- the loss ---------------------------------------------------------
    def loss(theta, radius_cells: float, beta: float):
        rho_phys = rho_phys_from_theta(theta, radius_cells, beta)
        s21 = s21_from_fields(*fields_from_rho(rho_phys))
        if args.surrogate == "power":
            j, terms = surrogate_power(s21, s21_empty_mag, seg, thr)
            il = il_db_surrogate(s21, s21_empty_mag)
        else:
            il = il_db_surrogate(s21, s21_empty_mag)
            j, terms = surrogate_metric(il, seg, thr, args.tau_db, args.kappa_db)
        aux = dict(terms=terms, il_db=il, rho=rho_phys)
        return j, aux

    grad_fn = jax.value_and_grad(loss, has_aux=True)      # NO jax.jit -- see docstring

    # ---- (gate) sigma_override must not be silently ignored ---------------
    sigma_gate = None
    if args.sigma_gate:
        s0 = schedule[0]
        rho0 = rho_phys_from_theta(theta0, s0["filter_radius_cells"], s0["beta"])
        occ0, sig0 = fields_from_rho(rho0)
        sig_big = jnp.zeros_like(sig0)
        for _, s in sides:
            sig_big = sig_big.at[s.ix_lo:s.ix_hi, s.iy_lo:s.iy_hi,
                                 s.iz:s.iz + s.nz].set(1.0e4)
        s_a = s21_from_fields(occ0, jnp.zeros_like(sig0))
        s_b = s21_from_fields(occ0, sig_big)
        budget.fwd(2)
        rel = float(jnp.max(jnp.abs(jnp.abs(s_a) - jnp.abs(s_b))
                            / (jnp.abs(s_a) + 1e-30)))
        sigma_gate = rel
        print(f"[armAB:{arm}] sigma-effect gate: max relative change {rel:.3e}")
        if rel < 1e-3:
            raise SystemExit(
                "sigma_override appears to be silently ignored alongside "
                "pec_occupancy_override -- STOP. The conductivity "
                "interpolation is the whole scheme; without it arm A is not "
                "arm A.")

    # ---- gradient at init -------------------------------------------------
    s0 = schedule[0]
    t0 = time.time()
    (j0, aux0), g0 = grad_fn(theta0, s0["filter_radius_cells"], s0["beta"])
    budget.grad()
    g0_np = np.asarray(g0)
    gnorm = float(np.linalg.norm(g0_np))
    ginf = float(np.max(np.abs(g0_np)))
    n_zero = int(np.sum(g0_np == 0.0))
    print(f"[armAB:{arm}] init: J={float(j0):.6f}  ||grad||_2={gnorm:.6e}  "
          f"||grad||_inf={ginf:.6e}  exactly-zero cells {n_zero}/{g0_np.size}  "
          f"[{time.time()-t0:.0f}s]")
    if not np.isfinite(gnorm) or gnorm == 0.0:
        raise SystemExit("the gradient at init is zero or non-finite -- the "
                         "differentiable path is broken, STOP.")

    # ---- directional-derivative Richardson check (PLAN Stage-2 gate) ------
    gradcheck = None
    if args.gradcheck:
        kd = jax.random.PRNGKey(args.seed + 977)
        d = jax.random.normal(kd, theta0.shape, dtype=jnp.float32)
        d = d / jnp.linalg.norm(d)
        ad = float(jnp.sum(g0 * d))
        rows = []
        for eps in (1e-2, 5e-3):
            jp, _ = loss(theta0 + eps * d, s0["filter_radius_cells"], s0["beta"])
            jm, _ = loss(theta0 - eps * d, s0["filter_radius_cells"], s0["beta"])
            budget.fwd(2)
            fd = float((jp - jm) / (2 * eps))
            rows.append(dict(eps=eps, fd=fd, ad=ad,
                             rel_err=abs(ad - fd) / (abs(fd) + 1e-30)))
            print(f"[armAB:{arm}] gradcheck eps={eps:.0e}  AD.d={ad:+.6e}  "
                  f"FD={fd:+.6e}  rel={rows[-1]['rel_err']:.4f}")
        # central differences are O(eps^2): (4*fd(h/2) - fd(h)) / 3
        rich = (4.0 * rows[1]["fd"] - rows[0]["fd"]) / 3.0
        gradcheck = dict(direction="random unit, all variables", ad=ad,
                         richardson=rich,
                         rel_err_richardson=abs(ad - rich) / (abs(rich) + 1e-30),
                         steps=rows)
        print(f"[armAB:{arm}] gradcheck Richardson FD={rich:+.6e}  "
              f"rel={gradcheck['rel_err_richardson']:.4f}   (a DIRECTIONAL "
              f"derivative over all {theta0.size} variables is far better "
              f"conditioned in float32 than the per-cell FD Phase-1 used)")

    # ---- the frozen imperative evaluation ---------------------------------
    def binarise(rho_phys) -> dict:
        r = np.asarray(rho_phys)
        return {n: (r[k] > 0.5).astype(np.uint8) for k, n in enumerate(fx.SIDES)}

    empty_refs: dict = {}

    def empty_ref(periods: float):
        """The IMPERATIVE empty line, solved once per window and cached.

        Fetched explicitly rather than left to ``score_design`` so the solve it
        may cost lands in the budget instead of vanishing inside a helper.
        """
        key = float(periods)
        if key not in empty_refs:
            e = cal.empty_reference(f_score, key, cache_dir=CACHE, verbose=True)
            if not e.cached:
                budget.imp(1)
            empty_refs[key] = e
            print(f"[armAB:{arm}] {e.summary()}")
            if e.z0_warnings:
                for w in e.z0_warnings:
                    print(f"[armAB:{arm}]   empty-line Z0 warning: {w[:160]}")
        return empty_refs[key]

    def frozen_score(rho_phys, periods: float, label: str) -> dict:
        """rho -> binarise -> hard PEC boxes -> imperative solve -> frozen M.

        THE ONLY SOURCE OF A REPORTED NUMBER IN THIS FILE.
        """
        mask = binarise(rho_phys)
        geo = realized(mask)
        t = time.time()
        sc = cal.score_design(mask, freqs_hz=f_score, num_periods=periods,
                              cache_dir=CACHE, label=label,
                              empty=empty_ref(periods),
                              require_settled_empty=not allow_unsettled,
                              verbose=False)
        budget.imp(1)                       # the design solve
        r = sc.result
        rec = dict(label=label, periods=periods,
                   M=float(r.M), S_L=float(r.S_L), S_U=float(r.S_U),
                   S_G=float(r.S_G), S_P=float(r.S_P), Omega=float(r.Omega),
                   R_L_raw=float(r.R_L_raw), R_U_raw=float(r.R_U_raw),
                   f_notch_L_MHz=float(r.f_notch_L_MHz),
                   f_notch_U_MHz=float(r.f_notch_U_MHz),
                   IL_gap_max=float(r.IL_gap_max),
                   IL_pass_max=float(r.IL_pass_max),
                   spec_pass=bool(r.spec_pass), degenerate=bool(r.degenerate),
                   valid=bool(sc.validity.ok),
                   settled=bool(sc.validity.settled),
                   settling_worst_db=float(sc.validity.settling_worst_db),
                   passivity_worst=float(sc.validity.passivity_worst),
                   empty_cal_max_db=float(sc.empty_cal_max_db),
                   empty_key=sc.empty_key, n_boxes=int(sc.n_boxes),
                   realized=geo, wall_s=round(time.time() - t, 1),
                   result=r.as_dict())
        print(f"[armAB:{arm}] FROZEN {label}: {sc.summary()}  "
              f"cells={geo['cells_total']}  ({rec['wall_s']:.0f}s)")
        return rec

    # ---- Adam with filter-radius (and, arm B, beta) continuation ----------
    import optax
    opt = optax.adam(args.lr)
    theta = theta0
    state = opt.init(theta)
    history, frozen = [], []
    t_run = time.time()
    last_stage = None
    for it in range(args.iters):
        st = stage_at(schedule, it)
        if st["stage"] != last_stage:
            print(f"[armAB:{arm}] --- stage {st['stage']}: r = "
                  f"{st['filter_radius_mm']:.3f} mm "
                  f"({st['filter_radius_cells']:.2f} cells), beta = "
                  f"{st['beta'] if st['beta'] else 'none'} ---")
            last_stage = st["stage"]
        t_it = time.time()
        (j_val, aux), g = grad_fn(theta, st["filter_radius_cells"], st["beta"])
        budget.grad()
        upd, state = opt.update(g, state)
        theta = optax.apply_updates(theta, upd)
        rho_np = np.asarray(aux["rho"])
        h = dict(iter=it, stage=st["stage"],
                 filter_radius_m=st["filter_radius_m"],
                 filter_radius_cells=st["filter_radius_cells"],
                 beta=st["beta"], J=float(j_val),
                 terms={k: float(v) for k, v in aux["terms"].items()},
                 grad_norm=float(jnp.linalg.norm(g)),
                 fill=float(rho_np.mean()), gray=grayness(rho_np),
                 binary_cells=int((rho_np > 0.5).sum()),
                 wall_s=round(time.time() - t_it, 1),
                 budget=budget.as_dict())
        history.append(h)
        print(f"[armAB:{arm}] it={it:4d} st={st['stage']} J={float(j_val):9.4f} "
              f"[L {h['terms']['s_L']:6.2f} U {h['terms']['s_U']:6.2f} "
              f"G {h['terms']['s_G']:6.2f} P {h['terms']['s_P']:6.2f}] "
              f"|g|={h['grad_norm']:.3e} fill={h['fill']:.3f} "
              f"gray={h['gray']:.3f} solves={budget.total_descent} "
              f"({h['wall_s']:.0f}s)", flush=True)

        if args.score_every and (it + 1) % args.score_every == 0 and it + 1 < args.iters:
            rho_now = rho_phys_from_theta(theta, st["filter_radius_cells"], st["beta"])
            rec = frozen_score(rho_now, args.periods, f"{tag}_it{it+1}")
            # The differentiable IL trace alongside the imperative one, on the
            # record, so surrogate/score divergence is a number rather than an
            # impression. This is the comparison the Phase-1 retraction says
            # was never made.
            rec.update(iter=it + 1, J_surrogate=float(j_val),
                       il_db_descent=[float(v) for v in np.asarray(aux["il_db"])],
                       budget=budget.as_dict())
            frozen.append(rec)
            _write(tag, args, arm, verify_periods, schedule, seg, budget,
                   history, frozen, None, theta, rho_now, gnorm, ginf,
                   gradcheck, sigma_gate, pre, box, n_steps, k_seg,
                   f_desc, f_score, allow_unsettled)

    # ---- final: binarise, score at the verification window ----------------
    st = schedule[-1]
    rho_final = rho_phys_from_theta(theta, st["filter_radius_cells"], st["beta"])
    final = frozen_score(rho_final, verify_periods, f"{tag}_final")
    final.update(iter=args.iters,
                 # from the LAST DESCENT ITERATION, i.e. one Adam step before
                 # this theta -- labelled rather than silently attached, since
                 # recomputing it at the final theta would cost another solve.
                 J_surrogate_last_iter=(float(history[-1]["J"]) if history
                                        else None),
                 budget=budget.as_dict())

    print(f"[armAB:{arm}] DONE in {time.time()-t_run:.0f}s   "
          f"descent budget = {budget.total_descent} Maxwell solves "
          f"({budget.forward} fwd + {budget.backward} bwd), plus "
          f"{budget.imperative} imperative 2-port monitoring solves")
    if SMOKE:
        print(f"[armAB:{arm}] SMOKE: {args.periods:g} periods does NOT settle. "
              f"Every number above is a ring-down artifact and is stamped "
              f"quotable=false. This run proves the gradient flows and the "
              f"binarise-and-score path executes; it proves nothing else.")

    path = _write(tag, args, arm, verify_periods, schedule, seg, budget,
                  history, frozen, final, theta, rho_final, gnorm, ginf,
                  gradcheck, sigma_gate, pre, box, n_steps, k_seg,
                  f_desc, f_score, allow_unsettled)
    print(f"[armAB:{arm}] wrote {path} (+ .npz)")
    return 0


def _write(tag, args, arm, verify_periods, schedule, seg, budget, history,
           frozen, final, theta, rho_final, gnorm, ginf, gradcheck, sigma_gate,
           pre, box, n_steps, k_seg, f_desc, f_score, allow_unsettled) -> Path:
    """One record per run. Rewritten at every scoring point so a long GPU run
    is recoverable from disk without waiting for the end."""
    quotable = bool(final is not None and not SMOKE and final.get("valid")
                    and not allow_unsettled)
    out = dict(
        tag=tag, arm=arm, smoke=SMOKE, quotable=quotable,
        arm_description=("A: conductivity interpolation + filter-radius "
                         "continuation, NO projection (the established scheme)"
                         if arm == "A" else
                         "B: A + Heaviside projection with beta continuation "
                         "(the ablation)"),
        args=vars(args), verify_periods=verify_periods,
        fixture=dict(dx_m=float(box.dx), n_vars=int(box.n_vars),
                     side_shape=list(box.lo.shape), iz=int(box.iz),
                     nz=int(box.lo.nz), n_steps=int(n_steps),
                     checkpoint_segments=int(k_seg),
                     kottke=os.environ.get("RFX_PEC_OCC_KOTTKE"),
                     preflight=pre),
        grids=dict(descent_mhz=[int(round(v / 1e6)) for v in f_desc],
                   scoring_mhz=[int(round(v / 1e6)) for v in f_score],
                   segments=dict(band_l=list(seg.band_l), band_u=list(seg.band_u),
                                 gap=list(seg.gap), pass_lo=list(seg.pass_lo),
                                 pass_hi=list(seg.pass_hi))),
        surrogate=dict(
            kind=args.surrogate, tau_db=args.tau_db, kappa_db=args.kappa_db,
            mapping={
                "s_L": "softplus(r_req - softmin_tau(il_capped over B_L))  ~  S_L",
                "s_U": "softplus(r_req - softmin_tau(il_capped over B_U))  ~  S_U",
                "s_G": "cap(softplus(softmax_tau(il_capped over gap) - il_gap))  ~  S_G",
                "s_P": "cap(bandwidth-weighted trapezoid of softplus(il_capped - il_pass))  ~  S_P",
                "J": "s_L + s_U + s_G + s_P  ~  M   (DESCENT ONLY)",
            },
            thresholds=dict(r_req_db=sd.SCORE.r_req_db, r_cap_db=sd.SCORE.r_cap_db,
                            il_gap_db=sd.SCORE.il_gap_db,
                            il_pass_db=sd.SCORE.il_pass_db,
                            term_cap_db=sd.SCORE.term_cap_db),
            note=("surrogate values are NEVER reported as results; every "
                  "reported number comes from phase2_calibrate.score_design "
                  "on the imperative hard-PEC path")),
        continuation=schedule,
        sigma_map=dict(kind="exponential", sigma_min_S_per_m=args.sigma_min,
                       sigma_max_S_per_m=args.sigma_max,
                       formula="sigma(rho) = sigma_min * (sigma_max/sigma_min)**rho"),
        init_gradient=dict(l2=gnorm, linf=ginf),
        gradcheck=gradcheck, sigma_gate_rel_change=sigma_gate,
        budget=budget.as_dict(),
        history=history, frozen_scores=frozen, final=final,
    )
    p = OUT / f"{tag}.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    np.savez(OUT / f"{tag}.npz", theta=np.asarray(theta),
             rho=np.asarray(rho_final),
             hard=(np.asarray(rho_final) > 0.5).astype(np.uint8))
    return p


if __name__ == "__main__":
    raise SystemExit(main())

"""The exact-lattice witness gate, shared by every slab-family crossval case.

Pre-declared in ``docs/design_notes/20260903_lattice_witness_standard.md``;
change the note (append-only) before changing a number here.

WHAT THIS IS. cv23 round 1 found that rfx's residual against the CONTINUUM
transfer matrix is not error at all: it is the Yee lattice's own second-order
term, reproduced to 3e-5 by an exact 1-D Yee-lattice time-harmonic solution
with no fitted parameter (cv23 note section 12.2). The continuum gate
(``|rfx - TMM| <= W_bin/W_mean + W_ADE``) stays exactly as it is and is not
widened anywhere. This module adds a SECOND, independent gate at every dx
rung a slab case runs:

    |rfx - lattice(f; material, d, dx, dt)|  <=  W_witness(f)

where ``lattice`` is the exact time-harmonic solution of the very lattice the
run stepped (``dispersive_eps.yee_lattice_slab_rt_model``) and ``W_witness``
is DERIVED from that model's own error budget -- the terms cv22 and cv23
already quantify -- never fitted to the residual it gates.

THE BUDGET. The lattice solution is the exact steady state of an INFINITE
record on an INFINITE lattice. Four things separate it from what the rig
measures, and only two of them are non-zero:

  (1) record truncation -- the rig records N steps; what is still ringing
      after step N is missing from both rFFTs. Bounded by the settling
      witness the case already gates (``tail.scat_refl_rel`` /
      ``tail.total_trans_rel`` against ``SETTLING_LIMIT``) and the slowest
      ring-down rate the case already derives (``run.record.rate_ring_1_s``,
      taken together with the measured ``tail.fitted_rate_*`` so a fit slower
      than the derivation -- cv22 note section 14.1 -- widens the bound, not
      the claim).
  (2) incident-reference truncation ("injection leakage" as the rig witnesses
      it) -- the 1-D auxiliary reference that forms the denominator of R and
      T is truncated at the same step. Bounded by ``tail.purity_inc_rel``
      against ``TAIL_PURITY_LIMIT``, with the differentiated Gaussian's own
      envelope rate at that level.
  (3) CPML round trip -- ZERO by construction, not by estimate: the rig sizes
      the box so the CPML round trip exceeds the record
      (``run.record.t_safe_cpml_steps >= run.n_steps``), so any CPML echo
      arrives after the record ends and is already inside term (1). Asserted,
      not modelled.
  (4) the 2-D rig vs the 1-D model -- ZERO by construction: at normal
      incidence with periodic y and a y-uniform TFSF plane wave, d/dy = 0 and
      the 2-D TMz update IS the 1-D Ez/Hy lattice; the TFSF auxiliary grid
      runs the same update at the same dx and dt, so the injected field is an
      exact lattice plane wave and R, T are lattice quantities. The vacuum
      lattice wavenumber is real over the whole band
      (``w_hat dx / 2c << 1``), so the 30-cell probe standoff is lossless and
      drops out of |.|^2.

  plus (5) float32 arithmetic -- the fields are float32
  (``rfx/core/yee.py::init_state`` default). Carried as a named term with the
  statistical (sqrt-accumulation) size; the coherent worst case is also
  computed and reported, and the note states that it is not reached.

FROM AMPLITUDE ERROR TO R AND T. With ``R = |S|^2/|I|^2`` (numpy rfft, no dt
factor), an absolute spectral error ``e_S`` on the scattered transform and
``e_I`` on the incident reference give, to first order,

    |dR| <= 2 sqrt(R) (e_S/|I|) + 2 R (e_I/|I|)

and likewise for T. The missing tail is bounded coherently:
``e_S <= sum_{n>=N} |s_n| <= A_tail * inc_peak / (1 - exp(-Gamma dt))``.
The denominator is the source's own spectrum: for the rig's differentiated
Gaussian ``s = -2 u exp(-u^2)``, ``u = (t-t0)/tau``, the continuous transform
is ``|S(w)| = w tau^2 sqrt(pi) exp(-w^2 tau^2/4)`` and the time-domain peak is
``sqrt(2) exp(-1/2)``, so

    |I(f)| = inc_peak * LAMBDA * a(f),   LAMBDA = sqrt(pi) tau / dt

with ``a(f)`` the relative incident amplitude the artifact already stores
(``inc_amp_rel``). Every level in the budget is therefore expressed relative
to the same ``inc_peak`` the witnesses use, and ``inc_peak`` cancels.

WHAT MAKES IT NOT-TUNED. Two statements, both checkable:
  * every input is either a pre-declared constant (SETTLING_LIMIT,
    TAIL_PURITY_LIMIT, the source tau, eps_f32) or a witness the case ALREADY
    gates for its own reasons (the tail levels, the derived ring rate, the
    CPML gate). None of them is the residual being gated.
  * ``ceiling_windows`` computes the same window with the DECLARED BARS in
    place of the measured levels. That number is available before any run and
    bounds every passing run's window from above.
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

import cv22_dispersive_gates as G  # noqa: E402
import dispersive_eps as de  # noqa: E402

SCHEMA = "lattice-witness/v1"

# float32 fields: rfx/core/yee.py::init_state(field_dtype=jnp.float32).
EPS_F32 = 2.0 ** -24

# The pre-declared witness bars this budget is built on (cv22 note sections 4,
# 13; cv04's #341 purity witness). Re-exported, never restated.
SETTLING_BAR = G.SETTLING_LIMIT          # 1e-2, the -40 dB settling witness
PURITY_BAR = G.TAIL_PURITY_LIMIT         # 1e-3, cv04's tail-purity witness

# The source the rig injects (rfx/sources/tfsf.py: s = -2 u exp(-u^2),
# u = (t - t0)/tau, tau = 1/(pi f0 bandwidth)); the same tau
# cv22_dispersive_gates.incident_amplitude_rel uses.
TAU_SRC_S = 1.0 / (math.pi * G.TFSF_F0_HZ * G.TFSF_BW)


# ---------------------------------------------------------------------------
# The lattice model
# ---------------------------------------------------------------------------

def lattice_rta(freqs_hz, model: str, params: dict, dx: float, dt: float,
                *, d_slab_m: float = G.D_SLAB_M):
    """R, T, A = 1 - R - T of the exact 1-D Yee lattice at (dx, dt).

    ``model`` is any of dispersive_eps.MODELS; the slab nodes realize the
    discrete-time permittivity of the update that ran (the ADE for cv22's
    poles, the semi-implicit sigma average for cv23's conductivity, the plain
    eps' for cv04's lossless slab).
    """
    R, T = de.yee_lattice_slab_rt_model(freqs_hz, model, params, d_slab_m, float(dx), float(dt))
    return R, T, 1.0 - R - T


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------

def source_spectral_gain(dt: float) -> float:
    """LAMBDA = sqrt(pi) tau / dt: the ratio of the discrete incident amplitude
    spectrum at its peak to the incident time-domain peak, for the rig's
    differentiated Gaussian. Analytic; the sampling correction at tau/dt ~ 27
    is below 1e-3 and the constant enters the window as a divisor."""
    return math.sqrt(math.pi) * TAU_SRC_S / float(dt)


def incident_tail_rate(purity_rel: float) -> float:
    """Envelope decay rate (1/s) of the differentiated-Gaussian incident at the
    level ``purity_rel`` of its peak: solve ``2 a exp(-a^2) = purity_rel`` for
    the late branch ``a > 1/sqrt(2)``; the envelope's logarithmic derivative
    there is ``2a/tau``."""
    y = float(purity_rel)
    if not (0.0 < y < math.sqrt(2.0) * math.exp(-0.5)):
        raise ValueError(f"purity_rel {y!r} is not on the decaying branch of the source envelope")
    lo, hi = 1.0 / math.sqrt(2.0), 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 2.0 * mid * math.exp(-mid * mid) > y:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    return 2.0 * a / TAU_SRC_S


def ringdown_rate(record: dict, tail: dict | None) -> tuple[float, str]:
    """The slowest amplitude decay rate to use for the truncation bound: the
    smaller of the case's own DERIVED ring-down rate
    (``run.record.rate_ring_1_s``) and any reliable fitted tail rate. cv22
    note section 14.1 measured a Debye tail decaying 1.5x SLOWER than the
    derivation, so taking the minimum is the conservative reading; the fit is
    a witness on the estimate, exactly as the record length treats it."""
    rates = {"derived": float(record["rate_ring_1_s"])}
    if tail:
        for k in ("fitted_rate_scat_refl_1_s", "fitted_rate_total_trans_1_s"):
            v = tail.get(k)
            if v is not None and np.isfinite(v) and float(v) > 0.0:
                rates[k] = float(v)
    src = min(rates, key=rates.get)
    return rates[src], src


def _geom_sum(rate: float, dt: float) -> float:
    """sum_{m>=0} exp(-rate m dt) = 1/(1 - exp(-rate dt))."""
    return 1.0 / (1.0 - math.exp(-float(rate) * float(dt)))


def budget_terms(freqs_hz, inc_amp_rel, *, dt: float, n_steps: int,
                 scat_tail_rel: float, trans_tail_rel: float, purity_rel: float,
                 rate_1_s: float) -> dict:
    """The four relative-amplitude error terms of the budget, per bin.

    Returns ``delta_scat``, ``delta_trans`` (record truncation on the two
    measured traces), ``delta_inc`` (incident-reference truncation) and
    ``delta_round`` (float32), each already divided by ``|I(f)|/inc_peak``.
    ``delta_round_coherent`` is the worst-case (fully in-phase) round-off
    alternative, reported, never gated.
    """
    a = np.asarray(inc_amp_rel, dtype=float)
    if np.any(a <= 0.0):
        raise ValueError("inc_amp_rel has a non-positive bin; the window would be undefined there")
    lam = source_spectral_gain(dt)
    kap = _geom_sum(rate_1_s, dt)
    rate_i = incident_tail_rate(purity_rel)
    kap_i = _geom_sum(rate_i, dt)
    n = int(n_steps)
    return {
        "lambda_source": lam,
        "kappa_ringdown": kap,
        "kappa_incident": kap_i,
        "rate_ringdown_1_s": float(rate_1_s),
        "rate_incident_1_s": float(rate_i),
        "delta_scat": float(scat_tail_rel) * kap / (lam * a),
        "delta_trans": float(trans_tail_rel) * kap / (lam * a),
        "delta_inc": float(purity_rel) * kap_i / (lam * a),
        "delta_round": n * EPS_F32 / math.sqrt(2.0) / (lam * a),
        "delta_round_coherent": n * n * EPS_F32 / 2.0 / (lam * a),
    }


def windows_from_terms(R_lat, T_lat, terms: dict):
    """W_witness,{R,T,A}(f) from the lattice prediction and the budget terms."""
    Rl = np.asarray(R_lat, dtype=float)
    Tl = np.asarray(T_lat, dtype=float)
    wR = 2.0 * np.sqrt(Rl) * (terms["delta_scat"] + terms["delta_round"]) + 2.0 * Rl * terms["delta_inc"]
    wT = 2.0 * np.sqrt(Tl) * (terms["delta_trans"] + terms["delta_round"]) + 2.0 * Tl * terms["delta_inc"]
    return wR, wT, wR + wT


def ceiling_windows(freqs_hz, inc_amp_rel, R_lat, T_lat, *, dt: float, n_steps: int,
                    rate_1_s: float):
    """The same window computed with the DECLARED BARS (settling 1e-2 on both
    traces, purity 1e-3) and the DERIVED ring rate only -- an a-priori number,
    available before any run, that bounds every passing run's window."""
    terms = budget_terms(freqs_hz, inc_amp_rel, dt=dt, n_steps=n_steps,
                         scat_tail_rel=SETTLING_BAR, trans_tail_rel=SETTLING_BAR,
                         purity_rel=PURITY_BAR, rate_1_s=rate_1_s)
    return windows_from_terms(R_lat, T_lat, terms)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def _f(x):
    return float(x)


def evaluate(arm_doc: dict, *, model: str | None = None, params: dict | None = None,
             d_slab_m: float = G.D_SLAB_M, tag: str | None = None) -> dict:
    """The lattice-witness gate for one arm at one dx rung.

    ``arm_doc`` is the per-arm block of a cv22 / cv23 ``rfx*.json`` (or the
    same shape built by any other slab case): ``freqs_hz``, ``gated``,
    ``R_rfx``, ``T_rfx``, ``dt_s``, ``inc_amp_rel``, ``tail``, ``params``,
    ``model`` and ``run`` (with ``n_steps``, ``dx_m`` and ``record``).

    Returns the same gate-record shape the other cv gates use: per-bin arrays,
    gated scalars, and a ``gates`` dict whose values are all booleans.
    """
    model = model or arm_doc["model"]
    params = params if params is not None else arm_doc["params"]
    f = np.asarray(arm_doc["freqs_hz"], dtype=float)
    g = np.asarray(arm_doc["gated"], dtype=bool)
    dt = float(arm_doc["dt_s"])
    run = arm_doc["run"]
    rec = run["record"]
    tail = arm_doc["tail"]
    dx = float(run["dx_m"])
    n_steps = int(run["n_steps"])

    R_x = np.asarray(arm_doc["R_rfx"], dtype=float)
    T_x = np.asarray(arm_doc["T_rfx"], dtype=float)
    A_x = 1.0 - R_x - T_x
    R_l, T_l, A_l = lattice_rta(f, model, params, dx, dt, d_slab_m=d_slab_m)

    rate, rate_src = ringdown_rate(rec, tail)
    terms = budget_terms(f, arm_doc["inc_amp_rel"], dt=dt, n_steps=n_steps,
                         scat_tail_rel=tail["scat_refl_rel"],
                         trans_tail_rel=tail["total_trans_rel"],
                         purity_rel=tail["purity_inc_rel"], rate_1_s=rate)
    wR, wT, wA = windows_from_terms(R_l, T_l, terms)
    cR, cT, cA = ceiling_windows(f, arm_doc["inc_amp_rel"], R_l, T_l, dt=dt,
                                 n_steps=n_steps, rate_1_s=float(rec["rate_ring_1_s"]))

    dR = np.abs(R_x - R_l)
    dT = np.abs(T_x - T_l)
    dA = np.abs(A_x - A_l)

    # The two terms the budget declares ZERO are asserted, not modelled.
    cpml_gate_ok = bool(int(rec.get("t_safe_cpml_steps", 0)) >= n_steps)
    tail_ok = bool(tail.get("ok", False))

    gates = {
        "precond_cpml_gate": cpml_gate_ok,
        "precond_tail_witness": tail_ok,
        "GL1_R": bool(np.all(dR[g] <= wR[g])),
        "GL1_T": bool(np.all(dT[g] <= wT[g])),
        "GL1_A": bool(np.all(dA[g] <= wA[g])),
        "GL2_R": bool(np.mean(dR[g]) <= np.mean(wR[g])),
        "GL2_T": bool(np.mean(dT[g]) <= np.mean(wT[g])),
        "GL2_A": bool(np.mean(dA[g]) <= np.mean(wA[g])),
    }
    out = {
        "schema": SCHEMA,
        "tag": tag,
        "model": model, "params": dict(params),
        "dx_m": dx, "dt_s": dt, "n_steps": n_steps, "dx_div": int(run.get("dx_div", 1)),
        "n_bins_gated": int(g.sum()),
        "R_lattice": R_l.tolist(), "T_lattice": T_l.tolist(), "A_lattice": A_l.tolist(),
        "dR_lattice": dR.tolist(), "dT_lattice": dT.tolist(), "dA_lattice": dA.tolist(),
        "W_witness_R": wR.tolist(), "W_witness_T": wT.tolist(), "W_witness_A": wA.tolist(),
        "budget": {
            "lambda_source": terms["lambda_source"],
            "kappa_ringdown": terms["kappa_ringdown"],
            "kappa_incident": terms["kappa_incident"],
            "rate_ringdown_1_s": terms["rate_ringdown_1_s"],
            "rate_ringdown_source": rate_src,
            "rate_incident_1_s": terms["rate_incident_1_s"],
            "scat_tail_rel": _f(tail["scat_refl_rel"]),
            "trans_tail_rel": _f(tail["total_trans_rel"]),
            "purity_rel": _f(tail["purity_inc_rel"]),
            "eps_f32": EPS_F32,
            "mean_delta_scat_gated": _f(np.mean(terms["delta_scat"][g])),
            "mean_delta_trans_gated": _f(np.mean(terms["delta_trans"][g])),
            "mean_delta_inc_gated": _f(np.mean(terms["delta_inc"][g])),
            "mean_delta_round_gated": _f(np.mean(terms["delta_round"][g])),
            "mean_delta_round_coherent_gated": _f(np.mean(terms["delta_round_coherent"][g])),
        },
        "mean_dR_lattice_gated": _f(dR[g].mean()), "max_dR_lattice_gated": _f(dR[g].max()),
        "mean_dT_lattice_gated": _f(dT[g].mean()), "max_dT_lattice_gated": _f(dT[g].max()),
        "mean_dA_lattice_gated": _f(dA[g].mean()), "max_dA_lattice_gated": _f(dA[g].max()),
        "mean_W_witness_R_gated": _f(wR[g].mean()), "mean_W_witness_T_gated": _f(wT[g].mean()),
        "mean_W_witness_A_gated": _f(wA[g].mean()),
        "mean_W_ceiling_R_gated": _f(cR[g].mean()), "mean_W_ceiling_T_gated": _f(cT[g].mean()),
        "mean_W_ceiling_A_gated": _f(cA[g].mean()),
        # The ceiling is a-priori in BOTH inputs: the declared bars AND the
        # DERIVED ring rate. A run whose tail is measured to decay slower than
        # the derivation (cv22 note section 14.1 saw exactly that on Debye) can
        # therefore carry a window above its own ceiling. Reported, never
        # silently absorbed: the flag says which term did it.
        "W_exceeds_ceiling_R": bool(np.mean(wR[g]) > np.mean(cR[g])),
        "W_exceeds_ceiling_T": bool(np.mean(wT[g]) > np.mean(cT[g])),
        "W_exceeds_ceiling_A": bool(np.mean(wA[g]) > np.mean(cA[g])),
        "worst_ratio_R": _f(np.max(dR[g] / wR[g])), "worst_ratio_T": _f(np.max(dT[g] / wT[g])),
        "worst_ratio_A": _f(np.max(dA[g] / wA[g])),
        "n_bins_R_over_window": int(np.sum(dR[g] > wR[g])),
        "n_bins_T_over_window": int(np.sum(dT[g] > wT[g])),
        "n_bins_A_over_window": int(np.sum(dA[g] > wA[g])),
        "gates": gates,
    }
    out["witness_ok"] = bool(all(gates.values()))
    return out


# ---------------------------------------------------------------------------
# Falsifiers -- analytic, no FDTD
# ---------------------------------------------------------------------------

def defective_params(kind: str, model: str, params: dict) -> dict:
    """F3's defect: the DISPERSIONLESS part of the permittivity off by 1 %."""
    if kind != "eps_x1p01":
        raise ValueError(kind)
    return {**params, "eps_inf": float(params["eps_inf"]) * 1.01}


def model_separation(freqs_hz, model: str, params: dict, dx: float, dt: float,
                     kind: str, *, d_slab_m: float = G.D_SLAB_M):
    """|defective model - declared lattice| per bin, for R, T and A -- the
    a-priori margin a falsifier must clear. ``kind``:

      ``thickness_plus_cell`` / ``thickness_minus_cell``
          the lattice built with one E node more / fewer in the slab: a
          one-cell thickness error at this rung.
      ``continuum``
          the CONTINUUM transfer matrix used as the witness model instead of
          the lattice -- the deliberately wrong model of F2. The separation is
          the lattice's own second-order term W_lat(f).
      ``eps_x1p01``
          the declared lattice with eps' (eps_inf) 1 % high.
    """
    R0, T0, A0 = lattice_rta(freqs_hz, model, params, dx, dt, d_slab_m=d_slab_m)
    if kind == "thickness_plus_cell":
        R1, T1, A1 = lattice_rta(freqs_hz, model, params, dx, dt, d_slab_m=d_slab_m + dx)
    elif kind == "thickness_minus_cell":
        R1, T1, A1 = lattice_rta(freqs_hz, model, params, dx, dt, d_slab_m=d_slab_m - dx)
    elif kind == "continuum":
        R1, T1 = de.tmm_slab_rt(freqs_hz, de.eps_analytic(freqs_hz, model, params), d_slab_m)
        A1 = 1.0 - R1 - T1
    elif kind == "eps_x1p01":
        R1, T1, A1 = lattice_rta(freqs_hz, model, defective_params(kind, model, params),
                                 dx, dt, d_slab_m=d_slab_m)
    else:
        raise ValueError(kind)
    return np.abs(R1 - R0), np.abs(T1 - T0), np.abs(A1 - A0)


def evaluate_falsifier(arm_doc: dict, kind: str, *, model: str | None = None,
                       params: dict | None = None, d_slab_m: float = G.D_SLAB_M) -> dict:
    """Re-run ``evaluate`` on the SAME committed measurement with the witness
    model replaced by the defective one of ``kind``. The gate must fail; the
    record says by how much and on which observable. No FDTD.

    ``continuum`` is expressed by swapping in the transfer matrix, which is
    what ``kind='continuum'`` means for the model -- so it is evaluated
    directly rather than through ``lattice_rta``.
    """
    f = np.asarray(arm_doc["freqs_hz"], dtype=float)
    g = np.asarray(arm_doc["gated"], dtype=bool)
    base = evaluate(arm_doc, model=model, params=params, d_slab_m=d_slab_m)
    mdl = model or arm_doc["model"]
    prm = params if params is not None else arm_doc["params"]
    dx, dt = float(arm_doc["run"]["dx_m"]), float(arm_doc["dt_s"])
    if kind == "thickness_plus_cell":
        R1, T1, A1 = lattice_rta(f, mdl, prm, dx, dt, d_slab_m=d_slab_m + dx)
    elif kind == "thickness_minus_cell":
        R1, T1, A1 = lattice_rta(f, mdl, prm, dx, dt, d_slab_m=d_slab_m - dx)
    elif kind == "continuum":
        R1, T1 = de.tmm_slab_rt(f, de.eps_analytic(f, mdl, prm), d_slab_m)
        A1 = 1.0 - R1 - T1
    elif kind == "eps_x1p01":
        R1, T1, A1 = lattice_rta(f, mdl, defective_params(kind, mdl, prm), dx, dt, d_slab_m=d_slab_m)
    else:
        raise ValueError(kind)
    R_x = np.asarray(arm_doc["R_rfx"], dtype=float)
    T_x = np.asarray(arm_doc["T_rfx"], dtype=float)
    A_x = 1.0 - R_x - T_x
    wR = np.asarray(base["W_witness_R"]); wT = np.asarray(base["W_witness_T"])
    wA = np.asarray(base["W_witness_A"])
    dR, dT, dA = np.abs(R_x - R1), np.abs(T_x - T1), np.abs(A_x - A1)
    sepR, sepT, sepA = model_separation(f, mdl, prm, dx, dt, kind, d_slab_m=d_slab_m)
    gates = {
        "GL1_R": bool(np.all(dR[g] <= wR[g])), "GL1_T": bool(np.all(dT[g] <= wT[g])),
        "GL1_A": bool(np.all(dA[g] <= wA[g])),
        "GL2_R": bool(np.mean(dR[g]) <= np.mean(wR[g])),
        "GL2_T": bool(np.mean(dT[g]) <= np.mean(wT[g])),
        "GL2_A": bool(np.mean(dA[g]) <= np.mean(wA[g])),
    }
    return {
        "kind": kind, "gates": gates,
        "witness_ok": bool(all(gates.values())),
        "n_bins_R_over_window": int(np.sum(dR[g] > wR[g])),
        "n_bins_T_over_window": int(np.sum(dT[g] > wT[g])),
        "n_bins_A_over_window": int(np.sum(dA[g] > wA[g])),
        "n_bins_gated": int(g.sum()),
        "mean_separation_R_gated": _f(sepR[g].mean()),
        "mean_separation_T_gated": _f(sepT[g].mean()),
        "mean_separation_A_gated": _f(sepA[g].mean()),
        "separation_over_window_R": _f(sepR[g].mean() / np.mean(wR[g])),
        "separation_over_window_T": _f(sepT[g].mean() / np.mean(wT[g])),
        "separation_over_window_A": _f(sepA[g].mean() / np.mean(wA[g])),
        "mean_dR_gated": _f(dR[g].mean()), "mean_dT_gated": _f(dT[g].mean()),
        "mean_dA_gated": _f(dA[g].mean()),
    }


FALSIFIER_KINDS = ("thickness_plus_cell", "thickness_minus_cell", "continuum", "eps_x1p01")


# ---------------------------------------------------------------------------
# Case-level driver: one JSON per case, one entry per arm x rung
# ---------------------------------------------------------------------------

def witness_document(case_id: str, entries: dict, *, commit: str | None = None,
                     falsifiers: bool = True, d_slab_m: float = G.D_SLAB_M) -> dict:
    """Build ``lattice_witness.json`` for a case.

    ``entries`` maps a rung name ("<arm>" or "<arm>_dx2") to the per-arm doc
    that ``evaluate`` takes. Every entry gets the gate and, unless
    ``falsifiers=False``, the four analytic falsifiers replayed against it.
    """
    doc = {"schema": SCHEMA, "case_id": case_id, "commit": commit,
           "d_slab_m": float(d_slab_m), "rungs": {}}
    ok = True
    for name, arm_doc in entries.items():
        rec = evaluate(arm_doc, d_slab_m=d_slab_m, tag=name)
        if falsifiers:
            rec["falsifiers"] = {k: evaluate_falsifier(arm_doc, k, d_slab_m=d_slab_m)
                                 for k in FALSIFIER_KINDS}
        doc["rungs"][name] = rec
        ok = ok and rec["witness_ok"]
    doc["verdict"] = {"all_rungs_ok": bool(ok), "n_rungs": len(doc["rungs"])}
    return doc


def witness_json_name() -> str:
    """The per-case artifact this module writes, beside the case's other JSONs."""
    return "lattice_witness.json"


def rungs_from_results(results_dir: str) -> dict:
    """Every committed dx rung of the DECLARED material in ``results_dir``.

    Reads ``rfx.json`` (one rung per arm, named after the arm) and every
    ``rfx__<tag>.json`` that is not a falsifier and not a smoke run (the dx
    ladder: ``rfx__tand3_dx2.json`` and friends, named after the tag). A
    falsifier artifact runs a DIFFERENT material from the declared one and is
    not a rung of this ladder; it is exercised by the gate test instead, which
    judges it against its own ``params_run``.
    """
    import glob
    import json as _json
    entries = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "rfx.json"))
                       + glob.glob(os.path.join(results_dir, "rfx__*.json"))):
        base = os.path.basename(path)
        if base.startswith("rfx__falsifier_"):
            continue
        with open(path) as fh:
            doc = _json.load(fh)
        if doc.get("falsifier") or doc.get("smoke"):
            continue
        tag = doc.get("tag")
        for arm, ad in doc.get("arms", {}).items():
            if "inc_amp_rel" not in ad or "record" not in (ad.get("run") or {}):
                continue
            entries[tag or arm] = ad
    return entries


def build_from_results(case_id: str, results_dir: str, *, commit: str | None = None,
                       d_slab_m: float = G.D_SLAB_M) -> dict:
    """``witness_document`` over every rung ``rungs_from_results`` finds. No FDTD:
    this is post-processing of the case's own committed artifacts, the same
    class as ``--meep-ladder-summary`` and ``--refit-tail-fits``."""
    return witness_document(case_id, rungs_from_results(results_dir),
                            commit=commit, d_slab_m=d_slab_m)

"""Gate arithmetic for the WR-90 chain battery (v1.8 WP2) — ONE copy.

Consumers:

* ``scripts/diagnostics/waveguide_chain_battery_measure.py`` — the measurement
  driver calls these to write the per-cell metrics, the ladder, the
  plane-shift rotations, the referee and the ``verdicts`` block of
  ``tests/fixtures/waveguide_chain_battery/fixture.json``.
* ``tests/test_waveguide_chain_battery.py`` — the replay layer recomputes every
  verdict from the stored numbers through the same functions and compares with
  the stored verdict; a disagreement is itself a failure (README, "verdicts").

Every tolerance here is an EXISTING gate imported from where it lives, or a
number the pre-declaration ``docs/design_notes/waveguide_chain_battery_predeclaration.md``
fixed before the first run (section and source quoted next to each). Nothing
is chosen here. Nothing here runs an FDTD step; the module is pure NumPy on
arrays the driver or the fixture provides.

Sign conventions used throughout (pre-declaration §5(b), §5(d)):

* time convention ``exp(+jωt)``; a forward (+x) wave carries ``exp(-jβx)``;
* the extractor shifts modal waves by ``exp(∓jβ·s)`` with
  ``s = shift_m · step_sign`` (``rfx/sources/waveguide_port.py::_shift_modal_waves``),
  so moving BOTH default reference planes inward by ``Δ_L`` / ``|Δ_R|`` rotates
  ``∠S11`` by ``+2β·Δ_L``, ``∠S22`` by ``+2β·|Δ_R|`` and ``∠S21 = ∠S12`` by
  ``+β·(Δ_L + |Δ_R|)``;
* the Airy oracle is moved to the default planes by ``exp(-2jβ_v d_L)`` (S11)
  and ``exp(-jβ_v (d_L + d_R))`` (S21), continuous vacuum β
  (``scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py``);
* the PEC-short phase oracle at a default plane ``d`` from the face is
  ``π − 2βd`` (reflection −1 at the face, round trip ``−2βd``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "diagnostics") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))

from build_waveguide_band_broad_e5_envelope import (  # type: ignore  # noqa: E402
    MAX_TOL,
    airy_slab,
)
from build_waveguide_band_broad_e5_phase_envelope import (  # type: ignore  # noqa: E402
    MAX_PHASE_TOL_DEG,
    PHASE_MAG_FLOOR,
)

from rfx.api._sparams import _SETTLING_WITNESS_DB  # noqa: E402
from rfx.sources.waveguide_port import C0_LOCAL, _compute_beta  # noqa: E402

from tests import _waveguide_chain_battery_fixture as F  # noqa: E402
from tests._gate_policy import gate_from_envelope  # noqa: E402
from tests.test_msl_ad_fd_converged import (  # noqa: E402
    _MIN_FD_ULP_SPAN,
    _fd_ulp_span,
)
from tests.test_waveguide_phase_gate import MAG_MASK_FRAC, PHASE_TOL_DEG  # noqa: E402
from build_waveguide_band_broad_e5_envelope import _committed_noise_floor  # type: ignore  # noqa: E402

# --- pre-declared gates, each with its source -------------------------------
SETTLING_DB_MAX = float(_SETTLING_WITNESS_DB)          # −40 dB, CLAUDE.md ring-down rule
NON_VACUITY_MIN_MAX_S11 = 0.20                          # twoport_contract_v1.py:266
ABS_S_INVARIANCE_RTOL = 1e-3                            # twoport_contract_v1.py:270
ABS_S_INVARIANCE_ATOL = 1e-4                            # twoport_contract_v1.py:270
ROTATION_TOL_YEE_DEG = 3.0                              # test_waveguide_phase_gate.py:259
ROTATION_TOL_CONTINUOUS_DEG = float(PHASE_TOL_DEG)      # 6.0, phase_gate.py:63
WRONG_SIGN_MIN_DEG = 10.0                               # phase_gate.py:266
ROTATION_MAG_MASK_FRAC = float(MAG_MASK_FRAC)           # 0.05 of the entry's band peak, phase_gate.py:68
ROTATION_NOISE_FLOOR = float(_committed_noise_floor())  # measured empty-guide |S| floor (~7e-4): an entry whose
                                                        # band peak sits below it carries no measurable phase
AD_FD_REL_GATE = 0.05                                   # sparam_ad_end_to_end.py:298, flux_ad.py:84
FD_ULP_FLOOR = float(_MIN_FD_ULP_SPAN)                  # 1e4, msl_ad_fd_converged.py:136
FORWARD_IDENTITY_RTOL = 1e-5                            # flux_ad.py:104
FORWARD_IDENTITY_ATOL = 1e-7                            # flux_ad.py:104
GRADIENT_REPORT_BAR = 1e-2                              # pre-declaration §5(b), report-first
GRADIENT_PIN_QUANTUM = 1000                             # gate_from_envelope(measured, quantum=1000)
LADDER_FLOOR_MAG = 0.005                                # port_validation_battery.py:474
LADDER_FLOOR_PHASE_DEG = 1.0                            # pre-declaration §5(c)
LADDER_RATIO_WINDOW = (0.15, 0.70)                      # pre-declaration §5(c), guard 2
PEC_SHORT_S11_MIN = 0.99                                # port_validation_battery.py:541
PEC_SHORT_S11_MAX = 1.03                                # port_validation_battery.py:550
PEC_SHORT_MEAN_TOL = 0.02                               # port_validation_battery.py:554
SLAB_AIRY_MAG_TOL = float(MAX_TOL)                      # 0.05, envelope.py:33
SLAB_AIRY_PHASE_TOL_DEG = float(MAX_PHASE_TOL_DEG)      # 15.0, phase_envelope.py:99
SLAB_AIRY_PHASE_MAG_FLOOR = float(PHASE_MAG_FLOOR)      # 0.30, phase_envelope.py — part of that gate
COLUMN_POWER_MAX = 1.02                                 # port_validation_battery.py:307
RECIPROCITY_MAG_MAX = 0.01                              # port_validation_battery.py:340
RECIPROCITY_COMPLEX_MAX = 0.01                          # pre-declaration §6, first measurement

RUNG_LABELS = ("coarse", "mid", "fine")
RUNG_DX = dict(zip(RUNG_LABELS, F.DX_LADDER))
LANE_LABELS = {False: "false", "flux": "flux"}
LANE_FROM_LABEL = {"false": False, "flux": "flux"}
CLAIMS_RUNG = "fine"                                     # pre-declaration §2.6
LEGS_RUNG_DEFAULT = "fine"                               # AD / FD / plane legs

# Distance from each DEFAULT reference plane to the near DUT face (§2.3).
D_PLANE_TO_PEC_FACE_M = F.PEC_SHORT_X_M[0] - F.REF_LEFT_DEFAULT_M     # 0.03810
D_PLANE_TO_SLAB_FACE_M = F.SLAB_X_M[0] - F.REF_LEFT_DEFAULT_M         # 0.03556
SLAB_THICKNESS_M = F.SLAB_X_M[1] - F.SLAB_X_M[0]                      # 0.01016
SHIFT_LEFT_M = F.REF_LEFT_SHIFTED_M - F.REF_LEFT_DEFAULT_M            # +0.01016
SHIFT_RIGHT_M = F.REF_RIGHT_SHIFTED_M - F.REF_RIGHT_DEFAULT_M         # −0.01270
FC_TE10_HZ = C0_LOCAL / (2.0 * F.A_M)

OBJECTIVES = {
    # name: (kind, port-entry (recv, drive), function of the complex entry)
    "s11_mag2": ("magnitude", (0, 0)),
    "s21_mag2": ("magnitude", (1, 0)),
    "re_s21": ("complex", (1, 0)),
    "im_s21": ("complex", (1, 0)),
    "re_s11": ("complex", (0, 0)),
    "im_s11": ("complex", (0, 0)),
}
# Legs of §5(a): (dut, theta_kind) -> objectives. The PEC-short |S11|² under
# a lossless eps θ is the pre-declared expected ULP-floor skip.
AD_LEGS = {
    ("slab", "eps"): ("s11_mag2", "s21_mag2", "re_s21", "im_s21"),
    ("pec_short", "sigma"): ("s11_mag2",),
    ("pec_short", "eps"): ("s11_mag2", "re_s11", "im_s11"),
}
EXPECTED_ULP_SKIP = {("pec_short", "eps", "s11_mag2")}
LADDER_OBSERVABLES = (
    # name, dut, entry, kind
    ("slab_s11_mag", "slab", (0, 0), "mag"),
    ("slab_s21_mag", "slab", (1, 0), "mag"),
    ("slab_s21_phase_deg", "slab", (1, 0), "phase"),
    ("pec_short_s11_mag", "pec_short", (0, 0), "mag"),
    ("pec_short_s11_phase_deg", "pec_short", (0, 0), "phase"),
)


# --- small helpers ----------------------------------------------------------

def wrap_deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.angle(np.exp(1j * np.radians(np.asarray(x, dtype=float)))))


def objective_value(S: np.ndarray, name: str):
    """Scalar objective of an S-matrix at the band-centre bin. Works on numpy
    and on jax arrays (only ``abs``/``real``/``imag`` and ``**`` are used)."""
    _, (i, j) = OBJECTIVES[name]
    s = S[i, j, F.BAND_CENTRE_BIN]
    if name.endswith("_mag2"):
        return abs(s) ** 2
    if name.startswith("re_"):
        return s.real
    return s.imag


def s_to_json(S: np.ndarray) -> dict:
    S = np.asarray(S)
    return {
        "S11": [[float(z.real), float(z.imag)] for z in S[0, 0]],
        "S21": [[float(z.real), float(z.imag)] for z in S[1, 0]],
        "S12": [[float(z.real), float(z.imag)] for z in S[0, 1]],
        "S22": [[float(z.real), float(z.imag)] for z in S[1, 1]],
    }


def s_from_json(d: dict) -> np.ndarray:
    def col(k):
        return np.array([complex(a, b) for a, b in d[k]])
    S = np.zeros((2, 2, len(d["S11"])), dtype=complex)
    S[0, 0], S[1, 0], S[0, 1], S[1, 1] = col("S11"), col("S21"), col("S12"), col("S22")
    return S


def beta_yee(freqs_hz, dt_s: float, dx_m: float) -> np.ndarray:
    """Yee-discrete β of the port cross-section (``_compute_beta(dt, dx)``)."""
    import jax.numpy as jnp
    b = _compute_beta(jnp.asarray(np.asarray(freqs_hz, dtype=float)), FC_TE10_HZ,
                      dt=float(dt_s), dx=float(dx_m))
    return np.real(np.asarray(b, dtype=complex))


def beta_continuous(freqs_hz, fc_hz: float = FC_TE10_HZ) -> np.ndarray:
    f = np.asarray(freqs_hz, dtype=float)
    k = 2.0 * np.pi * f / C0_LOCAL
    kc = 2.0 * np.pi * float(fc_hz) / C0_LOCAL
    return np.sqrt(np.maximum(k * k - kc * kc, 0.0))


def beta_yee_fc(freqs_hz, fc_hz: float, dt_s: float, dx_m: float) -> np.ndarray:
    """Yee-discrete β for an arbitrary cutoff (the extractor's own β when
    ``fc_hz`` is the port config's ``f_cutoff``)."""
    import jax.numpy as jnp
    b = _compute_beta(jnp.asarray(np.asarray(freqs_hz, dtype=float)), float(fc_hz),
                      dt=float(dt_s), dx=float(dx_m))
    return np.real(np.asarray(b, dtype=complex))


def discrete_guide_cutoff_hz(dx_m: float, a_m: float = F.A_M) -> float:
    """TE10 cutoff of the Yee-discretized guide of width ``a`` (walls on nodes):
    ``kc = (2/dx)·sin(π·dx/(2a))`` — the mode the FDTD run actually propagates."""
    kc = (2.0 / dx_m) * math.sin(math.pi * dx_m / (2.0 * a_m))
    return kc * C0_LOCAL / (2.0 * math.pi)


def fit_guide_cutoff(S21: np.ndarray, freqs_hz, dt_s: float, dx_m: float, length_m: float,
                     candidates_hz=None) -> dict:
    """Guide cutoff fitted from a thru's S21 phase between two planes
    ``length_m`` apart: ``unwrap(∠S21) = −β_yee(f; fc)·L + const``. Returns the
    best ``fc`` on a 1 MHz grid with its rms residual, plus the residuals at
    the analytic ``c/2a``, the discrete-guide and the port-config cutoffs
    when given. Pure NumPy; a mechanism witness, not a gate."""
    f = np.asarray(freqs_hz, dtype=float)
    ph = np.unwrap(np.angle(np.asarray(S21, dtype=complex)))

    def rms_deg(fc):
        model = -beta_yee_fc(f, fc, dt_s, dx_m) * length_m
        k = np.mean(ph - model)
        return float(np.degrees(np.sqrt(np.mean((ph - model - k) ** 2)))), float(np.degrees(k))

    if candidates_hz is None:
        candidates_hz = np.arange(5.0e9, 6.7e9, 1.0e6)
    best = min(((rms_deg(fc)[0], fc) for fc in candidates_hz), key=lambda x: x[0])
    out = {"fc_fit_hz": float(best[1]), "rms_deg_at_fit": best[0],
           "const_deg_at_fit": rms_deg(best[1])[1], "length_m": float(length_m),
           "fc_c_over_2a_hz": FC_TE10_HZ, "rms_deg_at_c_over_2a": rms_deg(FC_TE10_HZ)[0],
           "fc_discrete_guide_hz": discrete_guide_cutoff_hz(dx_m),
           "rms_deg_at_discrete_guide": rms_deg(discrete_guide_cutoff_hz(dx_m))[0]}
    return out


# --- per-cell metrics (§4, §6) ----------------------------------------------

def cell_metrics(S: np.ndarray) -> dict:
    S = np.asarray(S, dtype=complex)
    s11, s21, s12, s22 = S[0, 0], S[1, 0], S[0, 1], S[1, 1]
    col_power = np.stack([np.abs(s11) ** 2 + np.abs(s21) ** 2,
                          np.abs(s22) ** 2 + np.abs(s12) ** 2])
    m21, m12 = np.abs(s21), np.abs(s12)
    denom = np.maximum(np.maximum(m21, m12), 1e-12)
    max_abs_per_bin = np.max(np.abs(S), axis=(0, 1))
    return {
        "column_power_max": float(col_power.max()),
        "column_power_per_bin": [[float(v) for v in col_power[0]], [float(v) for v in col_power[1]]],
        "reciprocity_mag_mean": float(np.mean(np.abs(m21 - m12) / denom)),
        "reciprocity_complex_max": float(np.max(np.abs(s21 - s12) / np.maximum(max_abs_per_bin, 1e-12))),
        "reciprocity_complex_per_bin": [float(v) for v in np.abs(s21 - s12) / np.maximum(max_abs_per_bin, 1e-12)],
        "power_closure_max": float(np.max(np.abs(1.0 - col_power))),
        "non_vacuity_max_s11": float(np.abs(s11).max()),
    }


# --- oracles (§5(c), §5(d)) -------------------------------------------------

def airy_reference(freqs_hz) -> tuple[np.ndarray, np.ndarray]:
    """Airy slab moved to the DEFAULT reference planes (d_L = d_R = 35.56 mm),
    continuous vacuum β — the form of ``build_waveguide_band_broad_e5_phase_envelope.py``."""
    f = np.asarray(freqs_hz, dtype=float)
    s11_e, s21_e = airy_slab(f, F.SLAB_EPS_R, SLAB_THICKNESS_M, FC_TE10_HZ)
    b = beta_continuous(f)
    d_l = D_PLANE_TO_SLAB_FACE_M
    d_r = F.REF_RIGHT_DEFAULT_M - F.SLAB_X_M[1]
    return s11_e * np.exp(-2j * b * d_l), s21_e * np.exp(-1j * b * (d_l + d_r))


def pec_short_phase_oracle_deg(beta: np.ndarray) -> np.ndarray:
    """∠S11 at a default plane 38.10 mm from the short's face: π − 2βd."""
    return wrap_deg(np.degrees(np.pi - 2.0 * np.asarray(beta) * D_PLANE_TO_PEC_FACE_M))


# --- referee (§5(d)) -------------------------------------------------------

def referee_pec_short(S: np.ndarray, freqs_hz) -> dict:
    S = np.asarray(S, dtype=complex)
    m = np.abs(S[0, 0])
    m22 = np.abs(S[1, 1])
    f = np.asarray(freqs_hz, dtype=float)
    return {
        "min_s11": float(m.min()), "max_s11": float(m.max()), "mean_s11": float(m.mean()),
        "bins_above_1_03": [float(x) for x in f[m >= PEC_SHORT_S11_MAX]],
        "bins_below_0_99": [float(x) for x in f[m < PEC_SHORT_S11_MIN]],
        "s22_min": float(m22.min()), "s22_max": float(m22.max()), "s22_mean": float(m22.mean()),
        "gate_min": PEC_SHORT_S11_MIN, "gate_max": PEC_SHORT_S11_MAX, "gate_mean_tol": PEC_SHORT_MEAN_TOL,
    }


def referee_pec_short_pass(r: dict) -> bool:
    return (r["min_s11"] >= PEC_SHORT_S11_MIN and r["max_s11"] < PEC_SHORT_S11_MAX
            and abs(r["mean_s11"] - 1.0) < PEC_SHORT_MEAN_TOL)


def referee_slab_airy(S: np.ndarray, freqs_hz) -> dict:
    S = np.asarray(S, dtype=complex)
    f = np.asarray(freqs_hz, dtype=float)
    s11_ref, s21_ref = airy_reference(f)
    d11 = np.abs(np.abs(S[0, 0]) - np.abs(s11_ref))
    d21 = np.abs(np.abs(S[1, 0]) - np.abs(s21_ref))
    p11 = np.degrees(np.abs(np.angle(S[0, 0] * np.conj(s11_ref))))
    p21 = np.degrees(np.abs(np.angle(S[1, 0] * np.conj(s21_ref))))
    mask11 = np.abs(s11_ref) >= SLAB_AIRY_PHASE_MAG_FLOOR
    mask21 = np.abs(s21_ref) >= SLAB_AIRY_PHASE_MAG_FLOOR
    mag_worst = np.maximum(d11, d21)
    phase_masked = np.concatenate([p11[mask11], p21[mask21]])
    k_mag = int(np.argmax(mag_worst))
    return {
        "max_mag_abs_diff": float(mag_worst.max()),
        "s11_max_mag_abs_diff": float(d11.max()), "s21_max_mag_abs_diff": float(d21.max()),
        "worst_bin_hz": float(f[k_mag]),
        "max_phase_diff_deg": float(phase_masked.max()) if phase_masked.size else float("nan"),
        "max_phase_diff_deg_unmasked": float(max(p11.max(), p21.max())),
        "s11_phase_diff_deg_per_bin": [float(v) for v in p11],
        "s21_phase_diff_deg_per_bin": [float(v) for v in p21],
        "s11_mag_abs_diff_per_bin": [float(v) for v in d11],
        "s21_mag_abs_diff_per_bin": [float(v) for v in d21],
        "phase_mask_floor": SLAB_AIRY_PHASE_MAG_FLOOR,
        "phase_bins_masked_s11": [float(x) for x in f[~mask11]],
        "phase_bins_masked_s21": [float(x) for x in f[~mask21]],
        "oracle_shift_convention": "exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))",
        "d_left_m": D_PLANE_TO_SLAB_FACE_M,
        "d_right_m": float(F.REF_RIGHT_DEFAULT_M - F.SLAB_X_M[1]),
        "oracle_s11": [[float(z.real), float(z.imag)] for z in s11_ref],
        "oracle_s21": [[float(z.real), float(z.imag)] for z in s21_ref],
        "gate_mag": SLAB_AIRY_MAG_TOL, "gate_phase_deg": SLAB_AIRY_PHASE_TOL_DEG,
    }


def referee_slab_mag_pass(r: dict) -> bool:
    return r["max_mag_abs_diff"] <= SLAB_AIRY_MAG_TOL


def referee_slab_phase_pass(r: dict) -> bool:
    return r["max_phase_diff_deg"] <= SLAB_AIRY_PHASE_TOL_DEG


# --- plane shift (§5(b)) ----------------------------------------------------

def rotation_predictions_deg(beta: np.ndarray) -> dict[str, np.ndarray]:
    b = np.asarray(beta)
    return {
        "S11": wrap_deg(np.degrees(2.0 * b * SHIFT_LEFT_M)),
        "S22": wrap_deg(np.degrees(2.0 * b * abs(SHIFT_RIGHT_M))),
        "S21": wrap_deg(np.degrees(b * (SHIFT_LEFT_M + abs(SHIFT_RIGHT_M)))),
        "S12": wrap_deg(np.degrees(b * (SHIFT_LEFT_M + abs(SHIFT_RIGHT_M)))),
    }


_ENTRY = {"S11": (0, 0), "S22": (1, 1), "S21": (1, 0), "S12": (0, 1)}


def plane_shift_rotation(S_base: np.ndarray, S_shift: np.ndarray, freqs_hz,
                         dt_s: float, dx_m: float, fc_port_hz: float | None = None) -> dict:
    """Rotation of every entry under the shift, against the pre-declared
    prediction (β of the guide's TE10 cutoff c/2a: Yee-discrete and
    continuous) and, when ``fc_port_hz`` is given, against the extractor's
    OWN β (the port config's cutoff) as a mechanism witness."""
    S_base = np.asarray(S_base, dtype=complex)
    S_shift = np.asarray(S_shift, dtype=complex)
    f = np.asarray(freqs_hz, dtype=float)
    pred_yee = rotation_predictions_deg(beta_yee(f, dt_s, dx_m))
    pred_cont = rotation_predictions_deg(beta_continuous(f))
    pred_port = (rotation_predictions_deg(beta_yee_fc(f, fc_port_hz, dt_s, dx_m))
                 if fc_port_hz is not None else None)
    out = {
        "abs_s_max_diff": float(np.max(np.abs(np.abs(S_shift) - np.abs(S_base)))),
        "abs_s_allclose": bool(np.allclose(np.abs(S_base), np.abs(S_shift),
                                           rtol=ABS_S_INVARIANCE_RTOL, atol=ABS_S_INVARIANCE_ATOL)),
        "rotation_deg": {},
        "fc_port_hz": fc_port_hz,
        "fc_predeclared_hz": FC_TE10_HZ,
    }
    for name, (i, j) in _ENTRY.items():
        mag = np.abs(S_base[i, j])
        peak = float(mag.max())
        # An entry whose band peak is below the extractor's measured |S| noise
        # floor (the PEC-short's transmission) carries no measurable phase;
        # within a measurable entry the phase gate's own weak-signal mask applies.
        measurable = peak > ROTATION_NOISE_FLOOR
        mask = (mag >= ROTATION_MAG_MASK_FRAC * peak) if measurable else np.zeros_like(mag, dtype=bool)
        meas = wrap_deg(np.degrees(np.angle(S_shift[i, j] * np.conj(S_base[i, j]))))
        r_yee = np.abs(wrap_deg(meas - pred_yee[name]))
        r_cont = np.abs(wrap_deg(meas - pred_cont[name]))
        wrong = np.abs(wrap_deg(meas + pred_yee[name]))
        item = {
            "predicted_yee": [float(v) for v in pred_yee[name]],
            "predicted_continuous": [float(v) for v in pred_cont[name]],
            "measured": [float(v) for v in meas],
            "abs_s_base_peak": peak,
            "measurable": bool(measurable),
            "mask_frac": ROTATION_MAG_MASK_FRAC,
            "masked_bins_hz": [float(x) for x in f[~mask]] if measurable else [float(x) for x in f],
            "resid_yee_max": float(r_yee[mask].max()) if mask.any() else None,
            "resid_cont_max": float(r_cont[mask].max()) if mask.any() else None,
            "wrong_sign_resid_min": float(wrong[mask].min()) if mask.any() else None,
        }
        if pred_port is not None:
            item["predicted_port_beta"] = [float(v) for v in pred_port[name]]
            r_port = np.abs(wrap_deg(meas - pred_port[name]))
            item["resid_port_beta_max"] = float(r_port[mask].max()) if mask.any() else None
        out["rotation_deg"][name] = item
    live = [v for v in out["rotation_deg"].values() if v["resid_yee_max"] is not None]
    out["entries_measurable"] = [k for k, v in out["rotation_deg"].items() if v["measurable"]]
    out["resid_yee_max"] = max(v["resid_yee_max"] for v in live)
    out["resid_cont_max"] = max(v["resid_cont_max"] for v in live)
    out["wrong_sign_resid_min"] = min(v["wrong_sign_resid_min"] for v in live)
    if pred_port is not None:
        out["resid_port_beta_max"] = max(v["resid_port_beta_max"] for v in live)
    return out


def gradient_invariance_entry(kind: str, g_base_re, g_base_im, g_shift_re, g_shift_im,
                              phi_measured_rad: float | None,
                              phi_predeclared_rad: float | None = None) -> dict:
    """Magnitude objectives: invariant. Complex objectives: rotation-covariant,
    ``e^{jφ}·dS/dθ|base`` vs ``dS/dθ|shifted``.

    ``rel_change`` (the tested quantity) uses ``φ = ∠(S_shift/S_base)`` at the
    band-centre bin, i.e. the unit-modulus factor the extractor actually
    applied — that isolates the gradient property (a β on the tape, or a
    non-unit-modulus factor) from the value of β itself, which the rotation
    gate judges separately. ``rel_change_predeclared_phi`` repeats it with the
    pre-declared ``φ = 2β_yee(c/2a)·Δ`` for the record."""
    if kind == "magnitude":
        base, shift = float(g_base_re), float(g_shift_re)
        rel = abs(shift - base) / max(abs(base), 1e-300)
        return {"kind": kind, "value_base": base, "value_shifted": shift,
                "rel_change": float(rel), "report_bar": GRADIENT_REPORT_BAR, "pinned_gate": None}
    base = complex(float(g_base_re), float(g_base_im))
    shift = complex(float(g_shift_re), float(g_shift_im))
    rotated = base * np.exp(1j * float(phi_measured_rad))
    rel = abs(shift - rotated) / max(abs(base), 1e-300)
    out = {"kind": kind, "value_base": [base.real, base.imag],
           "value_shifted": [shift.real, shift.imag],
           "rotated_base": [float(rotated.real), float(rotated.imag)],
           "phi_measured_deg": float(np.degrees(phi_measured_rad)),
           "rel_change": float(rel), "report_bar": GRADIENT_REPORT_BAR, "pinned_gate": None}
    if phi_predeclared_rad is not None:
        rot2 = base * np.exp(1j * float(phi_predeclared_rad))
        out["phi_predeclared_deg"] = float(np.degrees(phi_predeclared_rad))
        out["rel_change_predeclared_phi"] = float(abs(shift - rot2) / max(abs(base), 1e-300))
    return out


# --- AD vs FD (§5(a)) -------------------------------------------------------

def ad_fd_entry(*, g_ad: float, f_plus: float, f_minus: float, h: float, loss_dtype) -> dict:
    span = float(_fd_ulp_span(float(f_plus), float(f_minus), loss_dtype))
    g_fd = (float(f_plus) - float(f_minus)) / (2.0 * float(h))
    rel = abs(float(g_ad) - g_fd) / max(abs(g_fd), 1e-12)
    if span < FD_ULP_FLOOR:
        verdict = "skipped_under_ulp_floor"
    else:
        verdict = "pass" if rel <= AD_FD_REL_GATE else "fail"
    return {"f_plus": float(f_plus), "f_minus": float(f_minus), "fd_ulp_span": span,
            "ulp_floor": FD_ULP_FLOOR, "g_ad": float(g_ad), "g_fd": g_fd, "rel": float(rel),
            "gate": AD_FD_REL_GATE, "verdict": verdict, "loss_dtype": str(np.dtype(loss_dtype))}


def forward_identity_pass(max_abs_diff_scaled: float) -> bool:
    """``max |S_traced − S_untraced| / (rtol·|S_untraced| + atol) ≤ 1``."""
    return max_abs_diff_scaled <= 1.0


def forward_identity_metric(S_traced, S_untraced) -> dict:
    a, b = np.asarray(S_traced, dtype=complex), np.asarray(S_untraced, dtype=complex)
    d = np.abs(a - b)
    scaled = d / (FORWARD_IDENTITY_RTOL * np.abs(b) + FORWARD_IDENTITY_ATOL)
    k = np.unravel_index(int(np.argmax(scaled)), scaled.shape)
    return {"max_abs_diff": float(d.max()), "max_scaled_diff": float(scaled.max()),
            "worst_entry": [int(k[0]), int(k[1]), int(k[2])], "abs_s_at_worst": float(np.abs(b[k])),
            "rtol": FORWARD_IDENTITY_RTOL, "atol": FORWARD_IDENTITY_ATOL,
            "pass": bool(scaled.max() <= 1.0)}


# --- dx ladder (§5(c)) ------------------------------------------------------

def _observable(S: np.ndarray, entry, kind: str) -> np.ndarray:
    z = np.asarray(S, dtype=complex)[entry[0], entry[1]]
    return np.abs(z) if kind == "mag" else np.degrees(np.angle(z))


def _delta(a: np.ndarray, b: np.ndarray, kind: str) -> np.ndarray:
    return np.abs(a - b) if kind == "mag" else np.abs(wrap_deg(a - b))


def _richardson(fine: np.ndarray, coarse: np.ndarray, kind: str) -> np.ndarray:
    if kind == "mag":
        return 2.0 * fine - coarse
    return wrap_deg(fine + wrap_deg(fine - coarse))


def ladder_eval(S_by_rung: dict[str, np.ndarray], entry, kind: str, freqs_hz,
                oracle_by_pair: dict[str, np.ndarray] | None,
                oracle_continuous_by_pair: dict[str, np.ndarray] | None = None) -> dict:
    """One observable across the three rungs; every bin evaluated, worst reported.

    ``oracle_by_pair`` maps ``"coarse-mid"`` / ``"mid-fine"`` to the per-bin
    oracle the Richardson estimate of that pair is compared with (None for the
    PEC-short magnitude, which is excluded, §5(c)).
    """
    f = np.asarray(freqs_hz, dtype=float)
    floor = LADDER_FLOOR_MAG if kind == "mag" else LADDER_FLOOR_PHASE_DEG
    v = {r: _observable(S_by_rung[r], entry, kind) for r in RUNG_LABELS}
    coarse_delta = _delta(v["coarse"], v["mid"], kind)
    fine_delta = _delta(v["mid"], v["fine"], kind)
    excess = fine_delta - coarse_delta
    k_worst = int(np.argmax(excess))
    gate_pass = bool(np.all(fine_delta <= coarse_delta + floor))
    # Witness (i): monotonicity and successive-delta ratio.
    d1 = (v["mid"] - v["coarse"]) if kind == "mag" else wrap_deg(v["mid"] - v["coarse"])
    d2 = (v["fine"] - v["mid"]) if kind == "mag" else wrap_deg(v["fine"] - v["mid"])
    monotone = (np.sign(d1) == np.sign(d2)) | (np.abs(d2) <= floor)
    conditioned = coarse_delta >= floor
    ratio = np.where(conditioned, fine_delta / np.maximum(coarse_delta, 1e-300), np.nan)
    lo, hi = LADDER_RATIO_WINDOW
    if conditioned.any():
        centre = math.sqrt(lo * hi)
        dist = np.abs(np.log(np.maximum(ratio, 1e-300) / centre))
        dist = np.where(conditioned, dist, -np.inf)
        k_ratio = int(np.argmax(dist))
        ratio_worst = float(ratio[k_ratio])
        interpretable = bool(np.all((ratio[conditioned] >= lo) & (ratio[conditioned] <= hi)))
    else:
        k_ratio = None
        ratio_worst = None
        interpretable = True     # every bin already inside the floor: nothing to interpret
    out = {
        "kind": kind, "floor": floor,
        "values_by_rung": {r: [float(x) for x in v[r]] for r in RUNG_LABELS},
        "coarse_delta_per_bin": [float(x) for x in coarse_delta],
        "fine_delta_per_bin": [float(x) for x in fine_delta],
        "coarse_delta_worst": float(coarse_delta.max()),
        "fine_delta_worst": float(fine_delta.max()),
        "excess_worst": float(excess[k_worst]),
        "worst_bin_hz": float(f[k_worst]),
        "gate_pass": gate_pass,
        "monotone_fraction_of_bins": float(monotone.mean()),
        "successive_ratio_per_bin": [None if np.isnan(x) else float(x) for x in ratio],
        "successive_ratio_worst": ratio_worst,
        "successive_ratio_worst_bin_hz": None if k_ratio is None else float(f[k_ratio]),
        "n_conditioned_bins": int(conditioned.sum()),
        "ratio_window": [lo, hi],
        "interpretable": interpretable,
    }
    if oracle_by_pair is not None:
        rich = {}
        for pair, (ra, rb) in (("coarse-mid", ("coarse", "mid")), ("mid-fine", ("mid", "fine"))):
            est = _richardson(v[rb], v[ra], kind)
            orc = np.asarray(oracle_by_pair[pair], dtype=float)
            diff = _delta(est, orc, kind)
            item = {"pair": [RUNG_DX[ra], RUNG_DX[rb]],
                    "estimate_per_bin": [float(x) for x in est],
                    "oracle_per_bin": [float(x) for x in orc],
                    "abs_diff_per_bin": [float(x) for x in diff],
                    "max_abs_diff": float(diff.max()),
                    "max_abs_diff_bin_hz": float(f[int(np.argmax(diff))]),
                    "finer_rung_abs_diff_max": float(_delta(v[rb], orc, kind).max())}
            if oracle_continuous_by_pair is not None:
                orc_c = np.asarray(oracle_continuous_by_pair[pair], dtype=float)
                item["oracle_continuous_per_bin"] = [float(x) for x in orc_c]
                item["max_abs_diff_continuous"] = float(_delta(est, orc_c, kind).max())
            rich[pair] = item
        out["richardson"] = rich
        out["richardson_max_abs_diff"] = max(r["max_abs_diff"] for r in rich.values())
    if not interpretable:
        out["verdict"] = "not_interpretable"
    else:
        out["verdict"] = "pass" if gate_pass else "fail"
    return out


# --- verdicts ---------------------------------------------------------------

def _cell_key(c: dict) -> str:
    return f"{c['dut']}|{rung_label(c['dx_m'])}|{c['lane']}"


def rung_label(dx_m: float) -> str:
    for r, dx in RUNG_DX.items():
        if abs(dx - float(dx_m)) < 1e-12:
            return r
    raise ValueError(dx_m)


def cell_settling_effective(c: dict) -> dict[str, float]:
    """The settling numbers that are claims-bearing for a cell: the doubled
    record where one exists (§2.5), the 40-period record otherwise."""
    if c.get("settling_rerun"):
        return {k: float(v) for k, v in c["settling_rerun"]["settling_db"].items()}
    return {k: float(v) for k, v in c["settling_db"].items()}


def recompute_verdicts(fx: dict) -> dict:
    """Every gate of the pre-declaration, recomputed from the stored numbers.

    Returns ``{gate_key: verdict}`` with verdict in
    ``{"pass", "fail", "report_only", "skipped", "not_interpretable"}``. The
    driver stores this dict under ``verdicts``; the replay test recomputes it
    with this same function and compares.
    """
    v: dict[str, str] = {}
    cells = {_cell_key(c): c for c in fx["cells"]}

    # settling (§2.5) per cell/drive, on the claims-bearing record
    for key, c in cells.items():
        eff = cell_settling_effective(c)
        v[f"settling|{key}"] = "pass" if all(x <= SETTLING_DB_MAX for x in eff.values()) else "fail"

    # non-vacuity (§4): both reflecting DUTs, every rung and lane
    for key, c in cells.items():
        if c["dut"] == "thru":
            continue
        v[f"non_vacuity|{key}"] = ("pass" if c["non_vacuity_max_s11"] > NON_VACUITY_MIN_MAX_S11
                                   else "fail")

    # physics gates (§6): gated at the claims rung, reported elsewhere
    for key, c in cells.items():
        if c["dut"] == "thru":
            continue
        gated = rung_label(c["dx_m"]) == CLAIMS_RUNG
        cp = c["column_power_max"] < COLUMN_POWER_MAX
        rm = c["reciprocity_mag_mean"] < RECIPROCITY_MAG_MAX
        rc = c["reciprocity_complex_max"] <= RECIPROCITY_COMPLEX_MAX
        v[f"column_power|{key}"] = ("pass" if cp else "fail") if gated else "report_only"
        v[f"reciprocity_mag|{key}"] = ("pass" if rm else "fail") if gated else "report_only"
        v[f"reciprocity_complex|{key}"] = ("pass" if rc else "fail") if gated else "report_only"
        v[f"power_closure|{key}"] = "report_only"

    # referee (§5(d)) at the claims rung; other rungs reported
    for key, c in cells.items():
        if c["dut"] == "thru":
            continue
        gated = rung_label(c["dx_m"]) == CLAIMS_RUNG
        S = s_from_json(c["s_params"])
        if c["dut"] == "pec_short":
            ok = referee_pec_short_pass(referee_pec_short(S, fx["fixture"]["freqs_hz"]))
            v[f"referee_pec_short|{key}"] = ("pass" if ok else "fail") if gated else "report_only"
        else:
            r = referee_slab_airy(S, fx["fixture"]["freqs_hz"])
            v[f"referee_slab_airy_mag|{key}"] = (
                ("pass" if referee_slab_mag_pass(r) else "fail") if gated else "report_only")
            v[f"referee_slab_airy_phase|{key}"] = (
                ("pass" if referee_slab_phase_pass(r) else "fail") if gated else "report_only")

    # AD vs FD (§5(a)) and the forward identity (criterion 1)
    for leg in fx["ad_vs_fd"]:
        key = f"{leg['dut']}|{leg['lane']}|{leg['theta_kind']}|{leg['objective']}"
        e = ad_fd_entry(g_ad=leg["g_ad"], f_plus=leg["f_plus"], f_minus=leg["f_minus"],
                        h=leg["h"], loss_dtype=np.dtype(leg["loss_dtype"]))
        v[f"ad_vs_fd|{key}"] = e["verdict"]
        v[f"forward_identity|{key}"] = (
            "pass" if forward_identity_pass(leg["forward_identity"]["max_scaled_diff"]) else "fail")

    # plane shift (§5(b))
    for key, p in fx["plane_shift"].items():
        if key == "cheap_refute":
            continue
        v[f"plane_shift_abs_s|{key}"] = "pass" if p["abs_s_allclose"] else "fail"
        v[f"plane_shift_rotation_yee|{key}"] = (
            "pass" if p["resid_yee_max"] <= ROTATION_TOL_YEE_DEG else "fail")
        v[f"plane_shift_rotation_continuous|{key}"] = (
            "pass" if p["resid_cont_max"] <= ROTATION_TOL_CONTINUOUS_DEG else "fail")
        v[f"plane_shift_wrong_sign|{key}"] = (
            "pass" if p["wrong_sign_resid_min"] > WRONG_SIGN_MIN_DEG else "fail")
        for obj, g in p["gradient_invariance"].items():
            gk = f"gradient_invariance|{key}|{obj}"
            if g.get("skipped_under_ulp_floor"):
                v[gk] = "skipped"
            elif g.get("pinned_gate") is None:
                v[gk] = "report_only"
            else:
                v[gk] = "pass" if g["rel_change"] <= g["pinned_gate"] else "fail"
    refute = fx["plane_shift"].get("cheap_refute")
    if refute is not None:
        # the gate must go RED under the flipped sign: every entry's residual > 10°
        v["cheap_refute_flip_shift_sign"] = (
            "pass" if refute["resid_yee_min_over_entries"] > WRONG_SIGN_MIN_DEG
            and not refute["rotation_gate_would_pass"] else "fail")

    # ladder (§5(c))
    for key, lad in fx["ladder"].items():
        v[f"ladder|{key}"] = lad["verdict"]
        pin = lad.get("pinned_richardson_gate")
        if "richardson" in lad:
            if pin is None:
                v[f"ladder_richardson|{key}"] = "report_only"
            else:
                v[f"ladder_richardson|{key}"] = (
                    "pass" if lad["richardson_max_abs_diff"] <= pin else "fail")
        pin_m = lad.get("pinned_monotone_fraction_min")
        if pin_m is None:
            v[f"ladder_monotone|{key}"] = "report_only"
        else:
            v[f"ladder_monotone|{key}"] = (
                "pass" if lad["monotone_fraction_of_bins"] >= pin_m else "fail")
    return v


def pin_from_envelope(values: list[float], quantum: int = GRADIENT_PIN_QUANTUM) -> float:
    """``gate_from_envelope(max(values), quantum)`` — the pin step of §5(b)/(c)."""
    return float(gate_from_envelope(max(float(x) for x in values), quantum=quantum))


def pin_lower_from_envelope(measured: float, *, quantum: int) -> float:
    """Lower-bound counterpart of ``gate_from_envelope`` for a quantity that must
    stay ABOVE its measured value (the monotone fraction of §5(c) witness (i)):
    ``floor(measured / ENVELOPE_GATE_MULTIPLIER · quantum) / quantum`` — the same
    shared multiplier, applied downward."""
    from tests import _gate_policy
    return math.floor(float(measured) / _gate_policy.ENVELOPE_GATE_MULTIPLIER * quantum) / quantum


RICHARDSON_PIN_QUANTUM = {"mag": 100, "phase": 10}   # cv18 precedent (0.0051 -> 0.01); 0.1 deg for phases
MONOTONE_PIN_QUANTUM = 100


def pin_fixture(fx: dict) -> dict:
    """The pin step (a separate commit, §5(b)/(c)): fill every ``pinned_*`` field
    from the measured report-first quantities and recompute the verdicts.

    * gradient invariance: ONE envelope over every resolvable, finite
      ``rel_change`` of every (dut, lane, objective) →
      ``gate_from_envelope(max, quantum=1000)`` written into each entry;
    * Richardson: per (observable, lane) ``gate_from_envelope(richardson_max_abs_diff,
      quantum=100 | 10)``;
    * monotone fraction: per (observable, lane) the lower bound
      ``pin_lower_from_envelope(measured, quantum=100)``.
    Nothing measured is touched; only the ``pinned_*`` fields and ``verdicts``.
    """
    rels = []
    for key, p in fx["plane_shift"].items():
        if key == "cheap_refute":
            continue
        for gi in p["gradient_invariance"].values():
            if gi.get("skipped_under_ulp_floor"):
                continue
            if np.isfinite(gi["rel_change"]):
                rels.append(gi["rel_change"])
    grad_pin = pin_from_envelope(rels, quantum=GRADIENT_PIN_QUANTUM) if rels else None
    for key, p in fx["plane_shift"].items():
        if key == "cheap_refute":
            continue
        for gi in p["gradient_invariance"].values():
            if gi.get("skipped_under_ulp_floor"):
                continue
            gi["pinned_gate"] = grad_pin if np.isfinite(gi["rel_change"]) else None
            gi["pinned_gate_envelope"] = max(rels) if rels else None
    for key, lad in fx["ladder"].items():
        if "richardson" in lad:
            lad["pinned_richardson_gate"] = float(gate_from_envelope(
                lad["richardson_max_abs_diff"], quantum=RICHARDSON_PIN_QUANTUM[lad["kind"]]))
        lad["pinned_monotone_fraction_min"] = pin_lower_from_envelope(
            lad["monotone_fraction_of_bins"], quantum=MONOTONE_PIN_QUANTUM)
    fx["pins"] = {"gradient_invariance_envelope": max(rels) if rels else None,
                  "gradient_invariance_gate": grad_pin, "gradient_quantum": GRADIENT_PIN_QUANTUM,
                  "richardson_quantum": RICHARDSON_PIN_QUANTUM, "monotone_quantum": MONOTONE_PIN_QUANTUM,
                  "policy": "tests/_gate_policy.py gate_from_envelope (x ENVELOPE_GATE_MULTIPLIER, rounded up); "
                            "lower bounds rounded down by the same multiplier"}
    fx["verdicts"] = recompute_verdicts(fx)
    return fx

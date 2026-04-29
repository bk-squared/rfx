#!/usr/bin/env python3
"""P-1 Spike v2 for strict waveguide-mode extractor closure.

This is a falsification harness, not production code.  It compares candidate
modal architectures against the strict plan in
``.omx/plans/waveguide-spec-strict-closure-initial-plan.md`` before any more
production churn:

* UNIFORM_FULL_YEE    — full Yee EVP under full aperture measure (old baseline)
* FULL_YEE_DROP_GS   — current compromise: full Yee cutoff/source matching,
                       dropped runtime aperture, aperture-space GS
* REDUCED_GEVP_DROP  — literal Section 6 reduced generalized EVP candidate
* BIORTHO_FULL_YEE   — full Yee modes with Gram-matrix/biorthogonal projection
                       in the algebraic spike only

Default/``--quick`` runs deterministic algebraic diagnostics.  ``--run-benchmarks``
adds expensive time-domain hooks for candidates that can be expressed by
monkeypatching the current production builders.  Strict completion is never
claimed by this script; it only prints GO/NO-GO evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sys
import time
import traceback
import warnings
from collections import namedtuple
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import jax.numpy as jnp
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module="matplotlib\\.projections",
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rfx.sources.waveguide_port as wg_mod  # noqa: E402
from rfx.sources.waveguide_port import WaveguidePort, WaveguidePortConfig  # noqa: E402


State = namedtuple("State", "ex ey ez hx hy hz")
Profile = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
CandidateKind = Literal[
    "uniform_full_yee",
    "full_yee_drop_gs",
    "reduced_gevp_drop",
    "biortho_full_yee",
]


@dataclass(frozen=True)
class CandidateSpec:
    """One spike candidate and the evidence it is allowed to claim."""

    name: str
    kind: CandidateKind
    description: str
    physical_supported: bool


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "UNIFORM_FULL_YEE",
        "uniform_full_yee",
        "full Yee EVP/cutoff under full aperture measure (old uniform baseline)",
        True,
    ),
    CandidateSpec(
        "FULL_YEE_DROP_GS",
        "full_yee_drop_gs",
        "full Yee EVP/cutoff with dropped aperture normalization and GS",
        True,
    ),
    CandidateSpec(
        "REDUCED_GEVP_DROP",
        "reduced_gevp_drop",
        "literal Section 6 reduced generalized EVP under dropped aperture",
        True,
    ),
    CandidateSpec(
        "BIORTHO_FULL_YEE",
        "biortho_full_yee",
        "full Yee modes plus recorder-matched dot-Gram inverse projection",
        True,
    ),
)


class SpikeUnavailable(RuntimeError):
    """Raised when an optional benchmark dependency/hook is unavailable."""


_ORIG_APERTURE_DA = wg_mod._aperture_dA
_ORIG_DISCRETE_TE_MODE_PROFILES = wg_mod._discrete_te_mode_profiles
_ORIG_DISCRETE_TM_MODE_PROFILES = wg_mod._discrete_tm_mode_profiles
_ORIG_GRAM_SCHMIDT_MODES = wg_mod._gram_schmidt_modes
_ORIG_INIT_WAVEGUIDE_PORT = wg_mod.init_waveguide_port


def _full_dA_np(u_widths: np.ndarray, v_widths: np.ndarray) -> np.ndarray:
    return np.asarray(u_widths, dtype=np.float64)[:, None] * np.asarray(
        v_widths, dtype=np.float64
    )[None, :]


def _partial_dA_np(u_widths: np.ndarray, v_widths: np.ndarray) -> np.ndarray:
    """Dropped +face aperture dA used by the current strict-closure plan."""

    u_w = np.asarray(u_widths, dtype=np.float64)
    v_w = np.asarray(v_widths, dtype=np.float64)
    u_weight = np.ones_like(u_w)
    v_weight = np.ones_like(v_w)
    if u_weight.size:
        u_weight[-1] = 0.0
    if v_weight.size:
        v_weight[-1] = 0.0
    return (u_w * u_weight)[:, None] * (v_w * v_weight)[None, :]


def _full_aperture_dA(cfg: Any) -> jnp.ndarray:
    dA = _full_dA_np(np.asarray(cfg.u_widths), np.asarray(cfg.v_widths))
    return jnp.asarray(dA, dtype=getattr(cfg.u_widths, "dtype", jnp.float32))


def _drop_both_aperture_dA(cfg: Any) -> jnp.ndarray:
    dA = _partial_dA_np(np.asarray(cfg.u_widths), np.asarray(cfg.v_widths))
    return jnp.asarray(dA, dtype=getattr(cfg.u_widths, "dtype", jnp.float32))


def _scipy_eigh(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.linalg import eigh  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise SpikeUnavailable(
            "REDUCED_GEVP_DROP requires scipy.linalg.eigh; import failed: "
            f"{exc}"
        ) from exc
    return eigh(a, b)


def _pick_weighted_overlap_mode(
    evecs_reduced: np.ndarray,
    analytic_reduced: np.ndarray,
    weights_reduced: np.ndarray,
    *,
    num_candidates: int = 60,
) -> int:
    """Pick a reduced-space mode by largest aperture-weighted overlap."""

    n_candidates = min(num_candidates, evecs_reduced.shape[1])
    w = np.asarray(weights_reduced, dtype=np.float64)
    ana = np.asarray(analytic_reduced, dtype=np.float64)
    ana_norm = np.sqrt(float(np.sum((ana * ana) * w)))
    if not np.isfinite(ana_norm) or ana_norm <= 1e-30:
        ana_norm = max(float(np.linalg.norm(ana)), 1e-30)
        weighted = ana / ana_norm
    else:
        weighted = ana * w / ana_norm
    overlaps = np.abs(evecs_reduced[:, :n_candidates].T @ weighted)
    return int(np.argmax(overlaps))


def _reduced_gevp_te_mode_profiles(
    a: float,
    b: float,
    m: int,
    n: int,
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    *,
    aperture_dA: np.ndarray | None = None,
    h_offset: tuple[float, float] = (0.0, 0.0),
    _orthogonalize: bool = True,
) -> Profile:
    """Literal Section-6-style reduced generalized EVP TE profile.

    This intentionally remains a spike candidate.  It may be physically wrong
    for source matching; the ledger exposes that via kc drift and physical gates.
    """

    _ = aperture_dA, _orthogonalize
    u_w = np.asarray(u_widths, dtype=np.float64)
    v_w = np.asarray(v_widths, dtype=np.float64)
    nu = len(u_w)
    nv = len(v_w)
    if nu == 0 or nv == 0:
        raise ValueError("cannot build TE mode profile for an empty aperture")

    d_uu = wg_mod._second_diff_1d(u_w, bc="neumann")
    d_vv = wg_mod._second_diff_1d(v_w, bc="neumann")
    lap = np.kron(d_uu, np.eye(nv)) + np.kron(np.eye(nu), d_vv)

    dA = _partial_dA_np(u_w, v_w)
    weights = dA.ravel()
    keep = weights > 0.0
    if int(np.count_nonzero(keep)) < 2:
        raise SpikeUnavailable(
            "reduced aperture has fewer than two non-zero cells: "
            f"shape=({nu}, {nv}), nonzero={int(np.count_nonzero(keep))}"
        )

    lap_reduced = lap[np.ix_(keep, keep)]
    positive = weights[keep]
    # Scale M to improve conditioning without changing eigenvectors.
    mass_diag = positive / float(np.median(positive))
    evals, evecs_reduced = _scipy_eigh(-lap_reduced, np.diag(mass_diag))

    u_c = np.cumsum(u_w) - 0.5 * u_w
    v_c = np.cumsum(v_w) - 0.5 * v_w
    hx_ana = (
        np.cos(m * np.pi * u_c / float(a))[:, None]
        * np.cos(n * np.pi * v_c / float(b))[None, :]
    )
    rank = _pick_weighted_overlap_mode(
        evecs_reduced,
        hx_ana.ravel()[keep],
        mass_diag,
    )
    kc2 = float(max(evals[rank], 0.0))

    hx_flat = np.zeros(nu * nv, dtype=np.float64)
    hx_flat[keep] = evecs_reduced[:, rank]
    hx = hx_flat.reshape(nu, nv)
    if float(np.sum(hx * hx_ana * dA)) < 0.0:
        hx = -hx

    d_hx_du = wg_mod._cell_centred_gradient(hx, u_w, axis=0, bc="neumann")
    d_hx_dv = wg_mod._cell_centred_gradient(hx, v_w, axis=1, bc="neumann")
    ey = d_hx_dv
    ez = -d_hx_du
    zero = ~keep.reshape(nu, nv)
    ey[zero] = 0.0
    ez[zero] = 0.0
    hy = -wg_mod._shift_profile_to_dual(ez, h_offset)
    hz = wg_mod._shift_profile_to_dual(ey, h_offset)
    hy[zero] = 0.0
    hz[zero] = 0.0

    power = float(np.sum((ey * ey + ez * ez) * dA))
    if not np.isfinite(power) or power <= 1e-30:
        raise SpikeUnavailable(
            "REDUCED_GEVP_DROP has zero/non-finite partial-dA power: "
            f"{power!r}"
        )
    norm = np.sqrt(power)
    ey /= norm
    ez /= norm
    hy /= norm
    hz /= norm
    hy, hz = wg_mod._scale_h_to_unit_cross(ey, ez, hy, hz, dA)
    return ey, ez, hy, hz, float(np.sqrt(kc2))


def _uniform_full_yee_te_mode_profiles(
    a: float,
    b: float,
    m: int,
    n: int,
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    *,
    aperture_dA: np.ndarray | None = None,
    h_offset: tuple[float, float] = (0.0, 0.0),
    _orthogonalize: bool = True,
) -> Profile:
    """Call production builder as if aperture_dA support did not exist."""

    _ = aperture_dA, _orthogonalize
    return _ORIG_DISCRETE_TE_MODE_PROFILES(
        a,
        b,
        m,
        n,
        u_widths,
        v_widths,
        aperture_dA=None,
        h_offset=h_offset,
        _orthogonalize=False,
    )


def _uniform_full_yee_tm_mode_profiles(
    a: float,
    b: float,
    m: int,
    n: int,
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    *,
    aperture_dA: np.ndarray | None = None,
    h_offset: tuple[float, float] = (0.0, 0.0),
) -> Profile:
    """Call production TM builder as if aperture_dA support did not exist."""

    _ = aperture_dA
    return _ORIG_DISCRETE_TM_MODE_PROFILES(
        a,
        b,
        m,
        n,
        u_widths,
        v_widths,
        aperture_dA=None,
        h_offset=h_offset,
    )


def _biortho_full_yee_te_mode_profiles(
    a: float,
    b: float,
    m: int,
    n: int,
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    *,
    aperture_dA: np.ndarray | None = None,
    h_offset: tuple[float, float] = (0.0, 0.0),
    _orthogonalize: bool = True,
) -> Profile:
    """Full-Yee TE profile normalized under the recorder aperture, no GS."""

    _ = _orthogonalize
    profile = _uniform_full_yee_te_mode_profiles(
        a, b, m, n, u_widths, v_widths, h_offset=h_offset
    )
    if aperture_dA is None:
        return profile
    return _normalize_profile_to_dA(profile, np.asarray(aperture_dA, dtype=np.float64))


def _biortho_full_yee_tm_mode_profiles(
    a: float,
    b: float,
    m: int,
    n: int,
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    *,
    aperture_dA: np.ndarray | None = None,
    h_offset: tuple[float, float] = (0.0, 0.0),
) -> Profile:
    """Full-Yee TM profile normalized under the recorder aperture, no GS."""

    profile = _uniform_full_yee_tm_mode_profiles(
        a, b, m, n, u_widths, v_widths, h_offset=h_offset
    )
    if aperture_dA is None:
        return profile
    return _normalize_profile_to_dA(profile, np.asarray(aperture_dA, dtype=np.float64))


def _profile_for_candidate(
    candidate: CandidateSpec,
    a: float,
    b: float,
    mode: tuple[int, int],
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    *,
    h_offset: tuple[float, float] = (0.0, 0.0),
) -> Profile:
    m, n = mode
    if candidate.kind == "uniform_full_yee":
        return _uniform_full_yee_te_mode_profiles(
            a, b, m, n, u_widths, v_widths, h_offset=h_offset
        )
    if candidate.kind == "biortho_full_yee":
        profile = _uniform_full_yee_te_mode_profiles(
            a, b, m, n, u_widths, v_widths, h_offset=h_offset
        )
        return _normalize_profile_to_dA(profile, _partial_dA_np(u_widths, v_widths))
    if candidate.kind == "full_yee_drop_gs":
        return _ORIG_DISCRETE_TE_MODE_PROFILES(
            a,
            b,
            m,
            n,
            u_widths,
            v_widths,
            aperture_dA=_partial_dA_np(u_widths, v_widths),
            h_offset=h_offset,
            _orthogonalize=True,
        )
    if candidate.kind == "reduced_gevp_drop":
        return _reduced_gevp_te_mode_profiles(
            a, b, m, n, u_widths, v_widths, h_offset=h_offset
        )
    raise AssertionError(f"unhandled candidate kind: {candidate.kind}")


def _normalize_profile_to_dA(profile: Profile, dA: np.ndarray) -> Profile:
    ey, ez, hy, hz, kc = profile
    ey = np.asarray(ey, dtype=np.float64).copy()
    ez = np.asarray(ez, dtype=np.float64).copy()
    hy = np.asarray(hy, dtype=np.float64).copy()
    hz = np.asarray(hz, dtype=np.float64).copy()
    c_self = float(np.sum((ey * hz - ez * hy) * dA))
    if abs(c_self) <= 1e-30:
        raise SpikeUnavailable("cannot normalize profile with zero Lorentz self-overlap")
    scale = np.sqrt(abs(c_self))
    ey /= scale
    ez /= scale
    hy /= scale
    hz /= scale
    hy, hz = wg_mod._scale_h_to_unit_cross(ey, ez, hy, hz, dA)
    return ey, ez, hy, hz, kc


def _candidate_dA(
    candidate: CandidateSpec,
    u_widths: np.ndarray,
    v_widths: np.ndarray,
) -> np.ndarray:
    if candidate.kind == "uniform_full_yee":
        return _full_dA_np(u_widths, v_widths)
    return _partial_dA_np(u_widths, v_widths)


def _lorentz_gram(
    profiles: list[Profile],
    dA: np.ndarray,
) -> np.ndarray:
    n_modes = len(profiles)
    gram = np.zeros((n_modes, n_modes), dtype=np.float64)
    for i, lhs in enumerate(profiles):
        ey_i, ez_i, _, _, _ = lhs
        for j, rhs in enumerate(profiles):
            _, _, hy_j, hz_j, _ = rhs
            gram[i, j] = float(np.sum((ey_i * hz_j - ez_i * hy_j) * dA))
    return gram


def _e_power(profile: Profile, dA: np.ndarray) -> float:
    ey, ez, _, _, _ = profile
    return float(np.sum((ey * ey + ez * ez) * dA))


def _h_power(profile: Profile, dA: np.ndarray) -> float:
    _, _, hy, hz, _ = profile
    return float(np.sum((hy * hy + hz * hz) * dA))


def _base_cfg_for_profile(profile: Profile, dA: np.ndarray):
    nu, nv = dA.shape
    dx = 1.0e-3
    port = WaveguidePort(
        x_index=2,
        y_slice=(0, nu),
        z_slice=(0, nv),
        a=nu * dx,
        b=nv * dx,
        mode=(1, 0),
        mode_type="TE",
        direction="+x",
    )
    ey, ez, hy, hz, kc = profile
    cfg = wg_mod.init_waveguide_port(
        port,
        dx=dx,
        freqs=jnp.array([12.0e9]),
        mode_profile="analytic",
        h_offset=(0.0, 0.0),
    )
    return cfg._replace(
        ey_profile=jnp.asarray(ey, dtype=jnp.float32),
        ez_profile=jnp.asarray(ez, dtype=jnp.float32),
        hy_profile=jnp.asarray(hy, dtype=jnp.float32),
        hz_profile=jnp.asarray(hz, dtype=jnp.float32),
        aperture_dA=jnp.asarray(dA, dtype=jnp.float32),
        h_offset=(0.0, 0.0),
        f_cutoff=float(kc) * wg_mod.C0_LOCAL / (2.0 * np.pi),
    )


def _state_with_profile(profile: Profile, *, x_index: int = 2) -> State:
    ey_p, ez_p, hy_p, hz_p, _ = profile
    nu, nv = ey_p.shape
    shape = (x_index + 2, nu, nv)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    ey = zeros.at[x_index, :, :].set(jnp.asarray(ey_p, dtype=jnp.float32))
    ez = zeros.at[x_index, :, :].set(jnp.asarray(ez_p, dtype=jnp.float32))
    hy = zeros.at[x_index, :, :].set(jnp.asarray(hy_p, dtype=jnp.float32))
    hy = hy.at[x_index - 1, :, :].set(jnp.asarray(hy_p, dtype=jnp.float32))
    hz = zeros.at[x_index, :, :].set(jnp.asarray(hz_p, dtype=jnp.float32))
    hz = hz.at[x_index - 1, :, :].set(jnp.asarray(hz_p, dtype=jnp.float32))
    return State(ex=zeros, ey=ey, ez=ez, hx=zeros, hy=hy, hz=hz)


def _vi_consistency(profile: Profile, dA: np.ndarray) -> dict[str, float]:
    cfg = _base_cfg_for_profile(profile, dA)
    state = _state_with_profile(profile, x_index=cfg.x_index)
    voltage = float(wg_mod.modal_voltage(state, cfg, cfg.x_index, cfg.dx))
    current = float(wg_mod.modal_current(state, cfg, cfg.x_index, cfg.dx))
    a_fwd, b_bwd = wg_mod.overlap_modal_amplitude(state, cfg, cfg.x_index, cfg.dx)
    z_mode_complex = complex(wg_mod._compute_mode_impedance(cfg.freqs[0], cfg.f_cutoff, cfg.mode_type))
    return {
        "voltage": voltage,
        "current": current,
        "a_fwd": float(a_fwd),
        "b_bwd_abs": abs(float(b_bwd)),
        "z_mode_abs": abs(z_mode_complex),
        "z_mode_phase": float(np.angle(z_mode_complex)),
        "vi_max_err": max(abs(voltage - 1.0), abs(current - 1.0), abs(float(a_fwd) - 1.0)),
    }


def _synthetic_projection_error(
    candidate: CandidateSpec,
    profiles: list[Profile],
    gram: np.ndarray,
    dA: np.ndarray,
) -> float:
    coeff = np.array([0.75, -0.30], dtype=np.float64)
    use_profiles = profiles[:2]
    if candidate.kind == "biortho_full_yee":
        # BIORTHO_FULL_YEE must match the actual recorder contract, not the
        # Lorentz-overlap algebra used by scalar V/I extraction.  Modal voltage
        # records E·e_m and modal current records H·h_m, so raw = G.T @ coeff
        # with separate E and H dot-product Grams.
        gram_e = _mode_dot_gram_np([(p[0], p[1]) for p in use_profiles], dA)
        gram_h = _mode_dot_gram_np([(p[2], p[3]) for p in use_profiles], dA)
        rhs_e = gram_e.T @ coeff
        rhs_h = gram_h.T @ coeff
        recovered_e = _solve_biorthogonal_coefficients(gram_e, rhs_e, label="synthetic E Gram")
        recovered_h = _solve_biorthogonal_coefficients(gram_h, rhs_h, label="synthetic H Gram")
        return float(max(np.max(np.abs(recovered_e - coeff)), np.max(np.abs(recovered_h - coeff))))

    e_y = sum(coeff[i] * use_profiles[i][0] for i in range(2))
    e_z = sum(coeff[i] * use_profiles[i][1] for i in range(2))
    rhs = np.zeros(2, dtype=np.float64)
    for j, mode in enumerate(use_profiles):
        _, _, hy_j, hz_j, _ = mode
        p1 = float(np.sum((e_y * hz_j - e_z * hy_j) * dA))
        # For a pure forward synthetic field P1=P2 in this convention.
        rhs[j] = p1

    gram2 = gram[:2, :2]
    recovered = np.array([
        rhs[i] / gram2[i, i] if abs(gram2[i, i]) > 1e-30 else np.nan
        for i in range(2)
    ])
    return float(np.max(np.abs(recovered - coeff)))


def _algebraic_ledger(candidate: CandidateSpec) -> dict[str, float | int | str]:
    fixture = "wr90ish_dx1mm_h_offset0"
    a = 22.86e-3
    b = 10.16e-3
    dx = 1.0e-3
    u_widths = np.full(int(round(a / dx)), dx, dtype=np.float64)
    v_widths = np.full(int(round(b / dx)), dx, dtype=np.float64)
    modes = [(1, 0), (2, 0), (3, 0)]
    profiles = [
        _profile_for_candidate(candidate, a, b, mode, u_widths, v_widths)
        for mode in modes
    ]
    dA = _candidate_dA(candidate, u_widths, v_widths)
    gram = _lorentz_gram(profiles, dA)
    diag = np.diag(gram)
    offdiag = gram - np.diag(diag)
    baseline = _profile_for_candidate(
        CANDIDATES[0], a, b, (1, 0), u_widths, v_widths
    )
    kc_te10 = profiles[0][4]
    kc_baseline = baseline[4]
    kc_drift = abs(kc_te10 - kc_baseline) / max(abs(kc_baseline), 1e-30)
    vi = _vi_consistency(profiles[0], dA)
    synth_err = _synthetic_projection_error(candidate, profiles, gram, dA)
    cond = float(np.linalg.cond(gram[:2, :2]))

    algebra_ok = (
        kc_drift <= 0.02
        and vi["vi_max_err"] <= 1.0e-5
        and vi["b_bwd_abs"] <= 1.0e-5
        and synth_err <= 1.0e-5
    )
    if candidate.kind != "biortho_full_yee":
        algebra_ok = algebra_ok and float(np.max(np.abs(offdiag))) <= 1.0e-5

    return {
        "candidate": candidate.name,
        "fixture": fixture,
        "nu": int(u_widths.size),
        "nv": int(v_widths.size),
        "dA": "full" if candidate.kind == "uniform_full_yee" else "drop+faces",
        "h_offset": "(0,0)",
        "nonzero_cells": int(np.count_nonzero(dA > 0.0)),
        "zero_cells": int(np.count_nonzero(dA == 0.0)),
        "kc_te10": float(kc_te10),
        "kc_te20": float(profiles[1][4]),
        "kc_te30": float(profiles[2][4]),
        "kc_drift_te10": float(kc_drift),
        "gram_diag_min": float(np.min(np.abs(diag))),
        "gram_offdiag_max": float(np.max(np.abs(offdiag))),
        "gram_cond_2mode": cond,
        "e_power_te10": _e_power(profiles[0], dA),
        "h_power_te10": _h_power(profiles[0], dA),
        "lorentz_self_te10": float(gram[0, 0]),
        "synthetic_proj_err": synth_err,
        "voltage": vi["voltage"],
        "current": vi["current"],
        "a_fwd": vi["a_fwd"],
        "b_bwd_abs": vi["b_bwd_abs"],
        "z_mode_abs_12ghz": vi["z_mode_abs"],
        "z_mode_phase_12ghz": vi["z_mode_phase"],
        "algebra_ok": "true" if algebra_ok else "false",
    }


def _degeneracy_ledger(candidate: CandidateSpec) -> dict[str, float | str]:
    fixture = "squareish_degeneracy_dx1mm_h_offset0"
    a = 16.0e-3
    b = 16.0e-3
    dx = 1.0e-3
    u_widths = np.full(int(round(a / dx)), dx, dtype=np.float64)
    v_widths = np.full(int(round(b / dx)), dx, dtype=np.float64)
    profiles = [
        _profile_for_candidate(candidate, a, b, mode, u_widths, v_widths)
        for mode in [(1, 0), (0, 1)]
    ]
    dA = _candidate_dA(candidate, u_widths, v_widths)
    gram = _lorentz_gram(profiles, dA)
    return {
        "fixture": fixture,
        "gram_offdiag_max": float(np.max(np.abs(gram - np.diag(np.diag(gram))))),
        "gram_cond_2mode": float(np.linalg.cond(gram)),
    }


def _format_row(prefix: str, row: dict[str, float | int | str]) -> str:
    parts = []
    for key, value in row.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return f"{prefix}: " + " ".join(parts)


# ---------------------------------------------------------------------------
# BIORTHO_FULL_YEE dot-product de-embedding helpers

def _aperture_weights_np(cfg: WaveguidePortConfig) -> np.ndarray:
    weights = cfg.aperture_dA
    if weights is None:
        weights = getattr(cfg, "mode_dA", None)
    if weights is None:
        raise SpikeUnavailable("waveguide config does not carry aperture_dA/mode_dA")
    return np.asarray(weights, dtype=float)


def _mode_dot_gram_np(profiles: Sequence[tuple[object, object]], weights: np.ndarray) -> np.ndarray:
    """Recorder-matched dot-product Gram used by modal_voltage/current."""

    matrix = np.zeros((len(profiles), len(profiles)), dtype=float)
    for i, (ui, vi) in enumerate(profiles):
        ui_np = np.asarray(ui, dtype=float)
        vi_np = np.asarray(vi, dtype=float)
        for j, (uj, vj) in enumerate(profiles):
            integrand = ui_np * np.asarray(uj, dtype=float) + vi_np * np.asarray(vj, dtype=float)
            matrix[i, j] = float(np.sum(integrand * weights))
    return matrix


def _solve_biorthogonal_coefficients(
    gram: np.ndarray,
    raw: np.ndarray,
    *,
    cond_threshold: float = 1.0e10,
    label: str = "Gram",
) -> np.ndarray:
    """Recover modal coefficients for raw[m]=<field,basis_m>.

    The raw recorder contract is ``raw = gram.T @ coeff`` because rows of the
    Gram are test modes and columns are expansion modes.  This helper is kept
    script-local until the physical spike proves the source path deserves
    production support.
    """

    gram_np = np.asarray(gram, dtype=np.complex128)
    raw_np = np.asarray(raw, dtype=np.complex128)
    cond = float(np.linalg.cond(gram_np))
    if not np.isfinite(cond) or cond > cond_threshold:
        raise SpikeUnavailable(f"{label} is ill-conditioned for BIORTHO_FULL_YEE (cond={cond:.3e})")
    return np.linalg.solve(gram_np.T, raw_np)


def _validate_biorthogonal_group(mode_cfgs: Sequence[WaveguidePortConfig]) -> None:
    if not mode_cfgs:
        raise SpikeUnavailable("empty waveguide mode group")
    first = mode_cfgs[0]
    first_weights = _aperture_weights_np(first)
    first_freqs = np.asarray(first.freqs)
    for cfg in mode_cfgs[1:]:
        if cfg.direction != first.direction or cfg.normal_axis != first.normal_axis:
            raise SpikeUnavailable("mixed physical-port group: direction/axis mismatch")
        if (cfg.u_lo, cfg.u_hi, cfg.v_lo, cfg.v_hi) != (first.u_lo, first.u_hi, first.v_lo, first.v_hi):
            raise SpikeUnavailable("mixed physical-port group: aperture slice mismatch")
        if not np.allclose(np.asarray(cfg.freqs), first_freqs):
            raise SpikeUnavailable("mixed physical-port group: frequency grid mismatch")
        if not np.allclose(_aperture_weights_np(cfg), first_weights):
            raise SpikeUnavailable("mixed physical-port group: aperture weights mismatch")


def _biorthogonal_group_waves(
    mode_cfgs: Sequence[WaveguidePortConfig],
    *,
    ref_shift: float | None = None,
    cond_threshold: float = 1.0e10,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Extract local modal waves by solving recorder-matched E/H Grams."""

    _validate_biorthogonal_group(mode_cfgs)
    first = mode_cfgs[0]
    weights = _aperture_weights_np(first)
    e_profiles = [(cfg.ey_profile, cfg.ez_profile) for cfg in mode_cfgs]
    h_profiles = [(cfg.hy_profile, cfg.hz_profile) for cfg in mode_cfgs]
    gram_e = _mode_dot_gram_np(e_profiles, weights)
    gram_h = _mode_dot_gram_np(h_profiles, weights)

    voltage_raw = np.stack(
        [np.asarray(wg_mod._rect_dft(cfg.v_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded)) for cfg in mode_cfgs],
        axis=0,
    )
    current_raw = np.stack(
        [np.asarray(wg_mod._rect_dft(cfg.i_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded)) for cfg in mode_cfgs],
        axis=0,
    )

    voltage_coeff = _solve_biorthogonal_coefficients(
        gram_e, voltage_raw, cond_threshold=cond_threshold, label="electric dot Gram"
    )
    current_coeff = _solve_biorthogonal_coefficients(
        gram_h, current_raw, cond_threshold=cond_threshold, label="magnetic dot Gram"
    )

    a_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    shift = 0.0 if ref_shift is None else float(ref_shift)
    step_sign = 1 if str(first.direction).startswith("+") else -1
    for idx, cfg in enumerate(mode_cfgs):
        forward, backward = wg_mod._extract_global_waves(
            cfg,
            jnp.asarray(voltage_coeff[idx]),
            jnp.asarray(current_coeff[idx]),
        )
        if str(cfg.direction).startswith("+"):
            a_ref, b_ref = forward, backward
        else:
            a_ref, b_ref = backward, forward
        if shift != 0.0:
            beta = wg_mod._compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx)
            a_ref, b_ref = wg_mod._shift_modal_waves(a_ref, b_ref, beta, shift, step_sign)
        a_rows.append(np.asarray(a_ref))
        b_rows.append(np.asarray(b_ref))

    diagnostics = {
        "cond_e": float(np.linalg.cond(gram_e)),
        "cond_h": float(np.linalg.cond(gram_h)),
        "min_diag_e": float(np.min(np.abs(np.diag(gram_e)))),
        "min_diag_h": float(np.min(np.abs(np.diag(gram_h)))),
    }
    return np.stack(a_rows, axis=0), np.stack(b_rows, axis=0), diagnostics


def _split_flat_groups(
    flat_cfgs: Sequence[WaveguidePortConfig], template_groups: Sequence[Sequence[WaveguidePortConfig]]
) -> list[list[WaveguidePortConfig]]:
    groups: list[list[WaveguidePortConfig]] = []
    offset = 0
    for group in template_groups:
        width = len(group)
        groups.append(list(flat_cfgs[offset : offset + width]))
        offset += width
    if offset != len(flat_cfgs):
        raise SpikeUnavailable("mode-group reconstruction mismatch")
    return groups


# ---------------------------------------------------------------------------
# Physical validation harness

def _load_validation_battery() -> Any:
    path = ROOT / "tests" / "test_waveguide_port_validation_battery.py"
    if not path.exists():
        raise SpikeUnavailable(f"missing validation hook: {path}")
    spec = importlib.util.spec_from_file_location("_rfx_validation_battery_spike", path)
    if spec is None or spec.loader is None:
        raise SpikeUnavailable(f"could not import validation hook: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_pec_short_validation() -> dict[str, float | int]:
    vb = _load_validation_battery()
    freqs = np.linspace(5.0e9, 7.0e9, 6)
    sim = vb._build_sim(
        freqs,
        pec_short_x=0.085,
        waveform="modulated_gaussian",
    )
    s, _, port_idx = vb._s_matrix(sim, num_periods=40, normalize=True)
    mag = np.abs(np.asarray(s[port_idx["left"], port_idx["left"], :], dtype=np.complex128))
    return {
        "n_freqs": int(np.asarray(freqs).size),
        "mean_abs_s11": float(np.mean(mag)),
        "min_abs_s11": float(np.min(mag)),
        "max_abs_s11": float(np.max(mag)),
        "max_dev_from_1": float(np.max(np.abs(mag - 1.0))),
    }


def _run_asym_recip_validation(*, num_periods: float) -> dict[str, float | int]:
    vb = _load_validation_battery()
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    obstacles = [((0.03, 0.0, 0.0), (0.05, 0.02, 0.02), 6.0)]
    sim = vb._build_sim(freqs, obstacles=obstacles, waveform="modulated_gaussian")
    s, _, port_idx = vb._s_matrix(sim, num_periods=num_periods, normalize=True)
    s21 = np.abs(s[port_idx["right"], port_idx["left"], :])
    s12 = np.abs(s[port_idx["left"], port_idx["right"], :])
    denom = np.maximum(np.maximum(s21, s12), 1e-12)
    return {
        "mean_reciprocity_rel_diff": float((np.abs(s21 - s12) / denom).mean()),
        "mean_abs_s21": float(np.mean(s21)),
        "mean_abs_s12": float(np.mean(s12)),
        "n_freqs": int(np.asarray(freqs).size),
    }


def _build_multimode_validation_sim(
    freqs_hz: np.ndarray,
    *,
    n_modes: int,
    obstacles: Sequence[tuple[tuple[float, float, float], tuple[float, float, float], float]] = (),
    pec_short_x: float | None = None,
    waveform: str = "modulated_gaussian",
):
    """Build the canonical validation geometry with real multimode ports."""

    from rfx.api import Simulation
    from rfx.geometry.csg import Box

    domain = (0.12, 0.04, 0.02)
    port_left_x = 0.01
    port_right_x = 0.09
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=max(float(freqs[-1]), f0),
        domain=domain,
        boundary="cpml",
        cpml_layers=10,
    )

    for idx, (lo, hi, eps_r) in enumerate(obstacles):
        name = f"diel_{idx}"
        sim.add_material(name, eps_r=eps_r, sigma=0.0)
        sim.add(Box(lo, hi), material=name)

    if pec_short_x is not None:
        thickness = 0.002
        sim.add(
            Box((pec_short_x, 0.0, 0.0), (pec_short_x + thickness, domain[1], domain[2])),
            material="pec",
        )

    port_freqs = jnp.asarray(freqs)
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        n_modes=int(n_modes),
        mode_profile="discrete",
    )
    sim.add_waveguide_port(
        port_left_x,
        direction="+x",
        name="left",
        **common,
    )
    sim.add_waveguide_port(
        port_right_x,
        direction="-x",
        name="right",
        **common,
    )
    return sim


def _flatten_cfgs(cfgs: Sequence[WaveguidePortConfig | Sequence[WaveguidePortConfig]]) -> list[WaveguidePortConfig]:
    out: list[WaveguidePortConfig] = []
    for cfg in cfgs:
        if isinstance(cfg, list):
            out.extend(cfg)
        elif isinstance(cfg, tuple) and cfg and isinstance(cfg[0], WaveguidePortConfig):
            out.extend(cfg)  # defensive; production returns list
        else:
            out.append(cfg)  # type: ignore[arg-type]
    return out


def _prepare_multimode_case(sim: Any, *, num_periods: float):
    """Mirror Simulation.compute_waveguide_s_matrix setup, but expose configs."""

    from rfx.core.yee import init_materials as init_vacuum_materials

    entries = list(sim._waveguide_ports)
    if len(entries) < 2:
        raise SpikeUnavailable("BIORTHO benchmark requires at least two waveguide ports")
    if not any(entry.n_modes > 1 for entry in entries):
        raise SpikeUnavailable("BIORTHO benchmark must be run with n_modes > 1")

    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, _, _ = sim._assemble_materials(grid)
    if pec_mask_wg is not None:
        base_materials = base_materials._replace(
            sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma)
        )
    materials = base_materials
    n_steps = grid.num_timesteps(num_periods=num_periods)
    _, debye, lorentz = sim._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

    def _resolve_freqs(entry: Any) -> jnp.ndarray:
        if entry.freqs is not None:
            return entry.freqs
        return jnp.linspace(sim._freq_max / 10, sim._freq_max, entry.n_freqs)

    freqs = _resolve_freqs(entries[0])
    for entry in entries[1:]:
        entry_freqs = _resolve_freqs(entry)
        if entry_freqs.shape != freqs.shape or not np.allclose(np.asarray(entry_freqs), np.asarray(freqs)):
            raise SpikeUnavailable("BIORTHO benchmark requires matching frequency grids")

    raw_cfgs = [sim._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]
    flat0 = _flatten_cfgs(raw_cfgs)
    ref_t0 = flat0[0].src_t0
    ref_tau = flat0[0].src_tau
    if any(c.src_t0 != ref_t0 or c.src_tau != ref_tau for c in flat0[1:]):
        raw_cfgs = [
            [c._replace(src_t0=ref_t0, src_tau=ref_tau) for c in cfg]
            if isinstance(cfg, list)
            else cfg._replace(src_t0=ref_t0, src_tau=ref_tau)
            for cfg in raw_cfgs
        ]

    port_mode_cfgs: list[list[WaveguidePortConfig]] = []
    for raw in raw_cfgs:
        if isinstance(raw, list):
            port_mode_cfgs.append(raw)
        else:
            port_mode_cfgs.append([raw])

    ref_shifts: list[float] = []
    for entry, mode_cfgs in zip(entries, port_mode_cfgs):
        first_cfg = mode_cfgs[0]
        planes = wg_mod.waveguide_plane_positions(first_cfg)
        desired_ref = entry.reference_plane if entry.reference_plane is not None else planes["source"]
        ref_shifts.append(float(desired_ref - planes["reference"]))

    ref_materials = init_vacuum_materials(grid.shape)
    common_run_kw = dict(
        boundary="cpml",
        cpml_axes=grid.cpml_axes,
        pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
        periodic=None,
    )
    return {
        "grid": grid,
        "materials": materials,
        "ref_materials": ref_materials,
        "debye": debye,
        "lorentz": lorentz,
        "port_mode_cfgs": port_mode_cfgs,
        "n_steps": n_steps,
        "ref_shifts": ref_shifts,
        "common_run_kw": common_run_kw,
        "freqs": freqs,
    }


def _biorthogonal_flat_waves(
    flat_cfgs: Sequence[WaveguidePortConfig],
    template_groups: Sequence[Sequence[WaveguidePortConfig]],
    ref_shifts: Sequence[float],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    groups = _split_flat_groups(flat_cfgs, template_groups)
    a_flat: list[np.ndarray] = []
    b_flat: list[np.ndarray] = []
    cond_e_values: list[float] = []
    cond_h_values: list[float] = []
    for group, shift in zip(groups, ref_shifts):
        a_group, b_group, diag = _biorthogonal_group_waves(group, ref_shift=shift)
        cond_e_values.append(float(diag["cond_e"]))
        cond_h_values.append(float(diag["cond_h"]))
        for mode_idx in range(a_group.shape[0]):
            a_flat.append(np.asarray(a_group[mode_idx]))
            b_flat.append(np.asarray(b_group[mode_idx]))
    return a_flat, b_flat, {
        "max_cond_e": float(max(cond_e_values) if cond_e_values else np.nan),
        "max_cond_h": float(max(cond_h_values) if cond_h_values else np.nan),
    }


def _biorthogonal_multimode_s_params_normalized(
    *,
    grid: Any,
    materials: Any,
    ref_materials: Any,
    port_mode_cfgs: list[list[WaveguidePortConfig]],
    n_steps: int,
    ref_shifts: Sequence[float],
    common_run_kw: dict[str, Any],
    debye: tuple | None,
    lorentz: tuple | None,
) -> tuple[np.ndarray, list[tuple[int, int, str, tuple[int, int]]], dict[str, float]]:
    """Two-run normalized multimode S using BIORTHO group projection."""

    from rfx.simulation import run as run_simulation

    flat_cfgs: list[WaveguidePortConfig] = []
    mode_map: list[tuple[int, int, str, tuple[int, int]]] = []
    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(mode_cfgs):
            flat_cfgs.append(cfg)
            mode_map.append((
                port_idx,
                mode_within,
                cfg.mode_type,
                getattr(cfg, "mode_indices", (0, 0)),
            ))

    n_total = len(flat_cfgs)
    if n_total < 2:
        raise SpikeUnavailable("BIORTHO normalized extraction requires at least two modal channels")
    n_freqs = len(flat_cfgs[0].freqs)
    s_matrix = np.zeros((n_total, n_total, n_freqs), dtype=np.complex128)

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros_t = jnp.zeros_like(cfg.v_probe_t)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    orig_amp = flat_cfgs[0].src_amp if flat_cfgs[0].src_amp != 0.0 else 1.0
    max_cond_e = 0.0
    max_cond_h = 0.0
    for drive_flat_idx in range(n_total):
        ref_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        ref_cfgs[drive_flat_idx] = ref_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        ref_result = run_simulation(
            grid,
            ref_materials,
            n_steps,
            debye=None,
            lorentz=None,
            waveguide_ports=ref_cfgs,
            **common_run_kw,
        )
        ref_final_cfgs = list(ref_result.waveguide_ports or ())
        if len(ref_final_cfgs) != n_total:
            raise RuntimeError(f"Expected {n_total} reference configs, got {len(ref_final_cfgs)}")
        a_ref_all, b_ref_all, ref_diag = _biorthogonal_flat_waves(
            ref_final_cfgs,
            port_mode_cfgs,
            ref_shifts,
        )
        max_cond_e = max(max_cond_e, ref_diag["max_cond_e"])
        max_cond_h = max(max_cond_h, ref_diag["max_cond_h"])
        a_inc_ref = np.asarray(a_ref_all[drive_flat_idx])
        safe_a_inc = np.where(np.abs(a_inc_ref) > 1e-30, a_inc_ref, np.ones_like(a_inc_ref))

        dev_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        dev_cfgs[drive_flat_idx] = dev_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        dev_result = run_simulation(
            grid,
            materials,
            n_steps,
            debye=debye,
            lorentz=lorentz,
            waveguide_ports=dev_cfgs,
            **common_run_kw,
        )
        dev_final_cfgs = list(dev_result.waveguide_ports or ())
        if len(dev_final_cfgs) != n_total:
            raise RuntimeError(f"Expected {n_total} device configs, got {len(dev_final_cfgs)}")
        _, b_dev_all, dev_diag = _biorthogonal_flat_waves(
            dev_final_cfgs,
            port_mode_cfgs,
            ref_shifts,
        )
        max_cond_e = max(max_cond_e, dev_diag["max_cond_e"])
        max_cond_h = max(max_cond_h, dev_diag["max_cond_h"])

        for recv_idx in range(n_total):
            b_recv_dev = np.asarray(b_dev_all[recv_idx])
            b_ref = np.asarray(b_ref_all[recv_idx])
            ref_nonzero = np.abs(b_ref) > 1e-30
            safe_b_ref = np.where(ref_nonzero, b_ref, np.ones_like(b_ref))
            with np.errstate(divide="ignore", invalid="ignore"):
                if recv_idx == drive_flat_idx:
                    s_matrix[recv_idx, drive_flat_idx, :] = (b_recv_dev - b_ref) / safe_a_inc
                else:
                    through_norm = b_recv_dev / safe_b_ref
                    conversion_norm = (b_recv_dev - b_ref) / safe_a_inc
                    s_matrix[recv_idx, drive_flat_idx, :] = np.where(
                        ref_nonzero,
                        through_norm,
                        conversion_norm,
                    )

    return s_matrix, mode_map, {"max_cond_e": max_cond_e, "max_cond_h": max_cond_h}


def _run_biortho_s_matrix(sim: Any, *, num_periods: float):
    case = _prepare_multimode_case(sim, num_periods=num_periods)
    s_matrix, mode_map, diag = _biorthogonal_multimode_s_params_normalized(
        grid=case["grid"],
        materials=case["materials"],
        ref_materials=case["ref_materials"],
        port_mode_cfgs=case["port_mode_cfgs"],
        n_steps=case["n_steps"],
        ref_shifts=case["ref_shifts"],
        common_run_kw=case["common_run_kw"],
        debye=case["debye"],
        lorentz=case["lorentz"],
    )
    return s_matrix, np.asarray(case["freqs"]), mode_map, diag


def _te10_channel_indices(mode_map: Sequence[tuple[int, int, str, tuple[int, int]]]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for flat_idx, (port_idx, _mode_within, mode_type, mode_indices) in enumerate(mode_map):
        if mode_type == "TE" and tuple(mode_indices) == (1, 0):
            if port_idx == 0:
                indices["left"] = flat_idx
            elif port_idx == 1:
                indices["right"] = flat_idx
    if set(indices) != {"left", "right"}:
        raise SpikeUnavailable(f"could not identify left/right TE10 channels in mode_map={mode_map}")
    return indices


def _run_biortho_pec_short_validation(*, n_modes: int, num_periods: float) -> dict[str, float | int | str]:
    freqs = np.linspace(5.0e9, 7.0e9, 6)
    sim = _build_multimode_validation_sim(
        freqs,
        n_modes=n_modes,
        pec_short_x=0.085,
        waveform="modulated_gaussian",
    )
    s, _, mode_map, diag = _run_biortho_s_matrix(sim, num_periods=num_periods)
    ch = _te10_channel_indices(mode_map)
    mag = np.abs(np.asarray(s[ch["left"], ch["left"], :], dtype=np.complex128))
    return {
        "n_modes": int(n_modes),
        "n_total_channels": int(s.shape[0]),
        "n_freqs": int(np.asarray(freqs).size),
        "mean_abs_s11": float(np.mean(mag)),
        "min_abs_s11": float(np.min(mag)),
        "max_abs_s11": float(np.max(mag)),
        "max_dev_from_1": float(np.max(np.abs(mag - 1.0))),
        "max_cond_e": float(diag["max_cond_e"]),
        "max_cond_h": float(diag["max_cond_h"]),
    }


def _run_biortho_asym_recip_validation(*, n_modes: int, num_periods: float) -> dict[str, float | int | str]:
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    obstacles = [((0.03, 0.0, 0.0), (0.05, 0.02, 0.02), 6.0)]
    sim = _build_multimode_validation_sim(
        freqs,
        n_modes=n_modes,
        obstacles=obstacles,
        waveform="modulated_gaussian",
    )
    s, _, mode_map, diag = _run_biortho_s_matrix(sim, num_periods=num_periods)
    ch = _te10_channel_indices(mode_map)
    s21 = np.abs(s[ch["right"], ch["left"], :])
    s12 = np.abs(s[ch["left"], ch["right"], :])
    denom = np.maximum(np.maximum(s21, s12), 1e-12)
    return {
        "n_modes": int(n_modes),
        "n_total_channels": int(s.shape[0]),
        "mean_reciprocity_rel_diff": float((np.abs(s21 - s12) / denom).mean()),
        "mean_abs_s21": float(np.mean(s21)),
        "mean_abs_s12": float(np.mean(s12)),
        "n_freqs": int(np.asarray(freqs).size),
        "max_cond_e": float(diag["max_cond_e"]),
        "max_cond_h": float(diag["max_cond_h"]),
    }


def _run_biortho_physical_benchmarks(
    *,
    n_modes: int,
    num_periods: float,
) -> dict[str, Any]:
    return {
        "pec_short": _run_biortho_pec_short_validation(n_modes=n_modes, num_periods=num_periods),
        "asym_recip": _run_biortho_asym_recip_validation(n_modes=n_modes, num_periods=num_periods),
    }


@contextlib.contextmanager
def patched_waveguide_port(candidate: CandidateSpec) -> Iterator[None]:
    """Patch production hooks enough for expensive benchmark candidates."""

    old_aperture = wg_mod._aperture_dA
    old_te = wg_mod._discrete_te_mode_profiles
    old_tm = wg_mod._discrete_tm_mode_profiles
    old_gs = wg_mod._gram_schmidt_modes
    try:
        if candidate.kind == "uniform_full_yee":
            wg_mod._aperture_dA = _full_aperture_dA
            wg_mod._discrete_te_mode_profiles = _uniform_full_yee_te_mode_profiles
        elif candidate.kind == "full_yee_drop_gs":
            # Current production compromise: no patch.
            pass
        elif candidate.kind == "reduced_gevp_drop":
            wg_mod._aperture_dA = _drop_both_aperture_dA
            wg_mod._discrete_te_mode_profiles = _reduced_gevp_te_mode_profiles
        elif candidate.kind == "biortho_full_yee":
            # Keep the production dropped aperture/source geometry, but disable
            # production Lorentz Gram-Schmidt and defer modal separation to the
            # script-local recorder-matched dot-Gram extraction.
            wg_mod._discrete_te_mode_profiles = _biortho_full_yee_te_mode_profiles
            wg_mod._discrete_tm_mode_profiles = _biortho_full_yee_tm_mode_profiles
            wg_mod._gram_schmidt_modes = lambda cfgs: list(cfgs)
        yield
    finally:
        wg_mod._aperture_dA = old_aperture
        wg_mod._discrete_te_mode_profiles = old_te
        wg_mod._discrete_tm_mode_profiles = old_tm
        wg_mod._gram_schmidt_modes = old_gs


def _run_benchmarks_for_candidate(
    candidate: CandidateSpec,
    *,
    num_periods: float,
    num_periods_dft: float | None,
    n_modes: int,
) -> dict[str, Any]:
    _ = num_periods_dft  # legacy CLI compatibility; current API uses full-record DFT.
    with patched_waveguide_port(candidate):
        if candidate.kind == "biortho_full_yee":
            return _run_biortho_physical_benchmarks(
                n_modes=n_modes,
                num_periods=num_periods,
            )
        return {
            "pec_short": _run_pec_short_validation(),
            "asym_recip": _run_asym_recip_validation(
                num_periods=num_periods,
            ),
        }


def _candidate_verdict(
    row: dict[str, float | int | str], bench: dict[str, Any] | None) -> str:
    if row.get("algebra_ok") != "true":
        return "NO_GO_ALGEBRA"
    if bench is None:
        return "GO_ALGEBRA_ONLY_PHYSICAL_UNRUN"
    pec = bench["pec_short"]
    asym = bench["asym_recip"]
    pec_ok = float(pec["min_abs_s11"]) >= 0.99
    asym_value = float(asym["mean_reciprocity_rel_diff"])
    asym_strict = asym_value < 0.02
    asym_headroom = asym_value < 0.03
    if row.get("candidate") == "BIORTHO_FULL_YEE":
        if pec_ok and asym_strict:
            return "BIORTHO_MULTIMODE_STRICT_PASSED"
        if pec_ok and asym_headroom:
            return "BIORTHO_MULTIMODE_REGRESSION_HEADROOM"
        return "NO_GO_SOURCE_PATH"
    if pec_ok and asym_strict:
        return "GO_STRICT"
    return "GO_COMPROMISE_ONLY" if pec_ok or asym_strict else "NO_GO_PHYSICAL"


def _run_candidate(candidate: CandidateSpec, args: argparse.Namespace) -> int:
    print(f"\n=== {candidate.name} ===")
    print(f"description: {candidate.description}")
    started = time.perf_counter()
    try:
        row = _algebraic_ledger(candidate)
        deg = _degeneracy_ledger(candidate)
        print(_format_row("ledger", row))
        print(_format_row("degeneracy", deg))
        bench = None
        if args.run_benchmarks:
            if not candidate.physical_supported:
                raise SpikeUnavailable(
                    f"{candidate.name} has no benchmark monkeypatch in this spike"
                )
            bench = _run_benchmarks_for_candidate(
                candidate,
                num_periods=args.num_periods,
                num_periods_dft=args.num_periods_dft,
                n_modes=args.n_modes,
            )
            pec = bench["pec_short"]
            asym = bench["asym_recip"]
            print(_format_row("PEC-short", pec))
            print(_format_row("asym-recip", asym))
        print(f"verdict={_candidate_verdict(row, bench)}")
        return 0
    except SpikeUnavailable as exc:
        print(f"SKIP {candidate.name}: {exc}")
        return 2 if args.strict else 0
    except Exception as exc:
        print(f"ERROR {candidate.name}: {exc}")
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        elapsed = time.perf_counter() - started
        print(f"elapsed_s={elapsed:.2f}")


def run(args: argparse.Namespace) -> int:
    print("Phase 2 / P-1 Spike v2: waveguide strict-closure falsification")
    print(f"root: {ROOT}")
    print(f"mode: {'benchmarks' if args.run_benchmarks else 'quick algebraic diagnostics'}")
    print("strict gates: PEC-short min|S11| >= 0.99 and asym-recip mean < 0.02")
    print("fixture contract: dx=1mm, h_offset=(0,0) algebraic smoke, deterministic no RNG")
    print(f"candidate filter: {args.candidate or 'ALL'}; multimode benchmark n_modes={args.n_modes}")

    selected = CANDIDATES
    if args.candidate:
        selected = tuple(c for c in CANDIDATES if c.name == args.candidate)
        if not selected:
            print(f"ERROR: unknown candidate {args.candidate!r}")
            return 1

    status = 0
    for candidate in selected:
        status = max(status, _run_candidate(candidate, args))

    if wg_mod._aperture_dA is not _ORIG_APERTURE_DA:
        print("ERROR: _aperture_dA monkeypatch leaked after run")
        status = 1
    if wg_mod._discrete_te_mode_profiles is not _ORIG_DISCRETE_TE_MODE_PROFILES:
        print("ERROR: _discrete_te_mode_profiles monkeypatch leaked after run")
        status = 1
    if wg_mod._discrete_tm_mode_profiles is not _ORIG_DISCRETE_TM_MODE_PROFILES:
        print("ERROR: _discrete_tm_mode_profiles monkeypatch leaked after run")
        status = 1
    if wg_mod._gram_schmidt_modes is not _ORIG_GRAM_SCHMIDT_MODES:
        print("ERROR: _gram_schmidt_modes monkeypatch leaked after run")
        status = 1
    if wg_mod.init_waveguide_port is not _ORIG_INIT_WAVEGUIDE_PORT:
        print("ERROR: init_waveguide_port monkeypatch leaked after run")
        status = 1
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only algebraic diagnostics (default).",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run expensive PEC-short and asymmetric-reciprocity hooks where supported.",
    )
    parser.add_argument(
        "--candidate",
        choices=[candidate.name for candidate in CANDIDATES],
        default=None,
        help="Run only one named candidate (default: all).",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=3,
        help="Physical multimode channel count for BIORTHO_FULL_YEE benchmarks (default: 3).",
    )
    parser.add_argument(
        "--num-periods",
        type=float,
        default=40.0,
        help="num_periods for the asymmetric-obstacle benchmark hook (default: 40).",
    )
    parser.add_argument(
        "--num-periods-dft",
        type=float,
        default=None,
        help="Optional DFT gate length for the asymmetric-obstacle benchmark hook.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat optional benchmark/helper skips as non-zero.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tracebacks for unexpected config errors.",
    )
    args = parser.parse_args(argv)
    if args.quick and args.run_benchmarks:
        parser.error("--quick and --run-benchmarks are mutually exclusive")
    if args.n_modes < 1:
        parser.error("--n-modes must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

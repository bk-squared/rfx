"""Issue #820 diagnosis driver — monostatic RCS translation sensitivity.

PRE-DECLARATION (written and committed BEFORE any run; the gate constants
below are not to be edited to match a measurement — a changed gate needs a
written root cause first, per the repo's no-silent-gate-loosening rule).

The finding under test
----------------------
cv16 (``validation/crossval/16_pec_sphere_mie_ka_sweep.py``) at ka = 1.0,
cells-per-radius 6.4, clearance 30, grid 91**3, CPML 8, 700 steps: translating
the PEC sphere by an integer number of cells inside the FIXED box moves the
monostatic RCS by 2.194 dB peak-to-peak with ``N_occupied = 1127`` unchanged.
Monostatic RCS is exactly translation-invariant in the continuum, so the
dependence belongs to the solver or to the extraction.

What this script is
-------------------
A read-only probe. It does not modify ``rfx/``. ``_rcs_complex`` below is a
replay of the NORMAL-INCIDENCE branch of ``rfx.rcs.compute_rcs``
(rfx/rcs.py:389-602) that additionally returns the COMPLEX backscatter far
field, which ``RCSResult`` discards. Arm ``equiv`` asserts the replay
reproduces ``compute_rcs(...).monostatic_rcs`` to <= 1e-9 dB before any other
number is reported; a failure there voids every other arm.

Hypotheses, discriminators, pre-declared outcomes and gates
-----------------------------------------------------------
Residuals are nonnegative and compared in the stated units.

H6  Rasterization is not byte-identical under integer translation.
    Check: arm ``raster`` — hash ``sigma > 0`` at every offset and compare with
    ``np.roll(mask(0), offset)``.
    Residual: ``n_cells_differing``. Gate 0.
    H6 TRUE if any offset has n_cells_differing > 0 (the issue's premise would
    be wrong and the raster is a live suspect). H6 FALSE if it is 0 everywhere.

H3  The far-field phase reference is tied to the box, not to the target.
    Monostatic |sigma| should be phase-insensitive: ``compute_far_field``
    (rfx/farfield.py:462-560) builds face coordinates as
    ``(idx - cpml_lo) * d``, so a change of coordinate origin multiplies every
    face contribution by the SAME scalar ``exp(j k rhat.d)`` and cannot change
    |E|. Check: arm ``origin`` — recompute the backscatter far field from ONE
    recorded ``ntff_data`` with the box's ``cpml_lo_*`` shifted by 5 cells.
    Residual: ``max | |E|_shifted / |E|_base - 1 |``. Gate 1e-12.
    H3 FALSE if residual <= 1e-12. H3 TRUE would need > 1e-3.

H1/H2  A target-INDEPENDENT additive contaminant on the NTFF surface (residual
    TFSF incident leakage, issue #280) sums coherently with a scattered far
    field whose phase rotates as ``exp(-2 j k x0)`` under an x translation.
    A fixed phasor plus a rotating phasor has a position-dependent MAGNITUDE
    even though each part is constant. (H2 as filed — "the incident-reference
    subtraction does not move with the target" — is the same mechanism seen
    from the fix side: ``monostatic_rcs`` is always taken from the RAW run,
    rfx/rcs.py:586-602, so the subtraction never runs for this number.)
    Check: arms ``xsweep`` + ``vacuum``. Fit the measured complex backscatter
    ``E_theta(x0) = A exp(-2 j k_fit x0) + B`` (k_fit scanned), and measure B
    independently as the backscatter far field of an EMPTY-domain run with the
    identical TFSF/NTFF setup.
    Residuals and gates:
      (a) ``resid_fit`` = ||E_meas - E_model||_2 / ||E_meas||_2.
          TRUE <= 0.15; FALSE > 0.40.
      (b) ``r_vac`` = | |B_fit| - |E_vacuum| | / |E_vacuum|  (independent
          witness; the vacuum run shares no quantity with the fit).
          TRUE <= 1.0 (within a factor of two); FALSE > 3.0.
      (c) ``pp_pred`` = 20 log10((1+r)/(1-r)), r = |B_fit|/|A_fit|.
          TRUE if | pp_pred - pp_meas | <= 0.30 dB.
      (d) fitted half-wave period within 15 % of ``lambda/2`` = 20.5 cells.

H4  CPML proximity asymmetry — the sphere approaches one absorber as it moves.
H7  NTFF-surface near-field sampling error — the Huygens surface sits
    ``ntff_offset = 1`` cell outside the TFSF box, i.e. in the reactive near
    field, and the target-to-face distance changes by +-10 cells.
    Both predict an EVEN dependence on the offset and a comparable effect for
    a transverse (y) translation, which does NOT rotate the backscatter phase.
    Checks: arm ``ysweep`` (same offsets along y) and arm ``cpmlladder``
    (CPML 8/16/24 at the extreme x offsets; the interior is unchanged, only the
    absorber deepens).
    Residuals and gates:
      (e) ``pp_y / pp_x``. Proximity TRUE >= 0.50; proximity FALSE < 0.20.
      (f) ``asym`` = max_n | sigma(+n) - sigma(-n) | dB over the x sweep.
          Proximity (even) TRUE <= 0.30 dB; coherent interference TRUE if
          ``asym`` is of the order of ``pp_x``.
      (g) CPML ladder: ``pp_x(cpml=24) / pp_x(cpml=8)``. H4 TRUE <= 0.50;
          H4 FALSE >= 0.80.

H5  The TF/SF auxiliary-grid absorber echo (#888 / PR #1005) reaching the
    target with a position-dependent delay or amplitude.
    Stated analytically first: the echo is a delayed replica of the incident
    field injected at the FIXED TFSF faces, so it multiplies the incident
    spectrum by ``(1 + rho exp(-j omega tau))`` with tau set by the aux-grid
    geometry, not by the target position. Direct and echoed illumination reach
    a translated target with the SAME relative delay, so the factor scales A
    and cannot produce translation dependence by itself.
    Check: it is subsumed by witness (b). If |B_fit| is accounted for by the
    vacuum run, no separate echo term is needed to explain the spread.
    H5 is declared NOT-PRIMARY if (a) and (b) pass; it is re-opened if (b)
    fails with |B_fit| >> |E_vacuum|.

H5-bis  CORRECTION to H5, written before the sweep was read and before the
    ``auxecho`` arm was run. The analytic argument above is WRONG, and its
    defect names the second attempt this hypothesis is allowed under R2: the
    1-D auxiliary absorber sits at a FIXED AUXILIARY-GRID index
    (``n_1d - n_cpml_1d``, rfx/sources/tfsf.py:264-274), and the target's
    auxiliary index moves one-for-one with its 3-D x index. So the direct/echo
    delay AT THE TARGET is ``tau(x0) = 2 (X_abs - x0) / c`` and the echo phase
    relative to the direct illumination rotates as ``exp(+2 j k x0)``.
    Collecting the far-field phase for backscatter observation:

        E_back(x0) = S_back E0 exp(-2 j k x0)  +  S_fwd E0 rho exp(-2 j k X_abs)

    -- the same rotating-plus-constant SHAPE as H1, because the echo arrives
    travelling -x and its contribution toward -x_hat is FORWARD scattering.
    The two are told apart by WHERE the constant term comes from, which is
    exactly witness (b): under H1 the constant term is present with no target
    (vacuum run); under H5-bis it is proportional to the target's own forward
    scattering and vanishes in vacuum.
    Quantitative PRE-DECLARED prediction (arm ``auxecho``; both inputs are
    measured/derived independently of the sweep being explained):
      rho and its phase are measured by replaying the 1-D auxiliary grid ALONE
      on this branch and time-gating the echo at the target's auxiliary index,
      and ``S_fwd / S_back`` comes from the committed exact-Mie oracle
      ``tests/fixtures/rcs_sphere_mie/mie_oracle.py::mie_S1_S2`` (complex;
      ``validate_oracle()`` is run first), as ``S1(theta=0) / S1(theta=pi)``.
      (h) ``r_pred`` = rho * |S_fwd / S_back|. Gate:
          ``|r_fit - r_pred| / r_pred`` <= 0.35 for H5-bis TRUE; >= 1.0 for
          H5-bis FALSE.
      (i) phase: ``arg(B_fit / A_fit)`` against
          ``arg(rho_complex * S_fwd / S_back)``. Gate: agreement within
          +-25 degrees MODULO 180 degrees for TRUE -- the modulo is declared
          because S1 and S2 coincide at theta = 0 but differ by a sign at
          theta = pi, so an overall sign is a convention hazard here and is
          not evidence either way.
      Admissibility of the time gate: the trace envelope minimum between the
      direct and echo arrivals must sit at least 20 dB below BOTH arrival
      peaks. If it does not, the split is not clean, the arm is recorded as
      NON-CLOSING and no rho is quoted.

Record-length / ring-down witness (mandatory before any DFT number is quoted)
----------------------------------------------------------------------------
Arm ``record``: re-run the two x offsets carrying sigma_max and sigma_min at
2x n_steps. Residual ``|sigma(2N) - sigma(N)|``, gate 0.10 dB each, and
``|pp(2N) - pp(N)|``, gate 0.20 dB. Exceeding either voids every verdict above
(the spread would be partly truncation).
Arm ``energy``: interior field energy ``sum(E^2 + H^2)`` over a ladder of
n_steps at offset 0, reported as end/peak in dB.

Usage
-----
  PYTHONPATH=<worktree> python3 scripts/diagnostics/issue820_rcs_translation_probe.py --arm raster
  ... --arm equiv | origin | xsweep | ysweep | vacuum | record | energy | cpmlladder | fit

Every arm appends one timestamped JSON file to
``scripts/diagnostics/issue820_results/`` (append-only; a re-run writes a new
file and never overwrites an earlier one).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import rfx  # noqa: E402

_RFX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(rfx.__file__)))
if _RFX_ROOT != _REPO_ROOT:
    raise RuntimeError(
        f"import rfx resolved outside this repo tree ({rfx.__file__}); "
        "refusing to report numbers for a different rfx build."
    )

import jax.numpy as jnp  # noqa: E402
from rfx.grid import Grid, C0  # noqa: E402
from rfx.geometry.csg import Sphere, rasterize  # noqa: E402
from rfx.core.yee import MaterialArrays  # noqa: E402
from rfx.farfield import NTFFBox, compute_far_field  # noqa: E402
from rfx.rcs import _incident_spectrum_amplitude, compute_rcs  # noqa: E402
from rfx.simulation import run  # noqa: E402
from rfx.sources.tfsf import init_tfsf  # noqa: E402

# --- cv16 operating point (copied from validation/crossval/16_*.py) ---------
F0 = 3e9
LAM = C0 / F0
CPML_LAYERS = 8
PEC_SIGMA = 1e7
BANDWIDTH = 0.5
COARSE_CPR = 6.4
KA = 1.0
CLEAR_CELLS = 30          # the issue's reproduction (91**3, N_occupied = 1127)

# --- pre-declared gate constants -------------------------------------------
GATE_RASTER_CELLS = 0
GATE_ORIGIN_REL = 1e-12
GATE_FIT_RESID_TRUE = 0.15
GATE_FIT_RESID_FALSE = 0.40
GATE_VAC_WITNESS_TRUE = 1.0
GATE_VAC_WITNESS_FALSE = 3.0
GATE_PP_PRED_DB = 0.30
GATE_PERIOD_REL = 0.15
GATE_PROX_RATIO_TRUE = 0.50
GATE_PROX_RATIO_FALSE = 0.20
GATE_ASYM_EVEN_DB = 0.30
GATE_CPML_RATIO_TRUE = 0.50
GATE_CPML_RATIO_FALSE = 0.80
GATE_RECORD_DB = 0.10
GATE_RECORD_PP_DB = 0.20

X_OFFSETS = list(range(-10, 11))
Y_OFFSETS = list(range(-10, 11))

_OUT = os.path.join(_SCRIPT_DIR, "issue820_results")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _emit(arm: str, payload: dict) -> str:
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, f"{_stamp()}_{arm}.json")
    payload = dict(payload)
    payload["arm"] = arm
    payload["rfx_file"] = rfx.__file__
    payload["git_head"] = subprocess.run(
        ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False).stdout.strip()
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[emit] {path}")
    return path


def _load_latest(arm: str) -> dict:
    files = sorted(f for f in os.listdir(_OUT) if f.endswith(f"_{arm}.json"))
    if not files:
        raise SystemExit(f"no {arm} result in {_OUT}; run that arm first")
    with open(os.path.join(_OUT, files[-1])) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
def build_case(offset_cells=(0, 0, 0), *, cpml_layers=CPML_LAYERS,
               vacuum=False, steps_mult=1.0):
    """cv16's ka=1.0 coarse point with the sphere translated by integer cells."""
    radius = KA * LAM / (2 * np.pi)
    res = max(15, int(np.ceil(2 * np.pi * COARSE_CPR / KA)))
    dx = LAM / res
    domain = 2 * radius + 2 * CLEAR_CELLS * dx
    grid = Grid(freq_max=F0 * 1.5, domain=(domain,) * 3, dx=dx,
                cpml_layers=cpml_layers)
    # n_steps is derived from the domain only, so it is IDENTICAL at every
    # offset and at every CPML depth (the interior is unchanged by either).
    n_steps = int(max(700, np.ceil(2.2 * domain / C0 / grid.dt)) * steps_mult)
    center = tuple(domain / 2 + n * dx for n in offset_cells)
    sphere = Sphere(center=center, radius=radius)
    if vacuum:
        eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    else:
        eps_r, sigma = rasterize(grid, [(sphere, 1.0, PEC_SIGMA)])
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma,
                          mu_r=jnp.ones(grid.shape, dtype=jnp.float32))
    mask = np.asarray(sigma) > 0
    meta = {
        "offset_cells": list(offset_cells),
        "cpml_layers": cpml_layers,
        "grid_shape": list(grid.shape),
        "dx": dx,
        "res": res,
        "n_steps": n_steps,
        "domain_over_lam": domain / LAM,
        "n_occupied": int(mask.sum()),
        "mask_sha256": hashlib.sha256(
            np.ascontiguousarray(mask)).hexdigest()[:16],
    }
    return grid, mats, n_steps, mask, meta


def _tfsf_and_ntff(grid, *, cpml_layers, tfsf_margin=3, ntff_offset=1,
                   freqs=(F0,)):
    """Exactly rfx.rcs.compute_rcs steps 1-2, normal-incidence branch."""
    tfsf_cfg, tfsf_st = init_tfsf(
        nx=grid.nx, dx=grid.dx, dt=grid.dt, cpml_layers=cpml_layers,
        tfsf_margin=tfsf_margin, f0=F0, bandwidth=BANDWIDTH, amplitude=1.0,
        polarization="ez", direction="+x", angle_deg=0.0,
        ny=grid.ny, nz=grid.nz, method="bloch",
    )
    fl = getattr(grid, "face_layers", None) or {
        k: grid.cpml_layers
        for k in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")}
    i_lo = max(tfsf_cfg.x_lo - ntff_offset, 1)
    i_hi = min(tfsf_cfg.x_hi + ntff_offset + 1, grid.nx - 2)
    j_lo = max(fl["y_lo"] + ntff_offset, 1)
    j_hi = min(grid.ny - fl["y_hi"] - ntff_offset, grid.ny - 2)
    k_lo = max(fl["z_lo"] + ntff_offset, 1)
    k_hi = min(grid.nz - fl["z_hi"] - ntff_offset, grid.nz - 2)
    box = NTFFBox.from_grid(grid, i_lo=i_lo, i_hi=i_hi, j_lo=j_lo, j_hi=j_hi,
                            k_lo=k_lo, k_hi=k_hi,
                            freqs=jnp.array(np.asarray(freqs, dtype=np.float64),
                                            dtype=jnp.float32))
    return (tfsf_cfg, tfsf_st), box


def _rcs_complex(grid, mats, n_steps, *, cpml_layers=CPML_LAYERS,
                 tfsf_margin=3, ntff_offset=1):
    """Replay of compute_rcs's normal-incidence path, returning complex E_back."""
    freqs_arr = np.array([F0], dtype=np.float64)
    tfsf, box = _tfsf_and_ntff(grid, cpml_layers=cpml_layers,
                               tfsf_margin=tfsf_margin,
                               ntff_offset=ntff_offset, freqs=freqs_arr)
    res = run(grid, mats, n_steps, boundary="cpml", tfsf=tfsf, ntff=box)
    # backscatter direction for +x incidence: (theta, phi) = (pi/2, pi)
    ff = compute_far_field(res.ntff_data, box, grid,
                           np.array([np.pi / 2]), np.array([np.pi]))
    e_th = np.asarray(ff.E_theta, dtype=np.complex128)[:, 0, 0]
    e_ph = np.asarray(ff.E_phi, dtype=np.complex128)[:, 0, 0]
    e_inc = _incident_spectrum_amplitude(F0, BANDWIDTH, freqs_arr,
                                         grid.dt, n_steps)
    p_inc = np.abs(e_inc) ** 2
    mono = 10.0 * np.log10(np.maximum(
        4.0 * np.pi * (np.abs(e_th) ** 2 + np.abs(e_ph) ** 2)
        / np.where(p_inc > 0, p_inc, 1e-30), 1e-30))
    out = {
        "monostatic_dbsm": float(mono[0]),
        "E_theta": [float(e_th[0].real), float(e_th[0].imag)],
        "E_phi": [float(e_ph[0].real), float(e_ph[0].imag)],
        "E_inc_abs": float(np.abs(e_inc[0])),
        "ntff_box": [int(box.i_lo), int(box.i_hi), int(box.j_lo),
                     int(box.j_hi), int(box.k_lo), int(box.k_hi)],
        "tfsf_x": [int(tfsf[0].x_lo), int(tfsf[0].x_hi)],
    }
    return out, res, box, ff


# --- arms ------------------------------------------------------------------
def arm_raster(_args):
    _, _, _, base_mask, base_meta = build_case((0, 0, 0))
    rows = []
    worst = 0
    for axis, offs in (("x", X_OFFSETS), ("y", Y_OFFSETS)):
        ax = "xyz".index(axis)
        for n in offs:
            off = [0, 0, 0]
            off[ax] = n
            _, _, _, mask, meta = build_case(tuple(off))
            rolled = np.roll(base_mask, n, axis=ax)
            ndiff = int(np.count_nonzero(mask != rolled))
            worst = max(worst, ndiff)
            rows.append({"axis": axis, "offset": n,
                         "n_occupied": meta["n_occupied"],
                         "mask_sha256": meta["mask_sha256"],
                         "n_cells_differing_vs_rolled_base": ndiff})
    verdict = ("H6 TRUE (raster moves)" if worst > GATE_RASTER_CELLS
               else "H6 FALSE (raster byte-identical under integer translation)")
    print(f"[raster] worst n_cells_differing = {worst} "
          f"(gate {GATE_RASTER_CELLS}) -> {verdict}")
    return _emit("raster", {"base": base_meta, "rows": rows,
                            "worst_n_cells_differing": worst,
                            "gate": GATE_RASTER_CELLS, "verdict": verdict})


def arm_equiv(_args):
    grid, mats, n_steps, _, meta = build_case((0, 0, 0))
    t0 = time.time()
    mine, _, _, _ = _rcs_complex(grid, mats, n_steps)
    t_mine = time.time() - t0
    t0 = time.time()
    ref = compute_rcs(grid, mats, n_steps, f0=F0, bandwidth=BANDWIDTH,
                      theta_inc=0.0, polarization="ez",
                      theta_obs=np.array([np.pi / 2]),
                      phi_obs=np.array([0.0, np.pi]),
                      freqs=np.array([F0]), boundary="cpml",
                      cpml_layers=CPML_LAYERS)
    t_ref = time.time() - t0
    d = abs(float(ref.monostatic_rcs[0]) - mine["monostatic_dbsm"])
    print(f"[equiv] replay {mine['monostatic_dbsm']:.9f} dBsm vs compute_rcs "
          f"{float(ref.monostatic_rcs[0]):.9f} dBsm  |delta| = {d:.3e} dB "
          f"({t_mine:.1f}s / {t_ref:.1f}s)")
    return _emit("equiv", {"meta": meta, "replay": mine,
                           "compute_rcs_dbsm": float(ref.monostatic_rcs[0]),
                           "abs_delta_db": d, "gate": 1e-9,
                           "pass": bool(d <= 1e-9),
                           "wall_s": [t_mine, t_ref]})


def _sweep(axis, offsets, cpml_layers=CPML_LAYERS, steps_mult=1.0):
    rows = []
    ax = "xyz".index(axis)
    for n in offsets:
        off = [0, 0, 0]
        off[ax] = n
        grid, mats, n_steps, _, meta = build_case(
            tuple(off), cpml_layers=cpml_layers, steps_mult=steps_mult)
        t0 = time.time()
        r, _, _, _ = _rcs_complex(grid, mats, n_steps,
                                  cpml_layers=cpml_layers)
        r.update(meta)
        r["wall_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  {axis}{n:+3d} cpml={cpml_layers} steps={n_steps} "
              f"sigma = {r['monostatic_dbsm']:9.4f} dBsm  ({r['wall_s']}s)")
    vals = [r["monostatic_dbsm"] for r in rows]
    return rows, float(max(vals) - min(vals))


def arm_xsweep(_args):
    rows, pp = _sweep("x", X_OFFSETS)
    print(f"[xsweep] peak-to-peak = {pp:.4f} dB")
    return _emit("xsweep", {"rows": rows, "pp_db": pp, "offsets": X_OFFSETS})


def arm_ysweep(_args):
    rows, pp = _sweep("y", Y_OFFSETS)
    print(f"[ysweep] peak-to-peak = {pp:.4f} dB")
    return _emit("ysweep", {"rows": rows, "pp_db": pp, "offsets": Y_OFFSETS})


def arm_vacuum(_args):
    grid, mats, n_steps, _, meta = build_case((0, 0, 0), vacuum=True)
    r, _, _, _ = _rcs_complex(grid, mats, n_steps)
    e_th = complex(*r["E_theta"])
    print(f"[vacuum] empty-domain backscatter |E_theta| = {abs(e_th):.6e}, "
          f"sigma = {r['monostatic_dbsm']:.4f} dBsm")
    return _emit("vacuum", {"meta": meta, "result": r,
                            "abs_E_theta": abs(e_th)})


def arm_origin(_args):
    grid, mats, n_steps, _, meta = build_case((0, 0, 0))
    _, res, box, _ = _rcs_complex(grid, mats, n_steps)
    th, ph = np.array([np.pi / 2]), np.array([np.pi])
    base = compute_far_field(res.ntff_data, box, grid, th, ph)
    shifted_box = box._replace(cpml_lo_x=box.cpml_lo_x + 5,
                               cpml_lo_y=box.cpml_lo_y + 5,
                               cpml_lo_z=box.cpml_lo_z + 5)
    sh = compute_far_field(res.ntff_data, shifted_box, grid, th, ph)

    def mag(f):
        return float(np.hypot(abs(np.asarray(f.E_theta)[0, 0, 0]),
                              abs(np.asarray(f.E_phi)[0, 0, 0])))

    rel = abs(mag(sh) / mag(base) - 1.0)
    ph_base = float(np.angle(np.asarray(base.E_theta)[0, 0, 0]))
    ph_sh = float(np.angle(np.asarray(sh.E_theta)[0, 0, 0]))
    verdict = ("H3 FALSE (|E| is coordinate-origin invariant)"
               if rel <= GATE_ORIGIN_REL else "H3 TRUE")
    print(f"[origin] |E| relative change under a 5-cell origin shift = "
          f"{rel:.3e} (gate {GATE_ORIGIN_REL}); phase moved "
          f"{ph_sh - ph_base:+.4f} rad -> {verdict}")
    return _emit("origin", {"meta": meta, "rel_magnitude_change": rel,
                            "phase_base_rad": ph_base,
                            "phase_shifted_rad": ph_sh,
                            "gate": GATE_ORIGIN_REL, "verdict": verdict})


def arm_record(_args):
    prev = _load_latest("xsweep")
    rows = prev["rows"]
    vals = [r["monostatic_dbsm"] for r in rows]
    i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
    out = []
    for idx in (i_max, i_min):
        n = rows[idx]["offset_cells"][0]
        grid, mats, n_steps, _, meta = build_case((n, 0, 0), steps_mult=2.0)
        r, _, _, _ = _rcs_complex(grid, mats, n_steps)
        d = abs(r["monostatic_dbsm"] - vals[idx])
        print(f"  x{n:+3d}: sigma(N={rows[idx]['n_steps']}) = {vals[idx]:.4f} "
              f"-> sigma(2N={n_steps}) = {r['monostatic_dbsm']:.4f} dBsm "
              f"|delta| = {d:.4f} dB (gate {GATE_RECORD_DB})")
        out.append({"offset": n, "sigma_N": vals[idx],
                    "sigma_2N": r["monostatic_dbsm"], "abs_delta_db": d,
                    "n_steps_2N": n_steps, "meta": meta})
    pp_n = max(vals) - min(vals)
    pp_2n = abs(out[0]["sigma_2N"] - out[1]["sigma_2N"])
    print(f"[record] pp(N) = {pp_n:.4f} dB, pp(2N at the same two offsets) = "
          f"{pp_2n:.4f} dB, |delta| = {abs(pp_2n - pp_n):.4f} dB "
          f"(gate {GATE_RECORD_PP_DB})")
    return _emit("record", {"rows": out, "pp_N_db": pp_n, "pp_2N_db": pp_2n,
                            "abs_pp_delta_db": abs(pp_2n - pp_n),
                            "gates": [GATE_RECORD_DB, GATE_RECORD_PP_DB]})


def arm_energy(_args):
    grid, mats, n_steps, _, meta = build_case((0, 0, 0))
    ladder = sorted(set(list(range(100, n_steps + 1, 100)) + [n_steps]))
    i0, i1 = grid.pad_x_lo, grid.nx - grid.pad_x_hi
    sl = (slice(i0, i1),) * 3
    rows = []
    for n in ladder:
        tfsf, box = _tfsf_and_ntff(grid, cpml_layers=CPML_LAYERS)
        res = run(grid, mats, n, boundary="cpml", tfsf=tfsf, ntff=box)
        s = res.state
        u = float(jnp.sum(s.ex[sl] ** 2 + s.ey[sl] ** 2 + s.ez[sl] ** 2
                          + s.hx[sl] ** 2 + s.hy[sl] ** 2 + s.hz[sl] ** 2))
        rows.append({"n_steps": n, "interior_energy": u})
        print(f"  n={n:5d}  U = {u:.6e}")
    peak_row = max(rows, key=lambda r: r["interior_energy"])
    peak = peak_row["interior_energy"]
    end = rows[-1]["interior_energy"]
    db = 10.0 * np.log10(max(end, 1e-300) / peak)
    print(f"[energy] end/peak = {db:.2f} dB (peak U = {peak:.4e} at n = "
          f"{peak_row['n_steps']}, end U = {end:.4e} at n = "
          f"{rows[-1]['n_steps']})")
    return _emit("energy", {"meta": meta, "rows": rows,
                            "peak_n_steps": peak_row["n_steps"],
                            "end_over_peak_db": float(db)})


def arm_cpmlladder(_args):
    prev = _load_latest("xsweep")
    vals = [r["monostatic_dbsm"] for r in prev["rows"]]
    offs = [prev["rows"][int(np.argmax(vals))]["offset_cells"][0],
            prev["rows"][int(np.argmin(vals))]["offset_cells"][0]]
    out = {}
    for cp in (8, 16, 24):
        rows, _ = _sweep("x", offs, cpml_layers=cp)
        pp = abs(rows[0]["monostatic_dbsm"] - rows[1]["monostatic_dbsm"])
        out[str(cp)] = {"rows": rows, "pp_db": pp}
        print(f"  cpml={cp}: sigma spread over offsets {offs} = {pp:.4f} dB")
    ratio = (out["24"]["pp_db"] / out["8"]["pp_db"]
             if out["8"]["pp_db"] else float("nan"))
    if ratio <= GATE_CPML_RATIO_TRUE:
        verdict = "H4 TRUE (CPML reflection dominates)"
    elif ratio >= GATE_CPML_RATIO_FALSE:
        verdict = "H4 FALSE (spread survives a 3x deeper absorber)"
    else:
        verdict = "H4 INCONCLUSIVE"
    print(f"[cpmlladder] pp(24)/pp(8) = {ratio:.3f} -> {verdict}")
    return _emit("cpmlladder", {"offsets": offs, "by_cpml": out,
                                "ratio_24_over_8": ratio, "verdict": verdict})


def arm_fit(_args):
    xs = _load_latest("xsweep")
    dx = xs["rows"][0]["dx"]
    offs = np.array([r["offset_cells"][0] for r in xs["rows"]], dtype=float)
    e_meas = np.array([complex(*r["E_theta"]) for r in xs["rows"]])
    x0 = offs * dx
    k0 = 2 * np.pi * F0 / C0
    best = None
    for k in np.linspace(0.7 * k0, 1.4 * k0, 7001):
        m = np.stack([np.exp(-2j * k * x0),
                      np.ones_like(x0, dtype=complex)], axis=1)
        coef, *_ = np.linalg.lstsq(m, e_meas, rcond=None)
        resid = np.linalg.norm(e_meas - m @ coef) / np.linalg.norm(e_meas)
        if best is None or resid < best[0]:
            best = (float(resid), float(k), complex(coef[0]), complex(coef[1]))
    resid, k_fit, a_fit, b_fit = best
    r = abs(b_fit) / abs(a_fit)
    pp_pred = 20.0 * np.log10((1 + r) / (1 - r)) if r < 1 else float("inf")
    pp_meas = xs["pp_db"]
    period_cells = (np.pi / k_fit) / dx           # lambda_fit / 2 in cells
    period_ref = (np.pi / k0) / dx
    try:
        vac = _load_latest("vacuum")
        e_vac = abs(complex(*vac["result"]["E_theta"]))
        r_vac = abs(abs(b_fit) - e_vac) / e_vac
    except SystemExit:
        e_vac, r_vac = None, None
    try:
        pp_y = _load_latest("ysweep")["pp_db"]
    except SystemExit:
        pp_y = None
    n = len(offs)
    asym = max(abs(xs["rows"][i]["monostatic_dbsm"]
                   - xs["rows"][n - 1 - i]["monostatic_dbsm"])
               for i in range(n // 2))
    print(f"[fit] resid = {resid:.4f} (TRUE<= {GATE_FIT_RESID_TRUE}, "
          f"FALSE> {GATE_FIT_RESID_FALSE})")
    print(f"[fit] |A| = {abs(a_fit):.6e}  |B| = {abs(b_fit):.6e}  r = {r:.4f}")
    print(f"[fit] pp_pred = {pp_pred:.4f} dB vs pp_meas = {pp_meas:.4f} dB "
          f"(gate {GATE_PP_PRED_DB})")
    print(f"[fit] fitted half-wave period = {period_cells:.3f} cells vs "
          f"lambda/2 = {period_ref:.3f} cells (rel "
          f"{abs(period_cells / period_ref - 1):.4f}, gate {GATE_PERIOD_REL})")
    if e_vac is not None:
        print(f"[fit] independent witness |E_vacuum| = {e_vac:.6e}, "
              f"r_vac = {r_vac:.4f} (TRUE<= {GATE_VAC_WITNESS_TRUE}, "
              f"FALSE> {GATE_VAC_WITNESS_FALSE})")
    if pp_y is not None:
        print(f"[fit] pp_y/pp_x = {pp_y / pp_meas:.4f} "
              f"(proximity TRUE>= {GATE_PROX_RATIO_TRUE}, "
              f"FALSE< {GATE_PROX_RATIO_FALSE})")
    print(f"[fit] asymmetry max|sigma(+n)-sigma(-n)| = {asym:.4f} dB "
          f"(even/proximity TRUE<= {GATE_ASYM_EVEN_DB})")
    return _emit("fit", {
        "resid": resid, "k_fit": k_fit, "k0": k0,
        "A": [a_fit.real, a_fit.imag], "B": [b_fit.real, b_fit.imag],
        "abs_A": abs(a_fit), "abs_B": abs(b_fit), "r": r,
        "pp_pred_db": pp_pred, "pp_meas_db": pp_meas,
        "period_cells": period_cells, "period_ref_cells": period_ref,
        "abs_E_vacuum": e_vac, "r_vac": r_vac,
        "pp_y_db": pp_y, "pp_y_over_pp_x": (pp_y / pp_meas) if pp_y else None,
        "asym_db": asym,
        "gates": {
            "fit_resid_true": GATE_FIT_RESID_TRUE,
            "fit_resid_false": GATE_FIT_RESID_FALSE,
            "vac_witness_true": GATE_VAC_WITNESS_TRUE,
            "vac_witness_false": GATE_VAC_WITNESS_FALSE,
            "pp_pred_db": GATE_PP_PRED_DB,
            "period_rel": GATE_PERIOD_REL,
            "prox_ratio_true": GATE_PROX_RATIO_TRUE,
            "prox_ratio_false": GATE_PROX_RATIO_FALSE,
            "asym_even_db": GATE_ASYM_EVEN_DB,
        },
    })


GATE_AUXECHO_R_TRUE = 0.35
GATE_AUXECHO_R_FALSE = 1.0
GATE_AUXECHO_PHASE_DEG = 25.0
GATE_AUXECHO_GATE_FLOOR_DB = -20.0


def arm_auxecho(_args):
    """Replay the 1-D auxiliary grid alone and predict the constant term."""
    sys.path.insert(0, os.path.join(_REPO_ROOT, "tests", "fixtures",
                                    "rcs_sphere_mie"))
    from mie_oracle import mie_S1_S2, validate_oracle
    witnesses = validate_oracle()
    print("[auxecho] Mie oracle self-check PASS:",
          {k: (round(float(v), 6) if np.isscalar(v) else v)
           for k, v in witnesses.items()})

    from rfx.sources.tfsf import update_tfsf_1d

    grid, _, n_steps, _, meta = build_case((0, 0, 0))
    tfsf, _ = _tfsf_and_ntff(grid, cpml_layers=CPML_LAYERS)
    cfg, st = tfsf
    dx, dt = grid.dx, grid.dt
    n_1d = int(np.asarray(st.e1d).shape[0])
    trace = np.zeros((n_steps, n_1d))
    for n in range(n_steps):
        st = update_tfsf_1d(cfg, st, dx, dt, n * dt)
        trace[n] = np.asarray(st.e1d)

    # target's 1-D auxiliary index at offset 0 (sphere centre cell)
    i_centre = grid.nx // 2
    p0 = int(cfg.i0) + (i_centre - int(cfg.x_lo))
    x_abs = n_1d - int(cfg.n_cpml)          # first absorbing cell, aux index
    rows = []
    k0 = 2 * np.pi * F0 / C0
    t = np.arange(n_steps) * dt
    kern = np.exp(-2j * np.pi * F0 * t)
    for off in (-10, -5, 0, 5, 10):
        p = p0 + off
        s = trace[:, p]
        env = np.abs(s)
        # direct arrival = first envelope peak; echo = the later one
        i_dir = int(np.argmax(env[: n_steps // 2]))
        i_echo = int(np.argmax(env[i_dir + 1:])) + i_dir + 1
        if i_echo <= i_dir + 5:
            rows.append({"offset": off, "gate_clean": False})
            continue
        i_split = i_dir + int(np.argmin(env[i_dir:i_echo]))
        floor_db = 20 * np.log10(max(env[i_split], 1e-300)
                                 / max(env[i_dir], 1e-300))
        clean = floor_db <= GATE_AUXECHO_GATE_FLOOR_DB
        d_dft = complex(np.sum(s[:i_split] * kern[:i_split]) * dt)
        e_dft = complex(np.sum(s[i_split:] * kern[i_split:]) * dt)
        rho = abs(e_dft) / abs(d_dft)
        rows.append({
            "offset": off, "aux_index": p, "i_direct": i_dir,
            "i_echo": i_echo, "i_split": i_split,
            "gate_floor_db": float(floor_db), "gate_clean": bool(clean),
            "rho": float(rho),
            "echo_over_direct": [float((e_dft / d_dft).real),
                                 float((e_dft / d_dft).imag)],
            "delay_steps": int(i_echo - i_dir),
            "delay_steps_predicted": int(round(2 * (x_abs - p) * dx
                                               / (C0 * dt))),
        })
        print(f"  aux offset {off:+3d} (index {p}): rho = {rho:.5f}, "
              f"gate floor {floor_db:6.1f} dB "
              f"({'clean' if clean else 'NOT CLEAN'}), delay "
              f"{i_echo - i_dir} steps vs predicted "
              f"{rows[-1]['delay_steps_predicted']}")

    good = [r for r in rows if r.get("gate_clean")]
    if not good:
        print("[auxecho] NON-CLOSING: no offset gave a clean direct/echo split")
        return _emit("auxecho", {"meta": meta, "rows": rows,
                                 "closing": False})
    rho_mean = float(np.mean([r["rho"] for r in good]))
    ka_eff = 2 * np.pi * (KA * LAM / (2 * np.pi)) / LAM
    s_fwd = mie_S1_S2(ka_eff, 0.0, n_max=20)[0]
    s_back = mie_S1_S2(ka_eff, np.pi, n_max=20)[0]
    ratio = s_fwd / s_back
    r_pred = rho_mean * abs(ratio)
    # phase of the constant term relative to the rotating one, from the replay
    # at the reference offset 0 (the fit's x0 = 0 point)
    ref = [r for r in good if r["offset"] == 0]
    phase_pred = None
    if ref:
        eo = complex(*ref[0]["echo_over_direct"])
        phase_pred = float(np.degrees(np.angle(eo * ratio)))
    out = {"meta": meta, "rows": rows, "closing": True,
           "rho_mean": rho_mean, "x_abs_aux_index": x_abs,
           "aux_n_1d": n_1d, "aux_i0": int(cfg.i0),
           "aux_n_cpml": int(cfg.n_cpml), "aux_src_idx": int(cfg.src_idx),
           "mie_S_fwd": [float(s_fwd.real), float(s_fwd.imag)],
           "mie_S_back": [float(s_back.real), float(s_back.imag)],
           "abs_S_fwd_over_S_back": float(abs(ratio)),
           "r_pred": r_pred, "phase_pred_deg": phase_pred, "k0": k0,
           "gates": {"r_true": GATE_AUXECHO_R_TRUE,
                     "r_false": GATE_AUXECHO_R_FALSE,
                     "phase_deg": GATE_AUXECHO_PHASE_DEG,
                     "gate_floor_db": GATE_AUXECHO_GATE_FLOOR_DB}}
    print(f"[auxecho] rho = {rho_mean:.5f}, |S_fwd/S_back| = {abs(ratio):.5f} "
          f"-> r_pred = {r_pred:.5f}")
    try:
        ft = _load_latest("fit")
        r_fit = ft["r"]
        rel = abs(r_fit - r_pred) / r_pred
        a_fit, b_fit = complex(*ft["A"]), complex(*ft["B"])
        phase_fit = float(np.degrees(np.angle(b_fit / a_fit)))
        dphi = abs((phase_fit - phase_pred + 90.0) % 180.0 - 90.0) \
            if phase_pred is not None else None
        out.update({"r_fit": r_fit, "rel_r": rel,
                    "phase_fit_deg": phase_fit, "phase_delta_mod180_deg": dphi})
        print(f"[auxecho] r_fit = {r_fit:.5f} -> |r_fit-r_pred|/r_pred = "
              f"{rel:.3f} (TRUE<= {GATE_AUXECHO_R_TRUE}, "
              f"FALSE>= {GATE_AUXECHO_R_FALSE})")
        if dphi is not None:
            print(f"[auxecho] arg(B/A) = {phase_fit:+.1f} deg vs predicted "
                  f"{phase_pred:+.1f} deg; |delta| mod 180 = {dphi:.1f} deg "
                  f"(gate {GATE_AUXECHO_PHASE_DEG})")
    except SystemExit:
        print("[auxecho] no fit result yet; prediction recorded alone")
    return _emit("auxecho", out)


ARMS = {
    "raster": arm_raster, "equiv": arm_equiv, "origin": arm_origin,
    "xsweep": arm_xsweep, "ysweep": arm_ysweep, "vacuum": arm_vacuum,
    "record": arm_record, "energy": arm_energy, "cpmlladder": arm_cpmlladder,
    "fit": arm_fit, "auxecho": arm_auxecho,
}


def main(argv=None):
    p = argparse.ArgumentParser(description="issue #820 translation probe")
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = p.parse_args(argv)
    ARMS[args.arm](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

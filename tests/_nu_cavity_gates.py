"""Non-uniform (graded-z) rectangular PEC cavity vs the exact Pozar
spectrum: the profiles, the closed-form mode list, mode identification, the
exact-lattice prediction and the estimator floor derived from it. Pure numpy;
no rfx import.

These are the gates of the former cv24 cross-validation case, removed
2026-09-21; the module is kept because
``tests/oracle/test_harminv_retained_cavity_records.py``,
``tests/unit/nonuniform/test_graded_metric_arrays_independent.py`` and
``scripts/diagnostics/harminv_cavity_adjudication.py`` judge retained records
with them. Its pre-declaration, for the derivations, is
``docs/design_notes/20260902_cv24_nu_cavity_predeclaration.md``.

Trimmed 2026-09-21 to the names those three import -- ``CHANNELS``,
``PROFILES``, ``declared_modes``, ``identify_modes``, ``estimator_floor``,
``lattice_freq``, ``inv_arrays``, ``swapped_inv_arrays`` -- plus what those
names call. The arms, the falsifiers, the per-transition allowance, the
record-length derivation, the envelope check and the gate evaluation went with
the case; the pre-declaration above still carries them.

Every number lives HERE, once. The
FDTD never enters this module: the exact-lattice prediction is a
frequency-domain eigenvalue computation on the SAME difference operators
``rfx.nonuniform._profile_to_inv_arrays`` builds (mirrored below in float64,
``inv_arrays``), and the identification windows are derived from committed
data and this model.

Model class of ``lattice_freq``: first principles (SPEC-00 0.2-2). For an
empty PEC box on a tensor Yee grid the discrete curl-curl operator separates
per axis (the per-component 1-D operators along the two "node" axes are the
primal Dirichlet operator ``A`` and along the "edge" axis the dual operator
``B = -inv_h . D_e^{-1} . inv_h^T``; ``A = -D_e^{-1} inv_h^T inv_h`` and ``B``
are ``XY`` and ``YX`` and therefore share their nonzero spectrum), so every
discrete eigenfrequency is

    sin(omega dt / 2) = (c0 dt / 2) sqrt(mu_x(m) + mu_y(n) + mu_z(l))

with ``mu_axis(i)`` the i-th eigenvalue of that axis's 1-D primal operator
(``mu(0) = 0``). The case's own gate test checked this separation against a
dense 3-D assembly of the same operators on a small graded box; that test left
with the case.
"""

from __future__ import annotations

import math

import numpy as np

C0 = 299792458.0

# ---------------------------------------------------------------------------
# cv14's cavity, verbatim (that case was removed 2026-09-21)
# ---------------------------------------------------------------------------
A_X, B_Y, D_Z = 0.050, 0.030, 0.040          # metres (x, y, z)
CHANNELS = ("ex", "ey", "ez")

# Gated band: every closed-form mode of the cavity below 8.5 GHz -- cv14's
# seven target modes exactly; the next mode (TE211, 8.658 GHz) is outside.
BAND_HZ = (4.0e9, 8.5e9)

# Transition chain 0.5 -> 0.7 -> 0.8 -> 1.0 mm: ratios 1.4, 1.143, 1.25.
CHAIN_MM = (0.7, 0.8)


def _mm(cells) -> np.ndarray:
    return np.asarray(cells, dtype=np.float64) * 1e-3


def _profile(parts) -> np.ndarray:
    out = []
    for size_mm, n in parts:
        out.extend([size_mm] * int(n))
    p = _mm(out)
    return p


UP = [(CHAIN_MM[1], 1), (CHAIN_MM[0], 1)]      # coarse -> fine  (0.8, 0.7)
DOWN = [(CHAIN_MM[0], 1), (CHAIN_MM[1], 1)]    # fine -> coarse  (0.7, 0.8)

PROFILES: dict[str, np.ndarray] = {
    # (a) cv14's mesh (the control): 40 x 1.0 mm
    "uniform": _profile([(1.0, 40)]),
    # (b) single fine band, off-centre: coarse 8 | 0.8 0.7 | fine 10 | 0.7 0.8 | coarse 24
    "single_band": _profile([(1.0, 8)] + UP + [(0.5, 10)] + DOWN + [(1.0, 24)]),
    # (c) small-large-small-large: fine 6 | chain | coarse 10 | chain | fine 9 | chain | coarse 18
    "multi_band": _profile([(0.5, 6)] + DOWN + [(1.0, 10)] + UP + [(0.5, 9)] + DOWN + [(1.0, 18)]),
    # (d) uniform-FINE at the graded arms' finest cell (cost control, #810)
    "uniform_fine": _profile([(0.5, 80)]),
}


# ---------------------------------------------------------------------------
# Pozar spectrum and mode identification
# ---------------------------------------------------------------------------


def pozar_freq(m: int, n: int, l: int, a: float = A_X, b: float = B_Y, d: float = D_Z) -> float:
    """f_mnl = (c/2) sqrt((m/a)^2 + (n/b)^2 + (l/d)^2)  [Pozar, re-derived]."""
    return (C0 / 2.0) * math.sqrt((m / a) ** 2 + (n / b) ** 2 + (l / d) ** 2)


def mode_name(m: int, n: int, l: int) -> str:
    """cv14's naming: TM for l = 0 and for the (TE/TM-degenerate) all-nonzero
    triples, TE when exactly one of m, n is zero."""
    fam = "TM" if (l == 0 or (m and n and l)) else "TE"
    return f"{fam}{m}{n}{l}"


def declared_modes(band_hz=BAND_HZ, max_index: int = 6, a: float = A_X, b: float = B_Y,
                   d: float = D_Z) -> list[dict]:
    """Every closed-form mode of the cavity in ``band_hz``, sorted by
    frequency. A triple with two zero indices is not a mode. All-nonzero
    triples are TE/TM-degenerate (one frequency, counted once)."""
    out = []
    for m in range(max_index + 1):
        for n in range(max_index + 1):
            for l in range(max_index + 1):
                if sum(1 for i in (m, n, l) if i == 0) >= 2:
                    continue
                f = pozar_freq(m, n, l, a, b, d)
                if band_hz[0] <= f <= band_hz[1]:
                    out.append({"name": mode_name(m, n, l), "mnl": (m, n, l), "f_hz": f,
                                "degenerate": bool(m and n and l),
                                "kz2_share": ((l / d) ** 2) / ((m / a) ** 2 + (n / b) ** 2 + (l / d) ** 2)})
    out.sort(key=lambda r: r["f_hz"])
    return out


def next_mode_above(band_hz=BAND_HZ, max_index: int = 6) -> dict:
    hi = declared_modes((band_hz[1], 3 * band_hz[1]), max_index)
    return hi[0]


def id_windows(modes: list[dict], band_hz=BAND_HZ) -> list[tuple[float, float]]:
    """Voronoi windows in frequency: each declared mode owns the interval up
    to the midpoint to its neighbours (the band edge below the first mode,
    the first mode ABOVE the band for the last). NOT argmin-nearest: a line
    outside every window belongs to no mode."""
    fs = [m["f_hz"] for m in modes]
    above = next_mode_above(band_hz)["f_hz"]
    wins = []
    for i, f in enumerate(fs):
        lo = band_hz[0] if i == 0 else 0.5 * (fs[i - 1] + f)
        hi = 0.5 * (f + (fs[i + 1] if i + 1 < len(fs) else above))
        wins.append((lo, hi))
    return wins


def identify_modes(lines: list[dict], modes: list[dict], band_hz=BAND_HZ,
                   amp_floor_rel: float | None = None) -> dict:
    """Assign harminv lines to declared modes by index.

    ``lines``: [{"f_hz", "amp", "channel", "error"}] from every channel.
    Lines below ``amp_floor_rel`` x the strongest line of their channel are
    dropped (the estimator's own rank floor, ``AMP_FLOOR_REL``). Lines are
    clustered across channels within ``CLUSTER_REL``; each cluster falls in at
    most one Voronoi window. Returns per-mode measured frequency (amplitude-
    weighted cluster mean), the cluster count in the band, and the orphans.
    """
    floor = AMP_FLOOR_REL if amp_floor_rel is None else amp_floor_rel
    keep = []
    by_ch: dict[str, float] = {}
    for ln in lines:
        by_ch[ln["channel"]] = max(by_ch.get(ln["channel"], 0.0), float(ln["amp"]))
    for ln in lines:
        if float(ln["amp"]) >= floor * by_ch[ln["channel"]] and band_hz[0] <= ln["f_hz"] <= band_hz[1]:
            keep.append(ln)
    keep.sort(key=lambda r: r["f_hz"])
    clusters: list[list[dict]] = []
    for ln in keep:
        if clusters and abs(ln["f_hz"] - clusters[-1][-1]["f_hz"]) <= CLUSTER_REL * ln["f_hz"]:
            clusters[-1].append(ln)
        else:
            clusters.append([ln])
    wins = id_windows(modes, band_hz)
    per_mode = {m["name"]: None for m in modes}
    orphans = []
    ambiguous = []
    for cl in clusters:
        amps = np.array([c["amp"] for c in cl])
        f = float(np.sum(amps * np.array([c["f_hz"] for c in cl])) / amps.sum())
        home = [m["name"] for m, (lo, hi) in zip(modes, wins) if lo <= f < hi]
        rec = {"f_hz": f, "n_lines": len(cl), "channels": sorted({c["channel"] for c in cl}),
               "amp_max": float(amps.max()), "error_max": float(max(c.get("error", 0.0) for c in cl))}
        if not home:
            orphans.append(rec)
        elif per_mode[home[0]] is not None:
            ambiguous.append((home[0], rec))
        else:
            per_mode[home[0]] = rec
    return {"per_mode": per_mode, "n_clusters_in_band": len(clusters), "orphans": orphans,
            "ambiguous": [(n, r) for n, r in ambiguous], "n_declared": len(modes)}


# rfx/harminv.py: sv_threshold = 1e-3 (rank floor) -- a line below 1e-3 of the
# channel's strongest is below the estimator's own model order; the dedup
# merges lines within 1 % (its ``< 0.01`` relative test), so cross-channel
# clustering uses a quarter of the closest declared pair instead.
AMP_FLOOR_REL = 1e-3
CLUSTER_REL = 0.005     # < (142.7 MHz / 7.0 GHz) / 4 = 0.51 %


# ---------------------------------------------------------------------------
# The exact Yee lattice on a tensor grid (rfx/core/yee.py update_h_nu /
# update_e_nu; metrics from rfx/nonuniform.py _profile_to_inv_arrays and the
# #562 bounding node _append_bounding_node)
# ---------------------------------------------------------------------------


def padded_profile(profile) -> np.ndarray:
    d = np.asarray(profile, dtype=np.float64)
    return np.concatenate([d, d[-1:]])


def inv_arrays(profile) -> tuple[np.ndarray, np.ndarray]:
    """float64 mirror of ``rfx.nonuniform._profile_to_inv_arrays`` on the
    padded profile: (inv_e, inv_h); inv_h[-1] = 0, inv_e[0] = 1/d[0]."""
    d = padded_profile(profile)
    inv_local = 1.0 / d
    inv_h = np.concatenate([inv_local[:-1], [0.0]])
    inv_e = np.concatenate([inv_local[:1], 2.0 / (d[:-1] + d[1:])])
    return inv_e, inv_h


def swapped_inv_arrays(profile) -> tuple[np.ndarray, np.ndarray]:
    """The pre-CORE-C2 defect (rfx/nonuniform.py _profile_to_inv_arrays
    docstring): H given the MEAN spacing, E given the LOCAL width -- the
    curl scaled by 2 d[k] / (d[k] + d[k +- 1]) on every graded cell. Used
    only to PREDICT the metric-defect falsifier (F1c)."""
    inv_e, inv_h = inv_arrays(profile)
    d = padded_profile(profile)
    inv_local = 1.0 / d
    inv_mean = np.concatenate([inv_local[:1], 2.0 / (d[:-1] + d[1:])])
    inv_h_bad = np.concatenate([inv_mean[:-1], [0.0]])      # H gets the mean spacing
    inv_e_bad = inv_local.copy()                             # E gets the local width
    return inv_e_bad, inv_h_bad


def operator_matrix(profile, swap_metrics: bool = False) -> np.ndarray:
    """1-D primal operator on the interior nodes 1..N-1 (Dirichlet ends),
    similarity-transformed to the symmetric S = D^{1/2} T D^{1/2}, D =
    diag(inv_e). With ``swap_metrics`` the defective pair is used (the
    operator is still of the form D^{-1}-weighted T, so still symmetrizable)."""
    inv_e, inv_h = swapped_inv_arrays(profile) if swap_metrics else inv_arrays(profile)
    n_cells = len(np.asarray(profile))
    ks = np.arange(1, n_cells)
    s = np.sqrt(inv_e[ks])
    diag = s * (inv_h[ks] + inv_h[ks - 1]) * s
    off = -inv_h[ks[:-1]] * s[:-1] * s[1:]
    return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)


def operator_eigenvalues(profile, n_modes: int, swap_metrics: bool = False) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(operator_matrix(profile, swap_metrics)))[:n_modes]


def uniform_mu(h: float, n_cells: int, i: int) -> float:
    """Closed form on a uniform axis: (2/h sin(i pi / (2 N)))^2."""
    return (2.0 / h * math.sin(i * math.pi / (2.0 * n_cells))) ** 2


def axis_mu(profile, i: int, swap_metrics: bool = False) -> float:
    if i == 0:
        return 0.0
    return float(operator_eigenvalues(profile, i, swap_metrics)[i - 1])


def cfl_dt(dx_min: float, dy_min: float, dz_min: float) -> float:
    """rfx.nonuniform.make_nonuniform_grid: 0.99 / (c0 sqrt(sum 1/d_min^2));
    identical to rfx.grid.Grid's 0.99 dx / (c0 sqrt(3)) on a uniform mesh."""
    return 0.99 / (C0 * math.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))


def leapfrog_freq(mu_sum: float, dt: float) -> float:
    arg = C0 * dt * math.sqrt(mu_sum) / 2.0
    if not 0.0 < arg < 1.0:
        raise ValueError(f"non-propagating / unstable argument {arg}")
    return math.asin(arg) / (math.pi * dt)


def lattice_freq(mnl: tuple[int, int, int], dz_profile, dxy: float, dt: float | None = None,
                 a: float = A_X, b: float = B_Y, swap_metrics: bool = False) -> dict:
    """Exact discrete eigenfrequency of the mode (m, n, l) on the tensor grid
    with uniform x/y cells ``dxy`` and the z profile; ``dt`` defaults to the
    grid's own CFL step. Returns the dt-free spatial frequency too."""
    m, n, l = mnl
    dz = np.asarray(dz_profile, dtype=np.float64)
    nx = int(round(a / dxy))
    ny = int(round(b / dxy))
    mu_x = uniform_mu(dxy, nx, m)
    mu_y = uniform_mu(dxy, ny, n)
    mu_z = axis_mu(dz, l, swap_metrics)
    if dt is None:
        dt = cfl_dt(dxy, dxy, float(dz.min()))
    f_sp = C0 * math.sqrt(mu_x + mu_y + mu_z) / (2 * math.pi)
    f_lf = leapfrog_freq(mu_x + mu_y + mu_z, dt)
    f_ex = pozar_freq(m, n, l, a, b, float(dz.sum()))
    return {"f_lattice_hz": f_lf, "f_spatial_hz": f_sp, "f_exact_hz": f_ex, "dt": dt,
            "mu_x": mu_x, "mu_y": mu_y, "mu_z": mu_z,
            "dev_lattice": f_lf / f_ex - 1.0, "dev_spatial": f_sp / f_ex - 1.0,
            "dev_time": f_lf / f_sp - 1.0}


# Committed anchors for the estimator floor: tests/oracle/test_nonuniform_cavity_accuracy.py
# (a = 40, b = 35 mm; TM111; dx = 1 mm; the docstring table) --
#   uniform z (40 x 1 mm)                -> 0.0011 % measured
#   4:1 graded z (0.25 mm band, smoothed 1.3) -> 0.0252 % measured (_MEASURED_ENVELOPE_PCT)
# The lattice model is evaluated on those exact grids (anchor_residuals) and
# the estimator floor is 1.5 x the worst |measured - lattice|, rounded up at
# 1e-6 (tests/_gate_policy.py rule, quantum 1e6).
ANCHOR_A, ANCHOR_B = 40e-3, 35e-3
ANCHOR_UNIFORM_PCT = 0.0011
ANCHOR_GRADED_PCT = 0.0252
ANCHOR_QUOTE_RESOLUTION = 0.5e-6        # both data are quoted to 1e-4 % = 1e-6


def _smooth_grading_1p3(cells, max_ratio: float = 1.3) -> np.ndarray:
    """Pure-numpy re-statement of ``rfx.auto_config.smooth_grading`` (no
    preserve_regions) for the anchor profile; the gate test asserts it
    equals the real one cell for cell."""
    cells = [float(c) for c in cells]
    smoothed = [cells[0]]
    for i in range(1, len(cells)):
        prev = smoothed[-1]
        target = cells[i]
        while target / prev > max_ratio + 1e-12:
            prev = prev * max_ratio
            smoothed.append(prev)
        while prev / target > max_ratio + 1e-12:
            prev = prev / max_ratio
            smoothed.append(prev)
        smoothed.append(target)
    return np.asarray(smoothed, dtype=np.float64)


def anchor_profile_graded(smoother=None) -> np.ndarray:
    fine, dx = 0.25e-3, 1e-3
    raw = [dx] * 17 + [fine] * 8 + [dx] * 17
    return np.asarray((smoother or _smooth_grading_1p3)(raw), dtype=np.float64)


def anchor_residuals(graded_profile=None) -> dict:
    """|measured - lattice| for the two committed TM111 anchors."""
    out = {}
    uni = np.full(40, 1e-3)
    grd = anchor_profile_graded() if graded_profile is None else np.asarray(graded_profile)
    for key, prof, meas_pct in (("uniform", uni, ANCHOR_UNIFORM_PCT), ("graded_4to1", grd, ANCHOR_GRADED_PCT)):
        d = float(prof.sum())
        r = lattice_freq((1, 1, 1), prof, 1e-3, a=ANCHOR_A, b=ANCHOR_B)
        f_ex = pozar_freq(1, 1, 1, ANCHOR_A, ANCHOR_B, d)
        dev_model = r["f_lattice_hz"] / f_ex - 1.0
        out[key] = {"dev_model": dev_model, "dev_measured_abs": meas_pct / 100.0,
                    "residual": abs(abs(dev_model) - meas_pct / 100.0), "d_m": d,
                    "n_cells": int(len(prof)), "dt": r["dt"]}
    return out


ENVELOPE_GATE_MULTIPLIER = 1.5   # tests/_gate_policy.py (the gate test asserts equality)


def gate_from_envelope(measured_envelope: float, *, quantum: float) -> float:
    return math.ceil(measured_envelope * ENVELOPE_GATE_MULTIPLIER * quantum) / quantum


def estimator_floor() -> float:
    """W_est: the lattice-vs-measurement floor from the committed anchors."""
    res = anchor_residuals()
    worst = max(max(v["residual"], ANCHOR_QUOTE_RESOLUTION) for v in res.values())
    return gate_from_envelope(worst, quantum=1e6)

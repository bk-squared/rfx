"""cv24 -- non-uniform (graded-z) rectangular PEC cavity vs the exact Pozar
spectrum: profiles, windows, allowance, exact-lattice prediction, mode
identification, record length and gates. Pure numpy; no rfx import.

Pre-declaration (read it first):
  docs/design_notes/20260902_cv24_nu_cavity_predeclaration.md

Every number the case script and its gate test use lives HERE, once. The
FDTD never enters this module: the exact-lattice prediction is a
frequency-domain eigenvalue computation on the SAME difference operators
``rfx.nonuniform._profile_to_inv_arrays`` builds (mirrored below in float64,
``inv_arrays``), the allowance is arithmetic on #785's frozen F-S2 table, and
the windows are derived from committed data and this model, before any arm
runs.

Model class of ``lattice_freq``: first principles (SPEC-00 0.2-2). For an
empty PEC box on a tensor Yee grid the discrete curl-curl operator separates
per axis (the per-component 1-D operators along the two "node" axes are the
primal Dirichlet operator ``A`` and along the "edge" axis the dual operator
``B = -inv_h . D_e^{-1} . inv_h^T``; ``A = -D_e^{-1} inv_h^T inv_h`` and ``B``
are ``XY`` and ``YX`` and therefore share their nonzero spectrum), so every
discrete eigenfrequency is

    sin(omega dt / 2) = (c0 dt / 2) sqrt(mu_x(m) + mu_y(n) + mu_z(l))

with ``mu_axis(i)`` the i-th eigenvalue of that axis's 1-D primal operator
(``mu(0) = 0``). ``tests/test_cv24_nu_cavity_gates.py`` checks this
separation against a dense 3-D assembly of the same operators on a small
graded box, so the separable formula is not taken on trust.
"""

from __future__ import annotations

import math

import numpy as np

C0 = 299792458.0
SCHEMA = "rfx.cv24_nu_rect_cavity_pozar.v1"
CASE_ID = "24_nu_rect_cavity_pozar"
RESULTS_DIRNAME = "_24_nu_cavity_results"

# ---------------------------------------------------------------------------
# cv14's cavity, verbatim (validation/crossval/14_rect_cavity_pozar.py)
# ---------------------------------------------------------------------------
A_X, B_Y, D_Z = 0.050, 0.030, 0.040          # metres (x, y, z)
DX_COARSE = 1.0e-3                            # cv14's cell (exact divisor)
DZ_FINE = 0.5e-3                              # the graded arms' finest cell
FREQ_MAX = 10e9                               # cv14: f0 = 5 GHz, bw 0.8
SRC_F0_HZ = FREQ_MAX / 2
SRC_BW = 0.8
SRC_CUTOFF = 3.0                              # GaussianPulse default
# cv14's sources / probe (absolute metres). Every z sits on a locally UNIFORM
# node of every profile below (no source/probe on a grading transition).
SOURCES = (((0.013, 0.011, 0.017), "ex"),
           ((0.019, 0.023, 0.013), "ey"),
           ((0.031, 0.013, 0.023), "ez"))
PROBE = (0.037, 0.017, 0.029)
CHANNELS = ("ex", "ey", "ez")

# Gated band: every closed-form mode of the cavity below 8.5 GHz -- cv14's
# seven target modes exactly; the next mode (TE211, 8.658 GHz) is outside.
BAND_HZ = (4.0e9, 8.5e9)

# ---------------------------------------------------------------------------
# #785 envelope (docs/design_notes/20260829_spec01_multiband_predeclaration.md,
# rfx/api/_preflight.py _MULTIBAND_RATIO_CAP): z only, ratio <= 1.4, <= 3
# fine bands, no face-adjacent grading. The battery had absorbing faces;
# this case has PEC walls and inherits the exclusion as a declared runway.
# ---------------------------------------------------------------------------
RATIO_CAP = 1.4
MAX_FINE_BANDS = 3
R_WALL_CELLS = 4        # uniform cells required against each PEC wall
EXTENT_TOL_M = 1e-9     # realized z extent must equal the declared profile sum

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

ARMS: dict[str, dict] = {
    "uniform": {"lane": "uniform", "dx": DX_COARSE, "profile": "uniform"},
    "single_band": {"lane": "nonuniform", "dx": DX_COARSE, "profile": "single_band"},
    "multi_band": {"lane": "nonuniform", "dx": DX_COARSE, "profile": "multi_band"},
    "uniform_fine": {"lane": "uniform", "dx": DZ_FINE, "profile": "uniform_fine"},
}
ARM_ORDER = ("uniform", "single_band", "multi_band", "uniform_fine")
GRADED_ARMS = ("single_band", "multi_band")

# ---------------------------------------------------------------------------
# Falsifier profiles (note section 6). Each is run as a graded arm judged
# against the UNIFORM control and the DECLARED cavity.
# ---------------------------------------------------------------------------
FALSIFIER_PROFILES: dict[str, np.ndarray] = {
    # F1  same bands as (b), ABRUPT 0.5 <-> 1.0 transitions (ratio 2.0)
    "ratio2_abrupt": _profile([(1.0, 9), (0.5, 10), (1.0, 26)]),
    # F2  (b)'s bands with the grading chain starting AT the z = 0 wall
    "grading_at_wall": _profile(UP + [(0.5, 10)] + DOWN + [(1.0, 32)]),
    # F3  (b) with ONE extra fine cell: realized d = 40.5 mm, oracle keeps 40.0
    "extent_plus_one_fine_cell": _profile([(1.0, 8)] + UP + [(0.5, 11)] + DOWN + [(1.0, 24)]),
}
FALSIFIERS: dict[str, dict] = {
    "ratio2_abrupt": {"kind": "profile", "profile": "ratio2_abrupt",
                      "expect": ("envelope",),
                      "note": "envelope ratio 2.0; physics excess vs allowance predicted in the note"},
    # F1c the pre-CORE-C2 metric swap injected into the solver for arm (b)
    "metric_defect": {"kind": "metric_swap", "profile": "single_band",
                      "expect": ("allowance", "lattice"),
                      "note": "H given the mean spacing, E the local width (rfx/nonuniform.py "
                              "_profile_to_inv_arrays docstring); predicted with swap_metrics=True"},
    "grading_at_wall": {"kind": "profile", "profile": "grading_at_wall",
                        "expect": ("envelope",),
                        "note": "transition chain within R_WALL_CELLS of the z=0 wall"},
    # F3 the RUN uses the 47-cell profile; the DECLARED profile (oracle, lattice
    #    prediction, allowance) stays (b): the geometry is mis-realized, not re-declared
    "extent_plus_one_fine_cell": {"kind": "profile", "profile": "extent_plus_one_fine_cell",
                                  "declared_profile": "single_band",
                                  "expect": ("lattice", "allowance", "extent"),
                                  "note": "l>=1 modes fail BY NAME; l=0 modes pass; realized d = 40.5 mm"},
    # F4  search band deliberately closed below TE102 (8.072 GHz): count 6 != 7
    "mode_count_drop_te102": {"kind": "search_band", "profile": "single_band",
                              "search_band_hz": (BAND_HZ[0], 8.0e9),
                              "expect": ("mode_count",),
                              "note": "the oracle still declares 7 modes in the band"},
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


def closest_pair_hz(modes: list[dict]) -> tuple[float, str, str]:
    fs = [m["f_hz"] for m in modes]
    best = None
    for i in range(len(fs) - 1):
        d = fs[i + 1] - fs[i]
        if best is None or d < best[0]:
            best = (d, modes[i]["name"], modes[i + 1]["name"])
    return best


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


def spatial_freq(f_measured: float, dt: float) -> float:
    """Invert the leapfrog relation exactly: the frequency the same spatial
    eigenvalue would have at dt -> 0 (equal-dt comparison, #810)."""
    k_disc = (2.0 / (C0 * dt)) * math.sin(math.pi * f_measured * dt)
    return C0 * k_disc / (2.0 * math.pi)


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


def second_order_mu(profile, k: float) -> dict:
    """Second-order perturbative eigenvalue of the 1-D primal operator for the
    continuum eigenfunction sin(k z) (note section 3):

        mu = k^2 - (k^4/12) <h~^2>_w + (k^3/3) <dh . sin cos>_w / <sin^2>_w

    with h~^2 = h_k^2 - h_k h_{k-1} + h_{k-1}^2, dh = h_k - h_{k-1} at node k,
    weights w_k = (h_{k-1} + h_k)/2 (the dual spacing, the metric A is
    self-adjoint in). Uniform mesh: mu = k^2 (1 - k^2 h^2 / 12) + O(h^4).
    """
    h = np.asarray(profile, dtype=np.float64)
    z = np.concatenate([[0.0], np.cumsum(h)])
    ks = np.arange(1, len(h))          # interior nodes
    hk, hkm = h[ks], h[ks - 1]
    w = 0.5 * (hk + hkm)
    s, c = np.sin(k * z[ks]), np.cos(k * z[ks])
    norm = np.sum(w * s * s)
    t_disp = -(k ** 4 / 12.0) * np.sum(w * (hk * hk - hk * hkm + hkm * hkm) * s * s) / norm
    t_trans = (k ** 3 / 3.0) * np.sum(w * (hk - hkm) * s * c) / norm
    return {"mu": k * k + t_disp + t_trans, "term_dispersion": float(t_disp),
            "term_transition": float(t_trans)}


# ---------------------------------------------------------------------------
# Allowance from #785's F-S2 per-transition reflection budget
# ---------------------------------------------------------------------------
# Frozen chain-model amplitudes at the battery's reference resolution
# (design note 20260829 section 3.3, abrupt column; R_model, not the 3x window):
FS2_R_MODEL = {1.1: 4.358e-4, 1.2: 9.139e-4, 1.4: 1.998e-3, 1.5: 2.605e-3, 2.0: 6.298e-3}
FS2_R_MEASURED_DB_AT_1P4 = -53.9           # PR #785, measured
FS2_REF_CELLS_PER_AXIAL_WAVELENGTH = 34.6  # "lambda_g / 34.6": the chain model's variable
FS2_REF_CELLS_PER_FREE_WAVELENGTH = 30.0   # the same point quoted in lambda_0
FS2_SCALING_EXPONENT = 2                    # (dz/lambda)^2, -12 dB per doubling


def fs2_reflection(ratio: float, cells_per_wavelength: float,
                   ref_cells_per_wavelength: float = FS2_REF_CELLS_PER_AXIAL_WAVELENGTH) -> float:
    """Amplitude reflection of ONE abrupt step of adjacent-cell ratio
    ``ratio`` at ``cells_per_wavelength`` (fine cell per wavelength; axial
    lambda_z against the 34.6 reference by default, free-space lambda_0
    against the 30 reference when asked), from the frozen table with the
    (dz/lambda)^2 law.
    Log-linear interpolation in the table; below r = 1.1 the reflection is
    linear in (r - 1) (the small-step limit of the chain model)."""
    r = float(ratio)
    if r <= 1.0:
        return 0.0
    keys = sorted(FS2_R_MODEL)
    if r < keys[0]:
        base = FS2_R_MODEL[keys[0]] * (r - 1.0) / (keys[0] - 1.0)
    elif r >= keys[-1]:
        base = FS2_R_MODEL[keys[-1]] * ((r - 1.0) / (keys[-1] - 1.0))
    else:
        for lo, hi in zip(keys[:-1], keys[1:]):
            if lo <= r <= hi:
                t = (r - lo) / (hi - lo)
                base = math.exp((1 - t) * math.log(FS2_R_MODEL[lo]) + t * math.log(FS2_R_MODEL[hi]))
                break
    return base * (ref_cells_per_wavelength / cells_per_wavelength) ** FS2_SCALING_EXPONENT


def transitions(profile) -> list[dict]:
    """Every adjacent unequal pair as one step: (index, ratio >= 1, fine cell)."""
    h = np.asarray(profile, dtype=np.float64)
    out = []
    for k in range(1, len(h)):
        if h[k] != h[k - 1]:
            out.append({"node": k, "ratio": float(max(h[k], h[k - 1]) / min(h[k], h[k - 1])),
                        "fine_cell": float(min(h[k], h[k - 1]))})
    return out


def allowance(mnl: tuple[int, int, int], profile, d: float = D_Z) -> dict:
    """Per-mode eigenfrequency allowance from the per-step reflections.

    A thin lossless scatterer of amplitude reflection rho at z_t in a 1-D
    cavity of length d (mode l) shifts its eigenfrequency by
    |df/f| = (2 rho / (l pi)) sin^2(k z_t) <= 2 rho / (l pi) (first-order
    energy perturbation of a thin slab, rho = (eps-1) k delta / 2). In 3-D
    only the z part of K^2 is perturbed: multiply by kz^2 / K^2. Steps add
    in amplitude (coherent worst case). l = 0 modes carry no allowance.
    Cells per axial wavelength use lambda_z = 2 d / l (the chain model's
    variable); the free-space alternative is reported for comparison.
    """
    m, n, l = mnl
    steps = transitions(profile)
    if l == 0 or not steps:
        return {"allowance": 0.0, "allowance_free_space": 0.0, "rho_steps": [], "n_steps": len(steps)}
    kz2 = (l / d) ** 2
    share = kz2 / ((m / A_X) ** 2 + (n / B_Y) ** 2 + kz2)
    lam_z = 2.0 * d / l
    f = pozar_freq(m, n, l, A_X, B_Y, d)
    lam_0 = C0 / f
    rho, rho_free = [], []
    for st in steps:
        n_ax = lam_z / st["fine_cell"]
        n_fr = lam_0 / st["fine_cell"]
        rho.append(fs2_reflection(st["ratio"], n_ax))
        rho_free.append(fs2_reflection(st["ratio"], n_fr, FS2_REF_CELLS_PER_FREE_WAVELENGTH))
    a_ax = share * (2.0 / (l * math.pi)) * float(np.sum(rho))
    a_fr = share * (2.0 / (l * math.pi)) * float(np.sum(rho_free))
    return {"allowance": a_ax, "allowance_free_space": a_fr, "rho_steps": rho, "n_steps": len(steps),
            "kz2_share": share, "lambda_z_m": lam_z}


# ---------------------------------------------------------------------------
# Envelope check (#785, transplanted to PEC walls)
# ---------------------------------------------------------------------------


def runs(profile) -> list[tuple[float, int]]:
    h = np.asarray(profile, dtype=np.float64)
    out: list[tuple[float, int]] = []
    for v in h:
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + 1)
        else:
            out.append((float(v), 1))
    return out


def envelope_check(profile, ratio_cap: float = RATIO_CAP, max_fine_bands: int = MAX_FINE_BANDS,
                   wall_runway: int = R_WALL_CELLS, d: float = D_Z) -> dict:
    """Violations of the declared envelope. A 'fine band' is a maximal run of
    the profile's minimum cell; a transition is any adjacent unequal pair."""
    h = np.asarray(profile, dtype=np.float64)
    viol = []
    ratios = [max(h[k], h[k - 1]) / min(h[k], h[k - 1]) for k in range(1, len(h))]
    r_max = max(ratios) if ratios else 1.0
    if r_max > ratio_cap + 1e-12:
        viol.append(f"ratio {r_max:.3f} > cap {ratio_cap}")
    rr = runs(h)
    n_fine = sum(1 for v, n in rr if v == h.min()) if r_max > 1.0 else 0
    if n_fine > max_fine_bands:
        viol.append(f"{n_fine} fine bands > {max_fine_bands}")
    if len(h) >= 2 * wall_runway:
        if np.any(h[:wall_runway] != h[0]):
            viol.append(f"grading within {wall_runway} cells of the z=0 wall")
        if np.any(h[-wall_runway:] != h[-1]):
            viol.append(f"grading within {wall_runway} cells of the z=d wall")
    extent = float(h.sum())
    if abs(extent - d) > 1e-9:
        viol.append(f"profile extent {extent*1e3:.4f} mm != declared d {d*1e3:.4f} mm")
    return {"ok": not viol, "violations": viol, "max_ratio": float(r_max), "n_fine_bands": int(n_fine),
            "n_transitions": len(transitions(h)), "n_cells": int(len(h)), "extent_m": extent}


# ---------------------------------------------------------------------------
# Record length (note section 5) and windows (section 4)
# ---------------------------------------------------------------------------
HARMINV_PENCIL_PARAMETER = 0.33      # rfx/harminv.py default L = 0.33 N
PENCIL_RESOLUTION_UNITS = 3.0        # pencil span L dt >= 1/df_min  ->  T >= 3/df_min
STATIONARITY_FRACTION = 2.0 / 3.0    # the two overlapping witness sub-windows
HARMINV_MAX_SAMPLES = 8000           # cv14's subsampling cap


def pulse_end_s() -> float:
    """cv14: start the record at 2 t0 = 2 x cutoff x tau of the default
    GaussianPulse (f0 = freq_max/2, bw 0.8, cutoff 3)."""
    tau = 1.0 / (math.pi * SRC_F0_HZ * SRC_BW)
    return 2.0 * SRC_CUTOFF * tau


def derive_record(dt: float, modes: list[dict] | None = None) -> dict:
    """Post-pulse record long enough that the closest declared pair sits at
    >= PENCIL_RESOLUTION_UNITS pencil-resolution units in EACH stationarity
    sub-window of length STATIONARITY_FRACTION x T_post:

        T_post = PENCIL_RESOLUTION_UNITS / (STATIONARITY_FRACTION df_min)
    """
    modes = declared_modes() if modes is None else modes
    df_min, m1, m2 = closest_pair_hz(modes)
    t_post = PENCIL_RESOLUTION_UNITS / (STATIONARITY_FRACTION * df_min)
    t_start = pulse_end_s()
    n_start = int(math.ceil(t_start / dt))
    n_steps = n_start + int(math.ceil(t_post / dt))
    return {"df_min_hz": df_min, "closest_pair": (m1, m2), "t_post_s": t_post, "t_start_s": t_start,
            "n_start": n_start, "n_steps": n_steps, "dt": dt,
            "n_sub": int(math.ceil(STATIONARITY_FRACTION * t_post / dt)),
            "pair_units_full": df_min * t_post * HARMINV_PENCIL_PARAMETER * 3.0,
            "pair_units_sub": df_min * t_post * STATIONARITY_FRACTION * HARMINV_PENCIL_PARAMETER * 3.0}


# Committed anchors for the estimator floor: tests/test_nonuniform_cavity_accuracy.py
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


# cv14's committed claims tolerance (validation/crossval/14_rect_cavity_pozar.py
# gate 1 / gate 2), unchanged on every arm.
CV14_TOL_TE101 = 0.01
CV14_TOL_HIGHER = 0.02
# #785 F-S1 energy envelope (validation/research/multiband_nu/w1_energy_drift.py)
FS1_K = 20.0
U32 = 2.0 ** -24
FS1_MIN_N = 1e4


def fs1_envelope(n_steps: int) -> float:
    return FS1_K * U32 * math.sqrt(float(n_steps))


# ---------------------------------------------------------------------------
# Predictions per arm (before the run)
# ---------------------------------------------------------------------------


def predict_arm(profile, dxy: float, modes: list[dict] | None = None) -> dict:
    modes = declared_modes() if modes is None else modes
    dz = np.asarray(profile, dtype=np.float64)
    dt = cfl_dt(dxy, dxy, float(dz.min()))
    rec = derive_record(dt, modes)
    per = {}
    for md in modes:
        lat = lattice_freq(md["mnl"], dz, dxy, dt)
        l = md["mnl"][2]
        so = second_order_mu(dz, l * math.pi / float(dz.sum())) if l else {"mu": 0.0, "term_dispersion": 0.0, "term_transition": 0.0}
        per[md["name"]] = dict(lat, allowance=allowance(md["mnl"], dz), second_order_mu_z=so,
                               f_pozar_hz=md["f_hz"])
    nx, ny = int(round(A_X / dxy)), int(round(B_Y / dxy))
    return {"dt": dt, "record": rec, "modes": per, "envelope": envelope_check(dz),
            "cells": [nx, ny, int(len(dz))], "nodes": [nx + 1, ny + 1, int(len(dz)) + 1],
            "n_cells_total": nx * ny * int(len(dz)), "cell_steps": nx * ny * int(len(dz)) * rec["n_steps"]}


# ---------------------------------------------------------------------------
# Gates on a measured arm
# ---------------------------------------------------------------------------


def evaluate_arm(measured: dict, profile, dxy: float, dt: float, control: dict | None,
                 modes: list[dict] | None = None, search_band_hz=None,
                 w_est: float | None = None, realized_d_m: float | None = None) -> dict:
    """``measured``: {"per_mode": {name: {"f_hz", ...} | None}, "n_clusters_in_band",
    "stationarity": {name: rel_scatter}, "energy": {"max_drift", "n_end", "fs1_fired"}}.
    ``profile`` is the DECLARED profile (predictions, allowance, envelope);
    ``realized_d_m`` the grid's realized z extent (the ``extent`` gate; None
    skips it). ``control``: the uniform arm's evaluation (None for the control).
    """
    modes = declared_modes() if modes is None else modes
    w_est = estimator_floor() if w_est is None else w_est
    dz = np.asarray(profile, dtype=np.float64)
    env = envelope_check(dz)
    rows = {}
    gates = {"cv14_te101": True, "cv14_higher": True, "allowance": True, "lattice": True,
             "mode_count": True, "stationarity": True, "energy": True, "envelope": env["ok"],
             "extent": (realized_d_m is None) or abs(realized_d_m - float(dz.sum())) <= EXTENT_TOL_M}
    n_higher_ok = 0
    n_higher = 0
    for md in modes:
        name = md["name"]
        lat = lattice_freq(md["mnl"], dz, dxy, dt)
        meas = (measured.get("per_mode") or {}).get(name)
        row = {"mnl": list(md["mnl"]), "f_pozar_hz": md["f_hz"], "f_lattice_hz": lat["f_lattice_hz"],
               "f_spatial_lattice_hz": lat["f_spatial_hz"], "pred_dev_lattice": lat["dev_lattice"],
               "pred_dev_spatial": lat["dev_spatial"], "allowance": allowance(md["mnl"], dz)["allowance"],
               "found": meas is not None}
        if meas is None:
            row.update({"f_meas_hz": None, "dev_raw": None, "dev_spatial": None, "resid_lattice": None})
            gates["mode_count"] = False
            if name == "TE101":
                gates["cv14_te101"] = False
            else:
                n_higher += 1
            rows[name] = row
            continue
        f = float(meas["f_hz"])
        dev = f / md["f_hz"] - 1.0
        dev_sp = spatial_freq(f, dt) / md["f_hz"] - 1.0
        resid = f / lat["f_lattice_hz"] - 1.0
        row.update({"f_meas_hz": f, "dev_raw": dev, "dev_spatial": dev_sp, "resid_lattice": resid,
                    "channels": meas.get("channels"), "n_lines": meas.get("n_lines")})
        # G1 cv14's tolerance
        if name == "TE101":
            gates["cv14_te101"] &= abs(dev) < CV14_TOL_TE101
        else:
            n_higher += 1
            n_higher_ok += int(abs(dev) < CV14_TOL_HIGHER)
        # G3 lattice
        row["lattice_ok"] = abs(resid) <= w_est
        gates["lattice"] &= row["lattice_ok"]
        # G2 allowance (graded arms only; equal-dt = spatial)
        if control is not None:
            crow = control["rows"].get(name) or {}
            cdev = crow.get("dev_spatial")
            if cdev is None:
                row["allowance_ok"] = False
            else:
                row["allowance_bound"] = abs(cdev) + row["allowance"] + w_est
                row["allowance_ok"] = abs(dev_sp) <= row["allowance_bound"]
            gates["allowance"] &= row["allowance_ok"]
        # witnesses
        sc = (measured.get("stationarity") or {}).get(name)
        row["stationarity"] = sc
        row["stationarity_ok"] = (sc is not None) and (abs(sc) <= w_est)
        gates["stationarity"] &= row["stationarity_ok"]
        rows[name] = row
    gates["cv14_higher"] = n_higher_ok >= 1
    n_found = measured.get("n_clusters_in_band")
    gates["mode_count"] &= (n_found == len(modes))
    en = measured.get("energy") or {}
    gates["energy"] = bool(en) and (not en.get("fs1_fired", True))
    ok = all(gates.values())
    return {"rows": rows, "gates": gates, "ok": ok, "w_est": w_est, "envelope": env,
            "realized_d_m": realized_d_m, "declared_d_m": float(dz.sum()),
            "n_clusters_in_band": n_found, "n_declared": len(modes), "search_band_hz": list(search_band_hz or BAND_HZ),
            "dt": dt, "dxy": dxy, "profile_mm": (dz * 1e3).tolist()}


def falsifier_expectation(name: str, evaluation: dict) -> dict:
    """Which gates the pre-declared falsifier must have failed, and whether
    the artifact says so (the gate test replays this)."""
    spec = FALSIFIERS[name]
    fired = {g: not evaluation["gates"][g] for g in spec["expect"]}
    return {"expected_failing_gates": list(spec["expect"]), "fired": fired, "as_declared": all(fired.values())}


def rfx_json_name(falsifier: str | None, arms: str | None = None) -> str:
    if falsifier:
        return f"rfx__falsifier_{falsifier}.json"
    if arms:
        return f"rfx__arms_{arms}.json"
    return "rfx.json"

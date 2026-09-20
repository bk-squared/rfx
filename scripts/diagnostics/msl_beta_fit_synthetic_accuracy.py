#!/usr/bin/env python3
"""How accurately does the N-probe beta fit recover a KNOWN beta?

Bench test of ``rfx.probes.msl_wave_decomp`` on synthetic two-wave
voltages -- no FDTD, no solver, no fixture physics.  The voltages are
built from the model the extractor itself fits,

    V_n = alpha * exp(-j*beta*x_n) + gamma * exp(+j*beta*x_n),

at the EXACT probe layout of the committed MSL phase-referee fixture (5
planes, 11-cell spacing, dx = 50 um), with ``beta`` known by
construction.  Any difference between the fitted and the constructed
beta is the extractor's own error, because the data contains nothing
else.

Why this bench exists.  The fitted beta on that fixture sits about
+1.3 % above the closed form for the board the solver actually builds.
``_estimate_beta`` reaches its answer by scanning 41 trial values across
``beta0 * [0.65, 1.35]`` -- one node is 1.75 % of ``beta0``, comparable
to the residual being attributed -- and refining the best node with a
3-point parabola.  That arrangement had never been measured against a
known answer, so it could not be ruled in or out as a contributor.

What is measured, and against what.  The fit is called exactly the way
production calls it (``extract_msl_nprobe`` with the Hammerstad-Jensen
``beta0`` anchor computed as ``rfx/sparams/msl.py`` computes it, at the
declared board h = 254 um), swept over true beta / beta0 in [0.97,
1.03], four backward/forward amplitude ratios and four backward-wave
phases, at the nine gated frequencies of the referee band (3.0-4.5 GHz).
Two independent recoveries of beta share no code with the fit: the slope
of the unwrapped voltage phase across the planes (valid only for a pure
forward wave) and a float64 Levenberg-Marquardt solve of the same model
from a deliberately different start.

Precision is separated from algorithm by re-running the SAME production
source lines with float64/complex128 substituted for float32/complex64
(see ``widened_precision``), rather than by re-implementing the fit.

Outputs:
    scripts/diagnostics/msl_beta_fit_synthetic_accuracy/
        msl_beta_fit_synthetic_accuracy.json
    docs/crossval/figures/msl_beta_fit_bias_vs_beta.png

The figure goes to ``docs/crossval/figures/`` because ``**/*.png`` is
gitignored everywhere else; that directory is one of the negations, and a
claims-bearing curve nobody can open from a fresh clone is not evidence.

Usage::

    PYTHONPATH=. python scripts/diagnostics/msl_beta_fit_synthetic_accuracy.py
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import jax

# Script-level (NOT test-level -- see tests/contracts/test_no_module_level_x64.py):
# the float64 arm below needs x64 available.  The float32 arm still runs at
# complex64 because the production source spells that dtype explicitly; the
# script asserts that below rather than assuming it.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import rfx  # noqa: E402
from rfx.probes import msl_wave_decomp as mwd  # noqa: E402
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json"
OUT_DIR = Path(__file__).resolve().parent / "msl_beta_fit_synthetic_accuracy"
FIG_DIR = REPO_ROOT / "docs" / "crossval" / "figures"

# The referee's gate band (validation/crossval/20_msl_phase_referee.py).
GATE_F_LO_HZ = 3.0e9
GATE_F_HI_HZ = 4.5e9
# Declared board -- what production anchors beta0 on (add_msl_port height).
EPS_R = 3.66
H_DECLARED_M = 254e-6
W_TRACE_M = 600e-6

# Sweep, as pre-declared on the issue.
RATIO_LO, RATIO_HI, RATIO_STEP = 0.97, 1.03, 0.0005
GAMMA_MAGS = (0.0, 0.01, 0.05, 0.2)
GAMMA_PHASE_LABELS = ("0", "pi/2", "pi", "random")
NOISE_REL = 1e-4
SEED = 830


def rfx_provenance() -> str:
    """Which rfx this ran against, as a repo-relative path.

    The check that matters is that the import resolved inside THIS checkout
    and not to an installed copy; recording the absolute path would put the
    author's machine layout in a committed artifact, so it is relative and
    says so loudly when it is not.
    """
    path = Path(rfx.__file__).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return f"NOT THIS CHECKOUT: an installed rfx at {path.name}"

# ---------------------------------------------------------------------------
# Inputs taken from the committed fixture / from production
# ---------------------------------------------------------------------------
def load_case() -> dict:
    """Probe layout, gated frequencies and the fixture's own beta ratio."""
    fx = json.loads(FIXTURE.read_text())
    meta = fx["meta"]
    geom = fx["reference_plane_geometry"]["msl_0"]
    dx = float(meta["dx_m"])
    x0 = float(geom["probe0_x_m"])
    n_probes = int(geom["n_probes"])
    spacing_cells = int(geom["n_probe_spacing"])
    x = x0 + np.arange(n_probes, dtype=np.float64) * spacing_cells * dx

    freqs = np.asarray(fx["freqs_hz"], dtype=np.float64)
    beta_fixture = np.asarray(
        [c[0] for c in fx["beta_first_port"]], dtype=np.float64)
    band = (freqs >= GATE_F_LO_HZ) & (freqs <= GATE_F_HI_HZ)

    beta0_all = production_beta0(freqs)
    return {
        "dx_m": dx,
        "x_m": x,
        "spacing_m": float(spacing_cells * dx),
        "n_probes": n_probes,
        "freqs_hz": freqs[band],
        "beta0": beta0_all[band],
        "beta_fixture": beta_fixture[band],
        # The operating point: where the committed fixture's own beta sits
        # relative to the anchor the scan is centred on.
        "operating_ratio": beta_fixture[band] / beta0_all[band],
    }


def production_beta0(freqs_hz: np.ndarray) -> np.ndarray:
    """``beta0`` exactly as ``rfx/sparams/msl.py`` builds it for this port.

    Same closed-form call, same declared width/height, same c -- not a
    retyped formula.
    """
    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
    _c0 = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
    _z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(
        W_TRACE_M, H_DECLARED_M, EPS_R)
    return 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) * float(
        np.sqrt(eps_eff_hj)) / _c0


# ---------------------------------------------------------------------------
# Precision arm: run the SAME production lines wider
# ---------------------------------------------------------------------------
class _WidenedJnp:
    """``jnp`` with float32 -> float64 and complex64 -> complex128.

    Everything else passes straight through, so substituting this for the
    production module's ``jnp`` executes the identical source lines at the
    wider precision.  Re-implementing the fit would confound precision with
    a second implementation; this cannot.
    """

    _WIDER = {"complex64": "complex128", "float32": "float64"}

    def __getattr__(self, name):  # pragma: no cover - thin proxy
        return getattr(jnp, self._WIDER.get(name, name))


@contextlib.contextmanager
def widened_precision():
    original = mwd.jnp
    mwd.jnp = _WidenedJnp()
    try:
        yield
    finally:
        mwd.jnp = original


# ---------------------------------------------------------------------------
# The fit, called the way production calls it
# ---------------------------------------------------------------------------
def fit_beta(v: np.ndarray, x: np.ndarray, beta0: np.ndarray) -> dict:
    """``extract_msl_nprobe`` on (n_cases, n_probes) voltages.

    Production's per-port call shape is (n_freqs, n_probes) with a
    (n_freqs,) anchor; the case axis plays the frequency axis here, which
    is what the extractor's own ``vmap`` consumes.
    """
    res = mwd.extract_msl_nprobe(
        jnp.asarray(v), jnp.asarray(x),
        jnp.ones((v.shape[0],), dtype=jnp.complex64), jnp.asarray(beta0),
    )
    return {
        "beta": np.asarray(jax.device_get(jnp.real(res["beta"]))),
        "railed": np.asarray(jax.device_get(res["beta_railed"])),
        "residual": np.asarray(jax.device_get(res["residual"])),
        "dtype": str(res["beta"].dtype),
    }


def fit_beta_chunked(v, x, beta0, chunk=2048):
    """Fixed-shape chunks so the vmap traces once and memory stays flat."""
    n = v.shape[0]
    beta, railed, resid, dtypes = [], [], [], set()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        v_c, b_c = v[start:stop], beta0[start:stop]
        pad = chunk - (stop - start)
        if pad:
            v_c = np.concatenate([v_c, np.repeat(v_c[-1:], pad, axis=0)])
            b_c = np.concatenate([b_c, np.repeat(b_c[-1:], pad)])
        out = fit_beta(v_c, x, b_c)
        dtypes.add(out["dtype"])
        keep = chunk - pad
        beta.append(out["beta"][:keep])
        railed.append(out["railed"][:keep])
        resid.append(out["residual"][:keep])
    return (np.concatenate(beta), np.concatenate(railed),
            np.concatenate(resid), sorted(dtypes))


def synth_voltages(x, beta_true, gamma):
    """V_n = exp(-j beta x_n) + gamma exp(+j beta x_n), alpha == 1."""
    bx = beta_true[:, None] * x[None, :]
    return np.exp(-1j * bx) + gamma[:, None] * np.exp(+1j * bx)


# ---------------------------------------------------------------------------
# Independent recoveries of beta (no code shared with the fit)
# ---------------------------------------------------------------------------
def beta_from_phase_slope(v_row: np.ndarray, x: np.ndarray) -> float:
    """-d(arg V)/dx by unwrap + straight-line fit.  Pure forward wave only.

    With gamma = 0 the model is V_n = alpha exp(-j beta x_n), so arg V is
    exactly linear in x with slope -beta.  Nothing here touches the scan,
    the parabola or lstsq.
    """
    phase = np.unwrap(np.angle(np.asarray(v_row, dtype=np.complex128)))
    slope = np.polyfit(np.asarray(x, dtype=np.float64), phase, 1)[0]
    return float(-slope)


def beta_from_least_squares(v_row, x, beta_start) -> float:
    """float64 Levenberg-Marquardt on the same model, different start.

    Free parameters (Re alpha, Im alpha, Re gamma, Im gamma, beta); the
    start is deliberately off the production anchor so a shared local
    minimum cannot be mistaken for agreement.
    """
    from scipy.optimize import least_squares

    xv = np.asarray(x, dtype=np.float64)
    vv = np.asarray(v_row, dtype=np.complex128)

    def residual(p):
        a = p[0] + 1j * p[1]
        g = p[2] + 1j * p[3]
        model = a * np.exp(-1j * p[4] * xv) + g * np.exp(+1j * p[4] * xv)
        d = vv - model
        return np.concatenate([d.real, d.imag])

    sol = least_squares(residual, x0=[1.0, 0.0, 0.0, 0.0, float(beta_start)],
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return float(sol.x[4])


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def build_sweep(case, rng):
    ratios = np.round(np.arange(
        RATIO_LO, RATIO_HI + 0.5 * RATIO_STEP, RATIO_STEP), 6)
    freqs = case["freqs_hz"]
    beta0 = case["beta0"]
    rows = []
    for r in ratios:
        for mag in GAMMA_MAGS:
            for ph_label in GAMMA_PHASE_LABELS:
                for fi in range(len(freqs)):
                    if ph_label == "random":
                        phase = float(rng.uniform(0.0, 2.0 * np.pi))
                    else:
                        phase = {"0": 0.0, "pi/2": 0.5 * np.pi,
                                 "pi": np.pi}[ph_label]
                    rows.append((float(r), float(mag), ph_label, phase, fi))
    ratio = np.array([r[0] for r in rows])
    mag = np.array([r[1] for r in rows])
    ph_label = [r[2] for r in rows]
    phase = np.array([r[3] for r in rows])
    fidx = np.array([r[4] for r in rows], dtype=int)
    beta_true = ratio * beta0[fidx]
    gamma = mag * np.exp(1j * phase)
    return dict(ratio=ratio, mag=mag, ph_label=ph_label, fidx=fidx,
                beta_true=beta_true, gamma=gamma, beta0=beta0[fidx],
                ratios=ratios)


def curve_key(arm: str, mag: float) -> str:
    return f"{arm}_gamma{mag:g}_fmid"


def curve_shape(ratio: np.ndarray, bias: np.ndarray, node_step: float) -> dict:
    """Is the bias curve a sawtooth locked to the scan nodes, or smooth?

    Reports the sign changes, the spacing between successive local maxima
    and where the bias crosses zero.  A refinement artefact repeats with the
    scan-node period; a genuine model error would not.
    """
    sign = np.sign(bias)
    n_sign_changes = int(np.count_nonzero(np.diff(sign) != 0))
    peaks = [i for i in range(1, len(bias) - 1)
             if bias[i] > bias[i - 1] and bias[i] >= bias[i + 1]]
    spacing = np.diff(ratio[peaks]) if len(peaks) > 1 else np.array([np.nan])
    zeros = []
    for i in range(len(bias) - 1):
        if bias[i] == 0.0 or bias[i] * bias[i + 1] < 0.0:
            frac = abs(bias[i]) / (abs(bias[i]) + abs(bias[i + 1]) + 1e-30)
            zeros.append(round(float(ratio[i] + frac * (ratio[i + 1] - ratio[i])), 5))
    return {
        "n_sign_changes": n_sign_changes,
        "n_local_maxima": len(peaks),
        "local_maxima_ratios": [round(float(r), 5) for r in ratio[peaks]],
        "mean_peak_spacing_ratio": float(np.mean(spacing)),
        "peak_spacing_over_node_step": float(np.mean(spacing) / node_step),
        "zero_crossing_ratios": zeros,
        "peak_bias_frac": float(np.max(bias)),
        "trough_bias_frac": float(np.min(bias)),
    }


def summarize(bias, sel, label):
    b = bias[sel]
    if b.size == 0:
        return None
    return {
        "label": label,
        "n": int(b.size),
        "max_abs_bias_frac": float(np.max(np.abs(b))),
        "signed_mean_bias_frac": float(np.mean(b)),
        "std_bias_frac": float(np.std(b)),
        "min_bias_frac": float(np.min(b)),
        "max_bias_frac": float(np.max(b)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--fig-dir", default=str(FIG_DIR))
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"rfx.__file__ = {rfx.__file__}")
    case = load_case()
    print(f"probe x (m) = {case['x_m'].tolist()}")
    print(f"spacing = {case['spacing_m']:.6e} m over {case['n_probes']} planes")
    print(f"gated freqs (GHz) = {(case['freqs_hz'] / 1e9).round(6).tolist()}")
    print("fixture beta / beta0(declared h=254um) per gated bin = "
          f"{case['operating_ratio'].round(6).tolist()}")
    print(f"  mean operating ratio = {case['operating_ratio'].mean():.6f}")

    # Where the scan nodes fall, so the curve can be read against them.
    nodes = np.linspace(1.0 - mwd._BETA_SCAN_FRAC, 1.0 + mwd._BETA_SCAN_FRAC,
                        mwd._BETA_SCAN_NODES)
    node_step = float(nodes[1] - nodes[0])
    print(f"scan: {mwd._BETA_SCAN_NODES} nodes over beta0*"
          f"[{nodes[0]:.4f}, {nodes[-1]:.4f}], step = {node_step:.6f}*beta0"
          f" ({100 * node_step:.4f} %)")

    rng = np.random.default_rng(SEED)
    sw = build_sweep(case, rng)
    x = case["x_m"]
    v_clean = synth_voltages(x, sw["beta_true"], sw["gamma"])

    results = {}
    for arm, ctx in (("complex64", contextlib.nullcontext()),
                     ("complex128", widened_precision())):
        with ctx:
            beta_fit, railed, resid, dtypes = fit_beta_chunked(
                v_clean, x, sw["beta0"])
        bias = beta_fit / sw["beta_true"] - 1.0
        results[arm] = dict(bias=bias, railed=railed, dtypes=dtypes)
        print(f"[{arm}] extractor beta dtype seen: {dtypes}; "
              f"railed bins: {int(np.count_nonzero(railed))}")

    # Noise arm (complex64, the production precision).
    noise = (rng.normal(size=v_clean.shape) + 1j * rng.normal(size=v_clean.shape))
    v_noisy = v_clean + NOISE_REL * np.max(np.abs(v_clean), axis=1)[:, None] * noise
    beta_noisy, railed_noisy, _r, _d = fit_beta_chunked(
        v_noisy, x, sw["beta0"])
    results["complex64_noise"] = dict(
        bias=beta_noisy / sw["beta_true"] - 1.0, railed=railed_noisy,
        dtypes=_d)

    # ---- tables -----------------------------------------------------------
    report = {
        "rfx_file": rfx_provenance(),
        "fixture": str(FIXTURE.relative_to(REPO_ROOT)),
        "probe_layout": {
            "x_m": case["x_m"].tolist(),
            "spacing_m": case["spacing_m"],
            "n_probes": case["n_probes"],
            "dx_m": case["dx_m"],
        },
        "gated_freqs_hz": case["freqs_hz"].tolist(),
        "beta0_rad_per_m": case["beta0"].tolist(),
        "operating_ratio_per_bin": case["operating_ratio"].tolist(),
        "operating_ratio_mean": float(case["operating_ratio"].mean()),
        "scan": {
            "nodes": int(mwd._BETA_SCAN_NODES),
            "frac": float(mwd._BETA_SCAN_FRAC),
            "node_step_frac_of_beta0": node_step,
        },
        "sweep": {
            "ratio_lo": RATIO_LO, "ratio_hi": RATIO_HI,
            "ratio_step": RATIO_STEP,
            "gamma_over_alpha": list(GAMMA_MAGS),
            "gamma_phases": list(GAMMA_PHASE_LABELS),
            "noise_relative": NOISE_REL,
            "seed": SEED,
            "n_cases": int(sw["ratio"].size),
        },
        "summaries": {},
        "operating_point": {},
    }

    mag_arr = sw["mag"]
    for arm, res in results.items():
        bias = res["bias"]
        rows = [summarize(bias, np.ones_like(bias, dtype=bool), "all")]
        for m in GAMMA_MAGS:
            rows.append(summarize(bias, mag_arr == m, f"|gamma/alpha|={m}"))
        for lab in GAMMA_PHASE_LABELS:
            sel = np.array([p == lab for p in sw["ph_label"]])
            rows.append(summarize(bias, sel, f"gamma phase={lab}"))
        report["summaries"][arm] = [r for r in rows if r]
        print(f"\n=== {arm} ===")
        for r in report["summaries"][arm]:
            print(f"  {r['label']:24s} n={r['n']:6d} "
                  f"max|bias|={100 * r['max_abs_bias_frac']:+.4f} % "
                  f"mean={100 * r['signed_mean_bias_frac']:+.4f} % "
                  f"sd={100 * r['std_bias_frac']:.4f} % "
                  f"[{100 * r['min_bias_frac']:+.4f}, "
                  f"{100 * r['max_bias_frac']:+.4f}] %")

    # ---- the operating point, at the fixture's own per-bin ratio ----------
    op_ratio = case["operating_ratio"]
    op_rows = []
    for mi, m in enumerate(GAMMA_MAGS):
        gam = m * np.exp(1j * 0.0) * np.ones(len(op_ratio))
        beta_true_op = op_ratio * case["beta0"]
        v_op = synth_voltages(x, beta_true_op, gam)
        for arm, ctx in (("complex64", contextlib.nullcontext()),
                         ("complex128", widened_precision())):
            with ctx:
                out = fit_beta(v_op, x, case["beta0"])
            bias_op = out["beta"] / beta_true_op - 1.0
            op_rows.append({
                "arm": arm, "gamma_over_alpha": float(m),
                "bias_per_bin_frac": bias_op.tolist(),
                "mean_bias_frac": float(np.mean(bias_op)),
                "max_abs_bias_frac": float(np.max(np.abs(bias_op))),
            })
            if mi == 0:
                print(f"\noperating point ({arm}, gamma=0), per gated bin:")
                for fi, f in enumerate(case["freqs_hz"]):
                    print(f"  f={f / 1e9:.4f} GHz  true/beta0={op_ratio[fi]:.6f}"
                          f"  bias={100 * bias_op[fi]:+.4f} %")
    report["operating_point"]["rows"] = op_rows
    op64 = [r for r in op_rows
            if r["arm"] == "complex64" and r["gamma_over_alpha"] == 0.0][0]
    print(f"\noperating-point mean bias (complex64, gamma=0) = "
          f"{100 * op64['mean_bias_frac']:+.4f} %")

    # ---- independent witnesses -------------------------------------------
    witness_rows = []
    probe_ratios = [0.98, 1.00, float(np.round(op_ratio.mean(), 6)), 1.02]
    f_mid = len(case["freqs_hz"]) // 2
    for r in probe_ratios:
        for m in (0.0, 0.05):
            beta_true = np.array([r * case["beta0"][f_mid]])
            v_row = synth_voltages(x, beta_true, np.array([m + 0j]))
            out = fit_beta(v_row, x, case["beta0"][f_mid:f_mid + 1])
            w_ls = beta_from_least_squares(
                v_row[0], x, 1.05 * case["beta0"][f_mid])
            w_ph = (beta_from_phase_slope(v_row[0], x) if m == 0.0 else None)
            witness_rows.append({
                "ratio": r, "gamma_over_alpha": m,
                "freq_hz": float(case["freqs_hz"][f_mid]),
                "beta_true": float(beta_true[0]),
                "beta_production_fit": float(out["beta"][0]),
                "beta_least_squares_f64": w_ls,
                "beta_phase_slope": w_ph,
                "bias_production_frac": float(out["beta"][0] / beta_true[0] - 1),
                "bias_least_squares_frac": float(w_ls / beta_true[0] - 1),
                "bias_phase_slope_frac": (
                    None if w_ph is None else float(w_ph / beta_true[0] - 1)),
            })
    report["independent_witnesses"] = witness_rows
    print("\n=== independent witnesses (f = "
          f"{case['freqs_hz'][f_mid] / 1e9:.4f} GHz) ===")
    print("  ratio  |g/a|   production fit     lstsq f64 (other start)   "
          "phase slope")
    for w in witness_rows:
        ph = ("      n/a" if w["bias_phase_slope_frac"] is None
              else f"{100 * w['bias_phase_slope_frac']:+9.5f} %")
        print(f"  {w['ratio']:.4f} {w['gamma_over_alpha']:.2f}  "
              f"{100 * w['bias_production_frac']:+9.5f} %   "
              f"{100 * w['bias_least_squares_frac']:+9.5f} %          {ph}")

    # ---- curves -----------------------------------------------------------
    curves = {"ratio": sw["ratios"].tolist()}
    sel_phase0 = np.array([p == "0" for p in sw["ph_label"]])
    for arm in ("complex64", "complex128"):
        bias = results[arm]["bias"]
        for m in GAMMA_MAGS:
            sel = sel_phase0 & (mag_arr == m) & (sw["fidx"] == f_mid)
            order = np.argsort(sw["ratio"][sel])
            curves[curve_key(arm, m)] = bias[sel][order].tolist()
    bias32 = results["complex64"]["bias"]
    for fi in range(len(case["freqs_hz"])):
        sel = sel_phase0 & (mag_arr == 0.0) & (sw["fidx"] == fi)
        order = np.argsort(sw["ratio"][sel])
        curves[f"complex64_gamma0_perbin_f{fi}"] = bias32[sel][order].tolist()
    report["curves"] = curves
    report["curve_shape"] = curve_shape(
        np.asarray(curves["ratio"]),
        np.asarray(curves[curve_key("complex64", 0.0)]), node_step)
    cs = report["curve_shape"]
    print(f"\ncurve shape (complex64, gamma=0, f_mid): sign changes="
          f"{cs['n_sign_changes']}, mean spacing between local maxima="
          f"{cs['mean_peak_spacing_ratio']:.6f} vs scan-node step "
          f"{node_step:.6f} (ratio {cs['peak_spacing_over_node_step']:.4f})")
    print(f"  zeros of the bias sit at ratios {cs['zero_crossing_ratios']}")

    json_path = out_dir / "msl_beta_fit_synthetic_accuracy.json"
    json_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\nwritten {json_path}")

    png_path = fig_dir / "msl_beta_fit_bias_vs_beta.png"
    plot(curves, case, nodes, node_step, png_path)
    print(f"written {png_path}")
    return 0


def plot(curves, case, nodes, node_step, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio = np.asarray(curves["ratio"])
    op_lo, op_hi = case["operating_ratio"].min(), case["operating_ratio"].max()
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)

    ax = axes[0]
    for arm, style in (("complex64", "-"), ("complex128", "--")):
        ax.plot(ratio, 100 * np.asarray(curves[curve_key(arm, 0.0)]),
                style, lw=1.3, label=f"{arm}")
    for n in nodes:
        if ratio[0] - node_step <= n <= ratio[-1] + node_step:
            ax.axvline(n, color="0.8", lw=0.8, zorder=0)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axvspan(op_lo, op_hi, color="tab:orange", alpha=0.22, zorder=0)
    ax.set_ylabel("fitted/true - 1  [%]")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.text(0.015, 0.06,
            "grey verticals: beta-scan nodes (step 1.75 % of beta0)\n"
            "orange band: the committed fixture's own beta/beta0",
            transform=ax.transAxes, fontsize=7.5, va="bottom")

    ax = axes[1]
    for m in GAMMA_MAGS:
        ax.plot(ratio, 100 * np.asarray(curves[curve_key("complex64", m)]),
                lw=1.1, label=f"|gamma/alpha| = {m}")
    for n in nodes:
        if ratio[0] - node_step <= n <= ratio[-1] + node_step:
            ax.axvline(n, color="0.8", lw=0.8, zorder=0)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axvspan(op_lo, op_hi, color="tab:orange", alpha=0.22, zorder=0)
    ax.set_xlabel("true beta / beta0 (Hammerstad-Jensen anchor, declared board)")
    ax.set_ylabel("fitted/true - 1  [%]")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

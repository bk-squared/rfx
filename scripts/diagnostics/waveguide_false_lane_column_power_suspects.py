"""Which factor carries the `normalize=False` empty-guide column-power excess.

Recomputes, from the FROZEN chain-battery artifact
``tests/fixtures/waveguide_chain_battery/fixture.json`` and the port configs the
battery builds, the contribution each named suspect makes to the spurious
reflection the ``normalize=False`` waveguide extractor reports on a thru.

NO FDTD SOLVE RUNS HERE. ``build_simulation`` is used only to build the port
configuration (mode profiles, aperture weights, ``f_cutoff``); the measured
S-parameters come from the committed fixture.

Why a spurious reflection is the right thing to model
----------------------------------------------------
``rfx/sources/waveguide_port.py::_extract_global_waves`` forms

    a = (V + Z·I)/2 ,  b = (V − Z·I)/2 ,  S_ij = b_i / a_j

so on a purely forward wave ``b`` vanishes only when the ``Z`` the extractor uses
equals the ``V/I`` the grid presents at the sampled plane.  Any mismatch shows up
as a reflection an empty guide cannot produce, and as column power above unity:

    Γ = (Z_seen − Z_used) / (Z_seen + Z_used)

Suspects, each a named factor of ``Z_seen / Z_used``
---------------------------------------------------
S1  modal cutoff → wave impedance.  ``Z_used`` is built from ``cfg.f_cutoff``, the
    discrete eigenvalue of the port APERTURE.  The aperture spans one cell more
    than the guide, so ``f_cutoff`` sits O(dx) below the cutoff the guide actually
    propagates.
S2  Yee half-cell offset along the port normal.  ``_plane_h_field`` averages H over
    ``x_idx−1`` and ``x_idx``; on a forward wave that scales I by ``cos(β·dx/2)``.
S3  aperture weighting / transverse pairing.  ``_plane_h_field_at_dual`` smooths the
    simulated H with ``[1,2,1]/4`` per transverse axis and pairs it with an already
    smoothed profile (``hy = −_shift_profile_to_dual(ez, h_offset)``), with the
    ``+face`` aperture cell dropped by weight.

The guide's own cutoff is not assumed: the battery fitted it from the thru's S21
phase (``port_cutoff.per_rung[*].fc_fit_hz``), and this script prints that fit's
residual next to the port config's cutoff.

Pre-declaration and decision rule:
``docs/design_notes/waveguide_false_lane_column_power_predeclaration.md``.

Usage (from the repository root)::

    PYTHONPATH=. python scripts/diagnostics/waveguide_false_lane_column_power_suspects.py \
        --out <scratch>/suspects.json

REPRODUCER IS PINNED TO THE MERGE TREE ``df6ca133`` (#904). The port config is
rebuilt LIVE from ``tests._waveguide_chain_battery_fixture``, and since #889 the
port solves its transverse mode on the guide's N cells instead of N+1 -- so on
current main the live cutoff EQUALS the guide's discrete cutoff (6.5239 GHz at
the coarse rung, where the frozen measurement used the N+1 port's 5.8772 GHz),
suspect S1 collapses to identically zero, and a regenerated report would claim
the suspects explain ~22 % of an excess that the (N+1) port largely produced.
Re-running here measures a DIFFERENT port from the one that produced the frozen
S-parameters: the result is mixed-provenance and must not be committed over
``tests/fixtures/waveguide_false_lane_column_power/suspects.json``. The earlier
instruction to re-run and re-commit the report if #868 was decided is retracted
(#889 strict-xfails the live tie instead); re-measuring needs a new
pre-declaration. Point ``--out`` at a scratch path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/waveguide_chain_battery/fixture.json"

# Tolerances of the pre-declared decision rule.
MAG_FACTOR = 1.25
RATIO_TOL = 0.20


def _smooth_transverse(arr: np.ndarray) -> np.ndarray:
    """The extractor's transverse H stencil, edge-clamped, per axis.

    Mirrors ``rfx/sources/waveguide_port.py::_plane_h_field_at_dual`` for
    ``h_offset=(0.5, 0.5)``: ``0.25*(plus + 2*out + minus)`` with the first/last
    row repeated instead of wrapped.
    """
    out = np.asarray(arr, dtype=float).copy()
    for axis in (0, 1):
        n = out.shape[axis]
        plus = np.concatenate(
            [np.take(out, [0], axis=axis), np.take(out, np.arange(n - 1), axis=axis)],
            axis=axis,
        )
        minus = np.concatenate(
            [np.take(out, np.arange(1, n), axis=axis), np.take(out, [n - 1], axis=axis)],
            axis=axis,
        )
        out = 0.25 * (plus + 2.0 * out + minus)
    return out


def _complex(rows) -> np.ndarray:
    return np.asarray([complex(re, im) for re, im in rows], dtype=complex)


def _cell(fixture: dict, dut: str, lane: str, rung: str) -> dict:
    for c in fixture["cells"]:
        if c["dut"] == dut and c["lane"] == lane and c["rung"] == rung:
            return c
    raise KeyError(f"no cell {dut}|{lane}|{rung}")


def _guide_mode(nu: int, nv: int, n_guide: int, half_cell_h: bool):
    """Transverse shape of the guide's TE10 at the extractor's array indices.

    Walls on nodes: the E sample at index ``j`` sits at ``y = j·dx`` and the guide
    spans ``n_guide`` cells, so the shape is ``sin(π·j/n_guide)`` and the samples at
    and beyond the far wall are PEC-zero.  ``half_cell_h`` is the sensitivity
    variant in which H is taken half a cell over in u instead of co-located.
    """
    j = np.arange(nu, dtype=float)
    phi_e = np.sin(np.pi * j / n_guide)
    phi_e[n_guide:] = 0.0
    if half_cell_h:
        jh = j + 0.5
        phi_h = np.sin(np.pi * jh / n_guide)
        phi_h[jh > n_guide] = 0.0
    else:
        phi_h = phi_e.copy()
    ones = np.ones((1, nv))
    return phi_e[:, None] * ones, phi_h[:, None] * ones


def write_report(result: dict, out_path: Path) -> None:
    """Write the JSON report to ``out_path`` via a temp file + rename.

    Called from ``main`` only AFTER the console report has finished, and the
    bytes land under a ``.partial`` name first. Before #904 ``analyse`` wrote
    ``--out`` in the middle of its own run and ``main`` then died in the ratio
    step below, so a probe run left 25 KB of half-finished, mixed-provenance
    JSON at the COMMITTED report path. Partial output is now either absent or
    parked under ``.partial``; the real path is only ever replaced by a
    complete report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    tmp.write_text(
        json.dumps(_json_safe(result), indent=1, sort_keys=False) + "\n"
    )
    tmp.replace(out_path)


def _json_safe(obj):
    """Replace non-finite floats with ``None`` so the report is strict JSON.

    The committed report carries none: the frozen (N+1)-cell port made every
    suspect nonzero. A LIVE rebuild on current main does (S1 is identically
    zero post-#889, so its successive ratios are 0/0), and ``json.dumps``
    would emit bare ``NaN`` tokens, which Python reads back but a strict
    parser rejects. A degenerate run should produce a readable artifact, not
    an unparseable one -- the run is refused by the warning in ``main``, not
    by a corrupt file.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _successive_ratio_strings(vals: list[float]) -> tuple[list[str], bool]:
    """``[coarse/mid, mid/fine]`` formatted, with a zero denominator spelled out.

    Post-#889 the port solves its transverse mode on the guide's N cells, so
    the live cutoff EQUALS the guide's discrete cutoff and suspect S1 --
    ``gamma = (z_guide - z_port)/(z_guide + z_port)`` -- is identically zero at
    every rung. The bare ``vals[0] / vals[1]`` that used to sit here then
    raised ``ZeroDivisionError`` before the verdict line printed (#904).
    Returns the formatted ratios and whether any denominator was zero.
    """
    out: list[str] = []
    degenerate = False
    for num, den in ((vals[0], vals[1]), (vals[1], vals[2])):
        if den == 0.0:
            out.append("n/a")
            degenerate = True
        else:
            out.append(f"{num / den:.2f}")
    return out, degenerate


def analyse() -> dict:
    import jax.numpy as jnp

    import tests._waveguide_chain_battery_fixture as F
    from rfx.sources.waveguide_port import _compute_beta, _compute_mode_impedance

    fixture = json.loads(FIXTURE.read_text())
    freqs = np.asarray(fixture["fixture"]["freqs_hz"], dtype=float)
    per_rung_fit = fixture["port_cutoff"]["per_rung"]
    dx_by_rung = dict(zip(("coarse", "mid", "fine"), fixture["fixture"]["dx_ladder_m"]))
    n_by_rung = dict(zip(("coarse", "mid", "fine"), fixture["fixture"]["n_ladder"]))

    result: dict = {
        "schema": "rfx.waveguide_false_lane_column_power_suspects",
        "schema_version": 1,
        "predeclaration": (
            "docs/design_notes/waveguide_false_lane_column_power_predeclaration.md"
        ),
        "source_fixture": "tests/fixtures/waveguide_chain_battery/fixture.json",
        "source_run_id": fixture["provenance"]["run_id"],
        "source_commit": fixture["provenance"]["commit"],
        "freqs_hz": freqs.tolist(),
        "decision_rule": {
            "band_mean_magnitude_factor": MAG_FACTOR,
            "successive_ratio_tol": RATIO_TOL,
            "statistic": "band mean over the 17 bins of |S11| on the thru",
        },
        "per_rung": {},
    }

    for rung in ("coarse", "mid", "fine"):
        dx = float(dx_by_rung[rung])
        n_guide = int(n_by_rung[rung])
        cell = _cell(fixture, "thru", "false", rung)
        flux = _cell(fixture, "thru", "flux", rung)

        s11 = _complex(cell["s_params"]["S11"])
        s21 = _complex(cell["s_params"]["S21"])
        s12 = _complex(cell["s_params"]["S12"])
        s22 = _complex(cell["s_params"]["S22"])
        colpow = np.asarray(cell["column_power_per_bin"], dtype=float)

        sim = F.build_simulation("thru", dx)
        grid = sim._build_grid()
        n_steps = int(grid.num_timesteps(F.NUM_PERIODS))
        cfg = sim._build_waveguide_port_config(
            sim._waveguide_ports[0], grid, jnp.asarray(F.FREQS), n_steps
        )
        dt = float(cfg.dt)
        fc_port = float(cfg.f_cutoff)
        p_ez = np.asarray(cfg.ez_profile, dtype=float)
        p_hy = np.asarray(cfg.hy_profile, dtype=float)
        dA = np.asarray(cfg.aperture_dA, dtype=float)
        nu, nv = p_ez.shape

        fit = per_rung_fit[f"{rung}|false"]
        fc_guide = float(fit["fc_discrete_guide_hz"])
        fc_fit = float(fit["fc_fit_hz"])

        beta_g = np.real(
            np.asarray(_compute_beta(jnp.asarray(freqs), fc_guide, dt=dt, dx=dx))
        )
        z_guide = np.real(
            np.asarray(
                _compute_mode_impedance(jnp.asarray(freqs), fc_guide, "TE", dt=dt, dx=dx)
            )
        )
        z_port = np.real(
            np.asarray(
                _compute_mode_impedance(jnp.asarray(freqs), fc_port, "TE", dt=dt, dx=dx)
            )
        )

        variants = {}
        for name, half in (("colocated_h", False), ("half_cell_h", True)):
            phi_e, phi_h = _guide_mode(nu, nv, n_guide, half)
            num = float(np.sum(phi_e * p_ez * dA))
            den = float(np.sum(_smooth_transverse(phi_h) * p_hy * dA))
            variants[name] = -num / den

        q = variants["colocated_h"]
        cos_half = np.cos(beta_g * dx / 2.0)

        g1 = (z_guide - z_port) / (z_guide + z_port)
        z2 = z_guide / cos_half
        g2 = (z2 - z_guide) / (z2 + z_guide)
        z3 = z_guide * q
        g3 = (z3 - z_guide) / (z3 + z_guide)
        z_tot = z_guide / cos_half * q
        g_tot = (z_tot - z_port) / (z_tot + z_port)

        # what Z the measured S11 implies, band-mean magnitude only
        meas = np.abs(s11)
        entry = {
            "dx_m": dx,
            "guide_cells_u": n_guide,
            "aperture_cells_u": nu,
            "aperture_cells_v": nv,
            "dt_s": dt,
            "n_steps": int(cell["n_steps"]),
            "settling_db": cell["settling_db"],
            "preflight": cell["preflight"],
            "warnings": cell["warnings"],
            "cutoffs_hz": {
                "port_config_f_cutoff": fc_port,
                "guide_discrete": fc_guide,
                "guide_fitted_from_s21_phase": fc_fit,
                "analytic_c_over_2a": float(fit["fc_c_over_2a_hz"]),
                "rms_deg_at_fit": float(fit["rms_deg_at_fit"]),
                "rms_deg_at_discrete_guide": float(fit["rms_deg_at_discrete_guide"]),
                "rms_deg_at_port_cutoff": float(fit["rms_deg_at_port_cutoff"]),
                "port_cutoff_effective_width_cells": float(
                    fit["port_cutoff_effective_width_cells"]
                ),
            },
            "measured": {
                "column_power_max": float(cell["column_power_max"]),
                "column_power_excess_max": float(cell["column_power_max"]) - 1.0,
                "column_power_excess_per_bin_col0": (colpow[0] - 1.0).tolist(),
                "abs_s11_per_bin": meas.tolist(),
                "abs_s22_per_bin": np.abs(s22).tolist(),
                "abs_s21_per_bin": np.abs(s21).tolist(),
                "abs_s11_band_mean": float(meas.mean()),
                "abs_s22_band_mean": float(np.abs(s22).mean()),
                "s11_sq_band_mean": float((meas**2).mean()),
                "s21_sq_minus_one_band_mean": float((np.abs(s21) ** 2 - 1.0).mean()),
                "reflection_share_of_excess_bin0": float(
                    meas[0] ** 2 / (colpow[0][0] - 1.0)
                ),
                "flux_lane_column_power_max": float(flux["column_power_max"]),
                "flux_lane_abs_s11_band_mean": float(
                    np.abs(_complex(flux["s_params"]["S11"])).mean()
                ),
            },
            "suspects": {
                "S1_modal_cutoff_impedance": {
                    "gamma_per_bin": g1.tolist(),
                    "gamma_band_mean": float(np.abs(g1).mean()),
                },
                "S2_normal_half_cell_h_average": {
                    "gamma_per_bin": g2.tolist(),
                    "gamma_band_mean": float(np.abs(g2).mean()),
                },
                "S3_aperture_transverse_pairing": {
                    "q_ratio": q,
                    "q_ratio_half_cell_h_variant": variants["half_cell_h"],
                    "gamma_per_bin": g3.tolist(),
                    "gamma_band_mean": float(np.abs(g3).mean()),
                },
                "product_S1_S2_S3": {
                    "gamma_per_bin": g_tot.tolist(),
                    "gamma_band_mean": float(np.abs(g_tot).mean()),
                },
            },
            "excess_decomposition_bin0_col0": {
                "column_power_excess": float(colpow[0][0] - 1.0),
                "reflection_term_abs_s11_sq": float(meas[0] ** 2),
                "transmission_term_abs_s21_sq_minus_one": float(np.abs(s21[0]) ** 2 - 1.0),
            },
            "excess_decomposition_worst_bin_col0": {
                "worst_bin": int(np.argmax(colpow[0])),
                "column_power_excess": float(colpow[0].max() - 1.0),
                "reflection_term_abs_s11_sq": float(meas[int(np.argmax(colpow[0]))] ** 2),
                "transmission_term_abs_s21_sq_minus_one": float(
                    np.abs(s21[int(np.argmax(colpow[0]))]) ** 2 - 1.0
                ),
            },
            "predicted_column_power_excess": {
                "product_S1_S2_S3_per_bin": (np.abs(g_tot) ** 2).tolist(),
                "product_S1_S2_S3_max": float((np.abs(g_tot) ** 2).max()),
                "S1_only_max": float((np.abs(g1) ** 2).max()),
            },
            "residual": {
                "measured_minus_product_band_mean": float(
                    meas.mean() - np.abs(g_tot).mean()
                ),
                "product_share_of_measured_band_mean": float(
                    np.abs(g_tot).mean() / meas.mean()
                ),
                # The impedance factor the three modelled channels leave over:
                # (1+|Γ|)/(1−|Γ|) measured, divided by the same from the product.
                "implied_extra_z_factor": float(
                    ((1.0 + meas.mean()) / (1.0 - meas.mean()))
                    / ((1.0 + np.abs(g_tot).mean()) / (1.0 - np.abs(g_tot).mean()))
                ),
                "implied_extra_z_factor_minus_one_over_dx_over_a": float(
                    (
                        ((1.0 + meas.mean()) / (1.0 - meas.mean()))
                        / ((1.0 + np.abs(g_tot).mean()) / (1.0 - np.abs(g_tot).mean()))
                        - 1.0
                    )
                    / (1.0 / n_guide)
                ),
            },
            "reciprocity_mag_mean": float(cell["reciprocity_mag_mean"]),
            "abs_s12_band_mean": float(np.abs(s12).mean()),
        }
        result["per_rung"][rung] = entry

    # ladders and the pre-declared decision
    rungs = ("coarse", "mid", "fine")
    meas_mean = np.asarray(
        [result["per_rung"][r]["measured"]["abs_s11_band_mean"] for r in rungs]
    )
    ladders = {
        "measured_abs_s11_band_mean": meas_mean.tolist(),
        "measured_column_power_excess_max": [
            result["per_rung"][r]["measured"]["column_power_excess_max"] for r in rungs
        ],
        "predicted_column_power_excess_max_product": [
            result["per_rung"][r]["predicted_column_power_excess"]["product_S1_S2_S3_max"]
            for r in rungs
        ],
        "predicted_column_power_excess_max_S1_only": [
            result["per_rung"][r]["predicted_column_power_excess"]["S1_only_max"]
            for r in rungs
        ],
    }
    verdict_rows = {}
    for key in (
        "S1_modal_cutoff_impedance",
        "S2_normal_half_cell_h_average",
        "S3_aperture_transverse_pairing",
        "product_S1_S2_S3",
    ):
        pred = np.asarray(
            [result["per_rung"][r]["suspects"][key]["gamma_band_mean"] for r in rungs]
        )
        ladders[key] = pred.tolist()
        mag_ok = bool(np.all(np.maximum(pred / meas_mean, meas_mean / pred) <= MAG_FACTOR))
        pr = pred[:-1] / pred[1:]
        mr = meas_mean[:-1] / meas_mean[1:]
        ratio_ok = bool(np.all(np.abs(pr - mr) <= RATIO_TOL))
        verdict_rows[key] = {
            "band_mean_gamma": pred.tolist(),
            "pred_over_measured": (pred / meas_mean).tolist(),
            "pred_successive_ratios": pr.tolist(),
            "measured_successive_ratios": mr.tolist(),
            "magnitude_within_1p25": mag_ok,
            "ratio_within_0p20": ratio_ok,
            "reproduces_ladder": bool(mag_ok and ratio_ok),
        }
    result["ladders"] = ladders
    result["decision"] = verdict_rows
    winners = [k for k, v in verdict_rows.items() if v["reproduces_ladder"]]
    result["verdict"] = {
        "reproducing": winners,
        "branch": (
            "i_identified" if len(winners) == 1
            else "ii_not_distinguished" if len(winners) > 1
            else "iii_non_closing"
        ),
    }

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    res = analyse()
    rungs = ("coarse", "mid", "fine")
    print(f"source run {res['source_run_id']} commit {res['source_commit'][:8]}")
    print()
    print("cutoff (GHz)        " + "".join(f"{r:>12}" for r in rungs))
    for label, key in (
        ("port config f_cutoff", "port_config_f_cutoff"),
        ("guide, discrete    ", "guide_discrete"),
        ("guide, S21-phase fit", "guide_fitted_from_s21_phase"),
    ):
        row = [res["per_rung"][r]["cutoffs_hz"][key] / 1e9 for r in rungs]
        print(f"{label} " + "".join(f"{v:>12.4f}" for v in row))
    print("phase-fit rms (deg) at port cutoff  " + "".join(
        f"{res['per_rung'][r]['cutoffs_hz']['rms_deg_at_port_cutoff']:>9.3f}" for r in rungs))
    print("phase-fit rms (deg) at guide cutoff " + "".join(
        f"{res['per_rung'][r]['cutoffs_hz']['rms_deg_at_discrete_guide']:>9.3f}" for r in rungs))
    print()
    degenerate_rows: list[str] = []
    print("band-mean |Gamma|   " + "".join(f"{r:>12}" for r in rungs) + "   ratios")
    for key, label in (
        ("measured_abs_s11_band_mean", "measured |S11|      "),
        ("S1_modal_cutoff_impedance", "S1 cutoff->Z        "),
        ("S2_normal_half_cell_h_average", "S2 normal H average "),
        ("S3_aperture_transverse_pairing", "S3 aperture pairing "),
        ("product_S1_S2_S3", "product S1*S2*S3    "),
    ):
        vals = res["ladders"][key]
        rat, degenerate = _successive_ratio_strings(vals)
        if degenerate:
            degenerate_rows.append(key)
        note = "  (suspect identically zero post-#889)" if degenerate else ""
        print(label + "".join(f"{v:>12.5f}" for v in vals)
              + "   " + " ".join(rat) + note)
    print()
    print("column power - 1   " + "".join(f"{r:>13}" for r in rungs))
    for key, label in (
        ("measured_column_power_excess_max", "measured (max)     "),
        ("predicted_column_power_excess_max_product", "product S1*S2*S3   "),
        ("predicted_column_power_excess_max_S1_only", "S1 only            "),
    ):
        print(label + "".join(f"{v:>13.5e}" for v in res["ladders"][key]))
    print()
    for key, row in res["decision"].items():
        print(f"{key:34s} mag_ok={row['magnitude_within_1p25']!s:5s} "
              f"ratio_ok={row['ratio_within_0p20']!s:5s} "
              f"pred/meas={['%.3f' % v for v in row['pred_over_measured']]}")
    print()
    print("VERDICT:", res["verdict"]["branch"], res["verdict"]["reproducing"])

    if degenerate_rows:
        print(
            "\nWARNING: " + ", ".join(degenerate_rows) + " is identically zero on "
            "this tree.\n"
            "         The port config is rebuilt LIVE, and since #889 the port "
            "solves its\n"
            "         transverse mode on the guide's N cells, so its cutoff equals "
            "the guide's\n"
            "         discrete cutoff and S1 = (z_guide - z_port)/(z_guide + z_port) "
            "vanishes.\n"
            "         The frozen S-parameters this report explains were measured "
            "with the\n"
            "         (N+1)-cell port, so THIS RUN IS MIXED-PROVENANCE: do not commit "
            "it over\n"
            "         the committed report (#904). The committed report reproduces "
            "only on the\n"
            "         merge tree, df6ca133.",
            file=sys.stderr,
        )

    if args.out is not None:
        write_report(res, args.out)


if __name__ == "__main__":
    main()

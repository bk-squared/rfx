"""Replay the two retained #947 runs; no FDTD and no amplitude re-fitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


def replay(root):
    from tests import _transverse_resonance_o3 as model
    from tests.oracle import test_leontovich_alpha_oracle as oracle

    result, profiles = {}, {}
    for arm in ("baseline", "absorber_repair"):
        directory = root / arm
        summary = json.loads((directory / "current_summary.json").read_text())
        path = directory / "current_raw.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == summary["raw_sha256"]
        raw = np.load(path)
        xs = raw["fit_xs"]
        dx = summary["dx"]
        pad_x, pad_y, pad_z = summary["lower_pads"]
        first = round(xs[0] / dx) + pad_x
        j = round(0.001 / dx) + pad_y
        # These receipts register Ez at z=3mm, with its Yee z offset +dx/2.
        z_e = raw["z_nodes"][round(0.003 / dx) + pad_z] + dx / 2
        rows = []
        for fi, (f, saved) in enumerate(zip(summary["frequencies"], summary["model_fits"])):
            fit = dict(saved)
            fit["modes"] = [complex(*pair) for pair in saved["modes"]]
            fit["amps"] = [complex(*pair) for pair in saved["amps"]]
            fit["x_reference"] = float(xs[0] + dx / 2)
            ez = model.predict_ez_from_hy_fit(
                fit, xs, [z_e], f, oracle.B_PLATE, oracle.G_STUB,
                oracle.RS0, oracle.ETA_0)[:, 0]
            predicted = model._fit_alpha_loglin(xs, np.abs(ez))
            measured = summary["alpha_ez"][fi]
            rows.append({
                "frequency_hz": f, "hy_fit_relative_rms": fit["rel_resid"],
                "alpha_hy_measured": fit["alpha_meas"],
                "alpha_hy_model": fit["alpha_model"],
                "alpha_ez_measured": measured, "alpha_ez_model": predicted,
                "ez_relative_alpha_error": abs(measured / predicted - 1),
            })
        prof = raw["fit_ez_magnitude"][2]
        result[arm] = {
            "rows": rows, "settle_db": summary["settle_db"],
            "f0_two_plane_alpha": float(np.log(prof[0] / prof[-1]) / (xs[-1] - xs[0])),
            "f0_ez_log_fit_rms": summary["fit_ln_rms_ez"][2],
        }
        # Actual measurement coordinates, not collocated E/H assertions.
        profiles[arm] = {
            "x_e": xs, "x_h": xs + dx / 2,
            "z_e": float(z_e), "z_h": float(raw["fit_zs_hy"][5]),
            "ez": raw["ez_midplane"][0, first:first + len(xs), j],
            "hy": raw["fit_hy_plane"][0, :, 5],
        }
    return result, profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", type=Path, help="Optional 8GHz complex profile CSV")
    args = parser.parse_args()
    result, profiles = replay(args.evidence_root)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    if args.profiles:
        with args.profiles.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["arm", "x_Ez_m", "z_Ez_m", "Ez_real", "Ez_imag",
                             "x_Hy_m", "z_Hy_m", "Hy_real", "Hy_imag"])
            for arm, p in profiles.items():
                for xe, xh, e, h in zip(p["x_e"], p["x_h"], p["ez"], p["hy"]):
                    writer.writerow([arm, xe, p["z_e"], e.real, e.imag,
                                     xh, p["z_h"], h.real, h.imag])

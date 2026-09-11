"""Judge a stored estimator replay using cv24's existing mode/lattice gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation.crossval.comparators import nu_cavity_gates as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists; choose a new path")
    report = json.loads(args.report.read_text())
    capture = json.loads(args.capture.read_text())
    assert report["case"] == capture["case"]
    assert [r["input"] for r in report["records"]] == capture["records"]
    arm = capture["arm"]
    profile = np.asarray(arm["profile_mm"]) * 1e-3
    a, b = ((arm["nodes"][i] - 1) * arm["dx"] for i in (0, 1))
    band = (capture["records"][0]["f_min"], capture["records"][0]["f_max"])
    declared = G.declared_modes(band, a=a, b=b, d=float(profile.sum()))
    gate = G.estimator_floor()
    output = dict(case=report["case"], source_commit=capture["source_commit"],
                  report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  comparator_sha256=hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest(),
                  gate=gate, variants={})
    for variant in report["records"][0]["variants"]:
        windows = []
        for w in range(3):
            lines = []
            for c, channel in enumerate(G.CHANNELS):
                for mode in report["records"][w * len(G.CHANNELS) + c]["variants"][variant]["modes"]:
                    lines.append(dict(f_hz=mode["freq"], amp=mode["amplitude"],
                                      error=mode["error"], channel=channel))
            windows.append(G.identify_modes(lines, declared, band))
        errors, scatter = {}, {}
        for mode in declared:
            name = mode["name"]
            full, wa, wb = (w["per_mode"][name] for w in windows)
            if full is None:
                continue
            # The intended metric is the oracle even for the deliberately
            # swapped-metric input. Never fit the reference to the defect.
            exact = G.lattice_freq(mode["mnl"], profile, arm["dx"],
                                   dt=arm["dt"], a=a, b=b)["f_lattice_hz"]
            errors[name] = (full["f_hz"] / exact - 1) * 1e6
            if wa and wb:
                scatter[name] = abs(wa["f_hz"] - wb["f_hz"]) / full["f_hz"] * 1e6
        counts = [w["n_clusters_in_band"] for w in windows]
        output["variants"][variant] = dict(
            cluster_counts=counts, residual_ppm=errors, stationarity_ppm=scatter,
            lattice_pass=len(errors) == len(declared) and all(abs(v) <= gate * 1e6 for v in errors.values()),
            stationarity_pass=len(scatter) == len(declared) and all(v <= gate * 1e6 for v in scatter.values()),
            count_pass=all(n == len(declared) for n in counts))
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

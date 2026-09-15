#!/usr/bin/env python3
"""Inspect all bins and independent time histories from the short closure lanes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for col, lane in enumerate(("coarse", "fine")):
        folder = args.evidence / lane
        data = json.loads((folder / (lane + ".json")).read_text())
        run_id = (folder / "run_id.txt").read_text().strip()
        s = np.asarray(data["s_real"]) + 1j*np.asarray(data["s_imag"])
        f = np.asarray(data["frequencies_hz"])/1e9
        for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
            axes[0, col].plot(f, np.abs(s[i, j]), "o-", label=f"|S{i+1}{j+1}|")
        axes[0, col].axhline(1, color="grey", linestyle="--", linewidth=.8)
        axes[0, col].set(xlabel="Frequency (GHz)", ylabel="S magnitude",
                         title=f"{lane}: VESSL {run_id}")
        axes[0, col].legend(ncol=2)
        records = np.load(folder / (lane + "-records.npz"))
        for drive in range(2):
            ax = axes[drive+1, col]
            for field, style in (("v_ref_t", "-"), ("i_ref_t", "--"),
                                 ("v_probe_t", ":"), ("i_probe_t", "-.")):
                own = records[f"drive{drive}_port{drive}_{field}"]
                other = records[f"drive{drive}_port{1-drive}_{field}"]
                peak = np.max(np.abs(own))
                # Block maxima preserve pulses. Opposite records are divided
                # by the driven peak, never normalized by their own zero peak.
                edges = np.linspace(0, len(own), 161, dtype=int)
                t = edges[:-1]
                def envelope(x):
                    return np.array([np.max(np.abs(x[a:b]))/peak
                                     for a, b in zip(edges[:-1], edges[1:])])
                ax.plot(t, envelope(own), style, label=field)
                ax.plot(t, envelope(other), color="black", linewidth=.5)
            ax.set(xlabel="Recorded sample index", ylabel="Envelope / driven peak",
                   title=f"Drive {drive+1}: local histories; opposite histories = 0")
            ax.legend(ncol=2, fontsize=8)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

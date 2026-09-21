"""Leader's re-plot of the #801 field dumps with a usable colour range.

Reads dumps/<label>/final_state.npz only. Energy-like map w = |E|^2 + (mu0/eps0)|H|^2 (vacuum
weights: enough to see WHERE the field is), log10(w / max w), clipped to [-8, 0].
Writes new files dumps/<label>/leader_z.png and leader_profiles.txt; overwrites nothing else.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/root/workspace/bk-workspace/.801-measure/dumps"
ETA2 = (376.730313668) ** 2

for label, pad in [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]:
    d = np.load(f"{BASE}/{label}/final_state.npz")
    e2 = sum(np.asarray(d[k], dtype=np.float64) ** 2 for k in ("ex", "ey", "ez"))
    h2 = sum(np.asarray(d[k], dtype=np.float64) ** 2 for k in ("hx", "hy", "hz"))
    w = e2 + ETA2 * h2
    nx, ny, nz = w.shape
    i, j, k = np.unravel_index(np.argmax(e2), e2.shape)
    rel = np.log10(np.maximum(w, 1e-300) / w.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, plane, title, extent_labels in (
        (axes[0], rel[:, :, k].T, f"z index {k} (through max |E|)", ("x index", "y index")),
        (axes[1], rel[:, j, :].T, f"y index {j} (through max |E|)", ("x index", "z index")),
    ):
        im = ax.imshow(plane, origin="lower", aspect="auto", vmin=-8, vmax=0, cmap="viridis")
        ax.set_xlabel(extent_labels[0]); ax.set_ylabel(extent_labels[1]); ax.set_title(title)
        fig.colorbar(im, ax=ax, label="log10(w / max w)")
    for ax, n_second in ((axes[0], ny), (axes[1], nz)):
        for v in (pad - 0.5, nx - pad - 0.5):
            ax.axvline(v, color="w", lw=0.8, ls="--")
        for v in (pad - 0.5, n_second - pad - 0.5):
            ax.axhline(v, color="w", lw=0.8, ls="--")
    axes[0].plot([i], [j], "r+", ms=12); axes[1].plot([i], [k], "r+", ms=12)
    fig.suptitle(f"{label}: final field, max |E| at ({i}, {j}, {k}); dashed = absorber inner boundary")
    fig.tight_layout()
    fig.savefig(f"{BASE}/{label}/leader_z.png", dpi=110)
    plt.close(fig)

    with open(f"{BASE}/{label}/leader_profiles.txt", "w") as fh:
        for name, axis_sum in (("x", (1, 2)), ("y", (0, 2)), ("z", (0, 1))):
            prof = w.sum(axis=axis_sum); prof = prof / prof.sum()
            fh.write(f"{name}: " + " ".join(f"{v:.3e}" for v in prof) + "\n")
    px = w.sum(axis=(1, 2)); px /= px.sum(); py = w.sum(axis=(0, 2)); py /= py.sum()
    print(label, "max|E| at", (int(i), int(j), int(k)), "grid", w.shape)
    print("   x profile, first 8:", " ".join(f"{v:.2e}" for v in px[:8]))
    print("   x profile, last 8: ", " ".join(f"{v:.2e}" for v in px[-8:]))
    print("   y profile, first 8:", " ".join(f"{v:.2e}" for v in py[:8]))
    print("   y profile, last 8: ", " ".join(f"{v:.2e}" for v in py[-8:]))

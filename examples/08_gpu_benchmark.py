"""Example 8: GPU Performance Benchmark

Measures FDTD throughput across increasing grid sizes, timing
JIT compilation and execution separately.

Grid sizes tested: 32³, 64³, 128³, 200³

2-panel figure:
  Panel 1 — Throughput (Mcells/s) vs grid size (log x-scale)
             Annotated with peak throughput value
  Panel 2 — Execution time (s) vs grid size

Save: examples/08_gpu_benchmark.png

Run on a GPU node for meaningful absolute numbers. CPU results are
also valid for relative scaling comparisons.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays, init_materials
from rfx.simulation import run, make_source, make_probe
from rfx.sources.sources import GaussianPulse

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# Grid edge lengths (cubic grids: N × N × N)
GRID_SIZES = [32, 64, 128, 200]

DX = 1e-3           # 1 mm cell size
N_STEPS = 100       # fixed step count for fair comparison
F_MAX = 10e9        # 10 GHz (sets grid dt via CFL)
F0 = 5e9            # source centre frequency


def _run_one(n: int, n_steps: int) -> dict:
    """Benchmark a single N³ grid.

    Returns a dict with keys: n, shape, n_cells, jit_time_s, exec_time_s,
    mcells_per_s, backend.
    """
    domain = (n * DX, n * DX, n * DX)
    grid = Grid(freq_max=F_MAX, domain=domain, dx=DX, cpml_layers=0)

    pulse = GaussianPulse(f0=F0, bandwidth=0.5)
    center = (domain[0] / 2, domain[1] / 2, domain[2] / 2)

    src = make_source(grid, center, "ez", pulse, n_steps)
    prb = make_probe(grid, center, "ez")

    # Use uniform vacuum materials
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    # ---- JIT compilation (first call) ----
    t_jit_start = time.perf_counter()
    result_jit = run(grid, mats, n_steps, sources=[src], probes=[prb],
                     boundary="pec")
    result_jit.time_series.block_until_ready()
    t_jit_end = time.perf_counter()
    jit_time = t_jit_end - t_jit_start

    # ---- Timed execution (second call — JIT already compiled) ----
    # Rebuild source for a fresh call (same n_steps → same JIT specialisation)
    src2 = make_source(grid, center, "ez", pulse, n_steps)
    t_exec_start = time.perf_counter()
    result_exec = run(grid, mats, n_steps, sources=[src2], probes=[prb],
                      boundary="pec")
    result_exec.time_series.block_until_ready()
    t_exec_end = time.perf_counter()
    exec_time = t_exec_end - t_exec_start

    n_cells = int(np.prod(grid.shape))
    total_cell_updates = n_cells * n_steps
    mcells_per_s = total_cell_updates / exec_time / 1e6

    return {
        "n": n,
        "shape": grid.shape,
        "n_cells": n_cells,
        "jit_time_s": jit_time,
        "exec_time_s": exec_time,
        "mcells_per_s": mcells_per_s,
        "backend": jax.default_backend(),
    }


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

backend = jax.default_backend()
devices = jax.devices()
print(f"JAX backend : {backend}")
print(f"Devices     : {devices}")
print(f"Steps/run   : {N_STEPS}")
print()
print(f"{'Grid':>8s}  {'Cells':>10s}  {'JIT (s)':>9s}  {'Exec (s)':>9s}  {'Mcells/s':>10s}")
print("-" * 55)

bench_results = []
for n in GRID_SIZES:
    res = _run_one(n, N_STEPS)
    bench_results.append(res)
    print(f"{str(res['shape']):>8s}  {res['n_cells']:>10,}  "
          f"{res['jit_time_s']:>9.3f}  {res['exec_time_s']:>9.3f}  "
          f"{res['mcells_per_s']:>10.1f}")

print()

# ---------------------------------------------------------------------------
# Extract arrays for plotting
# ---------------------------------------------------------------------------

sizes       = np.array([r["n"] for r in bench_results])
n_cells_arr = np.array([r["n_cells"] for r in bench_results])
jit_times   = np.array([r["jit_time_s"] for r in bench_results])
exec_times  = np.array([r["exec_time_s"] for r in bench_results])
throughputs = np.array([r["mcells_per_s"] for r in bench_results])

peak_idx = int(np.argmax(throughputs))
peak_tp  = throughputs[peak_idx]
peak_n   = sizes[peak_idx]

print(f"Peak throughput: {peak_tp:.1f} Mcells/s at {peak_n}³ grid")

# ---------------------------------------------------------------------------
# Figure: 2-panel
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"rfx FDTD Benchmark — {backend.upper()}  "
             f"({N_STEPS} steps/run)",
             fontsize=13)

x_labels = [f"{n}³" for n in sizes]

# -- Panel 1: Throughput (Mcells/s) vs grid size ---------------------------
ax1.semilogx(n_cells_arr, throughputs, "o-", color="#1f77b4",
             linewidth=2, markersize=8, markerfacecolor="white",
             markeredgewidth=2, label="execution")
ax1.semilogx(n_cells_arr, throughputs, "s--", color="#aec7e8",
             linewidth=1, markersize=6, alpha=0.7,
             label="(JIT incl.)",
             )
# Overplot JIT-inclusive throughput
jit_tp = n_cells_arr * N_STEPS / jit_times / 1e6
ax1.semilogx(n_cells_arr, jit_tp, "s--", color="#ff7f0e",
             linewidth=1.5, markersize=6, alpha=0.8,
             label="JIT + exec")

# Annotate peak
ax1.annotate(
    f"Peak\n{peak_tp:.0f} Mcells/s",
    xy=(n_cells_arr[peak_idx], peak_tp),
    xytext=(n_cells_arr[peak_idx] * 1.3, peak_tp * 0.85),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black"),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
              edgecolor="gray"),
)

ax1.set_xlabel("Grid cells (N³)")
ax1.set_ylabel("Throughput (Mcells/s)")
ax1.set_title("Throughput vs Grid Size")
ax1.set_xticks(n_cells_arr)
ax1.set_xticklabels(x_labels)
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)
ax1.set_ylim(bottom=0)

# -- Panel 2: Execution time vs grid size ----------------------------------
bar_width = 0.35
x_pos = np.arange(len(sizes))

bars_exec = ax2.bar(x_pos - bar_width / 2, exec_times,
                    width=bar_width, label="Execution",
                    color="#1f77b4", alpha=0.85, edgecolor="white")
bars_jit  = ax2.bar(x_pos + bar_width / 2, jit_times,
                    width=bar_width, label="JIT compile",
                    color="#ff7f0e", alpha=0.85, edgecolor="white")

# Value labels on bars
for bar in bars_exec:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + max(exec_times) * 0.01,
             f"{h:.2f}s", ha="center", va="bottom", fontsize=8)
for bar in bars_jit:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + max(jit_times) * 0.01,
             f"{h:.2f}s", ha="center", va="bottom", fontsize=8)

ax2.set_xlabel("Grid size")
ax2.set_ylabel("Time (s)")
ax2.set_title("Execution vs JIT Compilation Time")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels)
ax2.legend(fontsize=9)
ax2.grid(True, axis="y", alpha=0.3)

# Device info annotation
device_str = str(devices[0]) if devices else backend
ax2.text(0.98, 0.97, f"Device: {device_str}\nn_steps={N_STEPS}",
         transform=ax2.transAxes,
         va="top", ha="right", fontsize=8, color="gray",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                   edgecolor="lightgray"))

plt.tight_layout()
out_path = "examples/08_gpu_benchmark.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

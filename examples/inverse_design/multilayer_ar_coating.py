"""Inverse design — multilayer AR coating optimization (rfx vs analytic TMM).

Substantive demonstration of rfx's JAX-native differentiable FDTD on a
classical microwave EM problem: minimize broadband reflection from a
high-εr substrate using a 3-layer matching coating.  ``jax.value_and_grad``
runs through ``Simulation.forward(eps_override=...)`` and Adam converges
to within a few percent of the analytic transfer-matrix-method (TMM)
optimum.

Setup (1D-equivalent: thin transverse domain + periodic y, z; CPML on x):

    ┌─CPML─┬──vacuum──┬──[3 design layers]──┬──substrate (εr=12)──┬─CPML─┐
    │      │  src     │  εr1   εr2   εr3    │   trans probe       │      │
    │      │  refl    │  fixed thicknesses  │                     │      │
    └──────┴──────────┴─────────────────────┴─────────────────────┴──────┘

Each layer is one quarter-wave thick at f0 = 10 GHz in vacuum, scaled by
the geometric-ladder mean index ``n_ref = εr_sub^(1/(2(N+1)))``.  The
three layer permittivities are the design variables.  Reference for
cross-validation: TMM (transfer matrix method) optimised with L-BFGS-B
on the same scalar objective.

Pipeline:
  (1) one vacuum reference forward → ``ts_inc(t)`` at refl probe (frozen)
  (2) per-iteration design forward → ``ts_total(t)`` at refl probe
  (3) ``ts_scat = ts_total − ts_inc`` → FFT → ``R(f) = |S_scat / S_inc|²``
  (4) cost = mean ``R(f)`` over X-band [8, 12 GHz]
  (5) ``jax.value_and_grad`` → Adam step

Acceptance gates (the real cross-validation story):
  - rfx Adam-final cost ≤ 1.2 × analytic TMM optimum cost.  This proves
    the AD pipeline converges to the same X-band-mean reflection that
    TMM predicts, even though FDTD has a slightly different optimum
    point on its discretised grid.
  - rfx Adam-final cost ≤ 0.7 × geometric-ladder cost (i.e. the
    optimiser improves on the closed-form starting approximation).
  - At least 2 design layers in the intermediate εr range
    [1.5, εr_sub − 0.5], confirming a real impedance ladder rather
    than collapse to vacuum or substrate.

Note on εr per-layer match: the broadband AR cost surface is highly
degenerate (many (εr1, εr2, εr3) combinations give ≈ the same X-band
mean R²), so Adam and L-BFGS can land on different εr triples that
both reach the same cost.  The cost-level agreement is the meaningful
cross-validation; the per-layer εr is not.

The method comparison quoted by ``rfx.minimize`` documentation is
``--compare-methods --mesh fine`` (dx = 0.25 mm, 20 CPML cells).

Run: python examples/inverse_design/multilayer_ar_coating.py
Gradient-only CPU smoke: JAX_PLATFORMS=cpu python -B \
    examples/inverse_design/multilayer_ar_coating.py --gradient-witness-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from scipy.optimize import minimize as _scimin

from rfx import Simulation, GaussianPulse, gradient_record_length_witness, minimize
from rfx.boundaries.spec import Boundary, BoundarySpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
EPS_SUB = 12.0
N_LAYERS = 3
F_LO, F_HI = 8e9, 12e9
F0 = 0.5 * (F_LO + F_HI)
F_MAX = 15e9
LAMBDA0 = C0 / F0
N_REF = EPS_SUB ** (0.5 / (N_LAYERS + 1))
LAYER_THK = LAMBDA0 / (4.0 * N_REF)
DX = 0.5e-3

# x-axis layout (relative to domain origin)
X_PML_PAD = 6e-3
X_SRC = 10e-3
X_REFL = 22e-3
X_DESIGN_LO = 35e-3
X_DESIGN_HI = X_DESIGN_LO + N_LAYERS * LAYER_THK
X_TRANS = X_DESIGN_HI + 8e-3
# Substrate extends through the right-side CPML so the wave is absorbed
# inside the substrate (CPML works inside dielectrics).  Any partial
# substrate→vacuum interface inside the domain would create a strong
# spurious reflection back to the refl probe and inflate |R(f)|².
LX = X_TRANS + 18e-3
LY = LZ = DX

N_PERIODS = 30


# ---------------------------------------------------------------------------
# Simulation builder
# ---------------------------------------------------------------------------
def _simulation_kwargs(dx, cpml_cells):
    return dict(
        freq_max=F_MAX, domain=(LX, dx, dx), dx=dx,
        cpml_layers=cpml_cells,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="periodic", hi="periodic"),
            z=Boundary(lo="periodic", hi="periodic"),
        ),
    )


def _add_excitation(sim, dx):
    bw = (F_HI - F_LO) / F0 * 1.6
    sim.add_source(
        (X_SRC, dx / 2, dx / 2), "ez",
        waveform=GaussianPulse(f0=F0, bandwidth=bw, amplitude=1.0),
        amplitude_kind="current",
    )
    sim.add_probe((X_REFL, dx / 2, dx / 2), "ez")
    sim.add_probe((X_TRANS, dx / 2, dx / 2), "ez")
    return sim


def _build_simulation() -> Simulation:
    """Build the default mesh without solving (also used by fidelity checks)."""
    return _add_excitation(Simulation(**_simulation_kwargs(DX, 10)), DX)


class _ARMesh:
    """Simulation and all mesh-dependent geometry and frequency samples."""

    def __init__(self, sim, dx, cpml_cells):
        """``eps_override`` includes the CPML pad outside the declared domain."""
        self.sim = sim
        self.dx = dx
        self.cpml_cells = cpml_cells
        grid = sim.freeze_mesh()
        self.nx = grid.shape[0]
        self.dt = float(grid.dt)
        self.n_steps = int(round(N_PERIODS / F0 / self.dt))
        self.edges = [
            cpml_cells + int(round((X_DESIGN_LO + i * LAYER_THK) / dx))
            for i in range(N_LAYERS + 1)
        ]
        self.cells = np.diff(self.edges)
        self.nfft = 1 << (self.n_steps - 1).bit_length()
        all_bins = np.fft.rfftfreq(self.nfft, self.dt)
        self.band_indices = np.flatnonzero((all_bins >= F_LO) & (all_bins <= F_HI))
        self.bins = all_bins[self.band_indices]

    def vacuum(self):
        return jnp.ones((self.nx, 1, 1), dtype=jnp.float32)

    def base(self):
        return self.vacuum().at[self.edges[-1]:, :, :].set(EPS_SUB)

    def render(self, layer_eps):
        eps = self.base()
        for i in range(N_LAYERS):
            eps = eps.at[self.edges[i]:self.edges[i + 1], :, :].set(layer_eps[i])
        return eps

    def forward(self, eps):
        return self.sim.forward(
            eps_override=eps, n_steps=self.n_steps,
            skip_preflight=True).time_series

    def metadata(self):
        return dict(dx_m=self.dx, cpml_cells=self.cpml_cells, NX=self.nx,
                    DT=self.dt, N_STEPS=self.n_steps, layer_edges=self.edges,
                    layer_cells=self.cells.tolist(),
                    thicknesses_m=(self.cells * self.dx).tolist(),
                    nfft=self.nfft, bins_hz=self.bins.tolist())


def _build_mesh(dx, cpml_cells):
    """Build a mesh with one cell in each periodic transverse direction."""
    sim = _add_excitation(Simulation(**_simulation_kwargs(dx, cpml_cells)), dx)
    return _ARMesh(sim, dx, cpml_cells)


_DEFAULT_MESH = _ARMesh(_build_simulation(), DX, 10)
NX, DT, N_STEPS = _DEFAULT_MESH.nx, _DEFAULT_MESH.dt, _DEFAULT_MESH.n_steps
LAYER_EDGES_C = _DEFAULT_MESH.edges
LAYER_CELLS = _DEFAULT_MESH.cells.tolist()
i_design_lo, i_design_hi = LAYER_EDGES_C[0], LAYER_EDGES_C[-1]
i_sub_lo, i_sub_hi = i_design_hi, NX


def _print_setup(mesh=None) -> None:
    mesh = _DEFAULT_MESH if mesh is None else mesh
    DX, NX, DT, N_STEPS = mesh.dx, mesh.nx, mesh.dt, mesh.n_steps
    LAYER_CELLS, LAYER_EDGES_C = mesh.cells, mesh.edges
    i_sub_lo, i_sub_hi = mesh.edges[-1], mesh.nx
    print("=" * 72)
    print("Multilayer AR coating — rfx FDTD (JAX-grad-Adam) vs analytic TMM")
    print("=" * 72)
    print(f"  εr_sub = {EPS_SUB}, N_layers = {N_LAYERS}, "
          f"X-band = [{F_LO/1e9:.1f}, {F_HI/1e9:.1f}] GHz")
    print(f"  λ0 = {LAMBDA0*1e3:.2f} mm, n_ref = {N_REF:.3f}, "
          f"layer thk = {LAYER_THK*1e3:.3f} mm  ({LAYER_CELLS[0]} cells)")
    print(f"  dx = {DX*1e3:.2f} mm, LX = {LX*1e3:.1f} mm, "
          f"NX = {NX}, dt = {DT*1e12:.2f} ps, n_steps = {N_STEPS}, "
          f"CPML cells = {mesh.cpml_cells}")
    print(f"  layer-edge cells: {LAYER_EDGES_C}, "
          f"substrate cells [{i_sub_lo}, {i_sub_hi})")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _vacuum_eps_array() -> jnp.ndarray:
    """All-vacuum reference: pure incident — no substrate, no coating."""
    return _DEFAULT_MESH.vacuum()


def _base_eps_array() -> jnp.ndarray:
    """Vacuum design region + fixed substrate on the padded grid."""
    return _DEFAULT_MESH.base()


def _render_design_eps(layer_eps: jnp.ndarray) -> jnp.ndarray:
    return _DEFAULT_MESH.render(layer_eps)


def _latent_to_eps(latent: jnp.ndarray) -> jnp.ndarray:
    """Bounded sigmoid: ℝ → [1, EPS_SUB]."""
    return 1.0 + (EPS_SUB - 1.0) * jax.nn.sigmoid(latent)


# ---------------------------------------------------------------------------
# Meep cross-validation reference loader
# ---------------------------------------------------------------------------
# Meep R(f) reference for this geometry is computed by a VESSL job:
#   research/microwave-energy/meep_simulation/jobs/ar_coating_meep_for_rfx.yaml
#   →  research/microwave-energy/meep_simulation/ar_coating_reference.py
# Results are dropped into  docs/research_notes/ar_coating_meep_*.json
# (gitignored).  We load them here when available and overlay on the
# spectrum plot.  Local Meep is not used (NumPy 1.x/2.x ABI conflict).
# ---------------------------------------------------------------------------
def load_meep_reference(name: str) -> dict | None:
    """Load a Meep R(f) JSON dropped by the VESSL job.  None if absent."""
    p = os.path.join(SCRIPT_DIR, "..", "..", "docs", "research_notes",
                     f"ar_coating_meep_{name}.json")
    p = os.path.abspath(p)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Analytic TMM (numpy)
# ---------------------------------------------------------------------------
def tmm_R(layer_eps: np.ndarray, freqs: np.ndarray,
          eps_sub: float = EPS_SUB,
          layer_thk: float | np.ndarray = LAYER_THK) -> np.ndarray:
    """|R(f)|² for normal-incidence multilayer (vacuum-coatings-substrate)."""
    n_layers = len(layer_eps)
    thicknesses = np.broadcast_to(layer_thk, (n_layers,))
    n0 = 1.0
    n_sub = float(np.sqrt(eps_sub))
    R = np.zeros_like(freqs, dtype=float)
    for fi, f in enumerate(freqs):
        if f <= 0:
            R[fi] = 0.0
            continue
        M = np.eye(2, dtype=complex)
        for li in range(n_layers):
            n_l = float(np.sqrt(layer_eps[li]))
            beta = 2.0 * np.pi * f * n_l * thicknesses[li] / C0
            cos_b, sin_b = np.cos(beta), np.sin(beta)
            ML = np.array([[cos_b,            1j * sin_b / n_l],
                           [1j * n_l * sin_b, cos_b           ]], dtype=complex)
            M = M @ ML
        Y0, YN = n0, n_sub
        num = (Y0 * M[0, 0] + Y0 * YN * M[0, 1] - M[1, 0] - YN * M[1, 1])
        den = (Y0 * M[0, 0] + Y0 * YN * M[0, 1] + M[1, 0] + YN * M[1, 1])
        r = num / den
        R[fi] = float(np.abs(r) ** 2)
    return R


def tmm_band_cost(layer_eps: np.ndarray, eps_sub: float = EPS_SUB) -> float:
    fs = np.linspace(F_LO, F_HI, 51)
    return float(np.mean(tmm_R(layer_eps, fs, eps_sub=eps_sub)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def record_length_check(mesh=None):
    """Print the coating reflection gradient's record-length witness.

    Uses the existing smallest transverse domain (one cell per periodic
    axis). The tolerance is a demonstration choice, not an accuracy claim.
    Both records use the same frequency samples and their own vacuum reference.
    """
    mesh = _DEFAULT_MESH if mesh is None else mesh
    N_STEPS, DT = mesh.n_steps, mesh.dt
    sim_ref = sim_design = mesh.sim
    nfft = int(2 ** np.ceil(np.log2(N_STEPS)))
    bins = np.fft.rfftfreq(nfft, d=DT)
    freqs = jnp.asarray(bins[(bins >= F_LO) & (bins <= F_HI)])

    def objective(layer_eps, n_steps):
        incident = sim_ref.forward(
            eps_override=mesh.vacuum(), n_steps=n_steps,
            skip_preflight=True).time_series[:, 0]
        total = sim_design.forward(
            eps_override=mesh.render(layer_eps), n_steps=n_steps,
            skip_preflight=True).time_series[:, 0]
        phase = jnp.exp(-2j * jnp.pi * freqs[:, None]
                        * (jnp.arange(n_steps) * DT)[None, :])
        inc = phase @ incident
        scattered = phase @ (total - incident)
        return jnp.mean((jnp.abs(scattered) / (jnp.abs(inc) + 1e-30)) ** 2)

    layer_eps = jnp.asarray([
        EPS_SUB ** ((i + 1) / (N_LAYERS + 1)) for i in range(N_LAYERS)
    ], dtype=jnp.float32)
    witness = gradient_record_length_witness(
        objective, layer_eps, N_STEPS, tol=0.05)
    print(f"record-length witness: n_steps={N_STEPS}, "
          f"passed={witness.passed}, gradient change={witness.worst:.6g}, "
          f"value change={witness.worst_value_rel_change:.6g}")
    return witness


def main(mesh=None, output_dir=SCRIPT_DIR) -> int:
    mesh = _DEFAULT_MESH if mesh is None else mesh
    N_STEPS, DT = mesh.n_steps, mesh.dt
    _print_setup(mesh)
    record_length_check(mesh)

    # ---- TMM optimum ------------------------------------------------------
    print("\n[TMM] L-BFGS-B optimisation against analytic R(f)...")
    res_tmm = _scimin(
        tmm_band_cost,
        x0=np.array([2.5, 4.0, 7.0]),
        bounds=[(1.0, EPS_SUB)] * N_LAYERS,
        method="L-BFGS-B",
    )
    tmm_eps = res_tmm.x
    tmm_cost = res_tmm.fun
    print(f"  TMM εr = {tmm_eps},  mean R = {tmm_cost:.4e}")
    geo_ladder = np.array([
        EPS_SUB ** ((i + 1) / (N_LAYERS + 1)) for i in range(N_LAYERS)
    ])
    print(f"  Geometric ladder εr (closed-form approx): {geo_ladder}")
    print(f"  Geometric ladder mean R: {tmm_band_cost(geo_ladder):.4e}")

    # ---- Vacuum reference forward ----------------------------------------
    print("\n[ref] vacuum forward (one-shot, for incident-wave subtraction)...")
    sim_ref = mesh.sim
    t0 = time.time()
    result_ref = sim_ref.forward(
        eps_override=mesh.vacuum(),
        n_steps=N_STEPS,
        skip_preflight=True,
    )
    ts_inc_refl = jnp.asarray(result_ref.time_series[:, 0])
    ts_inc_trans = jnp.asarray(result_ref.time_series[:, 1])
    print(f"  done in {time.time() - t0:.1f}s, "
          f"max |ts_inc_refl| = {float(jnp.max(jnp.abs(ts_inc_refl))):.4e}")

    # ---- Differentiable design forward -----------------------------------
    nfft = int(2 ** np.ceil(np.log2(N_STEPS)))
    freqs_fft = jnp.fft.rfftfreq(nfft, d=DT)
    band_mask = (freqs_fft >= F_LO) & (freqs_fft <= F_HI)
    band_norm = float(jnp.sum(band_mask))

    sim_design = mesh.sim

    def _design_ts(layer_eps: jnp.ndarray) -> jnp.ndarray:
        eps = mesh.render(layer_eps)
        res = sim_design.forward(
            eps_override=eps,
            n_steps=N_STEPS,
            skip_preflight=True,
        )
        return res.time_series

    def cost_with_aux(layer_eps: jnp.ndarray):
        ts = _design_ts(layer_eps)
        ts_total_refl = ts[:, 0]
        ts_scat = ts_total_refl - ts_inc_refl
        S_inc = jnp.fft.rfft(ts_inc_refl, n=nfft)
        S_scat = jnp.fft.rfft(ts_scat, n=nfft)
        R = (jnp.abs(S_scat) / (jnp.abs(S_inc) + 1e-30)) ** 2
        cost = jnp.sum(R * band_mask) / band_norm
        return cost, R

    def cost_only(layer_eps):
        return cost_with_aux(layer_eps)[0]

    print("\n[rfx] Adam optimisation through differentiable forward...")
    n_iters = 60
    t0 = time.time()
    result = minimize(
        cost_only, np.asarray(geo_ladder, dtype=np.float32),
        method="adam", bounds=[(1.0, EPS_SUB)] * N_LAYERS,
        lr=0.15, n_iters=n_iters,
    )
    elapsed = time.time() - t0
    losses = result.history
    eps_trace = np.asarray(result.x_history)
    for it, (cost_val, eps_now) in enumerate(zip(losses, eps_trace)):
        if it % 5 == 0 or it == len(losses) - 1:
            print(f"  iter {it:3d}  cost = {cost_val:.4e}  εr = "
                  f"[{eps_now[0]:.3f}, {eps_now[1]:.3f}, {eps_now[2]:.3f}]")
    print(f"  {result.nfev} iters in {elapsed:.1f}s "
          f"({elapsed/result.nfev:.2f}s/iter)")

    final_eps = np.asarray(result.x)
    _, initial_R = cost_with_aux(result.x_history[0])
    _, R_final = cost_with_aux(result.x)
    initial_R, R_final = np.asarray(initial_R), np.asarray(R_final)
    final_cost = result.fun

    # ---- Meep cross-validation (loaded from VESSL job output) ------------
    meep_rfx = load_meep_reference("rfx_final")
    meep_tmm = load_meep_reference("tmm_optimum")
    meep_rfx_fine = load_meep_reference("rfx_final_fine")
    meep_at_rfx_eps = float(meep_rfx["mean_R_band"]) if meep_rfx else None
    meep_at_tmm_eps = float(meep_tmm["mean_R_band"]) if meep_tmm else None

    # ---- Comparison -------------------------------------------------------
    err_pct = np.abs(final_eps - tmm_eps) / tmm_eps * 100.0
    refl_drop_db = 10.0 * np.log10(losses[0] / max(final_cost, 1e-20))

    print("\n" + "=" * 72)
    print("Comparison")
    print("=" * 72)
    print(f"  εr (rfx final): [{final_eps[0]:.3f}, {final_eps[1]:.3f}, {final_eps[2]:.3f}]")
    print(f"  εr (TMM opt):   [{tmm_eps[0]:.3f}, {tmm_eps[1]:.3f}, {tmm_eps[2]:.3f}]")
    print(f"  per-layer error: [{err_pct[0]:.2f}, {err_pct[1]:.2f}, {err_pct[2]:.2f}] %")
    print(f"  cost: rfx initial = {losses[0]:.4e}, rfx final = {final_cost:.4e}, "
          f"TMM opt = {tmm_cost:.4e}")
    print(f"  reflection improvement: {refl_drop_db:.1f} dB")
    if meep_at_rfx_eps is not None:
        print(f"  Meep mean R @ rfx εr  = {meep_at_rfx_eps:.4e}  "
              f"(rfx-FDTD: {final_cost:.4e}, TMM: {tmm_band_cost(final_eps):.4e})")
    if meep_at_tmm_eps is not None:
        print(f"  Meep mean R @ TMM εr  = {meep_at_tmm_eps:.4e}  "
              f"(TMM: {tmm_cost:.4e}, ratio Meep/TMM = "
              f"{meep_at_tmm_eps/tmm_cost:.3f})")

    # ---- Plots ------------------------------------------------------------
    fs_plot = np.linspace(F_LO, F_HI, 51)
    R_tmm_init_geo = tmm_R(geo_ladder, fs_plot)
    R_tmm_opt = tmm_R(tmm_eps, fs_plot)
    R_tmm_rfx = tmm_R(final_eps, fs_plot)

    freqs_fft_arr = np.asarray(freqs_fft)
    mask = (freqs_fft_arr >= F_LO) & (freqs_fft_arr <= F_HI)
    fs_rfx = freqs_fft_arr[mask]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    axes[0].semilogy(fs_rfx / 1e9, initial_R[mask], "r-", lw=1.3,
                      label="rfx FDTD (initial)")
    axes[0].semilogy(fs_rfx / 1e9, R_final[mask], "g-", lw=2,
                      label="rfx FDTD (optimised)")
    axes[0].semilogy(fs_plot / 1e9, R_tmm_rfx, "g--", lw=1.2,
                      label="TMM @ rfx εr")
    axes[0].semilogy(fs_plot / 1e9, R_tmm_opt, "k-", lw=1.5,
                      label="TMM optimum")
    axes[0].semilogy(fs_plot / 1e9, R_tmm_init_geo, "b:", lw=1,
                      label="geometric ladder")
    if meep_rfx is not None:
        mf = np.asarray(meep_rfx["freqs_band_hz"]) / 1e9
        mr = np.real(np.asarray(meep_rfx["R_band"]))
        axes[0].semilogy(mf, np.clip(mr, 1e-7, None), "m-.", lw=1.5,
                          label=f"Meep @ rfx εr (res={meep_rfx['meta']['resolution_cells_per_mm']})")
    if meep_tmm is not None:
        mf = np.asarray(meep_tmm["freqs_band_hz"]) / 1e9
        mr = np.real(np.asarray(meep_tmm["R_band"]))
        axes[0].semilogy(mf, np.clip(mr, 1e-7, None), "c-.", lw=1.2,
                          label="Meep @ TMM εr")
    axes[0].set_xlabel("Frequency [GHz]")
    axes[0].set_ylabel("|R(f)|²")
    axes[0].set_title("Reflection magnitude (X-band)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(1e-6, 1.0)

    iters_x = np.arange(n_iters)
    axes[1].semilogy(iters_x, losses, "g-", lw=2, label="rfx Adam")
    axes[1].axhline(tmm_cost, color="k", ls="--", label="TMM optimum")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("mean |R|² over X-band")
    axes[1].set_title(f"Convergence  ({refl_drop_db:.1f} dB drop)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    eps_arr = np.asarray(eps_trace)
    for li in range(N_LAYERS):
        line, = axes[2].plot(iters_x, eps_arr[:, li], lw=2,
                              label=f"εr_{li+1}")
        axes[2].axhline(tmm_eps[li], color=line.get_color(), ls="--",
                         alpha=0.6)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("εr per layer")
    axes[2].set_title("εr trajectory (dashed = TMM optimum)")
    axes[2].legend(loc="best"); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.8, EPS_SUB + 0.5)

    fig.suptitle(
        f"Multilayer AR coating: rfx FDTD-Adam vs analytic TMM "
        f"(εr_sub={EPS_SUB}, X-band, 3 layers @ λ/4)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, "multilayer_ar_coating.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out_png}")

    # ---- Gates ------------------------------------------------------------
    geo_cost_tmm = tmm_band_cost(geo_ladder)
    cost_ratio_to_tmm = final_cost / max(tmm_cost, 1e-20)
    cost_ratio_to_geo = final_cost / max(geo_cost_tmm, 1e-20)
    gate_match_tmm = cost_ratio_to_tmm <= 1.2
    gate_beats_geo = cost_ratio_to_geo <= 0.7
    n_intermediate = int(np.sum((final_eps > 1.5) & (final_eps < EPS_SUB - 0.5)))
    gate_intermediate = n_intermediate >= 2
    # Meep cross-validation gate: when the VESSL Meep reference is present,
    # require Meep@TMM-εr to match analytic TMM within 10 % (full-wave FDTD
    # vs analytic transfer-matrix on the same configuration).  Skipped when
    # the Meep JSON isn't present so the demo still runs standalone.
    if meep_at_tmm_eps is not None:
        meep_vs_tmm = meep_at_tmm_eps / max(tmm_cost, 1e-20)
        gate_meep_vs_tmm = 0.9 <= meep_vs_tmm <= 1.1
    else:
        meep_vs_tmm = None
        gate_meep_vs_tmm = True   # not gating when reference absent
    all_ok = (gate_match_tmm and gate_beats_geo and gate_intermediate
              and gate_meep_vs_tmm)

    print("\nGates:")
    print(f"  rfx final cost ≤ 1.2 × TMM optimum: "
          f"{'PASS' if gate_match_tmm else 'FAIL'}  "
          f"(ratio = {cost_ratio_to_tmm:.3f})")
    print(f"  rfx final cost ≤ 0.7 × geometric-ladder cost: "
          f"{'PASS' if gate_beats_geo else 'FAIL'}  "
          f"(ratio = {cost_ratio_to_geo:.3f})")
    print(f"  ≥ 2 intermediate εr layers: "
          f"{'PASS' if gate_intermediate else 'FAIL'}  "
          f"({n_intermediate}/{N_LAYERS} layers in [1.5, {EPS_SUB-0.5}])")
    if meep_vs_tmm is not None:
        print(f"  Meep @ TMM-εr ≈ analytic TMM (0.9–1.1×): "
              f"{'PASS' if gate_meep_vs_tmm else 'FAIL'}  "
              f"(ratio = {meep_vs_tmm:.3f})")
    else:
        print(f"  Meep cross-validation: "
              f"<not present — run microwave-energy/meep_simulation/jobs/"
              f"ar_coating_meep_for_rfx.yaml to populate>")
    print()
    print(f"{'PASS' if all_ok else 'FAIL'}: multilayer_ar_coating — "
          f"{'AD pipeline matches analytic TMM optimum' if all_ok else 'gates failed'}")
    return 0 if all_ok else 1


def _ladder_start():
    return np.asarray([
        EPS_SUB ** ((i + 1) / (N_LAYERS + 1)) for i in range(N_LAYERS)
    ], dtype=np.float32)


def _reflectance_function(mesh):
    """Freeze one vacuum forward; return the plain 15-bin design forward."""
    incident = jnp.asarray(mesh.forward(mesh.vacuum())[:, 0])
    spectrum_inc = jnp.fft.rfft(incident, n=mesh.nfft)
    indices = jnp.asarray(mesh.band_indices)

    def vector(layer_eps):
        total = mesh.forward(mesh.render(layer_eps))[:, 0]
        scattered = jnp.fft.rfft(total - incident, n=mesh.nfft)
        return ((jnp.abs(scattered) / (jnp.abs(spectrum_inc) + 1e-30)) ** 2)[indices]

    return vector


def _comparison_specs(vector):
    def mean(layer_eps):
        return jnp.mean(vector(layer_eps))

    def q32(layer_eps):
        values = vector(layer_eps)
        scale = jax.lax.stop_gradient(jnp.max(values))
        return scale * jnp.mean((values / scale) ** 32) ** (1.0 / 32)

    return [
        ("adam_mean", mean, dict(method="adam", lr=0.15, n_iters=60)),
        ("projected_mean", mean, dict(method="projected_adam", lr=0.0375, n_iters=60)),
        ("mma_mean", mean, dict(method="mma", n_iters=60)),
        ("mma_worst", vector, dict(method="mma", worst_case=True, n_iters=40)),
        ("adam_q32", q32, dict(method="adam", lr=0.15, n_iters=60, best_iterate=True)),
    ]


def compare_methods(mesh, output_dir=SCRIPT_DIR):
    """Report plain-forward costs; diagnostic forwards are outside solves."""
    if len(mesh.bins) != 15:
        raise ValueError("method comparison requires the 15 X-band DFT bins")
    vector = _reflectance_function(mesh)
    report = dict(mesh=mesh.metadata(), vacuum_forwards=1, runs={})
    print("mesh " + json.dumps(report["mesh"], sort_keys=True))
    print("Iterations are 1-based evaluated points; trace forwards are uncounted.")
    for name, objective, settings in _comparison_specs(vector):
        result = minimize(objective, _ladder_start(),
                          bounds=[(1.0, EPS_SUB)] * N_LAYERS, **settings)
        trace = []
        passes = N_LAYERS if result.jacobian == "forward" else 2
        for iteration, x in enumerate(result.x_history, 1):
            R = np.asarray(vector(x))
            row = dict(iteration=iteration, solves=passes * iteration,
                       x=np.asarray(x).tolist(), R=R.tolist(),
                       f=float(np.mean(R)), w=float(np.max(R)))
            trace.append(row)
        metric = "w" if name in ("mma_worst", "adam_q32") else "f"
        final = trace[-1] if trace else None
        crossing = next((dict(iteration=row["iteration"], solves=row["solves"])
                         for row in trace if row[metric] <= 1.01 * final[metric]), None)
        run = dict(settings=settings, trace=trace, final=final, crossing_metric=metric,
                   first_within_one_percent=crossing, solves=result.solves,
                   setup_forwards=result.setup_forwards, subproblem_caps=result.subproblem_caps,
                   nfev=result.nfev, stopped_nonfinite=result.stopped_nonfinite,
                   jacobian=result.jacobian, scale=result.scale,
                   objective_history=result.history, diagnostic_forwards=len(trace))
        if final is not None:
            R_tmm = tmm_R(np.asarray(final["x"]), mesh.bins,
                          layer_thk=mesh.cells * mesh.dx)
            run["e_real"] = float(np.max(np.abs(np.asarray(final["R"]) - R_tmm)))
        if name == "adam_q32" and trace:
            best_index = int(np.argmin(result.history))
            run["best_iterate"] = dict(iteration=best_index + 1,
                                       x=np.asarray(result.x).tolist(),
                                       w=trace[best_index]["w"])
        if name == "mma_mean":
            xs = np.asarray(result.x_history)
            run["last_20_steps"] = [
                dict(iteration=i + 1, abs_delta_eps=np.abs(xs[i] - xs[i - 1]).tolist(),
                     abs_delta_box=(np.abs(xs[i] - xs[i - 1]) / (EPS_SUB - 1)).tolist())
                for i in range(max(1, len(xs) - 20), len(xs))
            ]
        report["runs"][name] = run
        print(f"{name} settings=" + json.dumps(settings, sort_keys=True))
        for row in trace:
            print(f"  {row['iteration']:02d} solves={row['solves']:3d} "
                  f"f={row['f']:.10g} w={row['w']:.10g}")
        print(f"  first within 1% of final {metric}: " + json.dumps(crossing))
        if final is not None:
            print("  final eps=" + json.dumps(final["x"]))
            print("  final R=" + json.dumps(final["R"]))
            print(f"  e_real={run['e_real']:.10g}")
        if "best_iterate" in run:
            print("  best_iterate=True " + json.dumps(run["best_iterate"]))
        for step in run.get("last_20_steps", []):
            print("  step " + json.dumps(step))
        print("  solves=" + json.dumps(result.solves, sort_keys=True)
              + f" setup_forwards={result.setup_forwards} "
              f"subproblem_caps={result.subproblem_caps} "
              f"stopped_nonfinite={result.stopped_nonfinite}")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "multilayer_ar_comparison.json")
    with open(path, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"JSON: {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--gradient-witness-only", action="store_true")
    routes.add_argument("--compare-methods", action="store_true")
    parser.add_argument("--mesh", choices=("default", "fine"), default="default")
    parser.add_argument("--output-dir", default=SCRIPT_DIR)
    args = parser.parse_args()
    mesh = _build_mesh(0.25e-3, 20) if args.mesh == "fine" else _DEFAULT_MESH
    if args.gradient_witness_only:
        _print_setup(mesh)
        record_length_check(mesh)
    elif args.compare_methods:
        compare_methods(mesh, args.output_dir)
    else:
        sys.exit(main(mesh, args.output_dir))

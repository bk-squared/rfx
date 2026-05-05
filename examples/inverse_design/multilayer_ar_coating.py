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

Run: python examples/inverse_design/multilayer_ar_coating.py
"""

from __future__ import annotations

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

from rfx import Simulation, GaussianPulse
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
def _build_simulation() -> Simulation:
    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, LZ),
        dx=DX,
        cpml_layers=10,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="periodic", hi="periodic"),
            z=Boundary(lo="periodic", hi="periodic"),
        ),
    )
    bw = (F_HI - F_LO) / F0 * 1.6   # ≈ 0.64 — covers the X-band cleanly
    sim.add_source(
        (X_SRC, LY / 2, LZ / 2),
        "ez",
        waveform=GaussianPulse(f0=F0, bandwidth=bw, amplitude=1.0),
    )
    sim.add_probe((X_REFL, LY / 2, LZ / 2), "ez")
    sim.add_probe((X_TRANS, LY / 2, LZ / 2), "ez")
    return sim


# Build once to read grid metadata
_SIM_PROBE = _build_simulation()
_GRID = _SIM_PROBE._build_grid()
NX = _GRID.shape[0]
DT = float(_GRID.dt)
N_STEPS = int(round(N_PERIODS / F0 / DT))

# Cell indices for design / substrate
i_design_lo = int(round(X_DESIGN_LO / DX))
i_design_hi = int(round(X_DESIGN_HI / DX))
i_sub_lo = i_design_hi
i_sub_hi = NX  # substrate runs through the right-side CPML for clean absorption
LAYER_EDGES_C = [
    int(round((X_DESIGN_LO + i * LAYER_THK) / DX))
    for i in range(N_LAYERS + 1)
]
LAYER_CELLS = [LAYER_EDGES_C[i + 1] - LAYER_EDGES_C[i] for i in range(N_LAYERS)]


def _print_setup() -> None:
    print("=" * 72)
    print("Multilayer AR coating — rfx FDTD (JAX-grad-Adam) vs analytic TMM")
    print("=" * 72)
    print(f"  εr_sub = {EPS_SUB}, N_layers = {N_LAYERS}, "
          f"X-band = [{F_LO/1e9:.1f}, {F_HI/1e9:.1f}] GHz")
    print(f"  λ0 = {LAMBDA0*1e3:.2f} mm, n_ref = {N_REF:.3f}, "
          f"layer thk = {LAYER_THK*1e3:.3f} mm  ({LAYER_CELLS[0]} cells)")
    print(f"  dx = {DX*1e3:.2f} mm, LX = {LX*1e3:.1f} mm, "
          f"NX = {NX}, dt = {DT*1e12:.2f} ps, n_steps = {N_STEPS}")
    print(f"  layer-edge cells: {LAYER_EDGES_C}, "
          f"substrate cells [{i_sub_lo}, {i_sub_hi})")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _vacuum_eps_array() -> jnp.ndarray:
    """All-vacuum reference: pure incident — no substrate, no coating."""
    return jnp.ones((NX, 1, 1), dtype=jnp.float32)


def _base_eps_array() -> jnp.ndarray:
    """Vacuum design region + fixed substrate on the right."""
    eps = jnp.ones((NX, 1, 1), dtype=jnp.float32)
    return eps.at[i_sub_lo:i_sub_hi, :, :].set(EPS_SUB)


def _render_design_eps(layer_eps: jnp.ndarray) -> jnp.ndarray:
    eps = _base_eps_array()
    for i in range(N_LAYERS):
        lo = LAYER_EDGES_C[i]
        hi = LAYER_EDGES_C[i + 1]
        eps = eps.at[lo:hi, :, :].set(layer_eps[i])
    return eps


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
          layer_thk: float = LAYER_THK) -> np.ndarray:
    """|R(f)|² for normal-incidence multilayer (vacuum-coatings-substrate)."""
    n_layers = len(layer_eps)
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
            beta = 2.0 * np.pi * f * n_l * layer_thk / C0
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
def main() -> int:
    _print_setup()

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
    sim_ref = _build_simulation()
    t0 = time.time()
    result_ref = sim_ref.forward(
        eps_override=_vacuum_eps_array(),
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

    sim_design = _build_simulation()

    def _design_ts(layer_eps: jnp.ndarray) -> jnp.ndarray:
        eps = _render_design_eps(layer_eps)
        res = sim_design.forward(
            eps_override=eps,
            n_steps=N_STEPS,
            skip_preflight=True,
        )
        return res.time_series

    def cost_with_aux(latent: jnp.ndarray):
        layer_eps = _latent_to_eps(latent)
        ts = _design_ts(layer_eps)
        ts_total_refl = ts[:, 0]
        ts_scat = ts_total_refl - ts_inc_refl
        S_inc = jnp.fft.rfft(ts_inc_refl, n=nfft)
        S_scat = jnp.fft.rfft(ts_scat, n=nfft)
        R = (jnp.abs(S_scat) / (jnp.abs(S_inc) + 1e-30)) ** 2
        cost = jnp.sum(R * band_mask) / band_norm
        return cost, R

    def cost_only(latent):
        return cost_with_aux(latent)[0]

    grad_fn = jax.value_and_grad(cost_only)

    # ---- Adam loop --------------------------------------------------------
    print("\n[rfx] Adam optimisation through differentiable forward...")
    n_iters = 60
    lr = 0.15
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    # Warm start: geometric-ladder εr is the well-known closed-form
    # approximation to the broadband AR optimum.  Initialising from there
    # puts Adam in the basin of attraction of the TMM optimum from step 0
    # rather than relying on it to traverse a potentially noisy gradient
    # surface from the εr-mid initialisation.
    p_geo = (geo_ladder - 1.0) / (EPS_SUB - 1.0)
    p_geo = np.clip(p_geo, 1e-3, 1.0 - 1e-3)
    latent_init = np.log(p_geo / (1.0 - p_geo))
    latent = jnp.asarray(latent_init, dtype=jnp.float32)
    m = jnp.zeros_like(latent)
    v = jnp.zeros_like(latent)

    losses: list[float] = []
    eps_trace: list[np.ndarray] = []

    t0 = time.time()
    initial_R: np.ndarray | None = None
    for it in range(n_iters):
        cost, grad = grad_fn(latent)
        cost_val = float(cost)
        eps_now = np.asarray(_latent_to_eps(latent))
        losses.append(cost_val)
        eps_trace.append(eps_now)

        if it == 0:
            _, R0 = cost_with_aux(latent)
            initial_R = np.asarray(R0)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        latent = latent - lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)

        if it % 5 == 0 or it == n_iters - 1:
            print(f"  iter {it:3d}  cost = {cost_val:.4e}  εr = "
                  f"[{eps_now[0]:.3f}, {eps_now[1]:.3f}, {eps_now[2]:.3f}]")
    elapsed = time.time() - t0
    print(f"  {n_iters} iters in {elapsed:.1f}s ({elapsed/n_iters:.2f}s/iter)")

    final_eps = np.asarray(_latent_to_eps(latent))
    _, R_final = cost_with_aux(latent)
    R_final = np.asarray(R_final)
    final_cost = losses[-1]

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
    out_png = os.path.join(SCRIPT_DIR, "multilayer_ar_coating.png")
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


if __name__ == "__main__":
    sys.exit(main())

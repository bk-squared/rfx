"""MSL open-stub notch tuning — differentiable inverse design.

Microwave-engineering inverse-design demo on a 2-port microstrip filter:
tune the open-stub length so the transmission notch lands at a chosen
design frequency.  Stub length is reformulated as a continuous sigmoid PEC
density mask via :meth:`Simulation.forward(pec_occupancy_override=…)` and
optimised with Adam through ``jax.grad``.  Cost is ``|S21(f_target)|²`` from
the plane-integrated JAX N-probe extractor.

Multimodal cost surface
-----------------------
The ``|S21(f_target=6 GHz)|²`` cost over L_stub ∈ [4, 12] mm is physically
**MULTIMODAL**, not convex:

  * GLOBAL minimum at L ≈ 7.0 mm, cost ≈ -45.9 dB — the in-band λ/4
    open-stub notch (analytic target 7.374 mm, ~5 % from the FDTD optimum).
  * SECONDARY valley near ~9.53 mm, ~290× shallower than the global min.
    This is the longer-stub branch notching BELOW band (e.g. a -28.6 dB
    notch near 3.98 GHz at L = 10.8 mm), so its in-band |S21| stays higher.

Because the surface is multimodal, a single-start Adam seeded above the
notch (e.g. at 9.5 mm) settles on the secondary valley and never reaches
the 7.0 mm global basin.  The demo therefore uses a best-of MULTI-START
Adam (see ``_multistart_adam``) with seeds spanning the band so one init
sits in the global basin near 7.0 mm.  The gradient itself is sign-correct
and device-independent (CPU==GPU); the multimodality — not any extractor or
AD issue — is what a single start fails to cross.

Differentiable PEC parameterisation
-----------------------------------
``pec_occupancy_override`` routes through the Kottke inv-eps machinery
(``compute_inv_eps_tensor_diag`` / ``_kottke_inv_eps_diag``) when
``RFX_PEC_OCC_KOTTKE=1``, giving a correct PEC limit with a smooth gradient.
The legacy ``apply_pec_occupancy`` E-tangential damping instead produces
sub-β wiggles in the high-Q cost surface that trap Adam in a ~β-period
oscillation.  Three ingredients combine to give correct PEC + smooth
gradient on the Kottke path:

  1. **Strict Kottke PEC limit** (``is_pec=True``) with a sigmoid-tail
     clamp (occ < 1e-3 → 0) so the floating-point sigmoid floor doesn't
     trip the f>0 selector.
  2. **Heaviside projection** centered at occ=0.5 (smooth_width=0.05) to
     force-zero interior cells, mirroring Stage 2's
     ``where(e_inside, 0, ...)`` for a hard ``Box(material='pec')``.
  3. **1-cell PEC dilation** via 6-neighbor max-pool of occupancy before
     projection — the AD-smooth analogue of the binary ``apply_pec_mask``
     ``mask & (roll | roll)`` rule the imperative ``compute_msl_s_matrix``
     uses for ``Box(material='pec')``.

The combination gives a global-min notch depth ≈ -45.9 dB at L ≈ 7.0 mm and
lets ``jax.grad`` flow cleanly through sigmoid → density → Yee → DFT
extractor.

Two-branch physics
------------------
An open stub of length L behind a feedline notches at the frequency where
the stub is an odd multiple of λ_g/4 (open end → short at the junction).  In
this band that gives two distinct branches inside [4, 12] mm:
  * L ≈ 7.0 mm → the IN-BAND λ/4 notch at the 6 GHz target (global cost
    minimum).
  * L ≳ 9.5 mm → the longer-stub branch whose λ/4 falls BELOW band (e.g.
    ~3.98 GHz at L = 10.8 mm), a shallow secondary valley in-band.

Validation chain (the point of this example):
  1. Best-of MULTI-START Adam minimises ``|S21_jax(f_target)|²`` w.r.t.
     ``L_stub`` (inits span the band so one is in the global basin near
     7.0 mm; see ``_multistart_adam``).
  2. Brute-force scan (same JAX extractor, N_SCAN ≥ 17) finds the global-min
     reference ``L_ref`` ≈ 7.0 mm.
  3. **Cross-solver gate**: at the best-basin ``L_opt``, run the imperative
     :meth:`Simulation.compute_msl_s_matrix` and check that the imperative
     notch is real (deep ≤ -15 dB) and near the design target.

Any change to ``pec_occupancy_override`` should be checked for mesh
robustness: single-mesh shortcuts (σ-loading, a sharper sigmoid) can work at
one dx and break at another.  Verify Adam descent at more than one dx and
FD-vs-AD agreement before trusting a new parameterisation.

Geometry: cv06b-class (uniform dx=127 µm = h_sub/2 = 2 substrate cells,
L_LINE=30 mm).  Long enough for each MSL port's 3-probe extractor to sit
outside the stub-junction standing-wave region (λ_g/4 reflector-clearance
check, enforced by `sim.preflight()` — see
`tests/test_msl_port_preflight.py::test_reflector_clearance_*`).  A shorter
5 mm line biases |S11|@notch to ≈ -7 dB instead of the physical 0 dB; this
geometry avoids that.

Run: ``python examples/inverse_design/msl_stub_notch_tuning.py``
"""

from __future__ import annotations

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.msl_wave_decomp import (
    register_msl_plane_probes,
    _v_from_plane,
    _i_from_plane,
    extract_msl_nprobe,
    MSLPlaneProbeSet,  # used in build_sim's return-type annotation
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# ---------------------------------------------------------------------------
# Problem constants — shrunken cv06b
# ---------------------------------------------------------------------------
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 127e-6                            # h_sub / 2 — 2 substrate cells.
                                        # Demo geometry of record.
                                        # Refining to 80 µm (cv06b
                                        # standard) gives a cleaner
                                        # ε_eff staircase but exposes a
                                        # dx-fragility in the plane lane
                                        # (|S21|² > 1 unphysical) that is
                                        # a follow-up; for now the demo
                                        # runs at h_sub/2 where the
                                        # extractor closure is verified
                                        # by
                                        # ``tests/test_msl_plane_extractor_jax.py``.
L_LINE = 30.0e-3                       # cv06b-class line length.  Each
                                        # MSL port's V₃ probe must sit
                                        # ≥ λ_g/4 from the stub PEC
                                        # reflector for the imperative
                                        # `compute_msl_s_matrix` 3-probe
                                        # extractor to read |S11|@notch
                                        # cleanly; preflight enforces
                                        # the bound (see
                                        # `_check_msl_port_geometry`,
                                        # `tests/test_msl_port_preflight
                                        # .py::test_reflector_clearance_*`).
PORT_MARGIN = 1.6e-3                   # ≥ cpml_extent (8·dx ≈ 1.02 mm)
                                        # + 2·h_sub safety; preflight
                                        # x-CPML clearance check.
F_MAX = 9e9

L_STUB_MAX = 14.0e-3
L_MIN, L_MAX = 4.0e-3, 12.0e-3
L_INIT = 9.5e-3                        # legacy single-start init —
                                        # NO LONGER the descent start.
                                        # 9.5 mm sits ON the SECONDARY
                                        # valley (~9.53 mm, the
                                        # below-band longer-stub
                                        # branch), NOT a clean descent
                                        # above the notch.  main() uses
                                        # best-of MULTI-START Adam
                                        # (`_multistart_adam`) with
                                        # inits spanning the band; this
                                        # constant is kept only as one
                                        # of those seeds (the
                                        # secondary-valley probe).  The
                                        # global λ/4 notch is at
                                        # L ≈ 7.0 mm (analytic 7.37 mm).
SIGMOID_BETA = max(DX * 0.25, 0.05 * H_SUB)
# Sigmoid PEC mask sharpness for the differentiable stub-length
# parameterisation.  The σ-loading path folds occ × σ_PEC into
# materials.sigma, so a broad β (e.g. ``DX * 0.7``) distorts the cost
# landscape: the sigmoid edge cells get partial σ-loading (lossy
# half-PEC at occ=0.5) that shifts the stub's effective characteristic
# and can produce an artefactual cost minimum near L ≈ 4-5 mm.  A
# sharper β (≈ ¼ dx, floored at 0.05·h_sub ≈ 13 µm) keeps the partial-σ
# edge narrow enough that the stub physics stays close to the hard-PEC
# Box reference: at β ≈ 5 µm the cost minimum lands at the 6 GHz λ/4
# length (L ≈ 7 mm at dx = 127 µm, L ≈ 6 mm at dx = 80 µm), matching
# the imperative-reference notch positions within 1 mm.  AD gradient
# through a sharper β remains finite because the sigmoid is still
# smooth over the cell-center samples used by pec_occupancy_override.

u = W_TRACE / H_SUB
EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5

# Default cost target — quarter-wave at 6.0 GHz → L_stub ≈ 7.4 mm,
# inside the design-bound interior.
F_TARGET = 6.0e9
L_TARGET_AN = C0 / (4 * F_TARGET * np.sqrt(EPS_EFF))

# Domain
LX = L_LINE + 2 * PORT_MARGIN
msl_clearance = 2 * (2 * H_SUB + 8 * DX)
LY = W_TRACE + msl_clearance + L_STUB_MAX + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.0e-3


# ---------------------------------------------------------------------------
# Sim builder + sigmoid stub mask
# ---------------------------------------------------------------------------
def build_sim(freqs: jnp.ndarray) -> tuple[
    Simulation, float, float, "MSLPlaneProbeSet", "MSLPlaneProbeSet",
]:
    """Build the through-line geometry (no stub Box) with 2 MSL ports
    + plane DFT probes (V₁/V₂/V₃ Ez planes + Hy plane per port).

    Returns the sim, the trace-y centre and trace-y top, and the two
    plane probe sets for post-forward extraction via
    :func:`extract_msl_nprobe`.
    """
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")

    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2.0
    trace_y_lo = y_trace - W_TRACE / 2.0
    trace_y_hi = y_trace + W_TRACE / 2.0
    sim.add(Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
            material="pec")

    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0)

    # Plane DFT probes — line-integrated V (Ez) + area-integrated I
    # (Hy) per port.  Mirrors the imperative `compute_msl_s_matrix`
    # plane integrals exactly, so the JAX-traceable N-probe extractor
    # in `extract_msl_nprobe` carries no scalar-Ez bias.
    d_set = register_msl_plane_probes(sim, port_index=0, freqs=freqs,
                                      name_prefix="d")
    p_set = register_msl_plane_probes(sim, port_index=1, freqs=freqs,
                                      name_prefix="p")

    # Drive only port 0 — disable port-1 excitation in the underlying
    # MSLPortEntry (frozen dataclass; bypass with object.__setattr__).
    object.__setattr__(sim._msl_ports[1], "excite", False)
    return sim, y_trace, trace_y_hi, d_set, p_set


def build_stub_occ(grid, trace_y_hi: float, L_stub: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid soft-PEC stub mask of commanded length ``L_stub``,
    rooted at ``trace_y_hi`` along +y, in the trace-x footprint."""
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_TRACE / 2.0
    stub_x_hi = stub_x_centre + W_TRACE / 2.0
    z_patch = H_SUB + 0.5 * DX

    x_centres = (np.arange(nx) - pad_x + 0.5) * DX
    y_centres = (np.arange(ny) - pad_y + 0.5) * DX
    z_centres = (np.arange(nz) - pad_z + 0.5) * DX

    in_x = ((x_centres >= stub_x_lo) & (x_centres <= stub_x_hi)).astype(np.float32)
    in_z = (np.abs(z_centres - z_patch) <= 0.5 * DX).astype(np.float32)
    in_x_j = jnp.asarray(in_x); in_z_j = jnp.asarray(in_z)
    y_far = jnp.asarray(y_centres - trace_y_hi, dtype=jnp.float32)
    sig_low = jax.nn.sigmoid(y_far / SIGMOID_BETA)
    sig_high = jax.nn.sigmoid((L_stub - y_far) / SIGMOID_BETA)
    sig_y = sig_low * sig_high
    return (in_x_j[:, None, None] * sig_y[None, :, None]
            * in_z_j[None, None, :]).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Best-of multi-start Adam (PURE — testable without FDTD)
# ---------------------------------------------------------------------------
def _multistart_adam(
    cost_fn,
    latent_inits,
    n_iters,
    lr,
    max_dL_per_step,
    latent_to_L,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    """Run Adam from each latent init and return the best-of (lowest
    final cost) trajectory.

    PURE in ``cost_fn``: takes the scalar-latent → scalar-cost
    callable as an argument, so it is testable against a synthetic
    bimodal cost WITHOUT any FDTD forward.  The ``[4, 12] mm`` MSL
    cost surface is multimodal (in-band λ/4 notch ~7.0 mm vs the
    below-band longer-stub valley ~9.5 mm), so single-start Adam can
    settle in the wrong basin — this best-of-N start sweep is the fix.

    Per step the update is clamped so the physical length moves by at
    most ``max_dL_per_step`` (default ~0.2 mm < the ~0.6 mm
    secondary-valley width), preventing the lr=0.4 latent step from
    overshooting a valley wall.

    Parameters
    ----------
    cost_fn : callable(latent) -> scalar cost (differentiable via jax.grad)
    latent_inits : sequence of scalar latent starting points
    n_iters : Adam iterations per start
    lr : Adam learning rate (applied to the latent)
    max_dL_per_step : max |ΔL_physical| per step, in the units of
        ``latent_to_L`` (metres for the MSL demo)
    latent_to_L : callable(latent) -> physical length (for the clamp +
        history; e.g. ``L_MIN + (L_MAX-L_MIN)*sigmoid(latent)``)

    Returns
    -------
    (best_L, best_cost, best_history, all_histories)
        ``best_L`` / ``best_cost`` are the winning start's BEST-SEEN
        iterate (lowest cost along its trajectory — NOT the final iterate,
        which can be an overshoot of a sharp resonant null); exposed on
        each history as ``L_best`` / ``cost_best`` alongside the diagnostic
        ``L_final`` / ``cost_final``.  ``best_history`` is that start's
        per-iter dict; ``all_histories`` is the list of every start's
        history dict (for the multi-basin plot witness).
    """
    cost_grad_fn = jax.value_and_grad(cost_fn)
    all_histories = []
    best_idx = None
    best_cost = None
    for s, latent0 in enumerate(latent_inits):
        latent = jnp.asarray(latent0, dtype=jnp.float32)
        m_buf = jnp.zeros_like(latent)
        v_buf = jnp.zeros_like(latent)
        hist = {"start": s, "L": [], "cost": [], "db": [], "latent": []}
        for it in range(n_iters):
            loss, grad = cost_grad_fn(latent)
            loss_v = float(loss)
            L_now = float(latent_to_L(latent))
            db = 10.0 * math.log10(max(loss_v, 1e-12))
            hist["L"].append(L_now * 1e3)
            hist["cost"].append(loss_v)
            hist["db"].append(db)
            hist["latent"].append(float(latent))
            m_buf = beta1 * m_buf + (1 - beta1) * grad
            v_buf = beta2 * v_buf + (1 - beta2) * grad ** 2
            m_hat = m_buf / (1 - beta1 ** (it + 1))
            v_hat = v_buf / (1 - beta2 ** (it + 1))
            step = lr * m_hat / (jnp.sqrt(v_hat) + eps)
            latent_trial = latent - step
            # Per-step physical-length clamp: scale the latent step so
            # |L(latent_trial) - L(latent)| <= max_dL_per_step.  Keep it
            # simple + monotone — bisect the step down if it overshoots.
            dL = abs(float(latent_to_L(latent_trial)) - L_now)
            scale = 1.0
            for _ in range(40):
                if dL <= max_dL_per_step:
                    break
                scale *= 0.5
                latent_trial = latent - scale * step
                dL = abs(float(latent_to_L(latent_trial)) - L_now)
            else:
                # The clamp did not converge in 40 halvings (e.g. a huge /
                # non-finite step) — take NO step this iter rather than
                # applying an unclamped move.
                latent_trial = latent
            latent = latent_trial
        # record the final landing point (post-loop) for this start
        final_loss = float(cost_fn(latent))
        final_L = float(latent_to_L(latent))
        # Append the final landing point into the per-iter arrays so the
        # plotted trajectory ends exactly at L_opt (the in-loop append
        # records the PRE-update point each iteration, so without this the
        # convergence plot stops one step short of the landing point).
        hist["L"].append(final_L * 1e3)
        hist["cost"].append(final_loss)
        hist["db"].append(10.0 * math.log10(max(final_loss, 1e-12)))
        hist["latent"].append(float(latent))
        hist["L_final"] = final_L
        hist["cost_final"] = final_loss
        hist["latent_final"] = float(latent)
        # BEST-ITERATE selection: Adam can OVERSHOOT a sharp resonant null.
        # A trajectory can hit the −46 dB null at L ≈ 7.0-7.1 mm but
        # momentum then carries it out to ~7.4 mm / −34 dB, so keeping the
        # FINAL iterate would discard the −46 dB point Adam actually
        # visited.  Keep the BEST-seen iterate along the trajectory instead
        # (standard best-checkpoint, not last-checkpoint).  Only finite
        # costs are eligible (NaN can never win — preserves the NaN guard).
        finite_pts = [
            (c, Lmm, lat)
            for c, Lmm, lat in zip(hist["cost"], hist["L"], hist["latent"])
            if math.isfinite(c)
        ]
        if finite_pts:
            c_best, Lmm_best, lat_best = min(finite_pts, key=lambda t: t[0])
            hist["cost_best"], hist["L_best"], hist["latent_best"] = (
                c_best, Lmm_best / 1e3, lat_best,
            )
        else:
            hist["cost_best"], hist["L_best"], hist["latent_best"] = (
                float("nan"), final_L, float(latent),
            )
        all_histories.append(hist)
        # NaN-safe best-OF-starts on the per-trajectory BEST iterate (not the
        # final, which may be an overshoot of a sharp null).
        if math.isfinite(hist["cost_best"]) and hist["cost_best"] < (
            best_cost if best_cost is not None else math.inf
        ):
            best_cost = hist["cost_best"]
            best_idx = s
    if best_idx is None:
        # Every start produced a non-finite cost everywhere — fail loudly
        # rather than indexing all_histories[None] with a NaN optimum.
        raise RuntimeError(
            f"multi-start Adam: all {len(latent_inits)} starts produced a "
            "non-finite cost at every iterate; no usable optimum."
        )
    best_history = all_histories[best_idx]
    best_L = best_history["L_best"]
    return best_L, best_cost, best_history, all_histories


# ---------------------------------------------------------------------------
# Cost + Adam
# ---------------------------------------------------------------------------
def main() -> int:
    # Defaults sized for ~12-15 min wall on this dev box.  The cost
    # landscape over L_stub ∈ [4, 12] mm is MULTIMODAL (in-band λ/4
    # notch ~7.0 mm = global min vs the below-band longer-stub valley
    # ~9.5 mm), so a single-start Adam can settle in the wrong basin.
    # We therefore run best-of N_START Adam (RFX_Y2B_NSTART) with inits
    # spanning the band, each capped to N_ITERS steps.  LR is 0.15 (a
    # larger 0.4 overshoots the ~0.3 mm valley wall); the per-step
    # physical-length clamp RFX_Y2B_MAXDL (default 0.2 mm < the ~0.6 mm
    # secondary-valley width) is the hard guard.  N_ITERS is 8: a seed
    # offset onto the basin wall (6.5 mm) must DESCEND ~0.5 mm into the
    # 7.0 mm notch under the 0.2 mm/step clamp, which needs several steps
    # to reach the global null floor (best-iterate keeps the deepest point
    # even if Adam then overshoots the sharp null).
    NUM_PERIODS = float(os.environ.get("RFX_Y2B_PERIODS", 10.0))
    N_ITERS = int(os.environ.get("RFX_Y2B_ITERS", 8))
    LR = float(os.environ.get("RFX_Y2B_LR", 0.15))
    N_SCAN = int(os.environ.get("RFX_Y2B_SCAN", 17))
    N_START = int(os.environ.get("RFX_Y2B_NSTART", 3))
    MAX_DL = float(os.environ.get("RFX_Y2B_MAXDL", 2e-4))

    f_target_arr = jnp.asarray([F_TARGET], dtype=jnp.float32)
    sim, y_trace, trace_y_hi, d_set, p_set = build_sim(f_target_arr)
    grid = sim._build_grid()
    print(f"Grid {grid.shape}  ({np.prod(grid.shape):,d} cells)  dt={float(grid.dt)*1e12:.2f}ps")

    # Preflight: surfaces MSL geometry warnings (lateral clearance, substrate
    # cells, x-CPML, reflector clearance — the last is what protects the
    # N-probe extractor from sitting in the stub-junction standing-wave
    # region; see `_check_msl_port_geometry` in rfx/api.py).
    pre_msgs = sim.preflight()
    if pre_msgs:
        print("\nPreflight warnings:")
        for m in pre_msgs:
            print(f"  - {m}")
    else:
        print("\nPreflight: clean.")
    print(f"εr={EPS_R}, h_sub={H_SUB*1e6:.0f}µm, W={W_TRACE*1e6:.0f}µm, "
          f"dx={DX*1e6:.0f}µm  ε_eff={EPS_EFF:.3f}")
    print(f"L_stub bounds [{L_MIN*1e3:.1f}, {L_MAX*1e3:.1f}] mm   "
          f"multi-start N={N_START} (band-spanning seeds)")
    print(f"f_target={F_TARGET/1e9:.2f} GHz   "
          f"analytic L_target={L_TARGET_AN*1e3:.3f} mm")
    # Pick an n_steps that's an exact multiple of a √n_steps-class
    # checkpoint K so the scan body can use segmented checkpointing.
    # checkpoint=True alone only does per-step rematerialisation; the
    # scan still keeps every step's carry, so peak GPU memory grows
    # linearly with n_steps and a 954 K-cell forward + value_and_grad
    # OOMs on a 24 GB RTX 4090.  Segmented checkpointing brings memory
    # back to O(√n_steps · |carry|).
    period = 1.0 / float(sim._freq_max)
    n_steps_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
    K_segments = max(8, int(math.isqrt(n_steps_raw)))
    n_steps_use = ((n_steps_raw + K_segments - 1) // K_segments) * K_segments
    print(f"Steps: n_steps={n_steps_use} ({K_segments} segments × "
          f"{n_steps_use // K_segments} steps each); "
          f"raw={n_steps_raw} → rounded up to be divisible by K_segments")

    def s21_at_f_target(L_stub):
        occ = build_stub_occ(grid, trace_y_hi, L_stub)
        fr = sim.forward(
            pec_occupancy_override=occ,
            n_steps=n_steps_use,
            checkpoint_segments=K_segments,
            skip_preflight=True,
        )
        # Assemble plane-integrated V phasors and I phasor for each port,
        # then call the canonical N-probe least-squares extractor.
        freqs_arr = f_target_arr
        beta0 = (2.0 * jnp.pi * freqs_arr * jnp.sqrt(jnp.asarray(EPS_EFF, dtype=jnp.float32))
                 / jnp.asarray(C0, dtype=jnp.float32))
        x_probes = jnp.array([0.0, d_set.delta, 2.0 * d_set.delta], dtype=jnp.float32)
        v_d = jnp.stack([
            _v_from_plane(fr, d_set.ez1_name, d_set),
            _v_from_plane(fr, d_set.ez2_name, d_set),
            _v_from_plane(fr, d_set.ez3_name, d_set),
        ], axis=-1)  # (n_freqs, 3)
        i1_d = _i_from_plane(fr, d_set.hy_name, d_set)
        v_p = jnp.stack([
            _v_from_plane(fr, p_set.ez1_name, p_set),
            _v_from_plane(fr, p_set.ez2_name, p_set),
            _v_from_plane(fr, p_set.ez3_name, p_set),
        ], axis=-1)  # (n_freqs, 3)
        i1_p = _i_from_plane(fr, p_set.hy_name, p_set)
        res_d = extract_msl_nprobe(v_d, x_probes, i1_d, beta0)
        res_p = extract_msl_nprobe(v_p, x_probes, i1_p, beta0)
        # S21 = alpha_passive / alpha_driven (forward wave amplitude ratio)
        s21 = res_p["alpha"] / (res_d["alpha"] + 1e-30)
        return s21[0]

    def cost_from_latent(latent):
        L_stub = L_MIN + (L_MAX - L_MIN) * jax.nn.sigmoid(latent)
        s21 = s21_at_f_target(L_stub)
        return jnp.abs(s21) ** 2

    def latent_from_L(L: float) -> float:
        u = (L - L_MIN) / (L_MAX - L_MIN)
        u = max(min(u, 0.999), 0.001)
        return float(math.log(u / (1.0 - u)))

    def latent_to_L(latent):
        return L_MIN + (L_MAX - L_MIN) * jax.nn.sigmoid(latent)

    # ---- Best-of multi-start Adam optimisation ----
    # MULTI-START IS REQUIRED: the [4, 12] mm cost surface is multimodal,
    # so a single-start Adam can settle in the below-band longer-stub
    # valley (~9.5 mm) instead of the in-band global λ/4 notch (~7.0 mm).
    # Seeds span the band so one lands in each physical branch — the
    # global basin's catchment and the 9.5 mm secondary trap.
    print("\n" + "=" * 70)
    print(f"Best-of multi-start Adam optimisation (N_START={N_START})")
    print("=" * 70)
    # Band-spanning seeds.  NO seed is pinned exactly at the global minimum
    # (7.0 mm) — that would make G2 self-fulfilling ("seeded at the
    # answer").  5.5 and 6.5 mm both sit on the monotone descending wall of
    # the global basin (brute scan: 5.5→7.31e-3, 6.5→1.17e-3, 7.0→2.58e-5)
    # so Adam must actually DESCEND into 7.0 mm, while 9.5 mm probes the
    # secondary trap.  Extra starts (N_START>3) fill the band uniformly.
    L_seeds_mm = [5.5, 6.5, 9.5]
    if N_START > len(L_seeds_mm):
        extra = np.linspace(L_MIN * 1e3 + 0.5, L_MAX * 1e3 - 0.5,
                            N_START - len(L_seeds_mm))
        L_seeds_mm = L_seeds_mm + [float(x) for x in extra]
    L_seeds_mm = L_seeds_mm[:N_START]
    latent_inits = [latent_from_L(Lmm * 1e-3) for Lmm in L_seeds_mm]
    print("  seeds (mm): " + ", ".join(f"{Lmm:.2f}" for Lmm in L_seeds_mm))

    t_total = time.time()
    L_opt, cost_opt, best_history, all_histories = _multistart_adam(
        cost_fn=cost_from_latent,
        latent_inits=latent_inits,
        n_iters=N_ITERS,
        lr=LR,
        max_dL_per_step=MAX_DL,
        latent_to_L=latent_to_L,
    )
    # Print every start's trace, marking the winner.
    for h in all_histories:
        tag = " ← BEST" if h["start"] == best_history["start"] else ""
        seed_mm = L_seeds_mm[h["start"]]
        print(f"  start {h['start']} (seed {seed_mm:.2f}mm):{tag}")
        for it in range(len(h["db"])):
            print(f"    iter {it:3d}  L={h['L'][it]:6.3f}mm  "
                  f"|S21|²={h['cost'][it]:.4e}  S21={h['db'][it]:+6.1f}dB")
        # Show final AND best-seen — they differ when Adam overshoots a
        # sharp null (the kept optimum is the BEST iterate, not the final).
        print(f"    final     L={h['L_final']*1e3:6.3f}mm  "
              f"|S21|²={h['cost_final']:.4e}   "
              f"best L={h['L_best']*1e3:6.3f}mm  |S21|²={h['cost_best']:.4e}")
    # `history` = the WINNING trajectory (used for G1 init→best cost drop
    # and the convergence plot).
    history = best_history
    _overshot = abs(best_history["L_final"] - best_history["L_best"]) > 1e-4
    print(f"\nMulti-start Adam done in {time.time() - t_total:.1f}s — "
          f"best start {best_history['start']} (seed "
          f"{L_seeds_mm[best_history['start']]:.2f}mm) → "
          f"L_opt={L_opt*1e3:.3f} mm (BEST-iterate),  |S21|²={cost_opt:.4e}"
          + (f"  [overshot to L_final={best_history['L_final']*1e3:.3f}mm; "
             f"kept best]" if _overshot else ""))

    # ---- Brute-force scan ----
    print("\n" + "=" * 70)
    print("Reference: brute-force scan via the same JAX extractor")
    print("=" * 70)
    L_scan = np.linspace(L_MIN, L_MAX, N_SCAN)
    scan_costs = np.zeros(N_SCAN)
    t0 = time.time()
    for i, L in enumerate(L_scan):
        s21 = s21_at_f_target(jnp.asarray(L, dtype=jnp.float32))
        scan_costs[i] = float(jnp.abs(s21) ** 2)
        db = 10.0 * math.log10(max(scan_costs[i], 1e-12))
        print(f"  L={L*1e3:6.3f} mm  |S21|²={scan_costs[i]:.4e}  S21={db:+6.1f}dB")
    i_min = int(np.argmin(scan_costs))
    L_ref = float(L_scan[i_min])
    print(f"\nScan done in {time.time()-t0:.1f}s  →  L_ref={L_ref*1e3:.3f} mm")

    # ---- Cross-solver gate: imperative compute_msl_s_matrix at L_opt ----
    print("\n" + "=" * 70)
    print("Cross-solver gate: imperative compute_msl_s_matrix at L_opt")
    print("=" * 70)
    sim_imp, _, trace_y_hi_imp, _, _ = build_sim(f_target_arr)
    # add a HARD-PEC stub Box of length L_opt to sim_imp
    stub_x_centre = LX / 2.0
    stub_x_lo = stub_x_centre - W_TRACE / 2.0
    stub_x_hi = stub_x_centre + W_TRACE / 2.0
    sim_imp.add(Box((stub_x_lo, trace_y_hi_imp, H_SUB),
                    (stub_x_hi, trace_y_hi_imp + L_opt, H_SUB + DX)),
                material="pec")
    # restore default both-port-driven for compute_msl_s_matrix
    object.__setattr__(sim_imp._msl_ports[1], "excite", True)
    t0 = time.time()
    res = sim_imp.compute_msl_s_matrix(n_freqs=80, num_periods=20.0)
    t_imp = time.time() - t0
    f_imp = np.asarray(res.freqs)
    s21_imp = np.asarray(res.S[1, 0, :])
    db_imp = 20 * np.log10(np.abs(s21_imp) + 1e-30)
    i_notch = int(np.argmin(db_imp))
    f_notch_imp = float(f_imp[i_notch])
    depth_imp = float(db_imp[i_notch])
    print(f"  imperative S-matrix done in {t_imp:.1f}s")
    print(f"  imperative notch: f={f_notch_imp/1e9:.3f} GHz  "
          f"|S21|={depth_imp:+.1f} dB")
    f_err = abs(f_notch_imp - F_TARGET) / F_TARGET * 100
    print(f"  Δf vs f_target=({F_TARGET/1e9:.2f} GHz): {f_err:.2f} %")

    # ---- Acceptance ----
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    L_err_ref = abs(L_opt - L_ref) / max(L_ref, 1e-12) * 100
    L_err_an = abs(L_opt - L_TARGET_AN) / L_TARGET_AN * 100
    # G1 is reported on the WINNING trajectory's own init→final cost
    # drop (``history`` == best_history), so it measures the basin that
    # actually produced L_opt, not an arbitrary start.
    cost_init = history["cost"][0]
    cost_drop_db = 10.0 * math.log10(cost_init / max(cost_opt, 1e-12))
    on_rail = L_opt <= L_MIN * 1.005 or L_opt >= L_MAX * 0.995
    # The gates avoid mixing an extractor metric with a mesh-staircase
    # metric.  ``L_TARGET_AN`` is the Hammerstad closed-form (ε_eff=2.869),
    # whereas FDTD on 2 substrate cells lands the imperative notch at a
    # freq ~10-12 % off because the staircased ε_eff is biased.  Gating
    # Adam against the analytic ``L_TARGET_AN`` would fold that mesh bias
    # into the AD-pipeline check, so the gates decouple the two:
    #
    #   * G1  cost descent ≥ 0.3 dB — verifies the AD pipeline does
    #     descend (cost landscape near a partial-notch minimum can be
    #     shallow on a coarse mesh; the meaningful test is ‘decreasing’
    #     not a fixed dB number).  Reported on the winning start.
    #   * G2  best-basin multi-start ``L_opt`` lands in the brute-scan
    #     GLOBAL-min grid cell (|ΔL| ≤ brute grid spacing) — i.e. Adam
    #     escaped the trap and reached the global λ/4 notch (~7.0 mm), not
    #     the below-band longer-stub valley (~9.5 mm, ~290× shallower).
    #     MULTI-START IS REQUIRED: the [4, 12] mm cost surface is
    #     physically MULTIMODAL, so single-start Adam from 9.5 mm settles
    #     in the secondary valley (NOT an extractor or AD bug; AD is
    #     sign-correct + device-independent).
    #     The stricter "L_opt within 1 % of L_ref" is only an
    #     informational diagnostic, not a gate.  The 7.0 mm notch is a
    #     FLAT-bottomed resonant null (brute 7.004→2.54e-5, 7.122→2.33e-5,
    #     both −46 dB), so sub-1 % (≤0.07 mm) L-precision is below the
    #     flat-null width (~0.12 mm) AND the brute grid spacing (0.5 mm) —
    #     it measures grid-quantization noise, not convergence quality.
    #     Best-iterate Adam reaches the −46 dB floor; the basin gate above
    #     + the G3 depth check are the physically meaningful success
    #     criteria.  The ≤1 % number is still printed below for the record.
    #   * G3  imperative-cross-solver notch depth ≤ -15 dB — verifies
    #     a real (i.e. FDTD-confirmed) deep notch exists at the
    #     Adam-found ``L_opt`` (independently extracted by the
    #     validated ``compute_msl_s_matrix`` path).
    #   * G4  L_opt strictly interior to ``[L_MIN, L_MAX]``.
    #
    # ``L_TARGET_AN`` and the imperative-notch frequency vs ``F_TARGET``
    # are still printed for diagnostics, but they live below the
    # gates as informational deltas — the ε_eff staircase that drives
    # them is a property of the chosen mesh, not of this demo's
    # AD pipeline.
    brute_dL_mm = (L_MAX - L_MIN) * 1e3 / (N_SCAN - 1)  # brute grid spacing
    L_err_ref_mm = abs(L_opt - L_ref) * 1e3
    g1 = cost_drop_db >= 0.3
    g2 = L_err_ref_mm <= brute_dL_mm  # basin: within the brute L-resolution
    g3 = depth_imp <= -15.0
    g4 = not on_rail
    print(f"  G1  Adam cost ↓ ≥ 0.3 dB:                "
          f"{cost_drop_db:.2f} dB  ({'PASS' if g1 else 'FAIL'})")
    print(f"  G2  L_opt in brute global-min basin:     "
          f"|ΔL|={L_err_ref_mm:.3f} ≤ grid {brute_dL_mm:.3f}mm  "
          f"({'PASS' if g2 else 'FAIL'})")
    print(f"  G3  Imperative notch depth ≤ -15 dB:     "
          f"{depth_imp:+.1f} dB  ({'PASS' if g3 else 'FAIL'})")
    print(f"  G4  L_opt strictly interior:             "
          f"{L_opt*1e3:.3f} mm  ({'PASS' if g4 else 'FAIL'})")
    all_ok = g1 and g2 and g3 and g4
    print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}")
    # Informational diagnostics — NOT gates (flat-null / mesh-staircase
    # properties, not AD-pipeline quality; see the G2 note above).
    print(f"  [info] L_opt={L_opt*1e3:.3f}mm  L_ref={L_ref*1e3:.3f}mm  "
          f"strict L-match={L_err_ref:.2f}%  "
          f"(flat −46 dB null: sub-1% L is below grid/null resolution, "
          f"not a convergence defect)")
    print(f"  [info] L_target(an)={L_TARGET_AN*1e3:.3f}mm  "
          f"(ε_eff staircase Δ ≈ {L_err_an:.1f}% on this mesh)")
    print(f"  [info] imperative notch f={f_notch_imp/1e9:.3f} GHz "
          f"(target {F_TARGET/1e9:.2f} GHz, Δ {f_err:.1f}%; mesh-staircase)")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    iters = np.arange(len(history["db"]))

    ax = axes[0]
    ax.semilogy(L_scan * 1e3, scan_costs, "k-o", lw=1.6, ms=5,
                label="brute-force scan (JAX extr.)")
    # Overlay EVERY multi-start trajectory so the visual witness shows
    # which basin each seed fell into; the winner is drawn bold.
    for h in all_histories:
        is_best = h["start"] == best_history["start"]
        ax.plot(h["L"], h["cost"],
                color=("b" if is_best else "0.6"),
                marker=".", lw=(1.6 if is_best else 0.9),
                alpha=(0.9 if is_best else 0.5),
                label=("Adam path (BEST basin)" if is_best else None))
    # Mark each seed (init basin) and the chosen global L_opt.
    for s, Lmm in enumerate(L_seeds_mm):
        ax.axvline(Lmm, color="m", ls="-", lw=0.6, alpha=0.35,
                   label=("multi-start seeds" if s == 0 else None))
    ax.axvline(L_TARGET_AN * 1e3, color="g", ls="--", alpha=0.6,
               label=f"L_target_an={L_TARGET_AN*1e3:.2f}mm")
    ax.axvline(L_opt * 1e3, color="r", ls=":", alpha=0.9, lw=1.8,
               label=f"L_opt (global)={L_opt*1e3:.2f}mm")
    ax.set_xlabel("L_stub (mm)")
    ax.set_ylabel(f"|S21(f={F_TARGET/1e9:.1f} GHz)|² (JAX extractor)")
    ax.set_title("Multimodal cost vs L_stub — multi-start basins")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, history["db"], "b-o", lw=1.4, ms=4)
    ax.set_xlabel("Adam iter"); ax.set_ylabel("|S21(f_target)| dB (JAX)")
    ax.set_title(f"Adam convergence (best start "
                 f"{best_history['start']}, seed "
                 f"{L_seeds_mm[best_history['start']]:.1f}mm)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(f_imp / 1e9, db_imp, "k-", lw=1.5,
            label=f"imperative @ L_opt={L_opt*1e3:.2f}mm")
    ax.axvline(F_TARGET / 1e9, color="r", ls=":", alpha=0.8,
               label=f"f_target={F_TARGET/1e9:.2f} GHz")
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("|S21| dB (imperative)")
    ax.set_title("Cross-solver gate — imperative notch at L_opt")
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        f"MSL stub notch tuning — density-PEC reformulation + JAX N-probe extractor\n"
        f"L_opt={L_opt*1e3:.2f}mm vs L_target_an={L_TARGET_AN*1e3:.2f}mm   "
        f"imperative notch @ {f_notch_imp/1e9:.2f}GHz ({depth_imp:+.1f}dB) — "
        f"{'PASS' if all_ok else 'FAIL'}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "msl_stub_notch_tuning.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nWrote: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

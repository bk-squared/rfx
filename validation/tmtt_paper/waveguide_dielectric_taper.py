"""Broadband WR-90 dielectric-taper matching via differentiable modal S-matrix.

Physics
-------
A WR-90 (X-band) rectangular waveguide is terminated by a high-permittivity
dielectric load (eps_r = 9) that fills the downstream half of the guide and
runs into the absorbing boundary (a quasi-semi-infinite eps_r-filled guide,
i.e. a matched dielectric load). The bare vacuum -> eps_r=9 interface reflects
strongly and broadbandly (|S11| ~ -4.5 dB across 8-12 GHz). This is the
textbook impedance-taper matching problem (Pozar, *Microwave Engineering*,
Ch. 5): match a guide to a mismatched termination over a band.

The design variable is a graded dielectric TAPER of ``N_SECTIONS`` axial
permittivity sections placed in the vacuum guide directly in front of the
eps_r=9 region. Each section's permittivity is a continuous design variable,
sigmoid-bounded to [1, EPS_LOAD], applied through ``eps_override`` (only eps_r
is touched; the WR-90 guide walls are the DOMAIN boundary condition, not a
conductor body, so ``eps_override`` cannot reach them at all). The objective minimizes the
band-averaged reflected power <|S11|^2> over the X-band, computed via the
public differentiable modal S-matrix ``Simulation.compute_waveguide_s_matrix``.
Reverse-mode AD (``jax.grad``) flows through the full FDTD solve + modal
S-extraction; Adam descent lowers the broadband |S11|.

Which guide each lane actually solves
-------------------------------------
Issue #1100. The declared WR-90 walls are 22.86 x 10.16 mm; a mesh that does
not divide them rasterizes a wider guide, and the run solves THAT.

  SMOKE (default), dx = 1.27 mm  ->  22.860 x 10.160 mm, exactly WR-90.
      gcd(22.86, 10.16) = 2.54 mm, so 1.27 mm divides both walls: 18 x 8
      transverse cells, no remainder. ``build_sim`` asserts the realized
      extent and preflight emits no port_aperture_snap row.
  paper (SMOKE=0), dx = 0.5 mm   ->  23.000 x 10.500 mm, +0.61 % / +3.35 %.
      NOT WR-90. Disclosed rather than re-meshed: see the QUOTE-REALIZED
      block beside the constants. The production point in the paper,
      dx = 0.25 mm, is not commensurate either (22.86/0.25 = 91.44).

Every analytic reference this file computes -- ``F_CUTOFF_TE10`` and anything
derived from it -- comes from ``A_WG_REALIZED``/``B_WG_REALIZED``, so it
always describes the guide the solver built.

Full-resolution result
----------------------
At dx = 0.5 mm the optimized 30-section taper reaches a band-mean
reflection of -26.7 dB, and -38.0 dB re-optimized at the production
resolution dx = 0.25 mm (cf. a discretized Klopfenstein taper of the same
electrical length at -36.6 dB at dx = 0.25 mm). Halving dx (~4x more cells)
lowers the Yee-dispersion reflection floor, deepening the achievable match
on the same layout. At a comparable coarse-grid solve budget, particle-swarm
and genetic search trail the gradient by at least 11.6 dB.

Those four numbers were measured on the 23.000 mm-wide guide those two meshes
rasterize, not on WR-90, and they are quoted here unchanged. A taper is a
broadband match to whatever guide it sits in, so they are not wrong -- they
belong to a structure 0.61 % wider than the one the title names. Re-measuring
them on a commensurate mesh is a GPU-scale re-run and a PI decision: #1122
carries it (refs #1100, #825). Not done here.

The reverse-mode tape over the full-resolution scan (~12-14k steps) is made
affordable by ``checkpoint_segments``: segmented gradient checkpointing reduces
peak reverse memory from O(n_steps) to ~O(sqrt(n_steps)) at ~2x backward cost.
This example wires the same kwarg so the SMOKE and full paths share one code
route.

SMOKE mode
----------
``SMOKE=1`` (default) uses a coarse commensurate grid (dx = 1.27 mm), a short
integration, and a handful of Adam steps so the example runs in ~1-3 min on
CPU. It still does real reverse-mode AD through the S-matrix and lowers the
broadband |S11|. Its numbers are a smoke signal, not a result: at 18 x 8
transverse cells the Yee-dispersion floor is well above the paper lane's.
``SMOKE=0`` switches to the paper's dx = 0.5 mm / 30-section / 120-iteration
settings (GPU required in practice for the full-resolution scan).

Run
---
  SMOKE=1 JAX_PLATFORMS=cpu python validation/tmtt_paper/waveguide_dielectric_taper.py
  SMOKE=0 python validation/tmtt_paper/waveguide_dielectric_taper.py   # paper (GPU)

Output: initial vs optimized band-averaged |S11| in dB, plus a figure of the
optimized permittivity profile and the |S11|(f) curves.
"""

from __future__ import annotations

import os
import time
import warnings

# complex64 scan carry (the reverse-mode S-matrix tape); keep x32.
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

from validation.crossval.comparators import realized_conductors as RC

C0 = 2.998e8

# --------------------------------------------------------------------------
# WR-90 geometry (X-band standard rectangular waveguide).
# --------------------------------------------------------------------------
A_WG = 22.86e-3          # broad-wall width (m), declared WR-90 standard
B_WG = 10.16e-3          # narrow-wall height (m), declared WR-90 standard

EPS_LOAD = 9.0           # high-permittivity matched dielectric termination

SMOKE = os.environ.get("SMOKE", "1") != "0"

if SMOKE:
    # Coarse CPU smoke: a single reverse-mode FDTD grad is a few tens of
    # seconds on this small domain, so the run is sized for 1-3 min total.
    #
    # COMMENSURATE MESH (issue #1100). gcd(22.86 mm, 10.16 mm) = 2.54 mm, so
    # dx = 1.27 mm puts BOTH walls exactly on the lattice: 22.86/1.27 = 18
    # and 10.16/1.27 = 8, no remainder. This lane therefore solves WR-90
    # itself rather than a guide rounded up to the next cell, and it is
    # coarser than the 1.0 mm mesh it replaces (18 x 8 transverse cells
    # instead of 23 x 11), so it is also faster. Measured build-only, the
    # realized aperture is 22.8600 x 10.1600 mm and preflight emits no
    # port_aperture_snap row; ``build_sim`` asserts that rather than
    # trusting this comment.
    DX_M = 1.27e-3
    # Every axial length is an exact cell count, so none of them is silently
    # snapped by ``position_to_index``. The old values are in the trailing
    # comments; each moved by less than one cell of the OLD mesh.
    CPML_LAYERS = 16
    DOMAIN_X = 71 * DX_M      #  90.170 mm (was 90.0)
    TAPER_X0 = 32 * DX_M      #  40.640 mm (was 40.0), taper start (vacuum side)
    FILL_X = 48 * DX_M        #  60.960 mm (was 60.0), eps_load fills from here
    PORT_OFFSET = 20 * DX_M   #  25.400 mm (was 25.0)
    REF_OFFSET = 26 * DX_M    #  33.020 mm (was 33.0)
    FREQS_HZ = np.linspace(8.5e9, 11.5e9, 4)
    N_SECTIONS = 12
    NUM_PERIODS = 60
    N_ADAM = 6
    LR = 0.3
    CHECKPOINT_SEGMENTS = 4   # tiny K for the short smoke scan
else:
    # Paper resolution (dx = 0.5 mm, 80 mm / 30-section taper). The
    # full-resolution dx = 0.25 mm run in the paper reaches -38.0 dB; that grid
    # is heavier still — start from this and halve dx for the production point.
    DX_M = 0.5e-3
    DOMAIN_X = 0.220
    CPML_LAYERS = 24
    FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
    N_SECTIONS = 30
    NUM_PERIODS = 120
    N_ADAM = 120
    LR = 0.2
    CHECKPOINT_SEGMENTS = 120   # ~sqrt(n_steps) for the ~12-14k-step scan
    TAPER_X0 = 0.060
    FILL_X = 0.140              # 80 mm taper (TAPER_X0 -> FILL_X), eps9 beyond
    PORT_OFFSET = 0.030
    REF_OFFSET = 0.040

# QUOTE-REALIZED (issue #1100, the cv11 convention at
# validation/crossval/11_waveguide_port_wr90.py): the walls this dx actually
# rasterizes, ``ceil(declared/dx)*dx``, not the declared WR-90 numbers. Every
# analytic reference below is derived from THESE, so a number this file prints
# always belongs to the guide the solver builds.
#
#   dx = 1.27 mm (SMOKE)  ->  22.860 x 10.160 mm   exactly WR-90
#   dx = 0.50 mm (paper)  ->  23.000 x 10.500 mm   +0.61 % / +3.35 %
#
# The paper lane is NOT re-meshed here. dx = 0.635 mm or 0.3175 mm would put
# it on the lattice, but every headline number that lane produces was measured
# at 0.5 mm (and the production point at 0.25 mm) on GPU-scale runs, so moving
# it is a PI decision, not a drive-by edit. Recorded, not done: #1122 (refs
# #1100, #825).
A_WG_REALIZED = float(np.ceil(A_WG / DX_M)) * DX_M
B_WG_REALIZED = float(np.ceil(B_WG / DX_M)) * DX_M
# TE10 cutoff of the guide actually solved. At dx = 1.27 mm this IS the
# declared WR-90 cutoff (6.5573 GHz) because the mesh is commensurate; at
# dx = 0.5 mm it is 6.5174 GHz against the declared 6.5573 GHz. The port's own
# discrete mode solve lands a little below either (Yee dispersion on a finite
# cell count): 6.5488 GHz at 1.27 mm, 6.5160 GHz at 0.5 mm.
F_CUTOFF_TE10 = C0 / (2.0 * A_WG_REALIZED)

F0_HZ = float(FREQS_HZ.mean())
BANDWIDTH_REL = 0.45


# --------------------------------------------------------------------------
# WR-90 two-port simulation: PEC y/z walls, CPML along x.
# --------------------------------------------------------------------------
def build_sim() -> Simulation:
    boundary = BoundarySpec(
        x=Boundary(lo="cpml", hi="cpml"),
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    )
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.05,
        domain=(DOMAIN_X, A_WG, B_WG),
        dx=DX_M,
        boundary=boundary,
        cpml_layers=CPML_LAYERS,
    )
    sim.add_waveguide_port(
        PORT_OFFSET, direction="+x",
        freqs=jnp.asarray(FREQS_HZ), f0=F0_HZ, bandwidth=BANDWIDTH_REL,
        waveform="modulated_gaussian", reference_plane=REF_OFFSET, name="left",
    )
    sim.add_waveguide_port(
        DOMAIN_X - PORT_OFFSET, direction="-x",
        freqs=jnp.asarray(FREQS_HZ), f0=F0_HZ, bandwidth=BANDWIDTH_REL,
        waveform="modulated_gaussian",
        reference_plane=DOMAIN_X - REF_OFFSET, name="right",
    )
    # Build-time (no solve) control: this fixture realizes NO conductor
    # body at all — the WR-90 walls are the domain boundary, which the
    # contract does not move (#931 §1.8). That makes the bit-identity
    # claim for this case checkable rather than assumed, and it is the
    # premise make_eps_builder's docstring relies on.
    RC.assert_no_conductor(sim, label="WR-90 taper (boundary PEC only)")
    _assert_realized_guide(sim)
    return sim


def _assert_realized_guide(sim) -> None:
    """The guide this lane SOLVES is the one A_WG_REALIZED names (#1100).

    ``A_WG_REALIZED`` / ``B_WG_REALIZED`` above are a PREDICTION
    (``ceil(declared/dx)*dx``) and every analytic reference in this file is
    derived from them, so the prediction has to be checked against what the
    grid built or the QUOTE-REALIZED block is just another comment. Cheap and
    build-only: the transverse walls are the domain faces (PEC, no absorber
    pad), so the realized wall-to-wall extent is the interior cell count times
    the cell size on that axis.

    On the SMOKE lane this also pins the commensurate mesh itself: 22.86 and
    10.16 mm both divide by 1.27 mm, so realized == declared and the run
    solves WR-90. If someone changes DX_M to a value that does not divide
    them, this raises here instead of silently solving a wider guide -- which
    is the defect #1100 was filed for.
    """
    grid = sim._build_grid()
    for axis, (n_nodes, declared, realized) in enumerate((
        (grid.ny, A_WG, A_WG_REALIZED),
        (grid.nz, B_WG, B_WG_REALIZED),
    ), start=1):
        pad = grid.axis_pads[axis]
        built = (n_nodes - 1 - 2 * pad) * DX_M
        assert abs(built - realized) < 1e-12, (
            f"{'yz'[axis - 1]}: realized guide {built * 1e3:.4f} mm is not the "
            f"{realized * 1e3:.4f} mm this file's analytic references are "
            f"derived from (declared {declared * 1e3:.4f} mm, dx = "
            f"{DX_M * 1e3:.4f} mm). Re-derive A_WG_REALIZED/B_WG_REALIZED."
        )


def _x_index(grid, x_m: float) -> int:
    return int(grid.position_to_index((float(x_m), A_WG / 2.0, B_WG / 2.0))[0])


def design_layout(grid) -> dict:
    """Axial cell-index edges of the taper sections + the eps_load fill start."""
    i_t0 = _x_index(grid, TAPER_X0)
    i_fill = _x_index(grid, FILL_X)
    edges = np.unique(
        np.round(np.linspace(i_t0, i_fill, N_SECTIONS + 1)).astype(int)
    )
    return dict(i_t0=i_t0, i_fill=i_fill, edges=edges, n_sec=edges.size - 1)


def make_eps_builder(grid, layout):
    """Return a differentiable ``eps_from_theta(theta)`` permittivity builder.

    Each taper section gets eps_r = 1 + (EPS_LOAD-1) * sigmoid(theta_s), so the
    permittivity stays bounded in [1, EPS_LOAD] for any real ``theta``. Beyond
    the taper, the guide is filled with eps_r = EPS_LOAD (the matched load).
    Only eps_r is set. The guide walls survive the override for a stronger
    reason than an ordering argument: they are the domain boundary
    condition (``BoundarySpec`` y/z = pec), which the lattice ownership
    contract fences off from body PEC entirely (#931 §1.8) — there is no
    conductor body in this fixture for an eps array to overwrite. The
    earlier wording ("the override is applied after the PEC fold inside
    compute_waveguide_s_matrix") described the pre-#931 waveguide lane,
    which folded pec_mask CELLS into sigma = 1e10 before stepping. That
    fold is gone; the lane now applies the realized PEC edges like every
    other lane. ``assert_no_conductor`` in ``build_sim`` states the
    premise the argument rests on, instead of asserting it in prose.
    """
    edges = layout["edges"]
    n_sec = layout["n_sec"]
    i_fill = layout["i_fill"]
    shape = grid.shape

    def eps_from_theta(theta):
        eps_sec = 1.0 + (EPS_LOAD - 1.0) * jax.nn.sigmoid(theta)
        er = jnp.ones(shape, dtype=jnp.float32)
        for s in range(n_sec):
            lo, hi = int(edges[s]), int(edges[s + 1])
            if hi > lo:
                er = er.at[lo:hi, :, :].set(eps_sec[s])
        er = er.at[i_fill:, :, :].set(EPS_LOAD)
        return er

    return eps_from_theta


def _checkpoint_n_steps(grid, num_periods: float, k: int):
    """Round the auto n_steps UP to a multiple of ``k``.

    ``checkpoint_segments=k`` must divide ``n_steps`` exactly (the runner
    rejects non-divisors; padding the recorded series would shift the modal
    V/I DFT windows). Rounding up only lengthens the integration window, which
    is always safe for the finite-energy rectangular full-record DFT.
    """
    auto = int(grid.num_timesteps(num_periods=num_periods))
    k = max(1, int(k))
    n_steps = ((auto + k - 1) // k) * k
    return n_steps, k


def _supports_checkpoint_segments(sim) -> bool:
    """``checkpoint_segments`` landed on the uniform waveguide AD path in PR
    #125/#172. Probe the signature so the example also runs against older pins
    (it simply pays full reverse memory there)."""
    import inspect
    params = inspect.signature(sim.compute_waveguide_s_matrix).parameters
    return "checkpoint_segments" in params


def make_objective(sim, eps_from_theta, grid):
    """Band-averaged reflected power <|S11|^2> as a scalar of ``theta``."""
    n_steps, k = _checkpoint_n_steps(grid, NUM_PERIODS, CHECKPOINT_SEGMENTS)
    use_ckpt = _supports_checkpoint_segments(sim)
    # Segmented reverse checkpointing keeps the full-resolution tape in memory;
    # pass it whenever main exposes it.
    ckpt_kw = {"checkpoint_segments": k} if use_ckpt else {}
    print(f"[taper] AD scan: n_steps={n_steps} "
          f"checkpoint_segments={k if use_ckpt else 'unavailable'} "
          f"(reverse-mem factor ~{k + n_steps // k} vs {n_steps} linear)")

    def s11(theta):
        with warnings.catch_warnings():
            # Mute only the expected normalize=False Yee-dispersion advisory
            # (the documented-correct choice for strong-reflector |S11|), so any
            # other warning still surfaces during the AD scan.
            warnings.filterwarnings("ignore", message=".*normalize=False.*")
            r = sim.compute_waveguide_s_matrix(
                n_steps=n_steps,
                normalize=False,            # |S11| of a strong reflector
                eps_override=eps_from_theta(theta),
                **ckpt_kw,
            )
        idx = {n: i for i, n in enumerate(r.port_names)}
        left = idx["left"]
        return jnp.abs(r.s_params[left, left, :])

    def loss(theta):
        return jnp.mean(s11(theta) ** 2)

    return s11, loss


def band_avg_db(s11_mag) -> float:
    return float(20.0 * np.log10(np.mean(np.asarray(s11_mag))))


def main() -> int:
    t0 = time.time()
    print(f"[taper] mode={'SMOKE' if SMOKE else 'FULL'} dx={DX_M*1e3:.2f}mm "
          f"freqs={len(FREQS_HZ)} eps_load={EPS_LOAD} N_req={N_SECTIONS}")

    sim = build_sim()
    grid = sim._build_grid()
    layout = design_layout(grid)
    n_sec = layout["n_sec"]
    print(f"[taper] grid.shape={grid.shape} taper cells=({layout['i_t0']},"
          f"{layout['i_fill']}) n_sections={n_sec} "
          f"taper_len={ (FILL_X-TAPER_X0)*1e3:.0f}mm")

    eps_from_theta = make_eps_builder(grid, layout)
    s11_fn, loss_fn = make_objective(sim, eps_from_theta, grid)

    # Initial design: a flat mid-permittivity block (sigmoid(0)=0.5 -> eps~5),
    # deliberately suboptimal so the optimizer has real work to do.
    theta = jnp.zeros(n_sec, dtype=jnp.float32)
    s11_init = np.asarray(s11_fn(theta))
    init_db = band_avg_db(s11_init)
    print(f"[taper] initial band-avg |S11| = {init_db:.2f} dB")

    # Adam descent with end-to-end reverse-mode gradients of the S-matrix.
    import optax
    opt = optax.adam(LR)
    state = opt.init(theta)
    value_and_grad = jax.value_and_grad(loss_fn)
    loss_hist = []
    for it in range(N_ADAM):
        v, g = value_and_grad(theta)
        updates, state = opt.update(g, state)
        theta = optax.apply_updates(theta, updates)
        loss_hist.append(float(v))
        print(f"[taper]   iter {it:3d}  <|S11|^2> = {float(v):.5e}  "
              f"({10.0*np.log10(float(v)):.2f} dB)")

    s11_opt = np.asarray(s11_fn(theta))
    opt_db = band_avg_db(s11_opt)

    print("\n" + "=" * 64)
    print("RESULT  (broadband WR-90 dielectric-taper matching)")
    print("=" * 64)
    print(f"  initial   band-avg |S11| : {init_db:8.2f} dB")
    print(f"  optimized band-avg |S11| : {opt_db:8.2f} dB")
    print(f"  improvement              : {init_db - opt_db:8.2f} dB")
    print(f"  wall time                : {time.time()-t0:8.1f} s")
    if SMOKE:
        print("\n  (SMOKE: coarse grid / few iters. The paper run at dx=0.5mm")
        print("   reaches -26.7 dB; at production dx=0.25mm it reaches -38.0 dB,")
        print("   beating a discretized Klopfenstein taper at -36.6 dB.)")

    _save_figure(grid, layout, eps_from_theta, theta, s11_init, s11_opt,
                 loss_hist)
    return 0


def _save_figure(grid, layout, eps_from_theta, theta, s11_init, s11_opt,
                 loss_hist):
    """Optimized permittivity profile + |S11|(f) before/after (optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib is optional for the example
        print(f"[taper] skipping figure (matplotlib unavailable: {exc})")
        return

    f_ghz = FREQS_HZ / 1e9
    eps_grid = np.asarray(eps_from_theta(theta))
    j, k = eps_grid.shape[1] // 2, eps_grid.shape[2] // 2
    x_mm = (np.arange(eps_grid.shape[0]) - grid.pad_x_lo) * DX_M * 1e3

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.0, 3.0))

    ax0.plot(x_mm, eps_grid[:, j, k], drawstyle="steps-mid", color="tab:blue")
    ax0.axvspan((layout["i_fill"] - grid.pad_x_lo) * DX_M * 1e3, x_mm[-1],
                alpha=0.18, color="tab:red", label=f"eps_r={EPS_LOAD:.0f} load")
    ax0.set_xlabel("x (mm)")
    ax0.set_ylabel("eps_r")
    ax0.set_title("optimized taper profile")
    ax0.legend(fontsize=7)

    ax1.plot(f_ghz, 20 * np.log10(np.maximum(s11_init, 1e-6)),
             "o-", color="tab:red", label="initial (flat block)")
    ax1.plot(f_ghz, 20 * np.log10(np.maximum(s11_opt, 1e-6)),
             "^-", color="tab:blue", label="optimized taper")
    ax1.set_xlabel("frequency (GHz)")
    ax1.set_ylabel("|S11| (dB)")
    ax1.set_title("broadband return loss")
    ax1.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "waveguide_dielectric_taper.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[taper] figure -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())

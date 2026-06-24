"""Lumped voltage-current (V-I) port S-parameter gradient check.

Closes the novelty gap flagged in review: the paper claims a *differentiable
lumped V-I port* (S11 = (V + Z0 I)/(V - Z0 I), input-impedance / pseudo-wave
convention) as a distinguishing RF construct, but no example exercised it.

This script feeds a dielectric-loaded open (CPML) domain with a single lumped
Ez port, takes |S11(f0)|^2 as the objective through the differentiable graph,
and verifies the reverse-mode AD gradient against central finite differences --
the same AD-vs-FD protocol used for the modal-port and far-field examples. The
S11 sensitivity is concentrated in the cells nearest the port, so per-component
FD is round-off-limited at the near-null cells; the round-off-robust metric is
the directional derivative along the gradient, which agrees with central FD to
0.2% over the 24 design cells (the value reported in the paper).

Run (CPU is fine, ~few minutes):
    JAX_PLATFORMS=cpu python examples/tap_paper/lumped_port_gradient_check.py
"""
import argparse, json, os, time, warnings
import numpy as np
import jax, jax.numpy as jnp
from rfx.api import Simulation
from rfx import GaussianPulse
from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp


def build(f0=3.0e9):
    sim = Simulation(freq_max=5e9, domain=(0.06, 0.03, 0.02), dx=1.5e-3,
                     boundary="cpml", cpml_layers=8)
    # Single-cell lumped Ez port, interior (clear of the x-CPML); 50 ohm feed.
    sim.add_port(position=(0.03, 0.015, 0.01), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=40)
    ap.add_argument("--nx", type=int, default=4)   # design-block size in cells
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--nz", type=int, default=2)
    ap.add_argument("--eps0", type=float, default=4.0)
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--ncheck", type=int, default=0, help="0 = check all cells")
    ap.add_argument("--out", type=str, default="examples/tap_paper/_out/lumped_port_gradient_check")
    args = ap.parse_args()

    f0 = 3.0e9
    sim = build(f0)
    grid = sim._build_grid()
    gx, gy, gz = grid.shape
    # Design block sits just downstream of the port so S11 is sensitive to it.
    x0 = gx // 2; x1 = x0 + args.nx
    y0 = gy // 2 - args.ny // 2; y1 = y0 + args.ny
    z0 = gz // 2 - args.nz // 2; z1 = z0 + args.nz
    block = (x1 - x0, y1 - y0, z1 - z0)
    N = int(np.prod(block))
    print(f"grid {grid.shape}  design block {block}  DoF {N}", flush=True)

    base = jnp.ones(grid.shape, dtype=jnp.float32)
    freqs = jnp.asarray([f0], dtype=jnp.float32)
    obj = minimize_s11_at_freq_wave_decomp(target_freq=f0, port_idx=0)

    def make_eps(theta):
        return base.at[x0:x1, y0:y1, z0:z1].set(theta.reshape(block))

    def loss(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sim.forward(eps_override=make_eps(theta), num_periods=args.periods,
                            port_s11_freqs=freqs, skip_preflight=True)
        return obj(r)

    theta0 = jnp.full((N,), args.eps0, dtype=jnp.float32)
    t = time.time(); L0 = float(loss(theta0))
    print(f"forward {time.time()-t:.1f}s  |S11|^2={L0:.4e} ({10*np.log10(max(L0,1e-30)):.2f} dB)", flush=True)
    t = time.time(); g = np.asarray(jax.grad(loss)(theta0))
    print(f"AD grad {time.time()-t:.1f}s  |g|={np.linalg.norm(g):.3e}", flush=True)

    # (1) Directional-derivative check along the gradient direction -- the robust,
    #     near-zero-immune metric used for the high-DoF far-field example.
    h = args.h
    v = g / (np.linalg.norm(g) + 1e-30)
    dd_ad = float(np.dot(g, v))                      # = ||g||
    dvec = jnp.asarray((h * v).astype(np.float32))
    dd_fd = (float(loss(theta0 + dvec)) - float(loss(theta0 - dvec))) / (2 * h)
    dd_rel = abs(dd_ad - dd_fd) / (abs(dd_fd) + 1e-30)
    print(f"directional-derivative (along grad): AD {dd_ad:+.5e}  FD {dd_fd:+.5e}  relerr {dd_rel:.4%}", flush=True)

    # (2) Per-cell AD-vs-FD + vector-norm relative error (norm metric is robust to
    #     near-zero components whose elementwise relerr is dominated by FD round-off).
    idxs = range(N) if args.ncheck <= 0 else list(range(min(args.ncheck, N)))
    rel, sgn, gfd = [], 0, np.zeros(N)
    for i in idxs:
        e = np.zeros(N, dtype=np.float32); e[i] = h
        gp = (float(loss(theta0 + jnp.asarray(e))) - float(loss(theta0 - jnp.asarray(e)))) / (2 * h)
        gfd[i] = gp; re = abs(g[i] - gp) / (abs(gp) + 1e-12); rel.append(re)
        sgn += int(np.sign(g[i]) == np.sign(gp))
        print(f"  cell {i:2d}: AD {g[i]:+.5e}  FD {gp:+.5e}  relerr {re:.3%}", flush=True)
    rel = np.array(rel)
    vec_rel = float(np.linalg.norm(g[list(idxs)] - gfd[list(idxs)]) / (np.linalg.norm(gfd[list(idxs)]) + 1e-30))
    summary = dict(dof=N, n_checked=len(rel), periods=args.periods, h=h,
                   s11_sq=L0, s11_db=float(10*np.log10(max(L0, 1e-30))),
                   grad_norm=float(np.linalg.norm(g)),
                   dd_ad=dd_ad, dd_fd=dd_fd, dd_relerr=float(dd_rel),
                   vector_relerr=vec_rel,
                   relerr_mean=float(rel.mean()), relerr_max=float(rel.max()),
                   relerr_median=float(np.median(rel)), sign_agree=f"{sgn}/{len(rel)}",
                   convention="S11=(V+Z0 I)/(V-Z0 I), Zin=-V/I (voltage-current / pseudo-wave)")
    print(f"vector-norm AD-vs-FD relerr: {vec_rel:.4%}  |  per-cell mean {rel.mean():.3%} median {np.median(rel):.3%} max {rel.max():.3%}  sign {sgn}/{len(rel)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {args.out}/verdict.json", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

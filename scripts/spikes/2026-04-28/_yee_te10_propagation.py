"""Yee-core TE10 standing-wave propagation diagnostic.

This is candidate #4 from the 2026-04-28 WR-90 architecture handoff.  It
intentionally bypasses waveguide ports, TFSF, DFT-plane probes, CPML, and the
production S-parameter extractor.  The only moving parts are:

  * analytic TE10 standing-wave initial fields (Ez, Hx, Hy) placed on the
    Yee-staggered rfx arrays,
  * rfx.core.yee.update_h / update_e,
  * PEC enforcement on the transverse waveguide walls and at a downstream
    short.

For each frequency, the script accumulates a gated DFT of Ez on a probe plane
and compares |Ez(y,z,f)| against the analytic TE10 standing-wave profile on
the same Yee grid.  The gate is chosen before any disturbance from the open
x_lo numerical boundary can reach the probe plane; this follows the
"travel-time first" rule from the CPML phantom postmortem.

Run examples:

    # fast smoke (5 representative frequencies)
    python scripts/spikes/2026-04-28/_yee_te10_propagation.py --quick

    # full 21-frequency cv11 band gate
    python scripts/spikes/2026-04-28/_yee_te10_propagation.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx.boundaries.pec import apply_pec_faces, apply_pec_mask
from rfx.core.yee import EPS_0, MU_0, FDTDState, MaterialArrays, update_e, update_h


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_yee_te10"
OUT.mkdir(parents=True, exist_ok=True)

C0 = 299_792_458.0
PEC_FACES = frozenset({"y_lo", "y_hi", "z_lo", "z_hi"})


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@jax.jit
def _yee_step(
    state: FDTDState,
    materials: MaterialArrays,
    pec_mask: jnp.ndarray,
    dt: float,
    dx: float,
) -> FDTDState:
    """One source-free Yee step with the same H→E ordering as rfx.run()."""
    state = update_h(state, materials, dt, dx, periodic=(False, False, False))
    state = update_e(state, materials, dt, dx, periodic=(False, False, False))
    state = apply_pec_faces(state, PEC_FACES)
    state = apply_pec_mask(state, pec_mask)
    return state


def _standing_wave_arrays(
    *,
    f_hz: float,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    a_mode: float,
    x_short: float,
    dt: float,
    amplitude: float = 1.0,
) -> tuple[FDTDState, np.ndarray]:
    """Build analytic TE10 standing-wave fields on rfx's Yee staggering.

    E fields are at integer Yee positions and H fields at the half-step time
    t=-dt/2 expected by the leapfrog H update.
    """
    omega = 2.0 * np.pi * f_hz
    kc = np.pi / a_mode
    k0 = omega / C0
    beta = float(np.sqrt(max(k0 * k0 - kc * kc, 0.0)))
    if beta <= 0.0:
        raise ValueError(f"frequency {f_hz:g} Hz is below TE10 cutoff")

    x_e = np.arange(nx, dtype=np.float64) * dx
    y_e = np.arange(ny, dtype=np.float64) * dx
    z_e = np.arange(nz, dtype=np.float64) * dx
    del z_e  # TE10 is invariant in z; nz only broadcasts the fields.

    # Yee component positions used by rfx's finite differences:
    #   Ez(i,j,k) at x_i, y_j
    #   Hx(i,j,k) at x_i, y_{j+1/2}
    #   Hy(i,j,k) at x_{i+1/2}, y_j
    x_hy = x_e + 0.5 * dx
    y_hx = y_e + 0.5 * dx

    t_e = 0.0
    t_h = -0.5 * dt

    sin_y = np.sin(kc * y_e)
    cos_y_half = np.cos(kc * y_hx)
    sin_x = np.sin(beta * (x_short - x_e))
    cos_x_half = np.cos(beta * (x_short - x_hy))

    active_e = (x_e < x_short).astype(np.float64)
    active_hy = (x_hy < x_short).astype(np.float64)

    ez = (
        amplitude
        * sin_x[:, None, None]
        * sin_y[None, :, None]
        * np.cos(omega * t_e)
        * active_e[:, None, None]
        * np.ones((1, 1, nz), dtype=np.float64)
    )
    hx = (
        -amplitude
        * kc
        / (MU_0 * omega)
        * sin_x[:, None, None]
        * cos_y_half[None, :, None]
        * np.sin(omega * t_h)
        * active_e[:, None, None]
        * np.ones((1, 1, nz), dtype=np.float64)
    )
    hy = (
        -amplitude
        * beta
        / (MU_0 * omega)
        * cos_x_half[:, None, None]
        * sin_y[None, :, None]
        * np.sin(omega * t_h)
        * active_hy[:, None, None]
        * np.ones((1, 1, nz), dtype=np.float64)
    )

    zeros = np.zeros((nx, ny, nz), dtype=np.float32)
    state = FDTDState(
        ex=jnp.asarray(zeros),
        ey=jnp.asarray(zeros),
        ez=jnp.asarray(ez.astype(np.float32)),
        hx=jnp.asarray(hx.astype(np.float32)),
        hy=jnp.asarray(hy.astype(np.float32)),
        hz=jnp.asarray(zeros),
        step=jnp.asarray(0, dtype=jnp.int32),
    )

    pec_mask_np = (x_e[:, None, None] >= x_short).repeat(ny, axis=1).repeat(nz, axis=2)
    return state, pec_mask_np


def _analytic_ez_plane(
    *,
    f_hz: float,
    dx: float,
    ny: int,
    nz: int,
    a_mode: float,
    x_short: float,
    probe_x: float,
    t: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    omega = 2.0 * np.pi * f_hz
    beta = np.sqrt((omega / C0) ** 2 - (np.pi / a_mode) ** 2)
    y = np.arange(ny, dtype=np.float64) * dx
    profile = (
        amplitude
        * np.sin(np.pi * y / a_mode)
        * np.sin(beta * (x_short - probe_x))
        * np.cos(omega * t)
    )
    return profile[:, None] * np.ones((1, nz), dtype=np.float64)


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(arr, dtype=np.float64)))))


def run_frequency(
    *,
    f_hz: float,
    dx: float,
    a_physical: float,
    b_physical: float,
    x_short: float,
    probe_x: float,
    total_cycles: float,
    gate_start_cycles: float,
    boundary_margin_cycles: float,
    use_physical_aperture: bool,
) -> dict:
    """Run one source-free TE10 diagnostic frequency."""
    # rfx's uniform grid uses fencepost nodes.  For the default diagnostic we
    # use the effective grid aperture so that this test isolates Yee update
    # equations rather than the already-refuted physical-wall placement issue.
    ny = int(np.ceil(a_physical / dx)) + 1
    nz = int(np.ceil(b_physical / dx)) + 1
    a_mode = float(a_physical if use_physical_aperture else (ny - 1) * dx)

    nx = int(np.ceil((x_short + 0.08) / dx)) + 1
    dt = dx / (C0 * np.sqrt(3.0)) * 0.99
    probe_i = int(round(probe_x / dx))

    omega = 2.0 * np.pi * f_hz
    beta = float(np.sqrt((omega / C0) ** 2 - (np.pi / a_mode) ** 2))
    vg = C0 * np.sqrt(max(1.0 - (C0 / (2.0 * a_mode) / f_hz) ** 2, 1e-9))
    boundary_arrival_s = probe_x / vg
    gate_start_s = gate_start_cycles / f_hz
    requested_end_s = total_cycles / f_hz
    safe_end_s = boundary_arrival_s - boundary_margin_cycles / f_hz
    gate_end_s = min(requested_end_s, safe_end_s)
    if gate_end_s <= gate_start_s:
        raise RuntimeError(
            f"unsafe gate for f={f_hz:g}: start={gate_start_s:g}, "
            f"end={gate_end_s:g}, boundary_arrival={boundary_arrival_s:g}"
        )

    n_steps = int(np.ceil(gate_end_s / dt))
    gate_start_step = int(np.ceil(gate_start_s / dt))
    gate_end_step = n_steps

    state, pec_mask_np = _standing_wave_arrays(
        f_hz=f_hz,
        dx=dx,
        nx=nx,
        ny=ny,
        nz=nz,
        a_mode=a_mode,
        x_short=x_short,
        dt=dt,
    )
    materials = MaterialArrays(
        eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
    )
    pec_mask = jnp.asarray(pec_mask_np)

    acc_sim = np.zeros((ny, nz), dtype=np.complex128)
    acc_ref = np.zeros((ny, nz), dtype=np.complex128)
    n_acc = 0

    for step in range(n_steps):
        state = _yee_step(state, materials, pec_mask, dt, dx)
        if step + 1 < gate_start_step or step + 1 > gate_end_step:
            continue
        t = float(state.step) * dt
        phase = np.exp(-1j * omega * t) * dt
        ez_sim = np.asarray(state.ez[probe_i, :, :], dtype=np.float64)
        ez_ref = _analytic_ez_plane(
            f_hz=f_hz,
            dx=dx,
            ny=ny,
            nz=nz,
            a_mode=a_mode,
            x_short=x_short,
            probe_x=probe_i * dx,
            t=t,
        )
        acc_sim += ez_sim * phase
        acc_ref += ez_ref * phase
        n_acc += 1

    # Ignore PEC-wall y endpoints where the reference is exactly zero.
    y_slice = slice(1, ny - 1)
    mag_sim = np.abs(acc_sim[y_slice, :])
    mag_ref = np.abs(acc_ref[y_slice, :])
    rel_mag_rms = _rms(mag_sim - mag_ref) / max(_rms(mag_ref), 1e-30)

    # Also report a shape-only error after best scalar magnitude fit.  This
    # distinguishes TE10 mode-shape corruption from a harmless global amplitude
    # drift caused by continuous-vs-discrete dispersion over the gate.
    denom = float(np.vdot(mag_ref.ravel(), mag_ref.ravel()).real)
    scale = float(np.vdot(mag_ref.ravel(), mag_sim.ravel()).real / max(denom, 1e-30))
    rel_shape_rms = _rms(mag_sim - scale * mag_ref) / max(_rms(mag_ref), 1e-30)

    return {
        "f_Hz": float(f_hz),
        "f_GHz": float(f_hz / 1e9),
        "dx_m": float(dx),
        "shape": [int(nx), int(ny), int(nz)],
        "a_mode_m": float(a_mode),
        "beta_1_per_m": float(beta),
        "x_short_m": float(x_short),
        "probe_x_m": float(probe_i * dx),
        "dt_s": float(dt),
        "n_steps": int(n_steps),
        "gate_start_step": int(gate_start_step),
        "gate_end_step": int(gate_end_step),
        "gate_start_cycles": float(gate_start_s * f_hz),
        "gate_end_cycles": float(gate_end_s * f_hz),
        "x_lo_boundary_arrival_cycles": float(boundary_arrival_s * f_hz),
        "n_accumulated_samples": int(n_acc),
        "rel_mag_rms": float(rel_mag_rms),
        "rel_shape_rms_after_scalar_fit": float(rel_shape_rms),
        "scalar_mag_fit": float(scale),
        "passes_2pct_mag_gate": bool(rel_mag_rms <= 0.02),
        "passes_2pct_shape_gate": bool(rel_shape_rms <= 0.02),
    }


def _select_freqs(freqs: np.ndarray, quick: bool, indices: str | None) -> np.ndarray:
    if indices:
        idx = [int(tok) for tok in indices.split(",") if tok.strip()]
        return freqs[idx]
    if quick:
        idx = np.linspace(0, len(freqs) - 1, 5).round().astype(int)
        return freqs[idx]
    return freqs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="run five representative frequencies instead of the full cv11 band")
    parser.add_argument("--freq-indices", type=str, default=None,
                        help="comma-separated cv11 frequency indices to run")
    parser.add_argument("--dx", type=float, default=1.0e-3)
    parser.add_argument("--x-short", type=float, default=0.600)
    parser.add_argument("--probe-x", type=float, default=0.505,
                        help="probe plane; default is 95 mm before the short, matching cv11 spacing")
    parser.add_argument("--total-cycles", type=float, default=10.0)
    parser.add_argument("--gate-start-cycles", type=float, default=2.0)
    parser.add_argument("--boundary-margin-cycles", type=float, default=2.0)
    parser.add_argument("--physical-aperture", action="store_true",
                        help="use physical WR-90 a in the analytic mode; default uses grid-effective a")
    args = parser.parse_args()

    cv = _load_cv11()
    freqs = _select_freqs(np.asarray(cv.FREQS_HZ, dtype=np.float64),
                          args.quick, args.freq_indices)

    print("=" * 78)
    print("Yee TE10 source-free standing-wave diagnostic")
    print("=" * 78)
    print(f"backend={jax.default_backend()}  dx={args.dx:g} m")
    print(f"frequencies={len(freqs)}  quick={args.quick}")
    print(f"x_short={args.x_short:g} m  probe_x={args.probe_x:g} m")
    print("gate: starts after "
          f"{args.gate_start_cycles:g} cycles; ends before x_lo boundary "
          f"arrival minus {args.boundary_margin_cycles:g} cycles")

    rows = []
    for i, f_hz in enumerate(freqs, start=1):
        print(f"\n[{i}/{len(freqs)}] f={f_hz/1e9:.3f} GHz ...", flush=True)
        row = run_frequency(
            f_hz=float(f_hz),
            dx=float(args.dx),
            a_physical=float(cv.A_WG),
            b_physical=float(cv.B_WG),
            x_short=float(args.x_short),
            probe_x=float(args.probe_x),
            total_cycles=float(args.total_cycles),
            gate_start_cycles=float(args.gate_start_cycles),
            boundary_margin_cycles=float(args.boundary_margin_cycles),
            use_physical_aperture=bool(args.physical_aperture),
        )
        rows.append(row)
        print(
            "    rel |Ez| RMS={rel_mag_rms:.4%}  "
            "shape-fit RMS={rel_shape_rms_after_scalar_fit:.4%}  "
            "scale={scalar_mag_fit:.6f}  "
            "gate={gate_start_cycles:.2f}-{gate_end_cycles:.2f} cycles  "
            "boundary={x_lo_boundary_arrival_cycles:.2f} cycles".format(**row),
            flush=True,
        )

    summary = {
        "description": "source-free rfx.core.yee TE10 standing-wave diagnostic",
        "acceptance_gate": "rel_mag_rms <= 0.02 per frequency",
        "shape_gate_auxiliary": "rel_shape_rms_after_scalar_fit <= 0.02 per frequency",
        "uses_physical_aperture": bool(args.physical_aperture),
        "rows": rows,
        "max_rel_mag_rms": float(max(r["rel_mag_rms"] for r in rows)),
        "max_rel_shape_rms_after_scalar_fit": float(
            max(r["rel_shape_rms_after_scalar_fit"] for r in rows)
        ),
        "all_pass_mag_gate": bool(all(r["passes_2pct_mag_gate"] for r in rows)),
        "all_pass_shape_gate": bool(all(r["passes_2pct_shape_gate"] for r in rows)),
    }

    suffix = "quick" if args.quick else "full"
    if args.freq_indices:
        suffix = "idx_" + args.freq_indices.replace(",", "_")
    if args.physical_aperture:
        suffix += "_physical_a"
    out_json = OUT / f"yee_te10_{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    f_ghz = np.array([r["f_GHz"] for r in rows])
    mag = np.array([r["rel_mag_rms"] for r in rows])
    shape = np.array([r["rel_shape_rms_after_scalar_fit"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0.02, color="k", ls="--", lw=1, label="2% gate")
    ax.plot(f_ghz, mag, "o-", label="|Ez| RMS vs analytic")
    ax.plot(f_ghz, shape, "s-", label="shape RMS after scalar fit")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("relative RMS error")
    ax.set_title("rfx Yee-core TE10 standing-wave preservation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = OUT / f"yee_te10_{suffix}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"max rel |Ez| RMS = {summary['max_rel_mag_rms']:.4%}")
    print("max shape-fit RMS = "
          f"{summary['max_rel_shape_rms_after_scalar_fit']:.4%}")
    print(f"all_pass_mag_gate = {summary['all_pass_mag_gate']}")
    print(f"all_pass_shape_gate = {summary['all_pass_shape_gate']}")
    print(f"[json] {out_json}")
    print(f"[plot] {out_png}")
    return 0 if summary["all_pass_mag_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

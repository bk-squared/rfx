"""Capture (x, y) field-slice snapshots at z = mid_z over time for Setup A
(internal wall via apply_pec_mask) vs Setup B (PEC at +x boundary face via
apply_pec_faces).

Goal: identify which field components diverge between A and B in the wall
vicinity, especially in the cells UPSTREAM of the LEFT FACE (i = 153, 154)
that drive the reflected wave back through the probe at i = 154.

Both setups share:
  - source plane at x = 40 mm (TFSF waveguide port)
  - reference plane at x = 50 mm
  - PEC LEFT FACE at x = 155 mm (Setup A: cell-155 mask; Setup B: i=Nx-1=155 boundary)
  - WR-90 cross-section, dx = 1 mm

Setup A has cells past i=155 (vacuum + CPML at +x); Setup B's domain ends at
i = 155.

Outputs npz at /tmp/pec_field_trace_{A,B}.npz containing:
  ez(snap, Nx, Ny), ey(snap, Nx, Ny), hy(snap, Nx, Ny), hz(snap, Nx, Ny),
  step_indices, dt
"""
from __future__ import annotations
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
from rfx.simulation import SnapshotSpec  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())
PORT_LEFT_X = 0.040
NUM_PERIODS = 200
CPML_LAYERS = 20

# Snapshot every 8 timesteps — at dt~1.9 ps that's ~15 ps per snap, enough to
# resolve the ~50 ps round-trip in the wall vicinity.
SNAP_INTERVAL = 8
# Slice axis 2 (z), index = floor(Nz/2). Nz = round(B_WG / dx) ≈ 10.
SLICE_AXIS = 2  # z
SLICE_INDEX_Z = int(round(B_WG / DX_M)) // 2  # ~5


def _build_setup(use_internal_wall: bool):
    if use_internal_wall:
        domain_x = 0.200
        bnd = BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        )
    else:
        domain_x = 0.156
        bnd = BoundarySpec(
            x=Boundary(lo="cpml", hi="pec"),  # PEC at +x face only
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        )
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(domain_x, A_WG, B_WG),
        boundary=bnd,
        cpml_layers=CPML_LAYERS,
        dx=DX_M,
    )
    if use_internal_wall:
        sim.add(
            Box((0.155, 0.0, 0.0), (0.155 + 2 * DX_M, A_WG, B_WG)),
            material="pec",
        )
    sim.add_waveguide_port(
        PORT_LEFT_X,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(FREQS_HZ),
        f0=F0,
        bandwidth=0.6,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        name="left",
    )
    return sim


def _run_with_snapshot(label: str, use_internal: bool, out_path: str):
    print(f"\n=== {label} ===", flush=True)
    sim = _build_setup(use_internal)
    snap = SnapshotSpec(
        interval=SNAP_INTERVAL,
        components=("ez", "ey", "hy", "hz", "ex"),
        slice_axis=SLICE_AXIS,
        slice_index=SLICE_INDEX_Z,
    )
    result = sim.run(num_periods=NUM_PERIODS, snapshot=snap, compute_s_params=False)
    snaps = result.snapshots
    if snaps is None:
        raise RuntimeError("snapshot dict is None — snapshot was not wired through")
    ez = np.asarray(snaps["ez"])
    ey = np.asarray(snaps["ey"])
    hy = np.asarray(snaps["hy"])
    hz = np.asarray(snaps["hz"])
    ex = np.asarray(snaps["ex"])
    n_snaps = ez.shape[0]
    print(f"  snaps={n_snaps}, ez.shape={ez.shape} (n_snap, Nx, Ny)")
    print(f"  Nx={ez.shape[1]}, Ny={ez.shape[2]}, slice z-index={SLICE_INDEX_Z}")
    # estimate dt
    dt_est = 0.99 / (3e8 * np.sqrt(3) / DX_M)
    np.savez(
        out_path,
        ez=ez, ey=ey, hy=hy, hz=hz, ex=ex,
        snap_interval=SNAP_INTERVAL,
        dt_estimate=dt_est,
        slice_axis=SLICE_AXIS,
        slice_index=SLICE_INDEX_Z,
    )
    print(f"  saved {out_path}")
    return ez, ey, hy, hz, ex


def main():
    ez_A, ey_A, hy_A, hz_A, ex_A = _run_with_snapshot(
        "SETUP A: internal wall (apply_pec_mask)",
        use_internal=True,
        out_path="/tmp/pec_field_trace_A.npz",
    )
    ez_B, ey_B, hy_B, hz_B, ex_B = _run_with_snapshot(
        "SETUP B: boundary PEC (apply_pec_faces x_hi)",
        use_internal=False,
        out_path="/tmp/pec_field_trace_B.npz",
    )

    # Quick comparison summary at key x-indices
    Ny_A = ez_A.shape[2]
    Ny_B = ez_B.shape[2]
    assert Ny_A == Ny_B, f"Ny mismatch: A={Ny_A}, B={Ny_B}"
    j_mid = Ny_A // 2  # mid-y for TE10 mode peak

    print(f"\n=== SUMMARY at j_mid={j_mid} (mid-y, TE10 peak) ===")
    print(f"  Setup A Nx = {ez_A.shape[1]}, Setup B Nx = {ez_B.shape[1]}")
    Nx_min = min(ez_A.shape[1], ez_B.shape[1])
    print(f"\n  Comparing common x range [0:{Nx_min}]:")

    # Check max abs difference per x-index per component, across all snapshots
    for comp_name, fA, fB in [
        ("ez", ez_A, ez_B), ("ey", ey_A, ey_B),
        ("hy", hy_A, hy_B), ("hz", hz_A, hz_B),
        ("ex", ex_A, ex_B),
    ]:
        diff = np.abs(fA[:, :Nx_min, :] - fB[:, :Nx_min, :])
        max_diff_per_x = diff.max(axis=(0, 2))  # per x-index
        # Check at indices around the wall
        x_indices = [150, 152, 153, 154, 155]
        print(f"\n  Component {comp_name}: max|A-B| at x-indices around wall:")
        for xi in x_indices:
            if xi < Nx_min:
                fA_max = np.abs(fA[:, xi, :]).max()
                fB_max = np.abs(fB[:, xi, :]).max()
                print(f"    x={xi}: max|A|={fA_max:.4e}, max|B|={fB_max:.4e}, "
                      f"max|A-B|={max_diff_per_x[xi]:.4e}")

    # Inspect Setup A's "interior of wall" cells (i = 156..159) — should be zero
    print(f"\n=== SETUP A: max field value at INTERIOR of wall (cells past LEFT FACE) ===")
    if ez_A.shape[1] > 159:
        for xi in [155, 156, 157, 158, 159]:
            print(f"  x={xi}: max|ez|={np.abs(ez_A[:, xi, :]).max():.4e}, "
                  f"max|ey|={np.abs(ey_A[:, xi, :]).max():.4e}, "
                  f"max|hy|={np.abs(hy_A[:, xi, :]).max():.4e}, "
                  f"max|hz|={np.abs(hz_A[:, xi, :]).max():.4e}, "
                  f"max|ex|={np.abs(ex_A[:, xi, :]).max():.4e}")


if __name__ == "__main__":
    main()

"""Stage-2 gate (b) witness: open-domain oblique TFSF (Method B) THROUGH run().

Proves that the integrated ``Simulation.run()`` scan reproduces the standalone
Method-B prototype physics (scripts/verify_methodB_jnp.py). Builds a Simulation
with a method='methodB' oblique TFSF, runs it via ``sim.run(...)`` with a
per-step field snapshot, and measures — with the SAME math as the standalone —
the time-averaged Poynting injection angle and the TFSF leakage.

Standalone baseline (numpy prototype, reproduced by the jnp module):
  normal  theta_eff~0.06 deg, Sx>0, leak -35.8 dB
  oblique 30/45/60 -> theta_eff 30.2/45.0/59.9, leak -37.6/-40.2/-41.5 dB
  domain-invariant 45 -> 45.01 @ny60 / 45.00 @ny100

Gate (b): through run(), theta_eff ~= requested within ~1 deg AND leak < -25 dB
(target ~ -40 dB), fields real float32.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import sys
sys.path.insert(0, "/root/workspace/bk-workspace/rfx-oblique-rcs")

import numpy as np

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.simulation import SnapshotSpec

C0 = 299_792_458.0
DX = 0.002
CPML = 10
MARGIN = 10


def run_angle_through_run(theta_deg, ny=None):
    """Build + run a Method-B oblique TFSF via sim.run(); return (Sx, Sy, leak, dtype)."""
    domain_y = 0.12 if ny is None else ny * DX
    # Grid mirror only to pick n_steps / kz exactly like the standalone driver.
    grid = Grid(freq_max=10e9, domain=(0.60, domain_y, 0.006), dx=DX, cpml_layers=CPML)
    dt = grid.dt
    nx, nyc, nz = grid.shape
    kz = nz // 2
    n_steps = int(1.6 * (nx * DX) / C0 / dt)

    sim = (
        Simulation(mode="3d", boundary="cpml", cpml_layers=CPML,
                   freq_max=10e9, domain=(0.60, domain_y, 0.006), dx=DX)
        .add_tfsf_source(f0=5e9, bandwidth=0.5, polarization="ez",
                         angle_deg=theta_deg, method="methodB",
                         waveform="modulated_gaussian", margin=MARGIN)
    )
    res = sim.run(
        n_steps=n_steps,
        snapshot=SnapshotSpec(interval=1, components=("ez", "hx", "hy"),
                              slice_axis=2, slice_index=kz),
    )

    ez = np.asarray(res.snapshots["ez"])   # (n_steps, nx, ny)
    hx = np.asarray(res.snapshots["hx"])
    hy = np.asarray(res.snapshots["hy"])

    # Box geometry: identical to init_tfsf_methodB (off = cpml + margin).
    off = CPML + MARGIN
    xl, xh = off, nx - off
    yl, yh = off, nyc - off
    ti0, ti1, tj0, tj1 = xl + 6, xh - 6, yl + 6, yh - 6

    # Peak tracking (leak) + steady-region Poynting sum (theta_eff) — exactly
    # the standalone's math, vectorised over the recorded step axis.
    sf = np.max(np.abs(ez[:, 13:xl - 3, tj0:tj1]), axis=(1, 2))
    tf = np.max(np.abs(ez[:, ti0:ti1, tj0:tj1]), axis=(1, 2))
    sf_peak = float(np.max(sf))
    tf_peak = float(np.max(tf))

    s0 = n_steps // 3
    ez_i = ez[s0 + 1:, ti0:ti1, tj0:tj1]
    hx_i = hx[s0 + 1:, ti0:ti1, tj0:tj1]
    hy_i = hy[s0 + 1:, ti0:ti1, tj0:tj1]
    Sx = float(np.sum(-ez_i * hy_i))   # S_x = -Ez*Hy
    Sy = float(np.sum(ez_i * hx_i))    # S_y = +Ez*Hx

    leak = 20.0 * np.log10(max(sf_peak, 1e-30) / max(tf_peak, 1e-30))
    return Sx, Sy, leak, res.state.ez.dtype


def main():
    print("=== GATE (b): Method B oblique TFSF THROUGH Simulation.run() ===")
    all_ok = True
    for th in (30.0, 45.0, 60.0):
        Sx, Sy, leak, dtype = run_angle_through_run(th)
        te = np.degrees(np.arctan2(Sy, Sx))
        real_f32 = (dtype == np.float32)
        ok = abs(te - th) < 1.5 and leak < -25.0 and Sx > 0 and real_f32
        all_ok = all_ok and ok
        print(f"  requested {th:4.0f} deg -> theta_eff={te:6.2f} deg, "
              f"leak={leak:6.1f} dB, Sx={Sx:.3e} (>0), dtype={dtype} "
              f"-> {'OK' if ok else 'FAIL'}")

    print("=== domain (ny) invariance at 45 deg (physical -> unchanged) ===")
    for ny in (60, 100):
        Sx, Sy, leak, _ = run_angle_through_run(45.0, ny=ny)
        te = np.degrees(np.arctan2(Sy, Sx))
        print(f"  ny={ny}: theta_eff={te:.2f} deg, leak={leak:.1f} dB")

    print("\nRESULT:", "ALL GATES OK" if all_ok else "SOME GATE FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""MSL thru-line mesh-convergence + energy-balance diagnostic.

Re-uses the geometry of ``tests/test_msl_port_integration.py`` and runs
the integration over a DX ladder.  For each DX it records:

    DX | mean |S11| | mean |S21| | mean Re(Z0) | residual = 1 - |S11|^2 - |S21|^2

Plus a single-mesh ablation: DX = sweep[-1] with port-1 sigma "killed"
(impedance bumped 1e9x so the σ-distribution becomes ~0).
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

# Make the rfx package importable when run from the repo root or from
# scripts/diagnostics itself.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

# Geometry constants — copied verbatim from the integration test.
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
F_MAX = 5e9
GATE_F_LO = 3.0e9
GATE_F_HI = 4.5e9


def _build_sim(dx: float, *, kill_port1_sigma: bool = False,
               probe_offset_um: float | None = None,
               probe_spacing_um: float | None = None) -> Simulation:
    LX = L_LINE + 2 * PORT_MARGIN
    LY = W_TRACE + 6 * dx          # 3-cell clearance each side
    LZ = H_SUB + 1.5e-3            # substrate + 1.5 mm air above
    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, LZ),
        dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)), material="ro4350b")
    y_centre = LY / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    # Full-LX PEC trace (matches the integration test as committed at 8882ef1).
    # Truncating to [PORT_MARGIN, PORT_MARGIN+L_LINE] (matching OpenEMS's
    # trace x-extent) was tested in v2 and made things much worse — |S11|
    # nearly doubled at every mesh and energy residual went NEGATIVE.
    # Reason: rfx's add_msl_port does NOT auto-extend trace metal across the
    # port-feed region (OpenEMS's MSLPort.metal_prop does, via a 6-cell box).
    # Truncating creates an open-circuit feed at port 0, much worse than the
    # small bias from PEC running into the CPML.
    sim.add(Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + dx)),
            material="pec")
    # Probe-cell counts derived from physical distances when overrides given.
    # Default behavior (None): use the API defaults (offset=5 cells, spacing=3 cells).
    msl_kwargs = {}
    if probe_offset_um is not None:
        msl_kwargs["n_probe_offset"] = max(3, int(round(probe_offset_um * 1e-6 / dx)))
    if probe_spacing_um is not None:
        msl_kwargs["n_probe_spacing"] = max(1, int(round(probe_spacing_um * 1e-6 / dx)))
    # Port 0: driven from left.
    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="+x", impedance=50.0,
        **msl_kwargs,
    )
    # Port 1: passive matched at right end.
    z1 = 50.0 if not kill_port1_sigma else 50.0 * 1e9
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_centre, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="-x", impedance=z1,
        **msl_kwargs,
    )
    return sim


def _gate_stats(result):
    freqs = np.asarray(result.freqs)
    S = np.asarray(result.S)
    Z0 = np.asarray(result.Z0)
    mask = (freqs >= GATE_F_LO) & (freqs <= GATE_F_HI)
    s11 = np.abs(S[0, 0, mask])
    s21 = np.abs(S[1, 0, mask])
    z0  = Z0[0, mask].real
    energy = s11**2 + s21**2
    residual = 1.0 - energy
    return dict(
        n_pts=int(mask.sum()),
        f_lo_GHz=float(freqs[mask][0] * 1e-9),
        f_hi_GHz=float(freqs[mask][-1] * 1e-9),
        mean_s11=float(np.mean(s11)),
        std_s11=float(np.std(s11)),
        mean_s21=float(np.mean(s21)),
        std_s21=float(np.std(s21)),
        mean_z0=float(np.mean(z0)),
        std_z0=float(np.std(z0)),
        mean_energy=float(np.mean(energy)),
        mean_residual=float(np.mean(residual)),
        max_residual=float(np.max(np.abs(residual))),
        per_freq_s11=s11.tolist(),
        per_freq_s21=s21.tolist(),
        per_freq_residual=residual.tolist(),
    )


def run_one(dx: float, *, kill_port1_sigma: bool = False, num_periods: float = 12.0,
            n_freqs: int = 30, label: str | None = None,
            probe_offset_um: float | None = None,
            probe_spacing_um: float | None = None) -> dict:
    label = label or f"dx={dx*1e6:.0f}um"
    sim = _build_sim(dx, kill_port1_sigma=kill_port1_sigma,
                     probe_offset_um=probe_offset_um,
                     probe_spacing_um=probe_spacing_um)
    nz_sub = int(round(H_SUB / dx))
    ny_trace = int(round(W_TRACE / dx))
    nx_total = int(round((L_LINE + 2 * PORT_MARGIN) / dx))
    print(f"\n[{label}] dx={dx*1e6:.0f}um  nz_sub={nz_sub}  ny_trace={ny_trace}  nx={nx_total}", flush=True)
    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=n_freqs, num_periods=num_periods)
    dt = time.time() - t0
    s = _gate_stats(res)
    s.update(dx=dx, label=label, nz_sub=nz_sub, ny_trace=ny_trace,
             nx_total=nx_total, kill_port1_sigma=kill_port1_sigma,
             runtime_s=dt)
    print(
        f"   |S11|={s['mean_s11']:.4f}  |S21|={s['mean_s21']:.4f}  "
        f"Re(Z0)={s['mean_z0']:.2f}  resid={s['mean_residual']:+.4f}  "
        f"(max|resid|={s['max_residual']:.4f})  [{dt:.1f}s]",
        flush=True,
    )
    return s


def main(out_json: str | None = None) -> list[dict]:
    # Trimmed ladder: {80, 60, 50} for runtime; if monotone trend is unclear
    # we can add 40e-6 in a follow-up.
    sweep = [80e-6, 60e-6, 50e-6]
    # Probe placement scaling: optionally fix physical offset/spacing rather
    # than cell counts. Defaults None → API defaults (5/3 cells).
    probe_offset_um = float(os.environ.get("PROBE_OFFSET_UM", "0")) or None
    probe_spacing_um = float(os.environ.get("PROBE_SPACING_UM", "0")) or None
    if probe_offset_um or probe_spacing_um:
        print(f"\n[probe override] offset={probe_offset_um}µm  spacing={probe_spacing_um}µm",
              flush=True)
    rows = [run_one(dx, probe_offset_um=probe_offset_um,
                    probe_spacing_um=probe_spacing_um) for dx in sweep]
    # Ablation at the finest dx.
    print("\n--- Ablation: kill port-1 sigma at finest dx ---")
    rows.append(run_one(sweep[-1], kill_port1_sigma=True,
                        probe_offset_um=probe_offset_um,
                        probe_spacing_um=probe_spacing_um,
                        label=f"dx={sweep[-1]*1e6:.0f}um(no_p1_sigma)"))
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote: {out_json}")
    return rows


if __name__ == "__main__":
    out = os.environ.get("OUT_JSON",
        os.path.join(HERE, "msl_thru_meshconv_results.json"))
    main(out)

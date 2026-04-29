"""Dump E_z field on the port plane for PEC-short geometry.

For TE10 mode, E_z(y, z) = sin(πy/a) (uniform in z). At a fixed y=a/2
(peak), E_z(z) should be a CONSTANT across all z cells. Any deviation
at the last cell (z = b - dz/2) quantifies the v_hi contamination
identified as the simulator-core blocker for follow-ups #1 and #2.

Output: per-z-cell |E_z| at y=a/2, port plane x=PORT_LEFT_X.
Comparison: ratio E_z[v_hi] / E_z[v_lo+1] should equal 1 for clean
TE10. Anything else IS the contamination.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp
import importlib.util


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    cv = _load_cv11()
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS,
        dx=cv.DX_M,
    )
    # PEC-short
    pec_short_x = cv.PORT_RIGHT_X - 0.005
    sim.add(
        Box((pec_short_x, 0.0, 0.0),
            (pec_short_x + 2 * cv.DX_M, cv.DOMAIN_Y, cv.DOMAIN_Z)),
        material="pec",
    )
    # Single left port
    port_freqs = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        name="left",
    )

    # DFT plane probe at the port plane
    sim.add_dft_plane_probe(
        axis="x",
        coordinate=cv.PORT_LEFT_X + 0.001,
        component="ez",
        freqs=port_freqs,
        name="ez_port",
    )

    result = sim.run(num_periods=cv.NUM_PERIODS_LONG, compute_s_params=False)
    print(f"sim.run done. dft_planes keys: {list(result.dft_planes or {})}")

    probe = result.dft_planes["ez_port"]
    # accumulator shape: (n_freqs, n1, n2). For axis=0 (x-normal): (nf, ny, nz)
    e_z = np.asarray(probe.accumulator)  # (nf, ny, nz)
    print(f"E_z accumulator shape: {e_z.shape}")
    # Reorder to (ny, nz, nf) for downstream code
    e_z = np.transpose(e_z, (1, 2, 0))

    # Pick one frequency (10 GHz) — index 9 of 21 in [8.2..12.4 GHz]
    freq_target = 10.0e9
    freq_idx = int(np.argmin(np.abs(np.asarray(probe.freqs) - freq_target)))
    print(f"Inspecting freq[{freq_idx}] = {float(probe.freqs[freq_idx])/1e9:.2f} GHz")

    # Pick y at peak: y = a/2
    a = cv.A_WG
    ny = e_z.shape[0]
    nz = e_z.shape[1]
    dy = cv.DOMAIN_Y / ny
    dz = cv.DOMAIN_Z / nz
    y_target_idx = ny // 2
    y_target = (y_target_idx + 0.5) * dy
    print(f"y slice: idx={y_target_idx}, y={y_target*1e3:.2f} mm (a/2={a*1e3/2:.2f} mm), dy={dy*1e3:.4f} mm, dz={dz*1e3:.4f} mm")

    # Profile along z at this y
    e_z_slice = e_z[y_target_idx, :, freq_idx]
    print(f"\nz_idx | |E_z|     | ∠E_z      | E_z_real    | E_z_imag")
    print("-" * 60)
    for k in range(len(e_z_slice)):
        ezk = complex(e_z_slice[k])
        print(f"  {k:3d} | {abs(ezk):.6e} | {np.degrees(np.angle(ezk)):+7.2f}° | {ezk.real:+.4e} | {ezk.imag:+.4e}")

    # Key ratio: E_z[v_hi] / E_z[v_hi - 1]
    if len(e_z_slice) >= 2:
        ratio = abs(complex(e_z_slice[-1])) / abs(complex(e_z_slice[-2])) if abs(complex(e_z_slice[-2])) > 0 else float("nan")
        ratio_complex = complex(e_z_slice[-1]) / complex(e_z_slice[-2]) if abs(complex(e_z_slice[-2])) > 0 else float("nan")
        print(f"\n=== v_hi vs v_hi-1 ratio at y=a/2, f=10 GHz ===")
        print(f"  |E_z[v_hi]| / |E_z[v_hi-1]| = {ratio:.4f}  (should be ~1 for TE10)")
        if abs(complex(e_z_slice[-2])) > 0:
            print(f"  E_z[v_hi] / E_z[v_hi-1]     = {ratio_complex.real:+.4f} {'+' if ratio_complex.imag >= 0 else '-'} {abs(ratio_complex.imag):.4f}j")
        # Bulk (interior) cells: median ratio of consecutive cells
        interior_ratios = [abs(complex(e_z_slice[i+1])) / abs(complex(e_z_slice[i])) for i in range(2, len(e_z_slice)-2)]
        if interior_ratios:
            print(f"  Interior median ratio       = {np.median(interior_ratios):.4f}")

    # Check v_lo cell too
    if len(e_z_slice) >= 2:
        ratio_lo = abs(complex(e_z_slice[0])) / abs(complex(e_z_slice[1])) if abs(complex(e_z_slice[1])) > 0 else float("nan")
        print(f"  |E_z[v_lo]| / |E_z[v_lo+1]| = {ratio_lo:.4f}  (should be ~1 for TE10)")


if __name__ == "__main__":
    main()

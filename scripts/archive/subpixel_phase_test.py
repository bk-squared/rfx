#!/usr/bin/env python3
"""Experiment 5: does enabling rfx's subpixel smoothing close the slab
phase offset vs Meep?

Hypothesis (from experiment 4):
  Empty S21 phase matches rfx <-> Meep perfectly. Only slab (with
  material boundary) introduces the ~-57° offset. Meep has subpixel
  averaging ON by default; rfx default is OFF. If subpixel smoothing
  is the cause, enabling it in rfx should shift the slab phase
  toward Meep.

Approach:
  - Build rfx WR-90 slab geometry (same as crossval/11).
  - Compute smoothed aniso_eps via rfx/geometry/smoothing.py.
  - Run FDTD dev and ref with aniso_eps passed through run_simulation.
  - Compute normalized S21.
  - Compare ∠S21_rfx_smoothed with ∠S21_Meep (from VESSL JSON).
  - Re-run phase fit; compare slope/intercept to baseline (slab without
    smoothing: -5.87mm / -57.3°).
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax.numpy as jnp
import numpy as np


C0 = 2.998e8


def _load_meep():
    path = ("/root/workspace/byungkwan-workspace/research/microwave-energy/"
            "results/rfx_crossval_wr90_meep/wr90_meep_reference.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _meep_complex(block):
    return np.array([complex(r, i) for r, i in block], dtype=np.complex128)


def _unwrap_deg(phases_deg):
    return np.degrees(np.unwrap(np.radians(phases_deg)))


def run_rfx_slab(*, subpixel: bool):
    """Build WR-90 slab and return (freqs_hz, s21_rfx).

    `subpixel=True` enables Kottke anisotropic eps smoothing at the
    slab boundary before running FDTD.
    """
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box
    from rfx.simulation import run as run_simulation
    from rfx.geometry.smoothing import compute_smoothed_eps
    from rfx.sources.waveguide_port import (
        extract_waveguide_s_params_normalized,
    )

    freqs = np.linspace(8.2e9, 12.4e9, 21)
    f0 = float(freqs.mean())
    bw = 0.5
    domain = (0.200, 0.02286, 0.01016)
    dx = 0.001

    sim = Simulation(
        freq_max=freqs[-1] * 1.1,
        domain=domain,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=20,
        dx=dx,
    )
    sim.add_material("slab", eps_r=2.0, sigma=0.0)
    sim.add(Box((0.095, 0.0, 0.0), (0.105, 0.02286, 0.01016)), material="slab")
    port_freqs = jnp.asarray(freqs)
    sim.add_waveguide_port(
        0.040, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=f0, bandwidth=bw,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
    )
    sim.add_waveguide_port(
        0.160, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=f0, bandwidth=bw,
        waveform="modulated_gaussian", reference_plane=0.150, name="right",
    )

    # Manually run the two-run normalized path so we can thread aniso_eps
    # through `run_simulation`.  Mirrors
    # `Simulation.compute_waveguide_s_matrix` but exposes the smoothing
    # hook that the public API does not currently take.
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, _, _ = sim._assemble_materials(grid)
    if pec_mask_wg is not None:
        base_materials = base_materials._replace(
            sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
    materials = base_materials
    n_steps = grid.num_timesteps(num_periods=40)
    _, debye, lorentz = sim._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

    # Compute smoothed eps for the slab (only slab is non-vacuum, non-PEC here)
    aniso_eps_dev = None
    if subpixel:
        shape_eps_pairs = [
            (entry.shape, sim._resolve_material(entry.material_name).eps_r)
            for entry in sim._geometry
        ]
        if shape_eps_pairs:
            aniso_eps_dev = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)

    # Now call extract_waveguide_s_params_normalized — but we need to
    # inject aniso_eps into the device run.  That function doesn't take
    # aniso_eps directly, so we duplicate its two-run logic here and
    # thread it.
    entries = list(sim._waveguide_ports)
    raw_cfgs = [sim._build_waveguide_port_config(e, grid, port_freqs, n_steps) for e in entries]
    cfgs = [c[0] if isinstance(c, list) else c for c in raw_cfgs]

    # Resolve reference planes (match crossval/11 defaults)
    from rfx.sources.waveguide_port import (
        waveguide_plane_positions,
        extract_waveguide_port_waves,
    )
    ref_shifts = []
    for entry, cfg in zip(entries, cfgs):
        planes = waveguide_plane_positions(cfg)
        desired_ref = entry.reference_plane if entry.reference_plane is not None else planes["source"]
        ref_shifts.append(desired_ref - planes["reference"])

    def _reset(cfg, drive_enabled):
        zeros = jnp.zeros_like(cfg.v_probe_dft)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_dft=zeros, v_ref_dft=zeros,
            i_probe_dft=zeros, i_ref_dft=zeros,
            v_inc_dft=zeros,
        )

    from rfx.core.yee import init_materials as _init_vac
    ref_materials = _init_vac(grid.shape)

    n_ports = len(cfgs)
    n_freqs = len(cfgs[0].freqs)
    s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)

    drive_idx = 0   # port 1 (left) drives; we only need S21 = s[1, 0]
    ref_cfgs = [_reset(c, idx == drive_idx) for idx, c in enumerate(cfgs)]
    dev_cfgs = [_reset(c, idx == drive_idx) for idx, c in enumerate(cfgs)]

    # Reference run (vacuum, no aniso_eps needed)
    ref_result = run_simulation(
        grid, ref_materials, n_steps,
        boundary="cpml", cpml_axes=grid.cpml_axes,
        pec_axes="".join(a for a in "xyz" if a not in grid.cpml_axes),
        debye=None, lorentz=None,
        waveguide_ports=ref_cfgs,
    )
    ref_final = ref_result.waveguide_ports or ()

    # Device run (slab, optionally with smoothed eps)
    dev_result = run_simulation(
        grid, materials, n_steps,
        boundary="cpml", cpml_axes=grid.cpml_axes,
        pec_axes="".join(a for a in "xyz" if a not in grid.cpml_axes),
        debye=debye, lorentz=lorentz,
        waveguide_ports=dev_cfgs,
        aniso_eps=aniso_eps_dev,
    )
    dev_final = dev_result.waveguide_ports or ()

    # S21 = b_out_dev[port2] / b_out_ref[port2]
    _, b_ref_port2 = extract_waveguide_port_waves(ref_final[1], ref_shift=ref_shifts[1])
    _, b_dev_port2 = extract_waveguide_port_waves(dev_final[1], ref_shift=ref_shifts[1])
    b_ref_arr = np.asarray(b_ref_port2)
    b_dev_arr = np.asarray(b_dev_port2)
    safe = np.where(np.abs(b_ref_arr) > 1e-30, b_ref_arr, np.ones_like(b_ref_arr))
    s21 = b_dev_arr / safe
    return freqs, s21


def _fit_phase(s_rfx, s_meep, beta):
    raw = np.degrees(np.angle(s_rfx)) - np.degrees(np.angle(s_meep))
    wrapped = ((raw + 180) % 360) - 180
    rad = np.unwrap(np.radians(wrapped))
    A = np.vstack([beta, np.ones_like(beta)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, rad, rcond=None)
    fit = A @ np.array([slope, intercept])
    resid = rad - fit
    rms = float(np.degrees(np.sqrt(np.mean(resid**2))))
    return slope, intercept, rms


def main():
    meep = _load_meep()
    if meep is None:
        print("Meep JSON not found.", file=sys.stderr)
        return 2
    r_keys = sorted(k for k in meep.keys() if k.startswith("r") and k != "meta")
    rk = r_keys[-1]
    s21_meep = _meep_complex(meep[rk]["slab"]["s21"])

    fc = C0 / (2.0 * 0.02286)  # WR-90 TE10 cutoff

    print("Running rfx slab WITHOUT subpixel smoothing...", flush=True)
    freqs, s21_off = run_rfx_slab(subpixel=False)
    print("Running rfx slab WITH subpixel smoothing...", flush=True)
    _, s21_on = run_rfx_slab(subpixel=True)

    omega = 2 * np.pi * freqs
    k0 = omega / C0
    beta = np.sqrt(np.maximum(k0**2 - (2*np.pi*fc/C0)**2, 0.0))

    sl_off, ic_off, rms_off = _fit_phase(s21_off, s21_meep, beta)
    sl_on, ic_on, rms_on = _fit_phase(s21_on, s21_meep, beta)

    print()
    print("=== rfx slab S21 vs Meep (resolution=r4) ===")
    print(f"  subpixel OFF: slope = {sl_off*1e3:+.2f} mm   intercept = {np.degrees(ic_off):+.1f}°   RMS = {rms_off:.2f}°")
    print(f"  subpixel ON : slope = {sl_on*1e3:+.2f} mm   intercept = {np.degrees(ic_on):+.1f}°   RMS = {rms_on:.2f}°")
    print(f"  |Δslope|  = {abs(sl_on-sl_off)*1e3:.2f} mm")
    print(f"  |Δintercept| = {abs(np.degrees(ic_on-ic_off)):.1f}°")

    # Also report magnitude change (should be near-identical)
    print(f"\n  |S21|_off mean = {np.abs(s21_off).mean():.4f}")
    print(f"  |S21|_on  mean = {np.abs(s21_on).mean():.4f}")

    print()
    if abs(sl_on - sl_off) * 1e3 < 0.5 and abs(np.degrees(ic_on - ic_off)) < 5.0:
        print("Verdict: subpixel smoothing does NOT shift the slab phase meaningfully.")
        print("         Hypothesis FALSIFIED — material boundary handling is not the cause.")
    elif abs(np.degrees(ic_on - ic_off)) > 20.0 or abs(sl_on - sl_off) * 1e3 > 1.5:
        print("Verdict: subpixel smoothing MOVES the slab phase substantially.")
        print("         Hypothesis CONFIRMED — rfx's hard-voxel boundary differs from")
        print("         Meep's default subpixel averaging by a measurable phase factor.")
    else:
        print("Verdict: subpixel smoothing has a modest effect — direction relevant but")
        print("         not sufficient to fully close the gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

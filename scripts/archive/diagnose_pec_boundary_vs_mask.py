"""Compare reflection coefficient: PEC via boundary vs PEC via geometry mask.

Setup A (current crossval/11): PEC short at internal x=155mm, enforced by
                               apply_pec_mask after each update_e.
Setup B (this script):         PEC short at +x DOMAIN BOUNDARY (no internal
                               wall). Domain shrinks so wall is at +x face.
                               PEC enforced by apply_pec(axes='x').

If the discrete reflection coefficient differs, rfx's geometry-mask path
has a bug; if identical, it's a universal Yee+staircased-PEC limit (which
contradicts OpenEMS getting 0.996+).
"""
from __future__ import annotations
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

C0 = 299_792_458.0
MU_0 = 4.0 * np.pi * 1e-7

from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_ref_t": np.asarray(cfg.v_ref_t).copy(),
        "i_ref_t": np.asarray(cfg.i_ref_t).copy(),
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "v_inc_t": np.asarray(cfg.v_inc_t).copy(),
        "n_steps_recorded": int(cfg.n_steps_recorded),
        "dt": float(cfg.dt), "dx": float(cfg.dx),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
        "ref_x_m": float(cfg.reference_x_m),
        "probe_x_m": float(cfg.probe_x_m),
        "src_x_m": float(cfg.source_x_m),
    })
    return a, b


wg.extract_waveguide_port_waves = _capture
import rfx.api as _api_mod  # noqa: E402
_api_mod.extract_waveguide_port_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
DOMAIN_Y, DOMAIN_Z = A_WG, B_WG
PORT_LEFT_X = 0.040
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())
BW = 0.6
NUM_PERIODS = 200
CPML_LAYERS = 20

# Setup B: place PEC at +x boundary. Need domain that ends right where the
# wall would be. To make geometry comparable to setup A (wall at 155mm
# internal), use domain_x = 155 mm so that +x boundary IS at 155 mm.
# Setup B v2: domain ends at 156mm so PEC boundary lives at i=155 (= 155mm).
# Probe at offset 114 → probe_x = 40 + 114 = 154 mm = ONE CELL upstream of
# the PEC face. This matches setup A's probe placement (154mm probe vs
# 155mm wall start).
DOMAIN_X_B = 0.156
PROBE_OFFSET_CELLS = 114


def _build(use_internal_wall: bool):
    if use_internal_wall:
        domain_x = 0.200
        bnd = BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        )
    else:
        domain_x = DOMAIN_X_B
        bnd = BoundarySpec(
            x=Boundary(lo="cpml", hi="pec"),   # PEC at +x face
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        )
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(domain_x, DOMAIN_Y, DOMAIN_Z),
        boundary=bnd,
        cpml_layers=CPML_LAYERS,
        dx=DX_M,
    )
    if use_internal_wall:
        sim.add(Box((0.155, 0.0, 0.0), (0.155 + 2*DX_M, DOMAIN_Y, DOMAIN_Z)),
                material="pec")
    port_freqs = jnp.asarray(FREQS_HZ)
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=BW,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        probe_offset=PROBE_OFFSET_CELLS,
    )
    # No RIGHT port for setup B (domain ends at 155 mm). For setup A we
    # also drop the RIGHT port to keep the comparison apples-to-apples.
    return sim


def _decompose(cap):
    n = cap["n_steps_recorded"]
    V_ref = cap["v_ref_t"][:n]
    I_ref = cap["i_ref_t"][:n]
    V_pro = cap["v_probe_t"][:n]
    I_pro = cap["i_probe_t"][:n]
    dt = cap["dt"]; fc = cap["f_cutoff"]
    freqs = cap["freqs"]
    omega = 2 * np.pi * freqs.astype(np.float64)
    beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
    Z_arr = omega * MU_0 / np.maximum(beta_arr, 1e-30)
    n_idx = np.arange(n)
    tt = n_idx * dt
    phase = np.exp(-1j * omega[None, :] * tt[:, None])

    def _dft(x):
        return 2.0 * dt * (x.astype(np.float64) @ phase)

    def _decomp(V_t, I_t):
        Vd = _dft(V_t)
        Id = _dft(I_t) * np.exp(+1j * omega * 0.5 * dt)
        return 0.5*(Vd + Z_arr*Id), 0.5*(Vd - Z_arr*Id)

    fwd_ref, bwd_ref = _decomp(V_ref, I_ref)
    fwd_pro, bwd_pro = _decomp(V_pro, I_pro)
    return freqs, fwd_ref, bwd_ref, fwd_pro, bwd_pro


for label, use_internal in [("internal wall (mask)", True),
                            ("boundary PEC (apply_pec)", False)]:
    print(f"\n=== Setup: {label} ===", flush=True)
    _CAPS.clear()
    sim = _build(use_internal)
    # 1-port sim, just run with src_amp on. No s_matrix path (would need 2 ports).
    # Use sim.run() instead and grab the LEFT port cfg.
    result = sim.run(num_periods=NUM_PERIODS)
    # Manually capture by calling extract_waveguide_port_waves on the result
    left_cfg = (result.waveguide_ports["left"]
                if isinstance(result.waveguide_ports, dict)
                else result.waveguide_ports[0])
    # _capture wrapper appends; trigger it:
    _ = wg.extract_waveguide_port_waves(left_cfg, ref_shift=0.0)
    cap = _CAPS[-1]
    freqs, fwd_ref, bwd_ref, fwd_pro, bwd_pro = _decompose(cap)
    # Reflection at wall-front (probe_x = 154 mm)
    wall_refl = np.abs(bwd_pro) / np.maximum(np.abs(fwd_pro), 1e-30)
    ref_refl  = np.abs(bwd_ref) / np.maximum(np.abs(fwd_ref), 1e-30)
    print(f"  ref_x = {cap['ref_x_m']*1000:.1f} mm, "
          f"probe_x = {cap['probe_x_m']*1000:.1f} mm")
    print(f"  |bwd|/|fwd| @ wall-front = mean {wall_refl.mean():.4f}  "
          f"range [{wall_refl.min():.4f}, {wall_refl.max():.4f}]")
    print(f"  |bwd|/|fwd| @ ref       = mean {ref_refl.mean():.4f}  "
          f"range [{ref_refl.min():.4f}, {ref_refl.max():.4f}]")
    for k in (0, 5, 10, 15, 20):
        print(f"    f={freqs[k]/1e9:5.2f} GHz: wall {wall_refl[k]:.4f}  "
              f"ref {ref_refl[k]:.4f}")

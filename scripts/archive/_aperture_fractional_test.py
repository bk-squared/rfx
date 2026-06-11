"""Phase 2 experiment: per-cell fractional aperture weight using port.a/port.b.

Compare against the current production half-weight implementation on the
PEC-short benchmark (target |r|=1.0). The regression tests are run via
pytest separately.

Variants (sys.argv[1]):
  PROD:       current production (half-weight at +face on both u, v)
  FRAC:       fractional weight = overlap of cell extent with [0, port.X] / dx,
              cell-centered-between-nodes (cell k extent [k*dx, (k+1)*dx])
  FRAC_NODE:  cell-centered-at-node (cell k extent [(k-0.5)*dx, (k+0.5)*dx])
              applied symmetrically at both faces
  V4:         drop +face on v only (legacy V4 reference)
  DROP_BOTH:  drop +face on both u and v (initial buggy implementation)
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

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "PROD"

import rfx.sources.waveguide_port as wg_mod  # noqa: E402

_orig_aperture_dA = wg_mod._aperture_dA


def _frac_between(widths_np, wall_m):
    """Cell-centered-between-nodes: cell k extent [cumsum_lo, cumsum_hi]."""
    cell_hi = np.cumsum(widths_np)
    cell_lo = cell_hi - widths_np
    inside = np.maximum(0.0, np.minimum(cell_hi, wall_m) - np.maximum(cell_lo, 0.0))
    return inside / np.maximum(widths_np, 1e-30)


def _frac_node(widths_np, wall_m):
    """Cell-centered-AT-node: cell k extent [center - 0.5*w, center + 0.5*w].

    For uniform widths this is [(k-0.5)*dx, (k+0.5)*dx] with cell 0 centered
    at z=0 (so half its area lies at z<0).
    """
    n = widths_np.size
    if n == 0:
        return widths_np
    # Node positions (cell centers in node-centered convention)
    nodes = np.zeros(n)
    nodes[0] = 0.0
    if n > 1:
        nodes[1:] = np.cumsum(widths_np[:-1])
    half = 0.5 * widths_np
    cell_lo = nodes - half
    cell_hi = nodes + half
    inside = np.maximum(0.0, np.minimum(cell_hi, wall_m) - np.maximum(cell_lo, 0.0))
    return inside / np.maximum(widths_np, 1e-30)


def _make_dA(cfg, u_w_method, v_w_method):
    u_w = np.asarray(cfg.u_widths)
    v_w = np.asarray(cfg.v_widths)
    u_weights = u_w_method(u_w, float(cfg.a))
    v_weights = v_w_method(v_w, float(cfg.b))
    return jnp.asarray(
        (u_w * u_weights)[:, None] * (v_w * v_weights)[None, :],
        dtype=jnp.float32,
    )


def _aperture_dA_FRAC(cfg):
    return _make_dA(cfg, _frac_between, _frac_between)


def _aperture_dA_FRAC_NODE(cfg):
    return _make_dA(cfg, _frac_node, _frac_node)


def _aperture_dA_V4(cfg):
    u_w = np.asarray(cfg.u_widths)
    v_w = np.asarray(cfg.v_widths)
    v_weight = np.ones_like(v_w)
    if v_w.size > 0:
        v_weight[-1] = 0.0
    return jnp.asarray(
        u_w[:, None] * (v_w * v_weight)[None, :],
        dtype=jnp.float32,
    )


def _aperture_dA_DROP_BOTH(cfg):
    u_w = np.asarray(cfg.u_widths)
    v_w = np.asarray(cfg.v_widths)
    u_weight = np.ones_like(u_w)
    v_weight = np.ones_like(v_w)
    if u_w.size > 0:
        u_weight[-1] = 0.0
    if v_w.size > 0:
        v_weight[-1] = 0.0
    return jnp.asarray(
        (u_w * u_weight)[:, None] * (v_w * v_weight)[None, :],
        dtype=jnp.float32,
    )


variants = {
    "PROD": _orig_aperture_dA,           # uses cfg.aperture_dA (half-weight at +face)
    "FRAC": _aperture_dA_FRAC,
    "FRAC_NODE": _aperture_dA_FRAC_NODE,
    "V4": _aperture_dA_V4,
    "DROP_BOTH": _aperture_dA_DROP_BOTH,
}
wg_mod._aperture_dA = variants[VARIANT]
print(f"[runner] variant {VARIANT}", flush=True)

from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS = []
_orig_extract = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig_extract(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "n": int(cfg.n_steps_recorded), "dt": float(cfg.dt),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
    })
    return a, b


wg.extract_waveguide_port_waves = _capture
import rfx.api as _api  # noqa: E402
_api.extract_waveguide_port_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())

sim = Simulation(
    freq_max=float(FREQS_HZ[-1]) * 1.1,
    domain=(0.200, A_WG, B_WG),
    boundary=BoundarySpec(
        x=Boundary(lo="cpml", hi="cpml"),
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    ),
    cpml_layers=20, dx=DX_M,
)
sim.add(Box((0.155, 0.0, 0.0), (0.155 + 2 * DX_M, A_WG, B_WG)),
        material="pec")
sim.add_waveguide_port(
    0.040, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=jnp.asarray(FREQS_HZ), f0=F0, bandwidth=0.6,
    waveform="modulated_gaussian", reference_plane=0.050, name="left",
    probe_offset=114,
)

result = sim.run(num_periods=200, compute_s_params=False)
wp = result.waveguide_ports
cfg = wp["left"] if isinstance(wp, dict) else wp[0]
_ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
cap = _CAPS[-1]

n = cap["n"]; dt = cap["dt"]; fc = cap["f_cutoff"]
freqs = cap["freqs"]
omega = 2 * np.pi * freqs.astype(np.float64)
beta = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
Z = omega * MU_0 / np.maximum(beta, 1e-30)
n_idx = np.arange(n)
phase = np.exp(-1j * omega[None, :] * n_idx[:, None] * dt)
Vd = 2.0 * dt * (cap["v_probe_t"][:n].astype(np.float64) @ phase)
Id_raw = 2.0 * dt * (cap["i_probe_t"][:n].astype(np.float64) @ phase)
Id = Id_raw * np.exp(+1j * omega * 0.5 * dt)
fwd = 0.5 * (Vd + Z * Id)
bwd = 0.5 * (Vd - Z * Id)
refl = np.abs(bwd) / np.maximum(np.abs(fwd), 1e-30)
arg = np.degrees(np.angle(Vd / (Z * Id)))
print(f"VARIANT_RESULT {VARIANT}: |r|={refl.mean():.4f} arg={arg[10]:.2f}deg "
      f"range=[{refl.min():.4f},{refl.max():.4f}]")

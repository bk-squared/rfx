"""A/B bit-identity harness for the CPML pad continuation on the smoothed lane.

The change this gates (#1043 stage B) moves the update permittivity for ONE
class of configuration: a dielectric that reaches an absorber pad, under
subpixel smoothing. Everything else has to be byte-identical, and "has to be"
is not a thing to assert from the shape of the diff -- the continuation is a
geometry transform feeding an SDF, and an SDF evaluated at a different
coordinate can move the last bits of a cell that was never meant to change.

So this drives eight configurations that must NOT move -- vacuum pads, an
interior dielectric under CPML and under UPML, the staircase lane on the very
geometry the change is about, a touching dielectric behind PEC walls (no pad
to continue into), 3-D, the Stage-2 ``kottke_pec`` tensor, and the non-uniform
mirror -- and SHA-256s the six final field arrays plus the probe trace.
``np.array_equal`` on raw bytes, never a tolerance: the methodology note's
§2.2 rule for a change that claims not to move existing numbers.

Both precision lanes, because x64 is process-global and a coordinate change
can be invisible in one and not the other. One process per lane.

Use, from the tree under test and from ``git archive origin/main`` extracted
somewhere::

    JAX_ENABLE_X64=1 PYTHONPATH=<tree> python3 \\
        scripts/diagnostics/pad_continuation_bit_identity.py > x64.json
    JAX_ENABLE_X64=0 PYTHONPATH=<tree> python3 \\
        scripts/diagnostics/pad_continuation_bit_identity.py > f32.json

then diff the two trees' JSON. Like
``tests/locks/test_runner_split_bit_identity.py``, the baseline is NOT
committed: the same FDTD code differs between hosts and XLA is free to re-fuse
a graph between versions, so bit identity is promised for A/B on ONE host with
ONE toolchain -- which is the question this change asks.

Measured 2026-09-15 on the remilab pod (linux x86_64, CPU, python 3.10, jax
0.6.2) against ``origin/main`` at 541f703f: all eight IDENTICAL in both lanes.
"""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np

import rfx
from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec

C0 = 2.998e8
A = 1.0e-6
DX = A / 10
FCEN = 0.15 * C0 / A
FWIDTH = 0.1 * C0 / A
N_STEPS = 60


def _digest(result) -> str:
    h = hashlib.sha256()
    for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
        arr = np.asarray(getattr(result.state, name))
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    ts = np.asarray(result.time_series)
    h.update(str(ts.dtype).encode())
    h.update(np.ascontiguousarray(ts).tobytes())
    return h.hexdigest()


def _src(sim, x, y, z=0.0):
    sim.add_source(position=(x, y, z), component="ez",
                   waveform=GaussianPulse(f0=FCEN, bandwidth=FWIDTH / FCEN,
                                          amplitude=1.0))
    sim.add_probe(position=(x + 2 * A, y, z), component="ez")


def cfg_vacuum_pads_2d():
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(8 * A, 6 * A, DX), dx=DX,
                     boundary=BoundarySpec.uniform("cpml"), cpml_layers=8,
                     mode="2d_tmz")
    _src(sim, 3 * A, 3 * A)
    return sim, dict(subpixel_smoothing=True)


def cfg_interior_dielectric_2d():
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(8 * A, 6 * A, DX), dx=DX,
                     boundary=BoundarySpec.uniform("cpml"), cpml_layers=8,
                     mode="2d_tmz")
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((2 * A, 2 * A, 0), (6 * A, 4 * A, DX)), material="d")
    _src(sim, 1 * A, 3 * A)
    return sim, dict(subpixel_smoothing=True)


def cfg_interior_dielectric_upml_2d():
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(8 * A, 6 * A, DX), dx=DX,
                     boundary=BoundarySpec.uniform("upml"), cpml_layers=8,
                     mode="2d_tmz")
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((2 * A, 2 * A, 0), (6 * A, 4 * A, DX)), material="d")
    _src(sim, 1 * A, 3 * A)
    return sim, dict(subpixel_smoothing=True)


def cfg_touching_dielectric_subpixel_off_2d():
    """The STAIRCASE lane on the very geometry stage B changes. Must not move:
    the array-side extension is untouched."""
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(8 * A, 6 * A, DX), dx=DX,
                     boundary=BoundarySpec.uniform("cpml"), cpml_layers=8,
                     mode="2d_tmz")
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((0, 2.5 * A, 0), (8 * A, 3.5 * A, DX)), material="d")
    _src(sim, 1 * A, 3 * A)
    return sim, dict(subpixel_smoothing=False)


def cfg_touching_dielectric_pec_walls_2d():
    """Touching dielectric, but PEC walls: no pad, so no continuation."""
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(8 * A, 6 * A, DX), dx=DX,
                     boundary="pec", mode="2d_tmz")
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((0, 2.5 * A, 0), (8 * A, 3.5 * A, DX)), material="d")
    _src(sim, 1 * A, 3 * A)
    return sim, dict(subpixel_smoothing=True)


def cfg_interior_dielectric_3d():
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(5 * A, 5 * A, 5 * A),
                     dx=DX, boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=6)
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((1.5 * A, 1.5 * A, 1.5 * A), (3.5 * A, 3.5 * A, 3.5 * A)),
            material="d")
    _src(sim, 1.0 * A, 2.5 * A, 2.5 * A)
    return sim, dict(subpixel_smoothing=True)


def cfg_kottke_pec_interior_3d():
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(5 * A, 5 * A, 5 * A),
                     dx=DX, boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=6)
    sim.add_material("d", eps_r=4.0)
    sim.add_material("metal", sigma=1e10)
    sim.add(Box((1.5 * A, 1.5 * A, 1.5 * A), (3.5 * A, 3.5 * A, 3.5 * A)),
            material="d")
    sim.add(Box((2.0 * A, 2.0 * A, 3.6 * A), (3.0 * A, 3.0 * A, 3.8 * A)),
            material="metal")
    _src(sim, 1.0 * A, 2.5 * A, 2.5 * A)
    return sim, dict(subpixel_smoothing="kottke_pec")


def cfg_nonuniform_interior_dielectric():
    L = 24 * DX
    dz = [DX] * 8 + [2 * DX] * 8 + [DX] * 8
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(L, L, 0.0), dx=DX,
                     dz_profile=dz, boundary="cpml", cpml_layers=6)
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((6 * DX, 6 * DX, 6 * DX), (18 * DX, 18 * DX, 14 * DX)),
            material="d")
    _src(sim, 4 * DX, 12 * DX, 10 * DX)
    return sim, dict(subpixel_smoothing=True)


CONFIGS = {
    "vacuum_pads_2d": cfg_vacuum_pads_2d,
    "interior_dielectric_2d": cfg_interior_dielectric_2d,
    "interior_dielectric_upml_2d": cfg_interior_dielectric_upml_2d,
    "touching_dielectric_subpixel_off_2d": cfg_touching_dielectric_subpixel_off_2d,
    "touching_dielectric_pec_walls_2d": cfg_touching_dielectric_pec_walls_2d,
    "interior_dielectric_3d": cfg_interior_dielectric_3d,
    "kottke_pec_interior_3d": cfg_kottke_pec_interior_3d,
    "nonuniform_interior_dielectric": cfg_nonuniform_interior_dielectric,
}

out = {"x64": bool(__import__("jax").config.jax_enable_x64),
       "rfx_file": rfx.__file__, "digests": {}}
for name, build in CONFIGS.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sim, kw = build()
            res = sim.run(n_steps=N_STEPS, skip_preflight=True, **kw)
            out["digests"][name] = _digest(res)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            out["digests"][name] = f"ERROR: {type(exc).__name__}: {exc}"
print(json.dumps(out, indent=2, sort_keys=True))

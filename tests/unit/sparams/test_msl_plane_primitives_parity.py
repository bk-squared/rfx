"""Parity guard: the MSL plane-primitive V/I must equal compute_msl_s_matrix's
own V/I (issue #514).

Uses a monkeypatched ``Simulation.run`` that fabricates DFT-plane
accumulators from the component and its physical Yee sampling position,
independent of probe names. Both ``register_msl_plane_probes``'s own probes and
``compute_msl_s_matrix``'s internally-registered probes read through the
SAME monkeypatch, so this compares the two paths' V/I INTEGRATION LOGIC
directly on synthetic, node-dependent markers -- no real FDTD forward
needed, so it belongs in the default (non-slow) lane.

The #726 witness adds longitudinal phase: H at index i is sampled at
x_i+dx/2. A same-index-only current no longer passes against production's
two-plane interpolation. Explicit offsets keep both observation ladders
fixed; this test does not compare the two paths' automatic placement rules.

Before issue #514's fix this failed on both meshes: V read ~4.33x high on
the aligned mesh (the old inclusive ``k_lo..k_hi`` span summed the
PEC-interior marker cell 3 the corrected span excludes) and matched by
coincidence on the bisecting mesh (two compounding wrong spans landing on
the same node); I was wrong on both meshes (a single pre-#80 Hy slab vs.
the closed Ampere loop). Post-fix, ``_v_from_plane``/``_i_from_plane`` call
the SAME ``msl_modal_voltage``/``msl_loop_current`` calls
``compute_msl_s_matrix`` makes, so V and I must match to float32
reassociation noise on BOTH meshes -- this test asserts that, not the
pre-fix numbers (which live in the PR record, not as a live assertion
against code that no longer exists).
"""
from __future__ import annotations

from types import MethodType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.msl_wave_decomp import (
    _i_from_plane,
    _v_from_plane,
    register_msl_plane_probes,
)
from rfx.probes.probes import DFTPlaneProbe

EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
MARGIN = 2e-3
F_MAX = 5e9

# Node-dependent synthetic markers so the closed Ampere loop is not
# identically zero and the V span's inclusive/exclusive boundary is
# discriminating: Ez varies by z-node (node 3 IS the realized trace plane
# on both meshes below — #931 §1.3 puts a foil sheet on the node nearest
# its declared plane, and round(254/84.67) = round(254/80) = 3), Hy varies
# by k (bottom vs. top leg differ), Hz varies by j (left vs. right leg
# differ).
_EZ_MARKER = {0: 1.0, 1: 1.0, 2: 1.0, 3: 10.0, 4: -1000.0}
_RTOL = 1e-6  # fast (fake-run) lane tolerance -- same-function-call parity
_BETA = 2500.0  # rad/m; resolves the half-cell error in both test meshes


def _fake_run(sim, *, n_steps=None, num_periods=1.0, compute_s_params=False,
              report_every=None, report_label=None):
    """Sample synthetic fields at their actual Yee positions (no FDTD)."""
    grid = sim._build_grid()
    planes = {}
    kk = jnp.arange(grid.nz, dtype=jnp.float32)
    jj = jnp.arange(grid.ny, dtype=jnp.float32)
    for entry in sim._dft_planes:
        index = grid.position_to_index((entry.coordinate, 0.0, 0.0))[0]
        sample_x = (index - grid.pad_x_lo) * grid.dx
        if entry.component.startswith("h"):
            sample_x += grid.dx / 2
        if entry.component == "ez":
            prof = jnp.asarray(
                [complex(_EZ_MARKER.get(k, 0.0)) for k in range(grid.nz)],
                dtype=jnp.complex64,
            )
            acc = jnp.broadcast_to(prof[None, None, :], (1, grid.ny, grid.nz))
        elif entry.component == "hy":
            acc = jnp.broadcast_to(
                ((1.0 + 0.1 * kk) * (1 + 0.2j)).astype(jnp.complex64)[None, None, :],
                (1, grid.ny, grid.nz),
            )
        else:  # hz
            acc = jnp.broadcast_to(
                ((0.5 + 0.01 * jj) * (1 - 0.3j)).astype(jnp.complex64)[None, :, None],
                (1, grid.ny, grid.nz),
            )
        acc = acc * jnp.asarray(np.exp(-1j * _BETA * sample_x), dtype=acc.dtype)
        planes[entry.name] = DFTPlaneProbe(
            accumulator=acc, freqs=entry.freqs, component=entry.component,
            axis=0, index=index, total_steps=1, window="rect", window_alpha=0.25,
        )
    return SimpleNamespace(dft_planes=planes)


def _build_thru(dx: float) -> Simulation:
    """Two-port MSL thru-line, cross-section only wide/tall enough to hold
    the port geometry -- no forward is ever run against it."""
    lx = L_LINE + 2 * MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * dx)
    lz = H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, H_SUB)), material="sub")
    y_c = ly / 2
    # 35 um foil -> a SHEET on the laminate face (#931 §1.3).
    sim.add(
        Box((0, y_c - W_TRACE / 2, H_SUB), (lx, y_c + W_TRACE / 2, H_SUB)),
        material="pec",
    )
    for x, d in ((MARGIN, "+x"), (MARGIN + L_LINE, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=W_TRACE, height=H_SUB,
                          direction=d, impedance=50.0,
                          n_probe_offset=10, n_probe_spacing=2, n_probes=3)
    return sim


@pytest.mark.parametrize("dx,label", [
    (H_SUB / 3, "aligned dx=h_sub/3"),   # h_sub/dx = 3.0 exactly: node-aligned
    (80e-6, "bisecting dx=80um"),         # h_sub/dx = 3.175: the sheet
                                          # snaps down to node 3, 14 um
                                          # below the declared face
])
@pytest.mark.parametrize("port_index", [0, 1], ids=["plus_x", "minus_x"])
def test_plane_path_v_and_i_match_production(dx, label, port_index, tmp_path):
    sim = _build_thru(dx)
    freqs = jnp.asarray([1.0e9], dtype=jnp.float32)
    ps = register_msl_plane_probes(sim, port_index=port_index, freqs=freqs, name_prefix="d")
    assert ps.hy_name == "d_hy" and ps.hz_name == "d_hz"
    assert [entry.name for entry in sim._dft_planes[:5]] == [
        "d_ez1", "d_ez2", "d_ez3", "d_hy", "d_hz",
    ]

    sim.run = MethodType(_fake_run, sim)
    fr = sim.run()
    for left_name, right_name in ((ps.hy_left_name, ps.hy_name),
                                 (ps.hz_left_name, ps.hz_name)):
        left = np.asarray(fr.dft_planes[left_name].accumulator)
        right = np.asarray(fr.dft_planes[right_name].accumulator)
        # Keep this fixture capable of refuting a right-plane-only reader.
        assert np.linalg.norm(left - right) > 0.05 * np.linalg.norm(right)
    v_copy = complex(np.asarray(_v_from_plane(fr, ps.ez1_name, ps))[0])
    i_copy = complex(np.asarray(_i_from_plane(fr, ps.hy_name, ps))[0])

    dump_path = tmp_path / "dump.npz"
    # sim.run stays monkeypatched: compute_msl_s_matrix's OWN dft-plane
    # registrations (different names, same components) are fed through the
    # identical fake accumulators, so this is a same-input comparison of
    # the two paths' integration logic, not a comparison across two
    # different (real vs. fake) field states.
    sim.compute_msl_s_matrix(
        n_steps=1, freqs=freqs, num_periods=1.0, enforce_passivity=False,
        raw_3probe_dump_path=str(dump_path),
    )
    with np.load(dump_path) as d:
        v_prod = complex(np.asarray(d["raw_v"])[0, port_index, 0, 0])
        i_prod = complex(np.asarray(d["raw_i1"])[0, port_index, 0])

    assert v_prod != 0.0 and i_prod != 0.0, (
        f"[{label}] fixture produced an all-zero production reference"
    )
    assert abs(v_copy / v_prod - 1.0) < _RTOL, (
        f"[{label}] V mismatch: plane-path={v_copy}  production={v_prod}  "
        f"k_lo={ps.k_lo} k_hi(excl)={ps.k_hi}"
    )
    assert abs(i_copy / i_prod - 1.0) < _RTOL, (
        f"[{label}] I mismatch: plane-path={i_copy}  production={i_prod}  "
        f"j=[{ps.j_lo},{ps.j_hi}] k_trace=[{ps.k_trace_lo},{ps.k_trace_hi}]"
    )


@pytest.mark.parametrize("missing", ["hy_left_name", "hz_left_name", "legacy_metadata"])
def test_plane_current_refuses_missing_bracketing_data_or_legacy_metadata(missing):
    sim = _build_thru(H_SUB / 3)
    ps = register_msl_plane_probes(sim, port_index=0, freqs=jnp.asarray([1e9]))
    fr = _fake_run(sim)
    if missing == "legacy_metadata":
        # The original constructor keyword set remains accepted, but its
        # absent H stencil must not silently select staggered extraction.
        legacy = vars(ps).copy()
        for name in ("hy_left_name", "hz_left_name", "h_weights"):
            legacy.pop(name)
        ps = type(ps)(**legacy)
    else:
        fr.dft_planes.pop(getattr(ps, missing))
    with pytest.raises(ValueError, match="requires bracketing H-plane"):
        _i_from_plane(fr, ps.hy_name, ps)

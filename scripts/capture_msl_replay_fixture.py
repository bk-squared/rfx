#!/usr/bin/env python3
"""Capture MSL DFT-plane accumulators and numpy-f64 golden for replay-based tests.

Run once (from repo root) to regenerate fixtures:
    python scripts/capture_msl_replay_fixture.py

Outputs:
    tests/fixtures/msl_replay_accumulators.npz  — complex64 DFT-plane arrays
    tests/fixtures/msl_replay_golden_f64.npy    — complex128 S-matrix from
                                                   numpy assembly on same data
                                                   (mirrors S1 V·I split logic)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

# No x64 — float32 production run.
import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

# Geometry constants (same as test_msl_port_integration.py).
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
DX = 80e-6
F_MAX = 5e9
LX = L_LINE + 2 * PORT_MARGIN
LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.5e-3

N_FREQS_REPLAY = 5
FREQS_REPLAY = np.linspace(F_MAX / 10, F_MAX, N_FREQS_REPLAY, dtype=np.float32)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
ACCUMULATORS_PATH = FIXTURES / "msl_replay_accumulators.npz"
GOLDEN_F64_PATH = FIXTURES / "msl_replay_golden_f64.npy"


def _build_sim() -> Simulation:
    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, LZ),
        dx=DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)), material="ro4350b")
    y_centre = LY / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )
    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="+x",
        impedance=50.0,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="-x",
        impedance=50.0,
    )
    return sim


def _intercept_run(accumulator_store: dict):
    """Return a monkeypatch for sim.run that stores DFT planes then returns result."""
    real_run_results: dict = {}

    def intercepting_run(self, *, n_steps=None, num_periods=40.0, compute_s_params=False):
        run_idx = len(real_run_results)
        result = type(self).run(self, n_steps=n_steps, num_periods=num_periods,
                                compute_s_params=False)
        if result.dft_planes:
            for name, probe in result.dft_planes.items():
                arr = np.asarray(probe.accumulator, dtype=np.complex64)
                accumulator_store[f"run{run_idx}__{name}"] = arr
        real_run_results[run_idx] = result
        return result

    return intercepting_run


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"Capturing MSL replay fixture with {N_FREQS_REPLAY} frequencies...")
    print(f"  freqs: {FREQS_REPLAY}")

    sim = _build_sim()
    freqs_jnp = jnp.asarray(FREQS_REPLAY, dtype=jnp.float32)

    accumulator_store: dict = {}
    sim.run = MethodType(_intercept_run(accumulator_store), sim)

    result = sim.compute_msl_s_matrix(
        freqs=freqs_jnp,
        num_periods=8,
    )

    S_production = np.asarray(result.S, dtype=np.complex128)
    print(f"  Production S shape: {S_production.shape}")
    print(f"  Captured {len(accumulator_store)} accumulator arrays")
    for k, v in sorted(accumulator_store.items()):
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

    np.savez_compressed(
        ACCUMULATORS_PATH,
        freqs=FREQS_REPLAY,
        **accumulator_store,
    )
    size_acc = ACCUMULATORS_PATH.stat().st_size / 1024
    print(f"\nSaved accumulators: {ACCUMULATORS_PATH} ({size_acc:.1f} KB)")

    print("\nComputing numpy-f64 golden from captured accumulators (S1 V·I split)...")
    golden_S = _compute_numpy_f64_golden_s1(accumulator_store, FREQS_REPLAY, sim)
    np.save(GOLDEN_F64_PATH, golden_S)
    size_gold = GOLDEN_F64_PATH.stat().st_size / 1024
    print(f"Saved f64 golden:   {GOLDEN_F64_PATH} ({size_gold:.1f} KB)")
    print(f"  Golden S shape: {golden_S.shape}, dtype: {golden_S.dtype}")

    delta = np.max(np.abs(S_production - golden_S))
    print(f"\n[f32 production vs f64 numpy golden] max_abs_dev = {delta:.3e}")
    print("  (This is the documented float32 deployment delta.)")


def _compute_numpy_f64_golden_s1(
    accumulator_store: dict,
    freqs: np.ndarray,
    sim_ref: Simulation,
) -> np.ndarray:
    """Reproduce the S1 V·I assembly in numpy-f64 from captured complex64 accumulators.

    Mirrors compute_msl_s_matrix's S1 logic:
      - voltage V = integral Ez dz at probe-0 plane, upcast to complex128
      - current I = msl_loop_current (closed Ampere ∮H·dl), upcast to complex128
      - S11 = b/a = (V - Z0*I/2) / (V + Z0*I/2)   [V·I wave split]
      - S21 = b_passive / a_driven
    This is identical to the jnp assembly except all arithmetic is numpy float64,
    so the two results agree to near-machine-epsilon (~1e-13).
    """
    from rfx.sources.msl_port import (
        MSLPort,
        _msl_yz_cells,
        msl_loop_current,
        msl_probe_x_coords_n,
    )
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0

    _C0 = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))

    entries = list(sim_ref._msl_ports)
    n_ports = len(entries)
    n_freqs = len(freqs)

    grid = sim_ref._build_grid()

    msl_ports: list[MSLPort] = []
    for pe in entries:
        x_feed, y_centre, z_lo = pe.position
        msl_ports.append(MSLPort(
            feed_x=float(x_feed),
            y_lo=float(y_centre - pe.width / 2),
            y_hi=float(y_centre + pe.width / 2),
            z_lo=float(z_lo),
            z_hi=float(z_lo + pe.height),
            direction=pe.direction,
            impedance=pe.impedance,
            excitation=pe.waveform,
        ))

    n_probes_per_port = [int(pe.n_probes) for pe in entries]
    probe_xs = [
        msl_probe_x_coords_n(
            grid, mp,
            n_probes=n_probes,
            n_offset_cells=pe.n_probe_offset,
            n_spacing_cells=pe.n_probe_spacing,
        )
        for mp, pe, n_probes in zip(msl_ports, entries, n_probes_per_port)
    ]

    dx = float(grid.dx)
    dy_arr = np.full(grid.ny, dx)
    dz_arr = np.full(grid.nz, dx)

    port_idx_meta = []
    for mp in msl_ports:
        cells = _msl_yz_cells(grid, mp)
        j_set = sorted({c[1] for c in cells})
        k_set = sorted({c[2] for c in cells})
        j_lo, j_hi = j_set[0], j_set[-1]
        k_lo, k_hi = k_set[0], k_set[-1]
        j_centre = (j_lo + j_hi) // 2
        k_top = k_hi
        port_idx_meta.append(dict(
            j_lo=j_lo, j_hi=j_hi,
            k_lo=k_lo, k_hi=k_hi,
            j_centre=j_centre, k_top=k_top,
            height=mp.z_hi - mp.z_lo,
        ))

    # Analytic Z0 per port (same logic as compute_msl_s_matrix).
    _msl_assembled = sim_ref._assemble_materials(grid)
    _msl_materials = _msl_assembled[0]
    _msl_pec_mask = (
        None if _msl_assembled[3] is None
        else np.asarray(_msl_assembled[3])
    )
    z0_hj_per_port: list[float] = []
    trace_k_per_port: list[tuple[int, int]] = []
    for p_idx, pe in enumerate(entries):
        meta = port_idx_meta[p_idx]
        if pe.eps_r_sub is not None:
            eps_r_ref = float(pe.eps_r_sub)
        else:
            k_mid = (meta["k_lo"] + meta["k_hi"]) // 2
            i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
            eps_r_ref = float(
                np.asarray(_msl_materials.eps_r[i_feed_p, meta["j_centre"], k_mid])
            )
        z0_hj, _ = hammerstad_jensen_z0_eps_eff(pe.width, pe.height, eps_r_ref)
        z0_hj_per_port.append(float(z0_hj))

        # Trace k span (same as compute_msl_s_matrix).
        i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
        col = (
            None if _msl_pec_mask is None
            else _msl_pec_mask[i_feed_p, meta["j_centre"], meta["k_top"]:]
        )
        k_pec = np.array([], dtype=int) if col is None else np.where(col)[0]
        if k_pec.size == 0:
            raise RuntimeError(f"No PEC trace for port {entries[p_idx].name!r}")
        trace_k_per_port.append((
            int(meta["k_top"] + int(k_pec.min())),
            int(meta["k_top"] + int(k_pec.max())),
        ))

    S = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)

    for driven in range(n_ports):
        def _get(name: str) -> np.ndarray:
            key = f"run{driven}__{name}"
            return accumulator_store[key].astype(np.complex128)

        # Build probe name lists (must match what compute_msl_s_matrix registers).
        ez_probe_names = [[] for _ in range(n_ports)]
        hy_probe_names = [None] * n_ports
        hz_probe_names = [None] * n_ports
        for p_idx, pxs in enumerate(probe_xs):
            for q_idx in range(len(pxs)):
                ez_probe_names[p_idx].append(f"_msl_run{driven}_p{p_idx}_ez{q_idx}")
            hy_probe_names[p_idx] = f"_msl_run{driven}_p{p_idx}_hy"
            hz_probe_names[p_idx] = f"_msl_run{driven}_p{p_idx}_hz"

        # Integrate V (Ez) and I (∮H·dl) for each port.
        v0_per_port: list[np.ndarray] = []  # voltage at probe 0
        i_per_port: list[np.ndarray] = []

        for p_idx, meta in enumerate(port_idx_meta):
            # Voltage at probe 0: integral Ez dz in centre column.
            ez_plane = _get(ez_probe_names[p_idx][0])  # (n_freqs, ny, nz), c128
            v_f = np.zeros(n_freqs, dtype=np.complex128)
            for k in range(meta["k_lo"], meta["k_hi"] + 1):
                v_f = v_f + ez_plane[:, meta["j_centre"], k] * float(dz_arr[k])
            v0_per_port.append(v_f)

            # Closed Ampere-loop current (S1 msl_loop_current, upcast to c128).
            hy_plane = _get(hy_probe_names[p_idx])
            hz_plane = _get(hz_probe_names[p_idx])
            # Leapfrog E/H half-step time correction — MIRROR of
            # compute_msl_s_matrix (rfx/api/_sparams.py): the DFT plane probe
            # timestamps H at t but H lives at t - dt/2, so I = ∮H·dl is missing
            # the exp(+jω·dt/2) the flux monitor applies. Ez (V) needs none.
            _hs_phase = np.exp(
                1j * 2.0 * np.pi * np.asarray(freqs) * (float(grid.dt) * 0.5))
            hy_plane = hy_plane * _hs_phase[:, None, None]
            hz_plane = hz_plane * _hs_phase[:, None, None]
            k_tr_lo, k_tr_hi = trace_k_per_port[p_idx]
            i_f = msl_loop_current(
                hy_plane, hz_plane,
                j_lo=meta["j_lo"], j_hi=meta["j_hi"],
                k_trace_lo=k_tr_lo, k_trace_hi=k_tr_hi,
                dy_arr=dy_arr, dz_arr=dz_arr,
                direction=msl_ports[p_idx].direction,
            )
            i_per_port.append(np.asarray(i_f, dtype=np.complex128))

        # V·I wave split — identical formula to S1 assembly.
        v0_d = v0_per_port[driven]
        i_d = i_per_port[driven]
        z0hj_d = z0_hj_per_port[driven]
        a_fwd_d = 0.5 * (v0_d + z0hj_d * i_d)
        b_ref_d = 0.5 * (v0_d - z0hj_d * i_d)
        S[driven, driven, :] = b_ref_d / (a_fwd_d + 1e-30)

        for j in range(n_ports):
            if j == driven:
                continue
            v0_p = v0_per_port[j]
            i_p = i_per_port[j]
            z0hj_p = z0_hj_per_port[j]
            b_out_p = 0.5 * (v0_p - z0hj_p * i_p)
            S[j, driven, :] = b_out_p / (a_fwd_d + 1e-30)

    return S


if __name__ == "__main__":
    main()

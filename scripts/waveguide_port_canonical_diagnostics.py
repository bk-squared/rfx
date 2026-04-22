#!/usr/bin/env python3
"""Canonical waveguide-port physics diagnostics.

Frozen baseline protocol for waveguide-port extractor work:
- frequency grid: np.linspace(4.5e9, 8.0e9, 20)
- num_periods: 40
- default acceptance policy: num_periods_dft=None

This script is diagnostic-first: it prints a concise text summary and can
optionally emit JSON for before/after comparisons.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.core.yee import init_materials as _init_vacuum_materials
from rfx.geometry.csg import Box
from rfx.simulation import run as run_simulation
from rfx.sources.waveguide_port import (
    _compute_beta,
    _extract_port_waves,
    _shift_modal_waves,
    waveguide_plane_positions,
)


FREQS_HZ = np.linspace(4.5e9, 8.0e9, 20)
DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09


@dataclass
class GeometryMetrics:
    name: str
    contract: str
    normalize: bool | None
    num_periods: float
    num_periods_dft: float | None
    mean_abs_s11: float
    max_abs_s11: float
    mean_abs_s21: float
    max_abs_s21: float
    mean_abs_s12: float
    max_abs_s12: float
    mean_abs_s22: float
    max_abs_s22: float
    mean_column_power: float
    max_column_power: float
    mean_reciprocity_rel_diff: float


def _port_bandwidth(freqs_hz: np.ndarray) -> float:
    f0 = float(freqs_hz.mean())
    return max(0.2, min(0.8, (freqs_hz[-1] - freqs_hz[0]) / max(f0, 1.0)))


def _make_base_sim(freqs_hz: np.ndarray) -> Simulation:
    sim = Simulation(
        freq_max=float(freqs_hz[-1]),
        domain=DOMAIN,
        boundary="cpml",
        cpml_layers=10,
    )
    f0 = float(freqs_hz.mean())
    bandwidth = _port_bandwidth(freqs_hz)
    port_freqs = jnp.asarray(freqs_hz)
    sim.add_waveguide_port(
        PORT_LEFT_X,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        name="left",
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        name="right",
    )
    return sim


def _add_pec_short(sim: Simulation) -> None:
    sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
    sim.add(Box((0.05, 0.0, 0.0), (0.055, 0.04, 0.02)), material="pec_like")


def _add_dielectric_slab(sim: Simulation) -> None:
    sim.add_material("eps4", eps_r=4.0, sigma=0.0)
    sim.add(Box((0.04, 0.0, 0.0), (0.06, 0.04, 0.02)), material="eps4")


def _add_asymmetric_obstacle(sim: Simulation) -> None:
    sim.add_material("eps6", eps_r=6.0, sigma=0.0)
    sim.add(Box((0.03, 0.0, 0.0), (0.05, 0.02, 0.02)), material="eps6")


def _build_geometry(name: str, freqs_hz: np.ndarray) -> Simulation:
    sim = _make_base_sim(freqs_hz)
    if name == "empty":
        return sim
    if name == "pec_short":
        _add_pec_short(sim)
        return sim
    if name == "dielectric_slab":
        _add_dielectric_slab(sim)
        return sim
    if name == "asymmetric_obstacle":
        _add_asymmetric_obstacle(sim)
        return sim
    raise ValueError(f"unknown geometry: {name}")


def _measure_geometry(
    name: str,
    *,
    freqs_hz: np.ndarray,
    num_periods: float,
    num_periods_dft: float | None,
    normalize: bool,
) -> GeometryMetrics:
    sim = _build_geometry(name, freqs_hz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.compute_waveguide_s_matrix(
            num_periods=num_periods,
            num_periods_dft=num_periods_dft,
            normalize=normalize,
        )
    s = np.asarray(result.s_params)
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    s12 = np.abs(s[0, 1, :])
    s22 = np.abs(s[1, 1, :])
    reciprocity = np.abs(s21 - s12) / np.maximum(np.maximum(s21, s12), 1e-12)
    return GeometryMetrics(
        name=name,
        contract="current_api",
        normalize=normalize,
        num_periods=num_periods,
        num_periods_dft=num_periods_dft,
        mean_abs_s11=float(np.mean(s11)),
        max_abs_s11=float(np.max(s11)),
        mean_abs_s21=float(np.mean(s21)),
        max_abs_s21=float(np.max(s21)),
        mean_abs_s12=float(np.mean(s12)),
        max_abs_s12=float(np.max(s12)),
        mean_abs_s22=float(np.mean(s22)),
        max_abs_s22=float(np.max(s22)),
        mean_column_power=float(np.mean(column_power)),
        max_column_power=float(np.max(column_power)),
        mean_reciprocity_rel_diff=float(np.mean(reciprocity)),
    )


def _reset_cfg(cfg, drive_enabled: bool):
    zeros = jnp.zeros_like(cfg.v_probe_dft)
    return cfg._replace(
        src_amp=cfg.src_amp if drive_enabled else 0.0,
        v_probe_dft=zeros,
        v_ref_dft=zeros,
        i_probe_dft=zeros,
        i_ref_dft=zeros,
        v_inc_dft=zeros,
    )


def _probe_port_waves(cfg):
    beta_loc = _compute_beta(cfg.freqs, cfg.f_cutoff)
    a_probe, b_probe = _extract_port_waves(cfg, cfg.v_probe_dft, cfg.i_probe_dft)
    return _shift_modal_waves(a_probe, b_probe, beta_loc, 0.0)


def _assemble_contract_v1(
    sim: Simulation,
    *,
    freqs_hz: np.ndarray,
    num_periods: float,
    num_periods_dft: float | None,
) -> np.ndarray:
    entries = list(sim._waveguide_ports)
    grid = sim._build_grid()
    mats, debye_spec, lorentz_spec, pec_mask_wg, _, _ = sim._assemble_materials(grid)
    if pec_mask_wg is not None:
        mats = mats._replace(sigma=jnp.where(pec_mask_wg, 1e10, mats.sigma))
    _, debye, lorentz = sim._init_dispersion(mats, grid.dt, debye_spec, lorentz_spec)
    n_steps = grid.num_timesteps(num_periods=num_periods)
    cfgs = [sim._build_waveguide_port_config(entry, grid, jnp.asarray(freqs_hz), n_steps) for entry in entries]
    ref_t0 = cfgs[0].src_t0
    ref_tau = cfgs[0].src_tau
    cfgs = [cfg._replace(src_t0=ref_t0, src_tau=ref_tau) for cfg in cfgs]
    if num_periods_dft is not None:
        n_dft = int(grid.num_timesteps(num_periods=num_periods_dft))
        cfgs = [cfg._replace(dft_end_step=n_dft) for cfg in cfgs]
    ref_materials = _init_vacuum_materials(grid.shape)
    common = dict(
        boundary="cpml",
        cpml_axes=grid.cpml_axes,
        pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
        periodic=None,
    )
    s = np.zeros((2, 2, len(freqs_hz)), dtype=np.complex128)
    for drive_idx in range(2):
        ref_cfgs = [_reset_cfg(cfg, idx == drive_idx) for idx, cfg in enumerate(cfgs)]
        ref_res = run_simulation(
            grid, ref_materials, n_steps,
            debye=None, lorentz=None,
            waveguide_ports=ref_cfgs, **common,
        )
        ref_finals = ref_res.waveguide_ports or ()
        a_drive_probe_ref, _ = _probe_port_waves(ref_finals[drive_idx])
        safe_a_ref = np.where(
            np.abs(np.asarray(a_drive_probe_ref)) > 1e-30,
            np.asarray(a_drive_probe_ref),
            1.0,
        )

        dev_cfgs = [_reset_cfg(cfg, idx == drive_idx) for idx, cfg in enumerate(cfgs)]
        dev_res = run_simulation(
            grid, mats, n_steps,
            debye=debye, lorentz=lorentz,
            waveguide_ports=dev_cfgs, **common,
        )
        dev_finals = dev_res.waveguide_ports or ()
        for recv_idx in range(2):
            _, b_recv_probe_dev = _probe_port_waves(dev_finals[recv_idx])
            s[recv_idx, drive_idx, :] = np.asarray(b_recv_probe_dev) / safe_a_ref

    # Directional amplitude normalization from the empty-guide contract itself.
    empty_tx_mag = [np.abs(s[1, 0, :]), np.abs(s[0, 1, :])]
    for drive_idx in range(2):
        tx_profile = np.where(empty_tx_mag[drive_idx] > 1e-30, empty_tx_mag[drive_idx], 1.0)
        s[:, drive_idx, :] /= tx_profile
    return s


def _measure_geometry_contract_v1(
    name: str,
    *,
    freqs_hz: np.ndarray,
    num_periods: float,
    num_periods_dft: float | None,
) -> GeometryMetrics:
    sim = _build_geometry(name, freqs_hz)
    s = _assemble_contract_v1(
        sim,
        freqs_hz=freqs_hz,
        num_periods=num_periods,
        num_periods_dft=num_periods_dft,
    )
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    s12 = np.abs(s[0, 1, :])
    s22 = np.abs(s[1, 1, :])
    reciprocity = np.abs(s21 - s12) / np.maximum(np.maximum(s21, s12), 1e-12)
    return GeometryMetrics(
        name=name,
        contract="v1_probe_ref",
        normalize=None,
        num_periods=num_periods,
        num_periods_dft=num_periods_dft,
        mean_abs_s11=float(np.mean(s11)),
        max_abs_s11=float(np.max(s11)),
        mean_abs_s21=float(np.mean(s21)),
        max_abs_s21=float(np.max(s21)),
        mean_abs_s12=float(np.mean(s12)),
        max_abs_s12=float(np.max(s12)),
        mean_abs_s22=float(np.mean(s22)),
        max_abs_s22=float(np.max(s22)),
        mean_column_power=float(np.mean(column_power)),
        max_column_power=float(np.max(column_power)),
        mean_reciprocity_rel_diff=float(np.mean(reciprocity)),
    )


def _print_summary(metrics: list[GeometryMetrics]) -> None:
    for item in metrics:
        print(
            f"[{item.name}] contract={item.contract} normalize={item.normalize} "
            f"num_periods={item.num_periods:g} num_periods_dft={item.num_periods_dft}"
        )
        print(
            "  "
            f"mean|S11|={item.mean_abs_s11:.6f} "
            f"max|S11|={item.max_abs_s11:.6f} "
            f"mean|S21|={item.mean_abs_s21:.6f} "
            f"mean|S12|={item.mean_abs_s12:.6f}"
        )
        print(
            "  "
            f"mean col power={item.mean_column_power:.6f} "
            f"max col power={item.max_column_power:.6f} "
            f"mean reciprocity rel diff={item.mean_reciprocity_rel_diff:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Measure only normalize=True. Default is both paths.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Measure only normalize=False.",
    )
    parser.add_argument(
        "--num-periods",
        type=float,
        default=40.0,
        help="Simulation length in source periods (default: 40).",
    )
    parser.add_argument(
        "--num-periods-dft",
        type=float,
        default=None,
        help="Optional DFT gate length in source periods.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write JSON metrics.",
    )
    parser.add_argument(
        "--contract",
        choices=("current", "v1", "both"),
        default="current",
        help="Which contract to evaluate (default: current).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.normalize and args.raw_only:
        raise SystemExit("--normalize and --raw-only are mutually exclusive")

    modes = [True, False]
    if args.normalize:
        modes = [True]
    elif args.raw_only:
        modes = [False]

    metrics: list[GeometryMetrics] = []
    geometries = (
        "empty",
        "pec_short",
        "dielectric_slab",
        "asymmetric_obstacle",
    )
    if args.contract in ("current", "both"):
        for normalize in modes:
            for geometry in geometries:
                metrics.append(
                    _measure_geometry(
                        geometry,
                        freqs_hz=FREQS_HZ,
                        num_periods=args.num_periods,
                        num_periods_dft=args.num_periods_dft,
                        normalize=normalize,
                    )
                )
    if args.contract in ("v1", "both"):
        for geometry in geometries:
            metrics.append(
                _measure_geometry_contract_v1(
                    geometry,
                    freqs_hz=FREQS_HZ,
                    num_periods=args.num_periods,
                    num_periods_dft=args.num_periods_dft,
                )
            )

    _print_summary(metrics)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([asdict(item) for item in metrics], fh, indent=2)


if __name__ == "__main__":
    main()

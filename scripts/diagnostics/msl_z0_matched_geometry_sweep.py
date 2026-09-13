#!/usr/bin/env python3
"""Fresh #752 matched-geometry measurements; historical records are never inputs.

See docs/research_notes/issue752/fresh/protocol.md. A new output directory is
required. Each --case is an independent process; run cases sequentially.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import warnings

import jax
import numpy as np

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
from rfx.sources.msl_port import msl_port_from_entry, msl_probe_x_coords_n

CASES = (
    ("h3", 254e-6 / 3),
    ("h4", 254e-6 / 4),
    ("h5", 254e-6 / 5),
    ("h6", 254e-6 / 6),
    ("dx80", 80e-6),
    ("dx60", 60e-6),
)
PERIODS = 20.0
FREQUENCIES = np.linspace(0.5e9, 5e9, 30, dtype=np.float32)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def array_signature(value):
    a = np.ascontiguousarray(value)
    return dict(
        shape=list(a.shape),
        dtype=a.dtype.str,
        sha256=hashlib.sha256(a.tobytes()).hexdigest(),
    )


def nearest_lower_tie(value):
    """Positive integer count, with exact half-integer ties toward the lower."""
    return int(math.ceil(value - 0.5))


def hj1980(width, height, eps_r):
    """Independent quasi-static zero-thickness HJ, Qucs equations 11.4–11.18.

    https://qucs.github.io/tech/node75.html ; this is a diagnostic reference,
    never passed into RFX's source, beta search or S normalization.
    """
    u = width / height
    a = 1 + math.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49
    a += math.log(1 + (u / 18.1) ** 3) / 18.7
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3)) ** 0.053
    ee = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 10 / u) ** (-a * b)
    fu = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    eta0 = 376.730313668  # SI free-space impedance; reference only
    z = (
        eta0
        / (2 * math.pi * math.sqrt(ee))
        * math.log(fu / u + math.sqrt(1 + (2 / u) ** 2))
    )
    return z, ee


def build_case(label):
    dx = dict(CASES)[label]
    nh = nearest_lower_tie(254e-6 / dx)
    nw = nearest_lower_tie(600e-6 / dx)
    # Choose integer faces before measuring anything. The trace edge, dielectric
    # interface and port top refer to the SAME node; the desired board remains
    # separate metadata. Side and top clearance >= 8 substrate heights.
    side = math.ceil(8 * nh)
    nx = math.ceil(0.040 / dx)
    ny = nw + 2 * side
    nz = nh + math.ceil(8 * nh)
    h, w = nh * dx, nw * dx
    y0, y1 = side * dx, (side + nw) * dx
    yc = (y0 + y1) / 2
    margin = math.ceil(0.002 / dx)
    feeds = (margin * dx, (nx - margin) * dx)
    offset, spacing = math.ceil(0.006 / dx), math.ceil(0.003 / dx)
    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0.0, 0.0, 0.0), (nx * dx, ny * dx, h)), material="substrate")
    sim.add(Box((0.0, y0, h), (nx * dx, y1, h)), material="pec")
    for index, (feed, direction) in enumerate(zip(feeds, ("+x", "-x"))):
        sim.add_msl_port(
            position=(feed, yc, 0.0),
            width=w,
            height=h,
            direction=direction,
            impedance=50.0,
            name=f"p{index}",
            n_probe_offset=offset,
            n_probe_spacing=spacing,
            n_probes=5,
            eps_r_sub=3.66,
            waveform=GaussianPulse(f0=3.75e9, bandwidth=0.8),
        )
    assembled = sim._msl_assemble_once()
    assert assembled is not None
    grid, materials, _, _, realized = assembled
    nodes = coords_from_uniform_grid(grid)
    i = grid.pad_x_lo + nx // 2
    k = nh  # bottom is a PEC domain face: no lower-z CPML padding
    assert grid.pad_z_lo == 0 and realized.pec_mask is None
    ey = np.asarray(realized.edges[1])[i, :, k]
    owned = np.flatnonzero(ey)
    assert len(owned) == nw and np.all(np.diff(owned) == 1)
    actual_width = float(nodes.y[owned[-1] + 1] - nodes.y[owned[0]])
    assert np.isclose(actual_width, w, rtol=0, atol=1e-17)
    assert np.flatnonzero(np.any(realized.edges[1], axis=(0, 1))).tolist() == [k]
    eps = np.asarray(materials.eps_r)
    j = grid.pad_y_lo + side + nw // 2
    assert np.all(eps[i, j, :k] == np.float32(3.66))
    assert np.all(eps[i, j, k:] == 1.0)
    gaps = [sim._msl_conductor_gap(pe, assembled) for pe in sim._msl_ports]
    assert all(g is not None and "h" in g for g in gaps), gaps
    assert all(np.isclose(g["h"], h, rtol=0, atol=1e-17) for g in gaps)
    xs = [
        list(
            map(
                float,
                msl_probe_x_coords_n(
                    grid,
                    msl_port_from_entry(pe),
                    pe.n_probes,
                    pe.n_probe_offset,
                    pe.n_probe_spacing,
                ),
            )
        )
        for pe in sim._msl_ports
    ]
    # Both ladders clear both feeds by > 5h and do not reach each other’s feed.
    assert min(abs(x - f) for ladder in xs for x in ladder for f in feeds) >= 5 * h
    legacy, ee = hammerstad_jensen_z0_eps_eff(actual_width, h, 3.66)
    hj, hj_ee = hj1980(actual_width, h, 3.66)
    plan = dict(
        label=label,
        desired_width_m=600e-6,
        desired_height_m=254e-6,
        dx_m=dx,
        height_m=h,
        width_m=actual_width,
        trace_thickness_m=0.0,
        height_intervals=nh,
        width_intervals=nw,
        domain_m=list(sim._domain),
        grid_shape=list(grid.shape),
        dt_s=float(grid.dt),
        n_steps=int(grid.num_timesteps(num_periods=PERIODS)),
        num_periods=PERIODS,
        actual_duration_s=int(grid.num_timesteps(num_periods=PERIODS)) * float(grid.dt),
        pulse_f0_hz=3.75e9,
        pulse_bandwidth=0.8,
        eps_r=3.66,
        feed_positions_m=list(feeds),
        probe_positions_m=xs,
        port_definitions=[asdict(p) for p in sim._msl_ports],
        repo_simplified_reference_ohm=legacy,
        repo_eps_eff=ee,
        hj1980_reference_ohm=hj,
        hj1980_eps_eff=hj_ee,
        material_signatures={
            name: array_signature(value)
            for name, value in zip(materials._fields, materials)
        },
        pec_edge_signatures=[array_signature(e) for e in realized.edges],
    )
    return sim, plan


def run_case(label, out, *, dry_run=False):
    import rfx.simulation as engine

    out.mkdir(parents=True, exist_ok=False)
    sim, plan = build_case(label)
    plan.update(
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,
        jax=jax.__version__,
        backend=jax.default_backend(),
        x64=bool(jax.config.x64_enabled),
        frequencies_hz=FREQUENCIES.tolist(),
    )
    write_json(out / "plan.json", plan)
    original = engine.run
    calls = []

    class BuildCaptured(Exception):
        pass

    def recorded(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        inp = bound.arguments
        drive = len(calls)
        mats = inp["materials"]
        # sigma legitimately contains source/load conductance. The dielectric
        # and the entire conductor edge set must equal this arm's own plan.
        for name in ("eps_r", "mu_r"):
            assert (
                array_signature(getattr(mats, name))
                == plan["material_signatures"][name]
            )
        edges = inp["pec_edge_masks"]
        assert edges is not None
        assert [array_signature(e) for e in edges] == plan["pec_edge_signatures"]
        assert list(inp["grid"].shape) == plan["grid_shape"]
        assert int(inp["n_steps"]) == plan["n_steps"]
        call = dict(
            drive=drive,
            n_steps=int(inp["n_steps"]),
            material_signatures={
                n: array_signature(v) for n, v in zip(mats._fields, mats)
            },
            probes=[p._asdict() for p in inp["probes"]],
            sources=[
                dict(
                    i=p.i,
                    j=p.j,
                    k=p.k,
                    component=p.component,
                    waveform=array_signature(p.waveform),
                )
                for p in inp["sources"]
            ],
            boundary=inp["boundary"],
            cpml_axes=inp["cpml_axes"],
            pec_axes=inp["pec_axes"],
            periodic=inp["periodic"],
            dft_planes=[
                dict(
                    component=p.component,
                    axis=p.axis,
                    index=p.index,
                    region=p.region,
                    freqs_hz=np.asarray(p.freqs).tolist(),
                    total_steps=p.total_steps,
                    window=p.window,
                    window_alpha=p.window_alpha,
                )
                for p in inp["dft_planes"]
            ],
        )
        calls.append(call)
        write_json(out / "consumed-plans.json", calls)
        if dry_run:
            raise BuildCaptured()
        inputs = {f"material_{n}": np.asarray(v) for n, v in zip(mats._fields, mats)}
        inputs.update({f"pec_{c}": np.asarray(v) for c, v in enumerate(edges)})
        inputs.update(
            {
                f"source_{n}": np.asarray(p.waveform)
                for n, p in enumerate(inp["sources"])
            }
        )
        np.savez_compressed(out / f"drive{drive}-inputs.npz", **inputs)
        t0 = time.monotonic()
        result = original(*args, **kwargs)
        arrays = {"time_series": np.asarray(result.time_series)}
        for n, p in enumerate(result.dft_planes or ()):
            arrays[f"dft_{n}"] = np.asarray(p.accumulator)
        np.savez_compressed(out / f"drive{drive}-records.npz", **arrays)
        call["wallclock_s"] = time.monotonic() - t0
        write_json(out / "consumed-plans.json", calls)
        return result

    engine.run = recorded
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sim.compute_msl_s_matrix(
                freqs=FREQUENCIES,
                num_periods=PERIODS,
                enforce_passivity=False,
                raw_3probe_dump_path=str(out / "raw-vi.npz"),
                report_every=5000,
            )
        arrays = {
            n: np.asarray(getattr(result, n))
            for n in (
                "S",
                "Z0",
                "beta",
                "freqs",
                "reliable",
                "settling_db",
                "cond_a",
                "beta_railed",
                "reference_impedances",
            )
            if getattr(result, n) is not None
        }
        np.savez_compressed(out / "result.npz", **arrays)
        write_json(
            out / "diagnostics.json",
            dict(
                warnings=[str(w.message) for w in caught],
                probe_clearance=[asdict(c) for c in result.probe_clearance],
                assembly=result.assembly,
                completed_drives=len(calls),
            ),
        )
        assert len(calls) == 2
        print(f"COMPLETE {label}", flush=True)
    except BuildCaptured:
        assert dry_run and len(calls) == 1
        print(f"BUILD CAPTURED {label}", flush=True)
    finally:
        engine.run = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=[x[0] for x in CASES])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.build_only:
        a.out.mkdir(parents=True, exist_ok=False)
        for label, _ in CASES:
            _, plan = build_case(label)
            write_json(a.out / f"{label}.json", plan)
            print(
                label, plan["height_m"], plan["width_m"], plan["grid_shape"], flush=True
            )
    else:
        if a.case is None:
            p.error("--case is required for a field measurement")
        run_case(a.case, a.out, dry_run=a.dry_run)


if __name__ == "__main__":
    main()

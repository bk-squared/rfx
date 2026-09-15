"""Paired observation-offset experiment on current cv06b (issue #726).

The source, load, material, mesh and DUT stay fixed. Only p1's probe offset
changes; p2 has the same ladder in both runs. Both input arms are certified
on the realized grid before either solve. Raw phasors and unprojected S are
saved even if settling fails. Such a failure prevents a numerical verdict.

This is a sensitivity experiment, not an exact |S11|=1 reference: an open
microstrip may radiate, and the fixed analytic reference impedance may
differ from the realized line impedance. S is reported at each first probe,
without translation to the feed plane. Even propagating fields can then
show magnitude variation with position; a difference cannot by itself
identify evanescent contamination.
The ideal quarter-wave circuit is a model, not the full-wave fixture.

Replaces the 2026-08-27 experiment, which moved the source as well as the
probes and printed a verdict despite failed settling. Historical output is
unchanged. No 2 dB winner-selection threshold is retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from rfx.api._preflight import (
    msl_min_probe_clearance,
    msl_source_near_field_standoff_cells,
)
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources.msl_port import (
    msl_cross_section_span,
    msl_port_from_entry,
    msl_probe_x_coords_n,
)

REPO = Path(__file__).resolve().parents[2]
CV06B = REPO / "validation/crossval/06b_msl_notch_filter_uniform.py"
N_PROBES = 5
SPACING = 2


def _load_case():
    spec = importlib.util.spec_from_file_location("_clearance_cv06b", CV06B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _realized(sim):
    # Reuse the fixture owner, which queries the solver's PEC realization.
    sys.path.insert(0, str(REPO / "tests"))
    from _realized_geometry import realized

    return realized(sim)


def _source_spec(entry):
    return {k: v for k, v in asdict(entry).items()
            if k not in {"n_probes", "n_probe_offset", "n_probe_spacing"}}


def _certificate(sim, rz, stub_bounds):
    x = np.asarray(coords_from_uniform_grid(rz.grid).x, dtype=float)
    ez_pec = np.asarray(rz.edge_masks[2])
    ports = []
    for entry in sim._msl_ports:
        mp = msl_port_from_entry(entry)
        span = msl_cross_section_span(rz.grid, mp)
        coords = msl_probe_x_coords_n(
            rz.grid, mp, n_probes=entry.n_probes,
            n_offset_cells=entry.n_probe_offset,
            n_spacing_cells=entry.n_probe_spacing,
        )
        indices = [int(np.argmin(abs(x - p))) for p in coords]
        sign = 1 if entry.direction == "+x" else -1
        front = stub_bounds[0 if sign > 0 else 1]
        gaps = sign * (front - np.asarray(coords))
        feed = float(x[span["i_feed"]])
        standoff = msl_source_near_field_standoff_cells(entry.height, rz.grid.dx)
        if len(set(indices)) != N_PROBES:
            raise ValueError("probe ladder clamps or repeats a realized plane")
        if any(g <= 0 for g in gaps):
            raise ValueError("probe ladder touches or crosses the realized stub front")
        if sign * (indices[0] - span["i_feed"]) < standoff:
            raise ValueError("first probe violates the existing source standoff")
        for i in indices:
            # All substrate-normal edges in the voltage-integration region
            # must remain live, including the point-witness column.
            if ez_pec[i, span["w_lo"]:span["w_hi"] + 1,
                      span["n_lo"]:span["n_hi"]].any():
                raise ValueError("a probe voltage path intersects realized PEC")
        ports.append(dict(
            name=entry.name, feed_x_m=feed, probe_x_m=list(coords),
            first_gap_m=float(gaps[0]), deepest_gap_m=float(gaps[-1]),
            source_standoff_cells=standoff, offset_cells=entry.n_probe_offset,
            spacing_cells=entry.n_probe_spacing,
            recommended_gap_m=msl_min_probe_clearance(sim._freq_max),
            source=_source_spec(entry),
        ))
    return ports


def prepare_inputs():
    cv = _load_case()
    base = cv._build_sim()
    metal = cv.assert_realized_metal(base)
    rz = _realized(base)
    x = np.asarray(coords_from_uniform_grid(rz.grid).x, dtype=float)
    stub_bounds = tuple(float(x[i]) for i in metal["stub_i"])
    clearance = msl_min_probe_clearance(cv.F_MAX)
    span_cells = (N_PROBES - 1) * SPACING
    clean_offsets = []
    near_offset = None
    for p, entry in enumerate(base._msl_ports):
        span = msl_cross_section_span(rz.grid, msl_port_from_entry(entry))
        sign = 1 if entry.direction == "+x" else -1
        front_i = metal["stub_i"][0 if sign > 0 else 1]
        distance_cells = sign * (front_i - span["i_feed"])
        lo = msl_source_near_field_standoff_cells(entry.height, cv.DX)
        hi = distance_cells - int(np.ceil(clearance / cv.DX)) - span_cells
        if hi < lo:
            raise ValueError("no compliant observation interval; extend the feed line")
        clean_offsets.append((lo + hi) // 2)
        if p == 0:
            # Stop one actual grid interval BEFORE the realized junction.
            near_offset = distance_cells - span_cells - 1
    arms = {}
    certificates = {}
    for label in ("clean", "near"):
        sim = copy.deepcopy(base)
        sim._msl_ports = [
            replace(entry, n_probes=N_PROBES, n_probe_spacing=SPACING,
                    n_probe_offset=(near_offset if label == "near" and p == 0
                                    else clean_offsets[p]))
            for p, entry in enumerate(base._msl_ports)
        ]
        # Preserve these explicit ladders through the production resolver.
        sim._msl_auto_offset_min = {}
        sim._msl_auto_probe_spacing = {}
        certificates[label] = _certificate(sim, rz, stub_bounds)
        arms[label] = sim
    if certificates["clean"][1] != certificates["near"][1]:
        raise ValueError("the passive-port observation ladder changed")
    for p in range(2):
        if certificates["clean"][p]["source"] != certificates["near"][p]["source"]:
            raise ValueError("source or termination changed between arms")
    if certificates["clean"][0]["deepest_gap_m"] < clearance:
        raise ValueError("control ladder fails the downstream layout recommendation")
    if certificates["near"][0]["deepest_gap_m"] >= clearance:
        raise ValueError("near ladder does not exercise the downstream warning")
    receipt = dict(
        realized_metal=metal, stub_bounds_m=stub_bounds, ports=certificates,
        case_sha256=hashlib.sha256(CV06B.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison="fixed DUT/source; only p1 observation offset changes",
        exact_full_wave_truth=False,
    )
    return cv, arms, receipt


def _readability(result):
    settling = np.asarray(result.settling_db, dtype=float)
    if settling.shape != (2,) or not np.all(np.isfinite(settling)):
        return "settling witness missing or nonfinite"
    if np.any(settling > -40.0):
        return "settling witness exceeds -40 dB"
    if not np.all(np.isfinite(result.S)):
        return "nonfinite raw S"
    return None


def measure(sim, label, out, *, freqs, periods):
    result = sim.compute_msl_s_matrix(
        freqs=freqs, num_periods=periods, enforce_passivity=False, report_every=5000,
        raw_3probe_dump_path=str(out / f"{label}-phasors.npz"),
    )
    arrays = dict(freqs=result.freqs, S=result.S, Z0=result.Z0,
                  settling_db=result.settling_db, reliable=result.reliable)
    for name in ("beta", "beta_railed", "cond_a", "passivity_correction"):
        value = getattr(result, name, None)
        arrays[name] = np.asarray([] if value is None else value)
    np.savez(out / f"{label}-result.npz", **arrays)
    (out / f"{label}-quality.json").write_text(json.dumps(dict(
        assembly=getattr(result, "assembly", None),
        readability_failure=_readability(result),
        passivity_projection=False,
    ), indent=2) + "\n")
    return result


def compare(clean, near):
    for label, result in (("clean", clean), ("near", near)):
        reason = _readability(result)
        if reason:
            return dict(status="not_read", reason=f"{label}: {reason}")
    f = np.asarray(clean.freqs)
    if not np.array_equal(f, near.freqs):
        raise ValueError("paired runs have different frequency bins")
    # Predeclared 3--5 GHz band brackets cv06b's ~3.7 GHz notch. Select ONE
    # common bin from the control's S21 minimum, never different arm peaks.
    band_indices = np.flatnonzero((f >= 3e9) & (f <= 5e9))
    if not len(band_indices):
        raise ValueError("frequency grid does not cover the notch band")
    k = int(band_indices[np.argmin(abs(clean.S[1, 0, band_indices]))])
    if k in (band_indices[0], band_indices[-1]):
        return dict(status="not_read", reason="control notch is at the band edge")
    for label, result in (("clean", clean), ("near", near)):
        reliable = np.asarray(result.reliable)
        if reliable.shape != (2, len(f)) or not np.all(reliable[:, k]):
            return dict(status="not_read", reason=f"{label}: low-signal screen at notch")
    db = [float(20 * np.log10(max(abs(r.S[0, 0, k]), 1e-300)))
          for r in (clean, near)]
    return dict(status="paired_observation", notch_bin_hz=float(f[k]),
                clean_s11_db=db[0], near_s11_db=db[1], delta_s11_db=db[1] - db[0],
                interpretation="observation-offset sensitivity; no universal bias or exact-truth claim")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-periods", type=float, default=100.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    if not np.isfinite(args.num_periods) or args.num_periods <= 0:
        ap.error("--num-periods must be finite and positive")
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"evidence directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "inputs.json"
    cv, arms, receipt = prepare_inputs()
    receipt["num_periods"] = args.num_periods
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.build_only:
        return 0
    freqs = np.linspace(cv.F_MAX / 10, cv.F_MAX, 161)
    clean = measure(arms["clean"], "clean", out, freqs=freqs, periods=args.num_periods)
    reason = _readability(clean)
    if reason:
        verdict = dict(status="not_read", reason=f"clean: {reason}; near arm skipped")
    else:
        near = measure(arms["near"], "near", out, freqs=freqs, periods=args.num_periods)
        verdict = compare(clean, near)
    (out / "comparison.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    return 0 if verdict["status"] == "paired_observation" else 2


if __name__ == "__main__":
    raise SystemExit(main())

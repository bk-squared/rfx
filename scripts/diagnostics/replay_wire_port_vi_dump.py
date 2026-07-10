#!/usr/bin/env python3
"""Replay a wire-port raw V/I dump into the production S-matrix convention.

The current wire-port extractor uses midpoint-cell voltage/current samples.
Diagonal entries are referenced to the total port impedance, while off-diagonal
entries use the per-cell impedance implied by the distributed wire feed.  This
script intentionally reimplements that math with NumPy only; it does not call
``extract_s_matrix_wire`` or import production port extractors.

Receive-wave convention (issue #308): at a passive receive port the b-wave is
the orthogonal channel ``(V - Z0c I) / (2 sqrt(Z0c))`` in FDTD-sign variables
(the historical ``(-V - Z0c I)`` channel structurally cancelled the arriving
wave at a matched port; the overall sign is pinned empirically by the DC
falsifier on the canonical thru, S21(DC) -> +1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _complex_to_jsonable(arr: np.ndarray) -> list[Any]:
    arr = np.asarray(arr)
    return [{"re": float(np.real(x)), "im": float(np.imag(x))} for x in arr.reshape(-1)]


def _load_dump(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return {
            "metadata": metadata,
            "freqs": np.asarray(data["freqs_hz"], dtype=np.float64),
            "raw_v": np.asarray(data["raw_voltages_fdt"], dtype=np.complex128),
            "raw_i": np.asarray(data["raw_currents"], dtype=np.complex128),
            "z0": np.asarray(data["port_impedances_ohm"], dtype=np.float64),
            "cell_counts": np.asarray(data["port_cell_counts"], dtype=np.int64),
            "production_s": np.asarray(data["production_smatrix"], dtype=np.complex128),
            "port_names": tuple(str(x) for x in data["port_names"].tolist()),
            "driven": tuple(int(x) for x in data["driven_port_indices"].tolist()),
        }


def replay_wire_port_vi_dump(path: Path, *, atol: float = 1e-9, rtol: float = 1e-6) -> dict[str, Any]:
    dump = _load_dump(path)
    freqs = dump["freqs"]
    raw_v = dump["raw_v"]
    raw_i = dump["raw_i"]
    z0 = dump["z0"]
    cell_counts = dump["cell_counts"]
    production_s = dump["production_s"]
    driven = dump["driven"]
    port_names = dump["port_names"]

    if raw_v.shape != raw_i.shape:
        raise ValueError(f"raw voltage/current shape mismatch: {raw_v.shape} vs {raw_i.shape}")
    if raw_v.ndim != 3:
        raise ValueError(
            "raw_voltages_fdt/raw_currents must have shape "
            f"(n_driven, n_ports, n_freqs); got {raw_v.shape}"
        )
    n_driven, n_ports, n_freqs = raw_v.shape
    if freqs.shape != (n_freqs,):
        raise ValueError(f"freqs shape {freqs.shape} incompatible with raw shape {raw_v.shape}")
    if z0.shape != (n_ports,):
        raise ValueError(f"port_impedances_ohm must have shape ({n_ports},); got {z0.shape}")
    if cell_counts.shape != (n_ports,) or np.any(cell_counts <= 0):
        raise ValueError(
            f"port_cell_counts must contain positive counts with shape ({n_ports},); "
            f"got {cell_counts}"
        )
    if production_s.shape != (n_ports, n_ports, n_freqs):
        raise ValueError(
            "production_smatrix shape must be "
            f"({n_ports}, {n_ports}, {n_freqs}); got {production_s.shape}"
        )
    if len(driven) != n_driven:
        raise ValueError(f"driven_port_indices length {len(driven)} != n_driven {n_driven}")

    replay_s = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    for drive_row, driven_port in enumerate(driven):
        v_drive = raw_v[drive_row, driven_port, :]
        i_drive = raw_i[drive_row, driven_port, :]
        safe_i = np.where(np.abs(i_drive) > 0.0, i_drive, 1e-30 + 0j)
        z_in = -v_drive / safe_i

        for receiver in range(n_ports):
            if receiver == driven_port:
                replay_s[receiver, driven_port, :] = (
                    (z_in - z0[receiver]) / (z_in + z0[receiver])
                )
                continue

            z_cell_receiver = z0[receiver] / float(cell_counts[receiver])
            z_cell_driven = z0[driven_port] / float(cell_counts[driven_port])
            # Passive-receive channel (issue #308), FDTD-sign variables.
            b_receiver = (
                raw_v[drive_row, receiver, :]
                - z_cell_receiver * raw_i[drive_row, receiver, :]
            ) / (2.0 * np.sqrt(z_cell_receiver))
            a_driven = (
                -v_drive + z_cell_driven * i_drive
            ) / (2.0 * np.sqrt(z_cell_driven))
            safe_a = np.where(np.abs(a_driven) > 0.0, a_driven, 1.0 + 0j)
            replay_s[receiver, driven_port, :] = b_receiver / safe_a

    diff = np.abs(replay_s - production_s)
    allowed = atol + rtol * np.maximum(np.abs(replay_s), np.abs(production_s))
    ok = bool(np.all(diff <= allowed))
    return {
        "status": "passed" if ok else "failed",
        "dump": str(path),
        "metadata": dump["metadata"],
        "n_ports": n_ports,
        "n_freqs": n_freqs,
        "freqs_hz": freqs.tolist(),
        "port_names": list(port_names),
        "port_cell_counts": cell_counts.astype(int).tolist(),
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "max_allowed": float(np.max(allowed)) if allowed.size else float(atol),
        "atol": float(atol),
        "rtol": float(rtol),
        "replayed_smatrix_complex_flat": _complex_to_jsonable(replay_s),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args(argv)

    payload = replay_wire_port_vi_dump(args.dump, atol=args.atol, rtol=args.rtol)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"{payload['status'].upper()} wire V/I replay: "
        f"max_abs_diff={payload['max_abs_diff']:.6g}, "
        f"max_allowed={payload['max_allowed']:.6g}, "
        f"ports={payload['n_ports']}, freqs={payload['n_freqs']}"
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay an MSL 3-probe raw dump into an S-matrix.

SUPERSEDED (issue #520 leg 3): this script is a dead evidence lane against
current dumps. It hardcodes the retired 3-probe ``raw_v123`` array and the
schema-v1 ``rfx.msl_3probe_dump`` layout; production dumps have shipped the
N-probe ``raw_v`` array under ``rfx.msl_nprobe_dump`` since schema v2
(issue #80 Fix C), currently v3 (issue #523), so loading a current dump
raises ``KeyError: 'raw_v123 is not a file in the archive'`` immediately.
Even patched to read ``raw_v``, its math would still be wrong: it
re-derives S by the single-ratio rule ``S[j,d] = b_j/a_d`` that issue #507
replaced with the multi-drive solve ``S = B @ inv(A)`` for the production
path, so an "independent" replay would no longer be answering the question
"does this match production" for anything but a ``single_ratio_fallback``
dump. Reimplementing the multi-drive solve independently (to make the
comparison meaningful again) is not a minimal fix; it has not been done.

The current independent check on the production V/I extraction is
``scripts/diagnostics/msl_vi_flux_oracle.py`` (the committed flux-oracle
falsifier, issue #520 leg 1 / #525 / #531). This module is kept for
historical reference and for replaying any dump still carrying the retired
schema v1 layout.

This was the E3 validation-side counterpart to
``Simulation.compute_msl_s_matrix(raw_3probe_dump_path=...)``.  It loads the
real simulation-derived V1/V2/V3/I1 phasors and recomputes the MSL S-matrix
without calling ``compute_msl_s_matrix`` or importing the production MSL
extractor helpers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _complex_to_jsonable(arr: np.ndarray) -> list[Any]:
    arr = np.asarray(arr)
    return [
        {"re": float(np.real(x)), "im": float(np.imag(x))}
        for x in arr.reshape(-1)
    ]


def _solve_3probe_independent(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    i1: np.ndarray,
    *,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent 3-probe recurrence.

    Returns ``alpha, gamma, z0, q`` at probe 1.  This intentionally duplicates
    the math instead of importing ``rfx.sources.msl_port.extract_msl_s_params``.
    """

    v1 = np.asarray(v1, dtype=np.complex128)
    v2 = np.asarray(v2, dtype=np.complex128)
    v3 = np.asarray(v3, dtype=np.complex128)
    i1 = np.asarray(i1, dtype=np.complex128)

    coeff = (v1 + v3) / (v2 + eps)
    disc = coeff * coeff - 4.0 + 0j
    sqrt_disc = np.sqrt(disc)
    q_plus = (coeff + sqrt_disc) / 2.0
    q_minus = (coeff - sqrt_disc) / 2.0

    ratio = v2 / (v1 + eps)
    err_plus = np.abs(q_plus - ratio)
    err_minus = np.abs(q_minus - ratio)
    use_plus = (err_plus < err_minus) | (
        np.isclose(err_plus, err_minus) & (np.abs(q_plus) <= np.abs(q_minus))
    )
    q = np.where(use_plus, q_plus, q_minus)

    denom = q * q - 1.0 + eps
    alpha = (q * v2 - v1) / denom
    gamma = q * (v1 * q - v2) / denom
    z0 = (alpha - gamma) / (i1 + eps)
    return alpha, gamma, z0, q


def replay_msl_3probe_dump(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if "raw_v123" not in data.files:
            raise ValueError(
                f"{path}: this dump uses schema "
                f"{metadata.get('schema')!r} v{metadata.get('schema_version')} "
                "(N-probe 'raw_v', not the retired 3-probe 'raw_v123'). "
                "replay_msl_3probe_dump.py is SUPERSEDED for these dumps — "
                "see the module docstring — and its single-ratio replay "
                "math no longer matches the production multi-drive solve "
                "(issue #507). Use scripts/diagnostics/msl_vi_flux_oracle.py "
                "for an independent check on a current dump."
            )
        freqs = np.asarray(data["freqs_hz"], dtype=np.float64)
        raw_v123 = np.asarray(data["raw_v123"], dtype=np.complex128)
        raw_i1 = np.asarray(data["raw_i1"], dtype=np.complex128)
        production_s = np.asarray(data["production_smatrix"], dtype=np.complex128)
        port_names = tuple(str(x) for x in data["port_names"].tolist())

    if raw_v123.ndim != 4 or raw_v123.shape[2] != 3:
        raise ValueError(
            "raw_v123 must have shape (n_driven, n_ports, 3, n_freqs); "
            f"got {raw_v123.shape}"
        )
    if raw_i1.shape != (raw_v123.shape[0], raw_v123.shape[1], raw_v123.shape[3]):
        raise ValueError(
            f"raw_i1 shape {raw_i1.shape} incompatible with raw_v123 "
            f"{raw_v123.shape}"
        )
    n_driven, n_ports, _, n_freqs = raw_v123.shape
    if n_driven != n_ports:
        raise ValueError("MSL replay currently expects one driven row per port")

    replay_s = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    replay_z0 = np.zeros((n_driven, n_ports, n_freqs), dtype=np.complex128)
    replay_q = np.zeros_like(replay_z0)
    for driven in range(n_driven):
        solved = []
        for port in range(n_ports):
            alpha, gamma, z0, q = _solve_3probe_independent(
                raw_v123[driven, port, 0, :],
                raw_v123[driven, port, 1, :],
                raw_v123[driven, port, 2, :],
                raw_i1[driven, port, :],
            )
            solved.append((alpha, gamma))
            replay_z0[driven, port, :] = z0
            replay_q[driven, port, :] = q

        alpha_d, gamma_d = solved[driven]
        replay_s[driven, driven, :] = gamma_d / (alpha_d + 1e-30)
        for port in range(n_ports):
            if port == driven:
                continue
            alpha_p, _ = solved[port]
            replay_s[port, driven, :] = alpha_p / (alpha_d + 1e-30)

    diff = np.abs(replay_s - production_s)
    return {
        "status": "passed" if bool(np.all(diff <= 1e-9 + 1e-6 * np.abs(production_s))) else "failed",
        "dump": str(path),
        "metadata": metadata,
        "n_ports": n_ports,
        "n_freqs": n_freqs,
        "freqs_hz": freqs.tolist(),
        "port_names": list(port_names),
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "replayed_smatrix_complex_flat": _complex_to_jsonable(replay_s),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args(argv)

    payload = replay_msl_3probe_dump(args.dump)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"{payload['status'].upper()} MSL 3-probe replay: "
        f"max_abs_diff={payload['max_abs_diff']:.6g}, "
        f"ports={payload['n_ports']}, freqs={payload['n_freqs']}"
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

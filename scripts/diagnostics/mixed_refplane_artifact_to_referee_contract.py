#!/usr/bin/env python3
"""Adapt an #498/#517 rfx measurement artifact to the openEMS referee's input contract.

WHY THIS EXISTS
---------------
The two #498/#517 lanes were built to two different names for the same content.
The rfx measurement driver (``mixed_refplane_measurement.py``) writes the
frequencies under ``fixture.freqs_hz`` and the unprojected S under
``s_matrix.S_raw`` (flat ``[re, im]`` pairs plus ``s_matrix.S_raw_shape``); the
referee (``probe_fed_msl_openems_referee.py``) declares its input contract
as top-level ``freqs_hz`` and ``s_raw`` nested ``[2][2][n_freqs]``.  VESSL run
369367257607 refused on exactly that mismatch and classified itself correctly:
"a script bug, NOT a physics finding".

This adapter is a pure re-shaping of an EXISTING artifact.  It computes nothing,
it changes no number, and it never touches the measurement file: it writes a new,
clearly-labelled derived file that records where it came from.  The driver also
emits the contract keys natively from now on, so this adapter is only needed for
artifacts produced before that change (VESSL 369367257597 and earlier).

WHAT IT REFUSES
---------------
* a ``port_families`` order other than (wire-family, MSL) -- the contract fixes
  index 0 = lumped/wire, index 1 = MSL, and silently transposing an S-matrix is
  precisely the class of defect this whole issue is about;
* a ``S_raw_shape`` that is not ``[2, 2, n_freqs]`` with ``n_freqs`` equal to the
  number of frequencies;
* any request to carry the PROJECTED ``S`` -- ``result.S`` is a joint SVD
  projection (~4.3x on this fixture) and only ``S_raw`` is comparator-eligible.
  This adapter never reads a projected field, exactly as the referee never does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_FAMILIES = ("wire", "msl")


def adapt(doc: dict, *, source: str) -> dict:
    """Return the referee-contract view of an rfx measurement document."""
    try:
        freqs = list(doc["fixture"]["freqs_hz"])
        sm = doc["s_matrix"]
        flat = sm["S_raw"]
        shape = list(sm["S_raw_shape"])
        families = tuple(sm["port_families"])
    except KeyError as exc:  # pragma: no cover - contract violation path
        raise SystemExit(f"REFUSED: source artifact lacks {exc!r}; not an #498 measurement document")

    if families != EXPECTED_FAMILIES:
        raise SystemExit(
            f"REFUSED: port_families {families!r} != {EXPECTED_FAMILIES!r}. The referee contract fixes "
            "index 0 = lumped/wire family, index 1 = MSL; transposing an S-matrix silently is the "
            "defect class this issue exists to remove."
        )
    if shape != [2, 2, len(freqs)]:
        raise SystemExit(f"REFUSED: S_raw_shape {shape} != [2, 2, {len(freqs)}]")
    if len(flat) != 4 * len(freqs):
        raise SystemExit(f"REFUSED: S_raw has {len(flat)} entries, expected {4 * len(freqs)}")

    # C order, verified against the run's own published per-bin table:
    # |S00| 0.3814 (bin 0) / 0.4027 (bin 4), |S10| 0.9133 / 0.9057, |S22| 0.0199 / 0.0341.
    n = len(freqs)
    s_raw = [[[list(map(float, flat[(i * 2 + j) * n + k])) for k in range(n)] for j in range(2)] for i in range(2)]

    return {
        "what_this_is": (
            "DERIVED VIEW of an #498/#517 rfx measurement artifact, re-shaped to the openEMS referee's "
            "input contract. No number is computed or altered here; see source_artifact."
        ),
        "source_artifact": source,
        "generated_by": Path(__file__).name,
        "freqs_hz": freqs,
        "s_raw": s_raw,
        "port_names": list(sm.get("port_names", [])),
        "port_families": list(families),
        "comparator_rule": sm.get("comparator_rule"),
        "projected_S_deliberately_omitted": (
            "result.S is a joint SVD projection under enforce_passivity=True; only S_raw is "
            "comparator-eligible, so no projected field is carried."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="path to mixed_refplane_measurement.json")
    ap.add_argument("--output", required=True, help="path of the derived contract view to write")
    args = ap.parse_args(argv)

    src = Path(args.source)
    doc = json.loads(src.read_text())
    view = adapt(doc, source=str(src))
    out = Path(args.output)
    out.write_text(json.dumps(view, indent=2))
    print(f"wrote {out} ({len(view['freqs_hz'])} bins) from {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

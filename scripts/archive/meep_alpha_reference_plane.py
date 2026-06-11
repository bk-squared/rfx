#!/usr/bin/env python3
"""Experiment 7: Meep α reference-plane diagnostic.

Hypothesis (handover v2 candidate 1): the remaining rfx-vs-Meep S21
phase residual (slope=-3.92 mm, intercept=-13°, after subtracting the
physical `β_slab·L` contribution) is a reference-plane mismatch
between rfx's port plane and Meep's `get_eigenmode_coefficients`
monitor plane. If α is monitor-anchored, `∠S21_meep` shifts by
`-β·Δx` when `mon_right` moves by `+Δx`. If α is invariant under
monitor moves, it is referenced to some fixed internal cell and the
4 mm slope residual cannot be absorbed into a monitor-plane alignment.

Input: `microwave-energy/results/rfx_crossval_wr90_meep/
wr90_meep_mon_sweep.json` produced by VESSL run 369367234937 with
`--mon-right-offsets-mm 0.0 5.0 10.0`.

Run:
    python scripts/meep_alpha_reference_plane.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


C0 = 2.998e8
F_CUTOFF_TE10 = C0 / (2.0 * 0.02286)  # WR-90


def _meep_complex(block):
    return np.array([complex(r, i) for r, i in block], dtype=np.complex128)


def _load_sweep():
    path = ("/root/workspace/byungkwan-workspace/research/microwave-energy/"
            "results/rfx_crossval_wr90_meep/wr90_meep_mon_sweep.json")
    if not os.path.exists(path):
        print(f"Sweep JSON not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        return json.load(f)


def main() -> int:
    data = _load_sweep()

    # Pick out the three slab blocks by mon_right_offset.
    offsets_sought = [0.0, 5.0, 10.0]
    blocks = {}
    for k, v in data.items():
        if k == "meta":
            continue
        off = v.get("mon_right_offset_mm")
        if off is None:
            continue
        blocks[off] = v

    if not all(o in blocks for o in offsets_sought):
        print(f"Missing offsets in JSON. Found: {list(blocks.keys())}",
              file=sys.stderr)
        return 2

    freqs_ghz = np.asarray(data["meta"]["freqs_ghz"])
    freqs_hz = freqs_ghz * 1e9
    omega = 2 * np.pi * freqs_hz
    k0 = omega / C0
    beta = np.sqrt(np.maximum(k0 ** 2 - (2 * np.pi * F_CUTOFF_TE10 / C0) ** 2, 0.0))

    # S21 for each offset (slab geometry).
    s21 = {off: _meep_complex(blocks[off]["slab"]["s21"]) for off in offsets_sought}

    # Baseline: offset 0.  Deltas vs offset 0.
    s21_0 = s21[0.0]
    phase_0 = np.unwrap(np.angle(s21_0))

    print("=== Meep α reference-plane diagnostic ===")
    print(f"  WR-90 slab (εr=2.0, 10 mm), resolution r3")
    print(f"  mon_right base = {data['meta']['mon_right_x_mm']} mm;"
          f" offsets swept = {data['meta']['mon_right_offsets_mm']}")
    print()
    print(" f/GHz   β/rad·m⁻¹    |S21|@0   |S21|@5   |S21|@10    "
          "∠S21@0    Δφ@5-0     Δφ@10-0    −β·5mm   −β·10mm")
    print("-" * 125)
    for i, f in enumerate(freqs_ghz):
        d5 = np.angle(s21[5.0][i] * np.conj(s21_0[i]))  # wrapped
        d10 = np.angle(s21[10.0][i] * np.conj(s21_0[i]))
        # Unwrap relative to the neighbouring sample to avoid 2π artefacts.
        predict_mm5 = -beta[i] * 5e-3  # radians
        predict_mm10 = -beta[i] * 10e-3
        print(
            f" {f:5.2f}   {beta[i]:9.2f}   "
            f"{abs(s21_0[i]):.4f}   {abs(s21[5.0][i]):.4f}   "
            f"{abs(s21[10.0][i]):.4f}    "
            f"{np.degrees(phase_0[i]):+8.2f}°  "
            f"{np.degrees(d5):+7.2f}°  "
            f"{np.degrees(d10):+7.2f}°    "
            f"{np.degrees(predict_mm5):+7.2f}°  {np.degrees(predict_mm10):+7.2f}°"
        )

    # Global comparison: for each offset, compare actual Δφ vs -β·Δx prediction.
    def _collapse(s_off, dx_mm):
        dphi = np.angle(s_off * np.conj(s21_0))  # wrapped
        # Compare modulo 2π against -β·Δx
        pred = -beta * (dx_mm * 1e-3)
        resid = np.angle(np.exp(1j * (dphi - pred)))  # wrap residual into [-π, π]
        return dphi, pred, resid

    d5, p5, r5 = _collapse(s21[5.0], 5.0)
    d10, p10, r10 = _collapse(s21[10.0], 10.0)

    rms_if_mon_anchored_5 = float(np.degrees(np.sqrt(np.mean(r5 ** 2))))
    rms_if_mon_anchored_10 = float(np.degrees(np.sqrt(np.mean(r10 ** 2))))
    rms_if_invariant_5 = float(np.degrees(np.sqrt(np.mean(d5 ** 2))))
    rms_if_invariant_10 = float(np.degrees(np.sqrt(np.mean(d10 ** 2))))

    print()
    print("=== Hypothesis test ===")
    print(f"  Monitor-anchored (Δφ = −β·Δx):")
    print(f"    RMS|Δφ_5  − (−β·5mm)|  = {rms_if_mon_anchored_5:6.2f}°")
    print(f"    RMS|Δφ_10 − (−β·10mm)| = {rms_if_mon_anchored_10:6.2f}°")
    print(f"  Invariant (Δφ = 0):")
    print(f"    RMS|Δφ_5|  = {rms_if_invariant_5:6.2f}°")
    print(f"    RMS|Δφ_10| = {rms_if_invariant_10:6.2f}°")
    print()

    anchored_better_5 = rms_if_mon_anchored_5 < rms_if_invariant_5
    anchored_better_10 = rms_if_mon_anchored_10 < rms_if_invariant_10

    if anchored_better_5 and anchored_better_10:
        print("Verdict: α is MONITOR-ANCHORED.")
        print("         ∠S21_meep shifts by −β·Δx when the monitor moves —")
        print("         the 3.92 mm rfx-vs-meep slope residual can be")
        print("         absorbed by aligning rfx's probe plane to Meep's.")
        print("         Next step: measure where Meep's mon_right sits vs")
        print("         rfx's probe/reference plane and de-embed.")
    elif not anchored_better_5 and not anchored_better_10:
        print("Verdict: α is INVARIANT under monitor moves.")
        print("         α is referenced to an internal Meep convention —")
        print("         monitor-plane alignment cannot close the residual.")
        print("         Next step: candidate 3 (extractor mode-projection")
        print("         audit), not a reference-plane alignment.")
    else:
        print("Verdict: MIXED. Hypothesis boundary between monitor-anchored")
        print("         and invariant is not clean; investigate before")
        print("         drawing conclusions.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

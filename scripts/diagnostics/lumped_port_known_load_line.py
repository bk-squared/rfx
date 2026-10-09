#!/usr/bin/env python3
"""Measure the #1162 internal through-line fixture and its pure-R residual.

The old PMC-width/open-stub construction and schema-1 JSON are historical.
This schema-2 run uses the same geometry, bins, reference planes and 40-period
record as tests/unit/ports/test_lumped_port_known_load_line.py. The pure-R
closed form is retained; L=0.214*mu0*dx predicts the cell residual separately.
DFT kernel exp(-j omega t), delayed wave exp(-j beta d).

Run with PYTHONPATH=. and --out pointing to untracked scratch. No record is
silently overwritten. The three-mesh verdict lives in the slow unit test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from tests._interior_tem_line import (
    build as build_line, input_reflection, element_inductance,
    assert_predicted_residual, residuals, assert_solved_ports,
)

FREQS = np.array([1., 2.5, 5., 7.5, 10.]) * 1e9
NUM_PERIODS = 40.


def build(kind, r_over_zc):
    return build_line(kind, ratio=r_over_zc, axial_positions=(.25e-3, 1.5e-3),
                      declared_separation=1.25e-3)[0]


def measure(kind, r_over_zc):
    start = time.monotonic()
    sim, line = build_line(kind, ratio=r_over_zc, axial_positions=(.25e-3, 1.5e-3),
                      declared_separation=1.25e-3)
    result = sim.forward(port_s11_freqs=FREQS, num_periods=NUM_PERIODS, skip_preflight=True)
    assert_solved_ports(result, line, kind)
    measured = np.asarray(result.s_params).reshape(-1)
    pure = input_reflection(line, FREQS, r_over_zc * line.zc)
    predicted = input_reflection(line, FREQS, r_over_zc * line.zc,
                                 element_l=element_inductance(line))
    assert_predicted_residual(measured, pure, predicted)
    return dict(kind=kind, ratio=r_over_zc, line=asdict(line),
                s11=np.stack([measured.real, measured.imag], axis=-1).tolist(),
                residuals=residuals(measured, pure, predicted),
                seconds=time.monotonic() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for ratio in (.5, 1., 2.):
        for kind in ('lumped', 'wire'):
            row = measure(kind, ratio)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(dict(
        schema='rfx.lumped_port_known_load_line', schema_version=2,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        freqs_hz=FREQS.tolist(), num_periods=NUM_PERIODS, records=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()

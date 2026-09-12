"""Exercise unequal physical H weights on the aligned live MSL coupon.

This is one explicit in-plane grading diagnostic, not a promotion of the
general in-plane grading support envelope. The board drawing, voltage
planes, source/load definitions, z profile and measurement band are held
fixed. Only four x cells around each first voltage plane change.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from tests.unit.autodiff import test_msl_sparam_ad as coupon
from tests._msl_fixture_qualification import qualify_result
from rfx.sources.msl_port import msl_h_plane_stencil, msl_port_from_entry


def prepare():
    sim = coupon._build_aligned_e2e_sim()
    # The builder has declared the complete board but has not evolved any
    # fields. Replace its explicit input profile; grid construction uses
    # this profile, not a previously realized material array.
    dx = coupon.E2E_SPACING[0]
    profile = np.array(sim._dx_profile, copy=True)
    assert profile.shape == (280,)
    np.testing.assert_allclose(profile, dx, rtol=0, atol=0)
    for node in (100, 180):
        profile[node - 2:node + 2] = dx * np.array([1.1, .9, 1.1, .9])
    sim._dx_profile = profile
    rz = coupon._assert_trace_sheet_realized(sim)
    grid = rz.grid
    entries = sim._resolve_msl_probe_entries(grid)
    stencils = [msl_h_plane_stencil(grid, msl_port_from_entry(pe), plane)
                for pe, plane in zip(entries, (.005, .009))]
    # The two local changes leave every original E measurement plane at
    # the same physical location; H is at unequal 22.5/27.5 um distances.
    for stencil, plane in zip(stencils, (.005, .009)):
        np.testing.assert_allclose(stencil['voltage_coordinate'], plane, atol=1e-14, rtol=0)
        np.testing.assert_allclose(stencil['weights'], [.55, .45], atol=1e-10, rtol=0)
    return sim, dict(
        scope=__doc__, x_profile_m=profile.tolist(), current_stencils=stencils,
        source_ports=[asdict(pe) for pe in entries],
        grid_shape=list(grid.shape), dt_s=float(grid.dt),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sim, receipt = prepare()
    (args.out / 'inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if args.build_only:
        return
    result = sim.compute_msl_s_matrix(
        freqs=np.linspace(coupon.F_MAX / 10, coupon.F_MAX, 10, dtype=np.float32),
        num_periods=coupon.E2E_PERIODS, enforce_passivity=False,
        raw_3probe_dump_path=str(args.out / 'raw-vi.npz'),
    )
    np.savez_compressed(args.out / 'result.npz', **{
        name: np.asarray(getattr(result, name)) for name in (
            'S', 'freqs', 'Z0', 'beta', 'settling_db', 'reliable', 'cond_a')})
    qualification = qualify_result(result)
    (args.out / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    print(json.dumps(qualification, indent=2), flush=True)
    if qualification['failures']:
        raise SystemExit('existing coupon physical screens failed; raw record retained')


if __name__ == '__main__':
    main()

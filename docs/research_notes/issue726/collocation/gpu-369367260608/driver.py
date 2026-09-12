"""Requalify the unchanged coupon after the authorized current-plane change.

Explicit GPU runs of the existing builder and physical screens. This driver
does not write a golden or alter any tolerance. Separate process invocations
produce named 1x confirmation and 2x refinement records for offline review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from tests.unit.autodiff import test_msl_sparam_ad as coupon
from tests._msl_fixture_qualification import qualify_result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refinement', type=int, choices=(1, 2), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference-record', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == 'gpu'
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    sim = coupon._build_aligned_e2e_sim(refinement=args.refinement)
    rz = coupon._assert_trace_sheet_realized(sim)
    issues = coupon._assert_e2e_preflight(sim)
    # Freeze the measured frequencies, not just a linspace recipe: NumPy
    # and different JAX builds/backends differ by 1-2 f32 ULP in some bins.
    # Reference and refined fields must receive the same actual inputs.
    with np.load(args.reference_record) as reference:
        recorded_freqs = reference['freqs']
    assert recorded_freqs.shape == (10,) and recorded_freqs.dtype == np.float32
    assert np.all(np.isfinite(recorded_freqs)) and np.all(np.diff(recorded_freqs) > 0)
    assert recorded_freqs[0] == coupon.F_MAX / 10 and recorded_freqs[-1] == coupon.F_MAX
    frequencies = jnp.asarray(recorded_freqs)
    np.testing.assert_array_equal(np.asarray(frequencies), recorded_freqs)
    receipt = dict(
        source_sha=sha, refinement=args.refinement,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(coupon.__file__).read_bytes()).hexdigest(),
        jax=jax.__version__, devices=str(jax.devices()), x64=bool(jax.config.x64_enabled),
        grid_shape=list(rz.grid.shape), dt_s=float(rz.grid.dt),
        num_periods=coupon.E2E_PERIODS, preflight=[i.to_dict() for i in issues],
        frequencies_hz=np.asarray(frequencies).tolist(),
        reference_record_sha256=hashlib.sha256(args.reference_record.read_bytes()).hexdigest(),
        drawing=dict(substrate_height_m=coupon.H_SUB, trace_width_m=coupon.W_TRACE,
                     launch_separation_m=coupon.L_LINE, domain_m=coupon.E2E_DOMAIN),
    )
    (args.out / 'inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
    result = sim.compute_msl_s_matrix(
        freqs=frequencies, num_periods=coupon.E2E_PERIODS,
        raw_3probe_dump_path=str(args.out / 'raw-vi.npz'),
    )
    arrays = {name: np.asarray(getattr(result, name)) for name in (
        'S', 'S_raw', 'freqs', 'Z0', 'beta', 'settling_db', 'reliable',
        'cond_a', 'beta_railed', 'passivity_correction')
        if getattr(result, name) is not None}
    np.savez_compressed(args.out / 'result.npz', **arrays)
    qualification = qualify_result(result)
    (args.out / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    print(json.dumps(qualification, indent=2), flush=True)
    if qualification['failures']:
        raise SystemExit('Existing physical qualification failed; evidence retained')


if __name__ == '__main__':
    main()

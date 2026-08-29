"""Revert-proof for the W1 witness (acceptance gate 2 evidence).

Two deliberate defect injections on P-A(r=1.4, reduced), float64 fields,
1000 steps:

(a) WITNESS-side corruption: the dual weight of one transition-node Ex
    plane replaced by the primal width -> the conservation gate must
    break (the witness is not trivially insensitive to its metric).
(b) SOLVER-side corruption (CORE-C2 defect class): the E-update inv_dz
    at one transition node replaced by the primal 1/d[k] -> the CORRECT
    witness must detect non-conservation (the witness fires on the
    defect family this lane guards against).

Expected: both drifts >> the 1e-12 f64 validity threshold, while the
uncorrupted pair conserves at ~1e-16.

Usage:
    PYTHONPATH=. JAX_ENABLE_X64=1 python -m validation.research.multiband_nu.revert_proof
"""

from __future__ import annotations

import json

import numpy as np
import jax
import jax.numpy as jnp

from . import fixtures as fx
from .harness import build_pec_fixture, te10_blob_ex, _build_nu_scan
from .remis_energy import energy_weights, remis_energy


def _f64_carry(setup, ex0):
    carry = dict(setup.carry_init)
    st = carry["fdtd"]
    st = st._replace(**{c: jnp.asarray(getattr(st, c) if c != "ex" else ex0,
                                       jnp.float64)
                        for c in ("ex", "ey", "ez", "hx", "hy", "hz")})
    carry["fdtd"] = st
    return carry


def _drift(grid_run, grid_witness, mats, ex0, weights, n_chunks=10, chunk=100):
    setup = _build_nu_scan(grid_run, mats, chunk, sources=[], probes=[])
    carry = _f64_carry(setup, ex0)
    xs = (jnp.arange(chunk, dtype=jnp.int32), jnp.zeros((chunk, 0), jnp.float32))
    run = jax.jit(lambda c: jax.lax.scan(setup.step_fn, c, xs)[0])
    es = [remis_energy(grid_witness, mats, carry["fdtd"], weights)]
    for _ in range(n_chunks):
        carry = run(carry)
        es.append(remis_energy(grid_witness, mats,
                               jax.device_get(carry["fdtd"]), weights))
    es = np.asarray(es)
    return float(np.abs(es[1:] - es[1]).max() / es[1])


def main():
    assert jax.config.read("jax_enable_x64"), "run with JAX_ENABLE_X64=1"
    prof = fx.pa_profile(1.4, "abrupt", n_fine=12, n_coarse=8)
    grid, mats = build_pec_fixture(prof, (fx.A_X, fx.B_Y), fx.DXY)
    ex0 = te10_blob_ex(grid, z_sigma_cells=5.0)
    k_tr = 12  # first fine->coarse transition node (no z padding)
    w = energy_weights(grid)

    baseline = _drift(grid, grid, mats, ex0, w)

    # (a) corrupted witness weight
    bad_w = dict(w)
    wex = w["ex"].copy()
    dual = 1.0 / float(np.asarray(grid.inv_dz)[k_tr])
    primal = 1.0 / float(np.asarray(grid.inv_dz_h)[k_tr])
    wex[:, :, k_tr] = wex[:, :, k_tr] / dual * primal
    bad_w["ex"] = wex
    drift_a = _drift(grid, grid, mats, ex0, bad_w)

    # (b) corrupted solver metric, correct witness
    inv_dz_bad = np.asarray(grid.inv_dz).copy()
    inv_dz_bad[k_tr] = float(np.asarray(grid.inv_dz_h)[k_tr])
    grid_bad = grid._replace(inv_dz=jnp.asarray(inv_dz_bad))
    drift_b = _drift(grid_bad, grid, mats, ex0, w)

    out = {
        "baseline_drift": baseline,
        "corrupted_witness_drift": drift_a,
        "corrupted_solver_metric_drift": drift_b,
        "threshold": 1e-12,
        "pass": bool(baseline < 1e-12 and drift_a > 1e-6 and drift_b > 1e-6),
    }
    print(json.dumps(out, indent=2))
    assert out["pass"], out
    with open("validation/research/multiband_nu/results/revert_proof.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

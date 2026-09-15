"""Re-measure the numbers group T3's docstrings quote, under #931.

Three fixtures in tests/unit/{nonuniform,runners} carry MEASURED values in
their prose that the lattice ownership contract moves. None of them is an
assertion — the gates in those files are ratios, self-comparisons or
one-sided thresholds — so nothing is red today; the prose is simply
pre-#931 and must not be carried across un-remeasured.

  1. test_nonuniform_pec_scatterer_limit.py — |S11|max for the WR-90
     inductive iris, uniform and graded-dy. Docstring quotes ~0.6-2.1
     (uniform) and ~1.4-1.6 (NU). Both the fins' drawing (snapped to the
     node line) and the waveguide S-matrix lane (which used to fold
     pec_mask cells into sigma=1e10 and now applies the realized PEC
     edges) changed under the contract.
  2. test_nu_wire_port_lane_parity.py — the three-load passive S11 table
     (vacuum / pec_plates / dielectric). The PEC plates realize both of
     their drawn faces now and short their own interior; the module's
     load-independence witness is re-measured against that.
  3. test_run_progress_reporting.py::_msl_thru — Z0 and beta of the
     thru-line fixture whose trace moved from a one-cell PEC Box to a
     sheet on the substrate-top node plane.

Writes one JSON. No fixture file is edited from this script and no number
is hand-copied into a test: the outputs are prose evidence.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "t3_measure.json"
res = {"commit": os.environ.get("T3_COMMIT", "?"), "cases": {}}


def _timed(name, fn):
    t0 = time.time()
    try:
        val = fn()
        err = None
    except Exception as exc:                     # noqa: BLE001
        val, err = None, f"{type(exc).__name__}: {exc}"
    dt = time.time() - t0
    res["cases"][name] = {"value": val, "error": err, "seconds": round(dt, 1)}
    print(f"[{name}] {dt:6.1f}s  {val if err is None else err}", flush=True)


# --- 1. WR-90 iris -------------------------------------------------------
def _iris(nonuniform):
    from tests.unit.nonuniform.test_nonuniform_pec_scatterer_limit import (
        _iris_s11_max)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(_iris_s11_max(nonuniform=nonuniform))


_timed("iris_s11max_uniform", lambda: _iris(False))
_timed("iris_s11max_nonuniform", lambda: _iris(True))


# --- 2. wire-port three-load table --------------------------------------
def _three_loads():
    from tests.unit.nonuniform import test_nu_wire_port_lane_parity as m
    out = {}
    for lane in (False, True):
        for extent, load in ((5e-3, "vacuum"), (5e-3, "pec_plates"),
                             (3e-3, "vacuum"), (3e-3, "dielectric")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = np.asarray(m._s11(m._build(lane, extent=extent,
                                               load=load)))
            key = f"{'nu' if lane else 'uniform'}_extent{extent*1e3:.0f}mm_{load}"
            out[key] = {"s11_0p2GHz": [float(s[0].real), float(s[0].imag)],
                        "abs_max": float(np.abs(s).max()),
                        "n_live": int(m._n_live(extent))}
    return out


_timed("wire_port_three_loads", _three_loads)


# --- 3. thru-line Z0 / beta ---------------------------------------------
def _msl_thru_z0_beta():
    from tests.unit.runners.test_run_progress_reporting import _msl_thru, _MSL_FREQS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _msl_thru().compute_msl_s_matrix(freqs=_MSL_FREQS,
                                             num_periods=6.0)
    z0 = np.atleast_1d(np.asarray(r.Z0))
    beta = np.atleast_1d(np.asarray(r.beta))
    s21 = np.asarray(r.S)[1, 0, :] if np.asarray(r.S).ndim == 3 else None
    return {"freqs_hz": [float(f) for f in np.asarray(_MSL_FREQS)],
            "z0_re": [float(np.real(v)) for v in z0.ravel()],
            "beta_re": [float(np.real(v)) for v in beta.ravel()],
            "abs_s21": None if s21 is None else [float(abs(v)) for v in s21]}


_timed("msl_thru_z0_beta", _msl_thru_z0_beta)


# --- 4. dual-spacing port sigma: oracle-2 table --------------------------
def _dual_spacing_closed_form():
    """The four rows of ``test_nu_port_sigma_dual_spacing``'s oracle-2 table.

    #931 R8 (half-open extent) dropped one realized cell from every
    declaration, so the extents were re-declared to realize the SAME
    counts the table was built on. ``n_live`` is read from the stamped
    array, never derived from the extent.
    """
    from tests.unit.nonuniform import test_nu_port_sigma_dual_spacing as d
    out = {}
    for comp, mult in (("ez", 5), ("ez", 9), ("ex", 3), ("ey", 4)):
        extent = mult * d.D
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s11, n_live = d._s11(comp, extent)
        s11 = np.asarray(s11, dtype=np.float64)
        expect = (1.0 - n_live) / (1.0 + n_live)
        out[f"{comp}_{mult}D"] = {
            "extent_mult_D": mult,
            "n_live": int(n_live),
            "expected": float(expect),
            "re_s11": [float(v) for v in s11.ravel()],
            "worst_rel": float(np.max(np.abs(s11 - expect) / abs(expect))),
        }
    return out


_timed("dual_spacing_closed_form", _dual_spacing_closed_form)

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print("wrote", OUT)

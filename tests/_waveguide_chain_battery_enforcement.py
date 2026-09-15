"""Audited replay metadata for the immutable #931 measurement artifact.

Pins are read from a committed record, never fitted to the values being tested.
The shift-sign falsifier needs no new field solve: reference-plane translation
acts on modal amplitudes after the solve, and is diagonal on the stored S matrix.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from tests import _waveguide_chain_battery_gates as G

FIXTURE = Path(__file__).parent / "fixtures/waveguide_chain_battery/fixture_931_realized_pec_forward2_run369367259427.json"
ENFORCEMENT = FIXTURE.with_name("enforcement_931_realized_pec_forward2_run369367259427.json")


def load_enforced_fixture() -> dict:
    raw = FIXTURE.read_bytes()
    record = json.loads(ENFORCEMENT.read_text())
    assert hashlib.sha256(raw).hexdigest() == record["measurement_sha256"]
    fx = json.loads(raw)
    for check in record["checks"]:
        row = fx
        for key in check["path"]:
            row = row[key]
        row[check["pin_field"]] = check["pin"]
        if check["pin_field"] == "pinned_richardson_gate":
            row["pinned_richardson_pair"] = "mid-fine"
    fx["plane_shift"]["cheap_refute"] = deepcopy(record["cheap_refute"])
    fx["verdicts"] = deepcopy(record["verdicts"])
    return fx


def replay_shift_sign_refute(fx: dict, *, flip_sign: bool = True) -> dict:
    """Apply the actual modal shift with reversed port signs to stored S.

    S_ij = b_i/a_j, so its translation is S_ij * B_i/A_j, where A and B
    are _shift_modal_waves(1, 1). This is the same post-solve operator the
    measurement driver's _FlippedShift mutates; neither DUT nor lane changes.
    flip_sign=False is the control and must reproduce the stored shifted S.
    """
    import jax.numpy as jnp
    from rfx.sources.waveguide_port import _shift_modal_waves

    cases = []
    for key, plane in fx["plane_shift"].items():
        if key == "cheap_refute":
            continue
        cell = next(c for c in fx["cells"] if (c["dut"], c["lane"], c["rung"]) ==
                    (plane["dut"], plane["lane"], plane["rung"]))
        base = G.s_from_json(cell["s_params"])
        beta = jnp.asarray(G.beta_yee_fc(fx["fixture"]["freqs_hz"], plane["fc_port_hz"],
                                       cell["dt_s"], cell["dx_m"]))
        incident, outgoing = [], []
        for distance, sign in zip(plane["shift_m"], (1, -1)):
            a, b = _shift_modal_waves(jnp.ones_like(beta), jnp.ones_like(beta), beta,
                                     distance, -sign if flip_sign else sign)
            incident.append(np.asarray(a))
            outgoing.append(np.asarray(b))
        shifted = np.asarray([[base[i, j] * outgoing[i] / incident[j]
                               for j in range(2)] for i in range(2)])
        rotation = G.plane_shift_rotation(
            base, shifted, fx["fixture"]["freqs_hz"], cell["dt_s"], cell["dx_m"],
            shift_left_m=plane["shift_m"][0], shift_right_m=plane["shift_m"][1])
        cases.append({"dut": plane["dut"], "lane": plane["lane"],
                      "resid_yee_per_entry": {k: v["resid_yee_max"]
                                               for k, v in rotation["rotation_deg"].items()},
                      "entries_measurable": rotation["entries_measurable"],
                      "abs_s_still_invariant": rotation["abs_s_allclose"],
                      "max_abs_diff_from_stored_shift": float(np.max(np.abs(
                          shifted - G.s_from_json(plane["s_params_shifted"]))))})
    residuals = [v for case in cases for v in case["resid_yee_per_entry"].values() if v is not None]
    return {"refute": "stored S translated through _shift_modal_waves with reversed step_sign",
            "recovery": "post-solve modal operator replay; no new field measurement",
            "resid_yee_min_over_entries": min(residuals),
            "resid_yee_max_over_entries": max(residuals),
            "rotation_gate_would_pass": any(
                max(v for v in case["resid_yee_per_entry"].values() if v is not None)
                <= G.ROTATION_TOL_YEE_DEG for case in cases),
            "abs_s_still_invariant": all(case["abs_s_still_invariant"] for case in cases),
            "per_case": cases}

"""WR-90 chain battery, THIRD run — the v1.8 closing artifact, replayed.

Pre-declaration ``docs/design_notes/20260905_v18_close_predeclaration.md``, first
committed at ``10b39787`` (14:20 UTC) and revised at ``04c42a57`` (14:24, report_only
for the zero-derivative leg) and ``f914a7ca`` (14:38, the CPU smoke) before the run
started at 14:39; ``f914a7ca`` is the commit the pod fetched and the version the
artifact stamps. ``ba463005`` (14:41, after the start, before any stage finished)
edited §3 row 2 and §3.1; §6 lists every revision. §6 is the outcome.
Artifact: ``tests/fixtures/waveguide_chain_battery/fixture_v18_close.json``
(schema_version 3, VESSL run 369367258638, gpu-rtx4090, 1350.5 s solve wall at
commit ``f914a7ca``).

What separates this artifact from run 2's is a declaration, not a measurement:
contract criterion 1 (forward identity) and 3(a) (AD-vs-FD) are read under x64
on the ``normalize="flux"`` lane (PI decision 2026-09-05,
``tests/_waveguide_chain_battery_gates.py::X64_DECLARED_LANES``), the float32
reading is stored beside the x64 one on every leg, and the pre-declared
zero-derivative leg is ``report_only``. The forward default stays float32. The
18 cells, every plane-shift rotation and every ladder are bit-identical to run
2's; every float32 gradient is bit-identical to run 2's too, so what the
declaration reads is precision, not run-to-run noise.

This module holds three things:

* the **identity** of the artifact and of its two predecessors, which stay as
  they were measured (run 1 frozen, run 2 float32-primary on every lane);
* the **adjudication** — every one of the 185 stored verdicts asserted at the
  value this run measured, one test per §3 row of the note pinning the number
  the row rests on, and the §4 falsifier replayed at zero cost from the stored
  float32 readings (must give exactly run 2's 9 red);
* the **live layer** (§5.11 of the run-2 note), moved here from run 2's module.

Three things found after the run are recorded in the note's §6 and pinned
below rather than smoothed over: the gate module's ``recompute_verdicts``
lacked the report_only branch the driver applied (added, schema_version ≥ 3
only); the plane-shift stage sourced its base gradient from the x64 primary
against a float32 shifted gradient, so the pin step rebuilt those six entries
from the stored float32 numbers and kept the mixed reading beside them; and the
strict-xfail marks §2 item 3 promised to remove live in run 1's replay against
the frozen run-1 artifact and cannot be removed — they stay.

No gate, tolerance, golden or pin is moved here: the gradient-invariance pin is
0.001 from the same envelope as run 2 and every ladder pin is run 2's.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests import _waveguide_chain_battery_fixture as F
from tests import _waveguide_chain_battery_gates as G
from tests._gate_policy import gate_from_envelope
from tests.oracle import test_waveguide_chain_battery as RUN1
from tests.oracle.test_waveguide_chain_battery import (
    LIVE_ABS_S_ENVELOPE,
    LIVE_ABS_S_TOL,
    _measure_cell,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture_v18_close.json"
RUN2 = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture_guide_cell_aperture.json"
FROZEN = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture.json"
PREDECLARATION = "docs/design_notes/20260905_v18_close_predeclaration.md"
PREDECLARATION_SHA = "f914a7ca"   # the note version the pod fetched (== provenance.commit); see note section 6
RUN_ID = "369367258638"
COMMIT = "f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7"
RUN2_ID = "369367258205"

# The pre-declared zero-derivative leg (|S11|² of a PEC short under eps_override):
# report_only on the declared lane, closing note §2.
ZERO_DERIVATIVE_LEG = ("pec_short", "flux", "eps", "s11_mag2")

# Run 2's red set: the section-4 falsifier must reproduce it from the float32 readings.
RUN2_RED = {
    "ad_vs_fd|pec_short|flux|eps|s11_mag2",
    "forward_identity|pec_short|flux|eps|im_s11",
    "forward_identity|pec_short|flux|eps|re_s11",
    "forward_identity|pec_short|flux|eps|s11_mag2",
    "forward_identity|pec_short|flux|sigma|s11_mag2",
    "forward_identity|slab|flux|eps|im_s21",
    "forward_identity|slab|flux|eps|re_s21",
    "forward_identity|slab|flux|eps|s11_mag2",
    "forward_identity|slab|flux|eps|s21_mag2",
}

# Every stored verdict at the value this run measured. 134 pass / 51 report_only.
# Against run 2's committed (pinned) dict exactly 9 keys differ: the 8 flux
# forward-identity legs (fail → pass) and the zero-derivative AD leg
# (fail → report_only). No key moves between pass and report_only.
ADJUDICATED_VERDICTS: dict[str, str] = {
    "ad_vs_fd|pec_short|false|eps|im_s11": "pass",
    "ad_vs_fd|pec_short|false|eps|re_s11": "pass",
    "ad_vs_fd|pec_short|false|eps|s11_mag2": "pass",
    "ad_vs_fd|pec_short|false|sigma|s11_mag2": "pass",
    "ad_vs_fd|pec_short|flux|eps|im_s11": "pass",
    "ad_vs_fd|pec_short|flux|eps|re_s11": "pass",
    "ad_vs_fd|pec_short|flux|eps|s11_mag2": "report_only",
    "ad_vs_fd|pec_short|flux|sigma|s11_mag2": "pass",
    "ad_vs_fd|slab|false|eps|im_s21": "pass",
    "ad_vs_fd|slab|false|eps|re_s21": "pass",
    "ad_vs_fd|slab|false|eps|s11_mag2": "pass",
    "ad_vs_fd|slab|false|eps|s21_mag2": "pass",
    "ad_vs_fd|slab|flux|eps|im_s21": "pass",
    "ad_vs_fd|slab|flux|eps|re_s21": "pass",
    "ad_vs_fd|slab|flux|eps|s11_mag2": "pass",
    "ad_vs_fd|slab|flux|eps|s21_mag2": "pass",
    "cheap_refute_flip_shift_sign": "pass",
    "column_power|pec_short|coarse|false": "report_only",
    "column_power|pec_short|coarse|flux": "report_only",
    "column_power|pec_short|fine|false": "pass",
    "column_power|pec_short|fine|flux": "pass",
    "column_power|pec_short|mid|false": "report_only",
    "column_power|pec_short|mid|flux": "report_only",
    "column_power|slab|coarse|false": "report_only",
    "column_power|slab|coarse|flux": "report_only",
    "column_power|slab|fine|false": "pass",
    "column_power|slab|fine|flux": "pass",
    "column_power|slab|mid|false": "report_only",
    "column_power|slab|mid|flux": "report_only",
    "forward_identity|pec_short|false|eps|im_s11": "pass",
    "forward_identity|pec_short|false|eps|re_s11": "pass",
    "forward_identity|pec_short|false|eps|s11_mag2": "pass",
    "forward_identity|pec_short|false|sigma|s11_mag2": "pass",
    "forward_identity|pec_short|flux|eps|im_s11": "pass",
    "forward_identity|pec_short|flux|eps|re_s11": "pass",
    "forward_identity|pec_short|flux|eps|s11_mag2": "pass",
    "forward_identity|pec_short|flux|sigma|s11_mag2": "pass",
    "forward_identity|slab|false|eps|im_s21": "pass",
    "forward_identity|slab|false|eps|re_s21": "pass",
    "forward_identity|slab|false|eps|s11_mag2": "pass",
    "forward_identity|slab|false|eps|s21_mag2": "pass",
    "forward_identity|slab|flux|eps|im_s21": "pass",
    "forward_identity|slab|flux|eps|re_s21": "pass",
    "forward_identity|slab|flux|eps|s11_mag2": "pass",
    "forward_identity|slab|flux|eps|s21_mag2": "pass",
    "gradient_invariance|pec_short|false|eps:s11_complex": "pass",
    "gradient_invariance|pec_short|false|eps:s11_mag2": "report_only",
    "gradient_invariance|pec_short|false|sigma:s11_mag2": "pass",
    "gradient_invariance|pec_short|flux|eps:s11_complex": "pass",
    "gradient_invariance|pec_short|flux|eps:s11_mag2": "report_only",
    "gradient_invariance|pec_short|flux|sigma:s11_mag2": "pass",
    "gradient_invariance|slab|false|eps:s11_mag2": "pass",
    "gradient_invariance|slab|false|eps:s21_complex": "pass",
    "gradient_invariance|slab|false|eps:s21_mag2": "pass",
    "gradient_invariance|slab|flux|eps:s11_mag2": "pass",
    "gradient_invariance|slab|flux|eps:s21_complex": "pass",
    "gradient_invariance|slab|flux|eps:s21_mag2": "pass",
    "ladder_monotone|pec_short_s11_mag|false": "pass",
    "ladder_monotone|pec_short_s11_mag|flux": "pass",
    "ladder_monotone|pec_short_s11_phase_deg|false": "pass",
    "ladder_monotone|pec_short_s11_phase_deg|flux": "pass",
    "ladder_monotone|slab_s11_mag|false": "pass",
    "ladder_monotone|slab_s11_mag|flux": "pass",
    "ladder_monotone|slab_s21_mag|false": "pass",
    "ladder_monotone|slab_s21_mag|flux": "pass",
    "ladder_monotone|slab_s21_phase_deg|false": "pass",
    "ladder_monotone|slab_s21_phase_deg|flux": "pass",
    "ladder_richardson|pec_short_s11_phase_deg|false": "pass",
    "ladder_richardson|pec_short_s11_phase_deg|flux": "pass",
    "ladder_richardson|slab_s11_mag|false": "pass",
    "ladder_richardson|slab_s11_mag|flux": "pass",
    "ladder_richardson|slab_s21_mag|false": "pass",
    "ladder_richardson|slab_s21_mag|flux": "pass",
    "ladder_richardson|slab_s21_phase_deg|false": "pass",
    "ladder_richardson|slab_s21_phase_deg|flux": "pass",
    "ladder|pec_short_s11_mag|false": "pass",
    "ladder|pec_short_s11_mag|flux": "pass",
    "ladder|pec_short_s11_phase_deg|false": "pass",
    "ladder|pec_short_s11_phase_deg|flux": "pass",
    "ladder|slab_s11_mag|false": "pass",
    "ladder|slab_s11_mag|flux": "pass",
    "ladder|slab_s21_mag|false": "pass",
    "ladder|slab_s21_mag|flux": "pass",
    "ladder|slab_s21_phase_deg|false": "pass",
    "ladder|slab_s21_phase_deg|flux": "pass",
    "non_vacuity|pec_short|coarse|false": "pass",
    "non_vacuity|pec_short|coarse|flux": "pass",
    "non_vacuity|pec_short|fine|false": "pass",
    "non_vacuity|pec_short|fine|flux": "pass",
    "non_vacuity|pec_short|mid|false": "pass",
    "non_vacuity|pec_short|mid|flux": "pass",
    "non_vacuity|slab|coarse|false": "pass",
    "non_vacuity|slab|coarse|flux": "pass",
    "non_vacuity|slab|fine|false": "pass",
    "non_vacuity|slab|fine|flux": "pass",
    "non_vacuity|slab|mid|false": "pass",
    "non_vacuity|slab|mid|flux": "pass",
    "plane_shift_abs_s|pec_short|false": "pass",
    "plane_shift_abs_s|pec_short|flux": "pass",
    "plane_shift_abs_s|slab|false": "pass",
    "plane_shift_abs_s|slab|flux": "pass",
    "plane_shift_rotation_continuous|pec_short|false": "pass",
    "plane_shift_rotation_continuous|pec_short|flux": "pass",
    "plane_shift_rotation_continuous|slab|false": "pass",
    "plane_shift_rotation_continuous|slab|flux": "pass",
    "plane_shift_rotation_yee|pec_short|false": "pass",
    "plane_shift_rotation_yee|pec_short|flux": "pass",
    "plane_shift_rotation_yee|slab|false": "pass",
    "plane_shift_rotation_yee|slab|flux": "pass",
    "plane_shift_wrong_sign|pec_short|false": "pass",
    "plane_shift_wrong_sign|pec_short|flux": "pass",
    "plane_shift_wrong_sign|slab|false": "pass",
    "plane_shift_wrong_sign|slab|flux": "pass",
    "power_closure|pec_short|coarse|false": "report_only",
    "power_closure|pec_short|coarse|flux": "report_only",
    "power_closure|pec_short|fine|false": "report_only",
    "power_closure|pec_short|fine|flux": "report_only",
    "power_closure|pec_short|mid|false": "report_only",
    "power_closure|pec_short|mid|flux": "report_only",
    "power_closure|slab|coarse|false": "report_only",
    "power_closure|slab|coarse|flux": "report_only",
    "power_closure|slab|fine|false": "report_only",
    "power_closure|slab|fine|flux": "report_only",
    "power_closure|slab|mid|false": "report_only",
    "power_closure|slab|mid|flux": "report_only",
    "reciprocity_complex|pec_short|coarse|false": "report_only",
    "reciprocity_complex|pec_short|coarse|flux": "report_only",
    "reciprocity_complex|pec_short|fine|false": "pass",
    "reciprocity_complex|pec_short|fine|flux": "pass",
    "reciprocity_complex|pec_short|mid|false": "report_only",
    "reciprocity_complex|pec_short|mid|flux": "report_only",
    "reciprocity_complex|slab|coarse|false": "report_only",
    "reciprocity_complex|slab|coarse|flux": "report_only",
    "reciprocity_complex|slab|fine|false": "pass",
    "reciprocity_complex|slab|fine|flux": "pass",
    "reciprocity_complex|slab|mid|false": "report_only",
    "reciprocity_complex|slab|mid|flux": "report_only",
    "reciprocity_mag|pec_short|coarse|false": "report_only",
    "reciprocity_mag|pec_short|coarse|flux": "report_only",
    "reciprocity_mag|pec_short|fine|false": "pass",
    "reciprocity_mag|pec_short|fine|flux": "pass",
    "reciprocity_mag|pec_short|mid|false": "report_only",
    "reciprocity_mag|pec_short|mid|flux": "report_only",
    "reciprocity_mag|slab|coarse|false": "report_only",
    "reciprocity_mag|slab|coarse|flux": "report_only",
    "reciprocity_mag|slab|fine|false": "pass",
    "reciprocity_mag|slab|fine|flux": "pass",
    "reciprocity_mag|slab|mid|false": "report_only",
    "reciprocity_mag|slab|mid|flux": "report_only",
    "referee_pec_short|pec_short|coarse|false": "report_only",
    "referee_pec_short|pec_short|coarse|flux": "report_only",
    "referee_pec_short|pec_short|fine|false": "pass",
    "referee_pec_short|pec_short|fine|flux": "pass",
    "referee_pec_short|pec_short|mid|false": "report_only",
    "referee_pec_short|pec_short|mid|flux": "report_only",
    "referee_slab_airy_mag|slab|coarse|false": "report_only",
    "referee_slab_airy_mag|slab|coarse|flux": "report_only",
    "referee_slab_airy_mag|slab|fine|false": "pass",
    "referee_slab_airy_mag|slab|fine|flux": "pass",
    "referee_slab_airy_mag|slab|mid|false": "report_only",
    "referee_slab_airy_mag|slab|mid|flux": "report_only",
    "referee_slab_airy_phase|slab|coarse|false": "report_only",
    "referee_slab_airy_phase|slab|coarse|flux": "report_only",
    "referee_slab_airy_phase|slab|fine|false": "pass",
    "referee_slab_airy_phase|slab|fine|flux": "pass",
    "referee_slab_airy_phase|slab|mid|false": "report_only",
    "referee_slab_airy_phase|slab|mid|flux": "report_only",
    "settling|pec_short|coarse|false": "pass",
    "settling|pec_short|coarse|flux": "pass",
    "settling|pec_short|fine|false": "pass",
    "settling|pec_short|fine|flux": "pass",
    "settling|pec_short|mid|false": "pass",
    "settling|pec_short|mid|flux": "pass",
    "settling|slab|coarse|false": "pass",
    "settling|slab|coarse|flux": "pass",
    "settling|slab|fine|false": "pass",
    "settling|slab|fine|flux": "pass",
    "settling|slab|mid|false": "pass",
    "settling|slab|mid|flux": "pass",
    "settling|thru|coarse|false": "pass",
    "settling|thru|coarse|flux": "pass",
    "settling|thru|fine|false": "pass",
    "settling|thru|fine|flux": "pass",
    "settling|thru|mid|false": "pass",
    "settling|thru|mid|flux": "pass",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


_FX = _load(FIXTURE)
_FX2 = _load(RUN2)


@pytest.fixture(scope="module")
def fx() -> dict:
    if _FX is None:
        pytest.skip(f"{FIXTURE} missing")
    return _FX


@pytest.fixture(scope="module")
def fx2() -> dict:
    if _FX2 is None:
        pytest.skip(f"{RUN2} missing")
    return _FX2


def _cell(fx_: dict, dut: str, rung: str, lane: str) -> dict:
    return next(c for c in fx_["cells"] if (c["dut"], c["rung"], c["lane"]) == (dut, rung, lane))


def _leg(fx_: dict, dut: str, lane: str, kind: str, obj: str) -> dict:
    return next(entry for entry in fx_["ad_vs_fd"]
                if (entry["dut"], entry["lane"], entry["theta_kind"], entry["objective"]) == (dut, lane, kind, obj))


def _leg_key(entry: dict) -> tuple:
    return (entry["dut"], entry["lane"], entry["theta_kind"], entry["objective"])


# ===========================================================================
# IDENTITY — this artifact, and the two it does not touch
# ===========================================================================

def test_identity_stamp(fx):
    assert fx["schema"] == "rfx.waveguide_chain_battery"
    assert fx["schema_version"] == 3
    assert fx["predeclaration"] == PREDECLARATION
    assert fx["predeclaration_sha"] == PREDECLARATION_SHA
    assert (REPO / PREDECLARATION).exists()
    assert fx["shift_pair_name"] == "sign_discriminating_pair"
    assert fx["supersedes"] == "tests/fixtures/waveguide_chain_battery/fixture_guide_cell_aperture.json"
    assert "x64" in fx["supersedes_reason"] and "report_only" in fx["supersedes_reason"]
    p = fx["provenance"]
    assert p["run_id"] == RUN_ID
    assert p["run_lane"] == "vessl"
    assert p["commit"] == COMMIT
    assert p["precision"] == "float32" and p["jax_enable_x64"] is False
    assert p["jax_devices"] == ["cuda:0"]
    assert p["wall_time_s"] == pytest.approx(1350.4885, abs=1e-3)
    assert p["recapture_entry_point"] == "scripts/diagnostics/waveguide_chain_battery_measure.py"
    assert p["recapture_vessl_yaml"] == "scripts/vessl_waveguide_chain_battery_v18_close.yaml"
    assert (REPO / p["recapture_vessl_yaml"]).exists() and (REPO / p["recapture_entry_point"]).exists()
    # Strings written after the pod's assemble step are named, and are strings only.
    assert "run_id" in p["post_run_edits"] and "supersedes" in p["post_run_edits"]
    assert p["recapture_command"].endswith("--fixture-out tests/fixtures/waveguide_chain_battery/fixture_v18_close.json")
    # every provenance.run_id anywhere in the artifact carries the run id (the cheap-refute
    # per-case records had kept the pod's placeholder; corrected post-run and listed)
    def _run_ids(node):
        if isinstance(node, dict):
            if "run_id" in node and isinstance(node["run_id"], str):
                yield node["run_id"]
            for v in node.values():
                yield from _run_ids(v)
        elif isinstance(node, list):
            for v in node:
                yield from _run_ids(v)
    assert set(_run_ids(fx)) == {RUN_ID}
    assert RUN_ID in p["run_id_note"]
    assert fx["legs_rung"] == "fine"


def test_the_two_predecessors_are_what_they_were(fx2):
    frozen = _load(FROZEN)
    assert frozen is not None and frozen["schema_version"] == 1
    assert fx2["schema_version"] == 2
    assert fx2["provenance"]["run_id"] == RUN2_ID
    # run 2 is float32-primary on every lane and carries none of the schema-3 keys
    for entry in fx2["ad_vs_fd"]:
        assert "primary_precision" not in entry and "ad_vs_fd_float32" not in entry
    assert "pins" in fx2


def test_the_declaration_is_the_committed_one(fx):
    assert G.X64_DECLARED_LANES == {"flux"}
    assert G.ZERO_DERIVATIVE_RATIO_MAX == 3.0
    assert G.EXPECTED_ULP_SKIP == {("pec_short", "eps", "s11_mag2")}
    for entry in fx["ad_vs_fd"]:
        if entry["lane"] == "flux":
            assert entry["primary_precision"] == "x64", _leg_key(entry)
            assert entry["x64_witness"] is not None, _leg_key(entry)
            assert entry["ad_vs_fd_float32"]["g_ad"] is not None
            assert entry["forward_identity_float32"]["max_scaled_diff"] is not None
            # the primary IS the witness, not a third computation
            assert entry["g_ad"] == entry["x64_witness"]["g_ad_x64"]
            assert entry["forward_identity"] == entry["x64_witness"]["forward_identity_x64"]
        else:
            assert entry["primary_precision"] == "float32", _leg_key(entry)
            assert entry["g_ad"] == entry["ad_vs_fd_float32"]["g_ad"]
            assert entry["forward_identity"] == entry["forward_identity_float32"]


# ===========================================================================
# WITNESS — the measurement did not move; only the reading did
# ===========================================================================

def test_all_18_cells_are_bit_identical_to_run_2(fx, fx2):
    """§3 row "everything else": the cells stage ran the same code on the same
    GPU type and reproduced run 2 to the bit — S-matrices, settling per drive,
    column power. The x64 context of the AD stage leaked into no forward path."""
    for c2 in fx2["cells"]:
        c3 = _cell(fx, c2["dut"], c2["rung"], c2["lane"])
        S2, S3 = G.s_from_json(c2["s_params"]), G.s_from_json(c3["s_params"])
        assert float(np.max(np.abs(S2 - S3))) == 0.0, (c2["dut"], c2["rung"], c2["lane"])
        assert c3["settling_db"] == c2["settling_db"]
        assert c3["column_power_max"] == c2["column_power_max"]
        assert c3["preflight"] == c2["preflight"]


def test_rotations_and_ladders_are_identical_to_run_2(fx, fx2):
    for key, p2 in fx2["plane_shift"].items():
        if key == "cheap_refute":
            continue
        p3 = fx["plane_shift"][key]
        for k in ("resid_yee_max", "resid_cont_max", "resid_port_beta_max", "wrong_sign_resid_min",
                  "abs_s_max_diff"):
            assert p3[k] == p2[k], (key, k)
    for key, l2 in fx2["ladder"].items():
        l3 = fx["ladder"][key]
        assert l3["verdict"] == l2["verdict"] == "pass", key
        assert l3["monotone_fraction_of_bins"] == l2["monotone_fraction_of_bins"]
        if "richardson" in l2:
            assert l3["richardson"]["mid-fine"]["max_abs_diff"] == l2["richardson"]["mid-fine"]["max_abs_diff"]
        assert l3.get("pinned_richardson_gate") == l2.get("pinned_richardson_gate")
        assert l3.get("pinned_monotone_fraction_min") == l2.get("pinned_monotone_fraction_min")


def test_every_float32_gradient_is_bit_identical_to_run_2(fx, fx2):
    """Free witness: the reverse-mode pass on this GPU is reproducible run to
    run in float32 AND in x64, so the difference the declaration reads between
    the two readings is precision, not noise."""
    for e2 in fx2["ad_vs_fd"]:
        e3 = _leg(fx, *_leg_key(e2))
        assert e3["ad_vs_fd_float32"]["g_ad"] == e2["g_ad"], _leg_key(e2)
        assert e3["forward_identity_float32"]["max_abs_diff"] == e2["forward_identity"]["max_abs_diff"]
        assert e3["f_plus"] == e2["f_plus"] and e3["f_minus"] == e2["f_minus"]
        if e2.get("x64_witness") is not None:
            assert e3["x64_witness"]["g_ad_x64"] == e2["x64_witness"]["g_ad_x64"], _leg_key(e2)


# ===========================================================================
# ADJUDICATION — every verdict at the value this run measured
# ===========================================================================

@pytest.mark.parametrize("key", sorted(ADJUDICATED_VERDICTS))
def test_adjudicated_verdict(fx, key):
    assert fx["verdicts"][key] == ADJUDICATED_VERDICTS[key], key


def test_stored_verdicts_equal_recomputed(fx):
    """The dict the artifact stores is what the shared gate module recomputes
    from the stored numbers — including the two branches added after the run
    (report_only on the declared zero-derivative leg; not_interpretable for a
    declared lane read at float32)."""
    recomputed = G.recompute_verdicts(fx)
    assert recomputed == fx["verdicts"]
    assert set(recomputed) == set(ADJUDICATED_VERDICTS)


def test_verdict_census(fx):
    """Closing note §3: 184 pass / report_only + 1 report_only (the
    zero-derivative leg), 0 fail, 0 not_interpretable."""
    census = {}
    for v in fx["verdicts"].values():
        census[v] = census.get(v, 0) + 1
    assert census == {"pass": 134, "report_only": 51}
    assert "fail" not in census and "not_interpretable" not in census
    # the +1 report_only against run 2's 50 is the zero-derivative leg and nothing else
    assert fx["verdicts"]["ad_vs_fd|pec_short|flux|eps|s11_mag2"] == "report_only"
    assert sum(v == "report_only" for k, v in fx["verdicts"].items() if k.startswith("ad_vs_fd|")) == 1


def test_a_declared_lane_read_at_float32_is_not_interpretable(fx):
    """The declaration cannot be satisfied vacuously: strip the x64 reading
    from one flux leg and the recompute reads it as not_interpretable, on
    both the AD-vs-FD and the forward-identity keys."""
    fx_c = json.loads(json.dumps(fx))
    entry = _leg(fx_c, "slab", "flux", "eps", "s21_mag2")
    entry["primary_precision"] = "float32"
    v = G.recompute_verdicts(fx_c)
    assert v["ad_vs_fd|slab|flux|eps|s21_mag2"] == "not_interpretable"
    assert v["forward_identity|slab|flux|eps|s21_mag2"] == "not_interpretable"
    # and a schema_version 2 artifact is not re-read under the declaration
    fx_c["schema_version"] = 2
    v2 = G.recompute_verdicts(fx_c)
    assert v2["ad_vs_fd|slab|flux|eps|s21_mag2"] == "pass"


# ---------------------------------------------------------------------------
# §3 row 1 — forward identity on the flux lane, under x64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group,scaled_x64,abs_x64,scaled_f32", [
    ("pec_short|flux|eps", 1.7380284854671055e-10, 1.7554167342883505e-15, 1.083215),
    ("pec_short|flux|sigma", 2.8349255023183324e-10, 2.3135566778103808e-15, 1.514875),
    ("slab|flux|eps", 1.0510831029970571e-08, 2.2353207304865925e-14, 1.760720),
])
def test_flux_forward_identity_under_x64(fx, group, scaled_x64, abs_x64, scaled_f32):
    """Predicted ≤ 1e-7 scaled on every flux leg (three measured on run 2 as
    witnesses: 1.7e-10 / 2.8e-10 / 1.05e-8); the fail branch was > 1e-3. Landed
    exactly on the three witnesses, now on all 8 legs. The float32 reading beside
    it is run 2's red to the last printed digit."""
    dut, lane, kind = group.split("|")
    legs = [entry for entry in fx["ad_vs_fd"] if (entry["dut"], entry["lane"], entry["theta_kind"]) == (dut, lane, kind)]
    assert legs
    assert len({entry["forward_identity"]["max_scaled_diff"] for entry in legs}) == 1
    fi = legs[0]["forward_identity"]
    assert fi["max_scaled_diff"] == pytest.approx(scaled_x64, rel=1e-9)
    assert fi["max_abs_diff"] == pytest.approx(abs_x64, rel=1e-9)
    assert fi["max_scaled_diff"] <= 1e-7
    assert G.forward_identity_pass(fi["max_scaled_diff"])
    for entry in legs:
        assert entry["forward_identity_float32"]["max_scaled_diff"] == pytest.approx(scaled_f32, rel=1e-6)
        assert not G.forward_identity_pass(entry["forward_identity_float32"]["max_scaled_diff"])
        conc = entry.get("forward_identity_concrete_override_vs_plain")
        if conc is not None:
            assert conc["max_abs_diff"] == 0.0


def test_flux_forward_identity_worst_over_all_eight_legs(fx):
    worst = max(entry["forward_identity"]["max_scaled_diff"] for entry in fx["ad_vs_fd"] if entry["lane"] == "flux")
    assert worst == pytest.approx(1.0510831029970571e-08, rel=1e-9)
    assert worst <= 1e-7


# ---------------------------------------------------------------------------
# §3 row 2 — normalize=False identity stays exactly zero (not declared)
# ---------------------------------------------------------------------------

def test_false_lane_identity_is_exactly_zero(fx):
    for entry in fx["ad_vs_fd"]:
        if entry["lane"] == "false":
            assert entry["forward_identity"]["max_abs_diff"] == 0.0, _leg_key(entry)
            assert entry["forward_identity"]["max_scaled_diff"] == 0.0


# ---------------------------------------------------------------------------
# §3 row 3 — the zero-derivative leg is report_only, and its ratio is NOT a pass
# ---------------------------------------------------------------------------

def test_zero_derivative_leg_is_report_only_and_its_ratio_fails_the_factor_3_band(fx):
    """Closing note §2: report_only on the remeasure note's exit (c). The
    sign / factor-3 entry is stored beside it and FAILS (5.709 > 3), written so
    the report-only status is not mistaken for a pass; widening the band to make
    it pass was the rejected alternative."""
    leg = _leg(fx, *ZERO_DERIVATIVE_LEG)
    assert leg["verdict"] == "report_only"
    assert leg["expected_ulp_floor_skip"] is True
    assert leg["primary_precision"] == "x64"
    assert "zero-derivative" in leg["report_only_reason"]
    assert leg["g_ad"] == pytest.approx(-2.942583137e-07, rel=1e-8)
    assert leg["g_fd"] == pytest.approx(-5.154156213e-08, rel=1e-8)
    assert leg["fd_ulp_span"] == pytest.approx(4.642451e7, rel=1e-6)
    assert leg["fd_ulp_span"] >= G.FD_ULP_FLOOR, "FD resolved above the floor, which is why the leg is not a skip"
    assert abs(leg["g_ad"]) <= 1e-6 and abs(leg["g_fd"]) <= 1e-6
    zd = leg["zero_derivative"]
    assert zd == G.zero_derivative_entry(g_ad_x64=leg["g_ad"], g_fd=leg["g_fd"], fd_ulp_span=leg["fd_ulp_span"])
    assert zd["same_sign"] is True
    assert zd["ratio"] == pytest.approx(5.709146, rel=1e-6)
    assert zd["ratio_max"] == 3.0
    assert zd["verdict"] == "fail", "outside the factor-3 band; a report, never the verdict"
    assert G.zero_derivative_report_only_admissible(g_ad_x64=leg["g_ad"], g_fd=leg["g_fd"])
    # the float32 reading beside it is run 2's red: wrong sign, 95x larger, rel 541.5
    f32 = leg["ad_vs_fd_float32"]
    assert f32["verdict"] == "fail"
    assert f32["g_ad"] == pytest.approx(2.786023e-05, rel=1e-6)
    assert f32["rel"] == pytest.approx(541.539, rel=1e-5)
    assert math.copysign(1.0, f32["g_ad"]) != math.copysign(1.0, leg["g_fd"])


def test_report_only_is_conditional_on_the_predeclared_branch(fx):
    """§3 row 3's fail branch is enforced by the gate module, not only by this
    module's pins: a future run whose zero-derivative leg flipped sign, or whose
    x64 AD or FD grew above 1e-5, recomputes as fail — the leg cannot ride
    report_only out of the branch that admits it."""
    key = "ad_vs_fd|pec_short|flux|eps|s11_mag2"
    base = json.loads(json.dumps(fx))
    assert G.recompute_verdicts(base)[key] == "report_only"
    flipped = json.loads(json.dumps(fx))
    _leg(flipped, *ZERO_DERIVATIVE_LEG)["g_ad"] = -_leg(flipped, *ZERO_DERIVATIVE_LEG)["g_ad"]
    assert G.recompute_verdicts(flipped)[key] == "fail"
    grown = json.loads(json.dumps(fx))
    _leg(grown, *ZERO_DERIVATIVE_LEG)["g_fd"] = -2e-5
    assert G.recompute_verdicts(grown)[key] == "fail"
    grown_ad = json.loads(json.dumps(fx))
    _leg(grown_ad, *ZERO_DERIVATIVE_LEG)["g_ad"] = -1.1e-5
    assert G.recompute_verdicts(grown_ad)[key] == "fail"
    assert G.ZERO_DERIVATIVE_ABS_MAX == 1e-5


# ---------------------------------------------------------------------------
# §3 row 4 — the other fifteen AD-vs-FD legs
# ---------------------------------------------------------------------------

def test_the_other_fifteen_ad_fd_legs_pass(fx):
    """rel 1.22e-4 … 1.074e-2 against 0.05; FD spans 8.7e13 … 4.8e15 ULP on
    fourteen legs and 3.44e11 on the zero-derivative leg's ``false``-lane
    sibling (its own number, run 2's note; still seven orders above the 1e4
    floor); and the finding branch ("a leg whose rel ROSE under x64") stayed
    empty above the noise floor (>10 % AND >1e-4 absolute)."""
    others = [entry for entry in fx["ad_vs_fd"] if _leg_key(entry) != ZERO_DERIVATIVE_LEG]
    assert len(others) == 15
    rels = sorted(entry["rel"] for entry in others)
    assert all(entry["verdict"] == "pass" for entry in others)
    assert rels[0] == pytest.approx(1.224937e-4, rel=1e-5)
    assert rels[-1] == pytest.approx(1.073658e-2, rel=1e-5)
    sibling = _leg(fx, "pec_short", "false", "eps", "s11_mag2")
    assert sibling["fd_ulp_span"] == pytest.approx(3.442241e11, rel=1e-5)
    assert min(entry["fd_ulp_span"] for entry in others) >= G.FD_ULP_FLOOR
    assert min(entry["fd_ulp_span"] for entry in others if _leg_key(entry) != _leg_key(sibling)) >= 8.7e13
    rose = [_leg_key(entry) for entry in others
            if entry["lane"] == "flux"
            and entry["rel"] > 1.1 * entry["ad_vs_fd_float32"]["rel"]
            and entry["rel"] - entry["ad_vs_fd_float32"]["rel"] > 1e-4]
    assert rose == []


def test_no_non_finite_float32_gradient_at_the_claims_rung(fx):
    """§3.1's watch: four PEC-short flux legs were NaN in float32 at the coarse
    rung on CPU; at the claims rung on the GPU every float32 gradient is finite."""
    for entry in fx["ad_vs_fd"]:
        assert math.isfinite(entry["ad_vs_fd_float32"]["g_ad"]), _leg_key(entry)
        assert math.isfinite(entry["g_ad"]), _leg_key(entry)


# ---------------------------------------------------------------------------
# §4 — the declaration's own falsifier, replayed from the file at zero cost
# ---------------------------------------------------------------------------

def test_section_4_falsifier_float32_primary_reproduces_run_2s_nine_red(fx):
    """Push the stored float32 readings through the run-2 gate (float32
    primary on every lane): exactly run 2's 9 red, no more, no fewer. The pod
    ran the same stage with RFX_CHAIN_PRIMARY=float32 and counted 9 too."""
    red = set()
    for entry in fx["ad_vs_fd"]:
        key = f"{entry['dut']}|{entry['lane']}|{entry['theta_kind']}|{entry['objective']}"
        f32 = entry["ad_vs_fd_float32"]
        e = G.ad_fd_entry(g_ad=f32["g_ad"], f_plus=entry["f_plus"], f_minus=entry["f_minus"],
                          h=entry["h"], loss_dtype=np.dtype(entry["loss_dtype"]))
        if e["verdict"] == "fail":
            red.add(f"ad_vs_fd|{key}")
        if not G.forward_identity_pass(entry["forward_identity_float32"]["max_scaled_diff"]):
            red.add(f"forward_identity|{key}")
    assert red == RUN2_RED
    assert len(red) == 9


def test_section_4_falsifier_as_run_in_the_pod_matches_the_replay(fx):
    """The pod ran the same AD stage with RFX_CHAIN_PRIMARY=float32; its record is
    attached as ``section_4_falsifier`` (independent review of PR #908, finding 7:
    the count alone is coarse — the leg-by-leg identity with the float32 readings
    stored on the primary legs is the evidence)."""
    f = fx["section_4_falsifier"]
    assert f["n_legs"] == 16 and f["n_red"] == 9
    assert set(f["red_keys"]) == RUN2_RED
    assert f["provenance"]["commit"] == COMMIT and f["provenance"]["precision"] == "float32"
    assert f["provenance"]["jax_devices"] == ["cuda:0"]
    for entry in fx["ad_vs_fd"]:
        key = f"{entry['dut']}|{entry['lane']}|{entry['theta_kind']}|{entry['objective']}"
        pod = f["legs"][key]
        f32 = entry["ad_vs_fd_float32"]
        assert pod["primary_precision"] == "float32"
        assert pod["g_ad"] == f32["g_ad"], key
        assert pod["g_fd"] == f32["g_fd"], key
        assert pod["rel"] == f32["rel"], key
        assert pod["verdict"] == f32["verdict"], key
        assert pod["forward_identity_max_scaled_diff"] == entry["forward_identity_float32"]["max_scaled_diff"], key
        assert pod["forward_identity_pass"] == G.forward_identity_pass(pod["forward_identity_max_scaled_diff"])


# ---------------------------------------------------------------------------
# post-run finding 2 — gradient invariance rebuilt at float32 on both sides
# ---------------------------------------------------------------------------

def test_gradient_invariance_is_float32_on_both_sides_and_equals_run_2(fx, fx2):
    """Criterion 3(b) is not under the declaration. The plane-shift stage of
    the closing run read its base gradient from the x64 primary against a
    float32 shifted gradient; the pin step rebuilt the six flux entries from the
    stored float32 numbers. Eleven of twelve rel_change values are bit-identical
    to run 2 and the twelfth is 2.2e-16 apart (a degrees→radians round trip on
    identical inputs)."""
    n = 0
    for key, p2 in fx2["plane_shift"].items():
        if key == "cheap_refute":
            continue
        p3 = fx["plane_shift"][key]
        for obj, g2 in p2["gradient_invariance"].items():
            g3 = p3["gradient_invariance"][obj]
            assert g3["base_precision"] == "float32" and g3["shift_precision"] == "float32", (key, obj)
            assert g3["value_base"] == g2["value_base"], (key, obj)
            assert g3["value_shifted"] == g2["value_shifted"], (key, obj)
            assert g3["rel_change"] == pytest.approx(g2["rel_change"], abs=1e-15), (key, obj)
            assert g3.get("pinned_gate") == g2.get("pinned_gate"), (key, obj)
            n += 1
    assert n == 12
    assert fx["pins"]["gradient_invariance_envelope"] == fx2["pins"]["gradient_invariance_envelope"]
    assert fx["pins"]["gradient_invariance_gate"] == fx2["pins"]["gradient_invariance_gate"] == 0.001
    assert G.rebase_gradient_invariance_float32(json.loads(json.dumps(fx)))["plane_shift"] == fx["plane_shift"], \
        "idempotent: nothing left to rebuild"


@pytest.mark.parametrize("key,obj,mixed", [
    ("pec_short|flux", "sigma:s11_mag2", 5.940562e-07),
    ("pec_short|flux", "eps:s11_mag2", 9.629483e+01),
    ("pec_short|flux", "eps:s11_complex", 3.393655e-06),
    ("slab|flux", "eps:s11_mag2", 4.730868e-06),
    ("slab|flux", "eps:s21_mag2", 2.649283e-06),
    ("slab|flux", "eps:s21_complex", 2.131703e-06),
])
def test_the_mixed_x64_base_reading_is_kept_as_a_report(fx, key, obj, mixed):
    """What the closing run stored, kept under its own name: the float32
    gradient's distance from x64 on the flux lane — 5.9e-7 … 4.7e-6 on the
    in-program legs, 96.3 on the zero-derivative leg, which is
    |+2.804e-5 − (−2.943e-7)| / 2.943e-7 with +2.804e-5 the float32
    shifted-plane gradient (the float32 base-plane one is +2.786e-5, 0.65 %
    away). Not a plane-invariance number."""
    m = fx["plane_shift"][key]["gradient_invariance_x64_base"][obj]
    assert m["base_precision"] == "x64" and m["shift_precision"] == "float32"
    assert m["rel_change"] == pytest.approx(mixed, rel=1e-6)
    if obj == "eps:s11_mag2" and key == "pec_short|flux":
        leg = _leg(fx, *ZERO_DERIVATIVE_LEG)
        assert m["value_base"] == leg["g_ad"], "the mixed base is the x64 primary"
        assert m["value_shifted"] == pytest.approx(2.804129508e-05, rel=1e-8)
        expected = abs(m["value_shifted"] - leg["g_ad"]) / abs(leg["g_ad"])
        assert m["rel_change"] == pytest.approx(expected, rel=1e-9)
    assert "gradient_invariance_x64_base" not in fx["plane_shift"][key.replace("flux", "false")]


# ---------------------------------------------------------------------------
# post-run finding 3 — run 1's strict marks stay where they are
# ---------------------------------------------------------------------------

def test_run_1s_strict_marks_stay_on_the_frozen_artifact():
    """§2 item 3 of the closing note named "the 8 + 1 xfail(strict=True)
    marks" for removal; they live in run 1's replay against the frozen run-1
    artifact, whose numbers cannot change, so they stay — as run 2 left run 1's
    rotation marks. What closes is this module: zero red, zero xfail."""
    assert RUN1.FIXTURE.name == "fixture.json"
    assert len(RUN1.KNOWN_RED["forward_identity"]) == 8
    assert len(RUN1.KNOWN_RED["ad_vs_fd"]) == 1
    assert all(m.name != "xfail" for m in getattr(test_adjudicated_verdict, "pytestmark", []))


# ---------------------------------------------------------------------------
# the physics gates at the claims rung, as run 2 measured them (unchanged)
# ---------------------------------------------------------------------------

def test_physics_gates_at_the_claims_rung(fx):
    for dut in ("pec_short", "slab", "thru"):
        for lane in ("false", "flux"):
            c = _cell(fx, dut, "fine", lane)
            m = G.cell_metrics(G.s_from_json(c["s_params"]))
            assert c["column_power_max"] < G.COLUMN_POWER_MAX
            assert m["reciprocity_mag_mean"] < G.RECIPROCITY_MAG_MAX
            assert m["reciprocity_complex_max"] <= G.RECIPROCITY_COMPLEX_MAX
            for v in G.cell_settling_effective(c).values():
                assert v <= G.SETTLING_DB_MAX
    assert _cell(fx, "pec_short", "fine", "false")["column_power_max"] == pytest.approx(1.000287, abs=1e-6)
    assert _cell(fx, "slab", "fine", "flux")["column_power_max"] == pytest.approx(1.000009, abs=1e-6)


# ===========================================================================
# LIVE layer — §5.11 of the run-2 note, re-pointed at THIS artifact
# ===========================================================================
# Moved from tests/oracle/test_waveguide_chain_battery_guide_cell_aperture.py.
# LIVE_ABS_S_ENVELOPE (5.000e-6) and the derived LIVE_ABS_S_TOL (1e-4) are
# imported from the frozen module unchanged; they are committed values from a
# measured cross-backend envelope and are not moved here.


def _live_compare(fx_, rung: str):
    for dut in F.DUTS:
        for lane in F.LANES:
            label = G.LANE_LABELS[lane]
            stored = _cell(fx_, dut, rung, label)
            sim, res, S, codes = _measure_cell(dut, rung, lane)
            assert codes == sorted(f["code"] for f in stored["preflight"]), (dut, rung, label, codes)
            S0 = G.s_from_json(stored["s_params"])
            d = float(np.max(np.abs(S - S0)))
            m = G.cell_metrics(S)
            print(f"[live {dut}-{rung}-{label}] max|S_live-S_fixture|={d:.3e} "
                  f"settling={np.asarray(res.settling_db)} colpow={m['column_power_max']:.5f} "
                  f"recip_c={m['reciprocity_complex_max']:.2e}")
            assert np.all(np.isfinite(S))
            assert d <= LIVE_ABS_S_TOL, (dut, rung, label, d, LIVE_ABS_S_TOL,
                                         "cross-backend excess is reported with both backends' "
                                         "numbers, never absorbed by widening the pin (§5.11)")


@pytest.mark.slow
@pytest.mark.parametrize("rung", ["coarse", "mid"])
def test_live_cells_reproduce_the_fixture_cpu(fx, rung):
    """§5.11 row 1 against this artifact (cells bit-identical to run 2's)."""
    _live_compare(fx, rung)


@pytest.mark.slow
@pytest.mark.gpu
def test_live_cells_reproduce_the_fixture_fine_rung(fx):
    """§5.11 row 2, on the GPU lane (the fine rung is 4x the steps)."""
    _live_compare(fx, "fine")


@pytest.mark.slow
@pytest.mark.parametrize("lane", F.LANES, ids=lambda l: G.LANE_LABELS[l])
def test_live_plane_shift_rotation_coarse_rung(lane):
    """§5.11 row 3: the coarse rung against physics, not against the artifact,
    with the sign-discriminating pair (predicted 0.512° / 0.669° / 64.07°)."""
    sim, res, S_base, _ = _measure_cell("slab", "coarse", lane)
    sim_s = F.build_simulation("slab", G.RUNG_DX["coarse"],
                               reference_planes=(F.REF_LEFT_SHIFTED_M, F.REF_RIGHT_SHIFTED_M))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_shift = np.asarray(sim_s.compute_waveguide_s_matrix(
            num_periods=F.NUM_PERIODS, normalize=lane).s_params).astype(complex)
    grid = sim._build_grid()
    rot = G.plane_shift_rotation(S_base, S_shift, F.FREQS, float(grid.dt), G.RUNG_DX["coarse"])
    print(f"[live plane-shift coarse {G.LANE_LABELS[lane]}] |S| max diff={rot['abs_s_max_diff']:.2e} "
          f"resid_yee={rot['resid_yee_max']:.3f}° resid_cont={rot['resid_cont_max']:.3f}° "
          f"wrong_sign_min={rot['wrong_sign_resid_min']:.1f}°")
    assert rot["abs_s_allclose"]
    assert rot["resid_yee_max"] <= G.ROTATION_TOL_YEE_DEG
    assert rot["resid_cont_max"] <= G.ROTATION_TOL_CONTINUOUS_DEG
    assert rot["wrong_sign_resid_min"] > G.WRONG_SIGN_MIN_DEG


def test_the_live_pin_is_the_committed_one():
    """§5.11: LIVE_ABS_S_ENVELOPE and its derived tolerance are not moved by this
    run. They live in the frozen module and are imported, not restated."""
    assert LIVE_ABS_S_ENVELOPE == 5.000e-6
    assert LIVE_ABS_S_TOL == gate_from_envelope(LIVE_ABS_S_ENVELOPE, quantum=10000) == 1e-4

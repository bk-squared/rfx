"""WR-90 chain battery, SECOND run — replay of the corrected-port artifact.

Pre-declaration (committed before the first S-parameter of this run, at
``68c8c340``): ``docs/design_notes/waveguide_chain_battery_remeasure_predeclaration.md``.
Artifact: ``tests/fixtures/waveguide_chain_battery/fixture_guide_cell_aperture.json``
(schema_version 2, VESSL run 369367258205, gpu-rtx4090, 1218.3 s).

What separates this artifact from the frozen one is the instrument, not the
geometry: ``WaveguidePort``'s transverse eigenproblem now runs on the guide's own
N cells instead of the node span's N+1 (PR #889), so ``f_cutoff`` is the guide's
discrete cutoff, and the reference-plane shift pair was re-declared (§4) because
the first pair put ``2βΔ`` through a half turn inside the band, where the
wrong-sign discriminator is degenerate by arithmetic.

The frozen artifact ``fixture.json`` and its replay
``tests/oracle/test_waveguide_chain_battery.py`` are untouched: it is the record
of a port that no longer exists.

This module holds three things:

* the **identity** of the artifact (§7) and the two discriminators that tell the
  two runs apart from the files alone — ``fc_port_hz`` and
  ``port_cutoff_effective_width_cells``, each against its closed form;
* the **adjudication** — every one of the 185 stored verdicts asserted at the
  value this run measured, plus one test per §5 leg pinning the number the
  adjudication rests on inside the branch that fired. A future change that moves
  any of them reds here rather than passing quietly;
* (moved on) the **live layer** (§5.11's success criterion) was re-pointed from
  the frozen artifact to this one here, with the three ``xfail(strict=True)``
  marks removed; it now reads the v1.8 closing artifact in
  ``tests/oracle/test_waveguide_chain_battery_v18_close.py`` (whose 18 cells are
  bit-identical to this one's). This module is the adjudication of run 2 as it
  was measured, float32 primary on every lane, and is not re-read under the
  closing declaration (``recompute_verdicts`` applies it from schema_version 3).

No gate, tolerance, golden or pin is moved here. Numbers that missed their
predicted branch are pinned at what they measured and named in the PR body as
misses, never widened.
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

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture_guide_cell_aperture.json"
FROZEN = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture.json"
PREDECLARATION = "docs/design_notes/waveguide_chain_battery_remeasure_predeclaration.md"
C0 = 299_792_458.0

# The guide's broad-wall cell count per rung, pinned by
# tests/unit/geometry/test_waveguide_chain_battery_geometry.py.
GUIDE_CELLS = {"coarse": 9, "mid": 18, "fine": 36}

ADJUDICATED_VERDICTS: dict[str, str] = {
    "ad_vs_fd|pec_short|false|eps|im_s11": "pass",
    "ad_vs_fd|pec_short|false|eps|re_s11": "pass",
    "ad_vs_fd|pec_short|false|eps|s11_mag2": "pass",
    "ad_vs_fd|pec_short|false|sigma|s11_mag2": "pass",
    "ad_vs_fd|pec_short|flux|eps|im_s11": "pass",
    "ad_vs_fd|pec_short|flux|eps|re_s11": "pass",
    "ad_vs_fd|pec_short|flux|eps|s11_mag2": "fail",
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
    "forward_identity|pec_short|flux|eps|im_s11": "fail",
    "forward_identity|pec_short|flux|eps|re_s11": "fail",
    "forward_identity|pec_short|flux|eps|s11_mag2": "fail",
    "forward_identity|pec_short|flux|sigma|s11_mag2": "fail",
    "forward_identity|slab|false|eps|im_s21": "pass",
    "forward_identity|slab|false|eps|re_s21": "pass",
    "forward_identity|slab|false|eps|s11_mag2": "pass",
    "forward_identity|slab|false|eps|s21_mag2": "pass",
    "forward_identity|slab|flux|eps|im_s21": "fail",
    "forward_identity|slab|flux|eps|re_s21": "fail",
    "forward_identity|slab|flux|eps|s11_mag2": "fail",
    "forward_identity|slab|flux|eps|s21_mag2": "fail",
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


def _load() -> dict | None:
    return json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None


_FX = _load()


@pytest.fixture(scope="module")
def fx() -> dict:
    if _FX is None:
        pytest.fail(f"{FIXTURE} is missing — regenerate with the driver named in "
                    "tests/fixtures/waveguide_chain_battery/README.md")
    return _FX


def _cell(dut: str, rung: str, lane: str) -> dict:
    for c in _FX["cells"]:
        if (c["dut"], c["rung"], c["lane"]) == (dut, rung, lane):
            return c
    raise KeyError((dut, rung, lane))


def _leg(dut: str, lane: str, kind: str, objective: str) -> dict:
    for entry in _FX["ad_vs_fd"]:
        if (entry["dut"], entry["lane"], entry["theta_kind"], entry["objective"]) == (dut, lane, kind, objective):
            return entry
    raise KeyError((dut, lane, kind, objective))


# ===========================================================================
# Identity — §7
# ===========================================================================

def test_identity_stamp(fx):
    """§7: the new run writes a new file whose header names the note that
    governs it, the pair it was measured with, and the artifact it supersedes."""
    assert fx["schema"] == "rfx.waveguide_chain_battery"
    assert fx["schema_version"] == 2
    assert fx["predeclaration"] == PREDECLARATION and (REPO / fx["predeclaration"]).exists()
    assert fx["shift_pair_name"] == "sign_discriminating_pair"
    assert fx["supersedes"] == "tests/fixtures/waveguide_chain_battery/fixture.json"
    assert fx["supersedes_reason"]
    assert fx["legs_rung"] == G.LEGS_RUNG_DEFAULT == "fine"
    p = fx["provenance"]
    # run_id names the job. The measurement stamped the literal "vessl" because
    # the yaml's ${VESSL_RUN_ID:-vessl} fallback fired on an unset env var; the
    # real id was recovered from the backed-up log and written in, with
    # run_id_source recording that it was recovered rather than stamped. Pin
    # both, so neither the id nor its provenance can be quietly dropped, and
    # reject the fallback strings outright.
    assert p["run_id"] == "369367258205", p["run_id"]
    assert p["run_id"] not in ("local", "vessl", "UNSET-see-log-filename")
    assert "recovered" in p["run_id_source"] and "369367258205" in p["run_id_source"]
    assert p["run_lane"] == "vessl"
    assert fx["predeclaration_sha"] not in ("unknown", "", p["commit"])
    assert p["commit"].startswith(fx["predeclaration_sha"]) is False, (
        "the pre-declaration must predate the run commit, not be it")
    # §3: the lane, backend and precision are declared unchanged from run 1. A
    # different backend or precision makes the run not a comparison at all.
    assert p["jax_default_backend"] == "gpu"
    assert p["precision"] == "float32" and p["jax_enable_x64"] is False
    assert p["wall_time_s"] > 30.0


def test_the_frozen_artifact_is_a_different_file_and_stays_at_schema_1():
    """§7: ``fixture.json`` is not edited, re-pinned, renamed or deleted. It
    keeps its own shift pair, so the builder guard binds for both files."""
    frozen = json.loads(FROZEN.read_text())
    assert frozen["schema_version"] == 1
    assert "shift_pair_name" not in frozen, "the frozen artifact gained a key — it is not frozen"
    assert frozen["predeclaration"] == "docs/design_notes/waveguide_chain_battery_predeclaration.md"
    assert frozen["fixture"]["reference_planes_shifted_m"] == list(F.SHIFT_PAIRS_M["half_turn_pair"])
    assert _FX["fixture"]["reference_planes_shifted_m"] == list(F.SHIFT_PAIRS_M["sign_discriminating_pair"])


def test_fixture_constants_match_the_named_shift_pair(fx):
    c = fx["fixture"]
    assert c["a_m"] == F.A_M and c["b_m"] == F.B_M
    assert c["dx_ladder_m"] == list(F.DX_LADDER) and c["n_ladder"] == list(F.N_LADDER)
    assert c["reference_planes_default_m"] == [F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M]
    assert c["reference_planes_shifted_m"] == list(F.SHIFT_PAIRS_M[fx["shift_pair_name"]])
    assert c["reference_planes_shifted_m"] == [F.REF_LEFT_SHIFTED_M, F.REF_RIGHT_SHIFTED_M]
    assert c["num_periods"] == F.NUM_PERIODS and c["lanes"] == ["false", "flux"]
    np.testing.assert_allclose(c["freqs_hz"], F.FREQS, rtol=0, atol=0)


@pytest.mark.parametrize("rung", ["coarse", "mid", "fine"])
def test_instrument_discriminator_reads_the_guides_own_cells(fx, rung):
    """§7 discriminator 1. ``fc_port_hz`` must be the discrete cutoff of an
    M = N cell aperture, ``fc(M) = (2/dx)·sin(π/(2M))·c/2π``; the effective-width
    metric is its continuous inversion ``π/(2·sin(π/(2M)))``, which reads a little
    ABOVE M and is not asserted at 9 / 18 / 36. Both are evaluated at M = N and at
    M = N+1 so an artifact matching neither is filed as neither run."""
    n_cells = GUIDE_CELLS[rung]
    dx = fx["fixture"]["dx_ladder_m"][("coarse", "mid", "fine").index(rung)]

    def fc(m):
        return (2.0 / dx) * math.sin(math.pi / (2 * m)) * C0 / (2 * math.pi)

    def eff(m):
        return math.pi / (2 * math.sin(math.pi / (2 * m)))

    for lane in ("false", "flux"):
        pc = fx["port_cutoff"]["per_rung"][f"{rung}|{lane}"]
        assert pc["fc_port_hz"] == pytest.approx(fc(n_cells), rel=1e-9), (rung, lane)
        assert pc["fc_port_hz"] != pytest.approx(fc(n_cells + 1), rel=1e-3), "this is the OLD aperture"
        assert pc["port_cutoff_effective_width_cells"] == pytest.approx(eff(n_cells), rel=1e-9)
        assert abs(pc["port_cutoff_effective_width_cells"] - eff(n_cells + 1)) > 0.9, (
            "the two apertures must stay a full cell apart")
        # §7's third sentence: "``rms_deg_at_port_cutoff`` collapses … to
        # ``rms_deg_at_discrete_guide``". That collapse IS the claim, so it is
        # asserted as the equality it is rather than as a loose bound this run
        # would have chosen for itself after seeing its own answer.
        assert pc["rms_deg_at_port_cutoff"] == pc["rms_deg_at_discrete_guide"]
        # §7 parenthesised the FROZEN artifact's own rms_deg_at_discrete_guide
        # (0.0797 / 0.0173 / 0.0041°) as the predicted new value. The measured
        # phase moved as well, so both keys read 0.033620 / 0.012387 / 0.003592°
        # — 2.4× / 1.4× / 1.2× off that parenthesis, in the better direction.
        # Pinned at what it measured; named as a miss in the PR body. §7's own
        # outcome branch is scoped to fc_port_hz and the effective width, both of
        # which sit on fc(N) / effw(N) above, so the discriminator itself is green.
        assert pc["rms_deg_at_port_cutoff"] == pytest.approx(
            {"coarse": 0.033620, "mid": 0.012387, "fine": 0.003592}[rung], rel=2e-3)
        assert pc["rms_deg_at_port_cutoff"] < {"coarse": 0.0797, "mid": 0.0173,
                                               "fine": 0.0041}[rung]


def test_the_absorber_reader_did_not_move(fx):
    """§3, checked not assumed: the CPML derivation reads preflight's
    wall-to-wall span, not the port's eigen-aperture, so PR #889 does not move it."""
    fc_wall = C0 / (2.0 * F.A_M)
    for c in fx["cells"]:
        assert c["fc_te10_numerical_hz"] == pytest.approx(fc_wall, rel=1e-12)
        assert c["cpml_layers"] == {"coarse": 17, "mid": 34, "fine": 68}[c["rung"]]
        assert c["n_steps"] == {"coarse": 713, "mid": 1425, "fine": 2849}[c["rung"]]


# ===========================================================================
# Adjudication — every stored verdict, at the value this run measured
# ===========================================================================

@pytest.mark.parametrize("key", sorted(ADJUDICATED_VERDICTS),
                         ids=lambda k: k.replace("|", "-").replace(":", "-"))
def test_adjudicated_verdict(fx, key):
    """Each of the 185 verdicts, pinned at what this run measured and adjudicated
    against its §5 branch in the PR body. 126 pass, 50 report_only, 9 fail — the
    nine reds are the flux lane's forward identity (8, §6: not closeable inside
    this run) and the zero-derivative AD leg (1)."""
    assert fx["verdicts"][key] == ADJUDICATED_VERDICTS[key]


def test_stored_verdicts_equal_recomputed(fx):
    recomputed = G.recompute_verdicts(fx)
    assert recomputed == fx["verdicts"], {
        k: (fx["verdicts"].get(k), recomputed.get(k))
        for k in set(recomputed) | set(fx["verdicts"])
        if fx["verdicts"].get(k) != recomputed.get(k)}
    assert len(recomputed) == len(ADJUDICATED_VERDICTS) == 185


def test_verdict_census(fx):
    """The census the PR body quotes: 24 red in four families became 9 red in two."""
    from collections import Counter
    census = Counter(fx["verdicts"].values())
    assert dict(census) == {"pass": 126, "report_only": 50, "fail": 9}
    reds = sorted(k for k, v in fx["verdicts"].items() if v == "fail")
    assert reds == [
        "ad_vs_fd|pec_short|flux|eps|s11_mag2",
        "forward_identity|pec_short|flux|eps|im_s11",
        "forward_identity|pec_short|flux|eps|re_s11",
        "forward_identity|pec_short|flux|eps|s11_mag2",
        "forward_identity|pec_short|flux|sigma|s11_mag2",
        "forward_identity|slab|flux|eps|im_s21",
        "forward_identity|slab|flux|eps|re_s21",
        "forward_identity|slab|flux|eps|s11_mag2",
        "forward_identity|slab|flux|eps|s21_mag2",
    ]
    assert not [k for k, v in fx["verdicts"].items() if v == "not_interpretable"]


# ===========================================================================
# §5 leg by leg — the number the adjudication rests on, in the branch that fired
# ===========================================================================

@pytest.mark.parametrize("key", ["pec_short|false", "pec_short|flux", "slab|false", "slab|flux"])
def test_rotation_residual_branch_one(fx, key):
    """§5.1 branch 1, "Residual ≤ 0.1° at the fine rung — as predicted".
    Measured 0.0317° against a 3° gate and a 0.032° point prediction; the
    continuous-β residual is 0.0407° against 6°. Run 1 read 6.602° / 6.565°.

    §5.1's last branch requires ``resid_port_beta_max`` beside it: it must stay at
    the ~1e-5° it had in run 1 and must NOT collapse together with
    ``resid_yee_max`` — if the two merged, the gate would be comparing a quantity
    against itself and would measure nothing. Measured 9.0e-6 … 2.3e-5°, three
    orders below the residual against the guide's own β."""
    p = fx["plane_shift"][key]
    assert p["resid_yee_max"] <= 0.1, p["resid_yee_max"]
    assert p["resid_yee_max"] == pytest.approx(0.0317, abs=5e-4)
    assert p["resid_cont_max"] == pytest.approx(0.0407, abs=5e-4)
    assert p["resid_yee_max"] <= G.ROTATION_TOL_YEE_DEG
    assert p["resid_cont_max"] <= G.ROTATION_TOL_CONTINUOUS_DEG
    assert 5e-6 <= p["resid_port_beta_max"] <= 5e-5, p["resid_port_beta_max"]
    assert p["resid_yee_max"] / p["resid_port_beta_max"] > 1e3, "the two β's have merged"
    # the two cutoffs must stay different numbers (§5.7's disclosure)
    assert p["fc_port_hz"] == pytest.approx(6.555060e9, rel=1e-6)
    assert p["fc_predeclared_hz"] == pytest.approx(6.557140e9, rel=1e-6)
    assert abs(p["fc_predeclared_hz"] - p["fc_port_hz"]) == pytest.approx(2.080e6, abs=1e4)


@pytest.mark.parametrize("key", ["pec_short|false", "pec_short|flux", "slab|false", "slab|flux"])
def test_wrong_sign_discriminator_branch_one(fx, key):
    """§5.2 branch 1, "≥ 60° — as predicted; the discriminator is live for the
    first time in this battery and a flipped shift sign cannot hide". Measured
    64.0549° against a 10° floor and a 64.05° prediction, set by ∠S22, the
    smallest lever. Run 1 read 0.734°, degenerate by arithmetic."""
    p = fx["plane_shift"][key]
    assert p["wrong_sign_resid_min"] >= 60.0
    assert p["wrong_sign_resid_min"] == pytest.approx(64.05, abs=0.05)
    assert p["wrong_sign_resid_min"] > G.WRONG_SIGN_MIN_DEG
    per_entry = {e: r["wrong_sign_resid_min"] for e, r in p["rotation_deg"].items() if r["measurable"]}
    assert per_entry["S22"] == pytest.approx(64.055, abs=0.01), "∠S22 is the binding margin"
    assert per_entry["S11"] == pytest.approx(126.446, abs=0.01)


def test_cheap_refute_still_reds_the_rotation_gate(fx):
    """§5.2's last bullet: the flipped-sign refute must red the rotation gate by
    ≥ 60°. Measured 117.27° … 178.00° per entry at the coarse rung. It is recorded
    because it catches a dead GATE; it is never quoted as corroboration that the
    discriminator is live — in run 1 it passed at 119.49° in the same file where
    ``wrong_sign_resid_min`` sat at 0.734°."""
    r = fx["plane_shift"]["cheap_refute"]
    assert r["rotation_gate_would_pass"] is False
    assert r["resid_yee_min_over_entries"] >= 60.0
    assert r["resid_yee_min_over_entries"] == pytest.approx(117.269, abs=0.01)
    assert r["abs_s_still_invariant"] is True


def test_settling_witness_per_drive(fx):
    """§5.3. 34 of the 36 drives land in the predicted −78 … −102 dB window; no
    rerun fired anywhere, no drive read 0.00 dB, and no ``settling_db`` is NaN.

    The two exceptions are recorded, not smoothed: ``pec_short|fine|false`` left
    reads −102.279 dB and ``slab|fine|false`` left −102.646 dB, i.e. 0.28 and
    0.65 dB below the window. That is NOT branch 2 ("outside −78 … −102 by more
    than 5 dB") and NOT the fine-rung trigger of the last branch ("at or below
    −106 dB", the level 80 periods produced in run 1), so no declared response
    fires — see the PR body, where the pair is reported as non-closing against
    §5.3's own partition rather than as "as predicted"."""
    for c in fx["cells"]:
        for port, db in c["settling_db"].items():
            assert db == db, (c["dut"], c["rung"], c["lane"], port)      # not NaN
            assert db <= G.SETTLING_DB_MAX, (c["dut"], c["rung"], c["lane"], port, db)
            assert db != 0.0
        assert c.get("settling_rerun") is None, "no cell triggered the 80-period rerun"
    outside = {(c["dut"], c["rung"], c["lane"], port): db
               for c in fx["cells"] for port, db in c["settling_db"].items()
               if not (-102.0 <= db <= -78.0)}
    assert set(outside) == {("pec_short", "fine", "false", "left"),
                            ("slab", "fine", "false", "left")}, outside
    for db in outside.values():
        assert -106.0 < db < -102.0, db
    assert min(db for c in fx["cells"] for db in c["settling_db"].values()) == pytest.approx(-102.646, abs=1e-3)
    assert max(db for c in fx["cells"] for db in c["settling_db"].values()) == pytest.approx(-79.585, abs=1e-3)


def test_settling_degenerate_record_set_did_not_move(fx):
    """§5.3's branch that is read whatever the dB says: the driver-side skip list
    must be the frozen one, name for name, or the dB was computed over a different
    population. 8 entries on each of the six ``pec_short`` cells — the four records
    of the far port on each of the two drives — and empty on all twelve
    ``thru`` / ``slab`` cells."""
    frozen = json.loads(FROZEN.read_text())
    for c in fx["cells"]:
        oc = [x for x in frozen["cells"]
              if (x["dut"], x["rung"], x["lane"]) == (c["dut"], c["rung"], c["lane"])][0]
        new = sorted(c.get("settling_degenerate_records") or [])
        old = sorted(oc.get("settling_degenerate_records") or [])
        assert new == old, (c["dut"], c["rung"], c["lane"], new, old)
        assert len(new) == (8 if c["dut"] == "pec_short" else 0)


def test_empty_guide_s11_is_the_unnamed_branch(fx):
    """§5.4 branch 3, "Anything else below the frozen 0.0320 — i.e. every
    value/ratio combination the two branches above do not name … This is a case in
    its own right and is not sorted into whichever branch it is nearest."

    Measured max_f|S11| on the empty guide, ``normalize=False``:
    0.041048 / 0.016405 / 0.007028 at coarse / mid / fine, ratios 2.502 and 2.334
    (order 1.32 and 1.22 in dx). Not branch 1 (≤ 0.004 with ratios ≈ 4) and not
    branch 2 (near 0.010 with ratios ≈ 2). Every rung is far below its frozen
    value (0.135383 / 0.065278 / 0.032022), so no order can be attributed and
    §2.2's first-order story is neither confirmed nor refuted by this leg."""
    vals = [_cell("thru", r, "false")["non_vacuity_max_s11"] for r in ("coarse", "mid", "fine")]
    assert vals == pytest.approx([0.041048, 0.016405, 0.007028], abs=5e-6)
    frozen = [0.135383, 0.065278, 0.032022]
    for v, f in zip(vals, frozen):
        assert v < f, (v, f)
    ratios = [vals[0] / vals[1], vals[1] / vals[2]]
    assert ratios == pytest.approx([2.502, 2.334], abs=5e-3)
    assert not (vals[2] <= 0.004 and 3.5 <= ratios[0] <= 4.5), "not branch 1"
    # The branch's declared response includes "the per-bin |S11| curve at each
    # rung". It lives in the artifact as cells[].s_params.S11; pinned here by its
    # length, its worst bin and that bin's value, so a later change to the curve
    # reds instead of passing quietly. The worst bin is 8.6 GHz at every rung —
    # the low end of the band, nearest cutoff, which is where a residual aperture
    # mismatch would sit.
    freqs = fx["fixture"]["freqs_hz"]
    for rung, expected in (("coarse", 0.041048), ("mid", 0.016405), ("fine", 0.007028)):
        curve = [abs(complex(re, im))
                 for re, im in _cell("thru", rung, "false")["s_params"]["S11"]]
        assert len(curve) == len(freqs)
        worst = max(range(len(curve)), key=curve.__getitem__)
        assert freqs[worst] == 8.6e9, (rung, freqs[worst])
        assert curve[worst] == pytest.approx(expected, abs=5e-6)


def test_empty_guide_column_power_is_the_unnamed_combination_branch(fx):
    """§5.4's LAST column-power branch, "Anything else below the frozen value at
    every rung — the case the four branches above do not name: every rung inside a
    factor 2 of its prediction but the ratio sequence away from 4 … This branch is
    a case in its own right and is not sorted into whichever branch it is nearest."

    Measured excess 6.135e-3 / 1.161e-3 / 2.546e-4 against a predicted
    6e-3 / 1.5e-3 / 4e-4 — 1.02× / 0.77× / 0.64×, so every rung is inside its
    factor 2 and no per-rung factor-2 branch fires. The ratios are 5.284 and 4.560.
    5.284 is 1.32× the declared 4; the contract's only calibration for "away from
    4" is its own worked example, "ratios ≈ 3", which is 0.76× — the same
    multiplicative distance. Reading 5.284 as "~4×" while
    ``test_empty_guide_s11_is_the_unnamed_branch`` refuses 2.502 as "≈ 2" on the
    same cells would be two standards for one run, so this leg is read by the
    branch written for exactly this shape.

    Its declared response, delivered here and in the PR body: the three per-rung
    excesses, the two ratios, the per-bin column-power curve at each rung, |S11|
    and |S21| separately at the worst bin, and the statement that the 4.06× ratio
    transferred from the PEC-short (§5.5) is NOT confirmed on this DUT."""
    ex = [_cell("thru", r, "false")["column_power_max"] - 1.0 for r in ("coarse", "mid", "fine")]
    assert ex == pytest.approx([6.135e-3, 1.161e-3, 2.546e-4], rel=2e-3)
    for e, pred in zip(ex, (6e-3, 1.5e-3, 4e-4)):
        assert 0.5 * pred <= e <= 2.0 * pred, (e, pred)
    ratios = [ex[0] / ex[1], ex[1] / ex[2]]
    assert ratios == pytest.approx([5.284, 4.560], abs=5e-3)
    # not the branch this leg was first reported under: 5.284 is as far from 4 in
    # log space as the contract's own "≈ 3" example of a sequence away from 4.
    assert abs(math.log(ratios[0] / 4.0)) >= abs(math.log(3.0 / 4.0)) * 0.95
    # every rung below its frozen value, which is what puts this in the LAST
    # branch rather than in "worse than the frozen value at that rung"
    for e, frozen in zip(ex, (1.8252530e-2, 4.0816620e-3, 9.8340770e-4)):
        assert e < frozen, (e, frozen)
    for r in ("coarse", "mid", "fine"):
        assert _cell("thru", r, "false")["column_power_max"] < G.COLUMN_POWER_MAX
    # the per-bin curve, and |S11| / |S21| at the worst bin. The excess is NOT a
    # mismatch term: at 11.6 GHz it is |S21|² − 1 that carries it (5.454e-3 of
    # 6.135e-3 coarse, 1.087e-3 of 1.161e-3 mid, 2.464e-4 of 2.546e-4 fine) while
    # |S11|² is 6.8e-4 / 7.5e-5 / 8.2e-6 — one to one and a half orders below.
    # §5.4's own words for that: "a column-power excess that does NOT track |S11|
    # points at |S21| — a transmitted-magnitude error — rather than at the
    # mismatch, and that distinction is the thing to report."
    freqs = fx["fixture"]["freqs_hz"]
    expected = {"coarse": (6.135092e-3, 0.026092, 1.002723),
                "mid": (1.161080e-3, 0.008636, 1.000543),
                "fine": (2.546142e-4, 0.002864, 1.000123)}
    for rung, (exc, s11_w, s21_w) in expected.items():
        c = _cell("thru", rung, "false")
        cols = c["column_power_per_bin"]
        assert len(cols) == 2 and all(len(col) == len(freqs) for col in cols)
        per_bin = [max(cols[0][i], cols[1][i]) - 1.0 for i in range(len(freqs))]
        worst = max(range(len(per_bin)), key=per_bin.__getitem__)
        assert freqs[worst] == 11.6e9, (rung, freqs[worst])
        assert per_bin[worst] == pytest.approx(exc, rel=2e-3)
        s11 = abs(complex(*c["s_params"]["S11"][worst]))
        s21 = abs(complex(*c["s_params"]["S21"][worst]))
        assert s11 == pytest.approx(s11_w, abs=5e-6)
        assert s21 == pytest.approx(s21_w, abs=5e-6)
        assert (s21 ** 2 - 1.0) > 6.0 * s11 ** 2, (
            "the excess is a |S21| term, not the mismatch")


def test_pec_short_column_power_branch_one(fx):
    """§5.5 branch 1, "Every rung's excess within a factor 2 of its prediction,
    sequence still ~4× per halving — as predicted". The predicted WORSENING landed:
    excess 4.716e-3 / 1.162e-3 / 2.867e-4 against a prediction of 4.7e-3 / 1.2e-3 /
    3e-4 (1.00× / 0.97× / 0.96×) and ratios 4.059 / 4.053 against the declared 4.06.
    Run 1 read 1.158e-4 / 3.205e-5 / 9.398e-6. The gate is not moved: the worst
    value is 1.0047 against 1.02, and ``max|S11| = √column_power`` = 1.000143 at the
    claims rung, inside the referee's 0.99 … 1.03."""
    ex = [_cell("pec_short", r, "false")["column_power_max"] - 1.0 for r in ("coarse", "mid", "fine")]
    assert ex == pytest.approx([4.716e-3, 1.162e-3, 2.867e-4], rel=2e-3)
    for e, pred in zip(ex, (4.7e-3, 1.2e-3, 3e-4)):
        assert 0.5 * pred <= e <= 2.0 * pred, (e, pred)
    assert [ex[0] / ex[1], ex[1] / ex[2]] == pytest.approx([4.059, 4.053], abs=5e-3)
    for r in ("coarse", "mid", "fine"):
        c = _cell("pec_short", r, "false")
        assert c["column_power_max"] < G.COLUMN_POWER_MAX
        assert c["non_vacuity_max_s11"] == pytest.approx(math.sqrt(c["column_power_max"]), rel=2e-6)


@pytest.mark.parametrize("group,scaled,x64", [
    ("pec_short|flux|eps", 1.083215, 1.7380284854671055e-10),
    ("pec_short|flux|sigma", 1.514875, 2.8349255023183324e-10),
    ("slab|flux|eps", 1.760720, 1.0510831029970571e-08),
])
def test_forward_identity_stays_red_on_the_flux_lane(fx, group, scaled, x64):
    """§5.6(i). The eight red legs sit in these three ``(dut, lane, θ-kind)``
    groups; the metric is a property of the traced forward pass, so it takes one
    value per group. ``pec_short|flux|eps`` lands in "Stays above 1.0 and at most
    1.5" (1.083, run 1 1.065); the other two land in "1.5 < scaled ≤ 10"
    (1.515 from 1.440, 1.761 from 1.440) — worse than "unchanged", below the
    blocking bar, and read there rather than folded into the first branch.

    In every group the x64 witness stays ≤ 1e-6 (1.7e-10 / 2.8e-10 / 1.1e-8), so
    the reassociation story is NOT falsified and §6's declaration may proceed;
    §6 governs closure, and this leg does not close inside this run either way."""
    dut, lane, kind = group.split("|")
    legs = [entry for entry in fx["ad_vs_fd"]
            if (entry["dut"], entry["lane"], entry["theta_kind"]) == (dut, lane, kind)]
    assert legs
    # The group reading is only licensed while the metric is invariant inside the
    # group — §5.6(i)'s pre-declared check, not a formality: a per-leg difference
    # here would mean the traced forward pass depends on which objective is
    # differentiated, which no reassociation argument predicts.
    assert len({entry["forward_identity"]["max_scaled_diff"] for entry in legs}) == 1
    assert len({entry["forward_identity"]["max_abs_diff"] for entry in legs}) == 1
    measured = legs[0]["forward_identity"]["max_scaled_diff"]
    assert measured == pytest.approx(scaled, rel=1e-6)
    assert not G.forward_identity_pass(measured)
    assert 1.0 < measured <= 10.0
    witness = [entry["x64_witness"]["forward_identity_x64"]["max_scaled_diff"]
               for entry in legs if entry.get("x64_witness")]
    assert witness == [pytest.approx(x64, rel=1e-6)]
    assert x64 <= 1e-6, "above 1e-6 the reassociation story is falsified (§5.6(i))"
    for entry in legs:
        conc = entry.get("forward_identity_concrete_override_vs_plain")
        if conc is not None:
            assert conc["max_abs_diff"] == 0.0, "a non-zero here survives without a tracer"
    # The "1.5 < scaled ≤ 10" branch asks for two more numbers than the first
    # branch does: "Report the worst entry, its ``abs_s_at_worst``, its group's
    # x64 witness, and the per-leg ``forward_identity_concrete_override_vs_plain``
    # where the group carries one." The first two were missing from the first
    # write-up of this run; pinned here and quoted in the PR body.
    fi = legs[0]["forward_identity"]
    worst_by_group = {
        "pec_short|flux|eps": ([1, 1, 16], 1.000006477, 1.094054e-05),
        "pec_short|flux|sigma": ([0, 0, 16], 0.747453662, 1.147447e-05),
        "slab|flux|eps": ([0, 0, 0], 0.202672520, 9.150247e-06),
    }
    entry_ix, abs_s, abs_diff = worst_by_group[group]
    assert fi["worst_entry"] == entry_ix
    assert fi["abs_s_at_worst"] == pytest.approx(abs_s, rel=1e-6)
    assert fi["max_abs_diff"] == pytest.approx(abs_diff, rel=1e-5)
    assert len({tuple(e["forward_identity"]["worst_entry"]) for e in legs}) == 1
    # ``worst_entry`` / ``abs_s_at_worst`` are taken at the argmax of the SCALED
    # metric (``forward_identity_metric``), so they say what the denominator was
    # doing. It was doing nothing: the primal at the worst entry is unchanged to
    # six digits across the two runs. The branch's own description therefore
    # applies literally — "the absolute difference has grown … while the primal
    # did not change scale, which a pure reassociation argument does not predict".
    frozen_group = next(
        e["forward_identity"] for e in json.loads(FROZEN.read_text())["ad_vs_fd"]
        if (e["dut"], e["lane"], e["theta_kind"]) == (dut, lane, kind))
    assert frozen_group["worst_entry"] == entry_ix, "the worst entry did not move"
    assert fi["abs_s_at_worst"] == pytest.approx(frozen_group["abs_s_at_worst"], rel=2e-6)
    assert fi["max_abs_diff"] > frozen_group["max_abs_diff"], "the error grew, not the scale"
    # What the branch asks next: is it still a float32 story? Yes — the x64
    # witness is eight orders inside and the concrete-override is exactly 0
    # wherever the group carries one, both asserted above. By the branch's own
    # words the leg "stays a float32 story with a larger constant and §6 proceeds".


def test_forward_identity_is_exactly_zero_on_the_false_lane(fx):
    """§5.6(i)'s control: the three ``false``-lane groups are bit-identical, which
    is what says the identity is a flux-lane property and not a general tape
    defect. A non-zero at any magnitude is reported before either lane is read."""
    for entry in fx["ad_vs_fd"]:
        if entry["lane"] != "false":
            continue
        assert entry["forward_identity"]["max_abs_diff"] == 0.0, entry["objective"]
        assert entry["forward_identity"]["max_scaled_diff"] == 0.0


def test_zero_derivative_leg_stays_red_but_its_x64_witness_moved(fx):
    """§5.6(ii), the branch "Red with ``|g_ad|`` in the declared 3e-6 … 3e-4 scale
    but the x64 AD no longer ~−1e-6 — its own branch, because the x64 AD is this
    leg's entire mechanism claim … The threshold: the x64 AD must stay within a
    factor 3 of −9.821e-7 **and** keep its sign."

    Measured ``g_ad`` = +2.786e-5 (in scale), ``g_fd`` = −5.154e-8, rel 541.5,
    span 4.64e7 ULP — red as predicted. But ``g_ad_x64`` = −2.943e-7: the sign is
    kept and it is the same order, yet it is 3.34× below the run-1 reference, just
    outside the declared factor-3 band whose lower edge is −3.274e-7. By §5.6(ii)
    this leg therefore does NOT stay on §6's x64 declaration on the strength of
    this run; it is reported in the PR body with ``g_ad``, ``g_fd``, ``g_ad_x64``
    and ``fd_ulp_span`` together, and root-caused separately."""
    leg = _leg("pec_short", "flux", "eps", "s11_mag2")
    assert leg["rel"] > G.AD_FD_REL_GATE
    assert 3e-6 <= abs(leg["g_ad"]) <= 3e-4
    assert leg["g_ad"] == pytest.approx(2.786023e-05, rel=1e-4)
    assert leg["g_fd"] == pytest.approx(-5.154156e-08, rel=1e-4)
    assert leg["fd_ulp_span"] == pytest.approx(4.642e7, rel=1e-3)
    g64 = leg["x64_witness"]["g_ad_x64"]
    assert g64 == pytest.approx(-2.942583e-07, rel=1e-4)
    assert g64 < 0.0, "the x64 AD keeps the sign of the residual derivative"
    assert abs(g64) < abs(-9.821e-7) / 3.0, (
        "pinned OUTSIDE the declared factor-3 band — this is the branch that fired")


def test_zero_derivative_siblings_gradient_grew_by_an_order(fx):
    """§5.6(ii)'s sibling branch, "it stays green with a LARGER ``g_fd`` — green,
    and against the prediction, so it is not folded into the branch above".

    ``pec_short|false|eps|s11_mag2``: ``g_fd`` = +7.643e-4 against run 1's
    +7.716e-5, a factor 9.9. §5.4 predicted this lane's spurious ``|S11|`` shrinks
    by roughly an order, so its ``|S11|²`` sensitivity to εr should have shrunk
    with it. The reading the branch says to rule out first — that the growth is the
    ``Z_TE`` change (1.24 % … 4.07 % at the fine rung) — does not survive a factor
    of 9.9. The leg is green (rel 8.65e-3 against 0.05) and the growth is reported
    as a miss beside §5.4's empty-guide numbers."""
    leg = _leg("pec_short", "false", "eps", "s11_mag2")
    assert leg["rel"] <= G.AD_FD_REL_GATE
    assert leg["rel"] == pytest.approx(8.6513e-3, rel=1e-3)
    # The branch names four numbers to report together: "Report ``g_ad``,
    # ``g_fd``, ``fd_ulp_span`` and the leg's ``|S11|`` at θ0 next to §5.4's
    # empty-guide numbers." All four, pinned:
    assert leg["g_ad"] == pytest.approx(7.709435e-4, rel=1e-5)
    assert leg["g_fd"] == pytest.approx(7.643310e-4, rel=1e-5)
    assert leg["fd_ulp_span"] == pytest.approx(3.442241e11, rel=1e-5)
    assert leg["fd_ulp_span"] >= G.FD_ULP_FLOOR
    # the objective is |S11|², so |S11| at θ0 is its square root: 1.0000451
    # against run 1's 0.9999984. The PEC short's |S11| moved OUTWARD from unity,
    # which is §5.5's predicted worsening seen on a single leg — and it is why
    # the sensitivity grew instead of shrinking with §5.4's mismatch.
    assert leg["value_at_theta0"] == pytest.approx(1.00009012, rel=1e-8)
    assert math.sqrt(leg["value_at_theta0"]) == pytest.approx(1.0000451, abs=5e-7)
    assert leg["g_fd"] / 7.716e-5 > 5.0, "grew by far more than the Z_TE share"


def test_the_other_fourteen_ad_fd_legs_are_green_as_predicted(fx):
    """§5.6(ii)'s last block, branch 1: "All 14 green with ``rel`` in
    1e-4 … 1.2e-2 — as predicted; the tape is undisturbed by the port correction".

    The 14 are the 16 ``ad_vs_fd`` legs minus the zero-derivative leg and its
    ``false``-lane sibling, both of which have their own branches and their own
    tests above. On that population: ``rel`` 1.225e-4 (``slab|false|eps|s21_mag2``)
    … 1.074e-2 (``pec_short|false|eps|re_s11``), FD spans 8.75e13 … 4.82e15 ULP,
    ten to eleven orders above the 1e4 floor, no leg skipped. The sibling's own
    span, 3.44e11, belongs to the sibling and is not quoted as one of the 14."""
    excluded = {("pec_short", "flux", "eps", "s11_mag2"),
                ("pec_short", "false", "eps", "s11_mag2")}
    legs = [entry for entry in fx["ad_vs_fd"]
            if (entry["dut"], entry["lane"], entry["theta_kind"], entry["objective"])
            not in excluded]
    assert len(legs) == 14
    for entry in legs:
        assert entry["rel"] <= G.AD_FD_REL_GATE, entry
        assert 1e-4 <= entry["rel"] <= 1.2e-2, (entry["dut"], entry["objective"], entry["rel"])
        assert entry["fd_ulp_span"] >= G.FD_ULP_FLOOR, entry
    rels = sorted(entry["rel"] for entry in legs)
    spans = sorted(entry["fd_ulp_span"] for entry in legs)
    assert rels[0] == pytest.approx(1.22494e-4, rel=1e-4)
    assert rels[-1] == pytest.approx(1.07366e-2, rel=1e-4)
    assert spans[0] == pytest.approx(8.746317e13, rel=1e-5)
    assert spans[-1] == pytest.approx(4.824417e15, rel=1e-5)
    # the sibling is green on the same band but is NOT one of the 14
    sibling = _leg("pec_short", "false", "eps", "s11_mag2")
    assert 1e-4 <= sibling["rel"] <= 1.2e-2
    assert sibling["fd_ulp_span"] < spans[0]


def test_the_three_uninterpretable_ladders_became_interpretable(fx):
    """§5.6(iii) branch 2, "One or more becomes interpretable — a finding to
    explain, not a relief: it would mean the coarse rung's dominant error was the
    port's first-order cutoff term rather than the slab's under-resolution".

    All three moved into the declared [0.15, 0.70] window:
    ``slab_s11_mag|false`` 0.0369 → 0.2156, ``slab_s21_mag|false`` 0.0975 → 0.1864,
    ``pec_short_s11_phase_deg|flux`` 0.0767 → 0.2078. Seven interpretable ladders
    stayed interpretable, so no ladder moved the other way. The non-increase gate
    passes on all ten, as predicted."""
    became = {"slab_s11_mag|false": 0.21562, "slab_s21_mag|false": 0.18640,
              "pec_short_s11_phase_deg|flux": 0.20783}
    for key, ratio in became.items():
        lad = fx["ladder"][key]
        assert lad["interpretable"] is True
        assert lad["successive_ratio_worst"] == pytest.approx(ratio, abs=1e-4)
        lo, hi = lad["ratio_window"]
        assert lo <= lad["successive_ratio_worst"] <= hi
    for key, lad in fx["ladder"].items():
        assert lad["interpretable"] is True, key
        assert lad["gate_pass"] is True, key
        assert lad["monotone_fraction_of_bins"] >= lad["pinned_monotone_fraction_min"], key


def test_ladder_richardson_three_false_lane_legs_rose_above_their_frozen_value(fx):
    """§5.10's second branch, "A ``false``-lane value above its frozen value while
    its verdict stays ``pass`` or ``report_only`` … Against the prediction either
    way, and it is the same shape of contradiction §5.7 describes: the measured
    side moved toward an unmoved oracle, so it cannot get worse for the reason this
    run claims."

    **The quantity is the MID-FINE Richardson ``max_abs_diff``**, not the
    top-level ``richardson_max_abs_diff``. §5.10 names it — "Frozen ``mid-fine``
    ``max_abs_diff``" — and the eight numbers it quotes (0.925 / 0.829, 0.01883 /
    0.01811, 0.01297 / 0.01169, 2.046 / 1.888) are exactly
    ``ladder[key]["richardson"]["mid-fine"]["max_abs_diff"]`` in the frozen
    artifact. The top-level key is the COARSE-MID value (3.6905 / 0.084866 /
    0.060135 / 9.0498), which the contract never names; the first write-up of this
    run read it, and got the population wrong in one leg. The derived pin is on
    the declared quantity either way — every pinned ladder carries
    ``pinned_richardson_pair == "mid-fine"``.

    On the declared quantity THREE of the four ``false``-lane legs rose:
    ``pec_short_s11_phase_deg`` 0.925009 → 0.994539 (+7.5 %), ``slab_s21_mag``
    0.0129711 → 0.0131110 (+1.1 %), ``slab_s21_phase_deg`` 2.046017 → 2.051128
    (+0.25 %). Only ``slab_s11_mag`` fell, 0.0188338 → 0.0175896 (−6.6 %). The
    branch class is unchanged — it fires on any of the four — but the population
    against the prediction is 3 of 4, not 2. Every ``flux``-lane leg stayed inside
    its ±20 % band (1.066 / 0.9998 / 1.0000 / 1.013), so §5.10's last branch does
    not fire and the ``false``-lane movement is not explained away by a lane the
    correction was not supposed to touch.

    The derived pin on ``pec_short_s11_phase_deg|false`` moves 1.4 → 1.5 by the
    same ``gate_from_envelope`` policy; the FROZEN pin is not touched."""
    mid_fine = {k: v["richardson"]["mid-fine"]["max_abs_diff"]
                for k, v in fx["ladder"].items() if "richardson" in v}
    frozen_mid_fine = {
        "pec_short_s11_phase_deg|false": 0.9250091885249533,
        "slab_s11_mag|false": 0.01883376476653381,
        "slab_s21_mag|false": 0.012971060638738319,
        "slab_s21_phase_deg|false": 2.046016852253139,
        "pec_short_s11_phase_deg|flux": 0.8294652794336201,
        "slab_s11_mag|flux": 0.018112579378638055,
        "slab_s21_mag|flux": 0.01168789898211342,
        "slab_s21_phase_deg|flux": 1.888190642486268,
    }
    frozen = json.loads(FROZEN.read_text())
    for key, value in frozen_mid_fine.items():
        assert frozen["ladder"][key]["richardson"]["mid-fine"]["max_abs_diff"] == value
    rose = {"pec_short_s11_phase_deg|false": 0.9945386406513278,
            "slab_s21_mag|false": 0.01311100865645587,
            "slab_s21_phase_deg|false": 2.051127509983207}
    fell = {"slab_s11_mag|false": 0.017589593070640785}
    assert set(rose) | set(fell) == {k for k in frozen_mid_fine if k.endswith("|false")}
    for key, new in rose.items():
        assert mid_fine[key] == pytest.approx(new, rel=1e-9)
        assert new > frozen_mid_fine[key], key
        assert fx["ladder"][key]["verdict"] in ("pass", "report_only")
        assert fx["ladder"][key]["pinned_richardson_pair"] == "mid-fine"
    for key, new in fell.items():
        assert mid_fine[key] == pytest.approx(new, rel=1e-9)
        assert new < frozen_mid_fine[key], key
    for key in (k for k in frozen_mid_fine if k.endswith("|flux")):
        old = frozen_mid_fine[key]
        assert 0.8 * old <= mid_fine[key] <= 1.2 * old, (key, mid_fine[key], old)
    assert fx["ladder"]["pec_short_s11_phase_deg|false"]["pinned_richardson_gate"] == 1.5
    assert frozen["ladder"]["pec_short_s11_phase_deg|false"][
        "pinned_richardson_gate"] == 1.4, "the frozen pin is not moved"


@pytest.mark.parametrize("key,frozen_value", [
    ("pec_short|false", 2.2206e-7), ("pec_short|flux", 5.6586e-8),
    ("slab|false", 1.9747e-7), ("slab|flux", 7.8814e-8),
])
def test_abs_s_invariance_branch_one(fx, key, frozen_value):
    """§5.8 branch 1, "All four in 5e-8 … 3e-7 with ``abs_s_allclose`` true — as
    predicted. The de-embedding factor is unit modulus and §5.1/§5.2's rotation
    readings are licensed."

    Measured 1.7605e-7 / 8.2258e-8 / 1.5837e-7 / 6.8021e-8. The second number the
    branch asks for is the ratio to the frozen value, predicted 0.7 … 1.5 because a
    float32 rounding residual does not depend on the lever: 0.793 / 1.454 / 0.802 /
    0.863, while the per-entry levers fell 2× (∠S11), 5× (∠S22) and 3× (∠S21). The
    residual did not track the lever, so the rounding basis of §5.8 stands.

    β is real at every bin: the band starts at 8.4 GHz, 1.845 GHz above the
    corrected fine-rung cutoff, so ``exp(∓jβs)`` cannot turn into a real
    exponential."""
    p = fx["plane_shift"][key]
    assert p["abs_s_allclose"] is True
    assert 5e-8 <= p["abs_s_max_diff"] <= 3e-7, p["abs_s_max_diff"]
    assert p["abs_s_max_diff"] != 0.0, "an exact zero would say the shifted build was not shifted"
    ratio = p["abs_s_max_diff"] / frozen_value
    assert 0.7 <= ratio <= 1.5, (key, ratio)
    assert p["fc_port_hz"] < 8.4e9


def test_gradient_invariance_branch_one(fx):
    """§5.9(a) branch 1, "all ten in-programme legs ≤ 3e-7 — as predicted; the
    gradient carries the same unit-modulus factor as the value, and the AD tape is
    undisturbed by the new lever". Measured envelope 2.3243e-7 (run 1 1.7938e-7);
    ``gate_from_envelope(x, quantum=1000)`` holds 0.001 for any envelope at or
    below 1/1500 = 6.66667e-4, so the derived pin reads 0.001 again by arithmetic.
    The FROZEN pin and envelope are not touched."""
    in_programme = [g["rel_change"] for k, p in fx["plane_shift"].items() if k != "cheap_refute"
                    for obj, g in p["gradient_invariance"].items()
                    if not (p["dut"] == "pec_short" and obj == "eps:s11_mag2")]
    assert len(in_programme) == 10
    assert max(in_programme) <= 3e-7, in_programme
    assert fx["pins"]["gradient_invariance_envelope"] == pytest.approx(max(in_programme), rel=1e-12)
    assert fx["pins"]["gradient_invariance_envelope"] == pytest.approx(2.3242906e-7, rel=1e-6)
    assert fx["pins"]["gradient_invariance_envelope"] <= 1.0 / 1500.0
    assert fx["pins"]["gradient_invariance_gate"] == 0.001
    assert fx["pins"]["gradient_invariance_gate"] == gate_from_envelope(
        fx["pins"]["gradient_invariance_envelope"], quantum=G.GRADIENT_PIN_QUANTUM)
    assert json.loads(FROZEN.read_text())["pins"]["gradient_invariance_envelope"] == pytest.approx(
        1.7938165062385892e-07, rel=1e-12), "the frozen envelope is not touched"


def test_gradient_invariance_report_only_legs(fx):
    """§5.9(a)'s last branch. The two ``report_only`` legs stay ``report_only`` and
    stay out of the envelope, as predicted. ``pec_short|flux|eps:s11_mag2`` reads
    6.499e-3, inside the declared 1e-3 … 1e-2. ``pec_short|false|eps:s11_mag2``
    reads 2.664e-4, BELOW that window (run 1: 7.234e-3) — the branch "Outside it,
    either side: their own reading, because they are the gradient-side view of
    §5.6(ii)'s physically-zero derivative and the two sections must agree". It is
    the same lane whose ``g_fd`` grew by a factor 9.9, and the two are reported
    together in the PR body."""
    false_leg = fx["plane_shift"]["pec_short|false"]["gradient_invariance"]["eps:s11_mag2"]
    flux_leg = fx["plane_shift"]["pec_short|flux"]["gradient_invariance"]["eps:s11_mag2"]
    assert flux_leg["rel_change"] == pytest.approx(6.4989e-3, rel=1e-3)
    assert 1e-3 <= flux_leg["rel_change"] <= 1e-2
    assert false_leg["rel_change"] == pytest.approx(2.6645e-4, rel=1e-3)
    assert false_leg["rel_change"] < 1e-3, "below the declared window — its own branch"
    for key in ("gradient_invariance|pec_short|false|eps:s11_mag2",
                "gradient_invariance|pec_short|flux|eps:s11_mag2"):
        assert fx["verdicts"][key] == "report_only"


@pytest.mark.parametrize("key,obj,phi_meas,phi_pre,predecl", [
    ("pec_short|false", "eps:s11_complex", 92.13631373833611, 92.11422972729224, 3.8544936645878183e-4),
    ("pec_short|flux", "eps:s11_complex", 92.136315598176, 92.11422972729224, 3.8534281259561157e-4),
    ("slab|false", "eps:s21_complex", 69.10223323772178, 69.08567229546918, 2.8921349381511523e-4),
    ("slab|flux", "eps:s21_complex", 69.10222627212148, 69.08567229546918, 2.8882389847919676e-4),
])
def test_gradient_invariance_predeclared_phi_branch_one(fx, key, obj, phi_meas, phi_pre, predecl):
    """§5.9(b) branch 1, "both complex legs in 2e-4 … 6e-4 — as predicted. The
    pre-declared β and the applied β agree at the centre bin to about 0.02°, which
    is §5.1's whole-band statement measured on a different quantity and a different
    code path."

    Predicted 3.85e-4 (∠S11) and 2.89e-4 (∠S21) from a frozen 6.480e-2 / 7.290e-2,
    factors of 168 and 252; measured 3.854e-4 / 3.853e-4 and 2.892e-4 / 2.888e-4.
    ``phi_measured_deg`` moves to +92.136° / +69.102° against a predicted +92.136° /
    +69.102°, and ``phi_predeclared_deg`` to 92.114° / 69.086°."""
    g = fx["plane_shift"][key]["gradient_invariance"][obj]
    assert 2e-4 <= g["rel_change_predeclared_phi"] <= 6e-4
    assert g["rel_change_predeclared_phi"] == pytest.approx(predecl, rel=1e-4)
    assert g["phi_measured_deg"] == pytest.approx(phi_meas, abs=1e-3)
    assert g["phi_predeclared_deg"] == pytest.approx(phi_pre, abs=1e-3)
    # the identity the section is built on: rel_change_predeclared_phi = 2|sin(Δφ/2)|
    dphi = math.radians(g["phi_measured_deg"] - g["phi_predeclared_deg"])
    assert g["rel_change_predeclared_phi"] == pytest.approx(2.0 * abs(math.sin(dphi / 2)), rel=2e-3)


def test_referee_at_the_claims_rung(fx):
    """§5.7. Four of the five ``false``-lane referee numbers land inside their
    predicted intervals (slab-vs-Airy magnitude 0.012620 in 0.00903 … 0.02072; slab
    column power 1.0003528 in 1.000014 … 1.000975; magnitude reciprocity 2.2507e-3
    in 7.7e-7 … 2.585e-3; complex reciprocity 4.8064e-3 in 3.28e-4 … 6.983e-3).
    The flux lane does NOT behave the same way — see
    ``test_flux_lane_column_power_is_below_its_predicted_band`` — so branch 1 is
    not claimed for this section as a whole.

    The fifth fires "Green and better than predicted — any ``false``-lane number
    below the bottom of its predicted interval": slab-vs-Airy PHASE reads 6.0986°,
    below the 6.58° floor, and below its own lane-mate's 6.1497°. The interval's
    lower bound was an argument — "the flux lane's residual is the part ``Z_TE``
    cannot explain" — and a ``false``-lane number below its flux counterpart
    falsifies it. Reported with both lanes' numbers in the PR body.

    The reading §5.7 says to rule out first is checked here: the oracle's
    ``FC_TE10_HZ`` and the extractor's ``fc_port_hz`` must stay 2.08 MHz apart and
    must not converge."""
    airy_false = fx["referee"]["slab_airy"]["fine|false"]
    airy_flux = fx["referee"]["slab_airy"]["fine|flux"]
    assert 0.00903 <= airy_false["max_mag_abs_diff"] <= 0.020721189
    assert airy_false["max_mag_abs_diff"] == pytest.approx(0.0126195, rel=1e-4)
    assert airy_false["max_phase_diff_deg"] == pytest.approx(6.098643, rel=1e-5)
    assert airy_false["max_phase_diff_deg"] < 6.578452302377118, "below the interval's floor"
    assert airy_false["max_phase_diff_deg"] < airy_flux["max_phase_diff_deg"], (
        "below its own flux counterpart — the interval's lower-bound argument is falsified")
    assert airy_false["max_mag_abs_diff"] <= G.SLAB_AIRY_MAG_TOL
    assert airy_false["max_phase_diff_deg"] <= G.SLAB_AIRY_PHASE_TOL_DEG
    for frozen_v, new_v in ((0.009029080292211028, airy_flux["max_mag_abs_diff"]),
                            (6.578452302377118, airy_flux["max_phase_diff_deg"])):
        assert 0.8 * frozen_v <= new_v <= 1.2 * frozen_v, (frozen_v, new_v)

    slab = fx["physics_gates"]["slab|fine|false"]
    assert 1.000014 <= slab["column_power_max"] <= 1.0009748557
    assert 7.7e-7 <= slab["reciprocity_mag_mean"] <= 2.5852644e-3
    assert 3.28e-4 <= slab["reciprocity_complex_max"] <= 6.9831665e-3
    assert slab["reciprocity_complex_max"] == pytest.approx(4.8064e-3, rel=1e-3)
    assert slab["reciprocity_complex_max"] <= G.RECIPROCITY_COMPLEX_MAX

    pec = fx["referee"]["pec_short"]["fine|false"]
    assert 0.99 <= pec["min_s11"] and pec["max_s11"] <= 1.03
    assert pec["max_s11"] == pytest.approx(1.0001433, rel=1e-6)

    plane = fx["plane_shift"]["slab|false"]
    assert plane["fc_predeclared_hz"] != plane["fc_port_hz"], (
        "the Airy oracle and the extractor must not share an input")


def test_flux_lane_column_power_is_below_its_predicted_band(fx):
    """§5.7's SECOND branch, "A flux-lane number outside its ±20 % band while the
    ``false`` lane behaves — its own reading, because the flux lane is the control
    for 'this is a ``Z_TE`` effect'. That lane carries no ``Z_TE`` term, so a move
    there is a move in something the correction was not supposed to touch. Report
    both lanes' per-bin residual curves side by side before attributing anything on
    the ``false`` lane to ``Z_TE``; not blocking on its own, and not absorbed into
    the branch above."

    §5.7's table gives the flux column as "a ±20 % band around its own frozen
    value". Read that way at the claims rung:

    * slab column power — excess 9.2485e-6 against the frozen 1.3651e-5, 0.68×,
      BELOW the printed band 1.00001 … 1.00002. The branch fires on this row.
    * slab complex reciprocity — 2.7303e-5 against 3.2772e-4, 0.083×. The table
      prints this row's flux prediction one-sided ("≤ 4e-4") and 2.73e-5 is under
      it, so it is inside the table and far outside the prose band; both readings
      are recorded rather than the convenient one.
    * slab magnitude reciprocity — 6.0224e-7 against 7.7072e-7, 0.78×, marginally
      under a ±20 % band and inside the printed "≤ 1e-6".
    * both Airy rows are inside on either reading (0.99996×, 0.935×).

    The per-bin curves the branch asks for say what the misses are: on the flux
    lane the column-power residual is 1e-7 … 9e-6 and changes sign bin to bin,
    and complex reciprocity is 1.0e-6 … 2.7e-5, also unsigned — a float32 noise
    floor, where a ±20 % envelope is not a meaningful prediction. The ``false``
    lane over the same bins is 5.3e-5 … 3.6e-4 and 1.4e-3 … 4.8e-3, smooth and
    single-signed. So nothing on the ``false`` lane is attributed to ``Z_TE`` on
    the strength of a flux-lane ratio, which is exactly what the branch asks."""
    flux = fx["physics_gates"]["slab|fine|flux"]
    false = fx["physics_gates"]["slab|fine|false"]
    frozen = json.loads(FROZEN.read_text())["physics_gates"]["slab|fine|flux"]

    excess = flux["column_power_max"] - 1.0
    frozen_excess = frozen["column_power_max"] - 1.0
    assert excess == pytest.approx(9.248524e-6, rel=1e-5)
    assert excess / frozen_excess == pytest.approx(0.6775, abs=5e-4)
    assert excess < 0.8 * frozen_excess, "below its ±20 % band — the branch that fired"
    assert flux["column_power_max"] < 1.0000109, "below the printed band 1.00001 … 1.00002"

    assert flux["reciprocity_complex_max"] == pytest.approx(2.730282e-5, rel=1e-5)
    assert flux["reciprocity_complex_max"] < 0.8 * frozen["reciprocity_complex_max"]
    assert flux["reciprocity_complex_max"] <= 4e-4, "inside the table's printed '≤ 4e-4'"
    assert flux["reciprocity_mag_mean"] == pytest.approx(6.022391e-7, rel=1e-5)
    assert flux["reciprocity_mag_mean"] <= 1e-6, "inside the table's printed '≤ 1e-6'"

    # the false lane behaves — which is the condition on this branch
    assert 1.000014 <= false["column_power_max"] <= 1.0009748557
    assert 3.28e-4 <= false["reciprocity_complex_max"] <= 6.9831665e-3

    # both lanes' per-bin residual curves, which the branch asks to see before
    # anything on the false lane is attributed to Z_TE
    n_bins = len(fx["fixture"]["freqs_hz"])
    for lane, lo, hi, signed in (("flux", 1e-7, 1e-5, False), ("false", 5e-5, 4e-4, True)):
        c = _cell("slab", "fine", lane)
        cols = c["column_power_per_bin"]
        per_bin = [max(cols[0][i], cols[1][i]) - 1.0 for i in range(n_bins)]
        assert len(per_bin) == n_bins
        assert lo <= max(abs(v) for v in per_bin) <= hi, (lane, per_bin)
        # the flux lane changes sign across the band; the false lane does too, but
        # smoothly and with an order more amplitude — it is a residual, not noise
        assert any(v > 0 for v in per_bin) and any(v < 0 for v in per_bin)
        recip = c["reciprocity_complex_per_bin"]
        assert len(recip) == n_bins
        if signed:
            assert min(recip) >= 1.4e-3 and max(recip) <= 4.9e-3
        else:
            assert min(recip) >= 1.0e-6 and max(recip) <= 2.8e-5


def test_non_vacuity_branch_one(fx):
    """§5.10's ``non_vacuity`` branch 1, "All 12 pass with ``slab`` inside ±5 % —
    as predicted; every other verdict on those cells is reading a real
    measurement". The slab moved +1.85 % / +0.74 % / +0.45 % on ``false`` and by
    under 0.01 % on ``flux``; ``pec_short`` is ``√column_power`` to seven digits and
    is read by §5.5, not here."""
    frozen = json.loads(FROZEN.read_text())
    for c in fx["cells"]:
        if c["dut"] == "thru":
            continue
        assert c["non_vacuity_max_s11"] > 0.20, (c["dut"], c["rung"], c["lane"])
        if c["dut"] != "slab":
            continue
        oc = [x for x in frozen["cells"]
              if (x["dut"], x["rung"], x["lane"]) == (c["dut"], c["rung"], c["lane"])][0]
        rel = abs(c["non_vacuity_max_s11"] - oc["non_vacuity_max_s11"]) / oc["non_vacuity_max_s11"]
        assert rel <= 0.05, (c["rung"], c["lane"], rel)


def test_power_closure_tracks_column_power(fx):
    """§5.10's ``power_closure`` branch 1, "Every cell tracking its own
    ``column_power_max`` within a factor 2 — as predicted, and it adds NO
    independent information: it is the same sum, so it is not a second witness for
    §5.4 or §5.5 and is not quoted as one". Worst ratio 1.99
    (``pec_short|fine|false``, the cell that already showed a two-sided error in
    miniature in run 1).

    Recorded because the number is in the artifact and no §5.10 leg reads it: the
    family is 12 verdicts and excludes ``thru`` by construction, and two ``thru``
    ``flux`` cells DO diverge from their own ``column_power_max`` by more than a
    factor 2 — ``thru|coarse|flux`` at 2.132 (frozen 2.124) and ``thru|mid|flux``
    at 2.715, which was 1.000 in run 1. A divergence means some bin lost power
    while another gained it; on these two the per-bin curve is a float32 noise
    floor at the 1e-6 … 5e-5 level, an order below the ``false``-lane residual.
    Not a verdict, not a witness for §5.4 — a line in the record."""
    unscored = {("thru", "coarse", "flux"): 2.132, ("thru", "mid", "flux"): 2.715}
    for (dut, rung, lane), ratio in unscored.items():
        c = _cell(dut, rung, lane)
        assert c["power_closure_max"] / (c["column_power_max"] - 1.0) == pytest.approx(
            ratio, abs=5e-3)
        assert f"power_closure|{dut}|{rung}|{lane}" not in fx["verdicts"]
    for c in fx["cells"]:
        if c["dut"] == "thru":
            continue
        excess = c["column_power_max"] - 1.0
        assert 0.5 <= c["power_closure_max"] / excess <= 2.0, (
            c["dut"], c["rung"], c["lane"], c["power_closure_max"], excess)
        assert fx["verdicts"][f"power_closure|{c['dut']}|{c['rung']}|{c['lane']}"] == "report_only"


def test_reciprocity_advisory_fired_on_one_cell_not_two(fx):
    """§3's declared warning surface. PR #882 added a warn-only advisory at 0.011,
    and §3 predicted "run 2 is expected to carry two per-cell advisory ``warnings``
    entries that run 1 does not, on those two cells and no others" —
    ``slab|coarse|false`` and ``slab|mid|false``.

    It fired on ``slab|coarse|false`` alone. ``slab|mid|false`` improved from
    1.986e-2 to 1.0867e-2, which is 1.2 % UNDER the 0.011 threshold, so it no
    longer warns. That is not "exactly those two", not "a third cell", and not
    "neither" — it is outside all three declared branches, and the PR body records
    it as NON-CLOSING rather than picking the nearest one. The claims-rung
    ``slab|fine|false`` cell (4.807e-3) stays well under the advisory, so the
    advisory's derivation is not put back on the table."""
    fired = {f"{c['dut']}|{c['rung']}|{c['lane']}" for c in fx["cells"]
             for w in (c.get("warnings") or []) if "reciprocity ADVISORY" in w["message"]}
    assert fired == {"slab|coarse|false"}
    assert fx["physics_gates"]["slab|mid|false"]["reciprocity_complex_max"] == pytest.approx(
        1.0867e-2, rel=1e-3)
    assert fx["physics_gates"]["slab|mid|false"]["reciprocity_complex_max"] < 0.011
    assert fx["physics_gates"]["slab|fine|false"]["reciprocity_complex_max"] < 0.011


def test_preflight_findings_are_recorded_verbatim(fx):
    """The preflight set is part of the result and is expected verbatim from run 1
    (§3): a changed set would be a finding, not a fixture edit."""
    expected = {
        ("thru", "coarse"): [], ("pec_short", "coarse"): [],
        ("slab", "coarse"): ["lossless_q", "mesh_resolution", "mesh_resolution", "mesh_resolution"],
        ("thru", "mid"): [], ("pec_short", "mid"): ["mesh_resolution"],
        ("slab", "mid"): ["lossless_q", "mesh_resolution", "mesh_resolution", "mesh_resolution"],
        ("thru", "fine"): [], ("pec_short", "fine"): [],
        ("slab", "fine"): ["lossless_q"],
    }
    for c in fx["cells"]:
        codes = sorted(f["code"] for f in c["preflight"])
        assert codes == expected[(c["dut"], c["rung"])], (c["dut"], c["rung"], c["lane"], c["preflight"])
        for f in c["preflight"]:
            assert f["severity"] == "warning" and f["message"]


# ===========================================================================
# LIVE layer — moved on
# ===========================================================================
# The three live tests that read THIS artifact — the two
# ``test_live_cells_reproduce_the_fixture_*`` and
# ``test_live_plane_shift_rotation_coarse_rung`` — and the pin guard
# ``test_the_live_pin_is_the_committed_one`` now live in
# ``tests/oracle/test_waveguide_chain_battery_v18_close.py`` and read the v1.8
# closing artifact ``fixture_v18_close.json`` (VESSL 369367258638), whose 18 cells
# are bit-identical to this one's (max|ΔS| = 0 on every cell). This module stays
# the adjudication of run 2 as it was measured.

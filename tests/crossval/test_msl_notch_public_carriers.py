"""cv06b -- public-carrier / docstring number correctness (issue #723).

Tracked carriers quoted `06b_msl_notch_filter_uniform.py`'s dx=80um run as
current MSL evidence: `validation/README.md`, `docs/guides/sparameter_
support_matrix.md` + `.json` (the support contract),
`docs/agent/port-selection.mdx`, and `docs/public/guide/benchmarks.mdx`.
Nothing coupled them to the script, so the #723 mesh change (dx 80um ->
63.5um = h_sub/4) would have left all of them stale with CI green -- that is
exactly how it was found, by hand, in review.

This test couples them. It locks:

  1. the script's own mesh convention (`DX == H_SUB / 4`) -- if the mesh
     moves again, this reds first;
  2. the realized board that convention buys, measured live from
     `sim.fidelity_report()` (no time stepping): substrate exactly 254.0um,
     main trace and stub both 635.0um. The 635.0 figure is what the analytic
     reference and every carrier quote, so it must be measured, not asserted
     from a formula -- `round(W_TRACE/DX)*DX` gives 571.5um here and is
     wrong (half-open [lo, hi) node rasterization);
  3. the committed run log's headline numbers, parsed from the log rather
     than retyped;
  4. each carrier quotes the CURRENT numbers, and carries the superseded
     dx=80um ones only inside text that frames them as history.

Fail-closed: every regex assert requires a match, so a carrier rewording
that drops the number reds this test instead of passing quietly.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CV06B = REPO_ROOT / "validation/crossval/06b_msl_notch_filter_uniform.py"
RUN_LOG = (REPO_ROOT / "validation/crossval/_06b_notch_uniform_logs"
           / "20260827T131217Z_run.log")
RUN_LOG_DX80 = (REPO_ROOT / "validation/crossval/_06b_notch_uniform_logs"
                / "20260828T054132Z_dx80_origin_main_cdc38bc8_run.log")
README = REPO_ROOT / "validation/README.md"
MATRIX_MD = REPO_ROOT / "docs/guides/sparameter_support_matrix.md"
MATRIX_JSON = REPO_ROOT / "docs/guides/sparameter_support_matrix.json"
PORT_SELECTION = REPO_ROOT / "docs/agent/port-selection.mdx"
BENCHMARKS = REPO_ROOT / "docs/public/guide/benchmarks.mdx"

# The superseded dx=80um headline. Any carrier may mention these, but only
# in a sentence that marks them as the old mesh.
STALE_NUMBERS = ("1.63", "34.2", "57.9")


def _z0_median(log_text: str) -> float:
    """Parse ``Re(Z0) median = <x> Ω`` out of a committed cv06b run log."""
    m = re.search(r"Re\(Z0\) median\s*=\s*(-?[\d.]+)", log_text)
    assert m, "Re(Z0) median line missing from the run log"
    return float(m.group(1))


def _load_cv06b():
    spec = importlib.util.spec_from_file_location("_cv06b_carriers", CV06B)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv06b():
    return _load_cv06b()


@pytest.fixture(scope="module")
def log_text():
    return RUN_LOG.read_text(encoding="utf-8")


def test_mesh_convention_is_h_sub_over_four(cv06b):
    """dx = h_sub/4 is the whole point of #723 -- an aligned substrate."""
    assert cv06b.DX == pytest.approx(cv06b.H_SUB / 4.0, rel=1e-12)
    assert cv06b.DX == pytest.approx(63.5e-6, rel=1e-12)
    # h_sub/dx must be an exact integer: that is the alignment property, not
    # merely "a finer mesh".
    n = cv06b.H_SUB / cv06b.DX
    assert n == pytest.approx(round(n), abs=1e-9)


def test_realized_board_is_measured_not_assumed(cv06b):
    """Substrate 254.0um exactly, trace and stub both 635.0um -- read from
    fidelity_report on the real build, which is where the analytic reference
    gets its width at runtime."""
    sim = cv06b._build_sim()
    report = sim.fidelity_report(print_report=False)
    axes = {item["entity"]: {a["axis"]: a for a in item["axes"]}
            for item in report if "axes" in item}

    sub_z = axes["geometry[0] 'ro4350b'"]["z"]
    assert sub_z["realized_extent_um"] == pytest.approx(254.0, abs=1e-6)
    assert max(sub_z["face_residual_um"]) == pytest.approx(0.0, abs=1e-9)

    trace_y = axes["geometry[1] 'pec'"]["y"]
    stub_x = axes["geometry[2] 'pec'"]["x"]
    assert trace_y["realized_extent_um"] == pytest.approx(635.0, abs=1e-6)
    assert stub_x["realized_extent_um"] == pytest.approx(635.0, abs=1e-6)

    # The live reader must agree with the report it reads.
    assert cv06b._realized_trace_width(sim) == pytest.approx(635.0e-6, rel=1e-12)

    # And it must NOT be reproducible by the round() formula -- if a future
    # edit "simplifies" _realized_trace_width into arithmetic, this reds.
    naive = round(cv06b.W_TRACE / cv06b.DX) * cv06b.DX
    assert naive == pytest.approx(571.5e-6, rel=1e-9)
    assert naive != pytest.approx(635.0e-6, rel=1e-6)


def test_z0_anchor_is_the_design_board_not_a_realized_one(cv06b):
    """#723 review, BLOCKING 1. What the fix buys is realized == declared on
    z, so the measured Z0 can be compared to the DESIGN's Hammerstad-Jensen.
    Evaluating HJ on each mesh's own realized board instead makes the
    dx=80um mesh look just as good (its realized 560/320um board has
    HJ = 57.46 ohm against a measured ~57.9), so that framing must not be
    used to claim a port-accuracy improvement.
    """
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

    z0_design, _ = hammerstad_jensen_z0_eps_eff(600e-6, 254e-6, cv06b.EPS_R)
    z0_realized_63, _ = hammerstad_jensen_z0_eps_eff(635e-6, 254e-6, cv06b.EPS_R)
    z0_realized_80, _ = hammerstad_jensen_z0_eps_eff(560e-6, 320e-6, cv06b.EPS_R)

    assert z0_design == pytest.approx(47.90, abs=0.02)
    assert z0_realized_63 == pytest.approx(46.18, abs=0.02)
    assert z0_realized_80 == pytest.approx(57.46, abs=0.02)

    # The measured medians, PARSED from the two committed logs rather than
    # retyped: the post-fix GPU run and the dx=80um re-measurement taken
    # live on origin/main (cdc38bc8) for the #723 review.
    z0_meas_63 = _z0_median(RUN_LOG.read_text(encoding="utf-8"))
    z0_meas_80 = _z0_median(RUN_LOG_DX80.read_text(encoding="utf-8"))
    assert z0_meas_63 == pytest.approx(46.5, abs=0.05)
    assert z0_meas_80 == pytest.approx(57.9, abs=0.05)
    dev_realized_63 = abs(z0_meas_63 - z0_realized_63) / z0_realized_63 * 100
    dev_realized_80 = abs(z0_meas_80 - z0_realized_80) / z0_realized_80 * 100
    # Both under 1% -- i.e. the realized-board comparison does not separate
    # the two meshes, which is why no "Nx bias reduction" may be claimed.
    assert dev_realized_63 < 1.0
    assert dev_realized_80 < 1.0
    assert dev_realized_80 / dev_realized_63 < 3.0

    # What IS true: on the design board the aligned mesh lands within 3%.
    dev_design_63 = abs(z0_meas_63 - z0_design) / z0_design * 100
    assert dev_design_63 == pytest.approx(2.9, abs=0.15)


def test_committed_log_reports_the_numbers_the_carriers_quote(log_text):
    """Parse the log; do not retype it."""
    def grab(label: str) -> float:
        m = re.search(rf"{label}\s*=\s*(-?[\d.]+)", log_text)
        assert m, f"{label!r} missing from {RUN_LOG.name}"
        return float(m.group(1))

    assert grab("Notch frequency error") == pytest.approx(1.40, abs=0.005)
    assert grab(r"Notch depth \|S21\|") == pytest.approx(-43.3, abs=0.05)
    assert grab(r"Re\(Z0\) median") == pytest.approx(46.5, abs=0.05)
    assert "W_realized=635.0µm" in log_text
    assert "mesh: dx=63.5µm, n_z_sub=4" in log_text
    assert "PASS: cv06b" in log_text


def test_committed_log_carries_the_warnings_the_contract_quotes(log_text):
    """R5: the headline is read inside a flagged band and after a passivity
    projection. Both must stay quotable from the same log."""
    assert "standing-wave null at the port plane: 9 bins in [3.6273, 7.0000] GHz" in log_text
    assert "63 of 100 frequency bins were non-passive as extracted" in log_text
    assert "worst sigma_max = 1.006" in log_text
    assert "'msl_0' = 61.02 ohm" in log_text
    assert "'msl_1' = 39.90 ohm" in log_text


@pytest.mark.parametrize(
    "carrier", [README, MATRIX_MD, MATRIX_JSON, PORT_SELECTION, BENCHMARKS])
def test_carrier_quotes_the_current_mesh(carrier):
    text = carrier.read_text(encoding="utf-8")
    assert re.search(r"63\.5\s*(µm|um)", text), (
        f"{carrier.relative_to(REPO_ROOT)} does not name the shipped "
        "dx=63.5um mesh")


@pytest.mark.parametrize("carrier", [MATRIX_MD, MATRIX_JSON, PORT_SELECTION])
def test_carrier_quotes_the_current_headline(carrier):
    text = carrier.read_text(encoding="utf-8")
    for number in ("1.40", "43.3", "46.5"):
        assert number in text, (
            f"{carrier.relative_to(REPO_ROOT)} is missing the current cv06b "
            f"figure {number}")


@pytest.mark.parametrize(
    "carrier", [README, MATRIX_MD, MATRIX_JSON, PORT_SELECTION, BENCHMARKS])
def test_carrier_does_not_present_the_dx80_numbers_as_current(carrier):
    """The stale figures may appear, but only in a sentence that marks them
    as the superseded mesh."""
    text = carrier.read_text(encoding="utf-8")
    history_markers = ("history", "historical", "earlier", "superseded",
                       "through 2026-08", "was dx", "did not solve")
    for number in STALE_NUMBERS:
        for m in re.finditer(re.escape(number), text):
            window = text[max(0, m.start() - 700):m.end() + 700].lower()
            assert any(k in window for k in history_markers), (
                f"{carrier.relative_to(REPO_ROOT)} quotes the superseded "
                f"dx=80um figure {number} without marking it as history")


def test_e4_board_mismatch_is_disclosed_where_the_comparison_is_cited():
    """Post-#723 cv06b solves h_sub=254um and the E4/openEMS leg still solves
    300um. Every carrier that cites that comparison must say so."""
    for carrier in (README, MATRIX_MD, MATRIX_JSON, BENCHMARKS):
        text = carrier.read_text(encoding="utf-8")
        if "msl_notch_e4" in text or "dx=50 um" in text or "dx=50um" in text:
            assert "300" in text, (
                f"{carrier.relative_to(REPO_ROOT)} cites the dx=50um E4 "
                "comparison without disclosing its 300um board")

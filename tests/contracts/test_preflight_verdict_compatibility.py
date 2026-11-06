"""Existing diagnostic codes and order survive mandatory realization admission."""
import json

from tests.locks import test_preflight_split_snapshot as snapshots


def render(name):
    fixture = next(row for row in snapshots._FIXTURES if row[0] == name)
    return json.loads(snapshots._render(*fixture))["preflight"]["issues"]


def test_outside_probe_keeps_absorber_code():
    issues = render("guards_probe_in_cpml")
    assert all(issue["code"] != "uncoded" for issue in issues)
    assert [(issue["code"], issue["severity"]) for issue in issues
            if issue["code"] == "absorber_overlap"] == [('absorber_overlap', 'warning'), ('absorber_overlap', 'warning')]


def test_aperture_keeps_its_code():
    issues = render("conformal_fine_dx")
    assert any(issue["code"] == "port_aperture_snap" for issue in issues)
    assert not any(issue["code"] == "realization" and
                   str(issue["loc"]).startswith("waveguide_port[") for issue in issues)


def test_retained_sheet_findings_keep_the_original_order():
    actual = render("congruence_on_lattice")
    expected = json.loads((snapshots._SNAPSHOT_DIR / "congruence_on_lattice.json").read_text())
    assert [issue["code"] for issue in actual] == [
        issue["code"] for issue in expected["preflight"]["issues"]]

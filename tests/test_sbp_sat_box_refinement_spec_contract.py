"""Regression locks for the Milestone 5 all-PEC box-refinement spec."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOX_SPEC = ROOT / "docs/guides/sbp_sat_all_pec_box_refinement_spec.md"
FULL_SPEC = ROOT / "docs/guides/sbp_sat_zslab_phase1_full_spec.md"


EXPECTED_FACE_ROWS = {
    "x_lo": ("-x̂", "(y, z)", "(Ey, Ez)", "(Hy, Hz)"),
    "x_hi": ("+x̂", "(y, z)", "(Ey, Ez)", "(Hy, Hz)"),
    "y_lo": ("-ŷ", "(x, z)", "(Ex, Ez)", "(Hx, Hz)"),
    "y_hi": ("+ŷ", "(x, z)", "(Ex, Ez)", "(Hx, Hz)"),
    "z_lo": ("-ẑ", "(x, y)", "(Ex, Ey)", "(Hx, Hy)"),
    "z_hi": ("+ẑ", "(x, y)", "(Ex, Ey)", "(Hx, Hy)"),
}


def _text(path: Path) -> str:
    return path.read_text()


def test_box_refinement_spec_has_required_sections():
    text = _text(BOX_SPEC)

    assert "# SBP-SAT all-PEC box refinement specification" in text
    for heading in (
        "## Status",
        "## General face-operator contract",
        "## Six-face component table",
        "## Edge and corner policy",
        "## Implementation plan",
        "## Benchmark matrix",
        "## Implementation gate",
    ):
        assert heading in text


def test_box_refinement_spec_records_all_six_face_mappings():
    text = _text(BOX_SPEC)

    for face, parts in EXPECTED_FACE_ROWS.items():
        assert face in text
        for part in parts:
            assert part in text


def test_box_refinement_spec_blocks_runtime_until_edge_corner_and_benchmarks_exist():
    text = _text(BOX_SPEC)

    assert "Each face SAT operator owns only the strict face interior" in text
    assert "Each of the 12 box edges is owned by a dedicated 1-D edge operator" in text
    assert "Each of the 8 box corners is owned by a dedicated corner operator" in text
    assert "arbitrary 6-face box" in text
    assert "refinement stays disabled" in text
    assert "Implementation remains blocked until all of the following are true" in text
    assert "the benchmark matrix above is implemented and passing" in text


def test_box_refinement_spec_locks_operator_contract_details():
    text = _text(BOX_SPEC)

    for token in (
        "`face`: one of `x_lo`, `x_hi`, `y_lo`, `y_hi`, `z_lo`, `z_hi`",
        "`normal_axis`: `x`, `y`, or `z`",
        "`normal_sign`: `-1` for `*_lo`, `+1` for `*_hi`",
        "`tangential_axes`: ordered pair `(t1, t2)`",
        "`tangential_e_components`: ordered pair `(E_t1, E_t2)`",
        "`tangential_h_components`: ordered pair `(H_t1, H_t2)`",
        "`coarse_shape`: `(n_t1, n_t2)` for that face",
        "`fine_shape`: `(n_t1 * ratio, n_t2 * ratio)`",
        "`coarse_slice(...)`: coarse-grid extractor for the face",
        "`fine_slice(...)`: fine-grid extractor for the face",
        "`ops`: the 2-D prolongation/restriction/norm bundle for that orientation",
        "define `R_t = P_t^T / ratio`",
        "define the 2-D face restriction as `R_face = R_t1 @ face @ R_t2.T`",
        "define the 2-D face prolongation as `P_face = P_t1 @ face @ P_t2.T`",
        "same `alpha_c = tau / (ratio + 1)`",
        "same `alpha_f = tau * ratio / (ratio + 1)`",
        "do **not** encode face-sign-dependent permutations in the trace arrays",
    ):
        assert token in text


def test_box_refinement_spec_locks_phase5a_to_phase5f_plan():
    text = _text(BOX_SPEC)

    for phase in (
        "### Phase 5A — orientation-general operator layer",
        "### Phase 5B — x/y/z face extraction and scatter",
        "### Phase 5C — face-interior SAT only",
        "### Phase 5D — edge operators",
        "### Phase 5E — corner resolution",
        "### Phase 5F — runtime enablement",
    ):
        assert phase in text

    for detail in (
        "Generalize the current z-face operator bundle into a 2-D face bundle",
        "Add explicit extraction/scatter helpers for all six faces",
        "Generalize SAT coupling to x-, y-, and z-oriented **face interiors**",
        "Introduce dedicated 1-D edge operators for the 12 coarse/fine edge pairs",
        "Introduce a dedicated corner rule for the 8 coarse/fine corners",
        "Only after Phases 5A-5E land may the runtime/API accept an arbitrary box",
    ):
        assert detail in text


def test_box_refinement_spec_locks_benchmark_matrix_details():
    text = _text(BOX_SPEC)

    for fixture in (
        "x-face oblique proxy benchmark",
        "y-face oblique proxy benchmark",
        "z-face regression benchmark",
        "edge stress benchmark",
        "corner stress benchmark",
    ):
        assert fixture in text

    for metric in (
        "point-probe DFT amplitude error",
        "point-probe DFT phase error",
        "at least one probe near each newly activated face / edge / corner region",
        "coarse-interior overlap sanity checks",
        "amplitude error `<= 5%`",
        "phase error `<= 5°`",
    ):
        assert metric in text


def test_full_spec_references_milestone5_artifact_and_contract_test():
    text = _text(FULL_SPEC)

    assert "docs/guides/sbp_sat_all_pec_box_refinement_spec.md" in text
    assert "tests/test_sbp_sat_box_refinement_spec_contract.py" in text

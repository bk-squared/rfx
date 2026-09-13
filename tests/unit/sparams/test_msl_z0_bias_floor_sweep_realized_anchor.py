"""Preserve the historical #752 record without treating it as a current oracle.

The source measurements and original realized-anchor JSON remain frozen.
Live fidelity-report checks measure geometric material/body extent drift,
not an RF conductor gap or current Z0 accuracy. The recorded 0.4%/0.25%
summaries below describe only the archived table. CLI generation is retired;
explicit archive inspection verifies hashes and never rebuilds geometry,
re-evaluates the model, evolves fields or overwrites these records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SOURCE_JSON = (
    REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"
    / "msl_z0_bias_floor_sweep.json"
)
ANCHOR_JSON = (
    REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"
    / "msl_z0_bias_floor_sweep_realized_anchor.json"
)
_SOURCE_SHA256 = (
    "f56f6b17691613d8782c1d5ce1241c1cd9bc10ef61715b203ed5cd6d4ab18362"
)

# Realized trace width, in um, that the lattice ownership contract's
# centre-sampled PEC VOLUME (#931 §1.1; ``fidelity_report`` reads that
# sampler since 20a8ccfc) produces at the three sweep points where it
# differs from the committed (pre-#802 f32) artifact. Re-derived live by
# ``realized_h_w_um()`` in both tests below -- these are the CURRENT
# expectation, so a future rasterizer move reds them; the committed
# artifact keeps the pre-#802 values as its as-solved record.
#
# Arithmetic (W = 600 um; a volume face rounds to the NEAREST cell plane;
# the trace's lower face is on a node at every aligned point):
#   h_sub/3  W/dx =  7.087 ->  7 cells  592.667  (committed 677.333: the f32
#                                                 node sampler took the hi face)
#   h_sub/4  W/dx =  9.449 ->  9 cells  571.5    (committed 635.0 = 10 nodes)
#   h_sub/5  W/dx = 11.811 -> 12 cells  609.6    = committed (558.8 under the
#                                                 #802/#834 node sampler)
#   h_sub/6  W/dx = 14.173 -> 14 cells  592.667  = committed (635.0 under it)
#   80 um    faces at 14.35 / 21.85 dx -> planes 14 / 22, 8 cells, 640.0
#                                                (committed 560.0 = 7 nodes)
#   60 um    faces at 16.467 / 26.467 dx -> 16 / 26, 10 cells, 600.0 = committed
# History: the #802/#834 exact-coordinate NODE sampler moved h_sub/3, /5
# and /6 (592.667 / 558.8 / 635.0) and left h_sub/4 and the misaligned
# pair alone; the centre sampler moves h_sub/3, /4 and 80 um instead and
# lands back on the committed widths at h_sub/5, /6 and 60 um. Measured
# 2026-09-07, fidelity_report(), no solve, JAX_PLATFORMS=cpu; h_sub is a
# dielectric (node-sampled, §1.1 unchanged) and reads the committed
# 254 / 254 / 254 / 254 / 320 / 300 um at every point.
_LIVE_W_UM = {
    "aligned h_sub/3": 592.667,
    "aligned h_sub/4": 571.5,
    "misaligned 80um": 640.0,
}


def test_pre_declared_sweep_json_is_frozen():
    """The pre-declared JSON and its as-run verdict block must never be
    edited -- they are auditable BECAUSE their criteria predate the
    data. This guards against exactly the mistake issue #752 warns
    against: fixing the reading of the data by rewriting the data."""
    digest = hashlib.sha256(SOURCE_JSON.read_bytes()).hexdigest()
    assert digest == _SOURCE_SHA256, (
        "scripts/diagnostics/msl_z0_bias_floor_sweep/"
        "msl_z0_bias_floor_sweep.json changed -- this file and its "
        "as-run verdict block must stay untouched; add corrections "
        "alongside it (see msl_z0_bias_floor_sweep_realized_anchor.py), "
        "never by editing it"
    )


def test_realized_anchor_json_exists_and_cites_source():
    assert ANCHOR_JSON.exists(), (
        f"{ANCHOR_JSON} missing -- restore the frozen historical record from git; "
        "current-geometry regeneration is retired"
    )
    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    assert out["source_json_sha256"] == _SOURCE_SHA256, (
        "the sibling artifact's recorded source sha256 no longer matches "
        "the frozen pre-declared JSON -- restore the recorded artifact"
    )
    assert len(out["rows"]) == 6


def test_realized_anchor_matches_fidelity_report_directly():
    """Re-derive h_sub/W_trace realized extents from fidelity_report()
    (no FDTD solve -- fast) for each of the six pre-declared dx points
    and check they agree with the committed sibling artifact. This is
    the "sweep's realized-board column agrees with fidelity_report to a
    stated tolerance" regression test.

    #766 review N3, re-adjudicated at #802/#834: the committed artifact
    declares its own precision --
    ``"jax_enable_x64=False, inferred from Z0 agreement with the
    realized-board Hammerstad-Jensen anchor (max |dev| = 0.377% over all
    six points, vs 8.6/5.6/4.4% at the alternative (x64) rasterization of
    the aligned class's trace width)"``. This test used to SKIP under
    JAX_ENABLE_X64=1 because the aligned points' trace width then
    rasterized differently (h_sub/3 read W = 592.667µm where the artifact
    has 677.333µm). #834 made realized geometry flag-independent, so that
    justification is gone and the skip is removed: the live re-derivation
    lands on the module-level expectations under BOTH flags (measured
    2026-09-01: this file passes under x64=0 and x64=1). The committed
    artifact itself stays the pre-#802 as-solved record, exactly as the
    re-pin note below explains.

    #931 (2026-09-07): the PEC trace is a one-cell VOLUME, so its width is
    now read from cell CENTRES (§1.1); ``_LIVE_W_UM`` carries the three
    points where that differs from the committed artifact (h_sub/3,
    h_sub/4, 80 um) and the arithmetic. The other three read the committed
    width again. h_sub (a dielectric) is unchanged at every point.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "msl_z0_bias_floor_sweep_realized_anchor",
        REPO / "scripts" / "diagnostics"
        / "msl_z0_bias_floor_sweep_realized_anchor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in out["rows"]}

    # Re-pinned at the exact-coordinate fix (#802) and again at #931
    # §1.1 (centre-sampled PEC volume). The committed artifact stays
    # untouched — it is the as-solved record of the frozen sweep (its Z0
    # numbers were measured on the pre-#802 f32 rasterization, and its
    # 0.4% bound test reads it as such). The live re-derivation therefore
    # matches the committed rows where the realization lands on the same
    # width (h_sub/5, h_sub/6, 60 um) and ``_LIVE_W_UM`` where it does not.
    # Refreshing the artifact itself requires RE-SOLVING the sweep on the
    # new rasterization — a re-measurement lane, not a value edit here.
    for label, dx in mod.DX_GRID:
        h_real_um, w_real_um = mod.realized_h_w_um(dx)
        committed = by_label[label]
        assert h_real_um == pytest.approx(
            committed["h_sub_realized_um"], abs=0.01
        ), f"{label}: h_sub realized drifted vs committed artifact"
        expected_w = _LIVE_W_UM.get(
            label, committed["w_trace_realized_um"])
        assert w_real_um == pytest.approx(expected_w, abs=0.01), (
            f"{label}: W_trace realized drifted vs the live (#931 §1.1) "
            "expectation (committed artifact = pre-#802 as-solved record)")


def test_realized_board_deviation_tolerances_hold():
    """STALE-ARTIFACT REGRESSION LOCK on the FROZEN committed JSON — NOT a
    live-extractor bound (audit 2026-09-02, finding A1).

    This asserts max|dev_vs_realized_board_pct| over the committed rows
    (0.377% all six, 0.197% misaligned pair). It reads the frozen column
    out of the artifact; it does NOT re-solve, so it does NOT measure the
    live extractor. That is fine as a lock that the committed file has not
    silently drifted, but it MUST NOT be read as proof the extractor still
    tracks within 0.4% on main:

      The artifact's rows were solved on the pre-#802 f32 rasterization.
      This tree (#931 §1.1 centre-sampled PEC volume, after #802/#834)
      rasterizes three of the six sweep points to DIFFERENT trace widths
      (h_sub/3 677.3->592.7um, h_sub/4 635.0->571.5um, 80um 560->640um
      -- see ``_LIVE_W_UM`` above, re-derived live; the node sampler had
      moved h_sub/3, /5 and /6 instead), so both the measured Z0 and the
      realized-board HJ anchor on those points move, and this column is
      a frozen PRE-#802 record. A LIVE bound requires
      RE-SOLVING the sweep on main's rasterization:
      ``scripts/diagnostics/msl_z0_bias_floor_sweep.py`` (6 FDTD points),
      then regenerating the anchor with
      ``msl_z0_bias_floor_sweep_realized_anchor.py``. That re-solve is
      OWED; the preflight advisories were corrected (audit A1) to make
      only the QUALITATIVE realized-board claim until it is done.

      The misaligned pair (dx=80,60um) did NOT move at #802 (W unchanged).
      Audit second pass (finding F3): unchanged geometry alone does not
      make a row live -- the MSL extractor lane changed after this sweep
      ran (#698, #771, #791, #798) -- so one of the two was RE-SOLVED
      instead of argued. dx=80um on the post-#802 tree (2026-09-02,
      jax_enable_x64=False, CPU, 149 s, ring-down settling -100.3/-101.0
      dB, mean|S11|raw 0.11607 vs the frozen 0.11609) reads Z0=57.572 ohm
      vs the frozen 57.576, i.e. +0.190% against HJ(560um,320um)=57.463
      where the frozen row records +0.197%. That row WAS
      live-representative, on evidence, on the node-sampled tree.
      dx=60um is untested and the aligned rows remain re-solve-owed.

      #931 (2026-09-07): the centre-sampled PEC volume (§1.1) realizes
      the dx=80um trace as 8 cells = 640um, not the 7 cells = 560um that
      re-solve ran on, so the 80um row is no longer live-representative
      either. The rows whose live width differs from the committed
      artifact are now h_sub/3, h_sub/4 and 80um (``_LIVE_W_UM``);
      h_sub/5, h_sub/6 and 60um realize the committed width again. The
      re-solve owed is the whole sweep.
    """
    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    rows = out["rows"]
    devs_all = [abs(r["dev_vs_realized_board_pct"]) for r in rows]
    devs_misaligned = [
        abs(r["dev_vs_realized_board_pct"])
        for r in rows if r["label"].startswith("misaligned")
    ]
    assert len(devs_misaligned) == 2
    max_all = max(devs_all)
    max_misaligned = max(devs_misaligned)
    # NOTE (audit 2026-09-02, finding F4): these two messages used to say
    # the numbers are what "the preflight advisory cites". They are not --
    # audit A1 retired every specific percentage from the advisories. They
    # are the maxima the FROZEN artifact itself records.
    assert max_all <= 0.4, (
        f"max|dev_vs_realized_board_pct| over all six points = {max_all}%, "
        "exceeds the 0.4% maximum recorded in the frozen artifact (a "
        "pre-#802 as-solved record, not a live extractor bound)"
    )
    assert max_misaligned <= 0.25, (
        f"max|dev_vs_realized_board_pct| over the misaligned pair = "
        f"{max_misaligned}%, exceeds the 0.25% maximum recorded in the "
        "frozen artifact"
    )
    # And the aggregate fields the artifact itself reports must match.
    assert out["max_abs_dev_vs_realized_board_pct_all_six"] == pytest.approx(
        max_all, abs=1e-9)
    assert out[
        "max_abs_dev_vs_realized_board_pct_misaligned_pair"
    ] == pytest.approx(max_misaligned, abs=1e-9)


def test_declared_board_column_unchanged_from_source():
    """z0_measured_ohm and z0_hj_declared_board_ohm in the sibling
    artifact must be copied VERBATIM from the pre-declared JSON, never
    re-solved or retyped by hand."""
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    src_by_label = {r["label"]: r for r in source["rows"]}
    for row in out["rows"]:
        src = src_by_label[row["label"]]
        assert row["z0_measured_ohm"] == src["z0_measured_ohm"]
        assert row["z0_hj_declared_board_ohm"] == src["z0_hj_ohm"]


def test_committed_aligned_rows_are_a_pre802_record_not_live():
    """Audit 2026-09-02 (finding A1) -- demonstrates WHY the tolerance
    lock above is a frozen record, not a live bound. Re-derive each
    point's realized trace width live (fidelity_report(), no FDTD solve)
    and confirm three aligned rows in the committed artifact no longer
    match main's rasterization. This is the staleness the corrected
    preflight advisories now disclose instead of the old absolute 0.4%
    claim; it is documentation of a known post-#802 drift, so it passes
    on any tree that carries #802/#834.

    AUDIT 2026-09-02 (finding F2, second pass): this test used to assert
    only ``abs(w_live - w_committed) > 1.0``, which a THIRD width would
    also satisfy -- so it did not actually detect the "moves AGAIN"
    failure its docstring named. It now pins the CURRENT realized width
    (``_LIVE_W_UM``, the same module-level expectation the older
    fidelity_report test uses) as well, so the test reds two ways:
      * live W != this tree's current rasterization -> a rasterizer moved
        the board again; the advisories' staleness disclosure and the
        artifact both need refreshing;
      * live W == the committed pre-#802 value  -> either the rasterizer
        reverted or the sweep was re-solved and the artifact refreshed, in
        which case a live "within X%" bound may be restored.

    #931 (2026-09-07) showed the second branch has a third cause the
    dichotomy above did not name: a rasterizer move that lands on the
    committed width at SOME points. The centre-sampled volume (§1.1)
    reads h_sub/5 and h_sub/6 at their committed widths again while
    moving h_sub/3, h_sub/4 and 80um, so the moved set is
    ``_LIVE_W_UM``'s keys — one aligned row fewer, one misaligned row
    more — and the name of this test now understates it: the misaligned
    80um row is a stale record too.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "msl_z0_bias_floor_sweep_realized_anchor",
        REPO / "scripts" / "diagnostics"
        / "msl_z0_bias_floor_sweep_realized_anchor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in out["rows"]}
    dx_by_label = dict(mod.DX_GRID)

    # Committed widths that this tree's rasterizer (#931 §1.1 centre
    # sampling, after #802/#834) has moved -- proving the committed
    # realized-board deviation column is stale for these points.
    assert set(_LIVE_W_UM) == {
        "aligned h_sub/3", "aligned h_sub/4", "misaligned 80um"}
    for label, w_expected_um in _LIVE_W_UM.items():
        _, w_live_um = mod.realized_h_w_um(dx_by_label[label])
        w_committed_um = by_label[label]["w_trace_realized_um"]
        # (a) live W is still main's CURRENT rasterization -- a third value
        # means the board moved again and every downstream disclosure is
        # stale (this is the assertion the first pass of this test lacked).
        assert w_live_um == pytest.approx(w_expected_um, abs=0.01), (
            f"{label}: live realized W {w_live_um}um is neither the "
            f"committed pre-#802 {w_committed_um}um nor the recorded "
            f"#931 §1.1 width {w_expected_um}um -- the rasterizer moved this "
            "point AGAIN; preserve the frozen records and update this geometry "
            "drift witness. A current accuracy bound needs a separately "
            "qualified measurement with matching geometry provenance"
        )
        # (b) and it still differs from the committed artifact, i.e. that
        # column really is a stale record.
        assert abs(w_live_um - w_committed_um) > 1.0, (
            f"{label}: live realized W {w_live_um}um now equals the "
            f"committed {w_committed_um}um -- reassess the geometry drift witness; "
            "agreement with an old width alone does not establish a live "
            "solver accuracy bound"
        )


def test_preflight_advisories_make_no_stale_absolute_realized_bound():
    """Anti-overclaim regression lock (audit 2026-09-02, finding A1).

    The MSL-port-geometry advisories in ``rfx/api/_preflight.py`` used to
    quote a specific "within 0.4% at every point" realized-board bound
    whose ONLY evidence was the committed anchor JSON -- stale for three
    aligned points post-#802 (see the test above). Until the sweep is
    RE-SOLVED on main, the advisories must state only the QUALITATIVE
    realized-board claim, not a specific unverified percentage. This locks
    the retired phrasings out of the module source.

    SCOPE (audit second pass): this lock covers the REALIZED-board 0.4%
    phrasings only. The DECLARED-board sequence retired in the same audit
    (-7.9%/-3.8%/-1.2%/+0.7%, ">5% expected") cannot be locked by a source
    grep, because the class docstring now QUOTES those strings while
    explaining why they were withdrawn. They are locked where it actually
    matters -- on the emitted message -- by
    ``test_substrate_resolution_warning_names_alignment_requirement`` in
    tests/unit/ports/test_msl_port_preflight.py, which asserts each is
    absent from the runtime check-2 warning."""
    src = (REPO / "rfx" / "api" / "_preflight.py").read_text(encoding="utf-8")
    for retired in (
        "to within 0.4% at every",
        "within 0.4% at EVERY",
        "within 0.4% at every one",
        "Hammerstad-Jensen to within 0.4%",
    ):
        assert retired not in src, (
            f"retired A1 overclaim phrasing {retired!r} is back in "
            "rfx/api/_preflight.py -- the realized-board 0.4% figure is a "
            "pre-#802 frozen record; a live extractor bound requires separate "
            "field data with matched geometry and model provenance"
        )
    # The honest replacement must be present: qualitative anchor language
    # plus an explicit requirement for new matched measurement evidence.
    assert "realized-board Hammerstad-Jensen anchor" in src
    assert "new matched-geometry measurement" in src
    assert "pre-#802" in src


def _load_historical_script(stem):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"history_{stem}", REPO / "scripts" / "diagnostics" / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HISTORICAL_SCRIPTS = (
    "msl_z0_bias_floor_sweep",
    "msl_z0_bias_floor_sweep_realized_anchor",
)


def _forbid_generation(*args, **kwargs):
    pytest.fail("historical inspection attempted geometry/model/field generation or a file write")


@pytest.mark.parametrize("stem", _HISTORICAL_SCRIPTS)
def test_historical_cli_refuses_implicit_generation_before_work(monkeypatch, stem):
    """A legacy command must not silently overwrite the auditable record."""
    mod = _load_historical_script(stem)
    monkeypatch.setattr(mod, "run_one", _forbid_generation, raising=False)
    monkeypatch.setattr(mod, "realized_h_w_um", _forbid_generation, raising=False)
    monkeypatch.setattr(Path, "write_text", _forbid_generation)
    monkeypatch.setattr(Path, "write_bytes", _forbid_generation)
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("stem", _HISTORICAL_SCRIPTS)
def test_explicit_archive_inspection_preserves_records_without_live_geometry(
    monkeypatch, capsys, stem,
):
    """The old field record is never paired with the current rasterizer."""
    mod = _load_historical_script(stem)
    paths = (SOURCE_JSON, ANCHOR_JSON)
    before = {path: path.read_bytes() for path in paths}
    expected_path = ANCHOR_JSON if stem.endswith("realized_anchor") else SOURCE_JSON
    monkeypatch.setattr(mod, "run_one", _forbid_generation, raising=False)
    monkeypatch.setattr(mod, "realized_h_w_um", _forbid_generation, raising=False)
    monkeypatch.setattr(Path, "write_text", _forbid_generation)
    monkeypatch.setattr(Path, "write_bytes", _forbid_generation)
    assert mod.main(["--show-archive"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["current_solver_validation"] is False
    assert out["record_kind"].startswith("historical_")
    assert out["historical_record"] == json.loads(before[expected_path])
    assert {path: path.read_bytes() for path in paths} == before
    assert out["source_json_sha256"] == hashlib.sha256(before[SOURCE_JSON]).hexdigest()
    if expected_path == ANCHOR_JSON:
        assert out["current_geometry_or_model_recomputed"] is False
        assert out["anchor_json_sha256"] == hashlib.sha256(before[ANCHOR_JSON]).hexdigest()
    else:
        assert out["new_field_solves"] == 0


@pytest.mark.parametrize("stem,attribute,original", [
    ("msl_z0_bias_floor_sweep", "SOURCE_JSON", SOURCE_JSON),
    ("msl_z0_bias_floor_sweep_realized_anchor", "SOURCE_JSON", SOURCE_JSON),
    ("msl_z0_bias_floor_sweep_realized_anchor", "ANCHOR_JSON", ANCHOR_JSON),
])
def test_archive_inspection_rejects_changed_provenance(monkeypatch, tmp_path, capsys,
                                                     stem, attribute, original):
    mod = _load_historical_script(stem)
    changed = tmp_path / "changed.json"
    # Even parse-equivalent bytes are not the recorded immutable artifact.
    changed.write_bytes(original.read_bytes() + b"\n")
    monkeypatch.setattr(mod, attribute, changed)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--show-archive"])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""

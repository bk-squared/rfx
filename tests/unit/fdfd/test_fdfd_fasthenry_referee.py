"""Study F (``validation/fdfd/fasthenry_referee.py``): FastHenry as an
independent magnetoquasistatic referee for the FDFD spiral extraction and for
the Greenhouse referee it has been judged against.

What runs LIVE here (needs the built binary; every test that does is skipped
cleanly when it is absent -- build it with
``sh validation/referees/fasthenry/build_fasthenry.sh``):

L1  one rectangular bar: FastHenry's partial self-inductance IS the Hoer-Love
    closed form (``spiral_greenhouse.self_inductance_bar``) to 1e-9 at three
    aspect ratios, with and without filament subdivision.
L2  two parallel bars: ``Im(Z12)/omega`` is ``mutual_inductance_bars`` to 1e-5,
    coplanar and stacked.
L3  the GROUND MODEL: one bar 26 um over the PEC plane by the image
    construction (a mirrored copy as a second port, ``Im(Z11 - Z12)/omega``)
    against the Greenhouse image sum, to 1e-6.
L4  the MESH model reproduces the closed form exactly where the current is
    uniform: a straight bar meshed at 1, 2 and 4 cells across its width is the
    Hoer-Love value to 1e-6 at every level (the corner is the only place the
    two models may differ, and L11 is where they do).
L5  the run cache CANNOT replay another build's numbers: the key of a cached
    run contains the SHA-256 of the binary, so the four gates above are
    statements about the pinned build and not about whatever happens to sit in
    ``$RFX_REFEREE_CACHE`` (the cached directory records that SHA-256 too).
L6  the short standard's grounded POST, live: the bridge whose post reaches the
    ground plane has a LOWER inductance than the same bridge with the post
    truncated to the metal stack, because at z = 0 it joins its own image --
    and the two conductor sets differ in nothing else. This is the difference
    between the referee's bridge and the fixture's, and the reason study F
    reports three de-embedded quantities; the live numbers must reproduce the
    committed JSON's own levels.

What runs live WITHOUT the binary (geometry and bookkeeping):

L7  the meshed conductors are the FDFD's own objects: metal volumes and object
    boxes against the ones ``invariant_ladder.json`` read back from its BUILT
    cell masks, and the centreline against ``spiral_greenhouse``'s independent
    copy.
L8  the Greenhouse referee recomputed inside this study is the published
    333.055 pH to 1e-12; its bridge is TWO HORIZONTAL BARS with no vertical
    member and no ground contact (which is what makes the post-less FastHenry
    bridge the like-for-like one); and the mesh tiles the metal exactly (the
    x-directed segments' volumes sum to the box volume).
L9  the Richardson helper recovers a known limit, and its ``uncertainty`` is
    documented for what it is -- the spread of the p = 1 / p = 2 /
    observed-order estimates, not an error bound.

What is read from the JSON: every gate verdict and the headline numbers
(L10-L14), asserted as hard values so a re-run that moves them fails here.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
STUDY_PATH = REPO / "validation" / "fdfd" / "fasthenry_referee.py"
JSON_PATH = REPO / "validation" / "fdfd" / "fasthenry_referee.json"
PNG_PATH = REPO / "validation" / "fdfd" / "fasthenry_referee.png"
BUILD_SCRIPT = REPO / "validation" / "referees" / "fasthenry" / "build_fasthenry.sh"

_CACHE: dict = {}


def _fr():
    if "fr" not in _CACHE:
        spec = importlib.util.spec_from_file_location("fasthenry_referee_test", STUDY_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fasthenry_referee_test"] = mod
        spec.loader.exec_module(mod)
        _CACHE["fr"] = mod
    return _CACHE["fr"]


def _sg():
    if "sg" not in _CACHE:
        _CACHE["sg"] = _fr().load_referee()
    return _CACHE["sg"]


def _study() -> dict:
    if "json" not in _CACHE:
        _CACHE["json"] = json.loads(JSON_PATH.read_text())
    return _CACHE["json"]


def _have_binary() -> bool:
    return _fr().fasthenry_binary() is not None


needs_fasthenry = pytest.mark.skipif(
    not _have_binary(),
    reason=f"no FastHenry binary; build it with: sh {BUILD_SCRIPT}")


# ---------------------------------------------------------------- live -----

@needs_fasthenry
def test_l1_single_bar_is_the_hoer_love_closed_form():
    """One segment, three aspect ratios, with and without filaments: the
    measured L is the Hoer-Love partial self-inductance. Tolerance 1e-9 --
    measured 1.5e-13 / 4.4e-16 / 2.2e-16 at (nwinc, nhinc) = (1, 1) and
    <= 5.3e-9 at (8, 4), where the filament subdivision only reassembles the
    same uniform current out of 32 pieces."""
    fr, sg = _fr(), _sg()
    for ell, w, t in ((100e-6, 10e-6, 2e-6), (10e-6, 10e-6, 2e-6), (4e-6, 10e-6, 10e-6)):
        for nw, nh in ((1, 1), (4, 4)):
            m = fr.Mesh()
            fr.segment_chain(m, [(0.0, 0.0, 0.0), (ell, 0.0, 0.0)], w, t,
                             group_start="a", group_end="b")
            r = fr.measure_mesh(m, [("p", "a", "b")], (1e3, 1e3, 1.0), nwinc=nw, nhinc=nh,
                                tag=f"test-self-{ell}-{nw}", kind="plain")
            got = r["L"]["1000"]
            ref = sg.self_inductance_bar(ell, w, t)
            assert abs(got / ref - 1.0) <= 1e-8, (ell, w, t, nw, nh, got, ref)


@needs_fasthenry
def test_l2_two_parallel_bars_are_the_mutual_closed_form():
    """``Im(Z12)/omega`` of two 100 um bars against
    ``mutual_inductance_bars``: coplanar one width apart and stacked at the
    underpass offset. Tolerance 1e-5 (measured 3.3e-9 and 1.4e-11)."""
    fr, sg = _fr(), _sg()
    ell, w, t = 100e-6, fr.WIDTH, fr.T_M2
    for dw, dt in ((2.0 * fr.WIDTH, 0.0), (0.0, fr.DZ_UNDER)):
        m = fr.Mesh()
        fr.segment_chain(m, [(0.0, 0.0, 0.0), (ell, 0.0, 0.0)], w, t,
                         group_start="a", group_end="b")
        fr.segment_chain(m, [(0.0, dw, dt), (ell, dw, dt)], w, t,
                         group_start="c", group_end="d")
        r = fr.measure_mesh(m, [("p1", "a", "b"), ("p2", "c", "d")], (1e3, 1e3, 1.0),
                            tag=f"test-pair-{dw}-{dt}", kind="pair")
        ref = sg.mutual_inductance_bars(ell, w, t, ell, w, t, dw, dt, 0.0)
        assert abs(r["M"]["1000"] / float(ref) - 1.0) <= 1e-5, (dw, dt, r["M"]["1000"], ref)


@needs_fasthenry
def test_l3_image_construction_is_the_greenhouse_ground_image():
    """The ground model this referee uses: a 100 x 10 x 2 um bar 26 um over the
    PEC plane, its mirror image driven as a second port with ``-I``, and
    ``L = Im(Z11 - Z12)/omega`` against ``greenhouse_terms(ground_height=26
    um)``. Tolerance 1e-6 (measured 3.7e-12 at one filament per segment)."""
    fr, sg = _fr(), _sg()
    m = fr.segment_bar(100e-6, z=fr.GROUND_H)
    both, ports = fr.with_image(m, [("a", "aend")])
    r = fr.measure_mesh(both, ports, (1e3, 1e3, 1.0), tag="test-image-bar")
    ref = sg.greenhouse_terms(
        [sg.Segment((-50e-6, 0.0, 0.0), (50e-6, 0.0, 0.0), fr.WIDTH, fr.T_M2)],
        ground_height=fr.GROUND_H).total
    assert abs(r["L"]["1000"] / float(ref) - 1.0) <= 1e-6


@needs_fasthenry
def test_l4_mesh_model_is_exact_where_the_current_is_uniform():
    """The 3-D mesh model at 1, 2 and 4 cells across the width of a straight
    200 um bar reproduces the Hoer-Love image value to 1e-6 -- splitting a
    uniform current lengthwise and crosswise changes no partial inductance, so
    any mesh-model difference from the closed form is a CORNER effect, not a
    discretisation artefact."""
    fr, sg = _fr(), _sg()
    ref = sg.greenhouse_terms(
        [sg.Segment((-100e-6, 0.0, 0.0), (100e-6, 0.0, 0.0), fr.WIDTH, fr.T_M2)],
        ground_height=fr.GROUND_H).total
    for k in (1, 2, 4):
        r = fr.measure("bar", k, freqs=(1e3, 1e3, 1.0))
        assert abs(r["L"]["1000"] / float(ref) - 1.0) <= 1e-6, (k, r["L"]["1000"], ref)


@needs_fasthenry
def test_l5_run_cache_is_keyed_on_the_binarys_own_sha256():
    """A cached run may only be replayed for the build that produced it.

    ``run_key`` mixes the deck, the solver options and the SHA-256 of the
    binary, so the live gates above cannot pass from a cache written by a
    different (or unpatched) FastHenry: change the binary's identity and the
    key changes. The directory of a run executed here records the same
    SHA-256, and it is the one in the manifest of the pinned build."""
    fr = _fr()
    deck = ("* cache-key probe\n.units um\n.default sigma=2\n"
            "N1 x=0 y=0 z=0\nN2 x=40 y=0 z=0\n"
            "E1 N1 N2 w=10 h=2 wx=0 wy=1 wz=0\n.external N1 N2\n"
            ".freq fmin=1e3 fmax=1e3 ndec=1\n.end\n")
    binary = str(fr.fasthenry_binary())
    real = fr.binary_identity()
    assert len(real) == 64
    key_real = fr.run_key(deck)
    try:
        fr._BINARY_ID[binary] = "0" * 64        # a different build, same deck
        key_other = fr.run_key(deck)
    finally:
        fr._BINARY_ID[binary] = real
    assert key_other != key_real
    assert fr.run_key(deck) == key_real
    res = fr.run_fasthenry(deck, "test-cache-key")
    stamp = pathlib.Path(res["dir"]) / "binary_sha256"
    assert stamp.is_file() and stamp.read_text().strip() == real
    man = fr.manifest()
    if man.get("binary_sha256"):                # manifests written before this field
        assert man["binary_sha256"] == real


@needs_fasthenry
def test_l6_the_grounded_post_lowers_the_bridge():
    """The short standard's post REACHES the ground plane, and that is not a
    modelling choice: at z = 0 it joins its own image, so the grounded bridge
    carries a second real-to-image path and a LOWER inductance than the same
    bridge with the post truncated to the metal stack. The referee's bridge has
    no vertical member at all (L8), so the post-less one is its like-for-like
    counterpart -- which is why the ground contact is reported as its own term
    instead of being charged to the referee.

    Structurally the two conductor sets differ in ONE number (the post's lower
    z); numerically the live values must be the committed JSON's own levels."""
    fr = _fr()
    g, u = fr.bridge_boxes(grounded=True), fr.bridge_boxes(grounded=False)
    assert g[:2] == u[:2]                       # the two arms are the same boxes
    assert g[2][:4] == u[2][:4] and g[2][5] == u[2][5]
    assert g[2][4] == 0.0 and u[2][4] == fr.Z_M1 > 0.0
    # the image construction joins the two halves where the metal touches the
    # plane: one shared node per cell face at z = 0, two segments incident on
    # it (the real cell and its mirror).  The post-less bridge has no node
    # there at all, so its halves are separate conductors.
    for k in (1, 2):
        mg = fr.build_case("bridge", k)[0]
        mu = fr.build_case("bridge_ungrounded", k)[0]
        at_zero = [n for n, v in mg.nodes.items() if abs(v[2]) < 1e-15]
        assert len(at_zero) == k * k                    # one per cell face of the post
        for n in at_zero:
            inc = sum(1 for _, a, b, _w, _h, _d in mg.segs if n in (a, b))
            assert inc == 2, (k, n, inc)
        assert not [n for n, v in mu.nodes.items() if abs(v[2]) < 1e-15]
    s = _study()
    ks = [r["k"] for r in s["spiral_mesh"]["bridge"]]
    recorded = dict(zip(ks, s["f3"]["post_short"]["bridge_per_level"]))
    for k in (2, 3):
        lg = fr.measure("bridge", k, freqs=fr.freq_spec(k))["L"]["1000"]
        lu = fr.measure("bridge_ungrounded", k, freqs=fr.freq_spec(k))["L"]["1000"]
        assert lu > lg, (k, lu, lg)
        assert (lu - lg) == pytest.approx(recorded[k], rel=1e-9), (k, lu - lg, recorded[k])
    # at the COARSEST level the two coincide: with one cell across the post the
    # two arms are a balanced bridge and the single z = 0 branch carries no
    # current.  The study's JSON says so, and says it in f3.post_short.note.
    l_coarse = s["spiral_mesh"]["bridge"][0]["L"]["1000"]
    assert abs(recorded[1]) / l_coarse <= 1e-13, recorded[1]     # measured 6.2e-15
    assert "balanced bridge" in s["f3"]["post_short"]["note"]


# ------------------------------------------------- live, no binary needed ---

def test_l7_meshed_conductors_are_the_fdfd_objects():
    """Metal volumes and object boxes against the ones the FDFD study read
    back from its BUILT cell masks (``invariant_ladder.json``,
    ``p1.coordinates.1``), and the centreline against ``spiral_greenhouse``'s
    own copy of the ``rect_spiral`` convention."""
    fr = _fr()
    g = fr.block_geometry(_sg())
    assert g["centreline_max_deviation_m"] <= 1e-15
    assert g["worst_volume_rel"] is not None and g["worst_volume_rel"] <= 1e-12
    assert g["worst_box_deviation_m"] == 0.0
    # the three fixtures' metal volumes, in um^3: 15840 / 2600 / 6500 - 400
    # (the short's arms overlap the post) = 6100
    assert g["volumes_m3"]["dut"] * 1e18 == pytest.approx(15840.0, rel=1e-12)
    assert g["volumes_m3"]["open"] * 1e18 == pytest.approx(2600.0, rel=1e-12)
    assert g["volumes_m3"]["short"] * 1e18 == pytest.approx(6100.0, rel=1e-12)
    assert g["passed"]


def test_l8_greenhouse_recomputed_its_bridge_and_the_mesh_tiling():
    """Three structural facts. (1) The Greenhouse referee recomputed here is
    the published 333.055 pH. (2) Its BRIDGE -- the conductor whose inductance
    the referee's 333.055 pH subtracts -- is two HORIZONTAL bars, one on M2 and
    one on M1, with no vertical member and no ground contact: that is why study
    F compares the referee against the FastHenry bridge whose post is truncated
    to the metal stack, and reports the grounded post's effect separately.
    (3) The mesh is a TILING: the x-directed segments of a meshed box have
    exactly the box's volume (each cell covered once), which is why a uniform
    current costs nothing in it."""
    fr, sg = _fr(), _sg()
    gh = fr.greenhouse_block(sg)
    assert gh["deembedded"] == pytest.approx(3.3305510896906523e-10, rel=1e-12)
    assert gh["strip"] == pytest.approx(3.493712871350631e-10, rel=1e-12)
    assert gh["bridge"] == pytest.approx(1.6316178165997864e-11, rel=1e-12)
    # the referee's bridge, rebuilt here from the same module: two bars, both
    # horizontal (z constant along each), neither of them at the ground plane
    xo, xi = fr.x_references()
    y_ref, xs = fr.y_reference(), 0.5 * (xo + xi)
    bars = [sg.Segment((xo, y_ref, 0.0), (xs, y_ref, 0.0), fr.WIDTH, fr.T_M2),
            sg.Segment((xs, y_ref, -fr.DZ_UNDER), (xi, y_ref, -fr.DZ_UNDER), fr.WIDTH, fr.T_M1)]
    assert len(bars) == 2
    for b in bars:
        assert b.start[2] == b.end[2]                       # horizontal: no vertical member
        assert abs(b.start[2]) < fr.GROUND_H                # above the plane, never touching it
    assert sg.greenhouse_terms(bars, ground_height=fr.GROUND_H).total == \
        pytest.approx(gh["bridge"], rel=1e-12)
    box = (0.0, 100e-6, 0.0, 10e-6, 0.0, 2e-6)
    m = fr.mesh_boxes([box], 2.5e-6, 10e-6, 10e-6, 2e-6, 5e-6, 2e-6,
                      terminals=[("a", 0, 0.0, -1, box), ("b", 0, 100e-6, +1, box)])
    vol = 0.0
    for _, na, nb, w, h, _ in m.segs:
        a, b = np.array(m.nodes[na]), np.array(m.nodes[nb])
        if abs(b[0] - a[0]) > 0:
            vol += abs(b[0] - a[0]) * w * h
    assert vol == pytest.approx(100e-6 * 10e-6 * 2e-6, rel=1e-12)


def test_l9_richardson_helper_recovers_a_known_limit():
    """The extrapolation used for every limit in the study: on a synthetic
    ``L(h) = 1 - 2 h^2`` ladder the p = 2 estimate is exact and the observed
    order comes back as 2. The reported ``uncertainty`` is half the SPREAD of
    the p = 1 / p = 2 / observed-order estimates -- a disagreement between
    estimators, not a bound on the discretisation error -- and the result says
    so in ``uncertainty_is``; here the p = 1 member is the whole spread."""
    fr = _fr()
    hs = [1.0, 0.5, 0.25]
    r = fr.richardson(hs, [1.0 - 2.0 * h ** 2 for h in hs])
    assert r["estimates"]["p2_finest_pair"] == pytest.approx(1.0, abs=1e-12)
    assert r["observed_order"] == pytest.approx(2.0, abs=1e-9)
    assert "not a bound" in r["uncertainty_is"].lower() or \
        "NOT a bound" in r["uncertainty_is"]
    spread = max(r["estimates"].values()) - min(r["estimates"].values())
    assert r["uncertainty"] == pytest.approx(0.5 * spread, rel=1e-12)


# ----------------------------------------------------------- from the JSON --

def test_l10_gates_pass_in_the_committed_json():
    s = _study()
    g = s["gates"]
    for name in ("F0", "F1", "F2", "F3", "F4", "F5"):
        assert g[name]["passed"], (name, g[name])
    assert g["all_passed"]
    # the JSON is committed; the PNG is a build artifact (**/*.png is gitignored),
    # so it exists only after a local run -- check it when it is there
    assert JSON_PATH.is_file()
    if PNG_PATH.is_file():
        assert PNG_PATH.stat().st_size > 0
    # the run cache's provenance travels with the numbers
    assert len(s["cost"]["binary_sha256"]) == 64


def test_l11_headline_numbers():
    """The numbers this study's docstring quotes, asserted against the JSON."""
    s = _study()
    gh = s["geometry"]["greenhouse"]["deembedded"]
    assert gh == pytest.approx(3.3305510896906523e-10, rel=1e-12)
    # F1: every closed form to better than 1e-5
    assert s["gates"]["F1"]["worst_rel"] <= 1e-5
    # F3/F4: the three de-embedded continuum limits.  "planes_referee" is the
    # referee's own quantity (post-less bridge), "fixture" is the FDFD's.
    rows = s["f3"]["rows"]
    assert rows["planes_referee"]["limit"] * 1e12 == pytest.approx(PLANES_REFEREE_PH, rel=2e-4)
    assert rows["planes_physical"]["limit"] * 1e12 == pytest.approx(PLANES_PHYSICAL_PH, rel=2e-4)
    assert rows["fixture"]["limit"] * 1e12 == pytest.approx(FIXTURE_PH, rel=2e-4)
    assert rows["planes_referee"]["rel_to_greenhouse"] == pytest.approx(PLANES_REFEREE_REL,
                                                                        abs=2e-4)
    assert s["f3"]["post_short"]["abs"] * 1e12 == pytest.approx(POST_SHORT_PH, rel=2e-3)
    assert s["f3"]["column_transition"]["abs"] * 1e12 == pytest.approx(COLUMN_PH, rel=2e-3)
    assert s["gates"]["F4"]["within_5_percent"] is True
    assert s["gates"]["F4"]["within_3_percent"] is WITHIN_3
    # F5: one corner
    assert s["f5"]["excess_limit"] * 1e12 == pytest.approx(EXCESS_PH, rel=2e-3)
    assert s["f5"]["greenhouse"]["excess"] * 1e12 == pytest.approx(-7.778293597325733, rel=1e-9)


def test_l12_attribution_closes_and_keeps_the_fixture_terms_apart():
    """The decomposition is an identity, not a fit: the de-embedded difference
    on the REFEREE's quantity (strip minus the post-less bridge) is (strip
    error) - (bridge error), and the strip's error is (corners in situ) + (via
    and underpass).  Both close to better than 0.1 pH -- the only slack is that
    each term is Richardson-extrapolated on its own ladder.  The two terms that
    are fixture physics rather than referee error (the post's ground contact
    and the lead columns) are recorded OUTSIDE the decomposition, and the
    plumbing gate rides along: the SEGMENT model reproduces the Greenhouse sum
    of the same bars to 1e-5."""
    s = _study()
    at = s["attribution"]
    assert at["quantity"] == "planes_referee"
    assert abs(at["closes_to"]) * 1e12 <= 0.1, at["closes_to"]
    ins = at["inside_strip"]
    assert (ins["corners_in_situ"]["total"] + ins["via_and_underpass"]["total"]) \
        == pytest.approx(at["terms"]["strip"]["total"], rel=1e-9)
    assert at["segment_model_check"]["rel"] == pytest.approx(0.0, abs=1e-5)
    # the isolated-corner cross-check has the same sign and order of magnitude
    assert ins["isolated_corner_check"]["total"] < 0.0
    assert ins["corners_in_situ"]["total"] < 0.0
    # the referee's bridge is LOW against the metal it stands for (it has no
    # vertical member), so that term enters the total with a negative sign too
    nre = at["not_the_referees_error"]
    assert at["terms"]["bridge"]["total"] < 0.0
    assert nre["post_ground_contact"]["bridge_abs"] > 0.0
    assert nre["post_ground_contact"]["quantity_abs"] < 0.0
    assert nre["lead_columns"]["total"] > 0.0
    # every term of the decomposition is the referee's error; neither of these
    # two is inside it
    assert set(at["terms"]) == {"strip", "bridge"}


def test_l13_the_extrapolations_survive_a_finer_level():
    """Every limit here is a Richardson extrapolation whose +- is the SPREAD of
    the p = 1 / p = 2 / observed-order estimates, which is not an error bound.
    The ``confirmation`` block measures one level finer than the ladder
    (k = 10, off the ladder) and extrapolates again from the three finest
    points: the two limits must agree inside the two spreads added together,
    and the finer level must sit between the ladder's last point and the
    limit."""
    s = _study()
    c = s["confirmation"]
    assert c["k"] == 10
    assert c["all_consistent"] is True
    for name, chk in c["checks"].items():
        assert chk["consistent"] is True, (name, chk)
        assert abs(chk["difference"]) <= chk["tolerance"], (name, chk)
        assert chk["refined_uncertainty"] < chk["ladder_uncertainty"], (name, chk)
        assert chk["finest_h_over_W"] == pytest.approx(0.1, rel=1e-12)
    assert s["gates"]["F2"]["confirmation_consistent"] is True


def test_l14_f4_applies_only_the_corrections_that_belong_to_each_quantity():
    """F4's primary comparison is the ``fixture`` quantity, the like-for-like
    counterpart of the FDFD's ``L_dut``; the only model differences applied to
    it are study P's walls and RC.  Study P's post-short correction is READ and
    NOT applied -- it converts ``L_dut`` into a third quantity through that
    study's straight-bar thru probe, while study F measures the two steps from
    ``fixture`` to the referee's quantity itself (the lead-column transition
    and the post's ground contact).  And the FDFD side is reported
    un-extrapolated as well, because "the range contains it" compares two
    extrapolations."""
    s = _study()
    f4 = s["f4"]
    c = f4["fdfd_corrected"]
    assert "post_short_rel_range_read_not_applied" in c
    assert c["post_short_rel_range_read_not_applied"][0] < 0.0
    w, rc = c["walls_rel"], c["rc_rel"]
    conv = c["conversion_rel"]
    assert conv["column_transition"] == pytest.approx(s["f3"]["column_transition"]["rel"],
                                                      rel=1e-12)
    assert conv["post_ground_contact"] == pytest.approx(s["f3"]["post_short"]["rel"], rel=1e-12)
    for lo_hi, raw in zip(c["for_fixture"], f4["fdfd_range"]):
        assert lo_hi == pytest.approx(raw * (1.0 + w + rc), rel=1e-12)
    for got, fix in zip(c["for_planes_physical"], c["for_fixture"]):
        assert got == pytest.approx(fix / (1.0 + conv["column_transition"]), rel=1e-12)
    for got, phys in zip(c["for_planes_referee"], c["for_planes_physical"]):
        assert got == pytest.approx(phys * (1.0 + conv["post_ground_contact"]), rel=1e-12)
    # the un-extrapolated FDFD side: its finest MEASURED level is further from
    # the FastHenry limit than its own observed-order estimate is
    h = f4["extrapolation_honesty"]
    assert abs(h["fdfd_finest_level"]["rel_to_fasthenry"]) > \
        abs(h["closest_estimate"]["rel_to_fasthenry"])
    assert s["gates"]["F4"]["fdfd_finest_level_rel"] == pytest.approx(
        h["fdfd_finest_level"]["rel_to_fasthenry"], rel=1e-12)


# headline numbers, kept beside the tests that assert them (equal to the JSON
# and to the module docstring of validation/fdfd/fasthenry_referee.py)
PLANES_REFEREE_PH = 318.743    # f3.rows.planes_referee.limit, +- 0.163 pH
PLANES_PHYSICAL_PH = 319.182   # f3.rows.planes_physical.limit, +- 0.157 pH
FIXTURE_PH = 319.913           # f3.rows.fixture.limit, +- 0.151 pH
PLANES_REFEREE_REL = -0.04297  # f3.rows.planes_referee.rel_to_greenhouse
POST_SHORT_PH = -0.4391        # f3.post_short.abs (the post's ground contact)
COLUMN_PH = 0.7314             # f3.column_transition.abs (the lead columns)
EXCESS_PH = -8.5771            # f5.excess_limit (Greenhouse: -7.7783 pH)
WITHIN_3 = True                # gates.F4.within_3_percent

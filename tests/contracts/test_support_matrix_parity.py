"""Structural parity gate between the S-parameter support-matrix carriers: the prose
carrier is the ``## S-parameters`` part of ``docs/guides/support_matrix.md``.

``docs/guides/sparameter_support_matrix.md`` (prose) and
``docs/guides/sparameter_support_matrix.json`` (machine-readable, consumed by
``test_sparameter_support_contract.py``) describe the same evidence and are
supposed to say the same thing, but nothing enforced that until now (issue
#554, filed from a #553 adversarial review: the two carriers drifted TWICE
after filing and were hand-fixed twice -- PR #559's fix commit updated only
the .md, and PR #581's review found matching-phrasing requirements enforced
by nothing).

Design, deliberately narrow (see issue #554's own scoping and the false
positives/false negatives this file's own construction and its PR #584
review ran into before landing):

- The .md is prose and the .json is structured data; they will never be
  byte-identical, and a full-text/substring check is a tautological "gate
  can bind an artifact" instrument, not a real check. This gate compares
  NUMBERS and STATUS TOKENS instead -- the atoms that actually drifted.
- Numeric parity is ONE-DIRECTION-primary: every distinctive numeric token in
  a json lane's ``numeric_metrics``/``ad_evidence`` (the fields that carry
  the lane's quantitative claims) must appear, verbatim after light
  formatting normalization, somewhere in the mapped .md section. The .md
  section legitimately carries MORE numeric detail than the compact json
  summary (setup restrictions, caveats) -- requiring the reverse for every
  number produced false positives on live, non-drifted content (e.g. the
  waveguide "Setup restrictions" subsection's iris-rasterization numbers)
  during this gate's own construction, so it is not required.
  LIMITATION (PR #584 review, M2): this checks SET MEMBERSHIP of numbers,
  not number-to-claim binding. It catches a number that VANISHED from a
  carrier (updated in one, left stale/absent in the other -- the #559 R1
  drift shape). It does NOT catch value reassignment where a number that
  already appears somewhere else in the section gets attached to the wrong
  claim, or two numbers get swapped between two sentences in the same
  section -- both would stay green because the SET of tokens is unchanged.
- Run IDs (VESSL job numbers) ARE checked in both directions: they are
  unambiguous provenance identifiers, not general physics quantities, so a
  run ID quoted in only one carrier is unreviewable from the other and is
  exactly the "run ID present in one carrier only" drift class this gate
  must catch. Extraction is anchored to the literal "VESSL" keyword
  (``VESSL[ ]+(?:run[ ]+)?<digits>``), not a bare long-integer scan: a
  free-scanning ``\\d{6,}`` window mined a phantom run ID out of the
  fractional digits of an unrelated decimal (PR #584 review, L2). Note this
  reverse check is only load-bearing where the .md section actually names a
  VESSL run at all -- today that is 1 of the 5 mapped lanes (MSL); the other
  4 pass vacuously because their sections don't cite a VESSL run yet, not
  because parity was verified (PR #584 review, L5).
- Status tokens (experimental / superseded / unresolved) must AGREE in
  presence between a lane's json ``support_status``/``validation_status``
  FIELDS and a per-lane DESIGNATED status region of the mapped .md section
  (the section header plus one or more hand-identified anchor paragraphs --
  see ``LANE_STATUS_ANCHORS_MD``), not the lane's full json entry
  (``json.dumps``) versus the whole .md section. The whole-entry/whole-
  section version shipped in the initial PR #584 and was proven unable to
  catch a headline status flip on review: the coaxial-port lane's json entry
  mentions "experimental" in FIVE different fields (support_status,
  supported_configurations/calculation_api, result_type, validation_status,
  known_limits, ad_evidence for other lanes), and its .md section separately,
  correctly states "deprecated and experimental" for the OLD
  ``compute_coaxial_s_matrix`` path a few paragraphs before the sentence that
  actually states the NEW ``compute_coaxial_two_port`` path's status -- so
  flipping just the intended (two-port) status sentence in either carrier
  left an unrelated, still-true mention of the word standing in the other,
  and the whole-blob check stayed green in both directions. Narrowing BOTH
  sides to their designated, hand-identified status text fixes the
  demonstrated miss (see the coax-lane falsifier in the PR #584 review
  follow-up commit). "pending" was tried and dropped during original
  construction: the json msl entry uses it ("...pending issue #519") where
  the .md paraphrases the same fact without the literal word, which is not
  drift and would have been a permanent false-positive tripwire.
  ROUND TWO (PR #588 review fix, P1 -- BLOCKING): narrowing the .md side to
  per-anchor paragraphs was not sufficient on its own once the coaxial-port
  lane's json entry started stating BOTH of its two sub-APIs' status words
  in the SAME whole-entry ``support_status``/``validation_status`` string --
  that combined json text is constant-True for "experimental" regardless of
  which sub-API's OWN status changed, proven by mutation on review (flipping
  either carrier's two-port-specific status text alone, with the other
  carrier untouched, stayed green both directions; a control mutation on the
  single-sub-API PRE-#588 prose correctly reds). ``LANE_STATUS_ANCHORS_JSON``
  narrows the JSON side the same way ``LANE_STATUS_ANCHORS_MD`` narrows the
  .md side, for any primitive whose json entry bundles multiple
  independently-evolving sub-API statuses: each json anchor must match
  EXACTLY ONE ``known_limits`` bullet (paired 1:1, same order, with the
  corresponding .md anchor), and THAT bullet -- not the whole entry -- is
  what gets scanned. A primitive absent from ``LANE_STATUS_ANCHORS_JSON``
  keeps the original whole-entry behavior (still correct for a lane with
  only one status-bearing sub-API).
- The section map (json ``primitive`` -> .md "## " header) is explicit and
  asserted complete in both directions for the lanes it covers. Lanes not
  yet mapped must be named in ``UNMAPPED_LANES_TODO``; a new json primitive
  in neither dict fails ``test_lane_map_is_complete_in_both_directions``
  loudly instead of silently escaping the gate.

- The "## API summary" table is a THIRD in-file copy of the same per-lane
  status claims (issue #586 item 2), and a demonstrated drift site: one of
  PR #584's five live-drift fixes was the coax section saying "no external
  referee" while the summary table's row for the same API already correctly
  said the referee was registered -- caught only by accident. The third
  comparison (``test_api_summary_status_cell_agrees_with_section_status``)
  pairs each mapped lane's summary-table row(s) (``API_SUMMARY_ROW_ANCHORS``,
  1:1 and in order with ``LANE_STATUS_ANCHORS_MD``) with the section's
  designated status block and requires STATUS_TOKENS presence-polarity
  agreement between the row's status cell and that block. The table parser
  is fail-loud by construction (the #303 "All checks passed with a skipped
  check family" class): the section, the exact 4-column header row, the
  4-cell shape of every row, and a nonzero data-row count are all asserted,
  and every mapped row key must match exactly one row -- a restructured
  table fails loudly instead of parsing to nothing and passing vacuously.
  Extending this check exposed live drift on first contact (the #584
  experience repeating): three rows understating the section's
  nonuniform-experimental status, the coax two-port row carrying ANOTHER
  lane's status word inside its own status cell, the transition row missing
  its section's "unresolved", and the compute_mixed_s_matrix lane having no
  summary row at all -- all fixed doc-side in the same change.

Not in scope (see the issue, and issue #586 filed from the PR #584 review for
the deferred items): auto-generating one carrier from the other, byte/full-
text parity, and covering every lane immediately (the map grows incrementally
as drift surfaces in a given lane).

Pure text parsing, no FDTD -- joins the contract-test family (same default
marker/lane as test_sparameter_support_contract.py; no slow/gpu marker).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.docs_consistency

MATRIX_JSON_PATH = Path("docs/guides/sparameter_support_matrix.json")
MATRIX_MD_PATH = Path("docs/guides/support_matrix.md")



LANE_SECTION_MAP: dict[str, str] = {
    "add_port(extent=None)": "### Lumped port",
    "add_port(extent=...)": "### Wire port",
    "add_msl_port(...)": "### Microstrip-line port",
    "add_waveguide_port(...)": "### Rectangular-waveguide port",
    "add_coaxial_port(...)": "### Coaxial port",
    "add_coaxial_port(...) + add_msl_port(...) driven by compute_coax_msl_transition(...)":
        "### Coax-to-microstrip transition — experimental, diagnostic only",
    "add_floquet_port(...)": "### Floquet/Bloch and non-port observables",
    "add_port(...) + add_msl_port(...) driven by compute_mixed_s_matrix(...)":
        "### Mixed port families — experimental",
}

UNMAPPED_LANES_TODO: dict[str, str] = {
    "add_source(...) / add_polarized_source(...)":
        "not_a_port family: numeric_metrics carries no numeric tokens (issue #586 kept it deferred)",
    "add_tfsf_source(...)":
        "not_a_port family: numeric_metrics carries no numeric tokens (issue #586 kept it deferred)",
    "add_probe(...) / add_dft_plane_probe(...) / add_flux_monitor(...)":
        "not_a_port family: numeric_metrics carries no numeric tokens (issue #586 kept it deferred)",
}

NUMBER_FREE_LANES: frozenset[str] = frozenset({
    "add_floquet_port(...)",
    "add_coaxial_port(...) + add_msl_port(...) driven by compute_coax_msl_transition(...)",
})

LANE_STATUS_ANCHORS_MD: dict[str, tuple[str, ...]] = {
    "add_port(extent=None)": (
        "Analytic extractor and V/I replay checks validate",
    ),
    "add_port(extent=...)": (
        "The nonuniform wire calculation is",
    ),
    "add_msl_port(...)": (
        'Nonuniform `mode="laplace"` and `mode="uniform"` have internal',
    ),
    "add_waveguide_port(...)": (
        "The calculation remains",
    ),
    "add_coaxial_port(...)": (
        "Use `compute_coaxial_line_reflection(...)` for the documented TEM-line result.",
        "every DUT it can currently gate against",
    ),
    "add_coaxial_port(...) + add_msl_port(...) driven by compute_coax_msl_transition(...)": (
        "drives a coaxial port and a microstrip port",
    ),
    "add_floquet_port(...)": (
        "`add_floquet_port(...)` has broadside modal bookkeeping",
    ),
    "add_port(...) + add_msl_port(...) driven by compute_mixed_s_matrix(...)": (
        "| off-diagonal magnitude |",
    ),
}

LANE_STATUS_ANCHORS_JSON: dict[str, tuple[str, ...]] = {
    "add_coaxial_port(...)": (
        "compute_coaxial_line_reflection(...) is not a general multi-port coaxial network solver",
        "compute_coaxial_two_port(...) drives each end of one through line",
    ),
}


API_SUMMARY_HEADER = "### API summary"
API_SUMMARY_EXPECTED_COLUMNS = ("Primitive", "Calculation API", "Result", "Current status")

API_SUMMARY_ROW_ANCHORS: dict[str, tuple[str, ...]] = {
    "add_port(extent=None)": (
        "Lumped `add_port(..., extent=None)` | `run(",
    ),
    "add_port(extent=...)": (
        "Wire `add_port(..., extent=...)` | `run(",
    ),
    "add_msl_port(...)": (
        "`add_msl_port(...)` | `compute_msl_s_matrix(...)`",
    ),
    "add_waveguide_port(...)": (
        "`add_waveguide_port(...)` | `compute_waveguide_s_matrix(...)`",
    ),
    "add_coaxial_port(...)": (
        "`add_coaxial_port(...)` | `compute_coaxial_line_reflection(...)`",
        "`add_coaxial_port(...)` | `compute_coaxial_two_port(...)`",
    ),
    "add_coaxial_port(...) + add_msl_port(...) driven by compute_coax_msl_transition(...)": (
        "`add_coaxial_port(...)` + `add_msl_port(...)` | `compute_coax_msl_transition(...)`",
    ),
    "add_floquet_port(...)": (
        "`add_floquet_port(...)` | no documented high-level",
    ),
    "add_port(...) + add_msl_port(...) driven by compute_mixed_s_matrix(...)": (
        "`add_port(...)` + `add_msl_port(...)` | `compute_mixed_s_matrix(...)`",
    ),
}



_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\d-])-\d+\.\d+(?:[eE][+-]?\d+)? ?%?"
    r"|\d+\.\d+(?:[eE][+-]?\d+)? ?%?"
    r"|(?<![\d-])-\d+[eE][+-]?\d+"
    r"|\d+[eE][+-]?\d+"
    r"|\d+ ?%"
    r"|\b\d{4,}\b"
)

_VESSL_RUN_ID_RE = re.compile(r"VESSL\s+(?:run\s+)?(\d{6,}(?:/\d{6,})*)")

STATUS_TOKENS = ("experimental", "superseded", "unresolved", "validated with scope")

_OVERCLAIM_PHRASES = (
    "differentiable end-to-end",
    "end-to-end differentiable",
    "fully differentiable",
    "ad-traceable",
    "ad traceable",
    "supports gradients",
    "gradients through",
)


def _normalize_numeric_token(token: str) -> tuple[float, bool]:
    """(value, was_percent) so '0.10' and '0.100' compare equal but 21 and
    21% do not collide. float() tolerates the optional space this regex
    allows before a trailing '%' or unit, so no separate whitespace strip
    is needed here."""
    is_percent = token.endswith("%")
    core = token[:-1] if is_percent else token
    return (round(float(core), 12), is_percent)


def _extract_numeric_tokens(text: str) -> set[tuple[float, bool]]:
    """Short bare integers are excluded because they are common enough across unrelated sentences to add noise without adding drift-detection value."""
    return {_normalize_numeric_token(m.group(0)) for m in _NUMERIC_TOKEN_RE.finditer(text)}


def _extract_vessl_run_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for m in _VESSL_RUN_ID_RE.finditer(text):
        ids.update(m.group(1).split("/"))
    return ids


def _md_sections(md_text: str) -> dict[str, str]:
    """Split only the unique ``## S-parameters`` part on ``### `` headers.
    Include each stripped header in its section text and use it as the key.
    """
    lines = md_text.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "## S-parameters"]
    assert len(starts) == 1, (
        f"expected exactly one '## S-parameters' heading, found {len(starts)}"
    )
    start = starts[0]
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    sections: dict[str, str] = {}
    header: str | None = None
    body_lines: list[str] = []
    for line in lines[start:end]:
        if line.startswith("### "):
            if header is not None:
                sections[header] = header + "\n" + "\n".join(body_lines)
            header = line.strip()
            body_lines = []
        else:
            body_lines.append(line)
    if header is not None:
        sections[header] = header + "\n" + "\n".join(body_lines)
    return sections


def _paragraph_blocks(section_text: str) -> list[str]:
    """Split a .md section into blank-line-delimited blocks (a prose
    paragraph, or a tightly-packed bullet list with no blank lines between
    items counts as one block), with each block's soft-wrapped lines joined
    by a single space so a sentence split across a line wrap can still be
    matched as one contiguous string."""
    raw_blocks = re.split(r"\n\s*\n", section_text)
    return [
        " ".join(line.strip() for line in block.splitlines() if line.strip())
        for block in raw_blocks
        if block.strip()
    ]


def _lane_status_blocks_md(
    section_text: str, header: str, anchors: tuple[str, ...]
) -> list[tuple[str, str]]:
    """[(anchor, header + '\\n' + matched_block), ...] -- one entry per
    anchor, each paired with the section header (several sections put their
    loudest status marker directly in the header, e.g. "## Coax<->MSL
    transition -- EXPERIMENTAL").

    Anchors are checked INDEPENDENTLY by the caller, not unioned into one
    combined text first: the coaxial-port lane bundles two sub-APIs with
    independently evolving status (the deprecated compute_coaxial_s_matrix
    and the experimental compute_coaxial_two_port) in ONE .md section and
    ONE json validation_status field. A first version of this check unioned
    both anchors' text before comparing against json, and review found it
    could NOT catch a flip confined to just the compute_coaxial_two_port
    paragraph: the still-correct, untouched "...deprecated and experimental"
    sentence from the OTHER anchor kept the combined text's status token
    present, so the flip on the intended anchor was invisible. Returning
    each anchor's own text separately lets the caller compare each one
    against json on its own, so a flip on any single anchor is visible on
    its own, independent of what the other anchor(s) still say.
    """
    blocks = _paragraph_blocks(section_text)
    result = []
    for anchor in anchors:
        matches = [b for b in blocks if anchor in b]
        assert len(matches) == 1, (
            f"status anchor {anchor!r} must match exactly one paragraph/bullet "
            f"block of the .md section; found {len(matches)}. If the doc prose "
            "changed, update the anchor in LANE_STATUS_ANCHORS_MD."
        )
        result.append((anchor, header + "\n" + matches[0]))
    return result


def _lane_status_text_json(entry: dict) -> str:
    """The json fields that carry the lane's authoritative status
    declaration (PR #584 review, M1) -- NOT the whole entry via
    json.dumps(), which also pulls in evidence_artifacts paths,
    numeric_metrics run logs, and caveats, any of which can carry an
    incidental, unrelated mention of a status word that masks a real flip
    in the field that actually declares status.

    This is the FALLBACK used only for primitives absent from
    LANE_STATUS_ANCHORS_JSON (PR #588 review fix, P1): for a lane bundling
    multiple independently-evolving sub-APIs in one entry, this combined
    whole-entry string is exactly the text that flip-mutation on review
    proved could not discriminate one sub-API's own status from another's
    -- see LANE_STATUS_ANCHORS_JSON's own comment for the full finding.
    """
    return f"{entry.get('support_status', '')} {entry.get('validation_status', '')}"


def _lane_status_blocks_json(entry: dict, primitive: str) -> list[str] | None:
    """[bullet_text, ...] in the SAME ORDER as LANE_STATUS_ANCHORS_MD[primitive],
    one per LANE_STATUS_ANCHORS_JSON[primitive] entry -- or None if
    ``primitive`` is not in LANE_STATUS_ANCHORS_JSON, signaling the caller to
    fall back to the whole-entry ``_lane_status_text_json`` (unchanged
    behavior for a lane with only one status-bearing sub-API).

    Mirrors ``_lane_status_blocks_md``'s own "exactly one match" discipline,
    scanning json ``known_limits`` (a list of independent bullets) instead of
    .md paragraphs -- see LANE_STATUS_ANCHORS_JSON's own comment for why this
    exists (PR #588 review fix, P1).
    """
    json_anchors = LANE_STATUS_ANCHORS_JSON.get(primitive)
    if json_anchors is None:
        return None
    limits = entry.get("known_limits", []) or []
    result: list[str] = []
    for anchor in json_anchors:
        matches = [b for b in limits if anchor in b]
        assert len(matches) == 1, (
            f"{primitive}: json status anchor {anchor!r} must match exactly "
            f"one known_limits entry; found {len(matches)}. If known_limits "
            "changed, update LANE_STATUS_ANCHORS_JSON."
        )
        result.append(matches[0])
    return result


def _api_summary_rows(md_sections: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """[(primitive_cell, api_cell, result_cell, status_cell), ...] parsed
    from the "## API summary" table, fail-loud at every stage (issue #586
    item 2; a parser that silently parses nothing and lets the comparison
    pass vacuously is the #303 "skipped check family" class):

    - the section itself must exist,
    - exactly one header row must carry exactly the 4 expected column
      titles (a renamed/reordered/re-widened table fails here, not by
      matching zero rows),
    - every table line must split into exactly 4 cells (split on UNESCAPED
      pipes only -- status cells legitimately contain ``\\|S21\\|``),
    - at least one data row must survive.
    """
    section = md_sections.get(API_SUMMARY_HEADER)
    assert section is not None, (
        f"section {API_SUMMARY_HEADER!r} not found in the .md -- if the table "
        "moved or was renamed, update API_SUMMARY_HEADER."
    )
    header_rows = 0
    data_rows: list[tuple[str, str, str, str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", stripped[1:-1])]
        assert len(cells) == len(API_SUMMARY_EXPECTED_COLUMNS), (
            f"API summary table row does not have exactly "
            f"{len(API_SUMMARY_EXPECTED_COLUMNS)} cells (unescaped-pipe "
            f"split): {stripped!r} -> {cells}"
        )
        if all(set(c) <= {"-", ":"} and c for c in cells):
            continue
        if tuple(cells) == API_SUMMARY_EXPECTED_COLUMNS:
            header_rows += 1
            continue
        data_rows.append((cells[0], cells[1], cells[2], cells[3]))
    assert header_rows == 1, (
        f"expected exactly one API summary header row with columns "
        f"{API_SUMMARY_EXPECTED_COLUMNS}, found {header_rows} -- if the "
        "table's columns changed, update API_SUMMARY_EXPECTED_COLUMNS and "
        "re-verify what the status comparison reads."
    )
    assert data_rows, "API summary table parsed to zero data rows"
    return data_rows


def _lane_evidence_text(entry: dict) -> str:
    """The json fields that carry the lane's numeric evidence claims."""
    metrics = " ".join(entry.get("numeric_metrics", []) or [])
    ad_evidence = str(entry.get("ad_evidence", "") or "")
    return f"{metrics} {ad_evidence}"




@pytest.fixture(scope="module")
def matrix_json() -> dict:
    return json.loads(MATRIX_JSON_PATH.read_text())


@pytest.fixture(scope="module")
def matrix_json_by_primitive(matrix_json) -> dict[str, dict]:
    return {entry["primitive"]: entry for entry in matrix_json["port_families"]}


@pytest.fixture(scope="module")
def matrix_md_sections() -> dict[str, str]:
    return _md_sections(MATRIX_MD_PATH.read_text())




def test_lane_map_is_complete_in_both_directions(matrix_json_by_primitive):
    """Every json port_families primitive must be either actively checked
    (LANE_SECTION_MAP) or explicitly deferred (UNMAPPED_LANES_TODO). A new
    primitive in neither dict fails here loudly instead of silently escaping
    the parity gate (issue #554).
    The not-a-port families stay deferred because their numeric_metrics value carries no numeric token to check."""
    json_primitives = set(matrix_json_by_primitive)
    mapped = set(LANE_SECTION_MAP)
    todo = set(UNMAPPED_LANES_TODO)

    overlap = mapped & todo
    assert not overlap, f"primitive(s) listed as BOTH mapped and TODO: {overlap}"

    accounted_for = mapped | todo
    missing = json_primitives - accounted_for
    assert not missing, (
        "json port_families primitive(s) not present in LANE_SECTION_MAP or "
        f"UNMAPPED_LANES_TODO -- add a mapping or a named TODO: {sorted(missing)}"
    )

    stale = accounted_for - json_primitives
    assert not stale, (
        "LANE_SECTION_MAP/UNMAPPED_LANES_TODO reference primitive(s) no "
        f"longer present in the json: {sorted(stale)}"
    )


@pytest.mark.docs_consistency
def test_lane_map_headers_exist_in_markdown(matrix_md_sections):
    missing = [h for h in LANE_SECTION_MAP.values() if h not in matrix_md_sections]
    assert not missing, f"LANE_SECTION_MAP header(s) not found in the .md: {missing}"


def test_lane_map_targets_are_unique_sections():
    headers = list(LANE_SECTION_MAP.values())
    assert len(headers) == len(set(headers)), (
        "two primitives in LANE_SECTION_MAP point at the same .md section"
    )


def test_json_status_anchors_reference_mapped_primitives_with_matching_arity():
    """LANE_STATUS_ANCHORS_JSON keys must be mapped primitives (a typo'd or
    stale key would otherwise silently never run), and each primitive's json
    anchor count must match its md anchor count 1:1 (same check the main
    test also runs per-primitive at collection time; this version fails even
    if pytest parametrization ever skips a primitive, e.g. via -k)."""
    unmapped = set(LANE_STATUS_ANCHORS_JSON) - set(LANE_SECTION_MAP)
    assert all(token not in anchor.lower() for mapping in (LANE_STATUS_ANCHORS_MD, LANE_STATUS_ANCHORS_JSON) for anchors in mapping.values() for anchor in anchors for token in STATUS_TOKENS), "Status anchors must exclude tracked status words"
    assert not unmapped, (
        f"LANE_STATUS_ANCHORS_JSON references primitive(s) not in "
        f"LANE_SECTION_MAP: {unmapped}"
    )
    mismatched = {
        primitive: (len(json_anchors), len(LANE_STATUS_ANCHORS_MD.get(primitive, ())))
        for primitive, json_anchors in LANE_STATUS_ANCHORS_JSON.items()
        if len(json_anchors) != len(LANE_STATUS_ANCHORS_MD.get(primitive, ()))
    }
    assert not mismatched, (
        f"LANE_STATUS_ANCHORS_JSON/LANE_STATUS_ANCHORS_MD anchor-count "
        f"mismatch (primitive -> (json_count, md_count)): {mismatched}"
    )




@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
@pytest.mark.docs_consistency
def test_json_numeric_claims_appear_in_markdown(
    primitive, header, matrix_json_by_primitive, matrix_md_sections
):
    """Every distinctive numeric token in the json lane's numeric_metrics/
    ad_evidence must appear verbatim (post-normalization) somewhere in the
    mapped .md section. Catches the #559 R1 drift shape: a number updated in
    one carrier (or removed) and left stale/absent in the other. See the
    module docstring (M2) for what this does NOT catch: value reassignment.
    The thresholds in Floquet's tests and the transition's tests bound synthetic inputs or a different fixture, not the result the matrix describes."""
    entry = matrix_json_by_primitive[primitive]
    json_tokens = _extract_numeric_tokens(_lane_evidence_text(entry))

    if primitive not in NUMBER_FREE_LANES:
        assert json_tokens, (
            f"{primitive}: numeric_metrics/ad_evidence produced NO extractable "
            "numeric tokens (PR #584 review, L4). This is either deleted/emptied "
            "evidence content (fix the json) or a genuinely number-free lane "
            "(add it to NUMBER_FREE_LANES with a comment saying why) -- an "
            "empty token set otherwise makes the check below a silent vacuous "
            "pass."
        )

    md_tokens = _extract_numeric_tokens(matrix_md_sections[header])

    missing = sorted(json_tokens - md_tokens)
    assert not missing, (
        f"{primitive}: numeric claim(s) in the json numeric_metrics/ad_evidence "
        f"not found anywhere in the .md section {header!r}: {missing}"
    )


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
@pytest.mark.docs_consistency
def test_markdown_run_ids_appear_in_json(
    primitive, header, matrix_json_by_primitive, matrix_md_sections
):
    """Run-ID-shaped tokens (VESSL job numbers) referenced in the .md section
    must also be traceable in the json lane's numeric_metrics/ad_evidence.
    This is the narrow reverse-direction check: general md prose legitimately
    carries far more numeric detail than the json summary (setup
    restrictions, caveats -- checking those bidirectionally produced false
    positives during construction), but a provenance run ID quoted in only
    one carrier is unreviewable from the other. Only load-bearing where the
    .md section actually names a VESSL run at all (today: MSL only -- the
    other 4 lanes pass vacuously, see the module docstring, L5)."""
    entry = matrix_json_by_primitive[primitive]
    json_ids = _extract_vessl_run_ids(_lane_evidence_text(entry))
    md_ids = _extract_vessl_run_ids(matrix_md_sections[header])

    missing = sorted(md_ids - json_ids)
    assert not missing, (
        f"{primitive}: VESSL run ID(s) referenced in the .md section {header!r} "
        f"not found in the json numeric_metrics/ad_evidence: {missing}"
    )




@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
@pytest.mark.docs_consistency
def test_status_token_polarity_agrees(
    primitive, header, matrix_json_by_primitive, matrix_md_sections
):
    """A status token's PRESENCE must agree between carriers: if the json
    side calls something experimental/superseded/unresolved/validated with
    scope, EACH of the mapped .md section's designated status anchors
    (LANE_STATUS_ANCHORS_MD) must say so too, and vice versa -- checked per
    anchor, not as one combined blob. Catches a status flip landed in only
    one carrier (PR #581 review finding: matching-phrasing requirements
    enforced by nothing) -- scoped to the fields/paragraphs that actually
    declare status (PR #584 review, M1) after the original whole-entry/
    whole-section version was shown unable to catch a headline flip on the
    coaxial-port lane specifically, and a first per-lane-combined-anchor
    revision was shown unable to catch a flip confined to just ONE of that
    lane's two anchors (both in the module docstring).

    The json side of the comparison is PER-ANCHOR when the primitive is in
    LANE_STATUS_ANCHORS_JSON (PR #588 review fix, P1 -- BLOCKING): a single
    whole-entry support_status+validation_status string was proven by
    mutation to be constant-True/False regardless of either sub-API's own
    status once the coaxial-port lane's json entry started separately
    stating BOTH sub-APIs' status words in that combined text (see
    LANE_STATUS_ANCHORS_JSON's own comment for the finding and the mutation
    outputs). Falls back to the original whole-entry text for any primitive
    not listed there (unchanged behavior for a lane with only one
    status-bearing sub-API).
    A carrier that silently reverted the validated-with-scope status word would go undetected by the experimental token alone."""
    entry = matrix_json_by_primitive[primitive]
    json_blocks = _lane_status_blocks_json(entry, primitive)
    whole_entry_json_text = _lane_status_text_json(entry).lower()
    anchor_blocks = _lane_status_blocks_md(
        matrix_md_sections[header], header, LANE_STATUS_ANCHORS_MD[primitive]
    )
    if json_blocks is not None:
        assert len(json_blocks) == len(anchor_blocks), (
            f"{primitive}: LANE_STATUS_ANCHORS_JSON has {len(json_blocks)} "
            f"anchor(s) but LANE_STATUS_ANCHORS_MD has {len(anchor_blocks)} -- "
            "the two must pair 1:1, same order."
        )

    disagreements = []
    for i, (anchor, block_text) in enumerate(anchor_blocks):
        json_text = json_blocks[i].lower() if json_blocks is not None else whole_entry_json_text
        block_text = block_text.lower()
        for token in STATUS_TOKENS:
            in_json = token in json_text
            in_md = token in block_text
            if in_json != in_md:
                disagreements.append((anchor, token, "json-only" if in_json else "md-only"))

    assert not disagreements, (
        f"{primitive}: status token polarity disagreement "
        f"(anchor, token, which carrier has it): {disagreements}"
    )


def test_api_summary_row_map_covers_all_mapped_lanes():
    """API_SUMMARY_ROW_ANCHORS must cover exactly the mapped primitives, with
    one row key per md status anchor (1:1, same order) -- the same
    completeness discipline test_json_status_anchors_... applies to the json
    side. A mapped lane missing here would silently escape the table check;
    a stale key would silently never run."""
    assert set(API_SUMMARY_ROW_ANCHORS) == set(LANE_SECTION_MAP), (
        "API_SUMMARY_ROW_ANCHORS keys must equal LANE_SECTION_MAP keys; "
        f"missing={set(LANE_SECTION_MAP) - set(API_SUMMARY_ROW_ANCHORS)}, "
        f"stale={set(API_SUMMARY_ROW_ANCHORS) - set(LANE_SECTION_MAP)}"
    )
    mismatched = {
        primitive: (len(row_keys), len(LANE_STATUS_ANCHORS_MD.get(primitive, ())))
        for primitive, row_keys in API_SUMMARY_ROW_ANCHORS.items()
        if len(row_keys) != len(LANE_STATUS_ANCHORS_MD.get(primitive, ()))
    }
    assert not mismatched, (
        f"API_SUMMARY_ROW_ANCHORS/LANE_STATUS_ANCHORS_MD count mismatch "
        f"(primitive -> (row_key_count, md_anchor_count)): {mismatched}"
    )


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
@pytest.mark.docs_consistency
def test_api_summary_status_cell_agrees_with_section_status(
    primitive, header, matrix_md_sections
):
    """Third comparison (issue #586 item 2), alongside json-vs-md-section
    numeric parity and status polarity: the "## API summary" table's status
    cell for each mapped (primitive, calculation API) row must agree with
    the mapped section's designated status block (LANE_STATUS_ANCHORS_MD) on
    STATUS_TOKENS presence -- row key i paired with anchor i. Catches the
    demonstrated #584 drift shape: the summary table and the detailed
    section, hundreds of lines apart in the SAME file, stating different
    statuses for the same API. The anchor side includes the section header
    (same semantics as the json-vs-md check): several lanes carry their
    loudest status marker there."""
    rows = _api_summary_rows(matrix_md_sections)
    row_keys = API_SUMMARY_ROW_ANCHORS[primitive]
    anchor_blocks = _lane_status_blocks_md(
        matrix_md_sections[header], header, LANE_STATUS_ANCHORS_MD[primitive]
    )
    assert len(row_keys) == len(anchor_blocks), (
        f"{primitive}: {len(row_keys)} summary row key(s) but "
        f"{len(anchor_blocks)} md status anchor(s) -- must pair 1:1, same order."
    )

    disagreements = []
    for row_key, (anchor, block_text) in zip(row_keys, anchor_blocks):
        matches = [r for r in rows if row_key in f"{r[0]} | {r[1]}"]
        assert len(matches) == 1, (
            f"{primitive}: API summary row key {row_key!r} must match exactly "
            f"one table row; found {len(matches)}. If the table changed, "
            "update API_SUMMARY_ROW_ANCHORS."
        )
        status_cell = matches[0][3].lower()
        block_lower = block_text.lower()
        for token in STATUS_TOKENS:
            in_table = token in status_cell
            in_section = token in block_lower
            if in_table != in_section:
                disagreements.append(
                    (row_key, anchor, token, "table-only" if in_table else "section-only")
                )

    assert not disagreements, (
        f"{primitive}: API-summary-table vs section status token disagreement "
        f"(row key, md anchor, token, which side has it): {disagreements}"
    )


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
@pytest.mark.docs_consistency
def test_ad_traceable_no_is_not_overclaimed_in_markdown(
    primitive, header, matrix_json_by_primitive, matrix_md_sections
):
    """If the json says ad_traceable: 'no' for a lane, its mapped .md section
    must not carry an unqualified differentiability overclaim."""
    entry = matrix_json_by_primitive[primitive]
    if entry.get("ad_traceable") != "no":
        pytest.skip(f"{primitive}: ad_traceable != 'no', overclaim check not applicable")

    md_text = matrix_md_sections[header].lower()
    hits = [phrase for phrase in _OVERCLAIM_PHRASES if phrase in md_text]
    assert not hits, (
        f"{primitive}: json says ad_traceable=no but .md section {header!r} "
        f"contains an unqualified differentiability claim: {hits}"
    )

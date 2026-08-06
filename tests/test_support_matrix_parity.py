"""Structural parity gate between the two S-parameter support-matrix carriers.

``docs/guides/sparameter_support_matrix.md`` (prose) and
``docs/guides/sparameter_support_matrix.json`` (machine-readable, consumed by
``test_sparameter_support_contract.py``) describe the same evidence and are
supposed to say the same thing, but nothing enforced that until now (issue
#554, filed from a #553 adversarial review: the two carriers drifted TWICE
after filing and were hand-fixed twice -- PR #559's fix commit updated only
the .md, and PR #581's review found matching-phrasing requirements enforced
by nothing).

Design, deliberately narrow (see issue #554's own scoping and the false
positives this file's own construction ran into before landing):

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
- Run IDs (VESSL job numbers) ARE checked in both directions: they are
  unambiguous provenance identifiers, not general physics quantities, so a
  run ID quoted in only one carrier is unreviewable from the other and is
  exactly the "run ID present in one carrier only" drift class this gate
  must catch.
- Status tokens (experimental / superseded / unresolved) must AGREE in
  presence between a lane's full json entry and its mapped .md section
  (section header included -- several sections put their loudest status
  marker directly in the header, e.g. "## Coax<->MSL transition --
  EXPERIMENTAL"). "pending" was tried and dropped during construction: the
  json msl entry uses it ("...pending issue #519") where the .md paraphrases
  the same fact without the literal word, which is not drift and would have
  been a permanent false-positive tripwire.
- The section map (json ``primitive`` -> .md "## " header) is explicit and
  asserted complete in both directions for the lanes it covers. Lanes not
  yet mapped must be named in ``UNMAPPED_LANES_TODO``; a new json primitive
  in neither dict fails ``test_lane_map_is_complete_in_both_directions``
  loudly instead of silently escaping the gate.

Not in scope (see the issue): auto-generating one carrier from the other,
byte/full-text parity, and covering every lane immediately -- the map grows
incrementally as drift surfaces in a given lane (issue #554's explicit
scoping: start with the lanes that drifted this week and generalize if it
holds up).

Pure text parsing, no FDTD -- joins the contract-test family (same default
marker/lane as test_sparameter_support_contract.py; no slow/gpu marker).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

MATRIX_JSON_PATH = Path("docs/guides/sparameter_support_matrix.json")
MATRIX_MD_PATH = Path("docs/guides/sparameter_support_matrix.md")


# ---------------------------------------------------------------------------
# Section map: json `primitive` -> .md "## " section header
# ---------------------------------------------------------------------------

# Lanes this parity gate actively checks -- the ones this week's drift
# actually hit (issue #554): MSL (ad_evidence), coax two-port, coax-MSL
# transition, wire/patch, and the waveguide near-cutoff group-delay figures
# (NU broad-E4/E5 lives in the same "## Rectangular-waveguide port" section).
LANE_SECTION_MAP: dict[str, str] = {
    "add_port(extent=...)": "## Wire port",
    "add_msl_port(...)": "## Microstrip-line port",
    "add_waveguide_port(...)": "## Rectangular-waveguide port",
    "add_coaxial_port(...)": "## Coaxial port",
    "add_coaxial_port(...) + add_msl_port(...) driven by compute_coax_msl_transition(...)":
        "## Coax<->MSL transition — EXPERIMENTAL, diagnostic-only (issue #489 leg 4)",
}

# json port_families primitives NOT yet mapped above, each with a short
# reason. Adding a primitive to the json's port_families without adding it to
# EITHER this dict or LANE_SECTION_MAP fails
# test_lane_map_is_complete_in_both_directions loudly rather than silently
# escaping the parity gate.
UNMAPPED_LANES_TODO: dict[str, str] = {
    "add_port(extent=None)": "TODO(#554): lumped-port section not yet mapped",
    "add_floquet_port(...)": "TODO(#554): Floquet section not yet mapped",
    "add_source(...) / add_polarized_source(...)":
        "TODO(#554): not-a-port family, no numeric_metrics field to compare",
    "add_tfsf_source(...)":
        "TODO(#554): not-a-port family, no numeric_metrics field to compare",
    "add_probe(...) / add_dft_plane_probe(...) / add_flux_monitor(...)":
        "TODO(#554): not-a-port family, no numeric_metrics field to compare",
    "add_port(...) + add_msl_port(...) driven by compute_mixed_s_matrix(...)":
        "TODO(#554): mixed lumped/wire+MSL lane not yet mapped",
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# A "distinctive" numeric token: decimals (optional scientific-notation
# exponent and/or trailing '%'), bare scientific notation, bare percentages,
# and long (>=4 digit) bare integers (run IDs, dates). Short bare integers
# ("2 ports", "3 probes") are deliberately excluded: common enough across
# unrelated sentences to add noise without adding drift-detection value
# (issue #554's "small, high-value token set" instruction). The `(?<!\d)`
# lookbehind stops a hyphenated range like "0.0002-0.0146" from being
# misread as a negative number ("-0.0146").
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<!\d)-?\d+\.\d+(?:[eE][+-]?\d+)?%?"
    r"|(?<!\d)-?\d+[eE][+-]?\d+"
    r"|\d+%"
    r"|\b\d{4,}\b"
)

# Run-ID extraction is scoped to the literal "VESSL" keyword rather than any
# bare long integer: a generic \d{4,} reverse check false-positived on plain
# dates (e.g. "promoted 2026-08-04" in the coaxial-port section) during this
# gate's construction. VESSL job numbers are always quoted near the word
# "VESSL" in this document ("VESSL 123.../456...", "VESSL run 123...").
_VESSL_KEYWORD_RE = re.compile(r"VESSL\b")
_RUN_ID_DIGITS_RE = re.compile(r"\d{6,}")

# Status tokens checked for presence-polarity agreement. Verified against
# live content before landing this gate (see module docstring for why
# "pending" was tried and dropped).
STATUS_TOKENS = ("experimental", "superseded", "unresolved")

_OVERCLAIM_PHRASES = (
    "differentiable end-to-end",
    "end-to-end differentiable",
    "fully differentiable",
    "ad-traceable",
)


def _normalize_numeric_token(token: str) -> tuple[float, bool]:
    """(value, was_percent) so '0.10' and '0.100' compare equal but 21 and
    21% do not collide."""
    is_percent = token.endswith("%")
    core = token[:-1] if is_percent else token
    return (round(float(core), 12), is_percent)


def _extract_numeric_tokens(text: str) -> set[tuple[float, bool]]:
    return {_normalize_numeric_token(m.group(0)) for m in _NUMERIC_TOKEN_RE.finditer(text)}


def _extract_vessl_run_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for m in _VESSL_KEYWORD_RE.finditer(text):
        window = text[m.end(): m.end() + 60]
        ids.update(_RUN_ID_DIGITS_RE.findall(window))
    return ids


def _md_sections(md_text: str) -> dict[str, str]:
    """Split the matrix markdown into {'## Header text': 'header\\nbody'}.

    The header line is included in the section text (not just the body):
    several sections carry their loudest status marker directly in the
    header (e.g. "## Coax<->MSL transition -- EXPERIMENTAL, diagnostic-only"),
    and excluding it produced a false-positive status-token mismatch during
    this gate's construction.
    """
    sections: dict[str, str] = {}
    header: str | None = None
    body_lines: list[str] = []
    for line in md_text.splitlines():
        if line.startswith("## "):
            if header is not None:
                sections[header] = header + "\n" + "\n".join(body_lines)
            header = line.strip()
            body_lines = []
        else:
            body_lines.append(line)
    if header is not None:
        sections[header] = header + "\n" + "\n".join(body_lines)
    return sections


def _lane_evidence_text(entry: dict) -> str:
    """The json fields that carry the lane's numeric evidence claims."""
    metrics = " ".join(entry.get("numeric_metrics", []) or [])
    ad_evidence = str(entry.get("ad_evidence", "") or "")
    return f"{metrics} {ad_evidence}"


def _lane_full_text(entry: dict) -> str:
    """Every string-bearing field, for status-token polarity checks."""
    return json.dumps(entry)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix_json() -> dict:
    return json.loads(MATRIX_JSON_PATH.read_text())


@pytest.fixture(scope="module")
def matrix_json_by_primitive(matrix_json) -> dict[str, dict]:
    return {entry["primitive"]: entry for entry in matrix_json["port_families"]}


@pytest.fixture(scope="module")
def matrix_md_sections() -> dict[str, str]:
    return _md_sections(MATRIX_MD_PATH.read_text())


# ---------------------------------------------------------------------------
# Section-map completeness
# ---------------------------------------------------------------------------


def test_lane_map_is_complete_in_both_directions(matrix_json_by_primitive):
    """Every json port_families primitive must be either actively checked
    (LANE_SECTION_MAP) or explicitly deferred (UNMAPPED_LANES_TODO). A new
    primitive in neither dict fails here loudly instead of silently escaping
    the parity gate (issue #554)."""
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


def test_lane_map_headers_exist_in_markdown(matrix_md_sections):
    missing = [h for h in LANE_SECTION_MAP.values() if h not in matrix_md_sections]
    assert not missing, f"LANE_SECTION_MAP header(s) not found in the .md: {missing}"


def test_lane_map_targets_are_unique_sections():
    headers = list(LANE_SECTION_MAP.values())
    assert len(headers) == len(set(headers)), (
        "two primitives in LANE_SECTION_MAP point at the same .md section"
    )


# ---------------------------------------------------------------------------
# Numeric parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
def test_json_numeric_claims_appear_in_markdown(
    primitive, header, matrix_json_by_primitive, matrix_md_sections
):
    """Every distinctive numeric token in the json lane's numeric_metrics/
    ad_evidence must appear verbatim (post-normalization) somewhere in the
    mapped .md section. Catches the #559 R1 drift shape: a number updated in
    one carrier (or removed) and left stale/absent in the other."""
    entry = matrix_json_by_primitive[primitive]
    json_tokens = _extract_numeric_tokens(_lane_evidence_text(entry))
    md_tokens = _extract_numeric_tokens(matrix_md_sections[header])

    missing = sorted(json_tokens - md_tokens)
    assert not missing, (
        f"{primitive}: numeric claim(s) in the json numeric_metrics/ad_evidence "
        f"not found anywhere in the .md section {header!r}: {missing}"
    )


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
def test_markdown_run_ids_appear_in_json(
    primitive, header, matrix_json_by_primitive, matrix_md_sections
):
    """Run-ID-shaped tokens (VESSL job numbers) referenced in the .md section
    must also be traceable in the json lane's numeric_metrics/ad_evidence.
    This is the narrow reverse-direction check: general md prose legitimately
    carries far more numeric detail than the json summary (setup
    restrictions, caveats -- checking those bidirectionally produced false
    positives during construction), but a provenance run ID quoted in only
    one carrier is unreviewable from the other."""
    entry = matrix_json_by_primitive[primitive]
    json_ids = _extract_vessl_run_ids(_lane_evidence_text(entry))
    md_ids = _extract_vessl_run_ids(matrix_md_sections[header])

    missing = sorted(md_ids - json_ids)
    assert not missing, (
        f"{primitive}: VESSL run ID(s) referenced in the .md section {header!r} "
        f"not found in the json numeric_metrics/ad_evidence: {missing}"
    )


# ---------------------------------------------------------------------------
# Status-token polarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
def test_status_token_polarity_agrees(
    primitive, header, matrix_json_by_primitive, matrix_md_sections
):
    """A status token's PRESENCE must agree between carriers: if the json
    lane's own text calls something experimental/superseded/unresolved, the
    mapped .md section must say so too, and vice versa. Catches a status
    flip landed in only one carrier (PR #581 review finding: matching-
    phrasing requirements enforced by nothing)."""
    entry = matrix_json_by_primitive[primitive]
    json_text = _lane_full_text(entry).lower()
    md_text = matrix_md_sections[header].lower()

    disagreements = []
    for token in STATUS_TOKENS:
        in_json = token in json_text
        in_md = token in md_text
        if in_json != in_md:
            disagreements.append((token, "json-only" if in_json else "md-only"))

    assert not disagreements, (
        f"{primitive}: status token polarity disagreement: {disagreements}"
    )


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
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

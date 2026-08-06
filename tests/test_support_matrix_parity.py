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

Not in scope (see the issue, and issue #586 filed from the PR #584 review for
the deferred items): auto-generating one carrier from the other, byte/full-
text parity, covering every lane immediately (the map grows incrementally as
drift surfaces in a given lane), and checking the separate "## API summary"
table (a second, currently-ungated in-file copy of the same per-primitive
claims -- issue #586).

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
# escaping the parity gate. issue #586 (filed from the PR #584 review) tracks
# promoting the 3 genuinely mappable lanes below (lumped, floquet, mixed) into
# LANE_SECTION_MAP; the remaining 3 are "not_a_port" families whose
# numeric_metrics field HOLDS a value -- ``["not an S-parameter calculation"]``
# -- but that value carries no numeric token to check, which is why they stay
# deferred rather than mapped.
UNMAPPED_LANES_TODO: dict[str, str] = {
    "add_port(extent=None)": "TODO(#586): lumped-port section not yet mapped",
    "add_floquet_port(...)": "TODO(#586): Floquet section not yet mapped",
    "add_source(...) / add_polarized_source(...)":
        "TODO(#586): not-a-port family, numeric_metrics carries no numeric tokens",
    "add_tfsf_source(...)":
        "TODO(#586): not-a-port family, numeric_metrics carries no numeric tokens",
    "add_probe(...) / add_dft_plane_probe(...) / add_flux_monitor(...)":
        "TODO(#586): not-a-port family, numeric_metrics carries no numeric tokens",
    "add_port(...) + add_msl_port(...) driven by compute_mixed_s_matrix(...)":
        "TODO(#586): mixed lumped/wire+MSL lane not yet mapped",
}

# Lanes explicitly exempt from the "numeric_metrics must extract at least one
# token" floor in test_json_numeric_claims_appear_in_markdown (PR #584
# review, L4). Empty on purpose: every currently-mapped lane has real
# numeric_metrics content: an empty allowlist means the floor is live on all
# 5 mapped lanes today. Add a primitive here only with a comment naming why
# its numeric_metrics is genuinely number-free (not merely emptied by a
# careless edit, which is exactly the silent-vacuous-pass class L4 closes).
NUMBER_FREE_LANES: frozenset[str] = frozenset()

# Per-lane, hand-identified .md status anchors for test_status_token_polarity_agrees
# (PR #584 review, M1). Each anchor is a literal substring that must appear
# in exactly one whitespace-normalized paragraph/bullet block of the mapped
# .md section; that block (not the whole section) is what gets scanned for
# STATUS_TOKENS. Most lanes need only one anchor (the section states its own
# status once). The coaxial-port lane needs two: its one .md section and one
# json entry bundle TWO sub-APIs with independently evolving status
# (compute_coaxial_s_matrix, deprecated; compute_coaxial_two_port,
# experimental) -- see the module docstring for why scanning only one of
# the two, or the whole section, both failed review.
#
# Anchors are deliberately chosen to NOT themselves contain any STATUS_TOKENS
# word: an anchor is a stable, unique PREFIX of its designated block, so
# editing the status word downstream of the anchor (the actual drift this
# check targets) changes the block's content without breaking the anchor
# lookup that finds the block in the first place. Anchoring on text that
# itself contains "experimental" was tried first and produced a confusing
# failure mode on review: flipping that exact word made the anchor stop
# matching at all (an "anchor not found" error) instead of exercising the
# intended status-token disagreement path.
LANE_STATUS_ANCHORS_MD: dict[str, tuple[str, ...]] = {
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
        "The older `compute_coaxial_s_matrix(...)` path is deprecated and",
        "every DUT it can currently gate against",
    ),
    "add_coaxial_port(...) + add_msl_port(...) driven by compute_coax_msl_transition(...)": (
        "the junction's own physical reflection also contributes is genuinely",
    ),
}

# Per-lane, hand-identified json ``known_limits`` anchors (PR #588 review
# fix, P1 -- BLOCKING). For a primitive listed here, each anchor below must
# match EXACTLY ONE entry of that primitive's json ``known_limits`` list
# (same "exactly one" discipline ``_lane_status_blocks_md`` already applies
# to the .md side), IN THE SAME ORDER as ``LANE_STATUS_ANCHORS_MD[primitive]``
# -- anchor i here pairs with md anchor i. THAT bullet, not the whole-entry
# ``support_status``/``validation_status`` blob, is what
# ``test_status_token_polarity_agrees`` scans for STATUS_TOKENS for that
# specific md anchor. Primitives NOT listed here keep the original
# whole-entry ``_lane_status_text_json`` behavior (unchanged, still correct
# for a lane with only one status-bearing sub-API).
#
# Why this exists: the whole-entry text is exactly what PR #588's own
# promotion prose defeated, proven by mutation on review. The coaxial-port
# json entry's ``support_status``/``validation_status`` mentions
# "experimental" (for the still-deprecated ``compute_coaxial_s_matrix``)
# regardless of what ``compute_coaxial_two_port``'s own status is, so
# ``in_json`` for the token "experimental" was CONSTANT True no matter how
# the two-port sub-API's own status was written -- flipping the two-port
# json status back to "experimental" with the .md left untouched, and
# flipping the .md status back with the json left untouched, BOTH stayed
# green (19 passed) because the combined string could not distinguish
# either sub-API's own claim from the other's. A control mutation on the
# PRE-#588 single-sub-API prose (removing "experimental" from the old
# one-sub-API paragraph) correctly reds, proving the check was live before
# this bundling defeated it -- the #584-M1 text-gaming class, recurring one
# level down. Splitting the json side into its own per-anchor bullet
# (mirroring what the .md side already does) restores real discrimination:
# each sub-API's own bullet only contains ITS OWN status word, so flipping
# either carrier's own text now changes ONLY that anchor's own polarity.
#
# Anchors here are chosen the SAME way the .md anchors are (see that dict's
# own comment): a stable prefix that does NOT itself contain the status
# word being tracked, so a later status-word edit changes the bullet's
# CONTENT without breaking the anchor LOOKUP that finds the bullet. The
# two-port anchor below deliberately does not include "VALIDATED WITH
# SCOPE" for this reason -- anchoring on "...is VALIDATED WITH SCOPE"
# would stop matching the moment that phrase is edited back to
# "EXPERIMENTAL" (or anything else), producing the same confusing
# "anchor not found" failure mode the .md side's own docstring already
# warns about, instead of a clean status-disagreement report.
LANE_STATUS_ANCHORS_JSON: dict[str, tuple[str, ...]] = {
    "add_coaxial_port(...)": (
        "compute_coaxial_s_matrix(...) remains",
        "validation/crossval/21_coax_two_port_referee.py, VESSL run-3 369367251629",
    ),
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# A "distinctive" numeric token: decimals (optional scientific-notation
# exponent, optional single space, optional trailing '%'), bare scientific
# notation, bare percentages, and long (>=4 digit) bare integers (run IDs,
# dates). Short bare integers ("2 ports", "3 probes") are deliberately
# excluded: common enough across unrelated sentences to add noise without
# adding drift-detection value (issue #554's "small, high-value token set"
# instruction).
#
# The signed and unsigned forms are SEPARATE alternatives (PR #584 review,
# L1): a single shared `(?<!\d)` lookbehind on an optional leading '-' still
# let a double-hyphen range like "3.0--4.5" (this doc's en-dash substitute,
# used throughout) mint a phantom "-4.5" token, because the lookbehind only
# checked the character immediately before the '-' itself (the OTHER '-'),
# not whether a '-'-preceded digit run was actually being read as a sign.
# Splitting into "signed, only when not preceded by a digit or another
# hyphen" and "unsigned, no lookbehind" alternatives means the unsigned
# branch is always free to grab the second half of ANY hyphenated range
# (single or double '-') as a plain positive number, while a genuine negative
# ("-34.3 dB", preceded by whitespace) still matches the signed branch.
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\d-])-\d+\.\d+(?:[eE][+-]?\d+)? ?%?"  # signed decimal
    r"|\d+\.\d+(?:[eE][+-]?\d+)? ?%?"  # unsigned decimal
    r"|(?<![\d-])-\d+[eE][+-]?\d+"  # signed bare scientific notation
    r"|\d+[eE][+-]?\d+"  # unsigned bare scientific notation
    r"|\d+ ?%"  # bare integer percent
    r"|\b\d{4,}\b"  # long bare integer (run IDs, dates)
)

# Run-ID extraction is scoped to the literal "VESSL" keyword rather than any
# bare long integer, and specifically to digits immediately following it
# (PR #584 review, L2): the original design free-scanned `\d{6,}` in a
# 60-character window after each "VESSL" occurrence, which mined a phantom
# run ID ("000110") out of the fractional digits of the UNRELATED decimal
# "0.000110" a few words later in the same sentence. Anchoring to
# "VESSL[ ]+(?:run[ ]+)?<digits>" (optionally "<digits>/<digits>" for the
# two-run-per-measurement citations this doc uses) only matches digits that
# are actually the run number, not anything else nearby.
_VESSL_RUN_ID_RE = re.compile(r"VESSL\s+(?:run\s+)?(\d{6,}(?:/\d{6,})*)")

# Status tokens checked for presence-polarity agreement. Verified against
# live content before landing this gate (see module docstring for why
# "pending" was tried and dropped). "validated with scope" added in PR #588
# review fix (P1): a lane's promotion to this new state must itself be
# checked for carrier agreement, not just the retired "experimental" token
# it replaces -- without this, a carrier that silently reverted the NEW
# status word would go undetected (the reverse of what "experimental"
# alone catches).
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
    return {_normalize_numeric_token(m.group(0)) for m in _NUMERIC_TOKEN_RE.finditer(text)}


def _extract_vessl_run_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for m in _VESSL_RUN_ID_RE.finditer(text):
        ids.update(m.group(1).split("/"))
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


def _lane_evidence_text(entry: dict) -> str:
    """The json fields that carry the lane's numeric evidence claims."""
    metrics = " ".join(entry.get("numeric_metrics", []) or [])
    ad_evidence = str(entry.get("ad_evidence", "") or "")
    return f"{metrics} {ad_evidence}"


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


def test_json_status_anchors_reference_mapped_primitives_with_matching_arity():
    """LANE_STATUS_ANCHORS_JSON keys must be mapped primitives (a typo'd or
    stale key would otherwise silently never run), and each primitive's json
    anchor count must match its md anchor count 1:1 (same check the main
    test also runs per-primitive at collection time; this version fails even
    if pytest parametrization ever skips a primitive, e.g. via -k)."""
    unmapped = set(LANE_STATUS_ANCHORS_JSON) - set(LANE_SECTION_MAP)
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
    one carrier (or removed) and left stale/absent in the other. See the
    module docstring (M2) for what this does NOT catch: value reassignment."""
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


# ---------------------------------------------------------------------------
# Status-token polarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("primitive,header", sorted(LANE_SECTION_MAP.items()))
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
    status-bearing sub-API)."""
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

"""Every number the crossval evidence documents *cite* must still be the number
the committed artifact holds.

Why this exists (#812, round-1 retrospective). Issue #812 shipped a wrong number
into a durable document four times: the audit's own cv11 claim, cv10's "1.6x /
at the arithmetic floor", cv17/cv18's sign-inverted measured claim in two
committed evidence JSONs, and cv15's regenerated leg whose ``s11_dip_db`` moved
without the prose moving with it. Of nine round-1 blocking findings, six were a
wrong or falsified claim in prose rather than a defective gate.

PR #814 closed the mechanical half of the same class for *pointers*: a
``file.py:NNN`` citation must land on a constant assignment whose NAME the
document also mentions. This gate is the same treatment for *values*.

The contract
------------
A document may cite a quantity by writing, inside a single-backtick code span,
an **artifact reference**::

    `validation/crossval/_18_wr90_iris_results/rfx.json::gates.fine_gate_abs`
    `validation/crossval/_18_wr90_iris_results/rfx.json::gates.fine_gate_abs = 0.04`
    `validation/crossval/_15_patch_results/rfx.json::f_primary_hz = 2.3139 GHz`

- the path is repo-relative and must name a committed JSON file;
- the key path walks that JSON (``a.b[2].c``);
- if a literal follows ``=``, the resolved value must equal it **to the
  precision written** -- tolerance is half a unit in the last decimal place of
  the literal as typed, times the (exact, tabulated) unit multiplier. Writing
  more digits is a tighter assertion; writing a sign is a signed assertion.

Any backtick span containing ``::`` that does not parse is an error, so a
malformed reference cannot be silently skipped.

What it does NOT catch (stated plainly, because the gap is the point)
--------------------------------------------------------------------
1. **Prose claims with no number** -- "the gate cannot fire", "this is at the
   arithmetic floor", "no committed run matches". Round 1's cv05 false absolute
   is exactly this shape and is invisible here.
2. **Wrong-but-self-consistent artifacts.** If the harness writes a wrong value
   and the prose quotes it faithfully, both agree and the gate is green. This
   gate checks provenance, never physics.
3. **A number that is correct today but describes the wrong quantity** -- citing
   ``mean_mag_abs_diff`` in a sentence about the max. The value resolves, the
   sentence is still false.
4. **Numbers nobody chose to reference.** Coverage is opt-in; the census below
   is what stops it from silently shrinking, not from being incomplete.
   Bare numerics in the opted-in sections are deliberately NOT flagged: these
   ``claim_scope`` strings carry dozens of legitimate bare numerics (issue and
   PR numbers, dates, drawn dimensions, cell counts, derived percentages), so a
   bare-numeric detector here is a false-positive storm, not a gate.

It asserts NO physics and changes NO gate value.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Reference syntax
# --------------------------------------------------------------------------

# Exact multipliers. ``artifact_value == literal * UNITS[unit]``. Choosing the
# wrong one turns the gate red, never green, so the table is a convenience for
# readable prose and not a place a wrong claim can hide.
UNITS: dict[str, float] = {
    "": 1.0,
    "Hz": 1.0,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
    "dB": 1.0,
    "dBi": 1.0,
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "%": 1e-2,
}

_SPAN = re.compile(r"`([^`\n]+)`")

_REFERENCE = re.compile(
    r"^(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*\.json)"
    r"::(?P<keypath>[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_][A-Za-z0-9_-]*|\[\d+\])*)"
    r"(?:\s*=\s*(?P<literal>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>%|[A-Za-z]+))?)?$"
)

_STEP = re.compile(r"\[(\d+)\]|([A-Za-z0-9_][A-Za-z0-9_-]*)")


class Reference:
    """One ``path::keypath [= literal unit]`` citation found in a document."""

    def __init__(self, doc: str, site: str, raw: str, path: str, keypath: str,
                 literal: str | None, unit: str) -> None:
        self.doc = doc
        self.site = site
        self.raw = raw
        self.path = path
        self.keypath = keypath
        self.literal = literal
        self.unit = unit

    def __repr__(self) -> str:  # pragma: no cover - test ids only
        return f"{self.doc}::{self.site}::{self.raw}"


def strip_code_blocks(text: str) -> str:
    """Blank out fenced and indented markdown code blocks.

    A document must be able to *show* the reference syntax without the gate
    trying to resolve the illustration. Only whole code blocks are exempt;
    inline spans in running prose are always checked.
    """
    kept: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            kept.append("")
            continue
        if fenced or line.startswith("    ") or line.startswith("\t"):
            kept.append("")
            continue
        kept.append(line)
    return "\n".join(kept)


def parse_references(doc: str, site: str, text: str) -> list[Reference]:
    """Every backtick span containing ``::`` in *text*, parsed or rejected."""
    out: list[Reference] = []
    for span in _SPAN.finditer(text):
        inner = span.group(1).strip()
        if "::" not in inner:
            continue
        match = _REFERENCE.match(inner)
        if match is None:
            raise ValueError(
                f"{doc} [{site}] contains `{inner}`, which looks like an "
                f"artifact reference but does not parse. The syntax is "
                f"`path/to.json::a.b[2].c` optionally followed by "
                f"` = <number>[ <unit>]`."
            )
        if ".." in match.group("path"):
            raise ValueError(
                f"{doc} [{site}] cites `{inner}`, whose path escapes the "
                f"repository. Artifact references are repo-relative."
            )
        unit = match.group("unit") or ""
        if unit not in UNITS:
            raise ValueError(
                f"{doc} [{site}] cites `{inner}` with unit {unit!r}, which is "
                f"not in the declared multiplier table {sorted(UNITS)}."
            )
        out.append(Reference(doc, site, inner, match.group("path"),
                             match.group("keypath"), match.group("literal"), unit))
    return out


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def resolve(root: Path, ref: Reference):
    """Walk ``ref.keypath`` into the committed JSON at ``ref.path``."""
    target = root / ref.path
    if not target.exists():
        raise AssertionError(
            f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but the artifact "
            f"{ref.path} does not exist."
        )
    data = json.loads(target.read_text(encoding="utf-8"))
    node = data
    walked = ""
    for step in _STEP.finditer(ref.keypath):
        index, name = step.group(1), step.group(2)
        if index is not None:
            walked += f"[{index}]"
            if not isinstance(node, list) or int(index) >= len(node):
                raise AssertionError(
                    f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but "
                    f"{ref.path}::{walked} is not a list index that exists "
                    f"(found {type(node).__name__})."
                )
            node = node[int(index)]
        else:
            walked = f"{walked}.{name}" if walked else name
            if not isinstance(node, dict) or name not in node:
                available = sorted(node)[:12] if isinstance(node, dict) else []
                raise AssertionError(
                    f"{ref.doc} [{ref.site}] cites `{ref.raw}`, but "
                    f"{ref.path}::{walked} does not exist. "
                    f"Keys available at that level: {available}"
                )
            node = node[name]
    return node


def tolerance(literal: str, unit: str) -> tuple[float, float]:
    """Expected value and half-a-last-place tolerance for *literal*."""
    dec = Decimal(literal)
    scale = UNITS[unit]
    expected = float(dec) * scale
    exponent = dec.as_tuple().exponent
    assert isinstance(exponent, int)
    tol = 0.5 * float(Decimal(1).scaleb(exponent)) * scale
    return expected, tol


def check(root: Path, ref: Reference) -> None:
    """Resolve *ref* and, if it carries a literal, assert the value matches."""
    value = resolve(root, ref)
    if ref.literal is None:
        if isinstance(value, (dict, list)) or value is None:
            raise AssertionError(
                f"{ref.doc} [{ref.site}] cites `{ref.raw}`, which resolves to "
                f"{type(value).__name__}, not a value a document can cite."
            )
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(
            f"{ref.doc} [{ref.site}] cites `{ref.raw}` with a numeric literal, "
            f"but {ref.path}::{ref.keypath} holds {value!r} "
            f"({type(value).__name__})."
        )
    expected, tol = tolerance(ref.literal, ref.unit)
    if isinstance(value, int) and Decimal(ref.literal).as_tuple().exponent >= 0 \
            and ref.unit in ("", "Hz", "dB", "dBi", "m"):
        if value != int(Decimal(ref.literal)):
            raise AssertionError(
                f"NUMERIC PROVENANCE FAILURE\n"
                f"  document : {ref.doc} [{ref.site}]\n"
                f"  reference: `{ref.raw}`\n"
                f"  document says: {ref.literal}\n"
                f"  artifact holds: {value}\n"
                f"  the artifact holds an integer; the citation must match it "
                f"exactly."
            )
        return
    if abs(value - expected) > tol * (1.0 + 1e-9):
        raise AssertionError(
            f"NUMERIC PROVENANCE FAILURE\n"
            f"  document : {ref.doc} [{ref.site}]\n"
            f"  reference: `{ref.raw}`\n"
            f"  document says: {ref.literal}{(' ' + ref.unit) if ref.unit else ''}"
            f"  (= {expected!r} in artifact units, +/- {tol!r})\n"
            f"  artifact holds: {value!r}\n"
            f"  at {ref.path}::{ref.keypath}\n"
            f"  |difference| = {abs(value - expected)!r}. Either the artifact "
            f"moved and the document was not updated with it, or the document "
            f"restates a number it never re-derived."
        )


# --------------------------------------------------------------------------
# The opted-in surface (coverage cannot silently shrink)
# --------------------------------------------------------------------------

MANIFEST = "validation/crossval/manifest.json"
CV11_NOTE = "docs/design_notes/20260831_cv11_broad_e4_artifact_provenance.md"

# Markdown documents, with the regex that cuts them into named sites.
MARKDOWN_SITES: dict[str, str] = {
    "validation/README.md": r"^\|\s*`(crossval/[^`]+)`",
    CV11_NOTE: r"^#+\s+(.*\S)\s*$",
    "docs/design_notes/20260901_numeric_provenance_gate.md": r"^#+\s+(.*\S)\s*$",
}

DOCUMENTS = (MANIFEST, *MARKDOWN_SITES)

# Sites that MUST carry at least this many value-checked references. Lowering a
# floor is a deliberate act that belongs in the same commit as the reason.
REQUIRED_SITES: dict[tuple[str, str], int] = {
    (MANIFEST, "11_waveguide_port_wr90"): 4,
    (MANIFEST, "15_patch_antenna_rt5880"): 3,
    (MANIFEST, "17_dielectric_sphere_mie"): 2,
    (MANIFEST, "18_wr90_iris_modematch"): 4,
    (MANIFEST, "19_wr90_iris_filter_aghanim"): 4,
    ("validation/README.md", "crossval/11_waveguide_port_wr90.py"): 4,
    # 2026-09-03: floors for the cv15/cv18 README rows re-based to the rows that
    # #847 / #846 rewrote after this gate was drafted (see the note, "Re-basing
    # two floors"); those rows cite in the value-checked form and carry 2 each.
    ("validation/README.md", "crossval/15_patch_antenna_rt5880.py"): 2,
    ("validation/README.md", "crossval/17_dielectric_sphere_mie.py"): 3,
    ("validation/README.md", "crossval/18_wr90_iris_modematch.py"): 2,
    ("validation/README.md", "crossval/19_wr90_iris_filter_aghanim.py"): 6,
    (CV11_NOTE, "7. Numeric provenance (appended 2026-09-01, #812 round 2 \u2014 no finding changed)"): 6,
}

# Anti-vacuity census. A green gate must mean the references are right, not that
# somebody deleted them.
MIN_REFERENCES = 44
MIN_VALUE_CHECKED = 44
MIN_DISTINCT_ARTIFACTS = 6


def _sites(root: Path, doc: str) -> list[tuple[str, str]]:
    text = (root / doc).read_text(encoding="utf-8")
    if doc == MANIFEST:
        cases = json.loads(text)["cases"]
        return [(case["id"], json.dumps(case, ensure_ascii=False)) for case in cases]
    text = strip_code_blocks(text)
    pattern = re.compile(MARKDOWN_SITES[doc], re.MULTILINE)
    marks = list(pattern.finditer(text))
    out: list[tuple[str, str]] = []
    if marks and marks[0].start() > 0:
        out.append(("<preamble>", text[: marks[0].start()]))
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((mark.group(1), text[mark.start():end]))
    if not marks:
        out.append(("<whole document>", text))
    return out


def collect(root: Path) -> list[Reference]:
    refs: list[Reference] = []
    for doc in DOCUMENTS:
        for site, text in _sites(root, doc):
            refs.extend(parse_references(doc, site, text))
    return refs


_REFS = collect(_REPO)


@pytest.mark.parametrize(
    "ref", _REFS, ids=[f"{r.doc.split('/')[-1]}:{r.site}:{r.keypath}" for r in _REFS]
)
def test_every_cited_number_matches_its_artifact(ref: Reference) -> None:
    check(_REPO, ref)


def test_the_cited_population_is_still_present() -> None:
    value_checked = [r for r in _REFS if r.literal is not None]
    artifacts = {r.path for r in _REFS}
    assert len(_REFS) >= MIN_REFERENCES, (
        f"only {len(_REFS)} artifact references across {list(DOCUMENTS)}; "
        f"expected at least {MIN_REFERENCES}. If references were legitimately "
        f"removed, lower MIN_REFERENCES in the same commit and say why."
    )
    assert len(value_checked) >= MIN_VALUE_CHECKED, (
        f"only {len(value_checked)} of {len(_REFS)} references carry a value to "
        f"check; expected at least {MIN_VALUE_CHECKED}. Existence-only "
        f"references keep a key alive but assert no number."
    )
    assert len(artifacts) >= MIN_DISTINCT_ARTIFACTS, (
        f"references reach only {len(artifacts)} distinct artifacts "
        f"({sorted(artifacts)}); expected at least {MIN_DISTINCT_ARTIFACTS}."
    )


@pytest.mark.parametrize("site,floor", sorted(REQUIRED_SITES.items()),
                         ids=lambda v: str(v) if not isinstance(v, tuple) else f"{v[0].split('/')[-1]}:{v[1]}")
def test_each_registered_site_still_carries_its_references(site, floor) -> None:
    doc, name = site
    found = [r for r in _REFS if r.doc == doc and r.site == name and r.literal is not None]
    assert len(found) >= floor, (
        f"{doc} [{name}] carries {len(found)} value-checked artifact "
        f"references, below its declared floor of {floor}. A rewrite that drops "
        f"references drops the only mechanical link between this claim and its "
        f"evidence."
    )


# --------------------------------------------------------------------------
# Criterion (B): the gate fails on the round-1 defects it exists for.
#
# These run against a scratch tree, never the repo, so they assert detection
# power without touching a committed artifact.
# --------------------------------------------------------------------------

# The measured round-1 cv15 finding: the lane regenerated the committed rfx leg
# and `s11_dip_db` moved from the value the documents quote to this one, with no
# prose moving with it.
CV15_ARTIFACT = "validation/crossval/_15_patch_results/rfx.json"
CV15_KEY = "s11_dip_db"
CV15_REGENERATED_VALUE = -0.3448069095611572


def _scratch(tmp_path: Path, artifact: str, key: str, value) -> Path:
    """A tree holding one artifact with one leaf replaced."""
    data = json.loads((_REPO / artifact).read_text(encoding="utf-8"))
    assert key in data, f"{artifact} no longer has a top-level {key}"
    data[key] = value
    dst = tmp_path / artifact
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def _cv15_dip_references() -> list[Reference]:
    return [r for r in _REFS
            if r.path == CV15_ARTIFACT and r.keypath == CV15_KEY and r.literal is not None]


def test_the_gate_fires_on_the_measured_cv15_regression(tmp_path: Path) -> None:
    """Round 1's cv15 defect, reproduced: artifact moves, prose does not."""
    refs = _cv15_dip_references()
    assert refs, (
        f"no committed document cites {CV15_ARTIFACT}::{CV15_KEY}; the (B) arm "
        f"of this gate's falsifier has nothing to fire on."
    )
    root = _scratch(tmp_path, CV15_ARTIFACT, CV15_KEY, CV15_REGENERATED_VALUE)
    for ref in refs:
        try:
            check(root, ref)  # (B): must go red on the regenerated leg
        except AssertionError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                f"the gate did NOT fire: {ref.doc} [{ref.site}] now cites the "
                f"regenerated value {CV15_REGENERATED_VALUE!r} itself, so this "
                f"falsifier no longer falsifies. If cv15's leg was legitimately "
                f"regenerated, re-anchor criterion (B) on another measured "
                f"defect in the same commit that explains the move -- do not "
                f"delete the arm."
            )
        assert ref.doc in message
        assert ref.raw in message
        assert ref.literal in message                       # what the document says
        assert repr(CV15_REGENERATED_VALUE) in message      # what the artifact holds


def test_the_gate_fires_on_a_sign_inversion(tmp_path: Path) -> None:
    """Round 1's cv17/cv18 shape: the magnitude survives, the sign is inverted."""
    refs = _cv15_dip_references()
    assert refs
    ref = refs[0]
    expected, _tol = tolerance(ref.literal, ref.unit)
    root = _scratch(tmp_path, CV15_ARTIFACT, CV15_KEY, -expected)
    with pytest.raises(AssertionError) as caught:
        check(root, ref)
    assert repr(-expected) in str(caught.value)


def test_the_gate_fires_on_a_drift_below_the_last_digit_written(tmp_path: Path) -> None:
    """Precision is what was typed: a move under half a last place is silent,
    a move over it is not. This pins the boundary rather than assuming it.

    Anchored on the literal the document carries, not on the tree, so it
    measures the rule and not today's artifact."""
    ref = _cv15_dip_references()[0]
    expected, tol = tolerance(ref.literal, ref.unit)
    inside = _scratch(tmp_path / "in", CV15_ARTIFACT, CV15_KEY, expected + 0.4 * tol)
    check(inside, ref)
    outside = _scratch(tmp_path / "out", CV15_ARTIFACT, CV15_KEY, expected + 2.0 * tol)
    with pytest.raises(AssertionError):
        check(outside, ref)


def test_an_unresolvable_key_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """A renamed key must fail loudly; a silent skip is how coverage rots."""
    ref = _cv15_dip_references()[0]
    data = json.loads((_REPO / CV15_ARTIFACT).read_text(encoding="utf-8"))
    data.pop(CV15_KEY)
    dst = tmp_path / CV15_ARTIFACT
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AssertionError, match="does not exist"):
        check(tmp_path, ref)


def test_a_malformed_reference_is_rejected() -> None:
    """A typo in the syntax must not read as 'no reference here'."""
    with pytest.raises(ValueError, match="does not parse"):
        parse_references("doc", "site", "the value is `results/x.json::a b c`")
    with pytest.raises(ValueError, match="not in the declared multiplier table"):
        parse_references("doc", "site", "the value is `results/x.json::a = 1.0 furlongs`")

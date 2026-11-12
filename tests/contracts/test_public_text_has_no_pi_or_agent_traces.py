"""Public source text states rules and facts without private attribution."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from time import perf_counter

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCOPE = ("rfx/", "tests/", "scripts/")
pytestmark = pytest.mark.source_scan

PATTERNS = {
    "C1": re.compile(r"/Users/byungkwankim/|-Users-byungkwankim|byungkwan-workspace|runs/byungkwan/|Project: byungkwan"),
    "C2": re.compile(r"\b(?:PI|PI-directed|PI-gated|PI-approved)\b|the professor"),
    "C3": re.compile(r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]"),
    "C4": re.compile(
        r"rfx-archive|agent-rules|pi-desk|rfx-lead|rfx-core-desk|\b[DL]:[0-9a-f]{6}\b"
        r"|MSL-PLANE\.md|rfx/records/|ledger archive|internal issue ledger|rfx-known-issues|\brecords?[ /]20\d{6}-[a-z]"
        r"|docs/research_notes/|docs/agent-memory|rfx-research|rfx-adv|rfx-fdfd-research"
        r"|bk-workspace|remilab-cnu/paper"
        r"|(?<![\w-])acc-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-\d{8}t\d{6}z-[0-9a-f]+(?![\w-])"
    ),
}
# Deliberately broader than PATTERNS; Git selects lines, Python judges them.
CANDIDATE_PATTERN = (
    r"byungkwan|/Users/|PI|professor|rfx|agent-rules|pi-desk|[DL]:|MSL-PLANE"
    r"|ledger|records?[ /]20|docs/agent-memory|research_notes|bk-workspace|remilab-cnu|acc-"
    r"|[^ -~]"  # All non-ASCII text includes every Hangul range, in any locale.
)
MATH_PI = re.compile(r"\bPI(?=\s*=)|\b2\s*\*\s*PI\b")


def matching_classes(line):
    """Allow mathematical uses without hiding attribution elsewhere on a line."""
    return [name for name, pattern in PATTERNS.items()
            if pattern.search(MATH_PI.sub("", line) if name == "C2" and "PI" in line else line)]


def test_public_text_has_no_pi_or_agent_traces():
    started = perf_counter()
    result = subprocess.run(
        ["git", "grep", "-n", "-I", "-z", "-E", "-i", CANDIDATE_PATTERN,
         "--", *SCOPE], cwd=ROOT, capture_output=True,
    )
    selected = perf_counter()
    assert result.returncode in (0, 1), result.stderr.decode(errors="replace")
    errors = []
    for record in result.stdout.split(b"\n"):
        if not record:
            continue
        name, number, source = record.split(b"\0", 2)
        name = name.decode("utf-8")
        try:
            line = source.decode("utf-8")
        except UnicodeDecodeError:
            continue
        exceptions = tuple(literal for path, literal, reason in NAMED_EXCEPTIONS
                           if path == name)
        if exceptions and any(literal in line for literal in exceptions):
            continue
        if classes := matching_classes(line):
            errors.append(f"{name}:{number.decode()}: {', '.join(classes)}: {line.strip()}")
    finished = perf_counter()
    print(f"public-text scan: git grep={selected - started:.3f}s "
          f"Python={finished - selected:.3f}s")
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("line", [
    "PI = math.pi", "PI =", "2 * PI", "math.pi", "np.pi", "PIL", "PIN",
    "API", "SPICE", "my_PI_value", "PI_value", "value_PI", "SPI2",
    "21^3", "20261016", "0.648", "acc-admem-g1-20261016t043022z-648d9-extra",
    "prefix-acc-admem-g1-20261016t043022z-648d9",
])
def test_math_and_identifiers_are_allowed(line):
    assert matching_classes(line) == []


PRIVATE_TEXT_CASES = [
    ("# /Users/byungkwankim/x", "C1"),
    ("# -Users-byungkwankim-x", "C1"),
    ("# byungkwan-workspace/x", "C1"),
    ("# runs/byungkwan/123", "C1"),
    ("# Project: byungkwan", "C1"),
    ("# (PI, 2026-09-24)", "C2"),
    ("# PI-directed", "C2"),
    ("# PI-gated", "C2"),
    ("# PI-approved", "C2"),
    ("# the professor", "C2"),
    ("PI = math.pi  # PI approved", "C2"),
    ("2 * PI  # PI approved", "C2"),
    ("# 지시", "C3"),
    ("# rfx-archive", "C4"),
    ("# agent-rules", "C4"),
    ("# pi-desk", "C4"),
    ("# rfx-lead", "C4"),
    ("# rfx-core-desk", "C4"),
    ("# D:abc123", "C4"),
    ("# L:abcdef", "C4"),
    ("# MSL-PLANE.md", "C4"),
    ("# rfx/records/sample/", "C4"),
    ("# ledger archive", "C4"),
    ("# see rfx-known-issues.md", "C4"),
    ("# records 20261012-pencil-stage-peaks", "C4"),
    ("# records/20261109-x/DESIGN.md", "C4"),
    ("# docs/agent-memory/x.md", "C4"),
    ("# internal issue ledger", "C4"),
    ("# docs/research_notes/sample.md", "C4"),
    ("# rfx-research", "C4"),
    ("# rfx-adv", "C4"),
    ("# rfx-fdfd-research", "C4"),
    ("# bk-workspace", "C4"),
    ("# remilab-cnu/paper", "C4"),
    ("# acc-admem-g1-20261016t043022z-648d9", "C4"),
]


@pytest.mark.parametrize(("line", "expected"), PRIVATE_TEXT_CASES)
def test_each_private_text_form_is_detected(line, expected):
    assert matching_classes(line) == [expected]


def test_git_prefilter_covers_every_class(tmp_path):
    assert {expected for _, expected in PRIVATE_TEXT_CASES} == set(PATTERNS)
    lines = [line for line, _ in PRIVATE_TEXT_CASES]
    # Include every endpoint of the Hangul ranges, not just a common syllable.
    lines.extend(chr(point) for point in (
        0x1100, 0x11ff, 0x3130, 0x318f, 0xa960, 0xa97f,
        0xac00, 0xd7af, 0xd7b0, 0xd7ff,
    ))
    sample = tmp_path / "samples.txt"
    sample.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for engine in ("-E", "-P"):
        result = subprocess.run(
            ["git", "grep", "--no-index", "-n", "-I", "-z", engine, "-i",
             CANDIDATE_PATTERN, "--", sample.name],
            cwd=tmp_path, capture_output=True,
        )
        assert result.returncode == 0, (engine, result.stderr)
        selected = [record.split(b"\0", 2)[2].decode("utf-8")
                    for record in result.stdout.split(b"\n") if record]
        assert selected == lines, engine


# A blank literal exempts a whole file while it still contains matching text.
NAMED_EXCEPTIONS = (
    ('scripts/memory_reduction_planning_artifact.py', 'default=Path(', "an argparse default into the repository's gitignored note folder; behaviour"),
    ('scripts/diagnostics/build_msl_openems_sparameter_comparison.py', 'DEFAULT_REFERENCE =', "an argparse default into the repository's gitignored note folder; behaviour"),
    ('tests/contracts/test_public_text_has_no_pi_or_agent_traces.py', '', 'The pin holds the patterns and synthetic cases, and its only hits are such text.'),
    ('tests/contracts/test_research_notes_are_not_tracked.py', 'docs/research_notes/', 'The contract proves the research-note directory is untracked.'),
    ('tests/contracts/test_research_notes_are_not_tracked.py', 'docs/agent-memory', 'The contract proves the local-only ledger directory is untracked.'),
    ('tests/contracts/test_source_carries_no_tracker_history.py', 'docs/research_notes/2000-01-02_sample.md', 'Synthetic citation proves private provenance is detected.'),
    ('tests/contracts/test_guide_citations_resolve.py', 'docs/research_notes/', 'Synthetic prose and its explanation prove a plain path is not a link.'),
)


def test_named_exceptions_are_not_stale():
    for path, literal, reason in NAMED_EXCEPTIONS:
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        assert any(literal in line and (literal or matching_classes(line))
                   for line in lines), (path, literal, reason)

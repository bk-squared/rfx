"""Public source text states rules and facts without private attribution."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCOPE = ("rfx/",)
pytestmark = pytest.mark.source_scan

PATTERNS = {
    "C1": re.compile(r"/Users/byungkwankim/|-Users-byungkwankim|byungkwan-workspace"),
    "C2": re.compile(r"\b(?:PI|PI-directed|PI-gated|PI-approved)\b|the professor"),
    "C3": re.compile(r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]"),
    "C4": re.compile(
        r"rfx-archive|agent-rules|pi-desk|rfx-lead|rfx-core-desk|\b[DL]:[0-9a-f]{6}\b"
        r"|MSL-PLANE\.md|rfx/records/|ledger archive|internal issue ledger|rfx-known-issues|\brecords? 20\d{6}-[a-z]"
        r"|docs/research_notes/|rfx-research|rfx-adv|rfx-fdfd-research"
        r"|bk-workspace|remilab-cnu/paper"
        r"|(?<![\w-])acc-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-\d{8}t\d{6}z-[0-9a-f]+(?![\w-])"
    ),
}
MATH_PI = re.compile(r"\bPI(?=\s*=)|\b2\s*\*\s*PI\b")


def matching_classes(line):
    """Allow mathematical uses without hiding attribution elsewhere on a line."""
    return [name for name, pattern in PATTERNS.items()
            if pattern.search(MATH_PI.sub("", line) if name == "C2" else line)]


def test_public_text_has_no_pi_or_agent_traces():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *SCOPE], cwd=ROOT,
    ).decode().split("\0")
    errors = []
    for name in filter(None, paths):
        data = (ROOT / name).read_bytes()
        if b"\0" in data:
            continue
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(source.splitlines(), 1):
            if classes := matching_classes(line):
                errors.append(f"{name}:{number}: {', '.join(classes)}: {line.strip()}")
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("line", [
    "PI = math.pi", "PI =", "2 * PI", "math.pi", "np.pi", "PIL", "PIN",
    "API", "SPICE", "my_PI_value", "PI_value", "value_PI", "SPI2",
    "21^3", "20261016", "0.648", "acc-admem-g1-20261016t043022z-648d9-extra",
    "prefix-acc-admem-g1-20261016t043022z-648d9",
])
def test_math_and_identifiers_are_allowed(line):
    assert matching_classes(line) == []


@pytest.mark.parametrize(("line", "expected"), [
    ("# /Users/byungkwankim/x", "C1"),
    ("# -Users-byungkwankim-x", "C1"),
    ("# byungkwan-workspace/x", "C1"),
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
    ("# internal issue ledger", "C4"),
    ("# docs/research_notes/sample.md", "C4"),
    ("# rfx-research", "C4"),
    ("# rfx-adv", "C4"),
    ("# rfx-fdfd-research", "C4"),
    ("# bk-workspace", "C4"),
    ("# remilab-cnu/paper", "C4"),
    ("# acc-admem-g1-20261016t043022z-648d9", "C4"),
])
def test_each_private_text_form_is_detected(line, expected):
    assert matching_classes(line) == [expected]

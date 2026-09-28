"""The showcase catalog and the pages that print its numbers.

``docs/public/showcase/showcase.json`` (schema ``rfx-showcase-catalog/1``) is
written by ``scripts/showcase/build_catalog.py`` from the share host's files: for
each showcase case, every file with its URL, size and SHA-256, the rfx commits,
and each number a page prints with its full-precision value.  The first tests
read only that JSON and hold it to itself.  The ``docs_consistency`` tests read
the pages: a number in a tile, a card, the home proof strip or the gradient-cost
table must be a catalog number for that page that rounds to what is printed,
every catalog number must be printed where the catalog says, and every file the
pages link on the share host must be a catalog file.  A mismatch there is a
documentation fix (PI, 2026-09-22), so those tests run only in the
docs-consistency workflow.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PUBLIC = REPO / "docs" / "public"
CATALOG = PUBLIC / "showcase" / "showcase.json"
SHARE = "https://remilab.cnu.ac.kr/share/7c02ad43c580/"
SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
#: A printed number: 20,592 / 0.097 / 2×10⁻⁵.  Not a digit inside a word
#: (RTX A6000, float64, TM010) or a date.
NUMBER = re.compile(r"(?<![\w.,/-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
                    r"(?:×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))?(?![\w,]|\.\d)")
#: Pages that print catalog numbers, by route.
PAGES = {
    "/rfx/": "index.mdx",
    "/rfx/showcase/": "showcase/index.mdx",
    "/rfx/showcase/sensitivity-map/": "showcase/sensitivity-map.mdx",
    "/rfx/showcase/gradient-cost/": "showcase/gradient-cost.mdx",
    "/rfx/showcase/descent-to-optimum/": "showcase/descent-to-optimum.mdx",
    "/rfx/showcase/forward-checks/": "showcase/forward-checks.mdx",
}


def catalog() -> dict:
    return json.loads(CATALOG.read_text())


def headlines() -> list[dict]:
    return [h for case in catalog()["cases"] for h in case["headlines"]]


def printed_bounds(text: str) -> tuple[Decimal, Decimal]:
    """The interval of values that print as ``text``: 1.4 -> [1.35, 1.45]."""
    match = NUMBER.fullmatch(text)
    if match is None:
        raise ValueError(f"not a printed number: {text!r}")
    mantissa = Decimal(match.group(1).replace(",", ""))
    scale = Decimal(10) ** int(match.group(2).translate(SUPERSCRIPT)) if match.group(2) else Decimal(1)
    half = Decimal(5) * Decimal(10) ** (mantissa.as_tuple().exponent - 1)
    return (mantissa - half) * scale, (mantissa + half) * scale


def rounds_to(value, text: str) -> bool:
    lo, hi = printed_bounds(text)
    ends = value if isinstance(value, list) else [value]
    return all(lo <= Decimal(repr(float(v))) <= hi for v in ends)


# ---------------------------------------------------------------- the catalog
def test_every_catalog_number_rounds_to_what_it_says_is_printed():
    found = headlines()
    assert len(found) >= 20, f"only {len(found)} catalog numbers"
    wrong = [(h["quantity"], h["value"], h["printed"]) for h in found if not rounds_to(h["value"], h["printed"])]
    assert not wrong, f"catalog values that do not print as their text: {wrong}"


def test_every_catalog_file_is_a_share_host_url_with_a_size_and_hash():
    data = catalog()
    assert data["schema"] == "rfx-showcase-catalog/1"
    assert data["share"]["base_url"] == SHARE
    files = 0
    for case in data["cases"]:
        assert case["route"] == f"/rfx/showcase/{case['id']}/"
        assert case["rfx_commits"] and all(re.fullmatch(r"[0-9a-f]{40}", c) for c in case["rfx_commits"])
        for source in case["sources"]:
            assert source["base_url"] == f"{SHARE}{source['folder']}/"
            names = {f["name"] for f in source["files"]}
            assert "result.json" in names and source["result_json"] == source["base_url"] + "result.json"
            for record in source["files"]:
                assert record["url"] == source["base_url"] + record["name"]
                assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                assert isinstance(record["bytes"], int) and record["bytes"] > 0
                files += 1
    assert files >= 34


def test_the_catalog_has_one_case_per_showcase_page():
    pages = {p.stem for p in (PUBLIC / "showcase").glob("*.mdx")} - {"index"}
    assert {case["id"] for case in catalog()["cases"]} == pages


@pytest.mark.parametrize("text,value,ok", [
    ("1.4", 1.4499, True), ("1.4", 1.451, False), ("1.4", 1.349, False),
    ("20,592", 20592.4, True), ("20,592", 20591.4, False),
    ("2×10⁻⁵", 1.9536e-05, True), ("2×10⁻⁵", 2.6e-05, False),
    ("0.090", 0.0897, True), ("0.090", 0.0904, True), ("0.090", 0.0906, False),
    ("8.0", 7.98, True), ("24", [23.68, 24.31], True), ("24", [23.4, 24.3], False),
])
def test_rounding_reads_the_printed_digits(text, value, ok):
    assert rounds_to(value, text) is ok


# ---------------------------------------------------------------- the pages
class _Numbers(HTMLParser):
    """Text inside the elements that carry headline numbers: the tiles, the
    proof strip, and a card's small metric line."""

    VOID = {"img", "source", "br", "hr", "input", "meta", "link", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str | None]] = []
        self.texts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append((tag, dict(attrs).get("data-rfx")))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        hooks = [hook for _, hook in self.stack]
        tags = [tag for tag, _ in self.stack]
        if "tiles" in hooks or "proof" in hooks or ("card" in hooks and "small" in tags):
            self.texts.append(data)


def _without_code(text: str) -> str:
    return re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)


def page_numbers(route: str) -> list[str]:
    text = _without_code((PUBLIC / PAGES[route]).read_text())
    parser = _Numbers()
    parser.feed(text)
    chunks = parser.texts
    # The cost table's rows (a Markdown table: header and delimiter rows skipped).
    rows = [line for line in text.splitlines() if line.startswith("|")]
    chunks += [cell for row in rows[2:] for cell in row.split("|")]
    return [m.group(0) for chunk in chunks for m in NUMBER.finditer(chunk)]


@pytest.mark.docs_consistency
def test_every_headline_number_on_a_page_is_a_catalog_number_for_that_page():
    missing = {}
    counted = 0
    for route in PAGES:
        known = [h for h in headlines() if route in h["routes"]]
        for text in page_numbers(route):
            counted += 1
            if not any(h["printed"] == text and rounds_to(h["value"], text) for h in known):
                missing.setdefault(route, []).append(text)
    assert counted >= 40, f"only {counted} headline numbers found on the pages"
    assert not missing, f"page numbers with no catalog number that rounds to them: {missing}"


@pytest.mark.docs_consistency
def test_every_catalog_number_is_printed_on_the_pages_it_names():
    printed = {route: page_numbers(route) for route in PAGES}
    absent = [(h["quantity"], h["printed"], route) for h in headlines()
              for route in h["routes"] if h["printed"] not in printed[route]]
    assert not absent, f"catalog numbers not printed where the catalog says: {absent}"


@pytest.mark.docs_consistency
def test_every_share_host_file_a_page_links_is_a_catalog_file():
    known = {f["url"] for case in catalog()["cases"] for s in case["sources"] for f in s["files"]}
    linked = {}
    for path in sorted(PUBLIC.rglob("*.md*")):
        for url in re.findall(re.escape(SHARE) + r"[^\s\"'()<>]+", path.read_text()):
            linked.setdefault(url, path.relative_to(REPO).as_posix())
    assert len(linked) >= 20, f"only {len(linked)} share-host links found"
    unknown = {url: page for url, page in linked.items() if url not in known}
    assert not unknown, f"share-host links that are not catalog files: {unknown}"


@pytest.mark.docs_consistency
def test_each_case_title_and_question_are_its_pages_front_matter():
    import yaml
    for case in catalog()["cases"]:
        text = (PUBLIC / "showcase" / f"{case['id']}.mdx").read_text()
        meta = yaml.safe_load(text[4:].split("\n---", 1)[0])
        assert (case["title"], case["question"]) == (meta["title"], meta["description"]), case["id"]

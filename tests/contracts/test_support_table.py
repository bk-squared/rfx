"""Pure parsing contract for the support page's refused-inputs tables.

Disposition fragments are executed by test_path_disposition_cells.py, not here.
The parser and checker accept in-memory inputs so mutations test the contract
without modifying the page, importing rfx, or executing pinned test functions.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re

import pytest

from tests.contracts import path_disposition as T

pytestmark = pytest.mark.docs_consistency

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs/guides/support_matrix.md"
PATH_SECTION = "Refused by solver path"
COMBINATION_SECTION = "Refused combinations"
PATH_HEADER = ("Input", "Row key", "Uniform", "Graded", "Multi-device",
               "Calculators", "What to do instead")
COMBINATION_HEADER = ("Input", "Row key", "Where", "What to do instead", "Pinned by")
COLUMNS = {
    "Uniform": ("run_uniform", "fwd_uniform"),
    "Graded": ("run_nonuniform", "fwd_nonuniform"),
    "Multi-device": ("run_distributed", "run_distributed_nu", "fwd_distributed_nu"),
}
WORDS = {
    "supported": {T.CARRIES}, "refused": {T.REFUSES},
    "falls back": {T.FALLS_BACK}, "n/a": {T.NOT_REACHABLE, T.IGNORABLE},
    "unclassified": {T.UNCLASSIFIED},
}
SETTINGS = {"_precision", "_stencil_order", "_mode", "_dt_pin", "_dt_min_cell"}
# All exemptions, with their reasons, live in this one dictionary.
EXEMPTIONS = {
    "run_adi": "ADI is experimental: one experimental row and no page, PI 2026-10-12",
    "fwd_adi": "ADI is experimental: one experimental row and no page, PI 2026-10-12",
    "run_subgridded": "refused by default, tracker 1465; one combination row",
    "coaxial_line_reflection": "build their own coaxial line and take no other structure",
    "coaxial_two_port": "build their own coaxial line and take no other structure",
    "coax_msl_transition": "experimental diagnostic",
    "_refinement/relaxed_validation": "subgridded research mode",
}


@dataclass(frozen=True)
class Row:
    section: str
    values: dict[str, str]
    text: str
    line: int

    @property
    def key(self):
        return self.values["Row key"].strip("`")


@dataclass(frozen=True)
class ParsedPage:
    rows: tuple[Row, ...]
    intro: str
    errors: tuple[str, ...]


def _fields(line):
    return tuple(part.strip() for part in line.strip().strip("|").split("|"))


def parse_page(text: str) -> ParsedPage:
    """Collect both refusal tables, retaining all parse errors for one report."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "## Refused inputs"]
    if len(starts) != 1:
        return ParsedPage((), "", ("expected exactly one ## Refused inputs section",))
    start = starts[0] + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    rows, errors, intro = [], [], []
    sections = set()
    section = None
    in_table = False
    header = None
    separator = False
    first_table = False
    for index in range(start, end):
        line = lines[index]
        if line.startswith("### "):
            section = line[4:]
            sections.add(section)
        if not first_table and not line.lstrip().startswith("|"):
            intro.append(line)
        if not line.lstrip().startswith("|"):
            in_table = False
            continue
        if section not in (PATH_SECTION, COMBINATION_SECTION):
            errors.append(f"line {index + 1}: table outside the two refusal subsections")
            continue
        first_table = True
        fields = _fields(line)
        expected = PATH_HEADER if section == PATH_SECTION else COMBINATION_HEADER
        if not in_table:
            in_table, header, separator = True, fields, False
            if header != expected:
                errors.append(f"line {index + 1}: unexpected {section} header {header!r}; expected {expected!r}")
            continue
        if not separator:
            separator = True
            if len(fields) == len(expected) and all(re.fullmatch(r":?-+:?", f) for f in fields):
                continue
            errors.append(f"line {index + 1}: missing table separator")
        if header != expected:
            continue
        if len(fields) != len(header):
            errors.append(f"line {index + 1}: expected {len(header)} cells, got {len(fields)}")
            continue
        rows.append(Row(section, dict(zip(header, fields)), line, index + 1))
    for name in (PATH_SECTION, COMBINATION_SECTION):
        if name not in sections:
            errors.append(f"missing ### {name}")
        elif not any(row.section == name for row in rows):
            errors.append(f"no rows under ### {name}")
    return ParsedPage(tuple(rows), "\n".join(intro), tuple(errors))


def parse_clauses(value: str, column: str):
    """Return (word, star, lanes) clauses; reject ambiguous or partial coverage."""
    lanes = COLUMNS[column]
    qualifiers = {
        "run": tuple(p for p in lanes if p.startswith("run_")),
        "forward": tuple(p for p in lanes if p.startswith("fwd_")),
    }
    if column == "Multi-device":
        qualifiers.update(uniform=("run_distributed",),
                          graded=("run_distributed_nu", "fwd_distributed_nu"))
    clauses, covered, used = [], [], set()
    for clause in value.split("; "):
        match = re.fullmatch(r"(supported|refused|falls back|n/a|unclassified)(\*)?(?: \(([^()]+)\))?", clause)
        if not match:
            raise ValueError(f"invalid clause {clause!r}")
        word, star, qualifier = match.groups()
        if qualifier is not None and qualifier not in qualifiers:
            raise ValueError(f"unknown qualifier ({qualifier})")
        selected = lanes if qualifier is None else qualifiers[qualifier]
        used.add(qualifier)
        covered.extend(selected)
        clauses.append((word, bool(star), selected))
    if used & {"uniform", "graded"} and used & {"run", "forward"}:
        raise ValueError("mixed mesh and entry-point qualifiers")
    if len(covered) != len(set(covered)) or set(covered) != set(lanes):
        raise ValueError("clauses must cover every lane exactly once")
    return clauses


def _star_errors(star, cells, text):
    issues = {c.wrong for c in cells if c.wrong}
    errors = []
    if star != bool(issues):
        errors.append(f"star must be {'present' if issues else 'absent'} for wrong={sorted(issues)!r}")
    for issue in sorted(issues):
        if issue not in text:
            errors.append(f"missing wrong= reference {issue}")
    return errors


def _calls(function):
    """Inspect its body, not decorators or bodies of nested definitions."""
    pending = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            yield node.func
        pending.extend(ast.iter_child_nodes(node))


def _has_raises(function):
    return any(isinstance(call, ast.Attribute) and call.attr == "raises"
               and isinstance(call.value, ast.Name) and call.value.id == "pytest"
               for call in _calls(function))


def _functions(body):
    return {node.name: node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def function_pins_refusal(module: ast.Module, names: list[str]) -> bool:
    """Resolve a module function or class method and follow at most one hop."""
    functions = _functions(module.body)
    methods = {}
    if len(names) == 2:
        classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
        if names[0] not in classes:
            return False
        methods = _functions(classes[names[0]].body)
    function = (methods if len(names) == 2 else functions).get(names[-1])
    if function is None:
        return False
    if _has_raises(function):
        return True
    for call in _calls(function):
        helper = None
        if isinstance(call, ast.Name):
            helper = functions.get(call.id)
        elif (isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name)
              and call.value.id in {"self", "cls"}):
            helper = methods.get(call.attr)
        if helper is not None and _has_raises(helper):
            return True
    return False


def _pin_errors(value, repo, cache):
    if not re.fullmatch(r"`[^`]+`(?:,\s*`[^`]+`)*", value):
        return ["Pinned by must be a non-empty comma-separated list of backticked pins"]
    errors, filename = [], None
    for item in re.findall(r"`([^`]+)`", value):
        parts = item.split("::")
        if parts[0]:
            filename = parts[0]
        names = parts[1:]
        if (filename is None or not filename.endswith(".py") or len(names) not in (1, 2)
                or any(not name.isidentifier() for name in names)):
            errors.append(f"invalid pin {item!r}")
            continue
        path = (repo / filename).resolve()
        if not path.is_relative_to(repo.resolve()) or not path.is_file():
            errors.append(f"pin file must exist under repo root: {filename}")
            continue
        if path not in cache:
            try:
                cache[path] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as exc:
                cache[path] = str(exc)
        module = cache[path]
        if not isinstance(module, ast.Module):
            errors.append(f"cannot parse pin file {filename}: {module}")
        elif not function_pins_refusal(module, names):
            errors.append(f"pin {filename}::{'::'.join(names)}: function missing or no pytest.raises directly or in a one-hop helper")
    return errors


def check_rows(page: ParsedPage, *, table=T.TABLE, cell=T.cell,
               row_class=T.ROW_CLASS, lanes=T.LANES, calculators=T.CALCULATORS,
               exemptions=EXEMPTIONS, repo=REPO) -> list[str]:
    """Check parsed rows against a table, using cell() for every disposition.

    Synthetic tables supply their matching cell callable (including '*' row
    resolution). The real table always uses path_disposition.cell unchanged.
    """
    errors = list(page.errors)
    for word in ("ADI", "Subgridded"):
        if word not in page.intro:
            errors.append(f"intro must mention {word}")
    seen, cache = set(), {}
    active_calculators = set(calculators) - exemptions.keys()
    active_paths = (set(lanes) | set(calculators)) - exemptions.keys()
    for row in page.rows:
        key = row.key
        prefix = f"{key} (line {row.line})"
        if not re.fullmatch(r"`[^`]+`", row.values["Row key"]):
            errors.append(f"{prefix}: Row key must be backticked")
        if key in seen:
            errors.append(f"{prefix}: duplicate row key")
        seen.add(key)
        if row.section == COMBINATION_SECTION:
            match = re.fullmatch(r"(admission|arg):([A-Za-z0-9_][A-Za-z0-9_-]*)", key)
            if not match:
                errors.append(f"{prefix}: invalid combination row key")
            elif match[1] == "arg" and (match[2] in table or "_" + match[2] in table):
                errors.append(f"{prefix}: arg collides with a TABLE attr")
            errors.extend(f"{prefix}: {error}" for error in _pin_errors(row.values["Pinned by"], repo, cache))
            continue
        attr, _, feature = key.partition("/")
        if (not re.fullmatch(r"_[^/`\s]+(?:/[^/`\s]+)?", key)
                or attr not in table or feature not in table[attr]):
            errors.append(f"{prefix}: row key does not exist in TABLE")
            continue
        for column, paths in COLUMNS.items():
            value = row.values[column]
            cells = {path: cell(attr, feature, path) for path in paths}
            context = f"{prefix}, {column}: page={value!r}; table=" + ", ".join(
                f"{path}={c.kind}" for path, c in cells.items())
            try:
                clauses = parse_clauses(value, column)
            except ValueError as exc:
                errors.append(f"{context}; {exc}")
                continue
            if any(cells[path].kind not in WORDS[word]
                   for word, _, selected in clauses for path in selected):
                errors.append(f"{context}; kind mismatch")
            for _, star, selected in clauses:
                errors.extend(f"{context}; {error}" for error in _star_errors(
                    star, [cells[path] for path in selected], row.text))
        value = row.values["Calculators"]
        cells = {name: cell(attr, feature, name) for name in sorted(active_calculators)}
        expected = {name for name, c in cells.items() if c.kind == T.REFUSES}
        context = f"{prefix}, Calculators: page={value!r}; table=" + ", ".join(
            f"{name}={c.kind}" for name, c in cells.items())
        match = re.fullmatch(r"refused \(([A-Za-z0-9_]+\*?(?:, [A-Za-z0-9_]+\*?)*)\)", value)
        if value != "—" and match is None:
            errors.append(f"{context}; invalid calculator grammar")
            continue
        listed = match[1].split(", ") if match else []
        names = [name.removesuffix("*") for name in listed]
        if len(set(names)) != len(names):
            errors.append(f"{context}; duplicate calculator name")
        if set(names) != expected or (value == "—") != (not expected):
            errors.append(f"{context}; expected refused calculators {sorted(expected)!r}")
        for token, name in zip(listed, names):
            if name not in cells:
                errors.append(f"{context}; unknown or exempt calculator {name}")
            else:
                errors.extend(f"{context}; {name}: {error}" for error in _star_errors(
                    token.endswith("*"), [cells[name]], row.text))
    missing = []
    for attr, features in table.items():
        if row_class[attr] not in (T.PHYSICS, T.OBSERVER) and attr not in SETTINGS:
            continue
        for feature in features:
            key = attr + ("/" + feature if feature else "")
            if key not in seen and key not in exemptions and any(
                    cell(attr, feature, path).kind == T.REFUSES for path in sorted(active_paths)):
                missing.append(key)
    if missing:
        errors.append("completeness: missing keys: " + ", ".join(sorted(missing)))
    return errors


@pytest.fixture(scope="module")
def page_text():
    return PAGE.read_text(encoding="utf-8")


def _mutate(text, key, column, value):
    row, = [row for row in parse_page(text).rows if row.key == key]
    header = PATH_HEADER if row.section == PATH_SECTION else COMBINATION_HEADER
    values = {**row.values, column: value}
    replacement = "| " + " | ".join(values[name] for name in header) + " |"
    return text.replace(row.text, replacement, 1)


def _assert_red(text, *fragments, **kwargs):
    errors = check_rows(parse_page(text), **kwargs)
    assert any(all(fragment in error for fragment in fragments) for error in errors), errors


def test_real_page_agrees(page_text):
    errors = check_rows(parse_page(page_text))
    assert not errors, "\n".join(errors)


def test_supported_changed_to_refused(page_text):
    text = _mutate(page_text, "_materials/kerr", "Uniform", "refused")
    _assert_red(text, "_materials/kerr", "Uniform", "kind mismatch")


def test_refused_star_removed(page_text):
    # Read the real TABLE once to select a page lane with wrong= if one exists.
    candidates = []
    for row in parse_page(page_text).rows:
        if row.section != PATH_SECTION:
            continue
        attr, _, feature = row.key.partition("/")
        for column in COLUMNS:
            for word, star, lanes in parse_clauses(row.values[column], column):
                if word == "refused" and any(T.cell(attr, feature, lane).wrong for lane in lanes):
                    candidates.append((row, column, star))
    if candidates:
        row, column, star = candidates[0]
        assert star, "the real wrong= refusal must first have its required star"
        text = _mutate(page_text, row.key, column, row.values[column].replace("refused*", "refused", 1))
        _assert_red(text, row.key, column, "star must be present")
        return
    # No page lane has a wrong= refusal today: synthetic mini-table + TABLE.
    assert not candidates
    table = {"_synthetic": {"": {"*": T.Cell(T.REFUSES, wrong="#1293")}}}
    text = "\n".join((
        "## Refused inputs", "ADI and Subgridded are exempt.",
        "### Refused by solver path", "| " + " | ".join(PATH_HEADER) + " |",
        "|---|---|---|---|---|---|---|",
        "| Synthetic #1293 | `_synthetic` | refused* | refused* | refused* | — | See #1293. |",
        "### Refused combinations", "| " + " | ".join(COMBINATION_HEADER) + " |",
        "|---|---|---|---|---|", next(row.text for row in parse_page(page_text).rows
                                    if row.key == "arg:exchange_interval"),
    ))
    def synthetic_cell(attr, feature, path):
        row = table[attr][feature]
        return row["*"] if "*" in row else row[path]
    kwargs = dict(table=table, cell=synthetic_cell, row_class={"_synthetic": T.PHYSICS}, calculators=())
    assert check_rows(parse_page(text), **kwargs) == []
    text = _mutate(text, "_synthetic", "Uniform", "refused")
    _assert_red(text, "_synthetic", "Uniform", "star must be present", **kwargs)


def test_stray_qualifier(page_text):
    text = _mutate(page_text, "_materials/kerr", "Uniform", "supported (cpu)")
    _assert_red(text, "_materials/kerr", "unknown qualifier (cpu)")


def test_mixed_multi_device_qualifiers(page_text):
    text = _mutate(page_text, "_materials/kerr", "Multi-device", "supported (uniform); refused (run)")
    _assert_red(text, "_materials/kerr", "mixed mesh and entry-point qualifiers")


def test_deleted_refusal_row(page_text):
    key = "_materials/kerr"
    assert T.cell("_materials", "kerr", "run_nonuniform").kind == T.REFUSES
    row, = [row for row in parse_page(page_text).rows if row.key == key]
    _assert_red(page_text.replace(row.text + "\n", "", 1), "completeness: missing keys:", key)


def test_dropped_calculator(page_text):
    text = _mutate(page_text, "_materials/kerr", "Calculators",
                   "refused (mixed_s_matrix, topology_optimize, waveguide_s_matrix, vmap_sweep_batched, material_fit)")
    _assert_red(text, "_materials/kerr", "expected refused calculators", "s_matrix_scan")


def test_pin_without_refusal(page_text):
    text = _mutate(page_text, "arg:exchange_interval", "Pinned by",
                   "`tests/contracts/test_support_table.py::parse_page`")
    _assert_red(text, "arg:exchange_interval", "no pytest.raises directly or in a one-hop helper")


def test_real_class_helper_pin(page_text):
    text = _mutate(page_text, "admission:multi-device-outer-jit", "Pinned by",
                   "`tests/unit/runners/test_distributed.py::TestExchangeInterval::test_interval_refused`")
    page = parse_page(text)
    assert not page.errors
    # Isolate the changed combination: unrelated real-page disagreements must
    # not make this hand-written green expectation fail.
    row, = [row for row in page.rows if row.key == "admission:multi-device-outer-jit"]
    assert check_rows(ParsedPage((row,), page.intro, ()), table={}, row_class={}) == []


def test_arg_collides_with_table(page_text):
    text = _mutate(page_text, "arg:exchange_interval", "Row key", "`arg:precision`")
    _assert_red(text, "arg:precision", "arg collides with a TABLE attr")


@pytest.mark.parametrize("column,value,reason", [
    ("Uniform", "unknown", "invalid clause"),
    ("Uniform", "supported (uniform)", "unknown qualifier"),
    ("Graded", "supported (run)", "cover every lane exactly once"),
    ("Multi-device", "refused (run); refused (run)", "cover every lane exactly once"),
    ("Uniform", "supported (run);refused (forward)", "invalid clause"),
    ("Uniform", "supported*", "star must be absent"),
])
def test_invalid_path_cells(page_text, column, value, reason):
    _assert_red(_mutate(page_text, "_materials/kerr", column, value), "_materials/kerr", reason)


@pytest.mark.parametrize("section,header", [
    (PATH_SECTION, PATH_HEADER), (COMBINATION_SECTION, COMBINATION_HEADER),
])
def test_changed_header(page_text, section, header):
    original = "| " + " | ".join(header) + " |"
    _assert_red(page_text.replace(original, original.replace("Input", "Feature"), 1),
                "unexpected", section, "header")


def test_duplicate_and_unknown_row_keys(page_text):
    row, = [row for row in parse_page(page_text).rows if row.key == "_materials/kerr"]
    _assert_red(page_text.replace(row.text, row.text + "\n" + row.text, 1),
                "_materials/kerr", "duplicate row key")
    _assert_red(_mutate(page_text, row.key, "Row key", "`_unknown/feature`"),
                "_unknown/feature", "does not exist in TABLE")


@pytest.mark.parametrize("value,reason", [
    ("—", "expected refused calculators"),
    ("refused (unknown)", "unknown or exempt calculator unknown"),
    ("refused (coaxial_two_port)", "unknown or exempt calculator coaxial_two_port"),
    ("refused (s_matrix_scan, s_matrix_scan)", "duplicate calculator name"),
    ("refused (s_matrix_scan*)", "star must be absent"),
    ("refused ()", "invalid calculator grammar"),
])
def test_invalid_calculators(page_text, value, reason):
    _assert_red(_mutate(page_text, "_materials/kerr", "Calculators", value), "_materials/kerr", reason)


@pytest.mark.parametrize("value,reason", [
    ("", "non-empty comma-separated"),
    ("`::test_interval_refused`", "invalid pin"),
    ("`tests/missing_support_pin.py::test_missing`", "pin file must exist"),
    ("`tests/unit/runners/test_distributed.py::test_interval_refused`", "function missing"),
])
def test_invalid_pins(page_text, value, reason):
    _assert_red(_mutate(page_text, "arg:exchange_interval", "Pinned by", value),
                "arg:exchange_interval", reason)


@pytest.mark.parametrize("body,names,expected", [
    ("def pin():\n    pytest.raises(ValueError, operation)", ["pin"], True),
    ("def helper():\n    with pytest.raises(ValueError):\n        operation()\n"
     "def pin():\n    helper()", ["pin"], True),
    ("class Pins:\n    def pin(cls):\n        cls.helper()\n"
     "    @staticmethod\n    def helper():\n        pytest.raises(ValueError, operation)",
     ["Pins", "pin"], True),
    ("def helper():\n    pytest.raises(ValueError, operation)\n"
     "def middle():\n    helper()\ndef pin():\n    middle()", ["pin"], False),
    ("def pin():\n    raises(ValueError, operation)", ["pin"], False),
    ("def outer():\n    def pin():\n        pytest.raises(ValueError, operation)", ["pin"], False),
])
def test_pin_ast_resolution(body, names, expected):
    assert function_pins_refusal(ast.parse(body), names) is expected


def test_real_bare_helper_and_shorthand_pins(page_text):
    page = parse_page(page_text)
    keys = {"admission:multi-device-outer-jit", "admission:wire-port-radius"}
    rows = tuple(row for row in page.rows if row.key in keys)
    assert len(rows) == 2
    assert check_rows(ParsedPage(rows, page.intro, page.errors), table={}, row_class={}) == []

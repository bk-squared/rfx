"""Source-reading tests must carry the mark used by the local gate selector."""
from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import pytest

from scripts.ci.source_scan import scan_source

ROOT = Path(__file__).resolve().parents[2]
NAMED_SCANS = (
    ("tests/unit/farfield/test_current_moment_monitor.py",
     "test_every_user_facing_return_carries_the_backstop"),
    ("tests/unit/materials/test_sheet_impedance.py",
     "test_every_fence_in_the_source_is_pinned"),
    ("tests/unit/sparams/test_settling_witness.py",
     "test_every_settling_db_producer_routes_through_the_shared_warner"),
    ("tests/unit/sparams/test_settling_witness.py",
     "test_the_known_lanes_are_all_covered"),
)


def violations(source: str, path: str) -> list[str]:
    scan = scan_source(source, path)
    errors = [f"{path}::{test.name}: unmarked source hit at line {test.hit_line}"
              for test in scan.tests if test.hit_line is not None and not test.marked]
    errors.extend(f"{path}::{test.name}: stale source_scan mark at line {test.lineno}"
                  for test in scan.tests if test.lineno in scan.stale_marks)
    errors.extend(f"{path}: stale source_scan mark at line {line}"
                  for line in scan.stale_marks if line not in {test.lineno for test in scan.tests})
    return errors


def test_source_reading_tests_are_marked():
    errors = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if not relative.startswith("tests/contracts/"):
            errors.extend(violations(path.read_text(encoding="utf-8"), relative))
    assert not errors, "\n" + "\n".join(errors)


@pytest.mark.parametrize("expression", [
    '"rfx/api/_execute.py"', '"rfx/core/a-b.py"',
    'rfx.__file__', 'package.__file__', 'execute.__file__', 'Simulation.__file__',
    'inspect.getsource(object)', 'ins.getsourcefile(object)', 'get_lines(object)',
    'getsource(object)', 'importlib.resources.files("rfx")',
    'resources.files("rfx.materials")', 'package_files("rfx.api")',
])
def test_each_source_hit(expression):
    source = """import rfx
import rfx.api as package
from rfx.api import _execute as execute
from rfx import Simulation
import inspect
import inspect as ins
from inspect import getsourcelines as get_lines
from inspect import *
import importlib.resources
from importlib import resources
from importlib.resources import files as package_files
def test_scan():
    VALUE
    open("other")
""".replace("VALUE", expression)
    scan = scan_source(source)
    assert scan.tests[0].hit_line == 13
    assert not scan.tests[0].marked
    assert scan.tests[0].reader_line is not None


@pytest.mark.parametrize("root", [
    '__import__("rfx.api._execute", fromlist=["_execute"])',
    'importlib.import_module("rfx.api")', 'import_module("rfx.api")', 'load("rfx.api")',
    'importlib.util.find_spec("rfx.api")', 'find_spec("rfx.api")', 'lookup("rfx.api")',
    'rfx', 'rfx.api._execute', 'package._execute', 'execute',
    'sys.modules["rfx.api._execute"]', 'modules["rfx.api"]',
])
@pytest.mark.parametrize("suffix", ['.__file__', '.origin', '.__spec__.origin'])
def test_each_new_source_handle(root, suffix):
    source = ('import rfx\nimport rfx.api as package\nfrom rfx.api import _execute as execute\n'
              'import importlib\nfrom importlib import import_module\n'
              'from importlib import import_module as load\n'
              'from importlib.util import find_spec\nfrom importlib.util import find_spec as lookup\n'
              'import sys\nfrom sys import modules\n'
              'def test_scan():\n    path = ' + root + suffix + '\n    open(path)\n')
    test = scan_source(source).tests[0]
    assert test.hit_line == 12
    assert test.reader_line == 13


@pytest.mark.parametrize("expression", [
    'other.__file__', 'other.origin', 'other.__spec__.origin',
    '__import__("other.api").__file__', 'importlib.import_module("other.api").origin',
    'importlib.util.find_spec("other.api").origin', 'sys.modules["other.api"].__file__',
    '__import__(name).__file__',
])
def test_non_package_handles_are_not_candidates(expression):
    source = 'def test_plain():\n    path = ' + expression + '\n    open(path)\n'
    assert scan_source(source).tests[0].hit_line is None


@pytest.mark.parametrize("expression", [
    'inspect.getfile(object)', 'inspect.getsource(object)', 'inspect.getsourcefile(object)',
    'inspect.getsourcelines(object)', 'getfile(object)', 'fetch(object)', 'getsource(object)',
])
def test_inspection_is_a_candidate_and_reader(expression):
    source = ('import inspect\nfrom inspect import *\nfrom inspect import getfile as fetch\n'
              'def test_scan():\n    ' + expression + '\n')
    test = scan_source(source).tests[0]
    assert test.hit_line == test.reader_line == 5


@pytest.mark.parametrize("expression", [
    '"rfx/api/_execute.py"', '"rfx/__init__.py"', 'rfx.__file__',
    'rfx.api._execute.__file__', '__import__("rfx.api").__file__',
    'importlib.util.find_spec("rfx.api").origin', 'importlib.resources.files("rfx")',
])
def test_candidate_alone_does_not_need_a_mark(expression):
    source = 'def test_plain():\n    location = ' + expression + '\n    assert location\n'
    assert scan_source(source).tests[0].hit_line is None
    assert not violations(source, "tests/test_plain.py")


@pytest.mark.parametrize("expression", [
    'Path("other").read_text()', 'open("other")', 'Path("other").glob("*.py")',
    'tokenize.open("other")',
])
def test_reader_alone_does_not_need_a_mark(expression):
    source = 'def test_plain():\n    ' + expression + '\n'
    test = scan_source(source).tests[0]
    assert test.hit_line is None
    assert test.reader_line == 2


def test_candidate_and_reader_in_one_helper_need_a_mark():
    test = scan_source('def helper():\n    return Path("rfx/api/a.py").read_text()\n'
                       'def test_scan():\n    return helper()\n').tests[0]
    assert test.hit_line == test.reader_line == 2


def test_fixture_candidate_and_parametrize_reader_are_combined():
    source = ('import pytest\n@pytest.fixture\ndef package_path():\n    return "rfx/api/a.py"\n'
              '@pytest.mark.parametrize("text", [open("other").read()])\n'
              'def test_scan(package_path, text):\n    pass\n')
    test = scan_source(source).tests[0]
    assert test.hit_line == 4
    assert test.reader_line == 5


def test_ast_walk_and_local_walk_are_not_file_readers():
    source = ('import ast\nPATH = "rfx/api/a.py"\n'
              'def walk(tree):\n    return tree\n'
              'def test_plain():\n    ast.walk(PATH)\n    walk(PATH)\n')
    assert scan_source(source).tests[0].hit_line is None


def test_tokenize_open_is_a_reader():
    test = scan_source('import tokenize\ndef test_scan():\n'
                       '    tokenize.open("rfx/api/a.py")\n').tests[0]
    assert test.hit_line == test.reader_line == 3


def test_module_reader_and_function_candidate_are_combined():
    test = scan_source('CONTENT = open("other").read()\ndef test_scan():\n'
                       '    assert CONTENT != "rfx/__init__.py"\n').tests[0]
    assert test.hit_line == 3
    assert test.reader_line == 1


def test_path_disposition_kernel_module_checks_are_not_source_reads():
    path = ROOT / "tests/unit/runners/test_path_disposition_cells.py"
    tests = {test.name: test for test in scan_source(path.read_text()).tests}
    names = ("test_cell", "test_relaxed_subgrid_validation_refuses_or_carries",
             "test_relaxed_subgrid_validation_refuses_an_absorbing_box",
             "test_the_guarded_lid_requires_experimental_validation",
             "test_a_slab_reaching_the_lid_is_refused_in_every_validation_mode")
    assert all(tests[name].hit_line is None for name in names)


def test_settling_witness_source_readers_are_detected():
    path = ROOT / "tests/unit/sparams/test_settling_witness.py"
    tests = {test.name: test for test in scan_source(path.read_text()).tests}
    for _, name in NAMED_SCANS[2:]:
        assert tests[name].hit_line is not None
        assert tests[name].reader_line is not None


@pytest.mark.parametrize("marker", ["slow", "slow_physics", "gpu", "docs_consistency"])
@pytest.mark.parametrize("scope", ["module", "class", "function"])
def test_stage2_exclusion_marks_are_inherited(marker, scope):
    source = 'import pytest\n'
    if scope == "module":
        source += f'pytestmark = [pytest.mark.{marker}, pytest.mark.source_scan]\n'
    if scope == "class":
        source += f'@pytest.mark.{marker}\n@pytest.mark.source_scan\nclass TestScan:\n'
    indent = "    " if scope == "class" else ""
    if scope == "function":
        source += f'@pytest.mark.{marker}\n@pytest.mark.source_scan\n'
    source += indent + 'def test_scan():\n' + indent + '    open("rfx/api/a.py")\n'
    test = scan_source(source).tests[0]
    assert test.deselected_markers == (marker,)
    assert test.marked
    assert test.hit_line is not None


@pytest.mark.parametrize("expression", [
    '"rfx"', '"rfx_like/api.py"', '"prefix/rfx/api.py"', '"rfx/api.py extra"',
    '"rfx/api.py\\n"', 'other.__file__', 'other.getsource(object)',
    'importlib.resources.files("other")',
])
def test_non_hits(expression):
    assert scan_source(f"def test_plain():\n    {expression}\n").tests[0].hit_line is None


def test_bare_package_directory_check_is_not_a_source_hit():
    scan = scan_source('def test_path(p):\n    return (Path(p) / "rfx").is_dir()\n')
    assert scan.tests[0].hit_line is None


def test_bare_package_directory_rglob_is_a_source_hit():
    scan = scan_source('def test_scan(root):\n    return (root / "rfx").rglob("*.py")\n')
    assert scan.tests[0].hit_line == 2
    assert not scan.tests[0].marked


def test_module_package_path_needs_a_mark_when_a_function_rglobs_it():
    source = ('PKG = ROOT / "rfx"\n'
              'def test_scan():\n    return PKG.rglob("*.py")\n'
              'def test_path():\n    return PKG.is_dir()\n')
    scan = scan_source(source)
    assert [test.hit_line for test in scan.tests] == [1, None]
    assert "tests/test_scan.py::test_scan: unmarked source hit at line 1" in violations(source, "tests/test_scan.py")


@pytest.mark.parametrize("reader", [
    'path.read_text()', 'path.read_bytes()', 'open(path)', 'path.open()',
    'path.rglob("*.py")', 'path.glob("*.py")', 'path.iterdir()', 'os.walk(path)',
    'os.listdir(path)', 'os.scandir(path)', 'inspect.getsource(path)',
    'inspect.getsourcefile(path)', 'inspect.getsourcelines(path)',
    'read(path)', 'walk_tree(path)',
])
def test_bare_package_directory_with_each_file_reader(reader):
    source = ('from pathlib import Path\nfrom os import walk as walk_tree\n'
              'from builtins import open as read\n'
              'def test_scan():\n    path = ROOT / "rfx"\n    ' + reader + '\n')
    assert scan_source(source).tests[0].hit_line == 5


def test_candidate_and_reader_can_be_reached_from_different_helpers():
    source = ('def path_helper():\n    return (ROOT / "rfx").is_dir()\n'
              'def read_helper():\n    return open("other")\n'
              'def test_scan():\n    path_helper()\n    read_helper()\n')
    test = scan_source(source).tests[0]
    assert test.hit_line == 2
    assert test.reader_line == 4


def test_bare_package_hit_requires_a_reader_in_the_same_module_statement():
    scan = scan_source('PATH = ROOT / "rfx"\nOTHER = open("other")\n'
                       'def test_plain():\n    return PATH.is_dir()\n')
    assert scan.tests[0].hit_line is None
    scan = scan_source('SOURCE = (ROOT / "rfx").rglob("*.py")\n'
                       'def test_scan():\n    return SOURCE\n')
    assert scan.tests[0].hit_line == 1


def test_module_package_path_aliases_and_reading_helpers_propagate():
    scan = scan_source('ALIAS = PKG\nPKG = ROOT / "rfx"\n'
                       'def helper():\n    return ALIAS.rglob("*.py")\n'
                       'def test_scan():\n    return helper()\n')
    assert scan.tests[0].hit_line == 2


def test_mark_on_a_bare_package_directory_check_is_stale():
    source = ('import pytest\n@pytest.mark.source_scan\n'
              'def test_path(p):\n    return (Path(p) / "rfx").is_dir()\n')
    assert "stale source_scan" in "\n".join(violations(source, "tests/test_plain.py"))


def test_module_helpers_fixtures_and_parametrize_propagate_to_fixed_point():
    scan = scan_source(textwrap.dedent('''\
        import pytest
        def second():
            return first()
        def first():
            return SOURCE
        SOURCE = "rfx/api/_execute.py"
        @pytest.fixture
        def scan_fixture():
            return second()
        @pytest.fixture(name="renamed")
        def alias_fixture(scan_fixture):
            return scan_fixture
        def test_helper():
            second().read_text()
        def test_fixture(scan_fixture):
            scan_fixture.read_text()
        def test_fixture_chain(renamed):
            renamed.read_text()
        @pytest.mark.usefixtures("scan_fixture")
        def test_usefixtures():
            open("other")
        @pytest.mark.parametrize("value", [SOURCE])
        def test_parameter(value):
            open(value)
        @pytest.mark.parametrize("value", ["rfx/model/a.py"])
        def test_direct_parameter(value):
            open(value)
        def test_unrelated():
            pass
        '''))
    assert [test.hit_line for test in scan.tests] == [6, 6, 6, 6, 6, 25, None]


@pytest.mark.parametrize("assignment", [
    'PATH = "rfx/api/a.py"', 'PATH: str = "rfx/api/a.py"',
    'LEFT = PATH = "rfx/api/a.py"',
    'if True:\n    PATH = "rfx/api/a.py"',
    'class Paths:\n    PATH = "rfx/api/a.py"\nPATH = Paths.PATH',
])
def test_module_bindings(assignment):
    scan = scan_source(assignment + "\ndef test_path():\n    return Path(PATH).read_text()\n")
    assert scan.tests[0].hit_line is not None


def test_function_locals_do_not_escape_into_module_names():
    scan = scan_source('def helper():\n    local = "rfx/api/a.py"\n'
                       'def test_plain():\n    local = "other"\n    return local\n')
    assert scan.tests[0].hit_line is None


@pytest.mark.parametrize("preamble,decorator,module_marked", [
    ("", "@pytest.mark.source_scan", False),
    ("pytestmark = pytest.mark.source_scan\n", "", True),
    ("pytestmark = [pytest.mark.slow, pytest.mark.source_scan]\n", "", True),
])
def test_direct_and_module_marks(preamble, decorator, module_marked):
    scan = scan_source("import pytest\n" + preamble + decorator
                       + '\ndef test_scan():\n    return Path("rfx/api/a.py").read_text()\n')
    assert scan.tests[0].marked
    assert scan.module_marked is module_marked
    assert not scan.stale_marks


def test_class_mark_and_test_node_id():
    scan = scan_source(textwrap.dedent('''\
        import pytest
        @pytest.mark.source_scan
        @pytest.mark.usefixtures("source_fixture")
        class TestScanner:
            def test_scan(self):
                pass
        @pytest.fixture
        def source_fixture():
            return Path("rfx/api/a.py").read_text()
        '''))
    assert scan.tests[0].name == "TestScanner::test_scan"
    assert scan.tests[0].marked
    assert scan.tests[0].hit_line == 9
    assert not scan.stale_marks


@pytest.mark.parametrize("source", [
    'import pytest\n@pytest.mark.source_scan\ndef test_plain():\n    pass\n',
    'import pytest\npytestmark = pytest.mark.source_scan\ndef test_plain():\n    pass\n',
    'import pytest\npytestmark = [pytest.mark.source_scan]\n',
    'import pytest\n@pytest.mark.source_scan\nclass TestPlain:\n    def test_plain(self):\n        pass\n',
])
def test_stale_marks_fail(source):
    assert "stale source_scan" in "\n".join(violations(source, "tests/test_plain.py"))


@pytest.mark.parametrize("path,name", NAMED_SCANS)
def test_named_scans_fail_when_their_mark_is_removed(path, name):
    source = (ROOT / path).read_text(encoding="utf-8")
    function = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    mark = next(node for node in function.decorator_list
                if ast.unparse(node) == "pytest.mark.source_scan")
    lines = source.splitlines(keepends=True)
    del lines[mark.lineno - 1:mark.end_lineno]
    errors = violations("".join(lines), path)
    assert any(f"{path}::{name}: unmarked source hit at line " in error for error in errors)


def test_sheet_impedance_depends_on_module_level_hit_propagation():
    path, name = NAMED_SCANS[1]
    source = (ROOT / path).read_text(encoding="utf-8")
    function = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    isolated = "import pytest\n" + ast.unparse(function)
    assert scan_source(isolated).tests[0].hit_line is None
    assert next(test for test in scan_source(source, path).tests if test.name == name).hit_line is not None

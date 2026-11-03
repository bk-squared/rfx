"""The process-wide realized capture hook is reserved for tests."""
import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _capture_calls(source, package):
    """Resolve absolute/relative imports and aliases without importing rfx."""
    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split('.')[0]] = (
                    name.name if name.asname else name.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                module = resolve_name('.' * node.level + module, package)
            for name in node.names:
                if name.name == '*':
                    if module == 'rfx._internal._realized':
                        aliases['capture'] = 'rfx._internal._realized.capture'
                    elif module == 'rfx._internal':
                        aliases['_realized'] = 'rfx._internal._realized'
                else:
                    aliases[name.asname or name.name] = f'{module}.{name.name}'

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f'{qualified(node.value)}.{node.attr}'
        return ''

    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and qualified(node.func) == 'rfx._internal._realized.capture']


@pytest.mark.source_scan
def test_rfx_never_enters_realized_capture():
    violations = []
    for path in sorted((ROOT / 'rfx').rglob('*.py')):
        relative = path.relative_to(ROOT)
        if relative == Path('rfx/_internal/_realized.py'):
            continue
        package = '.'.join(relative.parent.parts)
        for line in _capture_calls(path.read_text(), package):
            violations.append(f'{relative}:{line}')
    assert not violations, 'test-only realized capture entered by rfx: ' + ', '.join(violations)


@pytest.mark.parametrize('source', [
    'from rfx._internal._realized import capture; capture()',
    'from rfx._internal._realized import capture as observe; observe()',
    'from rfx._internal import _realized; _realized.capture()',
    'from rfx._internal import _realized as observed; observed.capture()',
    'import rfx._internal._realized; rfx._internal._realized.capture()',
    'import rfx._internal._realized as observed; observed.capture()',
    'import rfx._internal as package; package._realized.capture()',
    'from ._internal._realized import capture as observe; observe()',
    'from ._internal import _realized as observed; observed.capture()',
    'from .._internal._realized import capture; capture()',
    'from rfx._internal._realized import *; capture()',
    'from rfx._internal import *; _realized.capture()',
])
def test_capture_scan_recognizes_import_forms(source):
    package = 'rfx.probes' if source.startswith('from ..') else 'rfx'
    assert _capture_calls(source, package) == [1]


def test_capture_scan_ignores_unrelated_calls_and_mentions():
    source = '''
from another_module import capture
capture()
text = "rfx._internal._realized.capture()"
# rfx._internal._realized.capture()
'''
    assert _capture_calls(source, 'rfx') == []

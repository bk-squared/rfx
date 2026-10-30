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
                    if module == 'rfx._realized':
                        aliases['capture'] = 'rfx._realized.capture'
                    elif module == 'rfx':
                        aliases['_realized'] = 'rfx._realized'
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
            and qualified(node.func) == 'rfx._realized.capture']


@pytest.mark.source_scan
def test_rfx_never_enters_realized_capture():
    violations = []
    for path in sorted((ROOT / 'rfx').rglob('*.py')):
        relative = path.relative_to(ROOT)
        if relative == Path('rfx/_realized.py'):
            continue
        package = '.'.join(relative.parent.parts)
        for line in _capture_calls(path.read_text(), package):
            violations.append(f'{relative}:{line}')
    assert not violations, 'test-only realized capture entered by rfx: ' + ', '.join(violations)


@pytest.mark.parametrize('source', [
    'from rfx._realized import capture; capture()',
    'from rfx._realized import capture as observe; observe()',
    'from rfx import _realized; _realized.capture()',
    'from rfx import _realized as observed; observed.capture()',
    'import rfx._realized; rfx._realized.capture()',
    'import rfx._realized as observed; observed.capture()',
    'import rfx as package; package._realized.capture()',
    'from ._realized import capture as observe; observe()',
    'from . import _realized as observed; observed.capture()',
    'from .._realized import capture; capture()',
    'from rfx._realized import *; capture()',
    'from rfx import *; _realized.capture()',
])
def test_capture_scan_recognizes_import_forms(source):
    package = 'rfx.probes' if source.startswith('from ..') else 'rfx'
    assert _capture_calls(source, package) == [1]


def test_capture_scan_ignores_unrelated_calls_and_mentions():
    source = '''
from another_module import capture
capture()
text = "rfx._realized.capture()"
# rfx._realized.capture()
'''
    assert _capture_calls(source, 'rfx') == []

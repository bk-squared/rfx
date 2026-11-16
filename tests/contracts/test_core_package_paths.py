"""Private core paths disappear; realized state stays a live module binding."""
import ast
import importlib
from importlib.util import resolve_name
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE_MODULES = ('_realized', '_grid_metric', '_precision', '_periodic', '_uniform_specs')


@pytest.mark.parametrize('name', CORE_MODULES)
def test_old_core_paths_do_not_import(name, monkeypatch):
    # A shared editable install can otherwise find deleted files in another checkout.
    for finder in sys.meta_path:
        owner = sys.modules.get(finder.__module__)
        mapping = getattr(owner, 'MAPPING', None)
        if finder.__module__.startswith('__editable__') and isinstance(mapping, dict) and 'rfx' in mapping:
            monkeypatch.setitem(mapping, 'rfx', str(ROOT / 'rfx'))
    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.import_module(f'rfx.{name}')
    assert caught.value.name == f'rfx.{name}'
    assert importlib.import_module(f'rfx._internal.{name}') is not None


@pytest.mark.source_scan
def test_realized_state_is_never_imported_by_name():
    violations = []
    for path in sorted((ROOT / 'rfx').rglob('*.py')):
        package = '.'.join(path.relative_to(ROOT).parent.parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ''
            if node.level:
                module = resolve_name('.' * node.level + module, package)
            if module in {'rfx._realized', 'rfx._internal._realized'}:
                for name in node.names:
                    if name.name in {'ACTIVE', 'OWNED'}:
                        violations.append(f'{path.relative_to(ROOT)}:{node.lineno}: {name.name}')
    assert not violations, 'copied realized state binding: ' + ', '.join(violations)


@pytest.mark.parametrize('name', ['ACTIVE', 'OWNED'])
def test_realized_state_rebinding_is_seen_by_importer(monkeypatch, name):
    from rfx._internal import _realized
    importer = importlib.import_module('rfx.simulation')
    value = object()
    monkeypatch.setattr(_realized, name, value)
    assert importer._realized is _realized
    assert getattr(importer._realized, name) is value

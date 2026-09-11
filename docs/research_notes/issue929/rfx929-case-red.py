"""Negative control: restore the two missing contact checks' old behavior.

Only the in-process helper is changed. Repository files and case geometry
stay untouched; the regression tests themselves still use the real builders.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys

import pytest

root = Path('/root/rfx-codex')
sys.path.insert(0, str(root))
sys.path.insert(0, str(root/'validation/crossval'))
import _patch_feed_contract as contract

source = inspect.getsource(contract.assert_galvanic_patch_feed)
tree = ast.parse(source)
function = tree.body[0]
removed = []
kept = []
for statement in function.body:
    text = ast.unparse(statement)
    if isinstance(statement, ast.If) and any(
            token in text for token in ('no live source edge', 'stack ground/patch nodes')):
        removed.append(text)
    else:
        kept.append(statement)
assert len(removed) == 2, removed
function.body = kept
namespace = dict(contract.__dict__)
exec(compile(ast.fix_missing_locations(tree), '<contact-guard-negative-control>', 'exec'), namespace)
contract.assert_galvanic_patch_feed = namespace[function.name]
print('NEGATIVE CONTROL: omit only the stack-identity and live-source guards', flush=True)
print('helper_source_sha256', hashlib.sha256(source.encode()).hexdigest(), flush=True)
print('test_source_sha256', hashlib.sha256((root/'tests/crossval/test_patch_galvanic_feed.py').read_bytes()).hexdigest(), flush=True)
raise SystemExit(pytest.main([
    'tests/crossval/test_patch_galvanic_feed.py::test_cv05_cannot_substitute_an_intermediate_sheet_for_the_patch',
    'tests/crossval/test_patch_galvanic_feed.py::test_cv15_rejects_a_wire_wholly_inside_metal_even_with_valid_stack',
    '-q', '-o', 'addopts=', '--disable-warnings', '--tb=short',
]))

import ast, subprocess, textwrap
import pytest
import rfx.api._preflight as preflight

baseline = "0662c3b96d82ca65d7bd041b94c4ab984230e00d"
source = subprocess.check_output(["git", "show", baseline + ":rfx/api/_preflight.py"], text=True)
name = "_validate_cfg_off_lattice_design_edges"
node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
method = textwrap.dedent("\n".join(source.splitlines()[node.lineno-1:node.end_lineno]))
namespace = {}
exec(compile(method, "<retained-main-advisory-control>", "exec"), preflight.__dict__, namespace)
setattr(preflight._PreflightMixin, name, namespace[name])
result = pytest.main(["tests/unit/preflight/test_preflight_rasterization.py::TestOffLatticeCensus::test_nearest_node_residual_is_not_an_extent_error_bound", "-q", "-o", "addopts=", "-p", "no:cacheprovider"])
assert result == 1, f"expected old-advisory failures, got {result}"
print("Old main advisory restored in memory: expected regression failure; source files unchanged.")

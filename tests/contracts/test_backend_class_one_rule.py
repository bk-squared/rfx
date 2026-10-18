"""#1828: fake platform policy and a source guard; no Apple GPU measurements."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rfx.backend import BackendClass, backend_class


@pytest.mark.parametrize("platform,accelerator,writeback,memory", [
    ("cpu", False, "slab", "none"),
    ("gpu", True, "padded", "memory_stats"),
    ("cuda", True, "padded", "memory_stats"),
    ("rocm", True, "padded", "memory_stats"),
    ("mps", True, "slab", "memory_stats_if_available"),
    ("tpu-unknown", False, "slab", "unknown"),
])
def test_faked_platform_table(monkeypatch, platform, accelerator, writeback, memory):
    expected = BackendClass(platform, accelerator, writeback, memory)
    monkeypatch.setattr(jax, "default_backend", lambda: platform)
    assert backend_class() == expected
    # Explicit classification must not query the process's backend.
    def forbidden():
        pytest.fail("an explicit platform queried JAX")
    monkeypatch.setattr(jax, "default_backend", forbidden)
    assert backend_class(platform) == expected
    with pytest.raises(FrozenInstanceError):
        expected.accelerator = not accelerator


@pytest.mark.parametrize("platform,accelerator,form,gpu_mesh", [
    ("cpu", False, "slab", False),
    ("gpu", True, "padded", True),
    ("cuda", True, "padded", True),
    ("rocm", True, "padded", True),
    ("mps", True, "slab", False),
    ("tpu-unknown", False, "slab", False),
])
def test_reporting_cpml_and_rank(monkeypatch, platform, accelerator, form, gpu_mesh):
    from rfx import diagnostics, gpu
    from rfx.boundaries._cpml_slab import writeback_form
    from rfx.stepping.rank import _is_gpu_mesh

    devices = [SimpleNamespace(id=i, platform=platform, device_kind="gpu") for i in range(2)]
    monkeypatch.setattr(jax, "devices", lambda: devices)
    monkeypatch.setattr(jax, "default_backend", lambda: platform)
    assert writeback_form() == form
    assert gpu.device_info().gpu_available is accelerator
    report = diagnostics._Report()
    diagnostics._check_jax(report)
    backend_line = next(line for line in report.lines if "jax backend:" in line)
    assert ("[PASS]" in backend_line) is accelerator
    assert ("CPU-only" in backend_line) is (not accelerator)
    mesh = SimpleNamespace(devices=np.array(devices))
    assert _is_gpu_mesh(mesh) is gpu_mesh


def _raw_comparisons(source):
    """Find dispatch on raw backend values, including aliases and transforms.

    Track provenance to a fixed point in lexical scopes. Arbitrary consumers
    (including backend_class) end provenance; methods and string/collection
    wrappers preserve it. A device itself is only raw when rendered as text:
    membership/identity checks on device objects remain legitimate.
    """
    tree = ast.parse(source)
    scopes, parents, bindings = {}, {0: None}, {0: {}}
    collections = {"sorted", "set", "list", "tuple", "frozenset", "iter", "next", "reversed"}
    lookups = {"get", "__getitem__", "pop", "setdefault", "__contains__"}

    class Scopes(ast.NodeVisitor):
        scope = 0

        def visit(self, node):
            scopes[node] = self.scope
            super().visit(node)

        def nested(self, node):
            outer = self.scope
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings[outer].setdefault(node.name, set())
            self.scope = len(parents)
            parents[self.scope], bindings[self.scope] = outer, {}
            self.generic_visit(node)
            self.scope = outer

        visit_FunctionDef = nested
        visit_AsyncFunctionDef = nested
        visit_Lambda = nested
        visit_ClassDef = nested
        visit_ListComp = nested
        visit_SetComp = nested
        visit_DictComp = nested
        visit_GeneratorExp = nested

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                bindings[self.scope].setdefault(node.id, set())

        def visit_arg(self, node):
            bindings[self.scope].setdefault(node.arg, set())
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                kind = alias.name if node.module in ("jax", "builtins") else ""
                bindings[self.scope][alias.asname or alias.name] = {kind}

    Scopes().visit(tree)

    def lookup(node):
        scope = scopes[node]
        while scope is not None:
            if node.id in bindings[scope]:
                return bindings[scope][node.id]
            scope = parents[scope]
        return {node.id} if node.id in collections | {"str", "getattr"} else set()

    def kinds(node):
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            return lookup(node)
        if isinstance(node, ast.Constant):
            return {"string"} if isinstance(node.value, str) else set()
        if isinstance(node, ast.Attribute):
            if node.attr in ("platform", "device_kind"):
                return {"raw"}
            # Preserve bound method aliases as well as direct method calls.
            if "raw" in kinds(node.value):
                return {"raw", node.attr}
            if node.attr in {"default_backend", "devices", "local_devices"} | lookups:
                return {node.attr}
            return set()
        if isinstance(node, ast.Call):
            func = node.func
            called = kinds(func)
            args = set().union(*(kinds(arg) for arg in node.args))
            # Any method on a raw value preserves provenance, including aliases.
            if "raw" in called:
                return {"raw"}
            if "default_backend" in called:
                return {"raw"}
            if called.intersection({"devices", "local_devices"}):
                return {"device"}
            if "getattr" in called and len(node.args) > 1:
                attr = node.args[1]
                if isinstance(attr, ast.Constant) and attr.value in ("platform", "device_kind"):
                    return {"raw"}
            if "str" in called:
                return {"raw"} if args.intersection({"raw", "device"}) else {"string"}
            if called.intersection(collections):
                return args
            return set()
        if isinstance(node, ast.Compare):
            return set()  # The comparison's result is a boolean, not metadata.
        if isinstance(node, ast.Subscript):
            return kinds(node.value)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return kinds(node.elt)
        if isinstance(node, ast.DictComp):
            return {"dict"} | kinds(node.key) | kinds(node.value)
        if isinstance(node, ast.IfExp):
            return kinds(node.body) | kinds(node.orelse)
        if isinstance(node, ast.Dict):
            return {"dict"}.union(*(kinds(n) for n in (*node.keys, *node.values)))
        return set().union(*(kinds(child) for child in ast.iter_child_nodes(node)))

    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assignments.extend((target, node.value) for target in targets)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            assignments.append((node.target, node.iter))
    while True:
        changed = False
        for target, value in assignments:
            tags = kinds(value)
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                    bound = bindings[scopes[name]][name.id]
                    if not tags <= bound:
                        bound.update(tags)
                        changed = True
        if not changed:
            break

    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if any("raw" in kinds(n) for n in (node.left, *node.comparators)):
                hits.add(node.lineno)
        elif isinstance(node, ast.Match):
            if "raw" in kinds(node.subject):
                hits.add(node.lineno)
        elif isinstance(node, ast.Subscript):
            if "raw" in kinds(node.slice):
                hits.add(node.lineno)
        elif isinstance(node, ast.Call):
            called = kinds(node.func)
            if called.intersection({"devices", "local_devices"}):
                selectors = list(node.args) + [kw.value for kw in node.keywords if kw.arg == "backend"]
                if any(kinds(arg).intersection({"string", "raw"}) for arg in selectors):
                    hits.add(node.lineno)
            if "raw" in called and called.intersection({"startswith", "endswith"}):
                hits.add(node.lineno)
            if called.intersection(lookups):
                if any("raw" in kinds(arg) for arg in node.args):
                    hits.add(node.lineno)
    return sorted(hits)


@pytest.mark.source_scan
def test_no_raw_backend_string_comparisons():
    root = Path(__file__).resolve().parents[2]
    hits = []
    for path in sorted((root / "rfx").rglob("*.py")):
        if path == root / "rfx/backend.py":
            continue
        hits.extend(f"{path.relative_to(root)}:{line}" for line in _raw_comparisons(path.read_text()))
    assert not hits, "Read backend_class() instead: " + ", ".join(hits)


@pytest.mark.parametrize("source", [
    'import jax; x = jax.default_backend() != "cpu"',
    'from jax import default_backend as db; x = "mps" == db()',
    'x = d.platform in ("gpu", "cuda", "rocm")',
    'x = d.device_kind == "gpu"',
    'p = d.platform; q = p; x = q == "gpu"',
    'platforms = {d.platform for d in devices}; x = platforms == {"gpu"}',
    'x = getattr(d, "platform", None) == "gpu"',
    'platforms = sorted({d.platform for d in devices}); x = any(p == "gpu" for p in platforms)',
    pytest.param('x = d.platform.lower() == "gpu"', id="lower-platform"),
    pytest.param('x = jax.default_backend().upper() == "GPU"', id="upper-backend"),
    pytest.param('x = str(jax.default_backend()) == "gpu"', id="str-backend"),
    pytest.param('GPU = "gpu"; x = jax.default_backend() == GPU', id="module-constant"),
    pytest.param('def g(d):\n    GPU = "gpu"\n    return d.platform == GPU', id="local-constant"),
    pytest.param('f = jax.default_backend; x = f() == "gpu"', id="callable-alias"),
    pytest.param('match jax.default_backend():\n    case "gpu": pass', id="match"),
    pytest.param('raw = d.platform; x = {"gpu": 1}.get(raw)', id="dict-get"),
    pytest.param('raw = d.platform; x = raw.startswith("gpu")', id="startswith"),
    pytest.param('raw = d.platform; x = raw.endswith("gpu")', id="endswith"),
    pytest.param('x = "gpu" in str(jax.devices()[0])', id="str-device"),
    pytest.param('jax.devices("gpu")', id="devices-string"),
    pytest.param('x = d.platform == other', id="any-comparison"),
    pytest.param('x = None in {d.platform for d in devices}', id="non-string-membership"),
    pytest.param('x = {"gpu": 1}[d.platform]', id="dict-subscript"),
    pytest.param('GPU = "gpu"; jax.devices(GPU)', id="devices-constant"),
    pytest.param('jax.devices(backend="gpu")', id="devices-keyword"),
    pytest.param('jax.devices(d.platform)', id="devices-raw-selector"),
    pytest.param('from jax import devices as ds; ds("gpu")', id="devices-import-alias"),
    pytest.param('f = jax.devices; f("gpu")', id="devices-callable-alias"),
    pytest.param('x = "gpu" in str(jax.devices()[0]).lower()', id="str-device-method"),
    pytest.param('x = b.get_backend().platform == "gpu"', id="get-backend-platform"),
    pytest.param('x = f"{jax.default_backend()}" == "gpu"', id="fstring-backend"),
    pytest.param('raw = d.platform; f = raw.startswith; f("gpu")', id="predicate-alias"),
    pytest.param('get = {"gpu": 1}.get; get(d.platform)', id="dict-get-alias"),
    pytest.param('text = str; x = text(jax.default_backend()) == "gpu"', id="str-alias"),
    pytest.param('x = d.device_kind.lower() != "gpu"', id="device-kind-method"),
])
def test_source_guard_detects_raw_forms(source):
    assert _raw_comparisons(source)


def test_source_guard_allows_classification_and_metadata():
    assert not _raw_comparisons('''
info = {"platform": d.platform, "kind": d.device_kind}
backend = jax.default_backend()
rendered = str(jax.devices()[0]).lower()
saved = info["platform"]
log.info("backend: %s; platform: %s", backend, saved)
x = backend_class(d.platform).name == "mps"
y = backend_class().cpml_writeback == "slab"
platforms = {backend_class(d.platform) for d in devices}
z = next(iter(platforms)).name in ("gpu", "cuda", "rocm")
selected = jax.devices(backend_class("cpu").name)
member = device in jax.devices()
def shadow(backend):
    return backend == "a diagnostic label"
''')


@pytest.mark.parametrize("platforms,gpu", [
    ([], True), ([None, None], True), ([None, "cpu"], True),
    (["None", "None"], False), (["cpu", "mps"], True),
])
def test_mesh_metadata_checks_classification_records(platforms, gpu):
    from rfx.stepping.rank import _is_gpu_mesh

    devices = [SimpleNamespace(platform=platform) for platform in platforms]
    mesh = SimpleNamespace(devices=np.asarray(devices, dtype=object))
    assert _is_gpu_mesh(mesh) is gpu

"""Same-process old/new ports move checks; no stored measurements or golden bits.

Run with mac-serial and the repository Python. The base is the launch commit.
AST mode checks expanded calls, including the extracted eager loop, mechanically.
"""
from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import importlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = "4925b870792437563f171f556ad9a80b776e7f1c"
NAMES = {
    "rfx.probes.probes": ["extract_s_matrix", "extract_s_matrix_wire"],
    "rfx.probes.refplane": ["decompose_wire_s_matrix_with_reference_planes"],
    "rfx.probes.msl_wave_decomp": ["_lstsq_alpha_gamma", "extract_msl_nprobe"],
    "rfx.deembed": ["deembed_port_extension"],
    "rfx.sparams._common": ["_mixed_flux_magnitude_override"],
    "rfx.sparams.waveguide": ["_compute_waveguide_s_matrix_nu"],
    "rfx.io": ["_parse_pair"],
}


def source(module, old=False):
    path = module.replace(".", "/") + ".py"
    if old:
        return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT).decode()
    return (ROOT / path).read_text()


def function(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)


class Substitute(ast.NodeTransformer):
    def __init__(self, bindings):
        self.bindings = bindings

    def visit_Name(self, node):
        replacement = copy.deepcopy(self.bindings.get(node.id, node))
        if isinstance(replacement, ast.Name):
            replacement.ctx = node.ctx
        return replacement


class Expand(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "exp_j":
            kw = {k.arg: k.value for k in node.keywords}
            value = kw.get("unit", ast.Constant(1j))
            for factor in node.args:
                value = ast.BinOp(value, ast.Mult(), factor)
            return ast.Call(ast.Attribute(kw.get("xp", ast.Name("np", ast.Load())), "exp", ast.Load()), [value], [])
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.UAdd) and isinstance(node.operand, ast.Constant) and node.operand.value == 1j:
            return node.operand
        return node

    def visit_ImportFrom(self, node):
        if node.module == "rfx.stepping.probe_loop" and node.names[0].name == "run_probe_loop":
            node.names[0].name = "make_probe_step"
        return node

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "run_probe_loop":
            helper = function(source("rfx.stepping.probe_loop"), "run_probe_loop")
            bindings = dict(zip([a.arg for a in helper.args.args], node.value.args))
            bindings.update({k.arg: k.value for k in node.value.keywords})
            return [Substitute(bindings).visit(copy.deepcopy(n)) for n in helper.body[1:-1]]
        return self.generic_visit(node)


def ast_check():
    for module, names in NAMES.items():
        for name in names:
            old = function(source(module, True), name)
            target = "rfx.measurement.msl_waves" if name == "_lstsq_alpha_gamma" else module
            new = function(source(target), name)
            old_normal = Expand().visit(copy.deepcopy(old))
            new_normal = Expand().visit(copy.deepcopy(new))
            assert ast.dump(old_normal) == ast.dump(new_normal), (module, name)
            old_calls = [ast.unparse(n) for n in ast.walk(old) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "exp"]
            new_calls = [ast.unparse(n) for n in ast.walk(new) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "exp_j"]
            print(f"AST PASS {module}.{name}: " + (repr(list(zip(old_calls, new_calls))) if new_calls else "verbatim move" if target != module else "ImportFrom + builder/carry/For -> run_probe_loop (expanded AST identical)"), flush=True)


def pair(module, name):
    current = importlib.import_module(module)
    old_globals = dict(vars(current))
    # Old dependencies also execute their old bodies, rather than reusing the moved solve.
    if module == "rfx.probes.msl_wave_decomp":
        for dependency in ["_lstsq_alpha_gamma", "_estimate_beta"]:
            exec(compile(ast.fix_missing_locations(ast.Module([function(source(module, True), dependency)], [])), module, "exec"), old_globals)
        from rfx._precision import HIGHEST
        old_globals["HIGHEST"] = HIGHEST
    exec(compile(ast.fix_missing_locations(ast.Module([function(source(module, True), name)], [])), module, "exec"), old_globals)
    return old_globals[name], getattr(current, name)


def equal(a, b):
    import numpy as np
    if dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            equal(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif a is None or isinstance(a, str):
        assert a == b
    else:
        x, y = np.asarray(a), np.asarray(b)
        assert x.shape == y.shape and x.dtype == y.dtype
        assert x.tobytes() == y.tobytes(), (x, y)


def check(module, name, *args, **kwargs):
    old, new = pair(module, name)
    equal(old(*args, **kwargs), new(*args, **kwargs))
    print(f"BITS PASS {module}.{name}", flush=True)


def measurements():
    import numpy as np
    import jax.numpy as jnp
    from rfx.measurement.phasors import exp_j
    for xp in (np, jnp):
        for dtype in (xp.float32, xp.complex64):
            x = xp.asarray([0.13, -0.7, 1.5], dtype=dtype)
            for unit in (1j, -1j):
                equal(xp.exp(unit * x * x), exp_j(x, x, unit=unit, xp=xp))
    print("BITS PASS exp_j: numpy/jax x float32/complex64 x +/-1j", flush=True)
    for dtype in (np.float32, np.float64):
        for fmt in ("RI", "MA", "DB"):
            check("rfx.io", "_parse_pair", dtype(0.7), dtype(-37.2), fmt)
    s = np.full((2, 2, 3), 0.3 + 0.2j, dtype=np.complex64)
    check("rfx.deembed", "deembed_port_extension", s, np.array([1e9, 2e9, 3e9], np.float32), [0.003, -0.002])
    x = jnp.array([0.0, 0.001, 0.002, 0.003], jnp.float32)
    beta = jnp.asarray(130 + 0.01j, jnp.complex64)
    v = 0.8 * jnp.exp(-1j * beta * x) + (0.1 + 0.2j) * jnp.exp(1j * beta * x)
    check("rfx.probes.msl_wave_decomp", "_lstsq_alpha_gamma", v, x, beta)
    check("rfx.probes.msl_wave_decomp", "extract_msl_nprobe", v, x, jnp.array([0.014 - 0.004j], jnp.complex64), jnp.real(beta))
    box = np.array([[[2., 3., 4.]], [[-1., -2., -3.]]], np.float32)
    plane = np.array([[[-0.3, -0.4, -0.5]], [[1., 2., 3.]]], np.float32)
    check("rfx.sparams._common", "_mixed_flux_magnitude_override", s, box, plane, [("lw", 0), ("msl", 0)], np.array([1]), 1)
    fixture = importlib.import_module("tests.locks.test_refplane_port_waves")
    for gamma in (0.0, 0.18 * np.exp(0.4j)):
        data = fixture._two_port_synthetic(gamma)
        check("rfx.probes.refplane", "decompose_wire_s_matrix_with_reference_planes",
              data["v_all"], data["i_all"], data["z0"], data["counts"],
              **{key: data[key] for key in ("plane_v", "plane_im", "plane_ip", "freqs", "dt", "dx")},
              plane_enabled=np.array([True, True]), plane_offsets=np.array([data["n"], data["n"]]),
              outboard_signs=np.array([1, -1]), return_line_diagnostics=True)


def drivers():
    import jax.numpy as jnp
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.sources.sources import GaussianPulse, LumpedPort, WirePort
    grid = Grid(freq_max=2e9, domain=(0.1, 0.1, 0.1), dx=0.005, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=1e9, bandwidth=0.8)
    freqs = jnp.asarray([0.8e9, 1.2e9], jnp.float32)
    lumped = [LumpedPort((x, 0.05, 0.05), "ez", 50., pulse) for x in (0.025, 0.05, 0.075)]
    wire = [WirePort((x, 0.05, 0.04), (x, 0.05, 0.06), "ez", 50., pulse) for x in (0.025, 0.075)]
    print(f"driver grid={grid.shape}, steps=200; 3 lumped / 2 wire ports", flush=True)
    for name, ports in [("extract_s_matrix", lumped), ("extract_s_matrix_wire", wire)]:
        check("rfx.probes.probes", name, grid, materials, ports, freqs, n_steps=200, return_vi_dump=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["ast", "measurements", "drivers"], required=True)
    args = parser.parse_args()
    {"ast": ast_check, "measurements": measurements, "drivers": drivers}[args.case]()

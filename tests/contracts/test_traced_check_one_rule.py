"""#1830: the eleven real check closures obey one concrete/traced rule."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, _periodic, _realized, _traced_check
from rfx.api._source_semantics import guard_float16_source_increment
from rfx.backend import backend_class
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import MaterialArrays
from rfx.model import conductors, graded_grid, pole_stability
from rfx.preflight import sources


def invoke(site, x):
    if site == "periodic interval":
        _periodic.interval_coordinates(
            SimpleNamespace(periodic_axes="x", domain=(1.0,) * 3), 0, x, 2.0
        )
    elif site == "periodic fill":
        _periodic._fill_image_range(
            SimpleNamespace(domain=(1.0,) * 3), Box((x, 0.0, 0.0), (3.0, 1.0, 1.0)), 0
        )
    elif site == "realized record":
        _realized.Capture().observe("test", dict(waveforms=(x,), raw_waveform=x))
    elif site == "graded floor":
        graded_grid._assert_declared_floor_at_runtime(x, 1.0, "y", 0.1, 0.2)
    elif site == "graded fixed cells":
        graded_grid._assert_traced_cells_match_nominal(x, np.ones(2), (0, 1), "y")
    elif site == "graded node index":
        graded_grid._assert_traced_index_matches_realized(x, 0, 0, 1.0, 0, "y")
    elif site == "pole stability":
        rec = pole_stability.PoleSetExtremes(1, ((0.0, 0.0, 9.0),), x, 0.0, 0.0, 1, ())
        pole_stability.judge_pole_sets([rec], 1.0, 0.0, grid_kind="Grid")
    elif site == "sheet film overlap":
        ctx = SimpleNamespace(mask_ex=np.ones((1,), bool), mask_ey=None, mask_ez=None)
        obj = SimpleNamespace(
            sheet_impedance=(),
            periodic=(False,) * 3,
            materials=SimpleNamespace(sigma_film=(x, None, None)),
        )
        with patch(
            "rfx.materials.thin_conductor.build_sheet_impedance_ctx", return_value=ctx
        ):
            conductors.SheetConductors.sheet_context(obj, None)
    elif site in ("TFSF box", "TFSF slab"):
        cfg = SimpleNamespace(
            closed_box=site == "TFSF box",
            **{a + "_" + end: v for a in "xyz" for end, v in [("lo", 2), ("hi", 3)]},
        )
        materials = MaterialArrays(x, jnp.zeros_like(x), jnp.ones_like(x))
        sources._validate_tfsf_vacuum_boundary(materials, cfg)
    elif site == "source increment":
        guard_float16_source_increment(x, np.float16, "field")
    return jnp.float32(0.0)


INPUTS = {
    "periodic interval": np.float32(-1.0),
    "periodic fill": np.float32(-2.0),
    "realized record": np.zeros(2, np.float32),
    "graded floor": np.array([0.5], np.float32),
    "graded fixed cells": np.array([1.0, 2.0], np.float32),
    "graded node index": np.ones(2, np.float32),
    "pole stability": np.float32(1.0),
    "sheet film overlap": np.ones(1, np.float32),
    "TFSF box": np.full((6,) * 3, 2.0, np.float32),
    "TFSF slab": np.full((6,) * 3, 2.0, np.float32),
    "source increment": np.array([70000.0], np.float32),
}


# Captured by executing the original checks before the implementation changed.
MESSAGES = {
    "periodic interval": "periodic axis 'x': interval [-1.0, 2.0] m crosses the seam of period L=1.0 m; split the interval into declarations inside [0, L].",
    "periodic fill": "periodic axis 'x': fill bounds [-2.0, 3.0] m must lie inside [-L, 2L] for traced bounds; L=1.0 m",
    "realized record": "a zero waveform cannot measure a drive scale",
    "graded floor": "dt_min_cell=1000 mm was declared for the traced y profile, but the mesh this run built has a smallest y cell of 500 mm. The pinned dt=1.000000e-01 s was checked against the declared floor (Courant limit 2.000000e-01 s there) and is not guaranteed stable on the realized mesh — the fields would diverge with nothing said. Declare the smallest cell the deformation family actually reaches, or shorten dt.",
    "graded fixed cells": "traced y axis interior cell 1 has realized width 2.0, nominal width 1.0; outside declared traced_cells window (0, 1).",
    "graded node index": "position 1 m on the traced y axis resolves to interior node 0 on the nominal uniform mesh this lane must use for a structural index, but to node 1 on the mesh the run built. The two runs would place this source / probe / port at different points of the board. Declare the y deformation so the node lines outside it do not move (stretch and shrink inside a window), or give the position by node index. Or declare the concrete y profile with nominal_profiles=.",
    "pole stability": "Lorentz/Drude poles (omega_0=0, delta=0, kappa=9) on 1 cells, eps_low=1, sigma_low=0, sigma_high=0: unstable at this model's time step dt = 1 s; growth per step = 6.85410197; largest stable step for this set is 0.6666666666667496 s; dt/stable = 1.5. To run: use finer cells, or declare a dz_profile/dx_profile/dy_profile so that dt=0.6666666666667496 can be passed; or raise eps_inf.",
    "sheet film overlap": "an f0 sheet and a lossy film share edges on x at [[0]]",
    "TFSF box": "closed_box TFSF requires vacuum on and adjacent to all six faces; non-vacuum material found at x=1",
    "TFSF slab": "TFSF plane-wave source requires vacuum on and adjacent to the TFSF x boundaries; non-vacuum material found at x_lo-1",
    "source increment": "float16 soft-source one-step increment 70000 exceeds the float16 maximum (65504); scale the waveform.",
}


NAMES = {
    "periodic interval": ("periodic geometry", "interval seam"),
    "periodic fill": ("periodic geometry", "fill bounds"),
    "realized record": ("realized geometry", "record operands"),
    "graded floor": ("graded mesh", "declared cell floor"),
    "graded fixed cells": ("graded mesh", "fixed cells"),
    "graded node index": ("graded mesh", "node index"),
    "pole stability": ("Lorentz/Drude poles", "time-step stability"),
    "sheet film overlap": ("thin conductors", "sheet/film overlap"),
    "TFSF box": ("closed_box TFSF", "vacuum faces"),
    "TFSF slab": ("TFSF slab", "vacuum boundaries"),
    "source increment": ("float16 soft source", "one-step increment"),
}


@pytest.fixture
def no_callbacks(monkeypatch):
    monkeypatch.setattr(_traced_check, "backend_class", lambda: backend_class("mps"))


def _check_operands(site, *, invalid):
    x = float(invalid)
    if site == "periodic interval":
        return (-x, 2.0 if invalid else 0.5)
    if site == "periodic fill":
        return (np.float32(-2 * x), np.float32(3.0 if invalid else 0.5))
    if site == "realized record":
        samples = np.full(2, 1 - x, np.float32)
        return (dict(waveforms=(samples,), raw_waveform=samples),)
    if site == "graded floor":
        return (np.float32(0.5 if invalid else 1.1),)
    if site == "graded fixed cells":
        return (bool(invalid), np.int32(1), np.float32(1 + x), np.float32(1))
    if site == "graded node index":
        return (np.int32(invalid), not invalid)
    if site == "pole stability":
        return (1.0, 0.0, 0.0, 1.0 if invalid else 0.1, 0.0, 0.0, 9.0)
    if site == "sheet film overlap":
        return (np.array([invalid]),)
    if site in ("TFSF box", "TFSF slab"):
        group = tuple(np.full((1, 1, 1), v, np.float32) for v in (1 + x, 0.0, 1.0))
        return group * (12 if site == "TFSF box" else 4)
    return (np.array([70000.0 if invalid else 1.0], np.float32),)


class _Captured(Exception):
    pass


def _capture_check(monkeypatch, site):
    captured = []

    def capture(*args, **kwargs):
        captured.append((args[:3], kwargs))
        raise _Captured()

    with monkeypatch.context() as scoped:
        # Two leaf owners import the helper at module load; others import locally.
        for module in (_traced_check, _periodic, _realized):
            scoped.setattr(module, "traced_check", capture)
        with pytest.raises(_Captured):
            jax.make_jaxpr(lambda x: invoke(site, x))(INPUTS[site])
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("site", INPUTS)
def test_concrete_checks_run_with_original_messages(monkeypatch, no_callbacks, site):
    (feature, name, check), kwargs = _capture_check(monkeypatch, site)
    assert (feature, name) == NAMES[site]
    calls = []

    def observed(*operands):
        calls.append(True)
        return check(*operands)

    def forbidden(*args, **kwargs):
        pytest.fail("a concrete check scheduled a callback")

    monkeypatch.setattr(jax.debug, "callback", forbidden)
    monkeypatch.setattr("jax.experimental.io_callback", forbidden)
    _traced_check.traced_check(
        feature, name, observed, *_check_operands(site, invalid=False), **kwargs
    )
    with pytest.raises(ValueError) as caught:
        _traced_check.traced_check(
            feature, name, observed, *_check_operands(site, invalid=True), **kwargs
        )
    assert str(caught.value) == MESSAGES[site]
    assert len(calls) == 2


@pytest.mark.parametrize("site", INPUTS)
def test_each_owner_refuses_traced_operands(no_callbacks, site):
    with pytest.raises(NotImplementedError) as caught:
        jax.make_jaxpr(lambda x: invoke(site, x))(INPUTS[site])
    feature, check = NAMES[site]
    assert str(caught.value) == (
        f"{feature}: {check}: this backend (mps) cannot run a host check inside compiled code"
    )


@pytest.mark.parametrize("platform", ["cpu", "gpu", "cuda", "rocm"])
def test_callback_backends_keep_ordered_checks_and_io_tokens(monkeypatch, platform):
    monkeypatch.setattr(_traced_check, "backend_class", lambda: backend_class(platform))
    seen = []

    def check(x):
        seen.append(float(x))
        return np.int32(0)

    # Use a scalar check for the nested debug payload while exercising leaf traversal.
    def nested(x):
        check(x["nested"])

    def run(x):
        _traced_check.traced_check("test", "debug", nested, {"nested": x}, ordered=True)
        token = _traced_check.traced_check(
            "test", "token", check, x, result_shape=jax.ShapeDtypeStruct((), jnp.int32)
        )
        return x + token

    program = jax.make_jaxpr(run)(jnp.float32(2))
    assert any(e.primitive.name == "debug_callback" for e in program.jaxpr.eqns)
    assert any(e.primitive.name == "io_callback" for e in program.jaxpr.eqns)
    assert any(type(e).__name__ == "OrderedDebugEffect" for e in program.effects)
    assert float(jax.jit(run)(jnp.float32(2))) == 2.0
    jax.effects_barrier()
    assert seen == [2.0, 2.0]


_HOST_CALLBACKS = {
    "jax.debug.callback",
    "jax.debug.print",
    "jax.pure_callback",
    "jax.experimental.io_callback",
    "jax.experimental.host_callback",
}


def _callback_references(source):
    """Resolve static imports and name bindings, including references without calls.

    Keep every possible binding within a lexical scope: conditional assignments
    must not hide a callback. No source is imported or executed by this guard.
    """
    hits = set()
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    def scan(tree, outer):
        nodes, children = [], []

        def collect(node):
            nodes.append(node)
            if isinstance(node, scopes):
                children.append(node)
                # Defaults, annotations, decorators and class bases execute in
                # the enclosing scope, before parameters can shadow imports.
                for field, value in ast.iter_fields(node):
                    if field == "body":
                        continue
                    if field == "args":
                        value = (
                            value.defaults
                            + value.kw_defaults
                            + [
                                arg.annotation
                                for arg in ast.walk(value)
                                if isinstance(arg, ast.arg)
                            ]
                        )
                    for child in value if isinstance(value, list) else [value]:
                        if isinstance(child, ast.AST):
                            collect(child)
            else:
                for child in ast.iter_child_nodes(node):
                    collect(child)

        if isinstance(tree, scopes):
            body = tree.body if isinstance(tree.body, list) else [tree.body]
            if hasattr(tree, "args"):
                nodes.extend(
                    node for node in ast.walk(tree.args) if isinstance(node, ast.arg)
                )
        else:
            body = list(ast.iter_child_nodes(tree))
        for node in body:
            collect(node)
        # Local names shadow outer aliases. Imports and assignments below add
        # their possible paths before references or nested scopes are checked.
        aliases = {name: set(paths) for name, paths in outer.items()}
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                aliases[node.id] = set()
            elif isinstance(node, ast.arg):
                aliases[node.arg] = set()
            elif isinstance(node, scopes) and hasattr(node, "name"):
                aliases[node.name] = set()
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or (
                        alias.name.split(".")[0]
                        if isinstance(node, ast.Import)
                        else alias.name
                    )
                    aliases[name] = set()
        bindings = []
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    path = alias.name if alias.asname else name
                    aliases.setdefault(name, set()).add(path)
                    if alias.name in _HOST_CALLBACKS:
                        hits.add((node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and not node.level:
                for alias in node.names:
                    if alias.name == "*":
                        for api in _HOST_CALLBACKS:
                            module, name = api.rsplit(".", 1)
                            if module == node.module:
                                aliases.setdefault(name, set()).add(api)
                                hits.add((node.lineno, api))
                        continue
                    path = f"{node.module}.{alias.name}"
                    aliases.setdefault(alias.asname or alias.name, set()).add(path)
                    if path in _HOST_CALLBACKS:
                        hits.add((node.lineno, path))
            elif isinstance(node, ast.Assign):
                bindings.extend((target, node.value) for target in node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                bindings.append((node.target, node.value))

        def resolve(node):
            if isinstance(node, ast.Name):
                return aliases.get(node.id, set())
            if isinstance(node, ast.Attribute):
                return {f"{path}.{node.attr}" for path in resolve(node.value)}
            if isinstance(node, ast.NamedExpr):
                return resolve(node.value)
            return set()

        def bind(target, value):
            if isinstance(target, ast.Name):
                # Only prefixes of a forbidden API matter. This also bounds
                # fixed-point iteration for cyclic assignments such as x=x.y.
                aliases.setdefault(target.id, set()).update(
                    path
                    for path in resolve(value)
                    if any(
                        api == path or api.startswith(path + ".")
                        for api in _HOST_CALLBACKS
                    )
                )
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                value, (ast.Tuple, ast.List)
            ):
                for item, expression in zip(target.elts, value.elts):
                    bind(item, expression)

        while True:
            before = {name: set(paths) for name, paths in aliases.items()}
            for target, value in bindings:
                bind(target, value)
            if aliases == before:
                break
        for node in nodes:
            if isinstance(node, (ast.Attribute, ast.Name)):
                hits.update(
                    (node.lineno, path) for path in resolve(node) & _HOST_CALLBACKS
                )
        for child in children:
            scan(child, aliases)

    scan(ast.parse(source), {})
    return sorted(hits)


def _assert_callback_references_allowed(root):
    allowed = {"_traced_check.py", "_ringdown_jax.py"}
    forbidden = []
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).as_posix() in allowed:
            continue
        forbidden.extend(
            (str(path.relative_to(root)), line, api)
            for line, api in _callback_references(path.read_text())
        )
    assert not forbidden


@pytest.mark.source_scan
def test_callback_calls_stay_in_the_helper_and_ringdown():
    _assert_callback_references_allowed(Path(__file__).resolve().parents[2] / "rfx")


@pytest.mark.parametrize(
    "source, api",
    [
        ("from jax import debug; debug.callback(f)", "jax.debug.callback"),
        ("import jax.debug as d; d.callback(f)", "jax.debug.callback"),
        ("import jax; cb = jax.debug.callback; cb(f)", "jax.debug.callback"),
        ("import jax; jax.experimental.io_callback(f)", "jax.experimental.io_callback"),
        ("import jax; jax.debug.print('value')", "jax.debug.print"),
        (
            "from jax.experimental import host_callback",
            "jax.experimental.host_callback",
        ),
        ("import jax as j; j.pure_callback", "jax.pure_callback"),
        ("from jax.debug import print as p; p", "jax.debug.print"),
        (
            "from jax.experimental import io_callback as cb; cb",
            "jax.experimental.io_callback",
        ),
        (
            "import jax.experimental.host_callback as hc",
            "jax.experimental.host_callback",
        ),
        (
            "import jax as j; d = j.debug; other = d; other.callback",
            "jax.debug.callback",
        ),
        ("import jax; d: object = jax.debug; d.print", "jax.debug.print"),
        ("import jax; (d := jax.debug).callback", "jax.debug.callback"),
        ("import jax; d, x = jax.debug, 1; d.callback", "jax.debug.callback"),
        ("def f():\n    d.callback\nimport jax.debug as d", "jax.debug.callback"),
        ("from jax.debug import *\ncallback", "jax.debug.callback"),
        ("import jax\ndef f(jax=jax.debug.callback): pass", "jax.debug.callback"),
        (
            "def f():\n    from jax import debug as d\n    d.callback",
            "jax.debug.callback",
        ),
    ],
)
def test_callback_guard_rejects_references_then_restores(tmp_path, source, api):
    path = tmp_path / "owner.py"
    path.write_text(source)
    with pytest.raises(AssertionError, match=api.replace(".", r"\.")):
        _assert_callback_references_allowed(tmp_path)
    path.write_text("import jax\nx = jax.numpy.ones(1)\n")
    _assert_callback_references_allowed(tmp_path)


def test_callback_guard_resolves_name_uses_and_ignores_unrelated_names():
    assert _callback_references(
        "from jax.debug import callback as cb\nother = cb\nother\n"
    ) == [
        (1, "jax.debug.callback"),
        (2, "jax.debug.callback"),
        (3, "jax.debug.callback"),
    ]
    assert not _callback_references(
        "from elsewhere import callback, io_callback, host_callback\n"
        "callback(); io_callback(); host_callback()\n"
        "import jax as j\n"
        "def f(j):\n    j.debug.callback()\n"
    )


@pytest.mark.parametrize("name", ["_traced_check.py", "_ringdown_jax.py"])
def test_callback_guard_exempts_only_root_owners(tmp_path, name):
    (tmp_path / name).write_text("import jax; jax.debug.callback")
    _assert_callback_references_allowed(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / name).write_text("import jax; jax.debug.callback")
    with pytest.raises(AssertionError, match="jax.debug.callback"):
        _assert_callback_references_allowed(tmp_path)


@pytest.mark.parametrize("graded", [False, True])
def test_ringdown_refused_at_forward_admission(monkeypatch, no_callbacks, graded):
    from rfx.ringdown import RingdownSpec

    sim = Simulation(
        freq_max=15e9,
        domain=(0.008,) * 3,
        dx=0.001,
        boundary="pec",
        **({"dy_profile": np.full(8, 0.001)} if graded else {}),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("ringdown admission started a solve")

    monkeypatch.setattr(sim, "_build_grid", forbidden)
    monkeypatch.setattr(sim, "_build_nonuniform_grid", forbidden)
    with pytest.raises(NotImplementedError) as caught:
        sim.forward(n_steps=4, ringdown=RingdownSpec())
    assert str(caught.value) == (
        "forward(ringdown=): pole identification: this backend (mps) "
        "cannot run host code inside compiled code"
    )


def _forward_scene(feature):
    # Each scene has a real inventory check: TF/SF vacuum, periodic interval,
    # or sheet/film overlap. Debye alone has no check in this inventory.
    kwargs = {"dy_profile": np.full(8, 0.001)} if feature == "graded" else {}
    boundary = (
        BoundarySpec(x="periodic", y="pec", z="pec") if feature == "periodic" else "pec"
    )
    if feature == "Debye":
        boundary = "cpml"
        kwargs["cpml_layers"] = 2
    sim = Simulation(
        freq_max=15e9, domain=(0.008,) * 3, dx=0.001, boundary=boundary, **kwargs
    )
    if feature == "Debye":
        from rfx.materials.debye import DebyePole

        sim.add_tfsf_source(f0=5e9, bandwidth=0.8, margin=2, closed_box=True)
        sim.add_material("d", eps_r=2.0, debye_poles=[DebyePole(0.5, 1e-11)])
        sim.add(Box((0.003,) * 3, (0.004,) * 3), material="d")
    else:
        sim.add_source((0.002,) * 3, "ez", amplitude_kind="field")
    sim.add_probe((0.005,) * 3, "ez")
    if feature == "periodic":
        sim.add_flux_monitor(
            axis="z",
            coordinate=0.004,
            freqs=np.array([5e9]),
            size=(0.002, 0.002),
            center=(0.004, 0.004),
        )
    if feature == "graded":
        sim.add_thin_conductor(
            Box((0.001, 0.002, 0.004), (0.003, 0.004, 0.004)),
            sigma_bulk=1e5,
            thickness=1e-5,
            surface_impedance_f0=5e9,
        )
        sim.add_thin_conductor(
            Box((0.001, 0.002, 0.006), (0.003, 0.004, 0.006)),
            sigma_bulk=1e4,
            thickness=1e-5,
        )
    return sim


@pytest.mark.parametrize("feature", ["Debye", "periodic", "graded"])
def test_faked_mps_forward_requires_the_concrete_branch(no_callbacks, feature):
    result = _forward_scene(feature).forward(
        n_steps=4, skip_preflight=True, checkpoint=False
    )
    jax.block_until_ready(result.time_series)
    assert np.isfinite(result.time_series).all()

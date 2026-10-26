"""AST-only layer ratchet: real-source mutations and Git snapshot contracts."""

import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "layer_structure_gate", REPO / "scripts/ci/check_layer_structure.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
SIM = "rfx/simulation.py"
API = "rfx/api/__init__.py"
EXECUTE = "rfx/api/_execute.py"
CLEAN = "_uniform_cell_volume"
EXECUTE_CLEAN = "_ExecuteMixin._resolve_field_dtype"


@pytest.fixture(scope="module")
def generated():
    # --write-baseline generated this artifact. Expectations add fixed names,
    # never use a second counter run to manufacture the mutation's delta.
    return gate.baseline((REPO / gate.BASELINE).read_bytes())


class CopiedSources(gate.Sources):
    """Read mutated copies from tmp_path; unchanged import targets from REPO."""

    def __init__(self, tmp_path):
        super().__init__(REPO)
        self.copies = tmp_path
        self.paths.update(p.relative_to(tmp_path).as_posix()
                          for p in tmp_path.rglob("*.py"))

    def read(self, path):
        copy = self.copies / path
        return copy.read_bytes() if copy.exists() else super().read(path)


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def real_tree(path):
    return ast.parse((REPO / path).read_text())


def target_unit(tree, name):
    return next(node for unit, node, _ in gate.units(tree) if unit == name)


def mutate(tmp_path, path, unit, code):
    tree = real_tree(path)
    target = target_unit(tree, unit)
    index = len(target.body) - len(gate.body_without_docstring(target))
    target.body[index:index] = ast.parse(code).body
    write(tmp_path, path, ast.unparse(tree) + "\n")
    return gate.count_file(CopiedSources(tmp_path), path)


def body_from(path, unit):
    node = target_unit(real_tree(path), unit)
    return "\n".join(ast.unparse(n) for n in gate.body_without_docstring(node))


def assert_new_hit(generated, path, rules, name, observed):
    for rule in rules:
        assert name not in generated["files"][path][rule]
        assert observed[rule] == sorted([*generated["files"][path][rule], name])
    actual = deepcopy(generated)
    actual["files"][path] = observed
    errors = gate.violations(actual, generated, generated)
    assert any("new violation against BASE" in error for error in errors), errors


def test_uniform_step_body_and_restored_original_baseline_unchanged(tmp_path, generated):
    target = CLEAN
    assert target not in generated["files"][SIM]["S"]
    body = body_from("rfx/stepping/uniform.py", "make_uniform_step")
    observed = mutate(tmp_path, SIM, target, body)
    assert_new_hit(generated, SIM, ["S"], target, observed)
    # Restored original: baseline unchanged. This is an identity check, not
    # evidence that a call to a make_*_step builder would pass rule S.
    write(tmp_path, SIM, (REPO / SIM).read_text())
    assert gate.count_file(CopiedSources(tmp_path), SIM) == generated["files"][SIM]


def test_phase_body_and_bound_call_control(tmp_path, generated):
    target = CLEAN
    for rule in ("M", "G"):
        assert target not in generated["files"][SIM][rule]
    observed = mutate(tmp_path, SIM, target, body_from("rfx/measurement/dft.py", "_traced_unit"))
    assert_new_hit(generated, SIM, ["M", "G"], target, observed)
    # phase lives in dft.py; it is neither re-exported nor in measurement's
    # __all__. Bind the real submodule: the call has no M/G hit but DOES hit I.
    name = "rfx.measurement.dft:phase"
    assert name not in generated["files"][SIM]["I"]
    observed = mutate(tmp_path, SIM, target,
                      "from rfx.measurement import dft\ndft.phase(step, freqs, dt)")
    assert_new_hit(generated, SIM, ["I"], name, observed)
    assert observed["M"] == generated["files"][SIM]["M"]
    assert observed["G"] == generated["files"][SIM]["G"]


def test_model_mask_body(tmp_path, generated):
    target = EXECUTE_CLEAN
    for rule in ("A", "G"):
        assert target not in generated["files"][EXECUTE][rule]
    observed = mutate(tmp_path, EXECUTE, target,
                      body_from("rfx/model/occupancy.py", "occupancy_from_design"))
    assert_new_hit(generated, EXECUTE, ["A", "G"], target, observed)


def test_kernel_import(tmp_path, generated):
    name = "rfx.core.yee:update_h"
    assert name not in generated["files"][EXECUTE]["I"]
    observed = mutate(tmp_path, EXECUTE, EXECUTE_CLEAN, "from rfx.core.yee import update_h")
    assert_new_hit(generated, EXECUTE, ["I"], name, observed)


def test_module_attribute_import_is_per_file(tmp_path, generated):
    name = "rfx.model.conductors:_same_grid"
    assert name not in generated["files"][SIM]["I"]
    assert name not in generated["files"][EXECUTE]["I"]
    observed = mutate(tmp_path, SIM, CLEAN,
                      "from rfx.model import conductors as _c2\n_c2._same_grid")
    assert_new_hit(generated, SIM, ["I"], name, observed)
    assert name not in gate.count_file(CopiedSources(tmp_path), EXECUTE)["I"]


def test_declaration_grid_mutation(tmp_path, generated):
    target = "Simulation.add_thin_conductor"
    assert target not in generated["files"][API]["D1"]
    observed = mutate(tmp_path, API, target, "self._build_grid()")
    assert_new_hit(generated, API, ["D1"], target, observed)


def test_declaration_docstring_and_argument_check_control(tmp_path, generated):
    target = "Simulation.add_thin_conductor"
    for rule in ("D1", "D2"):
        assert target not in generated["files"][API][rule]
    tree = real_tree(API)
    node = target_unit(tree, target)
    node.body[0].value.value += "\n\nThe declared thickness must be positive."
    node.body[1:1] = ast.parse("if thickness <= 0:\n    raise ValueError('thickness')").body
    write(tmp_path, API, ast.unparse(tree) + "\n")
    assert gate.count_file(CopiedSources(tmp_path), API) == generated["files"][API]


def test_three_statement_public_utility(tmp_path, generated):
    target = "Simulation.layer_check_utility"
    assert target not in generated["files"][API]["D2"]
    tree = real_tree(API)
    simulation = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Simulation")
    simulation.body.extend(ast.parse(
        "def layer_check_utility(self):\n    a = 1\n    b = 2\n    return a + b\n",
    ).body)
    write(tmp_path, API, ast.unparse(tree) + "\n")
    observed = gate.count_file(CopiedSources(tmp_path), API)
    assert_new_hit(generated, API, ["D2"], target, observed)


@pytest.mark.parametrize("code,red", [
    ("np.asarray(a / b)", True), ("np.asarray(a)", False), ("np.any(a > b)", False),
])
def test_allowlist_mutations(tmp_path, generated, code, red):
    target = CLEAN
    assert target not in generated["files"][SIM]["G"]
    observed = mutate(tmp_path, SIM, target, code)
    if red:
        assert_new_hit(generated, SIM, ["G"], target, observed)
    else:
        assert observed == generated["files"][SIM]


def test_operator_only_known_miss(tmp_path, generated):
    target = CLEAN
    for rule in "SMAG":
        assert target not in generated["files"][SIM][rule]
    # DECLARE.md §5 item 1: pure operator arithmetic with no library call is
    # an intentional known miss, not evidence that this is orchestration.
    observed = mutate(tmp_path, SIM, target, "value = ce * e + cb * curl")
    assert observed == generated["files"][SIM]


@pytest.mark.parametrize("remove_elsewhere", [False, True])
def test_outside_new_unit_in_existing_file_cannot_net_removal(tmp_path, generated, remove_elsewhere):
    path, name = "rfx/runners/_distributed_common.py", "layer_check_outside_step"
    assert generated["outside"][path]["S"]
    assert name not in {unit for unit, _, _ in gate.units(real_tree(path))}
    write(tmp_path, path, (REPO / path).read_text()
          + "\ndef layer_check_outside_step():\n    return jax.lax.scan(body, carry, xs)\n")
    expected = deepcopy(generated)
    expected["outside"][path]["S"] = sorted([*expected["outside"][path]["S"], name])
    if remove_elsewhere:
        other, old = "rfx/subgridding/runner.py", "run_subgridded"
        assert generated["outside"][other] == {"S": [old], "M": []}
        tree = real_tree(other)
        tree.body.remove(target_unit(tree, old))
        write(tmp_path, other, ast.unparse(tree) + "\n")
        del expected["outside"][other]
    actual = gate.count(CopiedSources(tmp_path))["baseline"]
    assert actual == expected
    # Even updating the baseline to match the mutated code cannot admit the copy.
    errors = gate.violations(actual, expected, generated)
    assert errors == [f"{path} S: new violation against BASE {name}",
                      f"{path} S: baseline grew against BASE {name}"]


def test_outside_rename_reports_new_and_stale_names(tmp_path, generated):
    path, old, new = "rfx/subgridding/runner.py", "run_subgridded", "renamed_scan"
    tree = real_tree(path)
    target_unit(tree, old).name = new
    write(tmp_path, path, ast.unparse(tree) + "\n")
    expected = deepcopy(generated)
    expected["outside"][path] = {"S": [new], "M": []}
    actual = gate.count(CopiedSources(tmp_path))["baseline"]
    assert actual == expected
    assert gate.violations(actual, generated, generated) == [
        f"{path} S: unlisted violation {new}",
        f"{path} S: stale baseline entry {old}",
        f"{path} S: new violation against BASE {new}",
    ]


@pytest.mark.parametrize("destination,green", [
    ("rfx/stepping/moved_scan.py", True), ("rfx/runners/moved_scan.py", False),
])
def test_outside_move_requires_layer_destination(tmp_path, generated, destination, green):
    path, name = "rfx/subgridding/runner.py", "run_subgridded"
    tree = real_tree(path)
    node = target_unit(tree, name)
    tree.body.remove(node)
    write(tmp_path, path, ast.unparse(tree) + "\n")
    write(tmp_path, destination, ast.unparse(node) + "\n")
    expected = deepcopy(generated)
    del expected["outside"][path]
    if not green:
        expected["outside"][destination] = {"S": [name], "M": []}
    actual = gate.count(CopiedSources(tmp_path))["baseline"]
    assert actual == expected
    errors = gate.violations(actual, expected, generated)
    assert errors == ([] if green else [
        f"{destination} S: new violation against BASE {name}",
        f"{destination} S: baseline grew against BASE {name}",
    ])


def test_outside_new_file_with_measurement_hit(tmp_path, generated):
    path = "rfx/outside_measurement.py"
    write(tmp_path, path, "def spectrum(x):\n    return np.fft.fft(x)\n")
    expected = deepcopy(generated)
    expected["outside"][path] = {"S": [], "M": ["spectrum"]}
    actual = gate.count(CopiedSources(tmp_path))["baseline"]
    assert actual == expected
    assert gate.violations(actual, generated, generated) == [
        f"{path} M: unlisted violation spectrum",
        f"{path} M: new violation against BASE spectrum",
    ]


def test_calibration_from_declaration_and_leader_correction(generated):
    api = generated["files"][API]
    assert "Simulation.add_thin_conductor" not in api["D1"] + api["D2"]
    # The waveguide port's plane check moved to rfx.measurement.waveguide_planes;
    # D1 detection itself stays pinned by test_declaration_grid_mutation.
    assert "Simulation.add_waveguide_port" not in api["D1"]
    # The report moved to rfx.mesh_planner.mesh_intelligence_report; the method only returns its call.
    assert "Simulation.mesh_intelligence_report" not in api["D2"]
    node = target_unit(real_tree(API), "Simulation.mesh_intelligence_report")
    assert len(gate.body_without_docstring(node)) == 1
    assert "mesh_intelligence_report" in {name for name, _, _ in gate.units(real_tree("rfx/mesh_planner.py"))}
    moved = (
        "_ad_memory_static_accounting", "estimate_ad_memory", "explain_ad_memory",
        "plan_ad_memory", "ad_memory_preflight", "ad_memory_compiled_certificate",
    )
    definitions = {name for name, _, _ in gate.units(real_tree("rfx/api/_ad_memory.py"))}
    api_definitions = {name for name, _, _ in gate.units(real_tree(API))}
    for name in moved:
        assert f"Simulation.{name}" not in api["D2"]
        assert f"Simulation.{name}" not in api_definitions
        assert f"_AdMemoryMixin.{name}" in definitions
    nonuniform = generated["files"]["rfx/nonuniform.py"]
    all_units = set().union(*(nonuniform[r] for r in "SMAG"))
    assert "run_nonuniform" not in all_units
    assert "run_nonuniform" in {name for name, _, _ in gate.units(real_tree("rfx/stepping/graded_loop.py"))}
    assert "run_nonuniform" in gate.public_names(gate.Sources(REPO), "rfx.stepping")
    # Graded-grid construction (NonUniformGrid, its module-level tables) moved to
    # rfx/model/ in #1643; nonuniform.py now only imports it, counted under I.
    assert "NonUniformGrid.node_of" not in all_units
    assert "rfx.model.graded_grid:make_nonuniform_grid" in nonuniform["I"]
    assert "rfx.model.materials:_design_box_edge_coeffs" not in generated["files"][SIM]["I"]


def empty_baseline():
    return {"files": {p: {r: [] for r in gate.RULES} for p in gate.FIVE_FILES},
            "outside": {}, "reverse": {}}


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.delenv("HEAD_SHA", raising=False)
    for path in gate.FIVE_FILES:
        write(tmp_path, path, "")
    for package in gate.LAYERS:
        write(tmp_path, package.replace(".", "/") + "/__init__.py", "")
    write(tmp_path, gate.BASELINE, json.dumps(empty_baseline()))
    return tmp_path


def mock_base(monkeypatch, previous):
    def git(repo, *args):
        if args == ("ls-tree", "base", "--", gate.BASELINE):
            return b"present"
        assert args == ("show", f"base:{gate.BASELINE}")
        return json.dumps(previous).encode()
    monkeypatch.setattr(gate, "git", git)


@pytest.mark.parametrize("case,expected,message", [
    ("new", 1, "new violation against BASE new"),
    ("shrink", 0, ""),
    ("grow", 1, "baseline grew against BASE new"),
    ("stale", 1, "stale baseline entry old"),
])
def test_ratchet_synthetic_baselines(synthetic, monkeypatch, capsys, case, expected, message):
    previous, current = empty_baseline(), empty_baseline()
    previous["files"][SIM]["G"] = ["old"]
    if case in {"new", "grow"}:
        write(synthetic, SIM, "def new():\n    return np.zeros(1)\n")
        current["files"][SIM]["G"] = ["new"] if case == "grow" else []
    elif case == "stale":
        current["files"][SIM]["G"] = ["old"]
    write(synthetic, gate.BASELINE, json.dumps(current))
    mock_base(monkeypatch, previous)
    assert gate.main(["--repo", str(synthetic), "--base", "base"]) == expected
    output = capsys.readouterr()
    assert message in output.err
    assert "units=" in output.out and "outside totals:" in output.out and "surface sizes:" in output.out


def test_unreadable_base_revision_fails_closed(synthetic, capsys):
    assert gate.main(["--repo", str(synthetic), "--base", "layer-check-missing-revision"]) == 1
    assert "layer-structure ratchet failed:" in capsys.readouterr().err


def git_snapshot(repo, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
             "GIT_AUTHOR_NAME": "Layer test", "GIT_AUTHOR_EMAIL": "layer@example.invalid",
             "GIT_COMMITTER_NAME": "Layer test", "GIT_COMMITTER_EMAIL": "layer@example.invalid"},
    )
    return result.stdout.strip()


def test_initial_adoption_and_head_environment_use_git_sources(synthetic, monkeypatch, capsys):
    # Throwaway repositories under tmp_path are explicitly allowed by the leader.
    git_snapshot(synthetic, "init", "-q")
    (synthetic / gate.BASELINE).unlink()
    git_snapshot(synthetic, "add", ".")
    git_snapshot(synthetic, "commit", "-qm", "before adoption")
    base = git_snapshot(synthetic, "rev-parse", "HEAD")
    write(synthetic, "rfx/model/__init__.py", "__all__ = ['public']\ndef public():\n    pass\n")
    write(synthetic, SIM, "from rfx.model import public\n")
    write(synthetic, gate.BASELINE, json.dumps(empty_baseline()))
    git_snapshot(synthetic, "add", ".")
    git_snapshot(synthetic, "commit", "-qm", "adopt")
    head = git_snapshot(synthetic, "rev-parse", "HEAD")
    # Source, import target, and baseline all differ on disk: use Git for all.
    write(synthetic, "rfx/model/__init__.py", "def public():\n    pass\n")
    write(synthetic, SIM, "def new():\n    np.zeros(1)\n")
    write(synthetic, gate.BASELINE, "broken working-tree baseline")
    monkeypatch.setenv("BASE_SHA", base)
    monkeypatch.setenv("HEAD_SHA", head)
    assert gate.main(["--repo", str(synthetic)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "rfx.model=1" in output.out
    assert gate.main(["--repo", str(synthetic), "--head", "missing-head"]) == 1



@pytest.mark.parametrize("imports,read,expected", [
    ("from rfx.model import conductors as c", "c._Private", ["rfx.model.conductors:_Private"]),
    ("from .model import conductors as c", "c.hidden()", ["rfx.model.conductors:hidden"]),
    ("import rfx.model.conductors", "rfx.model.conductors.hidden()", ["rfx.model.conductors:hidden"]),
    ("import rfx.model.conductors as c", "c.exported()", []),
    ("from rfx.model import conductors as c", "c.Record; c.CONSTANT", []),
    ("from rfx.model.conductors import exported", "exported()", ["rfx.model.conductors:exported"]),
    ("from rfx.model import public", "public()", []),
    ("from rfx.model import hidden", "hidden()", ["rfx.model:hidden"]),
    ("from rfx.core.yee import update_h", "update_h()", ["rfx.core.yee:update_h"]),
    ("from rfx.core.dft_utils import phase", "phase()", ["rfx.core.dft_utils:phase"]),
    ("from rfx.model.conductors import Record, CONSTANT", "Record; CONSTANT", []),
])
def test_import_forms(synthetic, imports, read, expected):
    write(synthetic, "rfx/model/__init__.py",
          "__all__ = ['public', 'exported']\ndef public(): pass\ndef hidden(): pass\n")
    write(synthetic, "rfx/model/conductors.py",
          "class Record: pass\nclass _Private: pass\nCONSTANT = 1\n"
          "def hidden(): pass\ndef exported(): pass\n")
    write(synthetic, "rfx/core/yee.py", "def update_h(): pass\n")
    write(synthetic, "rfx/core/dft_utils.py", "def phase(): pass\n")
    write(synthetic, SIM, f"{imports}\ndef use():\n    {read}\n")
    assert gate.count_file(gate.Sources(synthetic), SIM)["I"] == expected


def test_module_reads_in_annotations_and_alias_scopes(synthetic):
    write(synthetic, "rfx/model/conductors.py", "class _Private: pass\ndef hidden(): pass\n")
    write(synthetic, SIM,
          "from rfx.model import conductors as c\n"
          "def annotated() -> c._Private: pass\n"
          "def shadowed(c): return c.hidden()\n"
          "def local():\n    from rfx.model import conductors as other\n    return other.hidden()\n"
          "def unrelated(): return other._unrelated\n")
    assert gate.count_file(gate.Sources(synthetic), SIM)["I"] == [
        "rfx.model.conductors:_Private", "rfx.model.conductors:hidden",
    ]


@pytest.mark.parametrize("operator", ["+", "-", "*", "/", "//", "%", "@", "**"])
def test_allowlist_arithmetic_keywords(synthetic, operator):
    write(synthetic, SIM, f"def use():\n    np.asarray(a, dtype=resolve(b {operator} c))\n")
    assert gate.count_file(gate.Sources(synthetic), SIM)["G"] == ["use"]


@pytest.mark.parametrize("call,rule", [
    ("rank_shard_map(f)", "S"), ("checkpointed_scan(f)", "S"),
    ("lorentz_e_component(x)", "S"), ("debye_p_component(x)", "S"),
    ("exchange_fields(x)", "S"), ("inject_source(x)", "S"),
    ("make_uniform_step(x)", "S"), ("state._replace(ex=e)", "S"),
    ("np.exp(x=-1j * angle)", "M"), ("jnp.fft.rfft(x)", "M"),
    ("a.at[i].multiply(b)", "A"), ("jnp.zeros(n).at[i].set(b)", "A"),
])
def test_named_syntax_in_nested_functions_belongs_to_outer_unit(synthetic, call, rule):
    write(synthetic, SIM, f"def outer():\n    def nested():\n        {call}\n        {call}\n")
    assert gate.count_file(gate.Sources(synthetic), SIM)[rule] == ["outer"]


def test_measurement_names_excluded_from_step(synthetic):
    write(synthetic, SIM,
          "def use():\n    update_waveguide_port_probe(x)\n    update_overlap_dft(x)\n"
          "    unrelated_component(x)\n    state._replace(other=x)\n")
    assert gate.count_file(gate.Sources(synthetic), SIM)["S"] == []


def test_baseline_matches_real_tree_and_gate_never_imports_rfx(generated):
    assert gate.count(gate.Sources(REPO))["baseline"] == generated
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c",
         "import runpy, sys; "
         f"runpy.run_path({str(REPO / 'scripts/ci/check_layer_structure.py')!r}); "
         "assert not any(n == 'rfx' or n.startswith(('rfx.', 'jax')) for n in sys.modules)"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_list_allowed_includes_arithmetic_for_human_review(synthetic, capsys):
    write(synthetic, SIM, "def use():\n    np.asarray(a / b)\n    np.any(a > b)\n")
    assert gate.main(["--repo", str(synthetic), "--write-baseline", "--list-allowed"]) == 0
    output = capsys.readouterr().out
    assert f"{SIM}:2: np.asarray(a / b)" in output
    assert f"{SIM}:3: np.any(a > b)" in output
    assert gate.baseline((synthetic / gate.BASELINE).read_bytes())["files"][SIM]["G"] == ["use"]


def test_wiring_matches_file_size_gate():
    workflow = yaml.safe_load((REPO / ".github/workflows/pr-tests.yml").read_text())
    steps = workflow["jobs"]["guards-and-preflight"]["steps"]
    calls = [i for i, step in enumerate(steps)
             if "scripts/ci/check_layer_structure.py" in step.get("run", "")]
    assert len(calls) == 1
    index = calls[0]
    step = steps[index]
    assert steps[index - 1]["name"] == "File-size ratchet"
    assert "pip install" in steps[index + 1]["run"]
    assert step["name"] == "Layer-structure ratchet"
    assert step["run"] == "python scripts/ci/check_layer_structure.py"
    assert not step.get("if") and not step.get("continue-on-error")
    assert step["env"] == steps[index - 1]["env"]
    assert set(step["env"]) == {"BASE_SHA"}
    local = (REPO / "scripts/ci/local.sh").read_text()
    step_four = local.split("begin 4\n", 1)[1].split("begin 5\n", 1)[0]
    assert 'BASE_SHA="$CHANGELOG_BASE" \\\n  "$PYTHON" scripts/ci/check_file_size_ratchet.py || fail\nBASE_SHA="$CHANGELOG_BASE" \\\n  "$PYTHON" scripts/ci/check_layer_structure.py || fail' in step_four
    assert "HEAD_SHA=" not in step_four
    tree = ast.parse((REPO / "tests/contracts/test_ci_workflows_contract.py").read_text())
    gates = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "BOTH_BRANCH_GATES" for t in n.targets))
    assert "scripts/ci/check_layer_structure.py" in gates["guards-and-preflight"]



@pytest.mark.parametrize("target,statement,name", [
    (CLEAN, "from rfx.boundaries import apply_cpml_e", "rfx.boundaries:apply_cpml_e"),
    (CLEAN, "from rfx.model.conductors import _same_grid", "rfx.model.conductors:_same_grid"),
    ("<module>", "from rfx.model.conductors import *", "rfx.model.conductors:*"),
])
def test_review_import_mutations_on_real_source(tmp_path, generated, target, statement, name):
    assert name not in generated["files"][SIM]["I"]
    if target == "<module>":
        # Star imports are only legal at module level.
        write(tmp_path, SIM, (REPO / SIM).read_text() + "\n" + statement + "\n")
        observed = gate.count_file(CopiedSources(tmp_path), SIM)
    else:
        observed = mutate(tmp_path, SIM, target, statement)
    assert_new_hit(generated, SIM, ["I"], name, observed)


@pytest.mark.parametrize("target,rules,code", [
    ("LayerReviewOuter.Inner.advance", ["S"],
     "class LayerReviewOuter:\n    class Inner:\n        def advance(self):\n            jax.lax.scan(f, c, xs)\n"),
    ("layer_review_conditional", ["S"],
     "if True:\n    def layer_review_conditional():\n        jax.lax.scan(f, c, xs)\n"),
    ("<module>", ["S"], "_X = jax.lax.scan(f, c, xs)\n"),
    ("LayerReviewLambda.<body>", ["M", "G"],
     "class LayerReviewLambda:\n    phase = lambda x: jnp.exp(1j * x)\n"),
])
def test_review_code_positions_on_real_source(tmp_path, generated, target, rules, code):
    for rule in rules:
        assert target not in generated["files"][SIM][rule]
    write(tmp_path, SIM, (REPO / SIM).read_text() + "\n" + code)
    observed = gate.count_file(CopiedSources(tmp_path), SIM)
    assert_new_hit(generated, SIM, rules, target, observed)


def test_additional_array_updates_on_real_source(tmp_path, generated):
    target = CLEAN
    assert target not in generated["files"][SIM]["A"]
    observed = mutate(tmp_path, SIM, target,
                      "a.at[i].divide(b)\na.at[i].power(b)\na.at[i].apply(f)")
    assert_new_hit(generated, SIM, ["A"], target, observed)
    # Each new spelling must independently trip A, not only their combination.
    for method in ("divide", "power", "apply"):
        assert "A" in gate.unit_hits(ast.parse(f"a.at[i].{method}(b)"))


@pytest.mark.parametrize("name", ["_Private", "_CONSTANT", "_function", "*"])
def test_private_names_and_stars_ignore_kind_and_exports(synthetic, name):
    write(synthetic, "rfx/model/__init__.py",
          "__all__ = ['_Private', '_CONSTANT', '_function']\n"
          "class _Private: pass\n_CONSTANT = 1\ndef _function(): pass\n")
    write(synthetic, SIM, f"from rfx.model import {name}\n")
    assert gate.count_file(gate.Sources(synthetic), SIM)["I"] == [f"rfx.model:{name}"]


def test_reexport_kinds_aliases_hops_and_cycles(synthetic):
    write(synthetic, "rfx/model/definitions.py",
          "def f(): pass\nclass Record: pass\nCONSTANT = 1\n")
    write(synthetic, "rfx/model/one.py",
          "from .definitions import f as alias, Record, CONSTANT\n"
          "import rfx.model.definitions as module\n")
    write(synthetic, "rfx/model/two.py", "from .one import alias as f\n")
    write(synthetic, "rfx/model/three.py", "from .two import f\n")
    write(synthetic, "rfx/model/four.py", "from .three import f\n")
    write(synthetic, "rfx/model/cycle_a.py", "from .cycle_b import f\n")
    write(synthetic, "rfx/model/cycle_b.py", "from .cycle_a import f\n")
    sources = gate.Sources(synthetic)
    assert sources.is_function("rfx.model.one", "alias")
    assert sources.is_function("rfx.model.one.module", "f")
    assert sources.is_function("rfx.model.three", "f")
    assert not sources.is_function("rfx.model.four", "f")
    assert not sources.is_function("rfx.model.one", "Record")
    assert not sources.is_function("rfx.model.one", "CONSTANT")
    assert not sources.is_function("rfx.model.one", "module")
    assert not sources.is_function("rfx.model.cycle_a", "f")


def test_units_in_control_blocks_and_pseudo_units_are_disjoint(synthetic):
    write(synthetic, SIM, """
if condition():
    async def conditional():
        jax.lax.scan(f, c, xs)
try:
    class Outer:
        with context():
            class Inner:
                try:
                    async def method(self):
                        def nested():
                            jax.lax.scan(f, c, xs)
                except Exception:
                    pass
        value = lambda x: jnp.exp(1j * x)
except Exception:
    pass
with context():
    def under_with():
        jax.lax.scan(f, c, xs)
_MODULE_PHASE = jnp.exp(1j * angle)
""")
    rules = gate.count_file(gate.Sources(synthetic), SIM)
    assert rules["S"] == ["Outer.Inner.method", "conditional", "under_with"]
    assert rules["M"] == ["<module>", "Outer.<body>"]
    assert rules["G"] == ["<module>", "Outer.<body>"]
    # The nested function belongs to its method, and no method body leaks into
    # a parent class or the module pseudo-unit.


def test_review_pseudo_unit_calibration(generated):
    # nonuniform.py lost its module-level hit when #1643 moved the grid tables out.
    assert "<module>" not in generated["files"]["rfx/nonuniform.py"]["G"]
    # waveguide_port.py lost both when its definitions moved to the layer
    # packages and the file kept only re-exports.
    waveguide = generated["files"]["rfx/sources/waveguide_port.py"]
    assert "<module>" not in waveguide["G"]
    assert "WaveguidePortConfig.<body>" not in waveguide["G"]


def test_missing_base_baseline_is_not_adoption_after_checker_exists(synthetic, monkeypatch, capsys):
    git_snapshot(synthetic, "init", "-q")
    (synthetic / gate.BASELINE).unlink()
    write(synthetic, gate.CHECKER, "# checker already adopted\n")
    git_snapshot(synthetic, "add", ".")
    git_snapshot(synthetic, "commit", "-qm", "checker without baseline")
    base = git_snapshot(synthetic, "rev-parse", "HEAD")
    write(synthetic, gate.BASELINE, json.dumps(empty_baseline()))
    git_snapshot(synthetic, "add", ".")
    git_snapshot(synthetic, "commit", "-qm", "head baseline")
    head = git_snapshot(synthetic, "rev-parse", "HEAD")
    monkeypatch.setenv("BASE_SHA", base)
    monkeypatch.setenv("HEAD_SHA", head)
    assert gate.main(["--repo", str(synthetic)]) == 1
    assert f"BASE contains {gate.CHECKER} but is missing {gate.BASELINE}" in capsys.readouterr().err


@pytest.mark.parametrize("case,expected,message", [
    ("new", 1, "new violation against BASE new"),
    ("grow", 1, "baseline grew against BASE new"),
    ("grow-stale", 1, "baseline grew against BASE new"),
    ("stale", 1, "stale baseline entry old"),
    ("shrink", 0, ""),
])
def test_outside_named_baseline_ratchet(synthetic, monkeypatch, capsys, case, expected, message):
    path = "rfx/runners/example.py"
    previous, current = empty_baseline(), empty_baseline()
    previous["outside"][path] = {"S": ["old"], "M": []}
    if case in {"new", "grow"}:
        write(synthetic, path, "def new():\n    jax.lax.scan(f, c, xs)\n")
    if case in {"grow", "grow-stale", "stale"}:
        current["outside"][path] = {"S": ["old" if case == "stale" else "new"], "M": []}
    write(synthetic, gate.BASELINE, json.dumps(current))
    mock_base(monkeypatch, previous)
    assert gate.main(["--repo", str(synthetic), "--base", "base"]) == expected
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("outside", [
    {"S": 77, "M": 44}, [],
    {"rfx/model/x.py": {"S": ["a"], "M": []}},
    {SIM: {"S": ["a"], "M": []}},
    {"rfx/../x.py": {"S": ["a"], "M": []}},
    {"rfx/x.py": {"S": 1, "M": []}},
    {"rfx/x.py": {"S": ["b", "a"], "M": []}},
    {"rfx/x.py": {"S": ["a", "a"], "M": []}},
    {"rfx/x.py": {"S": [""], "M": []}},
    {"rfx/x.py": {"S": ["a"]}},
    {"rfx/x.py": {"S": [], "M": [], "G": []}},
])
def test_outside_malformed_baseline_fails_closed(synthetic, monkeypatch, capsys, outside):
    invalid = empty_baseline()
    invalid["outside"] = outside
    write(synthetic, gate.BASELINE, json.dumps(invalid))
    assert gate.main(["--repo", str(synthetic)]) == 1
    assert "layer-structure ratchet failed:" in capsys.readouterr().err
    write(synthetic, gate.BASELINE, json.dumps(empty_baseline()))
    mock_base(monkeypatch, invalid)
    assert gate.main(["--repo", str(synthetic), "--base", "base"]) == 1
    assert "layer-structure ratchet failed:" in capsys.readouterr().err


def test_outside_pseudo_units_and_only_s_m_rules(synthetic):
    write(synthetic, "rfx/outside.py", """
from rfx.model import _private
scan(f, c, xs)
class Phase:
    value = np.exp(1j * angle)
    def arithmetic(self):
        return np.zeros(2).at[0].set(1)
""")
    actual = gate.count(gate.Sources(synthetic))["baseline"]
    assert actual["outside"] == {
        "rfx/outside.py": {"S": ["<module>"], "M": ["Phase.<body>"]},
    }


def test_outside_report_blocks_and_count_order(synthetic, capsys):
    write(synthetic, "rfx/a.py", "def a():\n    np.fft.fft(x)\n")
    write(synthetic, "rfx/z.py", "def z():\n    scan(f, c, xs)\n    np.fft.fft(x)\n")
    write(synthetic, "rfx/b.py", "def b():\n    scan(f, c, xs)\n")
    gate.report(gate.count(gate.Sources(synthetic)))
    output = capsys.readouterr().out
    five, outside = output.split("\noutside the layer packages — may only fall, not a 2.0 condition "
                                 "(PI to decide how far 2.0 reaches)\n")
    assert five.startswith("2.0 condition — five files\n")
    outside, reverse = outside.split("\nreverse imports from layers to callers — may only fall\n")
    assert reverse == "reverse totals: R=0\n"
    assert outside.splitlines() == [
        "outside rfx/z.py: S=1 M=1", "outside rfx/a.py: S=0 M=1",
        "outside rfx/b.py: S=1 M=0", "outside totals: S=2 M=2",
    ]


def test_outside_head_and_base_use_named_git_snapshots(synthetic, monkeypatch, capsys):
    path = "rfx/runners/example.py"
    write(synthetic, path, "def old():\n    scan(f, c, xs)\n")
    previous = empty_baseline()
    previous["outside"][path] = {"S": ["old"], "M": []}
    write(synthetic, gate.BASELINE, json.dumps(previous))
    git_snapshot(synthetic, "init", "-q")
    git_snapshot(synthetic, "add", ".")
    git_snapshot(synthetic, "commit", "-qm", "named base")
    base = git_snapshot(synthetic, "rev-parse", "HEAD")
    write(synthetic, path, "def new():\n    scan(f, c, xs, 2)\n")
    current = empty_baseline()
    current["outside"][path] = {"S": ["new"], "M": []}
    write(synthetic, gate.BASELINE, json.dumps(current))
    git_snapshot(synthetic, "add", ".")
    git_snapshot(synthetic, "commit", "-qm", "rename outside")
    head = git_snapshot(synthetic, "rev-parse", "HEAD")
    write(synthetic, path, "")
    write(synthetic, gate.BASELINE, json.dumps(empty_baseline()))
    monkeypatch.setenv("BASE_SHA", base)
    monkeypatch.setenv("HEAD_SHA", head)
    assert gate.main(["--repo", str(synthetic)]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{path} S: new violation against BASE new",
        f"{path} S: baseline grew against BASE new",
    ]


def rename_base(repo, files):
    for path, source in files.items():
        write(repo, path, source)
    assert gate.check(repo, write_baseline=True) == 0
    git_snapshot(repo, "init", "-q")
    git_snapshot(repo, "add", ".")
    git_snapshot(repo, "commit", "-qm", "rename base")
    return git_snapshot(repo, "rev-parse", "HEAD")


def test_outside_pure_rename_with_helper_and_docstrings(synthetic, capsys):
    path = "rfx/example.py"
    base = rename_base(synthetic, {path:
        'def helper(x):\n    "old helper"\n    return x + 1\n'
        'def old(x):\n    "old doc"\n    return np.exp(1j * helper(x))\n'})
    write(synthetic, path,
          'def renamed_helper(y):\n    "new helper"\n    return y + 1\n'
          'def new(y):\n    "new doc"\n    return np.exp(1j * renamed_helper(y))\n')
    capsys.readouterr()
    # A rename still needs baseline regeneration.
    assert gate.check(synthetic, base) == 1
    output = capsys.readouterr()
    assert output.err.splitlines() == [f"{path} M: unlisted violation new",
                                       f"{path} M: stale baseline entry old"]
    assert gate.check(synthetic, base, write_baseline=True) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert f"renamed: {path} M old -> new" in output.out
    assert gate.check(synthetic) == 0
    assert "renamed:" not in capsys.readouterr().out


@pytest.mark.parametrize("change", ["attribute", "constant", "statement", "copy", "third-party"])
def test_rename_with_real_change_is_rejected(synthetic, capsys, change):
    path = "rfx/example.py"
    old = "from numpy.fft import fft\ndef old(x):\n    return np.exp(1j * fft(jnp.fft(x)))\n"
    base = rename_base(synthetic, {path: old})
    new = old.replace("def old", "def new")
    if change == "attribute":
        new = new.replace("jnp.fft", "jnp.ifft")
    elif change == "constant":
        new = new.replace("1j", "2j")
    elif change == "statement":
        new = new.replace("    return", "    extra = 1\n    return")
    elif change == "copy":
        new += old.split("\n", 1)[1]
    else:
        new = new.replace("import fft", "import ifft").replace("* fft(", "* ifft(")
    write(synthetic, path, new)
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    output = capsys.readouterr()
    assert output.err.splitlines() == [f"{path} M: new violation against BASE new",
                                       f"{path} M: baseline grew against BASE new"]
    assert "renamed:" not in output.out


def test_two_identical_units_have_ambiguous_renames(synthetic, capsys):
    path = "rfx/example.py"
    source = "def {}(x):\n    return np.exp(1j * x)\n"
    base = rename_base(synthetic, {path: source.format("old_a") + source.format("old_b")})
    write(synthetic, path, source.format("new_a") + source.format("new_b"))
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    output = capsys.readouterr()
    assert output.err.splitlines() == [
        f"{path} M: new violation against BASE new_a",
        f"{path} M: new violation against BASE new_b",
        f"{path} M: baseline grew against BASE new_a",
        f"{path} M: baseline grew against BASE new_b",
    ]
    assert "renamed:" not in output.out


@pytest.mark.parametrize("expression", ["y + z", "z + y"])
def test_local_identifier_swap_requires_bijection(synthetic, capsys, expression):
    path = "rfx/example.py"
    base = rename_base(synthetic, {path: "def old(x, w):\n    return np.exp(1j * (x + x))\n"})
    write(synthetic, path, f"def new(y, z):\n    return np.exp(1j * ({expression}))\n")
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    assert f"{path} M: new violation against BASE new" in capsys.readouterr().err


@pytest.mark.parametrize("keep", ["alias", "wrapper"])
@pytest.mark.parametrize("changed_helper", [False, True])
def test_rename_keeps_old_helper_binding(synthetic, capsys, keep, changed_helper):
    path = "rfx/example.py"
    base = rename_base(synthetic, {path:
        "def helper(x):\n    return x + 1\n"
        "def old(x):\n    return np.exp(1j * helper(x))\n"})
    retained = "helper = renamed_helper\n" if keep == "alias" else (
        "def helper(x):\n    return renamed_helper(x)\n")
    write(synthetic, path,
          f"def renamed_helper(x):\n    return x + {2 if changed_helper else 1}\n"
          + retained + "def new(x):\n    return np.exp(1j * renamed_helper(x))\n")
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == int(changed_helper)
    output = capsys.readouterr()
    if changed_helper:
        assert f"{path} M: new violation against BASE new" in output.err
    else:
        assert output.err == ""
        assert f"renamed: {path} M old -> new" in output.out


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("changed_helper", [False, True])
def test_rfx_import_rename_checks_target_definitions(synthetic, capsys, relative, changed_helper):
    path, target = "rfx/example.py", "rfx/helpers.py"
    module = ".helpers" if relative else "rfx.helpers"
    base = rename_base(synthetic, {
        path: f"from {module} import helper\ndef old(x):\n    return np.exp(1j * helper(x))\n",
        target: "def helper(x):\n    return x + 1\n",
    })
    write(synthetic, path,
          f"from {module} import renamed_helper\ndef new(x):\n    return np.exp(1j * renamed_helper(x))\n")
    write(synthetic, target,
          f"def renamed_helper(x):\n    return x + {2 if changed_helper else 1}\nhelper = renamed_helper\n")
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == int(changed_helper)
    output = capsys.readouterr()
    assert (f"renamed: {path} M old -> new" in output.out) is not changed_helper


def test_move_to_another_file_is_not_a_rename(synthetic, capsys):
    source = "def same(x):\n    return np.exp(1j * x)\n"
    base = rename_base(synthetic, {"rfx/old_file.py": source})
    (synthetic / "rfx/old_file.py").unlink()
    write(synthetic, "rfx/new_file.py", source)
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    output = capsys.readouterr()
    assert "rfx/new_file.py M: new violation against BASE same" in output.err
    assert "renamed:" not in output.out


def test_five_file_g_rename_passes(synthetic, capsys):
    base = rename_base(synthetic, {SIM: "def old(x):\n    return np.zeros(x)\n"})
    write(synthetic, SIM, "def new(y):\n    return np.zeros(y)\n")
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert f"renamed: {SIM} G old -> new" in output.out


def test_method_own_name_does_not_allow_numpy_import_swap(synthetic, capsys):
    path = "rfx/example.py"
    source = ("from numpy.fft import {name}\nclass C:\n"
              "    def {name}(self, x):\n        return np.exp(1j * {name}(x))\n")
    base = rename_base(synthetic, {path: source.format(name="fft")})
    write(synthetic, path, source.format(name="ifft"))
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    output = capsys.readouterr()
    assert output.err.splitlines() == [f"{path} M: new violation against BASE C.ifft",
                                       f"{path} M: baseline grew against BASE C.ifft"]
    assert "renamed:" not in output.out


@pytest.mark.parametrize("binding", [
    "[fft for fft in ()]", "lambda fft: fft", "def inner(fft):\n        return fft",
])
def test_nested_local_binding_does_not_allow_module_import_swap(synthetic, capsys, binding):
    path = "rfx/example.py"
    source = (f"from numpy.fft import fft\ndef old(x):\n    {binding}\n"
              "    return np.exp(1j * fft(x))\n")
    base = rename_base(synthetic, {path: source})
    body = source.split("\n", 1)[1].replace("def old", "def new").replace("fft", "ifft")
    write(synthetic, path, "from numpy.fft import ifft\n" + body)
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    output = capsys.readouterr()
    assert f"{path} M: new violation against BASE new" in output.err
    assert "renamed:" not in output.out


@pytest.mark.parametrize("declaration", ["global", "nonlocal"])
def test_scope_declarations_are_never_local_renames(synthetic, capsys, declaration):
    path = "rfx/example.py"
    # The nonlocal declaration is inside a nested function of the counted unit.
    body = ("    global X\n    X = 3\n" if declaration == "global" else
            "    X = 3\n    def inner():\n        nonlocal X\n        X = 3\n")
    source = f"def old():\n{body}    return np.exp(1j * X)\n"
    base = rename_base(synthetic, {path: source})
    write(synthetic, path, source.replace("def old", "def new").replace("X", "Y"))
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    output = capsys.readouterr()
    assert f"{path} M: new violation against BASE new" in output.err
    assert "renamed:" not in output.out


@pytest.mark.parametrize("old,new", [("len", "size"), ("size", "len")])
def test_builtin_names_are_not_local_renames(synthetic, capsys, old, new):
    path = "rfx/example.py"
    source = "def {unit}(x):\n    lambda {name}: {name}\n    return np.exp(1j * {name}(x))\n"
    base = rename_base(synthetic, {path: source.format(unit="old", name=old)})
    write(synthetic, path, source.format(unit="new", name=new))
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    assert f"{path} M: new violation against BASE new" in capsys.readouterr().err


def test_plain_local_for_target_rename_passes(synthetic, capsys):
    path = "rfx/example.py"
    source = "def old(xs):\n    for v in xs:\n        np.exp(1j * v)\n"
    base = rename_base(synthetic, {path: source})
    write(synthetic, path, source.replace("def old", "def new").replace("v", "w"))
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert f"renamed: {path} M old -> new" in output.out


def test_top_level_recursive_rename_passes(synthetic, capsys):
    path = "rfx/example.py"
    source = "def old(x):\n    return np.exp(1j * old(x))\n"
    base = rename_base(synthetic, {path: source})
    write(synthetic, path, source.replace("old", "new"))
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert f"renamed: {path} M old -> new" in output.out


def test_existing_clean_unit_cannot_receive_counted_body(synthetic, capsys):
    path = "rfx/example.py"
    source = "def old(x):\n    return np.exp(1j * x)\n"
    base = rename_base(synthetic, {path: source + "def new(x):\n    return x\n"})
    write(synthetic, path, source.replace("def old", "def new"))
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == 1
    output = capsys.readouterr()
    assert f"{path} M: new violation against BASE new" in output.err
    assert "renamed:" not in output.out


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("body,name", [
    ("    def f(self, x):\n        return np.exp(1j * x)\n", "f"),
    ("    value = np.exp(1j * x)\n", "<body>"),
])
def test_class_prefix_requires_new_module_class(synthetic, capsys, existing, body, name):
    path = "rfx/example.py"
    base = rename_base(synthetic, {path: "class A:\n" + body
                                  + ("class B:\n    pass\n" if existing else "")})
    write(synthetic, path, "class B:\n" + body)
    capsys.readouterr()
    assert gate.check(synthetic, base, write_baseline=True) == int(existing)
    output = capsys.readouterr()
    if existing:
        assert f"{path} M: new violation against BASE B.{name}" in output.err
        assert "renamed:" not in output.out
    else:
        assert output.err == ""
        assert f"renamed: {path} M A.{name} -> B.{name}" in output.out


@pytest.mark.parametrize("path,code,entry", [
    ("rfx/stepping/__init__.py", "def local():\n    from rfx.simulation import X\n", "rfx.simulation:X"),
    ("rfx/stepping/__init__.py", "from ..progress import Y\n", "rfx.progress:Y"),
    ("rfx/stepping/nested/example.py", "from ...progress import Y\n", "rfx.progress:Y"),
    ("rfx/stepping/__init__.py", '_LAZY_EXPORTS = {"Z": "rfx.ringdown"}\n', "rfx.ringdown:Z"),
    ("rfx/stepping/__init__.py", "def local():\n    from rfx import simulation\n", "rfx.simulation"),
    ("rfx/stepping/__init__.py", "from .. import progress\n", "rfx.progress"),
], ids=["local", "relative", "nested-relative", "lazy", "package-attribute", "relative-package-attribute"])
def test_reverse_mutations_red_then_restored_green(tmp_path, generated, path, code, entry):
    assert path not in generated["reverse"]
    original = (REPO / path).read_text() if (REPO / path).exists() else ""
    write(tmp_path, path, original + "\n" + code)
    sources = CopiedSources(tmp_path)
    assert gate.count_file(sources, path) == {"R": [entry]}
    actual = gate.count(sources)["baseline"]
    expected = deepcopy(generated)
    expected["reverse"][path] = {"R": [entry]}
    assert actual == expected
    assert gate.violations(actual, generated) == [f"{path} R: unlisted violation {entry}"]
    write(tmp_path, path, original)
    restored = gate.count(CopiedSources(tmp_path))["baseline"]
    assert restored == generated
    assert gate.violations(restored, generated) == []


@pytest.mark.parametrize("code", [
    "from rfx.core.yee import update_h\n",
    "from rfx.api._spec import Spec\n",
    "import rfx.sources.waveguide_port_x\n",
    '_LAZY_EXPORTS = {"Spec": "rfx.api._spec"}\n',
    'import importlib\nimportlib.import_module("rfx.simulation")\n',
])
def test_reverse_exact_target_controls(tmp_path, generated, code):
    path = "rfx/stepping/__init__.py"
    write(tmp_path, path, (REPO / path).read_text() + "\n" + code)
    assert gate.count_file(CopiedSources(tmp_path), path) == {"R": []}
    assert gate.count(CopiedSources(tmp_path))["baseline"] == generated


@pytest.mark.parametrize("package", gate.LAYERS)
def test_reverse_all_import_forms_scopes_and_dedup(synthetic, package):
    path = package.replace(".", "/") + "/example.py"
    write(synthetic, path, '''
import rfx.simulation, rfx.progress as progress
from rfx.api._execute import a, b as alias
if TYPE_CHECKING:
    from rfx.nonuniform import c
class Nested:
    def local(self):
        from rfx.api import d
        from rfx.adjoint import e
        from rfx.sources.waveguide_port import f
        import rfx.simulation as sim
        from rfx.api._execute import a
_LAZY_EXPORTS: dict = {"tail": "rfx.ringdown", "ignored": dynamic}
''')
    entries = ["rfx.adjoint:e", "rfx.api._execute:a", "rfx.api._execute:b", "rfx.api:d",
               "rfx.nonuniform:c", "rfx.progress", "rfx.ringdown:tail", "rfx.simulation",
               "rfx.sources.waveguide_port:f"]
    assert gate.count_file(gate.Sources(synthetic), path) == {"R": entries}
    assert gate.count(gate.Sources(synthetic))["baseline"]["reverse"] == {path: {"R": entries}}


@pytest.mark.parametrize("case", ["adoption", "new", "grow", "stale", "grow-stale", "shrink"])
def test_reverse_base_adoption_and_ratchet(synthetic, monkeypatch, capsys, case):
    path = "rfx/stepping/example.py"
    previous, current = empty_baseline(), empty_baseline()
    if case == "adoption":
        del previous["reverse"]
    else:
        previous["reverse"][path] = {"R": ["rfx.simulation:old"]}
    if case in {"adoption", "new", "grow"}:
        write(synthetic, path, "from rfx.simulation import new\n")
    if case in {"adoption", "grow", "grow-stale", "stale"}:
        current["reverse"][path] = {"R": ["rfx.simulation:old" if case == "stale" else "rfx.simulation:new"]}
    write(synthetic, gate.BASELINE, json.dumps(current))
    mock_base(monkeypatch, previous)

    def no_renames(*args):
        pytest.fail("reverse imports must not use RenameMatcher")

    # Empty non-reverse groups also call pairs, so exempt those paths.
    monkeypatch.setattr(gate.RenameMatcher, "pairs",
                        lambda self, p, gone, new: no_renames() if p == path else set())
    assert gate.main(["--repo", str(synthetic), "--base", "base"]) == int(case not in {"adoption", "shrink"})
    errors = capsys.readouterr().err.splitlines()
    expected = {
        "adoption": [], "shrink": [],
        "new": [f"{path} R: unlisted violation rfx.simulation:new",
                f"{path} R: new violation against BASE rfx.simulation:new"],
        "grow": [f"{path} R: new violation against BASE rfx.simulation:new",
                 f"{path} R: baseline grew against BASE rfx.simulation:new"],
        "stale": [f"{path} R: stale baseline entry rfx.simulation:old"],
        "grow-stale": [f"{path} R: stale baseline entry rfx.simulation:new",
                       f"{path} R: baseline grew against BASE rfx.simulation:new"],
    }
    assert errors == expected[case]
    # Reverting each rejected source/baseline pair is green against the same BASE.
    if errors:
        write(synthetic, path, "from rfx.simulation import old\n")
        write(synthetic, gate.BASELINE, json.dumps(previous))
        assert gate.main(["--repo", str(synthetic), "--base", "base"]) == 0
        assert capsys.readouterr().err == ""


@pytest.mark.parametrize("reverse", [
    [], {"rfx/simulation.py": {"R": ["x"]}}, {"rfx/stepping_x/a.py": {"R": ["x"]}},
    {"rfx/stepping/../a.py": {"R": ["x"]}}, {"rfx/stepping//a.py": {"R": ["x"]}},
    {"rfx/stepping/a.txt": {"R": ["x"]}}, {"rfx/stepping/a.py": {"S": []}},
    {"rfx/stepping/a.py": {"R": [], "S": []}}, {"rfx/stepping/a.py": {"R": "x"}},
    {"rfx/stepping/a.py": {"R": ["b", "a"]}}, {"rfx/stepping/a.py": {"R": ["x", "x"]}},
    {"rfx/stepping/a.py": {"R": [""]}}, {"rfx/stepping/a.py": {"R": [1]}},
])
def test_reverse_malformed_baseline_red_then_restored_green(synthetic, monkeypatch, capsys, reverse):
    invalid = empty_baseline()
    invalid["reverse"] = reverse
    write(synthetic, gate.BASELINE, json.dumps(invalid))
    assert gate.main(["--repo", str(synthetic)]) == 1
    assert "layer-structure ratchet failed:" in capsys.readouterr().err
    write(synthetic, gate.BASELINE, json.dumps(empty_baseline()))
    assert gate.main(["--repo", str(synthetic)]) == 0
    assert capsys.readouterr().err == ""
    mock_base(monkeypatch, invalid)
    assert gate.main(["--repo", str(synthetic), "--base", "base"]) == 1
    assert "layer-structure ratchet failed:" in capsys.readouterr().err
    mock_base(monkeypatch, empty_baseline())
    assert gate.main(["--repo", str(synthetic), "--base", "base"]) == 0
    assert capsys.readouterr().err == ""


def test_reverse_required_at_head_even_when_empty(synthetic, capsys):
    legacy = empty_baseline()
    del legacy["reverse"]
    assert gate.baseline(json.dumps(legacy).encode()) == legacy
    write(synthetic, gate.BASELINE, json.dumps(legacy))
    assert gate.main(["--repo", str(synthetic)]) == 1
    assert capsys.readouterr().err == "HEAD baseline must contain reverse\n"
    write(synthetic, gate.BASELINE, json.dumps(empty_baseline()))
    assert gate.main(["--repo", str(synthetic)]) == 0
    assert capsys.readouterr().err == ""


def test_reverse_report_order_and_totals(synthetic, capsys):
    for name, code in [("a", "import rfx.simulation"), ("b", "import rfx.progress"),
                       ("z", "from rfx.simulation import a, b")]:
        write(synthetic, f"rfx/stepping/{name}.py", code + "\n")
    gate.report(gate.count(gate.Sources(synthetic)))
    outside, reverse = capsys.readouterr().out.split(
        "\nreverse imports from layers to callers — may only fall\n")
    assert outside.endswith("outside totals: S=0 M=0\n")
    assert reverse.splitlines() == [
        "reverse rfx/stepping/z.py: R=2", "reverse rfx/stepping/a.py: R=1",
        "reverse rfx/stepping/b.py: R=1", "reverse totals: R=4",
    ]

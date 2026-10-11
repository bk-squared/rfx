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


def test_uniform_step_body_and_restored_control(tmp_path, generated):
    target = CLEAN
    assert target not in generated["files"][SIM]["S"]
    body = body_from("rfx/stepping/uniform.py", "make_uniform_step")
    observed = mutate(tmp_path, SIM, target, body)
    assert_new_hit(generated, SIM, ["S"], target, observed)
    # Leader decision: make_*_step calls intentionally match S. The control
    # restores the original unit instead of exempting a step-builder call.
    write(tmp_path, SIM, (REPO / SIM).read_text())
    assert gate.count_file(CopiedSources(tmp_path), SIM) == generated["files"][SIM]


def test_phase_body_and_public_call_control(tmp_path, generated):
    target = CLEAN
    for rule in ("M", "G"):
        assert target not in generated["files"][SIM][rule]
    observed = mutate(tmp_path, SIM, target, body_from("rfx/measurement/dft.py", "_traced_unit"))
    assert_new_hit(generated, SIM, ["M", "G"], target, observed)
    # This call matches no rule, so the declared measurement control remains.
    observed = mutate(tmp_path, SIM, target, "rfx.measurement.phase(step, freqs, dt)")
    assert observed == generated["files"][SIM]


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
    name = "rfx.model.conductors:_PreparedSolve"
    assert name not in generated["files"][SIM]["I"]
    assert name in generated["files"][EXECUTE]["I"]
    observed = mutate(tmp_path, SIM, CLEAN,
                      "from rfx.model import conductors as _c2\n_c2._PreparedSolve")
    assert_new_hit(generated, SIM, ["I"], name, observed)


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


def test_outside_real_runner_mutation(tmp_path, generated):
    path = "rfx/runners/uniform.py"
    target = "layer_check_outside_step"
    assert target not in {name for name, _, _ in gate.units(real_tree(path))}
    write(tmp_path, path, (REPO / path).read_text()
          + "\ndef layer_check_outside_step():\n    return jax.lax.scan(body, carry, xs)\n")
    result = gate.count(CopiedSources(tmp_path))
    assert target in result["outside_files"][path]["S"]
    assert result["baseline"]["outside"] == {
        "S": generated["outside"]["S"] + 1, "M": generated["outside"]["M"],
    }
    assert any("outside S:" in error and "exceeds BASE" in error
               for error in gate.violations(result["baseline"], generated, generated))


def test_calibration_from_declaration_and_leader_correction(generated):
    api = generated["files"][API]
    assert "Simulation.add_thin_conductor" not in api["D1"] + api["D2"]
    assert "Simulation.add_waveguide_port" in api["D1"]
    # This method still computes/checks values before a final return at this commit.
    assert "Simulation.mesh_intelligence_report" in api["D2"]
    node = target_unit(real_tree(API), "Simulation.mesh_intelligence_report")
    assert len(gate.body_without_docstring(node)) > 1
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
    assert {"NonUniformGrid.node_of", "run_nonuniform"} <= all_units
    assert "rfx.model.materials:_design_box_edge_coeffs" in generated["files"][SIM]["I"]


def empty_baseline():
    return {"files": {p: {r: [] for r in gate.RULES} for p in gate.FIVE_FILES},
            "outside": {"S": 0, "M": 0}}


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
    ("outside-up", 1, "outside S: computed 1 exceeds BASE 0"),
    ("outside-down-stale", 1, "outside S: stored 1 differs from computed 0"),
    ("outside-down", 0, ""),
])
def test_ratchet_synthetic_baselines(synthetic, monkeypatch, capsys, case, expected, message):
    previous, current = empty_baseline(), empty_baseline()
    previous["files"][SIM]["G"] = ["old"]
    if case in {"new", "grow"}:
        write(synthetic, SIM, "def new():\n    return np.zeros(1)\n")
        current["files"][SIM]["G"] = ["new"] if case == "grow" else []
    elif case == "stale":
        current["files"][SIM]["G"] = ["old"]
    elif case == "outside-up":
        write(synthetic, "rfx/runners/example.py", "def outside():\n    jax.lax.scan(f, c, xs)\n")
        current["outside"]["S"] = 1
    elif case.startswith("outside-down"):
        previous["outside"]["S"] = 1
        current["outside"]["S"] = int(case.endswith("stale"))
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

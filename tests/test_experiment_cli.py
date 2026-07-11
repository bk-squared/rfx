from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

import pytest

from rfx import cli


FIXTURE = (
    Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_cpu_v1.json"
)
V2_FIXTURE = Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_v2.json"


def test_experiment_parser_exposes_cpu_lifecycle_commands(tmp_path):
    parser = cli.build_parser()

    run = parser.parse_args(
        ["experiment", "run", str(FIXTURE), "--workspace", str(tmp_path)]
    )
    status = parser.parse_args(
        ["experiment", "status", "00000000-0000-0000-0000-000000000001"]
    )

    assert run.func is cli._cmd_experiment_run
    assert run.workspace == str(tmp_path)
    assert status.func is cli._cmd_experiment_status


@pytest.mark.parametrize("fixture", [FIXTURE, V2_FIXTURE])
def test_experiment_validate_cli_prints_digests_and_preflight(capsys, fixture):
    exit_code = cli.main(["experiment", "validate", str(fixture)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["spec_sha256"]) == 64
    assert len(payload["compiled_sha256"]) == 64
    assert payload["preflight"]["ok"] is True


def test_all_extra_self_reference_uses_distribution_name():
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["optional-dependencies"]["all"] == [
        "rfx-fdtd[optimization,visualization,dashboard,agent,studio,dev]"
    ]

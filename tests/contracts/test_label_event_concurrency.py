"""Label events must not cancel required checks in a newer check suite."""

from pathlib import Path

import yaml


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def test_label_events_have_per_run_concurrency_groups() -> None:

    paths = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    checked = []
    for path in paths:
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = workflow.get("on", workflow.get(True))
        if not isinstance(triggers, dict):
            continue
        pull_request = triggers.get("pull_request") or {}
        types = pull_request.get("types", [])
        if "labeled" not in types:
            continue
        checked.append(path.name)
        actions = ["labeled", "unlabeled"]
        if "edited" in types:
            actions.append("edited")
        guard = " || ".join(f"github.event.action == '{action}'" for action in actions)


        group = "${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}"
        group += "${{ (" + guard + ") && format('-{0}', github.run_id) || '' }}"
        assert workflow["concurrency"]["group"] == group, path.name
        assert workflow["concurrency"]["cancel-in-progress"] == (
            "${{ github.event_name == 'pull_request' }}"
        ), path.name
    assert checked, "no workflows with labeled pull-request triggers found"

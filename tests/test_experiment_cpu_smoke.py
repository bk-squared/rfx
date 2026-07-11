from __future__ import annotations

import json
from pathlib import Path

from rfx.experiments import ExperimentService, verify_s11_artifact


FIXTURE = (
    Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_cpu_v1.json"
)


def test_isolated_cpu_worker_produces_immutable_s11(tmp_path):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    service = ExperimentService(tmp_path / "experiment-workspace")

    run = service.run_sync(document)

    assert run.state == "succeeded", run.error
    assert run.artifact_path is not None
    artifact = verify_s11_artifact(run.artifact_path)
    payload = json.loads(artifact.data_json.read_text(encoding="utf-8"))
    assert payload["runtime"]["backend"] == "cpu"
    assert {device["platform"] for device in payload["runtime"]["devices"]} == {"cpu"}
    assert len(payload["points"]) == document["model"]["frequency_sweep"]["points"]
    event_types = [event.event_type for event in service.repository.list_events(run.id)]
    assert event_types.index("preflight_completed") < event_types.index(
        "simulation_started"
    )
    assert event_types[-1] == "run_succeeded"


def test_started_worker_can_be_cancelled_from_reopened_service(tmp_path):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    workspace = tmp_path / "experiment-workspace"
    submitter = ExperimentService(workspace)
    run = submitter.submit(document)
    process = submitter.start(run.id)

    # Model a later CLI invocation: it has no in-memory Popen handle and must
    # verify the durable worker PID before signalling it.
    controller = ExperimentService(workspace)
    controller.cancel(run.id)
    process.wait(timeout=30)

    final = controller.get(run.id)
    assert final.state == "cancelled"
    assert final.cancel_requested is True

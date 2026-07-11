from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from rfx.studio.api import create_app


FIXTURE = Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_v2.json"
HEADERS = {"Accept": "application/json, text/event-stream"}


def _rpc(client, identifier, method, params=None):
    response = client.post(
        "/mcp/",
        headers=HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": identifier,
            "method": method,
            "params": params or {},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "error" not in payload, payload
    return payload["result"]


def _tool(client, identifier, name, arguments):
    return _rpc(
        client,
        identifier,
        "tools/call",
        {"name": name, "arguments": arguments},
    )["structuredContent"]


@pytest.mark.parametrize(
    ("client_name", "actor"),
    [
        ("openai-responses-release-smoke", "agent:openai-release-smoke"),
        ("anthropic-claude-release-smoke", "agent:claude-release-smoke"),
    ],
)
def test_provider_profile_approved_patch_cpu_journey(tmp_path, client_name, actor):
    from fastapi.testclient import TestClient

    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with TestClient(
        create_app(tmp_path / client_name), base_url="http://127.0.0.1:8765"
    ) as client:
        created = client.post("/api/experiments", json=spec).json()
        experiment_id = created["experiment"]["id"]
        first_revision_id = created["revision"]["id"]
        _rpc(
            client,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "1"},
            },
        )

        current = _tool(
            client,
            2,
            "get_experiment",
            {"experiment_id": experiment_id, "actor": actor},
        )
        assert current["current_revision"]["revision_id"] == first_revision_id

        patch_arguments = {
            "experiment_id": experiment_id,
            "base_revision_id": first_revision_id,
            "patch": [
                {
                    "op": "replace",
                    "path": "/metadata/title",
                    "value": f"{client_name} patch",
                }
            ],
            "message": "provider-neutral release smoke",
            "actor": actor,
        }
        proposed = _tool(client, 3, "apply_experiment_patch", patch_arguments)
        assert proposed["status"] == "approval_required"
        patch_approval = proposed["approval"]["id"]
        approved = client.post(
            f"/api/agent/approvals/{patch_approval}/approve",
            json={"actor": "human:release-smoke"},
        )
        assert approved.status_code == 200, approved.text
        patched = _tool(
            client,
            4,
            "apply_experiment_patch",
            {**patch_arguments, "approval_id": patch_approval},
        )
        assert patched["status"] == "ok"
        revision_id = patched["revision_id"]

        validated = _tool(
            client,
            5,
            "validate_experiment",
            {
                "experiment_id": experiment_id,
                "revision_id": revision_id,
                "actor": actor,
            },
        )
        assert validated["preflight"]["ok"] is True

        run_arguments = {
            "experiment_id": experiment_id,
            "revision_id": revision_id,
            "idempotency_key": f"{client_name}-run",
            "actor": actor,
        }
        run_proposal = _tool(client, 6, "start_run", run_arguments)
        assert run_proposal["status"] == "approval_required"
        run_approval = run_proposal["approval"]["id"]
        approved_run = client.post(
            f"/api/agent/approvals/{run_approval}/approve",
            json={"actor": "human:release-smoke"},
        )
        assert approved_run.status_code == 200, approved_run.text
        started = _tool(
            client,
            7,
            "start_run",
            {**run_arguments, "approval_id": run_approval},
        )
        run_id = started["run_id"]

        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            run = _tool(client, 8, "get_run", {"run_id": run_id, "actor": actor})
            if run["state"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert run["state"] == "succeeded", run

        analysis = _tool(
            client, 9, "analyze_sparameters", {"run_id": run_id, "actor": actor}
        )
        assert analysis["status"] == "ok"
        assert analysis["input_run_ids"] == [run_id]
        assert analysis["validation_status"] == "validated"
        assert analysis["limitations"]

        audit = client.get("/api/agent/audit?limit=1000").json()
        assert any(
            item["actor"] == actor
            and item["tool_name"] == "start_run"
            and item["outcome"] == "success"
            and item["run_id"] == run_id
            for item in audit
        )

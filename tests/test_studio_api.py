from __future__ import annotations

import json
from pathlib import Path
import time
import zipfile

import pytest

from rfx.studio.api import create_app
from rfx.studio.cli import build_parser, validate_launch_security


FIXTURE = Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_v2.json"


def _spec() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _client(tmp_path, *, static_dir=None, copilot_provider=None):
    from fastapi.testclient import TestClient

    return TestClient(
        create_app(
            tmp_path / "workspace",
            static_dir=static_dir,
            copilot_provider=copilot_provider,
        )
    )


class _FixedCopilotProvider:
    name = "test-llm"
    model = "test-structured-output"

    def __init__(self, patch=None):
        self.patch = patch or [
            {
                "op": "replace",
                "path": "/metadata/title",
                "value": "Copilot-reviewed RF design",
            }
        ]
        self.contexts = []

    def propose(self, *, intent, base_spec, run_context):
        self.contexts.append(
            {"intent": intent, "base_spec": base_spec, "run_context": run_context}
        )
        return {
            "answer": "I prepared one bounded design change.",
            "summary": "Retitle the canonical CPU experiment",
            "rationale": ["The requested label does not alter RF physics."],
            "patch": self.patch,
            "expected_effects": ["Generated Python keeps the same semantic plan."],
            "caveats": ["A validated run is still required for RF claims."],
            "needs_clarification": False,
            "question": "",
        }


def test_health_capabilities_and_empty_workspace(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/api/health").json() == {
            "status": "ok",
            "mode": "local",
            "backend": "cpu",
        }
        capabilities = client.get("/api/capabilities").json()
        assert capabilities["experiment_schema"] == "rfx-experiment/v2"
        assert set(capabilities["workflows"]) == {
            "patch_antenna",
            "wr90_waveguide",
            "multilayer_fresnel",
        }
        assert client.get("/api/experiments").json() == []


def test_copilot_initial_proposal_is_compiled_but_non_persistent(tmp_path):
    provider = _FixedCopilotProvider()
    with _client(tmp_path, copilot_provider=provider) as client:
        capabilities = client.get("/api/capabilities").json()["design_copilot"]
        assert capabilities == {
            "schema_version": "rfx-design-proposal/v1",
            "provider": "test-llm",
            "model": "test-structured-output",
            "llm": False,
            "store_provider_responses": False,
            "mutation_mode": "proposal-only",
            "protected_paths": [
                "/schema_version",
                "/kind",
                "/metadata/id",
                "/metadata/author",
                "/metadata/parent_revision",
                "/execution/backend",
            ],
        }
        response = client.post(
            "/api/copilot/proposals",
            json={"intent": "2.4 GHz patch antenna를 설계해줘"},
        )
        assert response.status_code == 200, response.text
        proposal = response.json()
        assert proposal["schema_version"] == "rfx-design-proposal/v1"
        assert proposal["workflow"] == "patch_antenna"
        assert proposal["base_revision_id"] is None
        assert proposal["candidate_spec"]["metadata"]["title"] == (
            "Copilot-reviewed RF design"
        )
        assert proposal["preview"]["preflight"]["ok"] is True
        assert "def build_simulation" in proposal["preview"]["generated_python"]
        assert proposal["cpu_estimate"]["backend"] == "cpu"
        assert proposal["cpu_estimate"]["estimated_cells"] > 0
        assert proposal["state_change"] == "none"
        assert client.get("/api/experiments").json() == []


def test_copilot_revision_proposal_uses_exact_base_without_mutating_it(tmp_path):
    provider = _FixedCopilotProvider()
    with _client(tmp_path, copilot_provider=provider) as client:
        created = client.post("/api/experiments", json=_spec()).json()
        experiment = created["experiment"]
        revision = created["revision"]
        response = client.post(
            "/api/copilot/proposals",
            json={
                "intent": "결과를 보기 좋게 이름을 바꿔줘",
                "revision_id": revision["id"],
            },
        )
        assert response.status_code == 200, response.text
        proposal = response.json()
        assert proposal["experiment_id"] == experiment["id"]
        assert proposal["base_revision_id"] == revision["id"]
        assert proposal["next_action"] == "review_then_load_draft"
        current = client.get(f"/api/experiments/{experiment['id']}").json()
        assert current["current_revision_id"] == revision["id"]
        assert current["revision_count"] == 1
        assert provider.contexts[-1]["base_spec"]["metadata"]["title"] == (
            "2.4 GHz FR4 patch antenna"
        )


def test_copilot_rejects_protected_model_patch(tmp_path):
    provider = _FixedCopilotProvider(
        patch=[
            {
                "op": "replace",
                "path": "/execution/backend",
                "value": "gpu",
            }
        ]
    )
    with _client(tmp_path, copilot_provider=provider) as client:
        response = client.post(
            "/api/copilot/proposals", json={"intent": "GPU로 바꿔줘"}
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "protected_patch_path"


def test_create_read_preview_generated_code_and_patch_conflict(tmp_path):
    with _client(tmp_path) as client:
        created = client.post("/api/experiments", json=_spec())
        assert created.status_code == 201, created.text
        payload = created.json()
        experiment = payload["experiment"]
        first = payload["revision"]

        revision = client.get(f"/api/revisions/{first['id']}").json()
        assert revision["validation_state"] == "validated"
        assert revision["scene"]["workflow"] == "patch_antenna"
        assert "def build_simulation" in revision["generated_python"]

        patch = {
            "base_revision_id": first["id"],
            "actor": "human:test",
            "message": "retitle",
            "patch": [
                {
                    "op": "replace",
                    "path": "/metadata/title",
                    "value": "Retitled patch experiment",
                }
            ],
        }
        second = client.post(f"/api/experiments/{experiment['id']}/patch", json=patch)
        assert second.status_code == 201, second.text
        assert second.json()["sequence"] == 2

        stale = client.post(f"/api/experiments/{experiment['id']}/patch", json=patch)
        assert stale.status_code == 409
        assert stale.json()["detail"]["code"] == "stale_revision"


def test_patch_run_journey_reaches_immutable_artifact(tmp_path):
    with _client(tmp_path) as client:
        created = client.post("/api/experiments", json=_spec()).json()
        experiment_id = created["experiment"]["id"]
        started = client.post(
            f"/api/experiments/{experiment_id}/runs",
            json={"idempotency_key": "studio-e2e-run"},
        )
        assert started.status_code == 202, started.text
        run_id = started.json()["id"]

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            run = client.get(f"/api/runs/{run_id}").json()
            if run["state"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert run["state"] == "succeeded", run
        assert run["progress"] == 1.0
        s11 = next(item for item in run["artifacts"] if item["kind"] == "s11")
        artifact = client.get(s11["url"])
        assert artifact.status_code == 200
        data = artifact.json()
        assert data["schema_version"] == "rfx-s11-artifact/v1"
        assert data["runtime"]["backend"] == "cpu"
        field = next(item for item in run["artifacts"] if item["kind"] == "field-slice")
        field_response = client.get(field["url"])
        assert field_response.status_code == 200
        field_data = field_response.json()
        assert field_data["schema_version"] == "rfx-field-slice-artifact/v1"
        assert field_data["component"] == "ez"
        assert field_data["slice_axis"] == "z"
        assert field_data["maximum_absolute"] > 0
        assert field_data["shape"] == [
            len(field_data["values"]),
            len(field_data["values"][0]),
        ]

        exported = client.post(f"/api/runs/{run_id}/export")
        assert exported.status_code == 201, exported.text
        bundle = client.get(exported.json()["url"])
        assert bundle.status_code == 200
        bundle_path = tmp_path / "run.zip"
        bundle_path.write_bytes(bundle.content)
        with zipfile.ZipFile(bundle_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["schema_version"] == "rfx-replay-bundle/v1"
            assert manifest["run_id"] == run_id
            assert {item["path"] for item in manifest["files"]} >= {
                "experiment/spec.json",
                "experiment/generated.py",
                "run/events.json",
                "analysis/metrics.json",
                "analysis/validation.json",
            }
            artifact_index = json.loads(archive.read("artifacts/index.json"))
            field_index = next(
                item for item in artifact_index if item["kind"] == "field-slice"
            )
            bundled_field = json.loads(archive.read(field_index["logical_path"]))
            assert bundled_field["schema_version"] == "rfx-field-slice-artifact/v1"


def test_static_spa_fallback_and_loopback_host_guard(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<main>rfx Studio</main>", encoding="utf-8")
    with _client(tmp_path, static_dir=static) as client:
        assert "rfx Studio" in client.get("/").text
        assert "rfx Studio" in client.get("/experiments/example").text
        metadata = client.get("/.well-known/oauth-protected-resource/mcp")
        assert metadata.status_code == 404
        assert metadata.content == b""

    parser = build_parser()
    assert parser.parse_args(["--host", "127.0.0.1"]).host == "127.0.0.1"
    with pytest.raises(ValueError, match="auth-token-file"):
        validate_launch_security(parser.parse_args(["--host", "0.0.0.0"]))

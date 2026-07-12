"""FastAPI transport over the durable rfx experiment application service."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import ipaddress
from pathlib import Path
from typing import Any

from rfx.agents import AgentControlPlane, TOOL_DESCRIPTORS
from rfx.agents.mcp import create_mcp_server
from rfx.experiments import (
    DurableStudyService,
    ExperimentService,
    RevisionConflictError,
    analyze_run as analyze_workflow_run,
    compare_sparameter_runs,
    compile_experiment,
    explain_validation,
    replay_bundle,
    verify_replay_bundle,
)
from rfx.studio.security import (
    RemoteSecurityConfig,
    install_remote_security,
    mcp_security_lists,
)
from rfx.studio.copilot import (
    CopilotError,
    CopilotProvider,
    CopilotProviderError,
    DesignCopilot,
)


def create_app(
    workspace: str | Path,
    *,
    static_dir: str | Path | None = None,
    remote_auth_token: str | None = None,
    allowed_origins: list[str] | None = None,
    mcp_host: str = "127.0.0.1",
    copilot_provider: CopilotProvider | None = None,
):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError(
            'rfx Studio requires the studio extra: pip install "rfx-fdtd[studio]"'
        ) from exc

    root = Path(workspace).expanduser().resolve()
    service = ExperimentService(root)
    study_service = DurableStudyService(service)
    agent_control = AgentControlPlane(service)
    design_copilot = DesignCopilot(service, provider=copilot_provider)
    remote_security = (
        RemoteSecurityConfig(
            token=remote_auth_token,
            allowed_origins=tuple(allowed_origins or ()),
        )
        if remote_auth_token is not None
        else None
    )
    if not _is_loopback_host(mcp_host) and remote_security is None:
        raise ValueError("non-loopback Studio requires remote authentication")
    mcp_hosts, mcp_origins = mcp_security_lists(
        mcp_host,
        remote_security.allowed_origins
        if remote_security
        else ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"),
    )
    mcp_server = create_mcp_server(
        agent_control,
        host=mcp_host,
        streamable_http_path="/",
        allowed_hosts=mcp_hosts,
        allowed_origins=mcp_origins,
    )
    mcp_http_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app):
        service.reconcile_stale_runs()
        async with mcp_server.session_manager.run():
            yield

    app = FastAPI(
        title="rfx Studio API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.study_service = study_service
    app.state.agent_control = agent_control
    app.state.design_copilot = design_copilot
    app.state.mcp_server = mcp_server
    app.state.workspace = root
    cors_origins = (
        list(remote_security.allowed_origins)
        if remote_security
        else [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8765",
            "http://localhost:8765",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Idempotency-Key"],
    )
    if remote_security:
        install_remote_security(app, remote_security)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        if remote_security:
            return {
                "status": "ok",
                "mode": "remote",
                "backend": "cpu",
                "authentication_required": True,
            }
        return {"status": "ok", "mode": "local", "backend": "cpu"}

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "schema_version": "rfx-studio-capabilities/v1",
            "experiment_schema": "rfx-experiment/v2",
            "workflows": [
                "patch_antenna",
                "wr90_waveguide",
                "multilayer_fresnel",
            ],
            "backend": "cpu",
            "loopback_only": True,
            "mcp_endpoint": "/mcp/",
            "design_copilot": design_copilot.capabilities(),
        }

    @app.post("/api/copilot/proposals")
    def propose_design(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Return a compiled, non-persistent design proposal for human review."""

        try:
            return design_copilot.propose(
                intent=payload.get("intent"),
                revision_id=payload.get("revision_id"),
                run_id=payload.get("run_id"),
            )
        except CopilotProviderError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        except CopilotError as exc:
            status = 409 if exc.code == "stale_revision" else 422
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    @app.get("/api/agent/tools")
    def agent_tools() -> list[dict[str, Any]]:
        return [item.to_dict() for item in TOOL_DESCRIPTORS.values()]

    @app.get("/api/agent/approvals")
    def list_agent_approvals(status: str | None = None) -> list[dict[str, Any]]:
        if status not in {
            None,
            "pending",
            "approved",
            "rejected",
            "consumed",
            "expired",
        }:
            raise HTTPException(status_code=422, detail={"code": "invalid_status"})
        return [
            agent_control.approval_payload(item.id)
            for item in agent_control.repository.list_approvals(status=status)
        ]

    @app.post("/api/agent/approvals/{approval_id}/approve")
    def approve_agent_action(
        approval_id: str, payload: dict[str, Any] = Body(default={})
    ) -> dict[str, Any]:
        try:
            return agent_control.approve(
                approval_id,
                decided_by=str(payload.get("actor", "human:studio")),
            )
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    @app.post("/api/agent/approvals/{approval_id}/reject")
    def reject_agent_action(
        approval_id: str, payload: dict[str, Any] = Body(default={})
    ) -> dict[str, Any]:
        try:
            return agent_control.reject(
                approval_id,
                decided_by=str(payload.get("actor", "human:studio")),
            )
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    @app.get("/api/agent/audit")
    def agent_audit(limit: int = 200) -> list[dict[str, Any]]:
        return agent_control.audit_payload(limit=limit)

    @app.post("/api/preview")
    def preview_experiment(spec: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Compile an unpersisted proposal for sub-second Studio feedback."""

        try:
            compiled = compile_experiment(spec)
            preflight = compiled.preflight()
        except Exception as exc:
            raise _http_error(exc) from exc
        return {
            "spec_sha256": compiled.spec.sha256,
            "semantic_fingerprint": compiled.semantic_fingerprint,
            "scene": compiled.scene_preview(),
            "generated_python": compiled.generated_python,
            # A draft must pass the same solver-aware gate as a persisted
            # revision before Studio presents it as safe to save or run.
            "preflight": preflight,
            "diagnostics": [item.to_dict() for item in compiled.diagnostics],
        }

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        return [
            _experiment_payload(service, record.id)
            for record in service.application_repository.list_experiments()
        ]

    @app.post("/api/experiments", status_code=201)
    def create_experiment(spec: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            experiment, revision = service.create_experiment(spec)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {
            "experiment": _experiment_payload(service, experiment.id),
            "revision": _revision_payload(service, revision.id),
        }

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        try:
            return _experiment_payload(service, experiment_id)
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc

    @app.get("/api/experiments/{experiment_id}/revisions")
    def list_revisions(experiment_id: str) -> list[dict[str, Any]]:
        try:
            revisions = service.application_repository.list_revisions(experiment_id)
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc
        return [
            _revision_payload(service, item.id, include_spec=False)
            for item in revisions
        ]

    @app.post("/api/experiments/{experiment_id}/patch", status_code=201)
    def patch_experiment(
        experiment_id: str, payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        try:
            revision = service.apply_experiment_patch(
                experiment_id,
                base_revision_id=str(payload["base_revision_id"]),
                patch=payload["patch"],
                actor=str(payload.get("actor", "local-user")),
                message=payload.get("message"),
            )
        except RevisionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "stale_revision",
                    "message": str(exc),
                    "current_revision_id": exc.current_revision_id,
                },
            ) from exc
        except Exception as exc:
            raise _http_error(exc) from exc
        return _revision_payload(service, revision.id)

    @app.get("/api/revisions/{revision_id}")
    def get_revision(revision_id: str) -> dict[str, Any]:
        try:
            return _revision_payload(service, revision_id)
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc

    @app.post("/api/revisions/{revision_id}/validate")
    def validate_revision(revision_id: str) -> dict[str, Any]:
        try:
            revision = service.application_repository.get_revision(revision_id)
            compiled = compile_experiment(revision.spec)
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc
        return {
            "revision_id": revision_id,
            "validation_state": revision.validation_state,
            "preflight": compiled.preflight(),
            "diagnostics": [item.to_dict() for item in compiled.diagnostics],
        }

    @app.post("/api/experiments/{experiment_id}/runs", status_code=202)
    def start_run(
        experiment_id: str, payload: dict[str, Any] = Body(default={})
    ) -> dict[str, Any]:
        try:
            linked = service.submit_revision(
                experiment_id,
                revision_id=payload.get("revision_id"),
                idempotency_key=payload.get("idempotency_key"),
            )
            if linked.run.state == "queued":
                service.start(linked.run.id)
            return _run_payload(service, linked.run.id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [
            _run_payload(service, item.id) for item in service.repository.list_runs()
        ]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return _run_payload(service, run_id)
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc

    @app.get("/api/runs/{run_id}/analysis")
    def analyze_run(run_id: str) -> dict[str, Any]:
        try:
            analysis = analyze_workflow_run(service, run_id)
            return {
                "analysis": analysis,
                "sparameters": (
                    analysis if analysis["analysis_kind"] == "sparameters" else None
                ),
                "validation": explain_validation(service, run_id),
            }
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    @app.post("/api/runs/compare")
    def compare_runs(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return compare_sparameter_runs(service, payload["run_ids"])
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            service.cancel(run_id)
            return _run_payload(service, run_id)
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc

    @app.post("/api/runs/{run_id}/export", status_code=201)
    def export_run(run_id: str) -> dict[str, Any]:
        try:
            service.export_run_snapshot(run_id)
            artifact = next(
                item
                for item in service.application_repository.list_artifacts(run_id)
                if item.kind == "replay-bundle"
            )
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc
        return {
            "artifact_id": artifact.id,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "url": f"/api/artifacts/{artifact.id}",
        }

    @app.get("/api/studies")
    def list_studies(experiment_id: str | None = None) -> list[dict[str, Any]]:
        return [
            study_service.payload(item.id) for item in study_service.list(experiment_id)
        ]

    @app.post("/api/experiments/{experiment_id}/studies", status_code=201)
    def create_study(
        experiment_id: str, payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        try:
            study = study_service.create(
                experiment_id,
                kind=str(payload["kind"]),
                parameter_path=str(payload["parameter_path"]),
                values=payload["values"],
                objective=str(payload.get("objective", "minimum_s11_db")),
                direction=str(payload.get("direction", "minimize")),
                budget_points=int(payload.get("budget_points", 20)),
                early_stop_value=payload.get("early_stop_value"),
            )
            return study_service.payload(study.id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/studies/{study_id}/run")
    def run_study(study_id: str) -> dict[str, Any]:
        try:
            return study_service.run(study_id)
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    @app.post("/api/studies/{study_id}/cancel", status_code=202)
    def cancel_study(study_id: str) -> dict[str, Any]:
        try:
            return study_service.cancel(study_id)
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    @app.post("/api/studies/{study_id}/promote")
    def promote_study(study_id: str) -> dict[str, Any]:
        try:
            return study_service.promote_best(study_id)
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    @app.get("/api/artifacts/{artifact_id}")
    def read_artifact(artifact_id: str):
        try:
            artifact = service.application_repository.get_artifact(artifact_id)
        except Exception as exc:
            raise _http_error(exc, not_found=True) from exc
        path = Path(artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "artifact_unavailable",
                    "message": "artifact is unavailable",
                },
            )
        actual_size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_size != artifact.size_bytes or actual_sha256 != artifact.sha256:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "artifact_integrity_error",
                    "message": "artifact bytes do not match the registered digest",
                },
            )
        media_type = {
            ".json": "application/json",
            ".zip": "application/zip",
        }.get(path.suffix, "text/plain")
        return FileResponse(path, media_type=media_type, filename=path.name)

    @app.post("/api/artifacts/{artifact_id}/replay")
    def replay_artifact(artifact_id: str) -> dict[str, Any]:
        try:
            artifact = service.application_repository.get_artifact(artifact_id)
            if artifact.kind != "replay-bundle":
                raise ValueError("artifact is not a replay bundle")
            verify_replay_bundle(artifact.path)
            replay_root = root / "replays" / artifact.id
            return replay_bundle(artifact.path, replay_root)
        except Exception as exc:
            raise _http_error(exc, not_found=isinstance(exc, KeyError)) from exc

    app.mount("/mcp", mcp_http_app, name="mcp")

    resolved_static = (
        Path(static_dir).expanduser().resolve()
        if static_dir is not None
        else Path(__file__).parent / "static"
    )
    if resolved_static.is_dir() and (resolved_static / "index.html").is_file():
        assets = resolved_static / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path == ".well-known" or path.startswith(".well-known/"):
                return Response(status_code=404)
            candidate = (resolved_static / path).resolve()
            if (
                path
                and candidate.is_relative_to(resolved_static)
                and candidate.is_file()
            ):
                return FileResponse(candidate)
            return FileResponse(resolved_static / "index.html")

    return app


def _experiment_payload(
    service: ExperimentService, experiment_id: str
) -> dict[str, Any]:
    experiment = service.application_repository.get_experiment(experiment_id)
    revisions = service.application_repository.list_revisions(experiment_id)
    return {
        "id": experiment.id,
        "title": experiment.title,
        "current_revision_id": experiment.current_revision_id,
        "revision_count": len(revisions),
        "created_at": experiment.created_at,
        "updated_at": experiment.updated_at,
    }


def _revision_payload(
    service: ExperimentService, revision_id: str, *, include_spec: bool = True
) -> dict[str, Any]:
    revision = service.application_repository.get_revision(revision_id)
    payload = {
        "id": revision.id,
        "experiment_id": revision.experiment_id,
        "sequence": revision.sequence,
        "parent_revision_id": revision.parent_revision_id,
        "spec_sha256": revision.spec_sha256,
        "semantic_fingerprint": revision.semantic_fingerprint,
        "validation_state": revision.validation_state,
        "preflight": revision.preflight,
        "actor": revision.actor,
        "message": revision.message,
        "created_at": revision.created_at,
    }
    if include_spec:
        compiled = compile_experiment(revision.spec)
        payload.update(
            {
                "spec": revision.spec,
                "scene": compiled.scene_preview(),
                "generated_python": compiled.generated_python,
                "diagnostics": [item.to_dict() for item in compiled.diagnostics],
            }
        )
    return payload


def _run_payload(service: ExperimentService, run_id: str) -> dict[str, Any]:
    run = service.refresh(run_id)
    try:
        linked = service.application_repository.get_linked_run(run_id)
    except KeyError:
        linked = None
    return {
        "id": run.id,
        "state": run.state,
        "experiment_id": linked.experiment_id if linked else None,
        "revision_id": linked.revision_id if linked else None,
        "progress": linked.progress if linked else None,
        "heartbeat_at": linked.heartbeat_at if linked else None,
        "artifact_sha256": run.artifact_sha256,
        "error": run.error,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "events": [
            {
                "sequence": event.sequence,
                "type": event.event_type,
                "state": event.state,
                "payload": event.payload,
                "created_at": event.created_at,
            }
            for event in service.repository.list_events(run_id)
        ],
        "artifacts": [
            {
                "id": artifact.id,
                "kind": artifact.kind,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "url": f"/api/artifacts/{artifact.id}",
            }
            for artifact in service.application_repository.list_artifacts(run_id)
        ],
    }


def _http_error(exc: Exception, *, not_found: bool = False):
    from fastapi import HTTPException

    status = 404 if not_found or isinstance(exc, KeyError) else 422
    code = getattr(exc, "code", "not_found" if status == 404 else "invalid_request")
    return HTTPException(
        status_code=status,
        detail={"code": code, "message": str(exc)},
    )


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

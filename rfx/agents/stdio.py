"""Stdio entrypoint for Claude Desktop and private OpenAI MCP tunnels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfx-agent-mcp",
        description="Run the approval-gated rfx Studio MCP server over stdio.",
    )
    parser.add_argument(
        "--workspace",
        default=".rfx-studio",
        help="durable rfx Studio workspace (default: .rfx-studio)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from rfx.agents.control import AgentControlPlane
    from rfx.agents.mcp import create_mcp_server
    from rfx.experiments import ExperimentService

    service = ExperimentService(Path(args.workspace))
    service.reconcile_stale_runs()
    create_mcp_server(AgentControlPlane(service)).run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the entrypoint
    raise SystemExit(main())

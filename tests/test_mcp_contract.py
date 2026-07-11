from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

from rfx.studio.api import create_app


SNAPSHOT = Path(__file__).parent / "fixtures" / "mcp" / "rfx_tools_v1.json"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def _rpc(client, identifier, method, params=None, *, headers=None):
    response = client.post(
        "/mcp/",
        headers={**MCP_HEADERS, **(headers or {})},
        json={
            "jsonrpc": "2.0",
            "id": identifier,
            "method": method,
            "params": params or {},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_openai_and_claude_clients_discover_identical_tool_schema(tmp_path):
    from fastapi.testclient import TestClient

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["tools"]
    with TestClient(
        create_app(tmp_path / "workspace"), base_url="http://127.0.0.1:8765"
    ) as client:
        discovered = []
        for index, provider in enumerate(("openai-responses", "anthropic-claude"), 1):
            initialized = _rpc(
                client,
                index * 10,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": provider, "version": "release-smoke"},
                },
            )
            assert initialized["result"]["protocolVersion"] == "2025-06-18"
            listed = _rpc(client, index * 10 + 1, "tools/list")["result"]["tools"]
            normalized = [
                {
                    "name": tool["name"],
                    "properties": sorted(tool["inputSchema"]["properties"]),
                    "required": sorted(tool["inputSchema"].get("required", [])),
                    "read_only": tool["annotations"]["readOnlyHint"],
                    "destructive": tool["annotations"]["destructiveHint"],
                }
                for tool in listed
            ]
            assert normalized == expected
            discovered.append(listed)
        assert discovered[0] == discovered[1]

        capability = _rpc(
            client,
            99,
            "tools/call",
            {"name": "get_capabilities", "arguments": {"actor": "agent:test"}},
        )
        assert capability["result"]["structuredContent"]["status"] == "ok"


def test_streamable_http_rejects_untrusted_host_and_origin(tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(
        create_app(tmp_path / "workspace"), base_url="http://127.0.0.1:8765"
    ) as client:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "security-test", "version": "1"},
            },
        }
        bad_origin = client.post(
            "/mcp/",
            headers={**MCP_HEADERS, "Origin": "https://evil.invalid"},
            json=payload,
        )
        assert bad_origin.status_code == 403
        bad_host = client.post(
            "/mcp/",
            headers={**MCP_HEADERS, "Host": "evil.invalid"},
            json=payload,
        )
        assert bad_host.status_code == 421


def test_stdio_entrypoint_exposes_the_same_tool_contract(tmp_path):
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["tools"]

    async def discover():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "rfx.agents.stdio",
                "--workspace",
                str(tmp_path / "workspace"),
            ],
            env={
                **os.environ,
                "JAX_PLATFORMS": "cpu",
                "JAX_PLATFORM_NAME": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                capability = await session.call_tool(
                    "get_capabilities", {"actor": "agent:stdio-test"}
                )
                return initialized, listed.tools, capability

    initialized, tools, capability = asyncio.run(discover())
    assert initialized.protocolVersion == types.LATEST_PROTOCOL_VERSION
    normalized = [
        {
            "name": tool.name,
            "properties": sorted(tool.inputSchema["properties"]),
            "required": sorted(tool.inputSchema.get("required", [])),
            "read_only": tool.annotations.readOnlyHint,
            "destructive": tool.annotations.destructiveHint,
        }
        for tool in tools
    ]
    assert normalized == expected
    assert capability.structuredContent["status"] == "ok"

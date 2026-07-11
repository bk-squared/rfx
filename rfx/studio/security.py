"""Remote lab-server authentication and Origin enforcement."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hmac
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class RemoteSecurityConfig:
    token: str
    allowed_origins: tuple[str, ...]

    def __post_init__(self):
        if len(self.token) < 32:
            raise ValueError("remote auth token must contain at least 32 characters")
        if not self.allowed_origins:
            raise ValueError("remote mode requires at least one allowed HTTPS origin")
        for origin in self.allowed_origins:
            parsed = urlparse(origin)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.path not in {"", "/"}
            ):
                raise ValueError(
                    "allowed remote origins must be HTTPS origins without paths"
                )

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(urlparse(origin).netloc for origin in self.allowed_origins)
        )


def load_auth_token_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if os.name != "nt" and source.stat().st_mode & 0o077:
        raise PermissionError(
            "remote auth token file must not be group/world accessible"
        )
    value = source.read_text(encoding="utf-8").strip()
    if len(value) < 32 or any(character.isspace() for character in value):
        raise ValueError(
            "remote auth token must be one whitespace-free value of 32+ characters"
        )
    return value


def install_remote_security(app, config: RemoteSecurityConfig) -> None:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    class RemoteSecurityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            origin = request.headers.get("origin")
            if origin and origin not in config.allowed_origins:
                return JSONResponse(
                    {
                        "detail": {
                            "code": "origin_rejected",
                            "message": "Origin is not allowlisted",
                        }
                    },
                    status_code=403,
                )
            if _public_path(request.url.path):
                return await call_next(request)
            candidate = _authorization_token(request.headers.get("authorization"))
            if candidate is None or not hmac.compare_digest(candidate, config.token):
                return JSONResponse(
                    {
                        "detail": {
                            "code": "authentication_required",
                            "message": "Valid Bearer token or rfx Basic password required",
                        }
                    },
                    status_code=401,
                    headers={
                        "WWW-Authenticate": 'Basic realm="rfx Studio", charset="UTF-8"'
                    },
                )
            return await call_next(request)

    app.add_middleware(RemoteSecurityMiddleware)


def mcp_security_lists(
    host: str, allowed_origins: Iterable[str]
) -> tuple[list[str], list[str]]:
    origins = list(dict.fromkeys(allowed_origins))
    hosts = [urlparse(origin).netloc for origin in origins]
    if ":" not in host or host.startswith("["):
        hosts.append(f"{host}:*")
    hosts.extend(["127.0.0.1:*", "localhost:*", "[::1]:*"])
    return list(dict.fromkeys(hosts)), origins


def _authorization_token(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer":
        return value
    if scheme.lower() != "basic":
        return None
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    _, separator, password = decoded.partition(":")
    return password if separator else None


def _public_path(path: str) -> bool:
    if path == "/api/health":
        return True
    return not (path.startswith("/api/") or path.startswith("/mcp"))

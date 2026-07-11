"""Launch the loopback-only rfx Studio API and packaged frontend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

from .security import RemoteSecurityConfig, load_auth_token_file


def _host(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise argparse.ArgumentTypeError("host must be a non-empty hostname or IP")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch local-first rfx Studio")
    parser.add_argument("--host", type=_host, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", default=".rfx-studio")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auth-token-file")
    parser.add_argument("--allowed-origin", action="append", default=[])
    parser.add_argument(
        "--tls-terminated",
        action="store_true",
        help="confirm that an HTTPS reverse proxy terminates TLS before remote Studio",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        print("error: port must be in 1..65535", file=sys.stderr)
        return 2
    try:
        remote_token = validate_launch_security(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        import uvicorn
    except ImportError:
        print(
            'error: rfx Studio requires: pip install "rfx-fdtd[studio]"',
            file=sys.stderr,
        )
        return 1

    from .api import create_app

    app = create_app(
        Path(args.workspace),
        remote_auth_token=remote_token,
        allowed_origins=args.allowed_origin,
        mcp_host=args.host,
    )
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def validate_launch_security(args) -> str | None:
    from .api import _is_loopback_host

    if _is_loopback_host(args.host):
        if args.auth_token_file or args.allowed_origin or args.tls_terminated:
            raise ValueError(
                "remote auth/TLS options are only valid with a non-loopback host"
            )
        return None
    if not args.auth_token_file:
        raise ValueError("non-loopback bind requires --auth-token-file")
    if not args.allowed_origin:
        raise ValueError("non-loopback bind requires --allowed-origin https://...")
    if not args.tls_terminated:
        raise ValueError("non-loopback bind requires --tls-terminated")
    token = load_auth_token_file(args.auth_token_file)
    RemoteSecurityConfig(token=token, allowed_origins=tuple(args.allowed_origin))
    return token


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the ``rfx-dashboard`` console entry point.

These tests never launch Streamlit or open a browser: ``subprocess.run`` and
``importlib.util.find_spec`` are monkeypatched so we only assert on the command
that *would* be invoked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfx.dashboard import cli


def test_help_returns_zero_and_prints_usage(capsys, monkeypatch):
    """--help prints usage and never touches streamlit or subprocess."""

    def _boom(*_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("streamlit must not be probed for --help")

    monkeypatch.setattr(cli.importlib.util, "find_spec", _boom)
    monkeypatch.setattr(cli.subprocess, "run", _boom)

    rc = cli.main(["--help"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "usage: rfx-dashboard" in out
    assert "streamlit" in out


def test_short_help_flag(capsys, monkeypatch):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: pytest.fail("subprocess.run called on -h"),
    )
    assert cli.main(["-h"]) == 0
    assert "usage: rfx-dashboard" in capsys.readouterr().out


def test_app_path_exists():
    """The resolved app.py path must point at a real file."""
    app_path = cli._app_path()
    assert app_path.is_file()
    assert app_path.name == "app.py"
    assert app_path.parent.name == "dashboard"


def test_launch_builds_streamlit_run_command(monkeypatch):
    """main([]) invokes `streamlit run <abs app.py>` and returns its code."""
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: object())

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    rc = cli.main([])
    assert rc == 0

    cmd = captured["cmd"]
    assert cmd[:2] == ["streamlit", "run"]
    app_arg = Path(cmd[2])
    assert app_arg.is_absolute()
    assert app_arg.is_file()
    assert app_arg.name == "app.py"


def test_extra_args_are_forwarded(monkeypatch):
    """Extra flags like --server.port are passed through to streamlit."""
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: object())

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    cli.main(["--server.port", "8502"])

    cmd = captured["cmd"]
    assert cmd[-2:] == ["--server.port", "8502"]
    assert cmd[:2] == ["streamlit", "run"]


def test_propagates_streamlit_exit_code(monkeypatch):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7)
    )
    assert cli.main([]) == 7


def test_missing_streamlit_returns_nonzero_with_actionable_message(capsys, monkeypatch):
    """When streamlit is absent, exit non-zero and name the dashboard extra."""

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: None)

    def _must_not_run(*_a, **_k):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run must not run without streamlit")

    monkeypatch.setattr(cli.subprocess, "run", _must_not_run)

    rc = cli.main([])
    assert rc != 0

    err = capsys.readouterr().err
    assert "rfx-fdtd[dashboard]" in err
    assert "not installed" in err


def test_console_script_fallback_to_module(monkeypatch):
    """If the `streamlit` script is missing, fall back to `python -m streamlit`."""
    calls: list[list[str]] = []

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: object())

    def _fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "streamlit":
            raise FileNotFoundError("no streamlit on PATH")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    rc = cli.main([])
    assert rc == 0
    assert len(calls) == 2
    assert calls[0][0] == "streamlit"
    assert calls[1][1:3] == ["-m", "streamlit"]
    assert "run" in calls[1]


def test_subprocess_module_is_the_real_one():
    """Sanity: cli references the stdlib subprocess (monkeypatch target check)."""
    assert cli.subprocess is subprocess

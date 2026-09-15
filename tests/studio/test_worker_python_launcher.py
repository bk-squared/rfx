"""A Windows venv redirector must not become the owned execution PID."""

from types import SimpleNamespace

import pytest

from rfx.experiments import _worker_child


def test_windows_venv_launch_preserves_environment_without_a_redirector(monkeypatch):
    executable = "C:/venv/Scripts/python.exe"
    base = "C:/Python/python.exe"
    monkeypatch.setattr(
        _worker_child,
        "sys",
        SimpleNamespace(
            platform="win32",
            executable=executable,
            _base_executable=base,
        ),
    )
    arguments = [executable, "-m", "rfx.experiments.worker", "--run-id", "owned-run"]
    environment = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}

    command, child_environment = _worker_child._python_command(arguments, environment)

    assert command == [base, *arguments[1:]]
    assert child_environment == {**environment, "__PYVENV_LAUNCHER__": executable}
    assert arguments[0] == executable
    assert "__PYVENV_LAUNCHER__" not in environment


@pytest.mark.parametrize(
    "platform,executable,base,requested",
    [
        ("linux", "/venv/bin/python", "/usr/bin/python", "/venv/bin/python"),
        (
            "win32",
            "C:/Python/python.exe",
            "C:/Python/python.exe",
            "C:/Python/python.exe",
        ),
        (
            "win32",
            "C:/venv/Scripts/python.exe",
            "C:/Python/python.exe",
            "other-helper.exe",
        ),
    ],
)
def test_python_launch_does_not_redirect_other_commands(
    monkeypatch,
    platform,
    executable,
    base,
    requested,
):
    monkeypatch.setattr(
        _worker_child,
        "sys",
        SimpleNamespace(
            platform=platform,
            executable=executable,
            _base_executable=base,
        ),
    )
    command, environment = _worker_child._python_command(
        [requested, "argument"], {"JAX_PLATFORMS": "cpu"}
    )
    assert command == [requested, "argument"]
    assert environment == {"JAX_PLATFORMS": "cpu"}

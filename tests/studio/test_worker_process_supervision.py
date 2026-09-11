"""Worker deadline controls that actually enter noncooperating native work."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from rfx.experiments import worker, service as service_module
from rfx.experiments.durable import ResourceBusyError, SQLiteApplicationRepository
from rfx.experiments.repository import SQLiteRunRepository
from rfx.experiments.service import ExperimentService


@pytest.fixture(scope="module")
def native_child(tmp_path_factory):
    if os.name != "posix":
        pytest.skip("the native control uses POSIX clock_gettime")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is needed for the native-blocking control")
    root = tmp_path_factory.mktemp("native-worker-control")
    source = root / "busy.c"
    source.write_text(
        """#include <time.h>
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
void busy_seconds(double seconds) {
    const double end = now() + seconds;
    volatile double value = 0.0;
    while (now() < end) {
        for (int i = 0; i < 1000; ++i) value += 0.000001;
    }
}
""",
        encoding="utf-8",
    )
    library = root / "busy.so"
    subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        timeout=30,
        capture_output=True,
    )
    script = root / "native.py"
    script.write_text(
        """import ctypes, json, os, runpy, signal, sys
from pathlib import Path
if len(sys.argv) == 5:
    runpy.run_path(sys.argv[3])["_bind_parent"](int(sys.argv[4]))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
library = ctypes.PyDLL(sys.argv[1])  # Hold the GIL throughout the native call.
library.busy_seconds.argtypes = [ctypes.c_double]
ready = Path(sys.argv[2])
temporary = ready.with_suffix(".tmp")
temporary.write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp()}))
temporary.replace(ready)
library.busy_seconds(30.0)
""",
        encoding="utf-8",
    )

    def command(ready: Path, *, parent_pid: int | None = None):
        result = [sys.executable, str(script), str(library), str(ready)]
        if parent_pid is not None:
            result += [
                str(Path(worker.__file__).with_name("_worker_child.py")),
                str(parent_pid),
            ]
        return result

    return command


def _run(tmp_path, *, timeout=1):
    database = tmp_path / "experiments.sqlite3"
    repository = SQLiteRunRepository(database)
    application = SQLiteApplicationRepository(database)
    run = repository.create_run(
        spec_json=json.dumps({"execution": {"timeout_seconds": timeout}}),
        spec_sha256="a" * 64,
        compiled_sha256="b" * 64,
    )
    (tmp_path / "runs" / run.id).mkdir(parents=True)
    application.acquire_cpu_lease(run.id)
    return repository, application, run


def _assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_native_timeout_is_durable_and_releases_lease_only_after_reaping(
    tmp_path,
    native_child,
    monkeypatch,
):
    repository, application, run = _run(tmp_path)
    other = repository.create_run(
        spec_json="{}", spec_sha256="c" * 64, compiled_sha256="d" * 64
    )
    ready = tmp_path / "native-ready.json"
    monkeypatch.setattr(worker, "_child_command", lambda **_: native_child(ready))
    original_stop = worker._stop_child
    checked_live_lease = []

    def stop_after_lease_check(process):
        if process.poll() is None:
            with pytest.raises(ResourceBusyError):
                application.acquire_cpu_lease(other.id)
            checked_live_lease.append(True)
        original_stop(process)

    monkeypatch.setattr(worker, "_stop_child", stop_after_lease_check)
    started = time.monotonic()
    result = worker.execute_run(
        database=repository.path, workspace=tmp_path, run_id=run.id
    )
    elapsed = time.monotonic() - started

    assert ready.is_file(), "the deadline must fire after native work starts"
    assert checked_live_lease
    _assert_reaped(json.loads(ready.read_text())["pid"])
    assert result == 1
    assert elapsed < 10, f"native work continued past the deadline: {elapsed}s"
    final = repository.get_run(run.id)
    assert final.state == "failed" and "timeout" in final.error
    terminal = [
        event for event in repository.list_events(run.id) if event.state == "failed"
    ]
    assert [event.event_type for event in terminal] == ["run_timed_out"]
    artifacts = application.list_artifacts(run.id)
    assert [artifact.kind for artifact in artifacts] == ["traceback"]
    assert "terminated and reaped" in Path(artifacts[0].path).read_text()
    application.acquire_cpu_lease(other.id)


@pytest.mark.parametrize("returncode", [0, 7])
def test_exit_without_outcome_is_a_durable_failure(tmp_path, monkeypatch, returncode):
    repository, application, run = _run(tmp_path, timeout=30)
    monkeypatch.setattr(
        worker,
        "_child_command",
        lambda **_: [sys.executable, "-c", f"raise SystemExit({returncode})"],
    )
    assert (
        worker.execute_run(database=repository.path, workspace=tmp_path, run_id=run.id)
        == 1
    )
    final = repository.get_run(run.id)
    assert final.state == "failed"
    assert f"status {returncode} without a complete outcome" in final.error
    assert [event.event_type for event in repository.list_events(run.id)][
        -1
    ] == "run_failed"
    assert {item.kind for item in application.list_artifacts(run.id)} == {"traceback"}


def test_diagnostic_write_failure_does_not_skip_terminal_state_or_lease_cleanup(
    tmp_path,
    monkeypatch,
):
    repository, application, run = _run(tmp_path, timeout=30)
    monkeypatch.setattr(
        worker, "_child_command", lambda **_: [sys.executable, "-c", "pass"]
    )

    def broken_diagnostic(*args):
        raise OSError("diagnostic volume unavailable")

    monkeypatch.setattr(worker, "_write_traceback_artifact", broken_diagnostic)
    assert (
        worker.execute_run(database=repository.path, workspace=tmp_path, run_id=run.id)
        == 1
    )
    assert repository.get_run(run.id).state == "failed"
    other = repository.create_run(
        spec_json="{}", spec_sha256="c" * 64, compiled_sha256="d" * 64
    )
    application.acquire_cpu_lease(other.id)


def test_terminal_commit_retry_preserves_registered_traceback_bytes(
    tmp_path, monkeypatch
):
    repository, application, run = _run(tmp_path, timeout=30)
    outcome_path = tmp_path / "runs" / run.id / "executor-outcome.json"
    proposal = {
        "state": "failed",
        "event_type": "run_failed",
        "error": "executor failed",
        "traceback": "original executor diagnostic\n",
    }
    child = (
        "from pathlib import Path\n"
        f"Path({str(outcome_path)!r}).write_text({json.dumps(proposal)!r})\n"
        "raise SystemExit(1)\n"
    )
    monkeypatch.setattr(
        worker, "_child_command", lambda **_: [sys.executable, "-c", child]
    )
    original_finish = SQLiteApplicationRepository.finish_worker_run
    calls = []

    def fail_first_commit(self, *args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            assert self.list_artifacts(run.id), (
                "first traceback must already be registered"
            )
            raise RuntimeError("terminal commit failed after diagnostic registration")
        return original_finish(self, *args, **kwargs)

    monkeypatch.setattr(
        SQLiteApplicationRepository, "finish_worker_run", fail_first_commit
    )
    assert (
        worker.execute_run(database=repository.path, workspace=tmp_path, run_id=run.id)
        == 1
    )
    assert len(calls) == 2
    artifacts = application.list_artifacts(run.id)
    assert len(artifacts) == 1
    data = Path(artifacts[0].path).read_bytes()
    assert artifacts[0].sha256 == hashlib.sha256(data).hexdigest()
    assert data == b"original executor diagnostic\n"
    assert b"terminal commit failed after diagnostic registration" not in data
    assert "terminal commit failed" in repository.get_run(run.id).error


def test_cancellation_stops_noncooperating_native_child(tmp_path, native_child):
    ready = tmp_path / "native-ready.json"
    process = subprocess.Popen(native_child(ready))
    try:
        observed = worker._wait_for_child(
            process,
            deadline=time.monotonic() + 10,
            cancelled=ready.is_file,
        )
        assert observed == "cancelled"
        assert process.returncode == -signal.SIGKILL
        _assert_reaped(json.loads(ready.read_text())["pid"])
    finally:
        worker._stop_child(process)


def test_interrupted_supervision_still_reaps_the_owned_child(tmp_path, native_child):
    ready = tmp_path / "native-ready.json"
    process = subprocess.Popen(native_child(ready))

    def interrupted():
        if ready.is_file():
            raise RuntimeError("supervisor observation failed")
        return False

    try:
        with pytest.raises(RuntimeError, match="observation failed"):
            worker._wait_for_child(
                process, deadline=time.monotonic() + 10, cancelled=interrupted
            )
        _assert_reaped(json.loads(ready.read_text())["pid"])
    finally:
        worker._stop_child(process)


def test_sqlite_lock_cannot_delay_stopping_native_computation(tmp_path, native_child):
    # Build the read-poll surface directly in a fresh rollback-journal DB.
    # Switching a repository's live WAL file to DELETE first needs every
    # other connection closed; relying on GC for that made setup itself fail
    # before this test had acquired the lock it intends to exercise.
    database = tmp_path / "cancel-control.sqlite3"
    run_id = "native-lock-control"
    ready = tmp_path / "native-ready.json"
    lock = sqlite3.connect(database)
    lock.execute("CREATE TABLE runs(id TEXT PRIMARY KEY, cancel_requested INTEGER)")
    lock.execute("INSERT INTO runs VALUES (?, 0)", (run_id,))
    lock.commit()
    assert lock.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    lock.execute("BEGIN EXCLUSIVE")
    contender = sqlite3.connect(database, timeout=0)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            contender.execute("SELECT count(*) FROM sqlite_master")
    finally:
        contender.close()
    process = subprocess.Popen(native_child(ready))
    started = time.monotonic()
    try:
        observed = worker._wait_for_child(
            process,
            deadline=started + 1,
            cancelled=lambda: worker._cancel_requested(database, run_id),
        )
        assert observed == "timeout"
        assert time.monotonic() - started < 5
        _assert_reaped(json.loads(ready.read_text())["pid"])
        assert lock.in_transaction, (
            "termination must precede waiting for DB finalization"
        )
    finally:
        worker._stop_child(process)
        lock.rollback()
        lock.close()


@pytest.mark.parametrize("durable_only", [False, True])
def test_reopened_service_cancels_native_execution_under_the_stable_worker_pid(
    tmp_path,
    native_child,
    monkeypatch,
    durable_only,
):
    service = ExperimentService(tmp_path)
    document = {
        "schema_version": "rfx-experiment/v1",
        "model": {"domain_m": [0.004] * 3, "cell_size_m": 0.002},
        "execution": {"timeout_seconds": 30},
    }
    run = service.repository.create_run(
        spec_json=json.dumps(document),
        spec_sha256="a" * 64,
        compiled_sha256="b" * 64,
    )
    (service.runs_root / run.id).mkdir()
    ready = tmp_path / "native-ready.json"
    # Substitute only the work being supervised. The service launches a real
    # worker interpreter, whose production execute_run owns the native child.
    command = native_child(ready, parent_pid=0)
    driver = (
        "import os, sys\nimport rfx.experiments.worker as worker\n"
        f"command = {command!r}\n"
        "def native_command(**kwargs):\n"
        "    command[-1] = str(os.getpid())\n    return command\n"
        "worker._child_command = native_command\n"
        "raise SystemExit(worker.main(sys.argv[1:]))\n"
    )
    original_popen = subprocess.Popen

    def start_supervisor(command, **kwargs):
        assert command[1:3] == ["-m", "rfx.experiments.worker"]
        return original_popen([command[0], "-c", driver, *command[3:]], **kwargs)

    monkeypatch.setattr(service_module.subprocess, "Popen", start_supervisor)
    process = service.start(run.id)
    # Restore the shared subprocess module before other service operations.
    monkeypatch.setattr(service_module.subprocess, "Popen", original_popen)
    try:
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline, "native child did not enter work"
            time.sleep(0.02)
        child = json.loads(ready.read_text())
        assert child["pgid"] == process.pid
        # Enter the same durable phases as production execution; this control
        # runs the native blocker in place of compile/preflight/FDTD.
        service.repository.transition(run.id, "preflighting", expected="queued")
        service.repository.transition(run.id, "running", expected="preflighting")
        controller = ExperimentService(tmp_path)
        assert controller.get(run.id).pid == process.pid
        if durable_only:
            # Exercise the Windows service branch without changing global os
            # semantics or pretending this is a Windows kernel execution test.
            monkeypatch.setattr(service_module, "os", SimpleNamespace(name="nt"))
        controller.cancel(run.id)
        final = service.wait(run.id, timeout=10)
        assert final.state == "cancelled" and final.cancel_requested
        assert [
            e.event_type
            for e in service.repository.list_events(run.id)
            if e.state == "cancelled"
        ] == ["run_cancelled"]
        _assert_reaped(child["pid"])
        assert {
            a.kind for a in service.application_repository.list_artifacts(run.id)
        } == {"stdout-log", "stderr-log"}
        other = service.repository.create_run(
            spec_json="{}", spec_sha256="c" * 64, compiled_sha256="d" * 64
        )
        service.application_repository.acquire_cpu_lease(other.id)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death kernel guard")
@pytest.mark.parametrize("kill_group", [False, True])
def test_supervisor_death_cannot_leave_native_execution_running(
    tmp_path,
    native_child,
    kill_group,
):
    ready = tmp_path / "native-ready.json"
    # A separate subreaper owns the whole control experiment. The pytest
    # process must not change its global orphan-adoption policy or leave a
    # zombie under a container's possibly non-reaping PID 1.
    probe = tmp_path / "parent-death-control.py"
    probe.write_text(
        """import ctypes, json, os, signal, subprocess, sys, time
from pathlib import Path
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
command = json.loads(sys.argv[1])
ready = Path(sys.argv[2])
kill_group = json.loads(sys.argv[3])
supervisor_code = '''import json, os, subprocess, sys
command = json.loads(sys.argv[1])
command[-1] = str(os.getpid())
child = subprocess.Popen(command)
child.wait()
'''
parent = subprocess.Popen([sys.executable, '-c', supervisor_code, json.dumps(command)], start_new_session=True)
child_pid = None
reaped = False
try:
    deadline = time.monotonic() + 10
    while not ready.exists():
        assert parent.poll() is None, 'supervisor exited before native work'
        assert time.monotonic() < deadline, 'native child did not start'
        time.sleep(0.02)
    child = json.loads(ready.read_text())
    child_pid = child['pid']
    assert child['pgid'] == parent.pid
    if kill_group:
        os.killpg(parent.pid, signal.SIGKILL)
    else:
        parent.kill()
    parent.wait(timeout=5)
    deadline = time.monotonic() + 5
    while True:
        pid, status = os.waitpid(child_pid, os.WNOHANG)
        if pid:
            reaped = True
            assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
            break
        assert time.monotonic() < deadline, 'native computation survived supervisor death'
        time.sleep(0.02)
    print(json.dumps({'native_entered': True, 'child_reaped': True, 'kill_group': kill_group}))
finally:
    if parent.poll() is None:
        parent.kill()
    parent.wait()
    if child_pid is not None and not reaped:
        pid, status = os.waitpid(child_pid, os.WNOHANG)
        if not pid:
            # waitpid just confirmed this is our own unreaped child.
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            json.dumps(native_child(ready, parent_pid=0)),
            str(ready),
            json.dumps(kill_group),
        ],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "native_entered": True,
        "child_reaped": True,
        "kill_group": kill_group,
    }

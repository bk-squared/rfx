"""Fresh execution bootstrap; run by file path before importing rfx/JAX.

The supervisor owns this process and its deadline. On Linux, parent death
also kills it in the kernel, including while native code holds the GIL.
Other platforms retain normal supervisor cleanup but lack this Linux guard.
"""

from __future__ import annotations

import os
import signal
import sys


def _python_command(
    command: list[str],
    environment: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Launch the actual Python process while preserving its virtual env.

    Windows venv python.exe is a redirector that creates another process.
    Follow CPython multiprocessing's bypass: the Popen handle must own the
    interpreter we will terminate/reap, and getppid must see its supervisor.
    """
    command = list(command)
    environment = dict(os.environ if environment is None else environment)
    base = getattr(sys, "_base_executable", None)
    if (
        sys.platform == "win32"
        and base
        and os.path.normcase(command[0]) == os.path.normcase(sys.executable)
        and os.path.normcase(base) != os.path.normcase(sys.executable)
    ):
        command[0] = base
        environment["__PYVENV_LAUNCHER__"] = sys.executable
    return command, environment


def _bind_parent(parent_pid: int) -> None:
    if sys.platform == "linux":
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        # PR_SET_PDEATHSIG. Arm before importing any numerical libraries.
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    # Close the race where the supervisor died before the kernel guard was
    # armed. This must precede all experiment work on every platform.
    if os.getppid() != parent_pid:
        raise SystemExit("experiment supervisor disappeared before child startup")


def main() -> int:
    parent_pid = int(sys.argv[1])
    _bind_parent(parent_pid)
    from rfx.experiments.worker import _execute_child, build_parser

    args = build_parser().parse_args(sys.argv[2:])
    return _execute_child(
        database=args.database.expanduser().resolve(),
        workspace=args.workspace.expanduser().resolve(),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    sys.exit(main())

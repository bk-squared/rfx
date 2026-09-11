"""Fresh execution bootstrap; run by file path before importing rfx/JAX.

The supervisor owns this process and its deadline. On Linux, parent death
also kills it in the kernel, including while native code holds the GIL.
Other platforms retain normal supervisor cleanup but lack this Linux guard.
"""

from __future__ import annotations

import os
import signal
import sys


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

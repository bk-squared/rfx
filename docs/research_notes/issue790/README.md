# Enforcing experiment deadlines outside native computation

2026-09-11. Baseline main: `9369e0aef78888c40e766be6ecf0f63beb7bb404`.
Issues #790 and #978 describe the same missing execution-timeout outcome.

## Evidence and its limits

The original PR #983 CI run 34626458847, job 103352640803, failed
`test_worker_timeout_is_durable_failed_outcome`: the one-second experiment
did not reach a terminal state within the test's unchanged 60-second wait.
Its last durable event was `progress`; stderr ended at `simulation.run()`.
The complete retained job log is `pr983-original-ci-timeout.log.gz`.
The disclosed single retry on the identical commit passed; it did not fix
or close this defect. These worker files were unchanged by PR #983.

The timestamp of a final pytest failure summary is not the exact instant
the worker stopped or timed out. In particular, it does not alone establish
the ten-minute worker interval inferred in the original #978 description.
The observed failure is exceeding the outer 60-second terminal-state wait.

`native-control/` retains a separate mechanism counterexample. A two-second
C computation holds the GIL through `ctypes.PyDLL`; a 0.1-second SIGALRM is
handled only after 2.0004 seconds. An external 0.3-second deadline stops and
reaps the same child in 0.3026 seconds. This is a native-control measurement,
not a new internal JAX stack trace or an electromagnetic validation run.
The Python signal limitation is described in the
[Python documentation](https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers).
To reproduce the retained control, copy that directory to a scratch location,
compile `busy.c` with `cc -O2 -shared -fPIC busy.c -o busy.so`, and run
`python probe.py` there; preserve the committed measurement.

## Runtime contract

The service's worker PID now belongs to the supervisor. One fresh child
executes the existing compile/preflight/solve/export path in the same process
group. The supervisor uses a monotonic deadline from the durable submitted
specification, so editing `spec.json` cannot extend its budget. Child startup
is included. FDTD dispatch, equations and numerical gates are unchanged.

The supervisor's wait loop does not rely on a Python handler in the child.
It checks the deadline and cancellation, kills an expired/cancelled owned
child, and waits for its exit before final persistence or lease release.
Cancellation polling uses a read-only SQLite connection with zero busy wait;
a database lock cannot turn that poll into a 30-second timeout extension.
An expired computation is stopped before durable finalization waits for locks.

The child writes immutable result files and a completion proposal, without
publishing successful artifact rows or a terminal state. The supervisor
accepts the proposal only after observing completion within the execution
budget and reaping the child. Missing, malformed or inconsistent proposals
fail the run, including an exit status of zero without an outcome.

One transaction decides terminal state, publishes successful artifacts and
final progress, and releases the owned CPU lease. Cancellation already
committed before this transaction wins over proposed success or timeout.
A terminal transaction committed first is immutable. Failure and cancellation
publish no primary results. Diagnostic/log artifacts remain available, and
retries preserve the first registered traceback's bytes and hash.

Queued cancellation does not signal a worker during imports. Windows records
the durable request instead of calling TerminateProcess on the supervisor.
On Linux, the child arms a kernel parent-death SIGKILL before importing rfx,
then rechecks its parent ID to close the startup race. The behavior follows
the [Linux parent-death signal contract](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html).

The execution deadline does not promise hard real-time kernel scheduling or
bounded database/filesystem finalization. Unexpected supervisor death is
guarded by the Linux kernel; equivalent abrupt-death protection on macOS or
Windows is not implemented or claimed by this change.

## Verification

The new tests replace computation with a C blocker that actually enters
native work, ignores termination signals and holds the GIL. They exercise the
production supervisor, durable repository and reopened-service control path.
Linux death controls use a separate subreaper so all test descendants are
reaped, including the parent-only and whole-process-group SIGKILL cases.

Additional controls cover database lock contention, CPU lease ownership before
reaping, missing outcomes, interruption during supervision, failed diagnostic
writes, retry hash integrity, atomic result visibility, cancellation races,
terminal idempotence, rollback and legacy v1 runs without a revision link.

The final packet passed 28 supervisor/transaction controls and 38 selected
Studio/CLI/repository/replay/lifecycle checks. This includes normal v1/v2 CPU
results, digest failure and the unchanged one-second worker timeout. Mypy
passed all 26 source files in the Studio quality-gate scope; Ruff also passed.
Commands, versions and file hashes are recorded in `manifest.json`. Earlier
8-control and 11-lifecycle logs are retained separately from final checks.

`traceback-overwrite-control.py` explicitly removes the first-write guard in
memory. The new retry test then fails because two artifact hashes refer to the
same overwritten path. This is a mutation of the new implementation, not a
claimed baseline checkout. Source files remain unchanged by the experiment.

The selected Studio run emitted an existing Starlette/httpx deprecation
warning and, at interpreter exit, a Popen ResourceWarning for PID 722917.
Read-only inspection mapped that PID to the API journey's successful run
`6552af42-6d33-49a6-85e8-c1b7dcec6f75`; its CPU lease was absent and the PID
was already absent. The API's unchanged refresh path can report durable
success before its Popen is polled again. CPython warns about an unset
returncode before polling in its destructor. This observation is not evidence
that computation survived supervision; the new success path reaps its
execution child before publishing success. No process was killed in this
inspection and no API lifetime change is included.

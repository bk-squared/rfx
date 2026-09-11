# Windows virtual-environment launch follow-up

The first PR #984 head, `fc418584ae95f4fd2ced5d50a11894a67c9a5be1`, passed
local controls but failed the Windows clean-wheel run: CI run 34634812244,
job 103380009017. The golden run moved from queued to failed with
`executor exited with status 1 without a complete outcome`. The original
smoke script did not include its temporary worker stderr in the failure
report. The full available CI log is retained as
`ci-fc418584-windows-failed.log.gz`; its early-exit symptom does not supply
the missing child traceback.

CPython's Windows venv `python.exe` is a redirector that creates an additional
process. A Popen handle to that redirector has a different PID from the
Python interpreter, and the latter's parent is the redirector. This violates
the new direct parent-ID check. That is a source-supported explanation of
the CI symptom; the corrected Windows run remains the platform verification.
The redirector also uses a kill-on-close Job Object, so this evidence does
not establish that killing it necessarily leaves native computation alive.

The fix follows [CPython's multiprocessing implementation](https://github.com/python/cpython/blob/v3.12.12/Lib/multiprocessing/popen_spawn_win32.py#L55):
launch `sys._base_executable` and preserve the virtual environment through
`__PYVENV_LAUNCHER__=sys.executable` in a copied environment. Both launch
boundaries apply this rule: service to supervisor and supervisor to executor.
CPython consumes the marker during startup, so setting it at only the first
boundary would not cover the second. The existing CPU environment and
parent-ID guard remain in place.

The packaged smoke now verifies that the owned handle's PID equals the
child-reported interpreter PID, that its actual parent matches the supervisor,
that its environment prefix is preserved, and that the existing deadline
loop terminates and waits for it. Windows uses a GIL-held native Sleep call
for this control. The separate startup allowance belongs to this diagnostic;
it does not alter any experiment timeout. Failed golden runs now include
bounded worker stderr tails before their temporary workspace disappears.

Local checks pass the Windows launch-selection regression and the existing
native/transaction controls (32 tests). A newly built wheel installed into a
separate Linux venv also passes the actual PID/prefix/deadline control. That
local venv reuses system dependencies; it is not a substitute for the three
clean CI installation environments. Current source hashes and the final
regression outcome are recorded in `launcher-followup.json`.

The original `manifest.json` and its evidence remain the frozen validation
packet for the first implementation. This follow-up addresses the platform
regression in that implementation; it adds no Studio product feature or
broader platform-hardening guarantee.

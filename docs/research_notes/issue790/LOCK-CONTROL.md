# Deterministic SQLite contention control

PR #984 head `7ce947bc0b08e4525167afbe4feb978c20654973` passed 20 of its
21 checks. The only failure was Python 3.11 shard 2, job 103386174797 in run
34636685878: 1 failed, 1712 passed, 4 skipped and 17 xfailed. It failed at
the test fixture's `PRAGMA journal_mode=DELETE`, before starting the native
child or exercising the deadline. Its full log is retained here.

The fixture had initialized an ordinary repository in WAL mode, then tried
to switch that live database to a rollback journal. Another open WAL
connection prevents this conversion, even without an active transaction.
Repository context-manager exits commit/roll back transactions but do not
explicitly close connections; depending on garbage collection for a mode
switch made this test's setup fragile. A separate local control retaining
one WAL connection reproduces `database is locked` at that switch.

The revised test creates a fresh rollback-journal database containing the
cancellation-read surface. It acquires an exclusive lock and separately
proves that another connection cannot read the schema. Only then does it
start the native blocker and exercise the production cancellation poll and
deadline loop. The full application repository, terminal transaction and
lease semantics remain covered by the other controls.

All 32 launcher/process/transaction controls pass. An explicit in-memory
mutation changing the poll's SQLite busy wait from zero to six seconds makes
the revised test fail: computation stops after 6.008 seconds instead of its
one-second budget. The test's existing five-second observation bound is
unchanged. Source files are unchanged by this mutation.

The corrected Windows installation job on the preceding head also passed:
job 103386174171 in run 34636685804 reports actual PID ownership, virtual
environment preservation, enforced deadline and a successful golden CPU run.
That full log is retained alongside the earlier Windows failure. This
follow-up changes the contention test and evidence only.

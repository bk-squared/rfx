"""tests/_durations_log.py -- every test phase's duration, written as it ends.

pytest-split stores --store-durations only at session end, so a weekly shard whose
pytest aborts stores nothing: twice the shard-5 prices had to be approximated from
log timestamps (runs 39220415396 and 39378910883). With RFX_DURATIONS_LOG=<path>
each phase is appended as one JSON line [nodeid, when, seconds];
scripts/ci/weekly_shard_budget.py --log rebuilds pytest-split's map from it when
the session did not reach its end.

The path is made absolute at configure time (the repo root), because tests that
change directory (monkeypatch.chdir) report while their directory is still changed.
A failed write is reported once on stderr (not a warning, which a warnings-as-errors
filter would turn into a session error) and never stops the session. pytest-xdist workers do not
log: the controller sees every relayed report once.
"""

import json
import os
import sys


class DurationsLog:
    def __init__(self, path: str) -> None:
        self._path = path
        self._failed = False

    def pytest_runtest_logreport(self, report) -> None:
        if self._failed:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps([report.nodeid, report.when, report.duration]) + "\n")
        except OSError as exc:
            self._failed = True
            print(f"RFX_DURATIONS_LOG: cannot write {self._path} ({exc}); "
                  "per-test durations stop here", file=sys.stderr)


def pytest_configure(config) -> None:
    path = os.environ.get("RFX_DURATIONS_LOG", "").strip()
    if path and not hasattr(config, "workerinput"):
        config.pluginmanager.register(DurationsLog(os.path.abspath(path)), "rfx-durations-log")

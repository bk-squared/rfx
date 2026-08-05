"""Tests for the weekly-lane RSS high-water reporter in root conftest.py.

Covers the properties issue #545 step 2 requires:
  - the reporter is opt-in via RFX_WEEKLY_RSS and registers nothing (zero
    hook-call overhead) when that env var is absent,
  - when enabled, it registers a real reporter that flags a nodeid whose
    RSS high-water mark grows past the threshold, reporting both the delta
    and the absolute peak, and
  - a malformed RFX_WEEKLY_RSS_THRESHOLD_MB never crashes collection --
    whether the reporter is on (falls back to the 500 MB default) or off
    (the value is never even parsed).

conftest.py sits at the repo root and pytest's default "prepend" import mode
has already loaded it as the top-level module ``conftest`` by the time this
test module is imported, so a plain ``import conftest`` reuses that same
module object rather than re-executing the file.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import conftest


def test_rss_env_enabled_reads_the_env_var(monkeypatch):
    monkeypatch.delenv("RFX_WEEKLY_RSS", raising=False)
    assert conftest._rss_env_enabled() is False

    for off_value in ("0", "false", "False", ""):
        monkeypatch.setenv("RFX_WEEKLY_RSS", off_value)
        assert conftest._rss_env_enabled() is False

    for on_value in ("1", "true", "yes", "anything"):
        monkeypatch.setenv("RFX_WEEKLY_RSS", on_value)
        assert conftest._rss_env_enabled() is True


def test_disabled_by_default_registers_nothing(monkeypatch):
    """Zero-cost when absent: no plugin registration happens at all."""
    monkeypatch.delenv("RFX_WEEKLY_RSS", raising=False)
    fake_config = MagicMock()

    conftest._maybe_register_rss_reporter(fake_config)

    fake_config.pluginmanager.register.assert_not_called()


def test_enabled_registers_the_reporter_plugin(monkeypatch):
    monkeypatch.setenv("RFX_WEEKLY_RSS", "1")
    monkeypatch.delenv("RFX_WEEKLY_RSS_THRESHOLD_MB", raising=False)
    fake_config = MagicMock()

    conftest._maybe_register_rss_reporter(fake_config)

    fake_config.pluginmanager.register.assert_called_once()
    call_args = fake_config.pluginmanager.register.call_args
    registered_plugin = call_args.args[0]
    assert isinstance(registered_plugin, conftest._RSSHighWaterReporter)
    assert registered_plugin.threshold_kb == 500.0 * 1024.0


def test_malformed_threshold_falls_back_to_default(monkeypatch):
    """Reporter ON + a garbage threshold value: registers anyway, using the
    500 MB default instead of raising (F2 fix, reporter-on side)."""
    monkeypatch.setenv("RFX_WEEKLY_RSS", "1")
    monkeypatch.setenv("RFX_WEEKLY_RSS_THRESHOLD_MB", "not-a-number")
    fake_config = MagicMock()

    conftest._maybe_register_rss_reporter(fake_config)

    fake_config.pluginmanager.register.assert_called_once()
    registered_plugin = fake_config.pluginmanager.register.call_args.args[0]
    assert registered_plugin.threshold_kb == 500.0 * 1024.0


def test_malformed_threshold_with_reporter_off_does_not_raise(monkeypatch):
    """Reporter OFF + a garbage threshold value: the value is never even
    parsed (the enable-gate check returns first), so this must not raise
    either -- the F2 bug was RFX_WEEKLY_RSS_THRESHOLD_MB being parsed at
    conftest.py MODULE scope, outside any gate, so it broke every pytest
    invocation regardless of whether RFX_WEEKLY_RSS was set."""
    monkeypatch.delenv("RFX_WEEKLY_RSS", raising=False)
    monkeypatch.setenv("RFX_WEEKLY_RSS_THRESHOLD_MB", "not-a-number")
    fake_config = MagicMock()

    conftest._maybe_register_rss_reporter(fake_config)

    fake_config.pluginmanager.register.assert_not_called()


def test_malformed_threshold_env_var_does_not_break_collection():
    """End-to-end regression test for F2: a fresh pytest process (so
    conftest.py is actually re-imported from scratch, reproducing the real
    bug) with a malformed RFX_WEEKLY_RSS_THRESHOLD_MB and RFX_WEEKLY_RSS
    unset must collect successfully -- before the fix, module-scope
    float(os.environ.get(...)) raised ValueError on import and reddened
    every pytest run in the repo, not just weekly-lane ones."""
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.pop("RFX_WEEKLY_RSS", None)
    env["RFX_WEEKLY_RSS_THRESHOLD_MB"] = "not-a-number"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_weekly_rss_reporter.py"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"collection failed with malformed RFX_WEEKLY_RSS_THRESHOLD_MB "
        f"(RFX_WEEKLY_RSS unset):\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_reporter_flags_nodeid_when_rss_high_water_crosses_threshold():
    """With a 0 MB threshold, any nonnegative delta is flagged and lands in
    the top-10 summary — proves the reporting path fires end to end without
    needing to actually allocate 500 MB in a unit test."""
    reporter = conftest._RSSHighWaterReporter(threshold_mb=0.0)
    fake_item = MagicMock()
    fake_item.nodeid = "tests/test_fake.py::test_thing"

    gen = reporter.pytest_runtest_protocol(fake_item, nextitem=None)
    next(gen)  # run up to the yield (records start_kb)
    try:
        next(gen)  # resume after yield (records end_kb, may report)
    except StopIteration:
        pass

    assert reporter.samples, "expected at least one sample at a 0 MB threshold"
    nodeid, delta_mb, peak_mb = reporter.samples[0]
    assert nodeid == "tests/test_fake.py::test_thing"
    assert delta_mb >= 0.0
    assert peak_mb > 0.0

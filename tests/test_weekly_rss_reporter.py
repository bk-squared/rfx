"""Tests for the weekly-lane RSS high-water reporter in root conftest.py.

Covers the two properties issue #545 step 2 requires:
  - the reporter is opt-in via RFX_WEEKLY_RSS and registers nothing (zero
    hook-call overhead) when that env var is absent, and
  - when enabled, it registers a real reporter that flags a nodeid whose
    RSS high-water mark grows past the threshold.

conftest.py sits at the repo root and pytest's default "prepend" import mode
has already loaded it as the top-level module ``conftest`` by the time this
test module is imported, so a plain ``import conftest`` reuses that same
module object rather than re-executing the file.
"""

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
    fake_config = MagicMock()

    conftest._maybe_register_rss_reporter(fake_config)

    fake_config.pluginmanager.register.assert_called_once()
    call_args = fake_config.pluginmanager.register.call_args
    registered_plugin = call_args.args[0]
    assert isinstance(registered_plugin, conftest._RSSHighWaterReporter)


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
    nodeid, delta_mb = reporter.samples[0]
    assert nodeid == "tests/test_fake.py::test_thing"
    assert delta_mb >= 0.0

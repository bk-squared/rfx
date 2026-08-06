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

And issue #545 step 3 (the live sampler, closing the discriminator gap the
2026-08-06 run 31103127491 writeup disclosed: ru_maxrss only updates in the
post-test hook, so the RSS at the exact SIGKILL instant is never captured):
  - _read_proc_vmhwm_mb reads a real number on this (Linux) box and returns
    None -- never raises -- when /proc is unavailable, and agrees with an
    INDEPENDENT measurement (resource.getrusage's ru_maxrss) to within a
    generous band, so a kB/MB unit-conversion mutation cannot silently pass,
  - _RSSLiveSampler prints the documented "[rss-live] VmHWM N MB [during
    nodeid]" line only once VmHWM has advanced past the threshold, using
    synthetic readings (no real GB allocation) for the pure line-format unit
    tests,
  - the sampler's thread is an actual daemon thread that starts and stops
    cleanly,
  - _RSSHighWaterReporter.pytest_sessionstart/pytest_sessionfinish start and
    stop that thread, gated the same way the reporter itself is gated, and
  - THE property PR #592 review demanded: a background thread's bare
    ``print()`` is invisible under pytest's own capture unless ``-s`` is
    passed (measured: 0 [rss-live] lines at -v -ra with default fd-capture,
    including under a `timeout -s KILL` rehearsal of the exact scenario, and
    even with --capture=tee-sys; 3 lines with -s). The unit tests above bind
    the STRING the sampler builds (via capsys, calling _sample_once
    directly) -- they do NOT prove that string is ever DELIVERED anywhere a
    human or CI log could see it. Only a real subprocess pytest run, killed
    with SIGKILL and inspected from OUTSIDE pytest's own capture layer
    (subprocess.run's stdout pipe), proves delivery. See
    test_live_sampler_line_survives_sigkill_with_dash_s and its
    without-`-s` regression-pin sibling below.

conftest.py sits at the repo root and pytest's default "prepend" import mode
has already loaded it as the top-level module ``conftest`` by the time this
test module is imported, so a plain ``import conftest`` reuses that same
module object rather than re-executing the file.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


# ---------------------------------------------------------------------------
# issue #545 step 3: the live sampler (background thread).


def test_read_proc_vmhwm_mb_returns_a_positive_number_on_linux():
    """This test process itself has a nonzero VmHWM by the time it runs."""
    mb = conftest._read_proc_vmhwm_mb()
    assert mb is not None, "/proc/self/status should be readable on Linux CI"
    assert mb > 0.0


def test_read_proc_vmhwm_mb_returns_none_when_proc_is_unreadable(monkeypatch):
    """Non-Linux / unreadable /proc guard: OSError is swallowed, not raised."""

    def _raise_oserror(*args, **kwargs):
        raise OSError("no such file or directory")

    monkeypatch.setattr("builtins.open", _raise_oserror)
    assert conftest._read_proc_vmhwm_mb() is None


def test_read_proc_vmhwm_mb_agrees_with_independent_ru_maxrss_band():
    """Cross-check against an INDEPENDENT measurement (resource.getrusage's
    ru_maxrss, KB on Linux -- the same call _RSSHighWaterReporter._maxrss_kb
    uses) so a unit-conversion mutation (e.g. dropping the /1024.0, or
    reading the wrong /proc/self/status field) cannot silently pass: VmHWM
    and ru_maxrss are two different kernel-accounting paths to the same
    process's peak RSS, not byte-identical, but they must never be off by
    an order of magnitude."""
    vmhwm_mb = conftest._read_proc_vmhwm_mb()
    assert vmhwm_mb is not None
    ru_maxrss_mb = conftest._RSSHighWaterReporter._maxrss_kb() / 1024.0
    assert ru_maxrss_mb > 0.0

    ratio = vmhwm_mb / ru_maxrss_mb
    assert 0.5 < ratio < 2.0, (
        f"VmHWM={vmhwm_mb:.1f} MB vs ru_maxrss={ru_maxrss_mb:.1f} MB, "
        f"ratio={ratio:.2f} outside the expected band -- possible unit-conversion bug"
    )


def test_live_sampler_line_format_with_nodeid(monkeypatch, capsys):
    """Synthetic reading (no real allocation): a delta past the threshold
    prints the documented "[rss-live] VmHWM N MB during <nodeid>" line."""
    monkeypatch.setattr(conftest, "_read_proc_vmhwm_mb", lambda: 1500.0)
    sampler = conftest._RSSLiveSampler(threshold_mb=500.0)
    sampler.current_nodeid = "tests/test_fake.py::test_thing"

    sampler._sample_once()

    out = capsys.readouterr().out
    assert out.strip() == "[rss-live] VmHWM 1500 MB during tests/test_fake.py::test_thing"


def test_live_sampler_line_format_without_nodeid(monkeypatch, capsys):
    """When no nodeid is cheaply known yet, the 'during ...' suffix is
    omitted entirely rather than printed as 'during None'."""
    monkeypatch.setattr(conftest, "_read_proc_vmhwm_mb", lambda: 1500.0)
    sampler = conftest._RSSLiveSampler(threshold_mb=500.0)
    assert sampler.current_nodeid is None

    sampler._sample_once()

    out = capsys.readouterr().out
    assert out.strip() == "[rss-live] VmHWM 1500 MB"


def test_live_sampler_skips_below_threshold_and_when_unavailable(monkeypatch, capsys):
    """No line at all when the delta is below threshold, or when the VmHWM
    reading is unavailable (None)."""
    sampler = conftest._RSSLiveSampler(threshold_mb=500.0)

    monkeypatch.setattr(conftest, "_read_proc_vmhwm_mb", lambda: 100.0)  # +100 MB < 500 MB
    sampler._sample_once()
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(conftest, "_read_proc_vmhwm_mb", lambda: None)
    sampler._sample_once()
    assert capsys.readouterr().out == ""


def test_live_sampler_thread_starts_is_daemon_and_stops_cleanly(monkeypatch):
    """The thread this sampler owns is a real daemon thread (never blocks
    process exit) that actually runs on its own clock and stops on demand."""
    readings = iter([100.0, 100.0, 100.0, 700.0])  # +600 MB on the 4th sample
    monkeypatch.setattr(conftest, "_read_proc_vmhwm_mb", lambda: next(readings, 700.0))
    sampler = conftest._RSSLiveSampler(threshold_mb=500.0, interval_s=0.02)
    sampler.current_nodeid = "tests/test_fake.py::test_thing"

    assert sampler.is_alive() is False
    sampler.start()
    try:
        assert sampler._thread.daemon is True
        deadline = time.time() + 2.0
        while time.time() < deadline and sampler._last_reported_mb == 0.0:
            time.sleep(0.02)
        assert sampler._last_reported_mb == 700.0, "sampler never observed the synthetic jump"
        assert sampler.is_alive() is True
    finally:
        sampler.stop(timeout=2.0)
    assert sampler.is_alive() is False


def test_reporter_sessionstart_starts_and_sessionfinish_stops_live_sampler(monkeypatch):
    """_RSSHighWaterReporter wires its live_sampler's lifecycle to the real
    pytest session, so a killed process is sampling for as long as the
    session itself runs."""
    monkeypatch.setattr(conftest, "_read_proc_vmhwm_mb", lambda: 100.0)
    reporter = conftest._RSSHighWaterReporter(threshold_mb=500.0)
    reporter.live_sampler.interval_s = 0.02

    assert reporter.live_sampler.is_alive() is False
    reporter.pytest_sessionstart(session=MagicMock())
    try:
        assert reporter.live_sampler.is_alive() is True
    finally:
        reporter.pytest_sessionfinish(session=MagicMock(), exitstatus=0)
    assert reporter.live_sampler.is_alive() is False


def test_reporter_updates_live_sampler_current_nodeid_during_test():
    """The reporter's own hookwrapper feeds the live sampler its "cheaply
    knowable" current nodeid before the test body runs, so a mid-test kill
    still has a nodeid to attribute the next live-sampler line to."""
    reporter = conftest._RSSHighWaterReporter(threshold_mb=1e9)  # never fires the post-test print
    fake_item = MagicMock()
    fake_item.nodeid = "tests/test_fake.py::test_other_thing"

    gen = reporter.pytest_runtest_protocol(fake_item, nextitem=None)
    next(gen)  # run up to the yield
    assert reporter.live_sampler.current_nodeid == "tests/test_fake.py::test_other_thing"
    try:
        next(gen)
    except StopIteration:
        pass


# ---------------------------------------------------------------------------
# THE property (PR #592 review, CRITICAL): does an [rss-live] line actually
# survive a real SIGKILL, observed from OUTSIDE pytest's own capture layer?
# The tests above bind the sampler's output STRING via capsys; none of them
# prove DELIVERY through pytest's real stdout capture machinery. Only a
# subprocess pytest run, killed for real and inspected via subprocess.run's
# own stdout pipe (a wholly separate capture layer from pytest's internal
# one), proves that.


def test_rss_live_sampler_helper_allocates_and_self_sigkills():
    """NOT a standalone test -- a subprocess helper for
    test_live_sampler_line_survives_sigkill_with_dash_s and
    test_live_sampler_line_lost_without_dash_s below. Skips immediately
    unless RFX_TEST_SIGKILL_HELPER=1 is set (only those two driver tests set
    it, in a CHILD subprocess), so this function is inert in every normal
    run of this file, including this file's own collection in the
    fast/weekly lanes.

    Allocates ~60 MB (comfortably above the low RFX_WEEKLY_RSS_THRESHOLD_MB
    the driver sets, and nowhere near "GB" scale), sleeps long enough for
    the live sampler's REAL ~2s poll cadence (this deliberately does not
    shorten interval_s -- it exercises the exact production default) to
    observe the jump and print at least one line, then SIGKILLs its own
    process -- reproducing the OOM-style hard kill mid-test that issue #545
    is about, without needing an actual multi-GB allocation.
    """
    if os.environ.get("RFX_TEST_SIGKILL_HELPER") != "1":
        pytest.skip("subprocess-only helper; set RFX_TEST_SIGKILL_HELPER=1 to run it")
    buf = bytearray(60 * 1024 * 1024)  # 60 MB
    for i in range(0, len(buf), 4096):
        buf[i] = 1  # touch every page so it's actually resident, not just reserved
    time.sleep(3.0)  # >= one full real _RSSLiveSampler interval_s (2.0s default), with margin
    os.kill(os.getpid(), signal.SIGKILL)


def _run_sigkill_helper_subprocess(extra_pytest_args):
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)  # an ambient -s here would leak into the child and invert the without--s regression pin
    env["RFX_TEST_SIGKILL_HELPER"] = "1"
    env["RFX_WEEKLY_RSS"] = "1"
    env["RFX_WEEKLY_RSS_THRESHOLD_MB"] = "10"  # low so the 60 MB helper alloc trips it
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", *extra_pytest_args, "-v",
            "tests/test_weekly_rss_reporter.py::test_rss_live_sampler_helper_allocates_and_self_sigkills",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_live_sampler_line_survives_sigkill_with_dash_s():
    """THE property (PR #592 review, CRITICAL fix): with -s (--capture=no),
    the live sampler's background-thread prints reach the real stdout fd, so
    flush=True actually lands the line before a SIGKILL. This is what
    validation.yml's -s flag (added in this fix) is load-bearing for --
    without it, this exact scenario prints nothing (see the sibling
    regression-pin test below)."""
    result = _run_sigkill_helper_subprocess(extra_pytest_args=["-s"])

    assert result.returncode == -signal.SIGKILL, (
        f"expected the child to die from SIGKILL, got returncode={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "[rss-live] VmHWM" in result.stdout, (
        "expected at least one [rss-live] line to survive the SIGKILL under -s\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_live_sampler_line_lost_without_dash_s():
    """Regression pin for WHY -s is required in validation.yml: under
    pytest's DEFAULT capture (no -s), the identical scenario prints nothing
    to the outer process's stdout -- pytest's fd-level capture redirects the
    real stdout fd to its own internal per-test buffer, which is lost (not
    replayed) on a SIGKILL mid-test. If a future edit drops -s from the
    workflow, THIS is the silent loss it would reintroduce; if this
    assertion ever starts failing (i.e. the line DOES show up here), that
    means pytest's capture behavior changed and validation.yml's -s flag
    may no longer be load-bearing -- re-verify before removing it."""
    result = _run_sigkill_helper_subprocess(extra_pytest_args=[])

    assert result.returncode == -signal.SIGKILL, (
        f"expected the child to die from SIGKILL, got returncode={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "[rss-live] VmHWM" not in result.stdout, (
        "expected NO [rss-live] line without -s (pytest's default capture "
        "should swallow it)\n"
        f"stdout={result.stdout}"
    )

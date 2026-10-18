"""Device-limit queries and advisory propagation for #1828, using fake devices."""

from types import SimpleNamespace

import jax
import pytest

from tests.unit.autodiff.test_estimate_ad_memory import _uniform_sim


@pytest.mark.parametrize("platform", ["gpu", "cuda", "rocm", "mps"])
def test_device_limit_is_used(monkeypatch, platform):
    calls = []
    def stats():
        calls.append(True)
        return {"bytes_limit": 8e9}
    monkeypatch.setattr(jax, "local_devices", lambda: [
        SimpleNamespace(platform=platform, memory_stats=stats)])
    estimate = _uniform_sim().estimate_ad_memory(4)
    assert estimate.available_gb == 8
    assert calls == [True]
    assert "cannot be read" not in (estimate.warning or "")


@pytest.mark.parametrize("stats", [None, {}, {"bytes_in_use": 3}, {"bytes_limit": None},
    {"bytes_limit": 0}, {"bytes_limit": -1}, {"bytes_limit": float("nan")},
    {"bytes_limit": float("inf")}, {"bytes_limit": "bad"}, {"bytes_limit": True},
    RuntimeError("unavailable"), "missing-method"])
def test_mps_unknown_limit_has_one_advisory(monkeypatch, stats):
    def memory_stats():
        if isinstance(stats, Exception):
            raise stats
        return stats
    device = SimpleNamespace(platform="mps")
    if stats != "missing-method":
        device.memory_stats = memory_stats
    monkeypatch.setattr(jax, "local_devices", lambda: [device, device])
    sim = _uniform_sim()
    estimate = sim.estimate_ad_memory(4)
    assert estimate.available_gb is None
    assert estimate.warning.count("device memory limit cannot be read on this backend") == 1
    report = sim.explain_ad_memory(4)
    assert sum("cannot be read" in text for text in report.recommendations) == 1


@pytest.mark.parametrize("platform", ["cpu", "tpu-unknown"])
def test_no_device_budget_query_or_new_advisory(monkeypatch, platform):
    def forbidden():
        pytest.fail("backend without memory stats queried the device budget")
    monkeypatch.setattr(jax, "local_devices", lambda: [
        SimpleNamespace(platform=platform, memory_stats=forbidden)])
    estimate = _uniform_sim().estimate_ad_memory(4)
    assert estimate.available_gb is None
    assert "cannot be read" not in (estimate.warning or "")


def test_explicit_budget_skips_device_query(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "mps")
    def forbidden():
        pytest.fail("explicit budget queried devices")
    monkeypatch.setattr(jax, "local_devices", forbidden)
    estimate = _uniform_sim().estimate_ad_memory(4, available_memory_gb=6)
    assert estimate.available_gb == 6
    assert "cannot be read" not in (estimate.warning or "")


def test_mps_device_query_failure_is_advised(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "mps")
    def unavailable():
        raise RuntimeError("devices unavailable")
    monkeypatch.setattr(jax, "local_devices", unavailable)
    estimate = _uniform_sim().estimate_ad_memory(4)
    assert estimate.available_gb is None
    assert estimate.warning.count("cannot be read") == 1


def test_backend_query_failure_remains_best_effort(monkeypatch):
    from rfx.api import _ad_memory

    def unavailable():
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(_ad_memory, "backend_class", unavailable)
    estimate = _uniform_sim().estimate_ad_memory(4)
    assert estimate.available_gb is None
    assert "cannot be read" not in (estimate.warning or "")

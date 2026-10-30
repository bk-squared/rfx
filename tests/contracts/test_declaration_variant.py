"""Private drive declarations preserve the caller during runs and failures."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from rfx import Simulation
from rfx._declarations import (
    DECLARATIONS, NOT_DECLARATIONS, declaration_key, declaration_variant,
    release_declaration_caches,
)
from tests.contracts.test_declaration_registry import model
from tests.locks import test_sparams_split_bit_identity as locks


@pytest.mark.parametrize("attr,form", [(a, f) for a, f in DECLARATIONS
                                      if f in {"entries", "mapping"}])
def test_variant_in_place_append_is_private(attr, form):
    sim = model()
    original = getattr(sim, attr)
    token = object()
    if form == "mapping":
        original["shared_entry"] = token
    else:
        original.append(token)
    before = dict(original) if form == "mapping" else list(original)
    variant = declaration_variant(sim)
    private = getattr(variant, attr)
    assert private is not original
    if form == "mapping":
        private["variant_only"] = object()
    else:
        private.append(object())
    assert original == before
    for a, f in DECLARATIONS:
        if f == "entries":
            assert all(x is y for x, y in zip(getattr(sim, a), getattr(variant, a)))


def test_variant_replacements_and_cache_release():
    sim = model()
    cache_names = [a for a, reason in NOT_DECLARATIONS if reason == "cache"]
    tokens = {a: object() for a in cache_names}
    for a, value in tokens.items():
        setattr(sim, a, value)
    ports = []
    variant = declaration_variant(sim, _ports=ports)
    assert variant is not sim and variant._ports is ports
    for a, value in tokens.items():
        assert getattr(variant, a) is None
        assert getattr(sim, a) is value
    release_declaration_caches(sim)
    assert all(getattr(sim, a) is None for a in cache_names)
    assert "_msl_default_memo" in cache_names
    with pytest.raises(TypeError, match="DECLARATIONS.*_typo"):
        declaration_variant(sim, _typo=[])


@pytest.mark.parametrize("rcs", [False, True])
def test_second_preflight_reuses_context(rcs):
    sim = model()
    sim.preflight()
    ctx = sim._campaign_ctx()
    key = declaration_key(sim)
    if rcs:
        sim.boundary_model(rcs=True)
        # The derived RCS view must reuse the context even before preflight
        # refreshes it back to the ordinary requirements.
        assert declaration_key(sim) == key
        assert sim._campaign_ctx() is ctx
    sim.preflight()
    assert sim._campaign_ctx() is ctx
    assert declaration_key(sim) == key


def _calculator_case(monkeypatch, family):
    if family == "waveguide":
        from tests.unit.sparams.test_waveguide_nu_sparam import _make_wr90_nu_sim
        return _make_wr90_nu_sim(solved_domain=True), {
            "normalize": True, "n_steps": 120, "skip_preflight": True}
    method = f"compute_{family}_s_matrix"
    # Reuse the lock's geometry, capturing its builder before it starts a solve.
    with monkeypatch.context() as patch:
        patch.setattr(Simulation, method, lambda self, **kwargs: (self, kwargs))
        return getattr(locks, f"_{family}_result")()


_SURFACES = ("_ports", "_msl_ports", "_probes", "_dft_planes",
             "_waveguide_ports", "_flux_monitors", "_internal_probe_indices")


def _snapshot(sim):
    return {a: (getattr(sim, a), tuple(getattr(sim, a))) for a in _SURFACES}


def _assert_original(sim, before):
    for a, (container, entries) in before.items():
        assert getattr(sim, a) is container, a
        assert tuple(getattr(sim, a)) == entries, a


class DriveFailure(Exception):
    pass


@pytest.mark.parametrize("family", ["msl", "mixed", "waveguide"])
@pytest.mark.parametrize("during_run", [True, False], ids=["caller_during_run", "exception"])
def test_calculator_preserves_caller(monkeypatch, family, during_run):
    sim, kwargs = _calculator_case(monkeypatch, family)
    before = _snapshot(sim)
    reached = []

    def drive(variant, *args, **kw):
        reached.append(variant)
        if during_run:
            _assert_original(sim, before)
            assert variant is not sim
        raise DriveFailure("injected drive failure")

    if family == "waveguide":
        monkeypatch.setattr("rfx.runners.nonuniform.run_nonuniform_path", drive)
    else:
        monkeypatch.setattr(Simulation, "run" if family == "msl" else "_forward_from_materials", drive)
    with pytest.raises(DriveFailure, match="injected drive failure"):
        getattr(sim, f"compute_{family}_s_matrix")(**kwargs)
    assert len(reached) == 1
    _assert_original(sim, before)


def test_two_waveguide_calls_share_no_drive_state(monkeypatch):
    sim, kwargs = _calculator_case(monkeypatch, "waveguide")
    before = _snapshot(sim)
    single = sim.compute_waveguide_s_matrix(**kwargs)
    from rfx.runners.nonuniform import run_nonuniform_path
    barrier = Barrier(2)

    def overlap(variant, **kw):
        barrier.wait(timeout=60)
        _assert_original(sim, before)
        return run_nonuniform_path(variant, **kw)

    monkeypatch.setattr("rfx.runners.nonuniform.run_nonuniform_path", overlap)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(sim.compute_waveguide_s_matrix, **kwargs) for _ in range(2)]
        results = [future.result(timeout=120) for future in futures]
    for result in results:
        assert np.array_equal(result.s_params, single.s_params)
    _assert_original(sim, before)


def test_release_set_preserves_input_keyed_mesh_resolution():
    sim = model()
    mesh = object()
    sim._mesh_resolution = mesh
    assert {a for a, reason in NOT_DECLARATIONS if reason == "cache"} == {
        "_pf_campaign_ctx", "_realized_geometry_record", "_coax_geometry", "_msl_default_memo",
    }
    assert declaration_variant(sim)._mesh_resolution is mesh
    release_declaration_caches(sim)
    assert sim._mesh_resolution is mesh

"""Bit-identity gate for the slab-family code motion (issue #928).

``docs/agent-memory/development_methodology.md`` section 2.2: a pure code
motion is merged only if a baseline captured BEFORE the edit compares
bit-identically after it. The move here took the cv04 rig block, the gated-band
mask and the ring-down helpers out of ``cv22_dispersive_gates`` -- a module
named after a consumer -- into the leaf ``slab_family``, and re-pointed the
producer at it.

Windows alone would not have been enough (round-2 review): a code motion can
leave the three scalars untouched and still move a mask edge, a derived record
length or a replayed verdict. The baseline therefore carries, and this file
compares:

  * the fourteen rig constants and both cases' derived windows, by exact float
    equality AND by float.hex();
  * the realized rig the runner builds without solving (dt, grid shape);
  * the derived record lengths, the masked bin grid and the gated bin count
    for every arm of both cases;
  * the replayed per-gate verdicts of the committed cv22 and cv23 baseline
    artifacts, plus the gated means and windows those verdicts came from.

Baseline: ``tests/fixtures/slab_family_windows_baseline.json``, captured at the
commit named inside it, before the move. Nothing here is a source for any
number -- it is a frozen copy of derived values, and if it ever disagrees with
the live derivation the live derivation is what changed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "tests/fixtures/slab_family_windows_baseline.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _load("cv22_dispersive_gates", "validation/crossval/comparators/cv22_dispersive_gates.py")
L = _load("cv23_lossy_gates", "validation/crossval/comparators/cv23_lossy_gates.py")


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(_BASELINE.read_text(encoding="utf-8"))


def _rig_dt_and_shape():
    from rfx.grid import Grid
    grid = Grid(freq_max=20e9, domain=(G.NX_INTERIOR * G.DX_M, 0.004, G.DX_M),
                dx=G.DX_M, cpml_layers=G.N_CPML, mode="2d_tmz")
    return float(grid.dt), [int(s) for s in grid.shape]


def test_rig_constants_are_bit_identical(baseline):
    for name, want in baseline["rig"].items():
        got = getattr(G, name)
        assert got == want, (name, got, want)
        if isinstance(want, float):
            assert float(got).hex() == baseline["float_hex"]["rig"][name], name
    # and the producer reads the same declaration the consumer re-exports
    slab_family = _load("slab_family_identity",
                        "validation/crossval/comparators/slab_family.py")
    for name in baseline["rig"]:
        assert getattr(slab_family, name) == baseline["rig"][name], name


@pytest.mark.parametrize("case", ["cv22", "cv23"])
def test_derived_windows_are_bit_identical(baseline, case):
    module = G if case == "cv22" else L
    for name, want in baseline[f"{case}_windows"].items():
        got = getattr(module, name)
        assert got == want, (case, name, got, want)
        if isinstance(want, float):
            assert float(got).hex() == baseline["float_hex"][f"{case}_windows"][name]


def test_adopted_envelope_values_are_bit_identical(baseline):
    for name, want in baseline["cv04_envelope_values"].items():
        assert G.CV04_ENVELOPE[name] == want, name
        assert float(G.CV04_ENVELOPE[name]).hex() == \
            baseline["float_hex"]["cv04_envelope_values"][name]


def test_realized_rig_is_bit_identical(baseline):
    dt, shape = _rig_dt_and_shape()
    assert float(dt).hex() == baseline["rig_realized"]["dt_s_hex"]
    assert shape == baseline["rig_realized"]["grid_shape"]


@pytest.mark.parametrize("case", ["cv22", "cv23"])
def test_records_and_masks_are_bit_identical(baseline, case):
    dt, _ = _rig_dt_and_shape()
    want = baseline["masks_and_records"][case]
    if case == "cv22":
        records = {a: G.derive_record_length(G.ARMS[a]["model"], G.ARMS[a]["params"], dt)
                   for a in G.ARM_ORDER}
    else:
        records = {a: L.derive_record_length(L.ARMS[a]["params"], dt) for a in L.ARM_ORDER}
    for arm, stored in want["records"].items():
        for key, value in stored.items():
            got = records[arm][key]
            assert got == value or float(got) == float(value), (case, arm, key, got, value)
    n_steps = max(r["n_steps"] for r in records.values())
    nfft = int(2 ** np.ceil(np.log2(n_steps)) * G.NFFT_OVERSAMPLE)
    f = np.fft.rfftfreq(nfft, d=dt)
    f = f[(f > G.MASK_F_LO_HZ) & (f < G.MASK_F_HI_HZ)]
    g = G.gated_mask(f)
    assert nfft == want["nfft"]
    assert int(f.size) == want["n_masked_bins"]
    assert int(g.sum()) == want["n_gated_bins"]
    assert float(f[0]) == want["f_masked_first_hz"]
    assert float(f[-1]) == want["f_masked_last_hz"]


@pytest.mark.parametrize("case", ["cv22", "cv23"])
def test_committed_artifacts_replay_to_the_same_verdicts(baseline, case):
    """The strongest arm: the committed baseline artifacts, replayed through
    the moved code, must reproduce every gate and every gated mean."""
    stored = baseline["artifact_verdicts"].get(case)
    if not stored:
        pytest.skip(f"{case} verdicts absent from the baseline")
    results = {"cv22": "_22_dispersive_results", "cv23": "_23_lossy_results"}[case]
    path = _REPO / "validation/crossval" / results / "rfx.json"
    if not path.is_file():
        pytest.skip(f"{case} baseline artifact absent")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for arm, want in stored.items():
        ad = doc["arms"][arm]
        if case == "cv22":
            e2 = G.evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], ad["model"],
                               ad["params"], ad["dt_s"], tail=ad["tail"])
        else:
            e2 = L.evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], ad["params"],
                               ad["dt_s"], tail=ad["tail"])
        assert e2["gates"] == want["gates"], (case, arm)
        assert e2["e2_ok"] == want["e2_ok"], (case, arm)
        for key, value in want.items():
            if key in ("gates", "e2_ok"):
                continue
            got = e2[key]
            if isinstance(value, str):
                assert float(got).hex() == value, (case, arm, key)
            else:
                assert got == value, (case, arm, key)

"""Synthetic counterexamples for the offline coupon review, not RF evidence."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'docs/research_notes/issue726/collocation/review_coupon_records.py'
spec = importlib.util.spec_from_file_location('coupon_review', SCRIPT)
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


def _json(path, value):
    path.write_text(json.dumps(value))


def _records(tmp_path, *, raw_error=0, plane_shift=0):
    base, qualification = tmp_path / 'base', tmp_path / 'qualification'
    (base / 'coupon').mkdir(parents=True)
    source = 'synthetic-contract-only'
    _json(base / 'environment.json', {'source_sha': source})
    (base / 'coupon.log.gz').write_bytes(gzip.compress(
        b"[WI-1 MSL raw qualification] {'failures': []}\n"))
    # Explicitly manufactured numbers. The review gate consumes a producer
    # verdict; this harness tests its repeat/mesh checks, not that verdict.
    freqs = np.linspace(.5e9, 5e9, 10, dtype=np.float32)
    s = np.broadcast_to(np.array([[.1, .8j], [.8j, -.1]])[..., None],
                        (2, 2, 10)).copy()
    base_record = base / 'coupon/call-0-result.npz'
    np.savez(base_record, S=s, freqs=freqs)
    planes = [.005, .009]

    def dump(path, raw_s, coordinates):
        meta = {'current_plane_stencils': [{'voltage_coordinate': x}
                                          for x in coordinates]}
        np.savez(path, production_smatrix=raw_s, metadata_json=json.dumps(meta))

    dump(base / 'coupon/call-0-raw-vi.npz', s, planes)
    for label, refinement in (('confirmation', 1), ('refinement', 2)):
        root = qualification / label
        root.mkdir(parents=True)
        _json(root / 'inputs.json', dict(
            source_sha=source, refinement=refinement, num_periods=12,
            drawing={'height_m': .000254},
            reference_record_sha256=hashlib.sha256(base_record.read_bytes()).hexdigest(),
        ))
        _json(root / 'qualification.json', {'failures': []})
        # A partial result export can omit S_raw while displaying the same
        # S. The raw dump still owns the pre-projection observations.
        np.savez(root / 'result.npz', S=s, freqs=freqs)
        raw = s.copy()
        shifted = planes
        if refinement == 2:
            raw[0, 0, 0] += raw_error
            shifted = [x + plane_shift for x in planes]
        dump(root / 'raw-vi.npz', raw, shifted)
    return base, qualification


def test_offline_gate_accepts_complete_consistent_contract_records(tmp_path):
    result = reviewer.review(*_records(tmp_path))
    assert result['accepted_for_golden_update']
    assert result['repeat_max_abs'] == result['refinement_raw_max_abs'] == 0


def test_displayed_s_cannot_hide_a_raw_complex_mesh_failure(tmp_path):
    result = reviewer.review(*_records(tmp_path, raw_error=.03j))
    assert not result['accepted_for_golden_update']
    assert result['repeat_max_abs'] == 0
    assert result['refinement_raw_max_abs'] == pytest.approx(.03)
    assert any('raw-complex mesh difference' in x for x in result['failures'])


def test_equal_s_cannot_hide_a_shifted_voltage_reference_plane(tmp_path):
    with pytest.raises(AssertionError):
        reviewer.review(*_records(tmp_path, plane_shift=25e-6))

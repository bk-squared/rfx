"""Physical anchor: a user plane costs its accumulator, with no n_steps factor."""

import numpy as np

from tests.unit.sparams.test_msl_internal_probe_advisories import FREQS, _thru


N_STEPS = 1200
USER_PLANE = "user_cross_section"


def test_msl_records_only_witness_planes(monkeypatch):
    sim = _thru()
    sim.add_dft_plane_probe(axis="x", coordinate=0.006, component="ez",
                            freqs=FREQS, name=USER_PLANE)
    captured = []
    original_run = type(sim).run

    def run(self, **kwargs):
        result = original_run(self, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(type(sim), "run", run)
    result = sim.compute_msl_s_matrix(freqs=FREQS, n_steps=N_STEPS)
    assert len(captured) == 2
    for drive, raw in enumerate(captured):
        records = raw.dft_time_records
        assert USER_PLANE not in records
        expected = {
            f"_msl_run{drive}_p{port}_{component}": shape
            for port in range(2)
            for component, shape in (
                [(f"ez{q}", (N_STEPS, 1, 4)) for q in range(5)]
                + [(h + side, (N_STEPS, 8, 2))
                   for h in ("hy", "hz") for side in ("", "_left")]
            )
        }
        assert set(records) == set(expected)
        for name, shape in expected.items():
            assert records[name].shape == shape
        assert USER_PLANE in raw.dft_planes
    assert np.isfinite(result.settling_db).all()
    assert [entry.name for entry in sim._dft_planes] == [USER_PLANE]
    assert "_time_recorded_dft_planes" not in sim.__dict__

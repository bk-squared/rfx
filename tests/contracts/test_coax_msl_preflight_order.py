"""A board's sheet refusal stays at the gate, before kernel assembly."""
import warnings

import pytest

from tests._coax_msl_instrument_fixture import (
    build_instrument_junction,
    instrument_kwargs,
)


def test_coax_msl_sheet_verdict_at_preflight_gate(monkeypatch):
    kwargs = instrument_kwargs(n_steps=1)
    kwargs["skip_preflight"] = False

    def forbidden_kernel(*args, **kwargs):
        pytest.fail("sheet-size refusal must precede coax_kernel_conductors")

    strict = build_instrument_junction()
    strict._snap = "strict"
    with monkeypatch.context() as patch:
        patch.setattr("rfx.sparams.coax.coax_kernel_conductors", forbidden_kernel)
        with pytest.raises(ValueError, match="conductor sheet dimension"):
            strict.compute_coax_msl_transition(**kwargs)

    class ReachedRunner(Exception):
        pass

    def runner(*args, **kwargs):
        raise ReachedRunner

    monkeypatch.setattr("rfx.simulation.run", runner)
    declared = build_instrument_junction()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(ReachedRunner):
            declared.compute_coax_msl_transition(**kwargs)
    sheet_notes = [w for w in emitted if "conductor sheet dimension" in str(w.message)]
    assert len(sheet_notes) == 1

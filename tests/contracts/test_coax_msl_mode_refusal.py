"""A declared MSL mode must not silently become a Laplace feed (#2619)."""
from dataclasses import replace

import pytest

from rfx.errors import RfxUnsupported
from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs
from tests.contracts import path_disposition as dispositions


@pytest.mark.parametrize("mode", ("uniform", "eigenmode", "laplace"))
def test_coax_msl_mode_contract(monkeypatch, mode):
    sim = build_instrument_junction(mode=mode)
    sim._msl_ports[0] = replace(sim._msl_ports[0], name="board_feed")
    cell = dispositions.cell("_msl_ports", "msl_port", "coax_msl_transition", mode=mode)

    class ReachedRunner(Exception):
        pass

    def runner(*args, **kwargs):
        raise ReachedRunner

    def forbidden_kernel(*args, **kwargs):
        pytest.fail("MSL mode refusal must precede coax_kernel_conductors")

    monkeypatch.setattr("rfx.simulation.run", runner)
    if mode == "laplace":
        assert cell.kind == dispositions.CARRIES
        with pytest.raises(ReachedRunner):
            sim.compute_coax_msl_transition(**instrument_kwargs(1))
    else:
        assert cell.kind == dispositions.REFUSES
        monkeypatch.setattr("rfx.sparams.coax.coax_kernel_conductors", forbidden_kernel)
        with pytest.raises(RfxUnsupported) as caught:
            sim.compute_coax_msl_transition(**instrument_kwargs(1))
        assert str(caught.value) == (
            "compute_coax_msl_transition(): MSL port 'board_feed' has "
            f"unsupported mode='{mode}'; use mode=\"laplace\" (the default)."
        )
        assert cell.raises in str(caught.value)

"""Public ADI notice and absorber refusal, shared by both execution entries."""
import warnings

import pytest

from rfx import Simulation
from rfx._adi_notice import ADI_SPONGE_REFUSAL, ADI_WARNING, ExperimentalADIWarning
from rfx.boundaries.spec import Boundary, BoundarySpec


def _sim(mode, **kwargs):
    return Simulation(freq_max=10e9, domain=(.006, .006, .006), dx=.001,
                      mode=mode, **kwargs)


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("boundary", [
    "cpml",
    BoundarySpec(x=Boundary(lo="cpml", hi="pec"), y="pec", z="pec"),
])
def test_absorber_refused_at_construction(mode, boundary):
    with pytest.raises(ValueError) as exc:
        _sim(mode, solver="adi", boundary=boundary)
    assert str(exc.value) == ADI_SPONGE_REFUSAL
    for reason in ("unmatched graded-conductivity sponge", "electric loss only",
                   "no magnetic loss", "-10.4 dB", "10 GHz", "16 layers",
                   "-3 dB", "2 GHz", "-93 dB"):
        assert reason in str(exc.value)


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("per_face", [False, True])
def test_preflight_and_execution_refuse_stale_absorber(mode, entry, per_face):
    sim = _sim(mode, solver="adi", boundary="pec")
    # Construction rejects new models; also guard restored/mutated models.
    if per_face:
        sim._boundary_spec = BoundarySpec(
            x=Boundary(lo="cpml", hi="pec"), y="pec", z="pec")
    else:
        sim._boundary = "cpml"
    issues = [i for i in sim.preflight() if i.code == "adi_absorber_unsupported"]
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert str(issues[0]) == ADI_SPONGE_REFUSAL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalADIWarning)
        with pytest.raises(ValueError, match="unmatched graded-conductivity sponge"):
            getattr(sim, entry)(n_steps=2, **({"skip_preflight": True} if entry == "run" else {}))


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_pec_execution_warns(mode, entry):
    sim = _sim(mode, solver="adi", boundary="pec")
    sim.add_source((.003, .003, .003), "ez", amplitude_kind="field")
    sim.add_probe((.004, .003, .003), "ez")
    with pytest.warns(ExperimentalADIWarning) as seen:
        getattr(sim, entry)(n_steps=2)
    notices = [w for w in seen if w.category is ExperimentalADIWarning]
    assert len(notices) == 1
    assert str(notices[0].message) == ADI_WARNING
    assert notices[0].filename == __file__


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_yee_emits_no_adi_warning(mode, entry):
    sim = _sim(mode, solver="yee", boundary="pec")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        getattr(sim, entry)(n_steps=2)
    assert not any(issubclass(w.category, ExperimentalADIWarning) for w in seen)

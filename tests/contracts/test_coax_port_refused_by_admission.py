"""Coax ports refuse at admission before stepping; no physical anchor (a refusal)."""

from pathlib import Path
import traceback
from unittest.mock import Mock

import jax
import numpy as np
import pytest

from rfx import Simulation


def _model(mesh, *, coax):
    profiles = ({"dx_profile": np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 10), 1e-3]}
                if mesh == "graded" else {})
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .012),
                     dx=.001, boundary="pec", **profiles)
    sim.add_probe((.008, .006, .006), "ez")
    if coax:
        sim.add_coaxial_port((.006, .006, 0.), face="bottom", pin_length=.003)
    return sim


@pytest.mark.parametrize("mesh", ("uniform", "graded"))
@pytest.mark.parametrize("entry,advice", (
    ("run", "for a single add_coaxial_port(), compute_s_matrix(method=...) "
     "dispatches to either. Use add_port(..., extent=...) for probe-feed S-parameters."),
    ("forward", "both take eps_scale= for jax.grad. Use add_port(..., extent=...) "
     "for differentiable probe-feed S11 objectives."),
))
def test_coax_refused_by_admission_before_scan(monkeypatch, mesh, entry, advice):
    sim = _model(mesh, coax=True)
    scan = Mock(side_effect=AssertionError("coax refusal reached jax.lax.scan"))
    monkeypatch.setattr(jax.lax, "scan", scan)

    with pytest.raises(NotImplementedError) as exc:
        getattr(sim, entry)(n_steps=1, skip_preflight=True)

    scan.assert_not_called()
    raising_frame = traceback.extract_tb(exc.value.__traceback__)[-1]
    assert Path(raising_frame.filename).parts[-3:] == ("rfx", "runners", "_admission.py")
    message = str(exc.value)
    assert "a coaxial port (add_coaxial_port) is not carried by the" in message
    assert ("A coaxial port's S-parameters come from "
            "compute_coaxial_line_reflection(...) (one-port reflection) "
            "or compute_coaxial_two_port(...) (through line); " + advice) in message


@pytest.mark.parametrize("mesh", ("uniform", "graded"))
@pytest.mark.parametrize("entry", ("run", "forward"))
def test_without_coax_runs_one_step(mesh, entry):
    sim = _model(mesh, coax=False)
    result = getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert result.time_series.shape == (1, 1)
    assert np.isfinite(np.asarray(result.time_series)).all()

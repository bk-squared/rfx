"""``sim._boundary`` is a plain ``str`` on every construction path (#2214).

Since #1944 the absorber scalar was replaced by a ``str`` subclass carrying its
face expansion. h5py refuses a ``str`` subclass, so ``save_simulation_dataset``
failed after the solve for every CPML/UPML simulation, the default included,
and ``rfx run`` on the shipped YAML example exited 1. v1.8.0 wrote the file.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.interop import design_to_dict, simulation_from_design

h5py = pytest.importorskip("h5py")

_EXAMPLE_YAML = Path(__file__).resolve().parents[3] / "examples" / "config" / "microstrip_thru.yaml"


def _sim(**kw):
    return Simulation(freq_max=10e9, domain=(0.012, 0.010, 0.008), dx=1e-3, **kw)


_FORMS = {
    "default": {},
    "cpml": {"boundary": "cpml"},
    "upml": {"boundary": "upml"},
    "pec": {"boundary": "pec"},
    # A caller's own str subclass (a numpy string read from a table) is a scalar too.
    "numpy str": {"boundary": np.str_("cpml")},
    "spec": {"boundary": BoundarySpec(x=Boundary(lo="cpml", hi="pec"), y="cpml", z="cpml")},
}


@pytest.mark.parametrize("form", sorted(_FORMS))
def test_boundary_field_is_plain_str_and_writes_to_hdf5(form, tmp_path):
    sim = _sim(**_FORMS[form])
    for label, s in (("built", sim), ("deepcopy", copy.deepcopy(sim)),
                     ("design round trip", simulation_from_design(design_to_dict(sim)))):
        assert type(s._boundary) is str, f"{form} {label}: {type(s._boundary).__name__}"
        with h5py.File(tmp_path / f"{form}-{label}.h5", "w") as f:
            f.attrs["boundary"] = s._boundary


def test_default_boundary_dataset_round_trip(tmp_path):
    from rfx.io import save_simulation_dataset

    sim = _sim()
    sim.add_source((0.005, 0.004, 0.003), "ez", amplitude_kind="current")
    sim.add_probe((0.007, 0.006, 0.005), "ez")
    res = sim.run(n_steps=30, compute_s_params=False, skip_preflight=True)
    path = tmp_path / "dataset.h5"
    save_simulation_dataset(path, sim, res)
    with h5py.File(path, "r") as f:
        assert f["input"].attrs["boundary"] == "cpml"
        np.testing.assert_array_equal(f["output"]["time_series"][:], np.asarray(res.time_series))


def test_yaml_example_run_and_save_writes_the_dataset(tmp_path):
    """The path ``rfx run examples/config/microstrip_thru.yaml`` takes, on a short record."""
    from rfx.config.runner import run_and_save

    out = tmp_path / "thru.h5"
    with pytest.warns(UserWarning):
        run_and_save(_EXAMPLE_YAML, out, n_steps=20)
    with h5py.File(out, "r") as f:
        assert f["input"].attrs["boundary"] == "cpml"
        assert f["output"]["s_params"].shape[:2] == (2, 2)

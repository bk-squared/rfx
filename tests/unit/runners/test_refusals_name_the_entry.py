"""Public refusals name the entered call and the current wall grid rule."""
import jax
import numpy as np
import pytest

from rfx import Simulation
from tests._multi_device_scene import host_devices
from tests.contracts.test_default_s_plain_source import model
from tests.unit.preflight.test_pec_face_short_of_domain_wall import A_WG, B_WG, CODE, _sim


def graded_scene(kind):
    sim = Simulation(freq_max=10e9, domain=(.02,) * 3, dx=.001,
                     dz_profile=[.0009, .0011] * 10, boundary="pec")
    if kind == "rlc":
        sim.add_source((.01, .01, .005), "ex", amplitude_kind="current")
        sim.add_lumped_rlc((.008,) * 3, "ez", R=50.)
    else:
        sim.add_port((.01, .01, .005), "ez", impedance=50., extent=.003,
                     reference_plane_cells=1, direction="+x")
    sim.add_probe((.01, .01, .014), "ez")
    return sim


@pytest.mark.parametrize("entry,kind,lane,expected", [
    ("run", "rlc", None,
     "add_lumped_rlc() element(s) (1): not supported on the distributed multi-device run() "
     "path; the lane never applies them and would solve the structure as if "
     "the elements were absent, with no warning ("
     "rfx.runners.distributed_v2.run_distributed / "
     "rfx.runners.distributed.run_distributed have no lumped-element update). "
     "Remove the elements, or omit devices=... (use a single-device run() instead)."),
    ("run", "wire", None,
     "Wire ports with reference_plane_cells are not supported with devices=...; "
     "the V_ref path needs pre-injection recordings. Use one device."),
    # Graded forward(distributed=True) is refused at the shared multi-device check,
    # as run() is, before the graded lane assembles (#2076).
    ("forward", "rlc", None,
     "add_lumped_rlc() element(s) (1): not supported on the distributed multi-device forward() "
     "path; the lane never applies them and would solve the structure as if "
     "the elements were absent, with no warning ("
     "rfx.runners.distributed_v2.run_distributed / "
     "rfx.runners.distributed.run_distributed have no lumped-element update). "
     "Remove the elements, or omit distributed=True (use a single-device forward() instead)."),
    ("forward", "wire", None,
     "Wire ports with reference_plane_cells are not supported with distributed=True; "
     "the V_ref path needs pre-injection recordings. Use one device."),
])
def test_graded_entry_refusal(entry, kind, lane, expected, monkeypatch):
    from rfx.model import distributed_ports
    original = distributed_ports.assemble
    lanes = []

    def recorded(*args, **kwargs):
        lanes.append(kwargs["lane"])
        return original(*args, **kwargs)

    monkeypatch.setattr(distributed_ports, "assemble", recorded)
    sim = graded_scene(kind)
    options = dict(n_steps=4, skip_preflight=True, devices=host_devices(2))
    with pytest.raises(NotImplementedError) as error:
        if entry == "run":
            sim.run(compute_s_params=False, **options)
        else:
            sim.forward(distributed=True, **options)
    assert str(error.value) == expected
    assert lanes == ([] if lane is None else [lane])


# One message for both entries: an equal request is refused alike (run/forward
# one-pipeline contract), and the message names the way out from either call.
PLAIN_SOURCE_S_REFUSAL = (
    "Lumped/wire S-matrix requests do not support plain sources / "
    "0-ohm ports (add_source, add_polarized_source or add_port(impedance=0)): "
    "they fire in every port drive. Remove the plain source, or drop the S request to run "
    "with the source included: run(compute_s_params=False) returns the raw port waves, "
    "forward() without port_s11_freqs the fields and probe records.")


@pytest.mark.parametrize("entry,expected", [
    ("forward", PLAIN_SOURCE_S_REFUSAL),
    ("run", PLAIN_SOURCE_S_REFUSAL),
])
def test_plain_source_s_request(entry, expected):
    sim = model()
    with pytest.raises(NotImplementedError) as error:
        if entry == "forward":
            sim.forward(n_steps=4, skip_preflight=True, port_s11_freqs=[5e9])
        else:
            sim.run(n_steps=4, skip_preflight=True, compute_s_params=True)
    assert str(error.value) == expected


def test_forward_plain_source_without_s_request():
    result = model().forward(n_steps=4, skip_preflight=True)
    jax.block_until_ready(result)
    assert result.time_series.shape == (4, 1)
    assert np.isfinite(np.asarray(result.time_series)).all()


def test_wall_message_names_current_grid_rule():
    hit, = _sim(A_WG, B_WG).preflight().by_code(CODE)
    text = str(hit)
    assert "nearest whole number of cells" in text
    assert "the WR-90 waveguide case" in text
    assert "ceil(" not in text
    assert "cv11" not in text

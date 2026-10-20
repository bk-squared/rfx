"""Public distributed refusals identify the call that reached them."""

import pytest

from rfx import Simulation
from tests._multi_device_scene import host_devices
from tests.unit.materials.test_sheet_impedance import _cube


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("forward", "on the distributed multi-device forward() lane"),
        ("run", "on the distributed multi-device run() lane"),
    ],
)
def test_uniform_sheet_refusal_names_entry(entry, expected):
    devices = host_devices(2)
    sim = _cube()
    kwargs = dict(n_steps=4, devices=devices, skip_preflight=True)
    if entry == "forward":
        kwargs["distributed"] = True
    with pytest.raises(ValueError) as exc:
        getattr(sim, entry)(**kwargs)
    assert expected in str(exc.value)


def _graded_sim():
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
        dz_profile=[0.9e-3, 1.1e-3] * 10, boundary="pec",
    )
    sim.add_source((10e-3, 10e-3, 5e-3), "ex", amplitude_kind="current")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ex")
    return sim


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("run", "run(devices=...) on a non-uniform mesh requires n_steps >= 1; got 0"),
        ("forward", "forward(distributed=True) on a non-uniform mesh requires n_steps >= 1; got 0"),
    ],
)
def test_graded_zero_steps_names_entry(entry, expected):
    devices = host_devices(2)
    kwargs = dict(n_steps=0, devices=devices, skip_preflight=True)
    if entry == "forward":
        kwargs["distributed"] = True
    with pytest.raises(ValueError) as exc:
        getattr(_graded_sim(), entry)(**kwargs)
    assert "n_warmup" not in str(exc.value)
    assert str(exc.value) == expected


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("run", "run(devices=...): duplicate device entries are not allowed; "
                "each device must appear at most once."),
        ("forward", "forward(distributed=True, devices=...): duplicate device "
                    "entries are not allowed; each device must appear at most once."),
    ],
)
def test_graded_duplicate_devices_names_entry(entry, expected):
    device = host_devices(2)[0]
    kwargs = dict(n_steps=4, devices=[device, device], skip_preflight=True)
    if entry == "forward":
        kwargs["distributed"] = True
    with pytest.raises(ValueError) as exc:
        getattr(_graded_sim(), entry)(**kwargs)
    assert str(exc.value) == expected


def test_graded_single_device_run_still_accepts_zero_steps():
    result = _graded_sim().run(n_steps=0, skip_preflight=True)
    assert result.time_series.shape[0] == 0


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize(
    "entry, expected, remedy",
    [
        ("run", "Simulation.run()", "omit devices= to run on one device"),
        ("forward", "Simulation.forward()", "omit distributed=True to run on one device"),
    ],
)
def test_smoothing_refusal_names_entry(graded, entry, expected, remedy):
    devices = host_devices(2)
    sim = (_graded_sim() if graded else
           Simulation(freq_max=10e9, domain=(0.02,) * 3, dx=1e-3, boundary="pec"))
    kwargs = dict(n_steps=4, devices=devices, skip_preflight=True, subpixel_smoothing=True)
    if entry == "forward":
        kwargs["distributed"] = True
    with pytest.raises(NotImplementedError) as exc:
        getattr(sim, entry)(**kwargs)
    assert expected in str(exc.value)
    assert remedy in str(exc.value)

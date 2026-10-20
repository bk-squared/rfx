"""Public distributed refusals identify the call that reached them."""

import warnings

import pytest

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
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


def _wire_sim(*, graded=False, pmc=False):
    sim = Simulation(
        freq_max=10e9, domain=(0.02,) * 3, dx=1e-3,
        boundary=BoundarySpec(x="pec", y="pmc", z="pec") if pmc else "pec",
        **({"dz_profile": [0.9e-3, 1.1e-3] * 10} if graded else {}),
    )
    sim.add_port((10e-3, 10e-3, 5e-3), "ez", impedance=50.0, extent=3e-3)
    sim.add_probe((10e-3, 10e-3, 14e-3), "ez")
    return sim


@pytest.mark.parametrize(
    "entry, option", [("run", "devices=..."), ("forward", "distributed=True")])
def test_pmc_wire_refusal_names_entry(entry, option):
    kwargs = dict(n_steps=4, devices=host_devices(2), skip_preflight=True)
    if entry == "forward":
        kwargs["distributed"] = True
    else:
        kwargs["compute_s_params"] = False
    with pytest.raises(NotImplementedError) as exc:
        getattr(_wire_sim(pmc=True), entry)(**kwargs)
    assert f"are not supported with {option};" in str(exc.value)
    assert f"Use one device (omit {option})." in str(exc.value)


@pytest.mark.parametrize(
    "graded, path, reason",
    [
        (False, "uniform single-wire-port S-parameter",
         "this lane computes S11 from a second distributed wire-port recording run"),
        (True, "graded single-wire-port S-parameter",
         "this lane reads S11 from the main run's wire-port record"),
    ],
)
def test_wire_s_param_n_steps_reason_matches_the_mesh(graded, path, reason):
    with pytest.raises(NotImplementedError) as exc:
        _wire_sim(graded=graded).run(
            n_steps=8, devices=host_devices(2), skip_preflight=True,
            compute_s_params=True, s_param_freqs=[5e9], s_param_n_steps=16)
    assert path in str(exc.value)
    assert reason in str(exc.value)


def test_forward_refinement_remedy_points_to_run():
    sim = Simulation(freq_max=10e9, domain=(0.02,) * 3, dx=1e-3, boundary="cpml")
    sim.add_source((10e-3, 10e-3, 5e-3), "ez", amplitude_kind="current")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ez")
    sim.add_refinement((8e-3, 12e-3), ratio=2)
    with pytest.raises(NotImplementedError) as exc:
        sim.forward(n_steps=4, distributed=True, devices=host_devices(2),
                    skip_preflight=True)
    assert ("Remove the refinement, or use run() on one device "
            "(forward() does not take add_refinement()).") in str(exc.value)


def test_uniform_forward_settling_warning_names_forward():
    sim = Simulation(freq_max=10e9, domain=(0.024,) * 3, dx=1e-3, boundary="cpml")
    sim.add_source((12e-3, 12e-3, 12e-3), "ez", amplitude_kind="current")
    sim.add_probe((14e-3, 12e-3, 12e-3), "ez")
    sim.add_ntff_box((6e-3, 6e-3, 6e-3), (18e-3, 18e-3, 18e-3), freqs=[5e9])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.forward(n_steps=20, distributed=True, devices=host_devices(2),
                    skip_preflight=True)
    settling = [str(w.message) for w in caught
                if "settling witness" in str(w.message)]
    assert settling
    assert all(t.startswith("no ring-down settling witness on this forward() call")
               for t in settling)


def test_pmc_lumped_s_refusal_names_forward():
    sim = Simulation(freq_max=10e9, domain=(0.02,) * 3, dx=1e-3,
                     boundary=BoundarySpec(x="pec", y="pmc", z="pec"))
    sim.add_port((10e-3, 10e-3, 5e-3), "ez", impedance=50.0)
    sim.add_probe((10e-3, 10e-3, 14e-3), "ez")
    with pytest.raises(NotImplementedError) as exc:
        sim.forward(n_steps=4, distributed=True, devices=host_devices(2),
                    skip_preflight=True, port_s11_freqs=[5e9])
    assert ("are not supported with distributed=True; use one device "
            "(omit distributed=True).") in str(exc.value)


def test_reference_plane_wire_refusal_names_forward():
    sim = Simulation(freq_max=10e9, domain=(0.02,) * 3, dx=1e-3, boundary="pec")
    sim.add_port((10e-3, 10e-3, 5e-3), "ez", impedance=50.0, extent=3e-3,
                 reference_plane_cells=1, direction="+x")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ez")
    with pytest.raises(NotImplementedError) as exc:
        sim.forward(n_steps=4, distributed=True, devices=host_devices(2),
                    skip_preflight=True)
    assert ("Wire ports with reference_plane_cells are not supported with "
            "distributed=True;") in str(exc.value)

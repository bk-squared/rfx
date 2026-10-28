"""Shared flux arithmetic, H ghosts, and the record-slot window clock."""
import json
from time import perf_counter

import jax
from tests._multi_device_scene import host_devices
import numpy as np
import pytest

from rfx.measurement import accumulators
from rfx.probes.probes import flux_spectrum
from tests.unit.runners.test_distributed_record_planes import model


def _pair(sim, n_steps, n_devices, entry):
    """One device, then ``n_devices``, through the same entry."""
    call = getattr(sim, entry)
    kwargs = dict(n_steps=n_steps, skip_preflight=True, **({"checkpoint": False} if entry == "forward" else {}))
    return call(**kwargs), call(**kwargs, devices=host_devices(n_devices),
                                **({"distributed": True} if entry == "forward" else {}))


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_flux(boundary, n_devices, entry, monkeypatch, tmp_path):
    sim, grid, seam = model(boundary, n_devices)
    for axis, index in ((0, seam - 1), (0, seam), (1, grid.shape[1] // 2)):
        coordinate = (index - (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[axis]) * grid.dx
        sim.add_flux_monitor(axis="xyz"[axis], coordinate=coordinate,
                             freqs=[3e9, 7e9], name=f"{axis}_{index}", dft_window="hann")
    clocks = []
    original = accumulators.flux

    def observe(state, accs, metadata, dt, step, *, window_step=None):
        width = state.ex.shape[0]
        jax.debug.callback(lambda n, s: clocks.append((width, int(n), int(s))), step, state.step)
        return original(state, accs, metadata, dt, step, window_step=window_step)

    monkeypatch.setattr(accumulators, "flux", observe)
    start = perf_counter()
    reference, actual = _pair(sim, 40, n_devices, entry)
    jax.block_until_ready(actual.flux_monitors)
    jax.effects_barrier()
    differences = {}
    for name, mon in reference.flux_monitors.items():
        other = actual.flux_monitors[name]
        for component in ("e1_dft", "e2_dft", "h1_dft", "h2_dft", "power"):
            a, b = ((flux_spectrum(other), flux_spectrum(mon)) if component == "power" else
                    (getattr(other, component), getattr(mon, component)))
            a, b = np.asarray(a), np.asarray(b)
            differences[f"{name}/{component}"] = dict(equal=bool(np.array_equal(a, b)),
                max_abs=float(np.max(np.abs(a - b))), peak=float(np.max(np.abs(b))))
    native_clock = sorted({(n, s) for width, n, s in clocks if width == grid.shape[0]})
    slab_clock = sorted({(n, s) for width, n, s in clocks if width != grid.shape[0]})
    report = dict(boundary=boundary, devices=n_devices, entry=entry, grid_shape=grid.shape, n_steps=40,
                  seconds=perf_counter() - start, flux=differences,
                  native_clock=native_clock, slab_clock=slab_clock)
    (tmp_path / "measurements.json").write_text(json.dumps(report))
    print(report)
    assert native_clock == slab_clock == [(n, n + 1) for n in range(40)], "A1 record-slot clock"
    assert all(v["equal"] for v in differences.values()), ("flux array_equal", differences)


@pytest.mark.parametrize("mutation", ["ghost_exchange", "window_clock"])
def test_flux_mutations(mutation, monkeypatch, tmp_path):
    if mutation == "ghost_exchange":
        from rfx.runners import distributed_v2
        monkeypatch.setattr(distributed_v2, "_exchange_h_ghosts_shmap", lambda state, *a, **kw: state)
    else:
        original = accumulators.flux

        def shifted(state, accs, metadata, dt, step, *, window_step=None):
            # The fixture has global nx=13; only local slab calls are mutated.
            if state.ex.shape[0] != 13:
                window_step = window_step - 1
            return original(state, accs, metadata, dt, step, window_step=window_step)
        monkeypatch.setattr(accumulators, "flux", shifted)
    with pytest.raises(AssertionError, match="flux array_equal") as failure:
        test_flux("pec", 2, "run", monkeypatch, tmp_path)
    measured = json.loads((tmp_path / "measurements.json").read_text())["flux"]
    assert any(not row["equal"] for name, row in measured.items() if name.startswith("0_7/")), "seam flux must turn red"
    (tmp_path / "mutation.json").write_text(json.dumps(dict(mutation=mutation, assertion=str(failure.value))))
    print(f"mutation={mutation}: RED {failure.value}")


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_finite_flux(boundary, n_devices, entry, tmp_path):
    from tests.unit.runners.test_distributed_record_planes import finite_model
    sim, grid, seam = finite_model(boundary, n_devices)
    freqs = [2.5e9, 6.1e9, 11.7e9]
    # At the first owned row, x-normal H sampling crosses the seam via the ghost.
    sim.add_flux_monitor(axis="x", coordinate=(seam - grid.pad_x_lo) * grid.dx,
                         freqs=freqs, size=(5.2e-3, 3.3e-3), center=(6.1e-3, 3.9e-3), name="finite_x")
    sim.add_flux_monitor(axis="z", coordinate=1.4e-3, freqs=freqs,
                         size=(11.1e-3, 6.2e-3), center=(7.9e-3, 5.3e-3),
                         name="finite_z", dft_window="hann")
    start = perf_counter()
    reference, actual = _pair(sim, 47, n_devices, entry)
    jax.block_until_ready(actual.flux_monitors)
    differences = {}
    for name, mon in reference.flux_monitors.items():
        if name == "finite_x":
            assert mon.index == seam
        else:
            for array in mon[:4]:
                assert mon.lo1 < seam < mon.lo1 + array.shape[1]
        other = actual.flux_monitors[name]
        for component in ("e1_dft", "e2_dft", "h1_dft", "h2_dft", "power"):
            a, b = ((flux_spectrum(other), flux_spectrum(mon)) if component == "power" else
                    (getattr(other, component), getattr(mon, component)))
            a, b = np.asarray(a), np.asarray(b)
            differences[f"{name}/{component}"] = dict(equal=bool(np.array_equal(a, b)),
                max_abs=float(np.max(np.abs(a - b))), peak=float(np.max(np.abs(b))))
        assert np.max(np.abs(mon.e1_dft)) + np.max(np.abs(mon.e2_dft)) > 0
    report = dict(boundary=boundary, devices=n_devices, entry=entry, grid_shape=grid.shape, n_steps=47,
                  seam=seam, seconds=perf_counter() - start, flux=differences)
    (tmp_path / "measurements.json").write_text(json.dumps(report))
    print(report)
    assert all(v["equal"] for v in differences.values()), ("finite flux array_equal", differences)

"""Shared flux arithmetic, H ghosts, and the record-slot window clock."""
import json
from time import perf_counter

import jax
import numpy as np
import pytest

from rfx.measurement import accumulators
from rfx.probes.probes import flux_spectrum
from tests.unit.runners.test_distributed_record_planes import model


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_flux(boundary, n_devices, monkeypatch, tmp_path):
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
    reference = sim.run(n_steps=40, skip_preflight=True)
    actual = sim.run(n_steps=40, devices=jax.devices("cpu")[:n_devices], skip_preflight=True)
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
    report = dict(boundary=boundary, devices=n_devices, grid_shape=grid.shape, n_steps=40,
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
        test_flux("pec", 2, monkeypatch, tmp_path)
    measured = json.loads((tmp_path / "measurements.json").read_text())["flux"]
    assert any(not row["equal"] for name, row in measured.items() if name.startswith("0_7/")), "seam flux must turn red"
    (tmp_path / "mutation.json").write_text(json.dumps(dict(mutation=mutation, assertion=str(failure.value))))
    print(f"mutation={mutation}: RED {failure.value}")

"""Always-on right-Hx observer contract for admitted slab H writers."""
from contextlib import redirect_stdout
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.runners import _admission
from rfx.runners._distributed_ntff import SlabNTFF
from rfx.stepping.slab import Slab


def _check_ntff_ghost(monkeypatch, *, tfsf):
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2, "ghost invariant requires two CPU host devices"
    sim = Simulation(freq_max=15e9, domain=(.021, .017, .015),
                     dx=.001, boundary="cpml", cpml_layers=4)
    if tfsf:
        sim.add_tfsf_source(f0=7.5e9, bandwidth=.8, polarization="ez",
                            direction="+x", closed_box=True, margin=3)
    else:
        sim.add_source((.010, .008, .007), "ez", amplitude_kind="field",
                       waveform=lambda t: jnp.exp(-((t - 3e-11) / 1e-11) ** 2))
    sim.add_ntff_box((.002, .002, .002), (.019, .015, .013),
                     freqs=[5e9, 10e9, 15e9])
    grid = sim._build_grid()
    layout = Slab(grid.nx, 2)
    observed = []
    original = SlabNTFF.update

    def capture(hx, step):
        rows = np.asarray(hx).reshape(2, layout.nx_local, grid.ny, grid.nz)
        ghost, owner = rows[0, -1], rows[1, 1]
        difference = np.abs(ghost - owner)
        delta = float(difference.max())
        peak = float(np.max(np.abs(owner)))
        ulp = float(np.spacing(np.float32(peak)))
        max_ulp = 0.0 if delta == 0 else (delta / ulp if ulp else None)
        first = np.argwhere(difference != 0)
        observed.append(dict(step=int(step), completed_step=int(step) + 1,
                             bit_equal=np.array_equal(ghost, owner),
                             max_abs=delta, peak=peak, max_ulp=max_ulp,
                             first_node=([layout.nx_per_rank, *first[0].tolist()]
                                         if len(first) else None)))

    def update(self, buffer, state, dt, step, **kwargs):
        # Observe the actual scan state immediately before the NTFF reader;
        # do not fill, recompute or otherwise repair a ghost in this hook.
        jax.debug.callback(capture, state.hx, step)
        return original(self, buffer, state, dt, step, **kwargs)

    monkeypatch.setattr(SlabNTFF, "update", update)
    with redirect_stdout(sys.stderr):
        result = sim.run(n_steps=160, devices=devices)
    jax.block_until_ready(result.state.hx)
    jax.effects_barrier()
    assert result.ntff_box.i_lo < layout.nx_per_rank < result.ntff_box.i_hi
    assert len(observed) == 160
    assert len({row["step"] for row in observed}) == 160
    assert max(row["peak"] for row in observed) > 0
    first = next((row for row in sorted(observed, key=lambda r: r["step"])
                  if not row["bit_equal"]), None)
    print(json.dumps(dict(test="closed_box_tfsf" if tfsf else "ez_dipole",
                          devices=2, shape=grid.shape, seam=layout.nx_per_rank,
                          steps=len(observed), bit_equal=first is None,
                          max_abs=max(row["max_abs"] for row in observed),
                          max_ulp=(max(row["max_ulp"] for row in observed)
                                   if all(row["max_ulp"] is not None for row in observed)
                                   else None),
                          first_difference=first)), file=sys.stderr)
    assert first is None, f"right Hx ghost differs from owner at NTFF read: {first}"


def test_closed_box_tfsf_right_hx_ghost_at_ntff(monkeypatch):
    _check_ntff_ghost(monkeypatch, tfsf=True)


def test_ez_dipole_right_hx_ghost_at_ntff(monkeypatch):
    _check_ntff_ghost(monkeypatch, tfsf=False)


def test_owner_only_h_writers_have_collected_ghost_invariants(request):
    """A new inventory entry without its NTFF ghost witness fails admission CI."""
    # Collect this module explicitly so a focused -k run still checks the
    # witnesses, without executing them recursively or depending on item order.
    collected = {item.name: item for item in request.node.parent.collect()}
    for writer in _admission.SLAB_OWNER_ONLY_H_WRITERS:
        name = f"test_{writer}_right_hx_ghost_at_ntff"
        assert name in collected, f"admitted owner-only H writer {writer!r} lacks {name}"
        assert callable(collected[name].obj)

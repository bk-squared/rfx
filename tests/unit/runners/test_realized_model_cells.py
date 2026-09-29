"""Kernel operands for an off-centre block and one excitation at a time.

Material comparisons are exact (0 ULP). Drive scales permit 4 ULP per
float32 element: the NU concrete source uses Python float arithmetic before
storing float32, whereas the reference evaluates Cb in JAX float32. A unit
waveform makes the injection table itself the scale; no division by a small
pulse tail enters the measurement. These are operand comparisons, not a
claim about S-parameter accuracy.
"""
from __future__ import annotations

import functools
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx import _realized
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import component_e_materials, e_update_coeffs
from tests.contracts import realized_model as T

MM = 1e-3


def unit_waveform(t):
    return jnp.ones_like(t)


def model(lane, row):
    sg = lane == "run_subgridded"
    kw = dict(freq_max=10e9, domain=(12*MM, 11*MM, (20 if sg else 12)*MM),
              dx=MM, boundary="cpml" if row.startswith("open_") else "pec",
              cpml_layers=3)
    if lane in ("run_nonuniform", "fwd_nonuniform", "fwd_distributed_nu"):
        kw["dx_profile"] = np.array([MM]*4 + [MM/2]*8 + [MM]*4)
    if lane in ("run_adi", "fwd_adi"):
        kw["solver"] = "adi"
    if row == "conformal":
        kw["boundary"] = BoundarySpec(
            x=Boundary(lo="pec", hi="pec", conformal=True),
            y=Boundary(lo="pec", hi="pec"), z=Boundary(lo="pec", hi="pec"))
    sim = Simulation(**kw)
    if sg:
        sim.add_refinement(z_range=(0., 14*MM), ratio=2)
    if row in ("eps", "sigma", "mu", "conformal", "sat", "override_drive", "soft_current"):
        mat = {"eps_r": 4.} if row in ("eps", "conformal", "sat", "override_drive", "soft_current") else (
            {"sigma": 0.2} if row == "sigma" else {"mu_r": 4.})
        sim.add_material("block", **mat)
        hi = (8*MM, 9*MM, (20 if row == "sat" else 10)*MM)
        sim.add(Box((5*MM, 3*MM, 4*MM), hi), material="block")
    if row == "conformal":
        sim.add(Box((9*MM, 2*MM, 2*MM), (10*MM, 4*MM, 4*MM)), material="pec")
    pos = (5*MM, 5*MM, 6*MM)
    if row.startswith(("lumped_", "wire_")):
        kind = row.split("_")[1]
        extra = {} if kind == "none" else {"amplitude_kind": kind}
        sim.add_port(pos, "ez", waveform=unit_waveform,
                     extent=2*MM if row.startswith("wire") else None, **extra)
    else:
        kind = row.split("_")[1] if row.startswith(("soft_", "open_")) else "field"
        if row == "override_drive":
            kind = "current"
        sim.add_source(pos, "ez", waveform=unit_waveform,
                       amplitude_kind=None if kind == "none" else kind)
    sim.add_probe((6*MM, 5*MM, 6*MM), "ez")
    return sim


def run(sim, lane, row, steps=4):
    kw = dict(n_steps=steps, skip_preflight=True)
    if lane.startswith("run"):
        kw["compute_s_params"] = False
        if lane == "run_distributed":
            kw["devices"] = jax.devices("cpu")[:2]
        if row == "conformal":
            kw["conformal_pec"] = True
        return sim.run(**kw)
    if lane == "fwd_distributed_nu":
        kw.update(distributed=True, devices=jax.devices("cpu")[:2])
        if row == "override_drive":
            grid = sim._build_nonuniform_grid()
            kw["eps_override"] = sim._assemble_materials_nu(grid)[0].eps_r
    return sim.forward(checkpoint=False, **kw)


@functools.lru_cache(maxsize=None)
def measured(lane, row):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = model(lane, row)
        with _realized.capture() as dump:
            result = run(sim, lane, row)
            jax.block_until_ready(result.time_series)
    assert dump.lane == lane
    return dump


def material_pairs(dump, row, axis):
    quantity = {"eps": "eps_e", "sigma": "sigma_e", "mu": "mu_h",
                "conformal": "eps_e", "sat": "eps_e"}[row]
    records = [r for r in dump.records if quantity in r
               and (r["site"].startswith("sat.") == (row == "sat"))]
    if row == "conformal":
        records = [r for r in records if r["site"] == "aniso.E"]
    assert records, (dump.lane, row, [r["site"] for r in dump.records])
    for r in records:
        if axis >= len(r[quantity]):
            continue  # SAT consumes tangential x/y, not normal z.
        mats = r["materials"]
        if row == "mu":
            ref = np.asarray(mats.mu_r)
        else:
            eps, sig = component_e_materials(mats, r["periodic"])
            ref = np.asarray(sig[axis] if row == "sigma" else eps[axis])
        got = r[quantity][axis]
        if r["region"] is not None:
            ref = ref[r["region"]]
        if row == "conformal":
            # Only the dielectric: the distant PEC weights are intentional.
            got, ref = got[ref != 1.], ref[ref != 1.]
        if r["site"].startswith("distributed"):
            got, ref = got[1:-1], ref[1:-1]  # owned rows; omit slab ghosts
        yield got, ref


def drive_pairs(dump, row):
    from rfx.nonuniform import current_source_volume
    from rfx.sources.sources import port_d_parallel
    records = [r for r in dump.records if "drive_scale" in r]
    assert records, (dump.lane, row)
    for r in records:
        grid, pe = r["grid"], r["declaration"]
        eps, sig = component_e_materials(r["materials"])
        expected = []
        n = len(r["cells"])
        for i, j, k, component in r["cells"]:
            axis = "xyz".index(component[1])
            cb = float(e_update_coeffs(eps[axis][i,j,k], sig[axis][i,j,k], grid.dt)[1])
            if pe.impedance:
                d = port_d_parallel(grid, (i,j,k), component)
                scale = cb / d / n
            else:
                if hasattr(grid, "dx_arr"):
                    dV = float(current_source_volume(grid, (i,j,k), component)[0])
                else:
                    dV = grid.dx * getattr(grid, "dy", grid.dx) * getattr(grid, "dz", grid.dx)
                kind = pe.amplitude_kind
                scale = (cb/dV if kind == "current" else 1. if kind == "field"
                         else cb if row.startswith("open") else 1.)
            expected.append(scale)
        yield np.asarray(r["drive_scale"], dtype=np.float32), np.asarray(expected, dtype=np.float32)


def assert_cell(dump, row, axis=0):
    pairs = (material_pairs(dump, row, axis) if row in
             ("eps", "sigma", "mu", "conformal", "sat") else drive_pairs(dump, row))
    count = 0
    for got, ref in pairs:
        count += 1
        delta = float(np.max(np.abs(got.astype(float) - ref.astype(float))))
        nonzero = ref != 0
        relative = float(np.max(np.abs((got[nonzero].astype(float) -
                        ref[nonzero]) / ref[nonzero]))) if np.any(nonzero) else 0.
        print(f"{dump.lane} {row} axis={axis} max_abs={delta:.9g} max_rel={relative:.9g}",
              file=sys.stderr)
        if row in ("eps", "sigma", "mu", "conformal", "sat"):
            assert np.array_equal(got, ref), (dump.lane, row, axis, delta)
        else:
            print(f"drive got={got.tolist()} reference={ref.tolist()}", file=sys.stderr)
            np.testing.assert_array_max_ulp(got, ref, maxulp=4)
    assert count, (dump.lane, row, axis)


def cases():
    for row in T.ROWS:
        for lane in T.LANES:
            cell = T.TABLE[row][lane]
            axes = range(2 if row == "sat" else 3) if row in (
                "eps", "sigma", "mu", "conformal", "sat") else (0,)
            for axis in axes:
                marks = (pytest.mark.xfail(strict=True, raises=AssertionError,
                         reason=f"{cell.issue} {cell.note}"),) if cell.issue else ()
                yield pytest.param(lane, row, axis, marks=marks,
                                   id=f"{row}-{lane}-{'xyz'[axis]}")


@pytest.mark.parametrize("lane,row,axis", list(cases()))
def test_realized_cell(lane, row, axis):
    cell = T.TABLE[row][lane]
    if cell.kind == "not reachable":
        if row.startswith(("lumped", "wire")):
            with pytest.raises(TypeError):
                model(lane, row)
        else:
            assert (lane != "run_subgridded" if row == "sat" else lane != "fwd_distributed_nu")
        return
    if cell.kind == "refuses":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises((ValueError, NotImplementedError)):
                sim = model(lane, row)
                run(sim, lane, row)
        return
    assert_cell(measured(lane, row), row, axis)
    if row == "override_drive":
        assert any("runtime_scale" in r for r in measured(lane, row).records)


@pytest.mark.parametrize("lane", T.LANES)
def test_dump_is_consumed(lane):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = model(lane, "eps")
        plain = np.asarray(run(sim, lane, "eps", steps=16).time_series)
        with _realized.capture() as saved:
            observed = np.asarray(run(sim, lane, "eps", steps=16).time_series)

        def replay(factor):
            def feed_back(site, quantity, arrays):
                if quantity != "eps_e" or site.startswith("sat"):
                    return arrays
                # Select the saved grid/slab by its operand values. This
                # also covers two equal-shaped device slabs without relying
                # on callback arrival order. An unrecognized operand yields
                # NaN, so a missing saved slab cannot silently pass.
                candidates = {}
                for record in saved.records:
                    if record["site"] == site and quantity in record:
                        for value in record[quantity]:
                            candidates[(value.shape, value.tobytes())] = value
                result = []
                for value in arrays:
                    restored = jnp.full_like(value, jnp.nan)
                    for stored in candidates.values():
                        if stored.shape == value.shape:
                            operand = jnp.asarray(stored)
                            restored = jnp.where(jnp.all(value == operand),
                                                 operand * factor, restored)
                    result.append(restored)
                return tuple(result)
            return feed_back

        with _realized.capture(transform=replay(1.0)) as same:
            unchanged = np.asarray(run(sim, lane, "eps", steps=16).time_series)
        with _realized.capture(transform=replay(1.01)) as changed:
            moved = np.asarray(run(sim, lane, "eps", steps=16).time_series)
    assert any("eps_e" in r for r in same.records)
    assert any("eps_e" in r for r in changed.records)
    assert plain.dtype == observed.dtype == unchanged.dtype
    assert plain.shape == observed.shape == unchanged.shape
    assert plain.tobytes() == observed.tobytes() == unchanged.tobytes()
    assert np.isfinite(moved).all()
    assert np.any(moved != unchanged)
    print(f"J4 {lane} unchanged_bits=True max_delta_V_per_m={np.max(np.abs(moved-unchanged)):.9g} "
          f"traced_sites={sorted({r['site'] for r in same.records if r['traced']})}", file=sys.stderr)

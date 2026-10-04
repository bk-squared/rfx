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
from rfx.core.yee import component_e_materials, component_h_materials, e_update_coeffs
from tests.contracts import realized_model as T

MM = 1e-3


def unit_waveform(t):
    return jnp.ones_like(t)


def model(lane, row, fill=False):
    """``fill`` puts the material over the whole domain instead of a block:
    the only material the ADI lanes carry (#1373), used for their
    consumption witness."""
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
        # #1465: explicitly opt in to exercise the experimental subgrid lane.
        sim.add_refinement(z_range=(0., 14*MM), ratio=2, validation="research")
    if row in ("eps", "sigma", "mu", "conformal", "sat", "override_drive", "soft_current", "soft_none"):
        mat = {"eps_r": 4.} if row in ("eps", "conformal", "sat", "override_drive", "soft_current", "soft_none") else (
            {"sigma": 0.2} if row == "sigma" else {"mu_r": 4.})
        sim.add_material("block", **mat)
        hi = (8*MM, 9*MM, (20 if row == "sat" else 10)*MM)
        lo = (5*MM, 3*MM, 4*MM)
        if fill:
            lo, hi = (0.0, 0.0, 0.0), tuple(kw["domain"])
        sim.add(Box(lo, hi), material="block")
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


def cell_materials(sim, lane):
    if lane in ("run_nonuniform", "fwd_nonuniform", "fwd_distributed_nu"):
        return sim._assemble_materials_nu(sim._build_nonuniform_grid())[0]
    return sim._assemble_materials(sim._build_grid())[0]


def override_epsilon(sim, lane):
    drawn = cell_materials(sim, lane).eps_r
    return jnp.where(drawn == 4., 6., drawn)


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
        kw["eps_override"] = override_epsilon(sim, lane)
    return sim.forward(checkpoint=False, **kw)


@functools.lru_cache(maxsize=None)
def measured(lane, row):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = model(lane, row)
        with _realized.capture() as dump:
            result = run(sim, lane, row)
            jax.block_until_ready(result.time_series)
    if dump.lane != lane:
        raise RuntimeError(f"expected lane {lane}, recorded {dump.lane}")
    if lane in ("run_distributed", "fwd_distributed_nu"):
        dump.full_materials = cell_materials(sim, lane)
    return dump


def material_pairs(dump, row, axis):
    quantity = {"eps": "eps_e", "sigma": "sigma_e", "mu": "mu_h",
                "conformal": "eps_e", "sat": "eps_e"}[row]
    records = [r for r in dump.records if quantity in r
               and (r["site"].startswith("sat.") == (row == "sat"))]
    if row == "conformal":
        records = [r for r in records if r["site"] == "aniso.E"]
    if not records:
        raise RuntimeError(f"no record at material site: {dump.lane} {row}")
    sites = {r["site"] for r in records}
    expected = ({"sat.c", "sat.f"} if row == "sat" else
                {"aniso.E"} if row == "conformal" else
                {"adi.E"} if dump.lane in ("run_adi", "fwd_adi") else
                {"distributed.E"} if row != "mu" and dump.lane == "run_distributed" else
                {"distributed_nu.E"} if row != "mu" and dump.lane == "fwd_distributed_nu" else
                set())
    if expected - sites:
        raise RuntimeError(f"no record at {sorted(expected - sites)}")
    if row != "mu" and dump.lane in ("run_distributed", "fwd_distributed_nu"):
        spans = sorted((int(r["owned_start"]), int(r["owned_count"])) for r in records)
        end = 0
        for start, count in spans:
            if start != end or count <= 0:
                raise RuntimeError(f"missing or repeated owned rows: {spans}")
            end += count
        if end != dump.full_materials.eps_r.shape[0]:
            raise RuntimeError(f"incomplete owned rows: {spans}")
    for r in records:
        if axis >= len(r[quantity]):
            continue  # SAT consumes tangential x/y, not normal z.
        distributed = r["site"].startswith("distributed")
        mats = dump.full_materials if distributed else r["materials"]
        if row == "mu":
            hi = np.asarray(r["materials"].mu_r)
            lo = np.roll(hi, 1, axis=axis)
            edge = [slice(None)] * hi.ndim
            edge[axis] = 0
            if not r["periodic"][axis]:
                lo[tuple(edge)] = hi[tuple(edge)]
            widths = r.get("cell_sizes")
            if widths is None:
                ref = 2 / (1 / lo + 1 / hi)
            else:
                dh = np.asarray(widths[axis])
                if len(dh) == hi.shape[axis] - 1:
                    dh = np.r_[dh, dh[-1]]
                dl = np.roll(dh, 1)
                if not r["periodic"][axis]:
                    dl[0] = dh[0]
                shape = [1] * hi.ndim
                shape[axis] = len(dh)
                dl, dh = dl.reshape(shape), dh.reshape(shape)
                ref = (dl + dh) / (dl / lo + dh / hi)
            ref = np.where(lo == hi, hi, ref)
            stamps = r["materials"].mu_r_wire
            if stamps is not None and stamps[axis] is not None:
                ref = ref + np.asarray(stamps[axis])
        else:
            eps, sig = component_e_materials(mats, r["periodic"])
            ref = np.asarray(sig[axis] if row == "sigma" else eps[axis])
        got = r[quantity][axis]
        if r["region"] is not None:
            ref = ref[r["region"]]
        if row == "conformal":
            # Only the dielectric: the distant PEC weights are intentional.
            got, ref = got[ref != 1.], ref[ref != 1.]
        if distributed:
            if "owned_start" not in r or "owned_count" not in r:
                raise RuntimeError(f"missing ownership at {r['site']}")
            start, count = int(r["owned_start"]), int(r["owned_count"])
            got, ref = got[1:1 + count], ref[start:start + count]
        yield got, ref


def drive_pairs(dump, row):
    from rfx.nonuniform import current_source_volume
    from rfx.sources.sources import port_sigma, port_d_parallel
    records = [r for r in dump.records if "drive_scale" in r]
    if not records:
        raise RuntimeError(f"no record at source site: {dump.lane} {row}")
    if row in ("soft_none", "open_none"):
        # The default's reference is this lane's explicit current declaration,
        # including ADI's own coefficient and remaining material convention.
        current = measured(dump.lane, row.replace("_none", "_current"))
        references = [r for r in current.records if "drive_scale" in r]
        if len(records) != len(references):
            raise RuntimeError("default/current source records differ")
        for got, ref in zip(records, references):
            assert got["cells"] == ref["cells"]
            yield np.asarray(got["drive_scale"], dtype=np.float32), np.asarray(
                ref["drive_scale"], dtype=np.float32)
        return
    for r in records:
        grid, pe = r["grid"], r["declaration"]
        # Main's helper includes add_lumped_eps on the stamped component;
        # neither the volume mean nor the lumped stamp is reimplemented here.
        materials = r["materials"]
        if row == "override_drive":
            materials = materials._replace(eps_r=override_epsilon(dump.sim, dump.lane))
        eps, sig = component_e_materials(materials)
        expected = []
        for i, j, k, component in r["cells"]:
            axis = "xyz".index(component[1])
            cb = float(e_update_coeffs(eps[axis][i,j,k], sig[axis][i,j,k], grid.dt)[1])
            if pe.impedance:
                d = port_d_parallel(grid, (i,j,k), component)
                scale = cb * port_sigma(grid, (i,j,k), component, pe.impedance) / d
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
    if row == "override_drive" and dump.lane == "fwd_distributed_nu":
        if not any("runtime_scale" in r for r in dump.records):
            raise RuntimeError("no record at distributed_nu.drive")
    pairs = (material_pairs(dump, row, axis) if row in
             ("eps", "sigma", "mu", "conformal", "sat") else drive_pairs(dump, row))
    # Materialize first so missing records/operands cannot be swallowed by
    # a strict xfail on an earlier numerical mismatch.
    pairs = list(pairs)
    if not pairs:
        raise RuntimeError(f"no operands at {dump.lane} {row} axis={axis}")
    for got, ref in pairs:
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


@pytest.mark.parametrize("lane,row", [(lane, row) for lane in T.LANES
                         for row in ("eps", "sigma", "mu")
                         if not (row == "mu" and lane in ("run_adi", "fwd_adi"))])
def test_dump_is_consumed(lane, row):
    quantity_name = {"eps": "eps_e", "sigma": "sigma_e", "mu": "mu_h"}[row]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ADI refuses a material interface (#1373): its witness is a fill.
        sim = model(lane, row, fill=lane in ("run_adi", "fwd_adi"))
        plain = np.asarray(run(sim, lane, row, steps=16).time_series)
        with _realized.capture() as saved:
            observed = np.asarray(run(sim, lane, row, steps=16).time_series)

        def replay(factor):
            def feed_back(site, quantity, arrays):
                if quantity != quantity_name or site.startswith("sat"):
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
            unchanged = np.asarray(run(sim, lane, row, steps=16).time_series)
        with _realized.capture(transform=replay(1.01)) as changed:
            moved = np.asarray(run(sim, lane, row, steps=16).time_series)
    assert any(quantity_name in r for r in same.records)
    assert any(quantity_name in r for r in changed.records)
    assert plain.dtype == observed.dtype == unchanged.dtype
    assert plain.shape == observed.shape == unchanged.shape
    assert plain.tobytes() == observed.tobytes() == unchanged.tobytes()
    assert np.isfinite(moved).all()
    assert np.any(moved != unchanged)
    print(f"J4 {lane} {row} unchanged_bits=True max_delta_V_per_m={np.max(np.abs(moved-unchanged)):.9g} "
          f"sites={sorted({r['site'] for r in same.records if quantity_name in r and not r['site'].startswith('sat')})} "
          f"traced_sites={sorted({r['site'] for r in same.records if r['traced']})}", file=sys.stderr)


@pytest.mark.parametrize("lane", ["run_distributed", "fwd_distributed_nu"])
@pytest.mark.parametrize("row", ["eps", "sigma"])
def test_ghost_row_mutation_is_detected(monkeypatch, lane, row):
    from rfx.runners import _distributed_common as dc

    original = dc._slab_x_lo_view

    def broken_ghost(arr, rank):
        view = original(arr, rank)
        # Keep both material helpers; corrupt rank 1's incoming ghost.
        return view.at[0].set(jnp.where(rank == 1, jnp.ones_like(view[0]), view[0]))

    measured.cache_clear()
    try:
        monkeypatch.setattr(dc, "_slab_x_lo_view", broken_ghost)
        dump = measured(lane, row)
        for axis in (1, 2):
            with pytest.raises(AssertionError):
                assert_cell(dump, row, axis)
    finally:
        measured.cache_clear()


@pytest.mark.parametrize("row", ["eps", "mu", "override_drive"])
def test_missing_record_is_not_a_departure(row):
    dump = _realized.Capture(lane="run_adi")
    with pytest.raises(RuntimeError, match="no record"):
        assert_cell(dump, row)


def test_missing_runtime_record_is_not_a_departure():
    dump = _realized.Capture(lane="fwd_distributed_nu", records=[
        {"site": "distributed_nu.sources", "drive_scale": np.array([0.])}])
    with pytest.raises(RuntimeError, match="no record at distributed_nu.drive"):
        assert_cell(dump, "override_drive")


def test_missing_sat_face_is_not_a_departure():
    dump = _realized.Capture(lane="run_subgridded", records=[
        {"site": "sat.c", "eps_e": (np.array([0.]),) * 2}])
    with pytest.raises(RuntimeError, match="sat.f"):
        assert_cell(dump, "sat")


@pytest.mark.parametrize("kernel,quantity", [
    ("precompute", "eps_e"), ("precompute", "sigma_e"), ("precompute", "mu_h"),
    ("aniso", "sigma_e"),
])
def test_kernel_dump_is_consumed(kernel, quantity):
    """Exercise material apply paths outside the standard CPU lane fixtures."""
    from rfx.core.yee import (init_materials, init_state, precompute_coeffs,
                              update_e_aniso, update_he_fast)

    shape = (4, 4, 4)
    ramp = jnp.arange(64, dtype=jnp.float32).reshape(shape) / 64
    materials = init_materials(shape)._replace(
        eps_r=2 + ramp, sigma=0.2 + ramp / 10, mu_r=1 + ramp)
    state = init_state(shape)._replace(ex=ramp, ey=ramp * 2, ez=ramp * 3,
                                      hx=ramp / 100, hy=ramp / 200, hz=ramp / 300)

    @jax.jit
    def advance(mats, fields):
        if kernel == "precompute":
            coeffs = precompute_coeffs(mats, 1e-12, MM)
            result = update_he_fast(fields, coeffs)
        else:
            result = update_e_aniso(fields, mats, mats.eps_r, mats.eps_r * 1.1,
                                    mats.eps_r * 1.2, 1e-12, MM)
        return jnp.stack([getattr(result, component)
                          for component in ("ex", "ey", "ez", "hx", "hy", "hz")])

    plain = np.asarray(advance(materials, state))
    with _realized.capture() as saved:
        observed = np.asarray(advance(materials, state))
    site = "aniso.E" if kernel == "aniso" else (
        "precompute.H" if quantity == "mu_h" else "precompute.E")
    records = [r for r in saved.records if r["site"] == site and quantity in r]
    if len(records) != 1:
        raise RuntimeError(f"expected one record at {site} for {quantity}")

    def replay(factor):
        def feed_back(actual_site, actual_quantity, arrays):
            if (actual_site, actual_quantity) != (site, quantity):
                return arrays
            return tuple(jnp.where(jnp.all(value == jnp.asarray(stored)),
                                   jnp.asarray(stored) * factor,
                                   jnp.full_like(value, jnp.nan))
                         for value, stored in zip(arrays, records[0][quantity]))
        return feed_back

    with _realized.capture(transform=replay(1.0)):
        unchanged = np.asarray(advance(materials, state))
    with _realized.capture(transform=replay(1.01)):
        moved = np.asarray(advance(materials, state))
    assert plain.tobytes() == observed.tobytes(), "observation changed kernel output"
    assert observed.tobytes() == unchanged.tobytes(), "saved replay changed kernel output"
    assert np.isfinite(moved).all()
    assert np.any(moved != unchanged)
    print(f"J4 kernel {site} {quantity} unchanged_bits=True "
          f"max_field_delta={np.max(np.abs(moved-unchanged)):.9g}", file=sys.stderr)

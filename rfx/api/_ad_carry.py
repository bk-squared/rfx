"""Read real forward scan avals; exclude declared linear accumulator leaves."""
from __future__ import annotations

from collections import defaultdict, deque
import math

import jax
import jax.numpy as jnp

from rfx.api._ad_memory_types import ADMemoryUnavailable


def linear_accumulators(sim, grid, n_steps):
    """Abstract shapes from the features' own linear-accumulator initializers.

    Forward's default request has no port_s11_freqs, hence no in-scan port
    DFTs. Port histories and probe time records stay saved, conservatively.
    An unrecognised accumulator also stays saved, never silently excluded.
    """
    records, warnings = [], []

    def add(name, initialize):
        try:
            state = jax.eval_shape(initialize)
        except Exception as error:
            path = "graded" if sim._uses_nonuniform_mesh else "uniform"
            warnings.append(f"Linear accumulator {name.replace('_', ' ')} on the {path} path "
                            f"could not be initialized ({type(error).__name__}: {error}); "
                            "its real carry leaves remain saved.")
            return
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
            records.append(dict(name=name + jax.tree_util.keystr(path),
                                shape=list(leaf.shape), dtype=str(leaf.dtype),
                                bytes=math.prod(leaf.shape) * leaf.dtype.itemsize))

    dtype = jnp.float64 if sim._precision == "float64" else jnp.float32
    if sim._ntff is not None:
        from rfx.farfield import make_ntff_box, init_ntff_data
        lo, hi, freqs = sim._ntff
        add("ntff", lambda: init_ntff_data(make_ntff_box(grid, lo, hi, freqs), field_dtype=dtype))
    if sim._dft_planes:
        from rfx.measurement.setup import build_planes
        add("dft_planes", lambda: tuple(p.accumulator for p in build_planes(sim, grid, n_steps)))
    if sim._flux_monitors:
        from rfx.measurement.flux_weights import build_flux_monitor_cfgs
        add("flux_monitors", lambda: tuple((m.e1_dft, m.e2_dft, m.h1_dft, m.h2_dft)
            for m in build_flux_monitor_cfgs(sim, grid, n_steps, periodic=sim._periodic_flags())))
    return records, warnings


def scan_carry_invars(eqn):
    """The carry operands of a ``scan`` equation, in either form of its parameters.

    jax 0.10 counts the operands (``num_consts``, ``num_carry``); jax 0.11
    gives their tree instead (``ft_in`` = consts, carry, xs).
    """
    params = eqn.params
    if "ft_in" in params:
        consts, carry, _ = params["ft_in"].unpack()
        start, count = len(consts), len(carry)
    else:
        start, count = params["num_consts"], params["num_carry"]
    return list(eqn.invars[start:start + count])


def _scans(jaxpr):
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "scan":
            yield [v.aval for v in scan_carry_invars(eqn)]
        for value in eqn.params.values():
            if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                yield from _scans(value)


def read_carry(sim, grid, n_steps, *, design_box, design, shape, checkpoint,
               checkpoint_every, checkpoint_segments, n_warmup, replay_tables=None):
    """Trace the requested forward with abstract overrides, without executing it."""
    try:
        if grid is None:
            raise ValueError("grid could not be built")
        names = (("design_occupancy_override",) if design_box is not None and design == "occupancy"
                 else ("pec_occupancy_override",) if design == "occupancy"
                 else ("design_eps_override", "design_sigma_override") if design_box is not None
                 else ("eps_override", "sigma_override"))
        inputs = tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for _ in names)

        def forward(*values):
            result = sim.forward(n_steps=n_steps, checkpoint=checkpoint,
                checkpoint_every=checkpoint_every, checkpoint_segments=checkpoint_segments,
                n_warmup=n_warmup, design_box=design_box, skip_preflight=True,
                **dict(zip(names, values)))
            if replay_tables is not None:
                replay_tables.extend(replay_table_accounting(result))
            return result.time_series, result.ntff_data

        traced = jax.make_jaxpr(forward)(*inputs)
        # Auxiliary waveform scans do not own six full-grid E/H arrays.
        candidates = [leaves for leaves in _scans(traced)
                      if sum(tuple(a.shape) == tuple(grid.shape) for a in leaves) >= 6]
        carry = max(candidates, key=lambda leaves: len(leaves))
        declared = defaultdict(deque)
        accumulators, warnings = linear_accumulators(sim, grid, n_steps)
        for record in accumulators:
            declared[tuple(record["shape"]), record["dtype"]].append(record)
        saved, excluded = [], []
        for index, leaf in enumerate(carry):
            key = tuple(leaf.shape), str(leaf.dtype)
            if declared[key]:
                excluded.append(dict(declared[key].popleft(), carry_index=index))
            else:
                saved.append(dict(name=f"scan_carry[{index}]", shape=list(leaf.shape),
                                  dtype=str(leaf.dtype), bytes=math.prod(leaf.shape) * leaf.dtype.itemsize,
                                  carry_index=index))
        return saved, excluded, warnings
    except Exception as error:
        raise ADMemoryUnavailable(f"AD memory carry could not be read: {error}") from error


def replay_table_accounting(result):
    """Read post-scan replay sizes from the configs realized by forward.

    extract_waveguide_sparams replays reference and probe V/I separately.
    Each replay keeps its own weight table, including the two E-stamped
    records and the two H-stamped records; equal stamps do not share leaves.
    Plane/flux DFTs and requested lumped/wire spectra accumulate in-scan.
    """
    from rfx.measurement.dft import _BLOCK

    size = 0
    for cfg in (result.waveguide_ports or {}).values():
        for name in ("v_ref_t", "i_ref_t", "v_probe_t", "i_probe_t"):
            record = getattr(cfg, name)
            padded = math.ceil(record.shape[0] / _BLOCK) * _BLOCK
            dtype = jnp.result_type(record.dtype, jnp.complex64)
            size += padded * cfg.freqs.size * dtype.itemsize
    return [("waveguide_port", size)] if result.waveguide_ports else []

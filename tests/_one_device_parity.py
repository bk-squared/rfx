"""Independent one-device measurements for multi-device parity tests (uniform forward media, graded run poles, graded derivatives, uniform directivity).

RFX_PARITY_REPORT optionally appends JSON measurements, including the two
negative controls. No solver helper supplies an expected field or derivative.
"""
import itertools
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rfx import Box, DebyePole, LorentzPole, Simulation
from rfx._grid_metric import field_index
from rfx.boundaries.spec import Boundary, BoundarySpec

STEPS = 48
COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")
PROFILE = np.array([1.] * 4 + [.9, 1.1, .9, 1.1] + [1.] * 4) * 1e-3


def record(metadata, mutation, measurements):
    row = dict(metadata, mutation=mutation, **measurements)
    print("UNVERIFIED " + json.dumps(row), flush=True)
    target = os.environ.get("RFX_PARITY_REPORT")
    if target:
        with Path(target).open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    return row


def scene(kind, *, boundary="pec", graded=False, drop=False, seam=None):
    faces = (BoundarySpec(x="pec", y=Boundary("pmc", "pec"), z="pec")
             if kind == "pmc" and not drop else boundary)
    sim = Simulation(freq_max=15e9, domain=(.012, .008, .008), dx=.001,
                     boundary=faces, cpml_layers=2,
                     **({"dx_profile": PROFILE} if graded else {}))
    source_x = .003 if seam is None else seam
    sim.add_source((source_x, .004, .004), "ez", amplitude_kind="field",
                   waveform=lambda t: jnp.exp(-((t - 8e-12) / 3e-12) ** 2))
    for pos, component in (((.009, .004, .004), "ez"),
                           ((.006, .003, .004), "ez"),
                           ((.004, .005, .003), "hy"),
                           ((source_x, .004, .004), "ez")):
        sim.add_probe(pos, component)
    poles = {}
    if kind == "debye" and not drop:
        poles["debye_poles"] = [DebyePole(delta_eps=1., tau=1e-11)]
    if kind == "lorentz" and not drop:
        poles["lorentz_poles"] = [LorentzPole(
            omega_0=2*np.pi*8e9, delta=2*np.pi*1e9, kappa=(2*np.pi*8e9)**2)]
    if kind in ("debye", "lorentz"):
        sim.add_material("pole", eps_r=3., **poles)
        # Continues through both x CPML faces, as in the accumulated-law scene.
        sim.add(Box((-1., .002, .002), (1., .006, .006)), material="pole")
    if kind == "film" and not drop:
        sim.add_thin_conductor(Box((.006, .002, .002), (.006, .006, .006)),
                               sigma_bulk=1e3, thickness=1e-4)
    return sim


def field_probes(sim, *, graded=False, magnetic=False, axes=None):
    """Record every distinct physical E cell, plus H for the exact-field bar.

    Positions come from the declared mesh, not from a field implementation.
    field_index is used only to deduplicate probes and label first divergences.
    The first four user probes are reserved for the independently computed DFT.
    """
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    if axes is None:
        xs = np.r_[0., np.cumsum(PROFILE)] if graded else np.arange(13) * .001
        axes = (xs, np.arange(9) * .001, np.arange(9) * .001)
    labels = []
    for component in COMPONENTS if magnetic else COMPONENTS[:3]:
        seen = set()
        for position in itertools.product(*axes):
            index = field_index(grid, position, component)
            if index not in seen:
                seen.add(index)
                sim.add_probe(tuple(float(v) for v in position), component)
                labels.append((component, *index))
    return labels


def perturb_cell(sim, n_devices, *, graded=False):
    """A declared +1% epsilon cell at the first slab cut, only on the actual side."""
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    seam = (grid.shape[0] + n_devices - 1) // n_devices
    x = grid.node_of("x", seam)
    # Base material is eps_inf=3 in the pole block, vacuum otherwise.
    base = sim._materials.get("pole")
    eps = base.eps_r if base is not None else 1.
    poles = (dict(debye_poles=base.debye_poles, lorentz_poles=base.lorentz_poles)
             if base is not None else {})
    sim.add_material("seam-change", eps_r=eps * 1.01, **poles)
    sim.add(Box((x, .004, .004), (x + grid.cells("x")[seam], .005, .005)),
            material="seam-change")
    return seam


def trace_metrics(one, multi, dt, labels, *, exact=False):
    """The predeclared accumulated law and a direct NumPy probe DFT."""
    a, b = np.asarray(one, np.float64), np.asarray(multi, np.float64)
    assert a.shape == b.shape and a.shape[1] == 4 + len(labels)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    delta = np.abs(b - a)
    field_delta = delta[:, 4:].max(axis=1)
    allowance = 1.2e-7 * np.cumsum(np.abs(a[:, 4:]).max(axis=1))
    ratio = np.divide(field_delta, allowance,
                      out=np.where(field_delta > 0, np.inf, 0.), where=allowance > 0)
    kernel = np.exp(-2j*np.pi*np.outer(np.linspace(2e9, 10e9, 20),
                                     np.arange(len(a))*dt)) * dt
    expected, actual = kernel @ a[:, :4], kernel @ b[:, :4]
    peak = np.abs(expected).max()
    spectral = float(np.abs(actual - expected).max() / peak) if peak else float("inf")
    first = None
    indices = np.argwhere(delta != 0)
    if len(indices):
        step, column = (int(v) for v in indices[0])
        first = dict(step=step + 1, cell=labels[column - 4] if column >= 4 else ["probe", column],
                     one=float(a[step, column]), multi=float(b[step, column]),
                     difference=float(delta[step, column]), allowance=float(allowance[step]))
    violations = np.flatnonzero(field_delta > allowance)
    first_violation = None
    if len(violations):
        step = int(violations[0])
        column = int(delta[step, 4:].argmax()) + 4
        first_violation = dict(step=step + 1, cell=labels[column - 4],
                               one=float(a[step, column]), multi=float(b[step, column]),
                               difference=float(delta[step, column]), allowance=float(allowance[step]))
    equal = bool(np.array_equal(a, b))
    return dict(finite=finite, nonzero=bool(np.any(a[:, 4:])), exact=exact,
                bit_equal=equal, max_abs=float(delta.max()), law_ratio=float(ratio.max()),
                law_bar=1., spectral=spectral, spectral_bar=1e-4,
                first_divergence=first, first_law_violation=first_violation,
                passed=finite and bool(np.any(a[:, 4:])) and
                (equal if exact else bool(ratio.max() <= 1. and spectral <= 1e-4)))


def derivative_metrics(one, multi):
    a, b = np.asarray(one, np.float64), np.asarray(multi, np.float64)
    assert a.shape == b.shape
    peak = float(np.abs(a).max())
    difference = float(np.abs(a - b).max())
    relative = difference / peak if peak else float("inf")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    nonzero = bool(np.any(a) and np.any(b))
    index = tuple(int(v) for v in np.unravel_index(np.abs(a - b).argmax(), a.shape))
    return dict(relative=relative, bar=1e-4, max_abs=difference, peak=peak,
                worst_index=index, one=float(a[index]), multi=float(b[index]),
                finite=finite, nonzero=nonzero, passed=finite and nonzero and relative <= 1e-4)


def assert_multi_device(sim, result=None):
    """The multi-device side really ran on several devices: no one-device fallback.

    A waveguide port or a TFSF source makes every multi-device entry fall back to
    one device (rfx.runners.distributed_v2.one_device_reason); the comparison would
    then pass trivially. None of these models has one, and no result carries the
    fallback record.
    """
    from rfx.runners.distributed_v2 import one_device_reason
    assert one_device_reason(sim) is None, one_device_reason(sim)
    if result is not None:
        codes = [d.code for d in getattr(result, "diagnostics", ())]
        assert "distributed.one_device_fallback" not in codes, codes

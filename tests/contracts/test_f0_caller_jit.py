"""Declaration methods preserve eager records and traced designs under caller jit."""

import ast
import inspect
import re
import textwrap
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation, Sphere
from rfx._declarations import DECLARATIONS
from rfx.errors import AccuracyWarning, RfxRefusal
from rfx.runners._admission import ADMITS
from tests.contracts.path_equivalence.builders import BUILDERS, build
from tests.unit.autodiff.test_pinned_sheet_on_traced_mesh import (
    D_MIN, DT_PIN, DX, K_MID, N_X, N_Y, N_Z, Z_PATCH, _prof_x,
)


DELTA0 = 0.5 * DX

ALWAYS_ON_WRITERS = {
    "add": "concrete_f0",
    "add_coaxial_port": ("_coaxial_ports", "coax_port"),
    "add_current_moment_monitor": ("_current_moments", "block_moments"),
    "add_floquet_port": ("_floquet_ports", "floquet_port"),
    "add_flux_monitor": ("_flux_monitors", "flux"),
    "add_lumped_rlc": ("_lumped_rlc", "series_RL"),
    "add_material": "concrete_f0",
    "add_msl_port": ("_msl_ports", "msl_port"),
    "add_ntff_box": ("_ntff", "ntff_box"),
    "add_pinned_sheet": ("_pinned_sheets", "pec_sheet"),
    "add_port": ("_ports", "passive_port"),
    "add_probe": "concrete_f0",
    "add_refinement": ("_refinement", "slab"),
    "add_source": "concrete_f0",
    "add_tfsf_source": ("_tfsf", "plane_wave"),
    "add_thin_conductor": "concrete_f0",
    "add_vector_probe": "add_vector_probe",
    "add_wall_loss": "add_wall_loss",
    "add_waveguide_port": ("_waveguide_ports", "waveguide_port"),
}
ALWAYS_ON_MATRIX_ROWS = {
    row for row in ALWAYS_ON_WRITERS.values() if isinstance(row, tuple)
} | {("_waveguide_ports", "n_modes")}
EXTRA_WRITER_CALLS = {
    "add_vector_probe": (("_probes", "probe"), ((.003, .002, .002),), {}),
    "add_polarized_source": (
        ("_probes", "probe"), ((.003, .002, .002),),
        dict(polarization=(1., 2.), amplitude_kind="current"),
    ),
}
SLOW_DELEGATE_ROWS = {
    "add_dft_plane_probe": ("_dft_planes", "dft_plane"),
    "add_polarized_source": "add_polarized_source",
}


def _series(widths, sheet=True, f0=10e9):
    sim = Simulation(
        freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
        dx=DX, dx_profile=widths, cpml_layers=0, boundary="pec",
        nominal_profiles=({"x": _prof_x(DELTA0)}
                          if isinstance(widths, jax.core.Tracer) else {}),
        dt=float(DT_PIN), dt_min_cell=float(D_MIN), snap="declared",
    )
    sim.add_material("diel", eps_r=4.0)
    sim.add_source((3 * DX, 8 * DX, K_MID * DX), "ez", amplitude_kind="current")
    sim.add_probe((20 * DX, 8 * DX, K_MID * DX), "ez")
    if sheet:
        sim.add_thin_conductor(
            Box((8 * DX, 5 * DX, Z_PATCH), (17.1 * DX, 11 * DX, Z_PATCH)),
            surface_impedance_f0=f0,
        )
    sim.add(Cylinder((18 * DX, 8 * DX, 16 * DX), 1.6 * DX, 2 * DX), material="diel")
    sim.add(Sphere((5 * DX, 8 * DX, 16 * DX), 1.6 * DX), material="pec")
    return sim.forward(n_steps=40).time_series


def _peak_bar(series):
    return 9 * float(np.spacing(np.float32(np.max(np.abs(series)))))


def _rs_text(caught):
    return [re.search(r"Rs0 = .*? ohm/sq", str(w.message)).group(0)
            for w in caught if isinstance(w.message, AccuracyWarning)
            and "Leontovich surface-" in str(w.message)]


def test_concrete_caller_jit_series_and_warning():
    """Row (i): nine float32 ULP at the eager series peak, identical Rs0 text."""
    def g(amplitude):
        return amplitude * _series(_prof_x(DELTA0))

    with warnings.catch_warnings(record=True) as eager_warnings:
        warnings.simplefilter("always")
        eager = np.asarray(g(1.0))
    with warnings.catch_warnings(record=True) as jit_warnings:
        warnings.simplefilter("always")
        compiled = np.asarray(jax.jit(g)(1.0))
    error, bar = float(np.max(np.abs(compiled - eager))), _peak_bar(eager)
    print(f"ROW i: max_abs_error={error!r}, bar={bar!r}, peak={np.max(np.abs(eager))!r}", file=sys.stderr)
    assert np.isfinite(eager).all() and np.isfinite(compiled).all()
    assert error <= bar
    eager_rs, jit_rs = _rs_text(eager_warnings), _rs_text(jit_warnings)
    print(f"ROW i warning: eager={eager_rs!r}, jit={jit_rs!r}", file=sys.stderr)
    assert len(eager_rs) == 1
    assert jit_rs == eager_rs


def test_traced_widths_caller_jit_loss_and_gradient():
    """Row (ii): summed loss and its width gradient have relative bar 1e-4."""
    def loss(widths):
        return jnp.sum(_series(widths) ** 2)

    widths = jnp.asarray(_prof_x(DELTA0), dtype=jnp.float32)
    eager_loss, eager_grad = jax.value_and_grad(loss)(widths)
    jit_loss, jit_grad = jax.jit(jax.value_and_grad(loss))(widths)
    for name, actual, expected in (
        ("loss", jit_loss, eager_loss), ("gradient", jit_grad, eager_grad),
    ):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        bar = 1e-4 * np.abs(expected.astype(np.float64))
        print(f"ROW ii {name}: eager={expected.tolist()!r}, jit={actual.tolist()!r}, "
              f"abs_error={error.tolist()!r}, bar={bar.tolist()!r}", file=sys.stderr)
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        assert np.all(error <= bar)


def test_eager_sheet_anchor():
    """A missing sheet changes the eager series by more than 100 row (i) bars."""
    eager = np.asarray(_series(_prof_x(DELTA0)))
    absent = np.asarray(_series(_prof_x(DELTA0), sheet=False))
    difference = float(np.max(np.abs(eager - absent)))
    bar = 100 * _peak_bar(eager)
    print(f"ANCHOR: max_abs_difference={difference!r}, bar={bar!r}", file=sys.stderr)
    assert np.isfinite(eager).all() and np.isfinite(absent).all()
    assert difference > bar


def _assert_relative(name, actual, expected):
    """Sum and derivative bars use only the corresponding eager reading."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bar = 1e-4 * np.abs(expected.astype(np.float64))
    print(f"{name}: eager={expected.tolist()!r}, jit={actual.tolist()!r}, "
          f"abs_error={error.tolist()!r}, bar={bar.tolist()!r}", file=sys.stderr)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    assert np.all(error <= bar)


def _eps_loss(eps):
    sim = build(("_probes", "probe"), "fwd_uniform")
    sim.add_material("block", eps_r=eps)
    return jnp.sum(sim.forward(n_steps=15).time_series ** 2)


def test_traced_eps_through_declaration():
    """A traced material value keeps its derivative through the wrapped body."""
    loss = _eps_loss
    eps = jnp.asarray(2.5, dtype=jnp.float32)
    eager_loss, eager_grad = jax.value_and_grad(loss)(eps)
    jit_loss, jit_grad = jax.jit(jax.value_and_grad(loss))(eps)
    _assert_relative("TRACED eps loss", jit_loss, eager_loss)
    _assert_relative("TRACED eps gradient", jit_grad, eager_grad)
    assert np.asarray(eager_grad) != 0


@pytest.mark.slow
def test_traced_eps_vmap_through_declaration():
    """Batching the same material scene agrees with two eager calls."""
    loss = _eps_loss
    values = jnp.asarray([2., 3.], dtype=jnp.float32)
    eager_values = np.asarray([loss(2.), loss(3.)])
    _assert_relative("TRACED eps vmap", jax.vmap(loss)(values), eager_values)


@pytest.mark.slow
def test_traced_f0_through_declaration():
    """A traced sheet frequency remains differentiable inside the wrapped body."""
    def loss(f0):
        return jnp.sum(_series(_prof_x(DELTA0), f0=f0) ** 2)

    f0 = jnp.asarray(10e9, dtype=jnp.float32)
    eager_loss, eager_grad = jax.value_and_grad(loss)(f0)
    jit_loss, jit_grad = jax.jit(jax.value_and_grad(loss))(f0)
    _assert_relative("TRACED f0 loss", jit_loss, eager_loss)
    _assert_relative("TRACED f0 gradient", jit_grad, eager_grad)
    assert np.asarray(eager_grad) != 0


def test_wall_loss_caller_jit_and_anchor():
    """All-PEC face loss gives fields and exceeds 100 nine-ULP bars.

    Use the 12 mm box and face parameters from test_wall_loss_copies.py;
    120 steps accumulate a loss effect above the anchor bar.
    """
    def series(wall_loss=True):
        sim = Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.0015, boundary="pec")
        sim.add_source((.003, .006, .006), "ez", amplitude_kind="current")
        sim.add_probe((.009, .006, .006), "ez")
        if wall_loss:
            sim.add_wall_loss(sigma_bulk=5.8e4, f0=8e9, faces=("y_lo", "y_hi"))
        return sim.forward(n_steps=120).time_series

    eager = np.asarray(series())
    compiled = np.asarray(jax.jit(series)())
    absent = np.asarray(series(False))
    error, bar = float(np.max(np.abs(compiled - eager))), _peak_bar(eager)
    difference = float(np.max(np.abs(eager - absent)))
    print(f"WALL LOSS: shape={eager.shape!r}, peak={np.max(np.abs(eager))!r}, "
          f"max_abs_error={error!r}, bar={bar!r}, "
          f"anchor_difference={difference!r}, anchor_bar={100 * bar!r}", file=sys.stderr)
    assert np.isfinite(eager).all() and np.isfinite(compiled).all() and np.isfinite(absent).all()
    assert error <= bar
    assert difference > 100 * bar


def test_subclass_declaration_descriptors_and_signatures():
    """Static and class descriptors retain binding on the instance and class."""
    class Subclass(Simulation):
        @staticmethod
        def add_helper(value):
            return value

        @classmethod
        def add_cfg(cls, value):
            return cls, value

    sim = Subclass(freq_max=10e9, domain=(.01,) * 3, dx=.001, boundary="pec")
    for receiver in (sim, Subclass):
        assert receiver.add_helper(3) == 3
        assert receiver.add_cfg(3) == (Subclass, 3)
    assert isinstance(inspect.getattr_static(Subclass, "add_helper"), staticmethod)
    assert isinstance(inspect.getattr_static(Subclass, "add_cfg"), classmethod)
    for name in dir(Simulation):
        if name == "add" or name.startswith("add_"):
            method = getattr(Simulation, name)
            assert inspect.signature(method) == inspect.signature(inspect.unwrap(method))
            assert getattr(Subclass, name) is method
    print("SUBCLASS: static/class calls preserve binding; all declaration signatures preserved", file=sys.stderr)


def _declaration_writes(method):
    """Find direct stores and container mutations independently of method names."""
    attributes = dict(DECLARATIONS)
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(method))))

    def root(node):
        while isinstance(node, ast.Subscript):
            node = node.value
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self" and node.attr in attributes):
            return node.attr
        return None

    writes = set()
    mutations = {"append", "extend", "insert", "remove", "pop", "clear", "sort",
                 "reverse", "update", "setdefault", "popitem", "add", "discard",
                 "difference_update", "intersection_update", "symmetric_difference_update",
                 "__setitem__", "__delitem__", "__iadd__", "__imul__", "__ior__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    attribute = root(part)
                    if attribute:
                        writes.add(attribute)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in mutations):
            attribute = root(node.func.value)
            if attribute:
                writes.add(attribute)
    return sorted(writes)


def test_declaration_wrapper_census():
    """Every declaration writer and every public add method carries the marker."""
    methods = {name: getattr(Simulation, name) for name in dir(Simulation)
               if not name.startswith("_") and callable(getattr(Simulation, name))}
    census = {name: writes for name, method in methods.items()
              if (writes := _declaration_writes(method))}
    declarations = {name for name in methods if name == "add" or name.startswith("add_")}
    print(f"CENSUS count={len(census)}: {census!r}", file=sys.stderr)
    wrapped = sorted(name for name, method in methods.items()
                     if getattr(method, "_rfx_declaration_setup", False))
    print(f"WRAPPED count={len(wrapped)}: {wrapped!r}", file=sys.stderr)
    assert not census.keys() - declarations, "STOP: declaration writer outside add/add_* rule"
    assert not (census.keys() | declarations) - set(wrapped)
    assert not census.keys() - ALWAYS_ON_WRITERS.keys(), "declaration writer needs an always-on row"
    covered = ALWAYS_ON_WRITERS | SLOW_DELEGATE_ROWS
    assert not set(wrapped) - covered.keys(), "wrapped method needs an always-on or slow row"
    assert set(SLOW_DELEGATE_ROWS.values()) <= BUILDERS.keys() | EXTRA_WRITER_CALLS.keys()
    assert ALWAYS_ON_MATRIX_ROWS <= BUILDERS.keys()
    assert set(ALWAYS_ON_WRITERS.values()) <= (
        BUILDERS.keys() | EXTRA_WRITER_CALLS.keys() | {"concrete_f0", "add_wall_loss"})


@pytest.mark.parametrize("row", [
    pytest.param(row, marks=() if row in ALWAYS_ON_MATRIX_ROWS else pytest.mark.slow)
    for row in BUILDERS
], ids=lambda row: ":".join(row))
def test_declaration_caller_jit_matrix(row):
    """Fifteen is the shortest shared record reaching both probes in every row.

    The limiting TF/SF plane wave first reaches probe two at index 14;
    the closed box reaches it at 13, and the other rows by index 8.
    """
    lane = "fwd_uniform" if row in ADMITS["fwd_uniform"] else "fwd_nonuniform"

    def g(amplitude):
        return amplitude * build(row, lane).forward(n_steps=15).time_series

    _assert_caller_jit(g, row, lane, require_arrivals=True)


@pytest.mark.parametrize("method", [
    pytest.param(method, marks=pytest.mark.slow if method in SLOW_DELEGATE_ROWS else ())
    for method in EXTRA_WRITER_CALLS
])
def test_declaration_caller_jit_extra_writer(method):
    """Cover declarations absent from BUILDERS on their uniform scene rows.

    Vector position: test_series_rlc_edge_coupling.py's boxed series element.
    Polarized source: real Jones weights exercise the delegate conversions.
    """
    row, args, kwargs = EXTRA_WRITER_CALLS[method]

    def g(amplitude):
        sim = build(row, "fwd_uniform")
        getattr(sim, method)(*args, **kwargs)
        return amplitude * sim.forward(n_steps=15).time_series

    _assert_caller_jit(g, method, "fwd_uniform")


def _assert_caller_jit(g, row, lane, *, require_arrivals=False):
    """Compare finite fields at nine peak ULP or the complete refusal reading."""
    try:
        eager = np.asarray(g(1.0))
    except (RfxRefusal, ValueError, NotImplementedError) as refusal:
        print(f"MATRIX {row!r} {lane}: eager={type(refusal).__name__}: {refusal}", file=sys.stderr)
        with pytest.raises(type(refusal)) as caught:
            np.asarray(jax.jit(g)(1.0))
        print(f"MATRIX {row!r} {lane}: jit={type(caught.value).__name__}: {caught.value}", file=sys.stderr)
        assert type(caught.value) is type(refusal)
        assert str(caught.value) == str(refusal)
        return
    arrivals = [np.flatnonzero(np.asarray(probe) != 0).tolist()[:1] for probe in eager.T]
    print(f"MATRIX {row!r} {lane}: eager=fields, shape={eager.shape}, arrivals={arrivals!r}", file=sys.stderr)
    if require_arrivals:
        assert all(arrivals), "record must reach both probes"
    try:
        compiled = np.asarray(jax.jit(g)(1.0))
    except Exception as error:
        print(f"MATRIX {row!r} {lane}: jit={type(error).__name__}: {error}", file=sys.stderr)
        raise
    error, bar = float(np.max(np.abs(compiled - eager))), _peak_bar(eager)
    print(f"MATRIX {row!r} {lane}: jit=fields, max_abs_error={error!r}, bar={bar!r}, "
          f"peak={np.max(np.abs(eager))!r}", file=sys.stderr)
    assert np.isfinite(eager).all() and np.isfinite(compiled).all()
    assert error <= bar

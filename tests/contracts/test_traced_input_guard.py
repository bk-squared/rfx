"""Admission contracts; traces stop before FDTD, with no new entry frame."""

import inspect
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx._traced_inputs import TracedStaticInput, admit_traced_inputs, walk
from rfx.ringdown import RingdownSpec

pytestmark = pytest.mark.source_scan


def transform_call(transform, call):
    if transform == "make_jaxpr":
        return jax.make_jaxpr(call)(1.0)
    if transform == "vmap":
        return jax.vmap(call)(jnp.ones(2))
    if transform == "grad":
        return jax.grad(call)(1.0)
    return jax.jvp(call, (1.0,), (1.0,))


@pytest.mark.parametrize("owner,parameter", [
    ("Simulation.forward", "eps_override"),
    ("Sphere", "radius"),
    ("Simulation.forward", "sigma_override"),
])
@pytest.mark.parametrize("transform", ["make_jaxpr", "vmap", "grad", "jvp"])
def test_t4_nonstatic_tracer_passes(owner, parameter, transform):
    def call(value):
        admit_traced_inputs(owner, {parameter: value})
        return value * 2

    transform_call(transform, call)


def test_t4_concrete_array_passes_static():
    admit_traced_inputs("Simulation.forward", {"n_steps": jnp.array(3.0)})


def test_t4_private_entry_accepts_resolved_spec_but_refuses_traced_count():
    @dataclass
    class ResolvedSpec:
        eps_r: object

    def call(value, *, n_steps):
        admit_traced_inputs("Simulation._forward_from_materials", {
            "design_box": ResolvedSpec(value), "n_steps": n_steps,
        })
        return value * 2

    jax.make_jaxpr(lambda t: call(t, n_steps=10))(1.0)
    with pytest.raises(TracedStaticInput, match=r"Simulation._forward_from_materials\(n_steps="):
        jax.make_jaxpr(lambda t: call(t, n_steps=t))(1.0)


def test_walker_containers_pytrees_cycles_and_depth():
    @dataclass
    class Record:
        value: object

    class Pair(NamedTuple):
        value: object

    @jax.tree_util.register_pytree_node_class
    class Tree:
        def __init__(self, value):
            self.value = value

        def tree_flatten(self):
            return (self.value,), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(children[0])

    for build in (lambda t: (t,), lambda t: [t], lambda t: {"v": t}, Record, Pair, Tree):
        def call(value):
            admit_traced_inputs("Simulation.forward", {"n_steps": build(value)})
            return value * 0
        with pytest.raises(TracedStaticInput, match=r"Simulation.forward\(n_steps="):
            jax.make_jaxpr(call)(1.0)
    root = []
    root.append(root)
    assert len(list(walk(root))) == 5
    marker = object()
    assert any(item is marker for item in walk([[[[marker]]]]))
    assert not any(item is marker for item in walk([[[[[marker]]]]]))


# These four decorators already exist on the base tree. No others may appear.
BASE_WRAPPED = {
    "forward", "_forward_from_materials", "_forward_nonuniform_from_materials",
    "compute_waveguide_s_matrix",
}


def test_t9_only_baseline_callables_have_wrapped():
    inventory = json.loads((Path(__file__).resolve().parents[2] / "docs/guides/api_symbol_inventory.json").read_text())
    names = {"__init__", *inventory["simulation_methods"],
             "_forward_from_materials", "_forward_nonuniform_from_materials"}
    assert {name for name in names if hasattr(getattr(Simulation, name), "__wrapped__")} == BASE_WRAPPED
    for name in BASE_WRAPPED:
        fn = getattr(Simulation, name)
        assert not hasattr(fn.__wrapped__, "__wrapped__"), name
    from tests.contracts.test_traced_input_classification import VALUE_CLASSES
    for cls in VALUE_CLASSES.values():
        for name in ("__init__", "__new__", "__post_init__"):
            if hasattr(cls, name):
                assert not hasattr(getattr(cls, name), "__wrapped__"), (cls, name)


# Literal baseline source and stack depths. The trace callback adds one frame
# to a body-issued stacklevel=2 warning; staging/flux wrappers add their existing
# frame. Independent source/name matching still finds a body hidden by a closure
# without __wrapped__, so adding such a wrapper cannot evade the attribution test.
FRAME_CASES = (
    ("run", "rfx/api/_execute.py", 0),
    ("forward", "rfx/api/_execute.py", 1),
    ("_forward_from_materials", "rfx/runners/uniform_forward.py", 1),
    ("compute_coaxial_line_reflection", "rfx/sparams/coax.py", 0),
    ("compute_coaxial_two_port", "rfx/sparams/coax.py", 0),
    ("compute_coax_msl_transition", "rfx/sparams/coax.py", 0),
    ("compute_msl_s_matrix", "rfx/sparams/msl.py", 0),
    ("compute_mixed_s_matrix", "rfx/sparams/mixed.py", 0),
    ("compute_waveguide_s_matrix", "rfx/sparams/waveguide.py", 1),
    ("add_tfsf_source", "rfx/sources/tfsf_discrete_plane_wave.py", 0),
    ("__post_init__", "rfx/ringdown/__init__.py", 0),
)


@pytest.mark.parametrize("method,source,baseline_frames", FRAME_CASES)
def test_t9_body_warning_keeps_caller(method, source, baseline_frames):
    sim = Simulation(domain=(.01, .01, .01), dx=.001, freq_max=1e9)
    entry = RingdownSpec().__post_init__ if method == "__post_init__" else getattr(sim, method)
    kwargs = {name: None for name, parameter in inspect.signature(entry).parameters.items()
              if parameter.default is inspect.Parameter.empty
              and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}

    class ReachedEntry(Exception):
        pass

    def trace(frame, event, arg):
        if event == "line" and frame.f_code.co_name == method and frame.f_code.co_filename.endswith(source):
            warnings.warn("entry caller diagnostic", UserWarning, stacklevel=2 + 1 + baseline_frames)
            raise ReachedEntry
        return trace

    previous = sys.gettrace()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sys.settrace(trace)
            with pytest.raises(ReachedEntry):
                entry(**kwargs)
        finally:
            sys.settrace(previous)
    warning, = caught
    assert warning.filename == __file__, (method, warning.filename, warning.lineno)

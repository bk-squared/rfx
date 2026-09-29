"""Every JAX matrix product rfx issues runs at full float32 precision (#1364).

On Ampere-class and newer NVIDIA GPUs, JAX's default precision for a float32
or complex64 matrix product is TF32, which rounds the operands to a 10-bit
mantissa. On one RTX 3090 that moved the completed S of the ring-down tests by
up to 4e-4, against float32 bars of 1.5e-5 and 2e-5. With
``JAX_DEFAULT_MATMUL_PRECISION=highest`` or ``NVIDIA_TF32_OVERRIDE=0``, the same
tests on the same card agreed to 1e-6 (VESSL 369367266046; rfx-archive
``rfx/records/20260929-tf32-ab/``). A CPU computes the same result at either
setting, so a numerical test on CPU CI cannot see this defect. The precision is
written into the lowered program, though. These tests read it there, on CPU.

Two checks:

1. Source scan. Every call to a JAX contraction by name (``jnp.einsum``,
   ``jnp.matmul``, ``jnp.dot``, ``jnp.tensordot``, ``jnp.inner``, ``jnp.vdot``,
   ``lax.dot``, ``lax.dot_general``) passes ``precision=``. A scan cannot tell
   ``a @ b`` on JAX arrays from ``a @ b`` on NumPy arrays, so the scan does not
   cover ``@``. The second check does.
2. Lowered programs. The entry points that reach the contractions are lowered
   with ``jax.jit(...).lower(...).as_text()``. Every ``stablehlo.dot_general``
   in the text must carry ``precision = [HIGHEST, HIGHEST]``. This covers ``@``
   on JAX arrays and any product a library routine issues on rfx's behalf.
   JAX 0.4.33, 0.6.2 and 0.10.2 print the attribute identically (checked
   2026-09-29).
"""

from __future__ import annotations

import ast
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

RFX = pathlib.Path(__file__).resolve().parents[2] / "rfx"

_CONTRACTIONS = {
    ("jnp", "einsum"), ("jnp", "matmul"), ("jnp", "dot"), ("jnp", "tensordot"),
    ("jnp", "inner"), ("jnp", "vdot"), ("lax", "dot"), ("lax", "dot_general"),
    ("lax", "conv"), ("lax", "conv_general_dilated"),
    ("lax", "conv_with_general_padding"),
}
_DOT = re.compile(r"stablehlo\.dot_general\b[^\n]*")
_CONV = re.compile(r"stablehlo\.convolution\b[^\n]*")
_CONV_HIGHEST = ("precision_config = [#stablehlo<precision HIGHEST>, "
                 "#stablehlo<precision HIGHEST>]")


def _unprecise_calls(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        # The last two names: jnp.einsum, lax.conv and jax.lax.conv alike.
        owner = (f.value.id if isinstance(f.value, ast.Name)
                 else f.value.attr if isinstance(f.value, ast.Attribute) else None)
        if (owner, f.attr) in _CONTRACTIONS and not any(
                k.arg == "precision" for k in node.keywords):
            lines.append(node.lineno)
    return lines


def _assert_all_highest(text: str, what: str) -> int:
    """Every product and every convolution in ``text`` at HIGHEST; returns their number."""
    dots, convs = _DOT.findall(text), _CONV.findall(text)
    loose = ([d for d in dots if "precision = [HIGHEST, HIGHEST]" not in d]
             + [c for c in convs if _CONV_HIGHEST not in c])
    n = len(dots) + len(convs)
    assert not loose, f"{what}: {len(loose)} of {n} products below HIGHEST: {loose[:3]}"
    return n


def test_every_named_jax_contraction_passes_a_precision():
    bad = {}
    for path in sorted(RFX.rglob("*.py")):
        hits = _unprecise_calls(path.read_text())
        if hits:
            bad[str(path.relative_to(RFX.parent))] = hits
    assert not bad, f"JAX contractions without precision= (TF32 on Ampere GPUs, #1364): {bad}"


def test_the_scan_sees_a_contraction_without_precision():
    """The scan's own check: a bare call is found, a precise one is not."""
    assert _unprecise_calls("import jax.numpy as jnp\nx = jnp.einsum('i,i', a, b)\n") == [2]
    assert _unprecise_calls(
        "x = jnp.matmul(a, b, precision=HIGHEST)\ny = np.matmul(a, b)\n") == []
    assert _unprecise_calls("y = jax.lax.conv(a, k, (1, 1), 'SAME')\n") == [1]


def test_the_lowered_check_reads_the_precision():
    a = jnp.ones((3, 4), jnp.float32)
    b = jnp.ones((4, 2), jnp.float32)
    plain = jax.jit(lambda a, b: a @ b).lower(a, b).as_text()
    with pytest.raises(AssertionError, match="below HIGHEST"):
        _assert_all_highest(plain, "plain @")
    high = jax.jit(lambda a, b: jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)).lower(a, b)
    assert _assert_all_highest(high.as_text(), "HIGHEST matmul") == 1


# ---------------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------------

def test_ring_down_completion_inside_forward():
    """``forward(ringdown=)``: the plain DFT, the tail and the residue solve."""
    from rfx.ringdown import RingdownSpec
    from tests.unit.sparams.test_ringdown_run import FREQS, _box
    sim = _box("uniform")
    shape = tuple(sim._build_grid().shape)

    def f(e):
        r = sim.forward(n_steps=300, skip_preflight=True, eps_override=e,
                        port_s11_freqs=FREQS, ringdown=RingdownSpec())
        # The completed S, not r.s_params: jit drops what the output does not use.
        return r.ringdown.s_params, r.ringdown.s_params_long

    text = jax.jit(f).lower(jnp.full(shape, 2.2, jnp.float32)).as_text()
    assert _assert_all_highest(text, "forward(ringdown=)") >= 5


def test_current_moment_monitor_and_its_far_field():
    """The in-loop block reduction and the block far field, in one program."""
    from rfx import Simulation, current_moment_far_field
    sim = Simulation(freq_max=12e9, domain=(12e-3, 12e-3, 12e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=6)
    sim.add_source((6e-3, 6e-3, 6e-3), "ex")
    sim.add_current_moment_monitor((2e-3, 2e-3, 2e-3), (10e-3, 10e-3, 10e-3),
                                   block_size=4e-3, freqs=np.array([4e9, 5e9]))
    theta, phi = np.array([0.3, 1.2]), np.array([0.4])
    box = ((7e-3, 5e-3, 5e-3), (8e-3, 7e-3, 7e-3))   # 2 x 3 x 3 cells, inside the slab

    def f(eps_r):
        fr = sim.forward(design_box=box, design_eps_override=jnp.ones((2, 3, 3)) * eps_r,
                         n_steps=20, checkpoint=False, skip_preflight=True)
        ff = current_moment_far_field(fr, theta, phi)
        return jnp.sum(jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2)

    text = jax.jit(f).lower(jnp.float32(2.0)).as_text()
    assert _assert_all_highest(text, "current-moment monitor") >= 4


def test_ntff_far_field():
    from rfx import Simulation
    from rfx.farfield import compute_far_field_jax
    sim = Simulation(freq_max=12e9, domain=(12e-3, 12e-3, 12e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_source((6e-3, 6e-3, 6e-3), "ez")
    sim.add_ntff_box((2.5e-3, 2.5e-3, 2.5e-3), (9.5e-3, 9.5e-3, 9.5e-3),
                     freqs=np.array([5e9, 6e9]))
    grid = sim._build_grid()
    theta, phi = jnp.array([0.3, 1.2]), jnp.array([0.0, 0.7])

    # The box comes from a concrete run: forward()'s own box carries traced
    # frequencies under jit (#1364's other half), so the transform is lowered
    # over the accumulated data with the box held fixed.
    fr = sim.forward(n_steps=10, skip_preflight=True)

    def f(data):
        ff = compute_far_field_jax(data, fr.ntff_box, grid, theta, phi)
        return jnp.sum(jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2)

    text = jax.jit(f).lower(fr.ntff_data).as_text()
    assert _assert_all_highest(text, "NTFF far field") >= 1


def test_port_extraction_cores():
    """The traced cores of the waveguide DFT, the coax S solve and the MSL fit."""
    from rfx.probes.msl_wave_decomp import _lstsq_alpha_gamma
    from rfx.sources.coaxial_port import (_coaxial_line_reflection_jnp,
                                          _solve_two_port_from_wave_amplitudes_jnp)
    from rfx.sources.waveguide_port import _rect_dft

    ts = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)
    freqs = jnp.array([1e9, 2e9], jnp.float32)
    text = jax.jit(lambda y: _rect_dft(y, freqs, 1e-11, 64)).lower(ts).as_text()
    _assert_all_highest(text, "waveguide _rect_dft")

    a = jnp.ones((2, 2, 3), jnp.complex64) + jnp.eye(2, dtype=jnp.complex64)[:, :, None]
    text = jax.jit(lambda a: _solve_two_port_from_wave_amplitudes_jnp(a, 0.5 * a).s_params).lower(a)
    assert _assert_all_highest(text.as_text(), "coax two-port solve") >= 1

    z = jnp.linspace(0.0, 0.01, 5, dtype=jnp.float32)
    v = jnp.ones((5,), jnp.complex64)
    text = jax.jit(lambda v: _coaxial_line_reflection_jnp(
        z, v, reference_plane_m=0.0, D=1e-3, z0=0.0, load_below=False)).lower(v)
    _assert_all_highest(text.as_text(), "coax line reflection")

    x = jnp.linspace(0.0, 0.01, 6, dtype=jnp.float32)
    v6 = jnp.ones((6,), jnp.complex64)
    text = jax.jit(lambda v: _lstsq_alpha_gamma(v, x, jnp.float32(100.0))).lower(v6)
    assert _assert_all_highest(text.as_text(), "MSL two-wave fit") >= 1


def test_ring_down_completion_gradient():
    """The gradient through the completion: QR's own derivative rule issues
    products that a ``precision=`` argument cannot reach (the Opus review of
    this change measured three at DEFAULT before the fix)."""
    from rfx.ringdown import RingdownSpec
    from tests.unit.sparams.test_ringdown_run import FREQS, _box
    sim = _box("uniform")
    shape = tuple(sim._build_grid().shape)

    def loss(e):
        r = sim.forward(n_steps=300, skip_preflight=True, eps_override=e,
                        port_s11_freqs=FREQS, ringdown=RingdownSpec())
        return jnp.sum(jnp.abs(r.ringdown.s_params) ** 2)

    text = jax.jit(jax.grad(loss)).lower(jnp.full(shape, 2.2, jnp.float32)).as_text()
    assert _assert_all_highest(text, "grad of forward(ringdown=)") >= 10


@pytest.mark.parametrize("shape", [(8, 8), (6, 6, 6)])
def test_topology_density_filter(shape):
    """The density filter's convolutions feed the permittivity of a design."""
    from rfx.topology import apply_density_filter

    def filt(r):
        # The filter turns its radius into an int with jnp.ceil: keep that
        # constant arithmetic concrete so the filter can be staged at all.
        with jax.ensure_compile_time_eval():
            return apply_density_filter(r, 2.0)

    text = jax.jit(filt).lower(jnp.ones(shape, jnp.float32)).as_text()
    # The normalising convolution of a constant is evaluated while staging;
    # the one on the density stays in the program.
    assert _assert_all_highest(text, f"density filter {len(shape)}-D") >= 1

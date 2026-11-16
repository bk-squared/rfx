"""Allocate the time step's arrays with extents XLA's update kernels vectorise.

The grid, the materials as built, the CPML profiles, ports, sources and
measurement plans stay physical. The pad is applied once here, after
assembly and before the scan: each field, psi buffer and per-cell operand the
step reads gets ``p`` cells at the high end of each padded axis. The physical
extent travels to every array-end site as a frame (:class:`HighPadFrame`),
each measurement reads the physical slice, and the final state is unpadded
before it leaves the step.

Physical cells and the pad couple only through the high ghost row, whose
update factor (``h_row_factors``) is 0 on every non-periodic high face. The
pad factors are 0 too, so a pad that starts at zero stays at zero and a
physical cell next to it reads the 0 that the zero-extending shift supplies.

Targets: ``"aligned"`` pads z to ``4 * ceil(nz / 4)``; ``"odd"`` pads by one
cell every axis whose length is even. No pad on a periodic (or Bloch) axis,
nor on z of a 2-D grid. ``"auto"`` applies the target the gate table holds
for the device; the table is empty until the step rates are measured, so
``"auto"`` pads nothing today. :func:`force_pad` is the internal override for
tests; it is not a ``run()`` or ``forward()`` argument.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from dataclasses import dataclass

import numpy as np

from rfx.boundaries._face_frame import HighPadFrame

TARGETS = ("off", "aligned", "odd")

#: (device kind, jax version) -> target. Filled only from a measured step
#: rate under the declared choosing rule, never from an estimate. Empty: every
#: device kind is unpadded under ``"auto"``.
_MEASURED_GAIN: dict = {}

_REQUESTED = contextvars.ContextVar("rfx_vector_pad_target", default="auto")
_RECORDER = contextvars.ContextVar("rfx_vector_pad_recorder", default=None)
_SEED = contextvars.ContextVar("rfx_vector_pad_seed", default=None)


@dataclass(frozen=True)
class PadPlan:
    """Physical and allocated extents of one run's step arrays (Python ints)."""

    shape: tuple
    padded: tuple
    pad: tuple
    target: str = "off"

    @property
    def active(self) -> bool:
        return any(self.pad)

    @property
    def frame(self):
        """``{axis: (0, n_phys)}`` for each padded axis."""
        return HighPadFrame({"xyz"[a]: (0, n) for a, (n, p) in
                             enumerate(zip(self.shape, self.pad)) if p})

    @property
    def physical(self) -> tuple:
        return tuple(slice(0, n) for n in self.shape)


def plan_pad(shape, *, target, periodic_axes, two_d=False) -> PadPlan:
    """The pad of ``target`` on a grid of ``shape``.

    ``periodic_axes``: three flags, or a string of axis letters. A periodic
    or Bloch axis is never padded (its wrap reads the physical end), nor z
    on a 2-D grid.
    """
    if target not in TARGETS:
        raise ValueError(f"vector padding target must be one of {TARGETS}, got {target!r}")
    shape = tuple(int(n) for n in shape)
    if isinstance(periodic_axes, str):
        periodic = tuple(a in periodic_axes for a in "xyz")
    else:
        periodic = tuple(bool(v) for v in periodic_axes)
    pad = [0, 0, 0]
    for axis, n in enumerate(shape):
        if periodic[axis] or n < 2 or (axis == 2 and two_d):
            continue
        if target == "aligned" and axis == 2:
            pad[axis] = (-n) % 4
        elif target == "odd" and n % 2 == 0:
            pad[axis] = 1
    pad = tuple(pad)
    return PadPlan(shape, tuple(n + p for n, p in zip(shape, pad)), pad, target)


def _device_kind():
    import jax
    device = jax.devices()[0]
    capability = getattr(device, "compute_capability", None)
    return (device.platform if capability is None else str(capability)), jax.__version__


def requested_target() -> str:
    """The forced target, or ``"auto"``."""
    return _REQUESTED.get()


def resolve_target() -> str:
    """The target a run applies: the forced one, else the gate table's."""
    requested = _REQUESTED.get()
    if requested != "auto":
        return requested
    return _MEASURED_GAIN.get(_device_kind(), "off")


@contextlib.contextmanager
def force_pad(target):
    """Internal override: pad every uniform run traced inside with ``target``.

    Read when a run is set up (trace time). Paths this stage does not pad
    refuse a forced ``"aligned"`` or ``"odd"``.
    """
    if target not in TARGETS:
        raise ValueError(f"force_pad target must be one of {TARGETS}, got {target!r}")
    token = _REQUESTED.set(target)
    try:
        yield
    finally:
        _REQUESTED.reset(token)


@contextlib.contextmanager
def record_plans():
    """Collect the :class:`PadPlan` of every uniform run set up inside (tests)."""
    plans = []
    token = _RECORDER.set(plans)
    try:
        yield plans
    finally:
        _RECORDER.reset(token)


@contextlib.contextmanager
def seed_pad(seed, magnitude=1e30):
    """Fill the pad cells of every field and psi array with finite random values.

    For the leak check only: with correct isolation the physical cells and
    every measurement do not depend on what the pad holds.
    """
    token = _SEED.set((int(seed), float(magnitude)))
    try:
        yield
    finally:
        _SEED.reset(token)


def refuse_if_forced(path):
    """A path this stage does not pad refuses a forced pad; it never ignores it."""
    if _REQUESTED.get() in ("aligned", "odd"):
        raise NotImplementedError(f"vector padding: {path} not padded yet")


def uniform_unsupported(ctx_kwargs, *, stencil_order, field_dtype):
    """Uniform-step features whose operands stage 1 does not pad."""
    import jax.numpy as jnp
    from rfx._internal import _realized
    tensor = ctx_kwargs["materials"].components.tensor
    named = (
        ("UPML", ctx_kwargs["use_upml"]),
        ("stencil_order=4", stencil_order != 2),
        ("TFSF plane-wave source", ctx_kwargs["use_tfsf"]),
        ("Bloch phase", ctx_kwargs["bloch"] is not None),
        ("complex fields", jnp.issubdtype(field_dtype, jnp.complexfloating)),
        ("waveguide ports", ctx_kwargs["use_waveguide_ports"]),
        ("surface-impedance sheets", ctx_kwargs["use_sheet_impedance"]),
        ("design box", ctx_kwargs["use_design_box"]),
        ("design occupancy", ctx_kwargs["use_design_occupancy"]),
        ("mixed Debye and Lorentz poles", ctx_kwargs["use_debye"] and ctx_kwargs["use_lorentz"]),
        ("Kerr nonlinearity", ctx_kwargs["use_kerr"]),
        ("tensor permittivity (subpixel smoothing)", tensor is not None and (
            tensor.eps is not None or tensor.inv_eps is not None
            or tensor.node_smoothing is not None)),
        ("realized-model capture", _realized.ACTIVE is not None),
    )
    return tuple(name for name, present in named if present)


def graded_unsupported(*, use_tfsf, use_waveguide_ports, use_sheet_impedance, use_design_box,
                       aniso_eps, use_debye, use_lorentz, field_dtype):
    """Graded-step features whose operands are not padded."""
    import jax.numpy as jnp
    from rfx._internal import _realized
    named = (
        ("TFSF plane-wave source", use_tfsf),
        ("waveguide ports", use_waveguide_ports),
        ("surface-impedance sheets", use_sheet_impedance),
        ("design box", use_design_box),
        ("tensor permittivity (subpixel smoothing)", aniso_eps is not None),
        ("mixed Debye and Lorentz poles", use_debye and use_lorentz),
        ("complex fields", jnp.issubdtype(field_dtype, jnp.complexfloating)),
        ("realized-model capture", _realized.ACTIVE is not None),
    )
    return tuple(name for name, present in named if present)


def uniform_plan(grid, periodic, *, unsupported=(), path="uniform") -> PadPlan:
    """Resolve this run's plan (recorded for tests); refuse what is not padded.

    ``path`` names the lane in the refusal (``"uniform"``, ``"graded"``).
    """
    target = resolve_target()
    if target != "off" and unsupported:
        if _REQUESTED.get() != "auto":
            raise NotImplementedError(
                f"vector padding: {path} with {', '.join(unsupported)} not padded yet")
        target = "off"
    plan = plan_pad(tuple(grid.shape), target=target, periodic_axes=periodic,
                    two_d=bool(getattr(grid, "is_2d", False)))
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.append(plan)
    return plan


# ---- pad helpers -------------------------------------------------------

def _module(arr):
    import jax.numpy as jnp
    return np if isinstance(arr, (np.ndarray, np.generic)) else jnp


def pad_cells(arr, plan, value):
    """Pad the trailing three (cell) axes of a per-cell array with ``value``.

    A scalar broadcasts and needs no pad. Any other shape is not a per-cell
    operand of this grid and is refused at trace time.
    """
    if arr is None or np.ndim(arr) == 0:
        return arr
    shape = tuple(np.shape(arr))
    if shape[-3:] != plan.shape:
        raise ValueError(
            f"vector padding: operand of shape {shape} is not per-cell on the "
            f"grid {plan.shape}")
    widths = [(0, 0)] * (len(shape) - 3) + [(0, p) for p in plan.pad]
    return _module(arr).pad(arr, widths, constant_values=value)


def pad_axis_vector(vec, axis, plan, value):
    """Pad a 1-D per-axis factor (None stays None).

    ``value="edge"`` repeats the last physical value.
    """
    if vec is None:
        return None
    if np.shape(vec) != (plan.shape[axis],):
        raise ValueError(
            f"vector padding: {'xyz'[axis]} factor of shape {np.shape(vec)} "
            f"does not match the grid's {plan.shape[axis]} cells")
    if isinstance(value, str):
        return _module(vec).pad(vec, (0, plan.pad[axis]), mode=value)
    return _module(vec).pad(vec, (0, plan.pad[axis]), constant_values=value)


def _cells(values, plan, value):
    return None if values is None else tuple(pad_cells(v, plan, value) for v in values)


def _seeded(arr, plan, keep_axis=None):
    """Zero-padded ``arr`` with its pad cells overwritten by the leak seed."""
    import jax.numpy as jnp
    seed = _SEED.get()
    if seed is None:
        return arr
    seed, magnitude = seed
    lead = arr.ndim - 3
    inside = np.ones(arr.shape[lead:], dtype=bool)
    for axis, n in enumerate(plan.shape):
        if axis == keep_axis or arr.shape[lead + axis] == n:
            continue
        index = np.arange(arr.shape[lead + axis]).reshape([-1 if b == axis else 1 for b in range(3)])
        inside &= index < n
    rng = np.random.RandomState(seed + int(np.prod(arr.shape)) % 9973)
    values = (rng.uniform(-1.0, 1.0, size=arr.shape) * magnitude).astype(arr.dtype)
    return jnp.where(jnp.asarray(inside), arr, jnp.asarray(values))


def pad_state(fdtd, plan):
    """FDTDState with every field padded with 0 (or the leak seed)."""
    return fdtd._replace(**{name: _seeded(pad_cells(getattr(fdtd, name), plan, 0.0), plan)
                            for name in ("ex", "ey", "ez", "hx", "hy", "hz")})


def unpad_state(fdtd, plan):
    """The physical cells of a padded FDTDState."""
    sl = plan.physical
    return fdtd._replace(**{name: getattr(fdtd, name)[sl]
                            for name in ("ex", "ey", "ez", "hx", "hy", "hz")})


physical_state = unpad_state


def physical_reader(read, plan):
    """``read`` applied to the physical cells of a (possibly padded) state."""
    if read is None or plan is None:
        return read
    return lambda st: read(physical_state(st, plan))


def _psi_axis(name):
    return "xyz".index(name.split("_")[-1][0])


def pad_cpml_state(cpml_state, plan):
    """Psi buffers keep their face depth; each transverse extent gets the pad."""
    padded = {}
    for name in cpml_state._fields:
        psi = getattr(cpml_state, name)
        axis = _psi_axis(name)
        widths = []
        for b in range(3):
            if b == axis:
                widths.append((0, 0))
                continue
            if psi.shape[b] != plan.shape[b]:
                raise ValueError(f"vector padding: psi {name} of shape {psi.shape} "
                                 f"does not span the grid {plan.shape}")
            widths.append((0, plan.pad[b]))
        padded[name] = _seeded(_module(psi).pad(psi, widths), plan, keep_axis=axis)
    return cpml_state._replace(**padded)


def unpad_cpml_state(cpml_state, plan):
    out = {}
    for name in cpml_state._fields:
        axis = _psi_axis(name)
        out[name] = getattr(cpml_state, name)[tuple(
            slice(None) if b == axis else slice(0, plan.shape[b]) for b in range(3))]
    return cpml_state._replace(**out)


def pad_pole_tree(tree, plan):
    """Pole states and assembly coefficients: 0 in the pad, so its P stays 0."""
    import jax
    return jax.tree_util.tree_map(lambda leaf: pad_cells(leaf, plan, 0.0), tree)


def unpad_pole_tree(tree, plan):
    import jax
    sl = plan.physical
    return jax.tree_util.tree_map(lambda leaf: leaf[(Ellipsis, *sl)], tree)


#: Carry entries that hold no per-cell array (accumulators, records, scalars).
_CARRY_PASS = frozenset({
    "ntff", "current_moments", "dft_planes", "flux_monitors", "waveguide_port_accs",
    "wire_sparam_accs", "lumped_sparam_accs", "wire_refplane_accs", "rlc_states",
    "dft_time_records", "sparam_time_records", "wire_sparams"})


def pad_carry(carry, plan):
    """Pad the scan carry once, before the scan."""
    out = {}
    for key, value in carry.items():
        if key == "fdtd":
            out[key] = pad_state(value, plan)
        elif key == "cpml":
            out[key] = pad_cpml_state(value, plan)
        elif key in ("debye", "lorentz"):
            out[key] = pad_pole_tree(value, plan)
        elif key in _CARRY_PASS:
            out[key] = value
        else:
            raise ValueError(f"vector padding: carry entry {key!r} has no pad rule")
    return out


def unpad_carry(carry, plan):
    """The carry with every per-cell entry back on the physical cells."""
    out = dict(carry)
    if "fdtd" in out:
        out["fdtd"] = unpad_state(out["fdtd"], plan)
    if "cpml" in out:
        out["cpml"] = unpad_cpml_state(out["cpml"], plan)
    for key in ("debye", "lorentz"):
        if key in out:
            out[key] = unpad_pole_tree(out[key], plan)
    return out


def _pad_tensor(tensor, plan):
    if tensor is None:
        return None
    if tensor.eps is not None or tensor.inv_eps is not None or tensor.node_smoothing is not None:
        raise NotImplementedError("vector padding: uniform with tensor permittivity not padded yet")
    return dataclasses.replace(tensor, conformal_weights=_cells(tensor.conformal_weights, plan, 1.0))


def pad_kernel_materials(materials, plan):
    """The step's material view: vacuum in the pad (eps, mu, keep 1; sigma 0)."""
    from rfx.model.materials import KernelComponents
    components = materials.components
    if not isinstance(components, KernelComponents):
        raise NotImplementedError("vector padding: uniform step materials without the kernel view not padded yet")
    for name in ("sigma", "mu_r", "sigma_lumped", "eps_r_lumped", "sigma_film"):
        if getattr(materials, name) is not None:
            raise ValueError(f"vector padding: kernel materials leaf {name!r} has no pad rule")
    return materials._replace(
        eps_r=pad_cells(materials.eps_r, plan, 1.0),
        edge_keep=_cells(materials.edge_keep, plan, 1.0),
        components=KernelComponents(
            _cells(components.eps_update, plan, 1.0),
            _cells(components.sigma_update, plan, 0.0),
            _cells(components.mu_update, plan, 1.0),
            _pad_tensor(components.tensor, plan)))


def _pad_loss(operands, plan):
    if operands is None:
        return None
    eps, sigma, inverse = operands
    return _cells(eps, plan, 1.0), _cells(sigma, plan, 0.0), inverse


def _pad_factors(factors, plan):
    return tuple(pad_axis_vector(f, a, plan, 0.0) for a, f in enumerate(factors))


#: Invariants read on the physical view of the state (index tables).
_INVARIANT_PASS = frozenset({"probe_table", "port_table", "waveguide_table"})


def pad_invariants(invariants, plan):
    """Every per-cell or per-axis operand the uniform step reads, padded.

    A key or leaf without a pad rule is an error at trace time.
    """
    out = {}
    for key, value in invariants.items():
        if key == "ctx":
            out[key] = pad_context(value, plan)
        elif key in ("magnetic_factors", "h_row_factors"):
            out[key] = _pad_factors(value, plan)
        elif key == "cpml_inv_eps_r":
            out[key] = _cells(value, plan, 1.0)
        elif key == "e_loss":
            out[key] = _pad_loss(value, plan)
        elif key == "conformal_zero_masks":
            out[key] = _cells(value, plan, False)
        elif key == "sheet_coeffs":
            if value is not None:
                raise NotImplementedError("vector padding: uniform with surface-impedance sheets not padded yet")
            out[key] = value
        elif key in _INVARIANT_PASS:
            out[key] = value
        else:
            raise ValueError(f"vector padding: step invariant {key!r} has no pad rule")
    return out


def pad_context(ctx, plan):
    """The step context with its per-cell operands padded and the plan attached."""
    for name in ("fast_coeffs", "upml_coeffs", "tfsf_cfg", "design_box", "design_occupancy",
                 "sheet_impedance", "bloch"):
        if getattr(ctx, name) is not None:
            raise NotImplementedError(f"vector padding: uniform step with {name} not padded yet")
    return dataclasses.replace(
        ctx,
        materials=pad_kernel_materials(ctx.materials, plan),
        debye_coeffs=None if ctx.debye_coeffs is None else pad_pole_tree(ctx.debye_coeffs, plan),
        lorentz_coeffs=None if ctx.lorentz_coeffs is None else pad_pole_tree(ctx.lorentz_coeffs, plan),
        pec_edge_masks=_cells(ctx.pec_edge_masks, plan, False),
        kerr_chi3=pad_cells(ctx.kerr_chi3, plan, 0.0),
        pad_plan=plan)


# ---- graded (non-uniform) step -------------------------------------------

def _tree_cells(value, plan, fill):
    """Every per-cell leaf of ``value`` (None, an array or a tuple) padded with ``fill``."""
    import jax
    return jax.tree_util.tree_map(lambda leaf: pad_cells(leaf, plan, fill), value)


def _must_be_absent(value, name):
    if value is not None:
        raise NotImplementedError(f"vector padding: graded step with {name} not padded yet")
    return value


#: Graded invariants read on the physical view of the state (index tables), or
#: not read by the step: ``dx_arr`` reaches only the component rule the step
#: skips when the realized components are present; left physical, any read of
#: it against padded arrays is a shape error at trace time.
_GRADED_PASS = frozenset({"drives", "probe_table", "port_table", "waveguide_table", "dx_arr"})


def pad_graded_invariants(invariants, plan):
    """Every per-cell or per-axis operand the graded step reads, padded.

    H metrics ``inv_d*_h`` get 0 (the zero ghost row reaches into the pad);
    E metrics ``inv_d*`` repeat the last physical value. A key without a pad
    rule is an error at trace time.
    """
    out = {}
    for key, value in invariants.items():
        if key == "inv_dx":
            out[key] = pad_axis_vector(value, 0, plan, "edge")
        elif key == "inv_dx_h":
            out[key] = pad_axis_vector(value, 0, plan, 0.0)
        elif key == "materials":
            out[key] = pad_kernel_materials(value, plan)
        elif key in ("pec_edge_masks", "pec_static_edge_masks"):
            out[key] = _tree_cells(value, plan, False)
        elif key == "pec_occupancy":
            out[key] = _tree_cells(value, plan, 0.0)
        elif key == "cpml_inv_eps_r":
            out[key] = _tree_cells(value, plan, 1.0)
        elif key == "e_loss":
            out[key] = _pad_loss(value, plan)
        elif key in ("debye_coeffs", "lorentz_coeffs"):
            out[key] = None if value is None else pad_pole_tree(value, plan)
        elif key in ("aniso_eps", "sheet_impedance", "sheet_coeffs", "design_box_coeffs"):
            out[key] = _must_be_absent(value, key)
        elif key in _GRADED_PASS:
            out[key] = value
        else:
            raise ValueError(f"vector padding: graded step invariant {key!r} has no pad rule")
    return out


def pad_graded_context(ctx, plan):
    """The graded step context with its y and z metrics padded and the plan attached."""
    for name in ("use_tfsf", "use_waveguide_ports", "use_sheet_impedance", "use_design_box"):
        if getattr(ctx, name):
            raise NotImplementedError(f"vector padding: graded step with {name[4:]} not padded yet")
    if ctx.x_face_frame is not None:
        raise NotImplementedError("vector padding: a graded slab step is padded by its own runner")
    return dataclasses.replace(
        ctx,
        inv_dy=pad_axis_vector(ctx.inv_dy, 1, plan, "edge"),
        inv_dz=pad_axis_vector(ctx.inv_dz, 2, plan, "edge"),
        inv_dy_h=pad_axis_vector(ctx.inv_dy_h, 1, plan, 0.0),
        inv_dz_h=pad_axis_vector(ctx.inv_dz_h, 2, plan, 0.0),
        pad_plan=plan)

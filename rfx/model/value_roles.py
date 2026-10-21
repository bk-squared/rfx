"""Host classifications and runtime validation of declaration VALUE leaves."""
import warnings
import math
from copy import copy
from dataclasses import fields, is_dataclass

import numpy as np

from rfx._traced_check import traced_check
from rfx.core.jax_utils import is_tracer


def is_pec_sigma(sigma, threshold=1e6, name="MaterialSpec.sigma"):
    """Concrete conductors retain their class; traced conductivity is resistive."""
    if not is_tracer(sigma):
        return sigma >= threshold

    def check(value):
        if np.any(np.asarray(value) >= threshold):
            raise ValueError(f"{name}: traced sigma must be below the PEC threshold {threshold:g} S/m")

    traced_check(name, "PEC threshold", check, sigma)
    return False


def is_pec_material(spec, threshold=1e6, name="MaterialSpec"):
    return is_pec_sigma(spec.sigma, threshold, name)


def has_nonzero(value):
    return is_tracer(value) or value != 0.0


def has_kerr(spec):
    return has_nonzero(getattr(spec, "chi3", 0.0))


def has_element(value):
    return is_tracer(value) or value > 0


def is_soft_impedance(value):
    return not is_tracer(value) and value == 0.0


def is_port_impedance(value):
    return is_tracer(value) or value > 0.0


def is_drude(pole):
    return not is_tracer(pole.omega_0) and pole.omega_0 == 0.0


def host_value(value):
    """Keep traced coefficients; retain the concrete path's float arithmetic."""
    return value if is_tracer(value) else float(value)


def nonvacuum_material(spec):
    """A traced coefficient counts as a declared non-vacuum material."""
    return any(is_tracer(v) for v in (spec.eps_r, spec.sigma, spec.mu_r)) or (
        abs(float(spec.eps_r) - 1.0) > 1e-12
        or abs(float(spec.sigma)) > 1e-18
        or abs(float(spec.mu_r) - 1.0) > 1e-12)


def validate_impedance(impedance):
    def check(value):
        if value <= 0:
            raise ValueError(f"impedance must be positive, got {value}")
    traced_check("port impedance", "positive impedance", check, impedance)


def validate_source_scalar(label, value, require_positive):
    def check(operand):
        try:
            numeric = float(operand)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a finite scalar, got {operand!r}") from None
        if not math.isfinite(numeric):
            raise ValueError(f"{label} must be finite, got {operand!r}")
        if require_positive and numeric <= 0:
            raise ValueError(f"{label} must be positive, got {operand}")
    traced_check("waveguide port", label, check, value)


def validate_tfsf_values(f0, bandwidth):
    def check(frequency, width):
        if frequency is not None and frequency <= 0:
            raise ValueError(f"f0 must be positive when provided, got {frequency}")
        if width <= 0:
            raise ValueError(f"bandwidth must be positive, got {width}")
    traced_check("TFSF source", "positive frequency and bandwidth", check, f0, bandwidth)


def validate_surface_sigma(sigma_bulk):
    def check(value):
        if value <= 0.0:
            raise ValueError(
                f"add_thin_conductor: sigma_bulk={value!r} must be "
                "> 0 when surface_impedance_f0 is set — Rs0 = "
                "sqrt(pi*f0*mu0/sigma_bulk) is undefined otherwise.")
    traced_check("ThinConductor.sigma_bulk", "positive surface conductivity", check, sigma_bulk)


def validate_rlc(R, L, C, topology):
    def check(r, l, c):
        if r < 0 or l < 0 or c < 0:
            raise ValueError(f"R, L, C must be non-negative, got R={r}, L={l}, C={c}")
        if r == 0 and l == 0 and c == 0:
            raise ValueError("At least one of R, L, C must be non-zero")
    traced_check("LumpedRLCSpec", "non-negative elements", check, R, L, C)
    present = [name for name, value in (("R", R), ("L", L), ("C", C)) if has_element(value)]
    if topology == "series" and len(present) == 1:
        warnings.warn(
            f"Series topology with single component ({present[0]}) uses material "
            "folding, not the series ADE. Add a second component or use "
            "topology='parallel' to suppress this warning.", stacklevel=4)


def check_traced_values(sim, checker, *args, attributes=None):
    """Run an existing host validation on concrete declaration operands."""
    operands = []

    def encode(value):
        if is_tracer(value):
            operands.append(value)
            return ("operand", len(operands) - 1)
        if is_dataclass(value):
            return ("record", type(value), [(f.name, encode(getattr(value, f.name))) for f in fields(value)])
        if hasattr(type(value), "_fields"):
            return ("record", type(value), [(f, encode(getattr(value, f))) for f in value._fields])
        if isinstance(value, dict):
            return ("mapping", [(k, encode(v)) for k, v in value.items()])
        if isinstance(value, (tuple, list)):
            return ("sequence", type(value), [encode(v) for v in value])
        return ("constant", value)

    def decode(node, values):
        kind, *parts = node
        if kind == "operand":
            return values[parts[0]]
        if kind == "constant":
            return parts[0]
        if kind == "mapping":
            return {k: decode(v, values) for k, v in parts[0]}
        if kind == "record":
            cls, items = parts
            return cls(**{k: decode(v, values) for k, v in items})
        cls, items = parts
        return cls(decode(v, values) for v in items)

    names = attributes or ("_ports", "_msl_ports", "_coaxial_ports", "_waveguide_ports", "_floquet_ports", "_tfsf")
    templates = {name: encode(getattr(sim, name)) for name in names}
    if not operands:
        return False
    original = copy(sim)

    def check(*values):
        concrete = copy(original)
        for name, template in templates.items():
            setattr(concrete, name, decode(template, values))
        checker(concrete, *args)

    traced_check("declaration values", checker.__name__, check, *operands)
    return True

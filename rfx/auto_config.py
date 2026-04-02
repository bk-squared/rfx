"""Automatic simulation configuration from geometry + frequency range.

Derives all simulation parameters (dx, domain, CPML, n_steps, source)
from user-specified geometry and frequency band. Based on validated rules
from Meep/OpenEMS best practices and rfx convergence testing.

Usage
-----
>>> config = auto_configure(
...     geometry=[(Box(...), "FR4"), (Box(...), "pec")],
...     freq_range=(1e9, 4e9),
...     accuracy="standard",
... )
>>> sim = Simulation(**config.to_sim_kwargs())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

C0 = 299792458.0
EPS_0 = 8.854187817e-12


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

class FeatureInfo(NamedTuple):
    """Geometry feature summary for auto-configuration."""
    min_thickness: float      # thinnest dimension (metres)
    max_extent: float         # largest dimension (metres)
    bbox: tuple               # ((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))
    max_eps_r: float          # highest relative permittivity
    has_pec: bool             # any PEC geometry present
    estimated_Q: float        # estimated cavity Q from material loss


def analyze_features(geometry: list, materials: dict, pec_threshold: float = 1e6) -> FeatureInfo:
    """Extract critical dimensions and material properties from geometry.

    Parameters
    ----------
    geometry : list of (Shape, material_name) tuples
    materials : dict of material_name -> MaterialSpec or dict
    pec_threshold : sigma above which material is PEC
    """
    thicknesses = []
    extents = []
    all_corners_lo = []
    all_corners_hi = []
    max_eps_r = 1.0
    has_pec = False
    max_loss_tangent = 0.0

    for shape, mat_name in geometry:
        mat = materials.get(mat_name, {})
        sigma = mat.get("sigma", 0.0) if isinstance(mat, dict) else getattr(mat, "sigma", 0.0)
        eps_r = mat.get("eps_r", 1.0) if isinstance(mat, dict) else getattr(mat, "eps_r", 1.0)

        if sigma >= pec_threshold:
            has_pec = True
            # PEC shapes still contribute to geometry bounds
        else:
            max_eps_r = max(max_eps_r, eps_r)
            if sigma > 0 and eps_r > 1:
                # Estimate loss tangent: tan_d ≈ sigma / (2*pi*f*eps_r*eps_0)
                # Use a representative frequency (center of band) later
                max_loss_tangent = max(max_loss_tangent, sigma / (eps_r * EPS_0))

        if hasattr(shape, "corner1") and hasattr(shape, "corner2"):
            c1, c2 = shape.corner1, shape.corner2
            dims = [abs(c2[i] - c1[i]) for i in range(3)]
            nonzero_dims = [d for d in dims if d > 1e-12]
            if nonzero_dims:
                thicknesses.append(min(nonzero_dims))
                extents.append(max(dims))
            all_corners_lo.append(c1)
            all_corners_hi.append(c2)

    # Bounding box of all geometry
    if all_corners_lo:
        bbox_lo = tuple(min(c[i] for c in all_corners_lo) for i in range(3))
        bbox_hi = tuple(max(c[i] for c in all_corners_hi) for i in range(3))
    else:
        bbox_lo = (0, 0, 0)
        bbox_hi = (0.01, 0.01, 0.01)

    min_thickness = min(thicknesses) if thicknesses else 0.001
    max_extent = max(extents) if extents else 0.01

    # Estimated Q from loss tangent
    if max_loss_tangent > 0:
        estimated_Q = 1.0 / max_loss_tangent  # Q ≈ 1/tan_d (rough)
    else:
        estimated_Q = 1000.0  # high Q if no loss specified

    return FeatureInfo(
        min_thickness=min_thickness,
        max_extent=max_extent,
        bbox=(bbox_lo, bbox_hi),
        max_eps_r=max_eps_r,
        has_pec=has_pec,
        estimated_Q=estimated_Q,
    )


# ---------------------------------------------------------------------------
# Auto-configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Auto-derived simulation configuration."""
    dx: float
    domain: tuple[float, float, float]
    cpml_layers: int
    n_steps: int
    freq_range: tuple[float, float]
    margin: float
    dt: float
    accuracy: str
    warnings: list[str] = field(default_factory=list)

    @property
    def cells_per_wavelength(self) -> float:
        f_max = self.freq_range[1]
        lambda_min = C0 / f_max
        return lambda_min / self.dx

    @property
    def sim_time_ns(self) -> float:
        return self.n_steps * self.dt * 1e9

    def to_sim_kwargs(self) -> dict:
        """Convert to keyword arguments for Simulation constructor."""
        return {
            "freq_max": self.freq_range[1],
            "domain": self.domain,
            "boundary": "cpml",
            "cpml_layers": self.cpml_layers,
            "dx": self.dx,
        }

    def summary(self) -> str:
        lines = [
            f"SimConfig (accuracy={self.accuracy!r}):",
            f"  dx = {self.dx*1e3:.3f} mm ({self.cells_per_wavelength:.0f} cells/λ_min)",
            f"  domain = {self.domain[0]*1e3:.1f} × {self.domain[1]*1e3:.1f} × {self.domain[2]*1e3:.1f} mm",
            f"  cpml = {self.cpml_layers} layers ({self.cpml_layers*self.dx*1e3:.1f} mm)",
            f"  n_steps = {self.n_steps} ({self.sim_time_ns:.1f} ns)",
            f"  freq = {self.freq_range[0]/1e9:.2f} – {self.freq_range[1]/1e9:.2f} GHz",
        ]
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def auto_configure(
    geometry: list,
    freq_range: tuple[float, float],
    materials: dict | None = None,
    accuracy: str = "standard",
    *,
    dx_override: float | None = None,
    margin_override: float | None = None,
    n_steps_override: int | None = None,
) -> SimConfig:
    """Derive all simulation parameters from geometry + frequency range.

    Parameters
    ----------
    geometry : list of (Shape, material_name) tuples
    freq_range : (f_min, f_max) in Hz
    materials : dict of material definitions (name -> {eps_r, sigma, ...})
    accuracy : "draft", "standard", or "high"
        draft:    10 cells/λ, 0.15λ margin  (fast, ~10% error)
        standard: 20 cells/λ, 0.25λ margin  (~3% error)
        high:     40 cells/λ, 0.50λ margin  (~1% error)
    dx_override : force specific cell size
    margin_override : force specific margin
    n_steps_override : force specific step count

    Returns
    -------
    SimConfig with all derived parameters.
    """
    if materials is None:
        materials = {}

    f_min, f_max = freq_range
    if f_min <= 0 or f_max <= f_min:
        raise ValueError(f"freq_range must be (f_min, f_max) with 0 < f_min < f_max, got {freq_range}")

    lambda_min = C0 / f_max
    lambda_max = C0 / f_min
    f_center = (f_min + f_max) / 2

    # Accuracy presets
    presets = {
        "draft":    {"cpw": 10, "cpf": 2, "margin_frac": 0.15, "pml_frac": 0.10},
        "standard": {"cpw": 20, "cpf": 4, "margin_frac": 0.25, "pml_frac": 0.20},
        "high":     {"cpw": 40, "cpf": 8, "margin_frac": 0.50, "pml_frac": 0.40},
    }
    if accuracy not in presets:
        raise ValueError(f"accuracy must be 'draft', 'standard', or 'high', got {accuracy!r}")
    preset = presets[accuracy]

    # Analyze geometry
    features = analyze_features(geometry, materials)
    warnings = []

    # 1. Cell size
    # λ_min in highest-eps medium
    lambda_min_medium = lambda_min / math.sqrt(features.max_eps_r)
    dx_wavelength = lambda_min_medium / preset["cpw"]
    dx_feature = features.min_thickness / preset["cpf"] if features.min_thickness > 0 else dx_wavelength
    dx = min(dx_wavelength, dx_feature)

    if dx_override is not None:
        dx = dx_override
    else:
        # Round to nice value
        dx = _round_dx(dx)

    # Check feature resolution
    if features.min_thickness > 0 and features.min_thickness / dx < 2:
        warnings.append(
            f"Thinnest feature ({features.min_thickness*1e3:.2f} mm) has only "
            f"{features.min_thickness/dx:.1f} cells — consider finer dx or subgridding"
        )

    # 2. Domain margins
    margin = lambda_max * preset["margin_frac"]
    if margin_override is not None:
        margin = margin_override

    bbox_lo, bbox_hi = features.bbox
    domain = tuple(
        (bbox_hi[i] - bbox_lo[i]) + 2 * margin
        for i in range(3)
    )
    # Ensure minimum domain > 4*dx per dimension
    domain = tuple(max(d, 8 * dx) for d in domain)

    # 3. CPML layers — based on cells needed for grading, not physical thickness
    # 8-16 cells is sufficient for well-tuned CPML regardless of wavelength
    cpml_cells = {"draft": 8, "standard": 12, "high": 16}[accuracy]
    cpml_layers = cpml_cells

    # 4. Timestep
    dt = 0.99 * dx / (C0 * math.sqrt(3))

    # 5. Number of steps
    # Source time: 6*tau for Gaussian pulse
    bw = 0.8  # default bandwidth
    tau = 1.0 / (f_center * bw * math.pi)
    t_source = 6 * tau

    # Ring-down time: Q / (pi * f_min)
    Q_est = min(features.estimated_Q, 1000)  # cap at 1000 for practicality
    t_ringdown = Q_est / (math.pi * f_min)

    t_total = t_source + t_ringdown
    n_steps = int(math.ceil(t_total / dt))

    if n_steps_override is not None:
        n_steps = n_steps_override

    # Sanity check
    if n_steps > 500000:
        warnings.append(f"Estimated {n_steps} steps (high Q={Q_est:.0f}). Consider reducing freq_range or using decay-based stopping.")
        n_steps = min(n_steps, 500000)

    return SimConfig(
        dx=dx,
        domain=domain,
        cpml_layers=cpml_layers,
        n_steps=n_steps,
        freq_range=freq_range,
        margin=margin,
        dt=dt,
        accuracy=accuracy,
        warnings=warnings,
    )


def _round_dx(dx: float) -> float:
    """Round dx to a nice value (1, 2, 2.5, or 5 × 10^n)."""
    exp = math.floor(math.log10(dx))
    mantissa = dx / (10 ** exp)
    nice = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    best = min(nice, key=lambda n: abs(n - mantissa))
    return best * (10 ** exp)

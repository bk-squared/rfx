"""Neural surrogate data export API.

Provides utilities to export parametric sweep results as training data
for neural network surrogates, and to export simulation geometry as
signed distance fields (SDFs) for geometry-conditioned neural operators.

Typical workflow::

    sr = parametric_sweep(factory, "width", widths, n_steps=500)
    export_training_data(sr, output_path="sweep_data.npz")

    sim = Simulation(freq_max=10e9, domain=(...))
    sim.add(Box(...), material="substrate")
    sdf = export_geometry_sdf(sim, resolution=1e-3)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Training data export
# ---------------------------------------------------------------------------

def export_training_data(
    sweep_result,
    *,
    output_path: str | Path,
    format: str = "npz",
) -> Path:
    """Export parametric sweep results as neural network training data.

    From a ``SweepResult`` or ``VmapSweepResult``, creates arrays:

    - **inputs** : parameter values, shape ``(n_samples, 1)``
    - **outputs** : time-series probe data, shape ``(n_samples, n_steps, n_probes)``
      (``SweepResult``) or ``(n_samples, n_steps, n_probes)``
      (``VmapSweepResult``).  When S-parameters are available (``SweepResult``
      with ports), an ``s_params`` array is also stored.
    - **metadata** : parameter name and frequency array (when available).

    Parameters
    ----------
    sweep_result : SweepResult or VmapSweepResult
        Output of ``parametric_sweep`` or ``vmap_material_sweep``.
    output_path : str or Path
        Destination file path.
    format : ``"npz"``
        Output format (currently only NumPy ``.npz`` is supported).

    Returns
    -------
    Path
        The written file path.

    Raises
    ------
    ValueError
        If format is unsupported or sweep_result type is unrecognised.
    """
    if format != "npz":
        raise ValueError(f"Unsupported format {format!r}; use 'npz'")

    output_path = Path(output_path)

    param_values = np.asarray(sweep_result.param_values)
    inputs = param_values.reshape(-1, 1)

    save_dict: dict[str, np.ndarray] = {
        "inputs": inputs,
        "param_name": np.array(sweep_result.param_name),
    }

    # Detect SweepResult vs VmapSweepResult
    if hasattr(sweep_result, "results"):
        # SweepResult — extract per-result time series
        ts_list = []
        s_params_list = []
        freqs = None
        for r in sweep_result.results:
            ts_list.append(np.asarray(r.time_series))
            sp = getattr(r, "s_params", None)
            if sp is not None:
                s_params_list.append(np.asarray(sp))
            f = getattr(r, "freqs", None)
            if f is not None:
                freqs = np.asarray(f)

        save_dict["outputs"] = np.stack(ts_list, axis=0)

        if s_params_list and len(s_params_list) == len(sweep_result.results):
            save_dict["s_params"] = np.stack(s_params_list, axis=0)

        if freqs is not None:
            save_dict["freqs"] = freqs

    elif hasattr(sweep_result, "time_series"):
        # VmapSweepResult — time_series already batched
        save_dict["outputs"] = np.asarray(sweep_result.time_series)
    else:
        raise ValueError(
            f"Unrecognised sweep_result type: {type(sweep_result).__name__}"
        )

    np.savez(str(output_path), **save_dict)
    return output_path


# ---------------------------------------------------------------------------
# Geometry SDF export
# ---------------------------------------------------------------------------

def export_geometry_sdf(
    sim,
    *,
    resolution: float = 1e-3,
) -> np.ndarray:
    """Export simulation geometry as a signed distance field.

    Builds a 3-D grid at the requested *resolution* covering the
    simulation domain and evaluates the geometry mask.  The SDF is
    approximated from the binary mask via a distance transform:

    - negative inside geometry (material regions)
    - positive outside
    - magnitude = approximate Euclidean distance to the nearest boundary
      (in metres)

    This is suitable as input to geometry-conditioned neural operators
    (e.g., DeepONet, Fourier Neural Operator).

    Parameters
    ----------
    sim : Simulation
        A configured simulation with geometry added.
    resolution : float
        Spatial resolution of the SDF grid (metres). Default 1 mm.

    Returns
    -------
    sdf : ndarray, shape (Nx, Ny, Nz)
        Signed distance field in metres.
    """
    domain = sim._domain
    nx = max(int(np.ceil(domain[0] / resolution)), 1)
    ny = max(int(np.ceil(domain[1] / resolution)), 1)
    nz = max(int(np.ceil(domain[2] / resolution)), 1)

    # Build a temporary grid at the SDF resolution to evaluate masks.
    # We construct coordinate arrays manually to avoid coupling to Grid's
    # CPML padding logic — the SDF should cover the physical domain only.
    x = np.linspace(0, domain[0], nx)
    y = np.linspace(0, domain[1], ny)
    z = np.linspace(0, domain[2], nz)

    # Evaluate geometry occupancy at each grid point
    occupied = np.zeros((nx, ny, nz), dtype=bool)

    for entry in sim._geometry:
        shape = entry.shape
        # Use the shape's corner-based geometry directly for Box, Sphere,
        # Cylinder. For arbitrary shapes fall back to mask() with a
        # temporary grid.
        if hasattr(shape, "corner_lo") and hasattr(shape, "corner_hi"):
            # Box — fast analytic test
            lo = shape.corner_lo
            hi = shape.corner_hi
            mx = (x >= lo[0]) & (x <= hi[0])
            my = (y >= lo[1]) & (y <= hi[1])
            mz = (z >= lo[2]) & (z <= hi[2])
            occupied |= mx[:, None, None] & my[None, :, None] & mz[None, None, :]
        elif hasattr(shape, "center") and hasattr(shape, "radius"):
            # Sphere or Cylinder — analytic
            cx, cy, cz = shape.center
            r = shape.radius
            dx_ = x[:, None, None] - cx
            dy_ = y[None, :, None] - cy
            dz_ = z[None, None, :] - cz
            if hasattr(shape, "height"):
                # Cylinder
                axis = getattr(shape, "axis", "z")
                if axis == "z":
                    r2 = dx_ ** 2 + dy_ ** 2
                    h_mask = np.abs(dz_) <= shape.height / 2
                elif axis == "y":
                    r2 = dx_ ** 2 + dz_ ** 2
                    h_mask = np.abs(dy_) <= shape.height / 2
                else:
                    r2 = dy_ ** 2 + dz_ ** 2
                    h_mask = np.abs(dx_) <= shape.height / 2
                occupied |= (r2 <= r ** 2) & h_mask
            else:
                # Sphere
                r2 = dx_ ** 2 + dy_ ** 2 + dz_ ** 2
                occupied |= r2 <= r ** 2
        # else: skip shapes we cannot evaluate analytically — they will
        # not appear in the SDF.  A future version could accept a Grid.

    # Convert binary mask to approximate SDF via distance transform.
    try:
        from scipy.ndimage import distance_transform_edt
        dist_outside = distance_transform_edt(~occupied) * resolution
        dist_inside = distance_transform_edt(occupied) * resolution
        sdf = dist_outside - dist_inside
    except ImportError:
        # Fallback without scipy: return +1/-1 indicator scaled by
        # resolution (no true distance).
        sdf = np.where(occupied, -resolution, resolution).astype(np.float64)

    return sdf.astype(np.float64)

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

def _is_pec_entry(sim, entry) -> bool:
    """Does the assembly route this geometry entry to the PEC branch?

    Keyed the way ``_assemble_materials`` keys it (the RESOLVED material's
    ``sigma`` against the PEC threshold), not on the literal name "pec" —
    a named copper at 5.8e7 S/m is PEC there too.
    """
    try:
        mat = sim._resolve_material(entry.material_name)
    except Exception:
        return entry.material_name == "pec"
    thr = float(getattr(sim, "_PEC_SIGMA_THRESHOLD", 1e6))
    return float(getattr(mat, "sigma", 0.0) or 0.0) >= thr


def _sheet_footprint_on_samples(shape, x, y, z, *, name):
    """A declared SHEET's footprint on the SDF sample lattice (#931 §1.3).

    Built by the contract's own :func:`sheet_spec_from_shape`, applied to
    the exporter's sample lines instead of the solver's node lines, so the
    exporter cannot drift from the realization rule: normal = the thinnest
    bounding-box axis, plane = the nearest sample plane to the shape's
    mid-plane (an exact half-sample tie resolves LOWER), footprint = the
    drawn rectangle sampled CLOSED on the two in-plane axes (any other
    shape: its cross-section at its own mid-plane).
    """
    from rfx.geometry.rasterize_grid import (
        GridCoords, axis_cell_sizes, sheet_spec_from_shape)

    coords = GridCoords(x=x, y=y, z=z, shape=(x.size, y.size, z.size))
    sizes = tuple(axis_cell_sizes(v) for v in (x, y, z))
    try:
        spec = sheet_spec_from_shape(shape, coords, sizes, name=name)
    except ValueError as exc:
        raise ValueError(
            f"export_geometry_sdf: sheet {name!r} cannot be placed on the "
            f"SDF sample lattice — {exc} A sheet is zero-thickness, so it is "
            "exported as ONE sample layer; if it falls between samples or "
            "its footprint covers none, pass a finer `resolution=`. It is "
            "not dropped silently (the #369 vanished-metal class)."
        ) from exc
    return np.asarray(spec.footprint, dtype=bool)


def export_geometry_sdf(
    sim,
    *,
    resolution: float = 1e-3,
) -> np.ndarray:
    """Export simulation geometry as a signed distance field.

    Builds a 3-D grid at the requested *resolution* covering the
    simulation domain and samples each shape at node coordinates. A Box
    face flush with the domain's high edge may fall just outside the last
    sample plane. A shape covering no sample point raises ValueError naming
    ``geometry[i]``, including dielectrics; use a finer ``resolution``.
    The SDF is approximated from the binary mask via a distance transform:

    - negative inside geometry (material regions)
    - positive outside
    - magnitude = approximate Euclidean distance to the nearest boundary
      (in metres)

    This is suitable as input to geometry-conditioned neural operators
    (e.g., DeepONet, Fourier Neural Operator).

    **Sheets.** A conductor declared as a sheet — ``add_thin_conductor``,
    or a zero-thickness PEC Box, which is the same declaration (lattice
    ownership contract §1.5) — has no interior, so a containment test
    finds it only if its plane happens to land exactly on a sample. Before
    this it did not land: ``add_thin_conductor`` sheets are not in
    ``sim._geometry`` at all and never reached the exporter, so every
    sheet-declared ground plane, patch and trace was missing from the
    exported training data with no error anywhere.

    A zero-thickness region cannot be represented in a sampled occupancy
    field as zero thickness. The convention here, stated rather than
    inferred: **a sheet occupies exactly ONE sample layer** — the sample
    plane nearest its declared mid-plane, with the drawn footprint sampled
    closed in-plane — so the sheet's apparent thickness in the SDF is one sample,
    not zero. That is the same realization rule the solve uses (a sheet is
    one node plane); only the lattice differs. A sheet whose plane or
    footprint cannot be placed on this lattice raises rather than
    vanishing — refine ``resolution``.

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
    if getattr(sim, "_pinned_sheets", ()):
        raise ValueError(
            "export_geometry_sdf cannot place add_pinned_sheet sheets; "
            "they would be missing: pinned sheets are declared by mesh node "
            "index, so their position is undefined on the exporter's own lattice"
        )

    if getattr(sim, "_coaxial_ports", ()):
        raise ValueError(
            "export_geometry_sdf does not export coaxial port conductors; "
            "they would be missing"
        )

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

    for gi, entry in enumerate(sim._geometry):
        shape = entry.shape
        # A PEC Box with exactly one zero-extent axis IS a sheet
        # declaration (§1.5), so it is realized as a sheet here too rather
        # than relying on its plane coinciding with a sample point.
        lo_b = getattr(shape, "corner_lo", None)
        hi_b = getattr(shape, "corner_hi", None)
        if lo_b is not None and hi_b is not None and _is_pec_entry(sim, entry):
            zero = [i for i in range(3)
                    if float(hi_b[i]) - float(lo_b[i]) == 0.0]
            if len(zero) == 1:
                occupied |= _sheet_footprint_on_samples(
                    shape, x, y, z, name=f"geometry[{gi}] 'pec'")
                continue
        # Use each shape's own node sampling, including half-open faces
        # and thin-volume snapping, rather than a separate analytic rule.
        try:
            mask = np.asarray(shape.mask_on_coords(x, y, z), dtype=bool)
            if mask.shape != occupied.shape:
                raise ValueError(
                    f"mask shape {mask.shape} does not match {occupied.shape}")
        except Exception as exc:
            raise ValueError(
                f"geometry[{gi}] ({type(shape).__name__}) cannot be "
                "evaluated on the SDF samples; it would be missing "
                "from the exported geometry"
            ) from exc
        if not np.any(mask):
            raise ValueError(
                f"geometry[{gi}] ({type(shape).__name__}) realizes no SDF "
                f"sample at resolution={resolution} m; it would be missing "
                "— refine resolution"
            )
        occupied |= mask

    # Sheets declared through add_thin_conductor are NOT in sim._geometry,
    # so the loop above never saw them: a sheet ground plane, patch or
    # trace was silently absent from the exported geometry. One sample
    # layer on the sheet's own plane (see the docstring).
    for ti, tc in enumerate(getattr(sim, "_thin_conductors", ()) or ()):
        occupied |= _sheet_footprint_on_samples(
            tc.shape, x, y, z, name=f"thin_conductor[{ti}]")

    # linspace includes both domain endpoints, so its pitch differs from
    # the requested resolution. Singleton axes retain the requested pitch.
    sampling = tuple(length / (n - 1) if n > 1 else resolution
                     for length, n in zip(domain, (nx, ny, nz)))

    # Convert binary mask to approximate SDF via distance transform.
    try:
        from scipy.ndimage import distance_transform_edt
        dist_outside = distance_transform_edt(~occupied, sampling=sampling)
        dist_inside = distance_transform_edt(occupied, sampling=sampling)
        sdf = dist_outside - dist_inside
    except ImportError:
        # Fallback without scipy: return +1/-1 indicator scaled by
        # the smallest sample spacing (no true distance).
        spacing = min(sampling)
        sdf = np.where(occupied, -spacing, spacing).astype(np.float64)

    return sdf.astype(np.float64)

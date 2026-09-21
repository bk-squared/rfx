"""Coaxial S-matrix calculators, moved verbatim out of ``rfx.api._sparams``.

Issue #980 Phase 2. ``compute_coaxial_s_matrix``,
``compute_coaxial_line_reflection``, ``compute_coaxial_two_port`` and
``compute_coax_msl_transition`` were methods on ``_SparamMixin``; they are
relocated here byte for byte, dedented by exactly four spaces, and nothing
else -- same order, same text, same docstrings, no rename, no cleanup. Each
is a MODULE-LEVEL function whose first parameter is still ``self``: that is
the ``Simulation`` instance, and the bodies still reach the rest of the class
through it. ``rfx.api._sparams`` binds each function back as a class
attribute at its original position, so ``sim.compute_coaxial_two_port(...)``
keeps the same signature, ``__doc__`` and bound-method behaviour. A class
wrapper may replace that binding in a later step.

The move is gated on bit identity of the S arrays every leg returns
(``tests/locks/test_sparams_split_bit_identity.py``).

Import contract, inherited from ``rfx.api._sparams``: the result dataclasses
come from the LEAF module ``rfx.api._spec``, never from the ``rfx.api``
package itself, which keeps ``rfx/api/__init__.py`` the sole composition
point. The bodies' function-local imports (including
``rfx.api._preflight``) stay inside the functions exactly as they were, so
they do not run at import time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import CoaxialPort

from rfx.api._spec import (
    CoaxialSMatrixResult,
    CoaxialLineReflectionResult,
    CoaxialTwoPortResult,
    CoaxMSLTransitionResult,
)

from rfx.sparams._common import (
    _msl_cell_profile,
    msl_modal_voltage,
    _validate_extra_flux_monitor_entries,
    _warn_if_ringdown_truncated,
    _finalize_sparam_result,
    _warn_ntff_box_dropped,
    _assemble_coaxial_two_port_from_voltages,
    _ladder_split_witness,
    _assemble_coax_msl_transition_from_voltages,
)


def compute_coaxial_s_matrix(
    self,
    *,
    n_steps: int = 320,
    freqs: jnp.ndarray | None = None,
    n_freqs: int = 21,
    field_scale: float = 1.0e4,
    magnetic_ratio: float = 1.0,
    signal_floor: float = 1.0e-12,
    reference_plane_axial_index_offset: int = 0,
    strict_passivity: bool = False,
) -> "CoaxialSMatrixResult":
    """Experimental coaxial S-matrix via distributed TEM plane sources.

    .. deprecated::
        This single-plane V/I path measures inside a closed PEC box around a
        short coaxial stub, which has no transmission line for a clean
        reflection — it reports non-physical ``|S11|>1`` for a lossless
        short (verified). Use :meth:`compute_coaxial_line_reflection`, which
        builds a real coax line with a matched CPML feed and extracts the
        reflection from a multi-plane matrix-pencil decomposition (validated
        short→Γ=-1, open→|Γ|=1, matched→0 across the band). This method is
        retained only for backward compatibility.

    For each registered ``add_coaxial_port(...)`` port, runs one FDTD
    simulation with that port driven and all other coaxial ports passive.
    A distributed transverse E/M plane source is injected on the port's
    cross-section (the M67 prototype scaffold promoted to the public
    API); DFT plane probes capture the resulting Ex/Ey/Hx/Hy on every
    coaxial port's reference plane; the V/I extractor recovers ``V`` and
    ``I`` via the radial line / azimuthal loop integrals; and the
    standard power-wave decomposition assembles the full S-matrix.

    Status: **experimental**. The plane source can produce a residual
    forward wave and the extracted reference-plane V/I has known
    amplitude bias for coarse grids; ``status="degraded"`` is reported
    when any V/I sample falls below ``signal_floor``. Use this API for
    development; do not promote claims beyond E2/E3 without an external
    cross-solver fixture (see ``port_external_reference_requirements``).

    Parameters
    ----------
    n_steps:
        FDTD timesteps per driven-port run. Default 320.
    freqs:
        Frequency grid (Hz). Defaults to a uniform grid covering
        ``[freq_max / 10, freq_max]``.
    n_freqs:
        Number of frequencies if ``freqs`` is None. Default 21.
    field_scale:
        Linear scale on the radial E waveform. Increase to lift the
        plane signal above DFT noise (V/I extraction is amplitude-linear
        so the S-matrix is invariant under this scale).
    magnetic_ratio:
        Multiplier on the ``H`` waveform after the analytic ``1/Z_TEM``
        factor. ``1.0`` injects the lossless-TEM Poynting-balanced
        amplitude; smaller values bias toward an E-only injection.
    signal_floor:
        Absolute V or I phasor magnitude below which the result is
        flagged as ``"degraded"``.
    reference_plane_axial_index_offset:
        Axial-index offset for the source/probe plane relative to the
        port pin centre.

    Returns
    -------
    CoaxialSMatrixResult
    """

    import warnings
    warnings.warn(
        "compute_coaxial_s_matrix() (single-plane V/I in a closed PEC box) is "
        "deprecated and reports non-physical |S11|>1 for a lossless short; use "
        "compute_coaxial_line_reflection() (validated coax-line method). "
        "It will be removed in rfx v2.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    from rfx.probes.probes import init_dft_plane_probe
    from rfx.simulation import run as _run
    from rfx.sources.coaxial_port import (
        build_coaxial_tem_plane_source_specs,
        extract_coaxial_plane_vi_from_dft,
    )

    if not self._coaxial_ports:
        raise ValueError(
            "No coaxial ports registered. Call add_coaxial_port() first."
        )
    if (
        self._ports
        or self._waveguide_ports
        or self._floquet_ports
        or self._msl_ports
    ):
        raise NotImplementedError(
            "compute_coaxial_s_matrix() is defined only for "
            "add_coaxial_port(...) families in the current simulation."
        )
    if self._tfsf is not None:
        raise NotImplementedError(
            "compute_coaxial_s_matrix() is not supported with TFSF; "
            "TFSF is a plane-wave source, not a coaxial port."
        )
    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ):
        raise NotImplementedError(
            "compute_coaxial_s_matrix() supports the uniform Yee lane only."
        )

    # Issue #704 audit: same silent NTFF drop class as the MSL path.
    _warn_ntff_box_dropped(self, "compute_coaxial_s_matrix()")

    ports: list[CoaxialPort] = list(self._coaxial_ports)
    n_ports = len(ports)

    # Build the working grid + materials with all coaxial geometries
    # stamped (PEC center pin, PTFE dielectric fill, PEC outer shell from
    # M66). ``_build_materials`` only assembles bulk materials and shapes;
    # ``add_coaxial_port`` only registers the port descriptor, so without
    # this loop the FDTD would run with the source dropped into pure
    # vacuum and the wave would radiate bidirectionally with no coax
    # structure to confine it (this is the real source of the
    # calibration-blocked status documented in the handover).
    from rfx.sources.coaxial_port import (
        setup_coaxial_port,
        add_coaxial_matched_termination,
        add_coaxial_open_termination,
        add_coaxial_pec_end_cap,
    )
    grid = self._build_grid()
    materials, _, _ = self._build_materials(grid)
    for p in ports:
        materials = setup_coaxial_port(grid, p, materials)
    for term_port_idx, term_R, term_offset_cells in self._coaxial_terminations:
        materials = add_coaxial_matched_termination(
            grid,
            ports[term_port_idx],
            materials,
            target_impedance=term_R,
            axial_offset_cells=term_offset_cells,
        )
    for term_port_idx, retract_cells in self._coaxial_open_terminations:
        materials = add_coaxial_open_termination(
            grid,
            ports[term_port_idx],
            materials,
            pin_retract_cells=retract_cells,
        )
    for cap_port_idx, cap_offset_cells in self._coaxial_pec_end_caps:
        materials = add_coaxial_pec_end_cap(
            grid,
            ports[cap_port_idx],
            materials,
            axial_offset_cells=cap_offset_cells,
        )

    # Frequency grid.
    if freqs is None:
        freqs = jnp.linspace(
            self._freq_max / 10.0,
            self._freq_max,
            int(n_freqs),
            dtype=jnp.float32,
        )
    else:
        freqs = jnp.asarray(freqs, dtype=jnp.float32)

    # Reference-plane axial indices per port (cross-section z-plane).
    from rfx.sources.coaxial_port import _coaxial_port_geometry
    plane_indices: list[int] = []
    for p in ports:
        _, _, _, pin_center, _, _ = _coaxial_port_geometry(grid, p)
        plane_indices.append(
            int(grid.position_to_index(pin_center)[2])
            + int(reference_plane_axial_index_offset)
        )

    # Output buffers.
    n_freqs_used = int(freqs.shape[0])
    s = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
    z_tem_arr = np.zeros((n_ports, n_freqs_used), dtype=np.complex128)
    v_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
    i_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)

    status = "passed"

    for driven in range(n_ports):
        spec = build_coaxial_tem_plane_source_specs(
            grid=grid,
            port=ports[driven],
            n_steps=int(n_steps),
            field_scale=float(field_scale),
            magnetic_ratio=float(magnetic_ratio),
            reference_plane_axial_index_offset=int(
                reference_plane_axial_index_offset
            ),
        )
        z_tem_arr[driven, :] = complex(spec.z_tem_ohm)

        # DFT plane probes on every port's cross-section.
        dft_planes = []
        for p_idx, p in enumerate(ports):
            for component in ("ex", "ey", "hx", "hy"):
                dft_planes.append(
                    init_dft_plane_probe(
                        axis=2,
                        index=plane_indices[p_idx],
                        component=component,
                        freqs=freqs,
                        grid_shape=grid.shape,
                        dft_total_steps=int(n_steps),
                    )
                )

        result = _run(
            grid,
            materials,
            int(n_steps),
            # Honour the simulation boundary (was hardcoded "pec", the closed
            # box that is the documented root cause); self._boundary is always
            # a str ("pec"/"cpml"/"upml"), even for the BoundarySpec path.
            boundary=self._boundary,
            sources=list(spec.electric_sources),
            mag_sources=list(spec.magnetic_sources),
            dft_planes=dft_planes,
            return_state=False,
        )
        if result.dft_planes is None:
            raise RuntimeError(
                "compute_coaxial_s_matrix(): runner returned no DFT planes"
            )

        # Slice DFT planes back into per-port (ex, ey, hx, hy) groups.
        per_port: list[dict[str, np.ndarray]] = []
        for p_idx in range(n_ports):
            start = p_idx * 4
            group = result.dft_planes[start : start + 4]
            comp_map = {
                probe.component: np.asarray(probe.accumulator, dtype=np.complex128)
                for probe in group
            }
            per_port.append(comp_map)

        # Extract V/I at each port's reference plane.
        voltages = []
        currents = []
        for p_idx, p in enumerate(ports):
            vi = extract_coaxial_plane_vi_from_dft(
                grid=grid,
                port=p,
                plane_axial_index=plane_indices[p_idx],
                ex_dft=per_port[p_idx]["ex"],
                ey_dft=per_port[p_idx]["ey"],
                hx_dft=per_port[p_idx]["hx"],
                hy_dft=per_port[p_idx]["hy"],
            )
            v = np.asarray(vi.vi.voltage, dtype=np.complex128)
            i = np.asarray(vi.vi.current, dtype=np.complex128)
            voltages.append(v)
            currents.append(i)
            v_dump[driven, p_idx, :] = v
            i_dump[driven, p_idx, :] = i
            if (
                float(np.max(np.abs(v))) <= float(signal_floor)
                or float(np.max(np.abs(i))) <= float(signal_floor)
            ):
                status = "degraded"

        # Power-wave decomposition at each receive port (a_j at driven, b_i
        # at receiver) using the analytic Z_TEM as Z0.
        z0 = complex(spec.z_tem_ohm)
        a_j = (voltages[driven] + z0 * currents[driven]) / (2.0 * np.sqrt(z0))
        for receiver in range(n_ports):
            b_i = (voltages[receiver] - z0 * currents[receiver]) / (
                2.0 * np.sqrt(z0)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                s[receiver, driven, :] = np.where(
                    np.abs(a_j) > 0.0,
                    b_i / a_j,
                    np.nan + 1j * np.nan,
                )

    # Report the plane actually measured (``plane_indices``, derived from
    # each port's ``pin_center`` — see ``_coaxial_port_geometry``), not
    # ``port.position``: the two differ by ``direction*pin_length/2``
    # whenever ``pin_length != 0``, and ``position_to_index`` already adds
    # ``pad_z_lo``, so multiplying that padded index by ``dx`` directly
    # (the previous formula) double-counted the padding offset too.
    # Neither defect is pinned by a committed test (only the array SHAPE
    # is asserted in test_coaxial_s_matrix.py) — see #489 stage-2 design
    # note, incidental defect 1.
    reference_planes = np.asarray(
        [
            (float(plane_indices[p_idx]) - float(grid.pad_z_lo)) * float(grid.dx)
            for p_idx in range(n_ports)
        ],
        dtype=float,
    )

    _res_coax = CoaxialSMatrixResult(
        s_params=s,
        freqs=np.asarray(freqs, dtype=float),
        port_names=tuple(f"coax_{i}" for i in range(n_ports)),
        port_faces=tuple(p.face for p in ports),
        reference_planes=reference_planes,
        z_tem_ohm=z_tem_arr,
        voltages=v_dump,
        currents=i_dump,
        status=status,
    )
    return _finalize_sparam_result(
        _res_coax,
        extractor="compute_coaxial_s_matrix",
        strict=strict_passivity,
    )

def compute_coaxial_line_reflection(
    self,
    *,
    termination: str = "short",
    n_steps: int = 6000,
    freqs: jnp.ndarray | None = None,
    n_freqs: int = 11,
    field_scale: float = 1.0e4,
    cpml_axes: str = "z",
    dut_offset_cells: int = 4,
    probe_count: int = 12,
    probe_start_cells: int = 8,
    probe_spacing_cells: int = 4,
    feed_impedance: float | None = None,
    dut_impedance: float | None = None,
    eps_scale: "jnp.ndarray | float | None" = None,
) -> "CoaxialLineReflectionResult":
    """One-port coaxial reflection on a real transmission line (broad-E5).

    Builds a coextensive coax line (face='top', pin along -z) terminated in a
    matched resistive feed near the +z boundary, drives a TEM TFSF source one
    cell below the feed, and reflects off a calibration ``termination`` at the
    -z end: ``"short"`` (Γ=-1), ``"open"`` (Γ=+1), or ``"matched"`` (Γ→0).
    With ``termination="matched"`` and ``dut_impedance=R`` the DUT is instead a
    known resistive load (analytic ``Γ=(R-Z0)/(R+Z0)``) — used by the broad-E5
    envelope to test non-trivial reflection magnitudes against exact truth.
    The reflection is read from the modal voltage ``V(z)=∫E_r dr`` sampled at
    ``probe_count`` equally spaced planes and a matrix-pencil estimate of the
    complex propagation constant (β self-measured, Z0-free).

    Unlike ``compute_coaxial_s_matrix`` (single-plane V/I in a hardcoded
    closed PEC box — non-physical |S11|>1), this uses an absorbing CPML feed
    so a real line exists. **Resolution recipe**: keep ≥~4 cells across the
    (outer-inner) annulus (raise ``freq_max`` to shrink ``dx``); the result
    ``status`` reports ``"under_resolved"`` below ~3.5 cells.

    The conductors deliberately stop ~2 cells short of the +z PML — running
    PEC into CPML is numerically unstable.

    The enclosing :class:`Simulation` must use float32 precision and the
    three-dimensional, second-order uniform Yee solver with
    ``boundary="cpml"`` and ``cpml_layers > 0``. Unsupported precision,
    solver, grid, boundary, TFSF, and refinement settings raise before the
    grid is built. The line feed requires positive CPML on both z faces,
    ``cpml_axes="z"``, and no periodic-axis override.

    This method constructs its own coaxial line, TEM source, DFT planes, and
    termination. Do not add separate geometry, thin conductors, lumped RLC
    elements, probes or field monitors, NTFF boxes, or ``add_coaxial_*``
    termination helpers; those registrations are rejected rather than
    ignored. Use the documented :class:`Simulation`, port, and method
    arguments instead.

    The registered coaxial port supplies its x/y centre, ``face``, inner
    and outer radii, and excitation waveform. The method derives its axial
    layout internally, so the port's z coordinate and ``pin_length`` do not
    place the line, and the port's ``impedance`` does not set either load.
    Use ``feed_impedance`` for the feed and ``dut_impedance`` only with
    ``termination="matched"``. ``probe_count`` must be an integer of at
    least three, and every requested plane must fit between the DUT and
    source; otherwise the method raises before starting the FDTD run.

    Differentiable (``eps_scale``)
    ------------------------------
    Pass ``eps_scale`` (a scalar or ``(nx, ny, nz)`` ``jnp`` array) to make
    the reflection differentiable w.r.t. the dielectric under ``jax.grad``.
    It MULTIPLIES the stamped ``eps_r`` (``eps_r <- eps_r * eps_scale``),
    applied AFTER the numpy conductor/dielectric stamps so the fixed geometry
    (PTFE fill in ``eps_r``, PEC pin/shell in ``sigma``) is preserved and only
    modulated — a well-conditioned design channel (unlike replacing the fill
    with air). When provided, the field→voltage→reflection assembly runs on
    the ``jax.numpy`` path (``coaxial_line_plane_voltage_jnp`` + the traced
    extractor) so the gradient flows design → FDTD → DFT planes → Γ. With
    ``eps_scale=None`` the result is byte-identical to the validated numpy
    path. The AD↔FD gate is ``tests/unit/autodiff/test_coax_end_to_end_ad.py``.
    """

    if self._boundary != "cpml" or self._cpml_layers <= 0:
        raise ValueError(
            "compute_coaxial_line_reflection() requires boundary='cpml' "
            "with cpml_layers > 0 for its absorbing feed."
        )
    z_boundary = self._boundary_spec.z
    if (
        z_boundary.lo != "cpml"
        or z_boundary.hi != "cpml"
        or z_boundary.resolved_lo_thickness(self._cpml_layers) <= 0
        or z_boundary.resolved_hi_thickness(self._cpml_layers) <= 0
    ):
        raise ValueError(
            "compute_coaxial_line_reflection() requires positive CPML "
            "thickness on both z faces."
        )
    if cpml_axes != "z":
        raise ValueError(
            "compute_coaxial_line_reflection() requires cpml_axes='z'."
        )
    if self._periodic_axes:
        raise ValueError(
            "compute_coaxial_line_reflection() does not support periodic "
            "boundary axes."
        )
    if any(token != "cpml" for _, _, token in self._boundary_spec.faces()):
        raise ValueError(
            "compute_coaxial_line_reflection() requires CPML tokens on all "
            "six boundary faces; mixed BoundarySpec faces are not supported."
        )
    if self._mode != "3d":
        raise ValueError(
            "compute_coaxial_line_reflection() requires mode='3d'."
        )
    if self._solver != "yee":
        raise ValueError(
            "compute_coaxial_line_reflection() supports solver='yee' only; "
            "solver='adi' is not supported."
        )
    if self._precision != "float32":
        raise ValueError(
            "compute_coaxial_line_reflection() requires precision='float32'."
        )
    if self._stencil_order != 2:
        raise ValueError(
            "compute_coaxial_line_reflection() requires stencil_order=2."
        )
    if self._tfsf is not None:
        raise ValueError(
            "compute_coaxial_line_reflection() creates its own TEM TFSF "
            "source and does not accept an existing TFSF source."
        )
    # This driver owns its geometry. Validate caller-supplied profiles
    # before planning a mesh from registrations that it will reject below.
    declared_mesh = self._declared_mesh
    if any(declared_mesh[name] is not None for name in (
        "_dx_profile", "_dy_profile", "_dz_profile"
    )):
        raise ValueError(
            "compute_coaxial_line_reflection() supports only a uniform Yee "
            "grid; dx_profile, dy_profile, and dz_profile are not supported."
        )
    if self._refinement is not None:
        raise ValueError(
            "compute_coaxial_line_reflection() does not support SBP-SAT "
            "refinement; remove add_refinement() from this simulation."
        )
    if self._geometry or self._thin_conductors:
        raise ValueError(
            "compute_coaxial_line_reflection() constructs the complete line "
            "geometry; registered geometry and thin conductors are not "
            "supported. Use the documented Simulation, port, and method "
            "arguments instead."
        )
    if self._lumped_rlc:
        raise ValueError(
            "compute_coaxial_line_reflection() does not support registered "
            "lumped RLC elements."
        )
    if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
        raise ValueError(
            "compute_coaxial_line_reflection() does not consume registered "
            "probes, DFT planes, flux monitors, or NTFF boxes."
        )
    if (
        self._coaxial_terminations
        or self._coaxial_open_terminations
        or self._coaxial_pec_end_caps
    ):
        raise ValueError(
            "compute_coaxial_line_reflection() does not consume registered "
            "add_coaxial_* termination helpers; use termination= and "
            "dut_impedance= instead."
        )

    from rfx.probes.probes import init_dft_plane_probe
    from rfx.simulation import run as _run
    from rfx.sources.coaxial_port import (
        CoaxialPort as _CoaxPort,
        build_coaxial_tem_plane_source_specs,
        coaxial_line_plane_voltage,
        coaxial_line_plane_voltage_jnp,
        coaxial_line_reflection_from_plane_voltages,
        coaxial_tem_characteristic_impedance,
        stamp_coaxial_line,
        stamp_coaxial_short_plane,
        stamp_coaxial_annular_resistor,
    )

    if termination not in ("short", "open", "matched"):
        raise ValueError(
            f"termination must be 'short', 'open' or 'matched', got {termination!r}"
        )
    if dut_impedance is not None and termination != "matched":
        raise ValueError(
            "dut_impedance is used only with termination='matched'; remove "
            "it for short or open terminations."
        )
    if isinstance(probe_count, bool) or not isinstance(
        probe_count, (int, np.integer)
    ):
        raise ValueError("probe_count must be an integer of at least 3.")
    requested_probe_count = int(probe_count)
    if requested_probe_count < 3:
        raise ValueError("probe_count must be at least 3.")
    if len(self._coaxial_ports) != 1:
        raise ValueError(
            "compute_coaxial_line_reflection() is a one-port method; register "
            "exactly one add_coaxial_port()."
        )
    if (
        self._ports or self._waveguide_ports or self._floquet_ports or self._msl_ports
    ):
        raise NotImplementedError(
            "compute_coaxial_line_reflection() is defined only for a single "
            "add_coaxial_port(...) family."
        )
    port = self._coaxial_ports[0]
    if port.face != "top":
        raise NotImplementedError(
            "compute_coaxial_line_reflection() currently supports face='top' "
            "(pin along -z, DUT at the -z end); face='bottom' is symmetric and "
            "not yet wired."
        )

    grid = self._build_grid()
    nz = grid.shape[2]
    dz = float(grid.dx)
    center_xy = (float(port.position[0]), float(port.position[1]))
    a, b = float(port.pin_radius), float(port.outer_radius)

    # Axial layout: DUT just above the -z PML; coax runs up to ~2 cells short
    # of the +z PML; matched feed one cell below the coax top; source below it.
    # The +z offset uses pad_z_hi (not pad_z_lo) so an asymmetric BoundarySpec
    # cannot run the conductors into the +z PML (verified unstable).
    z_dut = int(grid.pad_z_lo) + int(dut_offset_cells)
    z_hi_coax = nz - int(grid.pad_z_hi) - 2
    z_feed = z_hi_coax - 1
    z_src = z_hi_coax - 3
    if not (z_dut + probe_start_cells + 2 * probe_spacing_cells < z_src):
        raise ValueError(
            "domain too short for the requested line layout; increase the z "
            "domain or reduce probe_start_cells/probe_count."
        )
    probes_z = [
        z_dut + int(probe_start_cells) + int(probe_spacing_cells) * k
        for k in range(requested_probe_count)
    ]
    probes_z = [z for z in probes_z if z < z_src - 4]
    if len(probes_z) != requested_probe_count:
        raise ValueError(
            f"only {len(probes_z)} of {requested_probe_count} requested "
            "probe planes fit before the source; increase the z domain or "
            "reduce probe_count, probe_start_cells, or probe_spacing_cells."
        )

    z_tem = coaxial_tem_characteristic_impedance(a, b)
    R_feed = float(feed_impedance) if feed_impedance is not None else float(z_tem)
    # For termination='matched', the DUT load resistance defaults to the feed
    # (Γ→0); override with dut_impedance to place a known mismatch
    # (Γ = (R-Z0)/(R+Z0)) — used by the broad-E5 envelope's non-trivial loads.
    R_dut = float(dut_impedance) if dut_impedance is not None else R_feed

    materials, _, _ = self._build_materials(grid)
    materials, shell_inner = stamp_coaxial_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_dut,
        z_hi_index=z_hi_coax, pin_radius=a, outer_radius=b,
    )
    materials = stamp_coaxial_annular_resistor(
        grid, materials, center_xy=center_xy, z_index=z_feed, pin_radius=a,
        outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
    )
    if termination == "short":
        materials = stamp_coaxial_short_plane(
            grid, materials, center_xy=center_xy, z_index=z_dut, outer_radius=b,
        )
    elif termination == "matched":
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=z_dut, pin_radius=a,
            outer_radius=b, target_impedance=R_dut, shell_inner_radius=shell_inner,
        )
    # "open": conductors simply end at z_dut (open circuit) — no extra stamp.

    # Differentiable design channel: applied AFTER the (numpy) stamps as a
    # MULTIPLIER so the stamped dielectric (PTFE fill) + PEC (in sigma) are
    # preserved and only modulated — well-conditioned (vs replacing the fill).
    if eps_scale is not None:
        materials = materials._replace(eps_r=materials.eps_r * jnp.asarray(eps_scale))

    if freqs is None:
        freqs = jnp.linspace(
            0.1 * self._freq_max, 0.6 * self._freq_max, int(n_freqs), dtype=jnp.float32
        )
    else:
        freqs = jnp.asarray(freqs, dtype=jnp.float32)

    # TEM TFSF source at z_src (internal descriptor places pin_center there).
    src_port = _CoaxPort(
        position=(center_xy[0], center_xy[1], (z_src - grid.pad_z_lo) * dz),
        face="top", pin_length=dz, pin_radius=a, outer_radius=b,
        impedance=port.impedance, excitation=port.excitation,
    )
    spec = build_coaxial_tem_plane_source_specs(
        grid=grid, port=src_port, n_steps=int(n_steps), field_scale=float(field_scale),
        magnetic_ratio=1.0,
    )

    planes = []
    for z in probes_z:
        for comp in ("ex", "ey"):
            planes.append(
                init_dft_plane_probe(
                    axis=2, index=int(z), component=comp, freqs=freqs,
                    grid_shape=grid.shape, dft_total_steps=int(n_steps),
                )
            )
    result = _run(
        grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
        sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
        dft_planes=planes, return_state=False,
    )
    if result.dft_planes is None:
        raise RuntimeError("compute_coaxial_line_reflection(): runner returned no DFT planes")

    # Modal voltage V(z) at every probe plane, per frequency.
    n_f = int(freqs.shape[0])
    z_planes_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_z], dtype=np.float64)
    ref_m = (z_dut - grid.pad_z_lo) * dz
    annulus_cells = float((b - a) / dz)

    if eps_scale is not None:
        # --- differentiable path: jnp voltage + traced extractor (AD moat) ---
        # Gradient flows: eps_scale -> Yee update -> DFT plane accumulators
        # -> modal voltage -> matrix-pencil reflection -> Γ.
        V = jnp.stack(
            [
                coaxial_line_plane_voltage_jnp(
                    grid, result.dft_planes[pi * 2 + 0].accumulator,
                    center_xy=center_xy, pin_radius=a, outer_radius=b,
                )
                for pi in range(len(probes_z))
            ],
            axis=0,
        )  # (n_planes, n_freqs)
        s11_c, gamma_c, rec_c, fit_c, z0_c = [], [], [], [], []
        for fi in range(n_f):
            out = coaxial_line_reflection_from_plane_voltages(
                z_planes_m, V[:, fi], reference_plane_m=ref_m, _prefer_jnp=True,
            )
            s11_c.append(out.reflection)
            gamma_c.append(out.gamma)
            rec_c.append(out.recurrence_residual)
            fit_c.append(out.fit_residual)
            z0_c.append(
                R_dut * (1.0 - out.reflection) / (1.0 + out.reflection)
                if termination == "matched"
                else jnp.asarray(np.nan + 1j * np.nan)
            )
        # Status from concrete geometry only. NOTE: unlike the concrete path,
        # the traced rec_resid>0.1 "contaminated" check is NOT evaluated here
        # (can't Python-branch on a tracer), so ``"passed"`` on the eps_scale
        # path means geometry-resolved, NOT fit-clean — inspect
        # ``recurrence_residual`` (returned) if you need the contamination
        # signal. ``"differentiable"`` flags the AD path so it is not confused
        # with a fully-gated concrete ``"passed"``.
        status = "under_resolved" if annulus_cells < 3.5 else "differentiable"
        return CoaxialLineReflectionResult(
            s11=jnp.stack(s11_c),
            freqs=jnp.asarray(freqs),
            gamma=jnp.stack(gamma_c),
            recurrence_residual=jnp.stack(rec_c),
            fit_residual=jnp.stack(fit_c),
            annulus_cells=annulus_cells,
            z0_numerical_ohm=jnp.stack(z0_c),
            termination=termination,
            status=status,
        )

    # --- concrete path: NumPy (byte-identical to the validated result) ---
    v_by_plane = []
    for pi in range(len(probes_z)):
        ex = result.dft_planes[pi * 2 + 0].accumulator
        ey = result.dft_planes[pi * 2 + 1].accumulator
        v_by_plane.append(
            coaxial_line_plane_voltage(
                grid, ex, ey, center_xy=center_xy, pin_radius=a, outer_radius=b,
            )
        )
    V = np.stack(v_by_plane, axis=0)          # (n_planes, n_freqs)

    s11 = np.zeros(n_f, dtype=np.complex128)
    gamma = np.zeros(n_f, dtype=np.complex128)
    rec_resid = np.zeros(n_f, dtype=np.float64)
    fit_resid = np.zeros(n_f, dtype=np.float64)
    z0_num = np.full(n_f, np.nan + 1j * np.nan, dtype=np.complex128)
    for fi in range(n_f):
        out = coaxial_line_reflection_from_plane_voltages(
            z_planes_m, V[:, fi], reference_plane_m=ref_m,
        )
        s11[fi] = out.reflection
        gamma[fi] = out.gamma
        rec_resid[fi] = out.recurrence_residual
        fit_resid[fi] = out.fit_residual
        if termination == "matched":
            G = out.reflection
            z0_num[fi] = R_dut * (1.0 - G) / (1.0 + G)

    if annulus_cells < 3.5:
        status = "under_resolved"
    elif float(np.max(rec_resid)) > 0.1:
        status = "contaminated"
    else:
        status = "passed"

    return CoaxialLineReflectionResult(
        s11=s11,
        freqs=np.asarray(freqs, dtype=float),
        gamma=gamma,
        recurrence_residual=rec_resid,
        fit_residual=fit_resid,
        annulus_cells=annulus_cells,
        z0_numerical_ohm=z0_num,
        termination=termination,
        status=status,
    )

def compute_coaxial_two_port(
    self,
    *,
    n_steps: int = 6000,
    freqs: jnp.ndarray | None = None,
    n_freqs: int = 11,
    field_scale: float = 1.0e4,
    cpml_axes: str = "z",
    probe_count: int = 12,
    probe_start_cells: int = 8,
    probe_spacing_cells: int = 4,
    feed_impedance: float | None = None,
    cond_warn: float = 1.0e3,
    strict_passivity: bool = False,
    eps_scale: "jnp.ndarray | float | None" = None,
    extra_flux_monitors: "list | None" = None,
) -> "CoaxialTwoPortResult":
    """Two-drive coaxial 2-port S-parameters (#489 stage 2) — VALIDATED WITH SCOPE.

    .. note::
        **VALIDATED WITH SCOPE, not EXPERIMENTAL** (issue #489, PI
        decision 2026-08-06 —
        ``docs/guides/sparameter_support_matrix.md``, the S-parameter
        family companion where this row lives). Covered: the two-port
        through-line class on this single coax geometry family, bracketed
        by an external openEMS referee (crossval 21) on ``|S21|`` and,
        via the port's own measured ``beta``, phase; a mesh-refinement
        convergence witness (``p ~= 1.5``); an end-to-end ``eps_scale``
        AD gate (``GRAD_SAFE``); and this method's own measured
        reciprocity/``cond(A)``. NOT covered: every DUT this method can
        currently gate against is azimuthally symmetric (TM0n only) —
        transition discontinuities that excite TE11 are still outside
        this evidence (see :meth:`compute_coax_msl_transition`, which
        stays EXPERIMENTAL, diagnostic-only), nor does the evidence
        generalize to other coax geometry families. See the class
        docstring on the returned :class:`CoaxialTwoPortResult` for the
        full evidence chain and scope statement.

    Builds ONE through coax line spanning the z axis with a matched
    annular-resistor feed near EACH z end (mirroring the validated
    1-port :meth:`compute_coaxial_line_reflection` layout at both ends:
    each feed sits between that end's own TEM TFSF source and that end's
    own CPML, i.e. strictly on the scattered-only side of that drive's
    TFSF boundary — never in the path of that drive's own launched
    wave), then drives each end's source in turn (two separate FDTD
    runs). A probe array of ``probe_count`` equally spaced planes near
    each end recovers that array's local two-wave decomposition
    (matrix-pencil, Z0-free, same machinery as the 1-port method); the
    forward/back amplitudes are evaluated at each port's OWN reference
    plane (its feed's axial position) and assembled into a full 2x2
    S-matrix via :func:`rfx.sources.coaxial_port.
    solve_two_port_from_wave_amplitudes` — the two-drive solve that does
    NOT assume the non-driven port sees zero incident wave (unlike the
    naive ``S[j,i] = b_j/a_i`` ratio, which has a hard terminator-
    reflection floor on a through line; see that function's docstring
    and ``docs/research_notes/20260729_i489_coax_two_port_design.md``).

    Port 1 is the +z end (mirrors the 1-port fixture's own ``face='top'``
    orientation); port 2 is the -z end (mirror image, ``face='bottom'``).
    Both drives share the SAME registered ``add_coaxial_port(...)``
    geometry (x/y centre, pin/outer radii) and excitation waveform; the
    registered port's own ``position``/``face``/``pin_length`` do not
    place either end of the line (mirrors the 1-port method's own
    contract). Requires ``port.face == 'top'`` (arbitrary but consistent
    with the 1-port method; the value is not otherwise used to orient
    this method's own internally-built fixture).

    Unlike the 1-port method, the returned result is routed through
    :func:`_finalize_sparam_result` (which runs the passivity/finiteness
    self-check via :func:`_warn_if_nonpassive_smatrix`) before being
    returned — the 1-port ``CoaxialLineReflectionResult`` bypasses that
    check (design-note incidental defect 3); this result does not.

    This method constructs its own coaxial line, TEM sources, DFT
    planes, and feeds. Do not add separate geometry, thin conductors,
    lumped RLC elements, probes or field monitors, NTFF boxes, TFSF
    sources, or ``add_coaxial_*`` termination helpers; those
    registrations are rejected rather than ignored.

    Same solver/precision/boundary contract as
    :meth:`compute_coaxial_line_reflection` (float32, 3D uniform Yee,
    ``boundary='cpml'`` with positive CPML on all six faces,
    ``cpml_axes='z'``, no periodic axes, no non-uniform mesh, no
    refinement).

    Differentiable (``eps_scale``, #489 leg 3)
    -------------------------------------------
    Pass ``eps_scale`` (a scalar or ``(nx, ny, nz)`` ``jnp`` array) to make
    the S-matrix differentiable w.r.t. the dielectric under ``jax.grad`` —
    same name, semantics, and MULTIPLIES-the-stamped-``eps_r`` design as
    the 1-port :meth:`compute_coaxial_line_reflection`'s own ``eps_scale``
    (``eps_r <- eps_r * eps_scale``, applied once, after both feed stamps,
    so it reaches BOTH drives' FDTD runs — the through line has no DUT
    break to scope it to). When provided, both drives route their
    voltage extraction through :func:`rfx.sources.coaxial_port.
    coaxial_line_plane_voltage_jnp` (the same differentiable twin the
    1-port path uses) instead of the concrete
    :func:`~rfx.sources.coaxial_port.coaxial_line_plane_voltage`, and the
    assembly (:func:`_assemble_coaxial_two_port_from_voltages`) and the
    two-drive solve (:func:`~rfx.sources.coaxial_port.
    solve_two_port_from_wave_amplitudes`) both dispatch to their own jnp
    cores. With ``eps_scale=None`` the result is byte-identical to the
    validated numpy path (dual-path design, not a rewrite — mirrors the
    1-port method's own contract). The per-drive ring-down ``settling_db``
    witness needs a concrete time series and is skipped on this path
    (stays ``nan``, same reasoning as the 1-port method's own
    "can't Python-branch on a tracer" note on its ``rec_resid``
    contamination check); ``status`` is therefore ``"under_resolved"`` or
    ``"differentiable"`` here, never ``"contaminated"``/``"passed"``.
    ``cond_warn`` is also silently INERT on this path: the ill-
    conditioning warning it controls is Python control flow keyed on a
    concrete ``cond(A)`` value
    (:func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`'s
    NumPy branch), which the traced jnp core
    (``_solve_two_port_from_wave_amplitudes_jnp``) cannot evaluate and
    does not attempt to — this mirrors the ``settling_db``/``status``
    losses above (same "can't Python-branch on a tracer" reason), not a
    separate defect; ``cond_a`` is still RETURNED (as a tracer), so a
    caller can inspect it after concretizing the result if degeneracy
    matters to them. The AD gate is ``tests/unit/autodiff/test_coax_two_port_ad.py``.

    ``extra_flux_monitors`` (issue #589 flux-adjudication instrument):
    an ENERGY-WITNESS channel, not an extractor change. Pass the entry
    objects ``Simulation.add_flux_monitor`` registers (build them on a
    scratch ``Simulation`` with the same domain and hand over its
    ``._flux_monitors``); each internal drive run then accumulates the
    requested Poynting-flux planes, and per-drive spectra come back
    name-keyed on ``result.flux_monitors``
    (``{port_name: {monitor_name: (n_monitor_freqs,) float64}}``, net
    flux, positive = +axis). The S-parameter math is untouched — the
    non-perturbation witness (S bit-identical with and without
    monitors) is gated in
    ``tests/unit/sparams/test_coax_msl_transition.py::test_extra_flux_monitors_do_not_perturb_s``.
    The registered-monitor guard is unchanged: monitors registered ON
    this sim still raise, because this method builds its own probes.
    """

    if self._boundary != "cpml" or self._cpml_layers <= 0:
        raise ValueError(
            "compute_coaxial_two_port() requires boundary='cpml' "
            "with cpml_layers > 0 for its absorbing feeds."
        )
    z_boundary = self._boundary_spec.z
    if (
        z_boundary.lo != "cpml"
        or z_boundary.hi != "cpml"
        or z_boundary.resolved_lo_thickness(self._cpml_layers) <= 0
        or z_boundary.resolved_hi_thickness(self._cpml_layers) <= 0
    ):
        raise ValueError(
            "compute_coaxial_two_port() requires positive CPML "
            "thickness on both z faces."
        )
    if cpml_axes != "z":
        raise ValueError(
            "compute_coaxial_two_port() requires cpml_axes='z'."
        )
    if self._periodic_axes:
        raise ValueError(
            "compute_coaxial_two_port() does not support periodic "
            "boundary axes."
        )
    if any(token != "cpml" for _, _, token in self._boundary_spec.faces()):
        raise ValueError(
            "compute_coaxial_two_port() requires CPML tokens on all "
            "six boundary faces; mixed BoundarySpec faces are not "
            "supported."
        )
    if self._mode != "3d":
        raise ValueError(
            "compute_coaxial_two_port() requires mode='3d'."
        )
    if self._solver != "yee":
        raise ValueError(
            "compute_coaxial_two_port() supports solver='yee' only; "
            "solver='adi' is not supported."
        )
    if self._precision != "float32":
        raise ValueError(
            "compute_coaxial_two_port() requires precision='float32'."
        )
    if self._stencil_order != 2:
        raise ValueError(
            "compute_coaxial_two_port() requires stencil_order=2."
        )
    if self._tfsf is not None:
        raise ValueError(
            "compute_coaxial_two_port() creates its own TEM TFSF "
            "sources and does not accept an existing TFSF source."
        )
    # This driver owns its geometry. Validate caller-supplied profiles
    # before planning a mesh from registrations that it will reject below.
    declared_mesh = self._declared_mesh
    if any(declared_mesh[name] is not None for name in (
        "_dx_profile", "_dy_profile", "_dz_profile"
    )):
        raise ValueError(
            "compute_coaxial_two_port() supports only a uniform Yee "
            "grid; dx_profile, dy_profile, and dz_profile are not "
            "supported."
        )
    if self._refinement is not None:
        raise ValueError(
            "compute_coaxial_two_port() does not support SBP-SAT "
            "refinement; remove add_refinement() from this simulation."
        )
    if self._geometry or self._thin_conductors:
        raise ValueError(
            "compute_coaxial_two_port() constructs the complete line "
            "geometry; registered geometry and thin conductors are not "
            "supported. Use the documented Simulation, port, and "
            "method arguments instead."
        )
    if self._lumped_rlc:
        raise ValueError(
            "compute_coaxial_two_port() does not support registered "
            "lumped RLC elements."
        )
    if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
        raise ValueError(
            "compute_coaxial_two_port() does not consume registered "
            "probes, DFT planes, flux monitors, or NTFF boxes."
        )
    if (
        self._coaxial_terminations
        or self._coaxial_open_terminations
        or self._coaxial_pec_end_caps
    ):
        raise ValueError(
            "compute_coaxial_two_port() does not consume registered "
            "add_coaxial_* termination helpers; use feed_impedance= "
            "instead."
        )
    if isinstance(probe_count, bool) or not isinstance(
        probe_count, (int, np.integer)
    ):
        raise ValueError("probe_count must be an integer of at least 3.")
    requested_probe_count = int(probe_count)
    if requested_probe_count < 3:
        raise ValueError("probe_count must be at least 3.")
    if len(self._coaxial_ports) != 1:
        raise ValueError(
            "compute_coaxial_two_port() is built from exactly one "
            "add_coaxial_port() (its x/y centre, radii, and excitation "
            "waveform are shared by both drives); register exactly one."
        )
    if (
        self._ports or self._waveguide_ports or self._floquet_ports or self._msl_ports
    ):
        raise NotImplementedError(
            "compute_coaxial_two_port() is defined only for a single "
            "add_coaxial_port(...) family."
        )
    port = self._coaxial_ports[0]
    if port.face != "top":
        raise NotImplementedError(
            "compute_coaxial_two_port() currently requires the "
            "registered port's face='top' (the value is not otherwise "
            "used to orient this method's own internally-built "
            "two-ended fixture; kept for contract consistency with "
            "compute_coaxial_line_reflection)."
        )

    from rfx.probes.probes import init_dft_plane_probe
    from rfx.probes.probes import flux_spectrum as _flux_spectrum
    from rfx.runners.uniform import build_flux_monitor_cfgs
    from rfx.simulation import run as _run, ProbeSpec
    from rfx.sources.coaxial_port import (
        CoaxialPort as _CoaxPort,
        build_coaxial_tem_plane_source_specs,
        coaxial_line_plane_voltage,
        coaxial_line_plane_voltage_jnp,
        coaxial_tem_characteristic_impedance,
        stamp_coaxial_line,
        stamp_coaxial_annular_resistor,
    )
    _validate_extra_flux_monitor_entries(
        extra_flux_monitors, self._domain, "compute_coaxial_two_port"
    )
    flux_by_drive: dict = {}

    grid = self._build_grid()
    nz = grid.shape[2]
    dz = float(grid.dx)
    center_xy = (float(port.position[0]), float(port.position[1]))
    a, b = float(port.pin_radius), float(port.outer_radius)

    # Axial layout: two mirrored 1-port-style ends (source, feed, probe
    # array) sharing ONE continuous stamped line — no DUT break. Offsets
    # (2/1/3 cells) mirror compute_coaxial_line_reflection's own
    # (z_hi_coax, z_feed, z_src) spacing exactly, just doubled and
    # mirror-imaged. See docs/design_notes/
    # i489_stage2_two_port_fdtd_predeclaration.md for the derivation of
    # why each feed sits strictly on the scattered-only side of its own
    # drive's TFSF boundary.
    z_hi_coax_top = nz - int(grid.pad_z_hi) - 2
    z_feed_top = z_hi_coax_top - 1
    z_src_top = z_hi_coax_top - 3
    z_lo_coax_bot = int(grid.pad_z_lo) + 2
    z_feed_bot = z_lo_coax_bot + 1
    z_src_bot = z_lo_coax_bot + 3

    probes_top = sorted(
        z_src_top - int(probe_start_cells) - int(probe_spacing_cells) * k
        for k in range(requested_probe_count)
    )
    probes_bot = sorted(
        z_src_bot + int(probe_start_cells) + int(probe_spacing_cells) * k
        for k in range(requested_probe_count)
    )
    if z_lo_coax_bot >= z_hi_coax_top or probes_bot[0] <= z_lo_coax_bot:
        raise ValueError(
            "compute_coaxial_two_port(): domain too short for the "
            "two-feed line layout; increase the z domain."
        )
    if probes_bot[-1] >= probes_top[0]:
        raise ValueError(
            "compute_coaxial_two_port(): the two probe arrays overlap "
            f"(bottom array reaches index {probes_bot[-1]}, top array "
            f"starts at {probes_top[0]}); increase the z domain or "
            "reduce probe_count/probe_start_cells/probe_spacing_cells."
        )

    z_tem = coaxial_tem_characteristic_impedance(a, b)
    R_feed = float(feed_impedance) if feed_impedance is not None else float(z_tem)

    materials, _, _ = self._build_materials(grid)
    materials, shell_inner = stamp_coaxial_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_lo_coax_bot,
        z_hi_index=z_hi_coax_top, pin_radius=a, outer_radius=b,
    )
    materials = stamp_coaxial_annular_resistor(
        grid, materials, center_xy=center_xy, z_index=z_feed_top, pin_radius=a,
        outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
    )
    materials = stamp_coaxial_annular_resistor(
        grid, materials, center_xy=center_xy, z_index=z_feed_bot, pin_radius=a,
        outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
    )

    # Differentiable design channel (#489 leg 3): applied AFTER the
    # (numpy) stamps as a MULTIPLIER, same contract as the 1-port
    # compute_coaxial_line_reflection's own eps_scale — the stamped
    # dielectric (PTFE fill) + PEC (in sigma) are preserved and only
    # modulated. Applied ONCE, before either drive's _run() call below,
    # so it reaches both (the through line has no DUT break to scope it
    # to one end).
    if eps_scale is not None:
        materials = materials._replace(eps_r=materials.eps_r * jnp.asarray(eps_scale))

    if freqs is None:
        freqs = jnp.linspace(
            0.1 * self._freq_max, 0.6 * self._freq_max, int(n_freqs), dtype=jnp.float32
        )
    else:
        freqs = jnp.asarray(freqs, dtype=jnp.float32)
    n_f = int(freqs.shape[0])

    # TEM TFSF sources: port 1 (+z end) mirrors the 1-port fixture's own
    # face='top' source exactly; port 2 (-z end) is the mirror image,
    # face='bottom'.
    src_port_top = _CoaxPort(
        position=(center_xy[0], center_xy[1], (z_src_top - grid.pad_z_lo) * dz),
        face="top", pin_length=dz, pin_radius=a, outer_radius=b,
        impedance=port.impedance, excitation=port.excitation,
    )
    src_port_bot = _CoaxPort(
        position=(center_xy[0], center_xy[1], (z_src_bot - grid.pad_z_lo) * dz),
        face="bottom", pin_length=dz, pin_radius=a, outer_radius=b,
        impedance=port.impedance, excitation=port.excitation,
    )
    spec_top = build_coaxial_tem_plane_source_specs(
        grid=grid, port=src_port_top, n_steps=int(n_steps),
        field_scale=float(field_scale), magnetic_ratio=1.0,
    )
    spec_bot = build_coaxial_tem_plane_source_specs(
        grid=grid, port=src_port_bot, n_steps=int(n_steps),
        field_scale=float(field_scale), magnetic_ratio=1.0,
    )

    n_bot = len(probes_bot)
    n_top = len(probes_top)
    all_probes_z = list(probes_bot) + list(probes_top)

    # Settling witness: one point probe per array (ex, mid-annulus on the
    # +x ray), at each array's middle plane. Same worst end/peak E^2 (dB)
    # convention as the MSL/mixed lanes (rfx.api._sparams module docstring
    # of _warn_if_ringdown_truncated); -40 dB is the project's ring-down
    # settling rule (docs/guides/simulation_methodology.md).
    x_mid = center_xy[0] + 0.5 * (a + b)
    i_probe = int(round(x_mid / dz)) + int(grid.pad_x_lo)
    j_probe = int(grid.pad_y_lo) + int(round(center_xy[1] / dz))
    witness_probes = [
        ProbeSpec(i=i_probe, j=j_probe, k=int(probes_bot[n_bot // 2]), component="ex"),
        ProbeSpec(i=i_probe, j=j_probe, k=int(probes_top[n_top // 2]), component="ex"),
    ]

    z_planes_bot_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_bot], dtype=np.float64)
    z_planes_top_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_top], dtype=np.float64)
    ref_top_m = (z_feed_top - grid.pad_z_lo) * dz
    ref_bot_m = (z_feed_bot - grid.pad_z_lo) * dz
    annulus_cells = float((b - a) / dz)

    _traced_eps = eps_scale is not None
    if _traced_eps:
        # jnp lists, one entry per drive; stacked into (2, n_planes, n_f)
        # arrays after the loop (traced values can't be assigned into a
        # preallocated numpy array in place).
        v_bot_list: list = [None, None]
        v_top_list: list = [None, None]
    else:
        v_bot_by_drive = np.zeros((2, n_bot, n_f), dtype=np.complex128)
        v_top_by_drive = np.zeros((2, n_top, n_f), dtype=np.complex128)
    settling_db = np.full(2, np.nan, dtype=np.float64)

    # drive_idx 0 drives port 1 (top); drive_idx 1 drives port 2 (bot).
    for drive_idx, spec in enumerate((spec_top, spec_bot)):
        planes = []
        for z in all_probes_z:
            for comp in ("ex", "ey"):
                planes.append(
                    init_dft_plane_probe(
                        axis=2, index=int(z), component=comp, freqs=freqs,
                        grid_shape=grid.shape, dft_total_steps=int(n_steps),
                    )
                )
        # #589 flux-adjudication opt-in: fresh accumulators PER DRIVE
        # (init_flux_monitor zeroes the DFT carries; sharing cfgs across
        # drives would co-accumulate both drives into one spectrum).
        _flux_run_kwargs = (
            {"flux_monitors": build_flux_monitor_cfgs(
                self, grid, int(n_steps), entries=extra_flux_monitors)}
            if extra_flux_monitors else {}
        )
        result = _run(
            grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
            sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
            probes=witness_probes, dft_planes=planes, return_state=False,
            **_flux_run_kwargs,
        )
        if result.dft_planes is None:
            raise RuntimeError(
                "compute_coaxial_two_port(): runner returned no DFT planes"
            )
        if extra_flux_monitors:
            flux_by_drive[("port1", "port2")[drive_idx]] = {
                entry.name: np.asarray(_flux_spectrum(fm, exact_f64=True), dtype=np.float64)
                for entry, fm in zip(
                    extra_flux_monitors, result.flux_monitors or ()
                )
            }

        top_off = n_bot * 2
        if _traced_eps:
            # --- differentiable path: jnp voltage (AD moat, #489 leg 3) ---
            # Only the ex plane is needed (mirrors compute_coaxial_line_
            # reflection's own eps_scale branch): coaxial_line_plane_
            # voltage_jnp integrates E_r along the +x ray, ey is unused
            # for the voltage line-integral either way.
            v_bot_list[drive_idx] = jnp.stack(
                [
                    coaxial_line_plane_voltage_jnp(
                        grid, result.dft_planes[pi * 2 + 0].accumulator,
                        center_xy=center_xy, pin_radius=a, outer_radius=b,
                    )
                    for pi in range(n_bot)
                ],
                axis=0,
            )  # (n_bot, n_freqs)
            v_top_list[drive_idx] = jnp.stack(
                [
                    coaxial_line_plane_voltage_jnp(
                        grid, result.dft_planes[top_off + pi * 2 + 0].accumulator,
                        center_xy=center_xy, pin_radius=a, outer_radius=b,
                    )
                    for pi in range(n_top)
                ],
                axis=0,
            )  # (n_top, n_freqs)
            # Settling witness needs a concrete time series (np.asarray on
            # a traced result.time_series would raise
            # TracerArrayConversionError) -- skipped on this path, same
            # "can't Python-branch on a tracer" reasoning as the 1-port
            # eps_scale branch's own skipped contamination check.
            # settling_db stays nan for both drives.
        else:
            v_bot_by_drive[drive_idx] = np.stack(
                [
                    coaxial_line_plane_voltage(
                        grid, result.dft_planes[pi * 2 + 0].accumulator,
                        result.dft_planes[pi * 2 + 1].accumulator,
                        center_xy=center_xy, pin_radius=a, outer_radius=b,
                    )
                    for pi in range(n_bot)
                ],
                axis=0,
            )  # (n_bot, n_freqs)
            v_top_by_drive[drive_idx] = np.stack(
                [
                    coaxial_line_plane_voltage(
                        grid, result.dft_planes[top_off + pi * 2 + 0].accumulator,
                        result.dft_planes[top_off + pi * 2 + 1].accumulator,
                        center_xy=center_xy, pin_radius=a, outer_radius=b,
                    )
                    for pi in range(n_top)
                ],
                axis=0,
            )  # (n_top, n_freqs)

            ts = np.asarray(result.time_series, dtype=float)
            if ts.ndim == 2 and ts.shape[0] >= 10 and ts.shape[1] == len(witness_probes):
                power = ts ** 2
                tail = max(1, power.shape[0] // 10)
                end = power[-tail:, :].mean(axis=0)
                peak = power.max(axis=0)
                tiny = np.finfo(float).tiny
                ratio_db = 10.0 * np.log10((end + tiny) / (peak + tiny))
                settling_db[drive_idx] = float(np.max(ratio_db))

    if _traced_eps:
        v_bot_by_drive = jnp.stack(v_bot_list, axis=0)
        v_top_by_drive = jnp.stack(v_top_list, axis=0)

    s_params, cond_a, rec_resid, fit_resid, gamma = _assemble_coaxial_two_port_from_voltages(
        z_planes_bot_m=z_planes_bot_m, z_planes_top_m=z_planes_top_m,
        ref_bot_m=ref_bot_m, ref_top_m=ref_top_m,
        v_bot_by_drive=v_bot_by_drive, v_top_by_drive=v_top_by_drive,
        cond_warn=float(cond_warn), _prefer_jnp=_traced_eps,
    )

    if _traced_eps:
        # Status from concrete geometry only -- rec_resid is a jnp
        # tracer here, so the Python-branching "contaminated" check below
        # cannot run (mirrors the 1-port eps_scale branch's own status
        # derivation). "differentiable" flags the AD path so it is not
        # confused with a fully-gated concrete "passed".
        status = "under_resolved" if annulus_cells < 3.5 else "differentiable"
    elif annulus_cells < 3.5:
        status = "under_resolved"
    elif float(np.max(rec_resid)) > 0.1:
        status = "contaminated"
    else:
        status = "passed"

    reference_planes = np.asarray([ref_top_m, ref_bot_m], dtype=float)
    result_obj = CoaxialTwoPortResult(
        s_params=s_params,
        freqs=np.asarray(freqs, dtype=float),
        port_names=("port1", "port2"),
        reference_planes=reference_planes,
        cond_a=cond_a,
        recurrence_residual=rec_resid,
        fit_residual=fit_resid,
        gamma=gamma,
        annulus_cells=annulus_cells,
        settling_db=settling_db,
        status=status,
        flux_monitors=(flux_by_drive if extra_flux_monitors else None),
    )
    # Issue #662: the witness above is computed but was never compared to
    # the -40 dB bar this result's own docstring documents. NaN on the
    # eps_scale path is skipped by the warner's finite mask (this array is
    # host-side numpy on every path, so no tracer is branched on here).
    _warn_if_ringdown_truncated(
        settling_db, ("port1", "port2"), n_steps=int(n_steps),
    )
    return _finalize_sparam_result(
        result_obj,
        extractor="compute_coaxial_two_port",
        strict=strict_passivity,
    )


# Appended to the shared passivity guard's message when THIS lane refuses, so
# the exception a caller actually reads names the way back to the matrix. The
# shared message (rfx/sparams/_common.py) is used by five extractors and is
# left untouched; only this lane refuses by default (issue #838).
COAX_MSL_TRANSITION_REFUSAL_HINT = (
    "compute_coax_msl_transition is an EXPERIMENTAL cross-family lane and "
    "refuses a non-passive S by default (strict_passivity=True); pass "
    "strict_passivity=False to get the diagnostic matrix back with a "
    "UserWarning instead of this error."
)


def compute_coax_msl_transition(
    self,
    *,
    junction_x: float,
    eps_r_sub: float | None = None,
    n_steps: int | None = None,
    num_periods: float = 30.0,
    freqs: "jnp.ndarray | None" = None,
    n_freqs: int = 11,
    field_scale: float = 1.0e4,
    probe_count: int = 8,
    probe_start_cells: int = 6,
    probe_spacing_cells: int = 3,
    msl_probe_count: int | None = None,
    msl_probe_start_cells: int | None = None,
    msl_probe_spacing_cells: int | None = None,
    feed_impedance: float | None = None,
    cond_warn: float = 1.0e3,
    strict_passivity: bool = True,
    skip_preflight: bool = False,
    extra_flux_monitors: "list | None" = None,
    return_ladder_voltages: bool = False,
) -> "CoaxMSLTransitionResult":
    """EXPERIMENTAL coax<->microstrip transition 2-port S-parameters (issue #489 leg 4).

    .. warning::
        **EXPERIMENTAL — not in the validated set**
        (``docs/guides/sparameter_support_matrix.md`` / ``.json``). One
        pre-declared fixture has been run against this method; see that
        fixture's own predeclaration
        (``tests/unit/sparams/test_coax_msl_transition.py``) for the measured
        envelope. NOT_TRACEABLE (see :class:`~rfx.api._spec.
        CoaxMSLTransitionResult`'s class docstring for the full honesty
        contract, including why the MSL side is extracted via the coax
        matrix-pencil fit rather than the diagnostic-only N-probe fit
        #488 uses). Since issue #838 this lane REFUSES by default: a
        non-passive extracted S raises ``ValueError`` unless the caller
        passes ``strict_passivity=False``, which returns the diagnostic
        matrix with a ``UserWarning`` instead.

    Generalizes issue #488's mixed lumped/wire<->MSL assembler
    (:meth:`compute_mixed_s_matrix`) to a coax<->MSL pair by combining,
    UNCHANGED, the less-invasive half of each family's own validated
    machinery instead of writing a new geometry-specific extractor:

    * The **coax side** is built exactly like :meth:`compute_coaxial_two_port`'s
      own single-ended stub (CPML, TEM TFSF source, matched annular-
      resistor feed, then a probe array), reusing
      :func:`~rfx.sources.coaxial_port.stamp_coaxial_line`,
      :func:`~rfx.sources.coaxial_port.stamp_coaxial_annular_resistor`,
      and :func:`~rfx.sources.coaxial_port.build_coaxial_tem_plane_source_specs`
      verbatim — but only ONE end (this method has no second coax port).
    * The **MSL side** is consumed exactly like :meth:`compute_mixed_s_matrix`
      consumes its MSL ports: the caller registers arbitrary DUT
      geometry (substrate, trace, ground plane, and — unique to this
      transition — the ground-plane clearance hole and the vertical
      pin-to-trace post that connects the two families) via the
      ordinary ``sim.add(Box(...)/Cylinder(...), material=...)`` API,
      and this method reuses :func:`~rfx.sources.msl_port.compute_msl_mode_profile`,
      :func:`~rfx.sources.msl_port.setup_msl_port`, and
      :func:`~rfx.sources.msl_port.make_msl_port_sources` verbatim for
      the MSL port's own termination/excitation.

    This method does NOT build the junction geometry itself (the ground
    plane, its clearance hole, or the pin-to-trace post) — those are
    DUT-specific and belong to the caller's own registered geometry,
    exactly as :meth:`compute_mixed_s_matrix` never builds its own
    substrate/trace. What this method DOES fix, by construction, is
    WHERE each port's own S-parameter reference plane sits: both are
    placed AT the physical launch discontinuity (see ``junction_x``
    below and ``port.position[2]`` on the registered
    :meth:`add_coaxial_port`), specifically to minimize the
    pre-declared "reference-plane mismatch" failure mode (the coax's
    axial z-feed-plane convention has no direct analogue in the MSL's
    along-trace x reference plane — these are different geometric axes
    entirely).

    Registration contract
    ----------------------
    Exactly one :meth:`add_coaxial_port` (``face='bottom'`` — the
    physical convention this method assumes: the coax stub is built
    FROM the domain's low-z CPML face UP TO ``position[2]`` (rounded to
    the nearest grid z-node, half-cell ties to the lower node under
    the conductor-plane convention), where ``position[0], position[1]`` is the coax
    axis centre (x, y) and ``position[2]`` is the physical height of
    the caller's OWN registered ground-plane conductor — i.e. this is
    the ONE parameter that ties this method's auto-built coax stub to
    the caller's own junction geometry; get it wrong and the pin will
    either dangle in a gap or overlap the caller's own PEC). Exactly
    one :meth:`add_msl_port`, whose ``position[2]`` (substrate bottom /
    ground height) MUST equal the coax port's ``position[2]`` to within
    one grid cell — both refer to the SAME physical ground plane. No
    other ports (lumped/wire/waveguide/Floquet), no TFSF, no lumped
    RLC, no pre-registered probes/DFT planes/flux monitors/NTFF (this
    method builds its own). ``self._geometry`` must be non-empty (the
    caller's substrate/trace/ground-plane/pin-post Boxes and
    Cylinders). The MSL source interval must meet its realized conductor
    surfaces. The auto-built stub stops below the junction node and
    preserves the caller's materials at and above it. These checks do
    not certify the junction's connectivity or its modal accuracy.
    Same solver/precision/boundary contract as
    :meth:`compute_coaxial_two_port` (float32, 3D uniform Yee,
    ``boundary='cpml'`` with positive CPML on all six faces) EXCEPT
    this method needs absorption on all three axes (``cpml_axes="xyz"``,
    not ``"z"``): the coax's own far end needs an absorbing z face
    exactly like the coax lane, but the MSL trace radiates in x/y too,
    and the caller's own ground plane is an INTERNAL stamped/registered
    PEC layer rather than the domain's z boundary — this is also WHY
    this method cannot reuse a PEC ``z_lo`` domain boundary as the MSL
    ground reference the way :meth:`compute_mixed_s_matrix`'s
    ``magnitude_channel="flux"`` fixtures do (that would conflict with
    the coax's own need for an absorbing z_lo face).

    Two-drive extraction (never a naive single-ratio)
    ---------------------------------------------------
    Drives the coax source, then the MSL source, in two separate FDTD
    runs (never assuming the non-driven port sees zero incident wave —
    the terminator-floor problem :meth:`compute_coaxial_two_port`
    already solved for symmetric coax applies here too, and likely
    worse, since the coax feed resistor and the MSL Hammerstad-Jensen
    termination have no reason to share a termination quality). Each
    port's own forward/backward wave amplitudes are recovered from its
    own probe ladder via :func:`~rfx.sources.coaxial_port.
    coaxial_line_reflection_from_plane_voltages` (Z0-free matrix-pencil
    fit — see :class:`~rfx.api._spec.CoaxMSLTransitionResult` for why
    this is used for the MSL side too, not the lane's own N-probe fit),
    converted to POWER waves via each port's own analytic reference
    impedance, and assembled by
    :func:`~rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`
    (the same generic two-drive solve :meth:`compute_coaxial_two_port`
    uses). See :func:`_assemble_coax_msl_transition_from_voltages` for
    the full pure-assembly derivation.

    Parameters
    ----------
    junction_x : float
        Physical x-coordinate (metres) of the coax-to-trace launch
        discontinuity — the MSL side's own S-parameter reference plane.
        Must match wherever the caller's own registered pin-to-trace
        post / trace Box actually begins; this method does not infer it
        from geometry (mirrors the coax side's own reference plane
        being the registered port's ``position[2]``, not inferred).
    eps_r_sub : float, optional
        Substrate relative permittivity. If ``None``, taken from the
        registered :meth:`add_msl_port`'s own ``eps_r_sub`` (which must
        then be set explicitly — this method does not attempt the
        geometry-bounding-box auto-detection :meth:`add_msl_port`
        itself offers, to keep the eps anchor unambiguous for the
        Hammerstad-Jensen Z0 used in the power-wave normalization).
    probe_count, probe_start_cells, probe_spacing_cells : int
        The COAX side's own probe array (mirrors
        :meth:`compute_coaxial_two_port`'s identically-named
        parameters — this method's coax stub is short by
        construction, between the near-source feed and the junction,
        so these three rarely need to grow).
    msl_probe_count, msl_probe_start_cells, msl_probe_spacing_cells : int, optional
        The MSL side's OWN probe array, independent of the coax
        parameters above (added issue #489 leg 4 attempt 2 — the two
        families' probe ladders were coupled through one shared set
        of parameters through attempt 1, which is fine when both
        ladders are short but breaks as soon as the MSL side needs a
        ladder spanning a meaningful fraction of a guided wavelength,
        since the coax stub's own short z-extent cannot host the same
        span). Each defaults to ``None``, meaning "use the
        correspondingly-named coax parameter" — this preserves
        attempt 1's exact behavior (and its committed fixture's
        numbers) when left unset.

        NEAR-FIELD STANDOFF (issue #823) — the one constraint these
        three are NOT free of. Every probe must clear BOTH ends the
        ladder is referred to (the MSL port's own feed plane AND the
        reference plane at ``junction_x``) by at least
        ``max(3, round(5*h_sub/dx))`` cells. Both are launch
        discontinuities, and within a few substrate thicknesses of one
        the field is not the guided mode yet, so a matrix-pencil fit
        that includes such a probe reports the LADDER's error as the
        field's: on the settled attempt-3 run (VESSL 369367257533) a
        single probe 0.4 mm = 1.33*h_sub from the feed dragged the
        full-ladder residual from 3.7e-3 to 0.342, and this lane spent
        three attempts attributing that to junction physics. The
        threshold is the repo's EXISTING issue-#80 Fix B constant —
        see :func:`rfx.api._preflight.msl_source_near_field_standoff_cells`
        for the derivation that licenses reusing it here, and for the
        W/h limitation it carries. Preflight cannot enforce it on this
        lane (these are METHOD arguments and never reach the
        registered ``_MSLPortEntry``), so this method evaluates the
        same predicate on its own REALIZED ladder and emits a
        ``UserWarning`` naming the offending probes. REPORT-ONLY:
        nothing is refused. Read it together with
        ``result.ladder_split_gamma_dev`` /
        ``result.ladder_split_reflection_decades`` (computed when
        ``return_ladder_voltages=True``, ``None`` otherwise), which say
        whether the ladder actually disagrees with itself.
    strict_passivity : bool, default ``True``
        Default ``True``: this lane REFUSES a non-passive extracted S,
        raising ``ValueError`` from the shared guard
        (:func:`_warn_if_nonpassive_smatrix` via
        :func:`_finalize_sparam_result`) instead of returning the matrix.
        Pass ``strict_passivity=False`` to get the diagnostic matrix back
        with a ``UserWarning`` instead of the raise. The default is
        ``True`` here and ``False`` on the single-family coax lanes
        (:meth:`compute_coaxial_s_matrix`, :meth:`compute_coaxial_two_port`),
        which are unaffected by this (issue #838, PI decision 2026-09-20).

    ``extra_flux_monitors`` (issue #589 flux-adjudication instrument):
    an ENERGY-WITNESS channel, not an extractor change. Pass the entry
    objects ``Simulation.add_flux_monitor`` registers (build them on a
    scratch ``Simulation`` with the same domain and hand over its
    ``._flux_monitors``); each internal drive run then accumulates the
    requested Poynting-flux planes, and per-drive spectra come back
    name-keyed on ``result.flux_monitors``
    (``{"coax"|"msl": {monitor_name: (n_monitor_freqs,) float64}}``,
    net flux, positive = +axis). The S-parameter math is untouched —
    the non-perturbation witness (S bit-identical with and without
    monitors) is gated in
    ``tests/unit/sparams/test_coax_msl_transition.py::test_extra_flux_monitors_do_not_perturb_s``.
    The registered-monitor guard is unchanged: monitors registered ON
    this sim still raise, because this method builds its own probes.

    ``return_ladder_voltages`` (issue #589 label-independent ladder dump):
    a second read-only channel, additive in exactly the same sense. When
    ``True``, the RAW per-probe modal voltages this method already
    computed for both drives are attached to
    ``result.ladder_voltages`` (``None`` otherwise) so the ladders can be
    re-read OFFLINE — adjacent-pair phase slope (which way the dominant
    wave travels), standing-wave ratio, subset matrix-pencil fits —
    without a second FDTD run and without trusting any incident/outgoing
    LABEL. The dict is documented field-by-field on
    :class:`~rfx.api._spec.CoaxMSLTransitionResult`. It is built from
    ``.copy()`` of arrays that are complete before, and consumed by,
    :func:`_assemble_coax_msl_transition_from_voltages`, so every
    S-parameter number is bit-identical with the option off or on —
    gated by ``tests/unit/sparams/test_coax_msl_transition_ladder_dump.py::
    test_return_ladder_voltages_does_not_perturb_s`` (byte-identity A/B,
    the same discipline as the flux witness above; the round-trip
    assertion there also proves the dump IS what the assembler consumed).
    The same flag switches on the issue-#823 LADDER SELF-CONSISTENCY
    WITNESS, ``result.ladder_split_gamma_dev`` /
    ``result.ladder_split_reflection_decades`` (``None`` when the flag
    is off): a disjoint-half refit of each ladder, i.e. a Python loop of
    2 drives x n_freqs x 2 matrix pencils per ladder over exactly the
    arrays the dump exposes. It is computed AFTER the assembler from
    those arrays and moves no S-parameter number; it is opt-in so that
    a default call pays for no extra pencil solves. Documented
    field-by-field on :class:`~rfx.api._spec.CoaxMSLTransitionResult`.

    Returns
    -------
    CoaxMSLTransitionResult
    """
    from rfx.sources.coaxial_port import (
        CoaxialPort as _CoaxPort,
        build_coaxial_tem_plane_source_specs,
        coaxial_line_plane_voltage,
        coaxial_tem_characteristic_impedance,
        stamp_coaxial_line,
        stamp_coaxial_annular_resistor,
    )
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.sources.msl_port import (
        MSLPort as _MSLPortLL,
        msl_cross_section_span,
        compute_msl_mode_profile,
        setup_msl_port,
        make_msl_port_sources,
        msl_probe_x_coords_n,
    )
    from rfx.probes.probes import init_dft_plane_probe
    from rfx.probes.probes import flux_spectrum as _flux_spectrum
    from rfx.runners.uniform import build_flux_monitor_cfgs
    from rfx.simulation import run as _run, ProbeSpec

    _validate_extra_flux_monitor_entries(
        extra_flux_monitors, self._domain, "compute_coax_msl_transition"
    )
    flux_by_drive: dict = {}

    # ---- Registration guards ----------------------------------------
    from rfx.materials.thin_conductor import refuse_f0_sheets
    refuse_f0_sheets(self._thin_conductors, "coax-MSL transition")
    if self._boundary != "cpml" or self._cpml_layers <= 0:
        raise ValueError(
            "compute_coax_msl_transition() requires boundary='cpml' "
            "with cpml_layers > 0."
        )
    if any(token != "cpml" for _, _, token in self._boundary_spec.faces()):
        raise ValueError(
            "compute_coax_msl_transition() requires CPML tokens on all "
            "six boundary faces; mixed BoundarySpec faces are not "
            "supported (the coax stub needs an absorbing z_lo face and "
            "the MSL trace needs absorbing x/y faces; the ground plane "
            "is an internal registered/stamped PEC layer, not a domain "
            "boundary)."
        )
    if self._periodic_axes:
        raise ValueError(
            "compute_coax_msl_transition() does not support periodic "
            "boundary axes."
        )
    if self._mode != "3d":
        raise ValueError("compute_coax_msl_transition() requires mode='3d'.")
    if self._solver != "yee":
        raise ValueError(
            "compute_coax_msl_transition() supports solver='yee' only."
        )
    if self._precision != "float32":
        raise ValueError(
            "compute_coax_msl_transition() requires precision='float32'."
        )
    if self._stencil_order != 2:
        raise ValueError(
            "compute_coax_msl_transition() requires stencil_order=2."
        )
    if self._tfsf is not None:
        raise ValueError(
            "compute_coax_msl_transition() creates its own coax TEM "
            "source and does not accept an existing TFSF source."
        )
    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ):
        raise ValueError(
            "compute_coax_msl_transition() supports only a uniform "
            "Yee grid; dx_profile/dy_profile/dz_profile are not "
            "supported."
        )
    if self._refinement is not None:
        raise ValueError(
            "compute_coax_msl_transition() does not support SBP-SAT "
            "refinement."
        )
    if self._lumped_rlc:
        raise ValueError(
            "compute_coax_msl_transition() does not support registered "
            "lumped RLC elements."
        )
    if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
        raise ValueError(
            "compute_coax_msl_transition() does not consume registered "
            "probes, DFT planes, flux monitors, or NTFF boxes (it "
            "builds its own)."
        )
    if (
        self._coaxial_terminations
        or self._coaxial_open_terminations
        or self._coaxial_pec_end_caps
    ):
        raise ValueError(
            "compute_coax_msl_transition() does not consume registered "
            "add_coaxial_* termination helpers; use feed_impedance= "
            "instead."
        )
    if len(self._coaxial_ports) != 1:
        raise ValueError(
            "compute_coax_msl_transition() is built from exactly one "
            "add_coaxial_port()."
        )
    if len(self._msl_ports) != 1:
        raise ValueError(
            "compute_coax_msl_transition() is built from exactly one "
            "add_msl_port()."
        )
    if self._ports or self._waveguide_ports or self._floquet_ports:
        raise NotImplementedError(
            "compute_coax_msl_transition() is defined only for a "
            "coax + MSL port pair; no other port families."
        )
    if not self._geometry:
        raise ValueError(
            "compute_coax_msl_transition() consumes the caller's own "
            "registered DUT geometry (substrate, trace, ground plane, "
            "clearance hole, pin-to-trace post) — none is registered. "
            "This method builds only the coax stub; register the "
            "junction geometry via sim.add(...) first."
        )
    port = self._coaxial_ports[0]
    if port.face != "bottom":
        raise NotImplementedError(
            "compute_coax_msl_transition() currently requires the "
            "registered coax port's face='bottom' (the coax stub is "
            "built from the domain's low-z CPML face up to "
            "position[2])."
        )
    msl_pe = self._msl_ports[0]
    # The coax stub's own top face stops AT port.position[2] (rounded to
    # the nearest grid node); the caller's own registered ground-plane
    # conductor is expected to occupy that node and MSL's substrate
    # (msl_pe.position[2], its own z_lo / substrate-bottom convention)
    # to begin at or above it -- the exact gap is the caller's own
    # ground-plane thickness (fixture-specific, not knowable here), so
    # this only guards the ORDER and catches a grossly misaligned
    # ground reference (e.g. forgetting to raise msl_z_lo at all).
    if float(msl_pe.position[2]) < float(port.position[2]) - 1.5e-9:
        raise ValueError(
            "compute_coax_msl_transition(): the registered MSL port's "
            f"substrate-bottom height ({msl_pe.position[2]:.6g} m) sits "
            f"BELOW the coax port's own junction height "
            f"({port.position[2]:.6g} m) — both must reference the SAME "
            "physical ground plane, with the MSL substrate at or above "
            "it."
        )
    eps_r_sub_resolved = (
        float(eps_r_sub) if eps_r_sub is not None
        else (float(msl_pe.eps_r_sub) if msl_pe.eps_r_sub is not None else None)
    )
    if eps_r_sub_resolved is None:
        raise ValueError(
            "compute_coax_msl_transition() needs eps_r_sub, either "
            "passed directly or set on the registered add_msl_port() "
            "(this method does not auto-detect it from geometry)."
        )

    grid = self._build_grid()
    dz = float(grid.dx)
    _junction_gap_cells = (float(msl_pe.position[2]) - float(port.position[2])) / dz
    if _junction_gap_cells > 8.0:
        raise ValueError(
            "compute_coax_msl_transition(): the registered MSL port's "
            f"substrate-bottom height is {_junction_gap_cells:.1f} cells "
            f"above the coax port's junction height ({port.position[2]:.6g} "
            "m) — that is implausibly large for a single ground-plane "
            "layer; check both registrations reference the SAME "
            "physical ground plane."
        )
    _cx_pec_sheets: list = []
    _cx_pec_wires: list = []
    materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = \
        self._assemble_materials(
            grid, pec_sheets=_cx_pec_sheets, pec_wires=_cx_pec_wires)
    from rfx.boundaries.pec import (
        realized_pec_edge_masks as _rpem_cx,
    )
    _cx_pec_edge_masks = None
    if pec_mask is not None or _cx_pec_sheets or _cx_pec_wires:
        _cx_pec_edge_masks = _rpem_cx(
            pec_mask, sheets=tuple(_cx_pec_sheets),
            wires=tuple(_cx_pec_wires),
            periodic=self._periodic_flags())

    if freqs is None:
        freqs_arr = np.asarray(
            jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs)
        )
    else:
        freqs_arr = np.asarray(freqs)
    n_f = int(freqs_arr.shape[0])
    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)
    freqs_jnp = jnp.asarray(freqs_arr, dtype=jnp.float32)

    # ---- Coax stub (mirrors compute_coaxial_two_port's single end) --
    x_feed, y_centre, msl_z_lo = (float(c) for c in msl_pe.position)
    msl_port_base = _MSLPortLL(
        feed_x=x_feed,
        y_lo=y_centre - msl_pe.width / 2, y_hi=y_centre + msl_pe.width / 2,
        z_lo=msl_z_lo, z_hi=msl_z_lo + msl_pe.height,
        direction=msl_pe.direction, impedance=msl_pe.impedance,
        excitation=None,
    )
    center_xy = (float(port.position[0]), float(port.position[1]))
    a, b = float(port.pin_radius), float(port.outer_radius)
    from rfx.geometry.rasterize_grid import (
        _local_cell, _nearest_plane, coords_from_uniform_grid,
        cell_sizes_from_uniform_grid,
    )
    z_nodes = coords_from_uniform_grid(grid)[2]
    z_sizes = cell_sizes_from_uniform_grid(grid)[2]
    z_local = _local_cell(z_nodes, z_sizes, float(port.position[2]))
    z_junction_idx = _nearest_plane(
        z_nodes, float(port.position[2]), z_local,
        what="coax-MSL junction", axis=2)
    if z_junction_idx > msl_cross_section_span(grid, msl_port_base)["n_lo"]:
        raise ValueError(
            "compute_coax_msl_transition(): the realized coax junction is "
            "above the MSL ground plane; make both registrations refer "
            "to the same physical ground layer.")
    z_stub_lo = int(grid.pad_z_lo) + 2
    z_feed = z_stub_lo + 1
    z_src = z_stub_lo + 3
    z_stub_hi = z_junction_idx - 1
    if z_stub_hi <= z_src:
        raise ValueError(
            "compute_coax_msl_transition(): domain too short between "
            "the low-z CPML and the junction height for the coax "
            "stub's source/feed/probe layout; increase the z domain or "
            "lower position[2]."
        )
    probes_coax = sorted(
        z_src + int(probe_start_cells) + int(probe_spacing_cells) * k
        for k in range(int(probe_count))
    )
    if probes_coax[-1] >= z_stub_hi:
        raise ValueError(
            "compute_coax_msl_transition(): the coax probe array "
            f"reaches index {probes_coax[-1]}, at or past the junction "
            f"({z_stub_hi}); increase the z domain or reduce "
            "probe_count/probe_start_cells/probe_spacing_cells."
        )

    z_tem = coaxial_tem_characteristic_impedance(a, b)
    r_feed = float(feed_impedance) if feed_impedance is not None else float(z_tem)
    junction_materials = materials
    materials, shell_inner = stamp_coaxial_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_stub_lo,
        z_hi_index=z_stub_hi, pin_radius=a, outer_radius=b,
    )
    materials = stamp_coaxial_annular_resistor(
        grid, materials, center_xy=center_xy, z_index=z_feed,
        pin_radius=a, outer_radius=b, target_impedance=r_feed,
        shell_inner_radius=shell_inner,
    )
    # The shared line stamper includes axial padding for standalone
    # coax runs. Here the caller owns the junction, post and laminate:
    # stop the generated stub BELOW the junction node. Restore the
    # registered arrays, not air, so no DUT conductor/dielectric is cut.
    materials = materials._replace(
        eps_r=materials.eps_r.at[:, :, z_junction_idx:].set(
            junction_materials.eps_r[:, :, z_junction_idx:]),
        sigma=materials.sigma.at[:, :, z_junction_idx:].set(
            junction_materials.sigma[:, :, z_junction_idx:]),
    )

    src_port = _CoaxPort(
        position=(center_xy[0], center_xy[1], (z_src - grid.pad_z_lo) * dz),
        face="bottom", pin_length=dz, pin_radius=a, outer_radius=b,
        impedance=port.impedance, excitation=port.excitation,
    )
    spec_coax = build_coaxial_tem_plane_source_specs(
        grid=grid, port=src_port, n_steps=int(n_steps),
        field_scale=float(field_scale), magnetic_ratio=1.0,
    )
    ref_coax_m = (z_junction_idx - grid.pad_z_lo) * dz
    z_planes_coax_m = np.array(
        [(z - grid.pad_z_lo) * dz for z in probes_coax], dtype=np.float64
    )
    annulus_cells = float((b - a) / dz)
    if annulus_cells < 3.5:
        import warnings as _wa
        _wa.warn(
            f"compute_coax_msl_transition(): coax annulus resolution "
            f"{annulus_cells:.2f} cells is below the documented "
            "under-resolved threshold (3.5 cells, same convention as "
            "compute_coaxial_line_reflection/compute_coaxial_two_port) "
            "— reflection accuracy degrades at high frequency.",
            stacklevel=2,
        )

    # ---- MSL side (mirrors compute_mixed_s_matrix's MSL consumption) ---
    from rfx.sources.msl_port import validate_msl_port_geometry
    validate_msl_port_geometry(
        grid, msl_port_base, pec_edge_masks=_cx_pec_edge_masks,
        periodic=self._periodic_flags(),
        pec_faces=self._boundary_spec.pec_faces(), name=msl_pe.name)
    mode_profile = compute_msl_mode_profile(grid, msl_port_base, eps_r_sub_resolved)
    materials = setup_msl_port(grid, msl_port_base, materials, mode_profile=mode_profile)
    z0_msl, eps_eff_msl = hammerstad_jensen_z0_eps_eff(
        msl_pe.width, msl_pe.height, eps_r_sub_resolved
    )

    # Registered impedance= divergence advisory (issue #581 review N2):
    # add_coaxial_port(impedance=...) / add_msl_port(impedance=...) size
    # the feed resistor / termination sigma and (for coax) the TEM
    # source amplitude calibration — but the POWER-WAVE NORMALIZATION
    # (z0_ref, feeding sqrt(Z0) in the assembler) always uses the
    # ANALYTIC z_tem / z0_msl computed here, never the registered
    # impedance. A large silent divergence between the two is a
    # footgun: the source/termination is calibrated for one Z0 while
    # the extraction is normalized against another.
    for _label, _registered, _analytic in (
        ("coax", float(port.impedance), float(z_tem)),
        ("msl", float(msl_pe.impedance), float(z0_msl)),
    ):
        if _analytic > 0.0:
            _rel_dev = abs(_registered - _analytic) / _analytic
            if _rel_dev > 0.05:
                import warnings as _wz
                _wz.warn(
                    f"compute_coax_msl_transition(): the registered "
                    f"{_label} port impedance ({_registered:.2f} ohm) "
                    f"diverges {_rel_dev * 100:.1f}% from the analytic "
                    f"{_label} Z0 ({_analytic:.2f} ohm) this method "
                    "actually uses for the power-wave normalization "
                    "(z0_ref) and for sizing the feed resistor / "
                    "termination. The registered impedance= is NOT "
                    "the reference impedance of the returned "
                    "s_params; it only affects source/termination "
                    "sizing. Pass a matching pin_radius/outer_radius "
                    "(coax) or width/height/eps_r_sub (msl), or "
                    "reconcile the mismatch, before trusting a "
                    "specific reference-impedance interpretation.",
                    stacklevel=2,
                )

    # MSL-side probe ladder is independent of the coax-side one (issue
    # #489 leg 4 attempt 2) -- default to the coax values so an
    # existing caller (attempt 1's committed fixture) sees byte-
    # identical behavior when these new parameters are left unset.
    _msl_probe_count = int(probe_count if msl_probe_count is None else msl_probe_count)
    _msl_probe_start_cells = int(
        probe_start_cells if msl_probe_start_cells is None else msl_probe_start_cells
    )
    _msl_probe_spacing_cells = int(
        probe_spacing_cells if msl_probe_spacing_cells is None else msl_probe_spacing_cells
    )
    probe_xs = msl_probe_x_coords_n(
        grid, msl_port_base, n_probes=_msl_probe_count,
        n_offset_cells=_msl_probe_start_cells,
        n_spacing_cells=_msl_probe_spacing_cells,
    )
    xs_ladder = [float(x) for x in probe_xs]
    lx_dom = float(self._domain[0])
    mono = all(
        (xs_ladder[q + 1] - xs_ladder[q]) * (1 if msl_pe.direction == "+x" else -1)
        > 0.5 * dz
        for q in range(len(xs_ladder) - 1)
    )
    if (not mono) or min(xs_ladder) <= 0.0 or max(xs_ladder) >= lx_dom:
        raise ValueError(
            "compute_coax_msl_transition(): the MSL probe ladder "
            f"({', '.join(f'{x * 1e3:.2f}' for x in xs_ladder)} mm) "
            f"leaves the declared x-domain (0, {lx_dom * 1e3:.2f}) mm "
            "or was clamped at its edge. Face the port toward the "
            "junction (direction), reduce n_probe_offset/spacing, or "
            "enlarge the domain."
        )
    # coaxial_line_reflection_from_plane_voltages requires STRICTLY
    # INCREASING plane positions; a "-x"-facing port's own ladder comes
    # back decreasing in x (probe n steps AWAY from feed_x, toward the
    # junction). Sort once here and use this order everywhere below so
    # DFT-plane construction and the voltage array stay index-consistent.
    xs_sorted = sorted(xs_ladder)

    # ---- Issue #823: source near-field standoff on the REALIZED ladder.
    # ``msl_probe_count/start/spacing`` are METHOD arguments — they never
    # reach the registered ``_MSLPortEntry``, so preflight's own check 5
    # (``_check_msl_port_geometry``) is structurally blind to this ladder.
    # Evaluate the SAME predicate here, on the coordinates the extractor
    # will actually sample, at BOTH ends the ladder is referred to: the
    # port's own feed plane AND the reference plane at the junction.
    #
    # Emitted with ``warnings.warn``, deliberately NOT by constructing a
    # PreflightWarning: this is not a preflight check (this method never
    # calls preflight — it is DIAGNOSTIC_ONLY in
    # tests/unit/preflight/test_preflight_advisory_emission_contract.py's
    # EMISSION_CLASSIFICATION) and the emission-site freeze in that file
    # counts PreflightWarning/PreflightErrorWarning/PreflightIssue/
    # PreflightConfigError constructions only.
    #
    # REPORT-ONLY: no gate, no refusal, and no msl_fit_residual_max —
    # that gate is explicitly out of scope until the standoff rule and
    # the ladder self-consistency witness have both been exercised on a
    # settled run (PI sequencing).
    from rfx.api._preflight import (
        msl_source_near_field_standoff_cells as _msl_standoff_cells,
    )
    _standoff_cells = _msl_standoff_cells(float(msl_pe.height), float(dz))
    _standoff_m = _standoff_cells * float(dz)
    _feed_x_msl = float(msl_pe.position[0])
    _standoff_hits = []
    for _end_label, _end_x in (
        (f"the MSL port feed plane (x = {_feed_x_msl * 1e3:.2f} mm)", _feed_x_msl),
        (f"the reference plane at the junction "
         f"(x = {float(junction_x) * 1e3:.2f} mm)", float(junction_x)),
    ):
        _bad = [x for x in xs_sorted if abs(x - _end_x) < _standoff_m - 1e-12]
        if _bad:
            _standoff_hits.append(
                f"{len(_bad)} of {len(xs_sorted)} probes sit within "
                f"{_standoff_m * 1e3:.2f} mm of {_end_label} "
                f"(nearest {min(abs(x - _end_x) for x in _bad) * 1e3:.2f} mm "
                f"= {min(abs(x - _end_x) for x in _bad) / float(msl_pe.height):.2f}"
                f"·h_sub)"
            )
    if _standoff_hits:
        import warnings as _wnf
        _wnf.warn(
            "compute_coax_msl_transition(): the MSL probe ladder "
            f"(msl_probe_count={_msl_probe_count}, "
            f"msl_probe_start_cells={_msl_probe_start_cells}, "
            f"msl_probe_spacing_cells={_msl_probe_spacing_cells}; realized "
            f"x = {', '.join(f'{x * 1e3:.2f}' for x in xs_sorted)} mm) "
            "violates the source near-field standoff of "
            f"{_standoff_cells} cells ({_standoff_m * 1e3:.2f} mm = "
            "5·h_sub, the issue-#80 Fix B constant add_msl_port's own auto "
            "n_probe_offset already floors to): "
            + "; ".join(_standoff_hits)
            + ". Within a few substrate thicknesses of a launch "
            "discontinuity the field is not the guided mode yet, and the "
            "matrix-pencil fit reports the LADDER's error as the field's: "
            "measured on the settled attempt-3 run (VESSL 369367257533) a "
            "probe at 1.33·h_sub carried 82-128% two-wave model error and "
            "dragged the full-ladder fit_residual to 0.342/0.264/0.222, "
            "while every window excluding it fit to 1e-5..4e-3. Move the "
            "ladder inside [junction + 5·h_sub, feed - 5·h_sub] by raising "
            "msl_probe_start_cells and/or lowering msl_probe_count, and "
            "read result.ladder_split_gamma_dev / "
            "result.ladder_split_reflection_decades (computed when "
            "return_ladder_voltages=True) for whether this ladder "
            "actually disagrees with itself. REPORT-ONLY: nothing is "
            "refused.",
            stacklevel=2,
        )

    span = msl_cross_section_span(grid, msl_port_base)
    k_lo_msl, k_hi_msl = span["n_lo"], span["n_hi"]
    j_centre_msl = span["w_centre"]
    i_feed_msl = span["i_feed"]
    # #931 §1.9: realized wall planes locate the trace, not cells.
    from rfx.probes.msl_wave_decomp import (
        realized_trace_planes_on_column as _trace_planes_cx,
    )
    k_trace_lo, _ = _trace_planes_cx(
        _cx_pec_edge_masks, 2, (i_feed_msl, j_centre_msl), k_hi_msl,
        periodic=self._periodic_flags())
    if k_trace_lo is None:
        raise RuntimeError(
            "compute_coax_msl_transition(): no realized PEC trace "
            "conductor found above the substrate top at the registered "
            "MSL port's own feed plane; declare the microstrip trace as "
            "a Box(material='pec') (a volume) or as a zero-thickness "
            "Box / add_thin_conductor (a sheet, #931)."
        )
    dz_arr = _msl_cell_profile(grid, "z", grid.nz)
    _complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64

    # ---- Two-drive FDTD run -------------------------------------------
    msl_waveform = (
        msl_pe.waveform if msl_pe.waveform is not None
        else GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
    )
    import dataclasses as _dc
    msl_port_driven = _dc.replace(msl_port_base, excitation=msl_waveform)

    v_coax_by_drive = np.zeros((2, len(probes_coax), n_f), dtype=np.complex128)
    v_msl_by_drive = np.zeros((2, len(xs_sorted), n_f), dtype=np.complex128)
    settling_db = np.full(2, np.nan, dtype=np.float64)

    x_mid_coax = center_xy[0] + 0.5 * (a + b)
    i_probe_coax = int(round(x_mid_coax / dz)) + int(grid.pad_x_lo)
    j_probe_coax = int(grid.pad_y_lo) + int(round(center_xy[1] / dz))
    i_probe_msl = int(grid.position_to_index(
        (xs_sorted[len(xs_sorted) // 2], y_centre, msl_z_lo + msl_pe.height * 0.5)
    )[0])
    j_probe_msl = int(grid.pad_y_lo) + int(round(y_centre / dz))
    k_probe_msl = int(round((msl_z_lo + 0.5 * msl_pe.height) / dz)) + int(grid.pad_z_lo)

    # drive_idx 0 drives the coax port; drive_idx 1 drives the MSL port.
    for drive_idx in range(2):
        if drive_idx == 0:
            sources = list(spec_coax.electric_sources)
            mag_sources = list(spec_coax.magnetic_sources)
        else:
            sources = make_msl_port_sources(
                grid, msl_port_driven, materials, int(n_steps),
                mode_profile=mode_profile,
            )
            mag_sources = []

        planes = []
        for z in probes_coax:
            for comp in ("ex", "ey"):
                planes.append(init_dft_plane_probe(
                    axis=2, index=int(z), component=comp, freqs=freqs_jnp,
                    grid_shape=grid.shape, dft_total_steps=int(n_steps),
                ))
        n_coax_planes = len(planes)
        for x in xs_sorted:
            i_x = int(grid.position_to_index((x, y_centre, msl_z_lo))[0])
            planes.append(init_dft_plane_probe(
                axis=0, index=i_x, component="ez", freqs=freqs_jnp,
                grid_shape=grid.shape, dft_total_steps=int(n_steps),
            ))

        witness_probes = [
            ProbeSpec(i=i_probe_coax, j=j_probe_coax,
                      k=int(probes_coax[len(probes_coax) // 2]), component="ex"),
            ProbeSpec(i=i_probe_msl, j=j_probe_msl, k=k_probe_msl, component="ez"),
        ]

        # #589 flux-adjudication opt-in: fresh accumulators PER DRIVE
        # (init_flux_monitor zeroes the DFT carries; sharing cfgs across
        # drives would co-accumulate both drives into one spectrum).
        _flux_run_kwargs = (
            {"flux_monitors": build_flux_monitor_cfgs(
                self, grid, int(n_steps), entries=extra_flux_monitors)}
            if extra_flux_monitors else {}
        )
        result = _run(
            grid, materials, int(n_steps), boundary="cpml", cpml_axes="xyz",
            sources=sources, mag_sources=mag_sources, probes=witness_probes,
            dft_planes=planes, pec_edge_masks=_cx_pec_edge_masks,
            return_state=False,
            **_flux_run_kwargs,
        )
        if result.dft_planes is None:
            raise RuntimeError(
                "compute_coax_msl_transition(): runner returned no DFT "
                "planes"
            )
        if extra_flux_monitors:
            flux_by_drive[("coax", "msl")[drive_idx]] = {
                entry.name: np.asarray(_flux_spectrum(fm, exact_f64=True), dtype=np.float64)
                for entry, fm in zip(
                    extra_flux_monitors, result.flux_monitors or ()
                )
            }

        for pi, z in enumerate(probes_coax):
            v_coax_by_drive[drive_idx, pi, :] = np.asarray(
                coaxial_line_plane_voltage(
                    grid, result.dft_planes[pi * 2 + 0].accumulator,
                    result.dft_planes[pi * 2 + 1].accumulator,
                    center_xy=center_xy, pin_radius=a, outer_radius=b,
                )
            )
        for pi in range(len(xs_sorted)):
            ez_plane = jnp.asarray(result.dft_planes[n_coax_planes + pi].accumulator)
            v_q = msl_modal_voltage(
                ez_plane, j_centre=j_centre_msl, k_lo=k_lo_msl,
                k_hi=k_trace_lo, dz_arr=dz_arr, dtype=_complex_dtype,
            )
            v_msl_by_drive[drive_idx, pi, :] = np.asarray(v_q)

        ts = np.asarray(result.time_series, dtype=float)
        if ts.ndim == 2 and ts.shape[0] >= 10 and ts.shape[1] == len(witness_probes):
            power = ts ** 2
            tail = max(1, power.shape[0] // 10)
            end = power[-tail:, :].mean(axis=0)
            peak = power.max(axis=0)
            tiny = np.finfo(float).tiny
            settling_db[drive_idx] = float(np.max(
                10.0 * np.log10((end + tiny) / (peak + tiny))
            ))

    s_params, cond_a, cond_a_equilibrated, rec_resid, fit_resid, gamma, a_inc, b_out = \
        _assemble_coax_msl_transition_from_voltages(
            z_coax_planes_m=z_planes_coax_m, x_msl_planes_m=np.asarray(xs_sorted),
            ref_coax_m=ref_coax_m, ref_msl_m=float(junction_x),
            v_coax_by_drive=v_coax_by_drive, v_msl_by_drive=v_msl_by_drive,
            z0_coax=float(z_tem), z0_msl=float(z0_msl), cond_warn=float(cond_warn),
        )

    reference_planes = np.asarray([ref_coax_m, float(junction_x)], dtype=float)
    z0_ref = np.asarray([float(z_tem), float(z0_msl)], dtype=float)

    # ---- Issue #823 ladder self-consistency witness (REPORT-ONLY) ----
    # Refit each port's ladder on two disjoint contiguous halves with the
    # same reference plane and the same extractor, and report the
    # disagreement. Computed AFTER the assembler has produced every
    # number above, from the same arrays, so it cannot move one:
    # ``fit_residual`` cannot detect that its own window is the problem,
    # this can. See _ladder_split_witness for the measured separation.
    # OPT-IN with the ladder dump: it is a Python-loop refit (2 drives x
    # n_freqs x 2 matrix pencils per ladder) over the very arrays
    # ``return_ladder_voltages`` exposes, so a default call never runs
    # it and both fields stay None.
    ladder_split_gamma_dev = None
    ladder_split_reflection_decades = None
    if return_ladder_voltages:
        _split_g_coax, _split_d_coax = _ladder_split_witness(
            z_planes_coax_m, v_coax_by_drive, ref_coax_m)
        _split_g_msl, _split_d_msl = _ladder_split_witness(
            np.asarray(xs_sorted), v_msl_by_drive, float(junction_x))
        ladder_split_gamma_dev = np.stack([_split_g_coax, _split_g_msl])
        ladder_split_reflection_decades = np.stack([_split_d_coax, _split_d_msl])

    # #589 ladder dump (read-only). Taken AFTER the assembler consumed
    # the very same arrays, so no number above can depend on the flag.
    # ``msl_ladder_i`` is recomputed here with the same expression the
    # drive loop used (``i_x`` there is loop-local by construction).
    ladder_voltages = None
    if return_ladder_voltages:
        ladder_voltages = {
            "coax_ladder_v": v_coax_by_drive.copy(),
            "coax_ladder_z_m": np.asarray(z_planes_coax_m, dtype=np.float64).copy(),
            "coax_ladder_k": np.asarray(probes_coax, dtype=np.int64),
            "msl_ladder_v": v_msl_by_drive.copy(),
            "msl_ladder_x_m": np.asarray(xs_sorted, dtype=np.float64),
            "msl_ladder_i": np.asarray(
                [int(grid.position_to_index((x, y_centre, msl_z_lo))[0])
                 for x in xs_sorted],
                dtype=np.int64,
            ),
            "drive_order": ("coax", "msl"),
            "ref_coax_m": float(ref_coax_m),
            "ref_msl_m": float(junction_x),
            "z0_ref": z0_ref.copy(),
        }

    result_obj = CoaxMSLTransitionResult(
        s_params=s_params,
        freqs=np.asarray(freqs_arr, dtype=float),
        port_names=("coax", "msl"),
        reference_planes=reference_planes,
        z0_ref=z0_ref,
        cond_a=cond_a,
        cond_a_equilibrated=cond_a_equilibrated,
        recurrence_residual=rec_resid,
        fit_residual=fit_resid,
        gamma=gamma,
        a_inc=a_inc,
        b_out=b_out,
        settling_db=settling_db,
        ladder_split_gamma_dev=ladder_split_gamma_dev,
        ladder_split_reflection_decades=ladder_split_reflection_decades,
        status="experimental",
        flux_monitors=(flux_by_drive if extra_flux_monitors else None),
        ladder_voltages=ladder_voltages,
    )
    # Issue #662, same gap as compute_coaxial_two_port. ``n_steps`` here is
    # the RESOLVED record length (num_periods was folded into it above), and
    # it is the knob that overrides num_periods, so it is the actionable one.
    _warn_if_ringdown_truncated(
        settling_db, ("coax", "msl"), n_steps=int(n_steps),
    )
    try:
        return _finalize_sparam_result(
            result_obj,
            extractor="compute_coax_msl_transition",
            strict=strict_passivity,
        )
    except ValueError as exc:
        # Lane-local, on purpose. _finalize_sparam_result's message is shared
        # by five extractors and tells the reader to inspect the V/I dump --
        # good advice, but it cannot name an escape hatch that only this lane
        # has. Appending here keeps the other four messages byte-identical
        # (issue #838). ValueError is preserved and the original is chained,
        # so `except ValueError` callers and the traceback are unaffected.
        raise ValueError(f"{exc} {COAX_MSL_TRANSITION_REFUSAL_HINT}") from exc


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# The four functions above were ``def``s in the ``_SparamMixin`` class body,
# so their ``__qualname__`` read ``_SparamMixin.<name>``; a module-level
# ``def`` gets the bare ``<name>`` instead. ``rfx/api/__init__.py`` rewrites
# exactly ``_SparamMixin.<name>`` -> ``Simulation.<name>`` at
# class-composition time so that a bad keyword argument reports
# ``Simulation.compute_coaxial_two_port() got an unexpected keyword
# argument``, and it SKIPS any function whose qualname does not match that
# pattern. Leaving the bare name here would therefore change those TypeError
# messages -- a user-visible behaviour change in a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` pins it.
# ---------------------------------------------------------------------------
compute_coaxial_s_matrix.__qualname__ = "_SparamMixin.compute_coaxial_s_matrix"
compute_coaxial_line_reflection.__qualname__ = (
    "_SparamMixin.compute_coaxial_line_reflection"
)
compute_coaxial_two_port.__qualname__ = "_SparamMixin.compute_coaxial_two_port"
compute_coax_msl_transition.__qualname__ = (
    "_SparamMixin.compute_coax_msl_transition"
)

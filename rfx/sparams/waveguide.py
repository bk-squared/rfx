"""Waveguide S-matrix calculators bound onto the public Simulation API.

Drive-specific declarations and run state belong to private declaration
variants. The caller retains its declared port and observer containers.
"""

from __future__ import annotations

from rfx.sparams import _waveguide_stages

from rfx.errors import AccuracyWarning, RfxUnsupported
from rfx.stepping.step_count import steps_for_periods
from rfx._diagnostic_transport import report_diagnostics
from rfx._internal._precision import resolve_field_dtype
from rfx.preflight.realization import release_inspection
from rfx.model.thin_conductors import conductivity_envelope

from rfx.sparams._waveguide_planes import resolve_reported_planes
from rfx._declarations import declaration_variant

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.measurement.phasors import exp_j
from rfx.measurement.waveguide_s import (
    assemble_normalized_waveguide_column,
    _waveguide_s21_phase_residual,
)
from rfx.sources.waveguide_port import (
    extract_waveguide_s_matrix,
    extract_waveguide_s_matrix_flux,
    extract_waveguide_s_params_normalized,
    extract_multimode_s_matrix,
    extract_multimode_s_matrix_flux,
    waveguide_plane_positions,
)

from rfx.api._spec import (
    WaveguideSMatrixResult,
    _WaveguidePortEntry,
)

from rfx.sparams._common import (
    _finalize_sparam_result,
    _warn_if_ringdown_truncated,
    _warn_junction_probe_clearance,
    _warn_junction_cpml_thickness,
    _warn_ntff_box_dropped,
    _warn_thin_absorber_vs_guide_wavelength,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Type-only forward reference for the
    # ``port_reference_sims: "list[Simulation] | None"`` annotation, carried
    # over unchanged from ``rfx.api._sparams``. NOT a runtime import: the
    # module-import contract above forbids a runtime ``from rfx.api import
    # ...``, and a TYPE_CHECKING guard never executes.
    from rfx.api import Simulation


from rfx.model.waveguide_mesh import flux_mesh_entry


@jax.custom_jvp
def _flux_power_ratio(numerator, denominator):
    """Keep the primal division; avoid squaring tiny reference powers in AD."""
    return numerator / denominator


@_flux_power_ratio.defjvp
def _flux_power_ratio_jvp(primals, tangents):
    numerator, denominator = primals
    d_numerator, d_denominator = tangents
    ratio = _flux_power_ratio(numerator, denominator)
    # Mesh deformation traces the reference as well as the device. Powers
    # can be ~1e-24 in float32: d**-2 overflows even when dq is finite.
    return ratio, (d_numerator - ratio * d_denominator) / denominator


@flux_mesh_entry
def compute_waveguide_s_matrix(
    self,
    *,
    n_steps: int | None = None,
    num_periods: float = 20.0,
    normalize: bool | str = False,
    subpixel_smoothing: bool | str = False,
    conformal_pec: str | None = None,
    eps_override: "jnp.ndarray | None" = None,
    sigma_override: "jnp.ndarray | None" = None,
    checkpoint_segments: int | None = None,
    strict_passivity: bool = False,
    skip_preflight: bool = False,
    port_reference_sims: "list[Simulation] | None" = None,
) -> WaveguideSMatrixResult:
    """Compute a theoretically clean axis-normal boundary-aperture waveguide S-matrix.

    Requires ``solver="yee"``. On a uniform mesh, ``precision`` selects
    field storage for every device and reference run, as on ``run()``.
    A graded mesh supports ``precision="float32"`` and ``"float64"``.

    Parameters
    ----------
    skip_preflight : bool
        Skip advisories only when True; realization refusals still apply. Default False.
    num_periods : float
        Length of the FDTD run (in source-period multiples) used to
        derive ``n_steps`` when ``n_steps`` is not given. The
        spectra are computed POST-SCAN from the recorded modal V/I
        time series via a rectangular full-record DFT (matching
        OpenEMS's ``utilities.DFT_time2freq``); ``num_periods``
        therefore governs both the CPML drain horizon AND the DFT
        integration window. The rect
        full-record DFT is finite-energy on the recorded transient
        so no gating is needed even on strong reflectors.
    normalize : bool or "flux"
        Controls the S-parameter extraction algorithm:

        ``False`` (default) — modal V/I decomposition, no reference
        run.  Magnitude includes Yee impedance mismatch
        (Z_TE_num/Z_TE_exact ≈ 3 % at dx/λ = 0.07).  Use for
        |S11| of strong reflectors (PEC short, high-Q resonators)
        where this error is smaller than the ±10–20 % round-trip
        dispersion error introduced by ``normalize=True``.  On the
        differentiable chain: a traced ``eps_override`` /
        ``sigma_override`` flows through it.

        ``True`` — two-run modal normalization at the declared port planes.
        Every entry divides by the driven port's reference incident wave;
        the diagonal first subtracts the reference outgoing wave. Retains
        the same one-way Yee dispersion as ``False``. Unlike ``False``,
        the denominator excludes device-reflection leakage into the modal
        incident-wave split, and the diagonal removes reference reflection.
        Form the empty-guide insertion ratio from two calls: device S21
        divided by S21 of the same guide without the device, at the same planes.
        Host-assembled and **outside the differentiable chain**: a traced
        ``eps_override`` /
        ``sigma_override`` (``jax.grad``, ``jax.jit``) raises
        ``NotImplementedError`` at this call, before any FDTD run;
        a concrete override still runs forward.

        ``"flux"`` — hybrid power-flux extraction.  Magnitude from
        Poynting-vector DFT (|S|² = P_flux / P_inc), phase from
        modal V/I.  Corrects both the Z_TE impedance-mismatch error
        in S11 and the round-trip dispersion error in the
        ``normalize=True`` diagonal formula.  Costs 2 × N_ports
        FDTD runs (same as ``normalize=True``).  On the
        differentiable chain like ``False``. On the uniform lane the
        result promotes field, frequency, and JAX default float precision,
        like ``False``: complex64 with x64 off and at least complex128 with
        x64 on, including explicit float32 frequencies and float32/mixed fields.
        Modal records retain at least the frequency and default float precision.
        The non-uniform lane's result follows the ``freqs`` precision:
        complex64 by default, complex128 under ``JAX_ENABLE_X64``.

        **Reference impedance.** Each port decomposes waves with its own
        medium's Yee-discrete modal impedance. On ``False``, unequal
        media use the power-wave ratio ``(b_i / a_j) * sqrt(Z_j / Z_i)``.
        Equal media retain the voltage-wave ratio exactly. ``True`` and
        ``"flux"`` require equal port media and use a homogeneous reference
        filled with that medium. ``"flux"`` takes |S_ij| from
        the power ratio and its phase from the same modal waves.
        The returned matrix therefore equals a power-wave S only
        when every port shares one cross-section and mode — the
        validated scope (straight guides and same-guide junctions).
        Ports with dissimilar cross-sections are outside it.

        **Multimode.**  ``n_modes > 1`` on any port is assembled on
        the host for every ``normalize`` value and is outside the
        differentiable chain; ``eps_override`` / ``sigma_override``
        differentiation is single-mode only.
    conformal_pec : str or None
        None follows Boundary(conformal=True); the keyword only accepts
        "cut_cell_floor", which uses true edge lengths and floored
        cut-face areas on uniform device and boundary-wall reference scans.
        It requires Simulation(dt=q * Grid.courant_dt(realized_dx)), q=0.5
        or 0.25, and refuses custom port_reference_sims.
    checkpoint_segments : int or None
        Segmented gradient checkpointing for the **uniform** waveguide
        AD path.  Splits the ``n_steps`` scan
        into ``K`` segments that are rematerialised via
        ``jax.checkpoint`` during the backward pass, reducing peak
        reverse-mode memory from O(n_steps·|carry|) to
        O((K + n_steps/K)·|carry|) (≈ O(√n_steps·|carry|) at the
        optimal K ≈ √n_steps, at ≈ 2× backward compute cost).
        ``K`` is forwarded to ``rfx.simulation.run`` with
        ``checkpoint=True``; ``K`` MUST exactly divide the
        auto-computed ``n_steps`` (the runner rejects non-divisors —
        choose the nearest divisor of √n_steps; padding is rejected
        because it would shift the V/I DFT windows).  Default
        ``None`` is byte-identical to the pre-checkpoint scan.

        On a NON-uniform mesh (``dx_profile`` / ``dy_profile`` /
        ``dz_profile``) ``checkpoint_segments=K`` is now supported: ``K`` is
        translated to the NU runner's ``checkpoint_every`` chunk size — the
        divisor of ``n_steps`` nearest to ``n_steps/K`` — and applied to the
        *device* run only (the vacuum reference is constant in the design
        variable). The NU runner runs a non-divisor chunk as a real tail
        segment rather than padding the record; this calculator passes a
        divisor anyway. The result is forward-IDENTICAL
        and the ≈O(√n_steps) tape reduction is realised under ``jax.grad``
        with ``normalize='flux'`` + an ``eps_override`` / ``sigma_override``
        design variable.
    port_reference_sims : list[Simulation] or None
        Per-driven-port matched-straight-guide reference simulations for
        **interior-PEC multi-port** structures (T-junctions, branches,
        septa). ``port_reference_sims[i]`` is a ``Simulation`` describing
        the STRAIGHT continuation of driven port ``i``'s guide with **no
        junction** — same domain / ``dx`` / boundary, geometry = port
        ``i``'s guide walls extended straight through. Only valid with
        ``normalize='flux'`` (raises otherwise); single-mode ports only;
        uniform mesh only; not combinable with ``eps_override`` /
        ``sigma_override``.

        **Multi-port junctions (port_reference_sims).** The default flux
        path references incident power ``P_inc`` to a single shared VACUUM
        run (empty domain, no interior PEC). For a straight guide the walls
        come from ``pec_axes`` so that vacuum reference already carries the
        guided ``P_inc`` correctly. For a junction the interior PEC septum /
        branch is stripped from the vacuum reference, which then radiates
        into free space: ``P_inc`` is mis-normalized and every ``|S|``
        inflates hard (non-passive; ``normalize=True`` gave max|S|~230,
        ``normalize='flux'`` gave max|S|~9.8, |S11|~1.9 on the verified
        compact 3-port T-junction). Passing ``port_reference_sims[i]`` = the
        matched straight guide for driven arm ``i`` moves ``P_inc`` toward
        the true single-mode guided incident power (PEC-folded identically
        to the device run).

        **Far-port discipline (required for a physical S-matrix).** This
        plumbing is NECESSARY but NOT SUFFICIENT. A physical junction
        S-matrix additionally requires: (1) each port's probe plane placed
        >= 5 evanescent decay lengths of the next higher mode away from the
        junction; (2) CPML thickness >= ~0.5 guide wavelengths at band
        centre; (3) a converged mesh. On far-port geometry (arms 90/90/70 mm,
        48 mm CPML, dx 1.0/0.667 mm) the matched-reference flux path reaches
        passivity 1.006/1.002, reciprocity 0.001, mesh-convergence 0.0297
        and 0.087 vs MEEP (r2000, cross-device). On COMPACT geometry the
        matched reference fixes |S11| (1.86 → 0.49, physical) but the overall
        matrix stays non-physical (residual max|S|~3.9); the two in-method
        advisories (probe clearance, CPML thickness) warn when the far-port
        discipline is not met.

        The absorber half of that discipline is checked on **every**
        uniform path, not just this one: an in-method advisory
        fires whenever the absorber on a port's propagation axis is
        thinner than ``0.5 * lambda_g`` at the **lowest** measured
        frequency, which is where ``lambda_g`` is longest and the
        ``cpml_layers=16`` default weakest. It is checked in-method
        to retain the calculator-specific setup advisory alongside shared preflight.
        Treat ``0.5 * lambda_g`` as a floor, not a target: at the WR-90
        band edge the measured residual ``|S11|`` ripple was 0.0706 at
        0.30 ``lambda_g``, 0.0366 at 0.50 and 0.0093 at 0.75, so a
        0.5-``lambda_g`` absorber can still set the accuracy envelope
        instead of discretization. The advisory is silent on a
        non-uniform mesh (``cpml_layers * dx`` is ambiguous under a
        graded profile) and on a band that starts at or below cutoff
        (``lambda_g`` is undefined there; the ``port_freqs_below_cutoff``
        preflight owns that case).

        This enables junction measurements UNDER the documented
        discipline; it does NOT make arbitrary compact junctions
        valid. Consult ``docs/guides/sparameter_support_matrix.md`` for
        the current waveguide evidence and scope.

        On a graded mesh, flux normalization admits a traced x profile only
        with ``nominal_profiles={"x": NominalProfile(cells, traced_cells=window)}``.
        The window must avoid the port planes and interior ends and retain
        its nominal total width. Only total-preserving tangents describe
        derivatives with the port planes fixed in physical metres; a tangent
        changing that total instead moves downstream nodes with the mesh.
        Material rasterization is a cell-centre step test on realized nodes.
        ``run()`` and ``forward()`` still refuse traced waveguide meshes.
    """
    conformal_pec = _waveguide_stages._admit_waveguide_options(
        RfxUnsupported, conformal_pec, port_reference_sims, resolve_field_dtype, self, subpixel_smoothing,
    )
    if not normalize:
        import warnings
        warnings.warn(
            "compute_waveguide_s_matrix(normalize=False): S21 and "
            "S-parameter phase include Yee numerical dispersion. "
            "normalize=True also retains propagation between the declared "
            "planes, using the reference incident wave for every entry. "
            "For |S11| of strong reflectors (PEC short, resonators) "
            "normalize=False is more accurate — see the normalize "
            "parameter docstring.", AccuracyWarning,
            stacklevel=2,
        )
    entries = _waveguide_stages._validate_waveguide_ports(
        eps_override, normalize, port_reference_sims, self, sigma_override,
    )

    # audit: same silent NTFF drop class as the MSL path.
    _warn_ntff_box_dropped(self, "compute_waveguide_s_matrix()")

    # Non-uniform-mesh dispatch checks all three profiles, including a
    # dz_profile-only mesh. The dedicated NU two-run extractor below is enabled
    # when its supported scope is met (``normalize=True``, single-mode ports);
    # otherwise raise so the user is not given silently-wrong numbers. Resolve
    # the frequency grid here because the NU absorber advisory needs it; a
    # shared resolver also applies the matching-grid check consistently.
    from rfx.api._preflight import resolve_waveguide_port_freqs

    def _resolve_freqs(entry: _WaveguidePortEntry) -> jnp.ndarray:
        # ONE definition, shared with preflight_sparameters(calculator=
        # "waveguide") so the setup audits read the run's own band.
        return resolve_waveguide_port_freqs(self, entry)

    freqs = _resolve_freqs(entries[0])
    for entry in entries[1:]:
        entry_freqs = _resolve_freqs(entry)
        if entry_freqs.shape != freqs.shape or not np.allclose(np.asarray(entry_freqs), np.asarray(freqs)):
            raise ValueError("waveguide S-matrix requires matching frequency grids on all ports")

    if (any(profile is not None for profile in
            (self._dx_profile, self._dy_profile, self._dz_profile))
            and self._interface_eps == "dual_average"):
        raise ValueError("interface_eps='dual_average' is not supported on the S-parameter NU lane")
    from rfx.boundaries.pmc import refuse_waveguide_pmc
    refuse_waveguide_pmc(self)
    from rfx.runners._admission import admit
    admit(self, "waveguide_s_matrix", run_args={"conformal_pec": conformal_pec})
    if port_reference_sims is not None:
        for reference in port_reference_sims:
            refuse_waveguide_pmc(reference)
            admit(reference, "waveguide_s_matrix")

    if not skip_preflight:
        report_diagnostics(self._auto_preflight(
            context="compute_waveguide_s_matrix", check_ntff="advisory",
            tfsf_material_overrides=(eps_override, sigma_override, None)))

    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ):
        _waveguide_stages._validate_waveguide_nu(
            checkpoint_segments, entries, eps_override, normalize, port_reference_sims, sigma_override, subpixel_smoothing,
        )
        # Far-port absorber advisory, on the NU lane too.
        # It used to sit ~110 lines below this branch's return, i.e. it was
        # UNREACHABLE from here: the guard existed, its text named the exact
        # remedy, and no NU caller ever asked it. That silence is why both NU
        # fixture producers shipped 0.33 and 0.099 lambda_g stacks — patching
        # the two producers would have left the next one to repeat it. Pure
        # NumPy, no FDTD, so it costs a grid build and a few array ops.
        # No bare `except: pass` around it: an advisory that can fail
        # silently is the failure mode this finding IS. If the plumbing ever
        # breaks, the caller hears about it as a warning rather than getting
        # the same quiet all-clear that let 0.33 lambda_g ship.
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        try:
            _nu_grid = self._build_nonuniform_grid()
            _nu_cfgs = [
                # n_steps is still None here when the caller passed
                # num_periods; the advisory reads only the config's
                # propagation axis and modal cutoff, never anything derived
                # from the step count, so a placeholder is honest rather
                # than a lie about the run length.
                _build_waveguide_port_config_nu(
                    self, _e, _nu_grid, jnp.asarray(freqs),
                    int(n_steps) if n_steps else 1)
                for _e in entries
            ]
            from rfx.preflight.waveguide import resolve_preflight_port_configs
            _nu_cfgs = resolve_preflight_port_configs(self, _nu_grid, _nu_cfgs,
                eps_override=eps_override, sigma_override=sigma_override)
        except (TimeoutError,):
            raise
        except Exception as _exc:  # noqa: BLE001 - recorded in the UNCHECKED AccuracyWarning
            import warnings as _w
            _w.warn(
                "compute_waveguide_s_matrix: could not evaluate the far-port "
                f"absorber advisory on the non-uniform lane ({_exc!r}). The "
                "0.5 guide-wavelength discipline is therefore UNCHECKED for "
                "this run — verify the absorber depth by hand.", AccuracyWarning,
                stacklevel=2,
            )
        else:
            _warn_thin_absorber_vs_guide_wavelength(
                _nu_grid, _nu_cfgs, freqs, self._cpml_layers,
                self._boundary_spec,
            )
            # Record-vs-far-boundary and port-index mirror audits
            # (post-v1.8 plan item 2). Same shared hook the uniform lane
            # below calls and preflight_sparameters(calculator=
            # "waveguide") calls; the grid and configs this lane already
            # built are passed in so no second mode solve is paid for.
            import warnings as _wg_warnings_nu
            self._preflight_waveguide_setup(
                _wg_warnings_nu, freqs=freqs, num_periods=num_periods,
                grid=_nu_grid, cfgs=_nu_cfgs,
                n_steps=int(n_steps) if n_steps else None,
            )

        _res_nu = self._compute_waveguide_s_matrix_nu(
            n_steps=n_steps,
            num_periods=num_periods,
            normalize=normalize,
            eps_override=eps_override,
            sigma_override=sigma_override,
            checkpoint_segments=checkpoint_segments,
        )
        completed = _finalize_sparam_result(
            _res_nu,
            extractor="compute_waveguide_s_matrix",
            strict=strict_passivity,
            check_reciprocity=True,
            # normalize=False carries documented Yee-dispersion + band-edge
            # |S11| overshoot (validated paths reach ~1.4-1.7), so use a
            # loose bound there that still catches gross extractor bugs
            # (|S11|>>1); reference subtraction / flux use the tighter bound.
            passivity_tol=2.0 if normalize is False else 0.10,
        )
        release_inspection(self)
        return completed

    _waveguide_stages._validate_waveguide_uniform(
        eps_override, is_tracer, normalize, port_reference_sims, self, sigma_override,
    )

    grid = self._build_grid()
    _wg_sheet_specs: list = []
    _wg_pec_sheets: list = []
    _wg_pec_wires: list = []
    from rfx.model.conductors import solve_conductors, assembled_materials
    conductors = solve_conductors(
        self, grid, realization_options=(subpixel_smoothing, conformal_pec),
        realization_not_evaluated="conformal" if conformal_pec else None)
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, pec_shapes, boundary_pec_shapes, _ = assembled_materials(
        conductors, sheet_specs=_wg_sheet_specs, pec_sheets=_wg_pec_sheets,
        pec_wires=_wg_pec_wires)
    _wg_pec_sheets = tuple(_wg_pec_sheets)
    _wg_pec_wires = tuple(_wg_pec_wires)
    _waveguide_stages._validate_waveguide_materials(
        conformal_pec, debye_spec, eps_override, lorentz_spec, pec_shapes, self, subpixel_smoothing,
    )
    _wg_pec_edge_masks, _wg_geometry_pec_edges, _wg_sheet_ctx, materials = _waveguide_stages._prepare_waveguide_conductors(
        base_materials, conductors, eps_override, sigma_override, subpixel_smoothing,
    )

    ref_materials_per_port, ref_pec_edge_masks_per_port, ref_pec_masks = _waveguide_stages._prepare_waveguide_references(
        grid, port_reference_sims,
    )

    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)
    from rfx.model.pole_stability import courant_budget
    _, debye, lorentz = self._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec, budget=courant_budget(grid),
        periodic=self._periodic_flags(), grid=grid)

    has_multimode, raw_cfgs, reference_medium = _waveguide_stages._prepare_waveguide_configs(
        _reference_medium, debye, entries, freqs, grid, lorentz, materials,
        n_steps, normalize, pec_mask_wg, ref_materials_per_port, ref_pec_masks, self,
    )


    # Unify source waveform across all ports so that the S-matrix
    # extraction uses identical excitation.  Different source spectra
    # (from mismatched f0/bandwidth) cause S11 ≠ S22 artifacts in the
    # unnormalized path because V/I decomposition error varies with
    # frequency.  Use port 0's waveform as the canonical source.
    def _flatten_cfgs(cfgs):
        out = []
        for c in cfgs:
            if isinstance(c, list):
                out.extend(c)
            else:
                out.append(c)
        return out

    flat0 = _flatten_cfgs(raw_cfgs)
    ref_t0 = flat0[0].src_t0
    ref_tau = flat0[0].src_tau
    need_unify = any(
        c.src_t0 != ref_t0 or c.src_tau != ref_tau for c in flat0[1:]
    )
    if need_unify:
        raw_cfgs = [
            cfg._replace(src_t0=ref_t0, src_tau=ref_tau)
            if not isinstance(cfg, list)
            else [c._replace(src_t0=ref_t0, src_tau=ref_tau) for c in cfg]
            for cfg in raw_cfgs
        ]

    # Far-port absorber advisory for EVERY uniform two-port path. Emitted here, before the
    # FDTD runs and before the
    # single-mode / multimode split, because the functional entry points
    # run no sim.preflight() and the port_reference_sims advisory below
    # covers only the junction path.
    _warn_thin_absorber_vs_guide_wavelength(
        grid, raw_cfgs, freqs, self._cpml_layers, self._boundary_spec,
    )

    # Record-vs-far-boundary and port-index mirror audits (post-v1.8 plan
    # item 2). The same shared hook the non-uniform lane above and
    # preflight_sparameters(calculator="waveguide") call, given this
    # lane's already-built grid, configs and step count.
    import warnings as _wg_warnings
    self._preflight_waveguide_setup(
        _wg_warnings, freqs=freqs, num_periods=num_periods,
        grid=grid, cfgs=raw_cfgs, n_steps=int(n_steps),
    )

    if subpixel_smoothing:
        from rfx.geometry.smoothing import refuse_pmc_smoothing
        refuse_pmc_smoothing(self, grid)
    from rfx.model.tensor_operands import (
        form_tensor_operands, finish_tensor_operands, finish_conformal_mu, form_reference_operands,
    )
    aniso_eps, aniso_inv_eps, conformal_weights, _unextendable, conformal_areas = form_tensor_operands(
        self, grid, materials, subpixel_smoothing=subpixel_smoothing,
        conformal=conformal_pec, conformal_min_weight=0.1,
        pec_shapes=pec_shapes, zero_inverse_on=pec_mask_wg)
    if subpixel_smoothing:
        from rfx.geometry.smoothing import warn_unextendable_shapes
        warn_unextendable_shapes(_unextendable)
    if conformal_weights is not None:
        from rfx.model.materials import with_components
        from rfx.simulation import resolve_periodic
        materials = with_components(materials, grid, periodic=resolve_periodic(grid, None))
    materials = finish_conformal_mu(materials, conformal_areas)
    aniso_eps = finish_tensor_operands(materials, aniso_eps, conformal_weights)
    ref_aniso_eps, ref_aniso_inv_eps, ref_conformal = form_reference_operands(
        grid, materials, subpixel_smoothing=subpixel_smoothing,
        conformal_weights=conformal_weights, boundary_pec_shapes=boundary_pec_shapes,
        conformal_pec=conformal_pec,
        reference_eps=float(reference_medium.eps_r))

    from rfx.model.realization_table import finish_table
    realization_record = finish_table(self, conductors, lane="waveguide_s_matrix")
    from typing import Callable, cast
    # with_forward_geometry adds the metadata keyword to the tuple constructor.
    realization_result = cast(Callable[..., WaveguideSMatrixResult], WaveguideSMatrixResult)

    if has_multimode and _wg_sheet_ctx is not None:
        raise ValueError(
            "surface-impedance (surface_impedance_f0) sheets are not "
            "supported on the multimode waveguide S-matrix path"
            ": the multimode extractors do not thread the "
            "sheet operator ctx, so the runs would silently simulate "
            "NO sheet. Use n_modes=1 ports or drop the f0 sheet.")
    if has_multimode:
        s_params, reference_planes, port_names_mm, port_directions_mm, settling_db_mm, settling_details_mm = _waveguide_stages._extract_waveguide_multimode(
            _homogeneous_reference_materials, _wg_pec_edge_masks, aniso_eps, aniso_inv_eps, conformal_weights, debye, entries,
            extract_multimode_s_matrix, extract_multimode_s_matrix_flux, grid, lorentz, materials, n_steps, normalize,
            np, raw_cfgs, ref_aniso_eps, ref_aniso_inv_eps, ref_conformal, reference_medium, resolve_reported_planes,
            self, waveguide_plane_positions,
        )
        _warn_if_ringdown_truncated(
            settling_db_mm, tuple(port_names_mm), num_periods=float(num_periods),
            witnesses=settling_details_mm)
        _res_mm = realization_result(
            realized_geometry=realization_record,
            s_params=s_params,
            freqs=jnp.asarray(freqs),
            port_names=tuple(port_names_mm),
            port_directions=tuple(port_directions_mm),
            reference_planes=reference_planes,
            settling_db=settling_db_mm, settling_witness=tuple(settling_details_mm),
        )
        completed = _finalize_sparam_result(
            _res_mm,
            extractor="compute_waveguide_s_matrix",
            strict=strict_passivity,
            check_reciprocity=True,
            passivity_tol=2.0 if normalize is False else 0.10,
        )
        release_inspection(self)
        return completed

    # Single-mode path (original behavior)
    cfgs = raw_cfgs

    # Far-port discipline advisories for interior-PEC junction references.
    # Pure-numpy heuristics emitted BEFORE the FDTD runs; no simulation.
    if port_reference_sims is not None:
        _warn_junction_probe_clearance(
            grid, cfgs, conductivity_envelope(materials),
            [conductivity_envelope(m) for m in ref_materials_per_port], freqs,
            device_pec_edges=_wg_geometry_pec_edges,
            ref_pec_edges=ref_pec_edge_masks_per_port,
        )
        _warn_junction_cpml_thickness(
            grid, cfgs, freqs, self._cpml_layers,
        )

    # ``cfg.u_lo/u_hi`` are CELL spans (the entries ``_plane_indexer``
    # slices), so this compares exactly the field-array ranges the two
    # ports would write to. Two apertures that meet at a shared wall node
    # abut without sharing a cell and are NOT an overlap; the builders used
    # to hand over node spans, one entry too long, so such a pair was
    # rejected. ``test_abutting_apertures_on_one_face_share_no_cell``
    # (tests/unit/ports/test_waveguide_port_aperture_cell_count.py) pins it.
    def _slices_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return max(a[0], b[0]) < min(a[1], b[1])

    ref_shifts = _waveguide_stages._prepare_waveguide_apertures(
        _slices_overlap, cfgs, entries, resolve_reported_planes, waveguide_plane_positions,
    )

    _pec_axes = None
    # The CPML psi coefficient must read the same permittivity as the Yee half
    # of the timestep: conformal_eps_correction sets aniso_eps to eps_eff =
    # eps/w at wall cells, which is HIGHER than materials.eps_r, and those cells
    # run through the x CPML pads. Measured |S21| at dx 3 / 2 / 1.5 mm was
    # 0.7564 / 0.6974 / nan with mismatched permittivities, and 0.7796 / 0.6974
    # / 0.7274 with matching values. This stability evidence does not establish
    # conformal-PEC accuracy; accuracy measurements made with the mismatch
    # require remeasurement.

    settling_details = []
    if normalize == "flux":
        from rfx.core.yee import init_materials as _init_vacuum_materials
        ref_materials = _homogeneous_reference_materials(grid.shape, reference_medium)
        s_params = extract_waveguide_s_matrix_flux(
            grid,
            materials,
            ref_materials,
            cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_debye=None,
            ref_lorentz=None,
            ref_shifts=ref_shifts,
            aniso_eps=aniso_eps,
            ref_aniso_eps=ref_aniso_eps, ref_conformal=ref_conformal,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            ref_aniso_inv_eps=ref_aniso_inv_eps,
            ref_materials_per_port=ref_materials_per_port,
            pec_edge_masks=_wg_pec_edge_masks,
            ref_pec_edge_masks_per_port=ref_pec_edge_masks_per_port,
            checkpoint_segments=checkpoint_segments,
            return_settling=True, settling_details=settling_details,
            sheet_impedance=_wg_sheet_ctx,
            field_dtype=self._resolve_field_dtype(),
        )
        s_params, settling_db = s_params
    elif normalize:
        from rfx.core.yee import init_materials as _init_vacuum_materials
        ref_materials = _homogeneous_reference_materials(grid.shape, reference_medium)
        s_params = extract_waveguide_s_params_normalized(
            grid,
            materials,
            ref_materials,
            cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_debye=None,
            ref_lorentz=None,
            ref_shifts=ref_shifts,
            aniso_eps=aniso_eps,
            ref_aniso_eps=ref_aniso_eps, ref_conformal=ref_conformal,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            ref_aniso_inv_eps=ref_aniso_inv_eps,
            pec_edge_masks=_wg_pec_edge_masks,
            checkpoint_segments=checkpoint_segments,
            return_settling=True, settling_details=settling_details,
            sheet_impedance=_wg_sheet_ctx,
            field_dtype=self._resolve_field_dtype(),
        )
        s_params, settling_db = s_params
    else:
        s_params, settling_db = extract_waveguide_s_matrix(
            grid,
            materials,
            cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_shifts=ref_shifts,
            aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=_wg_pec_edge_masks,
            checkpoint_segments=checkpoint_segments,
            return_settling=True, settling_details=settling_details,
            sheet_impedance=_wg_sheet_ctx,
            field_dtype=self._resolve_field_dtype(),
        )
    reference_planes = np.array(
        [
            resolve_reported_planes(entry, cfg).reference
            for entry, cfg in zip(entries, cfgs)
        ],
        dtype=float,
    )
    _port_names = tuple(entry.name for entry in entries)
    # the energy ring-down witness reaches the waveguide
    # path — same aggregate truncation warning as the lumped/MSL path.
    # NaN entries (traced AD runs) are skipped by the warner's finite
    # mask; the array itself is always attached for the record.
    settling_db = np.asarray(settling_db, dtype=float)
    _warn_if_ringdown_truncated(
        settling_db, _port_names, num_periods=float(num_periods), witnesses=settling_details,
    )
    # Post-solve discretization witness (post-v1.8 plan item 5): one
    # banner line after the settling line. Report-only.
    _ph_rms, _ph_meta = _waveguide_s21_phase_residual(
        s_params, freqs, reference_planes, cfgs, normalize=normalize,
    )
    _res_sm = realization_result(
        realized_geometry=realization_record,
        s_params=s_params,
        freqs=jnp.asarray(freqs),
        port_names=_port_names,
        port_directions=tuple(entry.direction for entry in entries),
        reference_planes=reference_planes,
        settling_db=settling_db, settling_witness=tuple(settling_details),
        s21_phase_residual_deg_rms=_ph_rms,
        s21_phase_residual_meta=_ph_meta,
    )
    completed = _finalize_sparam_result(
        _res_sm,
        extractor="compute_waveguide_s_matrix",
        strict=strict_passivity,
        check_reciprocity=True,
        passivity_tol=2.0 if normalize is False else 0.10,
    )
    release_inspection(self)
    return completed


def _reference_medium(cfgs, normalize):
    from rfx.model.waveguide_modes import PortMedium
    media = [c.medium for group in cfgs for c in (group if isinstance(group, list) else [group])]
    if normalize and any(m != media[0] for m in media[1:]):
        raise ValueError("Waveguide normalize=True/'flux' requires all ports to share one medium.")
    return media[0] if media else PortMedium()


def _homogeneous_reference_materials(shape, medium):
    from rfx.core.yee import init_materials
    from rfx.model.waveguide_modes import PortMedium
    materials = init_materials(shape)
    if medium == PortMedium():
        return materials
    return materials._replace(eps_r=jnp.full(shape, medium.eps_r, dtype=materials.eps_r.dtype),
                              mu_r=jnp.full(shape, medium.mu_r, dtype=materials.mu_r.dtype))


def _empty_waveguide_reference(self, medium=None):
    """Build a straight guide at the resolved port medium."""
    # Construct from mesh/boundary configuration, never from device geometry.
    # Vacuum retains empty geometry; a fill extends through both CPML pads.
    # Only the mesh, boundary walls and modal ports
    # are shared; material poles, sheets and lumped elements cannot leak in.
    from rfx.boundaries.serialization import constructor_boundary
    reference = type(self)(
        freq_max=self._freq_max, domain=self._domain,
        boundary=constructor_boundary(self), cpml_layers=self._cpml_layers,
        cpml_kappa_max=self._cpml_kappa_max, dx=self._dx,
        dx_profile=self._dx_profile, dy_profile=self._dy_profile,
        dz_profile=self._dz_profile, dt=self._dt_pin,
        dt_min_cell=self._dt_min_cell, precision=self._precision,
        snap=self._snap, nominal_profiles=self._nominal_profiles,
    )

    from rfx.model.waveguide_modes import PortMedium
    if medium is not None and medium != PortMedium():
        from rfx import Box
        reference.add_material("waveguide_reference_fill", eps_r=medium.eps_r, mu_r=medium.mu_r)
        extent = max(self._domain) + 1.0
        reference.add(Box((-extent,) * 3, (extent,) * 3), material="waveguide_reference_fill")
    return reference


def _compute_waveguide_s_matrix_nu(
    self,
    *,
    n_steps: int | None,
    num_periods: float,
    normalize: bool,
    eps_override=None,
    sigma_override=None,
    checkpoint_segments: int | None = None,
) -> WaveguideSMatrixResult:
    """Non-uniform-mesh two-run S-matrix extraction.

    Drives each port in turn, running device + homogeneous-reference
    scans through ``run_nonuniform_path`` so ``dx_profile`` /
    ``dy_profile`` / ``dz_profile`` actually flow into the Yee update. The per-port
    drive uses a private declaration variant with zero ``amplitude`` on
    non-driven entries; the caller's port list remains unchanged. The reference's
    whole domain has the shared port medium. Unequal port media are refused before launching either scan.

    Current scope (matches the uniform path minus a few niceties):
      - ``normalize=True`` or ``normalize='flux'``.
      - Single-mode ports (``n_modes == 1``) only.
      - ``eps_override`` / ``sigma_override`` (the differentiable AD
        design variable) are wired only for ``normalize='flux'``: they
        are threaded into the *device* run so the traced eps flows
        through the jnp-native flux extraction and back to the
        S-matrix gradient. The reference keeps the fixed section medium. They are
        rejected for ``normalize=True`` (its diagonal a_inc_ref
        denominator carries the band-edge fragility).

    Extracts ``a_inc`` / ``b_out`` via the same
    ``extract_waveguide_port_waves`` helper as the uniform path and
    assembles one S at the declared planes: transmission divides by the
    driven port's reference incident wave, and reflection first subtracts
    its reference outgoing wave. One-way Yee dispersion remains, as on
    ``normalize=False``. See ``extract_waveguide_s_params_normalized``.
    """
    # The non-uniform scan has no Dey-Mittra update; the uniform lane of
    # this calculator has one.
    self._refuse_conformal_boundary(
        "non-uniform waveguide S-matrix",
        entry="compute_waveguide_s_matrix()",
        instead="use compute_waveguide_s_matrix() on a uniform mesh "
                "(no dx/dy/dz profile)")
    if self._interface_eps == "dual_average":
        raise ValueError("interface_eps='dual_average' is not supported on the S-parameter NU lane")
    from dataclasses import replace as _dc_replace
    from rfx.runners.nonuniform import run_nonuniform_path
    from rfx.sources.waveguide_port import (
        extract_waveguide_port_waves,
        settling_db_from_port_records,
        waveguide_plane_positions,
    )

    # ``normalize`` may be True (lumped V/I ratio) or "flux" (Poynting
    # power-ratio magnitude + modal phase). normalize=False is not
    # supported on the NU path.
    _flux_mode = (normalize == "flux")
    if not normalize:
        raise NotImplementedError(
            "compute_waveguide_s_matrix(normalize=False) is not yet "
            "supported on the non-uniform mesh path; use normalize=True, "
            "normalize='flux', or drop the dx/dy/dz profiles to stay on "
            "the uniform lane."
        )

    entries = list(self._waveguide_ports)
    if any(entry.n_modes > 1 for entry in entries):
        raise NotImplementedError(
            "Multi-mode waveguide ports are not yet supported on the "
            "non-uniform mesh path."
        )

    n_ports = len(entries)

    # Use the judged Simulation builder while restricting ``cpml_axes`` to
    # axes that are not fully PEC/PMC-bounded. The rasteriser (see
    # ``rfx/geometry/rasterize_grid.py::coords_from_nonuniform_grid``)
    # uses a single ``grid.cpml_layers`` offset for every axis;
    # when a fully PEC-bounded axis is shorter than
    # ``cpml_layers + 1`` cells the offset slice hits IndexError.
    # Dropping that axis from ``cpml_axes`` keeps the physical
    # grid identical (PEC faces already have pad=0) but zeroes the
    # offset so the rasteriser snaps cells to 0 cleanly.
    pec_set = (self._boundary_spec.pec_faces()
               if self._boundary_spec is not None else None) or set()
    pmc_set = (self._boundary_spec.pmc_faces()
               if self._boundary_spec is not None else None) or set()

    def _axis_fully_closed(ax: str) -> bool:
        return {f"{ax}_lo", f"{ax}_hi"}.issubset(pec_set | pmc_set)

    cpml_axes = "".join(
        ax for ax in "xyz"
        if ax not in (self._periodic_axes or "")
        and not _axis_fully_closed(ax)
    )
    # Missing z profiles are synthesized locally, preserving the declared
    # auto mesh and its cached resolution throughout device/reference runs.
    grid = self._build_nonuniform_grid(cpml_axes=cpml_axes)
    if n_steps is None:
        n_steps = steps_for_periods(grid.dt, self._freq_max, num_periods)

    from rfx.preflight.waveguide import resolve_preflight_port_configs
    from rfx.runners.nonuniform import _build_waveguide_port_config_nu
    from rfx.preflight.waveguide import resolve_waveguide_port_freqs
    medium_cfgs = [_build_waveguide_port_config_nu(
        self, e, grid, resolve_waveguide_port_freqs(self, e), n_steps) for e in entries]
    medium_cfgs = resolve_preflight_port_configs(
        self, grid, medium_cfgs, eps_override=eps_override, sigma_override=sigma_override)
    reference = _empty_waveguide_reference(self, _reference_medium(medium_cfgs, normalize))

    # Frequency grid must match across ports.
    port_freqs = entries[0].freqs
    if port_freqs is None:
        port_freqs = jnp.linspace(
            self._freq_max / 10, self._freq_max, entries[0].n_freqs,
        )
    for entry in entries[1:]:
        other = entry.freqs if entry.freqs is not None else jnp.linspace(
            self._freq_max / 10, self._freq_max, entry.n_freqs,
        )
        if other.shape != port_freqs.shape or not np.allclose(
            np.asarray(other), np.asarray(port_freqs)
        ):
            raise ValueError(
                "waveguide S-matrix requires matching frequency grids on all ports"
            )

    # jnp-functional: collect per-drive columns; stack after loop
    s_columns: list[list] = []  # s_columns[drive_idx] = list of (n_freqs,) jnp arrays over recv_idx
    settling_runs: list[float] = []
    settling_details = []
    ref_shifts: tuple[float, ...] | None = None
    reference_planes: list[float] = []
    final_cfgs: list | None = None

    original_entries = list(entries)
    for drive_idx in range(n_ports):
        run_entries = [
            _dc_replace(
                e,
                amplitude=(e.amplitude if idx == drive_idx else 0.0),
            )
            for idx, e in enumerate(original_entries)
        ]

        # Device run: thread the public eps/sigma override (the
        # traced design variable) into the device Yee update so the
        # gradient flows from eps_override -> device fields -> flux
        # -> S-matrix.
        # When eps_override is None the assembled device materials
        # are used unchanged (device fields identical); the np->jnp
        # flux extraction below matches the prior np path to
        # rtol<=1e-5 (float reassociation only).
        # translate the uniform `checkpoint_segments` (K
        # segments) → the NU runner's `checkpoint_every` (chunk size).
        # This calculator chooses a divisor of n_steps nearest to
        # n_steps/K. The NU runner handles non-divisors with a real tail
        # segment; it does not pad the time record. Checkpoint ONLY the
        # *device* run — the vacuum reference is constant in the design
        # variable so it carries no AD tape. The √N tape win is realised
        # under jax.grad (normalize='flux' + eps_override); plain forward
        # is identical.
        from rfx.simulation import _nearest_divisor
        _ckpt_every = None
        if checkpoint_segments is not None and checkpoint_segments > 1:
            _ck = _nearest_divisor(n_steps, max(1, n_steps // int(checkpoint_segments)))
            if 0 < _ck < n_steps:
                _ckpt_every = _ck
        # The warning promises no NTFF accumulation in this calculator.
        device = declaration_variant(self, _waveguide_ports=run_entries, _ntff=None)
        dev_result = run_nonuniform_path(
            device, n_steps=n_steps,
            eps_override=eps_override,
            sigma_override=sigma_override,
            attach_waveguide_flux=_flux_mode, waveguide_reference_only=True,
            checkpoint_every=_ckpt_every,
        )
        # Same active modal port and guide walls, independent vacuum physics.
        ref_variant = declaration_variant(reference, _waveguide_ports=list(run_entries))
        ref_result = run_nonuniform_path(
            ref_variant, n_steps=n_steps, attach_waveguide_flux=_flux_mode, waveguide_reference_only=True,
        )

        dev_wg = dev_result.waveguide_ports or {}
        ref_wg = ref_result.waveguide_ports or {}
        if len(dev_wg) != n_ports or len(ref_wg) != n_ports:
            raise RuntimeError(
                "NU waveguide S-matrix expected one final cfg per "
                "port on both device and reference runs"
            )

        # Energy ring-down witness per drive, worst over BOTH runs of
        # the pair -- the reference run feeds a_inc/P_inc, so its
        # truncation corrupts the same S values (same composition as
        # the uniform normalized/flux lanes). NaN under tracing
        # propagates instead of being max()-eaten.
        _sd_dev, _detail_dev = settling_db_from_port_records(
            [dev_wg[e.name] for e in original_entries], freq_max=self._freq_max, return_detail=True)
        _sd_ref, _detail_ref = settling_db_from_port_records(
            [ref_wg[e.name] for e in original_entries], freq_max=self._freq_max, return_detail=True)
        settling_runs.append(
            float("nan") if (np.isnan(_sd_dev) or np.isnan(_sd_ref))
            else max(_sd_dev, _sd_ref))
        from rfx.sparams._tail_witness import combine_witness_details
        settling_details.append(combine_witness_details(_detail_dev, _detail_ref))

        # Compute ref_shifts from the first drive's configs (same
        # measured planes for every drive / run).
        if ref_shifts is None:
            shifts = []
            for entry in original_entries:
                cfg = dev_wg[entry.name]
                planes = waveguide_plane_positions(cfg)
                desired = resolve_reported_planes(entry, cfg, grid=grid).reference
                reference_planes.append(desired)
                shifts.append(desired - planes["reference"])
            ref_shifts = tuple(shifts)
            # The same final configs, kept for the post-solve S21
            # phase residual below (its beta ingredients are the
            # extractor's own f_cutoff / dt / dx, not re-derived).
            final_cfgs = [dev_wg[e.name] for e in original_entries]

        drive_name = original_entries[drive_idx].name
        a_inc_ref, _ = extract_waveguide_port_waves(
            ref_wg[drive_name], ref_shift=ref_shifts[drive_idx],
        )
        safe_a_inc = jnp.where(
            jnp.abs(a_inc_ref) > 1e-30,
            a_inc_ref,
            jnp.ones_like(a_inc_ref),
        )

        if _flux_mode:
            # Power-flux magnitude + modal phase (mirrors the
            # uniform extract_waveguide_s_matrix_flux). Immune to
            # the band-edge a_inc_ref denominator collapse that
            # makes the normalize=True diagonal blow up:
            # P_inc = |F_ref[drive]| is large and well-conditioned
            # across the whole band, not source-spectrum-weighted.
            F_ref = ref_result.waveguide_port_flux
            F_dev = dev_result.waveguide_port_flux
            if F_ref is None or F_dev is None:
                raise RuntimeError(
                    "normalize='flux' on the NU path requires "
                    "per-port flux spectra; run_nonuniform_path did "
                    "not return waveguide_port_flux."
                )
            # jnp-native:
            # no np.asarray() concretization — keeps the whole flux
            # extraction on the AD tape so an eps_override-traced
            # device run yields finite gradients through the
            # S-matrix. Uses the DOUBLE-WHERE trick at sqrt(0) /
            # angle(0) (guard the INPUT, not just the output): a
            # single jnp.where still leaks NaN grad through the dead
            # branch. Forward values are identical
            # to the prior np version for P_inc / P > 0.
            P_inc = jnp.abs(F_ref[drive_idx])
            safe_P_inc = jnp.where(
                P_inc > 1e-60, P_inc, jnp.ones_like(P_inc)
            )
            recv_col = []
            for recv_idx in range(n_ports):
                recv_name = original_entries[recv_idx].name
                _, b_recv_dev = extract_waveguide_port_waves(
                    dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                )
                # AD-safe angle (double-where): angle() has an
                # undefined gradient at 0; angle(1)=0 matches
                # np.angle(0)=0 so the primal is unchanged.
                ratio = b_recv_dev / safe_a_inc
                ratio_ok = jnp.abs(ratio) > 0.0
                phase = jnp.angle(
                    jnp.where(ratio_ok, ratio, jnp.ones_like(ratio))
                )
                if recv_idx == drive_idx:
                    P_num = jnp.abs(F_ref[drive_idx] - F_dev[drive_idx])
                else:
                    P_num = jnp.abs(F_dev[recv_idx])
                # AD-safe sqrt (double-where): a perfect match/null
                # makes the power ratio exactly 0, where
                # d(sqrt)/dx = inf would leak 0*inf=nan through the
                # backward pass; primal stays exactly sqrt(x) for
                # x>0 and exactly 0 at x=0.
                p_ratio = _flux_power_ratio(P_num, safe_P_inc)
                p_ok = p_ratio > 0.0
                mag = jnp.where(
                    p_ok,
                    jnp.sqrt(
                        jnp.where(p_ok, p_ratio, jnp.ones_like(p_ratio))
                    ),
                    0.0,
                )
                recv_col.append(mag * exp_j(phase, xp=jnp))
            s_columns.append(recv_col)
            continue

        b_out_ref, b_out_dev = [], []
        for recv_idx in range(n_ports):
            recv_name = original_entries[recv_idx].name
            _, b_ref = extract_waveguide_port_waves(
                ref_wg[recv_name], ref_shift=ref_shifts[recv_idx],
            )
            _, b_dev = extract_waveguide_port_waves(
                dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
            )
            b_out_ref.append(b_ref)
            b_out_dev.append(b_dev)
        # The lower transmission floor preserves the graded lane's
        # small-signal guard; the diagonal keeps its existing floor.
        s_columns.append(assemble_normalized_waveguide_column(
            b_out_dev, b_out_ref, a_inc_ref, drive_idx,
            diagonal_floor=1e-30, transmission_floor=1e-60, xp=jnp,
        ))


    # The ring-down witness covers the NU lane -- same aggregate truncation warning and settling_db
    # field as the uniform single-mode lanes. NaN entries (traced
    # AD runs) are skipped by the warner's finite mask; the array itself
    # is always attached for the record.
    settling_db = np.asarray(settling_runs, dtype=float)
    _warn_if_ringdown_truncated(
        settling_db, tuple(e.name for e in original_entries), witnesses=settling_details,
        num_periods=float(num_periods),
    )
    _s_params_nu = jnp.stack(
        [jnp.stack(col) for col in s_columns], axis=1)
    _reference_planes_nu = np.asarray(reference_planes, dtype=float)
    # Post-solve discretization witness (post-v1.8 plan item 5), same
    # banner and same field as the uniform lane. Report-only.
    _ph_rms_nu, _ph_meta_nu = _waveguide_s21_phase_residual(
        _s_params_nu, port_freqs, _reference_planes_nu, final_cfgs or [],
        normalize=normalize,
    )
    return WaveguideSMatrixResult(
        s_params=_s_params_nu,
        freqs=jnp.asarray(port_freqs),
        port_names=tuple(e.name for e in original_entries),
        port_directions=tuple(e.direction for e in original_entries),
        reference_planes=_reference_planes_nu,
        settling_db=settling_db, settling_witness=tuple(settling_details),
        s21_phase_residual_deg_rms=_ph_rms_nu,
        s21_phase_residual_meta=_ph_meta_nu,
    )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Both functions above were ``def``s in the ``_SparamMixin`` class body, so
# their ``__qualname__`` read ``_SparamMixin.<name>``; a module-level ``def``
# gets the bare ``<name>`` instead. ``rfx/api/__init__.py`` rewrites exactly
# ``_SparamMixin.<name>`` -> ``Simulation.<name>`` at class-composition time so
# that a bad keyword argument reports ``Simulation.compute_waveguide_s_matrix()
# got an unexpected keyword argument``, and it SKIPS any function whose
# qualname does not match that pattern. Leaving the bare name here would
# therefore change those TypeError messages — a user-visible behaviour change
# in a pure code-motion step. ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` pins it.
# ---------------------------------------------------------------------------
compute_waveguide_s_matrix.__qualname__ = "_SparamMixin.compute_waveguide_s_matrix"
# ``flux_mesh_entry`` wraps the calculator, and Python names the INNER function
# in a bad-keyword TypeError; give it the name the rewrite gives the wrapper.
compute_waveguide_s_matrix.__wrapped__.__qualname__ = "Simulation.compute_waveguide_s_matrix"
_compute_waveguide_s_matrix_nu.__qualname__ = (
    "_SparamMixin._compute_waveguide_s_matrix_nu"
)

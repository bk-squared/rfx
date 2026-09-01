"""Batched material-parameter sweep using ``jax.vmap``.

For material-only sweeps (``eps_r``, ``sigma``, ``mu_r``) the grid geometry
is identical across all parameter values, so ``jax.vmap`` can batch the
full FDTD time loop and execute all simulations in a single GPU kernel.

This is dramatically faster than sequential sweeps for moderate batch sizes
(typically 5--50 values) because:

1. Grid construction happens **once**.
2. Source waveforms and probe specs are shared.
3. The ``jax.lax.scan`` body is vmapped over a leading batch axis in the
   material arrays, so all simulations run in parallel.

Limitations
-----------
- Only material parameters (``eps_r``, ``sigma``, ``mu_r``) can be swept.
  For geometry sweeps (different shapes/sizes) use ``parametric_sweep()``.
- ``until_decay`` is not supported (requires Python-level control flow).
- CPML and DFT plane probes (``add_dft_plane_probe``, #578) ARE supported on
  the fast path because they use the same grid topology as PEC/periodic
  walls. This bullet used to claim the WHOLE feature list below it was
  "fully supported" — that was a drifted inversion of the true limitation
  (the function docstring below and the code guards always disagreed with
  it). Lumped/MSL/coaxial/floquet ports, TFSF, dispersion, waveguide ports,
  flux monitors, NTFF, and RLC elements are NOT wired into the
  fast-path scan bodies and trigger the sequential fallback instead (still
  correct, just slower — and, as of #578, that fallback also carries DFT
  plane accumulators, so no feature silently vanishes on either path).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import MaterialArrays, init_state, update_e, update_h, EPS_0
from rfx.geometry.rasterize_grid import extend_cpml_pad_materials
from rfx.materials.thin_conductor import apply_thin_conductor
from rfx.probes.probes import DFTPlaneProbe, init_dft_plane_probe
from rfx.simulation import (
    SourceSpec,
    ProbeSpec,
    make_source,
    make_probe,
)
from rfx.boundaries.pec import apply_pec


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VmapSweepResult:
    """Result from a vmapped material sweep.

    Attributes
    ----------
    time_series : ndarray, shape (n_batch, n_steps, n_probes)
        Probe recordings for every sweep value.
    param_name : str
        Name of the swept parameter.
    param_values : ndarray, shape (n_batch,)
        The parameter values that were swept.
    final_fields : dict or None
        Never populated (``return_fields=True`` raises ``ValueError``
        instead of silently returning ``None`` here — see
        ``vmap_material_sweep``). Reserved for a future final-field-state
        implementation; use ``sim.run()``/``sim.forward()`` or
        ``parametric_sweep()`` for final fields today.
    dft_planes : dict[str, ndarray] or None
        DFT plane accumulators, keyed by the ``add_dft_plane_probe`` name,
        each shaped ``(n_batch, n_freqs, n1, n2)`` complex — a leading
        batch axis over the same accumulator ``Result.dft_planes[name]``
        exposes for a single run (#578). Populated on BOTH the vmap fast
        path and the sequential fallback (the fallback stacks each
        swept-value ``Result.dft_planes[name].accumulator``), so the field
        is uniform regardless of which path a given ``Simulation`` takes.
        ``None`` if no ``add_dft_plane_probe`` was registered. Appended
        AFTER ``final_fields`` (not inserted before it) to preserve the
        positional-construction meaning of this publicly exported
        dataclass for any existing caller using positional args.
    """
    time_series: np.ndarray
    param_name: str
    param_values: np.ndarray
    final_fields: dict | None = None
    dft_planes: dict | None = None

    def peak_field(self) -> np.ndarray:
        """Return peak |probe value| per batch element."""
        from rfx.sweep import peak_abs_field

        return peak_abs_field(self.time_series, axis=(1, 2))


# ---------------------------------------------------------------------------
# Parameter application helpers
# ---------------------------------------------------------------------------

_VALID_PARAMS = {"eps_r", "sigma", "mu_r"}


def _parse_param_name(param_name: str) -> tuple[str | None, str]:
    """Parse ``"eps_r"`` or ``"substrate.eps_r"`` into (material_name, field).

    Returns
    -------
    (material_name_or_None, field_name)
    """
    if "." in param_name:
        mat_name, field = param_name.rsplit(".", 1)
    else:
        mat_name, field = None, param_name

    if field not in _VALID_PARAMS:
        raise ValueError(
            f"param_name field must be one of {_VALID_PARAMS}, "
            f"got {field!r} (from {param_name!r})"
        )
    return mat_name, field


def _extend_batched_cpml_pad(
    eps_r: jnp.ndarray, sigma: jnp.ndarray, mu_r: jnp.ndarray, grid,
    dispersion_pole_mask: jnp.ndarray | None = None,
):
    """Extend BATCHED material arrays, each shape ``(n_batch, Nx, Ny, Nz)``,
    into the CPML padding, by running the package's single pad-extension
    rule — ``rfx.geometry.rasterize_grid.extend_cpml_pad_materials`` — once
    per batch element under ``jax.vmap``.

    ``jnp.where(mask, param_values, base_field)`` (the caller) only ever
    touches the physical-domain-interior cells that ``mask`` covers
    (``Shape.mask`` returns False everywhere in the padding, since padding
    cells map to physical coordinates outside the declared domain). The
    padding faces of ``base_materials`` were replicated from the BASE
    simulation's edge slice, so every batch element would otherwise
    inherit the base material's absorber there regardless of its own
    swept value (issue #637). Re-running the shared rule on the
    already-batch-correct interior makes each batch element's padding
    byte-identical to what ``Simulation.run()`` builds for that same
    swept value.

    **Why ``jax.vmap`` and not a hand-written batched copy (issue #643).**
    #637's original version of this function reproduced
    ``_assemble_materials``' rule *as it stood at the time* — a straight
    per-face edge-slice copy — in a second, hand-maintained
    implementation. #627 (PR #638) then changed that rule underneath it,
    adding a per-transverse-cell hi-face fallback (if the naive
    interior-edge column is vacuum but the column one further in is not,
    replicate from that inner column instead, recovering the node a
    ``Box`` flush with the domain's hi face loses to half-open ``[lo,
    hi)`` rasterization). The two copies then disagreed for exactly that
    geometry: ``run()``'s x-hi pad read the material, the batched path
    read vacuum. That is the defect this repo's own feature-discovery
    rule names — two hand-maintained copies of one rule is the defect,
    not which copy is right — so the copy is gone rather than resynced.

    ``vmap`` is what makes reuse possible at all. The shared helper slices
    axes 0/1/2 and evaluates its vacuum predicate on whole transverse
    planes; the batched arrays carry a leading sweep axis, and the
    predicate's answer differs per element (one swept value can leave a
    column vacuum where another does not). Mapping over that leading axis
    hides it from the helper entirely, so the helper sees exactly the
    ``(Nx, Ny, Nz)`` arrays it was written for and its fallback test is
    evaluated against each element's OWN materials — no axis-aware
    rewrite, no third implementation.

    All three arrays are extended together in ONE call, not one at a time:
    the shared helper takes a single ``use_inner`` decision from the joint
    vacuum predicate (``eps_r == 1 & sigma == 0 & mu_r == 1``) and applies
    it to all three, so a swept value that turns a column vacuum for one
    element changes that element's ``sigma``/``mu_r`` pad too. A per-field
    call cannot express that coupling.

    **The caller must hand this UN-extended arrays (issue #655).** The
    correctness argument is that the result depends only on the interior
    values, which the caller has already made batch-correct; pad cells are
    all overwritten by one of the three passes, so their incoming contents
    are irrelevant. #637/#643 relied on that to feed already-extended
    ``base_materials`` in. #655 then made the shared rule repair the
    dropped hi-face boundary NODE too — an INTERIOR cell, and one
    ``Shape.mask`` does not cover — so an already-extended input carries
    the BASE material at that node and the "interior is batch-correct"
    premise fails. The caller therefore now assembles with
    ``include_cpml_pad_extension=False``. Verified by the byte-identity
    matrix in
    ``tests/test_vmap_sweep_dft_planes.py::TestVmapBatchedPadByteIdentity``.

    A no-op on any face with zero pad depth (non-CPML boundary, or a
    reflector/periodic face with ``pad=0`` on that side), so this is safe
    to call unconditionally; with every face at zero it returns its inputs
    untouched rather than materialising broadcast views through ``vmap``.

    Returns
    -------
    (eps_r, sigma, mu_r) — same shapes as the inputs.
    """
    plx, phx = grid.pad_x_lo, grid.pad_x_hi
    ply, phy = grid.pad_y_lo, grid.pad_y_hi
    plz, phz = grid.pad_z_lo, grid.pad_z_hi
    if max(plx, phx, ply, phy, plz, phz) <= 0:
        return eps_r, sigma, mu_r

    def _one(e, s, m):
        # ``dispersion_pole_mask`` is batch-invariant (pole masks are
        # geometry-derived and the sweep only substitutes eps_r/sigma/mu_r
        # values), so it rides in as a closed-over constant rather than a
        # mapped operand — the helper sees the same (Nx, Ny, Nz) mask
        # ``run()`` hands it (#808 gate).
        return extend_cpml_pad_materials(
            e, s, m, plx, phx, ply, phy, plz, phz,
            dispersion_pole_mask=dispersion_pole_mask)

    return jax.vmap(_one)(eps_r, sigma, mu_r)


def _apply_batched_thin_conductors(
    sim, grid, eps_r: jnp.ndarray, sigma: jnp.ndarray, mu_r: jnp.ndarray,
):
    """Apply ``sim``'s thin conductors to BATCHED material arrays, each
    shape ``(n_batch, Nx, Ny, Nz)``, by running the package's single
    conductor rule — ``rfx.materials.thin_conductor.apply_thin_conductor``
    — once per batch element under ``jax.vmap``, in the same order
    ``Simulation._assemble_materials`` runs it.

    This is the second half of the #642 fix. ``_assemble_materials``
    extends the CPML pad and only THEN applies conductors, so ``run()``'s
    padding never contains one. The batched path re-extends the pad per
    swept value; to reproduce ``run()`` it must therefore re-extend the
    *pre-conductor* arrays and re-apply the conductors afterwards, which
    is exactly this call plus the ``include_thin_conductors=False``
    assembly in ``_build_batched_materials``. Doing it the other way
    round — extending the finished arrays, as the code did before #642 —
    replicates the conductor's own ``eps_r``/``sigma`` outward into a pad
    that ``run()`` fills with the background material instead.

    PEC thin conductors are a no-op here by construction: they route to
    ``pec_mask``, which the caller takes from the full (conductor-
    inclusive) assembly and which does not vary across the sweep. The
    returned mask is therefore discarded, not ignored by oversight.

    A no-op when the simulation declares no thin conductors — the caller
    guards on that, so this never materialises a batched copy for the
    overwhelmingly common case.

    Returns
    -------
    (eps_r, sigma, mu_r) — same shapes as the inputs.
    """
    conductors = tuple(sim._thin_conductors)
    if not conductors:
        return eps_r, sigma, mu_r

    def _one(e, s, m):
        mats = MaterialArrays(eps_r=e, sigma=s, mu_r=m)
        for tc in conductors:
            # #677: an f0 (surface-impedance) conductor is a no-op on the
            # material arrays by design — the sheet is a per-step operator
            # ctx now, and sims carrying one never reach the fast path
            # (has_f0_sheets ineligibility in _build_full_scan_fn).
            mats, _ = apply_thin_conductor(grid, tc, mats, pec_mask=None)
        return mats.eps_r, mats.sigma, mats.mu_r

    return jax.vmap(_one)(eps_r, sigma, mu_r)


def _build_batched_materials(
    sim,
    grid,
    base_materials: MaterialArrays,
    param_name: str,
    param_values: jnp.ndarray,
) -> MaterialArrays:
    """Create batched material arrays with shape (n_batch, Nx, Ny, Nz).

    For a global sweep (e.g. ``"eps_r"``), the parameter is applied to
    **all non-vacuum cells** that have the swept property differing from 1.0
    (for eps_r/mu_r) or 0.0 (for sigma). This already reaches the CPML
    padding without special-casing: ``base_materials`` replicates each
    material's actual (non-vacuum) value into its padding
    (``_assemble_materials``), so the ``!= 1.0`` / ``!= 0.0`` test is True
    there too and the global sweep overwrites those pad cells along with
    the interior ones (issue #637 scope note — verified by
    ``test_global_sweep_pad_cells_still_correct`` in
    ``tests/test_vmap_sweep_dft_planes.py``).

    For a material-specific sweep (e.g. ``"substrate.eps_r"``), the parameter
    is applied only to cells occupied by that named material — geometry
    cells only, via ``Shape.mask``. Unlike the global case, ``mask`` is
    False everywhere in the CPML padding (padding cells sit outside the
    physical domain ``Shape.mask`` tests against), so the padding is
    handled separately below via ``_extend_batched_cpml_pad`` (issue #637;
    issue #643 for why that now routes through the package's single shared
    pad rule rather than a second copy of it): without it, every swept
    batch element would run against a pad matched to the BASE material
    instead of its own.

    **Thin conductors and pipeline ORDER (issue #642).**
    ``_assemble_materials`` extends the pad and only THEN applies thin
    conductors, so ``run()``'s padding carries the background material,
    never the conductor. ``base_materials`` is the finished, post-conductor
    array; re-extending it would copy the conductor's own ``eps_r`` /
    ``sigma`` outward into the pad. #643 could not close that by making
    the extension rule more faithful, because the rule was never the
    problem — the batched path was handed the wrong INPUT. So when the
    simulation declares thin conductors, the material-named branch below
    re-derives the PRE-conductor arrays from the same assembler
    (``include_thin_conductors=False``), builds the sweep and the pad from
    those, and re-applies the same shared ``apply_thin_conductor``
    afterwards — reproducing ``run()``'s order rather than approximating
    its output. That also fixes a second consequence of the old order: a
    conductor overlapping the swept material's cells is overwritten by the
    swept value under the old code, whereas ``run()`` lets the conductor
    win (it is applied last).
    """
    mat_name, field = _parse_param_name(param_name)
    n_batch = len(param_values)

    eps_r = base_materials.eps_r   # (Nx, Ny, Nz)
    sigma = base_materials.sigma
    mu_r = base_materials.mu_r

    if mat_name is not None:
        # #642: build from the state run() extends the pad from, i.e.
        # before its own thin-conductor pass.
        # #655: and before the pad extension itself. The shared rule now
        # also repairs the hi-face boundary NODE the rasterizer dropped,
        # and that node is an INTERIOR cell ``mask`` does not cover, so
        # substituting into already-extended arrays leaves the base
        # material's value sitting there and then replicates it outward —
        # run() would have the swept value. Taking the un-extended arrays
        # restores the invariant ``_extend_batched_cpml_pad`` documents
        # (its result depends only on interior values, which must all be
        # batch-correct) instead of special-casing the node here.
        pre_materials, _pre_debye, _pre_lorentz, *_ = sim._assemble_materials(
            grid, include_thin_conductors=False,
            include_cpml_pad_extension=False)
        eps_r = pre_materials.eps_r
        sigma = pre_materials.sigma
        mu_r = pre_materials.mu_r

        # #808: run() gates the hi-face fallback on the combined pole mask,
        # so the batched re-extension below must hand the shared rule the
        # same mask or a dispersive sweep's pad would diverge from run().
        _pole_mask_any = None
        for _spec in (_pre_debye, _pre_lorentz):
            if _spec is not None:
                for _pmask in _spec[1]:
                    _pole_mask_any = (_pmask if _pole_mask_any is None
                                      else (_pole_mask_any | _pmask))

        # Build a mask for the specific material
        sim._resolve_material(mat_name)
        mask = jnp.zeros(grid.shape, dtype=jnp.bool_)
        for entry in sim._geometry:
            if entry.material_name == mat_name:
                mask = mask | entry.shape.mask(grid)

        # For each batch: where mask, use param_values[b]; else keep base.
        # The two unswept fields still have to be broadcast to the batch
        # shape, because the pad extension below takes ONE joint decision
        # across all three (see _extend_batched_cpml_pad) and a swept value
        # can change that decision per element.
        batch_eps = jnp.broadcast_to(eps_r[None], (n_batch,) + eps_r.shape)
        batch_sigma = jnp.broadcast_to(sigma[None], (n_batch,) + sigma.shape)
        batch_mu = jnp.broadcast_to(mu_r[None], (n_batch,) + mu_r.shape)
        swept = param_values[:, None, None, None]   # (n_batch, 1, 1, 1)
        if field == "eps_r":
            batch_eps = jnp.where(mask[None], swept, eps_r[None])
        elif field == "sigma":
            batch_sigma = jnp.where(mask[None], swept, sigma[None])
        else:  # mu_r
            batch_mu = jnp.where(mask[None], swept, mu_r[None])

        # #637: the mask above only covers the physical-domain interior --
        # extend the (now per-batch-correct) interior into the CPML pad the
        # same way the base simulation would for each swept value, instead
        # of leaving the base material's pad fill. #643: via the package's
        # single shared rule, vmapped over the sweep axis, not a second
        # hand-maintained copy of it.
        batch_eps, batch_sigma, batch_mu = _extend_batched_cpml_pad(
            batch_eps, batch_sigma, batch_mu, grid,
            dispersion_pole_mask=_pole_mask_any,
        )

        # #642: and only NOW the conductors, which is where
        # _assemble_materials applies them relative to the extension above.
        batch_eps, batch_sigma, batch_mu = _apply_batched_thin_conductors(
            sim, grid, batch_eps, batch_sigma, batch_mu,
        )
    else:
        # Global sweep: apply to all non-background cells
        if field == "eps_r":
            # Identify cells that have non-vacuum eps_r
            non_vac = eps_r != 1.0
            batch_eps = jnp.where(
                non_vac[None],
                param_values[:, None, None, None],
                eps_r[None],
            )
            batch_sigma = jnp.broadcast_to(sigma[None], (n_batch,) + sigma.shape)
            batch_mu = jnp.broadcast_to(mu_r[None], (n_batch,) + mu_r.shape)
        elif field == "sigma":
            non_zero = sigma != 0.0
            batch_eps = jnp.broadcast_to(eps_r[None], (n_batch,) + eps_r.shape)
            batch_sigma = jnp.where(
                non_zero[None],
                param_values[:, None, None, None],
                sigma[None],
            )
            batch_mu = jnp.broadcast_to(mu_r[None], (n_batch,) + mu_r.shape)
        else:  # mu_r
            non_vac = mu_r != 1.0
            batch_eps = jnp.broadcast_to(eps_r[None], (n_batch,) + eps_r.shape)
            batch_sigma = jnp.broadcast_to(sigma[None], (n_batch,) + sigma.shape)
            batch_mu = jnp.where(
                non_vac[None],
                param_values[:, None, None, None],
                mu_r[None],
            )

    return MaterialArrays(
        eps_r=batch_eps,
        sigma=batch_sigma,
        mu_r=batch_mu,
    )


# ---------------------------------------------------------------------------
# Core: vmapped FDTD scan
# ---------------------------------------------------------------------------
#
# W6.6: ``run()`` material sweeps need two FDTD loops — a PEC/periodic-walls
# loop and a CPML loop — that were previously two ~50-line near-identical scan
# bodies.  ``_build_vmap_scan_fn`` is the single parameterized builder; the two
# callers differ ONLY in:
#   * ``use_cpml`` — carry the per-step CPML psi-state and apply the CPML
#     H/E sub-steps (PEC/periodic carries just the field state);
#   * the J-source Cb-scaling pre-pass — CPML J-sources need their waveforms
#     scaled by the material-dependent Cb at the source cell, computed inside
#     ``run_one`` from the actual (batch-element) material arrays.
# Everything else — Yee H/E update, PEC walls, pec-mask, non-J soft sources,
# probe sampling, dtypes, sub-step ordering — is identical and shared.  This is
# pure code motion; numerics are unchanged (see W6.6 bit-identity gate).


def _build_vmap_scan_fn(
    grid,
    n_steps: int,
    *,
    use_cpml: bool = False,
    sources: list[SourceSpec] | None = None,
    j_source_raw_info: list[tuple] | None = None,
    probes: list[ProbeSpec] | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    cpml_axes: str = "xyz",
    pec_axes: str = "xyz",
    pec_mask=None,
    dft_probes: list[DFTPlaneProbe] | None = None,
):
    """Build a pure function ``f(materials) -> (time_series, dft_accs)``
    suitable for vmap.

    Constructs a minimal FDTD loop (H update -> [CPML H] -> E update ->
    [CPML E] -> PEC -> sources -> DFT-plane accumulation -> probes) without
    dispersion, TFSF, or waveguide ports.  For the common material-sweep use
    case (PEC cavity, CPML absorber, DFT-plane probe, or simple probe
    measurement) this covers the needed physics.

    Parameters
    ----------
    use_cpml : bool
        If True, the scan carry additionally holds the CPML psi-state and the
        CPML H/E sub-steps are applied; J-source waveforms are Cb-normalized
        per batch element from the actual material arrays.
    j_source_raw_info : list of (i, j, k, component, raw_waveform)
        Populated for legacy CPML J-sources AND for ``amplitude_kind='current'``
        soft sources on any boundary.  Raw (un-Cb-normalized) J-source
        waveforms + cell indices so Cb can be computed dynamically inside the
        vmapped function from the actual material arrays.
    dft_probes : list of DFTPlaneProbe or None
        Zero-initialized DFT plane probes (from ``init_dft_plane_probe``,
        the SAME helper ``run()`` uses — rfx/runners/uniform.py:489-497).
        Their ``.accumulator`` seeds the scan-carry accumulator for that
        plane; ``.component``/``.axis``/``.index``/``.freqs`` drive the
        per-step kernel, inlined here to match ``Simulation.run()``'s rect
        kernel exactly (rfx/simulation.py:1512-1526): ``t = st.step * dt``
        (the STATE's step counter, NOT the scan's ``xs`` step index — using
        the latter is an off-step, per-bin-phase-error class bug, #404).
        The returned ``run_one`` always returns a 2-tuple
        ``(time_series, dft_accs)`` where ``dft_accs`` is a tuple of final
        accumulators in the SAME order as ``dft_probes`` (empty tuple if
        no planes were registered).
    """
    dt = grid.dt
    dx = grid.dx
    sources = sources or []
    j_source_raw_info = j_source_raw_info or []
    probes = probes or []
    dft_probes = dft_probes or []

    use_pec_mask = pec_mask is not None
    use_dft = len(dft_probes) > 0
    dft_meta = tuple(
        (p.component, p.axis, p.index, p.freqs) for p in dft_probes
    )
    dft_acc_init = tuple(p.accumulator for p in dft_probes)

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h

    # Precompute source waveform matrix (these don't depend on materials)
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]

    # J-source metadata: cell indices + component + raw (un-Cb-normalized)
    # waveforms.  Populated for legacy CPML J-sources and for
    # amplitude_kind='current' soft sources on any boundary.
    j_src_meta = [(i, j, k, comp) for i, j, k, comp, _ in j_source_raw_info]
    if j_source_raw_info:
        j_src_raw_waveforms = jnp.stack(
            [raw_wf for _, _, _, _, raw_wf in j_source_raw_info], axis=-1
        )  # (n_steps, n_j_sources)
    else:
        j_src_raw_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # Initialize CPML once (shared across batch)
    if use_cpml:
        cpml_params, cpml_state_init = init_cpml(grid)

    def run_one(materials: MaterialArrays) -> tuple[jnp.ndarray, tuple]:
        """Run a single FDTD simulation with the given materials.

        Returns ``(time_series, dft_accs)``: time_series shaped
        ``(n_steps, n_probes)``; dft_accs a tuple of final ``(n_freqs, n1,
        n2)`` complex accumulators, one per registered DFT plane (empty
        tuple if none).
        """
        fdtd = init_state(grid.shape)

        # Compute Cb-normalized J-source waveforms from the actual materials.
        # Cb = (dt / eps) / (1 + loss) where eps = eps_r * EPS_0,
        # loss = sigma * dt / (2 * eps)
        if j_source_raw_info:
            j_cb_scales = []
            for si, sj, sk, sc in j_src_meta:
                eps = materials.eps_r[si, sj, sk] * EPS_0
                sigma_val = materials.sigma[si, sj, sk]
                loss = sigma_val * dt / (2.0 * eps)
                cb = (dt / eps) / (1.0 + loss)
                j_cb_scales.append(cb)
            j_cb_arr = jnp.stack(j_cb_scales)  # (n_j_sources,)
            # Scale raw waveforms: (n_steps, n_j_sources) * (n_j_sources,)
            j_src_waveforms = j_src_raw_waveforms * j_cb_arr[None, :]
        else:
            j_src_waveforms = j_src_raw_waveforms

        def step_fn(carry, xs):
            _step_idx, src_vals, j_src_vals = xs
            if use_cpml:
                st, cpml_st, dft_accs = carry
            else:
                st, dft_accs = carry

            # H update
            st = update_h(st, materials, dt, dx, periodic=periodic)
            if use_cpml:
                # Material-aware CPML: pass the per-element materials so the
                # absorber is impedance-matched to the local dielectric. Under
                # vmap, ``materials`` is the per-batch-element MaterialArrays
                # (the run_one arg), so vmap batches this transparently — no
                # batch-axis slicing needed. WITHOUT this, a dielectric that
                # fills the CPML region gets a free-space (eps_r x too strong)
                # absorber and the scan diverges to NaN (issue #205, same
                # mechanism as #203/#204 uniform and #208 non-uniform).
                st, cpml_st = apply_cpml_h(
                    st, cpml_params, cpml_st, grid, cpml_axes,
                    materials=materials)

            # E update
            st = update_e(st, materials, dt, dx, periodic=periodic)
            if use_cpml:
                st, cpml_st = apply_cpml_e(
                    st, cpml_params, cpml_st, grid, cpml_axes,
                    materials=materials)

            # PEC boundaries
            if pec_axes:
                st = apply_pec(st, axes=pec_axes)

            if use_pec_mask:
                from rfx.boundaries.pec import apply_pec_mask
                # #689: pass the run's periodic flags (see pec.py).
                st = apply_pec_mask(st, pec_mask, periodic)

            # Non-J soft sources (raw field add, no Cb dependence)
            for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
                field = getattr(st, sc)
                field = field.at[si, sj, sk].add(src_vals[idx_s])
                st = st._replace(**{sc: field})

            # J-sources (Cb-normalized, material-dependent): legacy cpml
            # route, plus amplitude_kind='current' on any boundary (#571)
            for idx_j, (si, sj, sk, sc) in enumerate(j_src_meta):
                field = getattr(st, sc)
                field = field.at[si, sj, sk].add(j_src_vals[idx_j])
                st = st._replace(**{sc: field})

            # DFT-plane accumulation (rect window always — matches
            # Simulation.run()'s inline kernel, rfx/simulation.py:1512-1526).
            # ``t`` is derived from the STATE's own step counter, not the
            # scan xs step index, so it matches run() bit-for-bit under vmap.
            if use_dft:
                t_plane = st.step * dt
                new_dft_accs = []
                for acc, (component, axis, index, freqs) in zip(dft_accs, dft_meta):
                    field = getattr(st, component)
                    if axis == 0:
                        plane = field[index, :, :]
                    elif axis == 1:
                        plane = field[:, index, :]
                    else:
                        plane = field[:, :, index]
                    phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs * t_plane)
                    new_dft_accs.append(
                        acc + plane[None, :, :] * phase[:, None, None] * dt)
                dft_accs = tuple(new_dft_accs)

            # Probe samples
            samples = [getattr(st, pc)[pi, pj, pk]
                       for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

            if use_cpml:
                return (st, cpml_st, dft_accs), probe_out
            return (st, dft_accs), probe_out

        xs = (
            jnp.arange(n_steps, dtype=jnp.int32),
            src_waveforms,
            j_src_waveforms,
        )
        if use_cpml:
            init_carry = (fdtd, cpml_state_init, dft_acc_init)
        else:
            init_carry = (fdtd, dft_acc_init)
        final_carry, time_series = jax.lax.scan(step_fn, init_carry, xs)
        final_dft_accs = final_carry[-1]
        return time_series, final_dft_accs

    return run_one


def _build_full_scan_fn(
    sim,
    grid,
    base_materials: MaterialArrays,
    n_steps: int,
    *,
    debye_spec=None,
    lorentz_spec=None,
    pec_mask=None,
):
    """Build ``(f, dft_names)`` where ``f(materials) -> (time_series,
    dft_accs)`` uses the full simulation runner (including CPML,
    dispersion, etc.), or ``(None, None)`` if this simulation must take the
    sequential fallback instead.

    This wraps ``simulation.run()`` into a vmappable form by making
    ``materials`` an explicit argument rather than a closure capture.
    """
    boundary = sim._boundary

    # Detect if simulation uses features incompatible with simple vmap
    has_ports = any(pe.impedance != 0.0 for pe in sim._ports)
    has_tfsf = sim._tfsf is not None
    has_waveguide = len(sim._waveguide_ports) > 0
    has_dispersion = debye_spec is not None or lorentz_spec is not None
    # MSL/floquet ports are genuine run()-consumed silent drops if allowed
    # onto the fast path (neither scan body launches/records them); coaxial
    # ports are NOT consumed by plain run() at all today, so this guard is
    # added for honesty/uniformity with the other port families rather than
    # because the fast path would corrupt anything (#578 design v2 PR B.3).
    has_msl_ports = len(getattr(sim, "_msl_ports", []) or []) > 0
    has_floquet_ports = len(getattr(sim, "_floquet_ports", []) or []) > 0
    has_coaxial_ports = len(getattr(sim, "_coaxial_ports", []) or []) > 0

    # Allowlist guard: the vmap scan bodies implement ONLY pec/periodic
    # walls, CPML, and (as of #578) DFT plane probes. Anything below must
    # take the sequential fallback (return None) or it would be SILENTLY
    # DROPPED from the swept runs:
    #   - boundary='upml': neither scan body applies a UPML step, so the
    #     fast path would run with no absorber at all;
    #   - lumped RLC elements: not wired into either scan body;
    #   - flux monitors / NTFF: frequency-domain accumulators exist only in
    #     the canonical Simulation.run() loop (DFT PLANE probes are the
    #     #578 exception — built into ``dft_probes`` below and threaded
    #     into the fast-path scan body);
    #   - non-uniform mesh profiles: scan bodies assume a uniform grid.
    if boundary == "upml":
        return None, None
    #   - surface-impedance (f0) sheets (#677): the sheet is a per-step
    #     node-thin operator ctx, not a materials.sigma fold, and neither
    #     vmap scan body applies it — the fast path would silently sweep a
    #     sheet-FREE model. The sequential fallback runs Simulation.run()
    #     per value, which applies the ctx (batch-invariant by
    #     construction there).
    from rfx.materials.thin_conductor import has_f0_sheets
    if has_f0_sheets(getattr(sim, "_thin_conductors", None)):
        return None, None
    if getattr(sim, "_lumped_rlc", None):
        return None, None
    if getattr(sim, "_flux_monitors", None):
        return None, None
    if getattr(sim, "_ntff", None) is not None:
        return None, None
    if (
        getattr(sim, "_dx_profile", None) is not None
        or getattr(sim, "_dy_profile", None) is not None
        or getattr(sim, "_dz_profile", None) is not None
    ):
        return None, None

    # Build sources and probes from the simulation.
    # For CPML J-sources the Cb coefficient depends on material properties
    # at the source cell, so we store *raw* (un-normalized) waveforms and
    # the source cell indices so Cb can be computed dynamically inside the
    # vmapped function from the actual material arrays.
    sources = []
    j_source_raw_info: list[tuple[int, int, int, str, jnp.ndarray]] = []
    probes = []

    for pe in sim._ports:
        if pe.impedance == 0.0:
            # amplitude_kind (issue #571) routing:
            #   None      -> legacy per-boundary routing, bit-identical
            #               (cpml: dynamic-Cb J-source; else raw make_source)
            #   'field'   -> raw make_source on EVERY boundary (E += w has
            #               no material dependence, so no dynamic path)
            #   'current' -> dynamic-Cb J-source on EVERY boundary, with the
            #               raw waveform prescaled by the static 1/dV part
            #               (Cb is material-dependent and varies per sweep
            #               element, so it stays dynamic; dV is geometry)
            if pe.amplitude_kind == "field":
                sources.append(make_source(grid, pe.position, pe.component,
                                           pe.waveform, n_steps))
            elif pe.amplitude_kind == "current" or (
                    pe.amplitude_kind is None and boundary == "cpml"):
                # Store raw waveform + cell info for dynamic Cb computation
                idx = grid.position_to_index(pe.position)
                times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
                raw_waveform = jax.vmap(pe.waveform)(times)
                if pe.amplitude_kind == "current":
                    from rfx.api._source_semantics import (
                        source_amplitude_scale)
                    from rfx.simulation import _uniform_cell_volume
                    # 'current' <- 'cb': static 1/dV (cb unused on this
                    # branch); the dynamic per-batch Cb multiply below in
                    # _build_vmap_scan_fn completes Cb*w/dV.
                    raw_waveform = source_amplitude_scale(
                        "current", "cb", cb=None,
                        dV=_uniform_cell_volume(grid)) * raw_waveform
                j_source_raw_info.append(
                    (idx[0], idx[1], idx[2], pe.component, raw_waveform)
                )
            else:
                sources.append(make_source(grid, pe.position, pe.component,
                                           pe.waveform, n_steps))

    for pe in sim._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    # DFT plane probes (#578): mirrors rfx/runners/uniform.py:478-498
    # exactly, including calling the SAME init_dft_plane_probe helper, so
    # the accumulator dtype/shape/init state is identical to run()'s by
    # construction (the #477/#484 x64-contract pin below is therefore a
    # sanity assertion, not a computed correction).
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    dft_probes: list[DFTPlaneProbe] = []
    dft_names: list[str] = []
    for pe in getattr(sim, "_dft_planes", []) or []:
        axis_idx = axis_to_index[pe.axis]
        plane_pos = [0.0, 0.0, 0.0]
        plane_pos[axis_idx] = pe.coordinate
        grid_index = grid.position_to_index(tuple(plane_pos))[axis_idx]
        freqs_arr = (
            pe.freqs
            if pe.freqs is not None
            else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
        )
        dft_probes.append(
            init_dft_plane_probe(
                axis=axis_idx,
                index=grid_index,
                component=pe.component,
                freqs=freqs_arr,
                grid_shape=grid.shape,
                dft_total_steps=n_steps,
            )
        )
        dft_names.append(pe.name)

    if dft_probes:
        _expected_dtype = (
            jnp.complex128 if jax.config.x64_enabled else jnp.complex64)
        for _probe in dft_probes:
            assert _probe.accumulator.dtype == _expected_dtype, (
                f"vmap DFT accumulator dtype {_probe.accumulator.dtype} != "
                f"run()'s {_expected_dtype} "
                f"(x64_enabled={jax.config.x64_enabled}) — #477/#484 "
                "x64-contract pin violated"
            )

    # For the simple (no-port, no-dispersion, no-TFSF, no-MSL/floquet/coax
    # port) case, use the lightweight scan function that can be cleanly
    # vmapped.
    if (
        not has_ports and not has_tfsf and not has_waveguide
        and not has_dispersion and not has_msl_ports
        and not has_floquet_ports and not has_coaxial_ports
    ):
        periodic = (False, False, False)
        if sim._periodic_axes:
            periodic = tuple(axis in sim._periodic_axes for axis in "xyz")
        if grid.is_2d:
            periodic = (periodic[0], periodic[1], True)

        cpml_axes = "xyz"
        axis_names = ("x", "y", "z")
        for axis_name, is_periodic in zip(axis_names, periodic):
            if is_periodic:
                cpml_axes = cpml_axes.replace(axis_name, "")
        pec_axes = "".join(
            axis_name for axis_name, is_periodic in zip(axis_names, periodic)
            if not is_periodic
        )

        if boundary == "cpml":
            # CPML requires its own state management.  For the vmapped path,
            # we build a scan function that includes CPML handling.
            run_one_fn = _build_vmap_scan_fn(
                grid, n_steps,
                use_cpml=True,
                sources=sources,
                j_source_raw_info=j_source_raw_info,
                probes=probes,
                periodic=periodic,
                cpml_axes=cpml_axes,
                pec_axes=pec_axes,
                pec_mask=pec_mask,
                dft_probes=dft_probes,
            )
        else:
            run_one_fn = _build_vmap_scan_fn(
                grid, n_steps,
                use_cpml=False,
                sources=sources,
                # non-empty only for amplitude_kind='current' soft sources
                # on a non-cpml boundary (issue #571): their Cb is material-
                # dependent and must be computed dynamically per sweep
                # element, exactly like the legacy cpml J-source path.
                j_source_raw_info=j_source_raw_info,
                probes=probes,
                periodic=periodic,
                pec_axes=pec_axes,
                pec_mask=pec_mask,
                dft_probes=dft_probes,
            )
        return run_one_fn, dft_names

    # Fallback: for complex sims, run sequentially (not vmapped)
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vmap_material_sweep(
    sim,
    param_name: str,
    param_values,
    *,
    n_steps: int | None = None,
    num_periods: float = 20.0,
    return_fields: bool = False,
) -> VmapSweepResult:
    """Batch-evaluate multiple material parameter values using ``jax.vmap``.

    Only works for material parameters (``eps_r``, ``sigma``, ``mu_r``)
    where the grid shape stays constant.  For geometry sweeps, use
    ``parametric_sweep()``.

    Parameters
    ----------
    sim : Simulation
        Base simulation (all geometry already added).
    param_name : str
        Material parameter to sweep: ``"eps_r"``, ``"sigma"``, ``"mu_r"``,
        or a material-specific name like ``"substrate.eps_r"``.
    param_values : array-like, shape (n_batch,)
        Values to evaluate.
    n_steps : int or None
        Timesteps.  If None, auto-computed from *num_periods*.
    num_periods : float
        Periods at freq_max for auto timestep count (default 20).
    return_fields : bool
        Must be False (the default). ``return_fields=True`` was documented
        here since this function's introduction but never implemented —
        neither the fast path nor the sequential fallback ever populated
        ``VmapSweepResult.final_fields``, so it silently returned ``None``
        instead of the promised snapshot. Passing ``True`` now raises
        ``ValueError`` (fail loud beats silent ``None``, #578). Use
        ``sim.run()``/``sim.forward()`` for a final-field snapshot of one
        configuration, or ``parametric_sweep()`` for full per-value
        ``Result`` objects across a swept parameter.

    Returns
    -------
    VmapSweepResult
        Result with ``.time_series`` of shape
        ``(n_batch, n_steps, n_probes)``, ``.dft_planes`` (dict of
        ``(n_batch, n_freqs, n1, n2)`` complex arrays keyed by
        ``add_dft_plane_probe`` name, or ``None`` if no plane was
        registered), and ``.param_values``.

    Notes
    -----
    The entire FDTD time loop is vmapped so all simulations execute in a
    single fused GPU kernel. Memory scales as
    ``n_batch * grid_size + n_batch * sum(n_freqs_p * n1_p * n2_p for each
    registered DFT plane p)`` (complex accumulators; no remat on this
    path). For large grids/plane counts, reduce batch size or plane
    frequency count to avoid OOM.

    Features supported in vmap path: PEC boundaries, CPML absorbing
    boundaries, soft sources (incl. ``amplitude_kind='current'``, #571),
    point probes, and DFT plane probes (``add_dft_plane_probe``, #578).
    Lumped ports, MSL/coaxial/floquet ports, TFSF, dispersion, waveguide
    ports, flux monitors, and NTFF are **not** supported in the vmap fast
    path and trigger a sequential fallback. As of #578 that fallback also
    populates ``.dft_planes`` (by stacking each swept value's
    ``Result.dft_planes`` accumulators), so registering a DFT plane never
    silently loses data on either path — it is only the FAST path that is
    conditional on eligibility, not the DFT-plane *output*.

    TFSF sweeps — angle batching is intentionally out of scope, not just
    unimplemented: it is a structural mismatch with this function's
    material-only batching model, not a missing feature. The TFSF incident
    field is itself an auxiliary FDTD solution advanced *inside* the scan
    (not closed-form); Method B's auxiliary-grid size is angle-dependent
    (a batch of angles would need a batch of scan shapes, which
    ``jax.vmap`` cannot express); and the oblique 2D-aux Bloch path
    concretizes its transverse phase factors from ``angle_deg`` at trace
    time (rfx/sources/tfsf_2d.py). More basically, rfx cannot even
    *express* a general incidence triple ``(theta, phi, psi)`` today — only
    ``angle_deg`` + a binary ``direction`` + a two-way ``polarization``
    choice — so there is no batchable angle parameter to expose in the
    first place. Route illumination/angle sweeps to ``parametric_sweep()``
    instead: it already returns full ``Result`` objects including
    ``dft_planes`` for each swept value. Caveat there too: oblique-Bloch
    TFSF (``angle_deg != 0``, ``method='bloch'``, the default) combined
    with DFT planes/flux monitors/NTFF raises ``NotImplementedError`` by
    design (rfx/simulation.py:728-745) because the accumulated envelope is
    not the physical spectrum — DFT-plane tensors from a TFSF-illuminated
    ``parametric_sweep`` are obtainable only at normal incidence
    (``angle_deg=0``) or with ``method='methodB'`` (open-domain, real
    fields). A MATERIAL sweep on a sim carrying a FIXED (non-swept) TFSF
    source still benefits from #578: it takes this function's sequential
    fallback (TFSF is not fast-path-eligible) but that fallback now
    carries ``dft_planes`` through, so per-material DFT tensors at fixed
    incidence ARE obtainable via ``vmap_material_sweep`` today.

    Point-frequency-domain probes are not a separate case here: rfx has no
    public builder for a point DFT probe (only plane DFT probes and flux
    monitors), and point *time series* — the actual point-probe primitive
    — are already fully batched via ``add_probe`` + ``.time_series`` on
    both paths.
    """
    # #706: neither the vmap fast path nor the sequential fallback is
    # witnessed with the two-plane slab realization — refuse loudly
    # rather than run the one-plane behaviour the user opted out of.
    sim._refuse_two_plane("vmap_material_sweep (batched)")
    if return_fields:
        raise ValueError(
            "return_fields=True is not implemented on vmap_material_sweep "
            "— it was accepted since this function's introduction but "
            "silently ignored (VmapSweepResult.final_fields was never "
            "populated on either the fast path or the sequential "
            "fallback). Use sim.run()/sim.forward() for a final-field "
            "snapshot, or parametric_sweep() for full per-value Result "
            "objects."
        )
    param_values = np.asarray(param_values, dtype=np.float32).ravel()
    if len(param_values) == 0:
        raise ValueError("param_values must not be empty")

    # Validate param_name
    _parse_param_name(param_name)

    # Build grid and base materials once
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask, *_ = sim._assemble_materials(grid)

    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)

    # Build the vmappable scan function
    run_one_fn, dft_names = _build_full_scan_fn(
        sim, grid, base_materials, n_steps,
        debye_spec=debye_spec,
        lorentz_spec=lorentz_spec,
        pec_mask=pec_mask,
    )

    jax_param_values = jnp.asarray(param_values)

    if run_one_fn is not None:
        # Fast path: vmap over material arrays
        batched_materials = _build_batched_materials(
            sim, grid, base_materials, param_name, jax_param_values,
        )

        # vmap run_one over the batch dimension of materials
        def run_one_from_materials(eps_r, sigma, mu_r):
            mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
            return run_one_fn(mats)

        batched_run = jax.vmap(run_one_from_materials)
        time_series, dft_accs = batched_run(
            batched_materials.eps_r,
            batched_materials.sigma,
            batched_materials.mu_r,
        )
        time_series_np = np.asarray(time_series)

        dft_planes_out = None
        if dft_names:
            dft_planes_out = {
                name: np.asarray(acc)
                for name, acc in zip(dft_names, dft_accs)
            }

        return VmapSweepResult(
            time_series=time_series_np,
            param_name=param_name,
            param_values=param_values,
            dft_planes=dft_planes_out,
        )
    else:
        # Fallback: sequential execution for complex simulations
        # Still uses the same interface but runs one at a time
        import warnings
        warnings.warn(
            "Simulation uses features not supported by vmap fast path "
            "(ports incl. MSL/coaxial/floquet, TFSF, dispersion, "
            "waveguide, flux monitors, NTFF). Falling back to sequential "
            "execution. Use parametric_sweep() for better sequential "
            "support.",
            stacklevel=2,
        )
        return _sequential_fallback(
            sim, param_name, param_values, n_steps=n_steps,
        )


def _sequential_fallback(
    sim,
    param_name: str,
    param_values: np.ndarray,
    *,
    n_steps: int,
) -> VmapSweepResult:
    """Sequential fallback when vmap is not possible.

    #578: also populates ``VmapSweepResult.dft_planes`` by stacking each
    swept value's ``Result.dft_planes[name].accumulator`` — the SAME
    carrier the fast path populates (dict[str, ndarray] with a leading
    batch axis), so a DFT plane registered on a simulation that must take
    this fallback (TFSF, lumped/MSL/coaxial/floquet ports, dispersion,
    waveguide ports, ...) still comes back through
    ``vmap_material_sweep()`` instead of only being reachable via
    ``parametric_sweep()``.
    """
    mat_name, field = _parse_param_name(param_name)

    all_ts = []
    dft_acc_lists: dict[str, list] = {}
    for _i, val in enumerate(param_values):
        # Clone the simulation and modify the material
        import copy
        sim_copy = copy.deepcopy(sim)
        # ``dataclasses.replace`` carries ALL MaterialSpec fields and
        # overrides only the swept one. The prior explicit field-by-field
        # reconstruction listed eps_r/sigma/mu_r/debye_poles/lorentz_poles
        # but NOT chi3 — a vmap fallback on a Kerr material silently
        # dropped the nonlinearity (Tier-3 silent-wrong-answer).
        if mat_name is not None:
            # Modify the named material.
            mat = sim_copy._resolve_material(mat_name)
            sim_copy._materials[mat_name] = replace(
                mat, **{field: float(val)})
        else:
            # Modify all custom materials.
            for name, mat in list(sim_copy._materials.items()):
                sim_copy._materials[name] = replace(
                    mat, **{field: float(val)})

        # Preflight only the first sim (this sequential fallback re-runs a
        # structurally-identical setup); skip thereafter to avoid per-iteration
        # preflight noise.
        result = sim_copy.run(n_steps=n_steps, skip_preflight=_i > 0)
        all_ts.append(np.asarray(result.time_series))

        if result.dft_planes:
            for name, probe in result.dft_planes.items():
                dft_acc_lists.setdefault(name, []).append(
                    np.asarray(probe.accumulator))

    # Stack into (n_batch, n_steps, n_probes)
    time_series = np.stack(all_ts, axis=0)

    dft_planes_out = None
    if dft_acc_lists:
        dft_planes_out = {
            name: np.stack(accs, axis=0)
            for name, accs in dft_acc_lists.items()
        }

    return VmapSweepResult(
        time_series=time_series,
        param_name=param_name,
        param_values=param_values,
        dft_planes=dft_planes_out,
    )

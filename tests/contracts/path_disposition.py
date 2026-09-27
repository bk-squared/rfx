"""How every time-stepping path treats every ``Simulation`` attribute.

A ``Simulation`` is a declaration: materials, conductors, ports, sources,
boundaries, a mesh and observers. Several paths step it in time, and each one
reads only part of the declaration. An input a path does not read is solved as
if it had not been declared, and nothing in the result says so: a Kerr block
comes back linear, a microstrip port launches nothing, a periodic wall is
solved as a metal one. The PI's rule (2026-09-24) is that a physics input a
path does not implement is refused before the first step. This table records,
for every attribute and every path, which of these holds:

``carries``        the path solves it;
``refuses``        the path raises before stepping;
``falls back``     the path hands the model to another path (named);
``not reachable``  no model with the attribute reaches the path (why);
``ignorable``      bookkeeping or a setting the path does not need (why);
``not yet classified``  only on the calculator columns, see below.

A cell whose behaviour today contradicts its disposition names the open issue
in ``wrong``; its executable check is a strict expected failure until the
issue is closed. For such a cell the disposition is the rule's minimum
(``refuses``) when today's behaviour is a drop, and ``carries`` when the path
solves the input but gets it wrong.

Where the cells are checked
---------------------------
* ``tests/contracts/test_path_disposition.py``: every attribute of a fresh
  ``Simulation`` has a row, every row has a cell on every column, the lane
  columns are exactly the lanes ``_dispatch_plan`` returns, and every function
  in ``rfx/`` that reaches a time-stepping kernel is registered.
* ``tests/unit/runners/test_path_disposition_cells.py``: the physics and
  observer rows. ``refuses`` runs a small model and expects a raise before any
  kernel scan starts; ``carries`` checks that the attribute changes the
  result (and, where cheap, that the lane agrees with ``run_uniform``);
  ``falls back`` checks the warning and the result of the named path.

Settings and bookkeeping rows (``ROW_CLASS``) carry only this entry. A few
settings are checked where they are refused, by the tests named in their
notes. Arguments to ``run()`` and ``forward()`` are not attributes; #1297's
``tests/unit/runners/test_silent_drop_warnings.py`` covers them.

Columns
-------
The eight lane columns are the lanes ``Simulation._dispatch_plan`` selects:
``run()`` on a uniform, graded, subgridded, ADI and multi-device model, and
``forward()`` on a uniform, graded and distributed graded model. This table
replaces the multi-device-only table of #1241
(``tests/unit/runners/test_distributed_admission_refusals.py``); its column
is ``run_distributed``.

Seven more entry points step a ``Simulation`` without ``_dispatch_plan``:
each assembles the model itself and calls a kernel. Their columns are
``not yet classified`` in this table; a follow-up classifies them.

* ``s_matrix_scan``: the lumped/wire S-matrix of ``run(compute_s_params=True)``
  (``compute_lumped_wire_s_matrix_via_scan`` → ``_forward_from_materials``).
* ``mixed_s_matrix``: ``compute_mixed_s_matrix`` → ``_forward_from_materials``.
* ``topology_optimize``: → ``_forward_from_materials``.
* ``waveguide_s_matrix``: ``compute_waveguide_s_matrix``; uniform mesh through
  ``rfx.sources.waveguide_port.extract_*`` → ``rfx.simulation.run``, graded
  mesh through ``run_nonuniform_path``.
* ``coax_calculators``: ``compute_coaxial_line_reflection``,
  ``compute_coaxial_two_port``, ``compute_coax_msl_transition`` →
  ``rfx.simulation.run``.
* ``vmap_sweep_batched``: the batched kernel of ``vmap_material_sweep``
  (its own scan; its sequential fallback calls ``run()``).
* ``material_fit``: ``differentiable_material_fit`` → ``rfx.simulation.run``.

Folded into the lane columns, because they call ``sim.run()`` or
``sim.forward()`` on the caller's model and add no assembly of their own:
``parametric_sweep``, ``ntff_sweep``, ``run_batch`` and
``run_batch_with_manifest``, ``config.runner.run_and_save``,
``convergence_study``, the ``RISUnitCell`` sweeps, the diagnostics smoke run,
``optimize`` (``forward(eps_override=...)``), ``profile_forward``, the
sequential fallback of ``vmap_material_sweep``, ``compute_s_matrix`` (it
dispatches to the calculators above or to ``run()``), ``compute_msl_s_matrix``
(``forward()`` or ``run()`` per drive) and ``run(ringdown=...)``.

Not ``Simulation`` paths, because they take a ``Grid``: ``rfx.run``,
``rfx.run_until_decay``, ``compute_rcs``, ``gpu.benchmark``, the eager
``extract_s_matrix`` / ``extract_s_matrix_wire`` and
``rfx.subgridding.runner.run_subgridded``.
"""

from __future__ import annotations

from typing import NamedTuple

CARRIES = "carries"
REFUSES = "refuses"
FALLS_BACK = "falls back"
NOT_REACHABLE = "not reachable"
IGNORABLE = "ignorable"
UNCLASSIFIED = "not yet classified"
KINDS = (CARRIES, REFUSES, FALLS_BACK, NOT_REACHABLE, IGNORABLE, UNCLASSIFIED)

LANES = (
    "run_uniform", "run_nonuniform", "run_subgridded", "run_adi",
    "run_distributed", "fwd_uniform", "fwd_nonuniform", "fwd_distributed_nu",
)
CALCULATORS = (
    "s_matrix_scan", "mixed_s_matrix", "topology_optimize",
    "waveguide_s_matrix", "coax_calculators", "vmap_sweep_batched",
    "material_fit",
)
PATHS = LANES + CALCULATORS

# A row's class decides what the cells test checks: physics and observer rows
# have executable cells, settings and bookkeeping rows only this entry.
PHYSICS, OBSERVER, SETTING, BOOKKEEPING = "physics", "observer", "setting", "bookkeeping"


class Cell(NamedTuple):
    kind: str
    note: str = ""
    wrong: str = ""   # open issue while today's behaviour contradicts ``kind``
    to: str = ""      # the path a ``falls back`` cell hands the model to


def carries(note="", *, wrong=""):
    return Cell(CARRIES, note, wrong)


def refuses(note="", *, wrong=""):
    return Cell(REFUSES, note, wrong)


def falls_back(to, note=""):
    return Cell(FALLS_BACK, note, to=to)


def not_reachable(why):
    return Cell(NOT_REACHABLE, why)


def ignorable(why):
    return Cell(IGNORABLE, why)


def lanes(*, run_uniform, run_nonuniform, run_subgridded, run_adi,
          run_distributed, fwd_uniform, fwd_nonuniform, fwd_distributed_nu):
    """One cell per lane; a missing lane is a TypeError at import."""
    return dict(run_uniform=run_uniform, run_nonuniform=run_nonuniform,
                run_subgridded=run_subgridded, run_adi=run_adi,
                run_distributed=run_distributed, fwd_uniform=fwd_uniform,
                fwd_nonuniform=fwd_nonuniform,
                fwd_distributed_nu=fwd_distributed_nu)


# Only a bookkeeping row may give one cell for every path, present and future:
# the completeness test accepts ``"*"`` for an ``ignorable`` cell only.
def every_path(cell):
    return {"*": cell}


# Recurring notes.
_PROFILE_TO_GRADED = "a dx/dy/dz profile sends the model to the graded lane"
_NO_PROFILE_DT = ("the constructor refuses it without a dx/dy/dz profile, and a "
                  "profile sends the model to the graded lane")
_SUBGRID_ENVELOPE = ("inside the production envelope: PEC walls, no CPML, a "
                     "slab touching one z wall")
_SUBGRID_DISPERSIVE = ("production validation refuses it "
                       "(dispersive_or_nonlinear_material); research/off drop "
                       "it, see _refinement 'relaxed_validation'")
_SHEET_OWNS_NO_CELL = "a sheet or wire owns no cell, and this lane applies PEC from a cell mask (#931)"
_ADI_INTERIOR_PEC = "adi_interior_pec_unsupported"
_ADI_SOFT_SOURCES = "ADI takes add_source() soft sources only"
_ADI_UNIFORM_ABSORBER = "ADI stamps one absorber on all six faces; a per-face layout is refused"
_DIST_FWD_PORTS = "lumped and wire ports are refused on distributed forward()"
_CONFORMAL_REFUSED = ("Boundary(conformal=True) is refused like conformal_pec=True: "
                      "this lane has no Dey-Mittra update (#1297)")


TABLE: dict[str, dict[str, dict[str, Cell]]] = {
    # ------------------------------------------------------------ settings
    "_freq_max": {"": lanes(
        run_uniform=carries("sets the step count from num_periods, the auto mesh and the default S band"),
        run_nonuniform=carries("as run_uniform"),
        run_subgridded=carries("as run_uniform; the count is scaled by the ratio"),
        run_adi=carries("as run_uniform"),
        run_distributed=carries("as run_uniform"),
        fwd_uniform=carries("as run_uniform"),
        fwd_nonuniform=carries("as run_uniform"),
        fwd_distributed_nu=carries("as run_uniform"),
    )},
    "_domain": {"": lanes(
        run_uniform=carries("the grid extent"),
        run_nonuniform=carries("the grid extent; a profile must sum to it"),
        run_subgridded=carries("the coarse grid extent"),
        run_adi=carries("the grid extent"),
        run_distributed=carries("the grid extent, padded to split along x"),
        fwd_uniform=carries("the grid extent"),
        fwd_nonuniform=carries("the grid extent"),
        fwd_distributed_nu=carries("the grid extent"),
    )},
    "_dx": {"": lanes(
        run_uniform=carries("the cell"),
        run_nonuniform=carries("the cell of every axis without a profile"),
        run_subgridded=carries("the coarse cell; the fine cell is dx/ratio"),
        run_adi=carries("the cell"),
        run_distributed=carries("the cell"),
        fwd_uniform=carries("the cell"),
        fwd_nonuniform=carries("the cell of every axis without a profile"),
        fwd_distributed_nu=carries("the cell of every axis without a profile"),
    )},
    "_dt_pin": {"": lanes(
        run_uniform=not_reachable(_NO_PROFILE_DT),
        run_nonuniform=carries("the graded grid's time step"),
        run_subgridded=not_reachable(_NO_PROFILE_DT + ", where a refinement is refused (#1282)"),
        run_adi=not_reachable(_NO_PROFILE_DT + "; ADI requires a uniform mesh"),
        run_distributed=carries("the distributed graded grid's time step (same probe as run())"),
        fwd_uniform=not_reachable(_NO_PROFILE_DT),
        fwd_nonuniform=carries("the graded grid's time step"),
        fwd_distributed_nu=carries("the distributed graded grid's time step"),
    )},
    "_dt_min_cell": {"": lanes(
        run_uniform=not_reachable("the constructor refuses it without dt="),
        run_nonuniform=carries("with dt=, read by the graded grid build"),
        run_subgridded=not_reachable("the constructor refuses it without dt=, and dt= needs a profile"),
        run_adi=not_reachable("the constructor refuses it without dt=, and dt= needs a profile"),
        run_distributed=carries("with dt=, read by _build_nonuniform_grid"),
        fwd_uniform=not_reachable("the constructor refuses it without dt="),
        fwd_nonuniform=carries("with dt=, read by the graded grid build"),
        fwd_distributed_nu=carries("with dt=, read by _build_nonuniform_grid"),
    )},
    "_precision": {"": lanes(
        run_uniform=carries("field dtype; float64 needs jax x64, without it every lane runs float32 and JAX warns"),
        run_nonuniform=refuses("non-float32 refused (#630)"),
        run_subgridded=refuses("non-float32 refused (#630)"),
        run_adi=refuses("not a named refusal: with x64 on, the ADI scan raises TypeError on the float64 carry"),
        run_distributed=refuses("non-float32 refused (#630)"),
        fwd_uniform=carries("field dtype"),
        fwd_nonuniform=refuses("non-float32 refused (#630)"),
        fwd_distributed_nu=refuses("non-float32 refused (#630)"),
    )},
    "_solver": {"": lanes(
        run_uniform=not_reachable("solver='adi' sends run() to run_adi"),
        run_nonuniform=refuses("solver='adi' requires a uniform mesh (_dispatch_plan)"),
        run_subgridded=refuses("run() dispatches solver='adi' before the refinement, and ADI refuses subgridding"),
        run_adi=carries("the lane solver='adi' selects"),
        run_distributed=refuses("solver='adi' does not support distributed execution"),
        fwd_uniform=carries("_forward_from_materials routes solver='adi' to the ADI kernel (same probe as run())"),
        fwd_nonuniform=refuses("solver='adi' requires a uniform mesh (_dispatch_plan)"),
        fwd_distributed_nu=refuses("solver='adi' requires a uniform mesh (_dispatch_plan)"),
    )},
    "_adi_cfl_factor": {"": lanes(
        run_uniform=ignorable("setting of solver='adi'"),
        run_nonuniform=ignorable("setting of solver='adi'"),
        run_subgridded=ignorable("setting of solver='adi'"),
        run_adi=carries("the ADI step is the Yee step times this factor"),
        run_distributed=ignorable("setting of solver='adi'"),
        fwd_uniform=carries("read when solver='adi' routes forward() to the ADI kernel"),
        fwd_nonuniform=ignorable("setting of solver='adi'"),
        fwd_distributed_nu=ignorable("setting of solver='adi'"),
    )},
    "_stencil_order": {"": lanes(
        run_uniform=carries("order 4 on the plain uniform PEC/periodic path"),
        run_nonuniform=refuses("stencil_order=4 refused (_check_stencil_order_supported)"),
        run_subgridded=refuses("stencil_order=4 refused"),
        run_adi=refuses("stencil_order=4 refused"),
        run_distributed=refuses("stencil_order=4 refused"),
        fwd_uniform=carries("order 4 on the plain uniform PEC/periodic path"),
        fwd_nonuniform=refuses("stencil_order=4 refused"),
        fwd_distributed_nu=refuses("stencil_order=4 refused"),
    )},
    "_mode": {"": lanes(
        run_uniform=carries("3d, 2d_tmz and 2d_tez"),
        run_nonuniform=carries(
            "2d_tmz on one z cell between PEC walls is solved as that 3-D box, other 2-D "
            "models are refused. That box differs from run_uniform's 2-D solve by 0.34 of "
            "the probe peak, as does run_uniform's own one-cell 3-D box (measured, field "
            "source): the lanes disagree on what a 2d_tmz model is"),
        run_subgridded=refuses("production validation: a slab across a one-cell z domain is never one-sided"),
        run_adi=carries("3d and 2d_tmz; 2d_tez refused"),
        run_distributed=carries("2d_tmz matched one device bit for bit (measured)"),
        fwd_uniform=carries("3d, 2d_tmz and 2d_tez"),
        fwd_nonuniform=carries("as run_nonuniform"),
        fwd_distributed_nu=carries("as run_nonuniform"),
    )},
    "_interface_eps": {"": lanes(
        run_uniform=ignorable("not read; 'dual_average' gave a field bit-identical to 'sampled' (εr 4 block, measured)"),
        run_nonuniform=carries("read by the graded assembly (assemble_interface_eps_nu)"),
        run_subgridded=ignorable("not read by the subgridded runner; not measured"),
        run_adi=ignorable("not read; bit-identical to 'sampled' (measured)"),
        run_distributed=refuses("'dual_average' refused"),
        fwd_uniform=ignorable("not read; bit-identical to 'sampled' (measured)"),
        fwd_nonuniform=carries("read by the graded assembly"),
        fwd_distributed_nu=refuses("'dual_average' refused"),
    )},

    # --------------------------------------------------------- bookkeeping
    "_internal_probe_indices": {"": every_path(ignorable("probe bookkeeping"))},
    "_msl_auto_offset_min": {"": every_path(ignorable("MSL port data, read with the port; the _msl_ports row decides it"))},
    "_msl_auto_probe_spacing": {"": every_path(ignorable("MSL port data, read with the port; the _msl_ports row decides it"))},
    "_msl_auto_probe_lengths": {"": every_path(ignorable("MSL port data, read with the port; the _msl_ports row decides it"))},
    "_boundary_model": {"": every_path(ignorable(
        "derived from the boundary declaration; the _boundary, _boundary_spec, _pec_faces "
        "and _periodic_axes rows decide it"))},

    # ----------------------------------------------------------- materials
    "_materials": {
        "eps": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries(_SUBGRID_ENVELOPE),
            run_adi=carries(),
            run_distributed=carries("each E edge takes ε from the cell that owns it", wrong="#1303"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("each E edge takes ε from the cell that owns it", wrong="#1303"),
        ),
        "sigma": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries(_SUBGRID_ENVELOPE),
            run_adi=carries("implicit conductivity in the ADI solve"),
            run_distributed=carries("each E edge takes σ from the cell that owns it", wrong="#1303"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("each E edge takes σ from the cell that owns it", wrong="#1303"),
        ),
        "mu": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries(_SUBGRID_ENVELOPE),
            run_adi=refuses("today dropped: the ADI kernel takes ε and σ only", wrong="#1308"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
        ),
        **{pole: lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=refuses(_SUBGRID_DISPERSIVE),
            run_adi=refuses("dispersive materials refused"),
            run_distributed=carries("in a CPML box the two-device field grows without bound", wrong="#1302"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
        ) for pole in ("debye", "lorentz", "drude")},
        "kerr": lanes(
            run_uniform=carries(),
            run_nonuniform=refuses("today dropped: the graded run() solves the material as linear", wrong="#1309"),
            run_subgridded=refuses(_SUBGRID_DISPERSIVE),
            run_adi=refuses("today dropped: the ADI kernel has no χ³ term", wrong="#1308"),
            run_distributed=refuses("Kerr χ³ refused (#1214)"),
            fwd_uniform=carries(),
            fwd_nonuniform=refuses("Kerr χ³ refused off the uniform forward lane"),
            fwd_distributed_nu=refuses("Kerr χ³ refused off the uniform forward lane"),
        ),
    },

    # ------------------------------------------------------------ geometry
    "_geometry": {
        "pec_volume": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries(),
            run_adi=refuses(_ADI_INTERIOR_PEC),
            run_distributed=carries("as a cell mask"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("as a cell mask"),
        ),
        **{shape: lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=refuses(_SHEET_OWNS_NO_CELL),
            run_adi=refuses(_ADI_INTERIOR_PEC),
            run_distributed=refuses(_SHEET_OWNS_NO_CELL),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=refuses(_SHEET_OWNS_NO_CELL),
        ) for shape in ("pec_sheet", "pec_wire")},
    },
    "_thin_conductors": {
        "lossy_sheet": lanes(
            run_uniform=carries("folded into σ"),
            run_nonuniform=carries("folded into σ"),
            run_subgridded=refuses("today dropped although production validation passes", wrong="#1311"),
            run_adi=refuses("thin-conductor corrections refused"),
            run_distributed=carries("folded into σ, which each E edge takes from the cell that owns it", wrong="#1303"),
            fwd_uniform=carries("folded into σ"),
            fwd_nonuniform=carries("folded into σ"),
            fwd_distributed_nu=carries("folded into σ, which each E edge takes from the cell that owns it", wrong="#1303"),
        ),
        "pec_sheet": lanes(
            run_uniform=carries("realized as a PEC sheet"),
            run_nonuniform=carries("realized as a PEC sheet"),
            run_subgridded=refuses(_SHEET_OWNS_NO_CELL),
            run_adi=refuses("thin-conductor corrections refused"),
            run_distributed=refuses(_SHEET_OWNS_NO_CELL),
            fwd_uniform=carries("realized as a PEC sheet"),
            fwd_nonuniform=carries("realized as a PEC sheet"),
            fwd_distributed_nu=refuses(_SHEET_OWNS_NO_CELL),
        ),
        "surface_impedance": lanes(
            run_uniform=carries("the per-step node-thin operator (#677)"),
            run_nonuniform=carries("the per-step node-thin operator"),
            run_subgridded=refuses("surface_impedance_f0 sheets refused"),
            run_adi=refuses("surface_impedance_f0 sheets refused"),
            run_distributed=refuses("surface_impedance_f0 sheets refused"),
            fwd_uniform=carries("the per-step node-thin operator"),
            fwd_nonuniform=carries("the per-step node-thin operator"),
            fwd_distributed_nu=refuses("surface_impedance_f0 sheets refused"),
        ),
    },
    "_pinned_sheets": {"pec_sheet": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses(_SHEET_OWNS_NO_CELL),
        run_adi=refuses(_ADI_INTERIOR_PEC),
        run_distributed=refuses(_SHEET_OWNS_NO_CELL),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses(_SHEET_OWNS_NO_CELL),
    )},

    # --------------------------------------------------- ports and sources
    "_ports": {
        "source": lanes(
            run_uniform=carries("add_source(): a soft point source"),
            run_nonuniform=carries(),
            run_subgridded=carries(_SUBGRID_ENVELOPE),
            run_adi=carries(),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
        ),
        "amplitude_kind": lanes(
            run_uniform=carries("'current' and 'field' scale the waveform differently (#571)"),
            run_nonuniform=carries(),
            run_subgridded=carries(),
            run_adi=refuses("today ignored: both kinds inject the same waveform", wrong="#1308"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
        ),
        "lumped_port": lanes(
            run_uniform=carries("drive and 50 Ω load"),
            run_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            run_subgridded=carries(_SUBGRID_ENVELOPE),
            run_adi=refuses(_ADI_SOFT_SOURCES),
            run_distributed=carries("single-cell excited ports"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            fwd_distributed_nu=refuses(_DIST_FWD_PORTS),
        ),
        "passive_port": lanes(
            run_uniform=carries("the 50 Ω load of excite=False"),
            run_nonuniform=carries(),
            run_subgridded=carries(_SUBGRID_ENVELOPE),
            run_adi=refuses(_ADI_SOFT_SOURCES),
            run_distributed=refuses("excite=False refused (#1241)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=refuses(_DIST_FWD_PORTS),
        ),
        "wire_port": lanes(
            run_uniform=carries("add_port(extent=...): drive and load along the wire"),
            run_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            run_subgridded=carries(_SUBGRID_ENVELOPE),
            run_adi=refuses(_ADI_SOFT_SOURCES),
            run_distributed=refuses("extended ports refused (#1241)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            fwd_distributed_nu=refuses(_DIST_FWD_PORTS),
        ),
    },
    "_msl_ports": {"msl_port": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("today dropped although production validation passes", wrong="#1311"),
        run_adi=refuses("a board with a PEC or thin-conductor trace is refused; today a port "
                        "declared without a trace is dropped", wrong="#1308"),
        run_distributed=refuses("MSL ports refused (#1241)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("today the port launches nothing", wrong="#1285"),
    )},
    "_waveguide_ports": {"waveguide_port": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("production validation (waveguide_port_unvalidated)"),
        run_adi=refuses("waveguide ports refused"),
        run_distributed=falls_back("run_uniform", "one device, with a warning; an explicit conformal_pec=False is overridden there (#1305)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("waveguide ports refused"),
    )},
    "_coaxial_ports": {"coax_port": lanes(
        run_uniform=refuses("run() refuses add_coaxial_port(); the coax calculators solve it"),
        run_nonuniform=refuses("run() refuses add_coaxial_port()"),
        run_subgridded=refuses("run() refuses add_coaxial_port()"),
        run_adi=refuses("run() refuses add_coaxial_port()"),
        run_distributed=refuses("run() refuses add_coaxial_port()"),
        fwd_uniform=refuses("forward() refuses add_coaxial_port()"),
        fwd_nonuniform=refuses("forward() refuses add_coaxial_port()"),
        fwd_distributed_nu=refuses("forward() refuses add_coaxial_port()"),
    )},
    "_floquet_ports": {"floquet_port": lanes(
        run_uniform=carries(),
        run_nonuniform=refuses("today launches nothing on a dx/dy profile; a dz profile is refused", wrong="#1312"),
        run_subgridded=refuses("production validation (floquet_port_unvalidated)"),
        run_adi=refuses("Floquet ports refused"),
        run_distributed=refuses("the periodic axes it sets are refused (#1241)"),
        fwd_uniform=carries(),
        fwd_nonuniform=refuses("today launches nothing on a dx/dy profile", wrong="#1312"),
        fwd_distributed_nu=refuses("today launches nothing on a dx/dy profile", wrong="#1312"),
    )},
    "_lumped_rlc": {element: lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("production validation (rlc_unvalidated); research/off drop it, "
                               "see _refinement 'relaxed_validation'"),
        run_adi=refuses("lumped RLC refused"),
        run_distributed=refuses("lumped RLC refused (#1239)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("lumped RLC refused"),
    ) for element in ("R", "series_RL")},
    "_tfsf": {"plane_wave": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("production validation (tfsf_unvalidated)"),
        run_adi=refuses("TFSF refused"),
        run_distributed=falls_back("run_uniform", "one device, with a warning; an explicit conformal_pec=False is overridden there (#1305)"),
        fwd_uniform=carries(),
        fwd_nonuniform=refuses("TFSF refused off the uniform forward lane"),
        fwd_distributed_nu=refuses("TFSF refused"),
    )},
    "_refinement": {
        "slab": lanes(
            run_uniform=not_reachable("a refinement sends run() on a uniform mesh to run_subgridded"),
            run_nonuniform=refuses("refinement on a graded mesh refused (#1282)"),
            run_subgridded=carries("the SBP-SAT fine slab"),
            run_adi=refuses("ADI refuses subgridding"),
            run_distributed=refuses("refinement refused (#1241)"),
            fwd_uniform=refuses("forward() has no subgridded lane (#1282)"),
            fwd_nonuniform=refuses("refinement on a graded mesh refused (#1282)"),
            fwd_distributed_nu=refuses("refinement on a graded mesh refused (#1282)"),
        ),
        "relaxed_validation": lanes(
            run_uniform=not_reachable("validation modes exist only on the subgridded lane; see 'slab'"),
            run_nonuniform=not_reachable("validation modes exist only on the subgridded lane; see 'slab'"),
            run_subgridded=refuses(
                "validation='research' and 'off' run the Debye, Lorentz and Drude poles, Kerr "
                "χ³ and lumped RLC that production refuses, and drop them", wrong="#1286"),
            run_adi=not_reachable("validation modes exist only on the subgridded lane; see 'slab'"),
            run_distributed=not_reachable("validation modes exist only on the subgridded lane; see 'slab'"),
            fwd_uniform=not_reachable("validation modes exist only on the subgridded lane; see 'slab'"),
            fwd_nonuniform=not_reachable("validation modes exist only on the subgridded lane; see 'slab'"),
            fwd_distributed_nu=not_reachable("validation modes exist only on the subgridded lane; see 'slab'"),
        ),
    },

    # ---------------------------------------------------------- boundaries
    "_boundary": {
        "cpml": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=refuses("production validation: no CPML in the guarded envelope"),
            run_adi=carries("a graded conductivity layer, not a CPML: tests/contracts/"
                            "test_realized_boundary.py cpml--adi is a strict xfail (#1221)"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
        ),
        "upml": lanes(
            run_uniform=carries(),
            run_nonuniform=refuses("the graded runner implements CPML only"),
            run_subgridded=refuses("production validation"),
            run_adi=refuses("boundary='upml' refused"),
            run_distributed=refuses("boundary='upml' refused"),
            fwd_uniform=carries(),
            fwd_nonuniform=refuses("the graded runner implements CPML only"),
            fwd_distributed_nu=refuses("the graded runner implements CPML only"),
        ),
    },
    "_pec_faces": {"pec_face": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("production validation: CPML on the other faces"),
        run_adi=refuses(_ADI_UNIFORM_ABSORBER),
        run_distributed=carries("the absorber backing differs from one device "
                                "(test_realized_boundary.py pec-zlo--distributed)", wrong="#1221"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
    )},
    "_boundary_spec": {
        "pmc_face": lanes(
            run_uniform=carries("the magnetic wall sits half a cell inside its face on every lane "
                                "that carries it: test_realized_boundary.py strict xfails (#1221)"),
            run_nonuniform=carries(),
            run_subgridded=refuses("production validation"),
            run_adi=refuses("today solved as an electric wall", wrong="#1221"),
            run_distributed=carries("on y/z faces; on x faces test_realized_boundary.py records "
                                    "tangential E held at zero (#1221)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("on y/z faces"),
        ),
        "conformal": lanes(
            run_uniform=carries("Dey-Mittra weights on every PEC shape"),
            run_nonuniform=refuses(_CONFORMAL_REFUSED),
            run_subgridded=refuses(_CONFORMAL_REFUSED),
            run_adi=refuses(_CONFORMAL_REFUSED),
            run_distributed=refuses(_CONFORMAL_REFUSED),
            fwd_uniform=refuses(_CONFORMAL_REFUSED),
            fwd_nonuniform=refuses(_CONFORMAL_REFUSED),
            fwd_distributed_nu=refuses(_CONFORMAL_REFUSED),
        ),
        "conformal_s_matrix": lanes(
            run_uniform=carries("the lumped-port S-matrix run() returns comes from "
                                "_forward_from_materials, which staircases", wrong="#1299"),
            run_nonuniform=refuses(_CONFORMAL_REFUSED),
            run_subgridded=refuses(_CONFORMAL_REFUSED),
            run_adi=refuses(_CONFORMAL_REFUSED),
            run_distributed=refuses(_CONFORMAL_REFUSED),
            fwd_uniform=refuses(_CONFORMAL_REFUSED),
            fwd_nonuniform=refuses(_CONFORMAL_REFUSED),
            fwd_distributed_nu=refuses(_CONFORMAL_REFUSED),
        ),
    },
    "_periodic_axes": {"periodic": lanes(
        run_uniform=carries("one cell longer than declared (#1221, B2)"),
        run_nonuniform=refuses("today solved as PEC walls (test_realized_boundary.py "
                               "periodic-xy--nonuniform)", wrong="#1221"),
        run_subgridded=refuses("today solved as PEC walls although production validation passes",
                               wrong="#1311"),
        run_adi=refuses("periodic axes refused"),
        run_distributed=refuses("periodic axes refused (#1241)"),
        fwd_uniform=carries("one cell longer than declared (#1221, B2)"),
        fwd_nonuniform=refuses("today solved as PEC walls", wrong="#1221"),
        fwd_distributed_nu=refuses("today solved as PEC walls", wrong="#1221"),
    )},
    "_cpml_layers": {"layers": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("production validation: no CPML in the guarded envelope"),
        run_adi=carries("thickness of ADI's conductivity layer"),
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
    )},
    "_cpml_kappa_max": {"kappa": lanes(
        run_uniform=carries(),
        run_nonuniform=refuses("today dropped: the graded grid build is never given it", wrong="#1310"),
        run_subgridded=refuses("production validation: no CPML in the guarded envelope"),
        run_adi=refuses("today dropped: ADI's absorber is not a CPML "
                        "(test_realized_boundary.py cpml--adi)", wrong="#1221"),
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=refuses("today dropped", wrong="#1310"),
        fwd_distributed_nu=refuses("today dropped", wrong="#1310"),
    )},

    # -------------------------------------------------------- mesh profiles
    **{profile: {"graded": lanes(
        run_uniform=not_reachable(_PROFILE_TO_GRADED),
        run_nonuniform=carries(),
        run_subgridded=refuses("the refinement is refused on a graded mesh (#1282)"),
        run_adi=refuses("ADI requires a uniform mesh"),
        run_distributed=carries("the distributed graded runner, grading up to 5:1"),
        fwd_uniform=not_reachable(_PROFILE_TO_GRADED),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
    )} for profile in ("_dx_profile", "_dy_profile", "_dz_profile")},

    # ------------------------------------------------------------ observers
    "_probes": {"probe": lanes(
        run_uniform=carries("every effect cell reads it"),
        run_nonuniform=carries(),
        run_subgridded=carries(),
        run_adi=carries(),
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
    )},
    "_dft_planes": {"dft_plane": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("production validation (dft_plane_unvalidated)"),
        run_adi=refuses("DFT planes refused"),
        run_distributed=refuses("DFT planes refused (#579)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("DFT planes refused (#579)"),
    )},
    "_flux_monitors": {"flux": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("today result.flux_monitors is None", wrong="#1313"),
        run_adi=refuses("today result.flux_monitors is None", wrong="#1313"),
        run_distributed=refuses("flux monitors refused (#1241)"),
        fwd_uniform=refuses("today ForwardResult has no flux field", wrong="#1313"),
        fwd_nonuniform=refuses("today ForwardResult has no flux field", wrong="#1313"),
        fwd_distributed_nu=refuses("flux monitors refused"),
    )},
    "_ntff": {"ntff_box": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=carries("inside the refined slab, clear of its artificial interface"),
        run_adi=refuses("NTFF refused"),
        run_distributed=refuses("NTFF refused (#1241)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("today ntff_data is None", wrong="#1313"),
    )},
}

ROW_CLASS: dict[str, str] = {
    **{attr: SETTING for attr in (
        "_freq_max", "_domain", "_dx", "_dt_pin", "_dt_min_cell", "_precision",
        "_solver", "_adi_cfl_factor", "_stencil_order", "_mode", "_interface_eps")},
    **{attr: BOOKKEEPING for attr in (
        "_internal_probe_indices", "_msl_auto_offset_min",
        "_msl_auto_probe_spacing", "_msl_auto_probe_lengths", "_boundary_model")},
    **{attr: OBSERVER for attr in (
        "_probes", "_dft_planes", "_flux_monitors", "_ntff")},
    **{attr: PHYSICS for attr in (
        "_materials", "_geometry", "_thin_conductors", "_pinned_sheets",
        "_ports", "_msl_ports", "_waveguide_ports", "_coaxial_ports",
        "_floquet_ports", "_lumped_rlc", "_tfsf", "_refinement", "_boundary",
        "_pec_faces", "_boundary_spec", "_periodic_axes", "_cpml_layers",
        "_cpml_kappa_max", "_dx_profile", "_dy_profile", "_dz_profile")},
}

# The calculator columns are classified in a follow-up. Found by reading, not
# run: topology_optimize and differentiable_material_fit discard Kerr χ³ from
# _assemble_materials; topology_optimize, s_matrix_scan and mixed_s_matrix skip
# forward()'s refusals, which may let coax ports and surface-impedance sheets
# through as well as #1299's conformal walls; differentiable_material_fit never
# reads TFSF, waveguide or Floquet ports, per-face boundaries, the solver or
# stencil_order. Known issues on these columns are named in their cells.
CALCULATOR_NOTES: dict[tuple[str, str], str] = {
    ("_boundary_spec", "s_matrix_scan"): "#1299: conformal walls staircased",
    ("_boundary_spec", "mixed_s_matrix"): "#1299: conformal walls staircased",
    ("_boundary_spec", "topology_optimize"): "#1299: conformal walls staircased",
    ("_solver", "waveguide_s_matrix"): "#1300: solver='adi' runs Yee",
    ("_dx", "vmap_sweep_batched"): "#1293: on an auto mesh each value runs its own dt for one step count (sequential path)",
    ("_ports", "material_fit"): "#1290: a wire port's extent and plain sources are dropped",
    ("_msl_ports", "material_fit"): "#1290: never read",
    ("_lumped_rlc", "material_fit"): "#1290: never read",
    **{("_refinement", calc): "refused (#1282)" for calc in (
        "s_matrix_scan", "topology_optimize", "waveguide_s_matrix",
        "vmap_sweep_batched", "material_fit")},
}


def cell(attr: str, feature: str, path: str) -> Cell:
    """The cell for ``path``; a calculator column is ``not yet classified``."""
    row = TABLE[attr][feature]
    if "*" in row:
        return row["*"]
    if path in CALCULATORS:
        return Cell(UNCLASSIFIED, CALCULATOR_NOTES.get((attr, path), ""))
    return row[path]

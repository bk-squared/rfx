"""Generate every requested admission cell; missing builders fail closed."""
from dataclasses import dataclass

from rfx.runners import _admission as admission
from tests.contracts import path_disposition as disposition

PAIRS = (
    ('run_uniform', 'run_nonuniform', False),
    ('run_uniform', 'run_distributed', False),
    ('fwd_uniform', 'fwd_nonuniform', False),
    ('run_nonuniform', 'run_distributed', True),
)
LONG_ROWS = {('_dft_planes', 'dft_plane'), ('_flux_monitors', 'flux'), ('_ports', 'wire_port')}


@dataclass(frozen=True)
class Cell:
    row: tuple
    a: str
    b: str
    graded: bool
    equivalence: bool
    refused: str | None
    steps: int

    @property
    def id(self):
        return ':'.join((*self.row, self.a, self.b, 'graded' if self.graded else 'constant', str(self.steps)))


def generate(builders):
    from .builders import build
    validated = set()
    cells = []
    for row, admitted in admission._ADMITTED_ON.items():
        for a, b, graded in PAIRS:
            on_a, on_b = a in admitted, b in admitted
            if not (on_a or on_b):
                continue
            assert row in builders, f'S0 missing builder: {row}'
            for lane in (a, b):
                assert (lane in admitted) == (row in admission.ADMITS[lane]), (row, lane)
                assert (lane in admitted) == (disposition.cell(*row, lane).kind in ('carries', 'ignorable')), (row, lane)
            equivalent = on_a and on_b
            if equivalent:
                for lane in (a, b):
                    key = (row, lane, graded)
                    if key not in validated:
                        sim = build(row, lane, graded=graded)
                        assert admission.DETECTORS[row](sim), f'S0 inactive builder: {row}, {lane}'
                        validated.add(key)
            for steps in ((12, 36) if equivalent and row in LONG_ROWS else (12,)):
                cells.append(Cell(row, a, b, graded, equivalent,
                                  None if equivalent else b if on_a else a, steps))
    # These paths are outside equivalence scope: only their refusals run.
    for row, admitted in admission._ADMITTED_ON.items():
        for lane in ('run_adi', 'fwd_adi', 'run_subgridded'):
            if lane not in admitted:
                assert row in builders, f'S0 missing refusal builder: {row}'
                cells.append(Cell(row, lane, 'refusal', False, False, lane, 0))
    return tuple(cells)


# Filled from the measured cold-solve cost of passing 12-step candidates.
PR_CHOICES = {
    "_adi_cfl_factor": "_adi_cfl_factor::run_uniform:run_nonuniform:constant:12",
    "_boundary": "_boundary:cpml:run_uniform:run_nonuniform:constant:12",
    "_boundary_spec": "_boundary_spec:absorbing_lid:run_uniform:run_nonuniform:constant:12",
    "_cpml_kappa_max": "_cpml_kappa_max:kappa:run_uniform:run_distributed:constant:12",
    "_cpml_layers": "_cpml_layers:layers:run_uniform:run_nonuniform:constant:12",
    "_current_moments": "_current_moments:block_moments:run_uniform:run_nonuniform:constant:12",
    "_domain": "_domain::run_uniform:run_nonuniform:constant:12",
    "_dt_min_cell": "_dt_min_cell::run_nonuniform:run_distributed:graded:12",
    "_dt_pin": "_dt_pin::run_nonuniform:run_distributed:graded:12",
    "_dx": "_dx::run_uniform:run_nonuniform:constant:12",
    "_dx_profile": "_dx_profile:graded:run_nonuniform:run_distributed:graded:12",
    "_dy_profile": "_dy_profile:graded:run_nonuniform:run_distributed:graded:12",
    "_dz_profile": "_dz_profile:graded:run_nonuniform:run_distributed:graded:12",
    "_freq_max": "_freq_max::run_uniform:run_nonuniform:constant:12",
    "_geometry": "_geometry:pec_wire:run_uniform:run_nonuniform:constant:12",
    "_lumped_rlc": "_lumped_rlc:series_RL:run_uniform:run_nonuniform:constant:12",
    "_materials": "_materials:mu:run_uniform:run_distributed:constant:12",
    "_msl_ports": "_msl_ports:msl_port:run_uniform:run_nonuniform:constant:12",
    "_ntff": "_ntff:ntff_box:run_uniform:run_nonuniform:constant:12",
    "_pec_faces": "_pec_faces:pec_face:run_nonuniform:run_distributed:graded:12",
    "_pinned_sheets": "_pinned_sheets:pec_sheet:run_uniform:run_nonuniform:constant:12",
    "_ports": "_ports:amplitude_kind:run_uniform:run_distributed:constant:12",
    "_probes": "_probes:probe:run_uniform:run_nonuniform:constant:12",
    "_tfsf": "_tfsf:plane_wave:run_uniform:run_nonuniform:constant:12",
    "_thin_conductors": "_thin_conductors:lossy_sheet:run_uniform:run_distributed:constant:12",
    "_waveguide_ports": "_waveguide_ports:waveguide_port:run_uniform:run_nonuniform:constant:12"
}


# Experimental ADI/subgrid refusals remain weekly unless selected as a cause witness.
PR_REFUSAL_LANES = frozenset((
    'run_uniform', 'fwd_uniform', 'run_nonuniform', 'fwd_nonuniform',
    'run_distributed', 'fwd_distributed_nu',
))


# Cheapest measured 12-step witness per cause (cold solve cost, including failures).
PR_FINDING_CHOICES = {
    "refusal-message-runs-adi-material-gate": '_geometry:pec_sheet:run_uniform:run_distributed:constant:12',
    "distributed-mode2d-broadcast": "_mode::run_uniform:run_distributed:constant:12",
    "flux-dA-shape": "_flux_monitors:flux:run_uniform:run_nonuniform:constant:12",
    "flux-dA2-missing": "_flux_monitors:flux:run_uniform:run_nonuniform:constant:12",
    "forward-flux-record-missing": "_flux_monitors:flux:fwd_uniform:fwd_nonuniform:constant:12",
    "graded-distributed-ntff-refused": "_ntff:ntff_box:run_nonuniform:run_distributed:graded:12",
    "graded-distributed-ports-refused": "_ports:wire_port:run_nonuniform:run_distributed:graded:12",
    "nu-dft-plane-accumulator": "_dft_planes:dft_plane:run_uniform:run_nonuniform:constant:12",
    "nu-flux-accumulator": "_flux_monitors:flux:run_uniform:run_nonuniform:constant:12",
    "nu-lumped-dft-record-missing": "_ports:lumped_port:fwd_uniform:fwd_nonuniform:constant:12",
    "nu-missing-vref": "_ports:wire_port:run_uniform:run_nonuniform:constant:12",
    "port-time-record-missing": "_ports:lumped_port:run_uniform:run_distributed:constant:12",
    "s-parameter-shape": "_ports:wire_port:fwd_uniform:fwd_nonuniform:constant:12",
    "uniform-wire-dft-record-missing": "_ports:wire_port:run_uniform:run_nonuniform:constant:12"
}


def pr_subset(cells, findings):
    """Main-path refusals, one strict witness per cause, and passing families."""
    chosen = {c.id for c in cells if not c.equivalence and c.refused in PR_REFUSAL_LANES}
    causes = {entry['cause'] for groups in findings.values()
              for entries in groups.values() for entry in entries}
    by_id = {c.id: c for c in cells}
    for cause in sorted(causes):
        assert cause in PR_FINDING_CHOICES, f'S0 needs PR finding cost selection: {cause}'
        identity = PR_FINDING_CHOICES[cause]
        assert identity in by_id and by_id[identity].steps == 12, (
            f'S0 stale PR finding cell: {cause}')
        assert any(entry['cause'] == cause for entries in findings.get(identity, {}).values()
                   for entry in entries), f'S0 stale PR finding selection: {cause}'
        chosen.add(identity)
    families = {c.row[0] for c in cells if c.equivalence}
    for family in sorted(families):
        candidates = [c for c in cells if c.equivalence and c.steps == 12
                      and c.row[0] == family and c.id not in findings]
        if not candidates:
            continue
        if PR_CHOICES:
            assert family in PR_CHOICES, f'S0 needs PR cost selection: {family}'
            identity = PR_CHOICES[family]
            assert identity in {c.id for c in candidates}, f'S0 stale PR selection: {family}'
        else:
            identity = candidates[0].id
        chosen.add(identity)
    return chosen

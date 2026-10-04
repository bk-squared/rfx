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


def pr_subset(cells, findings):
    """One equivalence per attribute family in declaration order, every
    refusal, and every cell with a strict expected-failure record."""
    chosen, families = set(), set()
    for cell in cells:
        if not cell.equivalence or cell.id in findings:
            chosen.add(cell.id)
        if cell.equivalence and cell.row[0] not in families:
            families.add(cell.row[0])
            chosen.add(cell.id)
    return chosen

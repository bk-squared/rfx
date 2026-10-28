"""Record-end coverage of returned frequency-domain accumulators."""
from dataclasses import dataclass
from itertools import product

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AccumulatorCoverage:
    """Static provenance survives forward JIT without string array leaves."""
    entries: tuple = ()

    def tree_flatten(self):
        return (), self.entries

    @classmethod
    def tree_unflatten(cls, entries, leaves):
        return cls(entries)

    def rows(self):
        return tuple(dict(zip(("name", "end_step", "n_record", "verdict"), row))
                     for row in self.entries)

    def max_end(self):
        ends = [row[1] for row in self.entries]
        return max(ends) if ends and None not in ends else None


def _corners(grid, spans):
    return tuple(tuple(grid.node_of(axis, int(i)) for axis, i in zip("xyz", corner))
                 for corner in product(*spans))


def accumulator_records(result):
    """Realized node corners, respecting each accumulator's slice convention."""
    grid = result.grid
    rows = []
    for attr, name in (("ntff_box", "NTFF box"),
                       ("current_moment_monitor", "current-moment monitor")):
        box = getattr(result, attr, None)
        if box is not None:
            spans = [(getattr(box, a + "_lo"), getattr(box, a + "_hi")) for a in "ijk"]
            if attr == "current_moment_monitor":
                spans[:2] = [(lo, hi - 1) for lo, hi in spans[:2]]
            rows.append((name, _corners(grid, spans)))
    for attr, label in (("dft_planes", "DFT plane"), ("flux_monitors", "flux monitor")):
        for name, plane in (getattr(result, attr, None) or {}).items():
            spans = [(0, size - 1) for size in grid.shape]
            spans[plane.axis] = (plane.index,)
            tangential = [a for a in range(3) if a != plane.axis]
            if attr == "dft_planes":
                if plane.region is not None:
                    lo1, hi1, lo2, hi2 = plane.region
                    spans[tangential[0]], spans[tangential[1]] = (lo1, hi1 - 1), (lo2, hi2 - 1)
            else:
                for axis, lo, hi in zip(tangential, (plane.lo1, plane.lo2), (plane.hi1, plane.hi2)):
                    spans[axis] = (lo, grid.shape[axis] - 1 if hi == -1 or hi == grid.shape[axis] else hi)
            rows.append((f"{label} {name}", _corners(grid, spans)))
    return tuple(rows)


def accumulator_coverage(result, *, sim=None, tfsf=None, n_steps=None):
    """Collect the same drives as the probe witness, at accumulator corners.

    The record length is the time series' length, else ``n_steps`` (a run
    that emits no time series, e.g. the graded multi-device forward with
    ``emit_time_series=False``); with neither, every row cannot judge.
    """
    from rfx.measurement.record_end import injection_end

    records = accumulator_records(result)
    if not records:
        return AccumulatorCoverage()
    series = getattr(result, "time_series", None)
    n_record = series.shape[0] if series is not None else n_steps
    if n_record is None:
        return AccumulatorCoverage(tuple(
            (name, None, None, "cannot judge: the record length is not available "
             "(no time series and no n_steps)") for name, _ in records))
    n_record = int(n_record)
    dt = getattr(result, "dt", None)
    if dt is None:
        dt = result.grid.dt
    entries = []
    for name, corners in records:
        end, reason, _ = injection_end(
            n_record, dt, records=corners, grid=result.grid, sim=sim, tfsf=tfsf,
            waveguide_ports=getattr(result, "waveguide_ports", None), return_detail=True)
        verdict = (reason if end is None else "covered" if 0 <= end < n_record else
                   f"cannot judge: the record ends at step {n_record} before the field leaves {name} (step {end})")
        entries.append((name, end, n_record, verdict))
    return AccumulatorCoverage(tuple(entries))


def with_coverage(db, witness, coverage):
    rows = coverage.rows()
    witness = {**witness, "coverage": rows}
    failed = next((r for r in rows if r["verdict"] != "covered"), None)
    if failed is not None and witness["status"] in {"pass", "absent"}:
        witness.update(status="undetermined", reason=failed["verdict"])
    return db, witness


def warn_uncovered(witness):
    """One warning for the first uncovered accumulator, before other advisories."""
    import warnings

    failed = next((r for r in witness.get("coverage", ()) if r["verdict"] != "covered"), None)
    if failed is None:
        return False
    db = witness.get("db")
    prefix = ("no ring-down settling witness; output is unguarded"
              if db is None or db != db else "record-end settling coverage")
    warnings.warn(f"{prefix}: {failed['name']}: {failed['verdict']} "
                  f"(record step {failed['n_record']}, end step {failed['end_step']})",
                  UserWarning, stacklevel=4)
    return True

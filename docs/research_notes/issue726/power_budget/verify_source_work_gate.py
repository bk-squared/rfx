"""Refute the local source-work gate with in-memory source mutations only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import jax
import tests.unit.ports.test_msl_source_work as gate


def main(out):
    if out.exists():
        raise FileExistsError(out)
    original = gate.make_msl_port_sources
    records = []
    for defect in ("detached_source_coefficient", "extra_sigma_source_factor",
                   "negative_source_force", "extra_source_outside_support"):
        def broken(*args, **kwargs):
            sources = original(*args, **kwargs)
            mats = args[2]
            if defect == "detached_source_coefficient":
                return [s._replace(waveform=jax.lax.stop_gradient(s.waveform)) for s in sources]
            if defect == "extra_sigma_source_factor":
                return [s._replace(waveform=s.waveform * mats.sigma[s.i, s.j, s.k]) for s in sources]
            if defect == "negative_source_force":
                return [s._replace(waveform=-s.waveform) for s in sources]
            return sources + [sources[0]._replace(i=0)]

        gate.make_msl_port_sources = broken
        try:
            gate.test_source_load_work_and_material_gradient("+x", True, True)
        except AssertionError as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            location = [frame for frame in frames if frame.filename == gate.__file__][-1]
            records.append(dict(defect=defect, rejected=True, assertion=str(exc),
                                gate_line=location.lineno, gate_expression=location.line))
        else:
            raise AssertionError(f"gate failed to reject {defect}")
        finally:
            gate.make_msl_port_sources = original
    record = dict(
        scope="In-memory source-function mutations; no production file changed. Local electric-substep checks only.",
        test_sha256=hashlib.sha256(Path(gate.__file__).read_bytes()).hexdigest(),
        cases=records,
    )
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)

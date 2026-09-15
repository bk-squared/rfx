#!/usr/bin/env python3
"""Three independent CPU lanes for the short witness retirement decision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_fixture_repair_evidence import junit_counts


def selection(lane):
    if lane == "policy":
        return ["tests/unit/sparams/test_sparam_passivity_guard.py",
                "tests/contracts/test_pec_short_advisory_geometry.py"]
    return ["tests/unit/sparams/test_pec_short_isolation.py::"
            f"test_pec_short_isolates_two_observable_ports[{lane}]"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("policy", "coarse", "fine"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert sha == os.environ["RFX_SHA"]
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    os.environ["RFX_SHORT_EVIDENCE_DIR"] = str(out)
    command = [sys.executable, "-m", "pytest", "-o", "addopts=",
               "--timeout=7200", "--timeout-method=thread", *selection(args.lane)]
    report = dict(source_sha=sha, lane=args.lane, commands=[],
                  run_id_source="submitter run_id.txt", started=time.time())
    rc = 1
    try:
        for label, extra in (
            ("collect", ["--collect-only", "-q"]),
            ("pytest", ["-n", "4" if args.lane == "policy" else "0", "-v",
                        "--capture=tee-sys", "-rA", "-o", "junit_logging=all",
                        "--junitxml=" + str(out / "junit.xml")]),
        ):
            cmd = command + extra
            print("RUN", label, cmd, flush=True)
            with (out / (label + ".log")).open("w") as log:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                rc = proc.wait()
            (out / (label + ".rc")).write_text(str(rc) + "\n")
            report["commands"].append(dict(name=label, command=cmd, returncode=rc))
            if rc:
                break
    finally:
        report.update(returncode=rc, finished=time.time(),
                      pytest=junit_counts(out / "junit.xml"))
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

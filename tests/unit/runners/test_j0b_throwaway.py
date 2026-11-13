"""Temporary one-device J0b observation; never an acceptance bar."""

import json
import os
import re
import subprocess
import sys


def _main():
    import inspect
    import time
    from unittest.mock import patch

    import jax
    import numpy as np

    from rfx.measurement import record_end
    from rfx.sparams import _msl_stages
    from tests.unit.runners._msl_multidevice_scene import FREQS, _json, build_scene

    def emit(kind, **data):
        print(json.dumps(_json(dict(kind=kind, **data)), allow_nan=False), flush=True)

    assert len(jax.devices()) == 1
    emit("metadata", purpose="J0b only; never a bar", devices=list(map(str, jax.devices())),
         freqs_hz=FREQS, lane="uniform", mode="laplace", excite_all=True)
    started = time.monotonic()
    original_score = _msl_stages._score_msl_record
    signature = inspect.signature(original_score)
    original_end = record_end.injection_end
    for scene in ("T1", "Y1"):
        for length in (1050, 1400, 2100, 2800):
            captures = {}

            def score(*args, **kwargs):
                bound = signature.bind(*args, **kwargs).arguments
                driven = bound["driven"]
                result = bound["_ts_result"]
                names = bound["ez_probe_names"][driven]
                records = result.dft_time_records
                peaks = {name: float(np.max(np.abs(np.asarray(records[name]))))
                         for name in names}
                capture = dict(
                    driven_port=f"p{driven + 1}",
                    realized_length=int(result.time_series.shape[0]),
                    plane_record_lengths={name: int(records[name].shape[0]) for name in names},
                    peak=max(peaks.values()), peak_kind="driven-port Ez ladder plane records",
                    plane_peaks=peaks, injection_end_calls=[],
                    excited_ports=[p.name for p in bound["variant"]._msl_ports if p.excite],
                )

                def spy(*end_args, **end_kwargs):
                    value = original_end(*end_args, **end_kwargs)
                    detail_kwargs = dict(end_kwargs, return_detail=True)
                    detail_value, reason, rows = original_end(*end_args, **detail_kwargs)
                    capture["injection_end_calls"].append(dict(
                        value=value, detail_value=detail_value, reason=reason, rows=rows))
                    return value

                with patch.object(record_end, "injection_end", spy):
                    scored = original_score(*args, **kwargs)
                captures[driven] = capture
                return scored

            emit("start", scene=scene, L=length)
            sim = build_scene(scene, mode="laplace", excite_all=True).sim
            with patch.object(_msl_stages, "_score_msl_record", score):
                s = sim.compute_msl_s_matrix(freqs=FREQS, n_steps=length)
            assert set(captures) == {0, 1}
            for driven in (0, 1):
                capture = captures[driven]
                witness = s.settling_witness[driven] if s.settling_witness is not None else None
                status = witness.get("status", "absent") if witness is not None else "absent"
                emit("witness", scene=scene, L=length, witness=witness, status=status,
                     abs_s_per_bin=np.abs(np.asarray(s.S)[:, driven, :]),
                     s_output_ports=["p1", "p2"], **capture)
                assert status in {"pass", "fail", "undetermined", "absent"}
                assert capture["realized_length"] == length
                assert all(n == length for n in capture["plane_record_lengths"].values())
                assert capture["peak"] > 0
                assert capture["excited_ports"] == [f"p{driven + 1}"]
                assert len(capture["injection_end_calls"]) == 1
                call = capture["injection_end_calls"][0]
                assert call["value"] == call["detail_value"]
            jax.clear_caches()
    emit("complete", wall_seconds=time.monotonic() - started)


def test_j0b():
    """Run in a fresh child with exactly one visible CPU device, as J0 did."""
    env = os.environ.copy()
    flags = re.sub(r"--xla_force_host_platform_device_count(?:=|\s+)\d+", "",
                   env.get("XLA_FLAGS", ""))
    env["XLA_FLAGS"] = flags + " --xla_force_host_platform_device_count=1"
    env["JAX_PLATFORMS"] = "cpu"
    code = "from tests.unit.runners.test_j0b_throwaway import _main; _main()"
    subprocess.run([sys.executable, "-u", "-c", code], env=env, check=True)

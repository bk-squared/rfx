"""One isolated real-GPU case of the tracker 1598 entry-point matrix."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tests._multi_device_scene import finite, scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    devices = jax.devices("gpu")[:args.devices]
    assert len(devices) == args.devices
    assert all(d.platform == "gpu" for d in devices)
    print(json.dumps({"jax": jax.__version__, "devices": [str(d) for d in devices],
                      "platforms": [d.platform for d in devices]}), flush=True)
    name = args.case.removeprefix("jit_").removeprefix("off_")
    options = {"checkpoint_every": 2} if name == "checkpoint" else {}
    eps, forward, objective, run = scene(devices, n=48, steps=30,
                                         uniform=name == "uniform_run", **options)
    arg = eps
    if name == "uniform_run":
        call = lambda e: run()
    elif name == "vmap":
        call = jax.vmap(forward)
        arg = jnp.stack((eps, eps * 1.01))
    elif name == "forward":
        call = forward
    else:
        call = jax.grad(objective)
    if args.case.startswith("jit_"):
        from rfx.stepping import rank
        reached_rank = []
        original_multi_device = rank._is_multi_device

        def observe_mesh(mesh):
            reached_rank.append(True)
            return original_multi_device(mesh)

        rank._is_multi_device = observe_mesh
        try:
            jax.jit(call)(arg)
        except NotImplementedError as exc:
            assert "1598" in str(exc) and "xla_gpu_enable_command_buffer" in str(exc)
            print(f"REFUSED: {exc}", flush=True)
            return
        except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as exc:
            if name != "uniform_run" or reached_rank:
                raise
            print(f"NOT_REACHABLE: {type(exc).__name__}: {exc}", flush=True)
            return
        raise AssertionError("enclosing jit was not refused")
    if args.case.startswith("off_"):
        call = jax.jit(call)
    result = call(arg)
    finite(result)
    np.save(args.out, np.asarray(result))
    print(f"RAN: {args.case}; finite", flush=True)


if __name__ == "__main__":
    main()

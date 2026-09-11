import os,sys,json,time,hashlib,subprocess
from pathlib import Path
out=Path(os.environ["RFX_OUT"])
if sys.argv[1] == "ready":
    import jax,numpy,scipy,rfx
    assert jax.default_backend() == "gpu"
    assert Path(rfx.__file__).resolve().is_relative_to(Path.cwd())
    sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert sha == os.environ["RFX_SHA"]
    receipt=dict(sha=sha,jax=jax.__version__,numpy=numpy.__version__,scipy=scipy.__version__,devices=str(jax.devices()),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(receipt,flush=True)
    (out/"environment.json").write_text(json.dumps(receipt,indent=2))
    (out/"ready").write_text("GPU ready\n")
    for _ in range(300):
        if (out/"proceed").exists():break
        time.sleep(1)
    else:raise SystemExit("handoff was not received")
    raise SystemExit(0)
import numpy as np
import tests.locks.test_patch_edgefed_resonance_harminv as patch
original=patch._census;saved=[]
def capture(signals,dt):
    name=("unfed","fed")[len(saved)]
    np.savez_compressed(out/(name+"-input.npz"),signals=np.stack(signals),dt=np.asarray(dt))
    result=original(signals,dt)
    saved.append(dict(arm=name,dt=float(dt),n_samples=len(signals[0]),census=result))
    (out/"census.json").write_text(json.dumps(saved,indent=2)+"\n")
    return result
patch._census=capture
import pytest
raise SystemExit(pytest.main(["tests/locks/test_patch_edgefed_resonance_harminv.py","-o","addopts=","-v","-s","-p","no:cacheprovider","--junitxml="+str(out/"junit.xml")]))

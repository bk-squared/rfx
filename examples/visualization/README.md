# Field movies from actual simulations

`field_movies.py` builds three small introductory examples with the public
`Simulation` API and renders the saved `Ez` fields with Matplotlib/ffmpeg.
They are qualitative single-mesh 2D TMz demonstrations, not accuracy benchmarks.

| Case | What to look for |
|---|---|
| `boundary-reflection` | The same pulse exits a CPML domain and returns from PEC walls. |
| `cavity-standing-wave` | A carrier-centered pulse excites a standing field inside a closed PEC rectangle. |
| `dielectric-interface` | A pulse crosses from vacuum into an ideal dielectric, compared with a vacuum reference. |

From the source checkout, install RFX with `python -m pip install -e .` and
install ffmpeg with your operating system's package manager. Then run:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python examples/visualization/field_movies.py \
  --case all --output /tmp/rfx-field-movies
```

The output directory receives a complete NPZ of actual field frames and their
physical coordinate/time arrays, probe CSV, poster, MP4, and model manifest.
Preflight output is visible. The dielectric example intentionally retains
resolution, ideal-lossless-material, and half-space/absorber advisories; it
makes no phase-accuracy, reflection coefficient, or Q claim.

Use `--render-only` with the same output directory to redraw without solving.
`--publish-to PATH` copies the video/poster/CSV, generator, and a compact NPZ
containing every 16th actual movie frame at full spatial resolution, with all
probe samples. Every published artifact is required to stay below 1 MB. A
publication directory is not a replacement for the full rendering archive.

All frame positions/times come from `Result.snapshot_axes`. There is no frame
interpolation or phasor replay. Compared models share one fixed linear color
scale across all frames. The scale excludes the 4 mm current-source
neighborhood; source pixels may saturate, while raw values are retained.

The manifest names solver commit `a9054338` only if the loaded solver files
match it. Otherwise that field is `null`; the script never silently labels a
changed solver as the original one. The recorded generator SHA-256 identifies
the rendering and model recipe. The original gallery runs used JAX 0.6.2 on CPU.

Published walkthroughs: [boundary reflection](https://remilab.ai/rfx/gallery/boundary-reflection/),
[cavity standing wave](https://remilab.ai/rfx/gallery/cavity-standing-wave/), and
[dielectric interface](https://remilab.ai/rfx/gallery/dielectric-interface/).

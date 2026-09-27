# Curated historical RFX visualization clips

Six small H.264 clips reused from their existing public share URLs. The
`media-catalog.json` records those URLs, SHA-256 hashes, byte counts, animation
types and limitations. Video bytes are preserved; JPEG posters are frames
extracted from the listed times with FFmpeg (`-frames:v 1 -q:v 2`).

This folder contains only selected public media. It does not contain a slide
deck, private notes, run logs, credentials or raw simulation arrays. The source
revision that generated the historical results is not established; related
versioned research-example links are not reproduction certificates.

To verify the local bundle, run `python scripts/check_public_docs_manifest.py`
from the repository root. The public pages label the notch field as a phasor
replay, the coarse notch trajectory as interpolation and the steering trajectory
as a latent-variable design. Numerical values embedded in the films retain
their historical scope.

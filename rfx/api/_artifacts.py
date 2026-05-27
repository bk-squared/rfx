"""Thin artifact-export mixin for :class:`rfx.api.Simulation`.

This module must remain a leaf/transitional mixin: do not import the package
root or ``rfx.api`` here.  Policy lives in ``rfx.artifacts`` and is imported
inside methods to keep the high-level API wiring additive and cycle-safe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class _ArtifactsMixin:
    """Simulation convenience methods for runtime artifact export."""

    def export_scene(self, path: str | Path | None = None, **kwargs: Any):
        """Return or write the native runtime scene artifact.

        ``path=None`` returns a JSON-serializable dict.  Otherwise the scene is
        written as indented JSON and the output :class:`~pathlib.Path` is
        returned.
        """

        from rfx.artifacts import build_scene_artifact

        scene = build_scene_artifact(self, **kwargs)
        if path is None:
            return scene
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scene, indent=2, sort_keys=True, default=str) + "\n")
        return out

    def artifact_report(self, result=None, **kwargs: Any):
        """Return a runtime artifact report dict for this simulation."""

        from rfx.artifacts import build_runtime_report

        return build_runtime_report(self, result=result, **kwargs)

    def export_artifact_bundle(self, path: str | Path, result=None, **kwargs: Any):
        """Write a runtime artifact bundle and return ``ArtifactBundle``."""

        from rfx.artifacts import export_artifact_bundle

        return export_artifact_bundle(path, self, result=result, **kwargs)

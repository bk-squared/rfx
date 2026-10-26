"""Shared pre-landing policy for repository changelog contracts."""

import os


def allow_topic_fragments() -> bool:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event == "pull_request":
        return not os.environ.get("GITHUB_BASE_REF", "").startswith("release")
    return not event and os.environ.get("RFX_CHANGELOG_CONTEXT") == "pre-landing"

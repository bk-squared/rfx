"""Classification data consumed by the traced-input admission mechanism."""

from collections import Counter

ROWS = ()
VALUE_TYPES = {}
DEFAULT_PARAMETERS = {}


def state_counts():
    """Count expanded parameters; explicit rows override a method default."""
    expanded = {}
    for owner, parameter, state, _ in ROWS:
        if parameter == "*":
            for name in DEFAULT_PARAMETERS[owner]:
                expanded[owner, name] = state
    for owner, parameter, state, _ in ROWS:
        if parameter != "*":
            expanded[owner, parameter] = state
    counts = dict.fromkeys(("traceable", "value_only", "static", "undecided", "no_number", "forwards"), 0)
    counts.update(Counter(state.split(":", 1)[0] for state in expanded.values()))
    return counts

"""MSL template rendering with SI values retained before display-unit rounding."""

from numbers import Real
from string import Formatter


class Message(str):
    """Rendered table fragments compose without discarding their observations."""

    def __new__(cls, text, values=(), templates=()):
        result = super().__new__(cls, text)
        result.values = dict(values)
        result.templates = tuple(templates)
        return result

    def __add__(self, other):
        return Message(
            str(self) + str(other),
            {**self.values, **getattr(other, "values", {})},
            self.templates + getattr(other, "templates", ()),
        )

    def __radd__(self, other):
        return Message(
            str(other) + str(self),
            {**getattr(other, "values", {}), **self.values},
            getattr(other, "templates", ()) + self.templates,
        )


class DisplayedValue(float):
    """An SI scalar retaining main's arithmetic before display rounding."""

    def __new__(cls, si_value, displayed):
        result = super().__new__(cls, si_value)
        result.displayed = displayed
        return result


class SIFormatter(Formatter):
    def format_field(self, value, spec):
        if isinstance(value, DisplayedValue):
            return format(value.displayed, spec.split("|", 1)[-1])
        if spec.startswith("GHz|"):
            return format(value / 1e9, spec.split("|", 1)[1])
        if "|" not in spec:
            return super().format_field(value, spec)
        unit, spec = spec.split("|", 1)
        if unit == "len":
            from rfx.preflight._common import _fmt_len

            return _fmt_len(value)
        if unit == "mm_tuple":
            return str(tuple(round(c * 1e3, 2) for c in value))
        return format(
            value * {"um": 1e6, "mm": 1e3, "percent": 100, "GHz": 1e-9}[unit], spec
        )


def render(key, template, fields):
    values = {}
    templates = [key]
    for name, value in fields.items():
        if isinstance(value, Message):
            values.update(value.values)
            templates.extend(value.templates)
        elif hasattr(value, "diagnostic"):
            values.update(value.diagnostic.values)
        elif isinstance(value, Real):
            values[name] = value
        elif isinstance(value, (tuple, list)):
            values.update(
                (f"{name}_{index}", item)
                for index, item in enumerate(value)
                if isinstance(item, Real)
            )
        elif isinstance(value, str):
            values[name] = value
    result = Message(SIFormatter().vformat(template, (), fields), values, templates)
    if template == "{detail}" and hasattr(fields["detail"], "diagnostic"):
        result.diagnostic = fields["detail"].diagnostic
    return result


def join_messages(separator, parts, *, indexed=False):
    result = Message("")
    for index, part in enumerate(parts):
        if indexed and isinstance(part, Message):
            part = Message(
                part,
                {f"port_{index}_{k}": v for k, v in part.values.items()},
                part.templates,
            )
        result = result + (separator if index else "") + part
    return result

from math import isfinite


class CatalogQueryError(RuntimeError):
    """A catalog request failed after its configured retry attempts."""


def finite_or_none(value):
    """Return a finite catalog scalar or None without coercing masked values."""
    try:
        if bool(getattr(value, 'mask', False)):
            return None
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if isfinite(converted) else None

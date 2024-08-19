import functools
from typing import Any


def rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """Recursively gets an attribute from an object.

    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427

    Args:
        obj: The object to retrieve the attribute from.
        attr: The attribute to retrieve. Can be a nested attribute separated by dots.
        *args: Optional default values to return if the attribute is not found.

    Returns:
        The value of the attribute if found, otherwise the default value(s) specified by *args.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

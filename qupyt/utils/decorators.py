"""Provide decorators shared by QuPyt hardware and measurement code."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Any, ParamSpec, TypeVar, cast

from numpy.typing import NDArray

P = ParamSpec("P")
R = TypeVar("R")


ArrayResult = TypeVar("ArrayResult", bound=NDArray[Any])


def log_sensor_acquisition(func: Callable[P, ArrayResult]) -> Callable[P, ArrayResult]:
    """Record acquisition details at the sensor boundary, including direct calls."""

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> ArrayResult:
        logging.debug(
            "Starting %s acquisition: measurements=%s",
            type(self).__name__,
            self.number_measurements,
        )
        started = perf_counter()
        try:
            data = func(self, *args, **kwargs)
        except Exception:
            logging.exception(
                "%s acquisition failed after %.3fs",
                type(self).__name__,
                perf_counter() - started,
            )
            raise
        logging.debug(
            "Acquired data: sensor=%s, shape=%s, dtype=%s, elapsed=%.3fs",
            type(self).__name__,
            data.shape,
            data.dtype,
            perf_counter() - started,
        )
        return data

    return cast(Callable[P, ArrayResult], wrapper)


def coerce_device_config_shape(func: Callable[P, R]) -> Callable[P, R]:
    """Normalize device configuration inputs to a common channel/value shape.

    Parameters
    ----------
    func : Callable
        Device setter receiving a list of channel/value pairs.

    Returns
    -------
    Callable
        Wrapped setter that accepts a scalar, a channel/value tuple, or a list.
        A scalar uses channel_1; a tuple becomes a one-element list.

    Raises
    ------
    ValueError
        If the input is not a scalar, tuple, or list.
    """
    adapted_func = cast(Callable[[Any, Any], R], func)

    @wraps(func)
    def wrapper(self: Any, arg: Any) -> R:
        if isinstance(arg, list):
            return adapted_func(self, arg)
        if isinstance(arg, tuple):
            return adapted_func(self, [arg])
        if isinstance(arg, (float, int, str)):
            return adapted_func(self, [("channel_1", arg)])
        raise ValueError(
            f"Expected a scalar, channel/value tuple, or list of pairs; got {arg!r}"
        )

    return cast(Callable[P, R], wrapper)


def loop_inputs(func: Callable[P, None]) -> Callable[P, None]:
    adapted_func = cast(Callable[[Any, tuple[str, Any]], None], func)

    @wraps(func)
    def wrapper(self: Any, arg: Any) -> None:
        for inp in arg:
            if (
                not isinstance(inp, (tuple, list))
                or len(inp) != 2
                or not isinstance(inp[0], str)
            ):
                raise ValueError(f"Expected a (channel, value) pair; got {inp!r}")
            channel = inp[0].split("_")[-1]
            value = inp[1]
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
            adapted_func(self, (channel, value))

    return cast(Callable[P, None], wrapper)

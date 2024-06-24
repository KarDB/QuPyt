"""
Contains all decorators used across QuPyt
"""

from functools import wraps


def coerce_device_config_shape(func):
    """
    Coerces specific configuration input
    to QuPyt devices to adhere to a unified shape.
    User configuration can thus be kept simple.
    Their shape is adapted in this decorator.

    Raises:
        ValueError
    """

    @wraps(func)
    def wrapper(self, arg):
        if isinstance(arg, list):
            return func(self, arg)
        if isinstance(arg, tuple):
            return func(self, [arg])
        if isinstance(arg, (float, int)):
            return func(self, [("channel_1", arg)])
        raise ValueError

    return wrapper


def loop_inputs(func):
    @wraps(func)
    def wrapper(self, arg):
        for inp in arg:
            channel = inp[0].split("_")[-1]
            if isinstance(inp[1], str):
                try:
                    inp[1] = float(inp[1])
                except:
                    pass
            func(self, (channel, inp[1]))

    return wrapper

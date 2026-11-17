"""Cleanup helpers that preserve the failure which triggered recovery."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


def record_failure(
    params: dict[str, Any], error: BaseException, stage: str, **context: Any
) -> dict[str, Any]:
    """Describe a failure in metadata while retaining the first recorded cause."""
    detail = {
        "type": type(error).__name__,
        "message": str(error),
        "stage": stage,
        **context,
    }
    params.setdefault("measurement_error", detail)
    return detail


def cleanup_after_error(callback: Callable[[], object], description: str) -> None:
    """Attempt secondary cleanup without replacing an active exception."""
    try:
        callback()
    except (Exception, SystemExit):
        logging.warning(
            "Additional error during %s; original failure retained",
            description,
            exc_info=True,
        )


def cleanup_all(operations: Iterable[tuple[str, Callable[[], object]]]) -> None:
    """Attempt every cleanup operation and then propagate the first failure."""
    first_error: BaseException | None = None
    first_traceback: TracebackType | None = None
    for description, callback in operations:
        try:
            callback()
        except (Exception, SystemExit) as error:
            logging.exception("Failed during %s", description)
            if first_error is None:
                first_error, first_traceback = error, error.__traceback__
    if first_error is not None:
        raise first_error.with_traceback(first_traceback)


@contextmanager
def cleanup_on_exit(callback: Callable[[], object], description: str) -> Iterator[None]:
    """Clean up on exit and propagate cleanup errors when the operation succeeded."""
    try:
        yield
    except BaseException:
        cleanup_after_error(callback, description)
        raise
    else:
        callback()


def close_on_initialization_error(func: Callable[P, R]) -> Callable[P, R]:
    """Release resources acquired by a constructor if later setup fails."""

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> R:
        # Derived constructors may call another decorated constructor. The
        # outermost call owns recovery so the same resource is not closed twice.
        if getattr(self, "_initialization_cleanup_active", False):
            return func(self, *args, **kwargs)
        self._initialization_cleanup_active = True
        try:
            return func(self, *args, **kwargs)
        except BaseException:
            logging.exception("%s initialization failed", type(self).__name__)
            cleanup_after_error(
                self.close, f"{type(self).__name__} initialization cleanup"
            )
            raise
        finally:
            self._initialization_cleanup_active = False

    return cast(Callable[P, R], wrapper)

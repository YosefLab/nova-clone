"""
Shared logging for the algorithm package.

Use ``set_log_level`` to control verbosity. To silence all algorithm logs:

    import logging
    from nova_clone.algorithm import set_log_level
    set_log_level(logging.WARNING)  # or logging.ERROR / logging.CRITICAL

Or pass ``verbose=False`` to ``clone_calling_algorithm``.
"""

import logging
import sys
from contextlib import contextmanager
from typing import Iterator

_LOGGER_NAME = "nova_clone.algorithm"


def _init_logger() -> None:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers on re-import (e.g. in notebooks).
    if logger.handlers:
        return
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Child loggers (if any) should not also hit the root logger.
    logger.propagate = False


_init_logger()
logger = logging.getLogger(_LOGGER_NAME)


def set_log_level(log_level: int) -> None:
    """Set the log level for all algorithm-package log messages."""
    logging.getLogger(_LOGGER_NAME).setLevel(level=log_level)


@contextmanager
def logging_verbosity(verbose: bool) -> Iterator[None]:
    """Temporarily silence algorithm logs when ``verbose`` is False."""
    if verbose:
        yield
        return
    previous_level = logger.level
    set_log_level(logging.CRITICAL)
    try:
        yield
    finally:
        set_log_level(previous_level)

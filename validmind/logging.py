# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""ValidMind logging module"""

import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

import sentry_sdk
from sentry_sdk.utils import event_from_exception, exc_info_from_error

from .__version__ import __version__

__dsn = "https://48f446843657444aa1e2c0d716ef864b@o1241367.ingest.sentry.io/4505239625465856"


def _get_log_level() -> int:
    """Get the log level from the environment variable."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

    if log_level_str not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid log level: {log_level_str}")

    return logging.getLevelName(log_level_str)


def get_logger(
    name: str = "validmind", log_level: Optional[int] = None
) -> logging.Logger:
    """Get a logger for the given module name."""
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s(%(name)s): %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(log_level or _get_log_level())

    # Clear existing handlers if any (or refine the existing logic as necessary)
    # TODO: move this to a yaml config and only configure once
    if not any(
        isinstance(h, type(handler)) and h.formatter._fmt == formatter._fmt
        for h in logger.handlers
    ):
        logger.addHandler(handler)

    # Prevent logger from propagating to root logger
    logger.propagate = False

    return logger


def init_sentry(server_config: Dict[str, Any]) -> None:
    """Initialize Sentry SDK for sending logs back to ValidMind.

    This will usually only be called by the API client module to initialize the
    Sentry connection after the user calls `validmind.init()`. This is because the DSN
    and other config options will be returned by the API.

    Args:
        server_config (Dict[str, Any]): The config dictionary returned by the API.
            - send_logs (bool): Whether to send logs to Sentry (gets removed).
            - dsn (str): The Sentry DSN.
            ...: Other config options for Sentry.

    Returns:
        None.
    """
    if os.getenv("VM_NO_TELEMETRY", False):
        return

    if not server_config.get("send_logs", False):
        return

    config = {
        "dsn": __dsn,
        "traces_sample_rate": 1.0,
        "release": f"validmind-python@{__version__}",
        "in_app_include": ["validmind"],
        "environment": "production",
    }
    config.update({k: v for k, v in server_config.items() if k != "send_logs"})

    try:
        sentry_sdk.init(**config)
    except Exception as e:
        logger = get_logger(__name__)
        logger.info("Sentry failed to initialize - ignoring...")
        logger.debug(f"Sentry error: {str(e)}")


F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Awaitable[Any]])


def log_performance(
    name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> Callable[[F], F]:
    """Decorator to log the time it takes to run a function.

    Args:
        name (str, optional): The name of the function. Defaults to None.
        logger (logging.Logger, optional): The logger to use. Defaults to None.
        force (bool, optional): Whether to force logging even if env var is off.

    Returns:
        Callable: The decorated function.
    """

    def decorator(func: F) -> F:
        # check if log level is set to debug
        if _get_log_level() != logging.DEBUG and not force:
            return func

        nonlocal logger
        if logger is None:
            logger = get_logger()

        nonlocal name
        if name is None:
            name = func.__name__

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            time1 = time.perf_counter()
            return_val = func(*args, **kwargs)
            time2 = time.perf_counter()

            logger.debug("%s function took %0.3f ms" % (name, (time2 - time1) * 1000.0))

            return return_val

        return wrapped

    return decorator


async def log_performance_async(
    func: AF,
    name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> AF:
    """Async version of log_performance decorator"""
    # check if log level is set to debug
    if _get_log_level() != logging.DEBUG and not force:
        return func

    if logger is None:
        logger = get_logger()

    if name is None:
        name = func.__name__

    async def wrap(*args: Any, **kwargs: Any) -> Any:
        time1 = time.perf_counter()
        return_val = await func(*args, **kwargs)
        time2 = time.perf_counter()

        logger.info("%s function took %0.3f ms" % (name, (time2 - time1) * 1000.0))

        return return_val

    return wrap


def log_api_operation(
    operation_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    extract_key: Optional[Callable] = None,
    force: bool = False,
) -> Callable[[F], F]:
    """Decorator to log API operations like figure uploads.

    Args:
        operation_name (str, optional): The name of the operation. Defaults to function name.
        logger (logging.Logger, optional): The logger to use. Defaults to None.
        extract_key (Callable, optional): Function to extract a key from args for logging.
        force (bool, optional): Whether to force logging even if env var is off.

    Returns:
        Callable: The decorated function.
    """

    def decorator(func: F) -> F:
        # check if log level is set to debug
        if _get_log_level() != logging.DEBUG and not force:
            return func

        nonlocal logger
        if logger is None:
            logger = get_logger()

        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Try to extract a meaningful identifier from the arguments
            identifier = ""
            if extract_key and args:
                try:
                    identifier = f": {extract_key(args[0])}"
                except (AttributeError, IndexError):
                    pass

            logger.debug(f"{operation_name}{identifier}")
            return await func(*args, **kwargs)

        return wrapped

    return decorator


def send_single_error(error: Exception) -> None:
    """Send a single error to Sentry.

    Args:
        error (Exception): The exception to send.
    """
    event, hint = event_from_exception(exc_info_from_error(error))
    client = sentry_sdk.Client(__dsn, release=f"validmind-python@{__version__}")
    client.capture_event(event, hint=hint)

    time.sleep(0.25)  # wait for the event to be sent

from __future__ import annotations

import functools
import time

from typing import Callable, TypeVar, Optional
from typing_extensions import ParamSpec

import logging

logger = logging.getLogger(__name__)

_FuncParams = ParamSpec("_FuncParams")
_FuncValue = TypeVar("_FuncValue")


def typed_cache(
    user_function: Callable[_FuncParams, _FuncValue]
) -> Callable[_FuncParams, _FuncValue]:
    return functools.cache(user_function)


def timeit(func: Callable[_FuncParams, _FuncValue]) -> Callable[_FuncParams, _FuncValue]:
    """
    Decorator that measures execution time of a method. This is a modified version from the one published at:
    https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    """

    @functools.wraps(func)
    def inner(*args: _FuncParams.args, **kwargs: _FuncParams.kwargs) -> _FuncValue:
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        logger.warning(
            f"{func.__name__}|Execution time: {te - ts:.2f}s"
        )
        return result

    return inner


def log_calling_args(func: Callable[_FuncParams, _FuncValue]) -> Callable[_FuncParams, _FuncValue]:

    @functools.wraps(func)
    def wrapper(*args: _FuncParams.args, **kwargs: _FuncParams.kwargs) -> _FuncValue:
        logger.debug(
            f"Calling method '{func.__name__}' with args={args} and kwargs={kwargs}"
        )
        return func(*args, **kwargs)

    return wrapper


def catch_exception_return_none(func: Callable[_FuncParams, _FuncValue]) -> Callable[_FuncParams, Optional[_FuncValue]]:

    @functools.wraps(func)
    def wrapper(*args: _FuncParams.args, **kwargs: _FuncParams.kwargs) -> Optional[_FuncValue]:
        logger.debug(
            f"Calling method '{func.__name__}' with args={args} and kwargs={kwargs}"
        )
        try:
            return func(*args, **kwargs)
        except:
            logger.exception(
                f"The function {func.__name__} raised an exception. "
                f"Debug info:"
                f"\n\t* {args=}"
                f"\n\t* {kwargs=}"
                f""
            )
            return None

    return wrapper

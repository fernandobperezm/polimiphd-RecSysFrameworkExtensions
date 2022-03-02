import logging
import os
import sys

import attr
import toml


def _convert_dir_name(
    dir_name: str
) -> str:
    return os.path.join(
        os.getcwd(),
        dir_name,
        "",
    )


@attr.s(frozen=True, kw_only=True)
class LoggingConfig:
    formatter: logging.Formatter = attr.ib(
        converter=logging.Formatter
    )
    dir_name: str = attr.ib(
        converter=_convert_dir_name,
    )
    file_name: str = attr.ib()

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self,
            "file_name",
            os.path.join(
                self.dir_name,
                self.file_name,
            )
        )


def _load_logger_config() -> LoggingConfig:
    with open(os.path.join(os.getcwd(), "pyproject.toml"), "r") as project_file:
        config = toml.load(
            f=project_file
        )
    logs_config = LoggingConfig(
        **config["logs"]
    )
    return logs_config


_LOGS_CONFIG = _load_logger_config()


def get_file_handler(
    filename: str,
    formatter: logging.Formatter
) -> logging.Handler:
    file_handler = logging.FileHandler(
        filename=filename,
        mode="a"
    )
    file_handler.setFormatter(
        fmt=formatter
    )
    return file_handler


def get_console_out_handler(
    formatter: logging.Formatter,
) -> logging.Handler:
    console_out_handler = logging.StreamHandler(
        stream=sys.stdout
    )
    console_out_handler.setFormatter(
        fmt=formatter
    )
    return console_out_handler


def get_console_error_handler(
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler:
    console_err_handler = logging.StreamHandler(
        stream=sys.stderr
    )
    console_err_handler.setFormatter(
        fmt=formatter,
    )
    console_err_handler.setLevel(
        level=level,
    )
    return console_err_handler


def get_logger(
    logger_name: str,
) -> logging.Logger:
    os.makedirs(
        name=_LOGS_CONFIG.dir_name,
        exist_ok=True,
    )

    logger = logging.getLogger(
        name=logger_name
    )
    logger.addHandler(
        get_file_handler(
            filename=_LOGS_CONFIG.file_name,
            formatter=_LOGS_CONFIG.formatter,
        )
    )
    logger.addHandler(
        get_console_out_handler(
            formatter=_LOGS_CONFIG.formatter,
        )
    )
    logger.addHandler(
        get_console_error_handler(
            level=logging.ERROR,
            formatter=_LOGS_CONFIG.formatter,
        )
    )
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger

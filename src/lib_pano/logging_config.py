import logging
import os


_DEFAULT_FMT = '[%(asctime)s] %(levelname)s | %(message)s'
_DEFAULT_DATEFMT = '%H:%M:%S'


def _normalize_level(level):
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        lv = level.strip().upper()
        if lv.isdigit():
            return int(lv)
        return getattr(logging, lv, logging.INFO)
    return logging.INFO


def configure_logging(level=None, fmt: str = _DEFAULT_FMT, datefmt: str = _DEFAULT_DATEFMT, force: bool = True):
    """Configure root logger once from a single entry point.

    Priority: explicit level arg > PANO_LOG_LEVEL env > INFO.
    """
    resolved = level if level is not None else os.getenv('PANO_LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=_normalize_level(resolved),
        format=fmt,
        datefmt=datefmt,
        force=force,
    )

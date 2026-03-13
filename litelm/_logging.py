"""Logging shim — DSPy configures litellm's verbose_logger at import time."""
import logging

verbose_logger = logging.getLogger("litelm.verbose")
verbose_logger.addHandler(logging.NullHandler())

"""Logging utility functions."""

import logging


def configure_logging_verbosity(verbose):
    """Set verbosity level.

    Args:
        verbose (bool): True for setting verbosity.

    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

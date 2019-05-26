"""Utility methods for the argparse functionality."""
import argparse


def string_to_bool(source):
    """Translate string to bools.

    [yes, true t, y, 1] are interpreted as True, whereas [no, false , f, n, 0] as False. No sensitive case.
    Credits go to: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954

    Args:
        source (str): Valid inputs are [('yes', 'true', 't', 'y', '1', 'no', 'false', 'f', 'n', '0']

    Returns:
        bool: Translated string.

    Raises:
        ArgumentTypeError: String was not valid input.

    """
    if source.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif source.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
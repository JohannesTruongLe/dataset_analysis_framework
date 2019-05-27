"""Utility methods for the argparse functionality."""
import argparse


def string_to_bool(source):
    """Translate string to bools.

    [yes, true t, y, 1] are interpreted as True, whereas [no, false , f, n, 0] as False. No
    sensitive case. Credits go to:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954

    Args:
        source (str): Valid inputs are [('yes', 'true', 't', 'y', '1', 'no', 'false', 'f', 'n', '0']

    Returns:
        bool: Translated string.

    Raises:
        ArgumentTypeError: String was not valid input.

    """
    if source.lower() in ('yes', 'true', 't', 'y', '1'):
        output = True
    elif source.lower() in ('no', 'false', 'f', 'n', '0'):
        output = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    return output

def default_config_parse(default_config_path):
    """Parse inline commands.

    Args:
        default_config_path (str): Path to default config.

    Returns:
        argparse.Namespace: Container holding
            config (dict): Dictionary holding configs
            verbose (bool): Bool value, which can be used for setting verbosity

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to config file.",
                        type=str,
                        default=default_config_path)
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args

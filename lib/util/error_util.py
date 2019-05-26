"""Error util functions."""


def check_instance_type(source, target):
    """Check if source and target are of same type.

    Adds nice Error message.

    Args:
        source: See isinstance() first input.
        target: See isinstance() second input.

    Raises:
        AttributeError: If type(source) != source(target)

    """
    if not isinstance(source, target):
        raise AttributeError("Wrong input type. It is %s, but should be %s" %
                             (source, target))

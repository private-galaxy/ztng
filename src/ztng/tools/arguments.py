"""Check the keyword arguments of a function."""


def updateKwargs(default_kwargs, input_kwargs):
    """Merge the keyword arguments, and name the ones that do not exist."""
    bad_keys = [key for key in input_kwargs if key not in default_kwargs]
    if bad_keys:
        raise TypeError(
            f'Invalid arguments: {bad_keys}. '
            f'Valid arguments: {list(default_kwargs)}.')
    return {**default_kwargs, **input_kwargs}

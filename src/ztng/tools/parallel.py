"""Run one function over many inputs, optionally in parallel."""

from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm


def parallelFunction(
        function, inputs,
        parallel=True, n_jobs=-1, backend='loky', verbose=10):
    """Apply function to every input and return the outputs in order.
    A tuple or a list input is unpacked as positional arguments, anything
    else is passed as the single argument.
    """

    def wrapper(input):
        if isinstance(input, (tuple, list)):
            return function(*input)
        return function(input)

    if parallel:
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            p = Parallel(verbose=verbose, return_as='generator')
            generator = p(delayed(wrapper)(input) for input in inputs)
            outputs = list(tqdm(
                generator, total=len(inputs), disable=(verbose > 0.)))
    else:
        outputs = [wrapper(input) for input in tqdm(inputs, total=len(inputs))]

    return outputs

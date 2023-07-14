import logging
from typing import Callable

import numpy as np


def sampler(
    n_samples,
    nq,
    random_generator: Callable[[], np.ndarray],
    accept_function: Callable[[np.ndarray], bool],
    max_tries: int,
):
    samples = np.empty(shape=(n_samples, nq), dtype=float)
    n_tries = 0
    n_collected = 0
    while n_collected < n_samples:
        if max_tries > 0 and n_tries > max_tries:  # noqa
            break
        n_tries += 1
        q = random_generator()
        if accept_function(q):
            samples[n_collected, :] = q
            n_collected += 1
    logging.debug(f"Tried {n_tries} initial states, Collected {n_collected}")
    return samples

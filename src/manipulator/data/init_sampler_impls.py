import logging

import numpy as np

from manipulator.model import ManipulatorDynamics
from manipulator.data.init_sampler import sampler
from manipulator.data.utils import get_is_collided_function


def sample_initial_states_simple(
    arm: ManipulatorDynamics,
    n_samples: int,
    sample_radius: float,
    sample_center: np.ndarray,
    max_tries: int = 0,
) -> np.ndarray:
    def random_generator() -> np.ndarray:
        variation = np.random.rand(len(sample_center))
        variation = 2.0 * variation - 1.0
        variation = variation / np.linalg.norm(variation)
        variation = variation * sample_radius
        return variation + sample_center

    is_collided = get_is_collided_function(arm)

    def accept_function(q_sample) -> bool:
        if is_collided(q_sample):
            return False
        return True

    return sampler(
        n_samples,
        nq=arm.nq,
        random_generator=random_generator,
        accept_function=accept_function,
        max_tries=max_tries,
    )


def sample_initial_states_cube(
    arm: ManipulatorDynamics,
    n_samples: int,
    half_side_length: float,
    sample_center: np.ndarray,
    max_tries: int = 0,
    valid: bool = False,
):
    def random_generator() -> np.ndarray:
        variation = np.random.rand(len(sample_center))  # sample in [0,1]^7
        variation = 2.0 * variation - 1.0  # sample in [-1,1]^7
        variation = half_side_length * variation  # sample in [-hs,hs]^7
        return variation + sample_center

    is_collided = get_is_collided_function(arm)
    if valid:
        logging.info("Sampling only those states in range.")
    lower_limit = arm.model.lowerPositionLimit
    upper_limit = arm.model.upperPositionLimit

    def accept_function(q_sample) -> bool:
        if valid:
            if not (
                (lower_limit <= q_sample).all() and (q_sample <= upper_limit).all()
            ):
                return False
        # accept if not collided
        return not is_collided(q_sample)

    return sampler(
        n_samples,
        nq=arm.nq,
        random_generator=random_generator,
        accept_function=accept_function,
        max_tries=max_tries,
    )

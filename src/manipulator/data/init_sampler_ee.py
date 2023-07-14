import numpy as np
import pinocchio as pin

from manipulator.model import ManipulatorDynamics
from manipulator.data.init_sampler import sampler
from manipulator.data.utils import get_is_collided_function


def random_configuration(model: pin.Model):
    lower_limit = model.lowerPositionLimit
    upper_limit = model.upperPositionLimit
    q = np.random.rand(model.nq)
    return lower_limit + (upper_limit - lower_limit) * q


def sample_initial_states_based_on_ee(
    arm: ManipulatorDynamics,
    n_samples: int,
    sample_radius: float,
    sample_center: np.ndarray,
    max_tries: int = 0,
) -> np.ndarray:
    radius_square = sample_radius**2

    def random_generator():
        return random_configuration(arm.model)

    is_collided = get_is_collided_function(arm)

    def accept_function(q_sample):
        if is_collided(q_sample):
            return False
        d = np.sum(
            (
                arm.get_ee(np.r_[q_sample, np.zeros_like(q_sample)]).translation
                - sample_center
            )
            ** 2
        )
        return d <= radius_square

    return sampler(
        n_samples,
        nq=arm.nq,
        random_generator=random_generator,
        accept_function=accept_function,
        max_tries=max_tries,
    )

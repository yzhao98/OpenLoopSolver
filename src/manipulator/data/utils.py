from typing import Union

import numpy as np

from manipulator.model import ManipulatorDynamics
from manipulator.data import ReachingSetup


def get_is_collided_function(arm: ManipulatorDynamics):
    def is_collided(q: np.ndarray):
        return arm.pinocchio_wrapper.is_collided(q)

    return is_collided


def get_initial_terminal_generalized_coordinates(
    select: Union[str, ReachingSetup] = ReachingSetup.default
):
    if isinstance(select, str):
        select = ReachingSetup[select]
    x0, x1 = select.get_states()
    return x0, x1

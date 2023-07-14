import abc

import crocoddyl

from manipulator.model import ManipulatorDynamics
from manipulator.data_process.dataset import ManipulatorTrajectories


class DataAugmenter:
    @classmethod
    @abc.abstractmethod
    def generate(
        cls,
        dataset: ManipulatorTrajectories,
        at_time_portion: float,
        cost_config_id: int,
        dynamics: ManipulatorDynamics,
        **kwargs,
    ) -> ManipulatorTrajectories:
        pass


class GeneralCostGetter:
    """This is to unify the interface of cost models getter"""

    def __init__(self, state, goal_frame_id, goal_state):
        self.state = state
        self.goal_frame_id = goal_frame_id
        self.goal_state = goal_state

    def __call__(self, *args, **kwargs) -> crocoddyl.CostModelAbstract:
        raise NotImplementedError

import dataclasses
import typing
from typing import List

import crocoddyl
import pinocchio as pin
import numpy as np

from manipulator.cost import CostConfig
from manipulator.data_process.states import ConfigurationOrPose
from manipulator.model import ManipulatorDynamics

__all__ = [
    "OpenLoopProblem",
    "DiscreteOpenLoopProblem",
    "DetailedDiscreteOpenLoopProblem",
    "FixedTimeOpenLoopProblem",
    "DiscreteFixedTimeOpenLoopProblem",
    "DetailedDiscreteFixedTimeOpenLoopProblem",
    "FreeTimeOpenLoopProblem",
    "DiscreteFreeTimeOpenLoopProblem",
    "DetailedDiscreteFreeTimeOpenLoopProblem",
    "OpenLoopSolution",
    "IntegrationDetails",
]


@dataclasses.dataclass(frozen=True, eq=False)
class OpenLoopProblem:
    arm: ManipulatorDynamics
    x0: np.ndarray
    x1: ConfigurationOrPose
    cost_config: CostConfig

    def __post_init__(self):
        if isinstance(self.x1, (np.ndarray, pin.SE3)):
            object.__setattr__(self, "x1", ConfigurationOrPose(self.x1))

    def change_problem(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def revise_initial(self, x0: np.ndarray):
        return self.change_problem(x0=x0)

    @property
    def nx(self):
        return self.arm.nx

    @property
    def nu(self):
        return self.arm.nu

    @property
    def nq(self):
        return self.arm.nq


@dataclasses.dataclass(frozen=True, eq=False)
class DiscreteOpenLoopProblem(OpenLoopProblem):
    dt: float


class IntegrationDetails:
    pass


@dataclasses.dataclass(frozen=True, eq=False)
class DetailedDiscreteOpenLoopProblem(DiscreteOpenLoopProblem):
    integration_details: IntegrationDetails = None

    @classmethod
    def from_discrete_open_loop_problem(
        cls,
        problem: DiscreteOpenLoopProblem,
        integration_details: IntegrationDetails = None,
    ):
        return cls(
            **problem.__dict__,
            integration_details=integration_details,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class FixedTimeOpenLoopProblem(OpenLoopProblem):
    total_times: float


@dataclasses.dataclass(
    frozen=True, eq=False
)  # though no new data is added, marking it as dataclass for multiple inheritance
class DiscreteFixedTimeOpenLoopProblem(
    FixedTimeOpenLoopProblem, DiscreteOpenLoopProblem
):
    @property
    def n_steps(self):
        return round(self.total_times / self.dt)

    def get_equilibrium_solution(self):
        assert self.x1.type == ConfigurationOrPose.Type.Configuration
        return OpenLoopSolution(
            xs=[self.x1.configuration] * (self.n_steps + 1),
            us=[self.arm.get_torque_grav_ref(self.x1.configuration)] * self.n_steps,
        )


@dataclasses.dataclass(
    frozen=True, eq=False
)  # though no new data is added, marking it as dataclass for multiple inheritance
class DetailedDiscreteFixedTimeOpenLoopProblem(
    DetailedDiscreteOpenLoopProblem, DiscreteFixedTimeOpenLoopProblem
):
    pass


@dataclasses.dataclass(
    frozen=True, eq=False
)  # mark as dataclass for multiple inheritance and consistency
class FreeTimeOpenLoopProblem(OpenLoopProblem):
    @typing.overload
    def fix_time(
        self: "DiscreteFreeTimeOpenLoopProblem", t: float
    ) -> DiscreteFixedTimeOpenLoopProblem:
        ...

    def fix_time(self, t: float) -> FixedTimeOpenLoopProblem:
        # find out the class of self dynamically, then converts it to its fixed time counterpart
        cls_name = self.__class__.__qualname__
        fixed_cls_name = cls_name.replace("FreeTime", "FixedTime")
        fixed_cls = globals()[fixed_cls_name]
        return fixed_cls(**self.__dict__, total_times=t)

    def fix_time_and_step(
        self, t: float, n_steps: int
    ) -> DiscreteFixedTimeOpenLoopProblem:
        # find out the class of self dynamically, then converts it to its fixed time counterpart
        assert t > 0, "total time must be positive"
        assert n_steps > 0, "number of steps must be positive"
        cls = DiscreteFixedTimeOpenLoopProblem
        dt = t / n_steps
        return cls(**self.__dict__, total_times=t, dt=dt)


@dataclasses.dataclass(
    frozen=True, eq=False
)  # mark as dataclass for multiple inheritance and consistency
class DiscreteFreeTimeOpenLoopProblem(FreeTimeOpenLoopProblem, DiscreteOpenLoopProblem):
    pass


@dataclasses.dataclass(
    frozen=True, eq=False
)  # mark as dataclass for multiple inheritance and consistency
class DetailedDiscreteFreeTimeOpenLoopProblem(
    DiscreteFreeTimeOpenLoopProblem, DetailedDiscreteOpenLoopProblem
):
    pass


@dataclasses.dataclass
class OpenLoopSolution:
    xs: List[np.ndarray]
    us: List[np.ndarray]
    cost: float = None
    ts: np.ndarray = None

    @classmethod
    def from_ddp(cls, ddp: crocoddyl.SolverDDP):
        running_models = ddp.problem.runningModels
        try:
            dts = [rm.dt for rm in running_models]
            ts = np.cumsum([0.0] + dts)
        except AttributeError:
            ts = None

        return cls(list(ddp.xs), list(ddp.us), cost=ddp.cost, ts=ts)

    def as_tuple(self):
        return self.xs, self.us

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

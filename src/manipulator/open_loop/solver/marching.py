import abc
from typing import Iterable, Optional, Union, List, overload

import numpy as np
from pydrake.trajectories import PiecewisePolynomial
from scipy.interpolate import interp1d

from manipulator.open_loop.problem import (
    DiscreteFixedTimeOpenLoopProblem,
    DiscreteFreeTimeOpenLoopProblem,
    DiscreteOpenLoopProblem,
    OpenLoopSolution,
    OpenLoopProblem,
)

__all__ = [
    "MarchingGenerator",
    "MarchingSolutionAdapter",
    "LineSegmentSpaceMarchGenerator",
    "LineSegmentUniformSpaceMarchGenerator",
    "IdentityMarchingSolutionAdapter",
    "TimeStepMarchGenerator",
    "InterpolationMarchingSolutionAdapter",
    "ScaleMarchingSolutionAdapter",
    "SmartTimeStepMarchGenerator",
]


class MarchingGenerator:
    """Note:
    1. The generator should yield the problem from easy to hard.
    2. The generator should NOT yield the source problem.
    """

    @overload
    def march(
        self, problem: DiscreteFixedTimeOpenLoopProblem
    ) -> Iterable[DiscreteFixedTimeOpenLoopProblem]:
        ...

    @overload
    def march(
        self, problem: DiscreteFreeTimeOpenLoopProblem
    ) -> Iterable[DiscreteFreeTimeOpenLoopProblem]:
        ...

    @overload
    def march(
        self, problem: DiscreteOpenLoopProblem
    ) -> Iterable[DiscreteOpenLoopProblem]:
        ...

    @abc.abstractmethod
    def march(self, problem: OpenLoopProblem) -> Iterable[OpenLoopProblem]:
        pass


class MarchingSolutionAdapter:
    # overload type hints
    @overload
    def adapt(
        self,
        solution: OpenLoopSolution,
        problem: Optional[DiscreteFixedTimeOpenLoopProblem],
        target_problem: DiscreteFixedTimeOpenLoopProblem,
    ) -> OpenLoopSolution:
        ...

    @overload
    def adapt(
        self,
        solution: OpenLoopSolution,
        problem: Optional[DiscreteFreeTimeOpenLoopProblem],
        target_problem: DiscreteFreeTimeOpenLoopProblem,
    ) -> OpenLoopSolution:
        ...

    @overload
    def adapt(
        self,
        solution: OpenLoopSolution,
        problem: Optional[DiscreteOpenLoopProblem],
        target_problem: DiscreteOpenLoopProblem,
    ) -> OpenLoopSolution:
        ...

    @abc.abstractmethod
    def adapt(
        self,
        solution,
        problem,
        target_problem,
    ):
        pass


class LineSegmentSpaceMarchGenerator(MarchingGenerator):
    def __init__(self, grids: Union[np.ndarray, List[float]]):
        """
        Args:
            grids: the grids of the time segment. The grids should be in descending order.
        """
        assert np.all(np.diff(grids) < 0)
        self.grids = grids

    def march(self, problem: OpenLoopProblem):
        x0, x1 = problem.x0, problem.x1.configuration
        interpolations = interp1d(
            np.array([0.0, 1.0]),
            np.array([x0, x1]),
            kind="linear",
            assume_sorted=True,
            axis=0,
        )(self.grids)
        for x in interpolations:
            yield problem.revise_initial(x)


class TimeStepMarchGenerator(MarchingGenerator):
    def __init__(self, dts: Union[np.ndarray, List[float]]):
        self.dts = dts

    def march(self, problem):
        for dt in self.dts:
            yield problem.change_problem(dt=dt)


class SmartTimeStepMarchGenerator(MarchingGenerator):
    def __init__(self, dts: Union[np.ndarray, List[float]]):
        self.dts = dts

    def march(self, problem):
        a_small_number = 1e-9
        dt = getattr(problem, "dt", a_small_number)
        for new_dt in self.dts:
            if new_dt > dt:
                yield problem.change_problem(dt=new_dt)


class LineSegmentUniformSpaceMarchGenerator(LineSegmentSpaceMarchGenerator):
    def __init__(self, k: int):
        super().__init__(np.linspace(0.0, 1.0, k + 1)[::-1][:-1])


class IdentityMarchingSolutionAdapter(MarchingSolutionAdapter):
    def adapt(
        self,
        solution,
        problem,
        target_problem,
    ):
        return solution


class InterpolationMarchingSolutionAdapter(MarchingSolutionAdapter):
    """
    This adapter interpolates the solution and then evaluate the solution at the new time grids.
    """

    @staticmethod
    def interpolate(
        ts: np.ndarray, ys: Union[np.ndarray, List[np.ndarray]], new_ts: np.ndarray
    ):
        # noinspection PyTypeChecker
        inter = PiecewisePolynomial.FirstOrderHold(ts, np.array(ys).T)
        new_ys = inter.vector_values(new_ts).T
        return new_ys

    def adapt(
        self,
        solution,
        problem,
        target_problem,
    ):
        if solution is None:
            return solution
        t0, tf = solution.ts[0], solution.ts[-1]
        new_ts = np.linspace(t0, tf, target_problem.n_steps + 1)
        new_xs = self.interpolate(solution.ts, solution.xs, new_ts)
        new_us = self.interpolate(solution.ts[:-1], solution.us, new_ts[:-1])
        return OpenLoopSolution(list(new_xs), list(new_us), ts=new_ts)


class ScaleMarchingSolutionAdapter(InterpolationMarchingSolutionAdapter):
    def adapt(
        self,
        solution,
        problem,
        target_problem,
    ):
        t0, tf = solution.ts[0], solution.ts[-1]
        new_ts = np.linspace(
            t0, t0 + target_problem.total_times, target_problem.n_steps + 1
        )
        scale_to_old_ts = (
            new_ts - t0
        ) / target_problem.total_times * problem.total_times + t0
        new_xs = self.interpolate(solution.ts, solution.xs, scale_to_old_ts)
        new_us = self.interpolate(solution.ts[:-1], solution.us, scale_to_old_ts[:-1])
        return OpenLoopSolution(list(new_xs), list(new_us), ts=new_ts)

import abc
from typing import Optional, Tuple, Union

from manipulator.open_loop.problem import (
    FreeTimeOpenLoopProblem,
    OpenLoopSolution,
)


class FreeTimeOpenLoopSolver:
    @abc.abstractmethod
    def solve(
        self,
        problem: FreeTimeOpenLoopProblem,
        initial_guess: Union[OpenLoopSolution, float],
        **kwargs,
    ) -> Tuple[Optional[OpenLoopSolution], dict]:
        pass

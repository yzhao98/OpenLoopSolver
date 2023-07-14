import abc
import itertools
from typing import Optional, Tuple

from manipulator.open_loop.problem import (
    DiscreteFixedTimeOpenLoopProblem,
    OpenLoopSolution,
)
from manipulator.open_loop.solver.marching import (
    MarchingGenerator,
    MarchingSolutionAdapter,
)


class FixedTimeOpenLoopSolver:
    @abc.abstractmethod
    def solve(
        self,
        problem: DiscreteFixedTimeOpenLoopProblem,
        initial_guess: Optional[OpenLoopSolution],
        **kwargs
    ) -> Tuple[Optional[OpenLoopSolution], dict]:
        pass


class MarchingBasedFixedTimeOpenLoopSolver(FixedTimeOpenLoopSolver):
    def __init__(
        self,
        marching_generator: MarchingGenerator,
        marching_solution_adapter: MarchingSolutionAdapter,
        solver: FixedTimeOpenLoopSolver,
        fail_on_sub_problem_failure: bool = True,
    ):
        self.space_march_generator = marching_generator
        self.space_march_solution_adapter = marching_solution_adapter
        self.solver = solver
        self.fail_on_sub_problem_failure = fail_on_sub_problem_failure

    def solve(
        self,
        problem: DiscreteFixedTimeOpenLoopProblem,
        initial_guess: Optional[OpenLoopSolution],
        **kwargs
    ) -> Tuple[Optional[OpenLoopSolution], dict]:
        solution = initial_guess
        info = {}
        previous_problem = None
        # add the source problem to the generator
        for sub_problem in itertools.chain(
            self.space_march_generator.march(problem), [problem]
        ):
            solution = self.space_march_solution_adapter.adapt(
                solution,
                previous_problem,
                target_problem=sub_problem,
            )
            solution, info = self.solver.solve(sub_problem, solution, **kwargs)
            if self.fail_on_sub_problem_failure:
                assert solution is not None, "Failed at sub-problem {}".format(
                    sub_problem
                )
            previous_problem = sub_problem
        return solution, info

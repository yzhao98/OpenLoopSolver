import dataclasses
import functools
from typing import Dict

import crocoddyl
import numpy as np
from torch import multiprocessing as mpp

from manipulator.constants import PreconfiguredModels
from manipulator.data_process.model import CollectedResultsV2
from manipulator.ddp_crocoddyl.solve import DDPOpenLoopSolver
from manipulator.model import ManipulatorDynamics
from manipulator.open_loop import (
    LineSegmentSpaceMarchGenerator,
    IdentityMarchingSolutionAdapter,
    TimeStepMarchGenerator,
    InterpolationMarchingSolutionAdapter,
    DiscreteFixedTimeOpenLoopProblem,
    MarchingBasedFixedTimeOpenLoopSolver,
    SmartTimeStepMarchGenerator,
)


def find_ddp_by_marching(
    initial_list,
    t0,
    dt,
    goal_state,
    cost_config: dict,
    robot_name: str,
    num_processors: int = 1,
    marching_method: str = "sm",
):
    solver = construct_solver(marching_method)
    n_samples = len(initial_list)

    # noinspection DuplicatedCode
    use_equilibrium_as_initial_guess = True if marching_method == "sm" else False
    if num_processors > 1:
        with mpp.Pool(num_processors) as pool:
            problem_defs = (
                solver,
                robot_name,
                cost_config,
                goal_state,
                dt,
                t0,
                use_equilibrium_as_initial_guess,
            )
            results = pool.map(
                functools.partial(
                    solve_one_with_solver,
                    *problem_defs,
                ),
                initial_list,
            )
    else:
        results = [
            solve_one_with_solver(
                solver,
                robot_name,
                cost_config,
                goal_state,
                dt,
                t0,
                use_equilibrium_as_initial_guess,
                initial_state,
            )
            for initial_state in initial_list
        ]

    collected_results: Dict[int, CollectedResultsV2] = {
        i: results[i] for i in range(n_samples)
    }

    return collected_results


def solve_one_with_solver(
    solver,
    robot_name,
    cost_config,
    goal_state,
    dt,
    t0,
    use_equilibrium_as_initial_guess: bool,
    initial_state,
):
    arm = ManipulatorDynamics.build(config_name=PreconfiguredModels[robot_name])
    problem = DiscreteFixedTimeOpenLoopProblem(
        arm,
        x0=initial_state,
        x1=goal_state,
        cost_config=cost_config,
        total_times=t0,
        dt=dt,
    )
    solution, info = solver.solve(
        problem,
        initial_guess=(
            problem.get_equilibrium_solution()
            if use_equilibrium_as_initial_guess
            else None
        ),
    )
    if solution is not None:
        print(solution.cost)
    else:
        print("None")

    if not isinstance(cost_config, dict):
        cost_config = dataclasses.asdict(problem.cost_config)

    result_v2 = CollectedResultsV2.build(
        initial_state=problem.x0,
        goal_state=problem.x1.value,
        distance=None,
        tf=problem.total_times,
        dt=problem.dt,
        n_steps=problem.n_steps,
        cost_config=cost_config,
        is_converged=solution is not None,
        solution=info["ddp"],
        problem=info["problem"],
    )
    return result_v2


def construct_solver(method: str):
    if method == "sm":
        solver = MarchingBasedFixedTimeOpenLoopSolver(
            LineSegmentSpaceMarchGenerator(np.power(1.2, -np.arange(32))),
            IdentityMarchingSolutionAdapter(),
            DDPOpenLoopSolver(solver_cls=crocoddyl.SolverFDDP),
        )
        return solver
    if method == "dt":
        solver = MarchingBasedFixedTimeOpenLoopSolver(
            TimeStepMarchGenerator(dts=[1e-2, 5e-3, 1e-3]),
            InterpolationMarchingSolutionAdapter(),
            DDPOpenLoopSolver(solver_cls=crocoddyl.SolverFDDP),
        )
        return solver
    if method == "smart_dt":
        solver = MarchingBasedFixedTimeOpenLoopSolver(
            SmartTimeStepMarchGenerator(dts=[1e-2, 5e-3, 1e-3]),
            InterpolationMarchingSolutionAdapter(),
            DDPOpenLoopSolver(solver_cls=crocoddyl.SolverFDDP),
        )
        return solver

import functools
import logging
import multiprocessing
from collections import namedtuple
from itertools import repeat
from typing import Union, List, Dict, Sequence, Tuple, Any, Optional, Iterable

import crocoddyl
import numpy as np

from manipulator.constants import PreconfiguredModels, DEFAULT_MAX_DDP_ITERATIONS
from manipulator.data_process.model import CollectedResults, CollectedResultsV2
from manipulator.model import ManipulatorDynamics
from manipulator.ddp_crocoddyl.solve import (
    ManipulatorDDPSolverFromCostConfig,
    ManipulatorDDPSolverFromCostConfigOverrideTerminalState,
)
from manipulator.util.warm_start import (
    WarmStartScheduler,
    NoWarmStartScheduler,
    SingleWarmStartScheduler,
)


def n_steps_estimate(dt, t0):
    n_steps = round(t0 / dt)
    if t0 - n_steps * dt >= 1e-2 * dt:
        n_steps = n_steps + 1
    return n_steps


def moving_time_estimation(
    arm,
    start_state,
    end_state,
    start_placement=None,
    end_placement=None,
    maximal_speed=None,
):
    """
    Simple moving time estimation
    Parameters
    ----------
    arm: ManipulatorDynamics
        Use to compute forward kinematics
    start_state: np.ndarray
        of dimension nx
    end_state: np.ndarray
        of dimension nx
    start_placement: pinocchio.Placement
        if not provided, will use arm to compute from start_state
    end_placement: pinocchio.Placement
        if not provided, will use arm to compute from end_state
    maximal_speed: float
        the speed that the robot's end-effector runs, defaults to 1.5 if pass None
    Returns
    -------
    (distance: float, time_estimate: float)
    distance is the distance of translation parts of the placements
    """
    if maximal_speed is None:
        maximal_speed = 1.5
    end_placement = (
        end_placement if end_placement is not None else arm.get_ee(end_state)
    )
    start_placement = (
        start_placement if start_placement is not None else arm.get_ee(start_state)
    )
    distance: float = np.linalg.norm(
        start_placement.translation - end_placement.translation
    )
    time_estimate = distance / maximal_speed
    return distance, time_estimate


def initial_states_unify_and_verify(x0s, expected_nx: int, nonzero_q_dot: bool = False):
    if nonzero_q_dot:
        x0s = [
            x0
            if len(x0) == expected_nx
            else np.r_[x0, 0.001 * np.random.randn(x0.shape[0])]
            for x0 in x0s
        ]
    else:
        x0s = [
            x0 if len(x0) == expected_nx else np.r_[x0, np.zeros_like(x0)] for x0 in x0s
        ]
    assert all(
        len(x0) == expected_nx for x0 in x0s
    ), f"Size of initial states does not match that of the expected size {expected_nx}"
    return x0s


def compute_time_and_distance_for_all_states(arm, x0s, goal_states, maximal_speed):
    distances: List[float, ...] = []
    times_estimate: List[float, ...] = []
    for x0, goal in zip(x0s, goal_states):
        goal_placement = arm.get_ee(goal)
        distance, time_estimate = moving_time_estimation(
            arm,
            x0,
            goal_states,
            end_placement=goal_placement,
            maximal_speed=maximal_speed,
        )
        distances.append(distance)
        times_estimate.append(time_estimate)
    return distances, times_estimate


def robust_ddp_solver_no_raise(
    arm: ManipulatorDynamics,
    initial_state: np.ndarray,
    goal_terminal_state: np.ndarray,
    dt: float,
    cost_config,
    moving_time: float,
    max_ddp_iterations: int = DEFAULT_MAX_DDP_ITERATIONS,
    n_trials: int = 10,
    max_trials: int = 200,
    noise_level: float = 1e-1,
    cost_hint: float = None,
    solution_hint=None,
):
    try:
        return robust_ddp_solver(
            arm=arm,
            initial_state=initial_state,
            goal_terminal_state=goal_terminal_state,
            dt=dt,
            cost_config=cost_config,
            moving_time=moving_time,
            max_ddp_iterations=max_ddp_iterations,
            n_trials=n_trials,
            max_trials=max_trials,
            noise_level=noise_level,
            cost_hint=cost_hint,
            solution_hint=solution_hint,
        )
    except AssertionError:
        return None


def robust_ddp_solver(
    arm: ManipulatorDynamics,
    initial_state: np.ndarray,
    goal_terminal_state: np.ndarray,
    dt: float,
    cost_config,
    moving_time: float,
    max_ddp_iterations: int = DEFAULT_MAX_DDP_ITERATIONS,
    n_trials: int = 10,
    max_trials: int = 200,
    noise_level: float = 1e-1,
    cost_hint: float = None,
    solution_hint=None,
    verbose: bool = False,
) -> Tuple[crocoddyl.SolverDDP, crocoddyl.ShootingProblem]:
    if verbose:
        print("Running robust ddp solver...")
    n_steps = n_steps_estimate(dt, moving_time)
    noise_level = noise_level * np.linalg.norm(goal_terminal_state - initial_state)
    Trial = namedtuple("Trial", ("cost", "solution"))
    trials = []

    n_tries = 0
    while len(trials) <= n_trials:
        n_tries += 1
        if n_tries > max_trials:
            logging.warning(f"Cannot find {n_trials} trials under {max_trials} tries.")
            break
        noise = np.random.rand(*initial_state.shape) * noise_level
        ddp, is_converged, _ = _solve_ddp(
            arm,
            x0=initial_state + noise,
            x1=goal_terminal_state,
            dt=dt,
            n_steps=n_steps,
            cost_config=cost_config,
            max_ddp_iterations=max_ddp_iterations,
            warm_start_guess=solution_hint,
        )
        if is_converged:
            trials.append(Trial(ddp.cost, ddp))

    if verbose:
        print("The costs found are: ")
        for t in trials:
            print(t.cost, end=";")
        print()
    if len(trials) <= n_trials:
        if cost_hint is None:
            assert False, "Cannot find enough guesses to robustly solve the problem."
        for t in trials:
            if t.cost <= 1.2 * cost_hint:
                break
        else:
            assert False, "Cannot find enough guesses or a cost that close to desired."

    guess = min(trials, key=lambda x: x.cost)
    ddp, is_converged, problem = _solve_ddp(
        arm,
        x0=initial_state,
        x1=goal_terminal_state,
        dt=dt,
        n_steps=n_steps,
        cost_config=cost_config,
        max_ddp_iterations=max_ddp_iterations,
        warm_start_guess=(guess.solution.xs, guess.solution.us),
    )
    assert is_converged
    return ddp, problem


def _construct_multiple_single_warm_start_schedulers(
    warm_start_scheduler: WarmStartScheduler,
    initial_states: List[np.ndarray],
    dts: List[float],
    moving_times: List[float],
) -> Iterable[WarmStartScheduler]:
    if isinstance(warm_start_scheduler, SingleWarmStartScheduler):
        return repeat(warm_start_scheduler)
    # TODO: this file should not depend on manipulator_utils package. We do this for simplicity. Will change later.
    from manipulator_utils import GridBasedWarmStartScheduler

    if isinstance(warm_start_scheduler, GridBasedWarmStartScheduler):
        logging.disable(logging.INFO)
        warm_start_schedulers = [
            SingleWarmStartScheduler(
                warm_start_scheduler.guess(
                    index=i, x0=x0, n_steps=n_steps_estimate(dt, t0)
                )
            )
            for i, (x0, dt, t0) in enumerate(zip(initial_states, dts, moving_times))
        ]
        logging.disable(logging.NOTSET)
        logging.info(
            "Constructed multiple single warm start schedulers from GridBasedWarnStartScheduler."
        )
        return warm_start_schedulers


def _get_all_warm_start_sols(
    warm_start_scheduler: WarmStartScheduler,
    initial_states: List[np.ndarray],
    dts: List[float],
    moving_times: List[float],
):
    warm_start_sols = [
        warm_start_scheduler.guess(
            index=i,
            x0=x0,
            n_steps=n_steps_estimate(dt, t0),
        )
        for i, (x0, dt, t0) in enumerate(zip(initial_states, dts, moving_times))
    ]
    return warm_start_sols


def find_ddp_trajectories(
    arm: ManipulatorDynamics,
    initial_states: Union[np.ndarray, List[np.ndarray]],
    goal_terminal_states: Union[np.ndarray, List[np.ndarray]],
    dts: Union[Sequence[float], float],
    cost_config,
    moving_times: Union[np.ndarray, List[float]] = None,
    max_ddp_iterations: int = DEFAULT_MAX_DDP_ITERATIONS,
    maximal_speed=None,
    warm_start_scheduler: Optional[WarmStartScheduler] = None,
    achieving_states: Optional[List[np.ndarray]] = None,
    num_processors=1,
    num_cube_points=-1,
    return_warm_sols=False,
    nonzero_q_dot=False,
) -> Tuple[Dict[int, CollectedResultsV2], Any]:
    """
    Parameters
    ----------
    return_warm_sols: set this to return a list of warm start solutions instead of a warm start scheduler

    """
    logging.info(f"In DDP solver:")
    logging.info(f"The Cost Configs is: {cost_config}")
    robot_name = arm.configured_model
    (
        n_samples,
        initial_states,
        goal_terminal_states,
        dts,
        moving_times,
        distances,
    ) = prepare_arguments(
        arm,
        initial_states,
        dts,
        moving_times,
        goal_terminal_states,
        maximal_speed,
        nonzero_q_dot,
    )
    if warm_start_scheduler is None:
        warm_start_scheduler = NoWarmStartScheduler()

    if achieving_states is not None:
        assert len(achieving_states) == n_samples

    pool = multiprocessing.Pool(num_processors)
    problem_defs = (
        robot_name,
        cost_config,
        max_ddp_iterations,
    )
    problem_specs = (
        repeat(None) if achieving_states is None else achieving_states,
        dts,
        initial_states,
        moving_times,
        distances,
        goal_terminal_states,
    )
    warm_start_sols = _get_all_warm_start_sols(
        warm_start_scheduler, initial_states, dts, moving_times
    )
    if num_cube_points < 2:
        collected_results_list = pool.starmap(
            functools.partial(
                find_one_ddp_trajectory,
                *problem_defs,
            ),
            zip(*problem_specs, warm_start_sols),
        )
    else:
        # GridBasedWarmStartScheduler relies on a dataset which does not support multiprocessing well.
        collected_results_list = pool.starmap(
            functools.partial(
                find_one_ddp_trajectory_by_steps,
                *problem_defs,
            ),
            zip(*problem_specs, repeat(num_cube_points), warm_start_sols),
        )
    collected_results: Dict[int, CollectedResultsV2] = {
        i: collected_results_list[i] for i in range(n_samples)
    }
    pool.close()
    pool.join()

    for traj_id in collected_results:
        traj = collected_results[traj_id]
        assert abs(traj.estimated_time - moving_times[traj_id]) <= 1e-15
        traj.validate()

    if return_warm_sols:
        return collected_results, warm_start_sols
    else:
        return collected_results, warm_start_scheduler


def find_one_ddp_trajectory(
    robot: PreconfiguredModels,
    cost_config,
    max_ddp_iterations,
    achieving_state,
    dt,
    initial_state,
    moving_time,
    distance,
    goal_terminal_state,
    warm_start_sol,
) -> CollectedResultsV2:
    arm = ManipulatorDynamics.build(robot)
    n_steps = n_steps_estimate(dt, moving_time)
    if n_steps == 0:
        # if the moving time is too small, we just skip it.
        ddp = None
        problem = None
        is_converged = True
    else:
        # warm_start_guess = warm_start_scheduler.guess(idx, initial_state, n_steps)
        ddp, is_converged, problem = _solve_ddp(
            arm,
            initial_state,
            goal_terminal_state,
            dt,
            n_steps,
            cost_config,
            max_ddp_iterations,
            warm_start_sol,
            achieving_state=achieving_state,
        )

    result = CollectedResultsV2.build(
        initial_state=initial_state,
        goal_state=goal_terminal_state,
        distance=distance,
        tf=moving_time,
        dt=dt,
        n_steps=n_steps,
        cost_config=cost_config,
        is_converged=is_converged,
        solution=ddp,
        problem=problem,
    )
    return result


def find_one_ddp_trajectory_by_steps(
    robot: PreconfiguredModels,
    cost_config,
    max_ddp_iterations,
    achieving_state,
    dt,
    initial_state,
    moving_time,
    distance,
    goal_terminal_state,
    num_cube_points,
    warm_start_sol,
) -> CollectedResultsV2:
    """This function only needs a warm start solution instead of a warm start scheduler."""
    arm = ManipulatorDynamics.build(robot)
    results = solve_ddp_by_coordinate_walking_warm_start(
        arm,
        initial_state,
        goal_terminal_state,
        achieving_state,
        dt,
        moving_time,
        cost_config,
        distance,
        max_ddp_iterations,
        num_cube_points,
        warm_start_sol,
    )
    return results[-1]


def solve_ddp_by_coordinate_walking_warm_start(
    arm,
    x0_init,
    goal,
    achieving_state,
    dt,
    t0,
    cost_config,
    d0,
    max_ddp_iterations,
    num_cube_points,
    warm_start_guess,
) -> List[CollectedResultsV2]:
    n_steps = n_steps_estimate(dt, t0)
    x0 = np.array(warm_start_guess[0][0])
    x_linspace = np.linspace(x0, x0_init, num_cube_points)
    result_list = []
    for i in range(arm.nq - 1, -1, -1):
        for n in range(1, num_cube_points):
            x0[i] = x_linspace[n][i]
            if n_steps == 0:
                # if the moving time is too small, we just skip it.
                ddp = None
                problem = None
                is_converged = True
            else:
                ddp, is_converged, problem = _solve_ddp(
                    arm,
                    x0,
                    goal,
                    dt,
                    n_steps,
                    cost_config,
                    max_ddp_iterations,
                    warm_start_guess,
                    achieving_state=achieving_state,
                )
                warm_start_guess = (ddp.xs, ddp.us)
            result = CollectedResultsV2.build(
                initial_state=x0,
                goal_state=goal,
                distance=d0,
                tf=t0,
                dt=dt,
                n_steps=n_steps,
                cost_config=cost_config,
                is_converged=is_converged,
                solution=ddp,
                problem=problem,
            )
            result_list.append(result)
            # if fail when walking, we shall mark the result as failed and stop here.
            if not is_converged:
                return result_list
            # if the moving time is too small, we just skip it and return.
            if n_steps == 0:
                return result_list
    return result_list


def must_solve_ddp(
    arm, x0, x1, dt, moving_time, cost_config, max_ddp_iterations, warm_start_guess=None
):
    ddp, is_converged, problem = _solve_ddp(
        arm,
        x0,
        x1,
        dt,
        n_steps_estimate(dt, moving_time),
        cost_config,
        max_ddp_iterations,
        warm_start_guess,
    )
    assert is_converged
    return ddp, problem


def _solve_ddp(
    arm,
    x0,
    x1,
    dt,
    n_steps,
    cost_config,
    max_ddp_iterations,
    warm_start_guess=None,
    achieving_state=None,
):
    solver_kwargs = {
        "arm": arm,
        "x0": x0,
        "x1": x1,
        "time_step": dt,
        "n_steps": n_steps,
        "cost_config": cost_config,
    }
    solver_cls = ManipulatorDDPSolverFromCostConfig
    if achieving_state is not None:
        solver_cls = ManipulatorDDPSolverFromCostConfigOverrideTerminalState
        solver_kwargs["achieving_state"] = achieving_state
    solver = solver_cls(**solver_kwargs)
    is_converged, ddp, problem = solver.solve(
        max_ddp_iterations, verbose=False, initial_guess=warm_start_guess
    )
    return ddp, is_converged, problem


def prepare_arguments(
    arm,
    initial_states,
    dts,
    moving_times,
    goal_terminal_states,
    maximal_speed,
    nonzero_q_dot=False,
):
    n_samples = (
        initial_states.shape[0]
        if hasattr(initial_states, "shape")
        else len(initial_states)
    )
    dts = [dts] * n_samples if isinstance(dts, float) else dts
    goal_terminal_states = np.array(goal_terminal_states)
    if len(goal_terminal_states.shape) > 1:
        assert goal_terminal_states.shape[0] == n_samples
    else:
        goal_terminal_states = np.tile(goal_terminal_states, (n_samples, 1))

    initial_states = initial_states_unify_and_verify(
        initial_states, expected_nx=arm.nx, nonzero_q_dot=nonzero_q_dot
    )
    distances, times_estimate = compute_time_and_distance_for_all_states(
        arm, initial_states, goal_terminal_states, maximal_speed
    )
    if moving_times is not None:
        assert maximal_speed is None, "Should provide only one of velocity, time to go"
        if isinstance(moving_times, float):
            logging.info(
                f"Got moving times: {moving_times}, will use it for all examples."
            )
            moving_times = [moving_times] * len(initial_states)
    else:
        moving_times = times_estimate
    return n_samples, initial_states, goal_terminal_states, dts, moving_times, distances


def convert_to_dict_and_save(
    path_or_buffer,
    collected_results: Dict[int, CollectedResults],
):
    CollectedResults.dict_bulk_save(collected_results, path_or_buffer)


def multi_slice(list_of_array: List[np.ndarray], list_of_length: Sequence[int]):
    """
    Slice a list of array into multiple list of array with given length.
    :param list_of_array: list of array
    :param list_of_length: list of length
    :return: list of  array
    """
    assert len(list_of_array) == len(list_of_length)
    list_of_array_shorter = []
    for i, length in enumerate(list_of_length):
        list_of_array_shorter.append(list_of_array[i][:length])
    return list_of_array_shorter

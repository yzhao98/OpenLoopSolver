import dataclasses
import enum
from typing import Union, List, Tuple, Optional

import crocoddyl
import numpy as np

from manipulator.ddp_crocoddyl.solve import DDPOpenLoopSolver
from manipulator.open_loop import (
    FreeTimeOpenLoopSolver,
    FixedTimeOpenLoopSolver,
    FreeTimeOpenLoopProblem,
    OpenLoopSolution,
    DiscreteFreeTimeOpenLoopProblem,
    DiscreteFixedTimeOpenLoopProblem,
    OpenLoopProblem,
)
from manipulator.simple_utils import StopWatch


class TimeGDSolverMethod(enum.Enum):
    var_dt = "var_dt"
    var_n = "var_n"


@dataclasses.dataclass
class TimeGDSolverSolveConfig:
    method: TimeGDSolverMethod
    kwargs: dict

    @classmethod
    def default(cls):
        return cls(
            method=TimeGDSolverMethod.var_dt,
            kwargs={"n_steps": 2000},
        )


class TimeGDSolver(FreeTimeOpenLoopSolver):
    """
    The free time solver is put here because it relies on the DDP fixed-time solver.
    """

    def __init__(
        self,
        fixed_time_solver: Union[DDPOpenLoopSolver, FixedTimeOpenLoopSolver],
        solution_adapter,
        max_iter: int = 50,
        penalty_weight_on_time: float = 0.0,
    ):
        """
        Parameters
        ----------
        fixed_time_solver: the solver to solve each fixed time problem.
            note that, though we hint on the type `FixedTimeOpenLoopSolver`,
             a fixed time solver must return the info dict containing the information we use to compute the gradient.
        solution_adapter: the adapter to convert the solution between fixed times with different time
        max_iter: maximum number of iterations
        penalty_weight_on_time: the constant weight on the time cost (added to running cost density function L)
        """
        self.fixed_time_solver = fixed_time_solver
        self.solution_adapter = solution_adapter
        self.max_iter = max_iter
        self.penalty_weight_on_time = penalty_weight_on_time

    # def compute_gradient(
    #     self, problem: FreeTimeOpenLoopProblem, solution: OpenLoopSolution, info: dict
    # ) -> float:
    #     shooting_problem: crocoddyl.ShootingProblem = info["problem"]
    #     running_model = shooting_problem.runningModels[-1]
    #     running_data = shooting_problem.runningDatas[-1]
    #     terminal_model = shooting_problem.terminalModel
    #     terminal_data = shooting_problem.terminalData
    #     x_T = solution.xs[-1]
    #     u_T = solution.us[-1]  # u is piecewise constant
    #     # get x', L at T, we shall use
    #     running_model.calc(running_data, x_T, u_T)
    #     running_model.calcDiff(running_data, x_T, u_T)
    #     x_prime = np.concatenate([x_T[problem.nq :], running_data.differential.xout])
    #     L = running_data.differential.cost
    #
    #     # get d(terminal_cost)/dx at T
    #     terminal_model.calc(terminal_data, x_T)
    #     terminal_model.calcDiff(terminal_data, x_T)
    #     phi_prime = terminal_data.Lx
    #
    #     return np.dot(phi_prime, x_prime) + L + self.penalty_weight_on_time

    def compute_gradient(
        self, problem: OpenLoopProblem, solution: OpenLoopSolution, info: dict
    ):
        from manipulator.data_process.model import CollectedResultsV2

        solution_ddp: crocoddyl.ShootingProblem = info["ddp"]
        result = CollectedResultsV2.build(
            initial_state=problem.x0,
            goal_state=problem.x1.value,
            distance=0.0,
            tf=solution.ts[-1],
            dt=solution.ts[1] - solution.ts[0],
            n_steps=len(solution.us),
            cost_config=problem.cost_config,  # noqa
            is_converged=True,
            solution=solution_ddp,
            problem=info["problem"],
        )

        lam = result.Vx[1:]
        shooting_pb = info["problem"]
        L = np.empty(len(solution.us))
        f = np.empty(shape=(len(solution.us), len(solution.xs[0])))
        for i, data in enumerate(shooting_pb.runningDatas):
            # TODO: for RK4, below not working
            L[i] = data.differential.cost
            f[i] = np.r_[solution.xs[i][problem.nq :], data.differential.xout]

        dts = np.array([model.dt for model in shooting_pb.runningModels])
        hamiltonian = ((L + np.sum(lam * f, axis=-1)) * dts).sum()
        return hamiltonian + self.penalty_weight_on_time * solution.ts[-1]

    def cost_with_time_penalty(self, cost, t):
        return cost + self.penalty_weight_on_time * t

    def solve_var_dt(
        self,
        problem: FreeTimeOpenLoopProblem,
        initial_guess: Union[OpenLoopSolution, float],
        n_steps: int,
        stop_if_dt_update_less_than: float = 1e-6,
        max_dt_update_ratio=0.2,
        tol: float = 1e-5,
    ):
        watch = StopWatch()
        sols = []
        watch.start("Initial Warm Start")
        if isinstance(initial_guess, float):
            t = initial_guess
            fix_time_problem = problem.fix_time_and_step(t, n_steps)
            solution, info = self.fixed_time_solver.solve(fix_time_problem, None)
        else:
            t = initial_guess.ts[-1]
            fix_time_problem = problem.fix_time_and_step(t, n_steps)
            solution, info = self.fixed_time_solver.solve(
                fix_time_problem, initial_guess
            )

        sols.append(solution)
        watch.stop("Initial Warm Start", report=True, clear=True)

        t_seq = [t]
        cost_seq = [solution.cost]
        gradient_seq = []
        watch.start("All Iterations")

        for _ in range(self.max_iter):
            # 1. find the gradient of the cost function with respect to time
            print("Iteration {} Start".format(_))
            watch.start("Iteration {}".format(_))
            gd = self.compute_gradient(problem, solution, info)
            gradient_seq.append(gd)
            if abs(gd) <= tol:
                print("Converged in {} iterations. Terminal GD is {}".format(_, gd))
                break
            sign = np.sign(gd)
            # 2. update
            (
                t,
                new_fix_problem,
                solution,
                info,
                success,
            ) = self.find_decreasing_var_dt_no_search(
                problem,
                n_steps=n_steps,
                previous_t=t,
                previous_solution=solution,
                previous_cost=cost_seq[-1],
                previous_fix_time_problem=fix_time_problem,
                sign=sign,
                gd=gd,
                max_dt_ratio=max_dt_update_ratio,
                stop_if_dt_update_less_than=stop_if_dt_update_less_than,
                all_gds=gradient_seq,
                all_ts=t_seq,
            )
            if not success:
                break

            watch.stop("Iteration {}".format(_), report=True, clear=True)
            print("t: {}, cost: {}, gd(was): {}".format(t, solution.cost, gd))
            t_seq.append(t)
            sols.append(solution)
            fix_time_problem = new_fix_problem
            cost_seq.append(solution.cost)
            print("Iteration {} End".format(_))
        else:
            print("Max iteration reached.")
            raise RuntimeError("Cannot converge.")

        watch.stop("All Iterations", report=True, clear=True)

        return solution, {
            "t_seq": t_seq,
            "cost_seq": cost_seq,
            "grad_seq": gradient_seq,
            "sols": sols,
            **info,
        }

    def solve_naive(
        self, problem: DiscreteFreeTimeOpenLoopProblem, time_sequences: List[float]
    ):
        solution = None
        previous_fix_time_problem = None
        info = {}
        gd_seqs = []
        cost_seqs = []
        sols = []
        for t in time_sequences:
            new_fix_problem = problem.fix_time(t)
            if solution is not None:
                solution_adapted = self.solution_adapter.adapt(
                    solution, previous_fix_time_problem, new_fix_problem
                )
            else:
                solution_adapted = None
            solution, info = self.fixed_time_solver.solve(
                new_fix_problem, solution_adapted
            )
            sols.append(solution)
            gd = self.compute_gradient(problem, solution, info)
            gd_seqs.append(gd)
            cost_seqs.append(solution.cost)
            previous_fix_time_problem = new_fix_problem
        return solution, {
            "grad_seq": gd_seqs,
            "cost_seq": cost_seqs,
            "sols": sols,
            **info,
        }

    def solve_var_n(
        self,
        problem: DiscreteFreeTimeOpenLoopProblem,
        initial_guess: Union[OpenLoopSolution, float],
        max_update_steps: int,
    ) -> Tuple[Optional[OpenLoopSolution], dict]:
        watch = StopWatch()
        sols = []
        watch.start("Initial Warm Start")
        if isinstance(initial_guess, float):
            t = initial_guess
            fix_time_problem = problem.fix_time(t)
            solution, info = self.fixed_time_solver.solve(fix_time_problem, None)
        else:
            t = initial_guess.ts[-1]
            fix_time_problem = problem.fix_time(t)
            solution, info = self.fixed_time_solver.solve(
                fix_time_problem, initial_guess
            )
        sols.append(solution)
        watch.stop("Initial Warm Start", report=True, clear=True)

        t_seq = [t]
        cost_seq = [solution.cost]
        gradient_seq = []
        watch.start("All Iterations")
        success = True
        gd = 1e4
        for _ in range(self.max_iter):
            # 1. find the gradient of the cost function with respect to time
            watch.start("Iteration {}".format(_))
            gd = self.compute_gradient(problem, solution, info)
            gradient_seq.append(gd)
            if abs(gd) <= problem.dt:
                print("Converged in {} iterations.".format(_))
                break
            sign = np.sign(gd)
            # 2. update
            t, new_fix_problem, solution, info, success = self.find_decreasing_step(
                problem,
                previous_t=t,
                previous_solution=solution,
                previous_cost=cost_seq[-1],
                previous_fix_time_problem=fix_time_problem,
                sign=sign,
                gd=gd,
                max_steps=max_update_steps,
            )
            if not success:
                break

            watch.stop("Iteration {}".format(_), report=True, clear=True)
            print("t: {}, cost: {}".format(t, solution.cost))
            t_seq.append(t)
            sols.append(solution)
            fix_time_problem = new_fix_problem
            cost_seq.append(solution.cost)
        else:
            print("Max iteration reached.")

        if not success:
            print("Energy does not decrease, but gradient is {}".format(gd))

        watch.stop("All Iterations", report=True, clear=True)

        return solution, {
            "t_seq": t_seq,
            "cost_seq": cost_seq,
            "grad_seq": gradient_seq,
            "sols": sols,
            **info,
        }

    @staticmethod
    def estimate_t_step_size(sign, gd, max_steps, dt: float):
        return -sign * min(round(abs(gd) / dt), max_steps) * dt

    @staticmethod
    def estimate_t_step_size_var_dt(sign, gd, max_dt_update: float) -> float:
        return -sign * min(abs(gd), max_dt_update)

    @staticmethod
    def estimate_t_step_size_var_dt_quasi_newton(
        ts, gds, max_dt_update: float
    ) -> float:
        return -np.clip(
            (ts[-1] - ts[-2]) / (gds[-1] - gds[-2]) * gds[-1],
            -max_dt_update,
            max_dt_update,
        )

    def solve(
        self,
        problem: FreeTimeOpenLoopProblem,
        initial_guess: Union[OpenLoopSolution, float],
        **kwargs,
    ) -> Tuple[Optional[OpenLoopSolution], dict]:
        config = kwargs.pop("config", TimeGDSolverSolveConfig.default())
        if config.method == TimeGDSolverMethod.var_dt:
            return self.solve_var_dt(
                problem,
                initial_guess,
                **kwargs,
                **config.kwargs,
            )

        if config.method == TimeGDSolverMethod.var_n:
            assert isinstance(problem, DiscreteFreeTimeOpenLoopProblem)
            return self.solve_var_n(
                problem,
                initial_guess,
                **kwargs,
                **config.kwargs,
            )

        raise NotImplementedError

    def find_decreasing_var_dt_no_search(
        self,
        problem: FreeTimeOpenLoopProblem,
        n_steps: int,
        previous_t: float,
        previous_solution,
        previous_cost,
        previous_fix_time_problem,
        sign,
        gd,
        max_dt_ratio: float,
        stop_if_dt_update_less_than: float = 1e-6,
        all_gds=None,
        all_ts=None,
    ):
        max_dt_updates = previous_t * max_dt_ratio
        # Compute Update Direction
        if all_gds is not None and len(all_gds) > 2:
            # Quasi-Newton update
            # x^{k+1} = x^{k} - (x^k - x^{k-1}) / (c^k - c^{k-1})c^k
            update_trail = self.estimate_t_step_size_var_dt_quasi_newton(
                (all_ts[-2], previous_t), (all_gds[-2], gd), max_dt_updates
            )
        else:
            # GD update one step
            update_trail = self.estimate_t_step_size_var_dt(sign, gd, max_dt_updates)

        update = update_trail
        t = previous_t + update
        new_fix_problem = problem.fix_time_and_step(t, n_steps)
        solution_adapted = self.solution_adapter.adapt(
            previous_solution, previous_fix_time_problem, new_fix_problem
        )
        solution, info = self.fixed_time_solver.solve(new_fix_problem, solution_adapted)

        return t, new_fix_problem, solution, info, True

    def find_decreasing_var_dt(
        self,
        problem: FreeTimeOpenLoopProblem,
        n_steps: int,
        previous_t: float,
        previous_solution,
        previous_cost,
        previous_fix_time_problem,
        sign,
        gd,
        max_dt_ratio: float,
        stop_if_dt_update_less_than: float = 1e-6,
        all_gds=None,
        all_ts=None,
    ):
        max_dt_updates = previous_t * max_dt_ratio
        # Compute Update Direction
        if all_gds is not None and len(all_gds) > 2:
            # Quasi-Newton update
            # x^{k+1} = x^{k} - (x^k - x^{k-1}) / (c^k - c^{k-1})c^k
            update_trail = self.estimate_t_step_size_var_dt_quasi_newton(
                (all_ts[-2], previous_t), (all_gds[-2], gd), max_dt_updates
            )
        else:
            # GD update one step
            update_trail = self.estimate_t_step_size_var_dt(sign, gd, max_dt_updates)

        # */2.0
        left_index = np.ceil(np.log2(abs(update_trail) / stop_if_dt_update_less_than))
        left_index = max(left_index, 0)
        right_index = 0

        j = right_index
        while True:
            # this loop is to ensure a solution is found.
            update = update_trail * np.power(1 / 2, j)
            t = previous_t + update
            new_fix_problem = problem.fix_time_and_step(t, n_steps)
            solution_adapted = self.solution_adapter.adapt(
                previous_solution, previous_fix_time_problem, new_fix_problem
            )
            try:
                solution, info = self.fixed_time_solver.solve(
                    new_fix_problem, solution_adapted
                )
            except AssertionError:
                solution = None
                info = {}

            if solution is None:
                print("Solver failed. Reduce t updates to {}".format(update / 2.0))
                j += 1
            else:
                break

        assert solution is not None, ""

        if self.cost_with_time_penalty(solution.cost, t) < self.cost_with_time_penalty(
            previous_cost, previous_t
        ):
            return t, new_fix_problem, solution, info, True

        # Do binary search
        right_index = j
        # Evaluate at left index (right index has cost increase)
        j = left_index
        if j > right_index:
            update = update_trail * np.power(1 / 2, j)
            t = previous_t + update
            new_fix_problem = problem.fix_time_and_step(t, n_steps)
            solution_adapted = self.solution_adapter.adapt(
                previous_solution, previous_fix_time_problem, new_fix_problem
            )
            solution, info = self.fixed_time_solver.solve(
                new_fix_problem, solution_adapted
            )

        if self.cost_with_time_penalty(solution.cost, t) >= self.cost_with_time_penalty(
            previous_cost, previous_t
        ):
            # if cose increase, means cost cannot decrease
            print(
                f"Cost cannot decrease (with dt update {update}). "
                f"Final Gradient is {gd: .2e}. "
                f"Ends."
            )
            return t, new_fix_problem, solution, info, False

        # Now: at left index cost decrease, at right index cost increase
        while True:
            # Binary search
            print(f"Binary search range ({left_index}-{right_index})")
            j = np.floor((left_index + right_index) / 2.0)

            last_t, last_fix_problem, last_solution, last_info = (
                t,
                new_fix_problem,
                solution,
                info,
            )

            update = update_trail * np.power(1 / 2, j)
            t = previous_t + update
            new_fix_problem = problem.fix_time_and_step(t, n_steps)
            solution_adapted = self.solution_adapter.adapt(
                previous_solution, previous_fix_time_problem, new_fix_problem
            )
            solution, info = self.fixed_time_solver.solve(
                new_fix_problem, solution_adapted
            )

            if self.cost_with_time_penalty(
                solution.cost, t
            ) < self.cost_with_time_penalty(previous_cost, previous_t):
                # cost decrease (2^-left - 2^-right)
                left_index = j
                if abs(right_index - left_index) <= 1:
                    return t, new_fix_problem, solution, info, True
            else:
                right_index = j
                if abs(right_index - left_index) <= 1:
                    return last_t, last_fix_problem, last_solution, last_info, True

    def find_decreasing_step(
        self,
        problem: DiscreteFreeTimeOpenLoopProblem,
        previous_t: float,
        previous_solution,
        previous_cost,
        previous_fix_time_problem,
        sign,
        gd,
        max_steps: int,
    ):
        while True:
            t = previous_t + self.estimate_t_step_size(sign, gd, max_steps, problem.dt)
            new_fix_problem: DiscreteFixedTimeOpenLoopProblem = problem.fix_time(t)
            solution_adapted = self.solution_adapter.adapt(
                previous_solution, previous_fix_time_problem, new_fix_problem
            )
            solution, info = self.fixed_time_solver.solve(
                new_fix_problem, solution_adapted
            )
            if max_steps > 1:
                max_steps = max_steps // 2
            else:
                return t, new_fix_problem, solution, info, False

            if solution is None:
                print("Solver failed. Reduce step size to {}".format(max_steps))
                continue
            if self.cost_with_time_penalty(
                solution.cost, t
            ) < self.cost_with_time_penalty(previous_cost, previous_t):
                break
            print("cost increased, reducing max step size to {}".format(max_steps))

        return t, new_fix_problem, solution, info, True

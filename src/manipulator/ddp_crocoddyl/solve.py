import abc
import logging
from typing import Tuple, Optional, List, Type, Union, Dict

import crocoddyl
import numpy as np

from manipulator.cost import CostConfig, CostTypes
from manipulator.data_process.states import (
    ConfigurationOrPoseInnerDataType,
    ConfigurationOrPose,
)
from manipulator.ddp_crocoddyl.discretization import CrocoddylIntegrationDetails
from manipulator.ddp_crocoddyl.simple_cost_models import (
    get_3d_goal_tracking_cost_model,
    get_6d_goal_tracking_cost_model,
    get_state_regularization_cost_model,
    get_control_regularization_cost_model,
    get_gravity_regularization_cost_model,
    get_acceleration_regularization_cost_model,
    get_frame_velocity_regularization_model,
)
from manipulator.ddp_crocoddyl.utils import get_each_cost_components
from manipulator.ddp_crocoddyl.utils import (
    get_state_representation,
    get_running_and_terminal_iam,
)
from manipulator.model import ManipulatorDynamics
from manipulator.open_loop import (
    FixedTimeOpenLoopSolver,
    DiscreteFixedTimeOpenLoopProblem,
    DetailedDiscreteFixedTimeOpenLoopProblem,
    OpenLoopSolution,
)
from manipulator.registry import registry
from manipulator.simple_utils import StopWatch


class ManipulatorDDPSolver:
    """Solver design:
    - create_shooting_problem:
        it will load available cost models from `get_cost_models` function and create the shooting problem to be solved
        Note: initial state is needed when creating shooting problem
    - get_cost_models:
        one should construct the cost models here: running and terminal
    - solve:
        solve the shooting problem
    """

    def __init__(
        self,
        arm: ManipulatorDynamics,
        x0: np.ndarray,
        x1: Union[ConfigurationOrPoseInnerDataType, ConfigurationOrPose],
        time_step: float,
        n_steps: int,
        integration_details: Optional[CrocoddylIntegrationDetails] = None,
    ):
        self.arm = arm
        self.time_step = time_step
        self.n_steps = n_steps
        self.solver_cls = crocoddyl.SolverDDP
        if integration_details is None:
            integration_details = CrocoddylIntegrationDetails()
        self.problem = self.create_shooting_problem(
            integration_details,
            initial_state=x0,
            goal_state=ConfigurationOrPose(x1)
            if not isinstance(x1, ConfigurationOrPose)
            else x1,
        )

    def set_solver(self, solver_cls: Type[crocoddyl.SolverAbstract]):
        self.solver_cls = solver_cls

    @abc.abstractmethod
    def get_cost_models(
        self,
        state: crocoddyl.StateMultibody,
        goal_frame_id: int,
        goal_state: ConfigurationOrPose,
    ) -> Tuple[List[crocoddyl.CostModelAbstract], crocoddyl.CostModelAbstract]:
        raise NotImplementedError

    def solve(
        self,
        max_iter=100,
        verbose=True,
        initial_guess: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None,
        callbacks=None,
        th_stop: float = None,
    ) -> Tuple[bool, crocoddyl.SolverAbstract, crocoddyl.ShootingProblem]:
        ddp = self.solver_cls(self.problem)
        if th_stop is not None:
            ddp.th_stop = th_stop
        callback_list = []
        if callbacks is not None:
            callback_list.extend(callbacks)
        if verbose:
            callback_list.insert(0, crocoddyl.CallbackVerbose())  # noqa
        if len(callback_list) > 0:
            ddp.setCallbacks(callback_list)

        if initial_guess:
            done = ddp.solve(initial_guess[0], initial_guess[1], max_iter)
        else:
            done = ddp.solve([], [], max_iter)
        return done, ddp, self.problem

    def evaluate(
        self,
        xs: Union[List[np.ndarray], np.ndarray],
        us: Union[List[np.ndarray], np.ndarray],
    ):
        return self.problem.calc(list(xs), list(us))

    def create_shooting_problem(
        self,
        integration_details: CrocoddylIntegrationDetails,
        initial_state: np.ndarray,
        goal_state: ConfigurationOrPose,
        benchmark: bool = False,
    ) -> crocoddyl.ShootingProblem:
        clock = StopWatch(active=benchmark)
        state = get_state_representation(self.arm)
        clock.start("CostModels")
        running_cost_models, terminal_cost_model = self.get_cost_models(
            state, self.arm.ee_index, goal_state
        )
        clock.stop("CostModels")
        clock.start("IAMs")
        actuation = crocoddyl.ActuationModelFull(state)
        running_models, terminal_model = get_running_and_terminal_iam(
            state,
            actuation,
            running_cost_models,
            terminal_cost_model,
            self.time_step,
            integration_details=integration_details,
        )
        clock.stop("IAMs")
        if len(running_models) == 1:
            running_models = running_models * self.n_steps
        else:
            assert len(running_models) == self.n_steps
        clock.start("Problem")
        problem = crocoddyl.ShootingProblem(
            initial_state, running_models, terminal_model
        )
        clock.stop("Problem")
        clock.report()
        return problem


class ManipulatorDDPSolverFromCostConfig(ManipulatorDDPSolver):
    cost_tol: float = 1e-12

    def __init__(
        self,
        arm: ManipulatorDynamics,
        x0: np.ndarray,
        x1: ConfigurationOrPoseInnerDataType,
        time_step: float,
        n_steps: int,
        cost_config: Union[dict, CostConfig],
        integration_details: Optional[CrocoddylIntegrationDetails] = None,
    ):
        self.cost_config: CostConfig = CostConfig.from_dict(cost_config)
        super().__init__(arm, x0, x1, time_step, n_steps, integration_details)

    @classmethod
    def from_open_loop_problem(
        cls, open_loop_problem: DiscreteFixedTimeOpenLoopProblem
    ):
        if isinstance(open_loop_problem, DetailedDiscreteFixedTimeOpenLoopProblem):
            if open_loop_problem.integration_details is not None:
                assert isinstance(
                    open_loop_problem.integration_details, CrocoddylIntegrationDetails
                )
        else:
            open_loop_problem = DetailedDiscreteFixedTimeOpenLoopProblem.from_discrete_open_loop_problem(
                open_loop_problem
            )
        return cls(
            open_loop_problem.arm,
            open_loop_problem.x0,
            open_loop_problem.x1.value,
            open_loop_problem.dt,
            open_loop_problem.n_steps,
            open_loop_problem.cost_config,
            open_loop_problem.integration_details,
        )

    def get_cost_model_dict(
        self,
        state: crocoddyl.StateMultibody,
        goal_frame_id: int,
        goal_state: ConfigurationOrPose,
    ):
        cost_model_dict = {
            CostTypes.eeVel: get_frame_velocity_regularization_model(
                state, goal_frame_id
            ),
            CostTypes.ctrlReg: get_control_regularization_cost_model(state),
            CostTypes.gravState: get_gravity_regularization_cost_model(state),
            CostTypes.accReg: get_acceleration_regularization_cost_model(state),
        }
        if goal_state.type == ConfigurationOrPose.Type.Pose:
            cost_model_dict.update(
                {
                    CostTypes.eeTrans: get_3d_goal_tracking_cost_model(
                        state, goal_frame_id, goal_state.pose.translation
                    ),
                    CostTypes.eePose: get_6d_goal_tracking_cost_model(
                        state, goal_frame_id, goal_state.pose
                    ),
                }
            )
        if goal_state.type == ConfigurationOrPose.Type.Configuration:
            pose = self.arm.get_ee(goal_state.configuration)
            cost_model_dict.update(
                {
                    CostTypes.eeTrans: get_3d_goal_tracking_cost_model(
                        state,
                        goal_frame_id,
                        pose.translation,
                    ),
                    CostTypes.stateReg: get_state_regularization_cost_model(
                        state, goal_state.configuration
                    ),
                    CostTypes.ctrlRegGravTf: get_control_regularization_cost_model(
                        state, self.arm.get_torque_grav_ref(goal_state.configuration)
                    ),
                    CostTypes.eePose: get_6d_goal_tracking_cost_model(
                        state, goal_frame_id, pose
                    ),
                }
            )
        return cost_model_dict

    def _is_skip(self, weight: float) -> bool:
        if abs(weight) <= self.cost_tol:
            if weight > 0.0:
                logging.warning(f"Cost weight {weight:.2e} is too small, removing it.")
            return True
        return False

    def get_cost_models(
        self,
        state: crocoddyl.StateMultibody,
        goal_frame_id: int,
        goal_state: ConfigurationOrPose,
    ) -> Tuple[List[crocoddyl.CostModelAbstract], crocoddyl.CostModelAbstract]:
        cost_model_dict = self.get_cost_model_dict(state, goal_frame_id, goal_state)

        # build terminal cost
        terminal_cost_model = crocoddyl.CostModelSum(state)
        for cost_name, cost_weight in self.cost_config.terminal.items():
            if cost_name == CostTypes.general:
                continue
            if self._is_skip(cost_weight):
                continue
            terminal_cost_model.addCost(
                cost_name, cost_model_dict[cost_name], cost_weight
            )
        assert len(self.cost_config.terminal.general) == 0

        # build running costs
        cost_model_general_dict: Dict[int, crocoddyl.CostModelAbstract] = {}
        running_cost_models = []
        assert len(self.cost_config.running) in (1, self.n_steps)
        for running_cost_config in self.cost_config.running:
            running_cost_model = crocoddyl.CostModelSum(state)
            for cost_name, cost_weight in running_cost_config.items():
                if cost_name == CostTypes.general:
                    continue
                if self._is_skip(cost_weight):
                    continue
                running_cost_model.addCost(
                    cost_name, cost_model_dict[cost_name], cost_weight
                )
            # handle general costs
            for cost_name, cost_details in running_cost_config.general.items():
                unique_id = cost_details.unique_id
                if unique_id in cost_model_general_dict:
                    cost_model = cost_model_general_dict[unique_id]
                else:
                    cost_cls = registry.get_cost_model_getter(cost_details.cls_name)
                    cost_model = cost_cls(state, goal_frame_id, goal_state)(
                        **cost_details.cls_kwargs
                    )
                    if unique_id is not None:
                        cost_model_general_dict[unique_id] = cost_model

                running_cost_model.addCost(cost_name, cost_model, cost_details.weight)
            running_cost_models.append(running_cost_model)

        return running_cost_models, terminal_cost_model


class ManipulatorDDPSolverFromCostConfigOverrideTerminalState(
    ManipulatorDDPSolverFromCostConfig
):
    """x1 is used to calculate the reference control force,
    and the achieving_state is set to the approaching state.
    """

    def __init__(
        self,
        arm: ManipulatorDynamics,
        x0: np.ndarray,
        x1: np.ndarray,
        time_step: float,
        n_steps: int,
        cost_config: dict,
        achieving_state: np.ndarray,
    ):
        self.achieving_state = achieving_state
        super().__init__(arm, x0, x1, time_step, n_steps, cost_config)

    def get_cost_model_dict(
        self,
        state: crocoddyl.StateMultibody,
        goal_frame_id: int,
        goal_state: ConfigurationOrPose,
    ):
        cost_model_dict = super().get_cost_model_dict(state, goal_frame_id, goal_state)
        cost_model_dict[CostTypes.stateReg] = get_state_regularization_cost_model(
            state, self.achieving_state
        )
        return cost_model_dict


class DDPOpenLoopSolver(FixedTimeOpenLoopSolver):
    def __init__(
        self,
        solver_cls: Type[crocoddyl.SolverAbstract] = crocoddyl.SolverDDP,
        return_solver_and_problem=True,
        build_cost_dict=False,
        **solve_kwargs,
    ):
        self.solver_cls = solver_cls
        self.solve_kwargs = solve_kwargs
        self.solve_kwargs.setdefault("verbose", False)

        self.return_solver_and_problem = return_solver_and_problem
        self.build_cost_dict = build_cost_dict

    def solve(
        self,
        problem: DiscreteFixedTimeOpenLoopProblem,
        initial_guess: Optional[OpenLoopSolution],
        **kwargs,
    ) -> Tuple[Optional[OpenLoopSolution], dict]:
        solver = ManipulatorDDPSolverFromCostConfig.from_open_loop_problem(problem)
        solver.set_solver(self.solver_cls)
        solve_kwargs = self.solve_kwargs.copy()
        solve_kwargs.update(kwargs)

        is_converged, ddp, problem = solver.solve(
            initial_guess=(initial_guess.xs, initial_guess.us)
            if initial_guess
            else None,
            **solve_kwargs,
        )
        solution = OpenLoopSolution.from_ddp(ddp) if is_converged else None
        info = {}
        if self.return_solver_and_problem:
            info["problem"] = problem
            info["ddp"] = ddp
            if self.build_cost_dict:
                info["costs_dict"] = get_each_cost_components(problem)

        return solution, info

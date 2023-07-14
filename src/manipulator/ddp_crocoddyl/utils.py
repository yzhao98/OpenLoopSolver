from collections import defaultdict
from typing import Tuple, List

import crocoddyl
import numpy as np
import pinocchio as pin
from crocoddyl import ShootingProblem

from manipulator.ddp_crocoddyl.discretization import CrocoddylIntegrationDetails
from manipulator.model import ManipulatorDynamics


def get_integrated_action_model(
    state: crocoddyl.StateAbstract,
    dt: float,
    cost_model: crocoddyl.CostModelAbstract,
    integration_details: CrocoddylIntegrationDetails = CrocoddylIntegrationDetails(),
    actuation: crocoddyl.ActuationModelAbstract = None,
):
    if actuation is None:
        actuation = crocoddyl.ActuationModelFull(state)
    differential_action_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, cost_model  # Free here means constraints-free
    )
    model = integration_details.construct_integrated_action_model(
        differential_action_model, dt
    )
    return model


def get_actuation_multi_body_data(state: crocoddyl.StateMultibody):
    actuation = crocoddyl.ActuationModelFull(state)
    act_data = actuation.createData()
    pin_data = pin.Data(state.pinocchio)
    multi_body_data = crocoddyl.DataCollectorActMultibody(pin_data, act_data)
    return multi_body_data


def get_state_representation(
    manipulator: ManipulatorDynamics,
) -> crocoddyl.StateMultibody:
    state = crocoddyl.StateMultibody(manipulator.model)
    return state


def get_running_and_terminal_iam(
    state: crocoddyl.StateMultibody,
    actuation: crocoddyl.ActuationModelAbstract,
    running_cost_models: List[crocoddyl.CostModelAbstract],
    terminal_cost_model: crocoddyl.CostModelAbstract,
    dt: float,
    integration_details: CrocoddylIntegrationDetails,
) -> Tuple[
    List[crocoddyl.IntegratedActionModelAbstract],
    crocoddyl.IntegratedActionModelAbstract,
]:
    running_models = [
        get_integrated_action_model(
            state,
            dt,
            cost_model=cost_model,
            integration_details=integration_details,
            actuation=actuation,
        )
        for cost_model in running_cost_models
    ]
    # dt for terminal model has no effect
    # only calc without control is being called for terminal model
    # which has no dt factor
    terminal_model = get_integrated_action_model(
        state,
        0.0,
        cost_model=terminal_cost_model,
        integration_details=integration_details,
        actuation=actuation,
    )
    return running_models, terminal_model


def get_each_cost_components(problem: ShootingProblem):
    running_datas = problem.runningDatas
    running_models = problem.runningModels
    costs = defaultdict(lambda: 0.0)
    for i, (data, model) in enumerate(zip(running_datas, running_models)):
        costs_dict = data.differential.costs.costs.todict()
        costs_item_dict = model.differential.costs.costs.todict()
        for key, value in costs_dict.items():
            costs[key] += value.cost * model.dt * costs_item_dict[key].weight
    costs["terminal"] = problem.terminalData.cost
    return costs


def get_cost_components_of_given_sols(xs, us, problem: ShootingProblem):
    if isinstance(xs, np.ndarray):
        xs = list(xs)
    if isinstance(us, np.ndarray):
        us = list(us)

    _ = problem.calc(xs, us)
    costs_dict = get_each_cost_components(problem)
    return costs_dict

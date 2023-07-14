from functools import wraps
from typing import Callable, Optional, Any

import crocoddyl
import numpy as np
import pinocchio as pin

from manipulator.ddp_crocoddyl.residual_models import (
    ResidualModelGrav,
    ResidualModelAcc,
)


def from_residual_to_cost_model(
    f: Callable[
        [crocoddyl.StateAbstract, Optional[Any]], crocoddyl.ResidualModelAbstract
    ]
):
    """The default activation applied on the residual is the function 0.5||r||^2"""

    @wraps(f)
    def get_cost_model(
        state: crocoddyl.StateAbstract,
        *args,
        activation: crocoddyl.ActivationModelAbstract = None,
        **kwargs,
    ) -> crocoddyl.CostModelResidual:
        residual_model: crocoddyl.ResidualModelAbstract = f(state, *args, **kwargs)
        if activation is None:
            return crocoddyl.CostModelResidual(state, residual_model)
        return crocoddyl.CostModelResidual(
            state,
            activation,
            residual_model,
        )

    return get_cost_model


@from_residual_to_cost_model
def get_3d_goal_tracking_cost_model(
    state: crocoddyl.StateMultibody, goal_index: int, goal_value: np.ndarray
) -> crocoddyl.CostModelResidual:
    goal_res = crocoddyl.ResidualModelFrameTranslation(state, goal_index, goal_value)
    return goal_res


@from_residual_to_cost_model
def get_6d_goal_tracking_cost_model(
    state: crocoddyl.StateMultibody, goal_index: int, goal_value: pin.SE3
) -> crocoddyl.CostModelResidual:
    goal_res = crocoddyl.ResidualModelFramePlacement(state, goal_index, goal_value)
    return goal_res


@from_residual_to_cost_model
def get_state_regularization_cost_model(
    state: crocoddyl.StateAbstract,
    x_ref: np.ndarray = None,
) -> crocoddyl.CostModelResidual:
    if x_ref is None:
        return crocoddyl.ResidualModelState(state)
    return crocoddyl.ResidualModelState(state, x_ref)


@from_residual_to_cost_model
def get_control_regularization_cost_model(
    state: crocoddyl.StateAbstract, u_ref: np.ndarray = None
) -> crocoddyl.CostModelResidual:
    if u_ref is None:
        return crocoddyl.ResidualModelControl(state)
    return crocoddyl.ResidualModelControl(state, u_ref)


@from_residual_to_cost_model
def get_gravity_regularization_cost_model(
    state: crocoddyl.StateMultibody,
):
    """This cost model is a(g(q)) here g(q) is the generalized gravity"""
    return ResidualModelGrav(state)


@from_residual_to_cost_model
def get_acceleration_regularization_cost_model(
    state: crocoddyl.StateMultibody,
):
    return ResidualModelAcc(state)


@from_residual_to_cost_model
def get_gravity_control_cost_model(
    state: crocoddyl.StateMultibody,
):
    """This cost model is a(u - g(q)) here g(q) is the generalized gravity"""
    return crocoddyl.ResidualModelControlGrav(state)


@from_residual_to_cost_model
def get_frame_velocity_regularization_model(
    state: crocoddyl.StateMultibody, frame_id: int
):
    return crocoddyl.ResidualModelFrameVelocity(
        state, frame_id, pin.Motion.Zero(), pin.ReferenceFrame.WORLD
    )

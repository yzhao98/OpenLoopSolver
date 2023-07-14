import dataclasses
from typing import Type, Union

import crocoddyl

from manipulator.open_loop import IntegrationDetails


@dataclasses.dataclass(frozen=True)
class CrocoddylIntegrationDetails(IntegrationDetails):
    integrator_type: Union[
        Type[crocoddyl.IntegratedActionModelAbstract], str
    ] = crocoddyl.IntegratedActionModelEuler
    integrator_kwargs: dict = dataclasses.field(default_factory=dict)
    control_model: crocoddyl.ControlParametrizationModelAbstract = None

    def __post_init__(self):
        if isinstance(self.integrator_type, str):
            if self.integrator_type in ("rk4",):
                object.__setattr__(
                    self, "integrator_type", crocoddyl.IntegratedActionModelRK
                )
                self.integrator_kwargs.update({"rktype": crocoddyl.RKType.four})

    def construct_integrated_action_model(
        self,
        differential_action_model: crocoddyl.DifferentialActionModelAbstract,
        dt: float,
    ) -> crocoddyl.IntegratedActionModelAbstract:
        if self.control_model is not None:
            model = self.integrator_type(
                diffModel=differential_action_model,
                control=self.control_model,
                stepTime=dt,
                **self.integrator_kwargs,
            )
        else:
            model = self.integrator_type(
                diffModel=differential_action_model,
                stepTime=dt,
                **self.integrator_kwargs,
            )
        return model

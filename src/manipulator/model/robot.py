import time
from typing import List, Tuple, Union

import numpy as np
import pinocchio as pin

from manipulator.constants import PreconfiguredModels, ModelDefinitionConfig
from manipulator.model.dynamics import (
    DifferentiableDynamics,
    DynamicsWithEE,
    IntegrableDynamics,
)
from manipulator.model.pinocchio_wrapper import PinocchioWrapper


class ManipulatorDynamics(DifferentiableDynamics, DynamicsWithEE, IntegrableDynamics):
    """
    The state consists of all joints configurations of the robot and their velocities.
    x = (q, v)

    """

    @classmethod
    def build(cls, config_name: Union[str, PreconfiguredModels] = None):
        if config_name is None:
            config_name = PreconfiguredModels.default.name
        if isinstance(config_name, str):
            config_name = PreconfiguredModels[config_name]
        model_config: ModelDefinitionConfig = config_name.value  # noqa
        return cls(
            model_config.model_file_path,
            model_config.dir_containing_pkg,
            model_config.ee_name,
            configured_model=config_name,
        )

    def __init__(
        self,
        urdf_path,
        dir_containing_pkg,
        ee_name,
        configured_model: PreconfiguredModels = None,
    ):
        self.pinocchio_wrapper: PinocchioWrapper = PinocchioWrapper.build_model(
            urdf_path, dir_containing_pkg
        )
        self.nq = self.pinocchio_wrapper.model.nq
        self.nv = self.pinocchio_wrapper.model.nv
        assert self.pinocchio_wrapper.model.nq == self.pinocchio_wrapper.model.nv
        self.ee_name = ee_name
        self.ee_index = self.pinocchio_wrapper.model.getFrameId(self.ee_name)
        self.q0 = pin.neutral(self.model)
        self.configured_model = configured_model

    def get_generalized_coordinates(self, x: np.ndarray) -> np.ndarray:
        return x[: self.nq]

    def get_torque_grav_ref(self, x: np.ndarray):
        q, v = self.position_velocity_split(x)
        assert (
            np.linalg.norm(v) <= 1e-8
        ), "Stable reference torque that counteracts gravity only support for zero velocity"
        return self.pinocchio_wrapper.get_torques_counter_gravity(q)

    def random_state(self) -> np.ndarray:
        q = pin.randomConfiguration(self.model)
        v = 2 * np.random.rand(self.nq) - 1
        return np.r_[q, v]

    def position_velocity_split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x[: self.nq], x[self.nq :]

    @property
    def nu(self) -> int:
        return self.nq

    @property
    def nx(self) -> int:
        return self.nq + self.nq

    @property
    def model(self) -> pin.Model:
        return self.pinocchio_wrapper.model

    @property
    def data(self) -> pin.Data:
        return self.pinocchio_wrapper.data

    def display(self, q) -> None:
        self.pinocchio_wrapper.display(q)

    def display_solutions(self, xs: List[np.ndarray], dts: List[float]):
        assert len(xs) == len(dts)
        for x, dt in zip(xs, dts):
            self.display(self.get_generalized_coordinates(x))
            time.sleep(dt)

    def get_value(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        q, v = self.position_velocity_split(x)
        a = self.pinocchio_wrapper.aba(q, v, u)
        return np.r_[v, a]

    def get_tau(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return self.pinocchio_wrapper.rnea(q, v, a)

    def get_derivative(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        q, v = self.position_velocity_split(x)
        self.pinocchio_wrapper.aba_with_derivatives(q, v, u)
        df_dx = np.empty(shape=(self.nx, self.nx), dtype=x.dtype)
        df_du = np.zeros(shape=(self.nx, self.nu), dtype=x.dtype)

        df_dx[: self.nq, : self.nq] = 0.0
        df_dx[: self.nq, self.nq :] = np.identity(self.nq)
        df_dx[self.nq :, : self.nq] = self.pinocchio_wrapper.da_dq
        df_dx[self.nq :, self.nq :] = self.pinocchio_wrapper.da_dv

        df_du[: self.nq, :] = 0.0
        df_du[self.nq :, :] = self.pinocchio_wrapper.da_du

        return df_dx, df_du

    def da(self, x, u):
        df_dx, df_du = self.get_derivative(x, u)
        da_dx = df_dx[self.nq :]
        da_du = df_du[self.nq :]
        return da_dx, da_du

    def get_linear_dynamics(
        self, x0: np.ndarray, u0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_derivative(x0, u0)

    def dx_symplectic_euler(self, dx_dt: np.ndarray, dt: float) -> np.ndarray:
        v, a = self.position_velocity_split(dx_dt)
        dv = a * dt
        dp = (v + dv) * dt
        return np.r_[dp, dv]

    def step_forward(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        dx_dt = self.get_value(x, u)
        dx = self.dx_symplectic_euler(dx_dt, dt)
        return x + dx

    def get_ee(self, x: np.ndarray) -> pin.SE3:
        return self.get_frame(x, self.ee_index)

    def get_frame(self, x, frame_id) -> pin.SE3:
        q = self.get_generalized_coordinates(x)
        return self.pinocchio_wrapper.get_frame(q, frame_id)


class ManipulatorDynamicsOnly(
    DifferentiableDynamics, IntegrableDynamics, DynamicsWithEE
):
    """This class serves as to only expose certain functions."""

    def __init__(self, manipulator: ManipulatorDynamics):
        self._manipulator = manipulator

    def get_value(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._manipulator.get_value(x, u)

    def get_ee(self, x: np.ndarray) -> pin.SE3:
        return self._manipulator.get_ee(x)

    @property
    def nu(self) -> int:
        return self._manipulator.nu

    @property
    def nx(self) -> int:
        return self._manipulator.nx

    def step_forward(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        return self._manipulator.step_forward(x, u, dt)

    def get_linear_dynamics(
        self, x0: np.ndarray, u0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._manipulator.get_linear_dynamics(x0, u0)

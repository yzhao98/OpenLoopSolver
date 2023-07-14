import abc
from typing import Tuple

import numpy as np
import pinocchio as pin


class Dynamics:
    r""" Abstract dynamics class
    Dynamics should define:
    - get_value: compute \dot{x} = f(x,u)
    - step_forward: do one-step simulation x[n] -> x[n+1]
    """

    @abc.abstractmethod
    def get_value(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def nu(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def nx(self) -> int:
        pass


class DynamicsWithEE(Dynamics):
    @abc.abstractmethod
    def get_ee(self, x: np.ndarray) -> pin.SE3:
        pass


class IntegrableDynamics(Dynamics):
    """ Implement the function step_forward to integrate forward """
    @abc.abstractmethod
    def step_forward(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        pass


class DifferentiableDynamics(Dynamics):
    """ In addition to Dynamics, DifferentiableDynamics needs:
        - get_linear_dynamics: f(x,u) = f_x(x0,u0)(x-x0) + f_u(x0,u0)(u-u0); This function should return the matrices
            f_x, f_u
    """
    @abc.abstractmethod
    def get_linear_dynamics(
        self, x0: np.ndarray, u0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

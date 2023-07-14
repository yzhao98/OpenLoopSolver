from typing import List

import crocoddyl
import numpy as np
import pinocchio as pin
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import Diagram

from manipulator.constants import PreconfiguredModels


# Don't inherit other classes that implemented calc and calcDiff function
# Boost::python does not support inherit C++ class in python (it cannot handle function hiding)
# https://github.com/loco-3d/crocoddyl/issues/1032
# when calling, they will be called instead of being overridden
class ResidualModelGrav(crocoddyl.ResidualModelAbstract):
    def __init__(self, state: crocoddyl.StateMultibody):
        crocoddyl.ResidualModelAbstract.__init__(
            self,
            state,
            state.nq,
            True,
            False,
            False
            # stateAbstract, nr: dimension of residuals, whether depends on q, v, u
        )
        self.pin_model = state.pinocchio

    def calc(self, data, x, u):  # noqa
        q = x[: self.state.nq]
        data.r[:] = pin.computeGeneralizedGravity(self.pin_model, data.pinocchio, q)

    def calcDiff(self, data, x, u):  # noqa
        q = x[: self.state.nq]
        Rq = pin.computeGeneralizedGravityDerivatives(self.pin_model, data.pinocchio, q)
        data.Rx[:, : self.state.nq] = Rq

    def createData(self, data):  # noqa
        return ResidualDataGrav(self, data)


class ResidualModelAcc(crocoddyl.ResidualModelAbstract):
    def __init__(self, state: crocoddyl.StateMultibody):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, state.nq, True, True, True
        )
        self.pin_model = state.pinocchio

    def calc(self, data, x, u):  # noqa
        # the acceleration data is already available in data.pinocchio.ddq
        data.r = data.pinocchio.ddq

    def calcDiff(self, data, x, u):  # noqa
        # the acceleration derivatives is not available in data.pinocchio
        # as ActionModel's calcDiff does not store the derivatives into data.pinocchio when computing
        # data.pinocchio.ddq_dq, data.pinocchio.ddq_dv, data.pinocchio.Minv
        # https://github.com/loco-3d/crocoddyl/issues/1119
        q = x[: self.state.nq]
        v = x[self.state.nq :]
        pin.computeABADerivatives(self.pin_model, data.pinocchio, q, v, u)
        Rq = data.pinocchio.ddq_dq
        Rv = data.pinocchio.ddq_dv
        Ru = data.pinocchio.Minv
        data.Rx[:, : self.state.nq] = Rq
        data.Rx[:, self.state.nq :] = Rv
        data.Ru[:] = Ru

    def createData(self, data):  # noqa
        return ResidualDataAcc(self, data)


class ResidualModelPosition(crocoddyl.ResidualModelAbstract):
    def __init__(self, state: crocoddyl.StateMultibody, q_ref: np.ndarray):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, state.nq, True, False, False
        )
        self.q_ref = q_ref

    def calc(self, data, x, u):  # noqa
        # self.state.diff: can be used to compute the rate of change (vector in tangent space)
        data.r = x[: self.state.nq] - self.q_ref

    def calcDiff(self, data, x, u):  # noqa
        data.Rx[:, : self.state.nq] = np.identity(self.state.nq)


class ResidualDataAcc(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ResidualDataAbstract.__init__(self, model, data)
        self.pinocchio = data.pinocchio


class ResidualDataGrav(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ResidualDataAbstract.__init__(self, model, data)
        self.pinocchio = data.pinocchio

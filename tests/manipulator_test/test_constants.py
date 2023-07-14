import numpy as np
import pytest

from manipulator.constants import PreconfiguredModels
from manipulator.model import ManipulatorDynamics


# test all PreconfiguredModels
@pytest.mark.parametrize("model_config", (config for config in PreconfiguredModels))
def test_ik_module(model_config: PreconfiguredModels):
    if model_config == PreconfiguredModels.drake_iiwa14:
        return
    robot = ManipulatorDynamics.build(model_config)
    model_config = model_config.value
    ik_spec = model_config.ik_module_spec
    if ik_spec is None:
        return
    ik_fn = ik_spec.get_ik_fn()
    fk_fn = ik_spec.get_fk_fn()
    assert ik_fn is not None
    assert fk_fn is not None

    import pinocchio as pin

    pin.seed(0)
    q = robot.random_state()[: robot.nq]
    trans, rot = fk_fn(q)

    pose = robot.get_ee(q)
    trans_robot = pose.translation.tolist()
    rot_robot = pose.rotation.tolist()
    assert np.allclose(trans_robot, trans)
    assert np.allclose(rot_robot, rot)

    qs = ik_fn(trans, rot, [q[i] for i in ik_spec.free_joint_indices])
    ees = [robot.get_ee(np.r_[qq, np.zeros(robot.nv)]) for qq in qs]
    trans_new = [ee.translation for ee in ees]
    rot_new = [ee.rotation for ee in ees]

    for t, r in zip(trans_new, rot_new):
        assert np.allclose(t, trans)
        assert np.allclose(r, rot)

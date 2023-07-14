import dataclasses
import enum
from typing import Union

import numpy as np
import pinocchio as pin

ConfigurationOrPoseInnerDataType = Union[np.ndarray, pin.SE3]


@dataclasses.dataclass
class ConfigurationOrPose:
    value: ConfigurationOrPoseInnerDataType

    class Type(str, enum.Enum):
        Configuration = "Configuration"
        Pose = "Pose"

    @property
    def type(self):
        if isinstance(self.value, np.ndarray):
            return self.Type.Configuration
        elif isinstance(self.value, pin.SE3):
            return self.Type.Pose
        else:
            raise ValueError(f"Unknown type {type(self.value)}")

    @property
    def configuration(self):
        assert self.type == self.Type.Configuration
        return self.value

    @property
    def pose(self):
        assert self.type == self.Type.Pose
        return self.value

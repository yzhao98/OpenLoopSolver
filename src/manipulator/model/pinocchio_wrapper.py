import dataclasses
from typing import Optional

import numpy as np
import pinocchio as pin


@dataclasses.dataclass
class PinocchioWrapper:
    model: pin.Model
    data: pin.Data
    visual_model: pin.GeometryModel
    collision_model: pin.GeometryModel
    viewer: pin.visualize.MeshcatVisualizer = dataclasses.field(init=False)
    collision_data: Optional[pin.GeometryData] = dataclasses.field(init=False)

    def __post_init__(self):
        self.viewer = pin.visualize.MeshcatVisualizer(
            self.model,
            visual_model=self.visual_model,
            collision_model=self.collision_model,
        )
        self.collision_data = None

    def init_viewer(self, q=None) -> None:
        q = q if q is not None else pin.neutral(self.model)
        self.viewer.initViewer(loadModel=True)
        self.display(q)

    def display(self, q) -> None:
        self.viewer.display(q)

    def aba(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return pin.aba(self.model, self.data, q, v, tau)

    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return pin.rnea(self.model, self.data, q, v, a)

    def aba_with_derivatives(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray
    ) -> None:
        """Information are stored in data"""
        pin.computeABADerivatives(self.model, self.data, q, v, tau)
        return None

    @property
    def da_dq(self) -> np.ndarray:
        return self.data.ddq_dq

    @property
    def da_dv(self) -> np.ndarray:
        return self.data.ddq_dv

    @property
    def da_du(self) -> np.ndarray:
        return self.data.Minv

    def get_frame(self, q: np.ndarray, frame_id) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        # position data also available in self.data.oMf[self.ee_index]
        return pin.updateFramePlacement(self.model, self.data, frame_id)

    def is_collided(self, q: np.ndarray) -> bool:
        if self.collision_data is None:
            self.collision_data = self.collision_model.createData()  # noqa
        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.collision_data, q
        )
        collided = pin.computeCollisions(
            self.collision_model, self.collision_data, True
        )  # stopAtFirstCollision = True
        return collided

    def get_torques_counter_gravity(self, q: np.ndarray) -> np.ndarray:
        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q)
        return tau_g

    @classmethod
    def build_model(cls, model_path, dir_containing_pkg) -> "PinocchioWrapper":
        model, visual_model, collision_model = pin.buildModelsFromUrdf(
            model_path,
            package_dirs=[dir_containing_pkg],
            geometry_types=[pin.GeometryType.VISUAL, pin.GeometryType.COLLISION],
        )
        return cls(
            model=model,
            data=model.createData(),
            visual_model=visual_model,
            collision_model=collision_model,
        )

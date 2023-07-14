import dataclasses
import enum
import functools
import os
import pathlib
import warnings
from importlib import import_module
from typing import Tuple, List

import numpy as np
import pandas as pd
from pinocchio import SE3, Quaternion
from pydrake.common import FindResourceOrThrow

PKG_PATH = pathlib.Path(__file__).parent
PROJECT_PATH = PKG_PATH.parent.parent

# override resources path from env
RESOURCES_PATH = PROJECT_PATH.joinpath("resources")
RESOURCES_PATH = pathlib.Path(
    os.environ.get("MANIPULATOR_RESOURCES_PATH", RESOURCES_PATH)
)
# override resources path from env
DATA_PATH = RESOURCES_PATH.joinpath("data")
DATA_PATH = pathlib.Path(os.environ.get("MANIPULATOR_DATA_PATH", DATA_PATH))

MODEL_PATH = RESOURCES_PATH.joinpath("model")

ROBOT_PATH = MODEL_PATH.joinpath("iiwa_description")
DRAKE_MOD_PATH = MODEL_PATH.joinpath("drake-mod")
INITIAL_STATES_FILE_NEW_PATTERN = "initial_states_%s_%s_%d.npy"
COLLECTED_DATA_FILE_NEW_PATTERN = "collected_data_%s_%s_%d.pkl"
FREE_TIME_SOLUTION_FILE_PATTERN = "free_time_%s_%s_%d.pkl"


DRAKE_SHARE_PATH = pathlib.Path("/opt/drake/share")
DRAKE_PATH = MODEL_PATH.joinpath("drake")

DEFAULT_MAX_DDP_ITERATIONS = 200

# This is not yet implemented by tqdm officially
# see: https://github.com/tqdm/tqdm/pull/950 and https://github.com/tqdm/tqdm/issues/614
TQDM_DISABLE = False
if os.environ.get("TQDM_DISABLE", False):
    TQDM_DISABLE = True


@dataclasses.dataclass
class WeldInfo:
    child: str
    xyz_rpy: Tuple[float, float, float, float, float, float] = None
    _parent: str = None  # None means WorldBody

    @property
    def parent(self):
        if self._parent is not None:
            raise ValueError(
                "Welding to frame other than world frame is not supported."
            )
        return self._parent


@dataclasses.dataclass
class IKModuleSpec:
    # ik_module_spec = (ik_module: str=module_name, free_joint_indices: List[int])
    # currently, we specify module name here as there is no well-established lazy module library in Python
    # the ik_module must contain a function named `get_ik` with signature:
    # (translation, rotation represented as SO3 matrix as a list of lists, free joint values) -> all joint values
    module_name: str
    free_joint_indices: List[int]
    # the ik_module is built to solve a specific link
    # for other links of interest, we need to apply a rigid transform
    # given pose of link a, wXa, i.e., pose of link a in world frame => wXb = wXa * aXb
    # given pose of link b, wXb, i.e., pose of link b in world frame => wXa = wXb * aXb.inverse()
    aXb: SE3 = None  # a: link of ik_module, b: link of interest

    @functools.cached_property
    def bXa(self):
        return self.aXb.inverse()  # noqa

    @functools.cached_property
    def module(self):
        return import_module(self.module_name)

    def get_ik_fn(self):
        if self.aXb is None:
            return self.module.get_ik

        def get_ik(translation, rotation, free_joint_values):
            pose = SE3(np.array(rotation), np.array(translation).reshape(-1, 1))  # wXb
            pose = pose * self.bXa  # wXa
            return self.module.get_ik(
                pose.translation.tolist(), pose.rotation.tolist(), free_joint_values
            )

        return get_ik

    def get_fk_fn(self):
        if self.aXb is None:
            return self.module.get_fk

        def get_fk(joint_values):
            trans, rotation = self.module.get_fk(joint_values)
            pose = SE3(np.array(rotation), np.array(trans).reshape(-1, 1))  # wXa
            pose = pose * self.aXb  # wXb
            return pose.translation.tolist(), pose.rotation.tolist()

        return get_fk


@dataclasses.dataclass
class ModelDefinitionConfig:
    # the path to model description file
    _model_file_path: str
    # this is used by Pinocchio: it will search package
    # (i.e., folder whose name is the same as package name) in this directory
    dir_containing_pkg: str
    # To load this model in Drake, one should configure the Parser as:
    # parser.package_map().Add(model_conf.pkg_name, model_conf.pkg_dir)
    pkg_dir: str  # the path to the package, used by Drake
    pkg_name: str  # This is the package name in the model description file when referring to meshes
    ee_name: str
    warn_msg: str = None
    weld_info: WeldInfo = None
    ik_module_spec: IKModuleSpec = None

    @property
    def model_file_path(self) -> str:
        if self.warn_msg:
            warning_msg = f"Model at {self._model_file_path}: {self.warn_msg}"
            warnings.warn(UserWarning(warning_msg))
        return self._model_file_path

    def print_to_str(self):
        return f"Model at: {self.model_file_path}"


class PreconfiguredModels(enum.Enum):
    # Since Drake v1.15.0, models data_process file are not ship with Drake.
    drake_iiwa14 = ModelDefinitionConfig(
        FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/urdf/iiwa14_no_collision.urdf"
        ),
        # from pydrake.multibody.parsing import PackageMap
        # PackageMap().GetPath("drake_models") returns the path to the drake_models package folder.
        # The folder is not named `drake_models` but some hash value.
        # Therefore, we can't provide the path to Pinocchio so that it can find the package `drake_models` in that path.
        # We thus disable loading this model using Pinocchio.
        "",
        pkg_dir="",
        pkg_name="",
        ee_name="iiwa_link_ee_kuka",
        weld_info=WeldInfo(
            "base",
        ),
    )
    drake_iiwa14_mod = ModelDefinitionConfig(
        str(DRAKE_MOD_PATH.joinpath("urdf", "iiwa14_no_collision.urdf")),
        str(DRAKE_MOD_PATH.parent),
        pkg_dir=str(DRAKE_MOD_PATH),
        pkg_name="drake-mod",
        ee_name="iiwa_link_ee_kuka",
        weld_info=WeldInfo(
            "base",
        ),
    )
    drake_iiwa14_mod_alpha = ModelDefinitionConfig(
        str(DRAKE_MOD_PATH.joinpath("urdf", "iiwa14_no_collision_alpha.urdf")),
        str(DRAKE_MOD_PATH.parent),
        pkg_dir=str(DRAKE_MOD_PATH),
        pkg_name="drake-mod",
        ee_name="iiwa_link_ee_kuka",
        weld_info=WeldInfo(
            "base",
        ),
    )
    planar = ModelDefinitionConfig(
        str(DRAKE_MOD_PATH.joinpath("urdf", "planar_iiwa14_no_collision.urdf")),
        str(DRAKE_MOD_PATH.parent),
        pkg_dir=str(DRAKE_MOD_PATH),
        pkg_name="drake-mod",
        ee_name="iiwa_link_ee_kuka",
        weld_info=WeldInfo(
            "base",
        ),
    )
    drake_iiwa14_mod_naive_collisions = ModelDefinitionConfig(
        str(DRAKE_MOD_PATH.joinpath("urdf", "iiwa14_naive_collision.urdf")),
        str(DRAKE_MOD_PATH.parent),
        pkg_dir=str(DRAKE_MOD_PATH),
        pkg_name="drake-mod",
        ee_name="iiwa_link_ee_kuka",
        weld_info=WeldInfo(
            "base",
        ),
    )


@dataclasses.dataclass(eq=False)
class ReachingConfigWithWaypoints:
    nq: int
    waypoints: List[np.ndarray]
    target_pose: SE3 = None

    def _complete_state(self, x: np.ndarray):
        if len(x) == self.nq:
            return np.r_[x, np.zeros(self.nq)]
        return x

    def __post_init__(self):
        self.waypoints = [self._complete_state(x) for x in self.waypoints]

    @property
    def initial(self):
        return self.waypoints[0]

    @property
    def terminal(self):
        if self.target_pose is not None:
            return self.target_pose
        return self.waypoints[-1]

    @property
    def q_initial(self):
        return self.initial[: self.nq]

    @property
    def q_terminal(self):
        return self.terminal[: self.nq]

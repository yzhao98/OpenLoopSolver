from typing import Dict, Any

import numpy as np

from manipulator.model import ManipulatorDynamics
from manipulator.data import (
    sample_initial_states_based_on_ee,
    sample_initial_states_simple,
    sample_initial_states_cube,
    ReachingSetup,
)


def get_sampler_ee_based(arm: ManipulatorDynamics, radius: float, center: np.ndarray):
    def sampler(n_samples: int):
        return sample_initial_states_based_on_ee(arm, n_samples, radius, center)

    return sampler


def get_sampler_simple(arm: ManipulatorDynamics, radius: float, center: np.ndarray):
    def sampler(n_samples: int):
        return sample_initial_states_simple(arm, n_samples, radius, center)

    return sampler


def get_sampler_cube(
    arm: ManipulatorDynamics, half_side: float, center: np.ndarray, **sampler_kwargs
):
    def sampler(n_samples: int):
        return sample_initial_states_cube(
            arm, n_samples, half_side, center, **sampler_kwargs
        )

    return sampler


def get_predefined_config(arm: ManipulatorDynamics, name):
    def get_positions_from_states(x):
        return arm.get_generalized_coordinates(x)

    x0, x1 = ReachingSetup.default.get_states()
    x0h, x1h = ReachingSetup.hard.get_states()
    x0r, x1r = ReachingSetup.reflect.get_states()
    name = name.split(".")[0]
    configs = {
        "setup1": {
            "center": np.array([-0.395, -0.117, 0.700]),
            "radius": 0.1,
            "max_v": 1.0,
            "dt": 1e-3,
            "sampling_freq": 1,
        },
        "setup2": {
            "center": np.array([-0.395, -0.117, 0.700]),
            "radius": 0.1,
            "max_v": 1.0,
            "dt": 1e-2,
            "sampling_freq": 1,
        },
        "setup3": {
            "center": np.array([-0.395, -0.117, 0.700]),
            "radius": 0.1,
            "max_v": 0.5,
            "dt": 0.001,
            "sampling_freq": 20,
        },
        "setup4": {
            "center": np.array([-0.395, -0.117, 0.700]),
            "radius": 0.1,
            "max_v": 0.5,
            "dt": 0.001,
            "sampling_freq": 1,
        },
        "setup5": {
            "center": get_positions_from_states(x0),
            "radius": 0.1,
            "max_v": 1.0,
            "dt": 0.001,
            "sampling_freq": 1,
            "sampler": "simple",
        },
        "setup6": {
            "center": get_positions_from_states((1 - 0.6) * x0 + 0.6 * x1),
            "radius": 0.1,
            "max_v": 0.5,
            "dt": 0.001,
            "sampling_freq": 1,
            "sampler": "simple",
        },
        "setup7": {
            "center": get_positions_from_states(x0),
            "radius": 0.04,
            "max_v": 1.0,
            "dt": 0.001,
            "sampling_freq": 1,
            "sampler": "simple",
        },
        "setup8": {
            "center": get_positions_from_states(x0),
            "radius": 0.01,
            # for cube, this is half side-length
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup108": {
            "center": get_positions_from_states(x0h),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 1.0,
            "sampling_freq": 1,
            "sampler": "cube",
            "goal": x1h,
        },
        "setup107": {
            "center": get_positions_from_states(x0h),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
            "goal": x1h,
        },
        "setup9": {
            "center": get_positions_from_states(x0),
            "radius": 0.001,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup10": {
            "center": get_positions_from_states(x0),
            "radius": 0.05,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup22": {
            "center": get_positions_from_states(x0),
            "radius": 0.2,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup11": {
            "center": get_positions_from_states((1 - 0.6) * x0 + 0.6 * x1),
            "radius": 0.04,
            "dt": 0.001,
            "t0": 0.40,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup12": {
            "center": get_positions_from_states((1 - 0.6) * x0 + 0.6 * x1),
            "radius": 0.04,
            "dt": 0.0001,
            "t0": 0.40,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup13": {
            "center": get_positions_from_states((1 - 0.6) * x0 + 0.6 * x1),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.40,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup113": {
            # 113 has initial state reflected from 13 around goal state
            "center": get_positions_from_states(x0r),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.40,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup14": {
            "center": get_positions_from_states((1 - 0.6) * x0 + 0.6 * x1),
            "radius": 0.04,
            "dt": 0.001,
            "t0": 0.40,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup15": {
            "center": get_positions_from_states((1 - 0.6) * x0 + 0.6 * x1),
            "radius": 0.1,
            "dt": 0.001,
            "t0": 0.40,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup21": {
            "center": get_positions_from_states((1 - 0.6) * x0 + 0.6 * x1),
            "radius": 0.2,
            "dt": 0.001,
            "t0": 0.40,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup16": {
            # this contains error though, but we keep it for tracing
            "center": get_positions_from_states((1 - 0.4) * x0 + 0.6 * x1),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.60,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup17": {
            "center": get_positions_from_states((1 - 0.4) * x0 + 0.4 * x1),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.60,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup18": {
            "center": get_positions_from_states((1 - 0.3) * x0 + 0.3 * x1),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.70,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup19": {
            "center": get_positions_from_states((1 - 0.5) * x0 + 0.5 * x1),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.50,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup20": {
            "center": get_positions_from_states((1 - 0.1) * x0 + 0.1 * x1),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setupE": {
            "center": get_positions_from_states((1 - 0.9) * x0 + 0.9 * x1),
            "radius": 0.01,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
        },
        "setup23": {
            "center": get_positions_from_states(x0),
            "radius": 0.20,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
            "sampler_kwargs": {"valid": True},
        },
        "setup24": {  # change from 8
            "center": get_positions_from_states(x0),
            "radius": 0.50,
            "dt": 0.001,
            "t0": 0.80,
            "sampling_freq": 1,
            "sampler": "cube",
            "sampler_kwargs": {"valid": True},
        },
    }
    if name not in configs:
        raise ValueError(f"Config {name} undefined!")
    config: Dict[str, Any] = configs[name]
    if "sampler_fn" not in config:
        if config.get("sampler") == "simple":
            sampler_fn = get_sampler_simple(arm, config["radius"], config["center"])
        elif config.get("sampler") == "cube":
            sampler_fn = get_sampler_cube(
                arm,
                config["radius"],
                config["center"],
                **config.get("sampler_kwargs", {}),
            )
        else:
            sampler_fn = get_sampler_ee_based(arm, config["radius"], config["center"])
        config["sampler_fn"] = sampler_fn
    if "goal" not in config:
        config["goal"] = x1

    return config

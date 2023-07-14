import enum
from typing import Final, Tuple, Dict

import numpy as np


class ReachingSetup(enum.Enum):
    default = "default"
    hard = "hard"
    planar = "planar"
    reflect = "reflect"
    wierd = "wierd"

    def get_states(self):
        return INITIAL_TERMINAL_STATES[self]


INITIAL_TERMINAL_POSITIONS: Final[
    Dict[ReachingSetup, Tuple[np.ndarray, np.ndarray]]
] = {
    ReachingSetup.default: (
        np.array(
            [
                1.67994461,
                1.25010269,
                2.4427646,
                -1.26689386,
                -0.97780785,
                1.1235575,
                -1.35749704,
            ]
        ),
        np.array(
            [
                2.77363793,
                0.58416175,
                1.54129758,
                -1.702825,
                -2.1665327,
                0.08465574,
                -2.57643323,
            ]
        ),
    ),
    ReachingSetup.hard: (
        np.array([1.6, 1.3, 2.7, -0.85, -1.9, 0.95, -1.6]),
        np.array([2.75, 0.6, 2.0, -1.55, -2.15, 0.0, -2.6]),
    ),
    # Limits: [-2.1,2.1]^3
    ReachingSetup.planar: (
        np.array([0.5, 0.45, 0.0]),
        np.array([1.2, -0.45, 0.9]),
    ),
    ReachingSetup.reflect: (
        np.array([2.35, 1.1, 1.9, -1.85, -1.7, -0.3, -2.1]),
        np.array(
            [
                2.77363793,
                0.58416175,
                1.54129758,
                -1.702825,
                -2.1665327,
                0.08465574,
                -2.57643323,
            ]
        ),
    ),
}
INITIAL_TERMINAL_POSITIONS[ReachingSetup.reflect] = (
    np.array([2.35, 1.1, 1.9, -1.85, -1.7, -0.3, -2.1]),
    INITIAL_TERMINAL_POSITIONS[ReachingSetup.default][1],
)
INITIAL_TERMINAL_POSITIONS[ReachingSetup.wierd] = (
    np.array([2.25, 1.90, -0.40, -1.95, -1.30, 0.6, -1.50]),
    INITIAL_TERMINAL_POSITIONS[ReachingSetup.default][1],
)


def fill_zero_velocity(q):
    return np.r_[q, np.zeros_like(q)]


INITIAL_TERMINAL_STATES: Final[Dict[ReachingSetup, Tuple[np.ndarray, np.ndarray]]] = {
    k: tuple(list(fill_zero_velocity(q) for q in v))
    for k, v in INITIAL_TERMINAL_POSITIONS.items()
}

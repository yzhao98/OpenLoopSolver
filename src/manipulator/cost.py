import copy
import dataclasses
from collections.abc import Mapping
from typing import Dict, Any, List
from rich.console import Console

import numpy as np

console = Console()


class CostTypes:
    eeTrans = "eeTrans"
    eeVel = "eeVel"
    stateReg = "stateReg"
    ctrlReg = "ctrlReg"
    gravState = "gravState"
    accReg = "accReg"
    ctrlRegGravTf = "ctrlRegGravTf"
    general = "general"
    eePose = "eePose"


@dataclasses.dataclass
class GeneralCost:
    cls_name: str
    cls_kwargs: Dict[str, Any]
    weight: float
    unique_id: int = None  # set this so that the cost model constructed will be re-used
    # we don't implement a hash function because we are not sure what will be in cls_kwargs.
    # the user defining a general cost should be aware if a cost model is re-used or not


@dataclasses.dataclass
class SingleCostConfig(Mapping):
    eeTrans: float = 0.0
    eeVel: float = 0.0
    stateReg: float = 0.0
    ctrlReg: float = 0.0
    gravState: float = 0.0
    accReg: float = 0.0
    ctrlRegGravTf: float = 0.0
    general: Dict[str, GeneralCost] = dataclasses.field(default_factory=dict)
    eePose: float = 0.0

    @classmethod
    def from_dict(cls, dikt) -> "SingleCostConfig":
        return cls(**dikt)

    def __iter__(self):
        for k in self.__annotations__:
            yield k

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self) -> int:
        return len(self.__annotations__)

    def print(self):
        dikt = self.__dict__.copy()
        for key in list(dikt.keys()):
            if dikt[key] == 0.0:
                dikt.pop(key)
        general = dikt.pop("general")
        console.print(dikt)
        if len(general) > 0:
            console.print(general)


@dataclasses.dataclass
class CostConfig:
    running: List[SingleCostConfig]
    terminal: SingleCostConfig

    @classmethod
    def from_dict(cls, dikt) -> "CostConfig":
        running_configs = dikt["running"]
        running_configs = (
            [running_configs]
            if not isinstance(running_configs, list)
            else running_configs
        )
        return cls(
            running=[SingleCostConfig.from_dict(config) for config in running_configs],
            terminal=SingleCostConfig.from_dict(dikt["terminal"]),
        )

    def __getitem__(self, key):
        return self.__dict__[key]

    def print(self):
        console.rule("[bold red]Cost")
        console.rule("terminal")
        self.terminal.print()
        console.rule("running")
        for config in self.running:
            config.print()


COST_CONFIG_1 = {
    "running": {
        CostTypes.eeTrans: 5e-2,
        CostTypes.stateReg: 5e-2,
        CostTypes.ctrlReg: 1e-3,
    },
    "terminal": {
        CostTypes.eeTrans: 5e5,
        CostTypes.stateReg: 5e-1,
        CostTypes.eeVel: 1e2,
    },
}

COST_CONFIG_2 = {
    "running": {
        CostTypes.eeTrans: 5e-2,
        CostTypes.stateReg: 5e1,
        CostTypes.ctrlReg: 1e-1,
    },
    "terminal": {
        CostTypes.eeTrans: 5e5,
        CostTypes.stateReg: 5e4,
        CostTypes.eeVel: 1e2,
    },
}

COST_CONFIG_3 = {
    "running": {
        CostTypes.stateReg: 5e1,
        CostTypes.ctrlReg: 1e-1,
    },
    "terminal": {
        CostTypes.stateReg: 5e5,
        CostTypes.eeVel: 1e2,
    },
}

COST_CONFIG_4 = {
    "running": {
        CostTypes.accReg: 1e-2,
        CostTypes.ctrlReg: 5e-2,
    },
    "terminal": {
        CostTypes.stateReg: 5e5,
    },
}

COST_CONFIG_41 = {
    "running": {
        CostTypes.accReg: 1e1,
        CostTypes.ctrlReg: 5e-2,
    },
    "terminal": {
        CostTypes.stateReg: 5e5,
    },
}

COST_CONFIG_42 = {
    "running": {
        CostTypes.accReg: 1e-2,
        CostTypes.ctrlReg: 5e-2,
    },
    "terminal": {
        CostTypes.eePose: 5e5,
    },
}

COST_CONFIG_5 = {
    "running": {
        CostTypes.accReg: 1e-2,
        CostTypes.ctrlRegGravTf: 5e-2,
    },
    "terminal": {
        CostTypes.stateReg: 5e5,
    },
}

COST_CONFIG_51 = {
    "running": {
        CostTypes.accReg: 5e-4,
        CostTypes.ctrlRegGravTf: 1e-3,
        CostTypes.stateReg: 5e-1,
    },
    "terminal": {
        CostTypes.stateReg: 5e5,
    },
}

COST_CONFIG_52 = {
    "running": {
        CostTypes.accReg: 1e-2,
        CostTypes.ctrlReg: 5e-2,
    },
    "terminal": {
        CostTypes.eePose: 5e5,
    },
}

COST_CONFIG_6 = {
    "running": {
        CostTypes.accReg: 1e-2,
        CostTypes.ctrlRegGravTf: 5e-2,
    },
    "terminal": {
        CostTypes.stateReg: 5e3,
    },
}

COST_CONFIG_7 = {
    "running": {
        CostTypes.accReg: 1e-2,
        CostTypes.ctrlRegGravTf: 1e-1,
    },
    "terminal": {
        CostTypes.stateReg: 5e5,
    },
}

COST_CONFIG_101 = {
    "running": {
        CostTypes.accReg: 1e-2,
    },
    "terminal": {
        CostTypes.stateReg: 5e1,
    },
}

COST_CONFIG_201 = copy.deepcopy(COST_CONFIG_5)
COST_CONFIG_201["running"]["general"] = {
    "simpleBall": GeneralCost(
        cls_name="BallObstacleSquarePotential",
        cls_kwargs={
            "center": np.array([-0.327, -0.015, 0.610]),
        },
        weight=1e3,
    )
}

COST_CONFIGS: Dict[int, dict] = {}
for var_name in dict(locals()):
    if var_name.startswith("COST_CONFIG_"):
        __config = locals()[var_name]
        COST_CONFIGS[int(var_name.split("_")[-1])] = __config

COST_CONFIGS_INSTANCE: Dict[int, CostConfig] = {
    k: CostConfig.from_dict(v) for k, v in COST_CONFIGS.items()
}

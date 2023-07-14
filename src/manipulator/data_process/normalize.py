import abc
import dataclasses

import numpy as np
import torch


class DataNormalizer:
    @abc.abstractmethod
    def scale(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def unscale(self, x_scaled: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def is_normal(self, x: np.ndarray) -> bool:
        pass


@dataclasses.dataclass
class MaxMinNormalizer(DataNormalizer):
    """m, M are one dimensional array"""

    m: np.ndarray
    M: np.ndarray

    def __post_init__(self):
        assert len(self.m) == len(self.M)
        self.length = self.M - self.m
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_torch = torch.tensor(self.m, device=device).float()
        self.M_torch = torch.tensor(self.M, device=device).float()
        self.length_torch = torch.tensor(self.length, device=device).float()

    def scale(self, x: np.ndarray) -> np.ndarray:
        return 2 * (x - self.m) / self.length - 1

    def unscale(self, x_scaled: np.ndarray) -> np.ndarray:
        return self.m + self.length * (x_scaled + 1) / 2

    def is_normal(self, x: np.ndarray) -> bool:
        return np.all((x <= 1) & (x >= -1))

    def scale_torch(self, x: torch.tensor) -> torch.tensor:
        return 2 * (x - self.m_torch.to(x.device)) / self.length_torch.to(x.device) - 1

    def unscale_torch(self, x_scaled: torch.tensor) -> torch.tensor:
        return (
            self.m_torch.to(x_scaled.device)
            + self.length_torch.to(x_scaled.device) * (x_scaled + 1) / 2
        )

import abc
import logging
from typing import Optional, Tuple, List, Union

import numpy as np


class WarmStartScheduler:
    def __init__(self):
        logging.info(f"Initializing warm start of class: {self.__class__}")

    @abc.abstractmethod
    def guess(
        self, index: int, x0: np.ndarray, n_steps: int
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Parameters
        ----------
        index: int
            If a group of warm-start guesses are provided, the index can be used to select from them.
        x0: np.ndarray
            The initial state to be solved.
        n_steps: int
            Number of steps in the guess
        Returns
        -------
        None or a tuple of x and u lists
        """

    @abc.abstractmethod
    def update(self, xs: List[np.ndarray], us: List[np.ndarray]):
        pass

    def force_update(self, xs: List[np.ndarray], us: List[np.ndarray]):
        self.update(xs, us)


class SingleWarmStartScheduler(WarmStartScheduler):
    def __init__(self, warm_start_solution):
        super().__init__()
        self.warm_start_solution = warm_start_solution
        self.n_u_steps: int = (
            len(self.warm_start_solution[1]) if self.warm_start_solution else 0
        )

    def guess(
        self, index: int, x0: np.ndarray, n_steps: int
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        if self.n_u_steps != n_steps:
            return None
        return self.warm_start_solution

    def update(self, xs: List[np.ndarray], us: List[np.ndarray]):
        if self.warm_start_solution is None:
            self.warm_start_solution = (xs, us)
            self.n_u_steps = len(us)

    def force_update(self, xs: List[np.ndarray], us: List[np.ndarray]):
        self.warm_start_solution = (xs, us)
        self.n_u_steps = len(us)


class NoWarmStartScheduler(WarmStartScheduler):
    def guess(
        self, index: int, x0: np.ndarray, n_steps: int
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        return None

    def update(self, xs: List[np.ndarray], us: List[np.ndarray]):
        pass


class OneToOneWarmStartScheduler(WarmStartScheduler):
    """Sometimes we are solving problems with a list of initial states.
    Each of these states has a corresponding warm-start-guess-solution.
    This class is used for this purpose.
    For performance, we suppose the order in the querying is the same as the order of the states and their guesses.
    """

    def __init__(
        self,
        x0s: Union[List[np.ndarray], np.ndarray],
        warm_start_solutions: List[Tuple[List[np.ndarray], List[np.ndarray]]],
    ):
        """
        Parameters
        ----------
        x0s:
            we are also storing those initial states, later when querying, we will verify that they match the order.
        warm_start_solutions
        """
        super().__init__()
        self._x0s = x0s
        self.warm_start_solutions = warm_start_solutions

    def guess(
        self, index: int, x0: np.ndarray, n_steps: int
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        x0_expected = self._x0s[index]
        if not (np.any(np.isnan(x0)) and np.any(np.isnan(x0_expected))):
            assert (
                np.linalg.norm(x0 - x0_expected) <= 1e-10
            ), "Order persistence is required for the guess to pair with initial state"
        return self.warm_start_solutions[index]

    def update(self, xs: List[np.ndarray], us: List[np.ndarray]):
        pass

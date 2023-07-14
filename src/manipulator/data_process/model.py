import dataclasses
import pickle
from typing import List, Dict, Any, Optional

import crocoddyl
import numpy as np

from manipulator.data_process.states import ConfigurationOrPoseInnerDataType


@dataclasses.dataclass
class ShootingProblemDataModel:
    cost: np.ndarray
    # the last action model is only used to compute cost
    Fu: np.ndarray
    Fx: np.ndarray
    Lu: np.ndarray
    Lx: np.ndarray

    @classmethod
    def parse_data(
        cls, problem: crocoddyl.ShootingProblem
    ) -> "ShootingProblemDataModel":
        running_datas = problem.runningDatas
        terminal_data = problem.terminalData

        def _get_field_data(field_name):
            field_datas = [getattr(data, field_name) for data in running_datas]
            field_datas.append(getattr(terminal_data, field_name))
            return np.array(field_datas)

        data_dict = {k: _get_field_data(k) for k in cls.__annotations__}
        return cls(**data_dict)


@dataclasses.dataclass
class CollectedResults:
    initial_state: np.ndarray
    goal_state: ConfigurationOrPoseInnerDataType
    distance: float  # this is the distance between initial and goal position (xyz)
    estimated_time: float
    time_step: float
    n_steps: int
    cost_config: dict

    xs: np.ndarray
    us: np.ndarray
    converged: bool

    def validate(self):
        assert round(self.estimated_time / self.time_step) == self.n_steps
        assert len(self.xs) == self.n_steps + 1
        assert len(self.us) == self.n_steps

    @staticmethod
    def _down_sample(
        arr: np.ndarray, sampling_freq: int, one_less: bool, full_length: int = None
    ) -> np.ndarray:
        if sampling_freq == 1:
            return arr
        arr = arr[::sampling_freq]
        if one_less:
            return arr[: full_length - 1]
        return arr

    def down_sampling(self, sampling_freq: int) -> "CollectedResults":
        if sampling_freq == 1:
            return self
        xs = self._down_sample(self.xs, sampling_freq, False)
        us = self._down_sample(self.us, sampling_freq, True, full_length=len(xs))
        assert len(xs) == len(us) + 1
        n_steps = len(us)

        return CollectedResults(
            initial_state=self.initial_state,
            goal_state=self.goal_state,
            distance=self.distance,
            estimated_time=self.estimated_time,
            time_step=self.time_step * sampling_freq,
            n_steps=n_steps,
            cost_config=self.cost_config,
            xs=xs,
            us=us,
            converged=self.converged,
        )

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def list_asdict(results: List["CollectedResults"]) -> List[dict]:
        return [r.asdict() for r in results]

    @staticmethod
    def dict_asdict(results: Dict[Any, "CollectedResults"]) -> Dict[Any, dict]:
        return {k: r.asdict() for k, r in results.items()}

    @staticmethod
    def dict_bulk_save(results: Dict[Any, "CollectedResults"], path_or_buffer):
        dict_results = CollectedResults.dict_asdict(results)
        if hasattr(path_or_buffer, "write"):
            pickle.dump(dict_results, path_or_buffer)
        else:
            with open(path_or_buffer, "wb") as f:
                pickle.dump(dict_results, f)

    @staticmethod
    def build(
        initial_state: np.ndarray,
        goal_state: ConfigurationOrPoseInnerDataType,
        distance: Optional[float],
        tf: float,
        dt: float,
        n_steps: int,
        cost_config: dict,
        is_converged: bool,
        solution: crocoddyl.SolverDDP = None,
        **kwargs,
    ):
        if solution is not None:
            xs, us = np.array(solution.xs), np.array(solution.us)
        else:
            xs = np.array([initial_state])
            us = np.empty(shape=(0, len(goal_state) // 2), dtype=xs.dtype)
        result = CollectedResults(
            initial_state,
            goal_state,
            distance,
            tf,
            dt,
            n_steps,
            cost_config=cost_config,
            xs=xs,
            us=us,
            converged=is_converged,
        )
        result.validate()
        return result


@dataclasses.dataclass
class CollectedResultsV2(CollectedResults):
    # N = n_steps
    cost: np.ndarray  # shape = (N+1, )
    # corresponds to cost at:
    # (t0, x0), (t1, x1), ..., (tN-1, xN-1), (tN, xN)
    # note that cost at (tN, xN) is the terminal cost
    k: np.ndarray  # shape = (N, nu)
    K: np.ndarray  # shape = (N, nu, nx)
    Vx: np.ndarray  # shape = (N+1, nx)
    # x_next = f(x, u)
    fx: np.ndarray  # shape = (N, nx, nx)
    fu: np.ndarray  # shape = (N, nx, nu)
    lx: np.ndarray  # shape = (N+1, nx)
    lu: np.ndarray  # shape = (N, nu)
    V_full_gradient: np.ndarray = None  # shape = (N+1, nx)

    def compute_value_full_gradient(self):
        # last value is terminal cost => full gradient is gradient of terminal cost
        full_gradient = np.empty(shape=(self.n_steps + 1, self.xs.shape[1]))
        full_gradient[-1] = self.lx[-1]
        for i in range(self.n_steps - 1, -1, -1):
            full_gradient[i] = (
                self.lx[i]
                + self.lu[i] @ self.K[i]
                + self.Vx[i + 1] @ (self.fx[i] + self.fu[i] @ self.K[i])
            )
        self.V_full_gradient = full_gradient

    def down_sampling(self, sampling_freq: int) -> "CollectedResultsV2":
        if sampling_freq == 1:
            # avoid unnecessary computation; besides, in super class, we also return self.
            # therefore, v1 will be an instance of CollectedResultsV2, which will introduce conflicts.
            return self
        v1 = super().down_sampling(sampling_freq)
        data_fields = v1.asdict()
        full_length = len(v1.xs)
        return CollectedResultsV2(
            cost=self._down_sample(
                self.cost,
                sampling_freq,
                False,
            ),
            k=self._down_sample(self.k, sampling_freq, True, full_length=full_length),
            K=self._down_sample(self.K, sampling_freq, True, full_length=full_length),
            Vx=self._down_sample(self.Vx, sampling_freq, False),
            fx=self._down_sample(self.fx, sampling_freq, True, full_length=full_length),
            fu=self._down_sample(self.fu, sampling_freq, True, full_length=full_length),
            lx=self._down_sample(self.lx, sampling_freq, False),
            lu=self._down_sample(self.lu, sampling_freq, True, full_length=full_length),
            **data_fields,
        )

    def validate(self):
        super().validate()
        assert len(self.cost) == self.n_steps + 1

        assert len(self.K) == len(self.k) == self.n_steps
        assert len(self.fx) == len(self.fu) == self.n_steps
        assert len(self.lx) == self.n_steps + 1
        assert len(self.lu) == self.n_steps

        assert len(self.Vx) == self.n_steps + 1
        if self.V_full_gradient is not None:
            assert len(self.V_full_gradient) == self.n_steps + 1

    @staticmethod
    def build(
        initial_state: np.ndarray,
        goal_state: ConfigurationOrPoseInnerDataType,
        distance: Optional[float],
        tf: float,
        dt: float,
        n_steps: int,
        cost_config: dict,
        is_converged: bool,
        solution: crocoddyl.SolverDDP = None,
        **kwargs,
    ):
        n_steps = round(tf / dt) if n_steps is None else n_steps
        v1 = super(CollectedResultsV2, CollectedResultsV2).build(
            initial_state,
            goal_state,
            distance,
            tf,
            dt,
            n_steps,
            cost_config,
            is_converged,
            solution,
            **kwargs,
        )

        if kwargs.get("problem", None) is None:
            assert solution is None, "V2 must be constructed with shooting problem"
            cost_cum = np.empty(shape=(1,))
            Vx = np.empty(shape=(1, v1.xs.shape[1]))
            fx = np.empty(shape=(0, v1.xs.shape[1], v1.xs.shape[1]))
            fu = np.empty(shape=(0, v1.xs.shape[1], v1.us.shape[1]))
            lx = np.empty(shape=(1, v1.xs.shape[1]))
            lu = np.empty(shape=(0, v1.us.shape[1]))
            K = np.empty(shape=(0, v1.us.shape[1], v1.xs.shape[1]))
            k = np.empty(shape=(0, v1.us.shape[1]))
        else:
            problem = kwargs.pop("problem")
            assert solution is not None
            extra_data = ShootingProblemDataModel.parse_data(problem)
            # cumulative sum extra_data.cost
            # cost_cum[i] = sum(extra_data.cost[:-i])
            cost_cum = np.cumsum(extra_data.cost[::-1])[::-1]
            Vx = np.array(solution.Vx)
            fx = extra_data.Fx[:n_steps]
            fu = extra_data.Fu[:n_steps]
            lx = extra_data.Lx
            lu = extra_data.Lu[:n_steps]
            K = np.array(solution.K)
            k = np.array(solution.k)

        result = CollectedResultsV2(
            cost=cost_cum,
            K=K,
            k=k,
            Vx=Vx,
            fx=fx,
            fu=fu,
            lx=lx,
            lu=lu,
            **v1.asdict(),
        )
        result.compute_value_full_gradient()
        result.validate()
        return result

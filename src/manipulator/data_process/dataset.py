import enum
import functools
import logging
import pathlib
import pickle
from collections import Counter
from typing import List, Union, Tuple, Optional, Sequence

import numpy as np
import torch
import torch.utils.data

from manipulator.constants import DATA_PATH
from manipulator.data_process import TrajectoriesLoader
from manipulator.data_process import DataNormalizer, MaxMinNormalizer


def check_dataset_compatibility(fn):
    """This function serves as a wrapper to ensure all arguments and keyword arguments of type ManipulatorTrajectories
    passed to the wrapped function are compatible with each other, namely, they are both of v1 or both of v2.
    To be specific, they should both have fields V_traj, dudx_traj, dVdx_traj or none of them.
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        is_version_v2 = None
        for arg in args:
            if isinstance(arg, ManipulatorTrajectories):
                if is_version_v2 is None:
                    is_version_v2 = arg.V_traj is not None
                else:
                    assert is_version_v2 == (
                        arg.V_traj is not None
                    ), "All arguments of type ManipulatorTrajectories should be of the same version."
        for kwarg in kwargs.values():
            if isinstance(kwarg, ManipulatorTrajectories):
                if is_version_v2 is None:
                    is_version_v2 = kwarg.V_traj is not None
                else:
                    assert is_version_v2 == (
                        kwarg.V_traj is not None
                    ), "All arguments of type ManipulatorTrajectories should be of the same version."
        return fn(*args, **kwargs)

    return inner


def handle_missing_fields_in_v2_version(fn):
    """
    It is assumed that the wrapped function is a member function of or having first argument of type
    ManipulatorTrajectories
    and that it returns a ManipulatorTrajectories object.
    This decorator can handle the case when the dataset is loaded from a v1 version of the dataset, i.e.
    the dataset does not have V_traj, dudx_traj, dVdx_traj. In this case, the decorator will create these fields,
    call the wrapped function, and then delete these fields from returned dataset.
    We also handle ManipulatorTrajectories objects that are passed as arguments to the wrapped function.
    """

    def fill_fake_data(dataset: "ManipulatorTrajectories"):
        dataset.V_traj = [np.empty(shape=(x_t.shape[0],)) for x_t in dataset.x_traj]
        dataset.dudx_traj = [
            np.empty(shape=(*u_t.shape, x_t.shape[1]))
            for u_t, x_t in zip(dataset.u_traj, dataset.x_traj)
        ]
        dataset.dVdx_traj = [np.empty(shape=x_t.shape) for x_t in dataset.x_traj]

    def undo_fill_fake_data(dataset: "ManipulatorTrajectories"):
        dataset.V_traj = None
        dataset.dudx_traj = None
        dataset.dVdx_traj = None

    @functools.wraps(fn)
    def inner(*args, **kwargs) -> "ManipulatorTrajectories":
        is_version_v2 = True

        args_v1 = []
        kw_v1 = []
        # Let us first determine if the dataset is of v1 version
        for arg in args:
            if isinstance(arg, ManipulatorTrajectories) and arg.V_traj is None:
                is_version_v2 = False
                break
        if is_version_v2:
            for kwarg in kwargs.values():
                if isinstance(kwarg, ManipulatorTrajectories) and kwarg.V_traj is None:
                    is_version_v2 = False
                    break

        if not is_version_v2:
            for arg_i, arg in enumerate(args):
                if isinstance(arg, ManipulatorTrajectories) and arg.V_traj is None:
                    fill_fake_data(arg)
                    args_v1.append(arg_i)
            for kw_k in kwargs:
                if (
                    isinstance(kwargs[kw_k], ManipulatorTrajectories)
                    and kwargs[kw_k].V_traj is None
                ):
                    fill_fake_data(kwargs[kw_k])
                    kw_v1.append(kw_k)

        dataset_rtn = fn(*args, **kwargs)
        if not is_version_v2:
            undo_fill_fake_data(dataset_rtn)
            # We shall also remove these fields from inputs
            if args_v1:
                for arg_i in args_v1:
                    undo_fill_fake_data(args[arg_i])
            if kw_v1:
                for kw_k in kw_v1:
                    undo_fill_fake_data(kwargs[kw_k])
        return dataset_rtn

    return inner


class ManipulatorTrajectories(torch.utils.data.Dataset, TrajectoriesLoader):
    # ======================= Start Implementation for Loader ==========================
    @classmethod
    def get_invalid_data(
        cls, trajectories: List[Tuple[np.ndarray, np.ndarray]], **configs
    ) -> Tuple[Sequence[int], dict]:
        (
            local_invalid_index,
            invalid_value,
            mean,
        ) = cls.find_invalid_trajectory_by_deviation_from_mean_of_max(
            trajectories[1],
            number_of_std=configs["number_of_std"],
        )
        return local_invalid_index, {"invalid_value": invalid_value, "mean": mean}

    @classmethod
    def clean_data_info_iter(cls, invalid_indexes, invalid_info):
        invalid_value, mean = invalid_info["invalid_value"], invalid_info["mean"]
        logging.info(
            f"Removing data_process at index {invalid_indexes} dut to their control out of distribution "
            f"(values {invalid_value} in data_process of current mean {mean})"
        )

    @classmethod
    def clean_data_info_summary(
        cls, all_invalid_indexes: List[List[int]], all_invalid_info: List[dict]
    ):
        super().clean_data_info_summary(all_invalid_indexes, all_invalid_info)
        n_remove = sum(map(len, all_invalid_indexes), 0)
        last_invalid_info = all_invalid_info[
            -1
        ]  # Even if no data_process is removed, will also report info once.
        logging.info(
            f"After cleaning {n_remove} trajectories, data_process has mean (of max u values of each trajectory): "
            f"{last_invalid_info['mean']}"
        )

    @classmethod
    def load_from_data_dict(
        cls,
        data_dict,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dtype=torch.float,
        remove_invalid_data: bool = True,
    ):
        logging.debug(f"Loading from data_dict to device {device} in type {dtype}")
        trajectories = super().load_from_data_dict(
            data_dict, remove_invalid_data, invalid_config={"number_of_std": 2}
        )
        return cls.construct_from_trajectories(trajectories, device, dtype)

    @classmethod
    def load_data(
        cls,
        filename,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dtype=torch.float,
        data_dir: pathlib.Path = DATA_PATH,
        remove_invalid_data: bool = True,
    ) -> "ManipulatorTrajectories":
        """This function will create a Dataset from pre-generated data_process in this project."""
        try:
            logging.debug(
                f"Loading {filename} in {data_dir.absolute()} to device {device} in type {dtype}"
            )
            trajectories = super().load_data(
                filename,
                data_dir,
                remove_invalid_data,
                invalid_config={"number_of_std": 2},
            )
            return cls.construct_from_trajectories(trajectories, device, dtype)
        except AttributeError as e:
            logging.warning(f"Loading failed. Try directly unpickling it...")
            with open(data_dir.joinpath(filename), "rb") as f:
                d = pickle.load(f)
            if isinstance(d, ManipulatorTrajectories):
                logging.info(f"Successfully unpickling dataset.")
                return d
            else:
                raise AttributeError from e

    @classmethod
    def construct_from_trajectories(
        cls,
        trajectories: TrajectoriesLoader.Trajectories,
        device: torch.device,
        dtype: torch.dtype,
    ):
        x_trajectories = trajectories.x_trajectories
        u_trajectories = trajectories.u_trajectories
        dt_array = trajectories.dt_array
        goal_state = trajectories.goal_state
        index_mapping = trajectories.index_mapping

        V_trajectories = trajectories.V_trajectories
        dudx_trajectories = trajectories.dudx_trajectories
        dVdx_trajectories = trajectories.dVdx_trajectories

        time_steps_to_go = [np.arange(0, x.shape[0])[::-1] for x in x_trajectories]
        time_trajectories = [t * dt for t, dt in zip(time_steps_to_go, dt_array)]

        return cls(
            x_traj=list(x_trajectories),
            u_traj=list(u_trajectories),
            t_traj=time_trajectories,
            dt_array=dt_array,
            goal_state=goal_state,
            device=device,
            dtype=dtype,
            index_mapping=tuple(index_mapping),
            V_traj=V_trajectories,
            dudx_traj=dudx_trajectories,
            dVdx_traj=dVdx_trajectories,
        )

    @staticmethod
    def find_invalid_trajectory_by_deviation_from_mean_of_max(
        trajectories: Sequence[np.ndarray],
        number_of_std: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        # If the traj is empty, we simply exclude it and record the index. Then we map the index back to the original.
        max_of_each_traj = []
        original_index = []
        for i, traj in enumerate(trajectories):
            if len(traj) > 0:
                max_of_each_traj.append(np.max(abs(traj)))
                original_index.append(i)
        max_of_each_traj = np.array(max_of_each_traj)
        original_index = np.array(original_index, dtype=int)

        mean = np.mean(max_of_each_traj).item()
        std = np.std(max_of_each_traj).item()
        fraction_of_mean = max(
            (1 - std / mean), 0.0
        )  # if std is very small, we tend to remove less data_process
        invalid_index: np.ndarray = np.where(
            max_of_each_traj >= mean + number_of_std * std + fraction_of_mean * mean
        )[0]
        invalid_value: np.ndarray = max_of_each_traj[invalid_index]
        # now we map invalid_index back to the original index
        invalid_original_index = original_index[invalid_index]
        return invalid_original_index, invalid_value, mean

    # ======================= End Implementation for Loader ==========================

    def get_trajectories(self, data_id: int):
        x_traj = self.x_traj[data_id]
        u_traj = self.u_traj[data_id]
        t_traj = self.t_traj[data_id]
        return x_traj, u_traj, t_traj

    def get_value_trajectories(self, data_id: int):
        return self.V_traj[data_id]

    @handle_missing_fields_in_v2_version
    def select_data(self, data_indexes, ignore_index=False):
        x_traj = [self.x_traj[i] for i in data_indexes]
        u_traj = [self.u_traj[i] for i in data_indexes]
        t_traj = [self.t_traj[i] for i in data_indexes]
        dt_array = [self.dt_array[i] for i in data_indexes]
        V_traj = [self.V_traj[i] for i in data_indexes]
        dudx_traj = [self.dudx_traj[i] for i in data_indexes]
        dVdx_traj = [self.dVdx_traj[i] for i in data_indexes]
        if ignore_index:
            index_mapping = None
        else:
            index_mapping = tuple([self.index_mapping[i] for i in data_indexes])
        return ManipulatorTrajectories(
            x_traj,
            u_traj,
            t_traj,
            dt_array,
            self.goal_state,
            self.device,
            self.dtype,
            index_mapping=index_mapping,
            V_traj=V_traj,
            dudx_traj=dudx_traj,
            dVdx_traj=dVdx_traj,
        )

    @handle_missing_fields_in_v2_version
    def select_portion(self, start: float, end: float):
        x_traj = []
        u_traj = []
        t_traj = []
        V_traj = []
        dudx_traj = []
        dVdx_traj = []
        for x_t, u_t, t_t, v_t, ux_t, vx_t in zip(
            self.x_traj,
            self.u_traj,
            self.t_traj,
            self.V_traj,
            self.dudx_traj,
            self.dVdx_traj,
        ):
            size = len(u_t)
            start_id = round(size * start)
            end_id = round(size * end)
            x_traj.append(x_t[start_id : end_id + 1])
            t_traj.append(t_t[start_id : end_id + 1])
            u_traj.append(u_t[start_id:end_id])
            V_traj.append(v_t[start_id : end_id + 1])
            dudx_traj.append(ux_t[start_id:end_id])
            dVdx_traj.append(vx_t[start_id : end_id + 1])
        return ManipulatorTrajectories(
            x_traj,
            u_traj,
            t_traj,
            self.dt_array,
            self.goal_state,
            self.device,
            self.dtype,
            self.index_mapping,
            V_traj=V_traj,
            dudx_traj=dudx_traj,
            dVdx_traj=dVdx_traj,
        )

    def scale(
        self,
        x_normalizer: Optional[DataNormalizer] = None,
        u_normalizer: Optional[DataNormalizer] = None,
        t_normalizer: Optional[DataNormalizer] = None,
        V_normalizer: Optional[DataNormalizer] = None,
    ):
        if self.dudx_traj is not None:
            raise NotImplementedError(
                "After scaling, the gradients are not valid anymore."
                " Please do not call this function."
            )
        x_traj = self.x_traj
        if x_normalizer is not None:
            x_traj = [x_normalizer.scale(x) for x in x_traj]
        u_traj = self.u_traj
        if u_normalizer is not None:
            u_traj = [u_normalizer.scale(u) for u in u_traj]
        t_traj = self.t_traj
        if t_normalizer is not None:
            t_traj = [t_normalizer.scale(t) for t in t_traj]
        V_traj = self.V_traj
        if V_normalizer is not None:
            V_traj = [V_normalizer.scale(V) for V in V_traj]
        return ManipulatorTrajectories(
            x_traj,
            u_traj,
            t_traj,
            self.dt_array,  # is this still meaningful?
            self.goal_state,  # NOTE: don't scale the goal state
            # x_normalizer.scale(self.goal_state)
            # if self.goal_state is not None
            # else None,  # should we scale also this?
            self.device,
            self.dtype,
            self.index_mapping,
            V_traj=V_traj,
            dudx_traj=self.dudx_traj,
            dVdx_traj=self.dVdx_traj,
        )

    @handle_missing_fields_in_v2_version
    def select_range(
        self, start_ids: Union[List[int], int], end_ids: Union[List[Optional[int]], int]
    ):
        x_traj = []
        u_traj = []
        t_traj = []
        V_traj = []
        dudx_traj = []
        dVdx_traj = []
        start_ids = (
            [start_ids] * self.n_traj if isinstance(start_ids, int) else start_ids
        )
        end_ids = [end_ids] * self.n_traj if isinstance(end_ids, int) else end_ids
        for x_t, u_t, t_t, start_id, end_id, V_t, ux_t, vx_t in zip(
            self.x_traj,
            self.u_traj,
            self.t_traj,
            start_ids,
            end_ids,
            self.V_traj,
            self.dudx_traj,
            self.dVdx_traj,
        ):
            sl = slice(start_id, end_id + 1 if end_id else None)
            ss = slice(start_id, end_id if end_id else None)
            x_traj.append(x_t[sl])
            t_traj.append(t_t[sl])
            u_traj.append(u_t[ss])
            V_traj.append(V_t[sl])
            dudx_traj.append(ux_t[ss])
            dVdx_traj.append(vx_t[sl])

        return ManipulatorTrajectories(
            x_traj,
            u_traj,
            t_traj,
            self.dt_array,
            self.goal_state,
            self.device,
            self.dtype,
            self.index_mapping,
            V_traj=V_traj,
            dudx_traj=dudx_traj,
            dVdx_traj=dVdx_traj,
        )

    def _data_check(self):
        assert (
            self.x.shape[0] == self.y.shape[0]
        ), "Size of inputs and labels must match"

    def get_all_duration(self):
        return [t_traj[0] for t_traj in self.t_traj]

    def get_all_t0_values(self):
        if self.V_traj is not None:
            return [v[0] for v in self.V_traj]
        else:
            return []

    def __str__(self) -> str:
        u_stats = self.u_statistics()
        x_stats = self.x_statistics()
        float_fmt = "8.3f"
        u_stats_str = self.convert_to_statistics_str(
            u_stats, name="u", float_fmt=float_fmt
        )
        x_stats_str = self.convert_to_statistics_str(
            x_stats, name="x", float_fmt=float_fmt
        )
        durations = self.get_all_duration()
        duration_freq = Counter(durations)
        duration_str = "t:\n"
        for k, v in duration_freq.items():
            duration_str += f"\t{k:4.3f}: {v}/{self.n_traj}\n"

        values = np.array(self.get_all_t0_values())
        values_str = (
            f"Values:\n\t{values.min(initial=np.inf):{float_fmt}}-{values.max(initial=-np.inf):{float_fmt}} "
            f"(mean={values.mean():{float_fmt}})\n"
        )

        return (
            f"{self.n_traj} trajectories on {self.device} of type {self.dtype}\n"
            f"{duration_str}"
            f"{u_stats_str}"
            f"{x_stats_str}"
            f"{values_str}"
        )

    def x_statistics(self):
        return self.get_statistics(self.x_traj)

    def u_statistics(self):
        return self.get_statistics(self.u_traj)

    @staticmethod
    def get_statistics(list_of_arrays: List[np.ndarray], axis=0):
        merged_array = np.concatenate(list_of_arrays, axis=axis)
        array_of_min = merged_array.min(axis=0, initial=+np.inf)
        array_of_max = merged_array.max(axis=0, initial=-np.inf)
        array_of_mean = merged_array.mean(axis=0)
        array_of_std = merged_array.std(axis=0)
        return array_of_min, array_of_max, array_of_mean, array_of_std

    @staticmethod
    def convert_to_statistics_str(
        statistics: Sequence[np.ndarray], name="", float_fmt="6.3f"
    ):
        stats_str = f"{name}:\n"
        for i, (u_min, u_max, u_mean, u_std) in enumerate(zip(*statistics)):
            stats_str += (
                f"\t{name}[{i:02d}]: {u_mean:{float_fmt}} +- {u_std:{float_fmt}} "
                f"({u_min:{float_fmt}}, {u_max:{float_fmt}})\n"
            )
        return stats_str

    def get_initial(self) -> Tuple[np.ndarray, np.ndarray]:
        x0s = [x[0] for x in self.x_traj]
        t0s = [t[0] for t in self.t_traj]
        return np.array(x0s), np.array(t0s)

    def get_terminal(self) -> np.ndarray:
        xfs = [x[-1] for x in self.x_traj]
        return np.array(xfs)

    def get_states_at_time(self, at: Union[float, int]) -> np.ndarray:
        if isinstance(at, int):
            return np.array([x[at] for x in self.x_traj])
        if isinstance(at, float):
            return np.array([x[round(len(x) * at)] for x in self.x_traj])
        raise ValueError(f"at={at} not supported.")

    @property
    def n_traj(self) -> int:
        return len(self.u_traj)

    def union(self, other: "ManipulatorTrajectories"):
        assert other.device == self.device
        assert other.dtype == self.dtype
        # if any of the goal state is None, or they are not equal, set to None
        if self.goal_state is None or other.goal_state is None:
            goal_state = None
        else:
            goal_state = (
                self.goal_state
                if np.allclose(self.goal_state, other.goal_state)
                else None
            )

        x_traj = self.x_traj + other.x_traj
        u_traj = self.u_traj + other.u_traj
        t_traj = self.t_traj + other.t_traj
        dt_array = np.r_[np.array(self.dt_array), np.array(other.dt_array)]
        if self.V_traj is None or other.V_traj is None:
            V_traj = None
            dudx_traj = None
            dVdx_traj = None
        else:
            V_traj = self.V_traj + other.V_traj
            dudx_traj = self.dudx_traj + other.dudx_traj
            dVdx_traj = self.dVdx_traj + other.dVdx_traj

        return ManipulatorTrajectories(
            x_traj,
            u_traj,
            t_traj=t_traj,
            dt_array=dt_array,
            goal_state=goal_state,
            device=self.device,
            dtype=self.dtype,
            index_mapping=None,
            V_traj=V_traj,
            dudx_traj=dudx_traj,
            dVdx_traj=dVdx_traj,
        )

    def is_same_length(self) -> bool:
        return len(set(map(len, self.x_traj))) == 1 and self._is_t0_same()

    @property
    def m(self) -> int:
        """get the length of trajectories, i.e. number of states (including initial and terminal) in each trajectory"""
        assert self.is_same_length(), "trajectories of different length"
        return len(self.x_traj[0])

    # ======================= Start Implementation for torch Dataset ==========================
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    # ======================= End Implementation for torch Dataset ==========================

    class Mode(enum.Enum):
        Uo = "control only"
        Vo = "value only"
        UV = "control and value"

        Uog = "control only with gradients"
        Vog = "value only with gradients"
        UVg = "control and value with gradients"

        Xn = "next state state"

    def __init__(
        self,
        x_traj: List[np.ndarray],
        u_traj: List[np.ndarray],
        t_traj: List[np.ndarray],
        dt_array: Union[List[float], np.ndarray],
        goal_state: Optional[np.ndarray],
        device,
        dtype,
        index_mapping: Optional[Tuple[int, ...]],
        V_traj: Optional[List[np.ndarray]] = None,
        dudx_traj: Optional[List[np.ndarray]] = None,
        dVdx_traj: Optional[List[np.ndarray]] = None,
    ):
        self.x_traj = x_traj
        self.u_traj = u_traj
        self.t_traj = t_traj
        self.dt_array = dt_array
        self.V_traj = V_traj
        self.dudx_traj = dudx_traj
        self.dVdx_traj = dVdx_traj

        self.device = device
        self.dtype = dtype
        self.goal_state = goal_state
        self.index_mapping = index_mapping

        self.mode = None
        # We don't initialize attributes x and y for an empty dataset
        self.len: int
        self.x: Optional[torch.Tensor]
        self.y: Optional[torch.Tensor]
        if len(self.x_traj) == 0:
            self.len, self.x, self.y = 0, None, None
        else:
            self.len, self.x, self.y = *self.get_inputs(), None  # noqa
            self.switch_labels(self.Mode.Uo)  # it will update y in-place

    def get_inputs(self) -> Tuple[int, torch.Tensor]:
        # concatenate times and states to get time-state pairs for each trajectory
        x_np = np.concatenate([x[:-1] for x in self.x_traj], axis=0)
        return x_np.shape[0], torch.tensor(x_np, dtype=self.dtype, device=self.device)

    def get_numpy_labels(self, mode: Mode):
        if mode == self.Mode.Uo:
            u_np = np.concatenate(self.u_traj, axis=0)
            return u_np
        elif mode == self.Mode.Vo:
            v_np = np.concatenate([v_traj[:-1] for v_traj in self.V_traj], axis=0)[
                :, np.newaxis
            ]
            return v_np
        elif mode == self.Mode.UV:
            u_np = np.concatenate(self.u_traj, axis=0)
            v_np = np.concatenate([v_traj[:-1] for v_traj in self.V_traj], axis=0)
            uv_np = np.concatenate((u_np, v_np[:, np.newaxis]), axis=1)
            return uv_np
        elif mode == self.Mode.Uog:
            u_np = np.concatenate(self.u_traj, axis=0)[:, :, np.newaxis]  # (N, nu, 1)
            dudx_np = np.concatenate(self.dudx_traj, axis=0)  # (N, nu, nx)
            return np.concatenate((u_np, dudx_np), axis=-1)  # (N, nu, 1+nx)
        elif mode == self.Mode.Vog:
            v_np = np.concatenate([v_traj[:-1] for v_traj in self.V_traj], axis=0)[
                :, np.newaxis, np.newaxis
            ]  # (N, 1, 1)
            dVdx_np = np.concatenate([dVdx[:-1] for dVdx in self.dVdx_traj], axis=0)[
                :, np.newaxis, :
            ]  # (N, 1, nx)
            return np.concatenate((v_np, dVdx_np), axis=-1)  # (N, 1, 1+nx)
        elif mode == self.Mode.UVg:
            u_dudx = self.get_numpy_labels(self.Mode.Uog)  # (N, nu, 1+nx)
            v_dVdx = self.get_numpy_labels(self.Mode.Vog)  # (N, 1, 1+nx)
            return np.concatenate((u_dudx, v_dVdx), axis=1)  # (N, nu+1, 1+nx)
        elif mode == self.Mode.Xn:
            x_np = np.concatenate([x[1:] for x in self.x_traj], axis=0)
            return x_np
        else:
            raise NotImplementedError(f"mode {mode} not implemented")

    def get_labels_at(self, k: int):
        if self.mode == self.Mode.Uo:
            return np.array([u[k] for u in self.u_traj])
        elif self.mode == self.Mode.UV:
            return np.array(
                [np.r_[u[k], v[k]] for u, v in zip(self.u_traj, self.V_traj)]
            )
        elif self.mode == self.Mode.Vo:
            return np.array([v[k] for v in self.V_traj])
        elif self.mode == self.Mode.Xn:
            return np.array([x[k + 1] for x in self.x_traj])
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")

    def iter_at_time(self, return_tensor="th"):
        assert (
            self.is_same_length()
        ), "This method can only be called when all trajectories are of same length"
        return_tensor_fn = (
            lambda x: torch.tensor(x, device=self.device, dtype=self.dtype)
            if return_tensor == "th"
            else np.array(x)
        )
        for k in range(len(self.x_traj[0]) - 1):
            x_at = np.array(
                [np.r_[t[k], x[k]] for x, t in zip(self.x_traj, self.t_traj)]
            )
            yield return_tensor_fn(x_at), return_tensor_fn(self.get_labels_at(k))

    def to(self, *, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        self.x = self.x.to(device=device, dtype=dtype)
        self.y = self.y.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return self

    def switch_labels(self, mode: Mode) -> None:
        """
        For gradient-mode, we have unified data_process to three dimensions:
            (samples dimension, data_process dimension, inputs dimension).
        - samples dimension: number of samples
        - data_process dimension: (dimension of data_process) for u, it is nu; for uv, it is nu+nv
        - inputs dimension: (dimension of inputs) this is always nx + 1 (1 for time)
        """
        if self.mode == mode and self.y is not None:
            return
        logging.info(f"switching data_process to {mode.value}")
        self.mode = mode
        self.y = torch.tensor(
            self.get_numpy_labels(mode), dtype=self.dtype, device=self.device
        )
        self._data_check()

    def __eq__(self, other: "ManipulatorTrajectories"):
        fields_to_compare = ("x_traj", "u_traj", "t_traj", "dt_array")
        for field in fields_to_compare:
            if not np.allclose(
                np.array(getattr(self, field)), np.array(getattr(other, field))
            ):
                return False

        fields_optional = ("V_traj", "dudx_traj", "dVdx_traj")
        # for field in fields_optional, they should either both be None or both not None and all close
        for field in fields_optional:
            if getattr(self, field) is None or getattr(other, field) is None:
                if getattr(self, field) != getattr(other, field):
                    return False
            else:
                if not np.allclose(
                    np.array(getattr(self, field)), np.array(getattr(other, field))
                ):
                    return False

        if self.dtype != other.dtype:
            return False
        return True

    def __getstate__(self):
        state = self.__dict__.copy()
        state["x"] = None
        state["y"] = None
        return state

    def __setstate__(self, state):
        if not torch.cuda.is_available():
            device = torch.device("cpu")
            if torch.device(state["device"]).type == "cuda":
                logging.warning("Cuda unavailable, device set to CPU.")
                state["device"] = device

        self.__dict__.update(state)
        # note that: x,y was removed in __getstate__; therefore, we need to add them back by calling switch_labels
        self.len, self.x = self.get_inputs()
        self.switch_labels(self.mode)

    @property
    def dt(self):
        # see if all dt are the same under a tolerance
        assert np.allclose(
            self.dt_array, self.dt_array[0]
        ), "Only if all dt are the same"
        return self.dt_array[0].item()

    def _is_t0_same(self):
        t0_array = [t[0] for t in self.t_traj]
        # see if all t0 are nearly the same under a tolerance
        return np.allclose(t0_array, t0_array[0])

    @property
    def t0(self) -> float:
        # see if all t0 are nearly the same under a tolerance
        assert self._is_t0_same(), "Only if all t0 are the same"
        return self.t_traj[0][0]


def get_scaled_dataset(
    dataset: ManipulatorTrajectories,
    x_normalizer=None,
    u_normalizer=None,
    t_normalizer=None,
    V_normalizer=None,
):
    # TODO: No support for gradient now
    dataset.dudx_traj = None
    dataset_scale = dataset.scale(
        x_normalizer, u_normalizer, t_normalizer, V_normalizer
    )
    return dataset_scale


def get_normalizer(
    dataset: ManipulatorTrajectories,
    working_dir,
):
    x_normalizer = MaxMinNormalizer(
        np.min(dataset.x_traj, axis=(0, 1)), np.max(dataset.x_traj, axis=(0, 1))
    )
    u_normalizer = MaxMinNormalizer(
        np.min(dataset.u_traj, axis=(0, 1)), np.max(dataset.u_traj, axis=(0, 1))
    )
    t_normalizer = MaxMinNormalizer(
        np.expand_dims(np.min(dataset.t_traj), 0),
        np.expand_dims(np.max(dataset.t_traj), 0),
    )
    V_normalizer = MaxMinNormalizer(
        np.expand_dims(np.min(dataset.V_traj), 0),
        np.expand_dims(np.max(dataset.V_traj), 0),
    )

    norm_info = {
        "x_min": x_normalizer.m,
        "x_max": x_normalizer.M,
        "u_min": u_normalizer.m,
        "u_max": u_normalizer.M,
        "t_min": t_normalizer.m,
        "t_max": t_normalizer.M,
        "V_min": V_normalizer.m,
        "V_max": V_normalizer.M,
    }
    np.save(working_dir.joinpath("norm_info.pkl"), norm_info)
    return x_normalizer, u_normalizer, t_normalizer, V_normalizer

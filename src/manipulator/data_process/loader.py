import abc
import collections
import dataclasses
import logging
import pathlib
import pickle
import warnings
from itertools import compress
from typing import List, Tuple, Sequence, final, Optional, Dict, Any

import numpy as np
import pinocchio as pin

from manipulator.constants import DATA_PATH


class TrajectoriesLoader:
    """
    TrajectoriesLoader does not have data member but implements several methods for loading data from pickle files.
    It also implements an outer loop for cleaning data but left the specific cleaning logics to its subclasses.
    """

    @dataclasses.dataclass
    class Trajectories:
        x_trajectories: Sequence[np.ndarray]
        u_trajectories: Sequence[np.ndarray]
        dt_array: np.ndarray
        valid: Sequence[bool]
        goal_state: Optional[np.ndarray]
        index_mapping: Sequence[int]
        V_trajectories: Optional[Sequence[np.ndarray]]
        dudx_trajectories: Optional[Sequence[np.ndarray]]
        dVdx_trajectories: Optional[Sequence[np.ndarray]]

        @classmethod
        def empty(cls):
            return cls([], [], np.empty(shape=0), [], None, [], [], [], [])

    @classmethod
    @abc.abstractmethod
    def get_invalid_data(
        cls, trajectories: Tuple[List[np.ndarray], List[np.ndarray]], **configs
    ) -> Tuple[Sequence[int], dict]:
        """
        This function accepts:
            - trajectories = (x_traj, u_traj)
        It should return a Tuple of two variables:
            - the index of data marked as invalid. This is the index of the argument trajectories[0], trajectories[1].
            - a dictionary containing information that is going to log out in clean_data_logger function
        """
        pass

    @classmethod
    @abc.abstractmethod
    def clean_data_info_iter(cls, invalid_indexes, invalid_info):
        pass

    @classmethod
    @abc.abstractmethod
    def clean_data_info_summary(
        cls, all_invalid_indexes: List[List[int]], all_invalid_info: List[dict]
    ):
        assert len(all_invalid_indexes) == len(all_invalid_info)

    @classmethod
    @final
    def clean_data(
        cls, x_traj, u_traj, valid, invalid_config
    ) -> Tuple[List[bool], List[List[int]], List[dict]]:
        """This function implements a  cleaning logics:
            it calls get_invalid_data with the current valid data repeatedly.
        It repeats:
         - find out the outlier w.r.t. the current valid data
         - remove outlier from the current valid data
        Returns:
         - valid: list of bool indicate which data are valid
         - all_invalid_indexes: list of lists of index, indexes of data being removed at each iteration
         - all_invalid_info: list of dicts, information from the clean function at each iteration
        """
        all_invalid_indexes = []
        all_invalid_info = []
        while True:
            (local_invalid_index, invalid_info) = cls.get_invalid_data(
                (list(compress(x_traj, valid)), list(compress(u_traj, valid))),
                **invalid_config,
            )
            # Note that the index is local, we need to convert it back to global one
            index_mapping = tuple(get_index_mapping(valid))
            global_invalid_index = [index_mapping[i] for i in local_invalid_index]
            all_invalid_indexes.append(global_invalid_index)
            all_invalid_info.append(invalid_info)
            if len(global_invalid_index) == 0:
                break
            cls.clean_data_info_iter(global_invalid_index, invalid_info)
            valid[global_invalid_index] = False
        return valid, all_invalid_indexes, all_invalid_info

    @classmethod
    @final
    def load_from_data_dict_impl(
        cls,
        data_dict,
        is_apply_cleaner: bool = True,
        invalid_config: dict = None,
    ):
        data_array, index_mapping = convert_data_to_numpy_array(data_dict)
        if len(data_array) == 0:
            return cls.Trajectories.empty()
        valid = data_array["converged"]
        x_trajectories = data_array["xs"]
        u_trajectories = data_array["us"]
        dt_array = data_array["time_step"]
        assert len(x_trajectories) == len(u_trajectories) == len(dt_array) == len(valid)

        # to be compatible with the old data, we need to check if the data has V and dudx
        V_trajectories = data_array.get("cost", None)
        dudx_trajectories = data_array.get("K", None)
        dVdx_trajectories = data_array.get("V_full_gradient", None)
        if V_trajectories is not None:
            assert (
                len(V_trajectories)
                == len(x_trajectories)
                == len(dudx_trajectories)
                == len(dVdx_trajectories)
            )

        if is_apply_cleaner:
            logging.info("Removing invalid data...")
            invalid_config = {} if invalid_config is None else invalid_config
            valid, all_removed_indexes, all_invalid_info = cls.clean_data(
                x_trajectories, u_trajectories, valid, invalid_config=invalid_config
            )
            cls.clean_data_info_summary(all_removed_indexes, all_invalid_info)

        index_mapping_local = tuple(get_index_mapping(valid))
        index_mapping = [index_mapping[i] for i in index_mapping_local]
        assert len(index_mapping) == np.sum(valid)

        x_trajectories = list(compress(x_trajectories, valid))
        u_trajectories = list(compress(u_trajectories, valid))

        if V_trajectories is not None:
            V_trajectories = list(compress(V_trajectories, valid))
            dudx_trajectories = list(compress(dudx_trajectories, valid))
            dVdx_trajectories = list(compress(dVdx_trajectories, valid))

        dt_array = dt_array[valid]

        goal_state = data_array["goal_state"][0]
        # check if goal_from all data are the same
        for i in range(1, len(data_array["goal_state"])):
            if not np.allclose(goal_state, data_array["goal_state"][i]):
                goal_state = None
                break

        return cls.Trajectories(
            x_trajectories,
            u_trajectories,
            dt_array,
            valid,
            goal_state,
            index_mapping,
            V_trajectories,
            dudx_trajectories,
            dVdx_trajectories,
        )

    load_from_data_dict = load_from_data_dict_impl

    @classmethod
    def load_data(
        cls,
        filename,
        data_dir: pathlib.Path = DATA_PATH,
        is_apply_cleaner: bool = True,
        invalid_config: dict = None,
    ):
        logging.info(f"Loading dataset from {filename}")
        data_path = data_dir.joinpath(filename)
        data_dict = load_data(datafile=data_path)
        return cls.load_from_data_dict_impl(data_dict, is_apply_cleaner, invalid_config)


class TrajectoriesDummyLoader(TrajectoriesLoader):
    @classmethod
    def get_invalid_data(
        cls, trajectories: List[Tuple[np.ndarray, np.ndarray]], **configs
    ) -> Tuple[Sequence[int], dict]:
        return [], {}

    @classmethod
    def clean_data_info_iter(cls, invalid_indexes, invalid_info):
        logging.debug("Dummy cleaning... nothing is cleaned")

    @classmethod
    def clean_data_info_summary(
        cls, all_invalid_indexes: List[List[int]], all_invalid_info: List[dict]
    ):
        logging.debug("Dummy cleaning... cleaning is done")


def load_data(datafile) -> dict:
    with open(datafile, "rb") as f:
        data = pickle.load(f)
    return data


def convert_data_to_numpy_array(
    data: Dict[int, dict]
) -> Tuple[Dict[str, Any], List[int]]:
    # noinspection PyTypeChecker
    data = collections.OrderedDict(sorted(data.items()))
    try:
        data1 = next(iter(data.values()))
    except StopIteration:
        return {}, []
    keys = tuple(data1.keys())
    data_dict = {}
    for k in keys:
        if isinstance(data1[k], (int, float)):
            data_dict[k] = np.array([data[d][k] for d in data])
            continue
        if isinstance(data1[k], np.ndarray):
            data_dict[k] = [data[d][k] for d in data]
            continue
        if isinstance(data1[k], pin.SE3):
            raise NotImplementedError("SE3 is not supported yet")
        if isinstance(data1[k], bool):
            assert k == "converged", "Only `converged` is known to be bool."
            continue
        if isinstance(data1[k], dict):
            assert (
                k == "cost_config"
            ), "Only `cost_config` is known to be dict (and ignored)."
            continue
        warnings.warn(f"Unknown data type for {k}: {type(data1[k])}. It is ignored.")
    return data_dict, list(data.keys())


def get_index_mapping(valid: List[bool]):
    """from local index to global index, index[local_i] = global_i"""
    index = [i for i, v in enumerate(valid) if v]
    return index

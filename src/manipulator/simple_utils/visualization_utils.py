import functools
from typing import List, Optional, Tuple, Union, Sequence

import numpy as np
from matplotlib import pyplot as plt


def show_or_save(plt_fcn):
    @functools.wraps(plt_fcn)
    def show_or_save_plot(*args, save_path=None, fig_title=None, **kwargs):
        results = plt_fcn(*args, **kwargs)
        if fig_title is not None:
            plt.gcf().suptitle(fig_title)
            plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()
        return results

    return show_or_save_plot


def filter_data_according_to_range(
    list_of_arr_list: List[List[np.ndarray]], range_list
):
    number_of_lists = len(list_of_arr_list)
    if number_of_lists == 0:
        return list_of_arr_list
    range_list = handle_range_list(range_list, len(list_of_arr_list[0]))
    list_of_filtered_arr_list = [[] for _ in range(len(list_of_arr_list))]
    # each list in list_of_arr_list should have the same number of elements as range_list
    for i, filter_range in enumerate(range_list):
        for n in range(number_of_lists):
            filtered_arr = list_of_arr_list[n][i]
            if filter_range:
                filtered_arr = filtered_arr[filter_range[0] : filter_range[1]]
            list_of_filtered_arr_list[n].append(filtered_arr)
    return list_of_filtered_arr_list, range_list


def compute_difference_within_range(x_list, x_ref, range_list):
    return [
        x - (x_ref[ran[0] : ran[1]] if ran else x_ref)
        for x, ran in zip(x_list, range_list)
    ]


def handle_range_list(range_list, length: int) -> List[Optional[Tuple[int, int]]]:
    range_list = range_list if range_list is not None else [None] * length
    if hasattr(range_list, "__len__") and len(range_list) == 2:
        if isinstance(range_list[0], int):
            # all trajectories share the same range
            range_list = [range_list] * length
    if isinstance(range_list, int):
        range_list = [(0, range_list)] * length
    return range_list


@show_or_save
def compare_raw_solution(
    x_arr_list,
    t_arr_list,
    label_names,
    ls_list,
    range_list: Optional[Union[int, Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    to_plot="raw",
    ref: np.ndarray = None,
    figsize=(12, 20),
    clip_value_at: float = None,
    u_arr_list=None,
    u_ref=None,
):
    plt_control = u_arr_list is not None
    n_plots = x_arr_list[0].shape[1] // 2
    n_cols = 2 + int(plt_control)
    figsize = (figsize[0] * n_cols / 2, figsize[1])
    fig, axes = plt.subplots(n_plots, n_cols, figsize=figsize)
    if plt_control:
        (
            x_arr_list,
            u_arr_list,
            t_arr_list,
        ), range_list = filter_data_according_to_range(
            [x_arr_list, u_arr_list, t_arr_list], range_list
        )
    else:
        (x_arr_list, t_arr_list), range_list = filter_data_according_to_range(
            [x_arr_list, t_arr_list], range_list
        )
        u_arr_list = [None] * len(x_arr_list)

    data_to_plot = x_arr_list
    if u_ref is not None:
        u_arr_list = compute_difference_within_range(u_arr_list, u_ref, range_list)
    if to_plot.startswith("err") or ref is not None:
        data_to_plot = compute_difference_within_range(x_arr_list, ref, range_list)
        if clip_value_at:
            data_to_plot = [
                np.clip(data, -clip_value_at, clip_value_at) for data in data_to_plot
            ]
        if "abs" in to_plot:
            data_to_plot = [abs(arr) for arr in data_to_plot]

    for i in range(n_plots):
        for data_arr, u_arr, label, t_arr, ls, plt_range in zip(
            data_to_plot, u_arr_list, label_names, t_arr_list, ls_list, range_list
        ):
            axes[i, 0].plot(t_arr, data_arr[:, i], ls=ls, label=label)
            axes[i, 1].plot(t_arr, data_arr[:, n_plots + i], ls=ls, label=label)
            if plt_control:
                axes[i, 2].plot(t_arr[: len(u_arr)], u_arr[:, i], ls=ls, label=label)
        axes[i, 0].set_ylabel(f"$q_{i + 1}$")
        axes[i, 1].set_ylabel(f"$v_{i + 1}$")
        if "abs" in to_plot:
            axes[i, 0].set_yscale("log")
            axes[i, 1].set_yscale("log")
        if plt_control:
            axes[i, 2].set_ylabel(f"$u_{i + 1}$")

        fig.legend(*axes[i, 0].get_legend_handles_labels(), loc="center right")
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.99, bottom=0.01)

"""Prints the commands to generate the data files required for each figure/table in the paper.

The plotting code calls this, to print out the required commands before attempting to plot.
"""
import os
from typing import List

import experiment_common


def command_printer(func):
    def wrapper(*args, **kwargs):
        print("#### COMMANDS TO GENERATE DATA FILES ####")
        func(*args, **kwargs)
        print("#### END OF COMMANDS ####")

    return wrapper


def train_command(
    dataset: str, split: int, depth: int, K: int, fixq: bool = False, dreg: bool = False
) -> str:
    arg_list = [  #
        f"--dataset={dataset}",  #
        f"--split={split}",  #
        f"--configuration=L1" + "_G5" * (depth - 1),  #
        f"--num_IW_samples={K}",  #
        f"--minibatch_size=64",  #
        f"--mode=IWAE",  #
        f"--iterations=300000",  #
    ]

    if fixq:
        arg_list.append("--qz_mode=fix_mu_sigma")
        arg_list.append("--name_suffix=fixq")

    if dreg:
        arg_list.append("--use_dreg")

    args = experiment_common.parse_arguments(arg_list)

    print(f'python train_model.py {" ".join(arg_list)}')

    return experiment_common.get_file_name(args)


def sample_command(
    path: str, ks: List[int], aggregate: bool, dreg: bool, num_datapoints: int = 1, batch_size=1
):
    aggregate_str = "--save_aggregates" if aggregate else ""
    dreg_str = "--dreg" if dreg else ""
    ks_str = ",".join([str(k) for k in ks])
    path_str = os.path.join("results", path)
    print(
        f"python sample_gradients.py --gpu {aggregate_str} --ks={ks_str} --num_datapoints={num_datapoints} "
        f"{dreg_str} --results_path={path_str} --batch_size={batch_size}"
    )


@command_printer
def figure_1():
    name = train_command("demo_normalized", split=0, depth=1, K=50)
    sample_command(
        name, ks=[1] + list(range(5, 101, 5)), aggregate=True, dreg=False, num_datapoints=10
    )
    sample_command(
        name, ks=[1] + list(range(5, 101, 5)), aggregate=True, dreg=True, num_datapoints=10
    )


@command_printer
def figure_2():
    name = train_command("demo_normalized", split=0, depth=2, K=50)
    sample_command(
        name, ks=[1] + list(range(5, 101, 5)), aggregate=False, dreg=False, num_datapoints=1
    )
    sample_command(
        name, ks=[1] + list(range(5, 101, 5)), aggregate=False, dreg=True, num_datapoints=1
    )


@command_printer
def figure_3(datasets: str, Ks: List[int], depths: List[int]):
    for dataset in datasets:
        for depth in depths:
            name = train_command(dataset, split=0, depth=depth, K=50)
            for dreg in [False, True]:
                sample_command(name, ks=Ks, aggregate=True, dreg=dreg, num_datapoints=10)


@command_printer
def figure_4(datasets: List[str], results_dir: str):
    for dataset in datasets:
        for fixq in [False, True]:
            for split in range(10):
                train_command(dataset, split=split, depth=2, fixq=fixq, K=50)
    print(f"python compute_tlls_and_elbos.py --figure=figure4 --results_root={results_dir}")


@command_printer
def figure_5():
    for dataset in ["wilson_forest", "winewhite"]:
        for fixq in [False, True]:
            train_command(dataset, split=0, depth=2, fixq=fixq, K=50)


@command_printer
def table_1(results_dir: str):
    for dataset in [
        "wilson_forest",
        "wilson_solar",
        "power",
        "wilson_pol",
        "winewhite",
        "winered",
    ]:
        for split in range(20):
            for dreg in [False, True]:
                train_command(dataset, split=split, depth=2, K=50, dreg=dreg)
    print(
        f"python compute_tlls_and_elbos.py --figure=table1 --results_root={results_dir} --metric=elbo"
    )
    print(
        f"python compute_tlls_and_elbos.py --figure=table1 --results_root={results_dir} --metric=tll"
    )


@command_printer
def figure_9(datasets: List[str], sample_batch_sizes: List[int]):
    for dataset in datasets:
        name = train_command(dataset, split=0, depth=3, K=30)
        for batch_size in sample_batch_sizes:
            sample_command(
                name,
                ks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
                aggregate=True,
                dreg=False,
                num_datapoints=10,
                batch_size=batch_size,
            )


@command_printer
def figure_8(names: List[str]):
    for name in names:
        for dreg in [False, True]:
            sample_command(
                name, ks=[50], aggregate=True, num_datapoints=10, batch_size=64, dreg=dreg
            )

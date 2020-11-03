"""Contains utility functions used by both training and gradient sampling."""
import csv
import os
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Iterable, List, Optional, Tuple

import bayesian_benchmarks
import numpy as np
from bayesian_benchmarks.data import get_regression_data
from numpy import ndarray

import demo_dataset
from demo_dataset import Data


def _parse_int_or_none(x: str) -> Optional[int]:
    if x == "None":
        return None
    else:
        try:
            return int(x)
        except ValueError as exception:
            raise ArgumentTypeError(f"Could not parse as int: '{x}'")


def parse_arguments(arg_list: Optional[List[str]] = None) -> Namespace:
    """Parses and returns the command line arguments."""
    parser = ArgumentParser()

    # model
    parser.add_argument("--configuration", default="L1", nargs="?", type=str)
    parser.add_argument("--mode", default="IWAE", nargs="?", type=str)
    parser.add_argument("--M", default=128, nargs="?", type=int)
    parser.add_argument("--likelihood_variance", default=1e-2, nargs="?", type=float)
    parser.add_argument("--num_IW_samples", default=5, nargs="?", type=int)
    parser.add_argument(
        "--encoder_dims",
        default="20,20",
        nargs="?",
        type=str,
        help="Dimensions of hidden layers in the encoder.",
    )
    parser.add_argument(
        "--qz_mode", choices=["encoder", "fix_mu_sigma"], type=str, default="encoder"
    )

    # training
    parser.add_argument("--minibatch_size", default=512, nargs="?", type=_parse_int_or_none)
    parser.add_argument(
        "--encoder_minibatch_size", default=None, nargs="?", type=_parse_int_or_none
    )
    parser.add_argument("--iterations", default=5000, nargs="?", type=int)
    parser.add_argument("--gamma", default=1e-2, nargs="?", type=float)
    parser.add_argument("--gamma_decay", default=0.98, nargs="?", type=float)
    parser.add_argument("--lr", default=5e-3, nargs="?", type=float)
    parser.add_argument("--encoder_lr", default=5e-3, nargs="?", type=float)
    parser.add_argument("--lr_decay", default=0.98, nargs="?", type=float)
    parser.add_argument("--encoder_lr_decay", default=0.98, nargs="?", type=float)
    parser.add_argument("--fix_linear", default=True, nargs="?", type=bool)
    parser.add_argument("--num_predict_samples", default=2000, nargs="?", type=int)
    parser.add_argument(
        "--predict_batch_size", default=1000, nargs="?", type=int
    )  ## was 10 for experiments
    parser.add_argument("--use_dreg", action="store_true")
    parser.add_argument(
        "--optimizer", default="adam", choices=["adam", "sgd"], nargs="?", type=str
    )
    parser.add_argument(
        "--encoder_optimizer", default="adam", choices=["adam", "sgd"], nargs="?", type=str
    )
    parser.add_argument("--assert_no_nans", action="store_true")
    parser.add_argument("--clip_encoder_grads", default=None, nargs="?", type=float)

    # data
    parser.add_argument("--dataset", default="energy", nargs="?", type=str)
    parser.add_argument("--split", default=0, nargs="?", type=int)
    parser.add_argument("--disable_split_fix", action="store_true")
    parser.add_argument("--seed", default=None, nargs="?", type=int)

    parser.add_argument("--results_path", default=None, nargs="?", type=str)
    parser.add_argument("--name_suffix", default="", nargs="?", type=str)
    parser.add_argument(
        "--override_name",
        default=None,
        type=str,
        nargs="?",
        help="Save/load checkpoints and tensorboard from this name, rather than the generated name",
    )
    parser.add_argument(
        "--log_snr_freq",
        default=None,
        nargs="?",
        type=int,
        help="How frequently (in iterations) to compute and log the SNR (omit to disable)",
    )
    parser.add_argument("--plot_freq", default=None, nargs="?", type=int)
    parser.add_argument("--log_main_freq", default=500, nargs="?", type=int)
    parser.add_argument("--log_test_freq", default=10000, nargs="?", type=int)

    parser.add_argument("--beta", default=0, nargs="?", type=float)  # for CIWAE

    parser.add_argument("--use_nat_grad_for_final_layer", default=True, nargs="?", type=bool)

    if arg_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_list)

    if args.seed is None:
        args.seed = args.split

    return args


def get_file_name(args) -> str:
    """Creates the base file name for this experiment from the command line arguments."""
    if args.override_name is not None and args.override_name != "":
        if args.name_suffix != "":
            raise ValueError("Cannot set --override_name and --name_suffix.")
        return args.override_name

    split_prefix = "S" if args.disable_split_fix else "SF"
    batch_size_text = "BSn" if args.minibatch_size is None else f"BS{args.minibatch_size}"
    if args.encoder_minibatch_size is not None:
        batch_size_text += f"_EBS{args.encoder_minibatch_size}"

    if args.mode == "IWAE":
        dreg_text = "_dreg" if args.use_dreg else ""
        file_name = (
            f"{args.dataset}_{args.configuration}_{args.mode}_{split_prefix}{args.split}"
            f"_K{args.num_IW_samples}_{batch_size_text}{dreg_text}"
        )
    elif args.mode == "VI":
        file_name = f"{args.dataset}_{args.configuration}_{args.mode}_{split_prefix}{args.split}_{batch_size_text}"
    elif args.mode == "CIWAE":
        file_name = (
            f"{args.dataset}_{args.configuration}_{args.mode}_{split_prefix}{args.split}"
            f"_K{args.num_IW_samples}_B{args.beta}_{batch_size_text}"
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if not args.use_nat_grad_for_final_layer:
        file_name += "_finvarnotnat"

    if args.name_suffix != "":
        file_name += f"_{args.name_suffix}"

    return file_name


def create_paths(file_name: str, results_path: Optional[str]) -> Tuple[str, str]:
    """Creates and returns the checkpoint and tensorboard paths

    :returns: (checkpoint path, tensorboard path)
    """
    if results_path is None:
        full_path = os.path.dirname(__file__)
        results_path_base = os.path.join(full_path, "results")
    else:
        results_path_base = results_path
    checkpoints_path_base = os.path.join(results_path_base, "checkpoints")
    tensorboard_path_base = os.path.join(results_path_base, "tensorboard")

    tensorboard_path = os.path.join(tensorboard_path_base, file_name)
    checkpoint_path = os.path.join(checkpoints_path_base, file_name)

    for p in [
        results_path_base,
        tensorboard_path_base,
        checkpoints_path_base,
        tensorboard_path,
        checkpoint_path,
    ]:
        if not os.path.isdir(p):
            os.mkdir(p)

    return checkpoint_path, tensorboard_path


def get_data(args) -> Data:
    """Gets the dataset specified in the command line arguments from Bayesian benchmarks."""
    if args.dataset == "two_points":
        X = np.array([[-1.0], [1.0]])
        Y = np.array([[1.0], [2.0]])
        return Data(X_train=X, Y_train=Y, X_test=X, Y_test=Y)

    elif args.dataset == "demo":
        X_train, Y_train, X_test, Y_test = demo_dataset.create_demo_data()
        return Data(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    elif args.dataset == "demo_normalized":
        X_train, Y_train, X_test, Y_test = _normalize(demo_dataset.create_demo_data())
        return Data(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

    else:
        if not args.disable_split_fix:
            dataset = get_regression_data(args.dataset, split=args.split)
        else:
            dataset = get_regression_data(args.dataset)
        X_test = dataset.X_test[:10000]
        Y_test = dataset.Y_test[:10000]
        return _remove_unused_dimensions(Data(dataset.X_train, dataset.Y_train, X_test, Y_test))


def _normalize(xs: Iterable[ndarray]) -> List[ndarray]:
    return [bayesian_benchmarks.data.normalize(x)[0] for x in xs]


def _remove_unused_dimensions(data: Data) -> Data:
    train_used_dims = np.argwhere(data.X_train.std(axis=0) != 0.0).squeeze(axis=1)
    test_used_dims = np.argwhere(data.X_test.std(axis=0) != 0.0).squeeze(axis=1)
    assert np.all(train_used_dims == test_used_dims)
    return Data(
        data.X_train[:, train_used_dims],
        data.Y_train,
        data.X_test[:, train_used_dims],
        data.Y_test,
    )


def parse_args_from_csv(name: str, results_path: Optional[str]) -> Namespace:
    """Restores arguments from a results csv file."""
    stats = ["test_loglik", "train_elbo", "test_elbo", "test_shapiro_W_median", "test_rmse"]
    store_true_flags = ["use_dreg", "assert_no_nans", "disable_split_fix"]
    none_flags = [
        "encoder_minibatch_size",
        "clip_encoder_grads",
        "seed",
        "override_name",
        "log_snr_freq",
        "plot_freq",
    ]

    rename_flags = {
        "lr_dreg": "encoder_lr",
        "lr_decay_dreg": "encoder_lr_decay",
    }

    delete_flags = ["sample_likelihood"]

    if results_path is None:
        results_path = "results"
    file_path = os.path.join(results_path, "checkpoints", name, f"{name}_0.csv")
    arg_list = []
    with open(file_path) as file:
        for key, value in csv.reader(file):
            if key in rename_flags:
                renamed_key = rename_flags[key]
                print(f'Renaming flag "{key}" to "{renamed_key}"')
                key = renamed_key

            if key in delete_flags:
                print(f'Deleting flag "{key}"')
                continue

            if key in stats:
                # The CSV file contains both configuration key/values and stats, we skip the stats.
                continue
            elif key in store_true_flags:
                # store_true flags either appear or don't.
                if value == "True":
                    arg_list.append(f"--{key}")
            elif key in none_flags and value == "":
                # These flags have None as the default value, however this gets saved to the CSV as ''.
                continue
            else:
                arg_list.append(f"--{key}={value}")

    args = parse_arguments(arg_list)

    expected_name = get_file_name(args)
    if expected_name != name:
        raise ValueError(f"Expected name {expected_name}, got name {name}")

    return args

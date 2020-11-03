"""Computes the test log-likelihood or train elbo from a model checkpoints.

Saves results to text files, which create_figures.py can load.
"""
import itertools
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Optional

import tensorflow as tf

import experiment_common
import metrics
from build_models import build_model


def table_1(results_root: str, metric: str) -> None:
    datasets = ["wilson_forest", "power", "wilson_pol", "wilson_solar", "winewhite", "winered"]

    values = defaultdict(lambda: {False: {}, True: {}})

    for dataset in datasets:
        n_splits = 0
        for split in itertools.count(0, 1):
            if split > 40:
                raise ValueError(f"Not enough splits for {dataset}")

            reg_value = None
            dreg_value = None
            for dreg in [False, True]:
                suffix = "_noise0.2" if dataset == "demo_normalized" else ""
                dreg_text = "_dreg" if dreg else ""
                name = f"{dataset}_L1_G5_IWAE_SF{split}_K50_BS64{dreg_text}{suffix}"

                checkpoint_path, _ = experiment_common.create_paths(name, results_root)
                final_iter_file = os.path.join(checkpoint_path, f"{name}_300000.csv")
                if not os.path.exists(final_iter_file):
                    # print(f"split {split} {dataset} {dreg} missing")
                    continue

                if metric == "elbo":
                    value = compute_train_elbos(results_root, name)
                elif metric == "tll":
                    value = compute_tlls(results_root, name)
                else:
                    raise ValueError

                if dreg:
                    dreg_value = value
                else:
                    reg_value = value

            if reg_value is None or dreg_value is None:
                continue
            values[dataset][False][split] = reg_value
            values[dataset][True][split] = dreg_value

            n_splits += 1
            if n_splits == 20:
                break

    with open(f"figure_data/table1_{metric}.txt", mode="w") as results_file:
        for dataset in datasets:
            for dreg in [False, True]:
                dreg_str = "T" if dreg else "F"
                for split, results in sorted(values[dataset][dreg].items()):
                    results_str = " ".join([str(r) for r in results])
                    results_file.write(f"{dataset} {dreg_str} {split} {results_str}")


def figure_4(results_root: str) -> None:
    datasets = ["wilson_forest", "power", "wilson_pol", "wilson_solar", "winewhite", "winered"]
    want_n_splits = 10

    values = defaultdict(lambda: {False: {}, True: {}})

    for dataset in datasets:
        n_splits = 0
        for split in itertools.count(0, 1):
            if split > 40:
                raise ValueError(f"Not enough splits for {dataset}")

            reg_value = None
            fixq_value = None
            for fixq in [False, True]:
                fixq_text = "_fixq" if fixq else ""
                name = f"{dataset}_L1_G5_IWAE_SF{split}_K50_BS64{fixq_text}"

                checkpoint_path, _ = experiment_common.create_paths(name, results_root)
                final_iter_file = os.path.join(checkpoint_path, f"{name}_300000.csv")
                if not os.path.exists(final_iter_file):
                    continue

                if fixq:
                    fixq_value = compute_tlls(results_root, name)
                else:
                    reg_value = compute_tlls(results_root, name)

            if reg_value is None or fixq_value is None:
                continue
            values[dataset][False][split] = reg_value
            values[dataset][True][split] = fixq_value

            n_splits += 1
            if n_splits == want_n_splits:
                break

    with open(f"figure_data/figure_4.txt", mode="w") as results_file:
        for dataset in datasets:
            for fixq in [False, True]:
                fixq_str = "T" if fixq else "F"
                for split, results in sorted(values[dataset][fixq].items()):
                    results_str = " ".join([str(r) for r in results])
                    results_file.write(f"{dataset} {fixq_str} {split} {results_str}")


def compute_tlls(results_root: str, name: str) -> List[float]:
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph).as_default():
            args = experiment_common.parse_args_from_csv(name, results_root)
            checkpoint_path, _ = experiment_common.create_paths(name, results_root)

            data = experiment_common.get_data(args)
            model = build_model(args, data.X_train, data.Y_train)
            sess = model.enquire_session()
            model.init_op(sess)

            step = 300000
            saver = tf.train.Saver()
            checkpoint_file = _get_checkpoint_file(checkpoint_path, step)
            saver.restore(sess, checkpoint_file)

            assert sess.run(model.global_step) == step

            return [
                metrics.compute_log_likelihood(model, data.X_test, data.Y_test, n_samples=10000)
                for _ in range(10)
            ]


def compute_train_elbos(results_root: str, name: str) -> List[float]:
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph).as_default():
            args = experiment_common.parse_args_from_csv(name, results_root)
            checkpoint_path, _ = experiment_common.create_paths(name, results_root)

            data = experiment_common.get_data(args)
            # Disable minibatching so we just compute the ELBO over the entire dataset at once.
            args.minibatch_size = None
            model = build_model(args, data.X_train, data.Y_train)
            sess = model.enquire_session()
            model.init_op(sess)

            step = 300000
            saver = tf.train.Saver(
                var_list=[v for v in tf.all_variables() if "dataholder" not in v.name]
            )
            checkpoint_file = _get_checkpoint_file(checkpoint_path, step)
            saver.restore(sess, checkpoint_file)

            assert sess.run(model.global_step) == step

            return [sess.run(model.likelihood_tensor).item() for _ in range(10)]


def _get_checkpoint_file(checkpoint_path: str, step_to_load: Optional[int]) -> str:
    if step_to_load is not None:
        return os.path.join(checkpoint_path, f"cp-{step_to_load}")
    else:
        return tf.train.latest_checkpoint(checkpoint_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--figure", choices=["table1", "figure4"], required=True)
    arg_parser.add_argument("--results_root", type=str, required=True)
    arg_parser.add_argument("--metric", choices=["elbo", "tll"], default="elbo")
    args = arg_parser.parse_args()

    if args.figure == "table1":
        table_1(args.results_root, args.metric)
    elif args.figure == "figure4":
        figure_4(args.results_root)

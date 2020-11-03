"""Trains DGP models.

You can use this to create checkpoints which you can analyze with sample_gradients.py

Adapted from
https://github.com/hughsalimbeni/DGPs_with_IWVI/blob/master/experiments/run_conditional_density_estimation.py
"""

import csv
import os
import random
import warnings
from typing import Dict

import numpy as np
import tensorflow as tf
from gpflow.training.monitor import (
    CallbackTask,
    CheckpointTask,
    LogdirWriter,
    ModelToTensorBoardTask,
    Monitor,
    MonitorContext,
    MonitorTask,
    PeriodicIterationCondition,
    PrintTimingsTask,
    restore_session,
)
from tensorflow_core.core.framework.summary_pb2 import Summary

import experiment_common
import metrics
import sample_gradients
from build_models import build_model
from dgps_with_iwvi.dreg_optimizer import DregModel
from dgps_with_iwvi.models import DGP_VI
import demo_dataset

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


def main():
    #################################### args

    ARGS = experiment_common.parse_arguments()
    print("\n", "ARGS:", "\n", ARGS, "\n")

    if ARGS.plot_freq is not None and not ARGS.dataset.startswith("demo"):
        raise ValueError("Plotting only supported for demo dataset.")

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    tf.set_random_seed(ARGS.seed)

    #################################### paths

    file_name = experiment_common.get_file_name(ARGS)
    checkpoint_path, tensorboard_path = experiment_common.create_paths(
        file_name, ARGS.results_path
    )

    #################################### data

    data = experiment_common.get_data(ARGS)

    #################################### model

    model = build_model(ARGS, data.X_train, data.Y_train)

    #################################### init

    sess = model.enquire_session()
    model.init_op(sess)

    #################################### monitoring

    def _write_dict_to_csv(data: Dict, step: int):
        csvsavepath = os.path.join(checkpoint_path, f"{file_name}_{step}.csv")
        with open(csvsavepath, "w") as file:
            writer = csv.writer(file)
            for key, val in data.items():
                writer.writerow([key, val])
        print("CSV WRITTEN " + csvsavepath)

    #################################### training

    tensorboard_writer = LogdirWriter(tensorboard_path)
    checkpoint_task = _create_checkpoint_task(checkpoint_path)
    snr_log_task = _create_snr_log_task(ARGS, checkpoint_path, tensorboard_writer, checkpoint_task)
    tensorboard_task = _create_tensorboard_task(model, tensorboard_writer, ARGS.log_main_freq)
    monitor_tasks = [
        checkpoint_task,  #
        snr_log_task,  #
        PrintTimingsTask()
        .with_name("print")
        .with_condition(PeriodicIterationCondition(interval=100)),
        tensorboard_task,
    ]

    with Monitor(monitor_tasks, sess, model.global_step, print_summary=True) as monitor:
        try:
            restore_session(sess, checkpoint_path)
        except ValueError:
            pass

        initial_global_step = sess.run(model.global_step)
        iterations_to_go = max([ARGS.iterations - initial_global_step, 0])

        if initial_global_step == 0:
            # Log initial values. Bit dodgy.
            tensorboard_task.run(monitor._context)

            if ARGS.log_snr_freq is not None:
                snr_log_task.run(monitor._context)

            if ARGS.plot_freq is not None:
                demo_dataset.plot_data_and_predictions(
                    data, model, tensorboard_writer, sess, step=0
                )

        print(
            "Already run {} iterations. Running {} iterations".format(
                initial_global_step, iterations_to_go
            )
        )

        epoch_train_elbos = []
        epoch_train_dreg_objectives = []
        datapoints_since_last_epoch = 0
        batching_enabled = ARGS.minibatch_size is not None
        minibatch_size = ARGS.minibatch_size if batching_enabled else len(data.X_train)

        for it in range(iterations_to_go):
            monitor()
            if isinstance(model, DregModel) and hasattr(model, "train_op"):
                _, train_elbo, train_dreg_objective = sess.run(
                    [
                        model.train_op,
                        model.likelihood_tensor,
                        model.get_dreg_objective_for_encoder_params(),
                    ]
                )
                epoch_train_elbos.append(train_elbo)
                epoch_train_dreg_objectives.append(train_dreg_objective)
            else:
                model.train_func(sess)

            global_step = sess.run(model.global_step)

            datapoints_since_last_epoch += minibatch_size
            # If batching is disabled then we use the entire dataset each iteration, so there is no point in recording
            # separate epoch statistics.
            if batching_enabled and datapoints_since_last_epoch >= len(data.X_train):
                # We have passed over the entire dataset, so compute epoch stats.
                epoch_train_elbo = np.mean(np.stack(epoch_train_elbos, axis=0))
                tensorboard_writer.add_summary(
                    _create_scalar_summary("optimisation/epoch_train_elbo", epoch_train_elbo),
                    global_step,
                )
                epoch_train_dreg_objective = np.mean(np.stack(epoch_train_dreg_objectives, axis=0))
                tensorboard_writer.add_summary(
                    _create_scalar_summary(
                        "optimisation/epoch_train_dreg_objective", epoch_train_dreg_objective
                    ),
                    global_step,
                )

                datapoints_since_last_epoch = 0
                epoch_train_elbos = []
                epoch_train_dreg_objectives = []

            if (
                ARGS.plot_freq is not None
                and (global_step - 1) % ARGS.plot_freq == 0
                or it == iterations_to_go
            ):
                demo_dataset.plot_data_and_predictions(
                    data, model, tensorboard_writer, sess, global_step
                )

            if (global_step - 1) % ARGS.log_test_freq == 0 or it == iterations_to_go:
                print("Iteration: {}".format(it))
                #################################### evaluation
                test_elbo = model.compute_log_likelihood(data.X_test)
                loglik, rmse, median_shapiro_W = metrics.compute_metrics(
                    model, data.X_test, data.Y_test, ARGS.num_predict_samples
                )
                res = {}
                res["test_loglik"] = loglik
                res["train_elbo"] = model.compute_log_likelihood(data.X_train)
                res["test_elbo"] = test_elbo
                res["test_shapiro_W_median"] = median_shapiro_W
                res["test_rmse"] = rmse
                res.update(ARGS.__dict__)
                print(res)
                _write_dict_to_csv(res, step=sess.run(model.global_step) - 1)

                tensorboard_writer.add_summary(
                    _create_scalar_summary("optimisation/test_elbo", test_elbo), global_step
                )
                tensorboard_writer.add_summary(
                    _create_scalar_summary("optimisation/test_loglik", loglik), global_step
                )

        model.anchor(sess)
        print(model.as_pandas_table())

    ####################################

    loglik, rmse, median_shapiro_W = metrics.compute_metrics(
        model, data.X_test, data.Y_test, ARGS.num_predict_samples
    )
    res = {}
    res["test_loglik"] = loglik
    res["train_elbo"] = model.compute_log_likelihood(data.X_train)
    res["test_elbo"] = model.compute_log_likelihood(data.X_test)
    res["test_shapiro_W_median"] = median_shapiro_W
    res["test_rmse"] = rmse

    res.update(ARGS.__dict__)
    print(res)

    ################################### save results as csv files for tighter bounds

    _write_dict_to_csv(res, step=sess.run(model.global_step))


def _create_checkpoint_task(checkpoint_path: str) -> MonitorTask:
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    return (
        CheckpointTask(checkpoint_dir=checkpoint_path, saver=saver)
        .with_name("checkpoint")
        .with_condition(PeriodicIterationCondition(interval=500))
        .with_exit_condition(True)
    )


def _create_tensorboard_task(
    model: DGP_VI, tensorboard_writer: LogdirWriter, freq: int
) -> MonitorTask:
    if isinstance(model, DregModel):
        model_summaries = model.get_tf_summaries()
        model_summaries.append(tf.summary.scalar("optimisation/learning_rate", model.lr))
    else:
        model_summaries = None

    return (
        ModelToTensorBoardTask(tensorboard_writer, model, additional_summaries=model_summaries)
        .with_name("tensorboard")
        .with_condition(PeriodicIterationCondition(interval=freq))
    )


def _create_snr_log_task(
    args, checkpoint_path: str, tensorboard_writer: LogdirWriter, checkpoint_task: MonitorTask
) -> MonitorTask:
    if args.log_snr_freq is None:
        # Return a task that does nothing.
        return CallbackTask(lambda *args, **kwargs: None)

    def log_snr(context: MonitorContext):
        # Create a checkpoint, as we depend on the checkpoint existing.
        # TODO: avoid creating duplicate checkpoints.
        checkpoint_task.run(context)

        snr_batch_size = 64
        param_snrs, mean_snr = sample_gradients.compute_snr_for_tensorboard(
            args, checkpoint_path, snr_batch_size
        )
        global_step = context.global_step
        tensorboard_writer.add_summary(
            _create_scalar_summary(f"snrs/mean:BS={snr_batch_size}", mean_snr), global_step
        )
        for param, snr in param_snrs.items():
            tensorboard_writer.add_summary(
                _create_scalar_summary(f"snrs/{param}:BS={snr_batch_size}", snr), global_step
            )

    return CallbackTask(log_snr).with_condition(PeriodicIterationCondition(args.log_snr_freq))


def _create_scalar_summary(tag: str, value) -> Summary:
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    return summary


if __name__ == "__main__":
    main()

"""Loads a model from a checkpoint and samples the gradients of the encoder parameters."""

import os
import random
from argparse import ArgumentParser
from copy import copy
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tensorflow as tf
from gpflow.training.monitor import restore_session
from numpy import ndarray
from tqdm import tqdm

import experiment_common
from build_models import build_model
from dgps_with_iwvi.dreg_optimizer import BatchRange
from experiment_common import parse_args_from_csv

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0


def _compute_and_save_aggregates(
    name: str,
    batch_size: int,
    num_datapoints: int,
    Ks: List[int],
    num_samples: int,
    results_path: Optional[str],
    step_to_load: Optional[int],
    dreg: bool,
    gpu: bool,
):
    print(f"Loading {name}")
    args = parse_args_from_csv(name, results_path)
    checkpoint_path, _ = experiment_common.create_paths(name, results_path)
    file_prefix = _get_file_prefix("gradags", batch_size, dreg)

    if num_datapoints != 1 and batch_size == 1:
        datapoints = range(num_datapoints)
        batch_ranges = [(i, i + 1) for i in datapoints]
    elif num_datapoints == 1:
        datapoints = [None]
        batch_ranges = [(0, batch_size)]
    else:
        raise ValueError(
            "Cannot aggregate snrs over more than one datapoint if batch size is not 1"
        )

    print("Computing snrs...")
    for K in _get_Ks_to_compute(Ks, checkpoint_path, file_prefix):
        for datapoint, batch_range in zip(datapoints, batch_ranges):
            print(f"Computing for datapoint {datapoint}, batch range {batch_range}")
            gradients = _sample_gradients_from_checkpoint(
                args,
                checkpoint_path,
                step_to_load,
                batch_size,
                batch_range,
                K,
                dreg,
                gpu,
                num_samples,
            )
            aggregates: Dict[str, Dict[str, ndarray]] = {"mean": {}, "std": {}}
            for param, grads in gradients.items():
                aggregates["mean"][param] = np.mean(grads, axis=0)
                aggregates["std"][param] = np.std(grads, axis=0)
                if np.any(aggregates["mean"][param] == 0.0):
                    print(f"Gradient mean was zero for {param} at K={K}")

            datapoint_suffix = "" if num_datapoints == 1 else f"_dp{datapoint}"
            file_path = os.path.join(checkpoint_path, f"{file_prefix}k{K}{datapoint_suffix}.npy")
            np.save(file_path, aggregates)


def _sample_and_save_gradients(
    name: str,
    batch_size: int,
    Ks: List[int],
    num_samples: int,
    results_path: Optional[str],
    step_to_load: Optional[int],
    dreg: bool,
    gpu: bool,
):
    print(f"Loading {name}")
    args = parse_args_from_csv(name, results_path)
    checkpoint_path, _ = experiment_common.create_paths(name, results_path)
    file_prefix = _get_file_prefix("gradients", batch_size, dreg)

    print("Computing gradients...")
    for K in _get_Ks_to_compute(Ks, checkpoint_path, file_prefix):
        gradients = _sample_gradients_from_checkpoint(
            args,
            checkpoint_path,
            step_to_load,
            batch_size,
            batch_range=(0, 1),
            K=K,
            dreg=dreg,
            gpu=gpu,
            num_samples=num_samples,
        )
        file_path = os.path.join(checkpoint_path, f"{file_prefix}k{K}.npy")
        np.save(file_path, gradients)


def _get_file_prefix(type: str, batch_size: int, dreg: bool) -> str:
    if dreg:
        dreg_ext = "_dreg"
    else:
        dreg_ext = ""
    return f"{type}_bs{batch_size}{dreg_ext}_"


def _get_Ks_to_compute(Ks: List[int], checkpoint_path: str, file_prefix: str) -> Set[int]:
    existing_Ks = set(_get_existing_Ks(checkpoint_path, file_prefix))
    missing_Ks = set(Ks).difference(existing_Ks)
    print(f"Already had K={existing_Ks}, computing gradients for K={missing_Ks}")
    return missing_Ks


def _get_existing_Ks(path: str, file_prefix: str) -> List[int]:
    ks = []
    for file_name in os.listdir(path):
        if file_name.startswith(file_prefix + "k") and file_name.endswith(".npy"):
            name_parts = file_name.split(".")[0].split("_")
            for part in name_parts:
                if part.startswith("k"):
                    ks.append(int(part.replace("k", "")))
                    break
    return ks


def _sample_gradients_from_checkpoint(
    args,
    checkpoint_path,
    step_to_load: Optional[int],
    batch_size: int,
    batch_range: BatchRange,
    K: int,
    dreg: bool,
    gpu: bool,
    num_samples: int,
) -> Dict[str, ndarray]:
    """Samples N gradients from the model, loaded from the latest available checkpoint.

    :returns: dict param_name -> [N x [grad dimension]]
    """
    config = tf.ConfigProto(device_count={"GPU": 0}) if not gpu else tf.ConfigProto()
    with tf.Graph().as_default() as graph:
        with tf.Session(config=config, graph=graph).as_default():
            return _sample_gradients_from_checkpoint_inner(
                args, checkpoint_path, step_to_load, batch_size, batch_range, K, dreg, num_samples
            )


def _sample_gradients_from_checkpoint_inner(
    args,
    checkpoint_path,
    step_to_load: Optional[int],
    batch_size: int,
    batch_range: BatchRange,
    K: int,
    dreg: bool,
    N: int,
) -> Dict[str, ndarray]:
    _set_random_seeds(0)

    # Disable minibatching, so we get the same data each time.
    args = copy(args)
    args.minibatch_size = batch_size
    args.num_IW_samples = K
    args.use_dreg = dreg

    data = experiment_common.get_data(args)
    model = build_model(args, data.X_train, data.Y_train)

    sess = model.enquire_session()
    model.init_op(sess)

    if dreg:
        objective = model.get_dreg_objective_for_encoder_params(fixed_batch=batch_range)
    else:
        objective = model.get_reg_objective_for_encoder_params(fixed_batch=batch_range)
    gradient_tensors = tf.gradients(objective, model.encoder_params)

    # As we use our own batch, don't restore variables related to the data from the checkpoint.
    saver = tf.train.Saver(var_list=[v for v in tf.all_variables() if "dataholder" not in v.name])
    checkpoint_file = _get_checkpoint_file(checkpoint_path, step_to_load)
    print(f"Loading checkpoint {checkpoint_file}")
    saver.restore(sess, checkpoint_file)

    print(f"K={K}, at iteration {sess.run(model.global_step)}, computing gradients")

    gradients = []
    for _ in tqdm(range(N)):
        gradients.append(sess.run(gradient_tensors))

    param_names = [param.name for param in model.encoder_params]
    gradients_stacked = [np.stack(grads_for_param, axis=0) for grads_for_param in zip(*gradients)]
    return {name: grads for name, grads in zip(param_names, gradients_stacked)}


def _get_checkpoint_file(checkpoint_path: str, step_to_load: Optional[int]) -> str:
    if step_to_load is not None:
        return os.path.join(checkpoint_path, f"cp-{step_to_load}")
    else:
        return tf.train.latest_checkpoint(checkpoint_path)


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def compute_snr_for_tensorboard(
    args, checkpoint_path: str, batch_size: int
) -> Tuple[Dict[str, float], float]:
    """Computes the SNR for the latest checkpoint of the model at the given path.

    The parameter SNRs are for the same parameters for which LatentVariableLayer logs the parameter values, for
    comparison purposes.

    :returns: (param snrs, mean snr), where param snrs is a dict param name -> snr
    """
    # _sample_gradients_from_checkpoint sets the random seed to zero, so sample a new seed now to reset after.
    new_random_seed = random.randint(0, 2 ** 30)
    gradients_by_param = _sample_gradients_from_checkpoint(
        args,
        checkpoint_path,
        step_to_load=None,
        batch_size=batch_size,
        batch_range=(0, batch_size),
        K=args.num_IW_samples,
        dreg=args.use_dreg,
        gpu=True,
        num_samples=1000,
    )
    _set_random_seeds(new_random_seed)
    return _compute_snr_for_params(gradients_by_param), _compute_mean_snr(gradients_by_param)


def _compute_snr_for_params(gradients_by_param: Dict[str, ndarray]) -> Dict[str, float]:
    snr_by_param = {}
    for param, gradients in gradients_by_param.items():
        if "Ws" in param:
            gradient = gradients[:, 0, 0]
            param_log_name = f"{param}_0_0"
        elif "bs" in param:
            gradient = gradients[:, 0]
            param_log_name = f"{param}_0"
        else:
            raise ValueError(f"Unknown parameter {param}")
        assert gradient.ndim == 1
        snr_by_param[param_log_name] = float(np.abs(np.mean(gradient)) / np.std(gradient))
    return snr_by_param


def _compute_mean_snr(gradients_by_param: Dict[str, ndarray]) -> float:
    flat_gradients = []
    for param, gradients in gradients_by_param.items():
        if "W" in param:
            N = gradients.shape[0]
            flat_gradients.append(gradients.reshape((N, -1)))
    gradients = np.concatenate(flat_gradients, axis=1)
    snr_by_param = np.abs(np.mean(gradients, axis=0)) / np.std(gradients, axis=0)
    return float(np.mean(snr_by_param))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("names", type=str, nargs="+")
    parser.add_argument("--batch_size", type=int, default=1, nargs="?")
    parser.add_argument("--num_datapoints", type=int, default=1, nargs="?")
    parser.add_argument("--ks", type=str, default="10,20,30,40,50", nargs="?")
    parser.add_argument("--num_samples", type=int, default=1000, nargs="?")
    parser.add_argument("--results_path", type=str, default=None, nargs="?")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        nargs="?",
        help="Which iteration to load the checkpoint from. By default loads the latest.",
    )
    parser.add_argument("--dreg", action="store_true")
    parser.add_argument("--save_aggregates", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    ks = []
    for k_str in args.ks.split(","):
        ks.append(int(k_str.strip()))

    for name in args.names:
        if args.save_aggregates:
            _compute_and_save_aggregates(
                name,
                args.batch_size,
                args.num_datapoints,
                ks,
                args.num_samples,
                args.results_path,
                args.step,
                args.dreg,
                args.gpu,
            )
        else:
            assert args.num_datapoints == 1, "Cannot save gradients for more than one datapoint."
            _sample_and_save_gradients(
                name,
                args.batch_size,
                ks,
                args.num_samples,
                args.results_path,
                args.step,
                args.dreg,
                args.gpu,
            )

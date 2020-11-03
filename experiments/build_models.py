"""
Adapted from https://github.com/hughsalimbeni/DGPs_with_IWVI/blob/master/experiments/build_models.py
"""

import numpy as np
import tensorflow as tf
from gpflow import defer_build
from gpflow.features import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Linear
from gpflow.multioutput.features import MixedKernelSharedMof
from gpflow.training import NatGradOptimizer
from scipy.cluster.vq import kmeans2
from tensorflow_core.python.ops.gen_control_flow_ops import NoOp

from dgps_with_iwvi.dreg_optimizer import DregOptimizer
from dgps_with_iwvi.layers import GPLayer, LatentVariableLayer
from dgps_with_iwvi.models import DGP_VI, DGP_IWVI, DGP_CIWAE
from dgps_with_iwvi.temp_workaround import SharedMixedMok


def build_model(ARGS, X, Y, apply_name=True):
    N, D = X.shape

    # first layer inducing points
    if N > ARGS.M:
        Z = kmeans2(X, ARGS.M, minit="points")[0]
    else:
        M_pad = ARGS.M - N
        Z = np.concatenate([X.copy(), np.random.randn(M_pad, D)], 0)

    #################################### layers
    P = np.linalg.svd(X, full_matrices=False)[2]

    layers = []

    DX = D
    DY = 1

    D_in = D
    D_out = D
    with defer_build():
        lik = Gaussian()
        lik.variance = ARGS.likelihood_variance

        if len(ARGS.configuration) > 0:
            for c, d in ARGS.configuration.split("_"):
                if c == "G":
                    num_gps = int(d)
                    A = np.zeros((D_in, D_out))
                    D_min = min(D_in, D_out)
                    A[:D_min, :D_min] = np.eye(D_min)
                    mf = Linear(A=A)
                    mf.b.set_trainable(False)

                    def make_kern():
                        k = RBF(D_in, lengthscales=float(D_in) ** 0.5, variance=1.0, ARD=True)
                        k.variance.set_trainable(False)
                        return k

                    PP = np.zeros((D_out, num_gps))
                    PP[:, : min(num_gps, DX)] = P[:, : min(num_gps, DX)]
                    ZZ = np.random.randn(ARGS.M, D_in)
                    ZZ[:, : min(D_in, DX)] = Z[:, : min(D_in, DX)]

                    kern = SharedMixedMok(make_kern(), W=PP)
                    inducing = MixedKernelSharedMof(InducingPoints(ZZ))

                    l = GPLayer(kern, inducing, num_gps, layer_num=len(layers), mean_function=mf)
                    if ARGS.fix_linear is True:
                        kern.W.set_trainable(False)
                        mf.set_trainable(False)

                    layers.append(l)

                    D_in = D_out

                elif c == "L":
                    d = int(d)
                    D_in += d
                    encoder_dims = [int(dim.strip()) for dim in ARGS.encoder_dims.split(",")]
                    layers.append(
                        LatentVariableLayer(
                            d, XY_dim=DX + 1, encoder_dims=encoder_dims, qz_mode=ARGS.qz_mode
                        )
                    )

        kern = RBF(D_in, lengthscales=float(D_in) ** 0.5, variance=1.0, ARD=True)
        ZZ = np.random.randn(ARGS.M, D_in)
        ZZ[:, : min(D_in, DX)] = Z[:, : min(D_in, DX)]
        layers.append(GPLayer(kern, InducingPoints(ZZ), DY))

        #################################### model
        name = "Model" if apply_name else None

        if ARGS.mode == "VI":
            model = DGP_VI(X, Y, layers, lik, minibatch_size=ARGS.minibatch_size, name=name)

        elif ARGS.mode == "IWAE":
            model = DGP_IWVI(
                X=X,
                Y=Y,
                layers=layers,
                likelihood=lik,
                minibatch_size=ARGS.minibatch_size,
                num_samples=ARGS.num_IW_samples,
                name=name,
                encoder_minibatch_size=ARGS.encoder_minibatch_size,
            )

        elif ARGS.mode == "CIWAE":
            model = DGP_CIWAE(
                X,
                Y,
                layers,
                lik,
                minibatch_size=ARGS.minibatch_size,
                num_samples=ARGS.num_IW_samples,
                name=name,
                beta=ARGS.beta,
            )

        else:
            raise ValueError(f"Unknown mode {ARGS.mode}.")

    global_step = tf.Variable(0, dtype=tf.int32)
    op_increment = tf.assign_add(global_step, 1)

    for layer in model.layers[:-1]:
        if isinstance(layer, GPLayer):
            layer.q_sqrt = layer.q_sqrt.read_value() * 1e-5

    model.compile()

    #################################### optimization

    # Whether to train the final layer with the other parameters, using Adam, or by itself, using natural
    # gradients.
    if ARGS.use_nat_grad_for_final_layer:
        # Turn off training so the parameters are not optimised by Adam. We pass them directly to the natgrad
        # optimiser, which bypasses this flag.
        model.layers[-1].q_mu.set_trainable(False)
        model.layers[-1].q_sqrt.set_trainable(False)

        gamma = tf.cast(
            tf.train.exponential_decay(
                ARGS.gamma, global_step, 1000, ARGS.gamma_decay, staircase=True
            ),
            dtype=tf.float64,
        )
        final_layer_vars = [[model.layers[-1].q_mu, model.layers[-1].q_sqrt]]
        final_layer_opt_op = NatGradOptimizer(gamma=gamma).make_optimize_tensor(
            model, var_list=final_layer_vars
        )
    else:
        final_layer_opt_op = NoOp()

    lr = tf.cast(
        tf.train.exponential_decay(
            ARGS.lr, global_step, decay_steps=1000, decay_rate=ARGS.lr_decay, staircase=True
        ),
        dtype=tf.float64,
    )

    encoder_lr = tf.cast(
        tf.train.exponential_decay(
            ARGS.encoder_lr,
            global_step,
            decay_steps=1000,
            decay_rate=ARGS.encoder_lr_decay,
            staircase=True,
        ),
        dtype=tf.float64,
    )

    dreg_optimizer = DregOptimizer(
        enable_dreg=ARGS.use_dreg,
        optimizer=ARGS.optimizer,
        encoder_optimizer=ARGS.encoder_optimizer,
        learning_rate=lr,
        encoder_learning_rate=encoder_lr,
        assert_no_nans=ARGS.assert_no_nans,
        encoder_grad_clip_value=ARGS.clip_encoder_grads,
    )
    other_layers_opt_op = dreg_optimizer.make_optimize_tensor(model)

    model.lr = lr
    model.train_op = tf.group(op_increment, final_layer_opt_op, other_layers_opt_op)
    model.init_op = lambda s: s.run(tf.variables_initializer([global_step]))
    model.global_step = global_step

    return model

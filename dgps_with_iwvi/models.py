"""
Adapted from https://github.com/hughsalimbeni/DGPs_with_IWVI/blob/master/dgps_with_iwvi/models.py
"""

from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import gpflow
import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow_core.core.framework.summary_pb2 import Summary
from tensorflow_core.python.framework.ops import Tensor

from dgps_with_iwvi.dreg_optimizer import BatchRange, DregModel
from dgps_with_iwvi.layers import Encoder, LatentVariableLayer, Layer, RegularizerType


class DGP_VI(gpflow.models.GPModel):
    def __init__(
        self,
        X: ndarray,
        Y: ndarray,
        layers: List[Layer],
        likelihood,
        num_samples=1,
        minibatch_size=None,
        name=None,
    ):
        gpflow.Parameterized.__init__(self, name=name)

        self.likelihood = likelihood

        self.num_data = X.shape[0]
        self.num_samples = num_samples
        self.minibatch_size = minibatch_size

        if minibatch_size is None:
            self.X = gpflow.params.DataHolder(X, name="DGP_VI/X/dataholder")
            self.Y = gpflow.params.DataHolder(Y, name="DGP_VI/Y/dataholder")
        else:
            self.X = gpflow.params.Minibatch(
                X, batch_size=minibatch_size, seed=0, name="DGP_VI/X/minibatch"
            )
            self.Y = gpflow.params.Minibatch(
                Y, batch_size=minibatch_size, seed=0, name="DGP_VI/Y/minibatch"
            )

        self.layers = gpflow.params.ParamList(layers)

    @gpflow.params_as_tensors
    def propagate(
        self,
        X,
        full_cov=False,
        inference_amorization_inputs=None,
        is_sampled_local_regularizer=False,
        dreg: bool = False,
    ):

        samples, means, covs, kls, kl_types = [X,], [], [], [], []

        for layer in self.layers:
            sample, mean, cov, kl = layer.propagate(
                samples[-1],
                full_cov=full_cov,
                inference_amorization_inputs=inference_amorization_inputs,
                is_sampled_local_regularizer=is_sampled_local_regularizer,
                dreg=dreg,
            )
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)

        return samples[1:], means, covs, kls, kl_types

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        X_tiled = tf.tile(self.X, [self.num_samples, 1])  # SN, Dx
        Y_tiled = tf.tile(self.Y, [self.num_samples, 1])  # SN, Dy

        XY = tf.concat([X_tiled, Y_tiled], -1)  # SN, Dx+Dy

        # Following Salimbeni 2017, the sampling is independent over N
        # The flag is_sampled_local_regularizer=False means that the KL is returned for the regularizer

        samples, means, covs, kls, kl_types = self.propagate(
            X_tiled,
            full_cov=False,
            inference_amorization_inputs=XY,
            is_sampled_local_regularizer=False,
        )

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.GLOBAL]

        var_exp = self.likelihood.variational_expectations(means[-1], covs[-1], Y_tiled)  # SN, Dy

        # Product over the columns of Y
        L_SN = tf.reduce_sum(var_exp, -1)  # SN, Dy -> SN

        shape_S_N = [self.num_samples, tf.shape(self.X)[0]]
        L_S_N = tf.reshape(L_SN, shape_S_N)

        if len(local_kls) > 0:
            local_kls_SN_D = tf.concat(local_kls, -1)  # SN, sum(W_dims)
            local_kls_SN = tf.reduce_sum(local_kls_SN_D, -1)
            local_kls_S_N = tf.reshape(local_kls_SN, shape_S_N)
            L_S_N -= local_kls_S_N  # SN

        scale = tf.cast(self.num_data, gpflow.settings.float_type) / tf.cast(
            tf.shape(self.X)[0], gpflow.settings.float_type
        )

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_mean(L_S_N, 0)

        return tf.reduce_sum(logp) * scale - tf.reduce_sum(global_kls)

    @gpflow.params_as_tensors
    def _build_predict(self, X, full_cov=False):
        _, means, covs, _, _ = self.propagate(X, full_cov=full_cov)
        return means[-1], covs[-1]

    @gpflow.params_as_tensors
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]), (gpflow.settings.int_type, ()))
    def predict_f_multisample(self, X, S):
        return self._predict_f_multisample(X, S)

    def _predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]

    @gpflow.params_as_tensors
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]), (gpflow.settings.int_type, ()))
    def predict_y_samples(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        m, v = self.likelihood.predict_mean_and_var(means[-1], covs[-1])
        z = tf.random_normal(tf.shape(means[-1]), dtype=gpflow.settings.float_type)
        return m + z * v ** 0.5

    @gpflow.params_as_tensors
    @gpflow.autoflow(
        (gpflow.settings.float_type, [None, None]),
        (gpflow.settings.float_type, [None, None]),
        (gpflow.settings.int_type, ()),
    )
    def sample_likelihood(self, X: Tensor, Y: Tensor, num_samples: int) -> Tensor:
        f_sample_means, f_sample_vars = self._predict_f_multisample(X, num_samples)
        return self.likelihood.predict_density(f_sample_means, f_sample_vars, Y)


class DGP_IWVI(DGP_VI, DregModel):
    """DGP model using importance-weighted variational inference."""

    def __init__(
        self, X: ndarray, Y: ndarray, encoder_minibatch_size: Optional[int], *args, **kwargs
    ):
        """Constructs a new instance.

        This model supports using a different batch size for the encoder and non-encoder parameters by setting
        encoder_minibatch_size. If encoder_minibatch_size is None, then the batch size will be minibatch_size for all
        parameters.
        """
        super().__init__(X, Y, *args, **kwargs)
        self._full_X = X
        self._full_Y = Y
        self._encoder_minibatch_size = encoder_minibatch_size

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return self._build_likelihood_for_batch_size(self.minibatch_size)

    def _build_likelihood_for_batch_size(
        self, batch_size: int, fixed_batch: Optional[BatchRange] = None
    ) -> Tensor:
        X, Y = self._get_batched_data(batch_size, fixed_batch)

        X_tiled = tf.tile(X[:, None, :], [1, self.num_samples, 1])  # [N x S x Dx]
        Y_tiled = tf.tile(Y[:, None, :], [1, self.num_samples, 1])  # [N x S x Dy]

        XY = tf.concat([X_tiled, Y_tiled], -1)  # [N x S x Dx+Dy]

        # While the sampling independent over N follows just as in Salimbeni 2017, in this
        # case we need to take full cov samples over the multisample dim S.
        # The flag is_sampled_local_regularizer=True means that the log p/q is returned
        # for the regularizer, rather than the KL
        samples, means, covs, kls, kl_types = self.propagate(
            X_tiled,
            full_cov=True,  # NB the full_cov is over the S dim
            inference_amorization_inputs=XY,
            is_sampled_local_regularizer=True,
            dreg=False,
        )

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.GLOBAL]

        # This could be made slightly more efficient by making the last layer full_cov=False,
        # but this seems a small price to pay for cleaner code. NB this is only a SxS matrix, not
        # an NxN matrix.
        cov_diag = tf.transpose(
            tf.matrix_diag_part(covs[-1]), [0, 2, 1]
        )  # [N x Dy x K x K -> [N x K x Dy]
        var_exp = self.likelihood.variational_expectations(
            means[-1], cov_diag, Y_tiled
        )  # [N x K x Dy]

        # Product over the columns of Y
        L_NK = tf.reduce_sum(var_exp, 2)  # [N x K x Dy] -> [N x K]

        if len(local_kls) > 0:
            local_kls_NKD = tf.concat(local_kls, -1)  # [N x K x sum(W_dims)]
            L_NK -= tf.reduce_sum(local_kls_NKD, 2)  # [N x K]

        scale = tf.cast(self.num_data, gpflow.settings.float_type) / tf.cast(
            tf.shape(X)[0], gpflow.settings.float_type
        )

        # This is reduce_mean in the VI case.
        logp = tf.reduce_logsumexp(L_NK, 1) - np.log(self.num_samples)

        return tf.reduce_sum(logp) * scale - tf.reduce_sum(global_kls)

    @lru_cache(maxsize=None)
    def _get_batched_data(
        self, batch_size: int, fixed_batch: Optional[BatchRange]
    ) -> Tuple[Tensor, Tensor]:
        if batch_size is None:
            X = gpflow.params.DataHolder(
                self._full_X, name=f"DGP_IWVI/X/dataholder/bs{batch_size}"
            )
            Y = gpflow.params.DataHolder(
                self._full_Y, name=f"DGP_IWVI/Y/dataholder/bs{batch_size}"
            )

        elif fixed_batch is not None:
            # In this case we want to return the same batch every time, so use a simple DataHolder containing a subset
            # of the data.
            start, end = fixed_batch
            assert (
                end - start == batch_size
            ), "Length of batch range must be equal to the batch size."
            X = gpflow.params.DataHolder(
                self._full_X[start:end], name=f"DGP_IWVI/X/dataholder/s{start}e{end}"
            )
            Y = gpflow.params.DataHolder(
                self._full_Y[start:end], name=f"DGP_IWVI/Y/dataholder/s{start}e{end}"
            )

        else:
            X = gpflow.params.Minibatch(
                self._full_X,
                batch_size=batch_size,
                seed=0,
                name=f"DGP_IWVI/X/minibatch/bs{batch_size}",
            )
            Y = gpflow.params.Minibatch(
                self._full_Y,
                batch_size=batch_size,
                seed=0,
                name=f"DGP_IWVI/Y/minibatch/bs{batch_size}",
            )

        return X.parameter_tensor, Y.parameter_tensor

    @lru_cache(maxsize=1)
    def get_reg_objective_for_encoder_params(
        self, fixed_batch: Optional[BatchRange] = None
    ) -> Tensor:
        # The superclass's objective function includes the priors over parameters, but as there aren't any in our case
        # we can ignore these and just return the negative of the likelihood.
        if self._encoder_minibatch_size is None:
            return tf.negative(
                self._build_likelihood_for_batch_size(self.minibatch_size, fixed_batch)
            )
        else:
            return tf.negative(
                self._build_likelihood_for_batch_size(self._encoder_minibatch_size, fixed_batch)
            )

    @lru_cache(maxsize=1)
    def get_reg_objective_for_other_params(self) -> Tensor:
        # The superclass's objective function includes the priors over parameters, but as there aren't any in our case
        # we can ignore these and just return the negative of the likelihood.
        return tf.negative(self._build_likelihood_for_batch_size(self.minibatch_size))

    @lru_cache(maxsize=1)
    def get_dreg_objective_for_encoder_params(
        self, fixed_batch: Optional[BatchRange] = None
    ) -> Tensor:
        if self._encoder_minibatch_size is None:
            return self._build_dreg_objective_for_batch_size(self.minibatch_size, fixed_batch)
        else:
            return self._build_dreg_objective_for_batch_size(
                self._encoder_minibatch_size, fixed_batch
            )

    @gpflow.params_as_tensors
    def _build_dreg_objective_for_batch_size(
        self, batch_size: int, fixed_batch: Optional[BatchRange]
    ) -> Tensor:
        """When differentiated, this results in the DReG gradient estimator.

        This gradient estimator is given in
        "On Signal-to-Noise Ratio Issues in Variational Inference for Deep Gaussian Processes" equation 9.

        To build an objective which differentiates to this estimator we apply tf.stop_gradient to various terms.
        This follows https://github.com/google-research/google-research/tree/master/dreg_estimators
        """
        self._assert_lv_first_layer_only()

        X, Y = self._get_batched_data(batch_size, fixed_batch)
        Y_tiled = tf.tile(Y[:, None, :], [1, self.num_samples, 1])  # [N x S x Dy]
        X_tiled = tf.tile(X[:, None, :], [1, self.num_samples, 1])  # [N x S x Dx]

        XY = tf.concat([X_tiled, Y_tiled], -1)  # [N x S x Dx+Dy]

        # While the sampling independent over N follows just as in Salimbeni 2017, in this
        # case we need to take full cov samples over the multisample dim S.
        # The flag is_sampled_local_regularizer=True means that the log p/q is returned
        # for the regularizer, rather than the KL
        samples, means, covs, kls, kl_types = self.propagate(
            X_tiled,  # NB the full_cov is over the S dim
            full_cov=True,
            inference_amorization_inputs=XY,
            is_sampled_local_regularizer=True,
            dreg=True,
        )

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.LOCAL]

        # This could be made slightly more efficient by making the last layer full_cov=False,
        # but this seems a small price to pay for cleaner code. NB this is only a SxS matrix, not
        # an NxN matrix.
        cov_diag = tf.transpose(tf.matrix_diag_part(covs[-1]), [0, 2, 1])  # N,Dy,K,K -> N,K,Dy
        var_exp = self.likelihood.variational_expectations(
            means[-1], cov_diag, Y_tiled
        )  # N, K, Dy

        # Product over the columns of Y.
        # In the paper, this is L_NK.
        likelihood_term = tf.reduce_sum(var_exp, 2)  # [N x K x Dy] -> [N x K]

        logw_NK = likelihood_term  # N, K
        if len(local_kls) > 0:
            local_kls_NKD = tf.concat(local_kls, -1)  # [N x K x sum(W_dims)]
            logw_NK -= tf.reduce_sum(local_kls_NKD, 2)  # [N x K]

        scale = tf.cast(self.num_data, gpflow.settings.float_type) / tf.cast(
            tf.shape(X)[0], gpflow.settings.float_type
        )

        # Sum over wk, wj using softmax, as we have log_wk.
        w_sum_sq_NK = tf.stop_gradient(tf.square(tf.nn.softmax(logw_NK, axis=1)))  # [N x K]

        likelihood = scale * tf.reduce_sum(w_sum_sq_NK * logw_NK)
        return tf.negative(likelihood)

    @property
    def encoder_params(self) -> Iterable[tf.Variable]:
        self._assert_lv_first_layer_only()
        return [var for var in self.trainable_tensors if "encoder" in var.name]

    def _assert_lv_first_layer_only(self):
        for i, layer in enumerate(self.layers):
            if i != 0 and isinstance(layer, LatentVariableLayer):
                raise ValueError("When using DReG, currently only support LVs in the first layer.")

    def get_tf_summaries(self) -> List[Summary]:
        summaries = []

        summaries.append(
            tf.summary.scalar(
                "optimisation/dreg_objective", self.get_dreg_objective_for_encoder_params()
            )
        )
        summaries.extend(
            self._get_grad_summaries("gradients/", self.get_reg_objective_for_encoder_params())
        )
        summaries.extend(
            self._get_grad_summaries(
                "dreg_gradients/", self.get_dreg_objective_for_encoder_params()
            )
        )
        for layer in self.layers:
            summaries.extend(layer.get_tf_summaries())

        return summaries

    def _get_grad_summaries(self, base_path: str, objective: Tensor) -> List[Summary]:
        param_names, param_values, indices = zip(
            *Encoder.get_params_to_log(self.trainable_parameters)
        )
        gradients = tf.gradients(objective, param_values)
        return [
            tf.summary.scalar(base_path + param_name, gradient[index])
            for param_name, gradient, index in zip(param_names, gradients, indices)
        ]


class DGP_CIWAE(DGP_VI):
    def __init__(
        self, X, Y, layers, likelihood, num_samples=1, minibatch_size=None, name=None, beta=0
    ):
        gpflow.Parameterized.__init__(self, name=name)

        self.likelihood = likelihood
        self.beta = beta
        self.num_data = X.shape[0]
        self.num_samples = num_samples

        if minibatch_size is None:
            self.X = gpflow.params.DataHolder(X)
            self.Y = gpflow.params.DataHolder(Y)
        else:
            self.X = gpflow.params.Minibatch(X, batch_size=minibatch_size, seed=0)
            self.Y = gpflow.params.Minibatch(Y, batch_size=minibatch_size, seed=0)

        self.layers = gpflow.params.ParamList(layers)

    @gpflow.params_as_tensors
    def VI_lik_build(self):
        X_tiled = tf.tile(self.X, [self.num_samples, 1])  # SN, Dx
        Y_tiled = tf.tile(self.Y, [self.num_samples, 1])  # SN, Dy

        XY = tf.concat([X_tiled, Y_tiled], -1)  # SN, Dx+Dy

        # Following Salimbeni 2017, the sampling is independent over N
        # The flag is_sampled_local_regularizer=False means that the KL is returned for the regularizer

        samples, means, covs, kls, kl_types = self.propagate(
            X_tiled,
            full_cov=False,
            inference_amorization_inputs=XY,
            is_sampled_local_regularizer=False,
        )

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.GLOBAL]

        var_exp = self.likelihood.variational_expectations(means[-1], covs[-1], Y_tiled)  # SN, Dy

        # Product over the columns of Y
        L_SN = tf.reduce_sum(var_exp, -1)  # SN, Dy -> SN

        shape_S_N = [self.num_samples, tf.shape(self.X)[0]]
        L_S_N = tf.reshape(L_SN, shape_S_N)

        if len(local_kls) > 0:
            local_kls_SN_D = tf.concat(local_kls, -1)  # SN, sum(W_dims)
            local_kls_SN = tf.reduce_sum(local_kls_SN_D, -1)
            local_kls_S_N = tf.reshape(local_kls_SN, shape_S_N)
            L_S_N -= local_kls_S_N  # SN

        scale = tf.cast(self.num_data, gpflow.settings.float_type) / tf.cast(
            tf.shape(self.X)[0], gpflow.settings.float_type
        )

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_mean(L_S_N, 0)

        return tf.reduce_sum(logp) * scale - tf.reduce_sum(global_kls)

    @gpflow.params_as_tensors
    def IWVI_lik_build(self):
        X_tiled = tf.tile(self.X[:, None, :], [1, self.num_samples, 1])  # N, S, Dx
        Y_tiled = tf.tile(self.Y[:, None, :], [1, self.num_samples, 1])  # N, S, Dy

        XY = tf.concat([X_tiled, Y_tiled], -1)  # N, S, Dx+Dy

        # While the sampling independent over N follows just as in Salimbeni 2017, in this
        # case we need to take full cov samples over the multisample dim S.
        # The flag is_sampled_local_regularizer=True means that the log p/q is returned
        # for the regularizer, rather than the KL
        samples, means, covs, kls, kl_types = self.propagate(
            X_tiled,
            full_cov=True,  # NB the full_cov is over the S dim
            inference_amorization_inputs=XY,
            is_sampled_local_regularizer=True,
        )

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is RegularizerType.GLOBAL]

        # This could be made slightly more efficient by making the last layer full_cov=False,
        # but this seems a small price to pay for cleaner code. NB this is only a SxS matrix, not
        # an NxN matrix.
        cov_diag = tf.transpose(tf.matrix_diag_part(covs[-1]), [0, 2, 1])  # N,Dy,K,K -> N,K,Dy
        var_exp = self.likelihood.variational_expectations(
            means[-1], cov_diag, Y_tiled
        )  # N, K, Dy

        # Product over the columns of Y
        L_NK = tf.reduce_sum(var_exp, 2)  # N, K, Dy -> N, K

        if len(local_kls) > 0:
            local_kls_NKD = tf.concat(local_kls, -1)  # N, K, sum(W_dims)
            L_NK -= tf.reduce_sum(local_kls_NKD, 2)  # N, K

        scale = tf.cast(self.num_data, gpflow.settings.float_type) / tf.cast(
            tf.shape(self.X)[0], gpflow.settings.float_type
        )

        # This is reduce_mean in the VI case.
        logp = tf.reduce_logsumexp(L_NK, 1) - np.log(self.num_samples)

        return tf.reduce_sum(logp) * scale - tf.reduce_sum(global_kls)

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return self.beta * self.VI_lik_build() + (1 - self.beta) * self.IWVI_lik_build()

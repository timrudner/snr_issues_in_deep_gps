"""
Adapted from https://github.com/hughsalimbeni/DGPs_with_IWVI/blob/master/dgps_with_iwvi/layers.py.
"""

import enum
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.params import Parameter
from tensorflow_core.core.framework.summary_pb2 import Summary
from tensorflow_core.python.framework.ops import Tensor

from dgps_with_iwvi.temp_workaround import multisample_sample_conditional, gauss_kl


class RegularizerType(enum.Enum):
    LOCAL = 0
    GLOBAL = 1


class Layer(ABC):
    @property
    @abstractmethod
    def regularizer_type(self) -> RegularizerType:
        pass

    @abstractmethod
    def get_tf_summaries(self) -> List[Summary]:
        pass


class GPLayer(gpflow.Parameterized, Layer):
    regularizer_type = RegularizerType.GLOBAL

    def __init__(self, kern, Z, num_outputs, mean_function=None, name=None, layer_num=None):
        gpflow.Parameterized.__init__(self, name=name)

        self.num_inducing = len(Z)

        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = gpflow.params.Parameter(q_mu)

        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        transform = gpflow.transforms.LowerTriangular(self.num_inducing, num_matrices=num_outputs)
        self.q_sqrt = gpflow.params.Parameter(q_sqrt, transform=transform)

        self.feature = (
            Z
            if isinstance(Z, gpflow.features.InducingFeature)
            else gpflow.features.InducingPoints(Z)
        )
        self.kern = kern
        self.mean_function = mean_function or gpflow.mean_functions.Zero()

        self.num_outputs = num_outputs

        self._layer_num = layer_num

    @gpflow.params_as_tensors
    def propagate(self, F, full_cov=False, **kwargs):
        samples, mean, cov = multisample_sample_conditional(
            F,
            self.feature,
            self.kern,
            self.q_mu,
            full_cov=full_cov,
            q_sqrt=self.q_sqrt,
            white=True,
        )

        kl = gauss_kl(self.q_mu, self.q_sqrt)

        mf = self.mean_function(F)
        samples += mf
        mean += mf

        return samples, mean, cov, kl

    def get_tf_summaries(self) -> List[Summary]:
        if self._layer_num is None:
            return []

        summaries = []
        lengthscales = next(
            param for param in self.kern.parameters if "lengthscales" in param.pathname
        )
        for i in range(lengthscales.shape[0]):
            summaries.append(
                tf.summary.scalar(
                    f"lengthscales/layer{self._layer_num}/{i}", lengthscales.parameter_tensor[i]
                )
            )
        return summaries


class LatentVariableLayer(gpflow.Parameterized, Layer):
    regularizer_type = RegularizerType.LOCAL

    def __init__(
        self,
        latent_dim,
        XY_dim=None,
        encoder=None,
        name=None,
        encoder_dims=[20, 20],
        qz_mode: str = "encoder",
    ):
        """Creates a new instance.

        :param qz_mode: one of 'encoder' or 'fix_mu_sigma'. If 'encoder', the variational distribution over the latent
                        variables, q(z), is paramterized by the encoder and learned as usual. If 'fix_mu_sigma', then
                        the distribution is fixed to the prior.
        """

        gpflow.Parameterized.__init__(self, name=name)

        self.latent_dim = latent_dim
        self._qz_mode = qz_mode

        # placeholders with default, where the default is the prior
        ones = tf.ones([1, 1], dtype=gpflow.settings.float_type)
        zeros = tf.zeros([1, 1], dtype=gpflow.settings.float_type)
        self.q_mu_placeholder = tf.placeholder_with_default(zeros, [None, None])
        self.q_sqrt_placeholder = tf.placeholder_with_default(ones, [None, None])

        if encoder is None:
            assert XY_dim, "must pass XY_dim or else an encoder"
            encoder = Encoder(latent_dim, XY_dim, encoder_dims)
        self.encoder = encoder

        self._kl_for_tb = None
        self._q_mu_for_tb = None
        self._q_sqrt_for_tb = None

    @gpflow.params_as_tensors
    def propagate(
        self,
        F,
        inference_amorization_inputs=None,
        is_sampled_local_regularizer=False,
        dreg: bool = False,
        **kwargs,
    ):
        if inference_amorization_inputs is None:
            """
            If there isn't an X and Y passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            shape = tf.concat([tf.shape(F)[:-1], [self.latent_dim]], 0)
            ones = tf.ones(shape, dtype=gpflow.settings.float_type)
            q_mu = self.q_mu_placeholder * ones  # TODO tf.broadcast_to
            q_sqrt = self.q_sqrt_placeholder * ones  # TODO tf.broadcast_to
        else:
            q_mu, q_sqrt = self.encoder(inference_amorization_inputs)

            if self._qz_mode == "encoder":
                pass
            elif self._qz_mode == "fix_mu_sigma":
                # Fix q(z) to the prior.
                # We include 0 * the original values so that q_mu and q_sqrt have a gradient wrt the encoder parameters.
                # If there is no connection in the graph between q_mu/q_sqrt and the encoder parameters, the gradient is
                # None and the optimiser throws an error.
                q_mu = tf.zeros_like(q_mu) + 0 * q_mu
                q_sqrt = tf.ones_like(q_sqrt) + 0 * q_sqrt
            else:
                raise ValueError(f"Unknown qz_mode {self._qz_mode}")

            self._q_mu_for_tb = q_mu
            self._q_sqrt_for_tb = q_sqrt

        # reparameterization trick to take a sample for W
        z = tf.random_normal(tf.shape(q_mu), dtype=gpflow.settings.float_type)
        W = q_mu + z * q_sqrt

        samples = tf.concat([F, W], -1)
        mean = tf.concat([F, q_mu], -1)
        cov = tf.concat([tf.zeros_like(F), q_sqrt ** 2], -1)

        # the prior regularization
        zero, one = [tf.cast(x, dtype=gpflow.settings.float_type) for x in [0, 1]]
        p = tf.contrib.distributions.Normal(zero, one)

        # If dreg is enabled we block a gradient here so that when the objective is differentiated we get the DReG
        # estimator. See DGP_IWVI._build_dreg_objective_for_batch_size().
        if dreg:
            q = tf.contrib.distributions.Normal(tf.stop_gradient(q_mu), tf.stop_gradient(q_sqrt))
        else:
            q = tf.contrib.distributions.Normal(q_mu, q_sqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(W) - p.log_prob(W)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tf.contrib.distributions.kl_divergence(q, p)
        self._kl_for_tb = tf.reduce_mean(kl)

        return samples, mean, cov, kl

    def get_tf_summaries(self) -> List[Summary]:
        summaries = []

        for W, b in zip(self.encoder.Ws, self.encoder.bs):
            summaries.append(tf.summary.histogram(W.pathname, W.parameter_tensor))
            summaries.append(tf.summary.histogram(b.pathname, b.parameter_tensor))

        # Pick some scalar parameters, This is easier to compare between runs than the histograms.
        for param_name, param_value, index in Encoder.get_params_to_log(self.parameters):
            summaries.append(tf.summary.scalar(param_name, param_value[index]))

        # The shape is (batch, importance samples, latent var dim)
        assert (
            self._q_mu_for_tb.shape[2] == 1
        ), "Logging only support a single latent variable dimension"
        assert (
            self._q_sqrt_for_tb.shape[2] == 1
        ), "Logging only support a single latent variable dimension"
        # q_mu and q_sqrt are equal across the importance sample dimension, so just look at the first value.
        summaries.append(tf.summary.histogram("encoder/q_mu", self._q_mu_for_tb[:, 0, 0]))
        summaries.append(tf.summary.histogram("encoder/q_sqrt", self._q_sqrt_for_tb[:, 0, 0]))

        summaries.append(tf.summary.scalar("encoder/qz_pz_kl", self._kl_for_tb))

        return summaries


class Encoder(gpflow.Parameterized):
    def __init__(self, latent_dim, input_dim, network_dims, activation_func=None, name=None):
        """
        Encoder that uses GPflow params to encode the features.
        Creates an MLP with input dimensions `input_dim` and produces
        2 * `latent_dim` outputs.
        :param latent_dim: dimension of the latent variable
        :param input_dim: the MLP acts on data of `input_dim` dimensions
        :param network_dims: dimensions of inner MLPs, e.g. [10, 20, 10]
        :param activation_func: TensorFlow operation that can be used
            as non-linearity between the layers (default: tanh).
        """
        gpflow.Parameterized.__init__(self, name=name)
        self.latent_dim = latent_dim
        self.activation_func = activation_func or tf.nn.tanh

        self.layer_dims = [input_dim, *network_dims, latent_dim * 2]

        Ws, bs = [], []

        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            xavier_std = (2.0 / (input_dim + output_dim)) ** 0.5
            W = np.random.randn(input_dim, output_dim) * xavier_std
            Ws.append(gpflow.Param(W))
            bs.append(gpflow.Param(np.zeros(output_dim)))

        self.Ws, self.bs = gpflow.params.ParamList(Ws), gpflow.params.ParamList(bs)

    @gpflow.params_as_tensors
    def __call__(self, Z):
        o = tf.ones_like(Z)[..., :1, :1]  # for correct broadcasting
        for i, (W, b, dim_in, dim_out) in enumerate(
            zip(self.Ws, self.bs, self.layer_dims[:-1], self.layer_dims[1:])
        ):
            Z0 = tf.identity(Z)
            Z = tf.matmul(Z, o * W) + o * b

            if i < len(self.bs) - 1:
                Z = self.activation_func(Z)

            if dim_out == dim_in:  # skip connection
                Z += Z0

        means, log_chol_diag = tf.split(Z, 2, axis=-1)
        q_sqrt = tf.nn.softplus(log_chol_diag - 3.0)  # bias it towards small vals at first
        q_mu = means
        return q_mu, q_sqrt

    @staticmethod
    def get_params_to_log(all_params: List[Parameter]) -> List[Tuple[str, Tensor, Tuple]]:
        """Returns a list of pairs of parameter names and scalar values to log to tensorboard.

        Rather than returning the tensor sliced to a scalar value, the entire tensor is returned along with an index
        which the caller should slice at. This is to allow taking the gradient of a value wrt the full parameter
        tensor.

        :returns: [(name, tensor, (index))], where (index) is a tuple giving the index to log in tensor.
        """
        params_to_log = []
        for param in all_params:
            if "encoder" in param.pathname and "W" in param.pathname:
                i1 = 0
                i2 = 0
                name = f"{param.pathname}_{i1}_{i2}"
                params_to_log.append((name, param.parameter_tensor, (i1, i2)))
            elif "encoder" in param.pathname and "b" in param.pathname:
                i1 = 0
                name = f"{param.pathname}_{i1}"
                params_to_log.append((name, param.parameter_tensor, (i1,)))
        return params_to_log

"""
Optimizer for models which support a doubly reparameterized gradient (DReG) estimator for some parameters.

DReG is introduced in 'Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives for Monte Carlo Objectics';
Tucker, Lawson, Gu, Maddison.

Models should implement the DregModel abstract class, which allows the model to provide seperate objectives for
different sets of parameters.
"""
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple

import tensorflow as tf
from gpflow import misc
from gpflow.training.optimizer import Optimizer
from tensorflow_core.core.framework.summary_pb2 import Summary
from tensorflow_core.python.framework.ops import Tensor
from tensorflow_core.python.training.adam import AdamOptimizer
from tensorflow_core.python.training.gradient_descent import GradientDescentOptimizer

BatchRange = Tuple[int, int]


class DregModel(ABC):
    """Interface implemented by models which support DReG optimisation of the parameters of the encoder network.

    This interface defines that the paremeters of the model are split into two groups: those of the encoder, and those
    of the rest of the model. For the encoder parameters, the model implements both a standard objective function, and
    an objective function which, when differentiated, results in the DReG estimator.
    """

    @abstractmethod
    def get_reg_objective_for_encoder_params(
        self, fixed_batch: Optional[BatchRange] = None
    ) -> Tensor:
        """Objective to use for the encoder parameters when DReG is turned off.

        :param fixed_batch: If not None, compute using the same batch of data every time this Tensor is evaluated, where
                            the batch is the given range in the dataset. This is useful when statically sampling
                            gradients. If None, use a different batch every evaluation like normal.
        """
        pass

    @abstractmethod
    def get_reg_objective_for_other_params(self) -> Tensor:
        """Objective to use for the non-encoder parameters.

        As we do not apply DReG to the non-encoder parameters, this should be used whether DReG is on or off.
        """
        pass

    @abstractmethod
    def get_dreg_objective_for_encoder_params(
        self, fixed_batch: Optional[BatchRange] = None
    ) -> Tensor:
        """Objective to use for the encoder parameters when DReG is enabled.

        When differentiated this objective yields the doubly reparameterized gradient estimator.

        :param fixed_batch: If not None, compute using the same batch of data every time this Tensor is evaluated, where
                            the batch is the given range in the dataset. This is useful when statically sampling
                            gradients. If None, use a different batch every evaluation like normal.
        """
        pass

    @property
    @abstractmethod
    def encoder_params(self) -> Iterable[tf.Variable]:
        """
        Parameter variables which the optimizer should update using the dreg_objective.

        The optimizer will update all other parameters using get_reg_objective_for_other_params().
        """
        pass

    def get_tf_summaries(self) -> List[Summary]:
        """Returns a list of TensorBoard summaries for this model."""
        return []


class DregOptimizer(Optimizer):
    """
    Optimizer which uses the doubly reparameterized gradient estimator for the encoder parameters (if enabled).

    This can only optimize models implementing DregModel, which specifies the DReG objective and identifies the encoder
    parameters.

    Based on gpflow.training.tensorflow_optimizer.
    """

    def __init__(
        self,
        optimizer: str,
        encoder_optimizer: str,
        learning_rate: float = 0.001,
        encoder_learning_rate: float = 0.001,
        enable_dreg: bool = True,
        assert_no_nans: bool = False,
        encoder_grad_clip_value: Optional[float] = None,
    ):
        """Constructs a new instance.

        :param optimizer: name of the optimizer to use for the non-encoder parameters, one of 'adam' or 'sgd'
        :param encoder_optimizer: name of the optimizer to use for the non-encoder parameters, one of 'adam' or 'sgd'
        :param assert_no_nans: if True then will assert that no parameters are NaN or inf after each gradient step
        :param encoder_grad_clip_value: if not None, then gradients of the encoder parameters larger in magnitude than
                                        the given value will be clipped to that value. If None, no clipping will be
                                        performed.
        """
        super().__init__()

        self._normal_optimizer = self._build_optimizer(optimizer, learning_rate)
        self._encoder_optimizer = self._build_optimizer(encoder_optimizer, encoder_learning_rate)
        self._enable_dreg = enable_dreg
        self._assert_no_nans = assert_no_nans
        self._encoder_grad_clip_value = encoder_grad_clip_value

    @staticmethod
    def _build_optimizer(optimizer: str, learning_rate: float) -> Optimizer:
        if optimizer == "adam":
            return AdamOptimizer(learning_rate)
        elif optimizer == "sgd":
            return GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer {optimizer}")

    def make_optimize_tensor(self, model, session=None, var_list=None, **kwargs):
        """
        Make Tensorflow optimization tensor.
        This method builds optimization tensor and initializes all necessary variables
        created by optimizer.

            :param model: GPflow model.
            :param session: Tensorflow session.
            :param var_list: List of variables for training.
            :param kwargs: Dictionary of extra parameters passed to Tensorflow
                optimizer's minimize method.
            :return: Tensorflow optimization tensor or operation.
        """
        assert isinstance(model, DregModel), "Model must be instance of DregModel."

        if var_list is not None:
            raise NotImplementedError("DregOptimizer does not support an additional var list.")

        session = model.enquire_session(session)

        encoder_vars, normal_vars = self._get_split_var_list(model)

        with session.as_default():
            other_param_objective, encoder_param_objective = self._get_objectives(model)

            encoder_grads_and_vars = self._encoder_optimizer.compute_gradients(
                encoder_param_objective, var_list=encoder_vars, **kwargs
            )
            encoder_optimize_step = self._encoder_optimizer.apply_gradients(
                self._clip_encoder_grads(encoder_grads_and_vars)
            )

            other_optimizer_step = self._normal_optimizer.minimize(
                other_param_objective, var_list=normal_vars, **kwargs
            )
            overall_optimize_step = tf.group(encoder_optimize_step, other_optimizer_step)

            if self._assert_no_nans:
                asserts = [
                    tf.debugging.assert_all_finite(var, message=f"{var.name} was not finite")
                    for var in encoder_vars + normal_vars
                ]
                overall_optimize_step = tf.group(overall_optimize_step, *asserts)

            model.initialize(session=session)
            self._initialize_optimizers(session)

            return overall_optimize_step

    def _clip_encoder_grads(self, grads_and_vars: List[Tuple[Tensor, tf.Variable]]):
        if self._encoder_grad_clip_value is None:
            return grads_and_vars
        else:
            return [
                (
                    tf.clip_by_value(
                        grad, -self._encoder_grad_clip_value, self._encoder_grad_clip_value
                    ),
                    var,
                )
                for grad, var in grads_and_vars
            ]

    def _get_objectives(self, model: DregModel) -> Tuple[Tensor, Tensor]:
        other_param_objective = model.get_reg_objective_for_other_params()
        if self._enable_dreg:
            encoder_param_objective = model.get_dreg_objective_for_encoder_params()
        else:
            encoder_param_objective = model.get_reg_objective_for_encoder_params()
        return other_param_objective, encoder_param_objective

    def _get_split_var_list(self, model: DregModel) -> Tuple[List[Tensor], List[Tensor]]:
        """Returns a pair ([dreg variables], [normal variables])."""
        # TODO: Simplify this function, don't need loop.
        encoder_vars = []
        normal_vars = []
        encoder_vars_unordered = model.encoder_params
        for var in self._gen_var_list(model, var_list=None):
            if var in encoder_vars_unordered:
                encoder_vars.append(var)
            else:
                normal_vars.append(var)
        return encoder_vars, normal_vars

    def _initialize_optimizers(self, session: tf.Session):
        misc.initialize_variables(
            self._encoder_optimizer.variables(), session=session, force=False
        )
        misc.initialize_variables(self._normal_optimizer.variables(), session=session, force=False)

    def minimize(
        self,
        model,
        session=None,
        var_list=None,
        feed_dict=None,
        maxiter=1000,
        initialize=True,
        anchor=True,
        step_callback=None,
        **kwargs,
    ):
        raise NotImplementedError

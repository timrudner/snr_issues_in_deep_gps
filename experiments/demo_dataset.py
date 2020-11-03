"""Contains the demo dataset."""
from typing import Tuple

import numpy as np
from numpy import ndarray
from dgps_with_iwvi.models import DGP_VI
from gpflow.training.monitor import LogdirWriter
import matplotlib.pyplot as plt
import tensorflow as tf
import io

Ns = 200
Xs = np.linspace(-4, 4, Ns).reshape(-1, 1)


class Data:
    def __init__(self, X_train: ndarray, Y_train: ndarray, X_test: ndarray, Y_test: ndarray):
        X_dim = X_train.shape[1]
        Y_dim = Y_train.shape[1]
        N_train = X_train.shape[0]
        N_test = X_test.shape[0]
        assert X_train.shape == (N_train, X_dim)
        assert Y_train.shape == (N_train, Y_dim)
        assert X_test.shape == (N_test, X_dim)
        assert Y_test.shape == (N_test, Y_dim)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test


def f(X):
    f1 = lambda x: np.sin(x * 4) / 3 + 0.2 * np.exp(np.random.randn(*x.shape))
    f2 = lambda x: -np.power(x * 3, 2) / 30 + 1.5 + 0.2 * np.exp(np.random.randn(*x.shape))

    ind = np.random.choice([True, False], size=X.shape, p=(0.6, 0.4))
    Y = np.empty(X.shape)
    Y[ind] = f1(X[ind])
    Y[np.invert(ind)] = f2(X[np.invert(ind)])
    return Y


def create_demo_data() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    # The data is [test data][train data] ... [test data][train data]
    # Each [test data][train data] pair is a "block" spanning 1 in input space.
    # As training and test data has equal density, the relative size of the blocks is determined by the ratio of the
    # number of points.
    num_train_points = 2000
    num_test_points = 200
    train_width = float(num_train_points) / float(num_train_points + num_test_points)
    test_width = 1 - train_width
    num_blocks = 6

    train_xs = []
    test_xs = []
    for block in range(num_blocks):
        start = -3 + block
        test_xs.append(
            np.random.uniform(
                low=start, high=start + test_width, size=(int(num_test_points / num_blocks), 1)
            )
        )
        train_xs.append(
            np.random.uniform(
                low=start + test_width,
                high=start + 1,
                size=(int(num_train_points / num_blocks), 1),
            )
        )

    train_X = np.concatenate(train_xs, axis=0)
    test_X = np.concatenate(test_xs, axis=0)
    train_Y = f(train_X)
    test_Y = f(test_X)
    return train_X, train_Y, test_X, test_Y


def plot_data_and_predictions(
    data: Data, model: DGP_VI, tensorboard_writer: LogdirWriter, sess, step: int
):
    xs = np.arange(-2.0, 2.0, 0.1).reshape((-1, 1))
    num_samples = 20
    samples = model.predict_y_samples(xs, num_samples, session=sess)

    plt.scatter(data.X_train, data.Y_train, marker=".", label="train")
    plt.scatter(data.X_test, data.Y_test, marker=".", label="test")

    plt.scatter(
        np.tile(xs, (num_samples, 1, 1)).reshape(-1, 1),
        samples.reshape((-1, 1)),
        marker="o",
        alpha=0.5,
    )

    plt.ylim(-1.5, 2.5)
    plt.legend()

    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format="png")
    plt.close()
    plot_buf.seek(0)

    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Graph().as_default() as graph:
        with tf.Session(config=config, graph=graph).as_default() as sess:
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            summary_op = tf.summary.image("demo/plot", image)
            tensorboard_writer.add_summary(sess.run(summary_op), step)

from typing import Tuple

import numpy as np
from dgps_with_iwvi.models import DGP_VI
from numpy import ndarray
from scipy.special import logsumexp
from scipy.stats import shapiro


def compute_metrics(
    model: DGP_VI, X: ndarray, Y: ndarray, num_samples: int
) -> Tuple[float, float, float]:
    """Computes log likelihood, RMSE and median Shapiro-Wilk value for the given data.

    :param num_samples: number of times to sample the model posterior at each test point
    :returns: (log lik, rmse, median Shapiro-Wilk)
    """
    assert len(X) == len(Y)
    rmse = np.empty(len(X))
    shapiro_W = np.empty(len(X))

    for i, (x, y) in enumerate(zip(X, Y)):
        samples = model.predict_y_samples(x.reshape(1, -1), num_samples)
        Ss = samples[:, :, 0]

        shapiro_W[i] = float(shapiro((Ss - np.average(Ss)) / np.std(Ss))[0])
        rmse[i] = (np.average(Ss) - float(y)) ** 2

    logp = compute_log_likelihood(model, X, Y, n_samples=1000)
    return logp, (np.average(rmse) ** 0.5).item(), np.median(shapiro_W).item()


def compute_log_likelihood(model: DGP_VI, X: ndarray, Y: ndarray, n_samples: int) -> float:
    """Computes the log likelihood of the given data using Monte Carlo sampling."""
    batch_size = 128
    logps_per_batch = []
    for batch_i in range(0, len(X), batch_size):
        xs = X[batch_i : batch_i + batch_size]
        ys = Y[batch_i : batch_i + batch_size]
        # Shape [num samples x batch size x output dimension]
        logp_samples = model.sample_likelihood(xs, ys, n_samples)
        assert logp_samples.shape == (n_samples, xs.shape[0], ys.shape[1])
        # Compute AND probability over the outputs.
        logp_samples = logp_samples.sum(2)
        assert logp_samples.shape == (n_samples, xs.shape[0])
        # Compute the mean in probability space over the samples
        batch_logps = -np.log(n_samples) + logsumexp(logp_samples, axis=0)
        assert batch_logps.shape == (xs.shape[0],)
        logps_per_batch.append(batch_logps)

    return np.concatenate(logps_per_batch).mean().item()

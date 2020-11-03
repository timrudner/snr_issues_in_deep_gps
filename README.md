# On Signal-to-Noise Ratio Issues in Variational Inference for Deep Gaussian Processes

This repository contains the code for

**_On Signal-to-Noise Ratio Issues in Variational Inference for Deep Gaussian Processes_**; Tim G. J. Rudner*, Oscar Key*, Yarin Gal, Tom Rainforth.

**Abstract:** We show that the gradient estimates used in training Deep Gaussian Processes (DGPs) with importance-weighted variational inference are susceptible to signal-to-noise ratio (SNR) issues. Specifically, we show both theoretically and empirically that the SNR of the gradient estimates for the latent variable's variational parameters decreases as the number of importance samples increases. As a result, these gradient estimates degrade to pure noise if the number of importance samples is too large. To address this pathology, we show how doubly-reparameterized gradient estimators, originally proposed for training variational autoencoders, can be adapted to the DGP setting and that the resultant estimators completely remedy the SNR issue, thereby providing more reliable training. Finally, we demonstrate that our fix can lead to improvements in the predictive performance of the model's predictive posterior.

<p align="center">
  <a href="https://arxiv.org/abs/2011.00515">View on arXiv</a>
</p>

In particular, the code implements:
- Tools for sampling and analyzing the gradient estimates of the variational distribution over the latent variable
- A doubly reparameterized gradient (DReG) estimator for importance weighted variational inference in latent-variable deep Gaussian process models

## License
This code is a derivative of the [code for "Deep Gaussian Processes with Importance-Weighted Variational Inference"](https://github.com/hughsalimbeni/DGPs_with_IWVI); Hugh Salimbeni, Vincent Dutordoir, James Hensman, Marc Peter Deisenroth. We have clearly marked the top of each file taken from the original code, and indicated if it has been modified.

The original Apache 2.0 license is included in LICENSE, and we release these modifications under the same license.

## Environment setup
If you have Conda installed, you can set up an appropriate environment by running:
```
conda env update -f environment.yaml
conda activate dgp-snr
```

Datasets named with the prefix 'wilson_' are not downloaded automatically. Please run the script `experiments/install_wilson_datasets.py`, which provides instructions for how to download the datasets and will install the downloaded files.

To format the code: `black --line-length 99`

## Reproducing the results from the paper
There are several main entry points:
- `experiments/train_model.py` Trains a DGP model, saves checkpoints, and logs statistics (ELBO, log-likelihood, etc.).
- `experiments/sample_gradients.py` Loads a model checkpoint, samples, and saves gradient estimates. This is required to plot gradient histograms or to compute the SNR.
- `experiments/compute_tlls_and_elbos.py` Loads a model checkpoint, computes the test log-likelihood or train-ELBO, and saves the results.
- `experiments/create_figures.py` Reproduces figures from the paper.

To reproduce a figure from the paper, run `experiments/create_figures.py [figure_x]`.
Depending on the figure to reproduce, `create_figures.py` may require checkpoints or data files created by `train_model.py`, `sample_gradients.py`, or `compute_tlls_and_elbos.py`.
For any given figure, `create_figures.py` will print the appropriate commands to generate these data files.

## Citation
```
@misc{rudner2020dgpsnr,
      title={On Signal-to-Noise Ratio Issues in Variational Inference for Deep Gaussian Processes},
      author={Tim G. J. Rudner and Oscar Key and Yarin Gal and Tom Rainforth},
      year={2020},
      eprint = {arXiv:2011.00515},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

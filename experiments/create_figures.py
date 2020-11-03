"""Produces the figures and tables in the main paper.

Creating each plot requires data files, such as model checkpoints, sampled gradient estimates, or computed test
log-likelihoods. This module will print the appropriate commands to generate the required data files before attempting
to create the plot. For more details, see the main README.
"""

import functools
import inspect
import os
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from tqdm import tqdm

import build_run_commands
import experiment_common
from build_models import build_model

##### Matplotlib configuration #####

plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["font.size"] = 16
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["lines.markersize"] = 8
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.color"] = "grey"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams[
    "text.latex.preamble"
] = "\\usepackage{amsmath} \n \\usepackage{amssymb} \n \\usepackage{graphicx}"
plt.rcParams["image.cmap"] = "tab20"
non_dreg_color = get_cmap("tab20").colors[0]
dreg_color = get_cmap("tab20").colors[2]
highlight_color = get_cmap("tab20").colors[4]

##### Utility functions #####


def get_k(file_name) -> int:
    parts = os.path.splitext(file_name)[0].split("_")
    for part in parts:
        if part.startswith("k"):
            return int(part.replace("k", ""))
    raise ValueError(f"missing k def: {file_name}")


def _flatten_gradients(d):
    flat_gradients = []
    for param, gradients in d.items():
        if "W" in param:
            N = gradients.shape[0]
            flat_gradients.append(gradients.reshape((N, -1)))
    return np.concatenate(flat_gradients, axis=1)


def load_gradients(results_dir, experiment_name, dreg=False, batch_size=1, max_k=100000):
    dreg_ext = "_dreg" if dreg else ""
    experiment_path = os.path.join(
        results_dir, "checkpoints", experiment_name, f"gradients_bs{batch_size}{dreg_ext}_k*.npy",
    )

    Ks = sorted([get_k(path) for path in glob(experiment_path)])

    gradients_by_K = {}

    for K in tqdm(Ks):
        if K > max_k:
            continue

        path = os.path.join(
            results_dir,
            "checkpoints",
            experiment_name,
            f"gradients_bs{batch_size}{dreg_ext}_k{K}.npy",
        )
        gradients_by_param = np.load(path, allow_pickle=True).item()
        gradients_by_K[K] = _flatten_gradients(gradients_by_param)

    return gradients_by_K


def group_by_M(gradient_samples, M):
    if gradient_samples.ndim == 1:
        num_samples = len(gradient_samples)
        divisible_samples = gradient_samples[: int(num_samples - num_samples % M)]
        grouped_samples = divisible_samples.reshape(-1, M)
        return np.mean(grouped_samples, axis=1)
    elif gradient_samples.ndim == 2:
        num_samples = gradient_samples.shape[0]
        num_params = gradient_samples.shape[1]
        divisible_samples = gradient_samples[: int(num_samples - num_samples % M), :]
        grouped_samples = divisible_samples.reshape(-1, M, num_params)
        return np.mean(grouped_samples, axis=1)
    else:
        raise ValueError


def _flatten_aggs(aggregates):
    flat_aggs = []
    for param, aggs in aggregates.items():
        if "W" in param:
            flat_aggs.append(aggs.reshape((-1)))
    return np.concatenate(flat_aggs, axis=0)


@functools.lru_cache(maxsize=128)
def load_snrs(results_dir: str, experiment_name, dreg=False, batch_size=1, max_k=100000):
    dreg_ext = "_dreg" if dreg else ""
    experiment_path = os.path.join(
        results_dir, "checkpoints", experiment_name, f"*_bs{batch_size}{dreg_ext}_k*.npy"
    )

    Ks = sorted([get_k(path) for path in glob(experiment_path)])

    results_by_K = {}

    for K in Ks:
        if K > max_k:
            continue

        base_path = os.path.join(results_dir, "checkpoints", experiment_name)
        grad_path = os.path.join(base_path, f"gradients_bs{batch_size}{dreg_ext}_k{K}.npy")
        agg_path = os.path.join(base_path, f"gradags_bs{batch_size}{dreg_ext}_k{K}.npy")

        if os.path.isfile(agg_path):
            aggs = np.load(agg_path, allow_pickle=True).item()
            mean_over_samples = _flatten_aggs(aggs["mean"])
            std_over_samples = _flatten_aggs(aggs["std"])
        elif os.path.isfile(grad_path):
            gradients = _flatten_gradients(np.load(grad_path, allow_pickle=True).item())
            mean_over_samples = gradients.mean(axis=0)
            std_over_samples = gradients.std(axis=0)
        else:
            raise ValueError(f"Missing {base_path} K={K}")
        snr = np.abs(mean_over_samples) / std_over_samples
        results_by_K[K] = {"snr": snr, "mean": mean_over_samples, "std": std_over_samples}

    return results_by_K


@functools.lru_cache(maxsize=128)
def load_multidp_snrs(results_dir, exp, dreg, num_points):
    dreg_ext = "_dreg" if dreg else ""
    base_path = os.path.join(results_dir, "checkpoints", exp)
    experiment_path = os.path.join(base_path, f"*_bs1{dreg_ext}_k*.npy")

    results_by_K = {}
    Ks = sorted(set([get_k(path) for path in glob(experiment_path)]))
    for K in Ks:
        snrs = []
        for dp in range(num_points):
            path = os.path.join(base_path, f"gradags_bs1{dreg_ext}_k{K}_dp{dp}.npy")
            aggs = np.load(path, allow_pickle=True).item()
            mean_over_samples = _flatten_aggs(aggs["mean"])
            std_over_samples = _flatten_aggs(aggs["std"])

            snr = np.abs(mean_over_samples) / std_over_samples

            # If the std deviation is zero but the mean is also zero, then the snr should be 0 (not NaN).
            zero_std = std_over_samples == 0
            if np.any(zero_std):
                assert np.all(mean_over_samples[zero_std] == 0)
                snr[zero_std] = 0

            snrs.append(snr)

        mean_snr = np.mean(np.stack(snrs, axis=0), axis=0)
        results_by_K[K] = {"snr": mean_snr, "mean": None, "std": None}

    return results_by_K


def save_fig(name, use_tight=True, **kwargs):
    if use_tight:
        plt.tight_layout(**kwargs)
    figures_root = "figures"
    if not os.path.isdir(figures_root):
        os.mkdir(figures_root)
    figure_path = os.path.join(figures_root, f"{name}.pdf")
    plt.savefig(figure_path, format="pdf", bbox_inches="tight")


def no_leading_zero(x, pos):
    """Formats the given float as a string with any leading zero removed (i.e. 0.1 -> .1) ."""
    ret = "%.1f" % x
    return re.sub("^0.", ".", ret)


##### Plotting functions #####


def figure_1(results_dir: str):
    build_run_commands.figure_1()

    plt.figure(figsize=(2.8, 1.5))
    plt.tight_layout()
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)

    axl = plt.gca()
    axr = axl.twinx()

    for dreg in [False, True]:
        name = f"demo_normalized_L1_IWAE_SF0_K50_BS64_noise0.2"
        snrs_by_K = load_multidp_snrs(results_dir, name, dreg, 10)

        if len(snrs_by_K) == 0:
            print(f"No gradients/snrs for {name}")

        for param in [0, 1, 2, 3]:
            Ks = []
            snrs = []
            for K, res in snrs_by_K.items():
                if K not in [1] + list(range(5, 101, 5)):
                    continue

                snr_per_param = res["snr"]
                Ks.append(K)
                snrs.append(snr_per_param[param])

            color = dreg_color if dreg else non_dreg_color

            ax = axr if dreg else axl
            if param == 0:
                legend_lab = "\\textsc{dreg}" if dreg else "\\textsc{reg}"
            else:
                legend_lab = ""
            ax.plot(Ks, snrs, color=color, linewidth=1, label=legend_lab)

    axl.grid(True)
    axl.set_xlabel("$K$")
    axl.set_ylabel("\\textsc{reg} \\textsc{snr}", labelpad=0)
    axr.set_ylabel("\\textsc{dreg} \\textsc{snr}", labelpad=0)
    axl.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))
    axr.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    axl.set_ylim(0.0, 0.4)
    axr.set_ylim(0.0, 2.0)

    axl.set_yticks([0.0, 0.2, 0.4])
    axr.set_yticks([0, 1, 2])

    axl.set_xlim(0, 100)
    axr.set_xlim(0, 100)
    x_ticks = [0, 50, 100]
    axl.set_xticks(x_ticks)
    axr.set_xticks(x_ticks)

    axl.legend(loc=(0.05, 0.7))
    axr.legend(loc=(0.5, 0.2))

    save_fig("snr_demo", use_tight=False)
    plt.show()


def figure_2(results_dir: str):
    build_run_commands.figure_2()

    def plot_grad_hists(dreg, Ks, x_min, x_max, bs, name, tbs=64):
        parameter_id = 4
        fig, axes = plt.subplots(len(Ks), 1, figsize=(5, 4), sharex=True)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)

        gradients_by_K = load_gradients(
            results_dir, f"demo_normalized_L1_G5_IWAE_SF0_K50_BS64", dreg=dreg, batch_size=bs
        )
        for ax, K in zip(axes, Ks):
            ax.set_xlim(x_min, x_max)
            aggregated_samples = gradients_by_K[K][:, parameter_id]
            color = dreg_color if dreg else non_dreg_color
            trimmed_samples = aggregated_samples[x_min <= aggregated_samples]
            trimmed_samples = trimmed_samples[trimmed_samples <= x_max]
            ax.hist(trimmed_samples, bins=200, density=True, color=color)
            # Plot the density function.
            x = np.arange(x_min, x_max, (x_max - x_min) / 100)
            ax.axvline(np.mean(aggregated_samples), linestyle="-", color="black")
            ax.axvline(0.0, linestyle="--", color="black")
            y = scipy.stats.norm.pdf(x, np.mean(aggregated_samples), np.std(aggregated_samples))
            ax.plot(x, y, color=highlight_color)

            ax.set_ylabel(f"$K = {K}$")
            ax.set_yticks([])

            mean = aggregated_samples.mean(0)
            std = np.std(aggregated_samples, 0)
            snr = np.abs(mean) / std
            snr_text = f"{{SNR}} = {snr:.3f}"
            ax.text(
                0.03,
                0.92,
                snr_text,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
            )

            if K == Ks[-1]:
                ax.set_xlabel(name)

            ax.grid(True, which="minor", axis="y", color="r", linestyle="-", linewidth=2)

    plot_grad_hists(False, [1, 10, 100], -20000, 20000, 1, "Reparameterised Gradient Estimates")
    save_fig("gradient_histograms_dreg_off", use_tight=False)
    plt.show()

    plot_grad_hists(
        True, [1, 10, 100], -20000, 20000, 1, "Doubly-reparameterized Gradient Estimates"
    )
    save_fig("gradient_histograms_dreg_on", use_tight=False)
    plt.show()


def figure_3(results_dir: str):
    datasets = [
        "winered",
        "wilson_solar",
        "demo_normalized",
        "wilson_forest",
        "power",
        "wilson_pol",
        "winewhite",
    ]
    depths = [1, 2, 3, 4]
    Ks_to_plot = [1, 10, 100, 1000]

    build_run_commands.figure_3(datasets, Ks_to_plot, depths)

    num_rows = 2
    num_cols = len(datasets) + (len(datasets) - 1)

    bar_width = 0.8
    gridspec = dict(hspace=0.08, width_ratios=([1.0] + [0.36]) * (len(datasets) - 1) + [1.0])
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(10, 3), gridspec_kw=gridspec, sharex=True
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.0, top=1.0, wspace=0.0)

    for col in range(1, num_cols, 2):
        for row in range(2):
            axes[row][col].set_axis_off()
            axes[row][col].set_yticks([])

    for row, dreg in enumerate([False, True]):
        for dataset_i, dataset in enumerate(datasets):
            col = dataset_i * 2
            max_snr = 0.1

            Ks = []
            snrs = []

            for depth in depths:
                Gs = "_G5" * (depth - 1)
                if dataset == "demo_normalized":
                    name = f"demo_normalized_L1{Gs}_IWAE_SF0_K50_BS64"
                else:
                    name = f"{dataset}_L1{Gs}_IWAE_SF0_K50_BS64"
                snrs_by_K = load_multidp_snrs(results_dir, name, dreg, 10)

                if len(snrs_by_K) == 0:
                    print(f"No gradients/snrs for {name}")

                for K, res in snrs_by_K.items():
                    if K not in Ks_to_plot:
                        continue

                    snr_per_param = res["snr"]
                    snr_mean_over_params = np.mean(snr_per_param)
                    Ks.append(f"d{depth}K{K}")
                    snrs.append(snr_mean_over_params)
                Ks.append(f"d{depth}end")
                snrs.append(0.0)

            max_snr = max(max(snrs), max_snr)

            x_coords = [bar_width * i for i, _ in enumerate(Ks)]
            color = dreg_color if dreg else non_dreg_color
            axes[row][col].bar(x_coords, snrs, width=bar_width, color=color)

            major_ticks_locs = [bar_width * 1.5 + bar_width * 5 * i for i in range(len(depths))]
            axes[row][col].set_xticks(major_ticks_locs, False)
            axes[row][col].set_xticklabels(depths)

            axes[row][col].grid(True, axis="y")

            f = FormatStrFormatter("%.0f") if dreg else FuncFormatter(no_leading_zero)
            axes[row][col].yaxis.set_major_formatter(f)
            axes[row][col].spines["left"].set_visible(True)

            if not dreg:
                axes[row][col].set_ylim(0.0, 0.5)
                axes[row][col].set_yticks([0.0, 0.2, 0.4])
            else:
                tick_map = {
                    "winered": 3.0,
                    "wilson_solar": 3.0,
                    "demo_normalized": 5.0,
                    "wilson_forest": 6.0,
                    "power": 8.0,
                    "wilson_pol": 30.0,
                    "winewhite": 60.0,
                }
                tick = tick_map[dataset]
                axes[row][col].set_yticks([0.0, tick / 2, tick])
                axes[row][col].set_ylim(0.0, tick * 1.2)

            axes[row][col].yaxis.get_major_ticks()[1].label1.set_visible(False)

            if row == 0:
                label = dataset.replace("wilson_", "").replace("_normalized", "")
                axes[row][col].set_title(label)

            if row == 1:
                axes[row][col].set_xlabel("layers", fontsize=14)

            if dataset_i == 0:
                if dreg:
                    pad = 12
                    axes[row][col].set_ylabel("\\textsc{dreg snr}", labelpad=pad)
                else:
                    pad = 0
                    axes[row][col].set_ylabel("\\textsc{reg snr}", labelpad=pad)
            axes[row][col].tick_params("y", pad=0)

    save_fig("snr_grid", use_tight=False)
    plt.show()


def figure_4(results_dir: str):
    datasets = ["winered", "solar", "forest", "power", "pol", "winewhite"]
    build_run_commands.figure_4(datasets, results_dir)

    results = defaultdict(lambda: {False: [], True: []})
    with open(os.path.join("figure_data", "figure_4.txt")) as figure4_results:
        for line in figure4_results:
            if len(line) == 0:
                continue

            if " " in line:
                cols = line.split(" ")
            else:
                cols = line.split("\t")

            dataset = cols[0]
            fixq_str = cols[1]
            split = int(cols[2])
            values = [float(val) for val in cols[3:]]
            assert len(values) == 1
            mean = np.mean(values)

            if fixq_str == "T":
                results[dataset][True].append(mean)
            elif fixq_str == "F":
                results[dataset][False].append(mean)
            else:
                raise ValueError

    fig, axes = plt.subplots(1, len(results), figsize=(10, 2))

    range_map = {
        "wilson_forest": (-1, 1),
        "power": (-0.1, 0.3),
        "wilson_pol": (1.5, 3.2),
        "wilson_solar": (0.2, 3.5),
        "winewhite": (-1.2, -1.0),
        "winered": (-0.8, 2.5),
    }

    for ax, dataset in zip(
        axes, ["wilson_forest", "wilson_solar", "wilson_pol", "power", "winewhite", "winered"]
    ):
        diff = 0.2
        for fixq in [False, True]:
            if fixq:
                xs = [0 + diff]
            else:
                xs = [0 - diff]

            values = np.array(results[dataset][fixq])

            parts = ax.violinplot(values, xs, showmeans=True, widths=[0.38])

            color = highlight_color if fixq else non_dreg_color
            for shaded_area in parts["bodies"]:
                shaded_area.set_color(color)
            parts["cbars"].set_color(color)
            parts["cmins"].set_color(color)
            parts["cmaxes"].set_color(color)
            parts["cmeans"].set_color(color)

        ax.set_title(dataset.replace("wilson_", ""))
        ax.set_xticks([-diff, diff])
        ax.set_xticklabels(["$q(z)$", "$p(z)$"])
        ax.set_ylim(range_map[dataset][0], range_map[dataset][1])

        ax.set_yticks(np.linspace(range_map[dataset][0], range_map[dataset][1], 4))
        for tick in ax.yaxis.get_major_ticks()[1:-1]:
            tick.label1.set_visible(False)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(True, axis="y", which="both")

    axes[0].set_ylabel("log-likelihood")
    fig.tight_layout(pad=0.1, w_pad=0.2)
    plt.savefig("fixq_2layer_violins.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def figure_5(results_dir: str):
    build_run_commands.figure_5()

    def _sample_ys_and_plot_histogram(ax, results_root: str, name: str, point_i: int):
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

                n_samples = 10000
                ys = model.predict_y_samples(data.X_test[point_i].reshape(1, -1), n_samples)

                if "fixq" in name:
                    color = highlight_color
                else:
                    color = non_dreg_color
                ax.hist(ys.reshape(-1), bins=200, density=True, color=color)

    def _get_checkpoint_file(checkpoint_path: str, step_to_load: Optional[int]) -> str:
        if step_to_load is not None:
            return os.path.join(checkpoint_path, f"cp-{step_to_load}")
        else:
            return tf.train.latest_checkpoint(checkpoint_path)

    result_names = [
        ("wilson_forest_L1_G5_IWAE_SF2_K50_BS64", 2),
        ("winewhite_L1_G5_IWAE_SF2_K50_BS64", 0),
        ("wilson_forest_L1_G5_IWAE_SF2_K50_BS64_fixq", 2),
        ("winewhite_L1_G5_IWAE_SF2_K50_BS64_fixq", 0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5, 3))
    axes_flat = axes.flatten()
    for ax, (name, point_i) in zip(axes_flat, result_names):
        _sample_ys_and_plot_histogram(ax, results_dir, name, point_i)

    for ax in axes_flat:
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        ax.set_xticks(np.linspace(left, right, 4))
        ax.set_yticks(np.linspace(bottom, top, 4))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        ax.grid(True)

    axes[0, 0].set_ylabel("$q(z)$")
    axes[1, 0].set_ylabel("$p(z)$")
    axes[1, 0].set_xlabel("forest")
    axes[1, 1].set_xlabel("winewhite")

    fig.tight_layout(w_pad=0.2, h_pad=0.2)
    plt.savefig("reg_fixq_marginals.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def table_1_tll_wilcoxon(results_dir: str):
    build_run_commands.table_1(results_dir)

    reg_results = []
    dreg_results = []
    with open(os.path.join("figure_data", "table_1_tlls.txt")) as table1_results:
        for line in table1_results:
            if len(line) == 0:
                continue

            if " " in line:
                cols = line.split(" ")
            else:
                cols = line.split("\t")

            dataset = cols[0]
            dreg_str = cols[1]
            split = int(cols[2])
            values = [float(val) for val in cols[3:]]
            assert len(values) == 10, f"Failed on line {cols}"
            mean = np.mean(values)

            if dreg_str == "T":
                dreg_results.append((dataset, split, mean))
            elif dreg_str == "F":
                reg_results.append((dataset, split, mean))
            else:
                raise ValueError

    assert len(reg_results) == len(dreg_results)
    for (reg_dataset, reg_split, _), (dreg_dataset, dreg_split, _) in zip(
        reg_results, dreg_results
    ):
        assert reg_dataset == dreg_dataset
        assert reg_split == dreg_split

    reg_tlls = [r[2] for r in reg_results]
    dreg_tlls = [r[2] for r in dreg_results]

    print("two sided:", scipy.stats.wilcoxon(reg_tlls, dreg_tlls))
    print("one sided:", scipy.stats.wilcoxon(reg_tlls, dreg_tlls, alternative="less"))


def figure_8(results_dir: str):
    datasets = [
        "winered_L1_G5_IWAE_SF1_K50_BS64",
        "wilson_solar_L1_G5_IWAE_SF1_K50_BS64",
        "demo_normalized_L1_G5_IWAE_SF1_K50_BS64_noise0.2",
        "wilson_forest_L1_G5_IWAE_SF1_K50_BS64",
        "power_L1_G5_IWAE_SF1_K50_BS64",
        "wilson_pol_L1_G5_IWAE_SF1_K50_BS64",
        "winewhite_L1_G5_IWAE_SF1_K50_BS64",
    ]
    build_run_commands.figure_8(datasets)

    fig, axes = plt.subplots(1, len(datasets), figsize=(10, 3))
    plt.tight_layout()
    plt.subplots_adjust(wspace=1.0)

    for col, dataset in enumerate(datasets):
        label = dataset
        label = label.replace("_L1_G5_IWAE_SF1_K50_BS64", "").replace("normalized", "")
        label = label.replace("wilson_", "")
        label = label.replace("_", " ")
        axes[col].set_title(label)

        for dreg in [False, True]:
            snr_per_param = load_snrs(results_dir, dataset, dreg, batch_size=64)[50]["snr"]
            snr_mean_over_params = np.mean(snr_per_param)

            x = 2 if dreg else 1
            bar_width = 0.8
            color = dreg_color if dreg else non_dreg_color
            axes[col].bar(
                [x], [snr_mean_over_params], width=bar_width, tick_label=[""], color=color
            )

        axes[col].grid(True, axis="y")
        axes[col].set_xticks([1, 2])

    axes[0].set_ylabel("\\textsc{snr}")
    save_fig("table1_snr_confirmation", use_tight=False)
    plt.show()


def figure_9(results_dir: str):
    datasets = ["power_L1_G5_G5_IWAE_S0_K30_bs1"]
    sample_batch_sizes = [1, 64, 128, 256, 512]
    build_run_commands.figure_9(datasets, sample_batch_sizes)

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    for ax, dataset in zip([axes], datasets):
        for i, batch_size in enumerate(sample_batch_sizes):
            snrs_by_K = load_snrs(results_dir, dataset, batch_size=batch_size, dreg=False)

            Ks2 = []
            snrs2 = []
            for (K, res) in snrs_by_K.items():
                Ks2.append(K)
                snrs2.append(res["snr"].mean(axis=0))
            ax.plot(Ks2, snrs2, label=f"$N = {batch_size}$", color=f"C{i}")
            ax.set_xlabel("K")
            ax.set_ylabel("$\\textrm{SNR}_{N,30}^{\\normalsize\\textrm{DGP}}(\phi)$")
            ax.set_xscale("log")
            ax.set_xticks([1, 10, 20, 30, 40, 50])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.legend()
    save_fig("snr_k_batch_size")
    plt.show()


if __name__ == "__main__":
    figures = {
        name: function
        for name, function in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.startswith("figure") or name.startswith("table")
    }

    parser = ArgumentParser()
    parser.add_argument("figure", choices=figures.keys(), help="Name of figure or table to plot")
    parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        required=False,
        help="Directory containing results, as set by train_model.py --results_path",
    )
    args = parser.parse_args()

    figures[args.figure](args.results_path)

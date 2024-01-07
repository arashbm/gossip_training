import argparse
import json

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

import sys
import os.path as path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from decoder import TorchTensorDecoder


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("results")

    args = parser.parse_args()

    indices = {}

    std_samples_per_round = []
    std_percentiles_per_round = []
    std_means_per_round = []
    mean_accuracy_per_round = []
    mean_loss_per_round = []
    node_accuracies_per_round = {}
    node_losses_per_round = {}
    with open(args.results) as f:
        for line in tqdm(f):
            r = json.loads(line, cls=TorchTensorDecoder)
            std_samples = {}
            std_percentiles = {}
            std_means = {}
            mean_std_volatilities = {}
            for name, tensor in r["stds"].items():
                if name not in indices:
                    flat_ind = torch.randperm(
                            tensor.numel())[:args.sample_size]
                    indices[name] = unravel_index(flat_ind, tensor.shape)

                std_samples[name] = tensor[indices[name]]
                std_percentiles[name] = np.percentile(
                        tensor.numpy(), [10, 50, 90])
                std_means[name] = torch.mean(tensor).item()
            std_samples_per_round.append(std_samples)
            std_percentiles_per_round.append(std_percentiles)
            std_means_per_round.append(std_means)
            mean_accuracy_per_round.append(np.mean(
                list(r["test_accuracies"].values())))
            mean_loss_per_round.append(np.mean(
                list(r["test_losses"].values())))
            for i, acc in r["test_accuracies"].items():
                if i not in node_accuracies_per_round:
                    node_accuracies_per_round[i] = []
                node_accuracies_per_round[i].append(acc)
            for i, acc in r["test_losses"].items():
                if i not in node_losses_per_round:
                    node_losses_per_round[i] = []
                node_losses_per_round[i].append(acc)

    fig, axes = plt.subplots(nrows=4, ncols=2,
                             figsize=(16, 20),
                             sharey=True,
                             sharex=True)
    vol_fig, vol_axes = plt.subplots(nrows=4, ncols=2,
                                     figsize=(16, 20),
                                     sharey=True,
                                     sharex=True)

    keys = std_samples_per_round[0].keys()
    for name, ax, vol_ax in zip(keys, axes.flatten(), vol_axes.flatten()):
        ax_acc = ax.twinx()
        stds = np.array([d[name] for d in std_samples_per_round]).T
        std_10_percentiles = [p[name][0] for p in std_percentiles_per_round]
        std_90_percentiles = [p[name][2] for p in std_percentiles_per_round]
        std_medians = [p[name][1] for p in std_percentiles_per_round]
        std_means = [d[name] for d in std_means_per_round]

        vols = np.zeros(len(stds[0]) - args.window_size)
        non_nan_vols = 0
        for std in stds:
            ax.plot(range(len(std)), std, marker='.', ms=2, alpha=0.2, ls='--')

            ret = (std[1:] - std[:-1])/std[:-1]
            windows = np.lib.stride_tricks.sliding_window_view(
                    ret, args.window_size)
            rolling_volatility = np.array([
                np.std(window) for window in windows])
            vol_ax.plot(
                    range(
                        args.window_size//2+1,
                        len(std)-args.window_size//2),
                    rolling_volatility,
                    marker='.', ms=2, alpha=0.2, ls='--')
            if not np.any(np.isnan(rolling_volatility)):
                vols += rolling_volatility
                non_nan_vols += 1
        ax.plot(range(len(std_medians)), std_medians,
                ms=2, label="median STD[parameter]")
        ax.plot(range(len(std_means)), std_means,
                ms=2, label="mean STD[parameter]")
        ax.legend()

        mean_vols = vols/non_nan_vols
        vol_ax.plot(
                range(
                    args.window_size//2+1,
                    len(std)-args.window_size//2),
                mean_vols,
                ms=2, label="mean Volatility[STD[parameter]")

        ax_acc.plot(range(len(mean_accuracy_per_round)),
                    mean_accuracy_per_round, ms=2,
                    label="Mean Accuracy")

        ax.set_title(name)
        ax.set_yscale("log")
        ax.set_xlabel("Round")
        ax.set_ylabel("Param STD")
        ax.set_ylim(1e-9, 1e-1)

        vol_ax.set_title(name)
        vol_ax.set_yscale("log")
        vol_ax.set_xlabel("Round")
        vol_ax.set_ylabel("Volatility[Param STD]")

        ax_acc.legend()
        ax_acc.set_ylabel("Mean Accuracy")
    fig.savefig(f"figures/{args.distribution}-param-std.pdf")
    vol_fig.savefig(f"figures/{args.distribution}-param-std-volatility.pdf")

    fig, ((ax_loss, ax_acc), (ax_loss_diff, ax_acc_diff)) = \
        plt.subplots(nrows=2, ncols=2, figsize=(16, 10),
                     sharex=True)
    ax_loss.plot(range(len(mean_loss_per_round)),
                 mean_loss_per_round, ms=2,
                 label="Mean Accuracy")
    for i, losses in node_losses_per_round.items():
        ax_loss.plot(range(len(losses)), losses,
                     ls='--', ms=2, alpha=0.3)
    ax_loss.set_ylabel("Test loss")
    ax_loss.set_xlabel("Round")

    print("Max diff[loss]",
          np.argmax(-np.diff(mean_loss_per_round))+1)
    ax_loss_diff.plot(range(1, len(mean_loss_per_round)),
                      -np.diff(mean_loss_per_round), ms=2,
                      label="Mean Accuracy")
    ax_loss_diff.set_yscale("log")
    ax_loss_diff.set_ylabel("-diff[Test loss]")
    ax_loss_diff.set_xlabel("Round")

    ax_acc.plot(range(len(mean_accuracy_per_round)),
                mean_accuracy_per_round, ms=2,
                label="Mean Accuracy")
    for i, accs in node_accuracies_per_round.items():
        ax_acc.plot(range(len(accs)), accs,
                    ls='--', ms=2, alpha=0.3)
    ax_acc.set_ylabel("Mean Accuracy")
    ax_acc.set_xlabel("Round")

    ax_acc_diff.plot(range(1, len(mean_accuracy_per_round)),
                     np.diff(mean_accuracy_per_round), ms=2,
                     label="Mean Accuracy")
    ax_acc_diff.set_ylabel("diff[Mean Accuracy]")
    ax_acc_diff.set_xlabel("Round")
    fig.savefig(f"figures/{args.distribution}-test-loss.pdf")

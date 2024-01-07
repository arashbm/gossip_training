import argparse
import json

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--exponent", type=float, default=1)
    parser.add_argument("results", nargs="+")

    args = parser.parse_args()

    size_accuracies = {}
    size_losses = {}
    for filename in tqdm(args.results):
        accuracies = {}
        losses = {}
        size = None
        with open(filename) as f:
            for line in tqdm(f):
                r = json.loads(line)
                size = len(r["test_accuracies"])

                for i, acc in r["test_accuracies"].items():
                    if i not in accuracies:
                        accuracies[i] = []
                    accuracies[i].append(acc)

                for i, loss in r["test_losses"].items():
                    if i not in losses:
                        losses[i] = []
                    losses[i].append(loss)

            if size is None:
                continue

            if size not in size_accuracies:
                size_accuracies[size] = []
            size_accuracies[size].extend(accuracies.values())

            if size not in size_losses:
                size_losses[size] = []
            size_losses[size].extend(losses.values())

    fig, ((ax_loss, ax_acc), (ax_loss_diff, ax_acc_diff)) = \
        plt.subplots(nrows=2, ncols=2, figsize=(16, 10), sharex=True)

    for size in sorted(size_losses):
        max_len = max(map(len, size_losses[size]))

        means = np.nanmean([vals + [np.nan]*(max_len - len(vals))
                            for vals in size_losses[size]],
                           axis=0)
        times = np.arange(max_len)

        ax_loss.plot(times*1/size**args.exponent, means,
                     ms=2, label=f"N={size}")
        ax_loss_diff.plot(times[1:]*1/size**args.exponent, -np.diff(means),
                          ms=2, label=f"N={size}")

    ax_loss.set_ylabel("Test loss")
    ax_loss.set_xlabel(f"$t N^{{-{args.exponent}}}$")
    ax_loss.set_xscale("log")
    ax_loss.legend()

    ax_loss_diff.set_xscale("log")
    ax_loss_diff.set_yscale("log")
    ax_loss_diff.set_ylabel("-diff[Test loss]")
    ax_loss_diff.set_xlabel(f"$t N^{{-{args.exponent}}}$")

    for size in sorted(size_accuracies):
        max_len = max(map(len, size_accuracies[size]))

        means = np.nanmean([vals + [np.nan]*(max_len - len(vals))
                            for vals in size_accuracies[size]],
                           axis=0)
        times = np.arange(max_len)

        ax_acc.plot(times*1/size**args.exponent, means,
                    ms=2, label=f"N={size}")
        ax_acc_diff.plot(times[1:]*1/size**args.exponent, np.diff(means),
                         ms=2, label=f"N={size}")

    ax_acc.set_ylabel("Mean Accuracy")
    ax_acc.set_xlabel("$t N^{-1}$")
    ax_acc.set_xscale("log")

    ax_acc_diff.set_ylabel("diff[Mean Accuracy]")
    ax_acc_diff.set_xlabel("$t N^{-1}$")
    ax_acc_diff.set_xscale("log")
    fig.savefig(f"figures/{args.distribution}-test-loss-scaling.pdf")

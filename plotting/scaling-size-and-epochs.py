import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-condition", type=str, default="homogenous",
                        choices=["homogenous", "heterogenous"])
    parser.add_argument("--epochs-exp", type=float, default=0)
    parser.add_argument("--n-exp", type=float, default=0)
    args = parser.parse_args()

    dd = 1

    rounds = {}
    loss_means = {}
    loss_stds = {}
    for size in [16, 32, 64]:
        for epochs in [2, 4, 8, 16, 32, 64]:
            with open(f"logs/aggregate/regular/nodes-{size}-"
                      f"degree-2-{args.initial_condition}-"
                      f"init-{epochs}-epochs-simple-balanced_iid-"
                      "100-items.jsonl") as f:
                for line in f:
                    j = json.loads(line)

                    if (size, epochs) not in loss_means:
                        loss_means[(size, epochs)] = []
                    loss_means[(size, epochs)].append(j["mean_loss"])

                    if (size, epochs) not in rounds:
                        rounds[(size, epochs)] = []
                    rounds[(size, epochs)].append(j["round"])

                    if (size, epochs) not in loss_stds:
                        loss_stds[(size, epochs)] = []
                    loss_stds[(size, epochs)].append(j["std_loss"])

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(10, 6),
                             sharey=True, sharex=True)
    max_y = -float('inf')
    min_y = float('inf')
    for ep, ax in zip([2, 4, 8, 16, 32, 64], axes.ravel()):
        for size in [16, 32, 64]:
            rs = np.array(rounds[(size, ep)])[1:]
            ls = np.array(loss_means[(size, ep)])[1:]

            min_y = min(min_y, np.min(ls[1:]))
            max_y = max(max_y, np.max(ls[1:]))

            yerrs = np.array(loss_stds[(size, ep)])[1:]
            ax.plot(rs[:-dd:dd]*(size**args.n_exp), -np.diff(ls[::dd]),
                    label=f"$N={size}$", marker="o", ls='--', ms=2)
            # ax.errorbar(rs, ls, yerr=yerrs, label=f"$N={size}$",
            #             errorevery=5, marker="o", ls='--', ms=2)
            ax.set_xlabel(f"$t N^{{{-args.n_exp}}}$")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"$N_{{epochs}}={ep}$")
    # margin = (max_y - min_y)*0.05
    # axes[0].set_ylim(min_y - margin, max_y + margin)
    axes.ravel()[0].set_ylabel("-Diff[Test loss]")
    axes.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(f"figures/scaling-{args.initial_condition}"
                f"-with-sizes-fixed-epochs-exp-{args.n_exp}.pdf")

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3),
                             sharey=True, sharex=True)

    max_y = -float('inf')
    min_y = float('inf')
    for size, ax in zip([16, 32, 64], axes.ravel()):
        for ep in [2, 4, 8, 16, 32, 64]:
            rs = np.array(rounds[(size, ep)])[1:]
            ls = np.array(loss_means[(size, ep)])[1:]

            min_y = min(min_y, np.min(ls[1:]))
            max_y = max(max_y, np.max(ls[1:]))

            yerrs = np.array(loss_stds[(size, ep)])[1:]
            ax.plot(rs[:-dd:dd]*(ep**args.epochs_exp), -np.diff(ls[::dd]),
                    label=f"$k={ep}$", marker="o", ls='--', ms=2)
            # ax.errorbar(rs, ls, yerr=yerrs, label=f"$k={ep}$",
            #             errorevery=5, marker="o", ls='--', ms=2)
            ax.set_xlabel(f"$t N_{{epochs}}^{{{-args.epochs_exp}}}$")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"$N={size}$")
    # margin = (max_y - min_y)*0.05
    # axes[0, 0].set_ylim(min_y - margin, max_y + margin)
    axes.ravel()[0].set_ylabel("-Diff[Test loss]")
    axes.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(f"figures/scaling-{args.initial_condition}"
                f"-with-epochs-fixed-sizes-exp-{args.epochs_exp}.pdf")

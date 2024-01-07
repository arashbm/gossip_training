import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-condition", type=str, default="homogenous",
                        choices=["homogenous", "heterogenous"])
    parser.add_argument("--k-exp", type=float, default=0)
    parser.add_argument("--n-exp", type=float, default=0)
    args = parser.parse_args()

    dd = 1

    rounds = {}
    loss_means = {}
    loss_stds = {}
    for size in [256, 512]:
        for degree in [16, 32, 64, 128]:
            with open(f"logs/aggregate/regular/nodes-{size}-"
                      f"degree-{degree}-{args.initial_condition}-"
                      "init-4-epochs-simple-balanced_iid-"
                      "100-items.jsonl") as f:
                for line in f:
                    j = json.loads(line)

                    if (size, degree) not in loss_means:
                        loss_means[(size, degree)] = []
                    loss_means[(size, degree)].append(j["mean_loss"])

                    if (size, degree) not in rounds:
                        rounds[(size, degree)] = []
                    rounds[(size, degree)].append(j["round"])

                    if (size, degree) not in loss_stds:
                        loss_stds[(size, degree)] = []
                    loss_stds[(size, degree)].append(j["std_loss"])

    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(14, 3),
                             sharey=True, sharex=True, squeeze=False)
    max_y = -float('inf')
    min_y = float('inf')
    for deg, ax in zip([16, 32, 64, 128], axes.ravel()):
        for size in [256, 512]:
            rs = np.array(rounds[(size, deg)])[1:]
            ls = np.array(loss_means[(size, deg)])[1:]

            min_y = min(min_y, np.min(ls[1:]))
            max_y = max(max_y, np.max(ls[1:]))

            yerrs = np.array(loss_stds[(size, deg)])[1:]
            # ax.plot(rs[:-dd:dd]*(size**args.n_exp), -np.diff(ls[::dd]),
            #         label=f"$N={size}$", marker="o", ls='--', ms=2)
            ax.errorbar(rs*(size**args.n_exp),
                        ls, yerr=yerrs, label=f"$N={size}$",
                        errorevery=5, marker="o", ls='--', ms=2)
            ax.set_xlabel(f"$t N^{{{args.n_exp}}}$")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"$k={deg}$")
    # margin = (max_y - min_y)*0.05
    # axes[0, 0].set_ylim(min_y - margin, max_y + margin)
    axes[0, 0].set_ylabel("-Diff[Test loss]")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(f"figures/scaling-{args.initial_condition}"
                f"-with-sizes-fixed-degree-exp-{args.n_exp}.pdf")

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 4),
                             sharey=True, sharex=True, squeeze=False)

    max_y = -float('inf')
    min_y = float('inf')
    for size, ax in zip([256, 512], axes.ravel()):
        for deg in [16, 32, 64, 128]:
            rs = np.array(rounds[(size, deg)])[1:]
            ls = np.array(loss_means[(size, deg)])[1:]

            min_y = min(min_y, np.min(ls[1:]))
            max_y = max(max_y, np.max(ls[1:]))

            yerrs = np.array(loss_stds[(size, deg)])[1:]
            # ax.plot(rs[:-dd:dd]*(deg**args.k_exp), -np.diff(ls[::dd]),
            #         label=f"$k={deg}$", marker="o", ls='--', ms=2)
            ax.errorbar(rs*(deg**args.k_exp),
                        ls, yerr=yerrs,
                        label=f"$k={deg}$",
                        errorevery=5, marker="o", ls='--', ms=2)
            ax.set_xlabel(f"$t k^{{{args.k_exp}}}$")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"$N={size}$")
    # margin = (max_y - min_y)*0.05
    # axes[0, 0].set_ylim(min_y - margin, max_y + margin)
    axes[0, 0].set_ylabel("-Diff[Test loss]")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(f"figures/scaling-{args.initial_condition}"
                f"-with-degress-fixed-sizes-exp-{args.k_exp}.pdf")

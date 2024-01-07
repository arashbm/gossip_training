import argparse
import json
import statistics

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def sem(nums):
    if len(nums) < 2:
        return float("nan")

    return statistics.stdev(nums)/(len(nums)**0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("results")

    args = parser.parse_args()

    indeces = np.array([])

    stds_per_round = {}
    std_percentiles_per_round = {}
    std_mean_per_round = {}
    accuracy_per_round = {}
    with open(args.results) as f:
        for line in tqdm(f):
            r = json.loads(line)
            if indeces.size == 0:
                indeces = np.random.choice(
                        range(len(r["stds"])), 100,
                        replace=False)
            stds_per_round[r["round"]] = [
                    r["stds"][i] for i in indeces]
            std_percentiles_per_round[r["round"]] = np.percentile(
                    r["stds"], [1, 5, 10, 50, 90, 95, 99])
            std_mean_per_round[r["round"]] = np.mean(r["stds"])
            accuracy_per_round[r["round"]] = np.mean(
                    list(r["test_accuracies"].values()))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax_acc = ax.twinx()
    stds = np.array([stds_per_round[round]
                     for round in sorted(stds_per_round)]).T
    std_10_percentiles = [std_percentiles_per_round[round][2]
                          for round in sorted(std_percentiles_per_round)]
    std_90_percentiles = [std_percentiles_per_round[round][4]
                          for round in sorted(std_percentiles_per_round)]
    std_medians = [std_percentiles_per_round[round][3]
                   for round in sorted(std_percentiles_per_round)]
    std_means = [std_mean_per_round[round]
                 for round in sorted(std_mean_per_round)]

    accuracues = np.array([accuracy_per_round[round]
                           for round in sorted(accuracy_per_round)])

    for std in stds:
        ax.plot(range(len(std)), std, marker='.', ms=2, alpha=0.2, ls='--')
    ax.plot(range(len(std_medians)), std_medians,
            ms=2, label="median STD[parameter]")
    ax.plot(range(len(std_means)), std_means,
            ms=2, label="mean STD[parameter]")
    ax.legend()

    ax_acc.plot(range(len(accuracues)), accuracues, ms=2,
                label="Mean Accuracy")

    ax.set_yscale("log")
    ax.set_xlabel("Round")
    ax.set_ylabel("Param STD")
    ax_acc.legend()
    ax_acc.set_ylabel("Mean Accuracy")
    fig.savefig(f"figures/{args.distribution}-param-std.pdf")

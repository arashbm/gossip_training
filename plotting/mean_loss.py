import argparse
import json

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("results")

    args = parser.parse_args()

    mean_accuracy_per_round = []
    mean_loss_per_round = []
    with open(args.results) as f:
        for line in tqdm(f):
            r = json.loads(line)
            mean_accuracy_per_round.append(
                np.mean(list(r["test_accuracies"].values())))
            mean_loss_per_round.append(
                np.mean(list(r["test_losses"].values())))

    fig, ((ax_loss, ax_acc), (ax_loss_diff, ax_acc_diff)) = \
        plt.subplots(nrows=2, ncols=2, figsize=(16, 10),
                     sharex=True)
    ax_loss.plot(range(len(mean_loss_per_round)),
                 mean_loss_per_round, ms=2,
                 label="Mean Accuracy")
    ax_loss.set_ylabel("Test loss")
    ax_loss.set_xlabel("Round")

    ax_loss_diff.plot(range(1, len(mean_loss_per_round)),
                      -np.diff(mean_loss_per_round), ms=2,
                      label="Mean Accuracy")
    ax_loss_diff.set_yscale("log")
    ax_loss_diff.set_ylabel("-diff[Test loss]")
    ax_loss_diff.set_xlabel("Round")

    ax_acc.plot(range(len(mean_accuracy_per_round)),
                mean_accuracy_per_round, ms=2,
                label="Mean Accuracy")
    ax_acc.set_ylabel("Mean Accuracy")
    ax_acc.set_xlabel("Round")

    ax_acc_diff.plot(range(1, len(mean_accuracy_per_round)),
                     np.diff(mean_accuracy_per_round), ms=2,
                     label="Mean Accuracy")
    ax_acc_diff.set_ylabel("diff[Mean Accuracy]")
    ax_acc_diff.set_xlabel("Round")
    fig.savefig(f"figures/{args.distribution}-test-loss.pdf")

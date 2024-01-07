import argparse
import json

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", type=str, nargs="+")
    args = parser.parse_args()

    round_losses = {}
    round_accuracies = {}

    for input_file in args.input_files:
        with open(input_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.decoder.JSONDecodeError:
                    break

                if data["round"] not in round_losses:
                    round_losses[data["round"]] = []
                round_losses[data["round"]].extend(
                        data["test_losses"].values())

                if data["round"] not in round_accuracies:
                    round_accuracies[data["round"]] = []
                round_accuracies[data["round"]].extend(
                        data["test_accuracies"].values())

    for round in sorted(round_losses):
        print(json.dumps(
            {"round": round,
             "mean_loss": np.mean(round_losses[round]),
             "std_loss": np.std(round_losses[round]),
             "mean_accuracy": np.mean(round_accuracies[round]),
             "std_accuracy": np.std(round_accuracies[round])}))

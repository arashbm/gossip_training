import argparse
import json
import statistics

# import matplotlib as mpl
import matplotlib.pyplot as plt


def sem(nums):
    if len(nums) < 2:
        return float("nan")

    return statistics.stdev(nums)/(len(nums)**0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("results", nargs="+")

    args = parser.parse_args()

    results_per_round = {}
    for result in args.results:
        with open(result) as f:
            for line in f:
                r = json.loads(line)
                if r["round"] not in results_per_round:
                    results_per_round[r["round"]] = []
                results_per_round[r["round"]].append(r["test_accuracies"])

    node_accuracy = {}
    for round in sorted(results_per_round):
        for res in results_per_round[round]:
            for node, acc in res.items():
                if node not in node_accuracy:
                    node_accuracy[node] = {}
                if round not in node_accuracy[node]:
                    node_accuracy[node][round] = []
                node_accuracy[node][round].append(res[node])

    # cmap = mpl.colormaps['tab10']
    # cols = [cmap(i) for i in range(cmap.N)]

    fig, ax = plt.subplots(figsize=(8, 6))
    points = [(round, acc)
              for round in sorted(node_accuracy[args.node])
              for acc in node_accuracy[args.node][round]]
    ax.plot(*zip(*points), marker='.', ms=2, alpha=0.5, ls='')

    means = [(round, statistics.mean(node_accuracy[args.node][round]))
             for round in sorted(node_accuracy[args.node])]
    sems = [sem(node_accuracy[args.node][round])*1.96
            for round in sorted(node_accuracy[args.node])]
    ax.errorbar(*zip(*means), yerr=sems, ls='--', errorevery=10)
    ax.axhline(0.1, ls='--', color="grey")

    ax.set_xlabel("round")
    ax.set_ylabel("accuracy")
    fig.savefig(f"figures/{args.distribution}-accuracy-{args.node}.pdf")

import argparse
from pathlib import Path

import numpy as np
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt


def loss(x):
    a = 10
    n = 1
    return a*n + x**2 - a*sp.cos(2*np.pi*x)
    # return -sp.sinc(x)


def gradient(x):
    return sp.diff(loss(x), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--edge-prob", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("figures_dir")

    args = parser.parse_args()

    x = sp.symbols('x')
    loss_func = sp.lambdify(x, loss(x), "numpy")
    gradient_func = sp.lambdify(x, gradient(x), "numpy")

    gen = np.random.default_rng(args.seed)
    nodes = gen.random(size=(args.nodes, 1))*100 - 200.0
    velocities = np.zeros_like(nodes)
    momentum = 0.7
    graph = nx.fast_gnp_random_graph(
            n=args.nodes, p=args.edge_prob,
            seed=args.seed)

    states = [nodes]

    for t in range(1, args.steps):
        print(np.mean(loss_func(nodes)))
        print(nodes)
        # velocities = momentum*velocities + \
        #     gradient_func(nodes)*args.learning_rate
        # nodes -= velocities
        nodes -= gradient_func(nodes)*args.learning_rate
        new_nodes = np.copy(nodes)
        for n in graph:
            neighbours = nx.neighbors(graph, n)
            neighbour_states = [nodes[i] for i in neighbours]
            new_nodes[n] = np.mean([*neighbour_states, nodes[n]])
        nodes = new_nodes
        states.append(nodes)

    fig, (ax, ax_end) = plt.subplots(
            ncols=2,
            width_ratios=[0.6, 0.4],
            gridspec_kw={"wspace": 0.05},
            sharey=True)
    ax.plot(range(60), [np.std(s) for s in states[:60]])
    ax.set_ylabel("STD[state]")
    ax.set_xlabel("time")
    ax.set_yscale("log")
    ax_end.plot(range(args.steps-40, args.steps),
                [np.std(s) for s in states[-40:]])
    ax_end.set_xlabel("time")
    fig.savefig(Path(args.figures_dir)/"std.pdf")

    fig, (ax, ax_loss) = plt.subplots(
            ncols=2,
            width_ratios=[0.8, 0.2],
            gridspec_kw={"wspace": 0.05},
            sharey=True)
    xs = np.linspace(
            np.min(np.array(states)),
            np.max(np.array(states)),
            num=200000)
    ax_loss.plot(loss_func(xs), xs)
    ax_loss.set_xlabel("loss")
    # ax_loss.set_xlim(0, 500)

    ax.set_xlabel("time")
    ax.set_ylabel("state")

    for ns in zip(*states):
        ax.plot(range(args.steps), ns, ls='--')

    fig.savefig(Path(args.figures_dir)/"states.pdf")

    fig, ax = plt.subplots()
    ax.set_xlabel("time")
    ax.set_ylabel("loss")

    for ns in zip(*states):
        ax.plot(range(args.steps), [loss_func(s) for s in ns], alpha=0.2)
    ax.plot(range(args.steps),
            [np.mean(loss_func(s)) for s in states],
            ls='--')
    fig.savefig(Path(args.figures_dir)/"loss.pdf")

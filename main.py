import argparse
import json
import sys

import numpy as np
import torchvision
import networkx as nx
from tqdm import tqdm

import sampler
from node import Node, SimpleModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--edge-prob", type=int, default=0.2)
    parser.add_argument("--t-max", type=int, default=1000)
    parser.add_argument("--local-validation-split", type=float, default=.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--zipf-alpha", type=float, default=1.6)

    parser.add_argument("--kd-alpha", type=float, default=1.0)
    parser.add_argument("--skd-beta", type=float, default=0.99)

    args = parser.parse_args()

    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))]))

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))]))

    rng = np.random.default_rng()
    partitions = sampler.zipf_sampler(
            args.zipf_alpha,
            args.users,
            dataset,
            args.local_validation_split,
            random_state=rng)
    # for t, v in partitions:
    #     print(len(t), len(v))
    #     train_labels = np.array([target for _, target in t])
    #     valid_labels = np.array([target for _, target in v])
    #     print("train:",
    #           "\t".join(f"{np.count_nonzero(train_labels == i)}"
    #                     for i in range(10)))
    #     print("valid:",
    #           "\t".join(f"{np.count_nonzero(valid_labels == i)}"
    #                     for i in range(10)))

    input_shape = dataset[0][0].numel()
    output_shape = len(dataset.classes)
    graph = nx.fast_gnp_random_graph(args.users, args.edge_prob, seed=rng)

    nodes = []
    for train, valid in partitions:
        nodes.append(Node(
            SimpleModel(input_shape, output_shape),
            train, valid))

    accuracy_over_time = []

    test_accuracy = []
    for i, node in enumerate(nodes):
        acc = node.test(test_dataset)
        test_accuracy.append(acc)
    accuracy_over_time.append(test_accuracy)
    print("mean test accuracy:", np.mean(accuracy_over_time, axis=1),
          file=sys.stderr)
    print(json.dumps({"round": 0, "test_accuracies": test_accuracy}))

    for t in tqdm(range(args.t_max)):
        new_states = []
        for i, node in enumerate(nodes):
            neighbours = [nodes[n] for n in graph[i]]
            trusts = [1.0 for _ in neighbours]
            new_states.append(
                    node.aggregate_neighbours(
                        neighbours, trusts))

        test_accuracy = []
        for i, node in enumerate(nodes):
            node.load_params(new_states[i])
            node.train(epochs=args.epochs,
                       learning_rate=args.learning_rate,
                       momentum=args.momentum,
                       skd_beta=args.skd_beta,
                       kd_alpha=args.kd_alpha)
            acc = node.test(test_dataset)
            test_accuracy.append(acc)
        accuracy_over_time.append(test_accuracy)

        print("mean test accuracy:", np.mean(accuracy_over_time, axis=1),
              file=sys.stderr)
        print(json.dumps({"round": t+1, "test_accuracies": test_accuracy}))

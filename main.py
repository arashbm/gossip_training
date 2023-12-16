import argparse
import json
import sys

import numpy as np
import torch
import torchvision
import networkx as nx
from tqdm import tqdm

import sampler
from node import Node, SimpleModel, calculate_model_std
from decoder import TorchTensorEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph")
    parser.add_argument("--t-max", type=int, default=1000)
    parser.add_argument("--validation-split", type=float, default=.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.5)

    parser.add_argument("--data-distribution",
                        choices=["zipf", "balanced_iid"],
                        required=True)
    parser.add_argument("--zipf-alpha", type=float, default=1.6)
    parser.add_argument("--items-per-user", type=int)

    parser.add_argument("--aggregation-method",
                        choices=["decdiff", "avg"],
                        required=True)

    parser.add_argument("--training-method",
                        choices=["vt", "simple"],
                        required=True)
    parser.add_argument("--kd-alpha", type=float, default=1.0)
    parser.add_argument("--skd-beta", type=float, default=0.99)

    parser.add_argument("--early-stopping",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--pretrained-model", type=str)

    args = parser.parse_args()

    graph = nx.read_edgelist(args.graph)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}", file=sys.stderr)

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

    partitions = None
    if args.data_distribution == "zipf":
        partitions = sampler.zipf_sampler(
                alpha=args.zipf_alpha,
                users=graph.number_of_nodes(),
                dataset=dataset,
                validation_split=args.validation_split,
                random_state=rng)
    elif args.data_distribution == "balanced_iid":
        partitions = sampler.balanced_iid_sampler(
                users=graph.number_of_nodes(), dataset=dataset,
                validation_split=args.validation_split,
                random_state=rng,
                items_per_user=args.items_per_user)
    else:
        raise ValueError(
                f"data distribution ``{args.data_distribution}''"
                " is not defined.")

    sampler.print_partition_counts(partitions)

    input_shape = dataset[0][0].numel()
    output_shape = len(dataset.class_to_idx)

    nodes = {}
    for (train, valid), node in zip(partitions, graph.nodes):
        model = SimpleModel(input_shape, output_shape).to(device)
        if args.pretrained_model:
            model.load_state_dict(torch.load(args.pretrained_model))
        nodes[node] = Node(model, train, valid)

    accuracy_over_time = []

    test_accuracy = {}
    test_loss = {}
    for i, node in nodes.items():
        loss, acc = node.test(test_dataset, device=device)
        test_accuracy[i] = acc
        test_loss[i] = loss

    accuracy_over_time.append(test_accuracy)
    print("mean test accuracy:", np.mean(
        [list(a.values()) for a in accuracy_over_time], axis=1),
          file=sys.stderr)
    print(json.dumps({
        "round": 0,
        "test_accuracies": test_accuracy,
        "test_losses": test_loss,
        "stds": calculate_model_std(list(nodes.values()))
        }, cls=TorchTensorEncoder))

    for t in tqdm(range(1, args.t_max)):
        new_states = {}
        for i, node in nodes.items():
            neighbours = [nodes[n] for n in graph[i]]
            trusts = [1.0 for _ in neighbours]

            if args.aggregation_method == "decdiff":
                new_states[i] = \
                    node.aggregate_neighbours_decdiff(
                        neighbours, trusts)
            elif args.aggregation_method == "avg":
                new_states[i] = \
                    node.aggregate_neighbours_simple_mean(
                        neighbours, trusts)
            else:
                raise ValueError(
                        f"aggregation method ``{args.aggregation_method}''"
                        " is not defined")

        test_accuracy = {}
        test_loss = {}
        for i, node in nodes.items():
            node.load_params(new_states[i])
            if args.training_method == "vt":
                node.train_virtual_teacher(
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        momentum=args.momentum,
                        skd_beta=args.skd_beta,
                        kd_alpha=args.kd_alpha,
                        device=device,
                        early_stopping=args.early_stopping)
            elif args.training_method == "simple":
                node.train_simple(
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        momentum=args.momentum,
                        device=device,
                        early_stopping=args.early_stopping)
            else:
                raise ValueError(
                        f"training method ``{args.training_method}''"
                        " is not defined")
            loss, acc = node.test(test_dataset, device=device)
            test_accuracy[i] = acc
            test_loss[i] = loss
        accuracy_over_time.append(test_accuracy)

        print("mean test accuracy:", np.mean(
            [list(a.values()) for a in accuracy_over_time], axis=1),
          file=sys.stderr)
        print(json.dumps({
            "round": t,
            "test_accuracies": test_accuracy,
            "test_losses": test_loss,
            "stds": calculate_model_std(list(nodes.values()))
            }, cls=TorchTensorEncoder))

import argparse
import json
import sys
import math

import numpy as np
import torch
import torchvision
import networkx as nx
from tqdm import tqdm
from torch.profiler import record_function

import sampler
from node import Node, SimpleModel, stds_across_nodes, stds_across_params
from decoder import TorchTensorEncoder


def save_state(nodes: dict[str, Node], round: int, filename: str):
    states = {}
    for v, node in nodes.items():
        states[v] = {
                "dict": node.model.state_dict(),
                "subsets": {
                    "training": node.training_dataset,
                    "validation": node.validation_dataset
                }}
    torch.save({"round": t, "states": states}, filename)


class PairDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    data: torch.Tensor
    targest: torch.Tensor

    def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        assert targets.size(0) == data.size(0), "Size mismatch between tensors"
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return (self.data[index], self.targets[index])

    def __getitems__(self, indices: list):
        return (self.data[indices], self.targets[indices])

    def __len__(self):
        return self.data.size(0)


def load_mnist(device: torch.device = torch.device("cpu")):
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))]))

    data, targets = zip(*train_dataset)
    train_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))]))
    data, targets = zip(*test_dataset)
    test_dataset = PairDataset(
            torch.stack(data).to(device),
            torch.tensor(targets).to(device))
    print("loaded datasets", file=sys.stderr)

    return train_dataset, test_dataset


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

    parser.add_argument("--gain-correction", choices=["none", "sqrt", "graph"],
                        default="sqrt")
    parser.add_argument("--early-stopping",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--pretrained-model", type=str)
    parser.add_argument("--parameter-samples", type=int, default=100)

    parser.add_argument("--test-every", type=int, default=1)
    parser.add_argument("--test-exponential",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--test-all-rounds-before", type=int, default=0)

    parser.add_argument("--checkpoint-file", type=str)

    args = parser.parse_args()

    graph = nx.read_edgelist(args.graph)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}", file=sys.stderr)

    dataset, test_dataset = load_mnist(device=device)

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
    output_shape = len(torch.unique(
        torch.tensor([t for _, t in dataset])))

    gain = 1.0
    if args.gain_correction == "sqrt":
        gain = graph.number_of_nodes()**0.5
    elif args.gain_correction == "none":
        gain = 1.0
    elif args.gain_correction == "graph":
        adj = nx.to_numpy_array(graph) + \
                np.eye(graph.number_of_nodes())
        adj /= adj.sum(axis=1, keepdims=True)
        vals, vecs = np.linalg.eig(adj.T)
        idx = np.argmax(vals)
        gain = np.abs(np.sum(vecs[:, idx]))
    else:
        raise NotImplementedError("gain correction not implemented")

    nodes = {}
    model_class = SimpleModel
    for (train, valid), node in zip(partitions, graph.nodes):
        model = model_class(input_shape, output_shape,
                            gain=gain).to(device)
        if args.pretrained_model:
            model.load_state_dict(torch.load(args.pretrained_model))
        nodes[node] = Node(model, train, valid)

    with torch.no_grad():
        node = nodes[list(graph.nodes())[0]]
        param_sample_indecies = {
            k: torch.randperm(v.numel())[:args.parameter_samples].to(device)
            for k, v in node.model.named_parameters()}

    test_accuracy = {}
    test_loss = {}
    for i, node in tqdm(nodes.items()):
        loss, acc = node.test(test_dataset, device=device)
        test_accuracy[i] = acc
        test_loss[i] = loss

    print("mean test accuracy:",
          np.mean(list(test_accuracy.values())),
          file=sys.stderr)
    print(json.dumps({
        "round": 0,
        "test_accuracies": test_accuracy,
        "test_losses": test_loss,
        "params": {
            n: nodes[n].model.param_sample(param_sample_indecies)
            for n in graph.nodes},
        "stds_across_nodes": stds_across_nodes(
            list(nodes.values()), param_sample_indecies),
        "stds_across_params": stds_across_params(
            list(nodes.values())),
        }, cls=TorchTensorEncoder))

    next_test = 1
    for t in tqdm(range(1, args.t_max)):
        new_states = {}
        aggregation_changes = {}
        for i, node in nodes.items():
            neighbours = [nodes[n] for n in graph[i]]
            trusts = [1.0 for _ in neighbours]

            with record_function("aggregation"):
                if args.aggregation_method == "decdiff":
                    new_states[i], aggregation_changes[i] = \
                        node.aggregate_neighbours_decdiff(
                            neighbours, trusts,
                            param_sample_indecies=param_sample_indecies)
                elif args.aggregation_method == "avg":
                    new_states[i], aggregation_changes[i] = \
                        node.aggregate_neighbours_simple_mean(
                            neighbours, trusts,
                            param_sample_indecies=param_sample_indecies)
                else:
                    raise ValueError(
                            f"aggregation method "
                            f"``{args.aggregation_method}''"
                            f" is not defined")

        test_accuracy = {}
        test_loss = {}
        training_changes = {}
        for i, node in tqdm(nodes.items()):
            node.load_params(new_states[i])
            with record_function("training"):
                if args.training_method == "vt":
                    training_changes[i] = node.train_virtual_teacher(
                            epochs=args.epochs,
                            learning_rate=args.learning_rate,
                            momentum=args.momentum,
                            skd_beta=args.skd_beta,
                            kd_alpha=args.kd_alpha,
                            device=device,
                            early_stopping=args.early_stopping,
                            param_sample_indecies=param_sample_indecies)
                elif args.training_method == "simple":
                    training_changes[i] = node.train_simple(
                            epochs=args.epochs,
                            learning_rate=args.learning_rate,
                            momentum=args.momentum,
                            device=device,
                            early_stopping=args.early_stopping,
                            param_sample_indecies=param_sample_indecies)
                else:
                    raise ValueError(
                            f"training method ``{args.training_method}''"
                            " is not defined")
            if t == next_test or t < args.test_all_rounds_before:
                with record_function("testing"):
                    loss, acc = node.test(test_dataset, device=device)
                    test_accuracy[i] = acc
                    test_loss[i] = loss

        if t == next_test or t < args.test_all_rounds_before:
            print("mean test accuracy:",
                  np.mean(list(test_accuracy.values())),
                  file=sys.stderr)
            print(json.dumps({
                "round": t,
                "test_accuracies": test_accuracy,
                "test_losses": test_loss,
                "training_changes": training_changes,
                "aggregation_changes": aggregation_changes,
                "params": {
                    n: nodes[n].model.param_sample(param_sample_indecies)
                    for n in graph.nodes},
                "stds_across_nodes": stds_across_nodes(
                    list(nodes.values()), param_sample_indecies),
                "stds_across_params": stds_across_params(
                    list(nodes.values())),
                }, cls=TorchTensorEncoder))
        if t == next_test:
            if args.test_exponential:
                next_test = math.ceil(next_test*1.5)
                print(next_test, file=sys.stderr)
            else:
                next_test += args.test_every
    if args.checkpoint_file:
        save_state(nodes=nodes, round=t, filename=args.checkpoint_file)

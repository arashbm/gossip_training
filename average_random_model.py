import argparse
import sys

import torch
import torchvision

from node import Node, SimpleModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--parameters-file", type=str)
    args = parser.parse_args()

    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))]))

    input_shape = dataset[0][0].numel()
    output_shape = len(dataset.classes)

    nodes = [
            Node(SimpleModel(input_shape, output_shape), dataset, dataset)
            for i in range(args.nodes)]

    new_state = nodes[0].model.state_dict()
    if args.nodes > 1:
        new_state = nodes[0].aggregate_neighbours_simple_mean(
                nodes[1:], [1.0 for _ in range(len(nodes)-1)])

    if args.parameters_file:
        torch.save(new_state, args.parameters_file)

import argparse
import json
import sys

import numpy as np
import torch
import torchvision

import sampler
from node import Node, SimpleModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-split", type=float, default=.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.5)

    parser.add_argument("--data-distribution",
                        choices=["zipf", "balanced_iid"],
                        required=True)
    parser.add_argument("--zipf-alpha", type=float, default=1.6)
    parser.add_argument("--items-per-user", type=int)

    parser.add_argument("--training-method",
                        choices=["vt", "simple"],
                        required=True)
    parser.add_argument("--kd-alpha", type=float, default=1.0)
    parser.add_argument("--skd-beta", type=float, default=0.99)
    parser.add_argument("--parameters-file", type=str)

    args = parser.parse_args()

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
                args.zipf_alpha,
                1,
                dataset,
                args.validation_split,
                random_state=rng)
    elif args.data_distribution == "balanced_iid":
        partitions = sampler.balanced_iid_sampler(
                users=1, dataset=dataset,
                validation_split=args.validation_split,
                random_state=rng, items_per_user=args.items_per_user)
    else:
        raise ValueError(
                f"data distribution ``{args.data_distribution}''"
                " is not defined.")

    sampler.print_partition_counts(partitions)

    input_shape = dataset[0][0].numel()
    output_shape = len(dataset.classes)

    train, valid = partitions[0]
    node = Node(
        SimpleModel(input_shape, output_shape).to(device),
        train, valid)

    accuracy_over_time = []

    test_loss, test_accuracy = node.test(
            test_dataset, device=device)
    accuracy_over_time.append({0: test_accuracy})

    print("mean test accuracy:", np.mean(
        [list(a.values()) for a in accuracy_over_time], axis=1),
          file=sys.stderr)
    print(json.dumps({
        "test_accuracies": test_accuracy,
        "test_losses": test_loss,
        }))

    if args.training_method == "vt":
        node.train_virtual_teacher(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                skd_beta=args.skd_beta,
                kd_alpha=args.kd_alpha,
                device=device,
                early_stopping=False)
    elif args.training_method == "simple":
        node.train_simple(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                device=device,
                early_stopping=False)
    else:
        raise ValueError(
                f"training method ``{args.training_method}''"
                " is not defined")
    test_loss, test_accuracy = node.test(
            test_dataset, device=device)
    accuracy_over_time.append({0: test_accuracy})
    print("mean test accuracy:", np.mean(
        [list(a.values()) for a in accuracy_over_time], axis=1),
      file=sys.stderr)
    print(json.dumps({
        "test_accuracies": test_accuracy,
        "test_losses": test_loss,
        }))

    if args.parameters_file:
        torch.save(node.model.state_dict(), args.parameters_file)

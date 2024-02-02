import argparse
import json
import sys

import numpy as np
import torch
from tqdm import tqdm

import sampler
from node import Node, SimpleModel, stds_across_params
from main import load_mnist
from decoder import TorchTensorEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    parser.add_argument("--training-method",
                        choices=["vt", "simple"],
                        required=True)
    parser.add_argument("--kd-alpha", type=float, default=1.0)
    parser.add_argument("--skd-beta", type=float, default=0.99)

    parser.add_argument("--early-stopping",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--parameter-samples", type=int, default=100)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}", file=sys.stderr)

    dataset, test_dataset = load_mnist(device)

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
    output_shape = len(torch.unique(
        torch.tensor([t for _, t in dataset])))

    train, valid = partitions[0]
    node = Node(
        SimpleModel(input_shape, output_shape, gain=1.0).to(device),
        train, valid)

    with torch.no_grad():
        param_sample_indecies = {
            k: torch.randperm(v.numel())[:args.parameter_samples].to(device)
            for k, v in node.model.named_parameters()}

    accuracy_over_time = []

    test_loss, test_accuracy = node.test(
            test_dataset, device=device)
    accuracy_over_time.append({0: test_accuracy})

    print("mean test accuracy:", np.mean(
        [list(a.values()) for a in accuracy_over_time], axis=1),
          file=sys.stderr)
    print(json.dumps({
        "round": 0,
        "test_accuracies": test_accuracy,
        "test_losses": test_loss,
        "params": node.model.param_sample(param_sample_indecies),
        "stds_across_params": stds_across_params([node]),
        }, cls=TorchTensorEncoder))

    for t in tqdm(range(1, args.t_max)):
        training_changes = None

        if args.training_method == "vt":
            training_changes = node.train_virtual_teacher(
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    momentum=args.momentum,
                    skd_beta=args.skd_beta,
                    kd_alpha=args.kd_alpha,
                    device=device,
                    early_stopping=args.early_stopping,
                    param_sample_indecies=param_sample_indecies)
        elif args.training_method == "simple":
            training_changes = node.train_simple(
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
        test_loss, test_accuracy = node.test(
                test_dataset, device=device)
        accuracy_over_time.append({0: test_accuracy})
        print("mean test accuracy:", np.mean(
            [list(a.values()) for a in accuracy_over_time], axis=1),
          file=sys.stderr)
        print(json.dumps({
            "round": t,
            "test_accuracies": test_accuracy,
            "test_losses": test_loss,
            "training_changes": training_changes,
            "params": node.model.param_sample(param_sample_indecies),
            "stds_across_params": stds_across_params([node]),
            }, cls=TorchTensorEncoder))

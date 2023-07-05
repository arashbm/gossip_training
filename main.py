from typing import Union
import copy

import numpy as np
import torchvision
import torch
import networkx as nx
from tqdm import tqdm

import sampler


DatasetLike = Union[torch.utils.data.Dataset,
                    torch.utils.data.Subset]


class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        layer_sizes = [512, 256, 128]

        self.input_size = input_size
        self.output_size = output_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, layer_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_sizes[0], layer_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_sizes[1], layer_sizes[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_sizes[2], output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class Node:
    def __init__(self, model: torch.nn.Module,
                 train_dataset: DatasetLike,
                 validation_dataset: DatasetLike):
        self.model = model
        self.training_dataset = torch.utils.data.dataloader.DataLoader(
                train_dataset, batch_size=32, shuffle=True)
        self.validation_dataset = torch.utils.data.dataloader.DataLoader(
                validation_dataset, batch_size=32)

    def aggregate_neighbours(self, neighbours: list["Node"],
                             trusts: list[float], alphas: list[float]):
        trusts = np.array(trusts)/sum(trusts)

        current_model_params = copy.deepcopy(self.model.state_dict())

        if len(neighbours) == 0:
            return current_model_params

        avg_params = self.model.state_dict()
        for key in avg_params.keys():
            for i, neighbour in enumerate(neighbours):
                neighbour_params = neighbour.model.state_dict()
                if i == 0:
                    avg_params[key] = \
                            alphas[i]*trusts[i]*neighbour_params[key]
                else:
                    avg_params[key] += \
                            alphas[i]*trusts[i]*neighbour_params[key]

        for key in avg_params.keys():
            dist = current_model_params[key] - avg_params[key]
            lp_dist = torch.norm(dist) + 1
            current_model_params[key] -= dist/lp_dist

        return current_model_params

    def load_params(self, params: dict):
        self.model.load_state_dict(params)

    def train(self, epochs: int, learning_rate: float, momentum: float):
        self.model.train()
        skd_beta = 0.99
        kd_alpha = 1.0

        optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        kl_loss = torch.nn.KLDivLoss(reduction='none')

        current_valid_loss = np.Inf
        prev_params = copy.deepcopy(self.model.state_dict())

        for i in range(epochs):
            epoch_loss = 0
            for data, target in self.training_dataset:
                optimizer.zero_grad()
                output = self.model(data)

                onehot = torch.nn.functional.one_hot(target, output_shape)
                t_prob = skd_beta*onehot + \
                    (1 - onehot)*(1 - skd_beta)/(output_shape - 1)
                kl = kl_loss(torch.log_softmax(output, dim=1), t_prob)
                loss = torch.nanmean(
                        (1 - kd_alpha)*criterion(output, target) +
                        kd_alpha * torch.sum(kl, dim=1), dim=0)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(f"epoch {i} loss:", epoch_loss)

            print(f"current best validation loss:\t\t{current_valid_loss:.8f}")
            val_loss, val_acc = self.validate()

            if val_loss == 0 or val_loss > current_valid_loss:
                self.model.load_state_dict(prev_params)
                print("breaking")
                break
            else:
                current_valid_loss = val_loss

    def validate(self):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total = 0
        corrects = 0
        total_loss = 0
        with torch.no_grad():
            for data, target in self.validation_dataset:
                output = self.model(data)
                loss = criterion(output, target)
                total += target.shape[0]
                corrects += \
                    torch.count_nonzero(
                        torch.argmax(
                            torch.softmax(output, dim=1),
                            dim=1) == target).item()
                total_loss += loss.item()
        accuracy = corrects/total
        print(f"validation accuracy:\t{accuracy:.8f}\t{total_loss:.8f}")
        return total_loss, accuracy

    def test(self, test_dataset: DatasetLike):
        self.model.eval()
        test_dataset = torch.utils.data.dataloader.DataLoader(
                test_dataset, batch_size=32)
        total = 0
        corrects = 0
        with torch.no_grad():
            for data, target in test_dataset:
                output = self.model(data)
                total += target.shape[0]
                corrects += np.sum(
                        (torch.argmax(
                            torch.softmax(output, dim=1),
                            dim=1) == target).numpy())
        accuracy = corrects/total
        print("test accuracy:", accuracy)
        return accuracy


if __name__ == "__main__":
    alpha = 1.6
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

    t_max = 1000
    users = 20
    validation_split = 0.2
    rng = np.random.default_rng()
    partitions = sampler.zipf_sampler(
            alpha, users, dataset, validation_split, rng)
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
    graph = nx.fast_gnp_random_graph(users, 0.2)

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
    print("mean test accuracy:", np.mean(accuracy_over_time, axis=1))

    for t in tqdm(range(t_max)):
        new_states = []
        for i, node in enumerate(nodes):
            neighbours = [nodes[n] for n in graph[i]]
            trusts = [1.0 for _ in neighbours]
            alphas = [1.0 for _ in neighbours]
            new_states.append(
                    node.aggregate_neighbours(neighbours, trusts, alphas))

        test_accuracy = []
        for i, node in enumerate(nodes):
            node.load_params(new_states[i])
            node.train(epochs=5, learning_rate=0.001, momentum=0.5)
            acc = node.test(test_dataset)
            test_accuracy.append(acc)
        accuracy_over_time.append(test_accuracy)

        print("mean test accuracy:", np.mean(accuracy_over_time, axis=1))

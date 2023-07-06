from typing import Union
import copy
import sys

import torch


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
        self.data_size = len(train_dataset)
        self.training_dataset = torch.utils.data.dataloader.DataLoader(
                train_dataset, batch_size=32, shuffle=True, num_workers=4)
        self.validation_dataset = torch.utils.data.dataloader.DataLoader(
                validation_dataset, batch_size=32, num_workers=4)

    def aggregate_neighbours(self,
                             neighbours: list["Node"],
                             trusts: list[float]):

        total_neighbour_data = sum(
                neighbour.data_size for neighbour in neighbours)
        alphas = [neighbour.data_size/total_neighbour_data
                  for neighbour in neighbours]

        current_model_params = copy.deepcopy(self.model.state_dict())

        if len(neighbours) == 0:
            return current_model_params

        avg_params = copy.deepcopy(self.model.state_dict())
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

    def train(self, epochs: int,
              learning_rate: float, momentum: float,
              skd_beta: float, kd_alpha: float,
              device: torch.device):
        self.model.train()
        optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        kl_loss = torch.nn.KLDivLoss(reduction='none')

        current_valid_loss = float("inf")
        prev_params = copy.deepcopy(self.model.state_dict())

        for i in range(epochs):
            epoch_loss = 0
            for data, target in self.training_dataset:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)

                onehot = torch.nn.functional.one_hot(
                        target, self.model.output_size)
                t_prob = skd_beta*onehot + \
                    (1 - onehot)*(1 - skd_beta)/(self.model.output_size - 1)
                kl = kl_loss(torch.log_softmax(output, dim=1), t_prob)
                loss = torch.nanmean(
                        (1 - kd_alpha)*criterion(output, target) +
                        kd_alpha * torch.sum(kl, dim=1), dim=0)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_loss, val_acc = self.validate(device=device)

            if val_loss == 0 or val_loss > current_valid_loss:
                self.model.load_state_dict(prev_params)
                current_valid_loss = val_loss
                print("breaking", file=sys.stderr)
                break
            else:
                current_valid_loss = val_loss

    def validate(self, device: torch.device):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total = 0
        corrects = 0
        total_loss = 0
        with torch.no_grad():
            for data, target in self.validation_dataset:
                data, target = data.to(device), target.to(device)
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
        print(f"validation accuracy:\t{accuracy:.8f}\t{total_loss:.8f}",
              file=sys.stderr)
        return total_loss, accuracy

    def test(self, test_dataset: DatasetLike, device: torch.device):
        self.model.eval()
        test_dataset = torch.utils.data.dataloader.DataLoader(
                test_dataset, batch_size=32, num_workers=4)
        total = 0
        corrects = 0
        with torch.no_grad():
            for data, target in test_dataset:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                total += target.shape[0]
                corrects += \
                    torch.count_nonzero(
                        torch.argmax(
                            torch.softmax(output, dim=1),
                            dim=1) == target).item()
        accuracy = corrects/total
        print("test accuracy:", accuracy, file=sys.stderr)
        return accuracy

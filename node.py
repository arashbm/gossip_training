from typing import Union
import copy
import sys
import statistics

import torch


DatasetLike = Union[torch.utils.data.Dataset,
                    torch.utils.data.Subset]


class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        h1 = 512
        h2 = 256
        h3 = 128
        self.fc1 = torch.nn.Linear(input_size, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, h3)
        self.fc4 = torch.nn.Linear(h3, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(self.relu(x))
        return x


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

    def aggregate_neighbours_decentralised(self,
                                           neighbours: list["Node"],
                                           trusts: list[float]):
        total_neighbourhood_data = sum(
                neighbour.data_size for neighbour in neighbours)
        total_neighbourhood_data += self.data_size

        # ratio of neighbours' data to total neighbourhood data (inc. self)
        alphas = [neighbour.data_size/total_neighbourhood_data
                  for neighbour in neighbours]

        current_model_params = copy.deepcopy(self.model.state_dict())

        if len(neighbours) == 0:
            return current_model_params

        self_trust = 1.
        self_alpha = self.data_size/total_neighbourhood_data
        avg_params = copy.deepcopy(self.model.state_dict())
        for key in avg_params.keys():
            avg_params[key] = torch.zeros_like(avg_params[key])

        for key in avg_params.keys():
            for i, neighbour in enumerate(neighbours):
                neighbour_params = neighbour.model.state_dict()
                avg_params[key] += \
                    alphas[i]*trusts[i]*neighbour_params[key]

            avg_params[key] += \
                self_alpha*self_trust*current_model_params[key]

        return avg_params

    def aggregate_neighbours_virtual_teacher(self,
                                             neighbours: list["Node"],
                                             trusts: list[float]):
        total_neighbour_data = sum(
                neighbour.data_size for neighbour in neighbours)

        # ratio of neighbours' data to total neighbourhood data (w/o self)
        alphas = [neighbour.data_size/total_neighbour_data
                  for neighbour in neighbours]

        current_model_params = copy.deepcopy(self.model.state_dict())

        if len(neighbours) == 0:
            return current_model_params

        avg_params = copy.deepcopy(self.model.state_dict())
        for key in avg_params.keys():
            avg_params[key] = torch.zeros_like(avg_params[key])

        for key in avg_params.keys():
            for i, neighbour in enumerate(neighbours):
                neighbour_params = neighbour.model.state_dict()
                avg_params[key] += \
                    alphas[i]*trusts[i]*neighbour_params[key]

        for key in avg_params.keys():
            dist = current_model_params[key] - avg_params[key]
            lp_dist = torch.norm(dist, p=2) + 1
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

        epoch_losses = []
        for i in range(epochs):
            batch_losses = []
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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(statistics.mean(batch_losses))

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
        losses = 0
        with torch.no_grad():
            for data, target in self.validation_dataset:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                losses += loss.item()
                preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
                corrects += torch.count_nonzero(preds == target).item()
                total += len(data)
        accuracy = corrects/total
        loss = losses/total
        print(f"validation accuracy:\t{accuracy:.8f}\t{loss:.8f}",
              file=sys.stderr)
        return loss, accuracy

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
                preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
                corrects += torch.count_nonzero(preds == target).item()
                total += len(data)
        accuracy = corrects/total
        print("test accuracy:", accuracy, file=sys.stderr)
        return accuracy

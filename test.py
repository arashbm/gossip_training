import itertools
import copy

import torch
import torchvision

import node


dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))]))


def state_dict_equal(a, b):
    for k, v in a.items():
        if k not in b:
            return False
        if not torch.allclose(v, b[k]):
            return False
    for k, v in b.items():
        if k not in a:
            return False
        if not torch.allclose(v, a[k]):
            return False
    return True


def test_node_aggregate_simple_mean():
    nodes = [node.Node(
        node.VerySimpleModel(2, 2), dataset, dataset) for i in range(10)]
    for i, j in itertools.combinations(range(10), 2):
        assert not state_dict_equal(
                nodes[i].model.state_dict(),
                nodes[j].model.state_dict())

    new_states = []
    for i in range(len(nodes)):
        new_states.append(
            nodes[i].aggregate_neighbours_simple_mean(
                nodes[:i] + nodes[i+1:],
                [1.0 for _ in range(len(nodes)-1)]))

    for i, j in itertools.combinations(range(10), 2):
        assert not state_dict_equal(
                nodes[i].model.state_dict(), new_states[i])
        assert not state_dict_equal(
                nodes[j].model.state_dict(), new_states[j])
        assert state_dict_equal(new_states[i], new_states[j])


def test_node_load_param():
    n = node.Node(
        node.VerySimpleModel(2, 2), dataset, dataset)
    new_params = node.VerySimpleModel(2, 2).state_dict()
    assert not state_dict_equal(
            n.model.state_dict(), new_params)

    old_params = copy.deepcopy(n.model.state_dict())

    n.load_params(new_params)
    assert n.model.state_dict() is not new_params
    assert state_dict_equal(
            n.model.state_dict(), new_params)
    assert not state_dict_equal(
            n.model.state_dict(), old_params)


if __name__ == "__main__":
    test_node_aggregate_simple_mean()
    test_node_load_param()
    print("done")

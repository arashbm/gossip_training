import argparse

import numpy as np
import networkx as nx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--degree', type=float, required=True)
    parser.add_argument('--in-degree', type=float, required=True)
    args = parser.parse_args()

    gen = np.random.RandomState(args.seed)
    p_in = args.in_degree/(args.nodes//2 - 1)
    p_out = (args.degree - args.in_degree)/(args.nodes//2)
    while not nx.is_connected(
            g := nx.random_partition_graph(
                (args.nodes//2,)*2, p_in, p_out, seed=gen)):
        pass

    for i, j in g.edges():
        print(i, j)

import argparse
import math

import networkx as nx
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--degree', type=int, required=True)
    parser.add_argument('--dims', type=int, required=True)
    args = parser.parse_args()

    unit_nball_vol = math.pi**(args.dims/2)/math.gamma(args.dims/2 + 1)
    r = (args.degree/(args.nodes*unit_nball_vol))**(1/args.dims)

    print(unit_nball_vol)
    print(r)

    gen = np.random.RandomState(args.seed)
    while not nx.is_connected(
            g := nx.random_geometric_graph(
                args.nodes, r, dim=args.dims, seed=gen)):
        print("retrying")

    # print average degree
    print(sum(d for _, d in g.degree())/g.number_of_nodes())

    # for i, j in g.edges():
    #     print(i, j)

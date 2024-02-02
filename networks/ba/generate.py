import reticula as ret
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--degree', type=int, required=True)
    args = parser.parse_args()

    gen = ret.mersenne_twister(args.seed)
    while not ret.is_connected(
            g := ret.random_barabasi_albert_graph[ret.int64](
                args.nodes, args.degree//2, gen)):
        pass

    for e in g.edges():
        print(*e.incident_verts())

import reticula as ret
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, required=True)
    args = parser.parse_args()

    g = ret.complete_graph[ret.int64](args.nodes)

    for e in g.edges():
        print(*e.incident_verts())

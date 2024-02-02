import reticula as ret
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', type=int, required=True)
    parser.add_argument('--dims', type=int, required=True)
    args = parser.parse_args()

    g = ret.square_grid_graph[ret.int64](args.side, args.dims, periodic=True)

    for e in g.edges():
        print(*e.incident_verts())

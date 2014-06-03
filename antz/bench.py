import csv
import time
import collections
import datetime
from antz import dijkstra
from antz.graph_gen import GridGraphGenerator
from antz.sim import WaypointEdge, AntColony, ShortestPathAlgorithm
from antz.sim import Waypoint


def main():
    def node_factory(name):
        if name == 'waypoint':
            return Waypoint
        else:
            raise ValueError('Node type %s does not exist' % name)

    def edge_factory(name):
        if name == 'waypoint':
            return WaypointEdge
        else:
            raise ValueError('Edge type %s does not exist' % name)

    graph_generator = GridGraphGenerator(500, 500,
                                         5, node_factory, edge_factory,
                                         min_food_hops=500,
                                         max_food_hops=500)

    nest_node, food_node, graph = graph_generator()

    n_ant_sol = 5
    n = 10

    ant_configs = []

    for i in range(0, n_ant_sol):
        alpha = i
        for j in range(0, n_ant_sol):
            beta = j
            for k in range(0, n_ant_sol):
                phero_dec = 0.01 * k
                ant_configs.append(
                    (('alpha', alpha),
                     ('beta', beta),
                     ('phero_decrease', phero_dec)))

    print(len(ant_configs))

    tsum_dijkstra = 0
    lensum_dijkstra = 0

    tsum_ant = collections.defaultdict(int)
    lensum_ant = collections.defaultdict(int)

    for j in range(0, n):
        print('Run %d' % j)
        start = time.time()
        dijkstra_dist, dijkstra_pred = dijkstra.shortest_path(graph, nest_node, food_node)
        dijkstra_len = dijkstra_dist[food_node]
        lensum_dijkstra += dijkstra_len
        tsum_dijkstra += time.time() - start

        colony = AntColony(100)
        algorithm = ShortestPathAlgorithm()
        algorithm.alpha = 4
        algorithm.beta = 2
        algorithm.gamma = 2
        for k, config in enumerate(ant_configs):
            print('Run %d.%d (%d)' % (j, k, len(ant_configs)))
            for vn, vv in config:
                setattr(algorithm, vn, vv)
            runner = colony.create_runner(algorithm, graph, nest_node)
            times_same = 0
            last_best = None
            start = time.time()
            for i in range(0, 1000):
                runner.next_step()
                if times_same > 50:
                    break
                if runner.best_solution != last_best:
                    last_best = runner.best_solution
                    times_same = 0
                    if runner.best_solution[1] == dijkstra_len:
                        break
                elif runner.best_solution:
                    times_same += 1
            lensum_ant[config] += runner.best_solution[1]
            tsum_ant[config] += time.time() - start

    with open('%s.csv' % datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f'), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_ALL)

        writer.writerow(['dijkstra', '', tsum_dijkstra / n, lensum_dijkstra / n])

        for config, lsum in sorted([(config, l) for config, l in lensum_ant.items()], key=lambda x: x[1]):
            writer.writerow(['ant', ', '.join(['%s=%s' % (k, v) for k, v in config]),
                             tsum_ant[config] / n, lensum_ant[config] / n])


if __name__ == '__main__':
    main()
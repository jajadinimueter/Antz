import time
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
                                         10, node_factory, edge_factory,
                                         min_food_hops=100,
                                         max_food_hops=500)

    nest_node, food_node, graph = graph_generator()

    start = time.time()
    dijkstra_dist, dijkstra_pred = dijkstra.shortest_path(graph, nest_node, food_node)
    dijkstra_len = dijkstra_dist[food_node]
    print('Dijstra took %.2f seconds. Solution %s' % (time.time() - start, dijkstra_len))

    start = time.time()
    colony = AntColony(100)
    runner = colony.create_runner(ShortestPathAlgorithm(), graph, nest_node)
    for i in range(0, 100):
        print('step %s' % i)
        runner.next_step()
        if runner.best_solution:
            if runner.best_solution[1] <= dijkstra_len:
                break
    print('Ant took %.2f seconds. Solution %s' % (time.time() - start, runner.best_solution[1]))


if __name__ == '__main__':
    main()
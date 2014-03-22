"""
"""


import sets
import .graph
import bisect
import itertools
import collections


class Environment(object):
    """
    Holds the environment. Ants are able to receive information
    about the environment especially the next possible moves 
    through the environment.
    """

    def __init__(self, graph):
        """
        :param graph: The whole environment graph
        """
        self._graph = graph

    def get_edges(self, node):
        """
        Returns the edges given a node
        """
        return self._graph[node]


class AntBehavior(object):
    """
    We need something like a behavior because we want to solve
    different problems with the ants.

    Such as:

        - Find the shortest path
        - Solve the TSP
        - Do war

    The behavior should be an attribte of the ant, like a part
    of the brain. We don't want subclasses here because those would
    be too static.
    """

    TYPE = None  # should be set by the implementation

    def choose_edge(self, edges):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

    def visit_edge(self, edge):
        """
        Called when the ant moves over the edge to another node.
        """

    def leave_edge(self, edge):
        """

        """

    def visit_node(self, node):
        """
        """

    def leave_node(self, node):
        """
        """

    def __repr__(self):
        """
        Default repr implementation for convienience 
        """
        return ('%s(TYPE=%s)' 
                    % (self.__class__.__name__,
                        self.TYPE))


class ShortestPathBehavior(AntBehavior):
    """
    Makes the ant behave in a way which leads to a shortest 
    path solution.
    """

    TYPE = 'shortest_path'

    def __init__(self):
        """
        """

    def choose_edge(self, ant, edges):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

        colony = ant.colony
        pkind = colony.pheromone_kind('default')
        # choose the random edge giving edges with
        # more pheromones a higher chance
        weighted_edges = [(e.pheromone_level(pkind), e) for e in edges]
        choices, weights = zip(*weighted_edges)
        cumdist = list(itertools.accumulate(weights))
        x = random.random() * cumdist[-1]

        return choices[bisect.bisect(cumdist, x)]


class AntColony(object):
    def __init__(self, name):
        # make a set out of the ants
        self._pheromone_kinds = {
            'default': PheromoneKind(name='%s:%s' % (name, 'default'))
        }

    def pheromone_kind(self, kind):
        return self._pheromone_kinds[kind]


class Ant(object):
    def __init__(self, colony, initial_node, behavior):
        """
        The ant is in fact a small wrapper around :cls:`.AntBehavior`.

        The behavior does the decisions and interactions with the environment.
        """
        if initial_node is None:
            raise ValueError('`initial_node` cannot be None')

        # already setup the path with the 
        # initial node as its only element
        self._current_node = initial_node
        self._path = [initial_node]
        self._behavior = behavior
        self._colony = colony

    @property
    def colony(self):
        return self._colony

    @property
    def current_node(self):
        """
        Returns the current node the ant stays on
        """
        return self._current_node

    def move(self):
        """
        Move to the next node
        """
        
        current_node = self.current_node
        edges = current_node.edges

        edge = self._behavior.choose_edge(edges)
        if not edge:
            # check this here for implementations
            raise ValueError('Behavior %s did not return an edge on `choose_edge` '
                             'called with edges %s'
                                % (self._behavior, edges))

        # returns the node which is not the current nod
        next_node = edge.other_node(current_node)

        # if the edge is unidirectional
        if next_node:
            self._behavior.leave_node()

            self._behavior.visit_edge(edge)
            self._behavior.leave_edge(edge)

            self._behavior.visit_node(next_node)

            self._current_node = next_node


def edge_has_pheromones(edge):
    return 


def node_is_food(node):
    """ Check whether a node is food """
    return node.TYPE == 'food'


def node_is_nest(node):
    """ Check whether a node is a nest """
    return node.TYPE == 'nest'


class Waypoint(graph.Node):
    TYPE = 'waypoint'

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _calculate_distance_to(self, other):
        x1, y1, x2, y2 = self.x, self.y, other.x, other.y
        x = abs(abs(x1) - abs(x2))
        y = abs(abs(y1) - abs(y2))
        distance = sqrt(x**2 + y**2)
        return distance


class Food(Waypoint):
    """
    Where the ants like to go to. When food is 
    reached, ants go back to nest
    """
    TYPE = 'food'


class Nest(Waypoint):
    """
    Where the ants have to spawn
    """
    TYPE = 'nest'


class WaypointEdge(graph.Edge):
    """
    Our edge implementation with pheromones
    """

    def __init__(self, pheromone_store):
        self._ps = pheromone_store

    def pheromone_level(self, kind):
        return self._ps.get_amount(kind)

    def increase_pheromones(self, pheromone):
        self._ps.increase(pheromone)

    def decrease_pheromones(self, pheromone):
        self._ps.decreases(pheromone)

    def evaporate_pheromones(self):
        self._ps.evaporate()


class EvaporationStrategy(object):
    def evaporate(self, current_amount):
        return current_amount // 5


DEFAULT_EVAPORATION_STRATEGY = EvaporationStrategy()


class PheromoneStore(object):
    def __init__(self, level=0, evaporation_strategy=None):
        level = level or 0
        evaporation_strategy = evaporation_strategy or DEFAULT_EVAPORATION_STRATEGY
        self._level = collections.defaultdict(int)
        self._es = evaporation_strategy

    def get_amount(self, kind):
        return self._level[kind] 

    def increase(self, pheromone):
        self._level[pheromone.kind] += pheromone.amount

    def decrease(self, pheromone):
        self._level[pheromone.kind] -= pheromone.amount

    def evaporate(self, kind):
        self._level[kind] = self._es.evaporate(self._level[kind])


class Pheromone(object):
    def __init__(self, kind, amount):
        self._kind = kind
        self._amount = amount

    @property
    def kind(self):
        reutrn self._kind

    @property
    def amount(self):
        return self._amount


class PheromoneKind(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


def main():
    """
    Main loop for the problem solver. This can be executed in 
    a different thread.
    """

    graph = graph.Graph()

    nest = Nest()
    wp1 = Waypoint()
    wp2 = Waypoint()
    wp3 = Waypoint()
    wp4 = Waypoint()
    wp5 = Waypoint()
    food = Food()

    graph.add_edge(nest, wp1)
    graph.add_edge(nest, wp2)
    graph.add_edge(wp1, wp2)
    graph.add_edge(wp1, wp3)
    graph.add_edge(wp2, wp3)
    graph.add_edge(wp3, wp4)
    graph.add_edge(wp3, wp5)
    graph.add_edge(wp3, food)
    graph.add_edge(wp5, food)


if __name__ == '__main__':
    main()



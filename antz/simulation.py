"""
"""


import sets
import bisect
import itertools
import collections

from antz import graph
from antz.util import *

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

    def init_ant(self, ant):
        """
        Initialize the ant
        """

    def choose_edge(self, ant, edges):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

    def visit_edge(self, ant, edge):
        """
        Called when the ant moves over the edge to another node.
        """

    def leave_edge(self, ant, edge):
        """
        Called when the ant leaves an edge
        """

    def visit_node(self, ant, node):
        """
        Called when the ant visits a node
        """

    def leave_node(self, ant, node):
        """
        Called when the ant leaves a node
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

    Behaviour is stateless.
    """

    TYPE = 'shortest_path'

    class AntState(object):
        def __init__(self):
            self.way_home = False
            self.edges = []

    def __init__(self):
        self._pheromone_increase = 10

    def init_ant(self, ant):
        if not ant._state:
            ant._state = ShortestPathBehavior.AntState()

    def choose_edge(self, ant, node):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

        edges = node.edges

        if ant._state.way_home:
            # just return the reversed path
            return ant._state.edges.pop()
        else:
            colony = ant.colony
            pkind = colony.pheromone_kind('default')
            # choose the random edge giving edges with
            # more pheromones a higher chance
            weighted_edges = [(e.pheromone_level(pkind), e) for e in edges]
           
            return max(weighted_edges, key=lambda x: x[0])[1]

    def visit_edge(self, ant, edge):
        """
        Just drop some pheromone on the edge
        """

        ant._state.edges.append(edge)

        # todo: pheromone increase should not be static
        edge.increase_pheromone(
            ant.create_pheromone(
                'default', self._pheromone_increase))

    def visit_node(self, ant, node):
        """
        Called when a node is visited
        """

        if node_is_food(node):
            ant._state.way_home = True
            ant._state.path = ant.path[0:]
        elif node_is_nest(node):
            ant._state.way_home = False
        
        if node in ant._path:
            ant._path.remove(node)
        ant._path.append(node)


class AntColony(object):
    def __init__(self, name):
        # make a set out of the ants
        self._pheromone_kinds = {
            'default': PheromoneKind(name='%s:%s' % (name, 'default'))
        }

    def pheromone_kind(self, kind):
        return self._pheromone_kinds[kind]


class NoNextStepError(Exception):
    pass

   
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

        # store your state here
        self._state = None

        self._behavior.init_ant(self)

    def create_pheromone(self, kind, amount):
        colony = self._colony
        kind = colony.pheromone_kind(kind)
        return Pheromone(kind, amount)

    @property
    def path(self):
        return self._path

    @property
    def colony(self):
        return self._colony

    @property
    def current_node(self):
        """
        Returns the current node the ant stays on
        """
        return self._current_node

    def _with_set_state(self, state):
        if state:
            self._state = state

    def move(self):
        """
        Move to the next node
        """
        
        current_node = self.current_node
        
        edge = self._behavior.choose_edge(self, current_node)
        if not edge:
            # check this here for implementations
            raise ValueError('Behavior %s did not return an edge on `choose_edge` '
                             'called with edges %s'
                                % (self._behavior, edges))

        # returns the node which is not the current nod
        next_node = edge.other_node(current_node)

        # if the edge is unidirectional
        if next_node:
            self._with_set_state(
                self._behavior.leave_node(
                    self, current_node))

            self._with_set_state(
                self._behavior.visit_edge(
                    self, edge))
            self._with_set_state(
                self._behavior.leave_edge(
                    self, edge))

            self._with_set_state(
                self._behavior.visit_node(
                    self, next_node))

            self._current_node = next_node
        else:
            raise NoNextStepError()


def node_is_food(node):
    """ Check whether a node is food """
    return node.TYPE == 'food'


def node_is_nest(node):
    """ Check whether a node is a nest """
    return node.TYPE == 'nest'


def node_is_waypoint(node):
    """ Check whether a node is a nest """
    return node.TYPE == 'waypoint'


class Waypoint(graph.Node):
    TYPE = 'waypoint'

    def __init__(self, x=None, y=None):
        graph.Node.__init__(self)
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _calculate_distance_to(self, other):
        if not all([self.x, self.y]):
            return None

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

    def __init__(self, node1, node2, pheromone_store=None, **kwargs):
        graph.Edge.__init__(self, node1, node2, **kwargs)
        self._ps = pheromone_store or PheromoneStore()

    def pheromone_level(self, kind):
        return self._ps.get_amount(kind)

    def increase_pheromone(self, pheromone):
        self._ps.increase(pheromone)

    def decrease_pheromone(self, pheromone):
        self._ps.decreases(pheromone)

    def evaporate_pheromone(self, kind=None):
        self._ps.evaporate(kind)


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

    def _evaporate(self, kind):
        self._level[kind] = self._es.evaporate(self._level[kind])
    
    def evaporate(self, kind=None):
        kind = aslist(kind)   
        if not kind:
            kind = self._level.keys()
        for k in kind:
            self._evaporate(k)


class Pheromone(object):
    def __init__(self, kind, amount):
        self._kind = kind
        self._amount = amount

    @property
    def kind(self):
        return self._kind

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

    g = graph.Graph()

    nest = Nest()
    wp1 = Waypoint()
    wp2 = Waypoint()
    wp3 = Waypoint()
    wp4 = Waypoint()
    wp5 = Waypoint()
    food = Food()

    g.add_edge(WaypointEdge(nest, wp1))
    g.add_edge(WaypointEdge(nest, wp2))
    g.add_edge(WaypointEdge(wp1, wp2))
    g.add_edge(WaypointEdge(wp1, wp3))
    g.add_edge(WaypointEdge(wp2, wp3))
    g.add_edge(WaypointEdge(wp3, wp4))
    g.add_edge(WaypointEdge(wp3, wp5))
    g.add_edge(WaypointEdge(wp3, food))
    g.add_edge(WaypointEdge(wp5, food))

    colony = AntColony('colony-1')
    shortest_path_behavior = ShortestPathBehavior()
    ant = Ant(colony, nest, shortest_path_behavior)    

    while True:
        ant.move()
        for edge in g.edges:
            edge.evaporate_pheromone()
        print('Current path: %s' % ant.path)

if __name__ == '__main__':
    main()



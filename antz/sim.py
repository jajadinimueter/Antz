"""
"""

import sys
import time
import sets
import math
import bisect
import random
import itertools
import collections

from decimal import Decimal

from antz import graph
from antz.util import *

DEBUG = True


class Algorithm(object):
    """
    We need something like a behavior because we want to solve
    different problems with the ants.

    Such as:

        - Find the shortest path
        - Solve the TSP
        - Do war

    The behavior should be an attribute of the ant, like a part
    of the brain. We don't want subclasses here because those would
    be too static.
    """

    TYPE = None  # should be set by the implementation

    def begin_round(self):
        """
        Begin a new round
        """
    def end_round(self):
        """
        Begin a new round
        """

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
        Called when the ant leaves an edge.
        """

    def visit_node(self, ant, node):
        """
        Called when the ant visits a node.
        """

    def leave_node(self, ant, node):
        """
        Called when the ant leaves a node.
        """

    def end_turn(self, ant):
        """
        Called after one turn
        """

    def __repr__(self):
        """
        Default repr implementation for convenience 
        """
        return ('%s(TYPE=%s, <%s>)' 
                    % (self.__class__.__name__,
                        self.TYPE, id(self)))


class ShortestPathAlgorithm(Algorithm):
    """
    Makes the ant behave in a way which leads to a shortest 
    path solution.
    """

    TYPE = 'shortest_path'
    PROB_NO_PHEROMONE = 0.05
    ALPHA = 2
    BETA = 1
    P = 0.01
    NUM_TOP_SOLUTIONS = 10

    class AntState(object):
        def __init__(self):
            self.way_home = False
            self.edges = []
            self.last_edge = None
            self.pathlen = 0
            self.turns = 0
            self.hit_pheromone = 0
            self.best_pathlen = sys.float_info.max

        def add_edge(self, edge):
            self.edges.append(edge)
            self.last_edge = edge
            self.pathlen += edge.cost

    def __init__(self, g, alpha=None, beta=None, p=None):
        self._pheromone_increase = 1
        self._pheromone_edges = set()
        self._alpha = alpha or self.ALPHA
        self._beta = beta or self.BETA
        self._p = p or self.P
        self._best_path_length = sys.float_info.max
        self._best_path = None
        self._edges_in_turn = set()
        self._g = g
        self._rounds = 1
        self._edge_counts = collections.defaultdict(int)
        self._edge_probs = {}
        self._phero_dec = 0.01
        self._alpha = alpha or self.ALPHA
        self._beta = beta or self.BETA

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def phero_dec(self):
        return self._phero_dec

    @phero_dec.setter
    def phero_dec(self, d):
        self._phero_dec = d

    @property
    def solutions(self):
        return self._solutions

    @property
    def top_solutions(self):
        return self._top_solutions

    @property
    def edges_in_turn(self):
        return self._edges_in_turn

    @property
    def best_path_length(self):
        if self._best_path_length == sys.float_info.max:
            return 0
        return self._best_path_length

    @property
    def best_path(self):
        if self._best_path:
            for n in self._best_path:
                if not self._can_pass(n):
                    self._best_path = []
                    self._best_path_length = sys.float_info.max

        return self._best_path

    @property
    def g(self):
        return self._g

    @property
    def pheromone_edges(self):
        return self._pheromone_edges

    def init_ant(self, ant):
        if not ant._state:
            ant._state = ShortestPathAlgorithm.AntState()

    def choose_edge(self, ant, node):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

        state = ant._state

        if state.way_home:
            next_edge = state.edges.pop()
            if node not in next_edge.nodes:
                raise Exception('%s not in next_edge.nodes %s %s' 
                    % (node, next_edge, next_edge.nodes))
            return next_edge
        else:
            probabilities = self._edge_probs.get(node)

            if not probabilities:
                edges = [e for e in node.edges
                            if self._can_pass(e.other_node(node))]
                edges = [e for e in edges
                         if e not in state.edges]

                if not edges:
                    return None

                probabilities = []
                colony = ant.colony
                pkind = colony.pheromone_kind('default')

                def prob(edge):
                    level_prob = edge.pheromone_level(pkind)
                    if not level_prob:
                        return 0
                    level_prob = level_prob ** self._alpha
                    level_prob *= (1/edge.cost) ** self._beta
                    return level_prob 

                probs = {e: prob(e) for e in edges}

                prob_sum = sum(probs.values())

                if not prob_sum:
                    l = len(edges)
                    probabilities = [(1.0/l, e) for e in edges]
                else:
                    probabilities = [(probs[e]/prob_sum, e) 
                                     for e in edges]

                self._edge_probs[node] = probabilities

            if probabilities:
                cdf = [probabilities[0][0]]
                for i in range(1, len(probabilities)):
                    cdf.append(cdf[i-1] + probabilities[i][0])

                r = random.random()
                ind = bisect.bisect(cdf, r)
                return probabilities[ind][1]
            
    def evaporate(self):
        to_remove = set()
        for edge in self._pheromone_edges:
            store = edge.pheromone_store
            kinds = store.kinds
            for k in kinds:
                level = store.get_amount(k)
                store.set(k, (1.0 - self._phero_dec) * level)
                if level < 0.001:
                    to_remove.add(edge)

        for edge in to_remove:
            self._pheromone_edges.remove(edge)

    def visit_edge(self, ant, edge):
        """
        Just drop some pheromone on the edge
        """

        self._edge_counts[edge] += 1

        state = ant._state
        if not state.way_home:
            state.add_edge(edge)
            ant._path_length = state.pathlen

        if state.way_home:
            colony = ant.colony
            pkind = colony.pheromone_kind('default')
            plevel = edge.pheromone_level(pkind)
        
            edge_ant_count = self._edge_counts[edge]
            phero_inc = 1.0/state.pathlen
            phero_inc *= 1.0/(edge_ant_count**6)

            edge.increase_pheromone(
                ant.create_pheromone(
                    'default', phero_inc))

            self._pheromone_edges.add(edge)

    def leave_edge(self, ant, edge):
        pass

    def _can_pass(self, node):
        return not node_is_obstacle(node)

    def visit_node(self, ant, node):
        """
        Called when a node is visited
        """

        state = ant._state

        for n in ant._path:
            if not self._can_pass(n):
                ant._reset()
                return

        if not state.way_home:
            ant._path.append(node)

        if node_is_food(node):
            # handle the thing when it's food
            state.way_home = True

            if state.pathlen <= state.best_pathlen:
                state.best_pathlen = state.pathlen
                ant._best_path = ant._path
                ant._best_path_length = state.best_pathlen

            if state.best_pathlen <= self._best_path_length:
                self._best_path_length = state.best_pathlen 
                self._best_path = ant._best_path
        elif node_is_nest(node):
            # when it's a nest and we are on our way
            # back -> reset the ant
            if state.way_home:
                ant._reset()

    def end_turn(self, ant):
        state = ant._state
        state.turns += 1
        turns = state.turns

        rand = random.random()
        if not state.way_home:
            if self.best_path:
                if rand > 0.8 and turns > random.randrange(200, 400):
                    ant._reset()
            
    def begin_round(self):
        self._edge_probs = {}
        self._edge_counts = collections.defaultdict(int)

    def end_round(self):
        """
        Decrease pheromone level on the stuff
        """
        self._rounds += 1
        self.evaporate()


class AntColony(object):
    def __init__(self, name):
        # make a set out of the ants
        self._pheromone_kinds = {
            'default': PheromoneKind(name='%s:%s' % (name, 'default'))
        }

    def pheromone_kind(self, kind):
        return self._pheromone_kinds[kind]


class NoNextStep(Exception):
    pass

   
class Ant(object):
    def __init__(self, colony, initial_node, algorithm):
        """
        The ant is in fact a small wrapper around :cls:`.AntBehavior`.

        The behavior does the decisions and interactions with the environment.
        """
        if initial_node is None:
            raise ValueError('`initial_node` cannot be None')

        # already setup the path with the 
        # initial node as its only element
        self._initial_node = initial_node
        self._current_node = initial_node
        self._current_edge = None
        self._path = [initial_node]
        self._algorithm = algorithm
        self._colony = colony
        self._best_path = None
        self._path_length = 0
        self._best_path_length = 0

        # store your state here
        self._state = None

        self._algorithm.init_ant(self)

    def _reset(self):
        self._current_node = self._initial_node
        self._current_edge = None

        self._path = [self._initial_node]
        self._path_length = 0
        self._state = None
        self._algorithm.init_ant(self)

    def create_pheromone(self, kind, amount):
        colony = self._colony
        kind = colony.pheromone_kind(kind)
        return Pheromone(kind, amount)

    @property
    def best_path_length(self):
        return self._best_path_length

    @property
    def path_length(self):
        return self._path_length

    @property
    def best_path(self):
        return self._best_path

    @property
    def path(self):
        return self._path

    @property
    def colony(self):
        return self._colony

    @property
    def current_edge(self):
        """
        Returns the current edge the ant stays on
        """
        return self._current_edge

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
        
        edge = self._algorithm.choose_edge(self, current_node)

        if edge:
            self._current_edge = edge
            
            # returns the node which is not the current nod
            next_node = edge.other_node(current_node)
            
            # if the edge is unidirectional
            if next_node:
                if self._current_edge:
                    self._algorithm.leave_edge(
                        self, self._current_edge)

                self._current_node = next_node

                self._algorithm.leave_node(self, current_node)
                self._algorithm.visit_edge(self, edge)
                self._algorithm.visit_node(self, next_node)
            else:
                # reset the ant
                self._reset()
            
            self._algorithm.end_turn(self)
        else:
            self._reset()


def node_is_food(node):
    """ Check whether a node is food """
    return node.TYPE == 'food'


def node_is_nest(node):
    """ Check whether a node is a nest """
    return node.TYPE == 'nest'


def node_is_waypoint(node):
    """ Check whether a node is a waypoint """
    return node.TYPE == 'waypoint'


def node_is_obstacle(node):
    """ Check whether a node is an obstacle """
    return node.obstacle


class Waypoint(graph.Node):
    TYPE = 'waypoint'

    def __init__(self, x=None, y=None, name=None):
        graph.Node.__init__(self, name=name)
        self._x = x
        self._y = y
        self._obstacle = False

    @property
    def obstacle(self):
        return self._obstacle

    @obstacle.setter
    def obstacle(self, obstacle):
        self._obstacle = obstacle

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _calculate_distance_to(self, other):
        if self.x is None or self.y is None:
            return None

        x1, y1, x2, y2 = self.x, self.y, other.x, other.y
        x = abs(abs(x1) - abs(x2))
        y = abs(abs(y1) - abs(y2))
        distance = math.sqrt(x**2 + y**2)

        return distance

    def __repr__(self):
        return ('%s(TYPE=%s, name=%s)'
                  % (self.__class__.__name__,
                     self.TYPE, self._name))

class Food(Waypoint):
    """
    Where the ants like to go to. When food is 
    reached, ants go back to the nest.
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

    def __init__(self, node_from, node_to, pheromone_store=None, 
                    evaporation_strategy=None, **kwargs):
        graph.Edge.__init__(self, node_from, node_to, **kwargs)
        self._ps = pheromone_store or PheromoneStore(
            evaporation_strategy=evaporation_strategy)

    @property
    def pheromone_store(self):
        return self._ps

    def pheromone_level(self, kind):
        return self._ps.get_amount(kind)

    def increase_pheromone(self, pheromone):
        self._ps.increase(pheromone)

    def decrease_pheromone(self, pheromone):
        self._ps.decreases(pheromone)

    def set_pheromone_level(self, kind, amount):
        self._ps.set(kind, amount)

    def evaporate_pheromone(self, kind=None):
        self._ps.evaporate(kind)

    def __repr__(self):
        return ('%s(node_from=%s, node_to=%s)'
                  % (self.__class__.__name__, 
                     self.node_from, self.node_to))


class EvaporationStrategy(object):
    def __init__(self, amount=10):
        self._amount = amount

    def amount(self, current_amount):
        return current_amount - self._amount


DEFAULT_EVAPORATION_STRATEGY = EvaporationStrategy()


class PheromoneStore(object):
    def __init__(self, evaporation_strategy=None):
        evaporation_strategy = evaporation_strategy or DEFAULT_EVAPORATION_STRATEGY
        self._level = {}
        self._es = evaporation_strategy

    @property
    def kinds(self):
        return self._level.keys()

    def get_amount(self, kind):
        return self._get(kind) 

    def increase(self, pheromone):
        k = pheromone.kind
        self._set(k, self._get(k) + pheromone.amount)

    def _decrease(self, kind, amount):
        self._set(kind, self._get(
            pheromone.kind) - amount)

    def _get(self, kind):
        level = self._level.get(kind, 0.0)
        #print(level)
        return level

    def _set(self, kind, amount):
        self._level[kind] = amount

    def set(self, kind, amount):
        self._set(kind, amount)

    def decrease(self, pheromone):
        self._decrease(pheromone.kind, pheromone.amount)

    def _evaporate(self, kind):
        self._set(kind, self._es.amount(
            self._level[kind]))
    
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


class AntCollection(set):
    def __init__(self, behavior):
        self._behavior = behavior

    def reset(self):
        for ant in self:
            ant._reset()

    def move(self):
        self._behavior.begin_round()
        start = time.time()
        for ant in self:
            ant.move()
        self._behavior.end_round()
 

def main():
    """
    Main loop for the problem solver. This can be executed in 
    a different thread.
    """

    g = graph.Graph()

    nest = Nest(name='nest')
    wp1 = Waypoint(name='wp-1')
    wp2 = Waypoint(name='wp-2')
    wp3 = Waypoint(name='wp-3')
    wp4 = Waypoint(name='wp-4')
    wp5 = Waypoint(name='wp-5')
    food = Food(name='food')

    evaporate_strategy = EvaporationStrategy(amount=2)

    # We need to create a waypoint factory
    g.add_edge(WaypointEdge(nest, wp1, evaporation_strategy=evaporate_strategy, cost=100))
    g.add_edge(WaypointEdge(nest, wp2, evaporation_strategy=evaporate_strategy, cost=20))
    g.add_edge(WaypointEdge(wp1, wp3, evaporation_strategy=evaporate_strategy, cost=200))
    g.add_edge(WaypointEdge(wp2, wp3, evaporation_strategy=evaporate_strategy, cost=2))
    g.add_edge(WaypointEdge(wp3, wp4, evaporation_strategy=evaporate_strategy, cost=11))
    g.add_edge(WaypointEdge(wp3, wp5, evaporation_strategy=evaporate_strategy, cost=15))
    g.add_edge(WaypointEdge(wp3, food, evaporation_strategy=evaporate_strategy, cost=10))
    g.add_edge(WaypointEdge(wp5, food, evaporation_strategy=evaporate_strategy, cost=22))

    colony = AntColony('colony-1')
    shortest_path_behavior = ShortestPathAlgorithm()
    
    ants = AntCollection()

    for i in range(0, 100):
        ants.add(Ant(colony, nest, shortest_path_behavior))
    
    pkind = colony.pheromone_kind('default')

    while True:
        ants.move()
        for edge in g.edges:
            edge.evaporate_pheromone()

        if shortest_path_behavior.best_path:
            print('Shortest path length: %s' % shortest_path_behavior.best_path_length)
            print('Shortest path: %s' % format_path(shortest_path_behavior.best_path))


if __name__ == '__main__':
    main()
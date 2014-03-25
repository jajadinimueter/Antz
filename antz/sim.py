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

    def end_turn(self, ant):
        """
        Called after one turn
        """

    def __repr__(self):
        """
        Default repr implementation for convienience 
        """
        return ('%s(TYPE=%s, <%s>)' 
                    % (self.__class__.__name__,
                        self.TYPE, id(self)))


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
            self.last_edge = None
            self.pathlen = 0
            self.turns = 0
            self.hit_pheromone = 0
            self.best_pathlen = sys.float_info.max

        def add_edge(self, edge):
            self.edges.append(edge)
            self.last_edge = edge
            self.pathlen += edge.cost

    def __init__(self):
        self._pheromone_increase = 1
        self._best_path_length = sys.float_info.max
        self._best_path = None

    @property
    def best_path_length(self):
        if self._best_path_length == sys.float_info.max:
            return 0
        return self._best_path_length

    @property
    def best_path(self):
        return self._best_path

    def init_ant(self, ant):
        if not ant._state:
            ant._state = ShortestPathBehavior.AntState()

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
            edges = node.edges[:]
            random_choice = random.random()
            edges = [e for e in edges
                if e not in state.edges]
            
            random.shuffle(edges)

            if not edges:
                return None

            propabilities = []
            colony = ant.colony
            pkind = colony.pheromone_kind('default')

            prop_sum = sum(e.pheromone_level(pkind)**2.0
                for e in edges)

            for edge in edges:
                if prop_sum == 0:
                    prop = 1.0/len(edges)
                else:
                    prop = edge.pheromone_level(pkind)**2/prop_sum
                
                propabilities.append((prop, edge))

            propabilities = list(reversed(sorted(propabilities,
                key=lambda x: x[0])))

            if propabilities:
                cdf = [propabilities[0]]
                for i in range(1, len(propabilities)):
                    cdf.append(cdf[i-1] + propabilities[i])

                ind = bisect.bisect(cdf, random.random())
                return propabilities[ind][1]
            
    def visit_edge(self, ant, edge):
        """
        Just drop some pheromone on the edge
        """

        colony = ant.colony
        pkind = colony.pheromone_kind('default')

        plevel = edge.pheromone_level(pkind)
    
        state = ant._state
        if not state.way_home:
            state.add_edge(edge)
            ant._path_length = state.pathlen

        # if state.way_home:
        # todo: pheromone increase should not be static
        if state.way_home:
            edge.increase_pheromone(
                ant.create_pheromone(
                    'default', self._pheromone_increase))
       
        if self.best_path:
            if plevel or state.way_home:
                state.hit_pheromone = 0
            else:
                state.hit_pheromone -= 1

    def visit_node(self, ant, node):
        """
        Called when a node is visited
        """

        state = ant._state

        if not state.way_home:
            ant._path.append(node)

        if node_is_food(node):
            # handle the thing when it's food
            state.way_home = True

            if state.pathlen < state.best_pathlen:
                state.best_pathlen = state.pathlen
                ant._best_path = ant._path
                ant._best_path_length = state.best_pathlen

            if state.best_pathlen < self._best_path_length:
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

        if self.best_path:
            if state.hit_pheromone < -100:
                ant._reset()
            

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
    def __init__(self, colony, initial_node, behavior):
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
        self._path = [initial_node]
        self._behavior = behavior
        self._colony = colony
        self._best_path = None
        self._path_length = 0
        self._best_path_length = 0

        # store your state here
        self._state = None

        self._behavior.init_ant(self)

    def _reset(self):
        self._current_node = self._initial_node
        self._path = [self._initial_node]
        self._path_length = 0
        self._state = None
        self._behavior.init_ant(self)

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
        
        edge = self._behavior.choose_edge(self, current_node)

        if edge:
            # returns the node which is not the current nod
            next_node = edge.other_node(current_node)
            
            # if the edge is unidirectional
            if next_node:
                self._current_node = next_node
                self._behavior.leave_node(self, current_node)
                self._behavior.visit_edge(self, edge)
                self._behavior.leave_edge(self, edge)
                self._behavior.visit_node(self, next_node)
            else:
                # reset the ant
                self._reset()
            
            self._behavior.end_turn(self)
        else:
            self._reset()


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

    def __init__(self, x=None, y=None, name=None):
        graph.Node.__init__(self, name=name)
        self._x = x
        self._y = y

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

    def __init__(self, node_from, node_to, pheromone_store=None, 
                    evaporation_strategy=None, **kwargs):
        graph.Edge.__init__(self, node_from, node_to, **kwargs)
        self._ps = pheromone_store or PheromoneStore(
            evaporation_strategy=evaporation_strategy)

    def pheromone_level(self, kind):
        return self._ps.get_amount(kind)

    def increase_pheromone(self, pheromone):
        self._ps.increase(pheromone)

    def decrease_pheromone(self, pheromone):
        self._ps.decreases(pheromone)

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
    def move(self):
        start = time.time()
        for ant in self:
            ant.move()


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

    # we need to create a waypoint factory
    g.add_edge(WaypointEdge(nest, wp1, evaporation_strategy=evaporate_strategy, cost=100))
    g.add_edge(WaypointEdge(nest, wp2, evaporation_strategy=evaporate_strategy, cost=20))
    g.add_edge(WaypointEdge(wp1, wp3, evaporation_strategy=evaporate_strategy, cost=200))
    g.add_edge(WaypointEdge(wp2, wp3, evaporation_strategy=evaporate_strategy, cost=2))
    g.add_edge(WaypointEdge(wp3, wp4, evaporation_strategy=evaporate_strategy, cost=11))
    g.add_edge(WaypointEdge(wp3, wp5, evaporation_strategy=evaporate_strategy, cost=15))
    g.add_edge(WaypointEdge(wp3, food, evaporation_strategy=evaporate_strategy, cost=10))
    g.add_edge(WaypointEdge(wp5, food, evaporation_strategy=evaporate_strategy, cost=22))

    colony = AntColony('colony-1')
    shortest_path_behavior = ShortestPathBehavior()
    
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
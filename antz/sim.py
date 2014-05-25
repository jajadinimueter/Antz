"""
"""

import math
import bisect
import random
import collections
import operator
from uuid import uuid4

from antz import graph as antz_graph
from antz.util import *


DEBUG = True


class AlgorithmContext(object):
    def __init__(self, graph, state):
        self.graph = graph
        self.state = state


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

    def begin_round(self, ctx):
        """
        Begin a new round
        """

    def end_round(self, ctx):
        """
        Begin a new round
        """

    def init_ant(self, ctx, ant):
        """
        Initialize the ant
        """

    def choose_edge(self, ctx, ant, edges):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

    def visit_edge(self, ctx, ant, edge):
        """
        Called when the ant moves over the edge to another node.
        """

    def leave_edge(self, ctx, ant, edge):
        """
        Called when the ant leaves an edge.
        """

    def visit_node(self, ctx, ant, node):
        """
        Called when the ant visits a node.
        """

    def leave_node(self, ctx, ant, node):
        """
        Called when the ant leaves a node.
        """

    def end_turn(self, ctx, ant):
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


def relative_cost(g, edge):
    if edge.cost == 0:
        raise ValueError()
    return edge.cost / g.max_cost


def get_weighted_choice(probabilities):
    if probabilities:
        cdf = [probabilities[0][1]]
        for i in range(1, len(probabilities)):
            assert probabilities[i][1] != 0
            cdf.append(cdf[i - 1] + probabilities[i][1])
        r = random.random()
        ind = bisect.bisect(cdf, r)
        item = probabilities[ind]
        return item[0]


class Attribute(object):
    def __init__(self, name, prop, converter, default=None):
        self.name = name
        self.prop = prop
        self.converter = converter
        self.default = default


class Reset(Exception):
    pass


class ShortestPathAlgorithm(Algorithm):
    """
    Makes the and behave in a way which leads to a shortes path solution
    through n nodes.
    """

    ALPHA = 4
    BETA = 2
    PHERO_DECREASE = 0.01
    PHERO_COST_DECREASE = False
    PHERO_COST_DECREASE_POW = 1
    COST_MULTIPLICATOR = 1

    # type and attributes are used by ui to show
    # additional configuration options
    NAME = 'Shortest Path'
    TYPE = 'shortest_path'
    ATTRIBUTES = [
        Attribute('Alpha', 'alpha', float, default=ALPHA),
        Attribute('Beta', 'beta', float, default=BETA),
        # Attribute('Cost-Multiplicator', 'cost_multiplicator', float, default=COST_MULTIPLICATOR),
        Attribute('Pheromone Decrease', 'phero_decrease', float, default=PHERO_DECREASE),
        Attribute('Phero Cost Decrease', 'phero_cost_decrease', bool, default=PHERO_COST_DECREASE),
        Attribute('Phero Cost Decrease Pow', 'phero_cost_decrease_pow', float, default=PHERO_COST_DECREASE_POW),
    ]

    class AlgorithmState(object):
        def __init__(self):
            self._solutions = []
            self._pheromone_edges = set()
            self._solution_counts = collections.defaultdict(int)
            self._best_solution = None
            self._rounds = 0

        @property
        def solutions(self):
            return self._solutions

        @property
        def rounds(self):
            return self._rounds

        @property
        def best_solution(self):
            return self._best_solution

        @property
        def updated_edges(self):
            return self._pheromone_edges

    class AntState(object):
        def __init__(self):
            self.edges = []
            self.turns = 0
            self.solution = None
            self.solution_length = 0

        def add_edge(self, edge):
            self.edges.append(edge)
            self.solution_length += edge.cost

    def create_state(self):
        """
        Returns a state object which has to be passed on every
        method call.
        """
        return self.__class__.AlgorithmState()

    def create_ant_state(self):
        """
        Returns a state object which should be attaced to an ant
        """
        return self.__class__.AntState()

    def __init__(self):
        self.alpha = self.ALPHA
        self.beta = self.BETA
        self.phero_decrease = self.PHERO_DECREASE
        self.cost_multiplicator = self.COST_MULTIPLICATOR
        self.phero_cost_decrease = self.PHERO_COST_DECREASE
        self.phero_cost_decrease_pow = self.PHERO_COST_DECREASE_POW

    def choose_edge(self, ctx, ant, node):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

        state = ant.state

        if state.solution:
            return state.edges.pop()
        else:
            edges = self._get_edges_to_consider(node, exclude_edges=state.edges)
            if edges:
                probabilities = ctx.state._edge_probs.get(node)
                if not probabilities:
                    pkind = self._get_default_pheromone_kind(ant)

                    pheromone_levels = [(edge, edge.pheromone_level(pkind))
                                        for edge in edges]
                    pheromone_levels_only = [level for _, level in pheromone_levels]

                    min_level = 0
                    max_level = max(pheromone_levels_only)

                    min_cost = 0
                    max_cost = min([edge.cost for edge in edges])

                    # noinspection PyShadowingNames
                    def calculate_probability(cost, level, min_level, max_level,
                                              min_cost, max_cost):
                        level = (level - min_level) / (max_level - min_level)
                        cost = 1 / ((cost - min_cost) / (max_cost - min_cost))
                        level **= self.alpha
                        if cost:
                            cost **= self.beta
                            level *= cost
                        return level

                    probabilities = [(edge, calculate_probability(edge.cost, level, min_level,
                                                                  max_level, min_cost, max_cost))
                                     for edge, level in pheromone_levels]

                    probablity_sum = sum([prob for _, prob in probabilities])
                    probablity_sum = probablity_sum or 1

                    probabilities = [(edge, prob / probablity_sum)
                                     for edge, prob in probabilities]

                    ctx.state._edge_probs[node] = probabilities

                return get_weighted_choice(probabilities)

    def evaporate(self, ctx):
        to_remove = set()
        for edge in ctx.state._pheromone_edges:
            store = edge.pheromone_store
            kinds = store.kinds
            for k in kinds:
                level = store.get_amount(k)
                if level < 10 ** -10:
                    level = 0
                    to_remove.add(edge)
                store.set(k, (1.0 - self.phero_decrease) * level)

        for edge in to_remove:
            ctx.state._pheromone_edges.remove(edge)

    def visit_edge(self, ctx, ant, edge):
        """
        Just drop some pheromone on the edge
        """

        state = ant.state

        if not state.solution:
            state.add_edge(edge)
        else:
            min_cost = 0
            max_cost = ctx.graph.max_cost

            cost_multiplicator = 1
            if self.phero_cost_decrease:
                cost_multiplicator = ((edge.cost - min_cost) / (max_cost - min_cost))
                cost_multiplicator **= self.phero_cost_decrease_pow

            phero_inc = 1 / (state.solution[1] * cost_multiplicator)
            # existing_multiplicator = (1 - edge.pheromone_level(ant.pheromone_kind('default')))
            # existing_multiplicator **= 4
            # phero_inc *= existing_multiplicator

            edge.increase_pheromone(
                ant.create_pheromone(
                    'default', phero_inc))

            ctx.state._pheromone_edges.add(edge)

    def visit_node(self, ctx, ant, node):
        """
        Called when a node is visited. We mainly check wether we
        hit a food or a nest node and react accordingly.
        """

        if not self._can_pass(node):
            # if the ant hits an obstacle, kill it
            raise Reset()
        else:
            solution = (tuple(ant.state.edges), ant.state.solution_length)

            if node_is_food(node):
                # the ant found a solution
                ant.state.solution = solution
            elif node_is_nest(node):
                # when it's a nest and we are on our way
                # back -> reset the ant
                if ant.state.solution:
                    raise Reset()

            # increase the solution count to check for most accessed
            # solutions at the end of the round
            if ant.state.solution:
                ctx.state._solution_counts[ant.state.solution] += 1

    def end_turn(self, ctx, ant):
        """
        Check whether the ant is hopelessly running around and
        not finding a solution. Let it die with a certain probability.
        """

        state = ant.state
        state.turns += 1

        rand = random.random()
        if not state.solution:
            if ctx.state._best_solution:
                if rand < 0.2 and state.turns > random.randrange(200, 400):
                    raise Reset()

    def _is_path_accessible(self, path):
        """
        Checks whether the passed path only consists of accessible nodes.
        """
        for edge in path:
            n1 = edge.node_from
            n2 = edge.node_to
            if not self._can_pass(n1) or not self._can_pass(n2):
                return False
        return True

    def begin_round(self, ctx):
        """
        Initialize some values which we create by round. This is the
        probability cache and the counts for active solutions.
        """
        ctx.state._edge_probs = {}
        ctx.state._solution_counts = collections.defaultdict(int)

    def end_round(self, ctx):
        """
        Determine the active, the best and the mostly used solutions.

        Evaporate the pheromone trails.
        """
        ctx.state._rounds += 1

        if ctx.state._solution_counts:
            ctx.state._solutions = sorted(ctx.state._solution_counts.keys(),
                                          key=lambda item: item[1])
            ctx.state._best_solution = ctx.state._solutions[0]
            ctx.state._most_solution = max(ctx.state._solution_counts.items(),
                                           key=operator.itemgetter(1))[0]
        else:
            ctx.state._best_solution = None

        # evaporate pheromones
        self.evaporate(ctx)

    def _can_pass(self, node):
        """
        Returns whether this noce can be passed by an ant.

        ATM checks only whether the node is an obstacle or not.
        """
        return not node_is_obstacle(node)

    def _get_edges_to_consider(self, node, exclude_edges=None):
        """
        Returns edges which are not yet accessed by the ant (passed by
        `exclude_edges` and which are accessible (not obstacles).
        """

        edges = node.edges
        exclude_edges = exclude_edges or []

        edges = [e for e in edges
                 if e not in exclude_edges]

        edges = [e for e in edges
                 if self._can_pass(e.other_node(node))]

        return edges

    def _get_default_pheromone_kind(self, ant):
        """
        Returns the default kind of pheromone of the ants colony
        """
        return ant.pheromones.pheromone_kind('default')


class Ant(object):
    """
    Represents the actual agent which is mapped to a ui entity.

    Ant ant is driven by the algorithm passed to it.
    """

    def __init__(self, initial_node, pheromones, state):
        """
        The ant is in fact a small wrapper around :cls:`.AntBehavior`.

        The behavior does the decisions and interactions with the environment.
        """

        self._pheromones = pheromones
        self._current_node = initial_node
        self._current_edge = None
        self._path_length = 0

        # store your state here
        self._state = state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def pheromones(self):
        return self._pheromones

    def reset(self, initial_node, state=None):
        """
        Resets the ant to initial state so it can be reused.
        """

        self._state = state
        self._current_node = initial_node
        self._current_edge = None

    def pheromone_kind(self, kind):
        return self.pheromones.pheromone_kind(kind)

    def create_pheromone(self, kind, amount):
        """
        Create a pheromone of the passed kind.

        Pass 'default' for the default kind.
        """
        kind = self.pheromones.pheromone_kind(kind)
        return Pheromone(kind, amount)

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

    def move(self, algorithm, ctx):
        """
        Move to the next node. This is entirely driven by algorithm.
        """

        current_node = self.current_node

        edge = algorithm.choose_edge(ctx, self, current_node)

        if edge:
            self._current_edge = edge

            # returns the node which is not the current nod
            next_node = edge.other_node(current_node)

            # if the edge is unidirectional
            if next_node:
                if self._current_edge:
                    algorithm.leave_edge(
                        ctx, self, self._current_edge)

                self._current_node = next_node

                algorithm.leave_node(ctx, self, current_node)
                algorithm.visit_edge(ctx, self, edge)
                algorithm.visit_node(ctx, self, next_node)
            else:
                # reset the ant
                raise Reset()

            algorithm.end_turn(ctx, self)
        else:
            raise Reset()


def node_is_food(node):
    """ Check whether a node is food """
    return node.food


def node_is_nest(node):
    """ Check whether a node is a nest """
    return node.nest


def node_is_waypoint(node):
    """ Check whether a node is a waypoint """
    return node.waypoint


def node_is_obstacle(node):
    """ Check whether a node is an obstacle """
    return node.obstacle


class Waypoint(antz_graph.Node):
    def __init__(self, x=None, y=None, name=None, node_type='waypoint'):
        antz_graph.Node.__init__(self, name=name)
        self._x = x
        self._y = y
        self._node_type = node_type
        self._obstacle = False

    @property
    def node_type(self):
        return self._node_type

    @node_type.setter
    def node_type(self, node_type):
        self._node_type = node_type

    @property
    def food(self):
        return self._node_type == 'food'

    @food.setter
    def food(self, food):
        if food:
            self._node_type = 'food'

    @property
    def nest(self):
        return self._node_type == 'nest'

    @nest.setter
    def nest(self, nest):
        if nest:
            self._node_type = 'nest'

    @property
    def waypoint(self):
        return self._node_type == 'waypoint'

    @waypoint.setter
    def waypoint(self, waypoint):
        if waypoint:
            self._node_type = 'waypoint'

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
        distance = math.sqrt(x ** 2 + y ** 2)

        return distance

    def __repr__(self):
        return ('%s(TYPE=%s, name=%s)'
                % (self.__class__.__name__,
                   self.node_type, self._name))


class WaypointEdge(antz_graph.Edge):
    """
    Our edge implementation with pheromones
    """

    def __init__(self, node_from, node_to, pheromone_store=None,
                 evaporation_strategy=None, **kwargs):
        antz_graph.Edge.__init__(self, node_from, node_to, **kwargs)
        self._ps = pheromone_store or PheromoneStore(
            evaporation_strategy=evaporation_strategy)

    @property
    def pheromone_store(self):
        return self._ps

    def pheromone_level(self, kind):
        return self._ps.get_amount(kind) or 10 ** -10

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
        self._set(kind, self._get(kind) - amount)

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


class Pheromones(object):
    def __init__(self, name=None):
        name = name or str(uuid4())
        # make a set out of the ants
        self._pheromone_kinds = {
            'default': PheromoneKind(name='%s:%s' % (name, 'default'))
        }

    def pheromone_kind(self, kind):
        return self._pheromone_kinds[kind]


def _reset_ant(ant, algorithm, nest_node):
    ant.reset(nest_node, algorithm.create_ant_state())
    return ant


class Runner(object):
    """
    Returned by AntColony.create_runner
    """

    def __init__(self, ants, nest_node, algorithm, graph):
        self._nest_node = nest_node
        self._state = algorithm.create_state()
        self._algorithm = algorithm
        self._ctx = AlgorithmContext(graph, self._state)
        self._ants = ants

    def _reset_ant(self, ant):
        """
        Resets an ant in one run. To reset the whole run,
        you just have to create a new runner
        """
        _reset_ant(ant, self._algorithm, self._nest_node)

    @property
    def ants(self):
        return self._ants

    @property
    def solutions(self):
        return self._state.solutions

    @property
    def rounds(self):
        return self._state.rounds

    @property
    def best_solution(self):
        return self._state.best_solution

    @property
    def updated_edges(self):
        return self._state.updated_edges

    def move(self):
        """
        Move all ants
        """
        self._algorithm.begin_round(self._ctx)
        for ant in self._ants:
            try:
                ant.move(self._algorithm,
                         self._ctx)
            except Reset:
                self._reset_ant(ant)
        self._algorithm.end_round(self._ctx)


class AntColony(object):
    def __init__(self, num_ants=1000):
        self._num_ants = num_ants
        self._pheromones = Pheromones()

    @property
    def pheromones(self):
        return self._pheromones

    def pheromone_kind(self, name):
        return self._pheromones.pheromone_kind(name)

    def create_runner(self, algorithm, graph, nest_node, num_ants=1000):
        ants = set([Ant(nest_node, self._pheromones, algorithm.create_ant_state())
                    for _ in range(0, num_ants or self._num_ants)])
        return Runner(ants, nest_node, algorithm, graph)

